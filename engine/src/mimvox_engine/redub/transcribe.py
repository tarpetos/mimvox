"""Transcription and translation steps of Redub, run by the persistent speech worker (``speech.py``): Whisper turbo
transcribes the file in its own language, NLLB-200 (1.3B or 600M) translates the segments. Segments that split one
sentence are translated together and the translation is cut back into them, so every segment keeps its own text and
times (never merged or dropped)."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from .. import aio, media, speech
from ..registry import EngineError
from . import langs

log = logging.getLogger("mimvox.redub.transcribe")

MODEL_TRANSCRIBE = speech.ASR_MODEL   # large-v3-turbo
MODEL_TRANSLATE = langs.NLLB_DIR_NAME   # the fallback; the result names the model that translated

Cancelled = speech.Cancelled


class Emit:
    """Turns stage-local percents into one monotonic overall ``redub.progress`` stream.

    ``ranges`` maps a stage name to its ``(lo, hi)`` share of 0-100; ``stage(name, pct)`` takes 0-100 inside the stage."""

    def __init__(self, ctx: Any, job: str, ranges: dict[str, tuple[float, float]]) -> None:
        self.ctx, self.job, self.ranges = ctx, job, ranges
        self.last = 0.0
        self.first: dict[str, float] = {}

    def timings(self) -> dict[str, float]:
        """Seconds spent in each stage so far (from its first event to the first event of the next stage)."""
        order = sorted(self.first.items(), key=lambda kv: kv[1])
        now = time.time()
        return {name: round((order[k + 1][1] if k + 1 < len(order) else now) - t, 2)
                for k, (name, t) in enumerate(order)}

    def stage(self, name: str, pct: float, **extra: Any) -> None:
        self.first.setdefault(name, time.time())
        lo, hi = self.ranges[name]
        overall = lo + (hi - lo) * min(100.0, max(0.0, pct)) / 100.0
        self.last = max(self.last, overall)
        self.ctx.emit("redub.progress", {"stage": name, "pct": round(self.last, 1), **extra}, job=self.job)


def decode_16k(path: str) -> tuple[np.ndarray, float]:
    audio, sr = media.read_audio(path, sr=16000)
    if len(audio) < 1600:
        raise EngineError("bad_request", "This file has no audible audio to transcribe.")
    return audio, len(audio) / float(sr)


def _save_npy(workdir: str, audio: np.ndarray) -> str:
    os.makedirs(workdir, exist_ok=True)
    npy = os.path.join(workdir, "audio16k.npy")
    np.save(npy, audio.astype(np.float32, copy=False))
    return npy


async def transcribe(worker: speech.SpeechWorker, audio: np.ndarray, language: str | None, workdir: str,
                     emit: Emit) -> dict[str, Any]:
    """Whisper turbo on 16 kHz audio -> ``{language, segments:[{start,end,text}], device}`` (raw, untrimmed)."""
    npy = await aio.run_io(_save_npy, workdir, audio)
    t0 = time.time()
    try:
        res = await worker.transcribe(npy, language, lambda st, pct: emit.stage(st, pct))
    except speech.SpeechError as exc:
        raise exc.engine_error() from exc
    log.info("whisper %s on %s: %.1fs of audio, %d segment(s), %.1fs total", MODEL_TRANSCRIBE, res.get("device"),
             len(audio) / 16000.0, len(res["segments"]), time.time() - t0)
    return res


def clean_segments(raw: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    segments = []
    for seg in raw:
        text = str(seg["text"]).strip()
        if not text:
            continue
        segments.append({"index": len(segments), "start": round(float(seg["start"]), 2),
                         "end": round(min(float(seg["end"]), duration), 2), "text": text})
    return segments


async def translate(worker: speech.SpeechWorker, segments: list[dict[str, Any]], source_language: str,
                    emit: Emit, target_language: str = "en", translator: str | None = None) -> tuple[list[str], str]:
    """The text of every segment in ``target_language``, in order (``len(result) == len(segments)``), and the model
    that translated it. ``translator``: ``nllb-1.3b`` | ``nllb-600m`` | None (the best installed)."""
    src = langs.nllb_code(source_language)
    if src is None:
        raise EngineError("bad_request", f"Translation from the language {source_language!r} is not supported "
                                         "(the translation model does not know it).")
    tgt = langs.nllb_code(target_language)
    if tgt is None:
        raise EngineError("bad_request", f"Translation to the language {target_language!r} is not supported.")
    t0 = time.time()
    gaps = [max(0.0, float(b["start"]) - float(a["end"])) for a, b in zip(segments, segments[1:])]
    try:
        res = await worker.translate([s["text"] for s in segments], src, lambda st, pct: emit.stage(st, pct), tgt,
                                     translator, gaps)
    except speech.SpeechError as exc:
        raise exc.engine_error() from exc
    out = [str(t) for t in res["texts"]]
    if len(out) != len(segments):
        raise EngineError("internal", "the translation returned a different number of segments")
    if res.get("untranslated"):
        log.warning("segment(s) %s came back empty from the translation model: the source text is kept",
                    res["untranslated"])
    model = str(res.get("model") or MODEL_TRANSLATE)
    log.info("%s %s->%s on %s: %d segment(s), %d joined sentence(s), in %.1fs", model, src, tgt, res.get("device"),
             len(out), len(res.get("joined") or []), time.time() - t0)
    return out, model


async def run(path: str, language: str | None, translate_to: str | None, workdir: str, emit: Emit,
              worker: speech.SpeechWorker, translator: str | None = None) -> dict[str, Any]:
    """Decode ``path``, transcribe it and (``translate_to`` set) translate every segment to that language with
    ``translator`` (None: the best installed). Cancel by cancelling the awaiting task (a running worker op is told to
    stop).

    Returns ``{language, duration_s, segments, model, translated}``; with a translation each segment has ``text`` (in
    ``translate_to``) and ``source_text``, and the result also carries ``source_language`` and ``target_language``."""
    emit.stage("decode", 0)
    audio, duration = await aio.run_io(decode_16k, path)
    emit.stage("decode", 100)
    res = await transcribe(worker, audio, language, workdir, emit)
    detected = langs.norm(res.get("language") or language or "")
    segments = clean_segments(res["segments"], duration)
    out: dict[str, Any] = {"language": detected, "duration_s": round(duration, 3), "segments": segments,
                           "model": MODEL_TRANSCRIBE, "translated": False, "source_language": detected}
    if translate_to and detected != translate_to and segments:
        texts, model = await translate(worker, segments, detected, emit, translate_to, translator)
        for seg, text in zip(segments, texts):
            seg["source_text"] = seg["text"]
            seg["text"] = text
        out.update(translated=True, target_language=translate_to, translation_model=model)
    if "translate" in emit.ranges:
        emit.stage("translate", 100)
    return out
