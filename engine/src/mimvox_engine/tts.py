"""tts.generate / tts.cancel / voices.warm: speech kept loaded, RVC per segment, streamed chunk by chunk; the model
status (``models`` event) and the GPU hand-over to training.

Everything here runs on the engine loop. Synthesis goes through a TTS backend (``backends/``; OmniVoice in its worker
process, whose request queue serializes model calls by design), RVC through the RVC worker client. A generation is a
producer task (synthesis, one segment ahead) and a consumer task (RVC, resample, ``tts.chunk``) joined by a bounded
queue; ``tts.cancel`` cancels the job task and with it both.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shutil
import time
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from . import aio, backends, config, media, rvc, speech, voices
from . import audio_util as au
from .backends import SynthParams
from .backends.omnivoice import OMNIVOICE
from .registry import Context, EngineError, method

log = logging.getLogger("mimvox.tts")

SPEED_MIN, SPEED_MAX = 0.8, 1.3  # speaking-speed range of tts.generate / redub.render ("speed" param)
PAUSE_SENTENCE_MS = 200   # silence appended to a chunk that ends mid-paragraph
PAUSE_PARAGRAPH_MS = 480  # ... and at a paragraph end
MIN_FIRST_TOKENS = 6  # the first segment is one sentence unless that is shorter than this
FIRST_TOKENS = 40   # cap for segment 0 (normally just the first sentence, ~13 tokens = ~4 s of speech)
GROWTH = float(os.environ.get("MIMVOX_SEGMENT_GROWTH", "2.5"))  # each segment may hold ~2.5x the previous target (measured: lowest total time)
LATER_TOKENS = 70   # cap for every later segment
MIN_PIECE_WORDS = 3   # a cut sentence never leaves a piece shorter than this many words...
MIN_PIECE_SHARE = 0.3  # ...or than this share of the segment budget
MIN_TAIL_TOKENS = 10  # a paragraph never ends in a segment shorter than this (merged into the previous one)

# Loop-thread state (only the event loop changes it, so no locks).
_RVC_STATUS = ["loading"]
_ACTIVE: list[str | None] = [None]   # the job holding the one generation slot (tts.generate, Redub GPU jobs)
_WARMING: set[str] = set()
_SETTLED: asyncio.Event | None = None  # set while no model is "loading"


def models_status() -> dict[str, str]:
    return {"tts": OMNIVOICE.status(), "rvc": _RVC_STATUS[0]}


def _settled() -> asyncio.Event:
    global _SETTLED
    if _SETTLED is None:
        _SETTLED = asyncio.Event()
        if "loading" not in models_status().values():
            _SETTLED.set()
    return _SETTLED


def _set_status(ctx: Context | None, **kv: str) -> None:
    if "tts" in kv:
        OMNIVOICE.set_status(kv["tts"])
    if "rvc" in kv:
        _RVC_STATUS[0] = kv["rvc"]
    snap = models_status()
    ev = _settled()
    if "loading" in snap.values():
        ev.clear()
    else:
        ev.set()
    if ctx is not None:
        ctx.emit("models", snap)


async def wait_settled(timeout: float) -> bool:
    """Wait (awaiting an event, not polling) until no model is ``loading``; False after ``timeout`` seconds."""
    try:
        await asyncio.wait_for(_settled().wait(), timeout)
        return True
    except asyncio.TimeoutError:
        return False


# --------------------------------------------------------------------------------------------- text segmentation

_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no", "inc", "ltd", "mt",
           "gen", "col", "lt", "capt", "sgt", "fig", "approx"}
_SENT_END = re.compile(r"""([.!?…]+["'”’)\]]*)\s+""")
_CLAUSE = re.compile(r"(?<=[,;:—–])\s+")


def _split_sentences(par: str) -> list[str]:
    out, start = [], 0
    for m in _SENT_END.finditer(par):
        head = par[start:m.start()] + m.group(1)
        last = re.findall(r"[A-Za-z.]+$", par[start:m.start()])
        word = last[0].lower().rstrip(".") if last else ""
        if m.group(1).startswith(".") and (word in _ABBREV or (len(word) == 1 and word.isalpha())):
            continue  # "Dr. Smith", "J. Smith", "e.g. this"
        out.append(head.strip())
        start = m.end()
    tail = par[start:].strip()
    if tail:
        out.append(tail)
    return out


def segment_text(text: str, count: Callable[[str], int]) -> list[tuple[str, int]]:
    """Split ``text`` into ``(segment, pause_after_ms)`` pairs.

    Sentences are never cut unless one exceeds the token budget (then at clause punctuation, then at words).
    The first segment is a single sentence so the first audio arrives fast; later ones are packed greedily up to
    the growing token budget, never across a paragraph break.
    """
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n|\n", text)]
    paragraphs = [p for p in paragraphs if p]
    segs: list[tuple[str, int]] = []

    target = [FIRST_TOKENS]

    def budget() -> int:
        """Growing schedule: segment 0 is the first sentence, then each segment may be ~GROWTH x the previous target (2.5 measured best), so
        playback of a chunk covers generation of the next (generation runs at ~1.05-1.4x real time)."""
        return target[0]

    def add(seg: str, pause: int) -> None:
        if not segs:
            target[0] = min(LATER_TOKENS, max(count(seg) + 6, round(count(seg) * GROWTH)))
        else:
            target[0] = min(LATER_TOKENS, max(target[0] + 6, round(target[0] * GROWTH)))
        segs.append((seg, pause))

    def words_split(words: list[str], parts: int) -> list[str]:
        """``words`` cut into ``parts`` runs of about equal length (never an empty or one-word run when avoidable)."""
        goal = count(" ".join(words)) / max(parts, 1)
        out: list[str] = []
        cur: list[str] = []
        for i, w in enumerate(words):
            left = len(words) - i  # words not placed yet, this one included
            if cur and len(out) < parts - 1 and left >= 2 and count(" ".join(cur + [w])) > goal:
                carry: list[str] = []
                # do not end a run on a preposition, conjunction or article ("з", "в", "the", "of"): it belongs with
                # the word after it
                while len(cur) > 1 and len(cur[-1].strip("\"'«»()")) <= 3 and not re.search(r"[,.;:!?…—–]$", cur[-1]):
                    carry.insert(0, cur.pop())
                out.append(" ".join(cur))
                cur = carry + [w]
            else:
                cur.append(w)
        out.append(" ".join(cur))
        return out

    def short(piece: str, limit: int) -> bool:
        return len(piece.split()) < MIN_PIECE_WORDS or count(piece) < limit * MIN_PIECE_SHARE

    def pieces(sentence: str, limit: int) -> list[str]:
        """Cut an over-budget sentence: at clause punctuation first (clauses stay whole), a clause longer than
        LATER_TOKENS into equal runs of words, then re-pack neighbouring clauses up to ``limit``. No piece is left
        as a stub (a last word or two on its own): a short piece joins its neighbour, or the pair is re-cut evenly."""
        if count(sentence) <= limit:
            return [sentence]
        out: list[str] = []
        for clause in _CLAUSE.split(sentence):
            n = -(-count(clause) // LATER_TOKENS)
            out.extend([clause] if n <= 1 else words_split(clause.split(" "), n))
        merged: list[str] = []
        for p in out:
            if merged and count(merged[-1] + " " + p) <= limit:
                merged[-1] += " " + p
            else:
                merged.append(p)
        # stubs: join them to a neighbour when the pair still fits (a little over is fine), else re-cut the pair evenly
        i = 0
        while i < len(merged):
            if len(merged) > 1 and short(merged[i], limit):
                j = i - 1 if i > 0 else i + 1
                a, b = (merged[j], merged[i]) if j < i else (merged[i], merged[j])
                pair = f"{a} {b}"
                k = min(i, j)
                if count(pair) <= limit + MIN_TAIL_TOKENS:
                    merged[k:k + 2] = [pair]
                else:
                    merged[k:k + 2] = words_split(pair.split(" "), 2)
                    i = k + 2
                    continue
                i = max(k, 0)
                continue
            i += 1
        return merged

    for par in paragraphs:
        cur = ""
        par_start = len(segs)
        for sent in _split_sentences(par):
            if not segs and cur and count(cur) >= MIN_FIRST_TOKENS:  # segment 0 is just the first sentence
                add(cur, PAUSE_SENTENCE_MS)
                cur = ""
            for piece in pieces(sent, budget()):
                if not cur:
                    cur = piece
                elif (not segs and count(cur) >= MIN_FIRST_TOKENS) or count(cur + " " + piece) > budget():
                    add(cur, PAUSE_SENTENCE_MS)
                    cur = piece
                else:
                    cur = f"{cur} {piece}"
        if cur:
            if len(segs) > max(par_start, 1) and count(cur) < MIN_TAIL_TOKENS \
                    and count(segs[-1][0]) + count(cur) <= LATER_TOKENS + MIN_TAIL_TOKENS:
                cur = f"{segs.pop()[0]} {cur}"  # no tiny trailing segment: its fixed cost is not worth it
            add(cur, PAUSE_PARAGRAPH_MS)
    return segs



# ----------------------------------------------------------------------------------------------- model loading

def _pick_warm_voice(data: config.DataDir) -> tuple[str | None, str | None]:
    """(reference path, voice id with an RVC model) to warm with: prefer a trained voice."""
    ref_only = None
    for vid in voices.list_voice_ids(data):
        ref = data.reference(vid)
        if not os.path.isfile(ref):
            continue
        if os.path.isfile(data.rvc_model(vid)[0]):
            return ref, vid
        ref_only = ref_only or ref
    return ref_only, None


async def load_models(ctx: Context) -> None:
    """Background start-up (and reload after a training): the speech model (OmniVoice worker) and the RVC worker start
    in parallel, then both are warmed up with a trained voice. Status changes are ``models`` events."""
    data = config.data_dir(ctx)
    errors: dict[str, str] = {}
    worker = rvc.get(ctx.runtime)
    _set_status(ctx, tts="loading", rvc="loading")

    async def do_tts() -> None:
        try:
            await OMNIVOICE.load(ctx.runtime)
            if OMNIVOICE.status() == "missing":
                errors["tts"] = "OmniVoice is not installed or failed to load (see the log); run setup"
        except Exception as exc:  # noqa: BLE001
            log.exception("the speech model failed to load")
            errors["tts"] = f"{type(exc).__name__}: {exc}"

    async def do_rvc() -> None:
        if worker is None:
            errors["rvc"] = "RVC runtime not found"
            return
        try:
            await worker.start()
        except Exception as exc:  # noqa: BLE001
            log.exception("RVC worker failed to start")
            errors["rvc"] = f"{type(exc).__name__}: {exc}"

    async def do_others() -> None:
        for b in backends.all_backends():
            if b is not OMNIVOICE:
                try:
                    await b.load(ctx.runtime)
                except Exception:  # noqa: BLE001 - an extra backend never blocks the main one
                    log.exception("TTS backend %s failed to load", b.name)

    t0 = time.time()
    await asyncio.gather(do_tts(), do_rvc(), do_others())
    warm_dir = os.path.join(data.tmp, "_warm")
    await aio.run_io(os.makedirs, warm_dir, exist_ok=True)
    wav: str | None = None
    trained_voice: str | None = None
    try:
        if "tts" not in errors:
            ref, trained_voice = await aio.run_io(_pick_warm_voice, data)
            if ref is not None:  # without any voice there is nothing to clone yet: the first generation warms up
                t1 = time.time()
                lang = await aio.run_io(_voice_language, data, trained_voice) if trained_voice else "en"
                wav = await OMNIVOICE.synthesize(_WARM_TEXT.get(lang, _WARM_TEXT["en"]), lang, ref,
                                                 SynthParams(out_path=os.path.join(warm_dir, "warm_tts.wav"),
                                                             first=True))
                log.info("speech model warm-up %.1fs", time.time() - t1)
            _set_status(ctx, tts="ready")
        else:
            _set_status(ctx, tts="missing")
        if "rvc" not in errors and worker is not None:
            if wav and trained_voice:
                pth, idx = data.rvc_model(trained_voice)
                t1 = time.time()
                await worker.convert(wav, os.path.join(warm_dir, "warm_rvc.wav"), pth, idx, 1.0)
                log.info("RVC warm-up %.1fs", time.time() - t1)
            _set_status(ctx, rvc="ready")
        else:
            _set_status(ctx, rvc="missing")
    except Exception:  # noqa: BLE001
        log.exception("warm-up failed")
        _set_status(ctx, tts="missing" if "tts" in errors else "ready",
                    rvc="ready" if "rvc" not in errors and worker is not None else "missing")
    finally:
        await aio.run_io(shutil.rmtree, warm_dir, ignore_errors=True)
    log.info("models ready in %.1fs: %s", time.time() - t0, models_status())
    if errors:
        log.warning("model start-up problems: %s", errors)
    aio.spawn(aio.run_io(_prewarm_redub), name="redub-prewarm")


def _prewarm_redub() -> None:
    """Redub's slot fitting trims clips with librosa, whose first call compiles numba kernels (measured ~20 s in the
    middle of the first dub), and stretches them with an ffmpeg filter (the filter list is read once). Pay both once in
    the background after start-up instead."""
    try:
        import librosa  # noqa: PLC0415

        t0 = time.time()
        y = (0.1 * np.sin(np.arange(22050) * 0.05)).astype(np.float32)
        librosa.effects.trim(y, top_db=40, frame_length=512, hop_length=128)
        log.info("librosa warmed up in %.1fs", time.time() - t0)
    except Exception:  # noqa: BLE001 - only an optimization
        log.exception("librosa warm-up failed")
    try:
        log.info("redub time-stretch: %s", ", ".join(media.stretch_filters()) or "none (long clips overrun)")
    except Exception:  # noqa: BLE001 - only an optimization
        log.exception("listing the ffmpeg filters failed")


_WARM_TEXT = {"en": "Warming up the voice engine.", "uk": "Розігріваю голосовий рушій."}


def _voice_language(data: config.DataDir, voice_id: str) -> str:
    try:
        return str(voices.read_meta(data, voice_id).get("language") or "en").lower()
    except EngineError:
        return "en"


# ----------------------------------------------------------------------------------------------------- methods

def duration_factor_for(speed: float) -> float:
    """``SynthParams.duration_factor`` (length of the speech relative to natural; below 1 is faster) for a speed."""
    return round(1.0 / min(SPEED_MAX, max(SPEED_MIN, float(speed))), 4)


def _num(params: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    v = params.get(key, default)
    if v is None:
        v = default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise EngineError("bad_request", f"'{key}' must be a number")
    return min(hi, max(lo, float(v)))


def speech_language(params: dict[str, Any], key: str = "language") -> tuple[str, backends.TtsBackend]:
    """The ``language`` parameter of a speech request (default ``en``) and the backend that speaks it."""
    lang = params.get(key) or "en"
    if not isinstance(lang, str):
        raise EngineError("bad_request", f"'{key}' must be a language code such as \"en\"")
    lang = lang.lower()
    backend = backends.for_language(lang)
    if backend is None:
        raise EngineError("bad_request", f"no voice model speaks {lang!r} (available: "
                                         f"{', '.join(sorted(backends.languages()))})")
    return lang, backend


def check_backend_ready(backend: backends.TtsBackend) -> None:
    state = backend.status()
    if state != "ready":
        raise EngineError("not_ready", "the speech model is still loading" if state == "loading"
                          else "speech is paused while a voice trains" if state == "paused"
                          else "the speech model is not available")


@method("tts.generate")
async def tts_generate(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    text = params.get("text")
    if not isinstance(text, str) or not text.strip():
        raise EngineError("bad_request", "'text' must be a non-empty string")
    voice_id = str(params.get("voice_id") or "")
    temperature = _num(params, "expressiveness", 0.8, 0.1, 1.5)
    index_rate = _num(params, "voice_match", 1.0, 0.0, 1.0)
    speed = _num(params, "speed", 1.0, SPEED_MIN, SPEED_MAX)
    quality = params.get("quality") or "best"
    if quality not in ("best", "fast"):
        raise EngineError("bad_request", "'quality' must be \"best\" or \"fast\"")
    lang, backend = speech_language(params)
    check_backend_ready(backend)
    meta = await aio.run_io(voices.read_meta, data, voice_id)  # not_found
    out_fmt = await aio.run_io(media.output_format, params)
    rvc_rate = rvc.index_rate_for(index_rate, lang, meta.get("language"))  # capped for foreign languages
    ref = data.reference(voice_id)
    pth, idx = data.rvc_model(voice_id)
    has_ref, has_pth = await aio.run_io(lambda: (os.path.isfile(ref), os.path.isfile(pth)))
    if not has_ref:
        raise EngineError("not_found", "this voice has no recording yet (reference.wav is missing)")
    use_rvc = has_pth and index_rate > 0
    worker = rvc.get(ctx.runtime) if use_rvc else None
    if use_rvc and (worker is None or models_status()["rvc"] != "ready"):
        raise EngineError("not_ready", "the voice-conversion model is not ready")
    segs = segment_text(text, backend.count_tokens)
    if not segs:
        raise EngineError("bad_request", "nothing to say in this text")
    # (no await between the busy check in start_exclusive and taking the slot: the loop makes it atomic)
    job = start_exclusive(ctx, lambda j: _generate(ctx, j, data, backend, lang, ref, segs, temperature, quality, pth,
                                                   idx, rvc_rate, worker, duration_factor_for(speed), out_fmt),
                          "another generation is running")
    return {"job": job, "segments": len(segs), "segment_chars": [len(t) for t, _ in segs]}


@method("tts.cancel")
async def tts_cancel(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    job = params.get("job")
    if not isinstance(job, str):
        raise EngineError("bad_request", "'job' is required")
    ctx.cancel(job)
    return {}


def _chunk_from(src: str, i: int, last: bool, pause: int, jobdir: str) -> tuple[np.ndarray, str, float]:
    """IO thread: the final 44.1 kHz chunk ``i`` from a synthesized / converted WAV (pause appended), written to disk."""
    x, sr = au.read_mono(src)
    x = au.resample(x, sr, au.OUT_SR)
    if not last:
        x = np.concatenate([x, au.silence(pause)])
    path = os.path.join(jobdir, f"chunk{i:03d}.wav")
    au.write_wav(path, x, au.OUT_SR)
    for p in (os.path.join(jobdir, f"s{i}_tts.wav"), os.path.join(jobdir, f"s{i}_rvc.wav")):
        with contextlib.suppress(OSError):
            os.remove(p)
    return x, path, au.duration_s(x)


async def _generate(ctx: Context, job: str, data: config.DataDir, backend: backends.TtsBackend, lang: str, ref: str,
                    segs: list[tuple[str, int]], temperature: float, quality: str, pth: str, idx: str,
                    index_rate: float, worker: rvc.RvcWorker | None, duration_factor: float, out_fmt: str) -> None:
    t_start = time.time()

    def finish(event: str, payload: dict[str, Any]) -> None:
        release_job(job)  # free the slot before the app can see the end of this job (it may start the next one)
        ctx.emit(event, payload, job=job)

    jobdir = os.path.join(data.tmp, job)
    try:
        await aio.run_io(_prepare_jobdir, data, jobdir)
        await speech.yield_gpu()  # Whisper / NLLB (if warm) leave the GPU to the speech model and RVC
        log.info("job %s: %s, %d segment(s), RVC index %s: %s", job, lang, len(segs),
                 index_rate if worker is not None else "-", [s for s, _ in segs])
        chunks: list[np.ndarray | None] = [None] * len(segs)
        todo: asyncio.Queue[tuple[int, str, int] | None] = asyncio.Queue(maxsize=2)
        first_chunk_at: list[float] = []
        last_ready = [0.0]

        async def produce() -> None:
            for i, (seg, pause) in enumerate(segs):
                t0 = time.time()
                path = await backend.synthesize(seg, lang, ref, SynthParams(
                    out_path=os.path.join(jobdir, f"s{i}_tts.wav"), temperature=temperature, quality=quality,
                    duration_factor=duration_factor, first=(i == 0)))
                log.info("job %s seg %d: %s %.2fs", job, i, backend.name, time.time() - t0)
                await todo.put((i, path, pause))  # waits while two segments are ahead of the consumer
            await todo.put(None)

        async def consume() -> None:
            """RVC + resample + emit, in order, overlapping with the synthesis of the next segment."""
            async for i, path, pause in aio.queue_items(todo):
                src = path
                if worker is not None:
                    src = os.path.join(jobdir, f"s{i}_rvc.wav")
                    await worker.convert(path, src, pth, idx, index_rate)
                last = i == len(segs) - 1
                x, chunk, dur = await aio.run_io(_chunk_from, src, i, last, pause, jobdir)
                chunks[i] = x
                now = time.time() - t_start
                if not first_chunk_at:
                    first_chunk_at.append(now)
                speech_s = max(dur - (0.0 if last else pause / 1000.0), 0.01)
                rtf = (now - last_ready[0]) / speech_s  # wall time since the previous chunk / this chunk's speech
                last_ready[0] = now
                ctx.emit("tts.chunk", {"index": i, "path": chunk, "duration_s": dur, "segments": len(segs),
                                       "ready_s": round(now, 2), "rtf": round(rtf, 2)}, job=job)

        await aio.gather_or_cancel(produce(), consume())
        full = np.concatenate([c for c in chunks if c is not None])
        saved = await aio.run_io(media.save_result, data, full, au.OUT_SR, "", out_fmt)
        finish("tts.done", {**saved, "duration_s": au.duration_s(full), "elapsed_s": round(time.time() - t_start, 2)})
        log.info("job %s done in %.1fs (first chunk %.1fs)", job, time.time() - t_start,
                 first_chunk_at[0] if first_chunk_at else -1)
    except asyncio.CancelledError:
        finish("tts.error", {"code": "cancelled", "message": "generation cancelled"})
    except EngineError as exc:
        finish("tts.error", {"code": exc.code, "message": exc.message})
    except Exception as exc:  # noqa: BLE001
        log.exception("job %s failed", job)
        finish("tts.error", {"code": "internal", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        release_job(job)


def _prepare_jobdir(data: config.DataDir, jobdir: str) -> None:
    os.makedirs(jobdir, exist_ok=True)
    _gc_tmp(data)


def start_exclusive(ctx: Context, fn: Callable[[str], Awaitable[None]],
                    busy_message: str = "another generation or Redub job is running") -> str:
    """Start an async GPU job that takes the one generation slot (``tts.generate``, Redub and training then answer
    ``busy``). ``fn(job)`` should call :func:`release_job` before it emits its final event; the slot is freed when it
    ends in any case. Loop thread only (the check-and-take is atomic there). Raises ``busy`` when taken."""
    if _ACTIVE[0] is not None:
        raise EngineError("busy", busy_message)

    async def body(job: str) -> None:
        try:
            await fn(job)
        finally:
            release_job(job)

    job = ctx.start_job(body)
    _ACTIVE[0] = job
    return job


def release_job(job: str) -> None:
    if _ACTIVE[0] == job:
        _ACTIVE[0] = None


def is_busy() -> bool:
    """A generation (or Redub / training GPU job) holds the generation slot."""
    return _ACTIVE[0] is not None


def _gc_tmp(data: config.DataDir) -> None:
    """Drop chunk folders of finished jobs older than an hour."""
    try:
        for name in os.listdir(data.tmp):
            p = os.path.join(data.tmp, name)
            if name.startswith("job-") and os.path.isdir(p) and time.time() - os.path.getmtime(p) > 3600:
                shutil.rmtree(p, ignore_errors=True)
    except OSError:
        pass


def forget_reference() -> None:
    """Drop cached speaker conditioning (call after ``reference.wav`` was replaced): the next synthesis conditions on
    the new clip. Callable from any thread. (The OmniVoice worker also notices a changed file by itself.)"""
    for b in backends.all_backends():
        b.forget_reference()


# ------------------------------------------------------------------------------------------------ voices.warm

@method("voices.warm")
async def voices_warm(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """Precompute the speaker conditioning of a voice in the background (call when the user selects a voice)."""
    data = config.data_dir(ctx)
    vid = str(params.get("id") or "")
    meta = await aio.run_io(voices.read_meta, data, vid)  # not_found
    lang = str(meta.get("language") or "en").lower()
    ref = data.reference(vid)
    pth, idx = data.rvc_model(vid)
    has_ref, has_pth = await aio.run_io(lambda: (os.path.isfile(ref), os.path.isfile(pth)))
    if not has_ref:
        raise EngineError("not_found", "this voice has no recording yet (reference.wav is missing)")
    if models_status()["tts"] != "ready":
        raise EngineError("not_ready", "the speech model is not ready")
    worker = rvc.get(ctx.runtime) if has_pth else None
    if worker is not None and models_status()["rvc"] != "ready":
        worker = None
    if OMNIVOICE.is_warm(ref) and (worker is None or worker.loaded_model == rvc.model_sig(pth)):
        return {"warming": False, "reason": "already_warm"}
    if _ACTIVE[0] is not None:
        return {"warming": False, "reason": "busy"}
    if vid in _WARMING:
        return {"warming": True}
    _WARMING.add(vid)
    aio.spawn(_warm(ctx, data, vid, ref, lang, worker, pth, idx), name=f"warm-{vid}")
    return {"warming": True}


async def _warm(ctx: Context, data: config.DataDir, vid: str, ref: str, lang: str, worker: rvc.RvcWorker | None,
                pth: str, idx: str) -> None:
    t0 = time.time()
    warm_dir = os.path.join(data.tmp, f"_warm_{vid}")
    try:
        await aio.run_io(os.makedirs, warm_dir, exist_ok=True)
        # The shortest real inference: builds the voice-clone prompt (and, the first time for a clip, its transcript).
        src = await OMNIVOICE.synthesize(_WARM_TEXT.get(lang, _WARM_TEXT["en"]), lang if lang in OMNIVOICE.languages
                                         else "en", ref, SynthParams(out_path=os.path.join(warm_dir, "in.wav"),
                                                                     first=True))
        if worker is not None:  # load the voice's RVC model and index too (about 16 s cold, 1.7 s warm)
            await worker.convert(src, os.path.join(warm_dir, "out.wav"), pth, idx, 1.0)
        ctx.emit("voice.warm", {"id": vid, "ok": True, "elapsed_s": round(time.time() - t0, 2)})
    except asyncio.CancelledError:
        ctx.emit("voice.warm", {"id": vid, "ok": False, "message": "cancelled"})
    except Exception as exc:  # noqa: BLE001
        log.exception("warm-up of voice %s failed", vid)
        ctx.emit("voice.warm", {"id": vid, "ok": False, "message": f"{type(exc).__name__}: {exc}"})
    finally:
        _WARMING.discard(vid)
        await aio.run_io(shutil.rmtree, warm_dir, ignore_errors=True)


# ------------------------------------------------------------------------------------- GPU hand-over (training)

async def release_gpu(ctx: Context) -> None:
    """Free the GPU and host RAM for training: unload the speech models (on the GPU thread, after any synthesis in
    flight) and stop the RVC and speech worker processes (their exits are awaited).

    Raises ``EngineError("busy")`` while a generation is running. Until :func:`reacquire_gpu` finishes, both models
    report ``"paused"`` in ``models_status()`` and ``tts.generate`` answers ``not_ready``."""
    if _ACTIVE[0] is not None:
        raise EngineError("busy", "a generation is running; wait for it or cancel it before training")
    _set_status(ctx, tts="paused", rvc="paused")
    for b in backends.all_backends():
        try:
            await b.release()
        except Exception:  # noqa: BLE001
            log.exception("TTS backend %s could not release the GPU", b.name)
    await asyncio.gather(rvc.shutdown(), speech.shutdown())


def reacquire_gpu(ctx: Context) -> asyncio.Task[None] | None:
    """Reload the speech models and start a fresh RVC worker in the background (about 50 s; ``models`` events, then
    ``ready``). A fresh worker process also reads a freshly trained ``rvc/model.pth`` from disk. Not while the engine
    shuts down."""
    if aio.closing():
        return None
    return aio.spawn(load_models(ctx), name="model-reload")


@contextlib.asynccontextmanager
async def gpu_released(ctx: Context):  # type: ignore[no-untyped-def]
    """``async with tts.gpu_released(ctx): <train>``: release before, reload in the background after (also on error)."""
    await release_gpu(ctx)
    try:
        yield
    finally:
        reacquire_gpu(ctx)
