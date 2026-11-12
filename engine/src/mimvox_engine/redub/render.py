"""Redub rendering: ``convert`` (RVC over the original audio, same timing) and ``respeak`` (TTS per transcript
segment, fitted into the original time slots).

Async, on the engine loop: synthesis is awaited from the TTS backend (GPU thread), RVC from the RVC worker client, and
the numpy work runs on the IO pool. Cancel by cancelling the awaiting task."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import time
from typing import Any, Callable

import numpy as np

from .. import aio, backends, config, rvc, tts
from .. import audio_util as au
from ..backends import SynthParams
from ..registry import EngineError

log = logging.getLogger("mimvox.redub.render")

SR = au.OUT_SR                  # 44.1 kHz working / output rate
CHUNK_MIN_S, CHUNK_MAX_S = 15.0, 30.0   # convert chunks are cut at the quietest point between these lengths
MIN_TAIL_S = 2.0                # a shorter remainder is merged into the previous chunk
SILENT_PEAK = 1e-4              # a chunk quieter than this is copied, not converted
GAP_S = 0.12                    # silence kept before the next original start when fitting a clip
MIN_AVAIL_S = 0.25              # never fit a clip into less than this
CLIP_GAP_S = 0.05               # minimum silence between two placed clips
FACTOR_MIN, FACTOR_MAX = 0.8, 1.25   # duration_factor range handed to the speech model for slot fitting (outside it quality drops)
FACTOR_LO, FACTOR_HI = 0.65, 1.4  # hard limits of the combined (slot x speed) factor
MAX_STRETCH = 1.2               # librosa time-stretch, faster only
INITIAL_SEC_PER_CHAR = 0.068    # prior for the speaking rate (about 14.7 characters per second)
PRIOR_CHARS = 40


Progress = Callable[..., None]


# ------------------------------------------------------------------------------------------------ helpers

def _fade(x: np.ndarray, ms: float = 5.0) -> np.ndarray:
    n = min(int(SR * ms / 1000.0), len(x) // 2)
    if n > 1:
        x = x.copy()
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=np.float32)])


def split_chunks(x: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Cut ``x`` into ~15-30 s pieces at the quietest 20 ms frame in each window (a pause when there is one)."""
    n = len(x)
    if n <= int(CHUNK_MAX_S * sr):
        return [(0, n)]
    frame = int(0.02 * sr)
    energy = np.convolve(np.add.reduceat(x.astype(np.float64) ** 2, np.arange(0, n, frame)), np.ones(3) / 3, "same")
    cuts = [0]
    while n - cuts[-1] > int(CHUNK_MAX_S * sr):
        lo = (cuts[-1] + int(CHUNK_MIN_S * sr)) // frame
        hi = (cuts[-1] + int(CHUNK_MAX_S * sr)) // frame
        cuts.append(int((lo + int(np.argmin(energy[lo:hi]))) * frame + frame // 2))
    if len(cuts) > 1 and n - cuts[-1] < int(MIN_TAIL_S * sr):
        cuts.pop()
    cuts.append(n)
    return list(zip(cuts[:-1], cuts[1:]))


def output_base(data: config.DataDir) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(data.outputs, f"redub_{stamp}")
    n = 1
    while os.path.exists(base + ".wav"):
        n += 1
        base = os.path.join(data.outputs, f"redub_{stamp}_{n}")
    os.makedirs(data.outputs, exist_ok=True)
    return base


def save_outputs(data: config.DataDir, track: np.ndarray) -> tuple[str, str]:
    track = np.clip(track, -1.0, 1.0)
    base = output_base(data)
    au.write_wav(base + ".wav", track, SR)
    au.write_mp3(base + ".mp3", track, SR)
    return base + ".wav", base + ".mp3"


def _write_clip(path: str, x: np.ndarray, sr: int) -> str:
    au.write_wav(path, x, sr)
    return path


def _read_converted(src: str, dst: str, n_in: int, sr_in: int) -> np.ndarray:
    try:
        y, ysr = au.read_mono(dst)
    finally:
        for p in (src, dst):
            with contextlib.suppress(OSError):
                os.remove(p)
    y = au.resample(y, ysr, SR)
    return _fit_length(y, int(round(n_in * SR / sr_in)))


async def _rvc_clip(worker: rvc.RvcWorker, x: np.ndarray, sr: int, workdir: str, tag: str, pth: str, idx: str,
                    index_rate: float, timeout: float = 180.0) -> np.ndarray:
    """One RVC call on an in-memory clip; returns 44.1 kHz audio of exactly the length of the input at 44.1 kHz."""
    src = await aio.run_io(_write_clip, os.path.join(workdir, f"{tag}_in.wav"), x, sr)
    dst = os.path.join(workdir, f"{tag}_out.wav")
    try:
        await worker.convert(src, dst, pth, idx, index_rate, timeout=timeout)
    except BaseException:
        for p in (src, dst):
            with contextlib.suppress(OSError):
                os.remove(p)
        raise
    return await aio.run_io(_read_converted, src, dst, len(x), sr)


# ------------------------------------------------------------------------------------------------ convert

async def convert(source: np.ndarray, workdir: str, worker: rvc.RvcWorker, pth: str, idx: str, index_rate: float,
                  progress: Progress) -> np.ndarray:
    """RVC over the whole source (44.1 kHz mono) in chunks; the result has exactly the source length."""
    chunks = await aio.run_io(split_chunks, source, SR)
    total = len(source)
    out = np.zeros(total, dtype=np.float32)
    done = 0
    for i, (a, b) in enumerate(chunks):
        piece = source[a:b]
        if float(np.max(np.abs(piece))) >= SILENT_PEAK:
            t0 = time.time()
            conv = await _rvc_clip(worker, piece, SR, workdir, f"c{i}", pth, idx, index_rate,
                                   timeout=max(180.0, 12.0 * (b - a) / SR))
            log.info("chunk %d/%d: %.1fs of audio in %.1fs", i + 1, len(chunks), (b - a) / SR, time.time() - t0)
            out[a:b] = _fade(conv) if len(chunks) > 1 else conv
        else:
            out[a:b] = piece
        done += b - a
        progress("convert", 5 + 90.0 * done / total, segment=i, segments=len(chunks))
    return out


# ------------------------------------------------------------------------------------------------ respeak

def _trim_clip(y: np.ndarray, sr: int) -> np.ndarray:
    """Cut leading/trailing silence of a TTS clip (keeps 20 ms before and 60 ms after the speech)."""
    import librosa  # noqa: PLC0415

    if len(y) == 0 or float(np.max(np.abs(y))) < 1e-4:
        return y
    _, (a, b) = librosa.effects.trim(y, top_db=40, frame_length=512, hop_length=128)
    return y[max(a - int(0.02 * sr), 0): min(b + int(0.06 * sr), len(y))]


def _stretch(y: np.ndarray, rate: float) -> np.ndarray:
    import librosa  # noqa: PLC0415

    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32, copy=False)


def _load_trimmed(path: str) -> tuple[np.ndarray, int]:
    y, sr = au.read_mono(path)
    with contextlib.suppress(OSError):
        os.remove(path)
    return _trim_clip(y.astype(np.float32, copy=False), sr), sr


class _Rate:
    """Running estimate of the voice's natural seconds per character (calibrates ``duration_factor``)."""

    def __init__(self) -> None:
        self.seconds = INITIAL_SEC_PER_CHAR * PRIOR_CHARS
        self.chars = float(PRIOR_CHARS)

    def update(self, natural_s: float, chars: int) -> None:
        if chars >= 8 and natural_s > 0:
            self.seconds += natural_s
            self.chars += chars

    @property
    def per_char(self) -> float:
        return self.seconds / self.chars


class _Track:
    """The output track; only the consumer task mixes into it (on the IO pool, one clip at a time)."""

    def __init__(self, total_s: float) -> None:
        self.data = np.zeros(int(total_s * SR) + 1, dtype=np.float32)
        self.end = 0   # furthest sample any clip reaches

    def mix(self, pos: int, y: np.ndarray) -> None:
        end = pos + len(y)
        self.end = max(self.end, end)
        if end > len(self.data):
            self.data = np.concatenate([self.data, np.zeros(end - len(self.data) + SR, dtype=np.float32)])
        self.data[pos:end] += y


async def respeak(segments: list[dict[str, Any]], total_s: float, ref: str, temperature: float, quality: str,
                  worker: rvc.RvcWorker | None, pth: str, idx: str, index_rate: float, workdir: str,
                  progress: Progress, speed: float = 1.0, lang: str = "en") -> tuple[np.ndarray, dict[str, Any]]:
    """Speak every segment and place it at its start on a silent track of ``total_s`` (it grows if the last clip
    overruns). Synthesis runs one segment ahead of a second task that does RVC and mixing (bounded queue).

    ``speed`` (0.8-1.3) is the user's pace: it multiplies the slot-fitting ``duration_factor`` by ``1/speed``; when that
    would no longer fit the slot, the factor is lowered to fit it (the slot always wins, within that range).
    ``lang`` is the language the texts are spoken in (it picks the TTS backend).

    Fitting order for a clip longer than its slot: ``duration_factor`` 0.8-1.25 (x the user's speed), then a librosa
    stretch of at most 1.2x, then the clip overruns and the following segments are pushed later. Speech is never
    cut and two clips never overlap. ``stats`` reports ``drift_s`` (how much longer the output is than the source) and
    ``max_push_s`` (the largest delay of any segment's start)."""
    backend = backends.for_language(lang)
    if backend is None:
        raise EngineError("bad_request", f"no voice model speaks {lang!r}")
    n = len(segments)
    track = _Track(total_s)
    todo: asyncio.Queue[tuple[int, int, np.ndarray, int] | None] = asyncio.Queue(maxsize=2)
    stats: dict[str, Any] = {"stretched": 0, "overrun": 0, "pushed": 0}
    max_push = [0.0]
    rate = _Rate()
    user_factor = tts.duration_factor_for(speed)

    async def produce() -> None:
        cursor = 0.0
        for i, seg in enumerate(segments):
            text = seg["text"]
            start = float(seg["start"])
            next_start = float(segments[i + 1]["start"]) if i + 1 < n else max(total_s, float(seg["end"]))
            pos_s = max(start, cursor)
            if pos_s > start + 0.01:
                stats["pushed"] += 1
                max_push[0] = max(max_push[0], pos_s - start)
            avail = max(next_start - GAP_S - pos_s, MIN_AVAIL_S)
            target = max(float(seg["end"]) - start, 0.4)
            aim = min(target, avail)
            est = rate.per_char * len(text)
            factor = min(FACTOR_MAX, max(FACTOR_MIN, aim / est)) if est > 0 else 1.0
            factor *= user_factor
            if user_factor != 1.0 and est > 0 and est * factor > avail:  # the user's pace does not fit this slot: hurry up
                factor = avail / est
            factor = min(FACTOR_HI, max(FACTOR_LO, factor))
            if abs(factor - 1.0) < 0.04:
                factor = 1.0
            t0 = time.time()
            path = await backend.synthesize(text, lang, ref, SynthParams(
                out_path=os.path.join(workdir, f"s{i}_tts.wav"), temperature=temperature, quality=quality,
                duration_factor=factor))
            clip, sr = await aio.run_io(_load_trimmed, path)
            dur = len(clip) / float(sr)
            rate.update(dur / factor, len(text))
            note = ""
            if dur > avail:
                r = min(dur / avail, MAX_STRETCH)
                clip = await aio.run_io(_stretch, clip, r)
                dur = len(clip) / float(sr)
                stats["stretched"] += 1
                note = f" stretched x{r:.2f}"
                if dur > avail + 1e-3:
                    stats["overrun"] += 1
                    note += " (overruns into the following silence)"
            log.info("segment %d/%d: factor %.2f, %.1fs (slot %.1fs, target %.1fs) in %.1fs%s", i + 1, n, factor,
                     dur, avail, target, time.time() - t0, note)
            cursor = pos_s + dur + CLIP_GAP_S
            await todo.put((i, int(round(pos_s * SR)), clip, sr))
        await todo.put(None)

    async def consume() -> None:
        async for i, pos, clip, sr in aio.queue_items(todo):
            if worker is not None:
                y = await _rvc_clip(worker, clip, sr, workdir, f"s{i}", pth, idx, index_rate)
            else:
                y = await aio.run_io(au.resample, clip, sr, SR)
            await aio.run_io(track.mix, pos, _fade(y))
            progress("speak", 5 + 90.0 * (i + 1) / n, segment=i, segments=n)

    await aio.gather_or_cancel(produce(), consume())
    out = track.data[:max(int(total_s * SR), track.end)]  # no padding: the output ends with the source or the last clip
    stats["drift_s"] = round(max(len(out) / float(SR) - total_s, 0.0), 2)
    stats["max_push_s"] = round(max_push[0], 2)
    return out, stats


def cleanup(workdir: str) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
