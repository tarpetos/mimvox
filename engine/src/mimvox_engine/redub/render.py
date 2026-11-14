"""Redub rendering: ``convert`` (RVC over the original audio) and ``respeak`` (TTS per transcript segment).

Runs on the engine loop with the numpy and ffmpeg work on the IO pool; cancel by cancelling the awaiting task."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import time
from typing import Any, Awaitable, Callable

import numpy as np

from .. import aio, backends, media, rvc, tts, zsvc
from .. import audio_util as au
from ..backends import SynthParams
from ..registry import EngineError

log = logging.getLogger("mimvox.redub.render")

SR = au.OUT_SR
CHUNK_MIN_S, CHUNK_MAX_S = 15.0, 30.0
MIN_TAIL_S = 2.0
SILENT_PEAK = 1e-4
GAP_S = 0.12
MIN_AVAIL_S = 0.25
CLIP_GAP_S = 0.05
FIT_TOL_S = 0.02
FIT_FLOOR = 0.85                # measured: faster than this OmniVoice drops weak consonants and short words
FACTOR_MAX = 1.25
FACTOR_LO, FACTOR_HI = 0.65, 1.4
NATURAL_PAUSE_S = 0.5
TIMINGS = ("keep", "natural")
MAX_STRETCH = 1.15
INITIAL_SEC_PER_CHAR = 0.068
PRIOR_CHARS = 40


Progress = Callable[..., None]


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
                    index_rate: float, timeout: float = 180.0, pitch: int = 0) -> np.ndarray:
    return await _converted(lambda src, dst: worker.convert(src, dst, pth, idx, index_rate, timeout=timeout,
                                                            pitch=pitch), x, sr, workdir, tag)


async def _zs_clip(zs: zsvc.Bound, x: np.ndarray, sr: int, workdir: str, tag: str) -> np.ndarray:
    return await _converted(zs.convert, x, sr, workdir, tag)


async def _converted(convert: Callable[[str, str], Awaitable[Any]], x: np.ndarray, sr: int, workdir: str,
                     tag: str) -> np.ndarray:
    src = await aio.run_io(_write_clip, os.path.join(workdir, f"{tag}_in.wav"), x, sr)
    dst = os.path.join(workdir, f"{tag}_out.wav")
    try:
        await convert(src, dst)
    except BaseException:
        for p in (src, dst):
            with contextlib.suppress(OSError):
                os.remove(p)
        raise
    return await aio.run_io(_read_converted, src, dst, len(x), sr)


async def convert(source: np.ndarray, workdir: str, worker: rvc.RvcWorker, pth: str, idx: str, index_rate: float,
                  progress: Progress) -> np.ndarray:
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


def _trim_clip(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa  # noqa: PLC0415

    if len(y) == 0 or float(np.max(np.abs(y))) < 1e-4:
        return y
    _, (a, b) = librosa.effects.trim(y, top_db=40, frame_length=512, hop_length=128)
    return y[max(a - int(0.02 * sr), 0): min(b + int(0.06 * sr), len(y))]


def _load_trimmed(path: str) -> tuple[np.ndarray, int]:
    y, sr = au.read_mono(path)
    with contextlib.suppress(OSError):
        os.remove(path)
    return _trim_clip(y.astype(np.float32, copy=False), sr), sr


def segment_factor(est: float, own: float, room: float, user_factor: float) -> float:
    """``est``: the clip's estimated natural length; ``own``: the original segment's length; ``room``: the time the
    clip may take (its slot and the silence after it); ``user_factor``: ``1 / speed``."""
    if est <= 0:
        return natural_factor(user_factor)
    at_pace = est * user_factor
    aim = min(own, room)
    if at_pace <= aim:
        factor = user_factor * min(FACTOR_MAX, aim / at_pace)
    elif at_pace <= room:
        factor = user_factor
    else:
        factor = max(room / est, min(FIT_FLOOR, user_factor))
    factor = min(FACTOR_HI, max(FACTOR_LO, factor))
    return 1.0 if abs(factor - 1.0) < 0.04 else factor


def natural_factor(user_factor: float) -> float:
    factor = min(FACTOR_HI, max(FACTOR_LO, user_factor))
    return 1.0 if abs(factor - 1.0) < 0.04 else factor


def stretch_rate(dur: float, room: float) -> float:
    if dur <= room + FIT_TOL_S:
        return 1.0
    return min(dur / room, MAX_STRETCH)


class _Rate:
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
    """Only the consumer task mixes into it, one clip at a time."""

    def __init__(self, total_s: float) -> None:
        self.data = np.zeros(int(total_s * SR) + 1, dtype=np.float32)
        self.end = 0

    def mix(self, pos: int, y: np.ndarray) -> None:
        end = pos + len(y)
        self.end = max(self.end, end)
        if end > len(self.data):
            self.data = np.concatenate([self.data, np.zeros(end - len(self.data) + SR, dtype=np.float32)])
        self.data[pos:end] += y


async def respeak(segments: list[dict[str, Any]], total_s: float, ref: str, temperature: float, quality: str,
                  worker: rvc.RvcWorker | None, pth: str, idx: str, index_rate: float, workdir: str,
                  progress: Progress, speed: float = 1.0, lang: str = "en", ref_text: str | None = None,
                  pitch: int = 0, zs: zsvc.Bound | None = None, timing: str = "keep") -> tuple[np.ndarray, dict[str, Any]]:
    """Speech is never cut and two clips never overlap: a clip that does not fit pushes the following ones later, and
    the track grows past ``total_s`` when it must."""
    backend = backends.for_language(lang)
    if backend is None:
        raise EngineError("bad_request", f"no voice model speaks {lang!r}")
    if timing not in TIMINGS:
        raise EngineError("bad_request", f"'timing' must be one of {', '.join(TIMINGS)}")
    keep = timing == "keep"
    n = len(segments)
    track = _Track(total_s)
    todo: asyncio.Queue[tuple[int, int, np.ndarray, int] | None] = asyncio.Queue(maxsize=2)
    stats: dict[str, Any] = {"stretched": 0, "overrun": 0, "pushed": 0}
    max_push = [0.0]
    rate = _Rate()
    user_factor = tts.duration_factor_for(speed)
    no_stretch = [False]

    async def produce() -> None:
        last_end = None
        for i, seg in enumerate(segments):
            text = seg["text"]
            start = float(seg["start"])
            if last_end is None:
                pos_s = start
            elif keep:
                pos_s = max(start, last_end + CLIP_GAP_S)
            else:
                pause = min(max(start - float(segments[i - 1]["end"]), CLIP_GAP_S), NATURAL_PAUSE_S)
                pos_s = max(start, last_end + pause)
            if pos_s > start + 0.01:
                stats["pushed"] += 1
                max_push[0] = max(max_push[0], pos_s - start)
            own = max(float(seg["end"]) - start, 0.4)
            if keep:
                next_start = float(segments[i + 1]["start"]) if i + 1 < n else max(total_s, float(seg["end"]))
                room = max(next_start - GAP_S - pos_s, MIN_AVAIL_S)
                factor = segment_factor(rate.per_char * len(text), own, room, user_factor)
            else:
                room = float("inf")
                factor = natural_factor(user_factor)
            t0 = time.time()
            async with zsvc.gpu_turn(zs.turn if zs is not None else None):  # a tight GPU: one model at a time
                path = await backend.synthesize(text, lang, ref, SynthParams(
                    out_path=os.path.join(workdir, f"s{i}_tts.wav"), temperature=temperature, quality=quality,
                    duration_factor=factor, reference_text=ref_text))
            clip, sr = await aio.run_io(_load_trimmed, path)
            dur = len(clip) / float(sr)
            rate.update(dur / factor, len(text))
            note = ""
            r = stretch_rate(dur, room)
            if r > 1.0:
                done = None if no_stretch[0] else await aio.run_io(media.time_stretch, clip, sr, r)
                if done is None:
                    if not no_stretch[0]:
                        log.warning("no ffmpeg time-stretch filter (rubberband / atempo): long clips overrun instead")
                    no_stretch[0] = True
                    note = " not stretched"
                else:
                    clip, how = done
                    dur = len(clip) / float(sr)
                    stats["stretched"] += 1
                    note = f" stretched x{r:.2f} ({how})"
            if not keep:
                note += f" (natural pace, at {pos_s:.1f}s)" if pos_s > start + 0.01 else " (natural pace)"
            elif dur > room + FIT_TOL_S:
                stats["overrun"] += 1
                note += f" (overruns by {dur - room:.2f}s, the next segments move later)"
            elif dur > own + 0.05:
                note += f" (runs {min(dur, room) - own:.1f}s into the following silence)"
            log.info("segment %d/%d: factor %.2f, %.1fs (segment %.1fs, room %.1fs) in %.1fs%s", i + 1, n, factor,
                     dur, own, min(room, 999.0), time.time() - t0, note)
            last_end = pos_s + dur
            await todo.put((i, int(round(pos_s * SR)), clip, sr))
        await todo.put(None)

    async def consume() -> None:
        async for i, pos, clip, sr in aio.queue_items(todo):
            if zs is not None:
                t0 = time.time()
                y = await _zs_clip(zs, clip, sr, workdir, f"s{i}")
                log.info("segment %d/%d: converted (zero-shot, %d steps) in %.1fs", i + 1, n, zs.steps, time.time() - t0)
            elif worker is not None:
                y = await _rvc_clip(worker, clip, sr, workdir, f"s{i}", pth, idx, index_rate, pitch=pitch)
            else:
                y = await aio.run_io(au.resample, clip, sr, SR)
            await aio.run_io(track.mix, pos, _fade(y))
            progress("speak", 5 + 90.0 * (i + 1) / n, segment=i, segments=n)

    await aio.gather_or_cancel(produce(), consume())
    out = track.data[:max(int(total_s * SR), track.end)]
    stats["drift_s"] = round(max(len(out) / float(SR) - total_s, 0.0), 2)
    stats["timing"] = timing
    stats["max_push_s"] = round(max_push[0], 2)
    return out, stats


def cleanup(workdir: str) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
