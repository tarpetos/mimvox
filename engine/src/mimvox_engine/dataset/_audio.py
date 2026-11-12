"""Small audio helpers for the dataset package (numpy / soundfile / scipy only)."""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import soundfile as sf

from ..registry import EngineError

FRAME_S = 0.025
HOP_S = 0.010
DB_FLOOR = -100.0


def read_audio(path: str) -> tuple[np.ndarray, int]:
    """Read any supported audio (or the audio track of a video) as mono float32 and its sample rate.

    Goes through the shared decoder (``media.read_audio``): libsndfile for wav/flac/aiff, ffmpeg for the rest.
    """
    from .. import media  # noqa: PLC0415

    if not isinstance(path, str) or not path:
        raise EngineError("bad_request", "Missing audio file path.")
    if not os.path.isfile(path):
        raise EngineError("not_found", f"Audio file not found: {path}")
    return media.read_audio(path)


def write_wav(path: str, y: np.ndarray, sr: int) -> None:
    """Atomically write a mono 16-bit PCM WAV."""
    tmp = f"{path}.tmp-{os.getpid()}.wav"
    try:
        sf.write(tmp, np.clip(y, -1.0, 1.0), sr, subtype="PCM_16", format="WAV")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def resample(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return y
    from scipy.signal import resample_poly

    g = math.gcd(sr, target)
    return resample_poly(y, target // g, sr // g).astype(np.float32)


def to_db(x: float | np.ndarray) -> Any:
    return np.maximum(20.0 * np.log10(np.maximum(x, 1e-12)), DB_FLOOR)


FRAME_BLOCK = 8192  # frames per block of frame_db: a float64 running sum of ~80 s of audio, not of the whole file


def frame_db(y: np.ndarray, sr: int) -> np.ndarray:
    """Frame RMS level in dBFS (25 ms frames, 10 ms hop). Computed block by block, so a long file (a 20-minute media
    pool) needs a few tens of MB of scratch memory instead of two float64 copies of itself."""
    n = max(int(sr * FRAME_S), 1)
    hop = max(int(sr * HOP_S), 1)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    starts = np.arange(0, len(y) - n + 1, hop)
    energy = np.empty(len(starts), dtype=np.float64)
    for i in range(0, len(starts), FRAME_BLOCK):
        st = starts[i:i + FRAME_BLOCK]
        a = int(st[0])
        seg = y[a:int(st[-1]) + n].astype(np.float64)
        c = np.concatenate([[0.0], np.cumsum(seg * seg)])
        energy[i:i + len(st)] = (c[st - a + n] - c[st - a]) / n
    return to_db(np.sqrt(np.maximum(energy, 0.0)))
