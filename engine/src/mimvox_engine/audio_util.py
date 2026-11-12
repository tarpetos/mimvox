"""Shared audio helpers: float32 mono arrays in memory, 16-bit WAV on disk (other formats: ``media.save_result``)."""
from __future__ import annotations

import math
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

OUT_SR = 44100


def resample(x: np.ndarray, sr_from: int, sr_to: int = OUT_SR) -> np.ndarray:
    if sr_from == sr_to:
        return x
    g = math.gcd(int(sr_from), int(sr_to))
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32, copy=False)


def int16_to_float(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32).reshape(-1)) / 32768.0


def read_mono(path: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1) if data.shape[1] > 1 else data[:, 0], sr


def silence(ms: float, sr: int = OUT_SR) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000.0), dtype=np.float32)


def concat(parts: list[np.ndarray], gap_ms: float = 0.0, sr: int = OUT_SR) -> np.ndarray:
    """Join clips with ``gap_ms`` of silence between them."""
    if not parts:
        return np.zeros(0, dtype=np.float32)
    if gap_ms <= 0:
        return np.concatenate(parts)
    gap = silence(gap_ms, sr)
    out: list[np.ndarray] = []
    for i, p in enumerate(parts):
        if i:
            out.append(gap)
        out.append(p)
    return np.concatenate(out)


def write_wav(path: str, data: np.ndarray, sr: int = OUT_SR) -> None:
    """16-bit WAV of mono ``(n,)`` or multichannel ``(channels, n)`` samples."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames = data.T if data.ndim == 2 else data
    sf.write(path, np.clip(frames, -1.0, 1.0), sr, subtype="PCM_16", format="WAV")


def duration_s(data: np.ndarray, sr: int = OUT_SR) -> float:
    """Length of mono ``(n,)`` or ``(channels, n)`` samples."""
    return round(data.shape[-1] / float(sr), 3)
