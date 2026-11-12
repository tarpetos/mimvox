"""Recording quality analysis: level, clipping, noise floor, SNR and silence, with friendly advice.

Calibration notes (measured on good takes, recorded quietly with a peak of about 0.1-0.15):
quiet-but-clean audio is fine because the dataset step normalises loudness. "Too quiet" therefore only fires when
the peak is so low that normalising would mostly amplify noise and 16-bit quantisation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..registry import EngineError, method
from ._audio import HOP_S, frame_db, read_audio, to_db

# Thresholds (dBFS unless noted)
MIN_DURATION_BAD_S = 3.0
MIN_DURATION_WARN_S = 10.0
PEAK_WARN = 0.03          # about -30 dBFS
PEAK_BAD = 0.006          # about -44 dBFS
CLIP_WARN_PCT = 0.05
CLIP_BAD_PCT = 0.5
SNR_WARN_DB = 25.0
SNR_BAD_DB = 15.0
NOISE_FLOOR_WARN_DB = -45.0
SILENCE_WARN_PCT = 50.0
SILENCE_BAD_PCT = 75.0
MIN_SAMPLE_RATE = 16000

_SEVERITY = {"ok": 0, "warn": 1, "bad": 2}


def _sum_squares(y: np.ndarray, block: int = 1 << 20) -> float:
    """Sum of squares in float64, a block at a time (no float64 copy of a long file)."""
    total = 0.0
    for i in range(0, len(y), block):
        seg = y[i:i + block].astype(np.float64)
        total += float(np.dot(seg, seg))
    return total


def measure(y: np.ndarray, sr: int) -> dict[str, Any]:
    """Raw metrics plus the frame-level arrays other modules reuse (keys starting with ``_``)."""
    n = len(y)
    peak = max(float(np.max(y)), -float(np.min(y))) if n else 0.0
    if peak >= 0.95:
        clipping_pct = float(100.0 * np.count_nonzero(np.abs(y) >= peak * 0.999) / n)
    else:
        clipping_pct = 0.0
    rms = float(np.sqrt(_sum_squares(y) / n)) if n else 0.0
    fdb = frame_db(y, sr)
    live = fdb[fdb > -90.0]  # ignore digitally silent frames (noise gates / zero padding)
    if live.size < 10:
        noise_floor, speech = -90.0, -90.0
    else:
        noise_floor = float(np.percentile(live, 10))
        speech = float(np.percentile(live, 90))
    snr = speech - noise_floor
    silence_thr = max(noise_floor + 6.0, speech - 30.0)
    silence_pct = float(100.0 * np.mean(fdb < silence_thr)) if fdb.size else 100.0
    return {
        "duration_s": n / sr if sr else 0.0,
        "sample_rate": sr,
        "peak": peak,
        "clipping_pct": clipping_pct,
        "rms_db": float(to_db(rms)),
        "noise_floor_db": noise_floor,
        "speech_db": speech,
        "snr_db": snr,
        "silence_pct": silence_pct,
        "_frame_db": fdb,
        "_silence_thr": silence_thr,
    }


def assess(m: dict[str, Any], expected_s: float | None = None) -> tuple[str, list[str]]:
    """Turn metrics into a verdict (ok / warn / bad) and friendly English issues."""
    found: list[tuple[str, str]] = []
    dur = m["duration_s"]
    if dur < MIN_DURATION_BAD_S:
        found.append(("bad", "Too short: this take is only a few seconds, please read the whole text"))
    elif dur < MIN_DURATION_WARN_S:
        found.append(("warn", "Very short take: read the whole text for the best result"))
    elif expected_s and dur < 0.6 * expected_s:
        found.append(("warn", "Shorter than expected: did you read the whole text?"))
    if m["sample_rate"] < MIN_SAMPLE_RATE:
        found.append(("warn", "Low sample rate: record at 44.1 kHz or 48 kHz if you can"))

    if m["speech_db"] <= -90.0 or m["snr_db"] < 6.0:
        found.append(("bad", "No clear speech found: check that the right microphone is selected and not muted"))
    else:
        if m["peak"] < PEAK_BAD:
            found.append(("bad", "Too quiet: move closer to the microphone or raise the input volume"))
        elif m["peak"] < PEAK_WARN:
            found.append(("warn", "Very quiet: move closer to the microphone or raise the input volume"))
        if m["clipping_pct"] > CLIP_BAD_PCT:
            found.append(("bad", "Clipping: the sound is distorted, move back from the microphone or lower the input volume"))
        elif m["clipping_pct"] > CLIP_WARN_PCT:
            found.append(("warn", "Clipping: some peaks are distorted, move back a little or lower the input volume"))
        if m["snr_db"] < SNR_BAD_DB:
            found.append(("bad", "Background noise: the room is too noisy, find a quieter place or switch off fans and music"))
        elif m["snr_db"] < SNR_WARN_DB or m["noise_floor_db"] > NOISE_FLOOR_WARN_DB:
            found.append(("warn", "Background noise: a quieter room or a closer microphone would help"))
        if m["silence_pct"] > SILENCE_BAD_PCT:
            found.append(("bad", "Mostly silence: speak continuously and avoid long pauses"))
        elif m["silence_pct"] > SILENCE_WARN_PCT:
            found.append(("warn", "A lot of silence: try to keep a steady reading pace"))

    verdict = "ok"
    for sev, _ in found:
        if _SEVERITY[sev] > _SEVERITY[verdict]:
            verdict = sev
    return verdict, [msg for _, msg in found]


def public(m: dict[str, Any], verdict: str, issues: list[str]) -> dict[str, Any]:
    return {
        "duration_s": round(m["duration_s"], 2),
        "peak": round(m["peak"], 4),
        "clipping_pct": round(m["clipping_pct"], 3),
        "rms_db": round(m["rms_db"], 1),
        "noise_floor_db": round(m["noise_floor_db"], 1),
        "snr_db": round(m["snr_db"], 1),
        "silence_pct": round(m["silence_pct"], 1),
        "verdict": verdict,
        "issues": issues,
    }


def check_array(y: np.ndarray, sr: int, expected_s: float | None = None) -> dict[str, Any]:
    m = measure(y, sr)
    verdict, issues = assess(m, expected_s)
    return public(m, verdict, issues)


def check_file(path: str, expected_s: float | None = None) -> dict[str, Any]:
    y, sr = read_audio(path)
    if len(y) == 0:
        raise EngineError("bad_request", "The audio file is empty.")
    return check_array(y, sr, expected_s)


@method("recording.check")
def recording_check(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    expected = params.get("expected_s")
    return check_file(params.get("path"), float(expected) if isinstance(expected, (int, float)) else None)


__all__ = ["measure", "assess", "check_array", "check_file", "HOP_S"]
