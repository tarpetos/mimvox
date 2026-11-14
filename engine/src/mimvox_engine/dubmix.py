"""Mixing a new voice over the background of the original (Redub's video output).

Arrays are float32 ``(channels, n)``; a mono voice is ``(n,)``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

HOP_S = 0.1              # four frames make a 400 ms BS.1770 block
ABS_GATE = -70.0
REL_GATE = -10.0
ACTIVE_BELOW_TOP_DB = 30.0
ACTIVE_FLOOR_DB = -60.0
DUCK_MIN_DB = 3.0
DUCK_MAX_DB = 10.0
DUCK_ATTACK_S, DUCK_RELEASE_S = 0.1, 0.4
GAIN_LIMIT_DB = 24.0
CEILING = 10 ** (-1.0 / 20)
LIMIT_RAMP_S = 0.05


def as2d(x: np.ndarray) -> np.ndarray:
    return x.reshape(1, -1) if x.ndim == 1 else x


def _biquads(sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The two BS.1770 K-weighting stages (high shelf, then the RLB high-pass) as (b, a)."""
    out = []
    for kind, gain, q, fc in (("shelf", 4.0, 1 / math.sqrt(2), 1500.0), ("hp", 0.0, 0.5, 38.0)):
        a_ = 10 ** (gain / 40.0)
        w0 = 2 * math.pi * fc / sr
        alpha = math.sin(w0) / (2 * q)
        c = math.cos(w0)
        if kind == "shelf":
            b = [a_ * ((a_ + 1) + (a_ - 1) * c + 2 * math.sqrt(a_) * alpha), -2 * a_ * ((a_ - 1) + (a_ + 1) * c),
                 a_ * ((a_ + 1) + (a_ - 1) * c - 2 * math.sqrt(a_) * alpha)]
            a = [(a_ + 1) - (a_ - 1) * c + 2 * math.sqrt(a_) * alpha, 2 * ((a_ - 1) - (a_ + 1) * c),
                 (a_ + 1) - (a_ - 1) * c - 2 * math.sqrt(a_) * alpha]
        else:
            b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
            a = [1 + alpha, -2 * c, 1 - alpha]
        out.append((np.array(b) / a[0], np.array(a) / a[0]))
    return out


def frame_power(x: np.ndarray, sr: int, weighted: bool = True) -> np.ndarray:
    """Mean square per ``HOP_S`` frame, summed over the channels: shape (frames,).
    Filtered in one-minute pieces, so a long track does not need a float64 copy of itself."""
    from scipy.signal import lfilter, lfilter_zi  # noqa: PLC0415

    x = as2d(x)
    hop = int(round(HOP_S * sr))
    n = x.shape[1]
    frames = max(n // hop, 0)
    out = np.zeros(frames, dtype=np.float64)
    if frames == 0:
        return out
    step = hop * 600
    filters = _biquads(sr) if weighted else []
    for ch in range(x.shape[0]):
        states = [lfilter_zi(b, a) * 0.0 for b, a in filters]
        for a0 in range(0, frames * hop, step):
            y = np.asarray(x[ch, a0:min(a0 + step, frames * hop)], dtype=np.float64)
            for k, (b, a) in enumerate(filters):
                y, states[k] = lfilter(b, a, y, zi=states[k])
            f0 = a0 // hop
            out[f0:f0 + len(y) // hop] += (y.reshape(-1, hop) ** 2).mean(axis=1)
    return out


def _db(p: np.ndarray | float) -> Any:
    return 10.0 * np.log10(np.maximum(p, 1e-12))


def lufs_from_frames(power: np.ndarray, mask: np.ndarray | None = None) -> float:
    """BS.1770 gated loudness of ``frame_power`` output. With ``mask`` only blocks whose frames are all in it count;
    ``-inf`` when nothing passes the gates."""
    if len(power) < 4:
        return float("-inf")
    blocks = (power[:-3] + power[1:-2] + power[2:-1] + power[3:]) / 4.0
    if mask is not None:
        m = mask[:-3] & mask[1:-2] & mask[2:-1] & mask[3:]
        blocks = blocks[m]
    loud = -0.691 + _db(blocks)
    gated = blocks[loud > ABS_GATE]
    if len(gated) == 0:
        return float("-inf")
    rel = -0.691 + float(_db(gated.mean())) + REL_GATE
    final = gated[(-0.691 + _db(gated)) > rel]
    return float(-0.691 + _db(final.mean())) if len(final) else float("-inf")


def lufs(x: np.ndarray, sr: int) -> float:
    return lufs_from_frames(frame_power(x, sr))


def active_frames(power: np.ndarray) -> np.ndarray:
    if len(power) == 0:
        return np.zeros(0, dtype=bool)
    db = _db(power)
    top = float(np.percentile(db, 95))
    act = db > max(top - ACTIVE_BELOW_TOP_DB, ACTIVE_FLOOR_DB)
    gap = ~act
    bridged = gap.copy()
    bridged[1:-1] = gap[1:-1] & (gap[:-2] | gap[2:])   # a lone silent frame counts as active
    return ~bridged


def _dilate(mask: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 0 or not mask.any():
        return mask
    from scipy.ndimage import maximum_filter1d  # noqa: PLC0415
    return maximum_filter1d(mask.astype(np.uint8), size=2 * frames + 1) > 0


def duck_depth(bg_power: np.ndarray, orig_active: np.ndarray) -> tuple[float, float]:
    """(ducking applied, difference measured) in dB of the original background under its voice."""
    n = min(len(bg_power), len(orig_active))
    bg, act = bg_power[:n], orig_active[:n]
    pause = ~_dilate(act, 3)
    if act.sum() < 20 or pause.sum() < 20:
        return 0.0, 0.0
    db = _db(bg)
    diff = float(np.median(db[pause]) - np.median(db[act]))
    return (min(diff, DUCK_MAX_DB) if diff >= DUCK_MIN_DB else 0.0), diff


def _duck_gain(new_active: np.ndarray, orig_active: np.ndarray, depth_db: float, n: int, sr: int) -> np.ndarray | None:
    if depth_db <= 0:
        return None
    frames = (n + int(HOP_S * sr) - 1) // int(HOP_S * sr)
    want = np.zeros(frames, dtype=bool)
    k = min(len(new_active), frames)
    want[:k] = new_active[:k]
    k2 = min(len(orig_active), frames)
    orig = np.zeros(frames, dtype=bool)
    orig[:k2] = orig_active[:k2]
    want &= ~orig   # where the original voice spoke, its ducking is already in the stem
    if not want.any():
        return None
    from scipy.ndimage import uniform_filter1d  # noqa: PLC0415
    held = want.copy()
    for k in range(1, max(int(round(DUCK_ATTACK_S / HOP_S)), 1) + 1):
        held[:-k] |= want[k:]
    for k in range(1, max(int(round(DUCK_RELEASE_S / HOP_S)), 1) + 1):
        held[k:] |= want[:-k]
    smooth = np.clip(uniform_filter1d(held.astype(np.float64), size=3), 0.0, 1.0)
    db = -depth_db * smooth
    centers = (np.arange(frames) + 0.5) * HOP_S * sr
    g = np.interp(np.arange(n), centers, db)
    return (10.0 ** (g / 20.0)).astype(np.float32)


def limit(x: np.ndarray, sr: int, ceiling: float = CEILING) -> tuple[np.ndarray, float]:
    """Look-ahead peak limiter, one gain for all channels. Returns (x, the largest reduction in dB)."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= ceiling:
        return x, 0.0
    from scipy.ndimage import minimum_filter1d, uniform_filter1d  # noqa: PLC0415
    need = np.minimum(1.0, ceiling / np.maximum(np.max(np.abs(x), axis=0), 1e-9)).astype(np.float32)
    r = max(int(LIMIT_RAMP_S * sr), 1)
    # A minimum over 2r then a mean over r never exceeds the gain a sample needs, and ramps instead of jumping.
    g = uniform_filter1d(minimum_filter1d(need, size=2 * r + 1), size=r + 1)
    g = np.minimum(g, need)   # float rounding
    out = (x * g).astype(np.float32, copy=False)
    np.clip(out, -ceiling, ceiling, out=out)
    return out, round(float(-20.0 * math.log10(max(float(g.min()), 1e-9))), 2)


def mix(voice: np.ndarray, sr: int, background: np.ndarray | None, vocals: np.ndarray | None,
        original: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """The voice at the loudness of the original ``vocals`` stem (else of the ``original`` mixture) over
    ``background``. Returns the mix ``(channels, max(n, m))`` and a report of the gains in dB."""
    channels = background.shape[0] if background is not None else 1
    voice = np.asarray(voice, dtype=np.float32).reshape(-1)
    v2 = np.broadcast_to(voice, (channels, len(voice)))
    v_pow = frame_power(v2, sr)
    v_lufs = lufs_from_frames(v_pow)
    report: dict[str, Any] = {"voice_lufs": _round(v_lufs)}
    target = float("-inf")
    o_pow = None
    if vocals is not None:
        voc = as2d(vocals)
        if voc.shape[0] != channels:
            voc = np.broadcast_to(np.asarray(voc, dtype=np.float32).mean(axis=0), (channels, voc.shape[1]))
        o_pow = frame_power(voc, sr)
        target = lufs_from_frames(o_pow)
    elif original is not None:
        orig = as2d(original)
        if orig.shape[0] != channels:
            orig = np.broadcast_to(np.asarray(orig, dtype=np.float32).mean(axis=0), (channels, orig.shape[1]))
        target = lufs(orig, sr)
    gain_db = 0.0
    if math.isfinite(target) and math.isfinite(v_lufs):
        gain_db = max(-GAIN_LIMIT_DB, min(GAIN_LIMIT_DB, target - v_lufs))
    report.update(target_lufs=_round(target), voice_gain_db=round(gain_db, 2))
    g = np.float32(10 ** (gain_db / 20.0))
    n = max(len(voice), background.shape[1] if background is not None else 0)
    out = np.zeros((channels, n), dtype=np.float32)
    duck, measured = 0.0, 0.0
    if background is not None:
        m = background.shape[1]
        bg = np.asarray(background, dtype=np.float32)
        b_pow = frame_power(bg, sr)
        report["background_lufs"] = _round(lufs_from_frames(b_pow))
        if o_pow is not None:
            orig_act = active_frames(o_pow)
            duck, measured = duck_depth(b_pow, orig_act)
            gain = _duck_gain(active_frames(v_pow), orig_act, duck, m, sr)
            out[:, :m] = bg if gain is None else bg * gain
        else:
            out[:, :m] = bg
    out[:, :len(voice)] += v2 * g
    report.update(duck_db=round(duck, 2), duck_measured_db=round(measured, 2))
    out, reduced = limit(out, sr)
    report["limited_db"] = reduced
    return out, report


def _round(v: float) -> float | None:
    return round(v, 2) if math.isfinite(v) else None
