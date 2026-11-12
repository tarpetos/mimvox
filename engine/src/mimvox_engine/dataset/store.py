"""Per-voice dataset store: ``voices/<id>/dataset/part<N>.wav`` plus the best-clip ``reference.wav``.

Stored takes are mono 16-bit PCM at the source rate (resampled to 44.1 kHz only above 48 kHz), with leading/trailing
silence trimmed (0.25 s kept) and loudness normalised (active-speech RMS about -20 dBFS, peak at most 0.9). Per-part
quality results live in ``dataset/meta.json`` so ``dataset.summary`` is cheap.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

import numpy as np
import soundfile as sf

from ..registry import EngineError, method
from . import pace, quality
from ._audio import HOP_S, read_audio, resample, write_wav
from .levels import level_for, message_for, recommended_epochs

TARGET_ACTIVE_RMS_DB = -20.0
MAX_PEAK = 0.9
MAX_GAIN_DB = 40.0
TRIM_PAD_S = 0.25
TRIM_MIN_FRAMES = 5            # speech must last 50 ms to count as the start/end (ignores clicks)
MAX_RATE = 48000
MEDIA_POOL = "reference_media.wav"   # voices.reference_from_media: the combined media files (not a dataset part)

_LOCK = threading.RLock()
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_PART_RE = re.compile(r"^part(\d+)\.wav$")


# ---------------------------------------------------------------------------------------------- paths and meta

def _voice_dir(ctx: Any, params: dict[str, Any]) -> str:
    vid = params.get("voice_id")
    if not isinstance(vid, str) or not _ID_RE.match(vid):
        raise EngineError("bad_request", "Missing or invalid 'voice_id'.")
    vdir = os.path.join(ctx.data_dir, "voices", vid)
    if not os.path.isdir(vdir):
        raise EngineError("not_found", f"Voice not found: {vid}")
    return vdir


def _part_index(params: dict[str, Any]) -> int:
    n = params.get("part_index")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise EngineError("bad_request", "'part_index' must be an integer starting at 1.")
    return n


def _ddir(vdir: str) -> str:
    return os.path.join(vdir, "dataset")


def _part_path(vdir: str, n: int) -> str:
    return os.path.join(_ddir(vdir), f"part{n}.wav")


def _meta_path(vdir: str) -> str:
    return os.path.join(_ddir(vdir), "meta.json")


def _load_meta(vdir: str) -> dict[str, Any]:
    try:
        with open(_meta_path(vdir), encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta, dict) and isinstance(meta.get("parts"), dict):
            return meta
    except (OSError, ValueError):
        pass
    return {"version": 1, "parts": {}}


def _save_meta(vdir: str, meta: dict[str, Any]) -> None:
    os.makedirs(_ddir(vdir), exist_ok=True)
    tmp = _meta_path(vdir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, _meta_path(vdir))


def _existing_parts(vdir: str) -> list[int]:
    try:
        names = os.listdir(_ddir(vdir))
    except OSError:
        return []
    return sorted(int(m.group(1)) for n in names if (m := _PART_RE.match(n)))


# ---------------------------------------------------------------------------------------------- processing

def trim_silence(y: np.ndarray, sr: int, m: dict[str, Any]) -> np.ndarray:
    """Cut leading/trailing silence, keeping ``TRIM_PAD_S`` around the speech."""
    if m["speech_db"] <= -90.0:
        return y
    fdb = m["_frame_db"]
    thr = max(m["noise_floor_db"] + 10.0, m["speech_db"] - 35.0)
    above = fdb >= thr
    k = TRIM_MIN_FRAMES
    if len(above) < k:
        return y
    run = np.convolve(above.astype(np.int8), np.ones(k, dtype=np.int8), mode="valid") == k
    idx = np.flatnonzero(run)
    if idx.size == 0:
        return y
    hop = max(int(sr * HOP_S), 1)
    start = max(int(idx[0] * hop - TRIM_PAD_S * sr), 0)
    end = min(int((idx[-1] + k) * hop + int(sr * 0.025) + TRIM_PAD_S * sr), len(y))
    return y[start:end] if end - start > sr * 0.3 else y


def normalise(y: np.ndarray, m: dict[str, Any]) -> np.ndarray:
    """Gain to about -20 dBFS active-speech RMS without exceeding peak 0.9 (or +40 dB)."""
    fdb = m["_frame_db"]
    act = fdb >= m["_silence_thr"]
    if m["speech_db"] <= -90.0 or not act.any():
        return y
    active_rms_db = float(10.0 * np.log10(np.mean(10.0 ** (fdb[act] / 10.0))))
    gain_db = min(TARGET_ACTIVE_RMS_DB - active_rms_db, MAX_GAIN_DB)
    gain = 10.0 ** (gain_db / 20.0)
    peak = float(np.max(np.abs(y))) or 1e-9
    gain = min(gain, MAX_PEAK / peak)
    return (y * gain).astype(np.float32)


def _fade(y: np.ndarray, sr: int, sec: float = 0.01) -> np.ndarray:
    k = min(int(sr * sec), len(y) // 2)
    if k > 1:
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
        y = y.copy()
        y[:k] *= ramp
        y[-k:] *= ramp[::-1]
    return y


def _titles(vdir: str) -> dict[int, str]:
    """Part number -> reading-text title, when the voice was made with a known preset (else empty)."""
    try:
        with open(os.path.join(vdir, "voice.json"), encoding="utf-8") as f:
            vmeta = json.load(f)
        preset_id = vmeta.get("preset")
        if not isinstance(preset_id, str):
            return {}
        from .presets import get_preset  # noqa: PLC0415

        language = vmeta.get("language")
        preset = get_preset(preset_id, language if isinstance(language, str) else None)
        return {p["index"]: p["title"] for p in preset["parts"]} if preset else {}
    except Exception:  # noqa: BLE001 - titles are only a hint for the reference pick
        return {}


def _usable(vdir: str, meta: dict[str, Any]) -> set[int]:
    """Stored parts that may supply the reference: not rated "bad" (all of them when every part is)."""
    have = {n for n in _existing_parts(vdir) if str(n) in meta["parts"]}
    ok = {n for n in have if meta["parts"][str(n)].get("verdict") != "bad"}
    return ok or have


def _ensure_analysed(vdir: str, meta: dict[str, Any]) -> bool:
    """Fill rate / candidate windows for parts stored before the pace analysis existed. True when meta changed."""
    changed = False
    for n in _existing_parts(vdir):
        info = meta["parts"].get(str(n))
        if info is not None and "cands" not in info:
            y, sr = read_audio(_part_path(vdir, n))
            info.update(pace.analyse_part(y, sr))
            info.pop("ref", None)
            changed = True
    return changed


def rank_candidates(vdir: str, meta: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    """Every stored candidate window of the voice, best first (see ``pace.rank``); ``meta`` must be analysed."""
    return pace.rank(meta["parts"], _titles(vdir), _usable(vdir, meta))


def write_reference(vdir: str, meta: dict[str, Any], part: int, start_s: float, end_s: float, **extra: Any) -> None:
    """Cut ``[start_s, end_s]`` of part ``part`` into ``reference.wav`` and record it in ``meta['reference']``."""
    y, sr = read_audio(_part_path(vdir, part))
    clip = _fade(y[int(start_s * sr): int(end_s * sr)], sr)
    write_wav(os.path.join(vdir, "reference.wav"), clip, sr)
    meta["reference"] = {"part": part, "start_s": round(start_s, 3), "end_s": round(end_s, 3),
                         "duration_s": round(len(clip) / sr, 2), **extra}


def cut_media(vdir: str, start_s: float, end_s: float) -> tuple[np.ndarray, int]:
    """``[start_s, end_s]`` of the media pool (``reference_media.wav``), faded at both ends; read without loading the
    whole pool. ``not_found`` when there is no pool."""
    path = os.path.join(vdir, MEDIA_POOL)
    try:
        sr = int(sf.info(path).samplerate)
        y, _ = sf.read(path, start=int(start_s * sr), stop=int(end_s * sr), dtype="float32", always_2d=False)
    except (OSError, RuntimeError):
        raise EngineError("not_found", "this voice has no combined media (reference_media.wav)") from None
    if y.ndim > 1:
        y = y.mean(axis=1)
    return _fade(np.ascontiguousarray(y, dtype=np.float32), sr), sr


def _keep_manual(vdir: str, meta: dict[str, Any], cur: dict[str, Any], changed_part: int | None) -> bool:
    """Whether a manual reference choice still stands (``reference.wav`` is rewritten from its source if it went
    missing). A clip from the media pool does not depend on the parts: it stays until another clip is chosen."""
    ref_path = os.path.join(vdir, "reference.wav")
    if cur.get("source") == "media":
        if os.path.isfile(ref_path):
            return True
        pool = meta.get("media") if isinstance(meta.get("media"), dict) else {}
        if not cur.get("pool") or cur.get("pool") != pool.get("pool"):
            return False  # cut from a pool that has been replaced or cleared since
        try:
            clip, sr = cut_media(vdir, float(cur["start_s"]), float(cur["end_s"]))
        except (EngineError, KeyError, TypeError, ValueError):
            return False
        write_wav(ref_path, clip, sr)
        return True
    part = cur.get("part")
    if isinstance(part, int) and os.path.isfile(_part_path(vdir, part)) and part != changed_part:
        if not os.path.isfile(ref_path):
            write_reference(vdir, meta, part, cur["start_s"], cur["end_s"], manual=True)
        return True
    return False


def _refresh_reference(vdir: str, meta: dict[str, Any], changed_part: int | None = None) -> None:
    """Choose ``reference.wav`` (best window by SNR, pace near the voice's median, few long pauses). A manual choice
    (``voices.set_reference``) is kept until its part is removed or re-recorded (``changed_part``); one from the media
    pool (``voices.reference_from_media``) until another clip is chosen."""
    ref_path = os.path.join(vdir, "reference.wav")
    cur = meta.get("reference")
    if isinstance(cur, dict) and cur.get("manual"):
        if _keep_manual(vdir, meta, cur, changed_part):
            return
        meta.pop("reference", None)
        cur = None
    _ensure_analysed(vdir, meta)
    ranked, _ = rank_candidates(vdir, meta)
    if not ranked:
        meta.pop("reference", None)
        if os.path.isfile(ref_path):
            os.remove(ref_path)
        return
    best = ranked[0]
    same = (isinstance(cur, dict) and cur.get("part") == best["part"] and cur.get("start_s") == round(best["start_s"], 3)
            and cur.get("end_s") == round(best["end_s"], 3) and best["part"] != changed_part)
    if same and os.path.isfile(ref_path):
        return  # unchanged: rewriting would only force the speech model to condition on the speaker again
    write_reference(vdir, meta, best["part"], best["start_s"], best["end_s"])


def _analyse_stored(vdir: str, n: int) -> dict[str, Any]:
    """Meta entry for a part file that has none (e.g. copied in by hand): analyse the stored audio."""
    y, sr = read_audio(_part_path(vdir, n))
    m = quality.measure(y, sr)
    verdict, issues = quality.assess(m)
    return {"verdict": verdict, "issues": issues, "snr_db": round(m["snr_db"], 1),
            "clipping_pct": round(m["clipping_pct"], 3), **pace.analyse_part(y, sr)}


# ---------------------------------------------------------------------------------------------- summary

def _summary(vdir: str, meta: dict[str, Any]) -> dict[str, Any]:
    changed = False
    parts = []
    for n in _existing_parts(vdir):
        info = meta["parts"].get(str(n))
        if info is None:
            info = meta["parts"][str(n)] = _analyse_stored(vdir, n)
            changed = True
        try:
            dur = sf.info(_part_path(vdir, n)).duration
        except Exception:  # noqa: BLE001
            dur = 0.0
        parts.append({"index": n, "duration_s": round(dur, 2), "verdict": info.get("verdict", "ok"),
                      "issues": info.get("issues", [])})
    stale = [k for k in meta["parts"] if int(k) not in {p["index"] for p in parts}]
    for k in stale:
        del meta["parts"][k]
        changed = True
    if changed:
        _refresh_reference(vdir, meta)
        _save_meta(vdir, meta)

    total = sum(p["duration_s"] for p in parts) / 60.0
    usable = sum(p["duration_s"] for p in parts if p["verdict"] != "bad") / 60.0
    level = level_for(usable)
    bad = [p["index"] for p in parts if p["verdict"] == "bad"]
    warn = [p["index"] for p in parts if p["verdict"] == "warn"]
    extra = ""
    if bad:
        extra = f"Re-record part {', '.join(map(str, bad))}: it has problems and is not counted."
    elif warn:
        extra = f"Part {', '.join(map(str, warn))} could be better; check the tips and re-record if you can."
    if not parts:
        message = "No recordings yet. Read the first part aloud to get started."
    else:
        message = message_for(usable, level, extra)
    out: dict[str, Any] = {
        "total_minutes": round(total, 2),
        "usable_minutes": round(usable, 2),
        "parts": parts,
        "level": level,
        "message": message,
        "recommended_epochs": recommended_epochs(usable),
        "has_reference": os.path.isfile(os.path.join(vdir, "reference.wav")),
    }
    if meta.get("reference"):
        out["reference"] = meta["reference"]
    return out


# ---------------------------------------------------------------------------------------------- handlers

@method("dataset.add")
def dataset_add(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    vdir = _voice_dir(ctx, params)
    n = _part_index(params)
    y, sr = read_audio(params.get("path"))
    if len(y) < sr * 0.3:
        raise EngineError("bad_request", "This recording is empty or too short.")
    if sr < quality.MIN_SAMPLE_RATE:
        raise EngineError("bad_request", f"Sample rate too low ({sr} Hz): record at 44.1 or 48 kHz.")
    if sr > MAX_RATE:
        y, sr = resample(y, sr, 44100), 44100
    y = y - float(np.mean(y))  # DC offset
    m0 = quality.measure(y, sr)
    y = trim_silence(y, sr, m0)
    m1 = quality.measure(y, sr)  # verdict is about the trimmed take (before the gain, which changes no ratio)
    expected = params.get("expected_s")
    verdict, issues = quality.assess(m1, float(expected) if isinstance(expected, (int, float)) else None)
    y = normalise(y, m1)
    with _LOCK:
        os.makedirs(_ddir(vdir), exist_ok=True)
        write_wav(_part_path(vdir, n), y, sr)
        meta = _load_meta(vdir)
        others = {k: v for k, v in meta["parts"].items() if k != str(n)}
        meta["parts"][str(n)] = {
            "verdict": verdict, "issues": issues, "snr_db": round(m1["snr_db"], 1),
            "clipping_pct": round(m1["clipping_pct"], 3),
            **pace.analyse_part(y, sr, pace.voice_median_rate(others)),
        }
        _refresh_reference(vdir, meta, changed_part=n)
        _save_meta(vdir, meta)
        summary = _summary(vdir, meta)
    return {
        "part": {"index": n, "duration_s": round(len(y) / sr, 2), "verdict": verdict, "issues": issues},
        "summary": summary,
    }


@method("dataset.remove")
def dataset_remove(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    vdir = _voice_dir(ctx, params)
    n = _part_index(params)
    with _LOCK:
        path = _part_path(vdir, n)
        if os.path.isfile(path):
            os.remove(path)
        meta = _load_meta(vdir)
        meta["parts"].pop(str(n), None)
        _refresh_reference(vdir, meta, changed_part=n)
        _save_meta(vdir, meta)
        return {"summary": _summary(vdir, meta)}


@method("dataset.summary")
def dataset_summary(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    vdir = _voice_dir(ctx, params)
    with _LOCK:
        return _summary(vdir, _load_meta(vdir))
