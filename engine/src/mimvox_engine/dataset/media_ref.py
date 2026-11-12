"""``voices.reference_from_media`` / ``voices.media_cancel`` / ``voices.clear_media``: a reference clip chosen from
many media files of the same speaker.

The speech model clones from one short clip, so a single noisy source (a compressed news intro, music under the voice)
makes the clone sound metallic. Here the user drops many audio / video files; each is decoded to mono 44.1 kHz,
DC-removed, trimmed at both ends, its long pauses shortened and its loudness normalised, and the files are joined with
short gaps into ``<voice>/reference_media.wav`` (the *media pool*, at most 20 minutes). Every file is analysed on its
own with the pace / SNR analysis of the dataset (``pace.py``), into pause-aligned windows of 8-12 s (the speech model
uses a clip of up to 12 s whole), so no window spans two files. The best window becomes the reference:
``reference.wav`` for the voice's own language, ``reference_<lang>.wav`` for another one.

The pool is not a dataset part: training data never changes. Its best windows are stored in ``dataset/meta.json``
(``media``) and offered by ``voices.reference_candidates`` (``source: "media"``), so the user can listen and pick
another one with ``voices.set_reference {"source": "media", ...}`` (see ``reference.py``). A reference taken from the
pool is a manual choice: ``dataset.add`` / ``dataset.remove`` keep it.

The job runs on the engine loop; decoding and analysis run on the IO pool, one file at a time (bounded memory: a file
is decoded only as far as the pool still has room). Cancelling stops at the next step; the pool, the meta entry and the
reference are installed together at the end (a cancelled job changes nothing).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import numpy as np
import soundfile as sf

from .. import aio, media
from ..registry import Context, EngineError, method
from . import pace, quality, store
from ._audio import HOP_S, write_wav

log = logging.getLogger("mimvox.media_ref")

SR = 44100
GAP_S = 0.4                    # silence between two files in the pool
MAX_POOL_S = 20 * 60.0         # the pool stops growing here (files after it are listed as unused)
DECODE_SLACK_S = 120.0         # decode a little more than the room left: trimming and shorter pauses remove some
MAX_FILES = 500
LONG_PAUSE_S, KEEP_PAUSE_S = 0.7, 0.3   # inner silences longer than LONG_PAUSE_S are shortened to KEEP_PAUSE_S
MIN_SPEECH_S = 3.0             # a file with less than this after cleaning is skipped
FADE_S = 0.005                 # ramps at the cuts of a shortened pause

# Reference windows: the speech model uses a clip of up to 12 s whole (longer ones it cuts to 7-12 s itself).
WIN_LO_S, WIN_HI_S, WIN_FALLBACK_HI_S, WIN_IDEAL_S = 8.0, 12.0, 15.0, 10.0
KEEP_PER_FILE = 12             # best distinct windows kept per file before the pool-wide ranking
KEEP_TOTAL = 24                # windows stored in meta.json for voices.reference_candidates
N_SHOW, MAX_SHOW_PER_FILE = 5, 2

_RUNNING: dict[str, str] = {}  # voice id -> job id (loop thread only)


class _Skip(Exception):
    """A file that cannot be used; the message is shown to the user."""


# --------------------------------------------------------------------------------------------- per-file processing

def _silent_runs(sil: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], sil.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _shorten_pauses(y: np.ndarray, sr: int, m: dict[str, Any]) -> np.ndarray:
    """Inner silences longer than LONG_PAUSE_S shortened to KEEP_PAUSE_S (half kept at each side, short ramps at the
    cut), so the pool is mostly speech and the windows hold more of it."""
    sil = m["_frame_db"] < m["_silence_thr"]
    hop = max(int(sr * HOP_S), 1)
    long_f, half_f = int(LONG_PAUSE_S / HOP_S), int(KEEP_PAUSE_S / 2.0 / HOP_S)
    ramp_n = max(int(sr * FADE_S), 1)
    ramp = np.linspace(0.0, 1.0, ramp_n, dtype=np.float32)
    pieces: list[np.ndarray] = []
    pos = 0
    for a, b in _silent_runs(sil):
        if b - a <= long_f or a == 0 or b >= len(sil):
            continue  # short, or at an edge (trim_silence keeps a little padding there on purpose)
        cut_a, cut_b = (a + half_f) * hop, (b - half_f) * hop
        if cut_b - cut_a < ramp_n * 2 or cut_a <= pos:
            continue
        piece = y[pos:cut_a].copy()
        if len(piece) >= ramp_n:
            piece[-ramp_n:] *= ramp[::-1]
        if pieces and len(piece) >= ramp_n:
            piece[:ramp_n] *= ramp
        pieces.append(piece)
        pos = cut_b
    if not pieces:
        return y
    tail = y[pos:].copy()
    if len(tail) >= ramp_n:
        tail[:ramp_n] *= ramp
    pieces.append(tail)
    return np.concatenate(pieces)


def _cut_at_pause(y: np.ndarray, sr: int, m: dict[str, Any], limit_s: float) -> np.ndarray:
    """``y`` cut to at most ``limit_s``, at the last pause in the 10 s before the limit when there is one."""
    hop = max(int(sr * HOP_S), 1)
    end_f = int(limit_s / HOP_S)
    sil = np.flatnonzero(m["_frame_db"][max(end_f - int(10.0 / HOP_S), 0):end_f] < m["_silence_thr"])
    end = (max(end_f - int(10.0 / HOP_S), 0) + int(sil[-1])) * hop if sil.size else int(limit_s * sr)
    return store._fade(y[:max(end, sr)], sr)


def _clean(path: str, room_s: float, analyse: bool = True) -> dict[str, Any]:
    """Decode and clean one file (IO thread). Returns ``{"y": float32 at SR, "truncated", "clip_pen", "an", "rate",
    "windows"}``; raises ``_Skip`` with a reason for the user."""
    if not os.path.isfile(path):
        raise _Skip("file not found")
    limit = room_s + DECODE_SLACK_S
    try:
        y, sr = media.read_audio(path, sr=SR, max_s=limit)
    except EngineError as exc:
        raise _Skip(exc.message.rstrip(".") if exc.message else "cannot be read") from None
    except Exception as exc:  # noqa: BLE001 - a broken file is skipped, not the whole job
        raise _Skip(f"cannot be read ({type(exc).__name__})") from None
    if len(y) < int(0.5 * sr):
        raise _Skip("no audio")
    truncated = len(y) >= int((limit - 0.5) * sr)
    y = (y - np.float32(np.mean(y, dtype=np.float64))).astype(np.float32, copy=False)  # DC offset
    m0 = quality.measure(y, sr)
    if m0["speech_db"] <= -90.0 or m0["snr_db"] < 6.0:
        raise _Skip("no clear speech found")
    clip_pen = pace.CLIP_PENALTY if m0["clipping_pct"] > 0.05 else 0.0
    y = store.trim_silence(y, sr, m0)
    del m0
    m = quality.measure(y, sr)
    if m["snr_db"] >= quality.SNR_BAD_DB:  # in a noisy file quiet syllables pass for silence: leave its pauses alone
        y = _shorten_pauses(y, sr, m)
        m = quality.measure(y, sr)
    if len(y) > room_s * sr:
        y = _cut_at_pause(y, sr, m, room_s)
        truncated = True
        m = quality.measure(y, sr)
    y = store.normalise(y, m)
    del m
    if len(y) < MIN_SPEECH_S * sr:
        raise _Skip(f"less than {MIN_SPEECH_S:g} s of speech")
    if not analyse:
        return {"y": y, "truncated": truncated, "clip_pen": clip_pen, "rate": 0.0, "windows": []}
    an = pace.PartAnalysis(y, sr)
    wins: list[dict[str, Any]] = []
    rate = pace.part_rate(an) if an.ok else 0.0
    if an.ok:
        scored = []
        for a, b in an.windows(WIN_LO_S, WIN_HI_S, WIN_FALLBACK_HI_S, WIN_IDEAL_S):
            mm = an.measure(a, b)
            scored.append({**mm, "score": pace.window_score(mm, rate, 0.0, clip_pen, WIN_IDEAL_S)})
        spoken = [c for c in scored if c["rate"] > 0]  # windows with under 3 s of speech have no rate: last resort
        scored = spoken or scored
        scored.sort(key=lambda c: -c["score"])
        wins = _distinct(scored, KEEP_PER_FILE)
    return {"y": y, "truncated": truncated, "clip_pen": clip_pen, "rate": rate, "windows": wins}


def _distinct(cands: list[dict[str, Any]], count: int, per_file: int | None = None) -> list[dict[str, Any]]:
    """Best-first selection of mostly non-overlapping windows (``pace.MAX_OVERLAP``), at most ``per_file`` per file."""
    out: list[dict[str, Any]] = []
    per: dict[Any, int] = {}
    for c in cands:
        if len(out) >= count:
            break
        if any(pace._overlap(c, o) > pace.MAX_OVERLAP for o in out):
            continue
        f = c.get("file")
        if per_file is not None and per.get(f, 0) >= per_file:
            continue
        out.append(c)
        per[f] = per.get(f, 0) + 1
    return out


def rank(cands: list[dict[str, Any]], median: float) -> list[dict[str, Any]]:
    """Pool windows re-scored against the pool's median rate, best first."""
    out = []
    for c in cands:
        m = {k: float(c[k]) for k in ("start_s", "end_s", "duration_s", "rate", "pauses", "snr_db")}
        out.append({**c, "score": pace.window_score(m, median, 0.0, float(c.get("clip_pen", 0.0)), WIN_IDEAL_S)})
    out.sort(key=lambda c: -c["score"])
    return out


def show_candidates(media_meta: dict[str, Any]) -> list[dict[str, Any]]:
    """The pool windows ``voices.reference_candidates`` offers (best first, a few per file when there are several)."""
    files = len(media_meta.get("files") or [])
    ranked = rank(list(media_meta.get("cands") or []), float(media_meta.get("rate_median") or 0.0))
    return _distinct(ranked, N_SHOW, MAX_SHOW_PER_FILE if files >= 3 else None)


# --------------------------------------------------------------------------------------------- targets

def target_name(code: str | None) -> str:
    """``reference.wav`` for the voice's own language (``code`` None), else ``reference_<code>.wav``."""
    return "reference.wav" if not code else f"reference_{code}.wav"


def _drop_sidecar(vdir: str, name: str) -> None:
    side = os.path.join(vdir, os.path.splitext(name)[0] + ".omnivoice.json")
    try:
        os.remove(side)
    except FileNotFoundError:
        pass


def write_target(vdir: str, meta: dict[str, Any], start_s: float, end_s: float, code: str | None,
                 info: dict[str, Any]) -> dict[str, Any]:
    """Cut ``[start_s, end_s]`` of the pool into the target file and record the choice (store lock held by the
    caller): ``meta['reference']`` for ``reference.wav`` (a manual choice), ``meta['media']['current']`` for a language
    reference. Returns the recorded entry."""
    clip, sr = store.cut_media(vdir, start_s, end_s)
    name = target_name(code)
    write_wav(os.path.join(vdir, name), clip, sr)
    _drop_sidecar(vdir, name)
    pool = meta["media"].get("pool") if isinstance(meta.get("media"), dict) else None
    entry = {"source": "media", "start_s": round(start_s, 3), "end_s": round(end_s, 3),
             "duration_s": round(len(clip) / sr, 2), "manual": True, "pool": pool, **info}
    if code is None:
        meta["reference"] = entry
    if isinstance(meta.get("media"), dict):
        meta["media"]["current"] = {"start_s": entry["start_s"], "end_s": entry["end_s"]}
    return entry


def file_at(media_meta: dict[str, Any], start_s: float, end_s: float) -> dict[str, Any] | None:
    """The pool file that holds ``[start_s, end_s]`` (most of it), or None."""
    best, best_len = None, 0.0
    for f in media_meta.get("files") or []:
        inter = min(end_s, float(f["end_s"])) - max(start_s, float(f["start_s"]))
        if inter > best_len:
            best, best_len = f, inter
    return best


# --------------------------------------------------------------------------------------------- the job

def _language(ctx: Context, vid: str, value: Any) -> str | None:
    """None for the voice's own language (reference.wav), else the code of another speech language."""
    from .. import config, voices  # noqa: PLC0415
    from .reference import _speech_language  # noqa: PLC0415

    own = str(voices.read_meta(config.data_dir(ctx), vid).get("language") or "en").lower()
    if value in (None, "", "auto"):
        return None
    code = _speech_language(value)
    return None if code == own.split("-")[0] else code


def _paths(params: dict[str, Any]) -> list[str]:
    paths = params.get("paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p for p in paths):
        raise EngineError("bad_request", "'paths' must be a non-empty list of file paths")
    if len(paths) > MAX_FILES:
        raise EngineError("bad_request", f"at most {MAX_FILES} files at once")
    seen, out = set(), []
    for p in paths:
        a = os.path.abspath(p)
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


@method("voices.reference_from_media")
async def reference_from_media(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    from .. import tts  # noqa: PLC0415

    vid = params.get("id", params.get("voice_id"))
    vdir = store._voice_dir(ctx, {"voice_id": vid})
    vid = str(vid)
    paths = _paths(params)
    code = await aio.run_io(_language, ctx, vid, params.get("language"))
    if tts.is_busy():
        raise EngineError("busy", "a generation is running; add media when it has finished")
    if vid in _RUNNING:
        raise EngineError("busy", "media files are already being combined for this voice")

    async def run(job: str) -> None:
        try:
            await _job(ctx, job, vid, vdir, paths, code)
        finally:
            if _RUNNING.get(vid) == job:
                del _RUNNING[vid]

    job = ctx.start_job(run)
    _RUNNING[vid] = job
    return {"job": job, "files": len(paths)}


@method("media.combine")
async def combine(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """Many media files of one speaker joined into one WAV, cleaned like the media pool (decoded to mono 44.1 kHz,
    DC removed, trimmed, long pauses shortened, loudness normalised, 0.4 s between files, at most 20 minutes) and
    written to ``<data>/tmp/combined/``. Record imports the result as one take (it then goes through
    ``recording.check`` -> ``dataset.add`` like any other file). Unreadable files are skipped and listed."""
    paths = _paths(params)
    out_dir = os.path.join(ctx.data_dir, "tmp", "combined")
    pieces: list[np.ndarray] = []
    files: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    unused: list[str] = []
    gap = np.zeros(int(GAP_S * SR), dtype=np.int16)
    pos = 0
    for k, path in enumerate(paths):
        room = MAX_POOL_S - pos / SR
        if room < MIN_SPEECH_S:
            unused.extend(os.path.basename(p) for p in paths[k:])
            break
        try:
            res = await aio.run_io(_clean, path, room, False)
        except _Skip as exc:
            skipped.append({"name": os.path.basename(path), "path": path, "reason": str(exc)})
            continue
        if pieces:
            pieces.append(gap)
            pos += len(gap)
        y = res.pop("y")
        pieces.append(np.round(np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16))
        pos += len(y)
        files.append({"name": os.path.basename(path), "duration_s": round(len(y) / SR, 2), "truncated": bool(res["truncated"])})
        del y, res
    if not files:
        raise EngineError("bad_request", "None of the files has usable speech: "
                          + "; ".join(f"{x['name']}: {x['reason']}" for x in skipped))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"combined-{time.strftime('%Y%m%d-%H%M%S')}-{len(files)}files.wav")
    await aio.run_io(_write_pool, out, pieces)
    log.info("media.combine: %d file(s) -> %s (%.1f s), %d skipped", len(files), out, pos / SR, len(skipped))
    return {"path": out, "duration_s": round(pos / SR, 2), "files": files, "skipped": skipped, "unused": unused}


@method("voices.media_cancel")
async def media_cancel(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    job = params.get("job")
    if not isinstance(job, str):
        raise EngineError("bad_request", "'job' is required")
    ctx.cancel(job)
    return {}


@method("voices.clear_media")
async def clear_media(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """Delete the media pool and its candidates. References already cut from it stay."""
    vid = params.get("id", params.get("voice_id"))
    vdir = store._voice_dir(ctx, {"voice_id": vid})
    if str(vid) in _RUNNING:
        raise EngineError("busy", "media files are being combined for this voice")

    def clear() -> None:
        with store._LOCK:
            meta = store._load_meta(vdir)
            if meta.pop("media", None) is not None:
                store._save_meta(vdir, meta)
            try:
                os.remove(os.path.join(vdir, store.MEDIA_POOL))
            except FileNotFoundError:
                pass

    await aio.run_io(clear)
    return {}


async def _job(ctx: Context, job: str, vid: str, vdir: str, paths: list[str], code: str | None) -> None:
    from .. import tts  # noqa: PLC0415

    n = len(paths)
    t0 = time.time()

    def progress(stage: str, pct: float, k: int | None = None) -> None:
        data: dict[str, Any] = {"id": vid, "stage": stage, "pct": round(min(max(pct, 0.0), 100.0), 1), "files": n}
        if k is not None:
            data.update(file=k + 1, name=os.path.basename(paths[k]))
        ctx.emit("voices.media.progress", data, job=job)

    pieces: list[np.ndarray] = []
    files: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    unused: list[str] = []
    cands: list[dict[str, Any]] = []
    rates: list[float] = []
    pos = 0  # samples in the pool so far (gaps included)
    gap = np.zeros(int(GAP_S * SR), dtype=np.int16)
    tmp = os.path.join(vdir, f"{store.MEDIA_POOL}.{job}.tmp.wav")
    try:
        for k, path in enumerate(paths):
            room = MAX_POOL_S - pos / SR
            if room < MIN_SPEECH_S:
                unused.extend(os.path.basename(p) for p in paths[k:])
                break
            progress("decode", 85.0 * k / n, k)
            try:
                res = await aio.run_io(_clean, path, room)
            except _Skip as exc:
                skipped.append({"name": os.path.basename(path), "path": path, "reason": str(exc)})
                continue
            if pieces:
                pieces.append(gap)
                pos += len(gap)
            y = res.pop("y")
            off = pos / SR
            pieces.append(np.round(np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16))
            pos += len(y)
            entry = {"index": k + 1, "name": os.path.basename(path), "start_s": round(off, 3),
                     "end_s": round(pos / SR, 3), "truncated": bool(res["truncated"])}
            files.append(entry)
            if res["rate"] > 0:
                rates.append(float(res["rate"]))
            for w in res["windows"]:
                cands.append({"file": k + 1, "start_s": round(off + w["start_s"], 3), "end_s": round(off + w["end_s"], 3),
                              "duration_s": round(w["duration_s"], 3), "rate": round(w["rate"], 3),
                              "pauses": round(w["pauses"], 3), "snr_db": round(w["snr_db"], 2),
                              "clip_pen": res["clip_pen"]})
            del y, res
        if not files:
            ctx.emit("voices.media.error", {"code": "bad_request", "message": "None of the files has usable speech.",
                                            "skipped": skipped}, job=job)
            return
        progress("combine", 88.0)
        await aio.run_io(_write_pool, tmp, pieces)
        pieces.clear()
        progress("analyse", 94.0)
        median = float(np.median(rates)) if rates else 0.0
        ranked = rank(cands, median)
        keep = _distinct(ranked, KEEP_TOTAL)
        best = keep[0] if keep else None
        busy = tts.is_busy()  # loop thread: a generation that started meanwhile keeps its reference
        progress("apply", 97.0)
        media_meta = {"version": 1, "pool": f"{int(time.time() * 1000):x}",
                      "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "language": code,
                      "duration_s": round(pos / SR, 2), "files": files, "skipped": skipped, "unused": unused,
                      "rate_median": round(median, 3), "cands": keep, "current": None}
        install = asyncio.ensure_future(aio.run_io(_install, vdir, tmp, media_meta, None if busy else best, code))
        try:
            ref = await asyncio.shield(install)
        except asyncio.CancelledError:
            ref = await install  # already writing: finish, so the pool, meta and reference stay consistent
    except asyncio.CancelledError:
        ctx.emit("voices.media.error", {"code": "cancelled", "message": "cancelled"}, job=job)
        _remove(tmp)
        return
    except EngineError as exc:
        ctx.emit("voices.media.error", {"code": exc.code, "message": exc.message}, job=job)
        _remove(tmp)
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("media reference job %s failed", job)
        ctx.emit("voices.media.error", {"code": "internal", "message": f"{type(exc).__name__}: {exc}"}, job=job)
        _remove(tmp)
        return

    warming = False
    if ref is not None:
        tts.forget_reference()
        if code is None:
            try:  # the existing warm path, as voices.set_reference does
                warming = bool((await tts.voices_warm(ctx, {"id": vid})).get("warming"))
            except EngineError:
                pass
    message = None
    if best is None:
        message = "No clip of 8-12 s with steady speech was found in these files."
    elif busy:
        message = "A generation was running, so the reference was not changed. Pick a clip from the list."
    elif unused:
        message = f"The 20-minute limit was reached; {len(unused)} file(s) were not used."
    ctx.emit("voices.media.done", {
        "id": vid, "files": len(files), "duration_s": round(pos / SR, 2), "language": code,
        "target": target_name(code), "applied": ref is not None, "reference": ref, "warming": warming,
        "preview_path": os.path.join(vdir, target_name(code)) if ref is not None else None,
        "skipped": skipped, "unused": unused, "truncated": [f["name"] for f in files if f["truncated"]],
        "message": message, "elapsed_s": round(time.time() - t0, 1),
    }, job=job)


def _write_pool(tmp: str, pieces: list[np.ndarray]) -> None:
    sf.write(tmp, np.concatenate(pieces) if len(pieces) > 1 else pieces[0], SR, subtype="PCM_16", format="WAV")


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _install(vdir: str, tmp: str, media_meta: dict[str, Any], best: dict[str, Any] | None,
             code: str | None) -> dict[str, Any] | None:
    """Put the new pool in place, record it in meta.json and write the best window as the reference (IO thread)."""
    with store._LOCK:
        os.replace(tmp, os.path.join(vdir, store.MEDIA_POOL))
        meta = store._load_meta(vdir)
        meta["media"] = media_meta  # a reference cut from an older pool keeps its file (its "pool" id differs)
        ref = None
        if best is not None:
            f = file_at(media_meta, best["start_s"], best["end_s"]) or {}
            info = {"file": f.get("index"), "title": f.get("name"), "rate": best["rate"], "pauses": best["pauses"],
                    "snr_db": best["snr_db"]}
            ref = write_target(vdir, meta, float(best["start_s"]), float(best["end_s"]), code, info)
        store._save_meta(vdir, meta)
        return ref
