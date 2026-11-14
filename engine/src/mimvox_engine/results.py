"""Saved Speak and Redub results: their ``.result.json`` sidecars, ``outputs.list`` and ``outputs.play``.

``outputs.play`` always returns a WAV: the app plays WAV only.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

from . import config, media
from .registry import EngineError, method

log = logging.getLogger("mimvox.results")

SIDECAR = ".result.json"
KINDS = ("speak", "redub")
LIMIT_DEFAULT, LIMIT_MAX = 50, 500
PLAY_SR = 44100
_AUDIO = {"wav", "mp3", "flac", "ogg", "opus", "m4a", "aiff"}
# Saved before sidecars: "<stamp>" or "redub_<stamp>", "_<n>" when the second was taken.
_LEGACY = re.compile(r"^(redub_)?\d{8}_\d{6}(_\d+)?$")
_FIELDS = ("voice_id", "voice_name", "text", "language", "accent", "mode", "source_name", "to")

_durations: dict[tuple[str, int, int], float] = {}
_dur_lock = threading.Lock()


def sidecar_path(result_path: str) -> str:
    return os.path.splitext(result_path)[0] + SIDECAR


def record(saved: dict[str, Any], fields: dict[str, Any]) -> None:
    """IO thread. Never raises: the result itself is already saved."""
    path = str(saved.get("path") or "")
    if not path:
        return
    out: dict[str, Any] = {"version": 1, "created": round(time.time(), 3), **fields,
                           "format": saved.get("format") or os.path.splitext(path)[1].lstrip(".").lower(),
                           "path": path}
    if fields.get("video_path") in (None, ""):
        out["video_path"] = None
    dest = sidecar_path(path)
    tmp = f"{dest}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        for attempt in range(5):  # Windows: a reader (an antivirus) may hold the target for a moment
            try:
                os.replace(tmp, dest)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except OSError as exc:
        log.warning("could not write %s: %s", dest, exc)
        try:
            os.remove(tmp)
        except OSError:
            pass


def _read_sidecar(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("ignoring %s: %s", path, exc)
        return None
    return obj if isinstance(obj, dict) and obj.get("kind") in KINDS else None


class _Group:
    def __init__(self) -> None:
        self.audio: list[tuple[str, os.stat_result]] = []
        self.video: list[tuple[str, os.stat_result]] = []
        self.sidecar: str | None = None


def _scan(outputs: str) -> dict[str, _Group]:
    groups: dict[str, _Group] = {}
    try:
        it = os.scandir(outputs)
    except FileNotFoundError:
        return groups
    with it:
        for e in it:
            name = e.name
            if name.startswith(".") or name.endswith(".tmp") or ".part" in name:
                continue
            try:
                if not e.is_file():
                    continue
                st = e.stat()
            except OSError:
                continue
            if name.endswith(SIDECAR):
                groups.setdefault(name[:-len(SIDECAR)], _Group()).sidecar = e.path
                continue
            stem, ext = os.path.splitext(name)
            ext = ext.lower().lstrip(".")
            if ext in _AUDIO:
                groups.setdefault(stem, _Group()).audio.append((e.path, st))
            elif media.is_video(name):
                groups.setdefault(stem, _Group()).video.append((e.path, st))
    return groups


def _pick(files: list[tuple[str, os.stat_result]], name: str | None = None) -> tuple[str, os.stat_result] | None:
    """``name`` when present, else the oldest: an export saved next to a result is newer."""
    if not files:
        return None
    if name:
        for f in files:
            if os.path.normcase(os.path.basename(f[0])) == os.path.normcase(name):
                return f
    return min(files, key=lambda f: (f[1].st_mtime, f[0]))


def _entry(stem: str, g: _Group) -> dict[str, Any] | None:
    side = _read_sidecar(g.sidecar) if g.sidecar else None
    if side is not None:
        audio = _pick(g.audio, os.path.basename(str(side.get("path") or "")))
        video = _pick(g.video, os.path.basename(str(side.get("video_path") or "")))
        main = audio or video
        if main is None:
            return None
        e: dict[str, Any] = {"id": stem, "kind": side["kind"], "legacy": False, "name": os.path.basename(main[0]),
                             "created": _num(side.get("created")) or main[1].st_mtime,
                             "path": main[0], "video_path": video[0] if video else None,
                             "duration_s": _num(side.get("duration_s")), "elapsed_s": _num(side.get("elapsed_s")),
                             "format": side.get("format") or os.path.splitext(main[0])[1].lstrip(".").lower(),
                             "_stat": main[1]}
        for k in _FIELDS:
            if side.get(k) not in (None, ""):
                e[k] = side[k]
        return e
    legacy = bool(_LEGACY.match(stem))
    audio, video = _pick(g.audio), _pick(g.video)
    main = audio or video
    if main is None:
        return None
    kind = ("redub" if stem.startswith("redub_") else "speak") if legacy else ("redub" if audio is None else "speak")
    return {"id": stem, "kind": kind, "legacy": True, "external": not legacy,
            "name": os.path.basename(main[0]), "created": main[1].st_mtime, "path": main[0],
            "video_path": video[0] if video else None, "duration_s": None, "elapsed_s": None,
            "format": os.path.splitext(main[0])[1].lstrip(".").lower(), "_stat": main[1]}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _duration(path: str, st: os.stat_result) -> float | None:
    key = (path, st.st_mtime_ns, st.st_size)
    with _dur_lock:
        if key in _durations:
            return _durations[key]
    d: float | None = None
    try:
        import soundfile as sf  # noqa: PLC0415

        d = float(sf.info(path).duration)
    except Exception:  # noqa: BLE001 - m4a, opus on an old libsndfile, a video: ffprobe
        try:
            d = float(media.probe(path)["duration_s"])
        except EngineError as exc:
            log.warning("could not read the length of %s: %s", path, exc.message)
    if d is not None:
        d = round(d, 3)
        with _dur_lock:
            _durations[key] = d
    return d


def _play_cached(data: config.DataDir, path: str) -> str | None:
    """Without decoding: the WAV itself, or its copy in tmp (emptied at every engine start)."""
    if path.lower().endswith(".wav"):
        return path
    dst = _play_dst(data, path)
    return dst if os.path.isfile(dst) else None


def _play_dst(data: config.DataDir, path: str) -> str:
    return os.path.join(data.tmp, media.PLAY_DIR, os.path.splitext(os.path.basename(path))[0] + ".wav")


def list_results(data: config.DataDir, kind: str | None = None, limit: int = LIMIT_DEFAULT,
                 before: float | None = None) -> dict[str, Any]:
    entries = []
    groups = _scan(data.outputs)
    for stem, g in groups.items():
        if stem.endswith("_export") and stem[:-len("_export")] in groups and not g.sidecar:
            continue   # media.export's default name for an export of that result
        e = _entry(stem, g)
        if e is None or (kind and e["kind"] != kind) or (before is not None and e["created"] >= before):
            continue
        entries.append(e)
    entries.sort(key=lambda e: (e["created"], e["id"]), reverse=True)
    page, more = entries[:limit], len(entries) > limit
    for e in page:
        st = e.pop("_stat", None)
        if e["duration_s"] is None and st is not None:
            e["duration_s"] = _duration(e["path"], st)
        e["path_play"] = _play_cached(data, e["path"])
    return {"results": page, "more": more}


@method("outputs.list")
def outputs_list(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    kind = params.get("kind")
    if kind not in (None, "", *KINDS):
        raise EngineError("bad_request", "'kind' must be \"speak\" or \"redub\"")
    limit = params.get("limit", LIMIT_DEFAULT)
    if limit is None:
        limit = LIMIT_DEFAULT
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise EngineError("bad_request", "'limit' must be a positive integer")
    before = params.get("before")
    if before is not None and (isinstance(before, bool) or not isinstance(before, (int, float))):
        raise EngineError("bad_request", "'before' must be a time in seconds (the 'created' of an entry)")
    return list_results(config.data_dir(ctx), kind or None, min(limit, LIMIT_MAX),
                        float(before) if before is not None else None)


def playable(data: config.DataDir, path: Any) -> dict[str, str]:
    if not isinstance(path, str) or not path or not os.path.isabs(path):
        raise EngineError("bad_request", "'path' must be an absolute file path")
    real = os.path.realpath(path)
    if not media._inside(real, data.outputs):
        raise EngineError("bad_request", "only results in the outputs folder can be played this way")
    if not os.path.isfile(real):
        raise EngineError("not_found", "the result file is gone")
    cached = _play_cached(data, real)
    if cached:
        return {"path": path, "path_play": cached if cached != real else path}
    dst = _play_dst(data, real)
    media.play_wav(real, dst, PLAY_SR)
    return {"path": path, "path_play": dst}


@method("outputs.play")
def outputs_play(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    return playable(config.data_dir(ctx), params.get("path"))
