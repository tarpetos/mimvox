"""Shared media layer: ffmpeg discovery, probing, one decoder for every engine input, and ``audio.export``.

Every module that reads a user's audio file goes through :func:`read_audio` (or :func:`decode_to_wav`), so any
format ffmpeg understands works everywhere: wav, mp3, m4a/aac, flac, ogg/vorbis, opus, wma, aiff and the audio track
of mp4, mkv, mov, webm, avi. Plain WAV / FLAC / AIFF files are read natively (no process); everything else, and
anything libsndfile refuses, is decoded by ffmpeg.

ffmpeg lookup order: ``MIMVOX_FFMPEG`` (a file or a folder), the Applio runtime root (``paths.applio`` in
runtime.toml; Applio's prerequisites put ``ffmpeg.exe`` / ``ffprobe.exe`` there on Windows), then ``PATH``
(Linux / macOS: the system ffmpeg that setup installs or verifies).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any

import numpy as np
import soundfile as sf

from . import config
from .registry import EngineError, method

log = logging.getLogger("mimvox.media")

_EXE = ".exe" if os.name == "nt" else ""
_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_runtime: dict[str, Any] = {}
_cache: dict[str, Any] = {}
_lock = threading.Lock()

# Extensions we advertise for import, with the ffmpeg demuxer names and decoders that make each one work.
# (kind, demuxers (any), decoders (any))
_IMPORT: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "wav": ("audio", ("wav",), ("pcm_s16le",)),
    "mp3": ("audio", ("mp3",), ("mp3", "mp3float")),
    "m4a": ("audio", ("mov",), ("aac",)),
    "aac": ("audio", ("aac",), ("aac",)),
    "flac": ("audio", ("flac",), ("flac",)),
    "ogg": ("audio", ("ogg",), ("vorbis", "libvorbis", "opus", "libopus")),
    "opus": ("audio", ("ogg",), ("opus", "libopus")),
    "wma": ("audio", ("asf",), ("wmav2", "wmav1", "wmapro")),
    "aiff": ("audio", ("aiff",), ("pcm_s16be",)),
    "mp4": ("video", ("mov",), ("aac",)),
    "mkv": ("video", ("matroska",), ("aac", "vorbis", "opus", "mp3", "flac", "pcm_s16le")),
    "mov": ("video", ("mov",), ("aac",)),
    "webm": ("video", ("matroska",), ("vorbis", "opus", "libopus", "libvorbis")),
    "avi": ("video", ("avi",), ("mp3", "mp3float", "pcm_s16le", "ac3", "aac")),
}
_IMPORT_ALIASES = {"aif": "aiff", "aifc": "aiff", "oga": "ogg", "mpeg": "mp3", "mp2": "mp3", "m4b": "m4a",
                   "mka": "mkv", "m4v": "mp4", "wave": "wav"}
_NATIVE_EXT = {".wav", ".flac", ".aiff", ".aif", ".aifc", ".ogg", ".oga"}  # libsndfile first, ffmpeg as fallback

# Export formats: ext, encoder options per quality (high / standard / small), encoders that can do it.
_EXPORT: dict[str, dict[str, Any]] = {
    "wav": {"ext": "wav", "args": ["-c:a", "pcm_s16le", "-ar", "44100"], "quality": False, "encoders": ("pcm_s16le",)},
    "mp3": {"ext": "mp3", "args": ["-c:a", "libmp3lame", "-ar", "44100"], "encoders": ("libmp3lame",),
            "q": {"high": ["-b:a", "320k"], "standard": ["-b:a", "192k"], "small": ["-b:a", "128k"]}},
    "flac": {"ext": "flac", "args": ["-c:a", "flac", "-sample_fmt", "s16"], "quality": False, "encoders": ("flac",)},
    "ogg": {"ext": "ogg", "args": ["-c:a", "libvorbis"], "encoders": ("libvorbis", "vorbis"),
            "q": {"high": ["-q:a", "7"], "standard": ["-q:a", "5"], "small": ["-q:a", "3"]}},
    "opus": {"ext": "opus", "args": ["-c:a", "libopus", "-ar", "48000"], "encoders": ("libopus",),
             "q": {"high": ["-b:a", "128k"], "standard": ["-b:a", "96k"], "small": ["-b:a", "64k"]}},
    "m4a": {"ext": "m4a", "args": ["-c:a", "aac", "-ar", "44100", "-movflags", "+faststart"], "encoders": ("aac",),
            "q": {"high": ["-b:a", "256k"], "standard": ["-b:a", "192k"], "small": ["-b:a", "128k"]}},
    "aiff": {"ext": "aiff", "args": ["-c:a", "pcm_s16be", "-ar", "44100"], "quality": False, "encoders": ("pcm_s16be",)},
}
QUALITIES = ("high", "standard", "small")


# ------------------------------------------------------------------------------------------- tool discovery

def configure(runtime: dict[str, Any]) -> None:
    """Remember runtime.toml (for the Applio ffmpeg) and forget earlier lookups."""
    with _lock:
        _runtime.clear()
        _runtime.update(runtime or {})
        _cache.clear()


def _find(name: str) -> str | None:
    env = os.environ.get("MIMVOX_FFMPEG")
    if env:  # a folder, or the ffmpeg file itself (its siblings are used)
        folder = env if os.path.isdir(env) else os.path.dirname(env)
        cand = os.path.join(folder, name + _EXE)
        if os.path.isfile(cand):
            return cand
    applio = config.applio_dir(_runtime) if _runtime else None
    if applio:
        cand = os.path.join(applio, name + _EXE)
        if os.path.isfile(cand):
            return cand
    return shutil.which(name)


def ffmpeg_path() -> str | None:
    with _lock:
        if "ffmpeg" not in _cache:
            _cache["ffmpeg"] = _find("ffmpeg")
        return _cache["ffmpeg"]


def ffprobe_path() -> str | None:
    with _lock:
        if "ffprobe" not in _cache:
            _cache["ffprobe"] = _find("ffprobe")
        return _cache["ffprobe"]


def add_to_path() -> None:
    """Put the ffmpeg folder first on PATH of this process (libraries that shell out to ``ffmpeg`` then find it)."""
    exe = ffmpeg_path()
    if exe:
        folder = os.path.dirname(os.path.abspath(exe))
        if folder not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")


def require_ffmpeg() -> str:
    exe = ffmpeg_path()
    if not exe:
        raise EngineError("not_ready", "ffmpeg was not found. On Windows it ships with the Applio runtime (re-run "
                                       "setup); on Linux/macOS install it (apt install ffmpeg / brew install ffmpeg).")
    return exe


def _run(cmd: list[str], timeout: float = 3600.0) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=_FLAGS, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise EngineError("internal", "ffmpeg timed out") from exc
    except OSError as exc:
        raise EngineError("not_ready", f"ffmpeg could not be started: {exc}") from exc


def _tail(err: bytes) -> str:
    lines = [ln.strip() for ln in err.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    return lines[-1] if lines else "unknown error"


def _no_audio(err: bytes) -> bool:
    text = err.decode("utf-8", errors="replace")
    return "matches no streams" in text or "does not contain any stream" in text


def _check_file(path: Any) -> str:
    if not isinstance(path, str) or not path:
        raise EngineError("bad_request", "'path' must be a file path")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise EngineError("not_found", f"file not found: {path}")
    return path


# ----------------------------------------------------------------------------------------------------- probe

def probe(path: str) -> dict[str, Any]:
    """``{duration_s, sample_rate, channels, has_audio, format}`` plus ``codec`` and ``has_video``."""
    path = _check_file(path)
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    probe_exe = ffprobe_path()
    if probe_exe is None:
        try:  # no ffprobe: libsndfile can still describe plain audio files
            info = sf.info(path)
            return {"duration_s": round(info.duration, 3), "sample_rate": int(info.samplerate),
                    "channels": int(info.channels), "has_audio": True, "format": ext or info.format.lower(),
                    "codec": info.subtype.lower(), "has_video": False}
        except Exception:  # noqa: BLE001
            require_ffmpeg()
            raise EngineError("not_ready", "ffprobe was not found next to ffmpeg") from None
    res = _run([probe_exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path], 60)
    if res.returncode != 0:
        raise EngineError("bad_request", f"This file cannot be read as audio or video ({_tail(res.stderr)}).")
    try:
        info = json.loads(res.stdout.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise EngineError("bad_request", "This file cannot be read as audio or video.") from exc
    streams = info.get("streams") or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    has_video = any(s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
                    for s in streams)
    fmt = info.get("format") or {}
    names = [n for n in str(fmt.get("format_name", "")).split(",") if n]
    # The extension the user knows beats ffprobe's "mov,mp4,m4a,3gp,3g2,mj2".
    known = _IMPORT_ALIASES.get(ext, ext)
    out_fmt = known if known in _IMPORT else (names[0] if names else ext)
    duration = _float(fmt.get("duration"))
    if not duration and audio is not None:
        duration = _float(audio.get("duration"))
    sample_rate = int(_float(audio.get("sample_rate"))) if audio else 0
    if audio is not None and not duration:
        try:  # e.g. a webm without a duration header: count the decoded samples
            y, sr = read_audio(path, sr=sample_rate or 44100)
            duration = len(y) / float(sr)
        except EngineError:
            duration = 0.0
    return {"duration_s": round(float(duration or 0.0), 3), "sample_rate": sample_rate,
            "channels": int(audio.get("channels") or 0) if audio else 0, "has_audio": audio is not None,
            "format": out_fmt, "codec": str(audio.get("codec_name", "")) if audio else "", "has_video": has_video}


def is_video(path: str) -> bool:
    """True for the video containers listed for import (mp4, mkv, mov, webm, avi and their aliases), by extension:
    cheap, no process."""
    ext = os.path.splitext(str(path))[1].lower().lstrip(".")
    ext = _IMPORT_ALIASES.get(ext, ext)
    return ext in _IMPORT and _IMPORT[ext][0] == "video"


def _float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------------------------------- decode

def decode_pcm(path: str, sr: int, max_s: float | None = None) -> np.ndarray:
    """ffmpeg decode of the first audio track to mono float32 at ``sr`` (only the first ``max_s`` seconds if given)."""
    exe = require_ffmpeg()
    limit = ["-t", f"{float(max_s):.3f}"] if max_s else []
    res = _run([exe, "-nostdin", "-v", "error", "-i", path, "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(int(sr)),
                *limit, "-f", "f32le", "pipe:1"])
    if res.returncode != 0 or not res.stdout:
        msg = _tail(res.stderr)
        if _no_audio(res.stderr):
            raise EngineError("bad_request", "This file has no audio track.")
        raise EngineError("bad_request", f"Cannot read this audio file ({msg}).")
    y = np.frombuffer(res.stdout, dtype="<f4")
    y = np.ascontiguousarray(y, dtype=np.float32)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def read_audio(path: str, sr: int | None = None, max_s: float | None = None) -> tuple[np.ndarray, int]:
    """Mono float32 samples of any supported audio / video file and their sample rate.

    ``sr=None`` keeps the file's own rate (what the dataset tools measure); an integer resamples to it. ``max_s`` reads
    only the first that many seconds (bounds the memory of a long file).
    Errors: ``bad_request`` (missing path, unreadable, no audio track), ``not_found``, ``not_ready`` (ffmpeg missing
    for a format libsndfile cannot read).
    """
    path = _check_file(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in _NATIVE_EXT:
        try:
            frames = -1
            if max_s:
                frames = int(float(max_s) * sf.info(path).samplerate)
            data, file_sr = sf.read(path, frames=frames, dtype="float32", always_2d=True)
            y = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            y = np.ascontiguousarray(y, dtype=np.float32)
            if not np.all(np.isfinite(y)):
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if sr is not None and sr != file_sr:
                from . import audio_util as au  # noqa: PLC0415
                y, file_sr = au.resample(y, int(file_sr), int(sr)), int(sr)
            return y, int(file_sr)
        except Exception:  # noqa: BLE001 - e.g. an Ogg Opus file on an old libsndfile: ffmpeg handles it
            pass
    target = sr
    if target is None:
        try:
            target = probe(path)["sample_rate"] or 44100
        except EngineError:
            target = 44100
    return decode_pcm(path, int(target), max_s), int(target)


def decode_to_wav(src: str, dst: str, sr: int = 44100) -> float:
    """Write the first audio track of ``src`` as a mono 16-bit WAV at ``sr`` to ``dst``; returns its duration."""
    y, rate = read_audio(src, sr=sr)
    if len(y) == 0:
        raise EngineError("bad_request", "This file has no audio.")
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    sf.write(dst, np.clip(y, -1.0, 1.0), rate, subtype="PCM_16", format="WAV")
    return len(y) / float(rate)


# ------------------------------------------------------------------------------------------ format support

def _ffmpeg_lists() -> tuple[set[str], set[str], set[str], str]:
    """(demuxers, decoders, encoders, version line) of the found ffmpeg; cached."""
    with _lock:
        if "lists" in _cache:
            return _cache["lists"]
    exe = ffmpeg_path()
    demuxers: set[str] = set()
    decoders: set[str] = set()
    encoders: set[str] = set()
    version = ""
    if exe:
        def listing(flag: str) -> list[str]:
            res = _run([exe, "-hide_banner", flag], 30)
            return res.stdout.decode("utf-8", errors="replace").splitlines()

        for ln in listing("-formats"):
            m = re.match(r"^\s([D ])([E ])\s+(\S+)\s", ln)
            if m and m.group(1) == "D":
                demuxers.update(m.group(3).split(","))
        for ln in listing("-decoders"):
            m = re.match(r"^\s(A)\S{5}\s+(\S+)\s", ln)
            if m:
                decoders.add(m.group(2))
        for ln in listing("-encoders"):
            m = re.match(r"^\s(A)\S{5}\s+(\S+)\s", ln)
            if m:
                encoders.add(m.group(2))
        ver = _run([exe, "-version"], 15).stdout.decode("utf-8", errors="replace").splitlines()
        version = ver[0] if ver else ""
    out = (demuxers, decoders, encoders, version)
    with _lock:
        _cache["lists"] = out
    return out


def formats() -> dict[str, Any]:
    """What this machine can read and write (see ``formats.list`` in the contract)."""
    demuxers, decoders, encoders, version = _ffmpeg_lists()
    if not version:  # no ffmpeg: libsndfile only
        avail = sf.available_formats()
        imp = [e for e, k in (("wav", "WAV"), ("flac", "FLAC"), ("ogg", "OGG"), ("aiff", "AIFF")) if k in avail]
        exp = [e for e, k in (("wav", "WAV"), ("mp3", "MP3"), ("flac", "FLAC"), ("ogg", "OGG"), ("aiff", "AIFF"))
               if k in avail]
        return {"import": imp, "import_video": [], "export": exp, "qualities": list(QUALITIES), "ffmpeg": None}
    imp, video = [], []
    for ext, (kind, dem, dec) in _IMPORT.items():
        if any(d in demuxers for d in dem) and any(d in decoders for d in dec):
            imp.append(ext)
            if kind == "video":
                video.append(ext)
    exp = [f for f, spec in _EXPORT.items() if any(e in encoders for e in spec["encoders"])]
    return {"import": imp, "import_video": video, "export": exp, "qualities": list(QUALITIES), "ffmpeg": version}


@method("formats.list")
def formats_list(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    return formats()


# ------------------------------------------------------------------------------------------------- export

def export(path: str, fmt: str, dest: str | None = None, quality: str = "standard") -> dict[str, Any]:
    path = _check_file(path)
    fmt = str(fmt or "").lower().lstrip(".")
    fmt = _IMPORT_ALIASES.get(fmt, fmt)
    spec = _EXPORT.get(fmt)
    if spec is None:
        raise EngineError("bad_request", f"unsupported export format {fmt!r} (use {', '.join(_EXPORT)})")
    if quality not in QUALITIES:
        raise EngineError("bad_request", f"'quality' must be one of {', '.join(QUALITIES)}")
    if dest in (None, ""):
        stem, old = os.path.splitext(path)
        dest = f"{stem}.{spec['ext']}"
        if os.path.normcase(dest) == os.path.normcase(path):
            dest = f"{stem}_export.{spec['ext']}"
    elif not isinstance(dest, str):
        raise EngineError("bad_request", "'dest' must be an absolute file path")
    else:
        if not os.path.isabs(dest):
            raise EngineError("bad_request", "'dest' must be an absolute file path")
        if os.path.isdir(dest):
            raise EngineError("bad_request", "'dest' must be a file path, not a folder")
    if os.path.normcase(os.path.abspath(dest)) == os.path.normcase(path):
        raise EngineError("bad_request", "'dest' must differ from the source file")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = f"{dest}.part-{os.getpid()}.{spec['ext']}"  # keeps the extension so ffmpeg picks the muxer
    try:
        if ffmpeg_path():
            _encode_ffmpeg(path, tmp, fmt, spec, quality)
        else:
            _encode_soundfile(path, tmp, fmt)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return {"path": dest, "size_bytes": os.path.getsize(dest)}


def _encode_ffmpeg(src: str, dst: str, fmt: str, spec: dict[str, Any], quality: str) -> None:
    exe = ffmpeg_path()
    assert exe
    _, _, encoders, _ = _ffmpeg_lists()
    args = list(spec["args"])
    if fmt == "ogg" and "libvorbis" not in encoders:
        if "vorbis" not in encoders:
            raise EngineError("not_ready", "this ffmpeg build has no Vorbis encoder")
        args = ["-c:a", "vorbis", "-strict", "-2"]  # the built-in encoder is experimental, mono/stereo only
    elif not any(e in encoders for e in spec["encoders"]):
        raise EngineError("not_ready", f"this ffmpeg build cannot encode {fmt}")
    args += spec.get("q", {}).get(quality, [])
    cmd = [exe, "-nostdin", "-v", "error", "-y", "-i", src, "-vn", "-map", "0:a:0", "-map_metadata", "-1", *args, dst]
    res = _run(cmd)
    if res.returncode != 0 or not os.path.isfile(dst):
        msg = _tail(res.stderr)
        if _no_audio(res.stderr):
            raise EngineError("bad_request", "This file has no audio track.")
        raise EngineError("bad_request", f"Could not convert to {fmt} ({msg}).")


def _encode_soundfile(src: str, dst: str, fmt: str) -> None:
    kinds = {"wav": "WAV", "mp3": "MP3", "flac": "FLAC", "ogg": "OGG", "aiff": "AIFF"}
    if fmt not in kinds or kinds[fmt] not in sf.available_formats():
        raise EngineError("not_ready", f"ffmpeg was not found and {fmt} cannot be written without it")
    data, sr = sf.read(src, dtype="float32", always_2d=True)
    sub = {"wav": "PCM_16", "flac": "PCM_16", "aiff": "PCM_16"}.get(fmt)
    sf.write(dst, data, sr, format=kinds[fmt], **({"subtype": sub} if sub else {}))


@method("audio.export")
def audio_export(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    return export(params.get("path"), params.get("format"), params.get("dest"), params.get("quality") or "standard")


# -------------------------------------------------------------------------------------------- saved results

PLAY_DIR = "play"  # <data>/tmp/play: playback WAVs of results saved in another format (tmp is emptied at start)


def output_format(params: dict[str, Any], key: str = "output_format") -> str:
    """The ``output_format`` of a job request (default ``wav``), checked against what this machine can write.
    Blocking (the first call lists the ffmpeg encoders): call it on the IO pool."""
    v = params.get(key)
    if v in (None, ""):
        return "wav"
    if not isinstance(v, str):
        raise EngineError("bad_request", f"'{key}' must be one of {', '.join(_EXPORT)}")
    fmt = v.lower().lstrip(".")
    fmt = _IMPORT_ALIASES.get(fmt, fmt)
    if fmt not in _EXPORT:
        raise EngineError("bad_request", f"'{key}' must be one of {', '.join(_EXPORT)}")
    if fmt != "wav" and fmt not in formats()["export"]:
        raise EngineError("not_ready", f"{fmt.upper()} cannot be written on this machine "
                                       f"({'install ffmpeg' if not ffmpeg_path() else 'this ffmpeg build lacks its encoder'})")
    return fmt


def save_result(data: config.DataDir, track: np.ndarray, sr: int, prefix: str, fmt: str) -> dict[str, str]:
    """Save a finished result as ONE file ``<data>/outputs/<prefix><stamp>.<ext>`` in ``fmt``.

    The app plays WAV only, so another format also gets a 16-bit playback WAV in ``<data>/tmp/play/`` (the same name),
    from which the saved file is encoded (``export``: ffmpeg, else libsndfile). ``track`` is mono float32 at ``sr``
    (44.1 kHz from Speak and Redub; MP3 cannot store 40 kHz). Returns ``{"path", "path_play", "format"}``; if the
    encoder fails anyway, the result is kept as WAV in outputs instead of being lost (``format`` says ``wav``)."""
    from . import audio_util as au  # noqa: PLC0415 - only results need it (it imports scipy)

    ext = _EXPORT[fmt]["ext"]
    play_dir = os.path.join(data.tmp, PLAY_DIR)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name, n = f"{prefix}{stamp}", 1

    def taken(nm: str) -> bool:
        return (os.path.exists(os.path.join(data.outputs, f"{nm}.{ext}"))
                or os.path.exists(os.path.join(play_dir, f"{nm}.wav")))

    while taken(name):  # a free name, not a wait
        n += 1
        name = f"{prefix}{stamp}_{n}"
    os.makedirs(data.outputs, exist_ok=True)
    path = os.path.join(data.outputs, f"{name}.{ext}")
    if fmt == "wav":
        au.write_wav(path, track, sr)
        return {"path": path, "path_play": path, "format": "wav"}
    play = os.path.join(play_dir, f"{name}.wav")
    au.write_wav(play, track, sr)
    try:
        export(play, fmt, path, "high")
    except EngineError as exc:
        log.warning("saving %s as %s failed (%s): keeping it as WAV", name, fmt, exc.message)
        path = os.path.join(data.outputs, f"{name}.wav")
        while os.path.exists(path):
            n += 1
            path = os.path.join(data.outputs, f"{prefix}{stamp}_{n}.wav")
        shutil.copyfile(play, path)
        return {"path": path, "path_play": play, "format": "wav"}
    return {"path": path, "path_play": play, "format": fmt}
