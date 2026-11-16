from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .. import aio, components, config, media, speech
from ..registry import Context, EngineError
from . import langs

log = logging.getLogger("mimvox.redub.video")

SR = 44100
SEP_PREFIX = "redub_sep_"     # <data>/tmp/redub_sep_<file key>/{vocals,background}.npy
OUTPUTS = ("video", "audio")

# Container (by the source's extension) -> audio encoders to try, best first: (encoder, options).
_AAC = ("aac", ["-b:a", "192k"])
_OPUS = ("libopus", ["-b:a", "160k"])
_CONTAINERS: dict[str, tuple[tuple[str, list[str]], ...]] = {
    "mp4": (_AAC,), "m4v": (_AAC,), "mov": (_AAC,),
    "mkv": (_OPUS, _AAC),
    "webm": (_OPUS, ("libvorbis", ["-q:a", "6"])),
    "avi": (("libmp3lame", ["-b:a", "192k"]), _AAC),
}
VIDEO_AUDIO_SR = 48000       # the video's audio track is written at the video standard rate
FALLBACK_CONTAINER = "mkv"   # a video in another container (ts, flv, ...) is written as Matroska
_FASTSTART = {"mp4", "m4v", "mov"}

# NLLB codes whose first three letters are not the ISO 639-2 code that containers use for the audio language.
_ISO639_2 = {"arb": "ara", "pes": "fas", "zsm": "msa", "swh": "swa", "khk": "mon", "azj": "aze", "lvs": "lav",
             "npi": "nep", "uzn": "uzb", "als": "sqi", "plt": "mlg", "pbt": "pus", "ydd": "yid", "gaz": "orm",
             "quy": "que", "ory": "ori", "yue": "zho", "nob": "nor", "est": "est"}


def file_key(path: str) -> str:
    """(path, mtime, size): the preview and the stems are cached by it."""
    st = os.stat(path)
    return hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8")).hexdigest()[:16]


def language_tag(code: str | None) -> str | None:
    """``en`` -> ``eng``, or None."""
    nllb = langs.nllb_code(code) if code else None
    if not nllb:
        return None
    three = nllb.split("_")[0]
    return _ISO639_2.get(three, three)


@dataclass
class Output:
    output: str
    keep_background: bool
    has_video: bool
    container: str        # the extension of the video result

    @property
    def video(self) -> bool:
        return self.output == "video"

    @property
    def mixes(self) -> bool:
        """Any video, or a kept background, goes through the mix (loudness, background, limiter)."""
        return self.video or self.keep_background


def options(params: dict[str, Any], path: str) -> Output:
    """Defaults: ``output`` audio, ``keep_background`` true for a video output. Blocking (probes)."""
    out = params.get("output")
    out = "audio" if out in (None, "") else out
    if out not in OUTPUTS:
        raise EngineError("bad_request", "'output' must be \"video\" or \"audio\"")
    keep = params.get("keep_background")
    if keep is None:
        keep = out == "video"
    if not isinstance(keep, bool):
        raise EngineError("bad_request", "'keep_background' must be true or false")
    has_video = False
    if out == "video" or keep:
        has_video = bool(media.probe(path).get("has_video"))
    if out == "video" and not has_video:
        raise EngineError("bad_request", "This file has no picture, so there is no video to write: use output "
                                         "\"audio\".")
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return Output(out, keep, has_video, ext if ext in _CONTAINERS else FALLBACK_CONTAINER)


@dataclass
class Stems:
    """A file split into ``vocals`` and ``background`` (drums + bass + other): float32 ``(channels, n)`` ``.npy``
    files at 44.1 kHz, read memory-mapped."""
    folder: str
    channels: int
    frames: int
    device: str
    elapsed_s: float           # the separation itself (model load excluded)
    total_s: float             # decode + load + separation
    cached: bool = False       # found ready (a finished preload, or an earlier job)
    wait_s: float = 0.0        # how long the caller waited for them (joining a preload in flight, or separating)

    @property
    def vocals_path(self) -> str:
        return os.path.join(self.folder, "vocals.npy")

    @property
    def background_path(self) -> str:
        return os.path.join(self.folder, "background.npy")

    def vocals(self) -> np.ndarray:
        return np.load(self.vocals_path, mmap_mode="r")

    def background(self) -> np.ndarray:
        return np.load(self.background_path, mmap_mode="r")

    def exists(self) -> bool:
        return os.path.isfile(self.vocals_path) and os.path.isfile(self.background_path)


@dataclass
class _Run:
    task: asyncio.Task[Stems] | None = None
    background: bool = True
    listeners: list[Callable[[str, float], None]] = field(default_factory=list)


_RUNS: dict[str, _Run] = {}      # file key -> separation in flight (loop thread)
_DONE: dict[str, Stems] = {}


def installed(data: config.DataDir) -> bool:
    return not components.off("music_removal") and os.path.isfile(speech.separator_path(data.root))


def _decode(path: str, folder: str) -> tuple[str, int, int]:
    """``<folder>/mix.npy``: float32 ``(channels, n)`` at 44.1 kHz, at most stereo; returns (path, channels, n)."""
    info = media.probe(path)
    channels = 1 if int(info.get("channels") or 0) == 1 else 2
    y = media.decode_pcm(path, SR, channels=channels)
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, "mix.npy")
    np.save(out, np.ascontiguousarray(y.reshape(channels, -1), dtype=np.float32))
    return out, channels, int(y.shape[-1])


def _drop_others(data: config.DataDir, keep: str) -> None:
    """One cached separation at a time: they are large."""
    try:
        names = os.listdir(data.tmp)
    except OSError:
        return
    for name in names:
        if name.startswith(SEP_PREFIX) and name != SEP_PREFIX + keep and name[len(SEP_PREFIX):] not in _RUNS:
            shutil.rmtree(os.path.join(data.tmp, name), ignore_errors=True)
            _DONE.pop(name[len(SEP_PREFIX):], None)


async def _separate(ctx: Context, data: config.DataDir, path: str, key: str, run: _Run) -> Stems:
    folder = os.path.join(data.tmp, SEP_PREFIX + key)
    sp = speech.get(ctx.runtime, data.root)
    t0 = time.time()

    def progress(stage: str, pct: float) -> None:
        for cb in list(run.listeners):
            cb(stage, pct)

    try:
        await aio.run_io(_drop_others, data, key)
        mix, channels, frames = await aio.run_io(_decode, path, folder)
        res = await sp.separate_vocals(mix, os.path.join(folder, "vocals.npy"), frames / SR, progress,
                                       background_out=os.path.join(folder, "background.npy"),
                                       tag=key if run.background else None)
        await aio.run_io(os.remove, mix)
        stems = Stems(folder, channels, frames, str(res.get("device") or ""), float(res.get("elapsed_s") or 0.0),
                      round(time.time() - t0, 2))
        _DONE[key] = stems
        log.info("separated %s: %.1fs of %d-channel audio in %.1fs on %s (%.1fs with decoding and loading)",
                 os.path.basename(path), frames / SR, channels, stems.elapsed_s, stems.device, stems.total_s)
        return stems
    except BaseException:
        await asyncio.shield(aio.run_io(shutil.rmtree, folder, True))
        raise
    finally:
        _RUNS.pop(key, None)
        sp.background_tags.discard(key)


async def separate(ctx: Context, data: config.DataDir, path: str, progress: Callable[[str, float], None] | None = None,
                   background: bool = False) -> Stems:
    """Cached, joined from a run in flight, or separated now. A ``background`` preload may be stopped by another GPU
    job; a job that was waiting for it starts it again."""
    t0 = time.time()
    key = await aio.run_io(file_key, path)
    done = _DONE.get(key)
    if done is not None and await aio.run_io(done.exists):
        return Stems(**{**done.__dict__, "cached": True, "wait_s": 0.0})
    run = _RUNS.get(key)
    joined = run is not None
    if run is None:
        run = _Run(background=background)
        _RUNS[key] = run
        if background:
            speech.get(ctx.runtime, data.root).background_tags.add(key)
        run.task = aio.spawn(_separate(ctx, data, path, key, run), name=f"redub-separate-{key}")
    elif not background and run.background:
        run.background = False     # a job waits for it now: no longer a preload another job may stop
        speech.get(ctx.runtime, data.root).background_tags.discard(key)
    if progress is not None:
        run.listeners.append(progress)
    assert run.task is not None
    try:
        stems = await asyncio.shield(run.task)
    except speech.Cancelled:
        if background or not joined:
            raise
        return await separate(ctx, data, path, progress, background=False)   # the preload was stopped: run it now
    finally:
        if progress is not None and progress in run.listeners:
            run.listeners.remove(progress)
    return Stems(**{**stems.__dict__, "cached": False, "wait_s": round(time.time() - t0, 2)})


def forget(key: str) -> None:
    _DONE.pop(key, None)


def write_mix_wav(path: str, x: np.ndarray, sr: int = SR) -> str:
    """``x`` (channels, n) as a 32-bit float WAV (no rounding before the encoder)."""
    import soundfile as sf  # noqa: PLC0415
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = x.reshape(1, -1) if x.ndim == 1 else x
    sf.write(path, np.ascontiguousarray(data.T, dtype=np.float32), sr, subtype="FLOAT", format="WAV")
    return path


def _audio_start(path: str) -> float:
    """The start time of the source's first audio stream (the decoded samples begin there), 0 when unknown."""
    exe = media.ffprobe_path()
    if not exe:
        return 0.0
    res = media._run([exe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=start_time",
                      "-of", "default=nw=1:nk=1", path], 60)
    try:
        return float(res.stdout.decode("utf-8", errors="replace").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0.0


def mux(source: str, audio: str, dest: str, container: str, language: str | None = None) -> dict[str, Any]:
    """Copy the picture of ``source`` (with subtitles, attachments, chapters where the container takes them) and
    encode ``audio`` as the only audio track. Blocking."""
    exe = media.require_ffmpeg()
    _, _, encoders, _ = media._ffmpeg_lists()
    choices = [c for c in _CONTAINERS.get(container, _CONTAINERS[FALLBACK_CONTAINER]) if c[0] in encoders]
    if not choices:
        raise EngineError("not_ready", f"this ffmpeg build has no audio encoder for {container.upper()} video")
    encoder, args = choices[0]
    start = _audio_start(source)
    offset = ["-itsoffset", f"{start:.6f}"] if abs(start) > 0.0005 else []
    tag = ["-metadata:s:a:0", f"language={language}"] if language else []
    fast = ["-movflags", "+faststart"] if container in _FASTSTART else []
    part = f"{dest}.part.{container}"
    last = ""
    # First with subtitles, attachments and chapters; if the container refuses one of them, the picture alone.
    for extra in (["-map", "0:s?", "-map", "0:t?"], []):
        cmd = [exe, "-nostdin", "-v", "error", "-y", "-i", source, *offset, "-i", audio,
               "-map", "0:v", *extra, "-map", "1:a:0", "-map_metadata", "0", "-map_chapters", "0" if extra else "-1",
               "-c", "copy", "-c:a", encoder, *args, "-ar", str(VIDEO_AUDIO_SR), *tag, "-disposition:a:0", "default", *fast, part]
        res = media._run(cmd)
        if res.returncode == 0 and os.path.isfile(part):
            os.replace(part, dest)
            return {"audio_codec": encoder, "faststart": bool(fast)}
        last = media._tail(res.stderr)
        log.warning("mux %s failed (%s)%s", os.path.basename(dest), last, "; trying the picture alone" if extra else "")
        if os.path.exists(part):
            os.remove(part)
    raise EngineError("internal", f"Could not write the video ({last}).")
