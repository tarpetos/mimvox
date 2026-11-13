"""``redub.probe`` / ``redub.prepare`` / ``redub.transcribe`` / ``redub.dub`` / ``redub.render`` / ``redub.cancel``
(see contract/engine-protocol.md).

The job-starting handlers are ``async`` (they run on the engine loop, so taking the one GPU slot is atomic); job bodies
are coroutines that await the speech worker, the TTS backend and the RVC worker. ``redub.cancel`` cancels the job task.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
import shutil
import time
from typing import Any

import numpy as np

from .. import accent, aio, backends, config, dubmix, media, reflang, results, rvc, speech, tts, voices, zsvc
from .. import audio_util as au
from ..registry import Context, EngineError, method
from . import render, transcribe, video
from . import langs

log = logging.getLogger("mimvox.redub")

_LANG = re.compile(r"^[a-z]{2,3}$")
MAX_SEGMENTS = 5000


def _path(params: dict[str, Any]) -> str:
    p = params.get("path")
    if not isinstance(p, str) or not p:
        raise EngineError("bad_request", "'path' must be a file path")
    p = os.path.abspath(p)
    if not os.path.isfile(p):
        raise EngineError("not_found", f"file not found: {p}")
    return p


def _num(params: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    v = params.get(key, default)
    if v is None:
        v = default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise EngineError("bad_request", f"'{key}' must be a number")
    return min(hi, max(lo, float(v)))


def _models_not_loading() -> None:
    st = tts.models_status()
    if "loading" in st.values() or "paused" in st.values():
        raise EngineError("not_ready", "the models are loading, or the GPU is in use by a training; try again soon")


def _reg_job(ctx: Context, fn: Any, error_event: str = "redub.error", speech_job: bool = False) -> str:
    """Start an exclusive GPU job (``async fn(job, finish)``) whose failures become ``redub.error`` and whose slot is
    freed before the end event. A job that does not use the speech worker (``speech_job=False``) first has it hand its
    GPU memory back."""

    async def run(job: str) -> None:
        def finish(event: str, payload: dict[str, Any]) -> None:
            tts.release_job(job)  # free the slot before the app can see the end of this job
            ctx.emit(event, payload, job=job)

        try:
            if not speech_job:
                await speech.yield_gpu()
            await fn(job, finish)
        except (asyncio.CancelledError, transcribe.Cancelled):
            finish(error_event, {"code": "cancelled", "message": "cancelled"})
        except EngineError as exc:
            finish(error_event, {"code": exc.code, "message": exc.message})
        except speech.SpeechError as exc:
            err = exc.engine_error()
            finish(error_event, {"code": err.code, "message": err.message})
        except Exception as exc:  # noqa: BLE001
            log.exception("redub job %s failed", job)
            finish(error_event, {"code": "internal", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            tts.release_job(job)

    return tts.start_exclusive(ctx, run)


@method("redub.probe")
def redub_probe(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    path = _path(params)
    info = media.probe(path)
    out = {k: info[k] for k in ("duration_s", "sample_rate", "channels", "has_audio", "has_video", "format")}
    out["preview_path"] = _preview(config.data_dir(ctx), path) if info["has_audio"] else None
    return out


def _preview(data: config.DataDir, path: str) -> str | None:
    """The source decoded to mono 16-bit 44.1 kHz WAV in <data>/tmp (the app plays only WAV); cached per file state."""
    dst = os.path.join(data.tmp, f"redub_preview_{video.file_key(path)}.wav")
    if not os.path.isfile(dst):
        part = dst + ".part.wav"
        try:
            media.decode_to_wav(path, part, render.SR)
            os.replace(part, dst)
        except EngineError:
            return None
        finally:
            if os.path.exists(part):
                os.remove(part)
    return dst


@method("redub.cancel")
async def redub_cancel(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    job = params.get("job")
    if not isinstance(job, str):
        raise EngineError("bad_request", "'job' is required")
    ctx.cancel(job)
    return {}


# ------------------------------------------------------------------------------------------------ transcribe / dub

_MODE_NOTE = "Run setup (setup.sh / setup.cmd) to download it."


def _language(params: dict[str, Any], key: str, default: str = "auto") -> str | None:
    """A language parameter: ``"auto"`` -> None, else a normalised two/three letter code."""
    lang = params.get(key, default)
    lang = default if lang in (None, "") else lang
    if not isinstance(lang, str) or (lang != "auto" and not _LANG.match(lang.lower())):
        raise EngineError("bad_request", f"'{key}' must be \"auto\" or a language code such as \"en\"")
    return None if lang == "auto" else langs.norm(lang)


def target_language(value: Any, key: str) -> str:
    """A translation target: a language both the translation model and the speech model know (``bad_request``
    otherwise; the message lists them)."""
    if not isinstance(value, str) or not _LANG.match(langs.norm(value)):
        raise EngineError("bad_request", f"'{key}' must be a language code such as \"en\"")
    code = langs.norm(value)
    if langs.nllb_code(code) is None or backends.for_language(code) is None:
        ok = ", ".join(sorted(c for c in backends.languages() if langs.nllb_code(c)))
        raise EngineError("bad_request", f"'{key}': cannot translate to {code!r} (targets: {ok})")
    return code


def _translate_to(params: dict[str, Any]) -> str | None:
    to = params.get("translate_to")
    legacy = params.get("translate_to_en", False)  # the old spelling
    if not isinstance(legacy, bool):
        raise EngineError("bad_request", "'translate_to_en' must be true or false")
    if to in (None, ""):
        return "en" if legacy else None
    return target_language(to, "translate_to")


def _translator(params: dict[str, Any]) -> str | None:
    """``translator``: ``"nllb-1.3b"`` (better), ``"nllb-600m"`` (faster) or absent / null (the best installed). A
    translator that is not installed falls back to the other one."""
    t = params.get("translator")
    if t in (None, ""):
        return None
    if t not in langs.TRANSLATORS:
        raise EngineError("bad_request", f"'translator' must be one of {', '.join(langs.TRANSLATORS)}")
    return str(t)


def _check_speech_ready(data: config.DataDir, translate: bool) -> None:
    _models_not_loading()
    if importlib.util.find_spec("whisper") is None:
        raise EngineError("not_ready", "openai-whisper is not installed in the engine environment")
    if translate:
        models = os.path.join(data.root, "runtime", "models")
        if not langs.installed_translators(models):
            folder = os.path.join(models, langs.NLLB_DIR_NAME)
            raise EngineError("not_ready", f"The translation model is not installed ({folder}). {_MODE_NOTE}")
        for mod in ("transformers", "sentencepiece"):
            if importlib.util.find_spec(mod) is None:
                raise EngineError("not_ready", f"'{mod}' is not installed in the engine environment (translation)")


@method("redub.prepare")
async def redub_prepare(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """Warm the speech worker (start it, load Whisper) and, for a video, split the file into voice and background, in
    the background; answers at once. Event ``redub.prepared``."""
    path = await aio.run_io(_path, params)
    data = config.data_dir(ctx)
    sep = params.get("separate")
    if sep is not None and not isinstance(sep, bool):
        raise EngineError("bad_request", "'separate' must be true or false")
    if sep is None:
        sep = await aio.run_io(lambda: bool(media.probe(path).get("has_video")))
    sep = sep and await aio.run_io(video.installed, data)
    warm = importlib.util.find_spec("whisper") is not None
    if "paused" in tts.models_status().values():
        return {"warming": False, "separating": False, "reason": "paused"}
    if tts.is_busy():
        return {"warming": False, "separating": False, "reason": "busy"}
    if not warm and not sep:
        return {"warming": False, "separating": False, "reason": "unavailable"}
    warm = warm and not _PREPARING[0]   # one Whisper preload at a time (a second file only adds its separation)
    if warm:
        _PREPARING[0] = True
    aio.spawn(_prepare(ctx, data, path, sep, warm), name="redub-prepare")
    return {"warming": True, "separating": sep}


async def _prepare(ctx: Context, data: config.DataDir, path: str, separate: bool, warm: bool) -> None:
    t0 = time.time()
    out: dict[str, Any] = {"ok": True}
    try:
        # The speech model may still be loading at engine start: do not compete with it (an awaited event, 3 min max).
        await tts.wait_settled(180.0)
        if "paused" in tts.models_status().values():
            ctx.emit("redub.prepared", {"ok": False, "message": "the GPU is in use by a training"})
            return
        if separate:
            try:
                stems = await video.separate(ctx, data, path, background=True)
                out.update(separated=True, separation_s=stems.total_s, separation_device=stems.device)
            except speech.Cancelled:
                out.update(separated=False, separation_note="the separation gave way to a GPU job")
            except speech.SpeechError as exc:
                log.warning("preload separation failed: %s", exc.message)
                out.update(separated=False, separation_note=exc.message)
        if warm:
            res = await speech.get(ctx.runtime, data.root).prepare()
            out.update(device=res.get("device"), model=res.get("model"))
        out["elapsed_s"] = round(time.time() - t0, 2)
        ctx.emit("redub.prepared", out)
    except speech.Cancelled:
        ctx.emit("redub.prepared", {**out, "ok": False, "message": "the preload gave way to a GPU job"})
    except Exception as exc:  # noqa: BLE001
        log.warning("prepare failed: %s", exc)
        ctx.emit("redub.prepared", {**out, "ok": False, "message": f"{type(exc).__name__}: {exc}"})
    finally:
        if warm:
            _PREPARING[0] = False


_PREPARING = [False]   # loop-thread state


@method("redub.transcribe")
async def redub_transcribe(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    path = await aio.run_io(_path, params)
    lang = _language(params, "language")
    translate_to = _translate_to(params)
    translator = _translator(params)
    if translate_to and lang == translate_to:
        translate_to = None  # already in that language
    data = config.data_dir(ctx)
    await aio.run_io(_check_speech_ready, data, translate_to is not None)

    async def work(job: str, finish: Any) -> None:
        ranges = ({"decode": (0, 5), "load_model": (5, 25), "transcribe": (25, 80), "translate": (80, 100)}
                  if translate_to else {"decode": (0, 5), "load_model": (5, 30), "transcribe": (30, 100)})
        emit = transcribe.Emit(ctx, job, ranges)
        t0 = time.time()
        workdir = os.path.join(data.tmp, job)
        try:
            result = await transcribe.run(path, lang, translate_to, workdir, emit, speech.get(ctx.runtime, data.root),
                                          translator)
        finally:
            await aio.run_io(shutil.rmtree, workdir, ignore_errors=True)
        log.info("job %s: transcript with %d segment(s) in %.1fs", job, len(result["segments"]), time.time() - t0)
        finish("redub.transcript", result)

    return {"job": _reg_job(ctx, work, speech_job=True)}


def _segments(params: dict[str, Any]) -> list[dict[str, Any]]:
    raw = params.get("segments")
    if not isinstance(raw, list) or not raw:
        raise EngineError("bad_request", "respeak needs 'segments' (run redub.transcribe first)")
    if len(raw) > MAX_SEGMENTS:
        raise EngineError("bad_request", f"too many segments (limit {MAX_SEGMENTS})")
    out = []
    for k, s in enumerate(raw):
        if not isinstance(s, dict):
            raise EngineError("bad_request", f"segments[{k}] must be an object")
        try:
            start, end = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            raise EngineError("bad_request", f"segments[{k}] needs numeric 'start' and 'end'") from None
        text = s.get("text")
        if not isinstance(text, str):
            raise EngineError("bad_request", f"segments[{k}] needs a 'text' string")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue  # an emptied segment stays silent
        if start < 0 or end < start:
            raise EngineError("bad_request", f"segments[{k}]: need 0 <= start <= end")
        out.append({"start": start, "end": end, "text": text})
    if not out:
        raise EngineError("bad_request", "all segments are empty")
    out.sort(key=lambda s: s["start"])
    return out


_LANG_NAMES = {"uk": "Ukrainian", "ru": "Russian", "de": "German", "fr": "French", "pl": "Polish", "it": "Italian",
               "pt": "Portuguese", "tr": "Turkish", "ko": "Korean", "nl": "Dutch"}


def _speak_language(params: dict[str, Any], segs: list[dict[str, Any]]) -> str:
    """The language the segment texts are spoken in. ``translated: true`` means they are translations into
    ``target_language`` (default English); else ``language`` (the transcript's language) or, when absent, a guess from
    the script. A language no backend speaks is refused instead of producing garbage."""
    translated = params.get("translated", False)
    if not isinstance(translated, bool):
        raise EngineError("bad_request", "'translated' must be true or false")
    if translated:
        target = params.get("target_language")
        return "en" if target in (None, "") else target_language(target, "target_language")
    code = _language(params, "language")
    if code is None:
        code = langs.script_language_guess(" ".join(s["text"] for s in segs)) or "en"
    if backends.for_language(code) is None:
        name = _LANG_NAMES.get(code, f"the language {code!r}")
        spoken = ", ".join(sorted(backends.languages()))
        raise EngineError("bad_request", f"The voice cannot speak {name} yet (it speaks {spoken}). Translate the "
                                         "text first: transcribe with translate_to (or use redub.dub), then render "
                                         "with translated: true and target_language.")
    return code


_STATS = ("timing", "stretched", "overrun", "pushed", "drift_s", "max_push_s")   # respeak stats in redub.done


# ------------------------------------------------------------------------------------------------ video output

_NO_SEPARATOR = ("The music-removal model is not installed (run setup again), so the background could not be kept: "
                 "the result has the new voice only.")


def ranges(parts: list[tuple[str, float]]) -> dict[str, tuple[float, float]]:
    """``Emit`` ranges from (stage, weight) pairs in order: the weights shared out over 0-100."""
    total = sum(w for _, w in parts) or 1.0
    out, acc = {}, 0.0
    for name, w in parts:
        out[name] = (100.0 * acc / total, 100.0 * (acc + w) / total)
        acc += w
    return out


def _out_stages(out: video.Output, separate: bool) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """(stages before, stages after) the voice work of a job with this output, with their progress weights."""
    before = [("separate", 7.0)] if separate else []
    after = ([("mix", 2.0)] if out.mixes else []) + [("save", 2.0)] + ([("mux", 4.0)] if out.video else [])
    return before, after


async def _stems(ctx: Context, data: config.DataDir, path: str, emit: transcribe.Emit,
                 need: bool) -> tuple[video.Stems | None, str | None]:
    """The voice / background stems a job needs (``need``), from the cache or separated now, and a note for the
    result when they cannot be had (model missing, separation failed: the job goes on without them)."""
    if not need:
        return None, None
    emit.stage("separate", 0)
    if not await aio.run_io(video.installed, data):
        return None, _NO_SEPARATOR
    try:
        stems = await video.separate(ctx, data, path, lambda st, pct: emit.stage("separate", pct if st == "separate" else 0))
    except speech.SpeechError as exc:
        log.warning("separation failed: %s", exc.message)
        return None, f"The background could not be separated ({exc.message}): the result has the new voice only."
    emit.stage("separate", 100)
    return stems, None


def _mix(track: np.ndarray, stems: video.Stems | None, keep_background: bool) -> tuple[np.ndarray, dict[str, Any]]:
    background = np.asarray(stems.background()) if stems is not None and keep_background else None
    vocals = np.asarray(stems.vocals()) if stems is not None else None
    return dubmix.mix(track, render.SR, background, vocals)


async def _finish(data: config.DataDir, path: str, track: np.ndarray, stems: video.Stems | None, out: video.Output,
                  out_fmt: str, emit: transcribe.Emit, workdir: str, language: str | None,
                  note: str | None) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """Mix (``out.mixes``: the loudness of the original voice, the background when kept, no clipping), save the audio
    result in ``out_fmt`` and, for a video output, mux it into a copy of the video next to it. Returns (the saved
    audio's fields, the final audio, the video fields of ``redub.done``)."""
    fields: dict[str, Any] = {"output": out.output, "video_path": None,
                              "keep_background": stems is not None and out.keep_background}
    if note:
        fields["background_note"] = note
    if stems is not None:
        fields.update(separation_s=stems.wait_s, separation_cached=stems.cached, separation_device=stems.device)
    final = track
    if out.mixes:
        emit.stage("mix", 0)
        t0 = time.time()
        final, report = await aio.run_io(_mix, track, stems, out.keep_background)
        fields.update(mix_s=round(time.time() - t0, 2), mix=report)
    emit.stage("save", 0)
    saved = await aio.run_io(media.save_result, data, final, render.SR, "redub_", out_fmt,
                             (out.container,) if out.video else ())
    if out.video:
        emit.stage("mux", 0)
        t0 = time.time()
        wav = await aio.run_io(video.write_mix_wav, os.path.join(workdir, "mix.wav"), final)
        dest = os.path.splitext(saved["path"])[0] + "." + out.container
        info = await aio.run_io(video.mux, path, wav, dest, out.container, video.language_tag(language))
        fields.update(video_path=dest, mux_s=round(time.time() - t0, 2), video_audio_codec=info["audio_codec"])
        emit.stage("mux", 100)
    else:
        emit.stage("save", 100)
    return saved, final, fields


def _record(saved: dict[str, Any], voice: "_Voice", source: str, mode: str, duration_s: float, elapsed_s: float,
            vfields: dict[str, Any], language: str | None = None, to: str | None = None,
            accent_name: str | None = None) -> None:
    """IO thread: the sidecar of a saved Redub result (results.py), so it comes back after a restart."""
    results.record(saved, {"kind": "redub", "voice_id": voice.voice_id, "voice_name": voice.name, "mode": mode,
                           "source_name": os.path.basename(source), "language": language, "to": to,
                           "accent": accent_name, "duration_s": duration_s, "elapsed_s": elapsed_s,
                           "video_path": vfields.get("video_path")})


class _Voice:
    """The voice-side settings shared by ``redub.render`` and ``redub.dub``."""

    def __init__(self, ctx: Context, data: config.DataDir, params: dict[str, Any]) -> None:
        self.voice_id = str(params.get("voice_id") or "")
        meta = voices.read_meta(data, self.voice_id)  # not_found
        self.language = meta.get("language") or "en"
        self.name = str(meta.get("name") or self.voice_id)
        quality = params.get("quality") or "best"
        if quality not in ("best", "fast"):
            raise EngineError("bad_request", "'quality' must be \"best\" or \"fast\"")
        self.temperature = _num(params, "expressiveness", 0.8, 0.1, 1.5)
        self.index_rate = _num(params, "voice_match", 1.0, 0.0, 1.0)
        self.speed = _num(params, "speed", 1.0, tts.SPEED_MIN, tts.SPEED_MAX)  # respeak only
        self.timing = params.get("timing") or "keep"                             # respeak / dub only
        if self.timing not in render.TIMINGS:
            raise EngineError("bad_request", "'timing' must be \"keep\" or \"natural\"")
        self.pth, self.idx = data.rvc_model(self.voice_id)
        self.trained = os.path.isfile(self.pth)
        self.ref = data.reference(self.voice_id)
        self.quality = quality
        self.ctx = ctx
        self.accent = accent.parse(params)              # respeak / dub only
        self.native_voice = accent.parse_voice(params)
        # The native accent of an untrained voice (or voice_match 0) uses the zero-shot converter, when installed.
        self.zs_worker = zsvc.get(ctx.runtime, data.root) if self.accent == "native-us" else None

    def index_rate_for(self, lang: str) -> float:
        """RVC index rate for speech in ``lang``: capped for languages the voice was not recorded in (rvc.py)."""
        return rvc.index_rate_for(self.index_rate, lang, self.language)

    def accent_for(self, lang: str) -> accent.Choice:
        """The accent speech in ``lang`` gets (``native-us`` needs English and RVC or the zero-shot converter, else
        ``own``; accent.py)."""
        return accent.choose(self.accent, self.native_voice, lang, self.trained, self.index_rate,
                             self.zs_worker is not None)

    async def ref_fields(self, choice: accent.Choice, lang: str) -> dict[str, Any]:
        """``reference_language`` / ``reference_foreign`` / ``reference_note`` of speech in ``lang`` (reflang.py)."""
        return await aio.run_io(reflang.fields, self.ref, lang, choice.accent == "native-us")

    async def speak_setup(self, choice: accent.Choice, lang: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """(extra ``redub.done`` fields, ``render.respeak`` keyword arguments) for speech in ``lang``: the voice's
        own reference, or the native clip with its transcript, RVC pitch and index rate (or the zero-shot converter,
        ``zs``, bound to the voice's reference)."""
        native = (await aio.run_io(accent.prepare, choice, self.ref, self.index_rate)
                  if choice.accent == "native-us" else None)
        if native is None:
            if choice.accent == "native-us":
                choice = accent.Choice("own", choice.native_voice, "unavailable")
            return choice.fields(), {"ref": self.ref, "index_rate": self.index_rate_for(lang)}
        how: dict[str, Any] = {"ref": native.ref.path, "ref_text": native.ref.text, "pitch": native.pitch,
                               "index_rate": native.index_rate}
        if native.converter == "zero-shot":
            how["zs"] = tts.zero_shot(self.ctx, choice, self.zs_worker, self.ref, self.quality)
        return {**choice.fields(), **native.fields()}, how

    def respeak_worker(self, lang: str = "en") -> rvc.RvcWorker | None:
        """Checks that a respeak can start (``not_found`` / ``not_ready``) and returns the RVC worker it needs, if any."""
        st = tts.models_status()
        if not os.path.isfile(self.ref):
            raise EngineError("not_found", "this voice has no recording yet (reference.wav is missing)")
        backend = backends.for_language(lang)
        if backend is None:
            raise EngineError("bad_request", f"no voice model speaks {lang!r}")
        try:
            tts.check_backend_ready(backend)
        except EngineError:
            raise EngineError("not_ready", "the speech model is not ready (loading, or the GPU is in use by a "
                                           "training)") from None
        if self.trained and self.index_rate > 0:
            worker = rvc.get(self.ctx.runtime)
            if worker is None or st["rvc"] != "ready":
                raise EngineError("not_ready", "the voice-conversion model is not ready")
            return worker
        return None


@method("redub.dub")
async def redub_dub(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """The whole pipeline in one job: transcribe -> translate every segment -> (review stop | speak and save)."""
    path = await aio.run_io(_path, params)
    data = config.data_dir(ctx)
    lang = _language(params, "source_language")
    target = params.get("target_language")
    target = "en" if target in (None, "") else target_language(target, "target_language")
    review = params.get("review", False)
    if not isinstance(review, bool):
        raise EngineError("bad_request", "'review' must be true or false")
    translator = _translator(params)
    voice = await aio.run_io(_Voice, ctx, data, params)
    out_fmt = await aio.run_io(media.output_format, params)
    out = await aio.run_io(video.options, params, path)
    forced = out.video and voice.timing != "keep"   # a video keeps the original timing (the picture does not wait)
    if out.video:
        voice.timing = "keep"
    worker = None if review else voice.respeak_worker(target)  # a review stop needs no voice models yet
    choice = voice.accent_for(target)
    await aio.run_io(_check_speech_ready, data, lang != target)

    async def work(job: str, finish: Any) -> None:
        t0 = time.time()
        separate = not review and out.keep_background
        before, after = _out_stages(out, separate)
        stages = ({"decode": (0, 5), "load_model": (5, 25), "transcribe": (25, 80), "translate": (80, 100)} if review
                  else ranges([("decode", 3.0), ("load_model", 5.0), ("transcribe", 16.0), ("translate", 6.0),
                               *before, ("speak", 58.0), *after]))
        emit = transcribe.Emit(ctx, job, stages)
        sp = speech.get(ctx.runtime, data.root)
        workdir = os.path.join(data.tmp, job)
        await aio.run_io(os.makedirs, workdir, exist_ok=True)
        try:
            res = await transcribe.run(path, lang, target, workdir, emit, sp, translator)
            segs = res["segments"]
            if not segs:
                raise EngineError("bad_request", "No speech was found in this file.")
            res["review"] = review
            res.setdefault("target_language", target)
            if review:
                res["timings_s"] = emit.timings()
                log.info("job %s: review transcript, %d segment(s) in %.1fs %s", job, len(segs), time.time() - t0,
                         res["timings_s"])
                finish("redub.transcript", res)
                return
            stems, note = await _stems(ctx, data, path, emit, separate)   # usually ready (redub.prepare)
            await sp.release_gpu()  # Whisper / NLLB / Demucs leave the GPU to the speech model and RVC
            # The text is in `target` now (translated, or the speech already was in that language).

            def progress(stage: str, pct: float, **extra: Any) -> None:
                if stage == "speak":
                    emit.stage("speak", (pct - 5.0) / 0.9, **extra)  # render reports speak as 5-95

            emit.stage("speak", 0)
            speak_segs = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs]
            accent_fields, how = await voice.speak_setup(choice, target)
            track, stats = await render.respeak(speak_segs, float(res["duration_s"]), how["ref"], voice.temperature,
                                                voice.quality, worker, voice.pth, voice.idx, how["index_rate"],
                                                workdir, progress, voice.speed, lang=target,
                                                ref_text=how.get("ref_text"), pitch=how.get("pitch", 0),
                                                zs=how.get("zs"), timing=voice.timing)
            accent_fields.update(await voice.ref_fields(accent.Choice(accent_fields["accent"]), target))
            saved, final, vfields = await _finish(data, path, track, stems, out, out_fmt, emit, workdir, target, note)
            timings = emit.timings()
            elapsed = round(time.time() - t0, 2)
            log.info("job %s (dub): %.1fs of audio in %.1fs %s%s", job, res["duration_s"], elapsed, timings,
                     f" -> {vfields['video_path']}" if vfields["video_path"] else "")
            await aio.run_io(_record, saved, voice, path, "dub", au.duration_s(final, render.SR), elapsed, vfields,
                             res["language"], target, accent_fields.get("accent"))
            finish("redub.done", {**saved, "duration_s": au.duration_s(final, render.SR),
                                  "elapsed_s": elapsed, "mode": "dub", "language": res["language"],
                                  "target_language": target, "translated": res["translated"], "segments": segs,
                                  "timings_s": timings, **accent_fields,
                                  **{k: stats[k] for k in _STATS}, **vfields, "timing_forced": forced})
        finally:
            await aio.run_io(shutil.rmtree, workdir, ignore_errors=True)

    start = {} if review else {**choice.fields(), **await voice.ref_fields(choice, target)}
    return {"job": _reg_job(ctx, work, speech_job=True), **start}


@method("redub.render")
async def redub_render(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    path = await aio.run_io(_path, params)
    data = config.data_dir(ctx)
    mode = params.get("mode")
    if mode not in ("convert", "respeak"):
        raise EngineError("bad_request", "'mode' must be \"convert\" or \"respeak\"")
    # The finish of a reviewed Translate & dub is a respeak the app still calls a dub (recorded so in the sidecar).
    result_mode = "dub" if mode == "respeak" and params.get("result_mode") == "dub" else mode
    voice = await aio.run_io(_Voice, ctx, data, params)
    out_fmt = await aio.run_io(media.output_format, params)
    out = await aio.run_io(video.options, params, path)
    forced = mode == "respeak" and out.video and voice.timing != "keep"
    if out.video:
        voice.timing = "keep"
    # Convert runs RVC over the voice alone whenever the result is a video or keeps the background (RVC over music
    # garbles it); re-speak only needs the background when it is kept.
    separate = out.keep_background or (mode == "convert" and out.video)
    pth, idx, trained, index_rate = voice.pth, voice.idx, voice.trained, voice.index_rate
    segs = _segments(params) if mode == "respeak" else []
    lang = _speak_language(params, segs) if mode == "respeak" else "en"
    st = tts.models_status()
    worker: rvc.RvcWorker | None = None
    if mode == "convert":
        if not trained:
            raise EngineError("bad_request", "This voice has no trained voice-conversion model, so a file cannot be "
                                             "converted to it. Train the voice first, or use mode \"respeak\".")
        worker = rvc.get(ctx.runtime)
        if worker is None or st["rvc"] != "ready":
            raise EngineError("not_ready", "the voice-conversion model is not ready (loading, or the GPU is in use "
                                           "by a training)")
    else:
        worker = voice.respeak_worker(lang)
    choice = voice.accent_for(lang) if mode == "respeak" else accent.Choice("own")

    async def work(job: str, finish: Any) -> None:
        t0 = time.time()
        workdir = os.path.join(data.tmp, job)
        await aio.run_io(os.makedirs, workdir, exist_ok=True)
        before, after = _out_stages(out, separate)
        emit = transcribe.Emit(ctx, job, ranges([("decode", 3.0), *before, (mode if mode == "convert" else "speak", 80.0),
                                                 *after]))

        def progress(stage: str, pct: float, **extra: Any) -> None:
            if stage in ("convert", "speak"):
                emit.stage(stage, (pct - 5.0) / 0.9, **extra)   # render reports these as 5-95

        try:
            emit.stage("decode", 0)
            source, _ = await aio.run_io(media.read_audio, path, sr=render.SR)
            if len(source) < render.SR * 0.2:
                raise EngineError("bad_request", "This file has no usable audio.")
            total_s = len(source) / float(render.SR)
            emit.stage("decode", 100)
            stems, note = await _stems(ctx, data, path, emit, separate)
            if separate:
                await speech.yield_gpu()   # Demucs leaves the GPU to RVC / the speech model
            if mode == "convert":
                assert worker is not None
                if stems is not None:   # the voice alone: the background is added back by the mix
                    source = await aio.run_io(lambda: np.asarray(stems.vocals(), dtype=np.float32).mean(axis=0))
                elif separate:
                    note = ("The voice could not be separated from the background, so the whole sound track was "
                            "converted, music included." + (" Run setup again to install the music-removal model."
                                                            if note == _NO_SEPARATOR else ""))
                track = await render.convert(source, workdir, worker, pth, idx, index_rate, progress)
                extra: dict[str, Any] = {}
            else:
                accent_fields, how = await voice.speak_setup(choice, lang)
                track, stats = await render.respeak(segs, total_s, how["ref"], voice.temperature, voice.quality,
                                                    worker, pth, idx, how["index_rate"], workdir, progress,
                                                    voice.speed, lang=lang, ref_text=how.get("ref_text"),
                                                    pitch=how.get("pitch", 0), zs=how.get("zs"), timing=voice.timing)
                extra = {k: stats[k] for k in _STATS}
                extra["language"] = lang
                extra.update(accent_fields)
                extra.update(await voice.ref_fields(accent.Choice(accent_fields["accent"]), lang))
                extra["timing_forced"] = forced
            saved, final, vfields = await _finish(data, path, track, stems, out, out_fmt, emit, workdir,
                                                  lang if mode == "respeak" else None, note)
            elapsed = round(time.time() - t0, 2)
            timings = emit.timings()
            log.info("job %s (%s): %.1fs of audio in %.1fs %s%s", job, mode, total_s, elapsed, timings,
                     f" -> {vfields['video_path']}" if vfields["video_path"] else "")
            spoken = lang if mode == "respeak" else None
            await aio.run_io(_record, saved, voice, path, result_mode, au.duration_s(final, render.SR), elapsed,
                             vfields, spoken, spoken, extra.get("accent"))
            finish("redub.done", {**saved, "duration_s": au.duration_s(final, render.SR),
                                  "elapsed_s": elapsed, "mode": mode, "timings_s": timings, **extra, **vfields})
        finally:
            await aio.run_io(shutil.rmtree, workdir, ignore_errors=True)

    start = {**choice.fields(), **await voice.ref_fields(choice, lang)} if mode == "respeak" else {}
    return {"job": _reg_job(ctx, work, speech_job=separate), **start}
