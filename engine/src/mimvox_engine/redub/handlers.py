"""``redub.probe`` / ``redub.prepare`` / ``redub.transcribe`` / ``redub.dub`` / ``redub.render`` / ``redub.cancel``
(see contract/engine-protocol.md).

The job-starting handlers are ``async`` (they run on the engine loop, so taking the one GPU slot is atomic); job bodies
are coroutines that await the speech worker, the TTS backend and the RVC worker. ``redub.cancel`` cancels the job task.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import logging
import os
import re
import shutil
import time
from typing import Any

from .. import accent, aio, backends, config, media, reflang, rvc, speech, tts, voices, zsvc
from .. import audio_util as au
from ..registry import Context, EngineError, method
from . import render, transcribe
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
    out = {k: info[k] for k in ("duration_s", "sample_rate", "channels", "has_audio", "format")}
    out["preview_path"] = _preview(config.data_dir(ctx), path) if info["has_audio"] else None
    return out


def _preview(data: config.DataDir, path: str) -> str | None:
    """The source decoded to mono 16-bit 44.1 kHz WAV in <data>/tmp (the app plays only WAV); cached per file state."""
    st = os.stat(path)
    key = hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8")).hexdigest()[:16]
    dst = os.path.join(data.tmp, f"redub_preview_{key}.wav")
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
    """Warm the speech worker (start it, load Whisper) in the background; answers at once. Event ``redub.prepared``."""
    await aio.run_io(_path, params)
    data = config.data_dir(ctx)
    if importlib.util.find_spec("whisper") is None:
        return {"warming": False, "reason": "unavailable"}
    if "paused" in tts.models_status().values():
        return {"warming": False, "reason": "paused"}
    if tts.is_busy():
        return {"warming": False, "reason": "busy"}
    if _PREPARING[0]:
        return {"warming": True}
    _PREPARING[0] = True
    aio.spawn(_prepare(ctx, data), name="redub-prepare")
    return {"warming": True}


async def _prepare(ctx: Context, data: config.DataDir) -> None:
    t0 = time.time()
    try:
        # The speech model may still be loading at engine start: do not compete with it (an awaited event, 3 min max).
        await tts.wait_settled(180.0)
        if "paused" in tts.models_status().values():
            ctx.emit("redub.prepared", {"ok": False, "message": "the GPU is in use by a training"})
            return
        res = await speech.get(ctx.runtime, data.root).prepare()
        ctx.emit("redub.prepared", {"ok": True, "device": res.get("device"), "model": res.get("model"),
                                    "elapsed_s": round(time.time() - t0, 2)})
    except speech.Cancelled:
        ctx.emit("redub.prepared", {"ok": False, "message": "the preload gave way to a GPU job"})
    except Exception as exc:  # noqa: BLE001
        log.warning("prepare failed: %s", exc)
        ctx.emit("redub.prepared", {"ok": False, "message": f"{type(exc).__name__}: {exc}"})
    finally:
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


class _Voice:
    """The voice-side settings shared by ``redub.render`` and ``redub.dub``."""

    def __init__(self, ctx: Context, data: config.DataDir, params: dict[str, Any]) -> None:
        self.voice_id = str(params.get("voice_id") or "")
        self.language = voices.read_meta(data, self.voice_id).get("language") or "en"  # not_found
        quality = params.get("quality") or "best"
        if quality not in ("best", "fast"):
            raise EngineError("bad_request", "'quality' must be \"best\" or \"fast\"")
        self.temperature = _num(params, "expressiveness", 0.8, 0.1, 1.5)
        self.index_rate = _num(params, "voice_match", 1.0, 0.0, 1.0)
        self.speed = _num(params, "speed", 1.0, tts.SPEED_MIN, tts.SPEED_MAX)  # respeak only
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
            how["zs"] = tts.zero_shot(self.ctx, choice, self.zs_worker, self.ref)
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
    worker = None if review else voice.respeak_worker(target)  # a review stop needs no voice models yet
    choice = voice.accent_for(target)
    await aio.run_io(_check_speech_ready, data, lang != target)

    async def work(job: str, finish: Any) -> None:
        t0 = time.time()
        ranges = ({"decode": (0, 5), "load_model": (5, 25), "transcribe": (25, 80), "translate": (80, 100)} if review
                  else {"decode": (0, 3), "load_model": (3, 8), "transcribe": (8, 24), "translate": (24, 30),
                        "speak": (30, 95), "save": (95, 100)})
        emit = transcribe.Emit(ctx, job, ranges)
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
            await sp.release_gpu()  # Whisper / NLLB leave the GPU to the speech model and RVC
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
                                                zs=how.get("zs"))
            accent_fields.update(await voice.ref_fields(accent.Choice(accent_fields["accent"]), target))
            emit.stage("save", 0)
            saved = await aio.run_io(media.save_result, data, track, render.SR, "redub_", out_fmt)
            emit.stage("save", 100)
            timings = emit.timings()
            elapsed = round(time.time() - t0, 2)
            log.info("job %s (dub): %.1fs of audio in %.1fs %s", job, res["duration_s"], elapsed, timings)
            finish("redub.done", {**saved, "duration_s": au.duration_s(track, render.SR),
                                  "elapsed_s": elapsed, "mode": "dub", "language": res["language"],
                                  "target_language": target, "translated": res["translated"], "segments": segs,
                                  "timings_s": timings, **accent_fields,
                                  **{k: stats[k] for k in ("stretched", "overrun", "pushed", "drift_s", "max_push_s")}})
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
    voice = await aio.run_io(_Voice, ctx, data, params)
    out_fmt = await aio.run_io(media.output_format, params)
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

        def progress(stage: str, pct: float, **extra: Any) -> None:
            ctx.emit("redub.progress", {"stage": stage, "pct": round(pct, 1), **extra}, job=job)

        try:
            progress("decode", 0)
            source, _ = await aio.run_io(media.read_audio, path, sr=render.SR)
            if len(source) < render.SR * 0.2:
                raise EngineError("bad_request", "This file has no usable audio.")
            total_s = len(source) / float(render.SR)
            progress("decode", 5)
            if mode == "convert":
                assert worker is not None
                track = await render.convert(source, workdir, worker, pth, idx, index_rate, progress)
                extra: dict[str, Any] = {}
            else:
                accent_fields, how = await voice.speak_setup(choice, lang)
                track, stats = await render.respeak(segs, total_s, how["ref"], voice.temperature, voice.quality,
                                                    worker, pth, idx, how["index_rate"], workdir, progress,
                                                    voice.speed, lang=lang, ref_text=how.get("ref_text"),
                                                    pitch=how.get("pitch", 0), zs=how.get("zs"))
                extra = {k: stats[k] for k in ("stretched", "overrun", "pushed", "drift_s", "max_push_s")}
                extra["language"] = lang
                extra.update(accent_fields)
                extra.update(await voice.ref_fields(accent.Choice(accent_fields["accent"]), lang))
            progress("save", 96)
            saved = await aio.run_io(media.save_result, data, track, render.SR, "redub_", out_fmt)
            elapsed = round(time.time() - t0, 2)
            log.info("job %s (%s): %.1fs of audio in %.1fs", job, mode, total_s, elapsed)
            finish("redub.done", {**saved, "duration_s": au.duration_s(track, render.SR),
                                  "elapsed_s": elapsed, "mode": mode, **extra})
        finally:
            await aio.run_io(shutil.rmtree, workdir, ignore_errors=True)

    start = {**choice.fields(), **await voice.ref_fields(choice, lang)} if mode == "respeak" else {}
    return {"job": _reg_job(ctx, work), **start}
