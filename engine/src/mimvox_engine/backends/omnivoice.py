"""OmniVoice backend (k2-fsa/OmniVoice): the speech model of the engine. Zero-shot voice cloning in ~60 well-covered
languages, English and Ukrainian among them.

The model runs in a persistent worker process (``omnivoice_worker.py``) started with the engine's own interpreter
(the engine venv, ``<data>/runtime/engine/.venv``: torch 2.8 + transformers 5), weights in
``<data>/runtime/models/omnivoice`` (``runtime.toml`` ``[paths] omnivoice_model``; ``[python] omnivoice`` may name
another interpreter). A process and not a module of the engine: a native crash cannot take the engine down, and
handing the GPU to a training is ending the process (all its VRAM and host RAM back at once). The worker is driven
through ``workerproc.WorkerClient``: one owner task, a reader task resolving futures, timers for timeouts; nothing here
blocks the engine loop or polls.

Measured on an RTX 4070 (12 GB, Windows 11), fp16, 32 steps: about 2.2 s per segment of up to ~10 s of speech
(RTF 0.2-0.33; 16 steps halves it), ~4 GB of VRAM with a 9 s reference (~7 GB with 19 s), ~2.2 GB of host RAM, 10-25 s to load.

Voice: the voice's ``reference.wav`` (used whole up to 20 s, else cut by the worker to 10-18 s of whole sentences;
transcript cached next to it in
``reference.omnivoice.json``). A per-language clip ``reference_<lang>.wav`` next to it wins when present: OmniVoice
carries the accent of the reference into the target language, so a Ukrainian clip makes the cleanest Ukrainian.

The voice's RVC model then converts the output like every backend's (pipeline). For languages other than the one the
RVC model was trained on, keep the index rate low (measured: index 1.0 garbles Ukrainian phonemes through an
English-only feature index; 0.3 keeps the words and most of the timbre): ``rvc.index_rate_for`` caps it, applied by
``tts.generate`` and Redub respeak from the voice's ``language`` (``voice.json``, default ``en``).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from .. import config
from ..registry import EngineError
from ..workerproc import WorkerCancelled, WorkerClient, WorkerDied, WorkerError, child_env
from . import SynthParams, TtsBackend

log = logging.getLogger("mimvox.omnivoice")

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnivoice_worker.py")

BEST_STEPS = int(os.environ.get("MIMVOX_OMNIVOICE_STEPS", "32"))
FAST_STEPS = int(os.environ.get("MIMVOX_OMNIVOICE_FAST_STEPS", "16"))
FIRST_STEPS = int(os.environ.get("MIMVOX_OMNIVOICE_FIRST_STEPS", "16"))  # segment 0: ~1.1 s instead of ~2.2 s
UNLOAD_AFTER_S = float(os.environ.get("MIMVOX_OMNIVOICE_UNLOAD_S", "0"))   # 0: keep the model loaded while idle

# "expressiveness" (0.1-1.5, default 0.8) -> OmniVoice's two sampling knobs:
# position_temperature (randomness of the order in which tokens are unmasked: phrasing, pauses; model default 5.0) and
# class_temperature (randomness of the tokens themselves; model default 0 = greedy). The default 0.8 gives the model's
# defaults; lower is flatter and steadier, higher adds variation (above ~1.2 words may start to slur).
EXPR_DEFAULT = 0.8
POS_TEMP_DEFAULT = 5.0


def sampling_for(expressiveness: float) -> dict[str, float]:
    e = max(0.1, min(1.5, float(expressiveness)))
    return {"position_temperature": round(POS_TEMP_DEFAULT * e / EXPR_DEFAULT, 3),
            "class_temperature": round(max(0.0, e - EXPR_DEFAULT), 3)}

SYNTH_TIMEOUT_S = 300.0   # one segment; includes a first-time reference transcription and a cold model load
LOAD_TIMEOUT_S = 300.0

# ISO 639-1 (Whisper / NLLB codes of redub/langs.py) -> OmniVoice language id, for the languages with at least ~100 h of
# OmniVoice training data (docs/languages.md of k2-fsa/OmniVoice; hours in the comments). The model knows 646, but
# below ~100 h quality drops; those are left out on purpose.
LANGS: dict[str, str] = {
    # >= 1000 h
    "en": "en", "zh": "zh", "ja": "ja", "es": "es", "fr": "fr", "de": "de", "ru": "ru", "pt": "pt", "yue": "yue",
    "th": "th", "it": "it", "ko": "ko", "vi": "vi", "id": "id", "no": "no", "ca": "ca", "hr": "hr", "lt": "lt",
    "sk": "sk", "sv": "sv", "el": "el", "nl": "nl", "bg": "bg", "sr": "sr", "uk": "uk",  # uk: 1852 h
    "be": "be", "da": "da", "kk": "kk", "lv": "lv", "sl": "sl", "ar": "arb",  # ar: Standard Arabic, 1484 h
    # 100-1000 h
    "et": "et", "pl": "pl", "bs": "bs", "is": "is", "mt": "mt", "eu": "eu", "fi": "fi", "ta": "ta", "sw": "sw",
    "fa": "fa", "bn": "bn", "as": "as", "mn": "mn", "hu": "hu", "ba": "ba", "te": "te", "ur": "ur", "gl": "gl",
    "ml": "ml", "ka": "ka", "mr": "mr", "cs": "cs", "pa": "pa", "cy": "cy", "kn": "kn", "tr": "tr", "hi": "hi",
    "uz": "uz", "ne": "npi",
}


def omnivoice_python(runtime: dict[str, Any]) -> str:
    """The worker's interpreter: ``[python] omnivoice`` when set, else the engine's own (the engine venv has OmniVoice)."""
    return config.runtime_path(runtime, "python", "omnivoice") or sys.executable


def omnivoice_model_dir(runtime: dict[str, Any]) -> str | None:
    d = config.runtime_path(runtime, "paths", "omnivoice_model")
    if d:
        return d
    root = _runtime_root(runtime)
    return os.path.join(root, "models", "omnivoice") if root else None


def _runtime_root(runtime: dict[str, Any]) -> str | None:
    """``<data>/runtime``: the parent of the Applio checkout named in runtime.toml."""
    d = config.applio_dir(runtime)
    return os.path.dirname(d) if d else None


def installed(runtime: dict[str, Any]) -> bool:
    py, model = omnivoice_python(runtime), omnivoice_model_dir(runtime)
    return bool(py and model and os.path.isfile(py) and os.path.isfile(os.path.join(model, "model.safetensors")))


class OmniVoiceBackend(TtsBackend):
    name = "omnivoice"
    languages = frozenset(LANGS)

    def __init__(self, unload_after_s: float = UNLOAD_AFTER_S) -> None:
        self._client: WorkerClient | None = None
        self._status = "loading"
        self._warm_sig: tuple | None = None
        self.unload_after_s = unload_after_s
        self.last_load: dict[str, Any] = {}

    # ---- state ----
    def status(self) -> str:
        return self._status

    def set_status(self, status: str) -> None:
        self._status = status

    def _ensure_client(self, runtime: dict[str, Any]) -> WorkerClient:
        if self._client is None:
            py, model = omnivoice_python(runtime), omnivoice_model_dir(runtime)
            whisper_dir = os.path.join(os.path.dirname(model), "whisper") if model else ""
            device = config.backend(runtime)
            argv = [py, "-I", WORKER, "--model-dir", model, "--whisper-dir", whisper_dir,
                    "--device", device if device in ("cuda", "mps", "cpu") else "cuda"]
            if self.unload_after_s > 0:
                argv += ["--unload-after", str(self.unload_after_s)]
            self._client = WorkerClient("omnivoice", argv, child_env(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                                        start_timeout=240.0)
        return self._client

    # ---- lifecycle ----
    async def load(self, runtime: dict[str, Any]) -> None:
        if not installed(runtime):
            self._status = "missing"
            log.warning("OmniVoice is not installed (venv %s, model %s); run setup", omnivoice_python(runtime),
                        omnivoice_model_dir(runtime))
            return
        self._status = "loading"
        client = self._ensure_client(runtime)
        try:
            self.last_load = await client.call("load", timeout=LOAD_TIMEOUT_S)
        except (WorkerError, WorkerDied, WorkerCancelled) as exc:
            self._status = "missing"
            log.error("OmniVoice failed to load: %s", exc)
            return
        self._status = "ready"
        log.info("OmniVoice ready: %s", self.last_load)

    async def release(self) -> None:
        """Training takes the GPU: end the worker process (frees VRAM and all its host RAM at once)."""
        self._status = "paused"
        self._warm_sig = None
        if self._client is not None:
            await self._client.stop()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    # ---- synthesis ----
    async def synthesize(self, text: str, lang: str, reference: str, params: SynthParams) -> str:
        lang = (lang or "").lower()
        model_lang = LANGS.get(lang)
        if model_lang is None:
            raise EngineError("bad_request", f"OmniVoice does not speak {lang!r}")
        if self._client is None or self._status in ("missing", "paused"):
            raise EngineError("not_ready", "the OmniVoice speech model is not loaded")
        ref = reference_for(reference, lang)
        steps = FAST_STEPS if params.quality == "fast" else BEST_STEPS
        if params.first:
            steps = min(steps, FIRST_STEPS)
        speed = 1.0 / max(0.5, min(2.0, float(params.duration_factor or 1.0)))
        t0 = time.time()
        try:
            res = await self._client.call("synth", {"text": text, "lang": lang, "model_lang": model_lang, "ref": ref,
                                                    "out": params.out_path, "steps": steps, "speed": round(speed, 4),
                                                    **sampling_for(params.temperature)},
                                          timeout=SYNTH_TIMEOUT_S)
        except WorkerCancelled as exc:
            raise EngineError("cancelled", "generation cancelled") from exc
        except WorkerError as exc:
            code = exc.code if exc.code in ("bad_request", "not_found") else "internal"
            raise EngineError(code, f"OmniVoice: {exc.message}") from exc
        except WorkerDied as exc:
            raise EngineError("internal", f"the OmniVoice worker stopped: {exc}") from exc
        self._warm_sig = _sig(ref)
        log.info("OmniVoice %s: %.2fs for %.2fs of audio (%d steps)", lang, time.time() - t0,
                 float(res.get("duration_s", 0.0)), steps)
        return str(res.get("out") or params.out_path)

    async def warm(self, reference: str, lang: str = "en") -> None:
        """Build the voice-clone prompt of a voice now (``voices.warm``), so its first sentence is fast."""
        if self._client is None or self._status != "ready":
            return
        ref = reference_for(reference, lang)
        await self._client.call("warm", {"ref": ref}, timeout=SYNTH_TIMEOUT_S)
        self._warm_sig = _sig(ref)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)

    def is_warm(self, reference: str) -> bool:
        return self._warm_sig is not None and self._warm_sig == _sig(reference)

    def forget_reference(self) -> None:
        self._warm_sig = None  # the worker re-reads a clip whose size or mtime changed (and its sidecar)


def reference_for(reference: str, lang: str) -> str:
    """``reference_<lang>.wav`` next to the voice's reference when it exists, else the reference itself."""
    alt = os.path.join(os.path.dirname(reference), f"reference_{(lang or '').lower()}.wav")
    return alt if lang and os.path.isfile(alt) else reference


def _sig(path: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


OMNIVOICE = OmniVoiceBackend()
