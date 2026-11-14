"""OmniVoice in a worker process: a native crash cannot take the engine down, and ending it frees the GPU at once."""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import OrderedDict
from typing import Any

from .. import config
from ..registry import EngineError
from ..workerproc import WorkerCancelled, WorkerClient, WorkerDied, WorkerError, child_env, isolated
from . import SynthParams, TtsBackend

log = logging.getLogger("mimvox.omnivoice")

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnivoice_worker.py")

BEST_STEPS = int(os.environ.get("MIMVOX_OMNIVOICE_STEPS", "32"))
FAST_STEPS = int(os.environ.get("MIMVOX_OMNIVOICE_FAST_STEPS", "16"))
FIRST_STEPS = int(os.environ.get("MIMVOX_OMNIVOICE_FIRST_STEPS", "16"))  # segment 0: ~1.1 s instead of ~2.2 s
UNLOAD_AFTER_S = float(os.environ.get("MIMVOX_OMNIVOICE_UNLOAD_S", "0"))
PROMPT_CACHE = 3  # the user's reference plus the two native-accent clips

# Expressiveness 0.8 gives OmniVoice's defaults (position_temperature 5.0, class_temperature 0 = greedy);
# above ~1.2 words may start to slur.
EXPR_DEFAULT = 0.8
POS_TEMP_DEFAULT = 5.0


def sampling_for(expressiveness: float) -> dict[str, float]:
    e = max(0.1, min(1.5, float(expressiveness)))
    return {"position_temperature": round(POS_TEMP_DEFAULT * e / EXPR_DEFAULT, 3),
            "class_temperature": round(max(0.0, e - EXPR_DEFAULT), 3)}

SYNTH_TIMEOUT_S = 300.0   # includes a first-time reference transcription and a cold model load
LOAD_TIMEOUT_S = 300.0

# Languages with at least ~100 h of OmniVoice training data (its docs/languages.md): quality drops below that.
LANGS: dict[str, str] = {
    "en": "en", "zh": "zh", "ja": "ja", "es": "es", "fr": "fr", "de": "de", "ru": "ru", "pt": "pt", "yue": "yue",
    "th": "th", "it": "it", "ko": "ko", "vi": "vi", "id": "id", "no": "no", "ca": "ca", "hr": "hr", "lt": "lt",
    "sk": "sk", "sv": "sv", "el": "el", "nl": "nl", "bg": "bg", "sr": "sr", "uk": "uk",
    "be": "be", "da": "da", "kk": "kk", "lv": "lv", "sl": "sl", "ar": "arb",
    "et": "et", "pl": "pl", "bs": "bs", "is": "is", "mt": "mt", "eu": "eu", "fi": "fi", "ta": "ta", "sw": "sw",
    "fa": "fa", "bn": "bn", "as": "as", "mn": "mn", "hu": "hu", "ba": "ba", "te": "te", "ur": "ur", "gl": "gl",
    "ml": "ml", "ka": "ka", "mr": "mr", "cs": "cs", "pa": "pa", "cy": "cy", "kn": "kn", "tr": "tr", "hi": "hi",
    "uz": "uz", "ne": "npi",
    # Under 100 h but tested: 13 % of words wrong, against 22 % for Polish and 39 % for Hungarian.
    "ro": "ro",
}


def omnivoice_python(runtime: dict[str, Any]) -> str:
    return config.runtime_path(runtime, "python", "omnivoice") or sys.executable


def omnivoice_model_dir(runtime: dict[str, Any]) -> str | None:
    d = config.runtime_path(runtime, "paths", "omnivoice_model")
    if d:
        return d
    root = _runtime_root(runtime)
    return os.path.join(root, "models", "omnivoice") if root else None


def _runtime_root(runtime: dict[str, Any]) -> str | None:
    return config.runtime_root(runtime)


def installed(runtime: dict[str, Any]) -> bool:
    py, model = omnivoice_python(runtime), omnivoice_model_dir(runtime)
    return bool(py and model and os.path.isfile(py) and os.path.isfile(os.path.join(model, "model.safetensors")))


class OmniVoiceBackend(TtsBackend):
    name = "omnivoice"
    languages = frozenset(LANGS)

    def __init__(self, unload_after_s: float = UNLOAD_AFTER_S) -> None:
        self._client: WorkerClient | None = None
        self._status = "loading"
        self._warm: OrderedDict[tuple, None] = OrderedDict()
        self.unload_after_s = unload_after_s
        self.last_load: dict[str, Any] = {}

    def status(self) -> str:
        return self._status

    def set_status(self, status: str) -> None:
        self._status = status

    def _ensure_client(self, runtime: dict[str, Any]) -> WorkerClient:
        if self._client is None:
            py, model = omnivoice_python(runtime), omnivoice_model_dir(runtime)
            whisper_dir = os.path.join(os.path.dirname(model), "whisper") if model else ""
            device = config.backend(runtime)
            argv = [py, *isolated(), WORKER, "--model-dir", model, "--whisper-dir", whisper_dir,
                    "--device", device if device in ("cuda", "mps", "cpu") else "cuda", "--prompt-cache", str(PROMPT_CACHE)]
            if self.unload_after_s > 0:
                argv += ["--unload-after", str(self.unload_after_s)]
            self._client = WorkerClient("omnivoice", argv, child_env(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                                        start_timeout=240.0)
            self._client.on_release.append(self._warm.clear)
        return self._client

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
        """Ending the worker frees its VRAM and host RAM at once."""
        self._status = "paused"
        self._warm.clear()
        if self._client is not None:
            await self._client.stop()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    async def synthesize(self, text: str, lang: str, reference: str, params: SynthParams) -> str:
        lang = (lang or "").lower()
        model_lang = LANGS.get(lang)
        if model_lang is None:
            raise EngineError("bad_request", f"OmniVoice does not speak {lang!r}")
        if self._client is None or self._status in ("missing", "paused"):
            raise EngineError("not_ready", "the OmniVoice speech model is not loaded")
        ref = reference if params.reference_text else reference_for(reference, lang)
        steps = FAST_STEPS if params.quality == "fast" else BEST_STEPS
        if params.first:
            steps = min(steps, FIRST_STEPS)
        speed = 1.0 / max(0.5, min(2.0, float(params.duration_factor or 1.0)))
        t0 = time.time()
        try:
            res = await self._client.call("synth", {"text": text, "lang": lang, "model_lang": model_lang, "ref": ref,
                                                    "out": params.out_path, "steps": steps, "speed": round(speed, 4),
                                                    **sampling_for(params.temperature),
                                                    **({"ref_text": params.reference_text} if params.reference_text
                                                       else {})},
                                          timeout=SYNTH_TIMEOUT_S)
        except WorkerCancelled as exc:
            raise EngineError("cancelled", "generation cancelled") from exc
        except WorkerError as exc:
            code = exc.code if exc.code in ("bad_request", "not_found") else "internal"
            raise EngineError(code, f"OmniVoice: {exc.message}") from exc
        except WorkerDied as exc:
            raise EngineError("internal", f"the OmniVoice worker stopped: {exc}") from exc
        self._mark_warm(ref)
        log.info("OmniVoice %s: %.2fs for %.2fs of audio (%d steps)", lang, time.time() - t0,
                 float(res.get("duration_s", 0.0)), steps)
        return str(res.get("out") or params.out_path)

    async def warm(self, reference: str, lang: str = "en", reference_text: str | None = None) -> None:
        if self._client is None or self._status != "ready":
            return
        ref = reference if reference_text else reference_for(reference, lang)
        try:
            await self._client.call("warm", {"ref": ref, **({"ref_text": reference_text} if reference_text else {})},
                                    timeout=SYNTH_TIMEOUT_S)
        except WorkerCancelled as exc:
            raise EngineError("cancelled", "warm-up cancelled") from exc
        except (WorkerError, WorkerDied) as exc:
            raise EngineError("internal", f"OmniVoice: {exc}") from exc
        self._mark_warm(ref)

    async def inspect(self, reference: str, detect: bool = True) -> str | None:
        """Whisper's language of the clip itself (no per-language swap), or None; never raises."""
        if self._client is None or self._status != "ready":
            return None
        try:
            res = await self._client.call("inspect", {"ref": reference, "detect": detect}, timeout=SYNTH_TIMEOUT_S)
        except (WorkerError, WorkerDied, WorkerCancelled) as exc:
            log.warning("OmniVoice could not inspect %s: %s", reference, exc)
            return None
        return str(res.get("language") or "") or None

    def _mark_warm(self, ref: str) -> None:
        sig = _sig(ref)
        if sig is None:
            return
        self._warm[sig] = None
        self._warm.move_to_end(sig)
        while len(self._warm) > PROMPT_CACHE:
            self._warm.popitem(last=False)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)

    def is_warm(self, reference: str) -> bool:
        sig = _sig(reference)
        return sig is not None and sig in self._warm

    def forget_reference(self) -> None:
        self._warm = OrderedDict()


def reference_for(reference: str, lang: str) -> str:
    alt = os.path.join(os.path.dirname(reference), f"reference_{(lang or '').lower()}.wav")
    return alt if lang and os.path.isfile(alt) else reference


def _sig(path: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


OMNIVOICE = OmniVoiceBackend()
