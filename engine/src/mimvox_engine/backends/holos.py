"""HolosTTS backend: Ukrainian with exact word stress (the optional ``ukrainian`` runtime component).

Why a second backend for one language: OmniVoice reads letters and guesses the stress of each Ukrainian word, right
about two times in three (``omnivoice_stress.py`` has the measurements), and no form of stress mark makes it reliable;
over a book that is a wrong stress in every sentence or two. HolosTTS (patriotyk/HolosTTS, MIT) speaks from IPA
phonemes that carry the stress, so the stress the text front-end decides is the stress that is spoken. The front-end
(``holos_worker.py``): the user's marks, numbers and abbreviations spelled out (the same Ukrainian rules as OmniVoice's
``omnivoice_text.py``), then lang-uk's stress dictionary with Stanza's part-of-speech and morphology tags choosing
between the readings of a heteronym, a small neural accentor for words the dictionary lacks, then ``ipa-uk``. The voice
is cloned from the voice's reference clip (HolosTTS's voice encoder). It reads evenly, like a narrator: it has no
sampling temperature, so "expressiveness" does not apply; speed does.

Routing (``backends.for_language``): when the component is installed and its worker loaded, Ukrainian goes here;
otherwise (not installed, failed to load, paused for a training) it goes to OmniVoice, as before this backend existed.
The voice's RVC model, pauses, slot fitting and everything else after synthesis are the pipeline's, as for any backend.

The model runs in a worker process with the interpreter of its own environment (``<data>/runtime/holos/.venv``,
``runtime.toml`` ``[python] holos``; weights in ``<data>/runtime/models/holos``, ``[paths] holos_model``): patriotyk's
fork of ``ukrainian-word-stress`` has the name of the lang-uk package the engine venv uses for OmniVoice's stress
hints, so the two cannot share an environment. Installed when setup's ``.mimvox-setup-ok`` stamp is in the weights
folder.
"""
from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import Any

from .. import config
from ..registry import EngineError
from ..workerproc import WorkerCancelled, WorkerClient, WorkerDied, WorkerError, child_env, isolated
from . import SynthParams, TtsBackend

log = logging.getLogger("mimvox.holos")

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holos_worker.py")
STAMP = ".mimvox-setup-ok"
REQUIRED = ("holos.pt", "voice_encoder.pt", "voices.pt")
VOICE_CACHE = 4
LOAD_TIMEOUT_S = 300.0
SYNTH_TIMEOUT_S = 600.0


def holos_python(runtime: dict[str, Any]) -> str | None:
    return config.runtime_path(runtime, "python", "holos")


def holos_model_dir(runtime: dict[str, Any]) -> str | None:
    d = config.runtime_path(runtime, "paths", "holos_model")
    if d:
        return d
    root = config.runtime_root(runtime)
    return os.path.join(root, "models", "holos") if root else None


def installed(runtime: dict[str, Any]) -> bool:
    """Setup's ``ukrainian`` component is complete: the interpreter, the weights and the stamp written last."""
    py, model = holos_python(runtime), holos_model_dir(runtime)
    return bool(py and model and os.path.isfile(py) and os.path.isfile(os.path.join(model, STAMP))
                and all(os.path.isfile(os.path.join(model, f)) for f in REQUIRED))


class HolosBackend(TtsBackend):
    name = "holos"
    languages = frozenset({"uk"})

    def __init__(self) -> None:
        self._client: WorkerClient | None = None
        self._status = "missing"   # until load(): not installed counts as missing, so Ukrainian goes to OmniVoice
        self._warm: OrderedDict[tuple, None] = OrderedDict()
        self.last_load: dict[str, Any] = {}

    # ---- state ----
    def status(self) -> str:
        return self._status

    def available(self) -> bool:
        return self._status == "ready"

    def _ensure_client(self, runtime: dict[str, Any]) -> WorkerClient:
        if self._client is None:
            device = config.backend(runtime)
            argv = [holos_python(runtime), *isolated(), WORKER, "--model-dir", holos_model_dir(runtime),
                    "--device", device if device in ("cuda", "mps", "cpu") else "cuda", "--voice-cache", str(VOICE_CACHE)]
            self._client = WorkerClient("holos", argv, child_env(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                                        start_timeout=240.0)
            self._client.on_release.append(self._warm.clear)
        return self._client

    # ---- lifecycle ----
    async def load(self, runtime: dict[str, Any]) -> None:
        if not installed(runtime):
            self._status = "missing"
            log.info("Ukrainian speech (HolosTTS) is not installed: Ukrainian is spoken by OmniVoice")
            return
        self._status = "loading"
        client = self._ensure_client(runtime)
        try:
            self.last_load = await client.call("load", timeout=LOAD_TIMEOUT_S)
        except (WorkerError, WorkerDied, WorkerCancelled) as exc:
            self._status = "missing"
            log.error("HolosTTS failed to load (Ukrainian falls back to OmniVoice): %s", exc)
            return
        self._status = "ready"
        log.info("HolosTTS ready: %s", self.last_load)

    async def release(self) -> None:
        """Training takes the GPU: end the worker process. ``load`` (after the training) starts it again."""
        if self._status == "ready" or self._status == "loading":
            self._status = "paused"
        self._warm.clear()
        if self._client is not None:
            await self._client.stop()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    # ---- synthesis ----
    async def synthesize(self, text: str, lang: str, reference: str, params: SynthParams) -> str:
        if (lang or "").lower() != "uk":
            raise EngineError("bad_request", f"HolosTTS speaks Ukrainian only, not {lang!r}")
        if self._client is None or self._status != "ready":
            raise EngineError("not_ready", "the Ukrainian speech model is not loaded")
        ref = _reference_for(reference)
        speed = 1.0 / max(0.5, min(2.0, float(params.duration_factor or 1.0)))
        t0 = time.time()
        try:
            res = await self._client.call("synth", {"text": text, "ref": ref, "out": params.out_path,
                                                    "speed": round(speed, 4)}, timeout=SYNTH_TIMEOUT_S)
        except WorkerCancelled as exc:
            raise EngineError("cancelled", "generation cancelled") from exc
        except WorkerError as exc:
            code = exc.code if exc.code in ("bad_request", "not_found") else "internal"
            raise EngineError(code, f"HolosTTS: {exc.message}") from exc
        except WorkerDied as exc:
            raise EngineError("internal", f"the HolosTTS worker stopped: {exc}") from exc
        self._mark_warm(ref)
        log.info("HolosTTS: %.2fs for %.2fs of audio", time.time() - t0, float(res.get("duration_s", 0.0)))
        return str(res.get("out") or params.out_path)

    async def warm(self, reference: str) -> None:
        """The voice's style vector now (``voices.warm``), so the first sentence does not wait for it."""
        if self._client is None or self._status != "ready":
            return
        ref = _reference_for(reference)
        try:
            await self._client.call("warm", {"ref": ref}, timeout=SYNTH_TIMEOUT_S)
        except (WorkerError, WorkerDied, WorkerCancelled) as exc:
            log.warning("HolosTTS could not warm %s: %s", ref, exc)
            return
        self._mark_warm(ref)

    async def stress(self, text: str) -> dict[str, Any]:
        """The text with the stress HolosTTS would speak, and the words left without one (``text.stress``)."""
        if self._client is None or self._status != "ready":
            raise EngineError("not_ready", "the Ukrainian speech model is not loaded")
        try:
            return await self._client.call("stress", {"text": text}, timeout=SYNTH_TIMEOUT_S)
        except (WorkerError, WorkerDied, WorkerCancelled) as exc:
            raise EngineError("internal", f"HolosTTS: {exc}") from exc

    def _mark_warm(self, ref: str) -> None:
        sig = _sig(ref)
        if sig is None:
            return
        self._warm[sig] = None
        self._warm.move_to_end(sig)
        while len(self._warm) > VOICE_CACHE:
            self._warm.popitem(last=False)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)

    def is_warm(self, reference: str) -> bool:
        sig = _sig(_reference_for(reference))
        return sig is not None and sig in self._warm

    def forget_reference(self) -> None:
        self._warm = OrderedDict()   # the worker re-encodes a clip whose size or mtime changed


def _reference_for(reference: str) -> str:
    """``reference_uk.wav`` next to the voice's reference when it exists (a cleaner Ukrainian clip), else the
    reference itself. HolosTTS takes only the timbre and manner from it, never the accent: the phonemes are Ukrainian."""
    alt = os.path.join(os.path.dirname(reference), "reference_uk.wav")
    return alt if os.path.isfile(alt) else reference


def _sig(path: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


HOLOS = HolosBackend()
