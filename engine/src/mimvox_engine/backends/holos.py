"""HolosTTS backend for Ukrainian with exact word stress.

Runs in its own venv: its fork of ``ukrainian-word-stress`` has the same package name as the one the engine venv uses.
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
    py, model = holos_python(runtime), holos_model_dir(runtime)
    return bool(py and model and os.path.isfile(py) and os.path.isfile(os.path.join(model, STAMP))
                and all(os.path.isfile(os.path.join(model, f)) for f in REQUIRED))


class HolosBackend(TtsBackend):
    name = "holos"
    languages = frozenset({"uk"})

    def __init__(self) -> None:
        self._client: WorkerClient | None = None
        self._status = "missing"
        self._warm: OrderedDict[tuple, None] = OrderedDict()
        self.last_load: dict[str, Any] = {}

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
        if self._status == "ready" or self._status == "loading":
            self._status = "paused"
        self._warm.clear()
        if self._client is not None:
            await self._client.stop()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

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
        self._warm = OrderedDict()


def _reference_for(reference: str) -> str:
    alt = os.path.join(os.path.dirname(reference), "reference_uk.wav")
    return alt if os.path.isfile(alt) else reference


def _sig(path: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


HOLOS = HolosBackend()
