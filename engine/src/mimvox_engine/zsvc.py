"""Client of the Seed-VC zero-shot converter. Seed-VC is GPL-3.0: setup installs it; it is not in this repository."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Any

from . import config
from .workerproc import WorkerCancelled, WorkerClient, WorkerDied, WorkerError

log = logging.getLogger("mimvox.zsvc")

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zsvc_worker.py")
CODE_DIR = "seed-vc"
MODELS_DIR = os.path.join("models", "seed-vc")
STAMP = ".mimvox-setup-ok"
REQUIRED = (os.path.join("dit", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
            os.path.join("dit", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"),
            os.path.join("bigvgan", "config.json"), os.path.join("bigvgan", "bigvgan_generator.pt"),
            os.path.join("whisper-small", "config.json"), os.path.join("whisper-small", "model.safetensors"),
            os.path.join("whisper-small", "preprocessor_config.json"))
TIGHT_VRAM_GB = 10.0
CONVERT_TIMEOUT_S = 180.0
STEPS = {"best": 12, "fast": 8}   # measured: 12 matched 25 in similarity and WER


class ZsvcError(RuntimeError):
    pass


def paths(runtime: dict[str, Any], data_root: str) -> tuple[str, str]:
    base = os.path.join(data_root, "runtime")
    return (config.runtime_path(runtime, "paths", "seed_vc") or os.path.join(base, CODE_DIR),
            config.runtime_path(runtime, "paths", "seed_vc_models") or os.path.join(base, MODELS_DIR))


def installed(runtime: dict[str, Any], data_root: str) -> bool:
    code, models = paths(runtime, data_root)
    return (os.path.isfile(os.path.join(code, "modules", "commons.py"))
            and os.path.isfile(os.path.join(code, "campplus_cn_common.bin"))
            and os.path.isfile(os.path.join(models, STAMP))
            and all(os.path.isfile(os.path.join(models, f)) for f in REQUIRED))


class ZsvcWorker:
    """Hosted by the OmniVoice worker process: one PyTorch process for both models instead of two."""

    def __init__(self, code: str, models: str, client: WorkerClient) -> None:
        self.where = {"seed_vc": code, "models": models}
        self.client = client
        self.warm_refs: set[tuple] = set()
        self.client.on_release.append(self.warm_refs.clear)

    def alive(self) -> bool:
        return self.client.alive()

    async def _call(self, op: str, params: dict[str, Any], timeout: float, start: bool = True) -> dict[str, Any]:
        try:
            return await self.client.call(f"zs_{op}", {**params, **self.where}, timeout=timeout, start=start)
        except (WorkerError, WorkerDied, WorkerCancelled) as exc:
            self.warm_refs.clear()
            raise ZsvcError(str(exc) or type(exc).__name__) from exc

    async def convert(self, src: str, dst: str, ref: str, timeout: float = CONVERT_TIMEOUT_S,
                      steps: int = STEPS["best"]) -> float:
        """``dst`` is a 22.05 kHz WAV of the same length. Returns the worker's elapsed seconds."""
        res = await self._call("convert", {"in": src, "out": dst, "ref": ref, "steps": int(steps)}, timeout)
        sig = ref_sig(ref)
        if sig:
            self.warm_refs.add(sig)
        return float(res.get("elapsed", 0.0))

    async def warm(self, ref: str) -> None:
        await self._call("warm", {"ref": ref}, 300.0)
        sig = ref_sig(ref)
        if sig:
            self.warm_refs.add(sig)

    def is_warm(self, ref: str) -> bool:
        return self.alive() and ref_sig(ref) in self.warm_refs

    async def release_gpu(self) -> None:
        if not self.alive():
            return
        self.warm_refs.clear()
        try:
            await self._call("release", {}, 60.0, start=False)
        except ZsvcError:
            log.exception("could not release the converter's GPU memory")

    async def stop(self) -> None:
        """Only the models: the process is the speech model's and ends with it."""
        await self.release_gpu()


def ref_sig(path: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


class Bound:
    def __init__(self, worker: ZsvcWorker, ref: str, turn: asyncio.Lock | None, quality: str = "best") -> None:
        self.worker = worker
        self.ref = ref
        self.turn = turn
        self.steps = STEPS.get(quality, STEPS["best"])

    async def convert(self, src: str, dst: str, timeout: float = CONVERT_TIMEOUT_S) -> float:
        async with gpu_turn(self.turn):
            return await self.worker.convert(src, dst, self.ref, timeout, self.steps)


_WORKER: ZsvcWorker | None = None
_TURN: asyncio.Lock | None = None


def get(runtime: dict[str, Any], data_root: str) -> ZsvcWorker | None:
    global _WORKER
    if not installed(runtime, data_root):
        return None
    if _WORKER is None:
        from .backends.omnivoice import OMNIVOICE, installed as speech_installed  # noqa: PLC0415 - no import cycle
        if not speech_installed(runtime):
            return None
        code, models = paths(runtime, data_root)
        _WORKER = ZsvcWorker(code, models, OMNIVOICE._ensure_client(runtime))
    return _WORKER


def turn_lock(vram_gb: float) -> asyncio.Lock | None:
    """``vram_gb`` is ``hello``'s total; 0 (unknown or not CUDA) means no lock. Loop thread."""
    global _TURN
    if not 0.0 < float(vram_gb or 0.0) < TIGHT_VRAM_GB:
        return None
    if _TURN is None:
        _TURN = asyncio.Lock()
    return _TURN


def gpu_turn(lock: asyncio.Lock | None) -> contextlib.AbstractAsyncContextManager[Any]:
    return lock if lock is not None else contextlib.nullcontext()


async def release_gpu() -> None:
    if _WORKER is not None:
        await _WORKER.release_gpu()


async def shutdown() -> None:
    if _WORKER is not None:
        await _WORKER.stop()
