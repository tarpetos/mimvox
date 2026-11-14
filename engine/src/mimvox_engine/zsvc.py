"""Client of the zero-shot voice-conversion worker (``zsvc_worker.py``: Seed-VC, run with the engine interpreter).

The native accent of an **untrained** voice (``accent.py``): OmniVoice speaks English from a native American clip and
this worker gives the result the timbre of the voice's own reference clip, with no training. A trained voice keeps
RVC (closer to the user). Measured on the owner's voice (11-18 s Ukrainian reference, RTX 4070): held-out ECAPA
speaker similarity 0.44 (native clip without conversion: 0.04; OmniVoice cloning the Ukrainian clip, with its accent:
0.63), Whisper WER 0 %, 1.5 GB of VRAM (docs/performance.md, "Zero-shot native accent"). Diffusion steps follow the
job's quality (``STEPS``): 12 matched 25 in similarity and WER at about 0.5 s per 4 s segment.

Installed by setup (optional runtime component, not part of this repository: Seed-VC is GPL-3.0):
the source checkout in ``<runtime>/seed-vc`` and the weights in ``<runtime>/models/seed-vc`` (``runtime.toml``
``[paths] seed_vc`` / ``seed_vc_models`` override them). ``installed`` needs setup's ``.mimvox-setup-ok`` stamp in the
weights folder, written once the engine venv has the two extra packages (munch, einops) and every file is there.

Async, on the engine loop (``workerproc.py``): requests are serialized by the worker's queue; a dead worker is
restarted by the next request (and that request retried once). The worker drops its models after 10 idle minutes.
On a GPU with less than ``TIGHT_VRAM_GB`` the speech model and the converter take turns (``gpu_turn``) instead of
overlapping a synthesis with the conversion of the previous segment.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from typing import Any

from . import config
from .workerproc import WorkerCancelled, WorkerClient, WorkerDied, WorkerError, child_env, isolated

log = logging.getLogger("mimvox.zsvc")

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zsvc_worker.py")
CODE_DIR = "seed-vc"                   # <runtime>/seed-vc
MODELS_DIR = os.path.join("models", "seed-vc")   # <runtime>/models/seed-vc
STAMP = ".mimvox-setup-ok"
REQUIRED = (os.path.join("dit", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
            os.path.join("dit", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"),
            os.path.join("bigvgan", "config.json"), os.path.join("bigvgan", "bigvgan_generator.pt"),
            os.path.join("whisper-small", "config.json"), os.path.join("whisper-small", "model.safetensors"),
            os.path.join("whisper-small", "preprocessor_config.json"))
TIGHT_VRAM_GB = 10.0   # below this the speech model and the converter do not run at the same time
CONVERT_TIMEOUT_S = 180.0
STEPS = {"best": 12, "fast": 8}   # diffusion steps per quality: 12 matched 25 (docs/performance.md, "Zero-shot native accent")


class ZsvcError(RuntimeError):
    pass


def paths(runtime: dict[str, Any], data_root: str) -> tuple[str, str]:
    """(Seed-VC checkout, weights folder)."""
    base = os.path.join(data_root, "runtime")
    return (config.runtime_path(runtime, "paths", "seed_vc") or os.path.join(base, CODE_DIR),
            config.runtime_path(runtime, "paths", "seed_vc_models") or os.path.join(base, MODELS_DIR))


def installed(runtime: dict[str, Any], data_root: str) -> bool:
    """Setup finished the converter: the checkout, every weight file and the stamp. A few stats (any thread)."""
    code, models = paths(runtime, data_root)
    return (os.path.isfile(os.path.join(code, "modules", "commons.py"))
            and os.path.isfile(os.path.join(code, "campplus_cn_common.bin"))
            and os.path.isfile(os.path.join(models, STAMP))
            and all(os.path.isfile(os.path.join(models, f)) for f in REQUIRED))


class ZsvcWorker:
    def __init__(self, code: str, models: str, device: str) -> None:
        env = child_env(TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        # -I: do not put the script directory (engine modules such as config.py) on sys.path.
        self.client = WorkerClient("zsvc", [sys.executable, *isolated(), WORKER, "--seed-vc", code, "--models", models,
                                            "--device", device], env, start_timeout=120.0)
        self.warm_refs: set[tuple] = set()   # reference states whose prompt the worker has built (cleared on exit)
        self.client.on_release.append(self.warm_refs.clear)

    def alive(self) -> bool:
        return self.client.alive()

    async def _call(self, op: str, params: dict[str, Any], timeout: float, start: bool = True) -> dict[str, Any]:
        try:
            return await self.client.call(op, params, timeout=timeout, start=start)
        except (WorkerError, WorkerDied, WorkerCancelled) as exc:
            self.warm_refs.clear()
            raise ZsvcError(str(exc) or type(exc).__name__) from exc

    async def convert(self, src: str, dst: str, ref: str, timeout: float = CONVERT_TIMEOUT_S,
                      steps: int = STEPS["best"]) -> float:
        """Convert ``src`` (wav) to the timbre of ``ref`` with ``steps`` diffusion steps; ``dst`` is a 22.05 kHz WAV of
        the same length. Returns the worker's elapsed seconds."""
        res = await self._call("convert", {"in": src, "out": dst, "ref": ref, "steps": int(steps)}, timeout)
        sig = ref_sig(ref)
        if sig:
            self.warm_refs.add(sig)
        return float(res.get("elapsed", 0.0))

    async def warm(self, ref: str) -> None:
        """Load the models and build ``ref``'s prompt (``voices.warm``)."""
        await self._call("warm", {"ref": ref}, 300.0)
        sig = ref_sig(ref)
        if sig:
            self.warm_refs.add(sig)

    def is_warm(self, ref: str) -> bool:
        return self.alive() and ref_sig(ref) in self.warm_refs

    async def release_gpu(self) -> None:
        """Drop the models (they reload on the next request). No-op when the process is not running."""
        if not self.alive():
            return
        self.warm_refs.clear()
        try:
            await self._call("release", {}, 60.0, start=False)
        except ZsvcError:
            log.exception("could not release the converter's GPU memory")

    async def stop(self) -> None:
        self.warm_refs.clear()
        await self.client.stop()


def ref_sig(path: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


class Bound:
    """The converter of one job: the worker plus the reference clip whose timbre it gives (``tts`` / Redub call
    ``convert(src, dst)`` per segment, like RVC), the diffusion steps of the job's quality (``STEPS``), and the GPU
    turn to take on a tight GPU."""

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
    """The shared worker, or None when setup has not installed the converter."""
    global _WORKER
    if not installed(runtime, data_root):
        return None
    if _WORKER is None:
        code, models = paths(runtime, data_root)
        _WORKER = ZsvcWorker(code, models, config.backend(runtime))
    return _WORKER


def turn_lock(vram_gb: float) -> asyncio.Lock | None:
    """The lock the speech model and the converter share on a GPU of less than ``TIGHT_VRAM_GB`` (``vram_gb`` is
    ``hello``'s total; 0 = unknown or not CUDA: no lock). Loop thread."""
    global _TURN
    if not 0.0 < float(vram_gb or 0.0) < TIGHT_VRAM_GB:
        return None
    if _TURN is None:
        _TURN = asyncio.Lock()
    return _TURN


def gpu_turn(lock: asyncio.Lock | None) -> contextlib.AbstractAsyncContextManager[Any]:
    """``async with gpu_turn(lock):`` around a GPU call; a no-op without a lock."""
    return lock if lock is not None else contextlib.nullcontext()


async def release_gpu() -> None:
    if _WORKER is not None:
        await _WORKER.release_gpu()


async def shutdown() -> None:
    if _WORKER is not None:
        await _WORKER.stop()
