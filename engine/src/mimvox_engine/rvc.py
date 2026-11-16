"""Client of the persistent RVC worker; a dead worker is restarted by the next request, which is retried once."""
from __future__ import annotations

import logging
import os
from typing import Any

from . import components, config
from .workerproc import WorkerCancelled, WorkerClient, WorkerDied, WorkerError, child_env, isolated

log = logging.getLogger("mimvox.rvc")

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_worker.py")


# The index pulls foreign phonemes towards the recorded language. Measured, Ukrainian on an English-trained voice:
# WER 25 % at index 1.0, 6 % at 0.3, timbre similarity 0.84 vs 0.87.
FOREIGN_INDEX_RATE = float(os.environ.get("MIMVOX_RVC_FOREIGN_INDEX", "0.3"))


def index_rate_for(voice_match: float, lang: str | None, voice_lang: str | None) -> float:
    if (lang or "en").lower() == (voice_lang or "en").lower():
        return voice_match
    return min(voice_match, FOREIGN_INDEX_RATE)


class RvcError(RuntimeError):
    pass


class RvcWorker:
    def __init__(self, python: str, applio: str) -> None:
        self.python = python
        self.applio = applio
        # -I: do not put the script directory (engine modules such as config.py) on sys.path.
        self.client = WorkerClient("rvc", [python, *isolated(), WORKER, "--applio", applio], child_env(), start_timeout=240.0)
        self.loaded_model: tuple | None = None
        self.client.on_release.append(self._released)

    def _released(self) -> None:
        self.loaded_model = None

    @property
    def last_error(self) -> str | None:
        return self.client.last_error

    def alive(self) -> bool:
        return self.client.alive()

    async def start(self) -> None:
        try:
            await self.client.call("ping", timeout=300.0)
        except (WorkerError, WorkerDied) as exc:
            raise RvcError(str(exc)) from exc

    async def convert(self, src: str, dst: str, pth: str, index: str, index_rate: float,
                      timeout: float = 180.0, pitch: int = 0) -> float:
        """Returns the worker's elapsed seconds."""
        try:
            res = await self.client.call("convert", {"in": src, "out": dst, "pth": pth, "index": index,
                                                     "idx": float(index_rate), "pitch": int(pitch)}, timeout=timeout)
        except (WorkerError, WorkerDied, WorkerCancelled) as exc:
            self.client.last_error = str(exc)
            raise RvcError(str(exc)) from exc
        self.loaded_model = model_sig(pth)
        return float(res.get("elapsed", 0.0))

    async def stop(self) -> None:
        self.loaded_model = None
        await self.client.stop()


def model_sig(pth: str) -> tuple | None:
    try:
        st = os.stat(pth)
    except OSError:
        return None
    return (os.path.abspath(pth), st.st_mtime_ns, st.st_size)


_WORKER: RvcWorker | None = None


def available(runtime: dict[str, Any], honour_off: bool = True) -> bool:
    if honour_off and components.off("training"):
        return False
    py, applio = config.rvc_python(runtime), config.applio_dir(runtime)
    return bool(py and applio and os.path.isfile(py) and os.path.isdir(applio))


def get(runtime: dict[str, Any]) -> RvcWorker | None:
    global _WORKER
    if components.off("training"):
        return None
    py, applio = config.rvc_python(runtime), config.applio_dir(runtime)
    if not py or not applio or not os.path.isfile(py) or not os.path.isdir(applio):
        return None
    if _WORKER is None:
        _WORKER = RvcWorker(py, applio)
    return _WORKER


async def shutdown() -> None:
    if _WORKER is not None:
        await _WORKER.stop()
