"""Client of the persistent RVC worker (``rvc_worker.py``, run with the ``python.rvc`` interpreter).

Async, on the engine loop: see ``workerproc.py`` for the owner-task design. Conversions are serialized by the worker's
request queue; a dead worker is restarted by the next request (and that request retried once)."""
from __future__ import annotations

import logging
import os
from typing import Any

from . import config
from .workerproc import WorkerCancelled, WorkerClient, WorkerDied, WorkerError, child_env, isolated

log = logging.getLogger("mimvox.rvc")

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_worker.py")


# RVC index rate for speech in a language other than the one the voice was recorded in. The retrieval index holds only
# ContentVec features of the recorded language and pulls foreign phonemes towards them: on an English-trained
# voice, Ukrainian speech came out garbled at index 1.0 (Whisper WER 25 % vs 6 % before RVC) and intact at 0.3 (6 %),
# with most of the timbre kept (ECAPA similarity 0.84 vs 0.87). Measured in the lang-tts evaluation (docs/performance.md).
FOREIGN_INDEX_RATE = float(os.environ.get("MIMVOX_RVC_FOREIGN_INDEX", "0.3"))


def index_rate_for(voice_match: float, lang: str | None, voice_lang: str | None) -> float:
    """The index rate to convert speech in ``lang`` with a voice recorded in ``voice_lang`` (``voice.json``
    ``language``, default English): ``voice_match`` itself in the voice's own language, at most
    ``FOREIGN_INDEX_RATE`` in any other."""
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
        self.loaded_model: tuple | None = None   # (path, mtime, size) of the voice model the worker has loaded

    @property
    def last_error(self) -> str | None:
        return self.client.last_error

    def alive(self) -> bool:
        return self.client.alive()

    async def start(self) -> None:
        """Start the process now (model start-up) instead of on the first conversion."""
        try:
            await self.client.call("ping", timeout=300.0)
        except (WorkerError, WorkerDied) as exc:
            raise RvcError(str(exc)) from exc

    async def convert(self, src: str, dst: str, pth: str, index: str, index_rate: float,
                      timeout: float = 180.0, pitch: int = 0) -> float:
        """Convert ``src`` (wav) to the voice of ``pth``, transposed by ``pitch`` semitones (the native accent moves
        a native speaker into the user's range, see accent.py); returns the worker's elapsed seconds."""
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


def available(runtime: dict[str, Any]) -> bool:
    """Voice training (Applio + its interpreter) is installed."""
    py, applio = config.rvc_python(runtime), config.applio_dir(runtime)
    return bool(py and applio and os.path.isfile(py) and os.path.isdir(applio))


def get(runtime: dict[str, Any]) -> RvcWorker | None:
    """The shared worker, or None when the runtime has no RVC interpreter / Applio."""
    global _WORKER
    py, applio = config.rvc_python(runtime), config.applio_dir(runtime)
    if not py or not applio or not os.path.isfile(py) or not os.path.isdir(applio):
        return None
    if _WORKER is None:
        _WORKER = RvcWorker(py, applio)
    return _WORKER


async def shutdown() -> None:
    if _WORKER is not None:
        await _WORKER.stop()
