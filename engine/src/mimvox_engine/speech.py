"""Client of the persistent speech worker (``speech_worker.py``: Whisper + NLLB-200 + Hybrid Demucs, run with the engine
interpreter).

Async, on the engine loop (see ``workerproc.py``): one owner task per worker serializes the requests through a queue,
replies resolve futures keyed by request id, progress lines call callbacks, and a crash fails every pending request
at once (the next request restarts the process). The worker drops its models after 3 idle minutes and exits after 10.

Concurrent ``redub.prepare`` + ``redub.dub`` + ``tts.generate`` + ``voices.warm`` are safe by construction: a
preload (``load_asr`` from ``redub.prepare``) is just the first request in the queue, a transcription queues behind it,
and ``release_gpu`` (before speech-model / RVC work) asks a running preload to stop and then queues behind it (behind
a running vocal separation too: that is one file of a media job, which then reloads the separator for its next file).
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import Any

from .registry import EngineError
from .workerproc import WorkerCancelled, WorkerClient, WorkerDied, WorkerError, child_env, isolated

log = logging.getLogger("mimvox.speech")

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_worker.py")
ASR_MODEL = "turbo"   # large-v3-turbo
TRANSLATE_BEAMS = 5   # 4 -> 5 costs about 5 % more time (measured); a little more search for the same VRAM
START_TIMEOUT_S = 120.0
OP_TIMEOUT_S = 900.0
SEPARATOR_FILE = os.path.join("demucs", "hdemucs_high_trained.pt")   # under <data>/runtime/models (setup fetches it)


def separator_path(data_root: str) -> str:
    """Where the vocal-separation weights (torchaudio HDEMUCS_HIGH_MUSDB_PLUS) live."""
    return os.path.join(data_root, "runtime", "models", SEPARATOR_FILE)


class Cancelled(Exception):
    pass


class SpeechError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def engine_error(self) -> EngineError:
        code = {"model_missing": "not_ready", "bad_request": "bad_request"}.get(self.code, "internal")
        return EngineError(code, self.message)


Progress = Callable[[str, float], None]


class SpeechWorker:
    def __init__(self, models_root: str, allow_cuda: bool) -> None:
        self.models_root = models_root
        self.allow_cuda = allow_cuda
        env = child_env(TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        # -I: do not put the script directory (engine modules such as config.py) on sys.path.
        self.client = WorkerClient("speech", [sys.executable, *isolated(), WORKER, "--models-root", models_root], env,
                                   start_timeout=START_TIMEOUT_S)
        # Tags of background separations (a Redub preload): release_gpu stops one that is running.
        self.background_tags: set[str] = set()

    @property
    def last_error(self) -> str | None:
        return self.client.last_error

    def alive(self) -> bool:
        return self.client.alive()

    async def request(self, op: str, params: dict[str, Any] | None = None, progress: Progress | None = None,
                      timeout: float = OP_TIMEOUT_S, start: bool = True) -> dict[str, Any]:
        """Run one op (cancel by cancelling the awaiting task). Errors become ``SpeechError`` / ``Cancelled``."""
        params = dict(params or {})
        params.setdefault("allow_cuda", self.allow_cuda)
        cb = None
        if progress is not None:
            def cb(p: dict[str, Any]) -> None:
                progress(str(p.get("stage")), float(p.get("pct", 0.0)))
        try:
            return await self.client.call(op, params, cb, timeout=timeout, start=start)
        except WorkerCancelled:
            raise Cancelled() from None
        except WorkerError as exc:
            raise SpeechError(exc.code, exc.message) from None
        except WorkerDied as exc:
            raise SpeechError("internal", f"the speech worker keeps crashing ({exc}); see the engine log") from None

    # ---- operations ----
    async def prepare(self) -> dict[str, Any]:
        return await self.request("load_asr", {"model": ASR_MODEL})

    async def transcribe(self, audio_npy: str, language: str | None, progress: Progress | None) -> dict[str, Any]:
        return await self.request("transcribe", {"audio": audio_npy, "language": language, "model": ASR_MODEL},
                                  progress)

    async def translate(self, texts: list[str], src: str, progress: Progress | None, tgt: str = "eng_Latn",
                        translator: str | None = None, gaps: list[float] | None = None) -> dict[str, Any]:
        """Translate ``texts`` (consecutive segments, in order). ``translator``: ``nllb-1.3b`` | ``nllb-600m`` | None
        (the best installed). With ``gaps`` (seconds of silence between text i and i+1) the segments of one sentence
        are translated together and the translation is cut back into them."""
        params: dict[str, Any] = {"texts": texts, "src": src, "tgt": tgt, "num_beams": TRANSLATE_BEAMS,
                                  "translator": translator}
        if gaps is not None:
            params.update(join=True, gaps=gaps)
        return await self.request("translate", params, progress)

    async def separate_vocals(self, audio_npy: str, out_npy: str, seconds: float, progress: Progress | None,
                              allow_cuda: bool | None = None, background_out: str | None = None,
                              tag: str | None = None) -> dict[str, Any]:
        """Vocals of a 44.1 kHz ``.npy`` of ``seconds`` (mono ``(n,)`` or ``(channels, n)``), written to ``out_npy``
        in the same shape; with ``background_out`` the accompaniment (drums + bass + other) too. ``tag``: a background
        run that ``release_gpu`` may stop while its tag is in ``background_tags``. The timeout grows with the length
        (the CPU fallback can be slower than real time)."""
        params: dict[str, Any] = {"audio": audio_npy, "out": out_npy}
        if background_out:
            params["background_out"] = background_out
        if tag:
            params["tag"] = tag
        if allow_cuda is not None:
            params["allow_cuda"] = bool(allow_cuda) and self.allow_cuda
        return await self.request("separate_vocals", params, progress, timeout=OP_TIMEOUT_S + 3.0 * seconds)

    async def release_gpu(self) -> None:
        """Hand the GPU memory back (models are dropped; they reload on the next request). No-op when not running.
        A background preload that is running (a Whisper load, or a Redub preload's separation) is asked to stop
        first, so the caller does not wait for it."""
        if not self.alive():
            return
        self.client.cancel_current("load_asr")
        self.client.cancel_current("separate_vocals", lambda msg: msg.get("tag") in self.background_tags)
        try:
            await self.request("release_gpu", timeout=120.0, start=False)
        except (SpeechError, Cancelled):
            log.exception("could not release the speech worker's GPU memory")

    async def status(self) -> dict[str, Any] | None:
        if not self.alive():
            return None
        try:
            return await self.request("status", timeout=30.0, start=False)
        except (SpeechError, Cancelled):
            return None

    async def stop(self) -> None:
        await self.client.stop()


_WORKER: SpeechWorker | None = None


def get(runtime: dict[str, Any], data_root: str) -> SpeechWorker:
    global _WORKER
    if _WORKER is None:
        from . import config  # noqa: PLC0415
        _WORKER = SpeechWorker(os.path.join(data_root, "runtime", "models"), config.backend(runtime) == "cuda")
    return _WORKER


async def yield_gpu() -> None:
    """Called before a job that needs the whole GPU (speech, voice conversion): drop the worker's models from it."""
    if _WORKER is not None:
        await _WORKER.release_gpu()


async def shutdown() -> None:
    if _WORKER is not None:
        await _WORKER.stop()
