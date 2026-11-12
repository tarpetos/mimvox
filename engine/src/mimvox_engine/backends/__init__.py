"""Text-to-speech backends: the seam between the speech pipeline and the models that speak.

The pipeline (``tts.generate``, ``voices.warm``, Redub ``respeak`` / ``dub``) never talks to a model directly. Per
segment it picks a backend for the segment's language and awaits

    path = await backend.synthesize(text, lang, reference, SynthParams(out_path=..., ...))

which writes one mono WAV (any sample rate) to ``params.out_path`` and returns that path. RVC (when the voice is trained),
resampling to 44.1 kHz, pauses and slot fitting happen afterwards in the pipeline, the same for every backend.

The one backend today is OmniVoice (``backends/omnivoice.py``): every language, Ukrainian and English included, in a
persistent worker process (IndexTTS-2.5 was dropped: OmniVoice matched or beat it on English at a quarter of the time
and memory, see docs/performance.md).

Adding a backend:

1. Write ``backends/<name>.py`` with a subclass of :class:`TtsBackend`: set ``name`` and ``languages`` (ISO 639-1
   codes) and implement ``synthesize``. In-process GPU work belongs on the engine's single GPU thread
   (``await aio.run_gpu(fn, ...)``), a model in another process behind a ``workerproc.WorkerClient``; blocking CPU
   work goes to ``aio.run_io``. Never block the event loop.
2. Optional hooks: ``load(runtime)`` (called at start-up and after a training hands the GPU back), ``release()``
   (called before a training takes the GPU; free VRAM and host RAM), ``status()``, ``is_warm`` / ``forget_reference``
   (speaker-conditioning caches), ``count_tokens`` (text segmentation).
3. Register it in :func:`_builtin` below. The first registered backend that speaks a language wins.

``languages()`` is what Redub's language guard and ``tts.generate``'s ``language`` parameter accept.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SynthParams:
    out_path: str                 # where to write the WAV (mono, 16-bit or float, any sample rate)
    temperature: float = 0.8      # "expressiveness" (0.1-1.5)
    quality: str = "best"         # "best" | "fast"
    duration_factor: float = 1.0  # length of the speech relative to natural (1 / speed; Redub slot fitting); <1 faster
    first: bool = False           # the first segment of a job: trade a little quality for a faster first sound


class TtsBackend(abc.ABC):
    """One text-to-speech model. All methods are called on the engine's event loop."""

    name: str = "?"
    languages: frozenset[str] = frozenset()

    @abc.abstractmethod
    async def synthesize(self, text: str, lang: str, reference: str, params: SynthParams) -> str:
        """Speak ``text`` in language ``lang`` (ISO 639-1) with the voice of ``reference`` (the voice's
        ``reference.wav``); write a WAV to ``params.out_path`` and return its path. Raise
        ``registry.EngineError("not_ready", ...)`` when the model is not loaded."""

    def status(self) -> str:
        """``ready`` | ``loading`` | ``paused`` | ``missing``."""
        return "ready"

    async def load(self, runtime: dict[str, Any]) -> None:
        """Load the model (start-up, and after a training gave the GPU back)."""

    async def release(self) -> None:
        """Free the GPU and host RAM (a training takes the GPU). ``load`` brings it back."""

    async def close(self) -> None:
        """Engine shutdown: stop any process the backend started and await its exit."""

    def count_tokens(self, text: str) -> int:
        """Rough size of ``text`` for the segmentation of ``tts.generate`` (model tokens when known)."""
        return max(1, len(text) // 3)

    def is_warm(self, reference: str) -> bool:
        """True when the speaker conditioning of ``reference`` is cached (``voices.warm`` has nothing to do)."""
        return False

    def forget_reference(self) -> None:
        """Drop cached speaker conditioning (``reference.wav`` was replaced). Callable from any thread."""


_BACKENDS: list[TtsBackend] = []


def register(backend: TtsBackend) -> None:
    _BACKENDS.append(backend)


def all_backends() -> list[TtsBackend]:
    if not _BACKENDS:
        _builtin()
    return list(_BACKENDS)


def for_language(lang: str) -> TtsBackend | None:
    """The backend that speaks ``lang`` (first registered wins), or None."""
    lang = (lang or "").lower()
    for b in all_backends():
        if lang in b.languages:
            return b
    return None


def languages() -> set[str]:
    out: set[str] = set()
    for b in all_backends():
        out |= set(b.languages)
    return out


def primary() -> "TtsBackend":
    """The main speech backend (OmniVoice): its state is the ``tts`` entry of the ``models`` event."""
    from .omnivoice import OMNIVOICE  # noqa: PLC0415
    all_backends()
    return OMNIVOICE


def _builtin() -> None:
    from .omnivoice import OMNIVOICE  # noqa: PLC0415
    register(OMNIVOICE)
