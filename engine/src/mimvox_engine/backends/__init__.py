"""Text-to-speech backends: the first registered backend that speaks a language and is ``available()`` wins."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SynthParams:
    out_path: str
    temperature: float = 0.8
    quality: str = "best"
    duration_factor: float = 1.0  # 1 / speed: below 1 is faster
    first: bool = False
    reference_text: str | None = None


class TtsBackend(abc.ABC):
    """One text-to-speech model. All methods are called on the engine's event loop."""

    name: str = "?"
    languages: frozenset[str] = frozenset()

    @abc.abstractmethod
    async def synthesize(self, text: str, lang: str, reference: str, params: SynthParams) -> str:
        """Write a WAV to ``params.out_path`` and return its path; ``EngineError("not_ready")`` when not loaded."""

    def status(self) -> str:
        return "ready"

    def available(self) -> bool:
        """The main backend always; an optional one only while ready, so its languages fall through otherwise."""
        return True

    async def load(self, runtime: dict[str, Any]) -> None:
        """Called at start-up and after a training gives the GPU back."""

    async def release(self) -> None:
        """Free the GPU and host RAM before a training takes the GPU."""

    async def close(self) -> None:
        """Stop any process the backend started and await its exit."""

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)

    def is_warm(self, reference: str) -> bool:
        return False

    def forget_reference(self) -> None:
        """Callable from any thread."""


_BACKENDS: list[TtsBackend] = []


def register(backend: TtsBackend) -> None:
    _BACKENDS.append(backend)


def all_backends() -> list[TtsBackend]:
    if not _BACKENDS:
        _builtin()
    return list(_BACKENDS)


def for_language(lang: str) -> TtsBackend | None:
    lang = (lang or "").lower()
    for b in all_backends():
        if lang in b.languages and b.available():
            return b
    return None


def languages() -> set[str]:
    out: set[str] = set()
    for b in all_backends():
        out |= set(b.languages)
    return out


def primary() -> "TtsBackend":
    from .omnivoice import OMNIVOICE  # noqa: PLC0415
    all_backends()
    return OMNIVOICE


def _builtin() -> None:
    from .holos import HOLOS  # noqa: PLC0415
    from .omnivoice import OMNIVOICE  # noqa: PLC0415
    register(HOLOS)
    register(OMNIVOICE)
