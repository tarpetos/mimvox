"""runtime.toml parsing and data-directory layout helpers."""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from typing import Any

from .registry import EngineError


def load_runtime(path: str) -> dict[str, Any]:
    """Parse runtime.toml (see contract/engine-protocol.md)."""
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def runtime_path(runtime: dict[str, Any], section: str, key: str) -> str | None:
    value = runtime.get(section, {}).get(key)
    return os.path.abspath(value) if value else None


def applio_dir(runtime: dict[str, Any]) -> str | None:
    return runtime_path(runtime, "paths", "applio")


def rvc_python(runtime: dict[str, Any]) -> str | None:
    return runtime_path(runtime, "python", "rvc")


def backend(runtime: dict[str, Any]) -> str:
    return str(runtime.get("device", {}).get("backend", "cuda")).lower()


@dataclass(frozen=True)
class DataDir:
    """Layout of ``<data>`` (see the contract)."""

    root: str

    @property
    def voices(self) -> str:
        return os.path.join(self.root, "voices")

    @property
    def outputs(self) -> str:
        return os.path.join(self.root, "outputs")

    @property
    def tmp(self) -> str:
        return os.path.join(self.root, "tmp")

    def voice(self, voice_id: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", voice_id or ""):
            raise EngineError("bad_request", f"invalid voice id {voice_id!r}")
        return os.path.join(self.voices, voice_id)

    def reference(self, voice_id: str) -> str:
        return os.path.join(self.voice(voice_id), "reference.wav")

    def rvc_model(self, voice_id: str) -> tuple[str, str]:
        d = os.path.join(self.voice(voice_id), "rvc")
        return os.path.join(d, "model.pth"), os.path.join(d, "model.index")

    def ensure(self) -> None:
        for d in (self.voices, self.outputs, self.tmp):
            os.makedirs(d, exist_ok=True)


def data_dir(ctx: Any) -> DataDir:
    return DataDir(os.path.abspath(ctx.data_dir))
