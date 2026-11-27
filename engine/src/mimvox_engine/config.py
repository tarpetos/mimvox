from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import EngineError


def load_runtime(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def runtime_path(runtime: dict[str, Any], section: str, key: str) -> Path | None:
    value = runtime.get(section, {}).get(key)
    return Path(value).absolute() if value else None


def applio_dir(runtime: dict[str, Any]) -> Path | None:
    return runtime_path(runtime, "paths", "applio")


def runtime_root(runtime: dict[str, Any]) -> Path | None:
    applio = applio_dir(runtime)
    if applio:
        return applio.parent
    model = runtime_path(runtime, "paths", "omnivoice_model")
    if model:
        return model.parent.parent
    engine = runtime_path(runtime, "python", "engine")
    return engine.parents[3] if engine and len(engine.parents) > 3 else None


def rvc_python(runtime: dict[str, Any]) -> Path | None:
    return runtime_path(runtime, "python", "rvc")


def backend(runtime: dict[str, Any]) -> str:
    return str(runtime.get("device", {}).get("backend", "cuda")).lower()


def vendor(runtime: dict[str, Any]) -> str:
    """ROCm torch also uses the ``cuda`` backend: this tells ``nvidia`` from ``amd`` (else ``""``)."""
    return str(runtime.get("device", {}).get("vendor", "")).lower()


@dataclass(frozen=True)
class DataDir:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def voices(self) -> Path:
        return self.root / "voices"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    def voice(self, voice_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", voice_id or ""):
            raise EngineError("bad_request", f"invalid voice id {voice_id!r}")
        return self.voices / voice_id

    def reference(self, voice_id: str) -> Path:
        return self.voice(voice_id) / "reference.wav"

    def rvc_model(self, voice_id: str) -> tuple[Path, Path]:
        d = self.voice(voice_id) / "rvc"
        return d / "model.pth", d / "model.index"

    def ensure(self) -> None:
        for d in (self.voices, self.outputs, self.tmp):
            d.mkdir(parents=True, exist_ok=True)


def data_dir(ctx: Any) -> DataDir:
    return DataDir(Path(ctx.data_dir).absolute())
