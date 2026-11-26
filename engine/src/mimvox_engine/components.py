"""Optional components the user turned off: they stay on disk and the engine ignores them until turned on again."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .registry import EngineError, method

log = logging.getLogger("mimvox.components")

FILE = "components.json"
SPEECH_MODELS = ("omnivoice", "voxcpm", "chatterbox")
OPTIONAL = (
    "omnivoice",
    "training",
    "translation",
    "better_translation",
    "native_accent",
    "music_removal",
    "ukrainian",
    "voxcpm",
    "chatterbox",
)

_off: set[str] = set()


def _path(data_root: str | Path) -> Path:
    return Path(data_root) / FILE


def _read(data_root: str | Path) -> set[str]:
    try:
        found = json.loads(_path(data_root).read_text(encoding="utf-8")).get("off") or []
    except FileNotFoundError:
        return set()
    except (OSError, ValueError, AttributeError) as exc:
        log.warning("cannot read %s: %s", _path(data_root), exc)
        return set()
    return {c for c in found if c in OPTIONAL}


def load(data_root: str | Path) -> None:
    """Read once at start; the app restarts the engine after a change."""
    _off.clear()
    _off.update(_read(data_root))
    if _off:
        log.info("turned off: %s", ", ".join(sorted(_off)))


def off(component: str) -> bool:
    return component in _off


def turned_off() -> list[str]:
    return [c for c in OPTIONAL if c in _off]


def _save(data_root: str | Path, names: set[str]) -> None:
    path = _path(data_root)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump({"off": [c for c in OPTIONAL if c in names]}, fh, indent=2)
    tmp.replace(path)


@method("components.set")
async def components_set(ctx, params):  # noqa: ANN001, ANN201 - registry handler
    name, on = params.get("id"), params.get("on")
    if name not in OPTIONAL or not isinstance(on, bool):
        raise EngineError("bad_request", f"'id' must be one of {', '.join(OPTIONAL)} and 'on' true or false")
    from . import aio

    names = await aio.run_io(_read, ctx.data_dir)
    names = (names - {name}) if on else (names | {name})
    if name in SPEECH_MODELS and not on:
        from . import backends

        left = {b.name for b in backends.selectable()} & set(SPEECH_MODELS) - names
        if not left:
            raise EngineError("bad_request", "at least one voice-cloning model has to stay on")
    await aio.run_io(_save, ctx.data_dir, names)
    return {"off": [c for c in OPTIONAL if c in names], "restart": True}
