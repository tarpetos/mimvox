"""Optional components the user turned off: they stay on disk and the engine ignores them until turned on again."""
from __future__ import annotations

import json
import logging
import os

from .registry import EngineError, method

log = logging.getLogger("mimvox.components")

FILE = "components.json"
OPTIONAL = ("training", "translation", "better_translation", "native_accent", "music_removal", "ukrainian")

_off: set[str] = set()


def _path(data_root: str) -> str:
    return os.path.join(data_root, FILE)


def _read(data_root: str) -> set[str]:
    try:
        with open(_path(data_root), encoding="utf-8") as fh:
            found = json.load(fh).get("off") or []
    except FileNotFoundError:
        return set()
    except (OSError, ValueError, AttributeError) as exc:
        log.warning("cannot read %s: %s", _path(data_root), exc)
        return set()
    return {c for c in found if c in OPTIONAL}


def load(data_root: str) -> None:
    """Read once at start; the app restarts the engine after a change."""
    _off.clear()
    _off.update(_read(data_root))
    if _off:
        log.info("turned off: %s", ", ".join(sorted(_off)))


def off(component: str) -> bool:
    return component in _off


def turned_off() -> list[str]:
    return [c for c in OPTIONAL if c in _off]


def _save(data_root: str, names: set[str]) -> None:
    tmp = _path(data_root) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"off": [c for c in OPTIONAL if c in names]}, fh, indent=2)
    os.replace(tmp, _path(data_root))


@method("components.set")
async def components_set(ctx, params):  # noqa: ANN001, ANN201 - registry handler
    name, on = params.get("id"), params.get("on")
    if name not in OPTIONAL or not isinstance(on, bool):
        raise EngineError("bad_request", f"'id' must be one of {', '.join(OPTIONAL)} and 'on' true or false")
    from . import aio  # noqa: PLC0415
    names = await aio.run_io(_read, ctx.data_dir)
    names = (names - {name}) if on else (names | {name})
    await aio.run_io(_save, ctx.data_dir, names)
    return {"off": [c for c in OPTIONAL if c in names], "restart": True}
