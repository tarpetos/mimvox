"""JSON for values that may hold paths: a ``Path`` is written as its string."""

from __future__ import annotations

from pathlib import PurePath
from typing import Any


def json_default(value: Any) -> str:
    if isinstance(value, PurePath):
        return str(value)
    msg = f"{type(value).__name__} is not JSON serializable"
    raise TypeError(msg)
