"""Reading presets (``presets/<lang>/<id>.md``) and the ``presets.list`` handler.

Layout: one folder per language, named by its ISO 639-1 code (``presets/en``, ``presets/uk``, ``presets/de``, ...),
each with ``quick.md``, ``standard.md`` and ``pro.md``. The texts are written for each language (not translated), so
their word counts differ while the reading times match. A flat legacy layout (``presets/*.md``) is read as English.

File format (hand-parsed, no YAML dependency)::

    ---
    id: standard
    title: Standard
    description: One sentence shown on the card.
    target_minutes: 6.5
    trains_rvc: true
    language: uk
    ---
    ## Part title
    First paragraph of the part.

    Second paragraph of the part.

    ## Next part title
    ...

``## `` headings start a new part; blank lines separate paragraphs. ``title`` and ``description`` are UI strings
(English); part titles and texts are in the preset's language. ``est_seconds`` is computed from the word count at the
language's reading pace (``dataset.lang``: 145 words per minute for English). The folder of the language comes first;
``language:`` in the front matter is informational. The presets root is ``$MIMVOX_PRESETS`` when set, otherwise
``presets/`` at the repository (or bundle) root, found by walking up from this file
(``engine/src/mimvox_engine/dataset`` -> repo root).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ..registry import EngineError, method
from . import lang
from .text import count_words, est_seconds

PRESET_ORDER = ["quick", "standard", "pro"]


def presets_dir() -> Path:
    env = os.environ.get("MIMVOX_PRESETS")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "presets"
        if cand.is_dir() and (any(cand.glob("*/*.md")) or any(cand.glob("*.md"))):
            return cand
    return here.parents[4] / "presets"


def _parse_bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def parse_preset(raw: str, fallback_id: str, language: str = lang.DEFAULT) -> dict[str, Any]:
    raw = raw.replace("\r\n", "\n").lstrip("﻿")
    meta: dict[str, str] = {}
    body = raw
    m = re.match(r"\s*---\n(.*?)\n---\n?(.*)\Z", raw, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
        body = m.group(2)
    wpm = lang.get(language).wpm
    parts: list[dict[str, Any]] = []
    title: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if title is None:
            return
        paras = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(buf)) if p.strip()]
        text = "\n\n".join(" ".join(p.split()) for p in paras)
        if text:
            parts.append({"index": len(parts) + 1, "title": title, "text": text, "est_seconds": est_seconds(count_words(text), wpm)})

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            title, buf = line[3:].strip(), []
        elif title is not None:
            buf.append(line)
    flush()
    total = sum(p["est_seconds"] for p in parts) / 60.0
    return {
        "id": meta.get("id", fallback_id),
        "title": meta.get("title", fallback_id.title()),
        "description": meta.get("description", ""),
        "target_minutes": float(meta["target_minutes"]) if meta.get("target_minutes") else round(total, 1),
        "trains_rvc": _parse_bool(meta.get("trains_rvc", "false")),
        "language": language,
        "parts": parts,
    }


def _language_dir(root: Path, code: str) -> Path | None:
    sub = root / code
    if sub.is_dir() and any(sub.glob("*.md")):
        return sub
    if code == lang.DEFAULT and any(root.glob("*.md")):  # legacy flat layout
        return root
    return None


def available_languages() -> list[str]:
    """Codes that have presets, in the order of ``lang.LANGUAGES`` (English first)."""
    root = presets_dir()
    if not root.is_dir():
        return []
    return [code for code in lang.LANGUAGES if _language_dir(root, code) is not None]


def load_presets(language: str = lang.DEFAULT) -> list[dict[str, Any]]:
    root = presets_dir()
    if not root.is_dir():
        raise EngineError("not_found", f"Presets folder not found: {root}")
    d = _language_dir(root, language)
    if d is None:
        return []
    found = []
    for f in d.glob("*.md"):
        if f.stem.lower() in ("readme",):
            continue
        p = parse_preset(f.read_text(encoding="utf-8"), f.stem, language)
        if p["parts"]:
            found.append(p)
    found.sort(key=lambda p: (PRESET_ORDER.index(p["id"]) if p["id"] in PRESET_ORDER else 99, p["id"]))
    return found


def get_preset(preset_id: str, language: str | None = lang.DEFAULT) -> dict[str, Any] | None:
    """The preset ``preset_id`` in ``language`` (English when ``None`` or when that language has no such preset)."""
    code = language if language in lang.LANGUAGES else lang.DEFAULT
    found = next((p for p in load_presets(code) if p["id"] == preset_id), None)
    if found is None and code != lang.DEFAULT:
        found = next((p for p in load_presets(lang.DEFAULT) if p["id"] == preset_id), None)
    return found


@method("presets.list")
def presets_list(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    try:
        code = lang.normalize_code(params.get("language")) or lang.DEFAULT
    except ValueError as exc:
        raise EngineError("bad_request", str(exc)) from None
    if not presets_dir().is_dir():
        raise EngineError("not_found", f"Presets folder not found: {presets_dir()}")
    available = available_languages()
    if code not in available:
        names = ", ".join(available) or "none"
        raise EngineError("bad_request", f"No reading presets for {lang.get(code).name} ({code}); available: {names}")
    return {
        "language": code,
        "languages": [lang.describe(c) for c in available],
        "presets": load_presets(code),
    }
