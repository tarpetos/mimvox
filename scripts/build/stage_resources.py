#!/usr/bin/env python3
"""Stage the runtime resources the installers bundle into app/target/bundle-resources/.

Packaging from a staged copy (instead of pointing cargo-packager at the repo folders) keeps caches
(__pycache__, *.pyc, .venv, ...) and user data out of the installer, and fails early with a clear
message when a required folder is missing.

Layout produced (this is also the layout the app looks for at runtime, see app/Packager.toml):

    engine/src/mimvox_engine/...   engine sources (run with the engine venv interpreter)
    presets/                           guided reading presets
    tools/setup/                       runtime setup (setup.py), triggered by the app on first run
    contract/                          engine protocol
    setup.sh, setup.cmd                optional, if present in the repo root
    DISCLAIMER.md, THIRD_PARTY_NOTICES.md, LICENSE*   optional, if present

Usage: python scripts/build/stage_resources.py   (stdlib only; the build scripts run it through `uv run`)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "app" / "target" / "bundle-resources"

REQUIRED_DIRS = ["engine/src", "presets", "tools/setup", "contract"]
OPTIONAL_FILES = ["setup.sh", "setup.cmd", "DISCLAIMER.md", "THIRD_PARTY_NOTICES.md", "LICENSE", "LICENSE.md"]
IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".venv", "venv", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "*.egg-info", ".DS_Store", "Thumbs.db",
)


def main() -> int:
    missing = [d for d in REQUIRED_DIRS if not (REPO / d).is_dir()]
    if missing:
        print(f"[stage] missing required folders in {REPO}: {', '.join(missing)}", file=sys.stderr)
        print("[stage] these are produced by other parts of the repo; check out a complete tree.", file=sys.stderr)
        return 1
    if not (REPO / "engine/src/mimvox_engine").is_dir():
        print("[stage] engine/src/mimvox_engine is missing.", file=sys.stderr)
        return 1

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    count = 0
    for rel in REQUIRED_DIRS:
        shutil.copytree(REPO / rel, STAGE / rel, ignore=IGNORE)
    for rel in OPTIONAL_FILES:
        src = REPO / rel
        if src.is_file():
            shutil.copy2(src, STAGE / rel)
    for path in STAGE.rglob("*"):
        count += path.is_file()
    print(f"[stage] {count} files staged in {STAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
