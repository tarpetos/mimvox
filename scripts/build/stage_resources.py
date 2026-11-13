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
    bin/uv(.exe)                       uv, which the app runs setup with (--uv TARGET; two targets: a macOS universal
                                       binary made with lipo; Linux: bin/uv.gz, see stage_uv)
    DISCLAIMER.md, THIRD_PARTY_NOTICES.md, LICENSE*   optional, if present

setup.sh / setup.cmd are for developers working in the repository and are not bundled: the installed app runs
tools/setup/setup.py itself (app/src/ui/setuprun.rs).

Usage: python scripts/build/stage_resources.py [--uv TARGET[,TARGET]]   (stdlib only; the build scripts run it
through `uv run` and pass the uv target of the platform they build for)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "app" / "target" / "bundle-resources"

REQUIRED_DIRS = ["engine/src", "presets", "tools/setup", "contract"]
OPTIONAL_FILES = ["DISCLAIMER.md", "THIRD_PARTY_NOTICES.md", "LICENSE", "LICENSE.md"]
# The uv the installers carry (keep it equal to UV_VERSION in setup.sh / setup.cmd).
UV_VERSION = "0.12.18"
UV_RELEASES = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}"
IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".venv", "venv", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "*.egg-info", ".DS_Store", "Thumbs.db",
)


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - a fixed https URL
        return response.read()


def uv_binary(target: str) -> bytes:
    """The uv executable from the pinned release for ``target``, checked against the release's SHA-256."""
    archive = f"uv-{target}.zip" if "windows" in target else f"uv-{target}.tar.gz"
    data = fetch(f"{UV_RELEASES}/{archive}")
    expected = fetch(f"{UV_RELEASES}/{archive}.sha256").decode().split()[0]
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit(f"[stage] {archive}: checksum mismatch")
    name = "uv.exe" if "windows" in target else "uv"
    if archive.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            member = next(m for m in z.namelist() if m.rsplit("/", 1)[-1] == name)
            return z.read(member)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        member = next(m for m in t.getmembers() if m.name.rsplit("/", 1)[-1] == name and m.isfile())
        extracted = t.extractfile(member)
        assert extracted is not None
        return extracted.read()


def stage_uv(targets: list[str]) -> None:
    """``bin/uv`` for one target, or a universal macOS binary (lipo) for two."""
    out_dir = STAGE / "bin"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "uv.exe" if any("windows" in t for t in targets) else "uv"
    out = out_dir / name
    if any("linux" in t for t in targets):
        # The AppImage tool (linuxdeploy) rewrites every ELF file in the AppDir and breaks the static uv, so Linux
        # gets it gzipped; the app unpacks it on first use (app/src/ui/setuprun.rs, uv_path).
        import gzip  # noqa: PLC0415
        packed = out_dir / "uv.gz"
        packed.write_bytes(gzip.compress(uv_binary(targets[0]), compresslevel=9))
        print(f"[stage] uv {UV_VERSION} ({targets[0]}) -> {packed}")
        return
    if len(targets) == 1:
        out.write_bytes(uv_binary(targets[0]))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            parts = []
            for t in targets:
                part = Path(tmp) / t
                part.write_bytes(uv_binary(t))
                parts.append(str(part))
            subprocess.run(["lipo", "-create", "-output", str(out), *parts], check=True)
    out.chmod(0o755)
    print(f"[stage] uv {UV_VERSION} ({', '.join(targets)}) -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage the installers' bundled resources.")
    parser.add_argument("--uv", metavar="TARGET[,TARGET]",
                        help="also bundle uv for this target (e.g. x86_64-unknown-linux-musl); two comma-separated "
                             "targets make a macOS universal binary")
    args = parser.parse_args()
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
    if args.uv:
        stage_uv([t.strip() for t in args.uv.split(",") if t.strip()])
    for path in STAGE.rglob("*"):
        count += path.is_file()
    print(f"[stage] {count} files staged in {STAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
