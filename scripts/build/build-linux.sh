#!/usr/bin/env bash
# Build the Mimvox desktop installers on Linux: an AppImage and a .deb.
# Run it from anywhere:  ./scripts/build/build-linux.sh
#
#   --formats appimage,deb   which installers to build (default: both; comma separated)
#   --skip-compile           reuse app/target/release (only re-stage resources and re-package)
#
# Output: dist/mimvox_<version>_amd64.deb and dist/*.AppImage (dist/ is git-ignored).
#
# What it does: checks the toolchain, installs cargo-packager when missing, stages the runtime resources
# (engine sources, presets, setup tools, contract; NOT the heavy models) into app/target/bundle-resources,
# compiles the release binary and runs `cargo packager` with app/Packager.toml. The header of that file documents
# where the installed app finds its bundled resources at runtime.
#
# Knobs: CARGO_PACKAGER_VERSION (default below), CARGO_BUILD_JOBS (default 6), CARGO_TARGET_DIR.
# Cargo runs under `flock /tmp/mimvox-cargo.lock` (when flock exists) so only one Rust build runs at a time on a shared machine.
set -euo pipefail

PACKAGER_VERSION="${CARGO_PACKAGER_VERSION:-0.11.8}"
BIN_NAME="mimvox"
# Cap parallel compiler jobs (a full Rust + wgpu build is memory hungry); override with CARGO_BUILD_JOBS.
JOBS="${CARGO_BUILD_JOBS:-6}"
export CARGO_BUILD_JOBS="$JOBS"
LOCK=()
if command -v flock >/dev/null 2>&1; then LOCK=(flock /tmp/mimvox-cargo.lock); fi

# The repo root is two folders above this script, so the build works from any directory
# and from a path that contains spaces.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SELF")/../.." && pwd)"
APP_DIR="$REPO_ROOT/app"
DIST_DIR="$REPO_ROOT/dist"
TARGET_DIR="${CARGO_TARGET_DIR:-$APP_DIR/target}"
cd "$APP_DIR"

FORMATS="appimage,deb"
SKIP_COMPILE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --formats) FORMATS="${2:?--formats needs a value}"; shift 2 ;;
        --skip-compile) SKIP_COMPILE=1; shift ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$SELF"; exit 0 ;;
        *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done

need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "$1 is required but is not installed. $2" >&2
        exit 1
    fi
}
need cargo "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
need uv "Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
need pkg-config "Install it: sudo apt install pkg-config"

if ! pkg-config --exists alsa; then
    echo "ALSA development files are missing (needed by the audio backend)." >&2
    echo "  Debian/Ubuntu: sudo apt install libasound2-dev   Fedora: sudo dnf install alsa-lib-devel" >&2
    exit 1
fi
if [[ ",$FORMATS," == *",deb,"* ]]; then
    need dpkg-deb "The .deb needs dpkg (Debian/Ubuntu). Use --formats appimage on other distributions."
fi
if [[ ",$FORMATS," == *",appimage,"* ]]; then
    need file "Install it: sudo apt install file"
    # cargo-packager downloads linuxdeploy on first use, so the first AppImage build needs the network.
    # Run the AppImage tools without FUSE (containers, WSL) and without their bundled, outdated `strip`,
    # which chokes on modern .relr.dyn sections.
    export APPIMAGE_EXTRACT_AND_RUN=1
    export NO_STRIP=true
fi

# The version in Packager.toml must match the crate, otherwise the installer lies about it.
cargo_version="$(sed -n 's/^version *= *"\(.*\)"/\1/p' Cargo.toml | head -n1)"
packager_version="$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' Packager.toml | head -n1)"
if [ "$cargo_version" != "$packager_version" ]; then
    echo "Version mismatch: app/Cargo.toml has $cargo_version, app/Packager.toml has $packager_version." >&2
    exit 1
fi

if ! command -v cargo-packager >/dev/null 2>&1; then
    echo "[build-linux] Installing cargo-packager $PACKAGER_VERSION..." >&2
    "${LOCK[@]}" cargo install cargo-packager --locked -j "$JOBS" --version "$PACKAGER_VERSION"
fi

echo "[build-linux] Staging bundled resources..." >&2
# The app runs setup with the bundled uv (a static musl build runs on every distribution).
case "$(uname -m)" in
    aarch64|arm64) UV_TARGET=aarch64-unknown-linux-musl ;;
    *) UV_TARGET=x86_64-unknown-linux-musl ;;
esac
uv run --no-project python "$REPO_ROOT/scripts/build/stage_resources.py" --uv "$UV_TARGET"

if [ "$SKIP_COMPILE" = "0" ]; then
    echo "[build-linux] Compiling the release binary..." >&2
    "${LOCK[@]}" cargo build --release --locked -j "$JOBS"
fi
if [ ! -x "$TARGET_DIR/release/$BIN_NAME" ]; then
    echo "Release binary not found: $TARGET_DIR/release/$BIN_NAME" >&2
    exit 1
fi

# cargo-packager resolves the binary through Packager.toml's binaries-dir ("target/release", relative to app/) even when
# --binaries-dir points elsewhere, so a build into another CARGO_TARGET_DIR is copied there first.
RELEASE_DIR="$TARGET_DIR/release"
if [ "$(realpath "$RELEASE_DIR")" != "$(realpath -m "$APP_DIR/target/release")" ]; then
    mkdir -p "$APP_DIR/target/release"
    cp -f "$RELEASE_DIR/$BIN_NAME" "$APP_DIR/target/release/$BIN_NAME"
    RELEASE_DIR="$APP_DIR/target/release"
fi

mkdir -p "$DIST_DIR"
echo "[build-linux] Packaging ($FORMATS)..." >&2
"${LOCK[@]}" cargo packager --config Packager.toml --formats "$FORMATS" \
    --binaries-dir "$RELEASE_DIR" --out-dir "$DIST_DIR"

echo "[build-linux] Done. Installers in $DIST_DIR:" >&2
ls -lh "$DIST_DIR" | sed 's/^/  /' >&2
