#!/usr/bin/env bash
# Build the Mimvox desktop installers on macOS: a .app bundle and a .dmg.
# Run it from anywhere:  ./scripts/build/build-mac.sh
#
#   --universal              build one binary for Apple Silicon + Intel (default: arm64 only, the supported target:
#                            GPU acceleration uses Metal/MPS, which Intel Macs lack)
#   --formats app,dmg        which packages to build (default: both; comma separated)
#   --skip-compile           reuse the compiled binary (only re-stage resources and re-package)
#
# Output: dist/Mimvox.app and dist/Mimvox_<version>_<aarch64|universal>.dmg
# (dist/ is git-ignored).
#
# Signing is opt-in. Set APPLE_SIGNING_IDENTITY (e.g. "Developer ID Application: Name (TEAMID)") to sign the app;
# cargo-packager then also notarizes it when APPLE_ID / APPLE_PASSWORD / APPLE_TEAM_ID (or the APPLE_API_* trio, or
# APPLE_KEYCHAIN_PROFILE) are set. Unsigned builds work for personal use: after copying the app from the .dmg, run
#   xattr -dr com.apple.quarantine "/Applications/Mimvox.app"
#
# What it does: checks the toolchain, installs cargo-packager when missing, stages the runtime resources
# (engine sources, presets, setup tools, contract; NOT the heavy models) into app/target/bundle-resources,
# compiles the release binary and runs `cargo packager` with app/Packager.toml. Resources end up in
# Contents/Resources of the .app (see the header of app/Packager.toml).
#
# Knobs: CARGO_PACKAGER_VERSION (default below), CARGO_BUILD_JOBS (default 6), CARGO_TARGET_DIR.
set -euo pipefail

PACKAGER_VERSION="${CARGO_PACKAGER_VERSION:-0.11.8}"
BIN_NAME="mimvox"
# Cap parallel compiler jobs (a full Rust + wgpu build is memory hungry); override with CARGO_BUILD_JOBS.
JOBS="${CARGO_BUILD_JOBS:-6}"
export CARGO_BUILD_JOBS="$JOBS"

# The repo root is two folders above this script, so the build works from any directory
# and from a path that contains spaces.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SELF")/../.." && pwd)"
APP_DIR="$REPO_ROOT/app"
DIST_DIR="$REPO_ROOT/dist"
TARGET_DIR="${CARGO_TARGET_DIR:-$APP_DIR/target}"
cd "$APP_DIR"

FORMATS="app,dmg"
UNIVERSAL=0
SKIP_COMPILE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --universal) UNIVERSAL=1; shift ;;
        --formats) FORMATS="${2:?--formats needs a value}"; shift 2 ;;
        --skip-compile) SKIP_COMPILE=1; shift ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$SELF"; exit 0 ;;
        *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done

if [ "$(uname -s)" != "Darwin" ]; then
    echo "build-mac.sh must run on macOS." >&2
    exit 1
fi
need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "$1 is required but is not installed. $2" >&2
        exit 1
    fi
}
need cargo "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
need rustup "Install Rust with rustup (needed to add the macOS targets): https://rustup.rs"
need uv "Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
if ! xcode-select -p >/dev/null 2>&1; then
    echo "The Xcode command line tools are missing. Install them: xcode-select --install" >&2
    exit 1
fi
if [[ ",$FORMATS," == *",dmg,"* ]]; then
    need hdiutil "hdiutil ships with macOS."
fi

# The oldest macOS the app declares (Packager.toml [macos] minimum-system-version).
export MACOSX_DEPLOYMENT_TARGET=13.0

# The version in Packager.toml must match the crate, otherwise the installer lies about it.
cargo_version="$(sed -n 's/^version *= *"\(.*\)"/\1/p' Cargo.toml | head -n1)"
packager_version="$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' Packager.toml | head -n1)"
if [ "$cargo_version" != "$packager_version" ]; then
    echo "Version mismatch: app/Cargo.toml has $cargo_version, app/Packager.toml has $packager_version." >&2
    exit 1
fi

if ! command -v cargo-packager >/dev/null 2>&1; then
    echo "[build-mac] Installing cargo-packager $PACKAGER_VERSION..." >&2
    cargo install cargo-packager --locked -j "$JOBS" --version "$PACKAGER_VERSION"
fi

echo "[build-mac] Staging bundled resources..." >&2
uv run --no-project python "$REPO_ROOT/scripts/build/stage_resources.py"

ARM=aarch64-apple-darwin
X86=x86_64-apple-darwin
if [ "$UNIVERSAL" = "1" ]; then
    PKG_TARGET=universal-apple-darwin
    BIN_DIR="$TARGET_DIR/universal-apple-darwin/release"
else
    PKG_TARGET=$ARM
    BIN_DIR="$TARGET_DIR/$ARM/release"
fi

if [ "$SKIP_COMPILE" = "0" ]; then
    targets=("$ARM")
    if [ "$UNIVERSAL" = "1" ]; then targets+=("$X86"); fi
    for t in "${targets[@]}"; do
        rustup target add "$t"
        echo "[build-mac] Compiling the release binary for $t..." >&2
        cargo build --release --locked -j "$JOBS" --target "$t"
    done
    if [ "$UNIVERSAL" = "1" ]; then
        echo "[build-mac] Merging into a universal binary (lipo)..." >&2
        mkdir -p "$BIN_DIR"
        lipo -create -output "$BIN_DIR/$BIN_NAME" \
            "$TARGET_DIR/$ARM/release/$BIN_NAME" "$TARGET_DIR/$X86/release/$BIN_NAME"
    fi
fi
if [ ! -x "$BIN_DIR/$BIN_NAME" ]; then
    echo "Release binary not found: $BIN_DIR/$BIN_NAME" >&2
    exit 1
fi

# Optional signing: cargo-packager takes the identity from the config only, so write a throw-away copy of
# Packager.toml (it must sit in app/ because paths in it are relative to its folder).
CONFIG="Packager.toml"
if [ -n "${APPLE_SIGNING_IDENTITY:-}" ]; then
    CONFIG="Packager.signed.tmp.toml"
    trap 'rm -f "$APP_DIR/Packager.signed.tmp.toml"' EXIT
    escaped="${APPLE_SIGNING_IDENTITY//\\/\\\\}"
    escaped="${escaped//\"/\\\"}"
    awk -v id="$escaped" '{print} /^\[macos\]/ {print "signing-identity = \"" id "\""}' Packager.toml > "$CONFIG"
    echo "[build-mac] Signing with the identity from APPLE_SIGNING_IDENTITY." >&2
else
    echo "[build-mac] APPLE_SIGNING_IDENTITY not set: building an unsigned app (fine for personal use)." >&2
fi

mkdir -p "$DIST_DIR"
echo "[build-mac] Packaging ($FORMATS, $PKG_TARGET)..." >&2
cargo packager --config "$CONFIG" --formats "$FORMATS" --target "$PKG_TARGET" \
    --binaries-dir "$BIN_DIR" --out-dir "$DIST_DIR"

echo "[build-mac] Done. Output in $DIST_DIR:" >&2
ls -lh "$DIST_DIR" | sed 's/^/  /' >&2
