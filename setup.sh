#!/bin/sh
# Set Mimvox up on macOS or Linux with one command: ./setup.sh [options]
# It installs uv when missing (Astral's installer; it adds uv to new terminals), then runs tools/setup/setup.py with
# uv's Python. That installs the build toolchains, the voice runtime (the engine with OmniVoice, Applio, their models) and writes
# runtime.toml, and ends with the command that starts the app. Safe to run again: it resumes where it stopped.
# Options: --data-dir DIR  --skip-toolchain  --skip-models  --cpu  --yes  --quick-check  (see tools/setup/setup.py)
set -eu
UV_VERSION=0.12.18
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] Installing uv $UV_VERSION..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
    else
        wget -qO- "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
    fi
    # Where the installer put it: the first of its candidates, in its own order, that holds uv.
    UV_BIN=""
    for candidate in "${UV_INSTALL_DIR:-}" "${XDG_BIN_HOME:-}" "${XDG_DATA_HOME:+$XDG_DATA_HOME/../bin}" "$HOME/.local/bin"; do
        if [ -n "$candidate" ] && [ -x "$candidate/uv" ]; then
            UV_BIN="$candidate"
            break
        fi
    done
    if [ -z "$UV_BIN" ]; then
        echo "[setup] uv was installed, but none of the folders its installer uses holds it; open a new terminal and run this again." >&2
        exit 1
    fi
    PATH="$UV_BIN:$PATH"
    MIMVOX_UV="$UV_BIN/uv"  # this terminal does not find it yet: setup.py uses it by its full path
    export PATH MIMVOX_UV
fi
exec uv run --no-project --python 3.12 python tools/setup/setup.py "$@"
