# Building the desktop installers

One script per OS builds the Rust app (`app/`) and packages it with
[cargo-packager](https://github.com/crabnebula-dev/cargo-packager) using `app/Packager.toml`. Output goes to
`dist/` (git-ignored). Each script installs `cargo-packager` itself when it is missing.

| OS | Script | Output |
| --- | --- | --- |
| Linux | `./scripts/build/build-linux.sh` | `.AppImage` + `.deb` |
| macOS | `./scripts/build/build-mac.sh` (`--universal` for arm64 + Intel) | `.app` + `.dmg` |
| Windows | `.\scripts\build\build-win.ps1` (`-Format msi` for an MSI) | NSIS setup `.exe` (default) or `.msi` |

Run them from anywhere; they locate the repo from their own path. Pass `--help` (Linux/macOS) to see the options.
Installers are built on the OS they target (no cross-compiling).

## What goes into an installer

The release binary plus the runtime resources, staged by `scripts/build/stage_resources.py` into
`app/target/bundle-resources/` (no `__pycache__`, venvs or data):

- `engine/src/` the Python engine
- `presets/` guided reading presets
- `tools/setup/` runtime setup (the app runs it on first launch)
- `contract/` the engine protocol
- `setup.sh`, `setup.cmd`, `DISCLAIMER.md`, `THIRD_PARTY_NOTICES.md` when present

The heavy parts (Python virtualenvs, Applio, model weights, several GB) are **not** bundled: first run
downloads them into the per-user data directory. The header of `app/Packager.toml` documents where each installer
puts the resources and the lookup order the app uses to find them.

The icon set in `tools/release/icons/` is committed. To regenerate it:
`uv run --with pillow python tools/release/make_icons.py`.

## Prerequisites

Common to all systems: **Rust** (stable, via [rustup](https://rustup.rs)) and **[uv](https://docs.astral.sh/uv/)**
(runs the staging script). The first packaging run needs internet access (cargo-packager downloads its tools).

### Linux (Debian/Ubuntu names)

```sh
sudo apt install build-essential pkg-config libasound2-dev libxkbcommon-dev libwayland-dev libx11-dev \
    libxcursor-dev libxi-dev libxrandr-dev libvulkan-dev file
```

- `.deb` needs `dpkg-deb` (preinstalled on Debian/Ubuntu). On other distributions build only the AppImage:
  `./scripts/build/build-linux.sh --formats appimage`.
- The AppImage tools (linuxdeploy) are downloaded on first use and run with `APPIMAGE_EXTRACT_AND_RUN=1`, so FUSE
  is not required. The script sets `NO_STRIP=true` because linuxdeploy's bundled `strip` fails on current toolchains.
- To run the AppImage itself, the target machine needs a Vulkan-capable GPU driver (the UI renders through wgpu).

### macOS

- Xcode command line tools: `xcode-select --install`
- Rust via rustup (the script adds `aarch64-apple-darwin`, and `x86_64-apple-darwin` for `--universal`).
- Minimum supported macOS is 13.0. Apple Silicon is the supported target (GPU acceleration uses Metal/MPS).
- Unsigned by default. For a signed (and notarized) build export `APPLE_SIGNING_IDENTITY` plus the notarization
  variables listed in the header of `build-mac.sh`. An unsigned app copied from the `.dmg` needs
  `xattr -dr com.apple.quarantine "/Applications/Mimvox.app"` once.

### Windows

- Rust with the **MSVC** toolchain (`rustup default stable-x86_64-pc-windows-msvc`).
- Visual Studio Build Tools with the "Desktop development with C++" workload:
  `winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`
- `uv`: `powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process -Bypass` once in that window.
- NSIS and the WiX toolset are downloaded by cargo-packager on first use; nothing to install by hand.
- The NSIS installer is per-user (no admin rights); the MSI is the choice for managed deployment.

## Releasing a new version

Bump `version` in `app/Cargo.toml` **and** `app/Packager.toml` (the scripts refuse to build when they differ), then run
the script on each OS.
