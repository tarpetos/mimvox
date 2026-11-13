"""Set Mimvox up fully on Linux, macOS and Windows: ``./setup.sh`` or ``setup.cmd``.

It installs the toolchains the app is built with (Rust, a C/C++ toolchain), then builds the ML runtime under
``<data dir>/runtime/``: the engine environment (OmniVoice, the speech model for every language, plus Whisper and NLLB
for Redub, Hybrid Demucs for removing background music from media files, Seed-VC for the native accent of untrained
voices) and Applio (RVC, at a pinned commit), each in its own uv virtual environment, plus every model they need, and
writes ``<data dir>/runtime.toml`` for the app (see contract/engine-protocol.md).

Every step checks what is already there and skips it, a step interrupted by Ctrl-C or a lost connection resumes
where it stopped (git fetches, uv and Hugging Face downloads are resumable), so running it again is always safe.
Standard library only: it runs before anything else is installed.

Examples:
    ./setup.sh                                   full setup (about 30 GB on disk, 33 GB with a GPU)
    ./setup.sh --skip-toolchain --yes            no Rust / compiler install, no questions
    ./setup.sh --quick-check                     verify an existing setup, download nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

APP_DIR_NAME = "mimvox"
DATA_ENV = "MIMVOX_DATA"
LOCATION_FILE = "location.toml"
UV_ENV = "MIMVOX_UV"
BIN_NAME = "mimvox"

# Pinned upstream revisions: the ones the prototype was validated with. Bump deliberately.
APPLIO_URL = "https://github.com/IAHispano/Applio.git"
APPLIO_COMMIT = "6d5d9511b7e719b1f613362bc767e841ad72fa72"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
# AMD GPUs on Linux: the two environments pin different torch versions, and each has wheels for another ROCm.
ENGINE_ROCM_INDEX = "https://download.pytorch.org/whl/rocm6.4"
APPLIO_ROCM_INDEX = "https://download.pytorch.org/whl/rocm7.2"
AMD_PCI_VENDOR = "0x1002"
# Integrated Radeon GPUs report a small VRAM carve-out; below this the card is not used.
AMD_MIN_VRAM_GB = 4.0
APPLIO_PYTHON = "3.12"

# Files that prove a download finished (relative to the repo they belong to).
APPLIO_REQUIRED = (
    "rvc/models/predictors/rmvpe.pt",
    "rvc/models/embedders/contentvec/pytorch_model.bin",
    "rvc/models/embedders/contentvec/config.json",
    "rvc/models/pretraineds/hifi-gan/f0G40k.pth",
    "rvc/models/pretraineds/hifi-gan/f0D40k.pth",
)
# Windows only: Applio's `prerequisites --exe` puts these in the Applio folder; the engine uses them for every audio
# and video format (Linux and macOS use the system ffmpeg and ffprobe).
APPLIO_WINDOWS_REQUIRED = ("ffmpeg.exe", "ffprobe.exe")
# Redub models, under <data>/runtime/models (the speech worker loads them from there, never from the user's cache).
NLLB_REPO = "facebook/nllb-200-distilled-600M"
NLLB_DIR = "nllb-200-distilled-600M"
NLLB_FILES = ("config.json", "generation_config.json", "pytorch_model.bin", "sentencepiece.bpe.model",
              "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")
NLLB_REQUIRED = ("config.json", "pytorch_model.bin", "sentencepiece.bpe.model", "tokenizer_config.json")
# "Better translation" (optional; a default with a GPU): NLLB-200 distilled 1.3B. The repo only has fp32
# weights (pytorch_model.bin, 5.5 GB), so they are downloaded, stored as fp16 safetensors (2.7 GB) and the .bin deleted.
NLLB_BIG_REPO = "facebook/nllb-200-distilled-1.3B"
NLLB_BIG_DIR = "nllb-200-distilled-1.3B"
NLLB_BIG_REQUIRED = ("config.json", "model.safetensors", "sentencepiece.bpe.model", "tokenizer_config.json")
NLLB_BIG_DOWNLOAD_GB = 5.5
# fp32 checkpoint -> fp16 model.safetensors in another folder (the tied embeddings are stored once). Engine interpreter.
NLLB_FP16_SCRIPT = (
    "import sys, torch, transformers\n"
    "from transformers import AutoModelForSeq2SeqLM\n"
    "key = 'dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'\n"
    "model = AutoModelForSeq2SeqLM.from_pretrained(sys.argv[1], local_files_only=True, low_cpu_mem_usage=True,\n"
    "                                              **{key: torch.float16})\n"
    "model.save_pretrained(sys.argv[2], max_shard_size='20GB')\n"
)
# Zero-shot voice conversion for the native accent of untrained voices (engine zsvc.py): Seed-VC (GPL-3.0), a separately
# downloaded runtime component, never part of this repository. Its source checkout at a pinned commit, the two packages
# its modules import that the engine venv lacks, and the weights of its speech model (v1: DiT + Whisper-small encoder +
# BigVGAN 22 kHz; CAM++ ships in the checkout) at pinned revisions. --no-zero-shot-vc leaves it out.
SEEDVC_URL = "https://github.com/Plachtaa/seed-vc.git"
SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
SEEDVC_DIR = "seed-vc"                 # <runtime>/seed-vc (code) and <runtime>/models/seed-vc (weights)
SEEDVC_PACKAGES = ("munch==4.0.0", "einops==0.8.0")
SEEDVC_MODELS = (   # (subfolder, Hugging Face repo, revision, files)
    ("dit", "Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e",
     ("DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")),
    ("bigvgan", "nvidia/bigvgan_v2_22khz_80band_256x", "633ff708ed5b74903e86ff1298cf4a98e921c513",
     ("config.json", "bigvgan_generator.pt")),
    ("whisper-small", "openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d",
     ("config.json", "preprocessor_config.json", "model.safetensors")),
)
SEEDVC_STAMP = ".mimvox-setup-ok"      # in the weights folder: the engine uses the converter only once it is there
ZERO_SHOT_GB = 2.0   # Seed-VC checkout (~0.1 GB) + weights (1.8 GB: DiT 0.44, BigVGAN 0.45, Whisper-small 0.97)

WHISPER_MODEL = "turbo"
# "Remove background music" (reference from media, Record's multi-file import): torchaudio's HDEMUCS_HIGH_MUSDB_PLUS
# weights, the very file that bundle downloads (download.pytorch.org/torchaudio/<its _model_path>), kept in
# <data>/runtime/models/demucs; the speech worker loads them from there with torchaudio.models.hdemucs_high.
DEMUCS_URL = "https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt"
DEMUCS_DIR = "demucs"
DEMUCS_FILE = "hdemucs_high_trained.pt"
DEMUCS_SIZE = 334_697_255
DEMUCS_SHA256 = "a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816"
# The engine environment, <runtime>/engine/.venv: the engine itself, the OmniVoice worker (k2-fsa/OmniVoice, zero-shot
# TTS in ~60 languages) and the speech worker (Whisper + NLLB for Redub).
ENGINE_DIR = "engine"
ENGINE_PYTHON = "3.11"
ENGINE_TORCH = ("torch==2.8.0", "torchaudio==2.8.0")
# pypdf (BSD-3-Clause, pure Python, no dependencies on 3.11): the text layer of PDF books for text.import (audiobooks);
# every other book format is read with the standard library.
ENGINE_PACKAGES = ("omnivoice==0.2.1", "openai-whisper==20250625", "num2words==0.5.14", "sentencepiece", "g2p_en==2.1.0",
                   "soundfile", "librosa", "scipy", "numpy", "marisa-trie==1.4.1", "pypdf==6.21.4")
# Ukrainian word stress for the OmniVoice text front-end (backends/omnivoice_stress.py, used when MIMVOX_STRESS_AUTO is
# set): lang-uk's dictionary (12.5 MB, inside the wheel, so it works offline) in dictionary-only mode with marisa-trie.
# Installed with --no-deps: its declared Stanza dependency (and the ~500 MB of models Stanza downloads) is only needed
# for the context mode, which is not used.
ENGINE_NO_DEPS = ("ukrainian-word-stress==2.1.0",)
OMNIVOICE_REPO = "k2-fsa/OmniVoice"
OMNIVOICE_DIR = "omnivoice"   # <runtime>/models/omnivoice
OMNIVOICE_REQUIRED = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                      "audio_tokenizer/config.json", "audio_tokenizer/model.safetensors",
                      "audio_tokenizer/preprocessor_config.json")

MIN_RUST = (1, 97)
RUNTIME_GB = 30
ENGINE_GB = 11    # engine venv (torch 2.8 CUDA, transformers 5, OmniVoice, Whisper; ~7.6 GB) + the OmniVoice model (3.3 GB)
APPLIO_GB = 12
SPEECH_GB = 4.5   # Whisper turbo (1.6 GB) + the NLLB-200 translation model (2.5 GB) for Redub
BETTER_TRANSLATION_GB = 2.7   # NLLB-200 1.3B in fp16 (a 5.5 GB download), with a GPU or --better-translation yes
SEPARATION_GB = 0.3   # Hybrid Demucs (335 MB) for "Remove background music"
LEGACY_INDEXTTS_DIR = "index-tts"   # the IndexTTS-2.5 runtime of earlier versions (about 19 GB), no longer used
DISK_NEEDED_GB = 34
UV_HTTP_TIMEOUT = "600"

# Linux packages: a C/C++ toolchain and the libraries Rune (wgpu, winit, cpal, fontconfig) builds and runs against,
# plus ffmpeg and libsndfile for the Python side. Keys are the package managers' names.
LINUX_PACKAGES = {
    "apt-get": [
        "build-essential", "pkg-config", "git", "curl", "ca-certificates", "ffmpeg", "libsndfile1",
        "libasound2-dev", "libxkbcommon-dev", "libxkbcommon-x11-0", "libwayland-dev", "libx11-dev",
        "libfontconfig1-dev", "libvulkan1", "mesa-vulkan-drivers", "libportaudio2",
    ],
    "dnf": [
        "gcc", "gcc-c++", "make", "pkgconf-pkg-config", "git", "curl", "ffmpeg-free", "libsndfile",
        "alsa-lib-devel", "libxkbcommon-devel", "libxkbcommon-x11", "wayland-devel", "libX11-devel",
        "fontconfig-devel", "vulkan-loader", "mesa-vulkan-drivers", "portaudio",
    ],
    "pacman": [
        "base-devel", "pkgconf", "git", "curl", "ffmpeg", "libsndfile", "alsa-lib", "libxkbcommon",
        "libxkbcommon-x11", "wayland", "libx11", "fontconfig", "vulkan-icd-loader", "vulkan-mesa-layers",
        "portaudio",
    ],
    "zypper": [
        "gcc", "gcc-c++", "make", "pkg-config", "git", "curl", "ffmpeg", "libsndfile1", "alsa-devel",
        "libxkbcommon-devel", "libxkbcommon-x11-0", "wayland-devel", "libX11-devel", "fontconfig-devel",
        "libvulkan1", "portaudio",
    ],
}
LINUX_INSTALL = {
    "apt-get": ["apt-get", "install", "-y"],
    "dnf": ["dnf", "install", "-y"],
    "pacman": ["pacman", "-S", "--needed", "--noconfirm"],
    "zypper": ["zypper", "--non-interactive", "install"],
}
WINGET = ["winget", "install", "--exact", "--accept-source-agreements", "--accept-package-agreements", "--id"]
WINGET_BUILD_TOOLS = "Microsoft.VisualStudio.2022.BuildTools"
BUILD_TOOLS_OVERRIDE = "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
VC_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
XCODE_WAIT_SECONDS = 30 * 60
XCODE_POLL_SECONDS = 10


class SetupError(RuntimeError):
    """A step could not complete; the message is the one line the user sees."""


def say(line: str) -> None:
    print(f"[setup] {line}", flush=True)


def warn(line: str) -> None:
    print(f"[setup] WARNING: {line}", flush=True)


def host_platform() -> str:
    """``linux``, ``macos`` or ``windows``."""
    if sys.platform.startswith("win"):
        return "windows"
    return "macos" if sys.platform == "darwin" else "linux"


def host_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        return "x86_64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine


def platform_data_dir(plat: str) -> Path:
    """Where the data directory goes by default. The runtime is about 28 GB, so on Windows this is the *local*
    app data (never the roaming profile, which can be synced or copied)."""
    home = Path.home()
    if plat == "windows":
        return Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    if plat == "macos":
        return home / "Library" / "Application Support"
    return Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")


def default_data_dir(plat: str) -> Path:
    return platform_data_dir(plat) / APP_DIR_NAME


def config_dir(plat: str) -> Path:
    """Fixed per-user folder of the pointer file ``location.toml`` (the app looks here for a custom data dir)."""
    home = Path.home()
    if plat == "windows":
        return Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local") / APP_DIR_NAME
    if plat == "macos":
        return home / "Library" / "Application Support" / APP_DIR_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config") / APP_DIR_NAME


def location_file(plat: str) -> Path:
    return config_dir(plat) / LOCATION_FILE


def read_location(plat: str) -> Path | None:
    """The data dir named by ``location.toml``, if the file exists and is valid."""
    try:
        import tomllib

        value = tomllib.loads(location_file(plat).read_text(encoding="utf-8")).get("data_dir")
    except (OSError, ValueError, ImportError):
        return None
    return Path(value) if isinstance(value, str) and value.strip() else None


def has_runtime(data_dir: Path) -> bool:
    """True when ``data_dir`` already holds an installed runtime (setup then reuses it and resumes)."""
    return (data_dir / "runtime.toml").is_file() or (data_dir / "runtime" / ENGINE_DIR).is_dir() \
        or (data_dir / "runtime" / LEGACY_INDEXTTS_DIR).is_dir()


def volumes(plat: str) -> list[tuple[str, float]]:
    """``(mount point or drive, free GB)`` of the volumes worth offering for the install."""
    roots: list[str] = []
    if plat == "windows":
        roots = [f"{c}:\\" for c in "CDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{c}:\\")]
    else:
        roots = ["/", str(Path.home())]
        for base in ("/mnt", "/media", "/run/media", "/Volumes"):
            try:
                for entry in sorted(Path(base).iterdir()):
                    if entry.is_dir():
                        roots.append(str(entry))
            except OSError:
                continue
    seen: set[tuple[int, int]] = set()
    out: list[tuple[str, float]] = []
    for root in roots:
        try:
            usage = shutil.disk_usage(root)
            dev = os.stat(root).st_dev
        except OSError:
            continue
        key = (dev, usage.total)
        if key in seen:
            continue
        seen.add(key)
        out.append((root, usage.free / 1e9))
    return out


def venv_python(venv: Path, plat: str) -> Path:
    return venv / ("Scripts/python.exe" if plat == "windows" else "bin/python")


def toml_str(value: object) -> str:
    """A TOML basic string (JSON string escapes are valid TOML)."""
    return json.dumps(str(value), ensure_ascii=False)


@dataclass
class Gpu:
    backend: str  # cuda | mps | cpu; an AMD GPU is "cuda" too, because ROCm torch uses the torch.cuda API
    name: str = ""
    vram_gb: float = 0.0
    vendor: str = ""  # nvidia | amd, for a cuda backend

    @property
    def build(self) -> str:
        """The torch build: cuda | rocm | mps | cpu."""
        return "rocm" if self.vendor == "amd" else self.backend


def amd_gpu() -> Gpu | None:
    """The AMD GPU with the most VRAM, from sysfs (no ROCm install needed: the torch wheels bring their own)."""
    best: tuple[float, Path] | None = None
    for dev in Path("/sys/class/drm").glob("card[0-9]*/device"):
        try:
            if (dev / "vendor").read_text().strip() != AMD_PCI_VENDOR:
                continue
            vram = int((dev / "mem_info_vram_total").read_text()) / 2**30
        except (OSError, ValueError):
            continue
        if vram >= AMD_MIN_VRAM_GB and (best is None or vram > best[0]):
            best = (vram, dev)
    if best is None:
        return None
    return Gpu("cuda", amd_gpu_name(best[1]), round(best[0], 1), "amd")


def amd_gpu_name(dev: Path) -> str:
    """The marketing name from rocminfo when ROCm is installed, else the bracket part of the lspci name."""
    rocminfo = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    try:
        out = subprocess.run([rocminfo], capture_output=True, text=True, timeout=10, check=False).stdout
        names = re.findall(r"Marketing Name:\s*(.*(?:Radeon|Instinct).*)", out)
        if names:
            return names[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        slot = dev.resolve().name
        out = subprocess.run(["lspci", "-mm", "-s", slot], capture_output=True, text=True, timeout=10,
                             check=False).stdout
        fields = re.findall(r'"([^"]*)"', out)
        match = re.search(r"\[(.+)\]", fields[2]) if len(fields) > 2 else None
        if match:
            return f"AMD {match.group(1)}"
    except (OSError, subprocess.SubprocessError):
        pass
    return "AMD GPU"


@dataclass
class Setup:
    """The setup steps. ``root`` is the checkout, ``data_dir`` the user's data directory."""

    root: Path
    plat: str
    arch: str
    data_dir: Path
    skip_toolchain: bool = False
    skip_models: bool = False
    force_cpu: bool = False
    assume_yes: bool = False
    force: bool = False
    gpu: Gpu = field(default_factory=lambda: Gpu("cpu"))
    uv_path: str = ""
    better_translation: str = "auto"   # auto (with a GPU) | yes | no
    zero_shot_vc: bool = True          # Seed-VC for the native accent of untrained voices (--no-zero-shot-vc)

    # ---- paths -----------------------------------------------------------------------------------------------

    @property
    def runtime(self) -> Path:
        return self.data_dir / "runtime"

    @property
    def legacy_indextts(self) -> Path:
        return self.runtime / LEGACY_INDEXTTS_DIR

    @property
    def applio(self) -> Path:
        return self.runtime / "applio"

    @property
    def runtime_toml(self) -> Path:
        return self.data_dir / "runtime.toml"

    @property
    def engine_venv(self) -> Path:
        return self.runtime / ENGINE_DIR / ".venv"

    @property
    def omnivoice_model(self) -> Path:
        return self.runtime / "models" / OMNIVOICE_DIR

    @property
    def seed_vc(self) -> Path:
        return self.runtime / SEEDVC_DIR

    @property
    def seed_vc_models(self) -> Path:
        return self.runtime / "models" / SEEDVC_DIR

    def engine_python(self) -> Path:
        return venv_python(self.engine_venv, self.plat)

    def rvc_python(self) -> Path:
        return venv_python(self.applio / ".venv", self.plat)

    # ---- process helpers -------------------------------------------------------------------------------------

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        for name in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME", "PYTHONPATH"):
            env.pop(name, None)
        env.setdefault("UV_HTTP_TIMEOUT", UV_HTTP_TIMEOUT)
        env.setdefault("UV_CONCURRENT_DOWNLOADS", "4")  # moderate: big wheels and source builds can exhaust RAM
        env.setdefault("UV_CONCURRENT_BUILDS", "2")
        env.setdefault("UV_CONCURRENT_INSTALLS", "4")
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        env.setdefault("GCM_INTERACTIVE", "never")
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        if extra:
            env.update(extra)
        return env

    def run(self, command: Sequence[object], cwd: Path | None = None, extra_env: Mapping[str, str] | None = None) -> int:
        """Run a command with its output on the console. Returns the exit code (127: program not found)."""
        argv = [str(part) for part in command]
        try:
            return subprocess.run(argv, cwd=cwd or self.root, env=self.env(extra_env), check=False).returncode
        except OSError:
            return 127

    def quiet(self, command: Sequence[object], cwd: Path | None = None) -> int:
        """Run a command with its output discarded (probing); returns the exit code."""
        try:
            return subprocess.run([str(p) for p in command], cwd=cwd or self.root, env=self.env(), check=False,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        except OSError:
            return 127

    def must(self, command: Sequence[object], what: str, cwd: Path | None = None,
             extra_env: Mapping[str, str] | None = None) -> None:
        code = self.run(command, cwd, extra_env)
        if code != 0:
            raise SetupError(f"{what} failed (exit {code}); the output above says why.")

    def read(self, command: Sequence[object], cwd: Path | None = None) -> str:
        """A command's stdout, or '' when it failed or is missing."""
        try:
            done = subprocess.run([str(p) for p in command], cwd=cwd or self.root, env=self.env(), check=False,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return done.stdout if done.returncode == 0 else ""

    def git(self, *args: object) -> list[str]:
        """A git command line with the quirks pinned: LF checkouts, long paths, schannel TLS on Windows."""
        config = ["-c", "core.autocrlf=false", "-c", "core.longpaths=true", "-c", "advice.detachedHead=false"]
        if self.plat == "windows":
            config += ["-c", "http.sslBackend=schannel"]  # MinGit ships without a CA bundle the default backend finds
        return ["git", *config, *[str(a) for a in args]]

    def uv(self, *args: object) -> list[str]:
        return [self.uv_path, *[str(a) for a in args]]

    # ---- step 1: detection -----------------------------------------------------------------------------------

    def detect(self) -> None:
        say(f"Platform: {self.plat} {self.arch} ({platform.platform()})")
        self.gpu = self._detect_gpu()
        if self.gpu.build == "rocm":
            say(f"GPU backend: rocm ({self.gpu.name}, {self.gpu.vram_gb:.0f} GB; AMD support is new, see the README)")
            if not os.access("/dev/kfd", os.R_OK | os.W_OK):
                warn("/dev/kfd is not accessible, so PyTorch cannot use the AMD GPU. Add your user to the 'render' "
                     "and 'video' groups (sudo usermod -aG render,video $USER), log in again and run setup again.")
        elif self.gpu.backend == "cuda":
            say(f"GPU backend: cuda ({self.gpu.name}, {self.gpu.vram_gb:.0f} GB)")
        elif self.gpu.backend == "mps":
            say("GPU backend: mps (Apple Silicon; best effort, parts of the engine may fall back to the CPU)")
        else:
            if self.plat == "macos":
                warn("Intel Macs are not supported: PyTorch 2.8 (OmniVoice) has no wheels for them; expect the install to fail.")
            warn("No supported GPU found: the runtime is set up for the CPU. Training a voice will be very slow "
                 "and speech generation several times slower than real time.")

    @staticmethod
    def _existing_parent(path: Path) -> Path:
        while not path.exists() and path != path.parent:
            path = path.parent
        return path

    def _detect_gpu(self) -> Gpu:
        if self.force_cpu:
            say("--cpu given: using the CPU backend.")
            return Gpu("cpu")
        if self.plat == "macos":
            return Gpu("mps") if self.arch == "arm64" else Gpu("cpu")
        smi = shutil.which("nvidia-smi")
        if smi:
            out = self.read([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).strip()
            if out:
                first = out.splitlines()[0].split(",")
                try:
                    vram = float(first[1]) / 1024
                except (IndexError, ValueError):
                    vram = 0.0
                return Gpu("cuda", first[0].strip(), vram, "nvidia")
        if self.plat == "linux":
            return amd_gpu() or Gpu("cpu")
        return Gpu("cpu")

    def free_gb(self) -> float:
        return shutil.disk_usage(self._existing_parent(self.data_dir)).free / 1e9

    def announce(self) -> bool:
        """Print where everything goes and how big it is, before anything is downloaded. Returns True when there is
        a big download ahead (False: a runtime already exists here)."""
        existing = has_runtime(self.data_dir)
        free = self.free_gb()
        say("=" * 78)
        say(f"Install location : {self.data_dir}")
        say(f"Free space there : {free:.0f} GB")
        if existing:
            say("Existing runtime found, nothing big to download (setup only fills in what is missing).")
            if not self._has(self.models_dir / NLLB_DIR, NLLB_REQUIRED) and not self.skip_models:
                say(f"Missing Redub models will be downloaded now (about {SPEECH_GB} GB: translation and Whisper turbo).")
            if self.wants_better_translation() and not self.has_better_translation() and not self.skip_models:
                say(f"The better translation model (NLLB-200 1.3B) will be downloaded now: about "
                    f"{NLLB_BIG_DOWNLOAD_GB} GB, kept as {BETTER_TRANSLATION_GB} GB.")
            if not (self.models_dir / DEMUCS_DIR / DEMUCS_FILE).is_file() and not self.skip_models:
                say(f"The music-removal model will be downloaded now (about {SEPARATION_GB} GB: Hybrid Demucs).")
            if self.zero_shot_vc and not self.has_zero_shot_vc() and not self.skip_models:
                say(f"The zero-shot voice converter (Seed-VC, GPL-3.0, about {ZERO_SHOT_GB:.0f} GB) will be installed "
                    "now: the native accent for voices that are not trained (--no-zero-shot-vc leaves it out).")
            if not self._has(self.omnivoice_model, OMNIVOICE_REQUIRED) and not self.skip_models:
                say(f"The OmniVoice speech model (every language, Ukrainian included) will be installed now: about "
                    f"{ENGINE_GB} GB (the engine environment in <data>/runtime/engine, the model in "
                    "<data>/runtime/models/omnivoice).")
            if self.legacy_indextts.is_dir():
                say(f"{self.legacy_indextts} (IndexTTS-2.5, about 19 GB) is no longer used; setup offers to delete it "
                    "at the end.")
        else:
            better = self.wants_better_translation()
            total = (RUNTIME_GB + (round(BETTER_TRANSLATION_GB) if better else 0)
                     - (0 if self.zero_shot_vc else round(ZERO_SHOT_GB)))
            say(f"This will download and install about {total} GB (engine + OmniVoice ~{ENGINE_GB} GB, "
                f"Applio/RVC ~{APPLIO_GB} GB, Redub models ~{SPEECH_GB} GB, music removal ~{SEPARATION_GB} GB"
                + (f", zero-shot native accent ~{ZERO_SHOT_GB:.0f} GB" if self.zero_shot_vc else "")
                + (f", better translation ~{BETTER_TRANSLATION_GB} GB" if better else "") + ") plus ~0.1 GB per voice.")
            say("OmniVoice speaks every language (Ukrainian, English and about 60 more): the engine environment in "
                "<data>/runtime/engine and the model (3.3 GB) in <data>/runtime/models/omnivoice.")
            say(f"Redub (dubbing a file) uses two models, about {SPEECH_GB} GB: Whisper turbo (transcription, 1.6 GB, "
                "<data>/runtime/models/whisper) and NLLB-200 (translation, 2.5 GB, "
                "<data>/runtime/models/nllb-200-distilled-600M).")
            if better:
                say(f"With a GPU Redub also gets the better translation model, NLLB-200 1.3B (a "
                    f"{NLLB_BIG_DOWNLOAD_GB} GB download kept as {BETTER_TRANSLATION_GB} GB in fp16, "
                    "<data>/runtime/models/nllb-200-distilled-1.3B; --better-translation no leaves it out).")
            say(f"Removing background music from media files uses Hybrid Demucs (about {SEPARATION_GB} GB, "
                "<data>/runtime/models/demucs).")
            if self.zero_shot_vc:
                say(f"The native accent of voices that are not trained uses Seed-VC, a zero-shot voice converter "
                    f"(about {ZERO_SHOT_GB:.0f} GB: its source in <data>/runtime/{SEEDVC_DIR}, GPL-3.0, downloaded "
                    f"from its own repository; weights in <data>/runtime/models/{SEEDVC_DIR}; --no-zero-shot-vc "
                    "leaves it out).")
            if self.skip_models:
                say("--skip-models: the model files (most of that size) are left out for now.")
        say("=" * 78)
        needed = DISK_NEEDED_GB + (round(NLLB_BIG_DOWNLOAD_GB) if self.wants_better_translation() else 0)
        if not existing and free < needed:
            if self.force:
                warn(f"Only {free:.0f} GB free, less than the {needed} GB needed; continuing because of --force.")
            else:
                raise SetupError(f"Only {free:.0f} GB free in {self.data_dir}; at least {needed} GB are needed. "
                                 f"Free up space, choose another place with --data-dir, or pass --force.")
        return not existing

    def confirm(self) -> None:
        """State the install location and size before the first download. Interactive: confirm or type another path.
        With ``--yes`` (or without a terminal) it only prints and checks the free space."""
        interactive = not self.assume_yes and sys.stdin.isatty()
        while True:
            try:
                self.announce()
            except SetupError as error:
                if not interactive:
                    raise SystemExit(f"[setup] {error}")
                say(f"{error}")
                answer = "n"
            else:
                if not interactive:
                    return
                try:
                    answer = input("[setup] Continue? [Y/n, or type another folder] ").strip().strip('"')
                except EOFError:
                    raise SystemExit("[setup] Cancelled.") from None
                if answer.lower() in ("", "y", "yes"):
                    return
            if answer.lower() in ("n", "no"):
                try:
                    answer = input("[setup] Install into which folder (empty to cancel)? ").strip().strip('"')
                except EOFError:
                    answer = ""
                if not answer:
                    raise SystemExit("[setup] Cancelled.")
            path = Path(answer).expanduser()
            if not path.is_absolute():
                say("Please give a full path (for example D:\\Voice or /data/voice).")
                continue
            self.data_dir = path.resolve()

    # ---- step 2: toolchains ----------------------------------------------------------------------------------

    def toolchains(self) -> None:
        """Rust and the C/C++ toolchain the app is built with, plus git."""
        if self.skip_toolchain:
            say("Skipping the toolchain install (--skip-toolchain).")
            if not shutil.which("git"):
                raise SetupError("git is not installed and is needed to fetch Applio; install it.")
            return
        failed: list[str] = []
        for step in (self._git, self._compiler, self._rust, self._ffmpeg):
            try:
                step()
            except SetupError as error:
                say(f"Failed: {error}")
                failed.append(str(error))
        if failed and not shutil.which("git"):
            raise SetupError("git is missing: " + "; ".join(failed))
        if failed:
            raise SetupError("; ".join(failed))

    def _git(self) -> None:
        if shutil.which("git"):
            return
        if self.plat == "windows":
            say("Installing Git for Windows with winget...")
            self.run([*WINGET, "Git.Git"])
            for candidate in (r"C:\Program Files\Git\cmd", r"C:\Program Files (x86)\Git\cmd"):
                if Path(candidate, "git.exe").exists():
                    os.environ["PATH"] = candidate + os.pathsep + os.environ["PATH"]
        elif self.plat == "linux":
            self._linux_install()  # the package set holds git
        if not shutil.which("git"):
            raise SetupError("git could not be installed; install it and run this again.")

    def _compiler(self) -> None:
        if self.plat == "windows":
            self._windows_build_tools()
        elif self.plat == "macos":
            self._xcode_tools()
        else:
            self._linux_install()

    def _linux_manager(self) -> str:
        found = next((name for name in LINUX_PACKAGES if shutil.which(name)), None)
        if found is None:
            raise SetupError("No known package manager (apt-get, dnf, pacman, zypper): install a C/C++ compiler, "
                             "pkg-config and the ALSA, xkbcommon, wayland, X11, fontconfig and Vulkan "
                             "development libraries by hand.")
        return found

    def _dpkg_installed(self, package: str) -> bool:
        return "install ok installed" in self.read(["dpkg-query", "-W", "-f=${Status}", package])

    def _linux_missing(self, manager: str) -> list[str]:
        packages = LINUX_PACKAGES[manager]
        if manager == "apt-get":
            return [p for p in packages if not self._dpkg_installed(p)]
        query = ["rpm", "-q"] if manager in ("dnf", "zypper") else ["pacman", "-Qq"]
        return [p for p in packages if not self.read([*query, p]).strip()]

    def _linux_install(self) -> None:
        manager = self._linux_manager()
        missing = self._linux_missing(manager)
        if not missing:
            say("System packages: all present.")
            return
        say(f"Missing system packages: {' '.join(missing)}")
        sudo = [] if hasattr(os, "geteuid") and os.geteuid() == 0 else ["sudo"]
        if sudo:
            say("Installing them needs administrator rights: sudo may ask for your password.")
        if manager == "apt-get":
            self.run([*sudo, "apt-get", "update"])
        if self.run([*sudo, *LINUX_INSTALL[manager], *missing]) != 0:
            raise SetupError(f"Installing the system packages failed; install them by hand: {' '.join(missing)}")

    def _windows_build_tools(self) -> None:
        vswhere = Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / (
            "Microsoft Visual Studio/Installer/vswhere.exe")
        query = [vswhere, "-latest", "-products", "*", "-requires", VC_COMPONENT, "-property", "installationPath"]
        if self.read(query).strip():
            say("Visual Studio Build Tools (C++): present.")
            return
        if not shutil.which("winget"):
            raise SetupError("winget is missing: install the Visual Studio Build Tools with the "
                             "'Desktop development with C++' workload by hand "
                             "(https://visualstudio.microsoft.com/visual-cpp-build-tools/).")
        say("Installing the Visual Studio Build Tools with the C++ workload (about 3 GB; this takes a while)...")
        self.run([*WINGET, WINGET_BUILD_TOOLS, "--override", BUILD_TOOLS_OVERRIDE])
        if not self.read(query).strip():  # winget's exit code is unreliable (reboot pending, already installed)
            raise SetupError("The Visual Studio Build Tools are still missing; install them with the C++ workload.")

    def _xcode_tools(self) -> None:
        if self.quiet(["xcode-select", "-p"]) == 0:
            say("Xcode Command Line Tools: present.")
            return
        say("Installing Xcode's Command Line Tools: confirm the dialog that opens; this waits for it...")
        self.run(["xcode-select", "--install"])
        for _ in range(XCODE_WAIT_SECONDS // XCODE_POLL_SECONDS):
            time.sleep(XCODE_POLL_SECONDS)
            if self.quiet(["xcode-select", "-p"]) == 0:
                return
        raise SetupError("Xcode's Command Line Tools are still missing; finish their installer and run this again.")

    def _ffmpeg(self) -> None:
        """ffmpeg + ffprobe decode and encode every audio/video format (Redub, imports, exports). Linux gets them with
        the system packages; macOS through Homebrew when it is installed; Windows uses Applio's copies (see
        ``applio_models``)."""
        if self.plat == "windows":
            return
        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            say("ffmpeg: present.")
            return
        if self.plat == "macos" and shutil.which("brew"):
            say("Installing ffmpeg with Homebrew...")
            self.run(["brew", "install", "ffmpeg"])
        elif self.plat == "linux":
            self._linux_install()
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            how = "brew install ffmpeg" if self.plat == "macos" else "your package manager (package: ffmpeg)"
            raise SetupError(f"ffmpeg and ffprobe are needed for audio and video files; install them with {how}.")
        say("ffmpeg: installed.")

    def _cargo(self) -> str | None:
        found = shutil.which("cargo")
        if found:
            return found
        candidate = Path.home() / ".cargo" / "bin" / ("cargo.exe" if self.plat == "windows" else "cargo")
        if candidate.exists():
            os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ["PATH"]
            return str(candidate)
        return None

    def _rust_version(self) -> tuple[int, int] | None:
        match = re.search(r"cargo (\d+)\.(\d+)", self.read(["cargo", "--version"]))
        return (int(match.group(1)), int(match.group(2))) if match else None

    def _rust(self) -> None:
        if not (self.root / "app" / "Cargo.toml").is_file():
            return  # an installed app (no sources to build): Rust is not needed
        if self._cargo():
            version = self._rust_version()
            if version and version >= MIN_RUST:
                say(f"Rust: cargo {version[0]}.{version[1]} present.")
                return
            if shutil.which("rustup"):
                say("Updating Rust (rustup)...")
                self.must(["rustup", "update", "stable"], "Updating Rust")
                self.run(["rustup", "default", "stable"])
                return
            raise SetupError(f"Rust is older than {MIN_RUST[0]}.{MIN_RUST[1]} and rustup is missing; update it.")
        say("Installing Rust with rustup...")
        with tempfile.TemporaryDirectory() as tmp:
            if self.plat == "windows":
                target = "aarch64" if self.arch == "arm64" else "x86_64"
                installer = Path(tmp) / "rustup-init.exe"
                self._download(f"https://win.rustup.rs/{target}", installer)
                command: list[object] = [installer, "-y", "--profile", "minimal", "--default-toolchain", "stable"]
            else:
                installer = Path(tmp) / "rustup-init.sh"
                self._download("https://sh.rustup.rs", installer)
                command = ["sh", installer, "-y", "--profile", "minimal", "--default-toolchain", "stable"]
            self.must(command, "Installing Rust")
        if not self._cargo():
            raise SetupError("Rust was installed but cargo was not found; open a new terminal and run this again.")
        say("Rust installed (new terminals find cargo by themselves).")

    @staticmethod
    def _download(url: str, target: Path) -> None:
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(target, "wb") as handle:  # noqa: S310
                shutil.copyfileobj(response, handle)
        except OSError as error:
            raise SetupError(f"Downloading {url} failed: {error}") from error

    # ---- step 3: runtime -------------------------------------------------------------------------------------

    def ensure_repo(self, name: str, url: str, commit: str, dest: Path) -> None:
        """Make ``dest`` a checkout of ``commit`` (shallow, resumable); re-running with the right commit is a no-op."""
        head = self.read(self.git("rev-parse", "HEAD"), dest).strip() if (dest / ".git").exists() else ""
        if head == commit:
            say(f"{name}: at the pinned commit {commit[:9]}.")
            return
        if dest.exists() and not (dest / ".git").exists() and any(dest.iterdir()):
            raise SetupError(f"{dest} exists and is not a git checkout; move it away.")
        say(f"{name}: fetching {url} @ {commit[:9]}...")
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / ".git").exists():
            self.must(self.git("init", "--quiet", dest), f"git init for {name}")
            self.must(self.git("remote", "add", "origin", url), f"git remote for {name}", cwd=dest)
        if self.run(self.git("fetch", "--quiet", "--depth", "1", "origin", commit), dest) != 0:
            say(f"{name}: shallow fetch of the commit failed, fetching the full history...")
            self.must(self.git("fetch", "--quiet", "origin"), f"Fetching {name}", cwd=dest)
        self.must(self.git("checkout", "--quiet", "--force", commit), f"Checking out {name}", cwd=dest)
        say(f"{name}: checked out {commit[:9]}.")

    def _stamp(self, venv_dir: Path) -> Path:
        return venv_dir / ".mimvox-setup-ok"

    def _stamp_value(self, commit: str) -> str:
        return f"{commit} {self.gpu.build} {self.plat}"

    def _build_note(self) -> str:
        return {"cuda": "CUDA 12.8 build; a few GB", "rocm": "ROCm build; a few GB"}.get(self.gpu.build, "")

    def _torch_args(self, rocm_index: str) -> list[str]:
        """uv arguments that pick the torch build. unsafe-best-match: the PyTorch index also lists other packages
        (requests, numpy...), older than the pins. The reinstall replaces a torch of another build: a pin such as
        torch==2.8.0 also matches an installed 2.8.0+cpu, so uv would keep it."""
        args = ["--reinstall-package", "torch", "--reinstall-package", "torchaudio"]
        if self.gpu.build in ("cuda", "rocm"):
            index = rocm_index if self.gpu.build == "rocm" else TORCH_CUDA_INDEX
            return [*args, "--extra-index-url", index, "--index-strategy", "unsafe-best-match"]
        if self.gpu.backend == "cpu" and self.plat != "macos":
            return [*args, "--torch-backend", "cpu"]
        return args

    def _venv_ready(self, repo: Path, commit: str) -> bool:
        stamp = self._stamp(repo / ".venv")
        py = venv_python(repo / ".venv", self.plat)
        return py.exists() and stamp.exists() and stamp.read_text(encoding="utf-8").strip() == self._stamp_value(commit)

    def applio_venv(self) -> None:
        """Applio's environment: Python 3.12 and its requirements (a different torch from the engine's)."""
        if self._venv_ready(self.applio, APPLIO_COMMIT):
            say("Applio environment: ready.")
            return
        venv = self.applio / ".venv"
        py = venv_python(venv, self.plat)
        if not py.exists():
            say(f"Applio environment: creating the Python {APPLIO_PYTHON} venv...")
            self.must(self.uv("venv", "--python", APPLIO_PYTHON, "--allow-existing", venv), "Creating the Applio venv",
                      cwd=self.applio)
        note = self._build_note()
        say(f"Applio environment: installing requirements (torch 2.11{' ' + note if note else ''})...")
        command = self.uv("pip", "install", "--python", py, "-r", "requirements.txt")
        command += self._torch_args(APPLIO_ROCM_INDEX)
        self.must(command, "Installing the Applio requirements", cwd=self.applio)
        self._stamp(venv).write_text(self._stamp_value(APPLIO_COMMIT), encoding="utf-8")

    def _has(self, repo: Path, files: Sequence[str]) -> bool:
        return all((repo / f).is_file() and (repo / f).stat().st_size > 0 for f in files)

    @property
    def models_dir(self) -> Path:
        return self.runtime / "models"

    def speech_models(self) -> None:
        """Redub: the NLLB-200 translation model (``hf download``) and the Whisper turbo checkpoint."""
        py = self.engine_python()
        if not py.exists():
            raise SetupError("The engine environment is missing, so the Redub models cannot be fetched.")
        nllb = self.models_dir / NLLB_DIR
        if self._has(nllb, NLLB_REQUIRED):
            say("Translation model (NLLB-200 distilled 600M): present.")
        else:
            say(f"Translation model: downloading {NLLB_REPO} (about 2.5 GB, resumable)...")
            nllb.mkdir(parents=True, exist_ok=True)
            # `hf download` through the interpreter (the venv's hf.exe stub breaks when the runtime was moved);
            # an older huggingface_hub without that CLI falls back to snapshot_download.
            code = self.run([py, "-I", "-m", "huggingface_hub.cli.hf", "download", NLLB_REPO, *NLLB_FILES,
                             "--local-dir", nllb], cwd=self.runtime)
            if code != 0:
                script = ("import sys; from huggingface_hub import snapshot_download; "
                          "snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=sys.argv[3:])")
                self.must([py, "-I", "-c", script, NLLB_REPO, nllb, *NLLB_FILES], "Downloading the translation model",
                          cwd=self.runtime)
            if not self._has(nllb, NLLB_REQUIRED):
                raise SetupError("The translation model download finished but files are missing; run this again.")
        whisper_dir = self.models_dir / "whisper"
        if any(whisper_dir.glob("large-v3-turbo.pt")):
            say("Whisper turbo checkpoint: present.")
            return
        say("Whisper turbo checkpoint: downloading (about 1.6 GB, resumable)...")
        whisper_dir.mkdir(parents=True, exist_ok=True)
        script = "import sys, whisper; whisper._download(whisper._MODELS[sys.argv[1]], sys.argv[2], False)"
        self.must([py, "-c", script, WHISPER_MODEL, whisper_dir], "Downloading the Whisper turbo checkpoint",
                  cwd=self.runtime)

    def wants_better_translation(self) -> bool:
        """The optional NLLB-200 1.3B: by default only with a CUDA or ROCm GPU (it runs on the GPU; without one Redub uses
        the 600M, which is about 3x faster on a CPU)."""
        if self.better_translation in ("yes", "no"):
            return self.better_translation == "yes"
        return self.gpu.backend == "cuda"

    def has_better_translation(self) -> bool:
        return self._has(self.models_dir / NLLB_BIG_DIR, NLLB_BIG_REQUIRED)

    def better_translation_model(self) -> None:
        """Redub's "better translation": NLLB-200 distilled 1.3B. ``hf download`` of the fp32 checkpoint (resumable,
        like the 600M), then stored as fp16 ``model.safetensors`` (half the disk) and the fp32 file deleted."""
        target = self.models_dir / NLLB_BIG_DIR
        fp32 = target / "pytorch_model.bin"
        if self.has_better_translation():
            if fp32.is_file():  # a conversion that was interrupted after the new file was in place
                fp32.unlink()
            say("Better translation model (NLLB-200 distilled 1.3B): present.")
            return
        py = self.engine_python()
        if not py.exists():
            raise SetupError("The engine environment is missing, so the better translation model cannot be fetched.")
        say(f"Better translation model: downloading {NLLB_BIG_REPO} (about {NLLB_BIG_DOWNLOAD_GB} GB, resumable; "
            f"kept as {BETTER_TRANSLATION_GB} GB)...")
        target.mkdir(parents=True, exist_ok=True)
        code = self.run([py, "-I", "-m", "huggingface_hub.cli.hf", "download", NLLB_BIG_REPO, *NLLB_FILES,
                         "--local-dir", target], cwd=self.runtime)
        if code != 0:
            script = ("import sys; from huggingface_hub import snapshot_download; "
                      "snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=sys.argv[3:])")
            self.must([py, "-I", "-c", script, NLLB_BIG_REPO, target, *NLLB_FILES],
                      "Downloading the better translation model", cwd=self.runtime)
        if not self._has(target, ("config.json", "pytorch_model.bin", "sentencepiece.bpe.model", "tokenizer_config.json")):
            raise SetupError("The better translation model download finished but files are missing; run this again.")
        say("Better translation model: storing it in fp16 (about a minute)...")
        tmp = target / ".fp16"
        shutil.rmtree(tmp, ignore_errors=True)
        self.must([py, "-I", "-c", NLLB_FP16_SCRIPT, target, tmp], "Converting the better translation model",
                  cwd=self.runtime, extra_env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        if not (tmp / "model.safetensors").is_file():
            raise SetupError(f"Converting the better translation model wrote no model.safetensors in {tmp}.")
        os.replace(tmp / "model.safetensors", target / "model.safetensors")
        shutil.rmtree(tmp, ignore_errors=True)
        fp32.unlink(missing_ok=True)
        if not self.has_better_translation():
            raise SetupError("The better translation model is incomplete; run this again.")
        size = (target / "model.safetensors").stat().st_size / 1e9
        say(f"Better translation model: ready ({size:.1f} GB).")

    def separation_model(self) -> None:
        """Hybrid Demucs weights for "Remove background music" (torchaudio HDEMUCS_HIGH_MUSDB_PLUS, 335 MB)."""
        target = self.models_dir / DEMUCS_DIR / DEMUCS_FILE
        if target.is_file() and target.stat().st_size == DEMUCS_SIZE:
            say("Music-removal model (Hybrid Demucs): present.")
            return
        say(f"Music-removal model: downloading Hybrid Demucs (about {DEMUCS_SIZE / 1e6:.0f} MB, resumable)...")
        fetch_demucs(self.models_dir)

    def has_zero_shot_vc(self) -> bool:
        """The converter is complete: checkout, weights and the stamp (as the engine's ``zsvc.installed``)."""
        files = [f"{sub}/{name}" for sub, _repo, _rev, names in SEEDVC_MODELS for name in names]
        return ((self.seed_vc / "modules" / "commons.py").is_file()
                and self._has(self.seed_vc, ("campplus_cn_common.bin",))
                and self._has(self.seed_vc_models, (*files, SEEDVC_STAMP)))

    def zero_shot_vc_runtime(self) -> None:
        """Seed-VC for the native accent of untrained voices: the source checkout (pinned commit), its two extra
        packages in the engine venv (munch, einops; it needs nothing else the engine does not have: torch 2.8,
        transformers 5, librosa), the weights (``hf download`` at pinned revisions, resumable), then the stamp that
        tells the engine it is complete. Runs offline afterwards."""
        py = self.engine_python()
        if not py.exists():
            raise SetupError("The engine environment is missing, so the zero-shot voice converter cannot be installed.")
        stamp = self.seed_vc_models / SEEDVC_STAMP
        stamp_value = f"{SEEDVC_COMMIT} {' '.join(SEEDVC_PACKAGES)} {' '.join(rev for _s, _r, rev, _f in SEEDVC_MODELS)}"
        if self.has_zero_shot_vc() and stamp.read_text(encoding="utf-8").strip() == stamp_value:
            say("Zero-shot voice converter (Seed-VC): present.")
            return
        self.ensure_repo("Seed-VC", SEEDVC_URL, SEEDVC_COMMIT, self.seed_vc)
        if self.quiet([py, "-I", "-c", "import munch, einops"], cwd=self.runtime) != 0:
            say("Zero-shot voice converter: adding munch and einops to the engine environment...")
            self.must(self.uv("pip", "install", "--python", py, *SEEDVC_PACKAGES),
                      "Installing the zero-shot voice converter's packages", cwd=self.runtime)
        if self.skip_models:
            return
        for sub, repo, rev, files in SEEDVC_MODELS:
            target = self.seed_vc_models / sub
            if self._has(target, files):
                continue
            say(f"Zero-shot voice converter: downloading {repo} ({', '.join(files)}, resumable)...")
            target.mkdir(parents=True, exist_ok=True)
            code = self.run([py, "-I", "-m", "huggingface_hub.cli.hf", "download", repo, *files, "--revision", rev,
                             "--local-dir", target], cwd=self.runtime, extra_env={"HF_HUB_DISABLE_XET": "1"})
            if code != 0:
                script = ("import sys; from huggingface_hub import hf_hub_download\n"
                          "for f in sys.argv[4:]: hf_hub_download(sys.argv[1], f, revision=sys.argv[2], local_dir=sys.argv[3])")
                self.must([py, "-I", "-c", script, repo, rev, target, *files],
                          f"Downloading {repo} for the zero-shot voice converter", cwd=self.runtime,
                          extra_env={"HF_HUB_DISABLE_XET": "1"})
            if not self._has(target, files):
                raise SetupError(f"The {repo} download finished but files are missing; run this again.")
        stamp.write_text(stamp_value, encoding="utf-8")
        if not self.has_zero_shot_vc():
            raise SetupError("The zero-shot voice converter is incomplete; run this again.")
        say("Zero-shot voice converter (Seed-VC): ready.")

    def engine_env(self) -> None:
        """The engine environment (Python 3.11, torch 2.8, transformers 5): the engine, the OmniVoice worker
        (omnivoice; num2words and the Ukrainian stress dictionary for the text front-end) and the speech worker
        (Whisper, NLLB via transformers + sentencepiece), g2p_en for the recording-text analysis."""
        stamp_value = (f"{' '.join(ENGINE_TORCH)} {' '.join(ENGINE_PACKAGES)} {' '.join(ENGINE_NO_DEPS)} "
                       f"{self.gpu.build} {self.plat}")
        stamp = self._stamp(self.engine_venv)
        py = self.engine_python()
        if py.exists() and stamp.exists() and stamp.read_text(encoding="utf-8").strip() == stamp_value:
            say("Engine environment: ready.")
            return
        if not py.exists():
            say(f"Engine environment: creating the Python {ENGINE_PYTHON} venv...")
            self.engine_venv.parent.mkdir(parents=True, exist_ok=True)
            self.must(self.uv("venv", "--python", ENGINE_PYTHON, "--allow-existing", self.engine_venv),
                      "Creating the engine venv", cwd=self.runtime)
        note = self._build_note()
        say(f"Engine environment: installing torch 2.8{f' ({note})' if note else ''}, OmniVoice, Whisper...")
        command = self.uv("pip", "install", "--python", py, *ENGINE_TORCH, *ENGINE_PACKAGES)
        command += self._torch_args(ENGINE_ROCM_INDEX)
        self.must(command, "Installing the engine environment", cwd=self.runtime)
        stress = self.uv("pip", "install", "--python", py, "--no-deps", *ENGINE_NO_DEPS)
        stress_ok = self.run(stress, cwd=self.runtime) == 0
        if not stress_ok:
            warn("The Ukrainian stress dictionary could not be installed (the next setup run tries again); Ukrainian "
                 "speech works without automatic stress.")
        # g2p_en fetches its NLTK data (CMUdict, tagger; a few MB) on first import: do it now, not in the app.
        if self.run([py, "-I", "-c", "import g2p_en"], cwd=self.runtime) != 0:
            warn("g2p_en could not fetch its NLTK data; the recording-text analysis falls back to a simpler method.")
        if stress_ok:
            stamp.write_text(stamp_value, encoding="utf-8")

    def omnivoice_models(self) -> None:
        """The OmniVoice weights (``hf download k2-fsa/OmniVoice``, 3.3 GB). The Whisper turbo checkpoint it uses once per
        reference clip comes with the Redub models (``speech_models``)."""
        py = self.engine_python()
        if not py.exists():
            raise SetupError("The engine environment is missing, so the OmniVoice model cannot be fetched.")
        target = self.omnivoice_model
        if self._has(target, OMNIVOICE_REQUIRED):
            say("OmniVoice model: present.")
        else:
            say(f"OmniVoice model: downloading {OMNIVOICE_REPO} (about 3.3 GB, resumable)...")
            target.mkdir(parents=True, exist_ok=True)
            code = self.run([py, "-I", "-m", "huggingface_hub.cli.hf", "download", OMNIVOICE_REPO, "--local-dir", target],
                            cwd=self.runtime, extra_env={"HF_HUB_DISABLE_XET": "1"})
            if code != 0:
                script = ("import sys; from huggingface_hub import snapshot_download; "
                          "snapshot_download(sys.argv[1], local_dir=sys.argv[2])")
                self.must([py, "-I", "-c", script, OMNIVOICE_REPO, target], "Downloading the OmniVoice model",
                          cwd=self.runtime, extra_env={"HF_HUB_DISABLE_XET": "1"})
            if not self._has(target, OMNIVOICE_REQUIRED):
                raise SetupError("The OmniVoice download finished but files are missing; run this again.")

    def applio_required(self) -> tuple[str, ...]:
        return APPLIO_REQUIRED + (APPLIO_WINDOWS_REQUIRED if self.plat == "windows" else ())

    def applio_models(self) -> None:
        """Applio's prerequisites: HiFi-GAN pretrained models, rmvpe, contentvec (and ffmpeg.exe on Windows)."""
        if not self.rvc_python().exists():
            raise SetupError("The Applio environment is missing, so its models cannot be fetched.")
        if self._has(self.applio, self.applio_required()):
            say("Applio prerequisites: present.")
            return
        say("Applio prerequisites: downloading pretrained HiFi-GAN, rmvpe, contentvec (about 1.5 GB)...")
        self.must([self.rvc_python(), "core.py", "prerequisites", "--pretraineds-hifigan", "--models", "--exe"],
                  "Downloading the Applio prerequisites", cwd=self.applio)
        if not self._has(self.applio, self.applio_required()):
            raise SetupError("The Applio prerequisites finished but files are missing; run this again.")

    # ---- step 4: runtime.toml --------------------------------------------------------------------------------

    def write_runtime_toml(self) -> None:
        for what, path in (("engine", self.engine_python()), ("Applio", self.rvc_python())):
            if not path.exists():
                raise SetupError(f"runtime.toml not written: the {what} interpreter {path} does not exist.")
        text = (
            "# Written by tools/setup/setup.py; setup rewrites it on every run.\n"
            "[python]\n"
            f"engine = {toml_str(self.engine_python())}\n"
            f"rvc = {toml_str(self.rvc_python())}\n"
            "\n[paths]\n"
            f"applio = {toml_str(self.applio)}\n"
            f"omnivoice_model = {toml_str(self.omnivoice_model)}\n"
            f"seed_vc = {toml_str(self.seed_vc)}\n"
            f"seed_vc_models = {toml_str(self.seed_vc_models)}\n"
            "\n[device]\n"
            f'backend = "{self.gpu.backend}"\n'
        )
        if self.gpu.vendor:
            text += f'vendor = "{self.gpu.vendor}"\n'
        if self.gpu.vendor == "amd":
            text += f"name = {toml_str(self.gpu.name)}\nvram_gb = {self.gpu.vram_gb}\n"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.runtime_toml.with_suffix(".toml.tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(self.runtime_toml)
        say(f"Wrote {self.runtime_toml}")
        self.write_location()

    def write_location(self) -> None:
        """Pointer file for the app: ``location.toml`` names the data dir, only when it is not the default one
        (a stale pointer is removed when the data dir is the default again)."""
        path = location_file(self.plat)
        if self.data_dir == default_data_dir(self.plat).resolve():
            try:
                path.unlink()
                say(f"Removed {path} (the data directory is the default again).")
            except OSError:
                pass
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".toml.tmp")
            tmp.write_text(f"data_dir = {toml_str(self.data_dir)}\n", encoding="utf-8", newline="\n")
            tmp.replace(path)
            say(f"Wrote {path} (tells the app where the data directory is).")
        except OSError as error:
            warn(f"could not write {path}: {error}. Set {DATA_ENV} to {self.data_dir} before starting the app.")

    def offer_legacy_removal(self, confirmed: bool) -> None:
        """``runtime/index-tts`` (IndexTTS-2.5 of earlier versions, ~19 GB) is no longer used. It is deleted only when
        the user says so: ``--remove-indextts``, or "y" at the prompt in an interactive terminal. Never automatically."""
        legacy = self.legacy_indextts
        if not legacy.is_dir():
            return
        if not confirmed:
            if self.assume_yes or not sys.stdin.isatty():
                say(f"{legacy} (IndexTTS-2.5, about 19 GB) is no longer used. Delete it yourself, or run setup with "
                    "--remove-indextts.")
                return
            try:
                answer = input(f"[setup] {legacy} (IndexTTS-2.5, about 19 GB) is no longer used. Delete it now? [y/N] ")
            except EOFError:
                answer = ""
            if answer.strip().lower() not in ("y", "yes"):
                say("Kept it. Run setup with --remove-indextts to delete it later.")
                return
        say(f"Deleting {legacy}...")
        shutil.rmtree(legacy, ignore_errors=True)
        if legacy.exists():
            warn(f"Could not delete everything in {legacy} (a file in use?); delete the rest by hand.")
        else:
            say("Deleted.")

    # ---- step 5: quick check ---------------------------------------------------------------------------------

    def quick_check(self) -> list[str]:
        """Verify an existing setup without loading a model: files, interpreters, torch and the GPU."""
        problems: list[str] = []
        if not self.runtime_toml.is_file():
            problems.append(f"{self.runtime_toml} is missing.")
        else:
            say(f"runtime.toml: {self.runtime_toml}")
            text = self.runtime_toml.read_text(encoding="utf-8")
            for key, path in (("engine", self.engine_python()), ("rvc", self.rvc_python())):
                if toml_str(path) not in text:
                    problems.append(f"runtime.toml does not point [python] {key} at {path}.")
        head = self.read(self.git("rev-parse", "HEAD"), self.applio).strip() if self.applio.exists() else ""
        if head == APPLIO_COMMIT:
            say(f"Applio: pinned commit {APPLIO_COMMIT[:9]}.")
        else:
            problems.append(f"Applio is not at the pinned commit {APPLIO_COMMIT[:9]} (found {head[:9] or 'nothing'}).")
        for label, folder, files in (("Applio prerequisites", self.applio, self.applio_required()),
                                     ("OmniVoice model", self.omnivoice_model, OMNIVOICE_REQUIRED)):
            missing = [f for f in files if not (folder / f).is_file()]
            if missing:
                problems.append(f"{label}: missing {', '.join(missing)} in {folder}")
            else:
                say(f"{label}: all files present.")
        for label, py in (("Engine", self.engine_python()), ("Applio", self.rvc_python())):
            problems += self._check_torch(label, py)
        if self.engine_python().exists():
            out = self.read([self.engine_python(), "-I", "-c",
                             "import omnivoice, num2words, transformers; print(transformers.__version__)"]
                            ).strip().splitlines()
            if out:
                say(f"Engine environment: OmniVoice + num2words, transformers {out[-1]}.")
            else:
                problems.append("The engine environment cannot import omnivoice / num2words / transformers.")
        if self.legacy_indextts.is_dir():
            say(f"Note: {self.legacy_indextts} (IndexTTS-2.5, about 19 GB) is no longer used and can be deleted "
                "(setup --remove-indextts).")
        problems += self._check_media()
        return problems

    def _check_media(self) -> list[str]:
        """Redub and the audio imports/exports: ffmpeg + ffprobe, and Whisper in the engine environment."""
        problems: list[str] = []
        if self.plat == "windows":
            folder = self.applio
            found = all((folder / exe).is_file() for exe in APPLIO_WINDOWS_REQUIRED)
        else:
            found = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        if found:
            say("ffmpeg and ffprobe: present.")
        else:
            where = "the Applio folder" if self.plat == "windows" else "PATH"
            problems.append(f"ffmpeg / ffprobe not found in {where}: audio and video files cannot be read or exported.")
        py = self.engine_python()
        if py.exists():
            out = self.read([py, "-c", "import whisper; print(whisper.__version__)"]).strip().splitlines()
            if out:
                say(f"openai-whisper {out[-1]} (Redub transcription).")
            else:
                problems.append("openai-whisper is missing from the engine environment (Redub transcription).")
            out = self.read([py, "-c", "import transformers, sentencepiece; print(transformers.__version__)"]
                            ).strip().splitlines()
            if out:
                say(f"transformers {out[-1]} + sentencepiece (Redub translation).")
            else:
                problems.append("transformers or sentencepiece is missing from the engine environment "
                                "(Redub translation).")
        nllb = self.models_dir / NLLB_DIR
        missing = [f for f in NLLB_REQUIRED if not (nllb / f).is_file()]
        if missing:
            problems.append(f"Translation model: missing {', '.join(missing)} in {nllb}")
        else:
            say("Translation model (NLLB-200 distilled 600M): all files present.")
        if self.has_better_translation():
            say("Better translation model (NLLB-200 distilled 1.3B): all files present.")
        elif self.wants_better_translation():
            say("Better translation model (NLLB-200 distilled 1.3B): not downloaded (optional; run setup to add it, "
                "Redub uses the 600M meanwhile).")
        if not (self.models_dir / "whisper" / "large-v3-turbo.pt").is_file():
            say("Whisper turbo checkpoint: not downloaded yet (setup fetches it; otherwise the first Redub does).")
        demucs = self.models_dir / DEMUCS_DIR / DEMUCS_FILE
        if demucs.is_file() and demucs.stat().st_size == DEMUCS_SIZE:
            say("Music-removal model (Hybrid Demucs): present.")
        else:
            say(f"Music-removal model: not downloaded yet ({demucs}); media files are used with their music until "
                "setup fetches it.")
        if self.has_zero_shot_vc():
            say("Zero-shot voice converter (Seed-VC): present.")
        else:
            say("Zero-shot voice converter (Seed-VC): not installed (optional; untrained voices then have no native "
                "accent until setup installs it).")
        return problems

    def _check_torch(self, label: str, py: Path) -> list[str]:
        if not py.exists():
            return [f"{label} interpreter missing: {py}"]
        script = (
            "import json, torch\n"
            "cuda = torch.cuda.is_available()\n"
            "mps = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())\n"
            "name = torch.cuda.get_device_name(0) if cuda else ''\n"
            "hip = bool(getattr(torch.version, 'hip', None))\n"
            "print(json.dumps({'torch': torch.__version__, 'cuda': cuda, 'mps': mps, 'name': name, 'hip': hip}))\n"
        )
        out = self.read([py, "-c", script]).strip().splitlines()
        if not out:
            return [f"{label}: importing torch failed with {py}."]
        info = json.loads(out[-1])
        wanted = {"cuda": info["cuda"], "mps": info["mps"], "cpu": True}[self.gpu.backend]
        detail = info["name"] if info["cuda"] else ("mps" if info["mps"] else "cpu")
        say(f"{label}: torch {info['torch']}, device {detail}.")
        if self.gpu.build == "rocm" and not info["hip"]:
            return [f"{label}: torch {info['torch']} is not the ROCm build; run setup again to reinstall it."]
        if not wanted and self.gpu.build == "rocm":
            return [f"{label}: the ROCm build of torch cannot use {self.gpu.name}. The card may be too old for these "
                    "ROCm wheels, or /dev/kfd may be inaccessible; run setup with --cpu to use the CPU."]
        if not wanted:
            return [f"{label}: torch {info['torch']} cannot use the {self.gpu.backend} backend."]
        return []

    # ---- step 6: the end -------------------------------------------------------------------------------------

    def ready_lines(self) -> list[str]:
        binary = self.root / "app" / "target" / "release" / (BIN_NAME + (".exe" if self.plat == "windows" else ""))
        lines = ["Start the app with:"]
        if self.data_dir != default_data_dir(self.plat).resolve():
            lines.insert(0, f"The app finds the data directory {self.data_dir} through {location_file(self.plat)}.")
        app = self.root / "app"
        if not (app / "Cargo.toml").is_file():
            return [*lines[:-1], "Go back to the Mimvox window: it notices the finished setup by itself."]
        if binary.exists():
            lines.append(f"  {binary}")
        elif self.plat == "windows":
            lines.append(f'  cd /d "{app}" && cargo run --release')
        else:
            lines.append(f"  cd {shlex_quote(str(app))} && cargo run --release")
        return lines


def fetch_demucs(models_dir: Path) -> Path:
    """Download the Hybrid Demucs weights to ``<models_dir>/demucs/hdemucs_high_trained.pt``: into a ``.part`` file that
    a later run continues (an HTTP range request), checked against the pinned SHA-256, then renamed into place."""
    folder = models_dir / DEMUCS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / DEMUCS_FILE
    part = folder / (DEMUCS_FILE + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > DEMUCS_SIZE:
        part.unlink()
        have = 0
    if have < DEMUCS_SIZE:
        request = urllib.request.Request(DEMUCS_URL, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                resumed = have > 0 and response.status == 206
                done = have if resumed else 0
                step = DEMUCS_SIZE // 10
                next_note = (done // step + 1) * step
                with open(part, "ab" if resumed else "wb") as handle:
                    for block in iter(lambda: response.read(1 << 20), b""):
                        handle.write(block)
                        done += len(block)
                        if done >= next_note and done < DEMUCS_SIZE:
                            say(f"  {100 * done // DEMUCS_SIZE}% of {DEMUCS_SIZE / 1e6:.0f} MB")
                            next_note = (done // step + 1) * step
        except OSError as error:
            raise SetupError(f"Downloading the music-removal model failed ({error}); run setup again to resume."
                             ) from error
    digest = hashlib.sha256()
    with open(part, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != DEMUCS_SHA256:
        part.unlink()
        raise SetupError("The music-removal model download is corrupt (checksum mismatch); run setup again.")
    os.replace(part, target)
    say(f"Music-removal model: saved to {target}.")
    return target


def shlex_quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="setup", description="Full setup of Mimvox: toolchains, runtime, models, runtime.toml.")
    parser.add_argument("--data-dir", metavar="DIR",
                        help=f"data directory (default: ${DATA_ENV}, else asked for; the default is the local app data dir + "
                             f"{APP_DIR_NAME}, or the one in location.toml)")
    parser.add_argument("--skip-toolchain", action="store_true", help="do not install Rust / compilers / system packages")
    parser.add_argument("--skip-models", action="store_true", help="set the environments up but download no models")
    parser.add_argument("--cpu", action="store_true", help="ignore any GPU and set up the CPU backend")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="do not ask for confirmation (then --data-dir is required: no silent default)")
    parser.add_argument("--force", action="store_true",
                        help=f"continue even when the drive has less than {DISK_NEEDED_GB} GB free")
    parser.add_argument("--quick-check", action="store_true",
                        help="verify an existing setup (files, interpreters, torch on the GPU) and download nothing")
    parser.add_argument("--runtime-toml-only", action="store_true",
                        help="only (re)write runtime.toml for a runtime that already exists in the data directory")
    parser.add_argument("--move-to", metavar="DIR",
                        help="move the installed data directory (from --data-dir, location.toml or the default) to DIR: "
                             "renames on the same volume, else copies with '[move] 37% 11.4/31.0 GB' progress lines, "
                             "fixes the paths inside the virtual environments, rewrites runtime.toml and location.toml, "
                             "checks the result and only then deletes the old copy")
    parser.add_argument("--no-check", action="store_true", help="with --move-to: skip the quick check (testing)")
    parser.add_argument("--better-translation", choices=("auto", "yes", "no"), default="auto",
                        help="the optional better translation model for Redub (NLLB-200 1.3B, a 5.5 GB download kept "
                             "as 2.7 GB): auto = only with a CUDA or ROCm GPU (default), yes = also without one, no = never")
    parser.add_argument("--no-zero-shot-vc", action="store_true",
                        help="leave out Seed-VC (GPL-3.0, about 2 GB), the zero-shot voice converter that gives voices "
                             "that are not trained a native accent")
    parser.add_argument("--remove-indextts", action="store_true",
                        help="delete <data>/runtime/index-tts (the IndexTTS-2.5 runtime of earlier versions, about 19 GB) "
                             "without asking; without this flag setup only offers it in an interactive terminal")
    return parser.parse_args(argv)


def find_uv() -> str:
    installed = os.environ.get(UV_ENV)
    if installed and Path(installed).exists():
        return installed
    found = shutil.which("uv")
    if found:
        return found
    raise SetupError("uv was not found; run setup.sh / setup.cmd, which installs it.")


def choose_data_dir(args: argparse.Namespace, plat: str) -> Path:
    """``--data-dir`` > ``$MIMVOX_DATA`` > ask (interactive) > nothing: a full setup never picks a place silently.
    Only the checks that download nothing (``--quick-check``, ``--runtime-toml-only``) fall back to location.toml or
    the default directory."""
    explicit = args.data_dir or os.environ.get(DATA_ENV)
    default = (read_location(plat) or default_data_dir(plat)).expanduser().resolve()
    if explicit:
        chosen = Path(explicit).expanduser().resolve()
    elif args.quick_check or args.runtime_toml_only:
        chosen = default
    elif not args.yes and sys.stdin.isatty():
        chosen = ask_data_dir(plat, default)
    else:
        raise SetupError("no install location given: pass --data-dir DIR (a full setup needs about "
                         f"{RUNTIME_GB} GB, so the place is never chosen silently), or run it in a terminal "
                         "without --yes to be asked.")
    return chosen


def ask_data_dir(plat: str, default: Path) -> Path:
    say(f"Where should the voice runtime go? It needs about {RUNTIME_GB} GB (models and two Python environments).")
    say("Free space per volume:")
    for root, free in volumes(plat):
        say(f"  {root:<28} {free:7.0f} GB free")
    say(f"Default: {default}")
    while True:
        try:
            answer = input("[setup] Data directory (Enter for the default): ").strip().strip('"')
        except EOFError:
            raise SetupError("no install location chosen (no input)") from None
        chosen = Path(answer).expanduser() if answer else default
        if not chosen.is_absolute():
            say("Please give a full path (for example D:\\Voice or /data/voice).")
            continue
        return chosen.resolve()


def build_setup(args: argparse.Namespace) -> Setup:
    plat = host_platform()
    root = Path(__file__).resolve().parents[2]
    data_dir = choose_data_dir(args, plat)
    return Setup(root=root, plat=plat, arch=host_arch(), data_dir=data_dir,
                 skip_toolchain=args.skip_toolchain, skip_models=args.skip_models, force_cpu=args.cpu,
                 assume_yes=args.yes, force=args.force, uv_path=find_uv(),
                 better_translation=args.better_translation, zero_shot_vc=not args.no_zero_shot_vc)


def run_move(args: argparse.Namespace) -> int:
    """``--move-to``: see move_runtime.py. Prints ``[move]`` lines; exit code 0 only when everything went through."""
    import tomllib

    import move_runtime as mv

    plat = host_platform()
    windows = plat == "windows"
    explicit = args.data_dir or os.environ.get(DATA_ENV)
    current = Path(explicit).expanduser() if explicit else (read_location(plat) or default_data_dir(plat))
    new = Path(args.move_to).expanduser().resolve()
    try:
        old, resume, same = mv.prepare(current.resolve(), new, windows, args.force)
        mv.move_files(old, new, resume, same, args.force)
        fixer = mv.fix_venv_paths(new, old, windows)
        mv.out(f"Updated {len(fixer.fixed)} file(s) that held the old path")
        for path in fixer.unfixed:
            mv.out(f"warning: could not update {path}; it still points to the old place (reinstall that package to refresh it)")
        device: dict = {}
        try:
            device = tomllib.loads((new / "runtime.toml").read_text(encoding="utf-8")).get("device", {})
        except (OSError, ValueError):
            pass
        gpu = Gpu(str(device.get("backend", "cuda")), str(device.get("name", "")), float(device.get("vram_gb", 0.0)),
                  str(device.get("vendor", "")))
        setup = Setup(root=Path(__file__).resolve().parents[2], plat=plat, arch=host_arch(), data_dir=new, gpu=gpu)
        setup.write_runtime_toml()  # also writes (or removes) location.toml
        mv.out("Rewrote runtime.toml and the location pointer")
        if not args.no_check:
            problems = setup.quick_check()
            if problems:
                raise mv.MoveError("the moved runtime failed the quick check, the old copy was kept: " + "; ".join(problems))
            mv.out("Quick check passed")
        (new / mv.MARKER).unlink(missing_ok=True)
        if old.exists():
            mv.out(f"Deleting the old copy {old}")
            mv.remove_tree(old)
            setup.write_location()  # the old folder may have held location.toml (the default dir is also the config dir)
        mv.out("done")
        return 0
    except mv.MoveError as error:
        mv.out(f"failed: {error}")
        return 1
    except OSError as error:
        mv.out(f"failed: {error}. Run the same command again to resume.")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.move_to:
        return run_move(args)
    try:
        setup = build_setup(args)
    except SetupError as error:
        say(f"Failed: {error}")
        return 1
    setup.detect()

    if args.quick_check:
        problems = setup.quick_check()
        if problems:
            say(f"Quick check found {len(problems)} problem(s):")
            for line in problems:
                say(f"  - {line}")
            say("Run setup again (without --quick-check) to fix them.")
            return 1
        say("Quick check passed.")
        return 0

    if args.runtime_toml_only:
        try:
            setup.write_runtime_toml()
        except SetupError as error:
            say(f"Failed: {error}")
            return 1
        return 0

    setup.confirm()
    failed: list[str] = []

    def attempt(step) -> bool:  # noqa: ANN001 - a bound method
        try:
            step()
            return True
        except SetupError as error:
            say(f"Failed: {error}")
            failed.append(str(error))
            return False

    attempt(setup.toolchains)
    applio_cloned = attempt(lambda: setup.ensure_repo("Applio", APPLIO_URL, APPLIO_COMMIT, setup.applio))
    if attempt(setup.engine_env) and not args.skip_models:
        if attempt(setup.omnivoice_models):
            if attempt(setup.speech_models) and setup.wants_better_translation():
                attempt(setup.better_translation_model)
        attempt(setup.separation_model)
    if setup.zero_shot_vc and setup.engine_python().exists():
        attempt(setup.zero_shot_vc_runtime)
    if applio_cloned and attempt(setup.applio_venv) and not args.skip_models:
        attempt(setup.applio_models)
    attempt(setup.write_runtime_toml)
    if not failed:
        setup.offer_legacy_removal(args.remove_indextts)
    if args.skip_models:
        warn("--skip-models: no model files were downloaded; run setup again without it before using the app.")

    if failed:
        say(f"Setup finished with {len(failed)} step(s) failed; fix them and run it again (it resumes):")
        for line in failed:
            say(f"  - {line}")
        return 1
    for line in setup.ready_lines():
        say(line)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("Interrupted. Run it again to resume where it stopped.")
        sys.exit(130)
