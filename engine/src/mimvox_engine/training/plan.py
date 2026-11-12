"""Training plan: how many epochs, which batch size, whether the dataset fits in VRAM."""
from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from pathlib import Path

# The validated reference run: 300 epochs on 6.4 minutes of speech (about 7200 steps at batch 8).
REF_MINUTES = 6.4
REF_EPOCHS = 300
MIN_EPOCHS = 150
MAX_EPOCHS = 450
MIN_TRAIN_MINUTES = 1.0


def recommended_epochs(minutes: float) -> int:
    """Epochs for a dataset of ``minutes`` of speech.

    Fewer steps per epoch on small data means it needs more epochs; big data converges in fewer. The curve is
    ``300 * sqrt(6.4 / minutes)`` clamped to 150..450 and rounded to 10: 1 min -> 450, 3 min -> 440, 6.4 min -> 300,
    10 min -> 240, 15 min -> 200, 20+ min -> 170 or fewer (floor 150).
    """
    if minutes <= 0:
        return MAX_EPOCHS
    raw = REF_EPOCHS * (REF_MINUTES / minutes) ** 0.5
    return int(round(min(MAX_EPOCHS, max(MIN_EPOCHS, raw)) / 10.0)) * 10


def wav_seconds(path: Path) -> float:
    """Duration of a WAV file (stdlib only; falls back to soundfile for float/24-bit files)."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError):
        import soundfile as sf  # engine venv has it

        info = sf.info(str(path))
        return info.frames / float(info.samplerate)


def dataset_parts(dataset_dir: Path) -> list[Path]:
    return sorted(p for p in dataset_dir.glob("part*.wav") if p.is_file())


def free_vram_gb() -> tuple[float | None, float | None]:
    """(free, total) GPU memory in GiB of GPU 0 via nvidia-smi, or (None, None) when unknown."""
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform != "win32":
        for cand in ("/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi"):
            if Path(cand).exists():
                exe = cand
                break
    if exe is None:
        return None, None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        free_mib, total_mib = (float(x) for x in out.strip().splitlines()[0].split(","))
        return free_mib / 1024.0, total_mib / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None


# Measured on an RTX 4070 (HiFi-GAN, 40 kHz, fp32): the training process itself needs roughly
# BASE_GB + PER_SAMPLE_GB * batch; the cached dataset needs about CACHE_MB_PER_SEC per second of audio.
BASE_GB = 2.0
PER_SAMPLE_GB = 0.35
CACHE_MB_PER_SEC = 1.3
BATCH_CHOICES = (8, 6, 4, 2)


def pick_batch_size(free_gb: float | None, dataset_seconds: float, backend: str = "cuda") -> tuple[int, bool]:
    """Return ``(batch_size, cache_data_in_gpu)`` for the free VRAM.

    Batch 8 is the validated default (more samples per step does not make the model better, it only makes fewer
    steps per epoch), so it is lowered only when memory is short. The dataset is cached in VRAM when it fits with
    a safety margin, which removes the per-batch host-to-device copies.
    """
    if backend != "cuda":
        return 4, False
    if free_gb is None:
        return 8, False
    batch = BATCH_CHOICES[-1]
    for b in BATCH_CHOICES:
        if free_gb >= BASE_GB + PER_SAMPLE_GB * b + 0.8:
            batch = b
            break
    cache_gb = dataset_seconds * CACHE_MB_PER_SEC / 1024.0
    need = BASE_GB + PER_SAMPLE_GB * batch + cache_gb + 1.0
    return batch, free_gb >= need
