"""Pause / resume support: checkpoint bookkeeping in Applio's log folder and the run markers in ``voice.json``.

How Applio resumes (``rvc/train/train.py``): when ``core.py train`` starts and ``logs/<model>/`` holds ``G_*.pth`` and
``D_*.pth``, it loads the newest pair (highest number in the file name), reads the ``iteration`` stored inside (the epoch
that was finished) and continues with ``epoch + 1``. Pretrained weights are loaded only when no checkpoint exists.

Applio's ``--save-only-latest`` overwrites one fixed file name in place (``G_2333333.pth``), so a crash or a kill during
the save destroys the only checkpoint. This module instead lets Applio write ``G_<step>.pth`` / ``D_<step>.pth`` and
removes the older pair only once the next epoch is running (the save is then complete), so a valid checkpoint always
exists. Torn files (no zip directory at the end) are detected without loading them and the previous pair is used.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

log = logging.getLogger("mimvox.training")

# epochs between G/D checkpoints (about 1.3 GB written each time at 40 kHz); MIMVOX_CHECKPOINT_EVERY for tests
CHECKPOINT_EVERY = max(1, int(os.environ.get("MIMVOX_CHECKPOINT_EVERY", "5")))
PAUSE_PATIENCE_S = 90.0       # a pause waits at most this long for the next checkpoint save
_CKPT = re.compile(r"^(?P<kind>[GD])_(?P<step>\d+)\.pth$")
_WEIGHTS = re.compile(r"^(?P<m>.+)_(?P<e>\d+)e_(?P<s>\d+)s\.pth$")
FEATURE_DIRS = ("sliced_audios", "extracted", "f0", "f0_voiced")


def model_name(voice_id: str) -> str:
    return "mimvox_" + re.sub(r"[^A-Za-z0-9_]", "_", voice_id)


def logdir_for(runtime: dict[str, Any], voice_id: str) -> Path | None:
    applio = runtime.get("paths", {}).get("applio")
    return Path(applio) / "logs" / model_name(voice_id) if applio else None


def checkpoint_every(total_epochs: int) -> int:
    return max(1, min(total_epochs, CHECKPOINT_EVERY))


# ------------------------------------------------------------------------------------------------ checkpoints

def _is_zip(path: Path) -> bool:
    """A finished ``torch.save`` file is a zip archive whose directory sits at the end; a torn write has none."""
    try:
        if path.stat().st_size < 1024:
            return False
        with zipfile.ZipFile(path):
            return True
    except (OSError, zipfile.BadZipFile):
        return False


def _pairs(logdir: Path) -> dict[int, dict[str, Path]]:
    found: dict[int, dict[str, Path]] = {}
    for p in logdir.glob("*.pth"):
        m = _CKPT.match(p.name)
        if m:
            found.setdefault(int(m.group("step")), {})[m.group("kind")] = p
    return found


def newest_valid_step(logdir: Path) -> int | None:
    """Step number of the newest G/D pair that is complete on disk, or ``None``."""
    pairs = _pairs(logdir)
    for step in sorted(pairs, reverse=True):
        pair = pairs[step]
        if "G" in pair and "D" in pair and _is_zip(pair["G"]) and _is_zip(pair["D"]):
            return step
    return None


def prune_checkpoints(logdir: Path, keep_step: int | None) -> None:
    """Delete every G/D pair except ``keep_step`` and the per-save weight exports of other steps.

    With ``keep_step=None`` all checkpoints go. Applio's own pick is "highest number", so leaving a torn newer pair
    next to a valid older one would make it load the torn one: call this with the valid step before every start.
    """
    for p in list(logdir.glob("*.pth")):
        c = _CKPT.match(p.name)
        w = _WEIGHTS.match(p.name)
        if c and int(c.group("step")) != keep_step:
            p.unlink(missing_ok=True)
        elif w and not c and int(w.group("s")) != keep_step:
            p.unlink(missing_ok=True)  # intermediate weight export; the final one is written again at the last epoch


def repair(logdir: Path) -> int | None:
    """Make the folder safe to resume from: keep only the newest valid pair. Returns its step (``None`` = no pair)."""
    step = newest_valid_step(logdir)
    prune_checkpoints(logdir, step)
    return step


def features_ready(logdir: Path) -> bool:
    """The preprocess and extract outputs Applio's ``train`` needs are all there."""
    if not (logdir / "config.json").is_file() or not (logdir / "filelist.txt").is_file():
        return False
    for name in FEATURE_DIRS:
        d = logdir / name
        try:
            if not d.is_dir() or not any(d.iterdir()):
                return False
        except OSError:
            return False
    return True


# ---------------------------------------------------------------------------------------------- orphan workers

def _pid_is_applio(pid: int) -> bool:
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
            ).stdout.lower()
            return "python" in out
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return b"train.py" in fh.read()
    except (OSError, subprocess.SubprocessError):
        return False


def kill_orphans(logdir: Path) -> None:
    """Stop Applio training workers left over from an engine that was killed without cleaning up.

    ``train.py`` records its worker PIDs in ``config.json``; only python processes among them are touched.
    """
    try:
        pids = json.loads((logdir / "config.json").read_text(encoding="utf-8")).get("process_pids") or []
    except (OSError, ValueError, AttributeError):
        return
    for pid in pids:
        if not isinstance(pid, int) or pid <= 0 or not _pid_is_applio(pid):
            continue
        log.warning("stopping an orphaned Applio training worker (pid %s)", pid)
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=30,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                os.kill(pid, 9)
        except (OSError, subprocess.SubprocessError):
            pass


def discard_logdir(runtime: dict[str, Any], voice_id: str) -> None:
    """Delete the Applio folder of a voice (paused or abandoned run), after stopping workers that might still use it."""
    logdir = logdir_for(runtime, voice_id)
    if logdir is None or not logdir.exists():
        return
    kill_orphans(logdir)
    shutil.rmtree(logdir, ignore_errors=True)
