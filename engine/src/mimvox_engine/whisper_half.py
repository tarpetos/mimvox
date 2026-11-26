"""Whisper with fp16 weights built directly on the GPU (1.6 GB of VRAM for turbo instead of 3.2 GB). Standalone: the
workers load this file by its path. LayerNorm stays fp32 (whisper casts around it). ``half=False`` builds fp32 (the
CPU) from the same mapped checkpoint, without the checkpoint's 1.6 GB copy that ``whisper.load_model`` makes."""

import os
from pathlib import Path


def default_root():
    """Where ``whisper.load_model`` keeps its checkpoints when no folder is given."""
    return Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"


def load(whisper, name, root, device, half=True):
    import torch

    path = whisper._download(whisper._MODELS[name], root or default_root(), False)
    try:  # mapped from the file: the 1.6 GB of weights never become process memory on their way to the GPU
        ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        with Path(path).open("rb") as fh:
            ckpt = torch.load(fh, map_location="cpu", weights_only=True)
    dims = whisper.model.ModelDimensions(**ckpt["dims"])
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16 if half else prev)
    try:
        with torch.device(device):
            model = whisper.model.Whisper(dims)
    finally:
        torch.set_default_dtype(prev)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    for m in model.modules():
        if isinstance(m, whisper.model.LayerNorm):
            m.float()
    alignment = getattr(whisper, "_ALIGNMENT_HEADS", {}).get(name)
    if alignment is not None:
        model.set_alignment_heads(alignment)
    return model.eval()
