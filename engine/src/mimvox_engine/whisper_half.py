"""Whisper with fp16 weights built directly on the GPU: 1.6 GB of VRAM for turbo instead of 3.2 GB (no fp32 copy is
ever made). Shared by the standalone workers (``speech_worker.py``, the OmniVoice worker),
which load this file by its path. LayerNorm stays fp32 (whisper casts around it)."""
import os


def default_root():
    """Where ``whisper.load_model`` keeps its checkpoints when no folder is given."""
    cache = os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    return os.path.join(cache, "whisper")


def load(whisper, name, root, device):
    import torch

    path = whisper._download(whisper._MODELS[name], root or default_root(), False)
    with open(path, "rb") as fh:
        ckpt = torch.load(fh, map_location="cpu", weights_only=True)
    dims = whisper.model.ModelDimensions(**ckpt["dims"])
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
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
