"""What an RVC voice model (.pth) must hold, read with pickletools: nothing in the file is unpickled or run."""

import io
import pickletools
import zipfile
from pathlib import Path

MAX_PICKLE = 16 * 1024 * 1024
_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
# Applio's VoiceConverter reads cpt["weight"]["emb_g.weight"] and cpt["config"].
_NEEDED = {"weight", "config", "emb_g.weight"}
_CHECKPOINT = {"optimizer", "iteration", "learning_rate"}

NOT_TORCH = "The model file is not a .pth model. If it came in a .zip, unpack it and choose the .pth inside."
CHECKPOINT = (
    "The model file is a training checkpoint (G_ or D_), not a voice model. Choose the smaller .pth that "
    "RVC or Applio saves for inference."
)
NOT_RVC = "The model file is not an RVC voice model (RVC v1/v2 or Applio)."


def _strings(stream: io.BytesIO, pickles: int) -> set[str]:
    out: set[str] = set()
    for _ in range(pickles):
        for _op, arg, _pos in pickletools.genops(stream):
            if isinstance(arg, str):
                out.add(arg)
    return out


def _zip_pickle(path: str | Path) -> bytes | None:
    with zipfile.ZipFile(path) as z:
        found = [i for i in z.infolist() if i.filename.count("/") == 1 and i.filename.endswith("/data.pkl")]
        if len(found) != 1 or found[0].file_size > MAX_PICKLE:
            return None
        return z.read(found[0])


def _legacy_strings(path: str | Path) -> set[str] | None:
    with Path(path).open("rb") as fh:
        stream = io.BytesIO(fh.read(MAX_PICKLE))
    magic = [arg for _op, arg, _pos in pickletools.genops(stream) if isinstance(arg, int)]
    if _LEGACY_MAGIC not in magic:
        return None
    return _strings(stream, 3)  # protocol version, sys info, the object


def model_problem(path: str | Path) -> str | None:
    """Why ``path`` cannot be an RVC voice model, or None when it looks like one."""
    try:
        if zipfile.is_zipfile(path):
            data = _zip_pickle(path)
            strings = None if data is None else _strings(io.BytesIO(data), 1)
        else:
            strings = _legacy_strings(path)
    except (ValueError, EOFError, NotImplementedError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return NOT_TORCH
    if strings is None:
        return NOT_TORCH
    if strings >= _NEEDED:
        return None
    if strings & _CHECKPOINT:
        return CHECKPOINT
    return NOT_RVC
