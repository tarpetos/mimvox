"""RVC worker, run in the Applio venv (a separate torch): imports nothing from ``mimvox_engine``.

    python -I rvc_worker.py --applio <dir>

Ops (wire format: ``worker_kit.py``):
  convert {"in","out","pth","index","idx","protect","f0","pitch"} -> {"elapsed":s}
  release {}                                                       -> {}
  ping    {}                                                       -> {}
"""

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path


def _load_kit():
    path = Path(__file__).absolute().parent / "worker_kit.py"
    spec = importlib.util.spec_from_file_location("mimvox_worker_kit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kit = _load_kit()
INP, OUT = kit.open_channels()  # before Applio's imports: they print a lot, also from C code

ap = argparse.ArgumentParser()
ap.add_argument("--applio", required=True)
args = ap.parse_args()

APPLIO = Path(args.applio).absolute()
os.chdir(APPLIO)
sys.path.insert(0, str(APPLIO))

import faiss  # noqa: E402
from rvc.infer.infer import VoiceConverter  # noqa: E402

# Applio re-reads the FAISS index from disk on every call; cache it (searching is read-only).
_orig_read_index = faiss.read_index
_index_cache = {}


def _cached_read_index(path, *a, **kw):
    st = Path(path).stat()
    key = (Path(path).absolute(), st.st_mtime_ns, st.st_size)
    if key not in _index_cache:
        _index_cache.clear()
        _index_cache[key] = _orig_read_index(path, *a, **kw)
    return _index_cache[key]


faiss.read_index = _cached_read_index

vc = VoiceConverter()
_real_cleanup = vc.cleanup_model
# convert_audio() unloads HuBERT and the voice model after every call; a persistent worker keeps them.
vc.cleanup_model = lambda: None

_safe_dir = APPLIO / "logs" / "_mimvox_index"


def safe_index(path):
    """Applio rewrites 'trained' to 'added' anywhere in the index path; sidestep that for odd data dirs."""
    if not path or "trained" not in path:
        return path
    _safe_dir.mkdir(parents=True, exist_ok=True)
    st = Path(path).stat()
    key = hashlib.sha1(f"{Path(path).absolute()}{st.st_mtime_ns}".encode(), usedforsecurity=False).hexdigest()[:16]
    dst = _safe_dir / (key + ".index")
    if not dst.exists():
        shutil.copyfile(path, dst)
    return str(dst)


def op_convert(req):
    t0 = time.time()
    kit.gpu_fence(start=True)  # Applio loads HuBERT and the voice model on the first conversion
    index = req.get("index") or ""
    if index and not Path(index).is_file():
        index = ""
    vc.convert_audio(
        audio_input_path=req["in"],
        audio_output_path=req["out"],
        model_path=req["pth"],
        index_path=safe_index(index),
        index_rate=float(req.get("idx", 1.0)),
        protect=float(req.get("protect", 0.33)),
        f0_method=req.get("f0", "rmvpe"),
        pitch=int(req.get("pitch", 0)),
        export_format="WAV",
        embedder_model="contentvec",
    )
    if not Path(req["out"]).exists():
        raise kit.OpError("internal", "no output produced")
    return {"elapsed": round(time.time() - t0, 3)}


def op_release(_req):
    _real_cleanup()
    return {}


worker = kit.Worker(
    "rvc",
    INP,
    OUT,
    {"convert": op_convert, "release": op_release, "ping": lambda _req: {}},
    idle=[(kit.IDLE_EXIT_S, "exit")] if kit.IDLE_EXIT_S > 0 else [],
)
try:
    worker.run()
except BaseException as exc:
    worker.send({"event": "fatal", "error": f"{type(exc).__name__}: {exc}"})
    sys.stderr.flush()
    os._exit(1)
sys.stderr.flush()
os._exit(0)  # do not wait for CUDA teardown or the stdin reader thread
