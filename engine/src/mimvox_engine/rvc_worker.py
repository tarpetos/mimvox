"""Persistent RVC worker. Runs in the Applio virtualenv (separate torch), launched by ``rvc.py`` as
``python -I rvc_worker.py --applio <dir>``.

Standalone on purpose: it must not import anything from ``mimvox_engine`` (different interpreter); the shared
event-driven skeleton ``worker_kit.py`` is loaded by its file path. Wire format: see ``worker_kit.py``.

Ops:  convert {"in","out","pth","index","idx","protect","f0","pitch"} -> {"elapsed":s}
      release {}                                                       -> {} (drops the voice model and HuBERT)
      ping    {}                                                       -> {}
"""
import argparse
import hashlib
import importlib.util
import os
import shutil
import sys
import time


def _load_kit():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker_kit.py")
    spec = importlib.util.spec_from_file_location("mimvox_worker_kit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kit = _load_kit()
INP, OUT = kit.open_channels()  # before Applio's imports: they print a lot, also from C code

ap = argparse.ArgumentParser()
ap.add_argument("--applio", required=True)
args = ap.parse_args()

APPLIO = os.path.abspath(args.applio)
os.chdir(APPLIO)
sys.path.insert(0, APPLIO)

import faiss  # noqa: E402
from rvc.infer.infer import VoiceConverter  # noqa: E402

# Applio re-reads the FAISS index from disk on every call; cache it (searching is read-only).
_orig_read_index = faiss.read_index
_index_cache = {}


def _cached_read_index(path, *a, **kw):
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key not in _index_cache:
        _index_cache.clear()
        _index_cache[key] = _orig_read_index(path, *a, **kw)
    return _index_cache[key]


faiss.read_index = _cached_read_index

vc = VoiceConverter()
_real_cleanup = vc.cleanup_model
# convert_audio() unloads HuBERT and the voice model after every call; a persistent worker keeps them.
vc.cleanup_model = lambda: None

_safe_dir = os.path.join(APPLIO, "logs", "_mimvox_index")


def safe_index(path):
    """Applio rewrites 'trained' to 'added' anywhere in the index path; sidestep that for odd data dirs."""
    if not path or "trained" not in path:
        return path
    os.makedirs(_safe_dir, exist_ok=True)
    st = os.stat(path)
    key = hashlib.sha1(f"{os.path.abspath(path)}{st.st_mtime_ns}".encode()).hexdigest()[:16]
    dst = os.path.join(_safe_dir, key + ".index")
    if not os.path.exists(dst):
        shutil.copyfile(path, dst)
    return dst


def op_convert(req):
    t0 = time.time()
    index = req.get("index") or ""
    if index and not os.path.isfile(index):
        index = ""
    vc.convert_audio(
        audio_input_path=req["in"], audio_output_path=req["out"],
        model_path=req["pth"], index_path=safe_index(index),
        index_rate=float(req.get("idx", 1.0)), protect=float(req.get("protect", 0.33)),
        f0_method=req.get("f0", "rmvpe"), pitch=int(req.get("pitch", 0)),
        export_format="WAV", embedder_model="contentvec",
    )
    if not os.path.exists(req["out"]):
        raise kit.OpError("internal", "no output produced")
    return {"elapsed": round(time.time() - t0, 3)}


def op_release(_req):
    _real_cleanup()
    return {}


worker = kit.Worker("rvc", INP, OUT, {"convert": op_convert, "release": op_release, "ping": lambda _req: {}})
try:
    worker.run()
except BaseException as exc:  # noqa: BLE001
    worker.send({"event": "fatal", "error": f"{type(exc).__name__}: {exc}"})
    sys.stderr.flush()
    os._exit(1)
sys.stderr.flush()
os._exit(0)  # do not wait for CUDA teardown or the stdin reader thread
