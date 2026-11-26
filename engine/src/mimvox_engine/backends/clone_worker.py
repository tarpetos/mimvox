"""Shared by the voice-cloning workers (VoxCPM2, Chatterbox). Standalone: they load it by path in their own
interpreters, so it imports nothing from ``mimvox_engine``.

Ops (wire format: ``worker_kit.py``):
  load     {}                                                  -> {"load_s","device","vram_gb"}
  synth    {"text","lang","ref","out","ref_text"?,"start"?,"end"?,"quality"?,"expressiveness"?}
                                                               -> {"out","sr","duration_s","gen_s"}
           ``start`` / ``end``: the stretch of ``ref`` that ``ref_text`` says (seconds).
  release  {}                                                  -> {}
  status   {}                                                  -> {"loaded","vram_gb"}
"""

import argparse
import contextlib
import importlib.util
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kit = load_by_path("mimvox_worker_kit", HERE.parent / "worker_kit.py")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return args


def run(name, inp, out, args, load, speak):
    """``load(model_dir, device)`` -> model; ``speak(model, text, lang, prompt, req)`` -> (float32 audio, sample rate),
    with ``prompt`` = {"wav", "text"?}."""
    import numpy as np
    import soundfile as sf
    import torch

    text_norm = load_by_path("mimvox_omnivoice_text", HERE / "omnivoice_text.py")
    state = {"model": None, "device": None}

    def log(msg):
        print(f"[{name}] {msg}", file=sys.stderr, flush=True)

    def vram_gb():
        if state["device"] != "cpu" and torch.cuda.is_available():  # asking CUDA makes a context, even for the CPU
            free, total = torch.cuda.mem_get_info()
            return round((total - free) / 2**30, 2)
        return 0.0

    def device():
        want = args.device
        if want == "cuda" and not torch.cuda.is_available():
            want = "cpu"
        if want == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            want = "cpu"
        return want

    def op_load(_req=None):
        if state["model"] is not None:
            return {"load_s": 0.0, "device": state["device"], "vram_gb": vram_gb()}
        t0 = time.time()
        dev = device()
        state["model"], state["device"] = load(args.model_dir, dev), dev
        log(f"loaded on {dev} in {time.time() - t0:.1f}s, VRAM {vram_gb()} GB")
        return {"load_s": round(time.time() - t0, 2), "device": dev, "vram_gb": vram_gb()}

    def op_release(_req=None):
        state["model"] = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {}

    def prompt_clip(req, out_path):
        ref, start, end = req["ref"], req.get("start"), req.get("end")
        if start is None or end is None:
            return None
        info = sf.info(ref)
        a, b = int(float(start) * info.samplerate), int(float(end) * info.samplerate)
        x, sr = sf.read(ref, start=a, stop=b, dtype="float32", always_2d=True)
        clip = out_path.with_name(out_path.stem + "_prompt.wav")
        sf.write(clip, x.mean(axis=1), sr, subtype="PCM_16")
        return clip

    def op_synth(req):
        text, out_path, ref = req.get("text"), req.get("out"), req.get("ref")
        if not isinstance(text, str) or not text.strip() or not out_path or not ref:
            raise kit.OpError("bad_request", "synth needs 'text', 'ref' and 'out'")
        if not Path(ref).is_file():
            raise kit.OpError("not_found", f"reference clip not found: {ref}")
        op_load()
        lang = str(req.get("lang") or "en").lower()
        Path(out_path).absolute().parent.mkdir(parents=True, exist_ok=True)
        clip = prompt_clip(req, Path(out_path))
        t0 = time.time()
        try:
            with torch.inference_mode():
                audio, sr = speak(
                    state["model"],
                    text_norm.normalize(text, lang),
                    lang,
                    {"wav": str(clip) if clip else ref, "text": req.get("ref_text")},
                    req,
                )
        finally:
            if clip is not None:
                with contextlib.suppress(OSError):
                    clip.unlink()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise kit.OpError("internal", "the model returned no audio")
        peak = float(np.abs(audio).max())
        if peak > 0.99:
            audio = audio * (0.99 / peak)
        sf.write(out_path, audio, sr, subtype="PCM_16")
        return {"out": out_path, "sr": sr, "duration_s": round(len(audio) / sr, 3), "gen_s": round(time.time() - t0, 3)}

    def op_status(_req=None):
        return {"loaded": state["model"] is not None, "vram_gb": vram_gb()}

    worker = kit.Worker(
        name,
        inp,
        out,
        {"load": op_load, "synth": op_synth, "release": op_release, "status": op_status, "ping": lambda _r: {}},
    )
    worker.run()
