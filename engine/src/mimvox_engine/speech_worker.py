"""Persistent speech worker: Whisper (transcription), NLLB (translation) and Hybrid Demucs (vocal separation) in one
child process of the engine.

Started lazily by ``speech.py`` as ``python -I speech_worker.py --models-root <data>/runtime/models`` with the engine
interpreter (the engine venv). Standalone on purpose: it imports nothing from ``mimvox_engine``.

Why a process: the OmniVoice worker and the RVC worker already share the 12 GB card; a separate
process can be told to hand its GPU memory back, exits on its own when idle, and a native crash cannot take the
engine down. Models load on demand, stay warm, and are dropped after ``UNLOAD_S`` of idleness (the process itself
exits after ``EXIT_S``).

Wire format and the event-driven skeleton (stdin reader thread -> loop -> single op thread, idle timers): see
``worker_kit.py``, which this script loads by its file path.

Ops (all params optional unless stated):
  load_asr    {"model":"turbo","allow_cuda":true}                              -> {"device","half","model","load_s"}
  transcribe  {"audio":"<.npy 16 kHz mono float32>"(required),"language":null|"uk","model":"turbo","allow_cuda":true}
                                                                               -> {"language","device","segments":[{start,end,text}]}
  translate   {"texts":[...](required),"src":"ukr_Cyrl"(required),"tgt":"eng_Latn","allow_cuda":true,"num_beams":4}
                                                                               -> {"texts":[...],"device","untranslated":[i]}
  separate_vocals {"audio":"<.npy 44.1 kHz mono float32>"(required),"out":"<.npy to write>"(required),
                   "allow_cuda":true}                                          -> {"device","seconds","elapsed_s",
                                                                                   "peak_vram_gb"} (vocals in "out")
  release_gpu / unload                                                         -> {"free_gb"}
  status                                                                       -> {"asr","mt","sep","rss_gb","peak_rss_gb",
                                                                                   "free_gb"}
Progress replies: {"id":N,"progress":{"stage":"load_model|transcribe|translate|separate","pct":0-100}}.
Events: {"event":"ready"}, {"event":"unloaded","reason":"idle"} (models dropped after UNLOAD_S idle),
{"event":"exit","reason":"idle"} (the process exits after EXIT_S idle).
"""
import argparse
import gc
import importlib.util
import os
import re
import sys
import time
import types


def _load_kit():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker_kit.py")
    spec = importlib.util.spec_from_file_location("mimvox_worker_kit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kit = _load_kit()
INP, OUT = kit.open_channels()  # protocol on private descriptors; library prints go to stderr

ap = argparse.ArgumentParser()
ap.add_argument("--models-root", required=True)
args = ap.parse_args()

UNLOAD_S = float(os.environ.get("MIMVOX_SPEECH_UNLOAD_S", "180"))   # models are dropped after this much idleness
EXIT_S = float(os.environ.get("MIMVOX_SPEECH_EXIT_S", "600"))       # the process exits after this much idleness
ASR_NEED_HALF_GB = float(os.environ.get("MIMVOX_WHISPER_NEED_GB", "2.6"))   # fp16 weights (1.6 GB) + decoding activations
ASR_NEED_FP32_GB = 4.0
MT_NEED_GB = float(os.environ.get("MIMVOX_NLLB_NEED_GB", "2.0"))            # fp16 weights (1.2 GB) + beam search
NLLB_DIR = os.path.join(args.models_root, "nllb-200-distilled-600M")
WHISPER_DIR = os.path.join(args.models_root, "whisper")
# Vocal separation: torchaudio's HDEMUCS_HIGH_MUSDB_PLUS weights (setup downloads them; never fetched from here).
SEP_PATH = os.path.join(args.models_root, "demucs", "hdemucs_high_trained.pt")
SEP_SOURCES = ("drums", "bass", "other", "vocals")
SEP_SR = 44100
SEP_NEED_GB = float(os.environ.get("MIMVOX_DEMUCS_NEED_GB", "1.2"))   # 0.76 GB peak measured (fp32 weights + one 10 s
                                                                      # segment) plus the CUDA context
SEP_SEGMENT_S = 10.0     # audio per forward pass: bounds the memory whatever the file length
SEP_OVERLAP = 0.25       # neighbouring segments overlap by this share and are cross-faded (as Demucs' apply does)
SPLIT_OVER_CHARS = 160   # a longer text is translated sentence by sentence (NLLB drops content of long inputs)
MERGE_MIN_CHARS = 50     # ... but a sentence shorter than this is joined to the next one

Cancelled = kit.Cancelled
OpError = kit.OpError
WORKER = None  # the kit.Worker, created at the bottom


def log(msg):
    print(f"[speech] {msg}", file=sys.stderr, flush=True)


def check_cancel():
    WORKER.check_cancel()


def progress(stage, pct):
    WORKER.progress({"stage": stage, "pct": round(float(pct), 1)})


# ------------------------------------------------------------------------------------------------ memory helpers

def _torch():
    import torch
    return torch


def free_vram_gb():
    try:
        torch = _torch()
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / 2**30
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def rss_gb():
    """(working set, peak working set) of this process in GB; (0, 0) when unreadable."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class C(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            c = C()
            c.cb = ctypes.sizeof(c)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
            if ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return c.WorkingSetSize / 2**30, c.PeakWorkingSetSize / 2**30
            return 0.0, 0.0
        with open("/proc/self/status") as f:
            vals = {ln.split(":")[0]: int(ln.split()[1]) for ln in f if ln.startswith(("VmRSS", "VmHWM"))}
        return vals.get("VmRSS", 0) / 2**20, vals.get("VmHWM", 0) / 2**20
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def trim_working_set():
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        ctypes.windll.psapi.EmptyWorkingSet(k32.GetCurrentProcess())
    except Exception:  # noqa: BLE001
        pass


def collect():
    gc.collect()
    gc.collect()
    if "torch" in sys.modules:
        try:
            torch = sys.modules["torch"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    trim_working_set()


# ------------------------------------------------------------------------------------------------ state

class Slot:
    """A loaded model: ``obj`` (and ``aux``), the ``device`` it lives on and a ``key`` naming what it is."""

    def __init__(self, key, obj, device, aux=None, half=False):
        self.key, self.obj, self.device, self.aux, self.half = key, obj, device, aux, half


ASR = [None]   # Slot of Whisper
MT = [None]    # Slot of NLLB
SEP = [None]   # Slot of Hybrid Demucs
_half_broken = [False]   # the fp16 Whisper loader failed once: use the stock loader from then on


def unload(which):
    slot = which[0]
    which[0] = None
    if slot is not None:
        slot.obj = slot.aux = None
    del slot
    collect()


def unload_all(reason=""):
    had = ASR[0] is not None or MT[0] is not None or SEP[0] is not None
    unload(ASR)
    unload(MT)
    unload(SEP)
    collect()
    if had:
        log(f"models unloaded ({reason or 'requested'}); {free_vram_gb():.1f} GB VRAM free")
    return had


def pick_device(need_gb, allow_cuda, *others):
    """cuda when ``need_gb`` of VRAM is free (dropping the other models from the GPU if that makes it fit), else cpu
    (dropping the other models too when they also sit in host RAM, so the process stays small)."""
    torch = _torch()
    if allow_cuda and torch.cuda.is_available():
        free = free_vram_gb()
        if free >= need_gb:
            return "cuda"
        for other in others:
            if other[0] is not None and other[0].device == "cuda":
                unload(other)
                free = free_vram_gb()
                if free >= need_gb:
                    return "cuda"
        log(f"only {free:.1f} GB of VRAM free (need {need_gb:.1f}): using the CPU")
    for other in others:
        if other[0] is not None and other[0].device == "cpu":
            unload(other)
    return "cpu"


# ------------------------------------------------------------------------------------------------ Whisper

class Bar:
    """Stands in for tqdm inside whisper: reports progress and honours cancel."""
    stage = "load_model"

    def __init__(self, total=None, **_):
        self.total, self.n = max(int(total or 1), 1), 0

    def update(self, n=1):
        check_cancel()
        self.n += n
        progress(Bar.stage, 100.0 * min(self.n / self.total, 1.0))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def _patch_whisper(whisper):
    import inspect
    tr_mod = sys.modules["whisper.transcribe"]  # (the package attribute `whisper.transcribe` is the function)
    ns = types.SimpleNamespace(tqdm=Bar)
    tr_mod.tqdm = ns
    whisper.tqdm = ns if inspect.ismodule(getattr(whisper, "tqdm", None)) else Bar


def _load_whisper_half(whisper, name, device):
    """fp16 weights built directly on the GPU (1.6 GB instead of 3.2 GB). LayerNorm stays fp32 (whisper casts around it)."""
    torch = _torch()
    path = whisper._download(whisper._MODELS[name], WHISPER_DIR, False)
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


def ensure_asr(name, allow_cuda):
    """Load Whisper ``name`` if it is not loaded yet; returns the Slot."""
    slot = ASR[0]
    if slot is not None and slot.key == name:
        progress("load_model", 100)
        return slot
    unload(ASR)
    torch = _torch()
    import whisper
    _patch_whisper(whisper)
    Bar.stage = "load_model"
    os.makedirs(WHISPER_DIR, exist_ok=True)
    progress("load_model", 0)
    t0 = time.time()
    model, device, half = None, "cpu", False
    if allow_cuda and torch.cuda.is_available() and not _half_broken[0]:
        device = pick_device(ASR_NEED_HALF_GB, True, MT, SEP)
        if device == "cuda":
            try:
                model, half = _load_whisper_half(whisper, name, "cuda"), True
            except Exception as exc:  # noqa: BLE001
                log(f"fp16 loader failed ({type(exc).__name__}: {exc}); using the stock loader")
                _half_broken[0] = True
                model = None
                collect()
    if model is None:
        device = pick_device(ASR_NEED_FP32_GB, allow_cuda and torch.cuda.is_available(), MT, SEP)
        if device == "cuda":
            with torch.device("cuda"):
                model = whisper.load_model(name, device="cuda", download_root=WHISPER_DIR)
        else:
            model = whisper.load_model(name, device="cpu", download_root=WHISPER_DIR)
    collect()
    log(f"whisper {name} loaded on {device}{' (fp16 weights)' if half else ''} in {time.time() - t0:.1f}s; "
        f"{free_vram_gb():.1f} GB VRAM free")
    progress("load_model", 100)
    ASR[0] = Slot(name, model, device, half=half)
    return ASR[0]


def op_load_asr(req):
    check_cancel()
    t0 = time.time()
    slot = ensure_asr(req.get("model", "turbo"), req.get("allow_cuda", True))
    return {"device": slot.device, "half": slot.half, "model": slot.key, "load_s": round(time.time() - t0, 2)}


def op_transcribe(req):
    import numpy as np
    import whisper
    audio = np.load(req["audio"])
    name = req.get("model", "turbo")
    allow = req.get("allow_cuda", True)
    for attempt in (1, 2):
        check_cancel()
        slot = ensure_asr(name, allow)
        Bar.stage = "transcribe"
        progress("transcribe", 0)
        t1 = time.time()
        try:
            res = whisper.transcribe(slot.obj, audio, language=req.get("language"), task="transcribe",
                                     fp16=(slot.device == "cuda"), verbose=False, condition_on_previous_text=False)
            break
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            if attempt == 1 and slot.half:  # the fp16 weights did not work here: reload the stock way and retry
                log(f"fp16 transcription failed ({type(exc).__name__}: {exc}); reloading with the stock loader")
                _half_broken[0] = True
                unload(ASR)
                continue
            raise
    log(f"whisper transcribed {len(audio) / 16000:.1f}s of audio in {time.time() - t1:.1f}s on {slot.device}")
    progress("transcribe", 100)
    return {"language": res.get("language"), "device": slot.device,
            "segments": [{"start": float(s["start"]), "end": float(s["end"]), "text": str(s["text"])}
                         for s in res.get("segments", [])]}


# ------------------------------------------------------------------------------------------------ NLLB

def _transformers_major():
    import transformers
    try:
        return int(transformers.__version__.split(".")[0])
    except ValueError:
        return 4


def ensure_mt(allow_cuda, force_cpu=False):
    slot = MT[0]
    if slot is not None and (not force_cpu or slot.device == "cpu"):
        return slot
    unload(MT)
    need = ("config.json", "pytorch_model.bin", "sentencepiece.bpe.model", "tokenizer_config.json")
    missing = [n for n in need if not os.path.isfile(os.path.join(NLLB_DIR, n))
               and not (n == "pytorch_model.bin" and os.path.isfile(os.path.join(NLLB_DIR, "model.safetensors")))]
    if missing:
        raise OpError("model_missing", f"the translation model is not installed ({', '.join(missing)} missing in "
                                       f"{NLLB_DIR}); run setup to download it")
    torch = _torch()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    progress("translate", 0)
    t0 = time.time()
    device = "cpu" if force_cpu else pick_device(MT_NEED_GB, allow_cuda, ASR, SEP)
    tok = AutoTokenizer.from_pretrained(NLLB_DIR, local_files_only=True)
    kw = {"local_files_only": True, "low_cpu_mem_usage": True}
    kw["dtype" if _transformers_major() >= 5 else "torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_DIR, **kw)
    model = model.to(device).eval()
    collect()
    log(f"nllb-200-distilled-600M loaded on {device} in {time.time() - t0:.1f}s; {free_vram_gb():.1f} GB VRAM free")
    MT[0] = Slot("nllb", model, device, aux=tok)
    return MT[0]


_SENT = re.compile(r"(?<=[.!?…])\s+")


def split_pieces(text):
    """One text -> pieces translated separately and joined with a space (short texts stay whole)."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= SPLIT_OVER_CHARS:
        return [text]
    out, buf = [], ""
    for sent in _SENT.split(text):
        buf = f"{buf} {sent}".strip()
        if len(buf) >= MERGE_MIN_CHARS:
            out.append(buf)
            buf = ""
    if buf:
        if out and len(buf) < 20:
            out[-1] = f"{out[-1]} {buf}"
        else:
            out.append(buf)
    return out or [text]


def _translate_batch(slot, batch, src, bos, beams):
    torch = _torch()
    tok, model = slot.aux, slot.obj
    tok.src_lang = src
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=400).to(slot.device)
    n_in = int(enc["input_ids"].shape[1])
    with torch.inference_mode():
        gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=beams, do_sample=False,
                             max_new_tokens=min(512, int(n_in * 2.5) + 24))
    return [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]


def _translate_all(slot, pieces, src, tgt, beams):
    bos = slot.aux.convert_tokens_to_ids(tgt)
    if bos is None or bos == slot.aux.unk_token_id:
        raise OpError("bad_request", f"unknown target language code {tgt!r}")
    order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
    size = 8 if slot.device == "cuda" else 4
    result = [""] * len(pieces)
    done = 0
    k = 0
    while k < len(order):
        check_cancel()
        idx = order[k:k + size]
        try:
            outs = _translate_batch(slot, [pieces[i] for i in idx], src, bos, beams)
        except _torch().cuda.OutOfMemoryError:
            if size == 1:
                raise
            collect()
            size = max(1, size // 2)
            continue
        for i, t in zip(idx, outs):
            result[i] = t
        k += len(idx)
        done += len(idx)
        progress("translate", 100.0 * done / len(order))
    return result


def op_translate(req):
    texts = req["texts"]
    src, tgt = req["src"], req.get("tgt", "eng_Latn")
    beams = int(req.get("num_beams", 4))
    allow = req.get("allow_cuda", True)
    owners, pieces = [], []
    for i, t in enumerate(texts):
        for p in split_pieces(str(t)):
            if p:
                owners.append(i)
                pieces.append(p)
    force_cpu = False
    for attempt in (1, 2):
        slot = ensure_mt(allow, force_cpu)
        progress("translate", 0)
        t1 = time.time()
        try:
            out = _translate_all(slot, pieces, src, tgt, beams)
            break
        except _torch().cuda.OutOfMemoryError:
            if attempt == 2:
                raise
            log("out of VRAM while translating: retrying on the CPU")
            unload(MT)
            force_cpu = True
    log(f"translated {len(pieces)} piece(s) of {len(texts)} text(s) on {slot.device} in {time.time() - t1:.1f}s")
    joined = [[] for _ in texts]
    for o, t in zip(owners, out):
        if t:
            joined[o].append(t)
    result, untranslated = [], []
    for i, parts in enumerate(joined):
        text = " ".join(parts).strip()
        if not text and str(texts[i]).strip():
            untranslated.append(i)
            text = str(texts[i]).strip()  # never drop a segment: keep the source text
        result.append(text)
    return {"texts": result, "device": slot.device, "untranslated": untranslated}


# ------------------------------------------------------------------------------------------------ Hybrid Demucs

def ensure_sep(allow_cuda, force_cpu=False):
    """Load the separation model (torchaudio's ``hdemucs_high`` with the HDEMUCS_HIGH_MUSDB_PLUS weights from
    ``SEP_PATH``, fp32) if it is not loaded yet; returns the Slot."""
    slot = SEP[0]
    if slot is not None and (not force_cpu or slot.device == "cpu"):
        progress("load_model", 100)
        return slot
    unload(SEP)
    if not os.path.isfile(SEP_PATH):
        raise OpError("model_missing", f"the music-removal model is not installed ({SEP_PATH} is missing); "
                                       "run setup again to download it")
    torch = _torch()
    from torchaudio.models import hdemucs_high
    progress("load_model", 0)
    t0 = time.time()
    device = "cpu" if force_cpu else pick_device(SEP_NEED_GB, allow_cuda, ASR, MT)
    with open(SEP_PATH, "rb") as fh:
        state = torch.load(fh, map_location="cpu", weights_only=True)
    model = hdemucs_high(sources=list(SEP_SOURCES))
    model.load_state_dict(state)
    del state
    model = model.to(device).eval()
    collect()
    log(f"hdemucs_high loaded on {device} in {time.time() - t0:.1f}s; {free_vram_gb():.1f} GB VRAM free")
    progress("load_model", 100)
    SEP[0] = Slot("hdemucs", model, device)
    return SEP[0]


def _separate(slot, mix):
    """Vocals of mono ``mix`` (float32 at SEP_SR) as a float32 numpy array of the same length.

    The model takes stereo: the mono signal is fed on both channels and the two vocal channels are averaged. The
    signal is normalised once for the whole file and cut into SEP_SEGMENT_S segments that overlap by SEP_OVERLAP;
    each segment's output is weighted with a triangle and the overlaps are summed and divided by the summed weights
    (Demucs' ``apply_model`` with ``segment`` / ``overlap``), so memory stays bounded for any length."""
    import math
    torch = _torch()
    model, dev = slot.obj, slot.device
    vocals = SEP_SOURCES.index("vocals")
    x = torch.from_numpy(mix)
    mean, std = float(x.mean()), float(x.std())
    std = std if std > 1e-8 else 1.0
    x = (x - mean) / std
    n = int(x.shape[0])
    seg = int(SEP_SEGMENT_S * SEP_SR)
    stride = max(int(seg * (1.0 - SEP_OVERLAP)), 1)
    half = seg // 2
    weight = torch.cat([torch.arange(1, half + 1), torch.arange(seg - half, 0, -1)]).float()
    weight /= weight.max()
    starts = [i * stride for i in range(max(math.ceil((n - seg) / stride), 0) + 1)]
    out = torch.zeros(n)
    total = torch.zeros(n)
    progress("separate", 0)
    with torch.inference_mode():
        for i, s in enumerate(starts):
            check_cancel()
            piece = x[s:s + seg]
            length = int(piece.shape[0])
            if length < seg:
                piece = torch.nn.functional.pad(piece, (0, seg - length))
            inp = piece.to(dev).view(1, 1, seg).expand(1, 2, seg).contiguous()
            voc = model(inp)[0, vocals].mean(0)[:length].float().cpu()
            del inp
            w = weight[:length]
            out[s:s + length] += voc * w
            total[s:s + length] += w
            progress("separate", 100.0 * (i + 1) / len(starts))
    y = out / total.clamp_min(1e-8) * std
    return y.numpy().astype("float32", copy=False)


def op_separate_vocals(req):
    import numpy as np
    torch = _torch()
    audio = np.load(req["audio"])
    if audio.ndim != 1:
        raise OpError("bad_request", "'audio' must hold mono samples")
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    allow = req.get("allow_cuda", True)
    force_cpu = False
    for attempt in (1, 2):
        check_cancel()
        slot = ensure_sep(allow, force_cpu)
        if slot.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        try:
            y = _separate(slot, audio)
            break
        except torch.cuda.OutOfMemoryError:
            if attempt == 2 or slot.device != "cuda":
                raise
            log("out of VRAM while separating: retrying on the CPU")
            unload(SEP)
            force_cpu = True
    elapsed = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2**30 if slot.device == "cuda" else 0.0
    np.save(req["out"], y)
    log(f"separated the vocals of {len(audio) / SEP_SR:.1f}s of audio in {elapsed:.1f}s on {slot.device}"
        + (f" (peak {peak:.2f} GB VRAM)" if peak else ""))
    progress("separate", 100)
    return {"device": slot.device, "seconds": round(len(audio) / SEP_SR, 2), "elapsed_s": round(elapsed, 2),
            "peak_vram_gb": round(peak, 2)}


# ------------------------------------------------------------------------------------------------ main

def op_status(_req):
    cur, peak = rss_gb()
    return {"asr": None if ASR[0] is None else {"model": ASR[0].key, "device": ASR[0].device, "half": ASR[0].half},
            "mt": None if MT[0] is None else {"device": MT[0].device},
            "sep": None if SEP[0] is None else {"device": SEP[0].device},
            "rss_gb": round(cur, 2), "peak_rss_gb": round(peak, 2), "free_gb": round(free_vram_gb(), 2)}


def op_release(_req):
    unload_all("released for the GPU")
    return {"free_gb": round(free_vram_gb(), 2)}


OPS = {"load_asr": op_load_asr, "transcribe": op_transcribe, "translate": op_translate,
       "separate_vocals": op_separate_vocals, "status": op_status, "release_gpu": op_release, "unload": op_release}


def on_idle_unload():
    if unload_all("idle"):
        return {"event": "unloaded", "reason": "idle"}
    return None


WORKER = kit.Worker("speech", INP, OUT, OPS, idle=[(UNLOAD_S, on_idle_unload), (EXIT_S, "exit")])
try:
    WORKER.run()
except BaseException as exc:  # noqa: BLE001
    WORKER.send({"event": "fatal", "error": f"{type(exc).__name__}: {exc}"})
    sys.stderr.flush()
    os._exit(1)
sys.stderr.flush()
os._exit(0)  # do not wait for CUDA teardown or the stdin reader thread
