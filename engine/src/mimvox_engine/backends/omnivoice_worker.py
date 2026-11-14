"""Persistent OmniVoice worker (k2-fsa/OmniVoice, zero-shot TTS in 600+ languages): the engine's speech model.

Runs with the engine venv's interpreter (``<data>/runtime/engine/.venv``: torch 2.8, transformers 5, omnivoice,
openai-whisper, num2words), started by ``backends/omnivoice.py`` as

    python -I omnivoice_worker.py --model-dir <data>/runtime/models/omnivoice --whisper-dir <data>/runtime/models/whisper
                                  [--device cuda|cpu|mps] [--unload-after S] [--exit-after S] [--prompt-cache N]

Standalone on purpose: it imports nothing from ``mimvox_engine``; the event-driven skeleton ``worker_kit.py`` and
the text normalizer ``omnivoice_text.py`` are loaded by their file paths. Wire format: see ``worker_kit.py``.

Ops (one at a time, on the op thread):
  load     {}                                         -> {"load_s","device","vram_gb"}   (no-op when loaded)
  synth    {"text","lang","ref","out", "steps":32, "speed":1.0, "duration":null, "normalize":true, "ref_text":null,
            "model_lang", "position_temperature", "class_temperature", "guidance_scale" (optional sampling knobs)}
                                                      -> {"out","sr","duration_s","gen_s","text"}
           ``ref``: a WAV of the voice. A clip longer than REF_MAX_S (12 s) is cut to a 7-12 s run of whole sentences
           (Whisper turbo finds them once; the cut, its transcript and Whisper's detected language are cached next
           to the clip in ``<clip>.omnivoice.json``). ``ref_text``, the transcript of a clip of at most REF_MAX_S, skips Whisper and
           the sidecar (the shipped native-accent clips, whose folder may be read-only). The voice-clone prompts of
           the last ``--prompt-cache`` references (default 3: the user's clip and the native ones) stay in memory.
  warm     {"ref","ref_text"?}                        -> {"elapsed_s"}  (builds the voice-clone prompt only)
  inspect  {"ref","detect"?:true}                     -> {"language","text","elapsed_s"}  (the clip's sidecar, built
           when missing: Whisper only, the TTS model is not loaded; with ``detect`` a sidecar without ``language``
           gets Whisper's language detection added)
  release  {}                                         -> {}  (drops the models and frees the GPU; ``load`` or the next
                                                              ``synth`` loads them again)
  status   {}                                         -> {"loaded","rss_gb","vram_gb"}
Events: {"event":"ready"}, {"event":"unloaded","reason":"idle"}, {"event":"exit","reason":"idle"}.
"""
import argparse
import collections
import gc
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kit = _load_by_path("mimvox_worker_kit", os.path.join(os.path.dirname(HERE), "worker_kit.py"))
INP, OUT = kit.open_channels()  # before torch / transformers: they print, also from C code

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--whisper-dir", default="")
ap.add_argument("--device", default="cuda")
ap.add_argument("--unload-after", type=float, default=0.0)  # 0: keep the model while the process lives
ap.add_argument("--exit-after", type=float, default=0.0)    # 0: never exit on idleness
ap.add_argument("--prompt-cache", type=int, default=3)      # voice-clone prompts kept in memory (least recently used out)
args = ap.parse_args()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

text_norm = _load_by_path("mimvox_omnivoice_text", os.path.join(HERE, "omnivoice_text.py"))
whisper_half = _load_by_path("mimvox_whisper_half", os.path.join(os.path.dirname(HERE), "whisper_half.py"))

# Reference length: OmniVoice's docs suggest 3-10 s. On a test voice the whole 19 s clip cloned a little better
# than a 7 s cut (ECAPA similarity after RVC 0.86 vs 0.84) but needs ~3 GB more VRAM (peak 6.9 vs 3.9 GB), and next to
# the RVC worker that filled the 12 GB card (11.6 GB in use, segments 2-3x slower). Clips up to REF_MAX_S are used
# whole, longer ones are cut to whole sentences around REF_CUT_TARGET_S.
REF_MAX_S = float(os.environ.get("MIMVOX_OMNIVOICE_REF_MAX_S", "12"))
REF_CUT_MIN_S, REF_CUT_MAX_S, REF_CUT_TARGET_S = 7.0, REF_MAX_S, 10.0
# Whisper turbo for a new reference clip: on the GPU with fp16 weights (1.6 GB + decoding) only when that fits beyond
# the desktop's reserve next to everything else on the card, else on the CPU (about 10-20 s for a clip, once per clip).
WHISPER_NEED_GB = float(os.environ.get("MIMVOX_WHISPER_NEED_GB", "2.6"))
SIDECAR_VERSION = 3

_state = {"model": None, "device": None}
_prompts = collections.OrderedDict()  # (sig, ref_text) -> voice-clone prompt, most recently used last


def _log(msg):
    print(f"[omnivoice] {msg}", file=sys.stderr, flush=True)


def _vram_gb():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return round((total - free) / 2**30, 2)
    return 0.0


def _rss_gb():
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class _C(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t)] + \
                           [(f"f{i}", ctypes.c_size_t) for i in range(6)]

            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            c = _C()
            c.cb = ctypes.sizeof(c)
            psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return round(c.WorkingSetSize / 2**30, 2)
        import resource
        with open("/proc/self/statm") as f:
            return round(int(f.read().split()[1]) * resource.getpagesize() / 2**30, 2)
    except Exception:  # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------------------------------------- model

def _device():
    want = args.device
    if want == "cuda" and not torch.cuda.is_available():
        want = "cpu"
    if want == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        want = "cpu"
    return want


def op_load(_req=None):
    if _state["model"] is not None:
        return {"load_s": 0.0, "device": _state["device"], "vram_gb": _vram_gb()}
    from omnivoice import OmniVoice

    t0 = time.time()
    dev = _device()
    if dev == "cuda":
        kit.gpu_fence(start=True)
    dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
    model = OmniVoice.from_pretrained(args.model_dir, device_map=dev if dev != "cuda" else "cuda:0", dtype=dtype)
    _state.update(model=model, device=dev)
    _prompts.clear()
    gc.collect()
    _trim_working_set()  # the weights passed through host RAM on their way to the GPU
    _log(f"loaded on {dev} in {time.time() - t0:.1f}s, VRAM in use {_vram_gb()} GB")
    return {"load_s": round(time.time() - t0, 2), "device": dev, "vram_gb": _vram_gb()}


def _trim_working_set():
    if os.name == "nt":
        try:
            import ctypes

            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            ctypes.windll.psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
            ctypes.windll.psapi.EmptyWorkingSet(k32.GetCurrentProcess())
        except Exception:  # noqa: BLE001
            pass


def op_release(_req=None):
    _state.update(model=None)
    _prompts.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _trim_working_set()
    return {}


def _idle_unload():
    if _state["model"] is None:
        return None
    op_release()
    return {"event": "unloaded", "reason": "idle"}


# ------------------------------------------------------------------------------------------------ reference

def _sig(path):
    st = os.stat(path)
    return [os.path.abspath(path), st.st_mtime_ns, st.st_size]


def _load_mono(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(1), sr


def _whisper_device():
    if (_state["device"] or _device()) != "cuda":  # `inspect` runs without the TTS model
        return "cpu"
    free = kit.gpu_free_gb()
    if free >= WHISPER_NEED_GB:
        kit.gpu_fence(start=True)
        return "cuda"
    _log(f"only {free:.1f} GB of VRAM free beyond the {kit.VRAM_RESERVE_GB:.1f} GB reserve (Whisper needs "
         f"{WHISPER_NEED_GB:.1f}): transcribing the reference on the CPU")
    return "cpu"


def _with_whisper(use):
    """``use(model, device)`` with Whisper turbo loaded for this call only (fp16 on the GPU when it fits, else the CPU),
    retried on the CPU when the GPU runs out of memory midway; the model is dropped afterwards either way."""
    import whisper

    dev = _whisper_device()
    for attempt in (dev, "cpu") if dev == "cuda" else ("cpu",):
        wm = None
        try:
            if attempt == "cuda":
                wm = whisper_half.load(whisper, "turbo", args.whisper_dir or None, "cuda")
            else:
                wm = whisper.load_model("turbo", download_root=args.whisper_dir or None, device="cpu")
            return use(wm, attempt)
        except Exception as exc:
            if attempt != "cuda" or not kit.is_vram_error(exc):
                raise
            _log(f"Whisper ran out of GPU memory ({exc}); transcribing the reference on the CPU")
        finally:
            del wm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _trim_working_set()  # Whisper's weights passed through host RAM: give those pages back now
    raise AssertionError("unreachable")


def _16k(x, sr):
    import torchaudio

    x16 = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy() if sr != 16000 else x
    return x16.astype(np.float32)


def _transcribe(x, sr):
    """Whisper turbo on a clip, language auto-detected (not forced): ([(start, end, text)], detected language code or
    None)."""
    x16 = _16k(x, sr)
    res = _with_whisper(lambda wm, dev: wm.transcribe(x16, fp16=dev == "cuda", condition_on_previous_text=False))
    segs = [(float(s["start"]), float(s["end"]), s["text"].strip()) for s in res["segments"] if s["text"].strip()]
    return segs, (str(res.get("language") or "").lower() or None)


def _detect_language(x, sr):
    """Whisper turbo's language detection alone (the first 30 s of a clip): an ISO code or None."""
    import whisper

    x16 = _16k(x, sr)

    def detect(wm, _dev):
        audio = whisper.pad_or_trim(torch.from_numpy(x16))
        mel = whisper.log_mel_spectrogram(audio, n_mels=wm.dims.n_mels).to(wm.device)
        if next(wm.parameters()).dtype == torch.float16:
            mel = mel.half()
        _, probs = wm.detect_language(mel)
        return max(probs, key=probs.get) if probs else None

    return _with_whisper(detect)


def _pick_window(segs, total):
    """The run of whole Whisper segments closest to REF_CUT_TARGET_S within [MIN, MAX]; statements (ending with
    '.') are preferred over questions and exclamations, which make every generated sentence sound like one."""
    best, best_score = None, None
    for i in range(len(segs)):
        for j in range(i, len(segs)):
            d = segs[j][1] - segs[i][0]
            if d > REF_CUT_MAX_S:
                break
            if d < REF_CUT_MIN_S:
                continue
            questions = sum(1 for s in segs[i:j + 1] if s[2].rstrip().endswith(("?", "!")))
            score = abs(d - REF_CUT_TARGET_S) + 3.0 * questions
            if best_score is None or score < best_score:
                best, best_score = (i, j), score
    if best is None:  # no fitting run: the first REF_CUT_MAX_S seconds with whatever was said in them
        keep = [s for s in segs if s[1] <= REF_CUT_MAX_S] or segs[:1]
        return 0.0, min(total, keep[-1][1] if keep else REF_CUT_MAX_S), " ".join(s[2] for s in keep)
    i, j = best
    return max(0.0, segs[i][0] - 0.05), min(total, segs[j][1] + 0.15), " ".join(s[2] for s in segs[i:j + 1])


def _reference(ref, ref_text=None):
    """(waveform float32 mono, sample rate, transcript) for the voice-clone prompt: a clip of at most REF_MAX_S with
    a known ``ref_text`` as it is, else via the sidecar cache."""
    sig = _sig(ref)
    side = os.path.splitext(ref)[0] + ".omnivoice.json"
    x, sr = _load_mono(ref)
    total = len(x) / sr
    if ref_text and total <= REF_MAX_S:
        return x, sr, ref_text
    cached = None
    try:
        with open(side, encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("version") != SIDECAR_VERSION or cached.get("sig") != sig[1:]:
            cached = None
    except (OSError, ValueError):
        cached = None
    if cached is None:
        t0 = time.time()
        segs, language = _transcribe(x, sr)
        if not segs:
            raise kit.OpError("bad_request", "the reference clip has no recognizable speech")
        if total > REF_MAX_S:
            a, b, text = _pick_window(segs, total)
        else:
            a, b, text = 0.0, total, " ".join(s[2] for s in segs)
        cached = {"version": SIDECAR_VERSION, "sig": sig[1:], "start": round(a, 3), "end": round(b, 3), "text": text,
                  "language": language}
        _write_sidecar(side, cached)
        _log(f"reference {ref} ({language or '?'}): {a:.2f}-{b:.2f}s of {total:.1f}s chosen in "
             f"{time.time() - t0:.1f}s: {text!r}")
    cut = x[int(cached["start"] * sr):int(cached["end"] * sr)]
    return cut, sr, cached["text"]


def _write_sidecar(side, cached):
    try:
        tmp = side + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cached, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, side)
    except OSError as exc:
        _log(f"could not cache the reference transcript: {exc}")


def _prompt(ref, ref_text=None):
    key = (tuple(_sig(ref)), ref_text or None)
    if key in _prompts:
        _prompts.move_to_end(key)
        return _prompts[key]
    wav, sr, text = _reference(ref, ref_text)
    kit_worker.check_cancel()
    prompt = _state["model"].create_voice_clone_prompt((torch.from_numpy(wav), sr), ref_text=text)
    _prompts[key] = prompt
    while len(_prompts) > max(1, args.prompt_cache):
        _prompts.popitem(last=False)
    return prompt


# ------------------------------------------------------------------------------------------------------ ops

def op_warm(req):
    t0 = time.time()
    op_load()
    _prompt(req["ref"], req.get("ref_text"))
    return {"elapsed_s": round(time.time() - t0, 2)}


def op_inspect(req):
    """The sidecar of a user clip, built when missing (Whisper only: the TTS model is not loaded for this). With
    ``detect`` (default) a sidecar written before the worker recorded the language (version 3 without ``language``)
    gets Whisper's language detection added."""
    ref = req.get("ref")
    if not ref or not os.path.isfile(ref):
        raise kit.OpError("not_found", f"reference clip not found: {ref}")
    t0 = time.time()
    _reference(ref)
    side = os.path.splitext(ref)[0] + ".omnivoice.json"
    try:
        with open(side, encoding="utf-8") as fh:
            cached = json.load(fh)
    except (OSError, ValueError) as exc:  # the folder is read-only: _reference could not write it
        raise kit.OpError("internal", f"no sidecar for {ref}: {exc}") from None
    if not cached.get("language") and req.get("detect", True):
        x, sr = _load_mono(ref)
        cached["language"] = _detect_language(x, sr)
        _write_sidecar(side, cached)
        _log(f"reference {ref}: language {cached['language'] or '?'} detected in {time.time() - t0:.1f}s")
    return {"language": cached.get("language"), "text": cached.get("text"), "elapsed_s": round(time.time() - t0, 2)}


def op_synth(req):
    text, out, ref = req.get("text"), req.get("out"), req.get("ref")
    if not isinstance(text, str) or not text.strip() or not out or not ref:
        raise kit.OpError("bad_request", "synth needs 'text', 'ref' and 'out'")
    if not os.path.isfile(ref):
        raise kit.OpError("not_found", f"reference clip not found: {ref}")
    lang = str(req.get("lang") or "").lower() or None
    op_load()
    prompt = _prompt(ref, req.get("ref_text"))
    if req.get("normalize", True) and lang:
        text = text_norm.normalize(text, lang)
    kit_worker.check_cancel()
    kw = {"num_step": int(req.get("steps") or 32)}
    for knob in ("position_temperature", "class_temperature", "guidance_scale"):
        if req.get(knob) is not None:
            kw[knob] = float(req[knob])
    if req.get("duration"):
        kw["duration"] = float(req["duration"])
    elif req.get("speed") and abs(float(req["speed"]) - 1.0) > 1e-3:
        kw["speed"] = float(req["speed"])
    t0 = time.time()
    audio = _state["model"].generate(text=text, language=req.get("model_lang") or lang, voice_clone_prompt=prompt,
                                     **kw)[0]
    gen_s = time.time() - t0
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # hand cached blocks back: the RVC worker shares the card
    sr = int(_state["model"].sampling_rate)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sf.write(out, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
    return {"out": out, "sr": sr, "duration_s": round(len(audio) / sr, 3), "gen_s": round(gen_s, 3), "text": text}


def op_status(_req=None):
    return {"loaded": _state["model"] is not None, "rss_gb": _rss_gb(), "vram_gb": _vram_gb()}


idle = []
if args.unload_after > 0:
    idle.append((args.unload_after, _idle_unload))
if args.exit_after > 0 or kit.IDLE_EXIT_S > 0:
    idle.append((args.exit_after or kit.IDLE_EXIT_S, "exit"))
kit_worker = kit.Worker("omnivoice", INP, OUT,
                        {"load": op_load, "synth": op_synth, "warm": op_warm, "inspect": op_inspect, "release": op_release,
                         "status": op_status, "ping": lambda _r: {}},
                        idle=idle)
kit_worker.run()
os._exit(0)
