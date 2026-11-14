"""Persistent zero-shot voice-conversion worker: Seed-VC (Plachtaa/seed-vc, GPL-3.0), the speech model (v1, DiT with a
Whisper-small content encoder, CAM++ speaker style, BigVGAN 22 kHz vocoder). It turns any speech into the timbre of a
short reference clip without training; the native accent of an untrained voice uses it (``accent.py``).

Runs with the engine venv's interpreter, started by ``zsvc.py`` as

    python -I zsvc_worker.py --seed-vc <runtime>/seed-vc --models <runtime>/models/seed-vc [--device cuda|cpu|mps]
                             [--unload-after S]

``--seed-vc`` is the Seed-VC source checkout that setup fetches (its ``modules`` package is imported from there; it
is not part of this repository), ``--models`` the weights setup downloads (offline afterwards):
``dit/`` (Plachta/Seed-VC ``DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth`` + its config),
``bigvgan/`` (nvidia/bigvgan_v2_22khz_80band_256x), ``whisper-small/`` (openai/whisper-small; only the encoder is
kept in memory). CAM++ (``campplus_cn_common.bin``) ships in the checkout.

Standalone on purpose: it imports nothing from ``mimvox_engine``; the event-driven skeleton ``worker_kit.py`` is loaded
by its file path. Wire format: see ``worker_kit.py``.

Ops (one at a time, on the op thread):
  load     {}                                     -> {"load_s","device","vram_gb"}   (no-op when loaded)
  convert  {"in","out","ref","steps":12,"cfg":0.7} -> {"elapsed","duration_s","sr"}
           ``in`` speech (any rate) converted to the timbre of ``ref``; ``out`` a 22.05 kHz mono WAV of the same
           length. The reference is used up to REF_MAX_S; its prompt (content, mel, speaker style) is cached per file
           state. Pitch follows the reference (the model generates it; no transpose needed).
  warm     {"ref"}                                -> {"elapsed_s"}   (loads the models, builds the reference prompt and
                                                                     runs one short conversion, so the first real one
                                                                     does not pay for the first DiT / vocoder run)
  release  {}                                     -> {}   (drops the models and frees the GPU; the next op reloads)
  status   {}                                     -> {"loaded","vram_gb"}
Events: {"event":"ready"}, {"event":"unloaded","reason":"idle"}.
"""
import argparse
import collections
import gc
import importlib.util
import os
import sys
import time
import types


def _load_kit():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker_kit.py")
    spec = importlib.util.spec_from_file_location("mimvox_worker_kit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Hosted: the OmniVoice worker loads this file into its own process (one PyTorch and one CUDA context for both models:
# about 1.5 GB of commit and 1 GB of RAM less than a process of its own) and sets ``__hosted__`` first:
# {"kit": its worker_kit, "worker": its kit.Worker, "argv": this file's arguments}. It then serves ``OPS`` itself.
HOST = globals().get("__hosted__")
kit = HOST["kit"] if HOST else _load_kit()
if not HOST:
    INP, OUT = kit.open_channels()

ap = argparse.ArgumentParser()
ap.add_argument("--seed-vc", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--unload-after", type=float, default=float(os.environ.get("MIMVOX_ZSVC_UNLOAD_S", "0")))  # 0: exit instead
args = ap.parse_args(HOST["argv"] if HOST else None)

SEED_VC = os.path.abspath(args.seed_vc)
MODELS = os.path.abspath(args.models)
SR = 22050                # the speech model's rate (mel hop 256)
HOP = 256
REF_MAX_S = 20.0          # reference used (Seed-VC allows 25 s, but prompt + source share a 30 s context)
CONTEXT_FRAMES = SR // HOP * 30
OVERLAP_FRAMES = 16       # crossfade between windows of a long source
STEPS = 12                # diffusion steps (the engine sends 12 for Best, 8 for Fast): 12 to 25 gave the same held-out
                          # speaker similarity and WER, at half the time (docs/performance.md, "Zero-shot native accent")
WARM_STEPS = 4            # the warm-up conversion (its output is thrown away)
CFG = 0.7                 # inference_cfg_rate (upstream default)
PROMPT_CACHE = 2

sys.path.insert(0, SEED_VC)


def _stub_unused_imports():
    """Seed-VC's modules import two packages the speech model never uses: descript-audio-codec's ``VectorQuantize``
    (length regulator, only with ``vector_quantize``) and matplotlib (BigVGAN's plotting helpers). Install-free
    stand-ins keep them out of the engine venv."""
    if importlib.util.find_spec("dac") is None:
        dac, nn_, quant = (types.ModuleType(n) for n in ("dac", "dac.nn", "dac.nn.quantize"))

        class VectorQuantize:  # noqa: D401 - a stand-in
            def __init__(self, *_a, **_k):
                raise RuntimeError("vector quantization is not available")

        quant.VectorQuantize = VectorQuantize
        dac.nn, nn_.quantize = nn_, quant
        sys.modules.update({"dac": dac, "dac.nn": nn_, "dac.nn.quantize": quant})
    if importlib.util.find_spec("matplotlib") is None:
        mpl, pylab = types.ModuleType("matplotlib"), types.ModuleType("matplotlib.pylab")
        mpl.use = lambda *_a, **_k: None
        mpl.pylab = pylab
        sys.modules.update({"matplotlib": mpl, "matplotlib.pylab": pylab})


_stub_unused_imports()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

_state = {"model": None, "ran": False}   # ran: one conversion has run since the models were loaded
_prompts = collections.OrderedDict()   # (path, mtime, size) -> (prompt_condition, mel2, style)


def _device():
    want = (args.device or "cuda").lower()
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _vram_gb():
    return round(torch.cuda.memory_reserved() / 2**30, 2) if torch.cuda.is_available() else 0.0


def _need(*parts):
    path = os.path.join(MODELS, *parts)
    if not os.path.isfile(path):
        raise kit.OpError("model_missing", f"{path} is missing; run setup")
    return path


def op_load(_req=None):
    if _state["model"] is not None:
        return {"load_s": 0.0, "device": str(_state["model"]["device"]), "vram_gb": _vram_gb()}
    t0 = time.time()
    import yaml  # noqa: PLC0415
    from modules.audio import mel_spectrogram  # noqa: PLC0415
    from modules.bigvgan import bigvgan  # noqa: PLC0415
    from modules.campplus.DTDNN import CAMPPlus  # noqa: PLC0415
    from modules.commons import build_model, load_checkpoint, recursive_munch  # noqa: PLC0415
    from transformers import AutoFeatureExtractor, WhisperModel  # noqa: PLC0415

    dev = _device()
    half = dev.type == "cuda"
    if half:
        kit.gpu_fence(start=True)
    with open(_need("dit", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    mp = recursive_munch(config["model_params"])
    mp.dit_type = "DiT"
    dit = build_model(mp, stage="DiT")
    dit, *_ = load_checkpoint(dit, None, _need("dit", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
                              load_only_params=True, ignore_modules=[], is_distributed=False)
    for k in dit:
        dit[k].eval().to(dev)
    dit.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    camp_path = os.path.join(SEED_VC, "campplus_cn_common.bin")
    if not os.path.isfile(camp_path):
        raise kit.OpError("model_missing", f"{camp_path} is missing; run setup")
    camp = CAMPPlus(feat_dim=80, embedding_size=192)
    camp.load_state_dict(torch.load(camp_path, map_location="cpu"))
    camp.eval().to(dev)

    voc = bigvgan.BigVGAN(bigvgan.load_hparams_from_json(_need("bigvgan", "config.json")), use_cuda_kernel=False)
    voc.load_state_dict(torch.load(_need("bigvgan", "bigvgan_generator.pt"), map_location="cpu")["generator"])
    voc.remove_weight_norm()
    voc = voc.eval().to(dev)

    wdir = os.path.join(MODELS, "whisper-small")
    _need("whisper-small", "model.safetensors")
    wm = WhisperModel.from_pretrained(wdir, torch_dtype=torch.float16 if half else torch.float32)
    del wm.decoder
    wm = wm.eval().to(dev)
    wfe = AutoFeatureExtractor.from_pretrained(wdir)

    sp = config["preprocess_params"]["spect_params"]
    mel_args = {"n_fft": sp["n_fft"], "win_size": sp["win_length"], "hop_size": sp["hop_length"],
                "num_mels": sp["n_mels"], "sampling_rate": SR, "fmin": sp.get("fmin", 0), "fmax": None, "center": False}
    _state["model"] = {"dit": dit, "camp": camp, "voc": voc, "wm": wm, "wfe": wfe, "device": dev, "half": half,
                       "mel": lambda x: mel_spectrogram(x, **mel_args)}
    worker.log(f"Seed-VC loaded on {dev} in {time.time() - t0:.1f}s ({_vram_gb()} GB VRAM reserved)")
    return {"load_s": round(time.time() - t0, 2), "device": str(dev), "vram_gb": _vram_gb()}


def _semantic(m, w16):
    """Whisper-small encoder features (50 per second) of 16 kHz audio of at most 30 s."""
    inp = m["wfe"]([w16.squeeze(0).cpu().numpy()], return_tensors="pt", return_attention_mask=True,
                   sampling_rate=16000)
    feats = m["wm"]._mask_input_features(inp.input_features, attention_mask=inp.attention_mask).to(m["device"])
    out = m["wm"].encoder(feats.to(m["wm"].encoder.dtype), return_dict=True)
    return out.last_hidden_state.float()[:, :w16.size(-1) // 320 + 1]


def _semantic_long(m, w16):
    """Encoder features of any length: 30 s windows overlapping by 5 s (upstream's scheme)."""
    if w16.size(-1) <= 16000 * 30:
        return _semantic(m, w16)
    parts, buf, pos = [], None, 0
    while pos < w16.size(-1):
        if buf is None:
            chunk = w16[:, pos:pos + 16000 * 30]
        else:
            chunk = torch.cat([buf, w16[:, pos:pos + 16000 * 25]], dim=-1)
        s = _semantic(m, chunk)
        parts.append(s if buf is None else s[:, 50 * 5:])
        buf = chunk[:, -16000 * 5:]
        pos += 16000 * 30 if pos == 0 else chunk.size(-1) - 16000 * 5
        worker.check_cancel()
    return torch.cat(parts, dim=1)


def _load(path, max_s=None):
    import librosa  # noqa: PLC0415
    x, _ = librosa.load(path, sr=SR, mono=True, duration=max_s)
    if len(x) < SR // 4:
        raise kit.OpError("bad_request", f"{path} is too short to convert")
    return x


def _prompt(m, ref):
    st = os.stat(ref)
    key = (os.path.abspath(ref), st.st_mtime_ns, st.st_size)
    if key in _prompts:
        _prompts.move_to_end(key)
        return _prompts[key]
    dev = m["device"]
    r = torch.tensor(_load(ref, REF_MAX_S))[None].to(dev)
    r16 = torchaudio.functional.resample(r, SR, 16000)
    mel2 = m["mel"](r.float())
    s_ori = _semantic(m, r16)
    fb = torchaudio.compliance.kaldi.fbank(r16, num_mel_bins=80, dither=0, sample_frequency=16000)
    style = m["camp"]((fb - fb.mean(dim=0, keepdim=True))[None])
    pc = m["dit"].length_regulator(s_ori, ylens=torch.LongTensor([mel2.size(2)]).to(dev), n_quantizers=3, f0=None)[0]
    _prompts[key] = (pc, mel2, style)
    while len(_prompts) > PROMPT_CACHE:
        _prompts.popitem(last=False)
    return _prompts[key]


def _crossfade(a, b, n):
    fade_out = np.cos(np.linspace(0, np.pi / 2, n)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, n)) ** 2
    k = min(len(b), n)
    b = b.copy()
    b[:k] = b[:k] * fade_in[:k] + (a[-n:] * fade_out)[:k]
    return b


def _convert(m, x, prompt, steps, cfg):
    """``x`` (22.05 kHz float32 numpy) in the timbre of ``prompt`` (``_prompt``): float32 numpy of the same length."""
    dev = m["device"]
    pc, mel2, style = prompt
    x = torch.tensor(x)[None].to(dev)
    s_alt = _semantic_long(m, torchaudio.functional.resample(x, SR, 16000))
    n_frames = m["mel"](x.float()).size(2)
    cond = m["dit"].length_regulator(s_alt, ylens=torch.LongTensor([n_frames]).to(dev), n_quantizers=3, f0=None)[0]
    window = max(CONTEXT_FRAMES - mel2.size(2), 4 * OVERLAP_FRAMES)
    overlap = OVERLAP_FRAMES * HOP
    pieces, prev, done = [], None, 0
    while done < cond.size(1):
        worker.check_cancel()
        chunk = cond[:, done:done + window]
        last = done + window >= cond.size(1)
        cat = torch.cat([pc, chunk], dim=1)
        amp = torch.autocast(device_type="cuda", dtype=torch.float16) if m["half"] else _null()
        with amp:
            vm = m["dit"].cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(dev), mel2, style, None, steps,
                                        inference_cfg_rate=cfg)[:, :, mel2.size(-1):]
        w = m["voc"](vm.float()).squeeze().float().cpu().numpy()
        if prev is None and last:
            pieces.append(w)
            break
        if prev is None:
            pieces.append(w[:-overlap])
        elif last:
            pieces.append(_crossfade(prev, w, overlap))
            break
        else:
            pieces.append(_crossfade(prev, w[:-overlap], overlap))
        prev = w[-overlap:]
        done += vm.size(2) - OVERLAP_FRAMES
    _state["ran"] = True
    return np.clip(np.concatenate(pieces), -1.0, 1.0).astype(np.float32)


@torch.no_grad()
def op_convert(req):
    t0 = time.time()
    src, out, ref = req.get("in"), req.get("out"), req.get("ref")
    if not (src and out and ref):
        raise kit.OpError("bad_request", "'in', 'out' and 'ref' are required")
    for p in (src, ref):
        if not os.path.isfile(p):
            raise kit.OpError("bad_request", f"{p} does not exist")
    steps = int(req.get("steps") or STEPS)
    cfg = float(req.get("cfg", CFG))
    op_load()
    m = _state["model"]
    prompt = _prompt(m, ref)
    worker.check_cancel()
    y = _convert(m, _load(src), prompt, steps, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sf.write(out, y, SR, subtype="PCM_16")
    return {"elapsed": round(time.time() - t0, 3), "duration_s": round(len(y) / SR, 3), "sr": SR}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@torch.no_grad()
def op_warm(req):
    t0 = time.time()
    ref = req.get("ref")
    if not ref or not os.path.isfile(ref):
        raise kit.OpError("bad_request", "'ref' must be an existing file")
    op_load()
    m = _state["model"]
    prompt = _prompt(m, ref)
    if not _state["ran"]:  # the first DiT and vocoder run costs more (measured: +0.2 s alone, +0.8 s next to OmniVoice)
        _convert(m, _load(ref, 1.5), prompt, WARM_STEPS, CFG)
    return {"elapsed_s": round(time.time() - t0, 2)}


def op_release(_req=None):
    had = _state["model"] is not None
    _state["model"] = None
    _state["ran"] = False
    _prompts.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if had:
        worker.log(f"models released ({_vram_gb()} GB VRAM still reserved)")
    return {}


def op_status(_req=None):
    return {"loaded": _state["model"] is not None, "vram_gb": _vram_gb()}


def _idle_unload():
    if _state["model"] is None:
        return None
    op_release()
    return {"event": "unloaded", "reason": "idle"}


OPS = {"load": op_load, "convert": op_convert, "warm": op_warm, "release": op_release, "status": op_status}
if HOST:
    worker = HOST["worker"]
else:
    idle = [(args.unload_after, _idle_unload)] if args.unload_after > 0 else []
    if kit.IDLE_EXIT_S > 0:
        idle.append((kit.IDLE_EXIT_S, "exit"))
    worker = kit.Worker("zsvc", INP, OUT, {**OPS, "ping": lambda _req: {}}, idle=idle)
    try:
        worker.run()
    except BaseException as exc:  # noqa: BLE001
        worker.send({"event": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        sys.stderr.flush()
        os._exit(1)
    sys.stderr.flush()
    os._exit(0)  # do not wait for CUDA teardown or the stdin reader thread
