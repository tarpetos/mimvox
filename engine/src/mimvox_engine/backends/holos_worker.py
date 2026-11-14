"""Persistent HolosTTS worker: Ukrainian speech with exact word stress (the ``ukrainian`` runtime component).

OmniVoice reads letters and guesses the stress of every word (measured on Ukrainian: about 65-72 % of the words on
the right syllable, and stress marks only nudge it, see ``omnivoice_stress.py``). HolosTTS (patriotyk/HolosTTS, MIT)
reads IPA phonemes instead, and the stress is part of each phoneme: what this worker decides is what is spoken.
The stress comes from patriotyk's ``ukrainian-word-stress``: lang-uk's stress dictionary, with Stanza's Ukrainian
part-of-speech and morphology tagger choosing between the readings of a heteronym from the sentence (ру́ки / руки́ by
case, за́мок / замо́к stays the model's guess only when the tags cannot tell), and a small neural accentor for words the
dictionary lacks. The phonemes come from ``ipa-uk``. The voice is cloned from the voice's reference clip by HolosTTS's
voice encoder (a style vector; the longer the clean clip, the closer).

Runs with the interpreter of its own environment (``<data>/runtime/holos/.venv``: torch 2.8, holos-tts, stanza,
ipa-uk, ukrainian-word-stress, num2words), started by ``backends/holos.py`` as

    python -I holos_worker.py --model-dir <data>/runtime/models/holos [--device cuda|cpu|mps]

Standalone on purpose: it imports nothing from ``mimvox_engine``; ``worker_kit.py`` and the Ukrainian text front-end
of the OmniVoice backend (``omnivoice_text.py``: numbers, units, dates and abbreviations spelled out with cases) are
loaded by their file paths. Wire format: see ``worker_kit.py``. Offline: HF_HUB_OFFLINE, and Stanza reads its models
from ``<model-dir>/stanza`` (STANZA_RESOURCES_DIR) without downloading.

Ops (one at a time, on the op thread):
  load     {}                                   -> {"load_s","device","vram_gb"}   (no-op when loaded)
  synth    {"text","ref","out","speed":1.0}      -> {"out","sr","duration_s","gen_s","text","stressed"}
           ``text`` Ukrainian, as the user wrote it: "+" before a vowel or an acute after it marks the stress by hand
           (the user's mark always wins over the dictionary). ``ref`` a WAV of the voice. ``stressed`` is the text
           with the stress that was spoken (U+0301 after each stressed vowel).
  warm     {"ref"}                               -> {"elapsed_s"}   (the voice's style vector only)
  stress   {"text"}                              -> {"stressed","unsure":[...]}   (no synthesis: what would be said,
           and the words of two or more syllables that got no stress from the dictionary, the tagger or the accentor)
  release  {}                                    -> {}   (drops the models and frees the GPU)
  status   {}                                    -> {"loaded","vram_gb"}
Events: {"event":"ready"}.
"""
import argparse
import collections
import importlib.util
import os
import re
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kit = _load_by_path("mimvox_worker_kit", os.path.join(os.path.dirname(HERE), "worker_kit.py"))
INP, OUT = kit.open_channels()  # before torch / stanza: they print, also from C code

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--voice-cache", type=int, default=4)   # style vectors of the last N reference clips
args = ap.parse_args()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["STANZA_RESOURCES_DIR"] = os.path.join(args.model_dir, "stanza")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

text_norm = _load_by_path("mimvox_omnivoice_text", os.path.join(HERE, "omnivoice_text.py"))

SR = 24000
ACUTE = "́"
VOWELS = "аеєиіїоуюяАЕЄИІЇОУЮЯ"
# Within one segment the engine sends (a few sentences), HolosTTS speaks sentence by sentence (it was trained on
# sentences; long inputs drift): split after .?!: followed by a space, keeping pieces of at least MIN_PIECE characters
# together, the way HolosTTS's own demo does. Sentences are joined with SENTENCE_GAP_S of silence.
MIN_PIECE = 100
SENTENCE_GAP_S = 0.12
_DASHES = re.compile(r"[᠆‐‑‒–—―⁻₋−⸺⸻]")
# The user's marks (as in the OmniVoice backend): "+" right before a vowel, or an acute (U+0301, or the spacing "´")
# right after one. HolosTTS's demo writes "+" after the vowel; Mimvox keeps one convention for both backends.
_PLUS = re.compile(rf"\+([{VOWELS}])")
_SPACING = re.compile(rf"([{VOWELS}])[´ˊ]")

_state = {"model": None, "encoder": None, "stressify": None, "ipa": None, "device": None}
_voices = collections.OrderedDict()   # (path, mtime_ns, size) -> style tensor, most recently used last


def _log(msg):
    print(f"[holos] {msg}", file=sys.stderr, flush=True)


def _vram_gb():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return round((total - free) / 2**30, 2)
    return 0.0


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
    t0 = time.time()
    from holos.models import HolosTTS, VoiceEncoder
    from ipa_uk import ipa
    from ukrainian_word_stress import Stressifier

    dev = _device()
    d = args.model_dir
    _state["model"] = HolosTTS(os.path.join(d, "holos.pt"), device=dev, voices_path=os.path.join(d, "voices.pt"))
    _state["encoder"] = VoiceEncoder(os.path.join(d, "voice_encoder.pt"), device=dev)
    _state["stressify"] = Stressifier()      # loads Stanza's Ukrainian tokenizer + tagger (REUSE_RESOURCES: offline)
    _state["ipa"] = ipa
    _state["device"] = dev
    _log(f"loaded on {dev} in {time.time() - t0:.1f}s, VRAM {_vram_gb()} GB")
    return {"load_s": round(time.time() - t0, 2), "device": dev, "vram_gb": _vram_gb()}


def op_release(_req=None):
    for key in ("model", "encoder", "stressify"):
        _state[key] = None
    _voices.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {}


def _sig(path):
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


def _voice(ref):
    sig = _sig(ref)
    if sig in _voices:
        _voices.move_to_end(sig)
        return _voices[sig]
    with torch.inference_mode():
        style = _state["encoder"].encode(ref)
    _voices[sig] = style
    while len(_voices) > max(1, args.voice_cache):
        _voices.popitem(last=False)
    return style


def _user_marks(text):
    """The user's stress marks as U+0301 after the vowel, the form the stressifier keeps and ipa-uk reads."""
    text = _PLUS.sub(lambda m: m.group(1) + ACUTE, text)
    return _SPACING.sub(lambda m: m.group(1) + ACUTE, text)


def _prepare(text):
    """Text as written -> text with the stress of every word that has one, ready for ipa-uk."""
    text = unicodedata.normalize("NFC", text)
    text = _user_marks(text)
    text = text_norm._uk(text)                       # numbers, units, dates, abbreviations -> words, with cases
    text = text.replace('"', "").replace("«", "").replace("»", "").replace("„", "").replace("“", "").replace("”", "")
    text = unicodedata.normalize("NFKC", text)
    text = _DASHES.sub("-", text)
    text = re.sub(r" - ", ": ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _state["stressify"](text) if text else text


def _pieces(text):
    """Sentences of at least MIN_PIECE characters (short ones join the next)."""
    out, cur = [], ""
    for i, ch in enumerate(text):
        cur += ch
        if ch in ".?!:" and i + 1 < len(text) and text[i + 1] == " " and len(cur) > MIN_PIECE:
            out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return [p if p[-1] in ".?!:-…" else p + "." for p in out]


def _unsure(stressed):
    words = re.findall(rf"[\w'’{ACUTE}]+", stressed)
    return sorted({w for w in words if sum(c in VOWELS for c in w) >= 2 and ACUTE not in w})


def op_stress(req):
    text = req.get("text")
    if not isinstance(text, str):
        raise kit.OpError("bad_request", "stress needs 'text'")
    op_load()
    stressed = _prepare(text)
    return {"stressed": stressed, "unsure": _unsure(stressed)}


def op_warm(req):
    ref = req.get("ref")
    if not ref or not os.path.isfile(ref):
        raise kit.OpError("not_found", f"reference clip not found: {ref}")
    op_load()
    t0 = time.time()
    _voice(ref)
    return {"elapsed_s": round(time.time() - t0, 3)}


def op_synth(req):
    text, out, ref = req.get("text"), req.get("out"), req.get("ref")
    if not isinstance(text, str) or not text.strip() or not out or not ref:
        raise kit.OpError("bad_request", "synth needs 'text', 'ref' and 'out'")
    if not os.path.isfile(ref):
        raise kit.OpError("not_found", f"reference clip not found: {ref}")
    speed = float(req.get("speed") or 1.0)
    if not (0.25 <= speed <= 4.0):
        raise kit.OpError("bad_request", "speed must be between 0.25 and 4")
    op_load()
    style = _voice(ref)
    stressed = _prepare(text)
    t0 = time.time()
    gap = np.zeros(int(SR * SENTENCE_GAP_S), dtype=np.float32)
    chunks = []
    with torch.inference_mode():
        for piece in _pieces(stressed):
            kit_worker.check_cancel()
            phonemes = _state["ipa"](piece)
            if not phonemes:
                continue
            wav = _state["model"].generate(phonemes, style, speed=speed).squeeze().float().cpu().numpy()
            if chunks:
                chunks.append(gap)
            chunks.append(wav.astype(np.float32))
    if not chunks:
        raise kit.OpError("bad_request", "nothing to say in this text")
    audio = np.concatenate(chunks)
    peak = float(np.abs(audio).max())
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sf.write(out, audio, SR, subtype="PCM_16")
    return {"out": out, "sr": SR, "duration_s": round(len(audio) / SR, 3), "gen_s": round(time.time() - t0, 3),
            "text": text, "stressed": stressed}


def op_status(_req=None):
    return {"loaded": _state["model"] is not None, "vram_gb": _vram_gb()}


kit_worker = kit.Worker("holos", INP, OUT,
                        {"load": op_load, "synth": op_synth, "warm": op_warm, "stress": op_stress,
                         "release": op_release, "status": op_status, "ping": lambda _r: {}})
kit_worker.run()
os._exit(0)
