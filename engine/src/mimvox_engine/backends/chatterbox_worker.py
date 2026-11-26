"""Chatterbox Multilingual worker (MIT, Resemble AI): 23 languages. Its output carries Resemble's inaudible Perth
watermark.

    python -I chatterbox_worker.py --model-dir <data>/runtime/models/chatterbox [--device cuda|cpu|mps]

Ops: see ``clone_worker.py``.
"""

import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).absolute().parent
_spec = importlib.util.spec_from_file_location("mimvox_clone_worker", HERE / "clone_worker.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
INP, OUT = common.kit.open_channels()  # before torch: it prints, also from C code
args = common.parse_args()

SPEECH_TOKENS_PER_S = 25
_conds = {"key": None}
_fast = {"module": None, "decoder": None}


def _patch(model_dir):
    from chatterbox.models.t3.inference import alignment_stream_analyzer as asa
    from chatterbox.models.tokenizers import tokenizer

    real = tokenizer.hf_hub_download

    def local_first(*a, filename=None, **kw):
        # It looks for Cangjie5_TC.json (Chinese) in an HF cache layout; setup keeps the files flat in model_dir.
        path = Path(model_dir) / (filename or "")
        return str(path) if filename and path.is_file() else real(*a, filename=filename, **kw)

    tokenizer.hf_hub_download = local_first
    step = asa.AlignmentStreamAnalyzer.step

    def tolerant_step(self, logits, next_token=None):
        # Its repetition check indexes past the text on 5 tokens or fewer ("One."): IndexError (chatterbox-tts 0.1.7).
        try:
            return step(self, logits, next_token=next_token)
        except IndexError:
            return logits

    asa.AlignmentStreamAnalyzer.step = tolerant_step


def load(model_dir, device):
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    _patch(model_dir)
    _conds["key"] = None
    model = ChatterboxMultilingualTTS.from_local(model_dir, device)
    _fast["decoder"] = None
    if device == "cuda":
        # fp16 gave the same speech tokens as fp32 on every test sentence; the decoder to audio stays fp32.
        model.t3.tfmr.to(torch.float16)
        model.t3.speech_head.to(torch.float16)
        _fast["module"] = common.load_by_path("mimvox_chatterbox_fast", HERE / "chatterbox_fast.py")
        _fast["decoder"] = _fast["module"].Decoder(model.t3)
    return model


def _slow_inference(model, kw):
    """The library's own loop, which needs the fp32 weights."""
    import torch

    half = next(model.t3.tfmr.parameters()).dtype == torch.float16
    if half:
        model.t3.tfmr.float()
        model.t3.speech_head.float()
    try:
        return type(model.t3).inference(model.t3, **kw)
    finally:
        if half:
            model.t3.tfmr.half()
            model.t3.speech_head.half()


def _inference(model, cap):
    def run(**kw):
        kw["max_new_tokens"] = cap
        fast, decoder = _fast["module"], _fast["decoder"]
        if decoder is not None:
            try:
                return fast.inference(model.t3, decoder, **kw)
            except fast.Unsupported:
                pass
        return _slow_inference(model, kw)

    return run


def speak(model, text, lang, prompt, req):
    exaggeration = round(max(0.25, min(1.0, 0.5 * float(req.get("expressiveness") or 0.8) / 0.8)), 3)
    st = Path(prompt["wav"]).stat()
    key = (Path(req["ref"]).absolute(), req.get("start"), req.get("end"), st.st_size, exaggeration)
    if _conds["key"] != key:
        model.prepare_conditionals(prompt["wav"], exaggeration=exaggeration)
        _conds["key"] = key
    # A slow reading of the text with some room: on a word or two it otherwise ran on for up to 10 s of babble.
    cap = min(1000, int(SPEECH_TOKENS_PER_S * (0.12 * len(text) + 0.8)))
    model.t3.inference = _inference(model, cap)
    try:
        wav = model.generate(text, language_id=lang, exaggeration=exaggeration)
    finally:
        del model.t3.inference
    return wav.squeeze().float().cpu().numpy(), model.sr


common.run("chatterbox", INP, OUT, args, load, speak, release=lambda: _fast.update(decoder=None))
os._exit(0)
