"""Faster OmniVoice decoding loop: same model and sampling, the CFG rows packed into one sequence with per-block
attention, and short passes replayed as a CUDA graph (measurements: docs/performance.md, "Speech speed")."""
import math
import os

import torch
import torch.nn.functional as F

# Below ~1100 tokens a pass is bound by its ~40 ms of CPU-side launches; a graph capture costs ~0.12 s.
GRAPH_MAX_TOKENS = int(os.environ.get("MIMVOX_OMNIVOICE_GRAPH_TOKENS", "1100"))  # 0: never a CUDA graph
GRAPH_MIN_STEPS = 12
ATTN_NAME = "mimvox_blocks"

_split = [None]  # the conditional block's length while a packed pass runs (read by _block_attention)


class Unsupported(Exception):
    """The fast path cannot run this request: use the library's ``generate``."""


def _block_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Packed pass: each block attends only to itself (two unmasked SDPA calls); otherwise the library's SDPA."""
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

    c = _split[0]
    if c is None or attention_mask is not None or query.shape[0] != 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      **kwargs)
    key = repeat_kv(key, module.num_key_value_groups)      # the unmasked kernels of torch 2.8 take no GQA
    value = repeat_kv(value, module.num_key_value_groups)
    b, h, n, d = query.shape
    out = torch.empty((b, n, h, d), dtype=query.dtype, device=query.device)
    out[:, :c] = F.scaled_dot_product_attention(query[:, :, :c], key[:, :, :c], value[:, :, :c],
                                                scale=scaling).transpose(1, 2)
    out[:, c:] = F.scaled_dot_product_attention(query[:, :, c:], key[:, :, c:], value[:, :, c:],
                                                scale=scaling).transpose(1, 2)
    return out, None


def _install(model):
    """Route the model's attention through :func:`_block_attention` (once per model)."""
    if model.llm.config._attn_implementation == ATTN_NAME:
        return
    if model.llm.config._attn_implementation != "sdpa":
        raise Unsupported(f"attention {model.llm.config._attn_implementation!r}")
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(ATTN_NAME, _block_attention)
    model.llm.set_attn_implementation(ATTN_NAME)


@torch.inference_mode()
def generate(model, text, language=None, voice_clone_prompt=None, speed=None, duration=None, **kwargs):
    """Drop-in for ``model.generate`` with one voice-clone prompt on CUDA; raises :class:`Unsupported` otherwise."""
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    if model.device.type != "cuda" or voice_clone_prompt is None or isinstance(voice_clone_prompt, list):
        raise Unsupported("needs CUDA and one voice-clone prompt")
    cfg = OmniVoiceGenerationConfig.from_dict(kwargs)
    model.eval()
    _install(model)
    task = model._preprocess_all(text=text, language=language, voice_clone_prompt=voice_clone_prompt,
                                 preprocess_prompt=cfg.preprocess_prompt, speed=speed, duration=duration)
    short_idx, long_idx = task.get_indices(cfg, model.audio_tokenizer.config.frame_rate)
    results = [None] * task.batch_size
    for i in short_idx:
        results[i] = _iterate(model, task.slice_task([i]), cfg)
    if long_idx:  # over 30 s: the library's chunked decoding
        for i, t in zip(long_idx, model._generate_chunked(task.slice_task(long_idx), cfg)):
            results[i] = t
    return [model._decode_and_post_process(results[i], task.ref_rms[i], cfg) for i in range(task.batch_size)]


def _schedule(t_len, n_cb, cfg):
    """How many tokens each step unmasks (the library's schedule, verbatim)."""
    from omnivoice.models.omnivoice import _get_time_steps

    timesteps = _get_time_steps(t_start=0.0, t_end=1.0, num_step=cfg.num_step, t_shift=cfg.t_shift).tolist()
    total = t_len * n_cb
    rem, sched = total, []
    for step in range(cfg.num_step):
        num = rem if step == cfg.num_step - 1 else min(math.ceil(total * (timesteps[step + 1] - timesteps[step])), rem)
        sched.append(int(num))
        rem -= int(num)
    return sched


def _iterate(model, task, cfg):
    """The library's ``_generate_iterative`` for one item on the packed layout; returns the tokens ``(C, T)``."""
    from omnivoice.models.omnivoice import _gumbel_sample

    dev = model.device
    n_cb, mask_id = model.config.num_audio_codebook, model.config.audio_mask_id
    inp = model._prepare_inference_inputs(task.texts[0], task.target_lens[0], task.ref_texts[0],
                                          task.ref_audio_tokens[0], task.langs[0], task.instructs[0], cfg.denoise)
    c, t = inp["input_ids"].size(2), task.target_lens[0]
    # [conditional row | unconditional row (its last t tokens)], positions restarting for the second block
    ids = torch.cat([inp["input_ids"], inp["input_ids"][:, :, -t:]], dim=2).contiguous()
    audio_mask = torch.cat([inp["audio_mask"], inp["audio_mask"][:, -t:]], dim=1).contiguous()
    pos = torch.cat([torch.arange(c, device=dev), torch.arange(t, device=dev)]).unsqueeze(0)
    gather = torch.cat([torch.arange(c - t, c, device=dev), torch.arange(c, c + t, device=dev)])
    no_mask = {"full_attention": None}

    def targets():
        emb = model._prepare_embed_inputs(ids, audio_mask)
        hid = model.llm(inputs_embeds=emb, attention_mask=no_mask, position_ids=pos, return_dict=True)[0]
        return hid[0].index_select(0, gather)

    tokens = torch.full((1, n_cb, t), mask_id, dtype=torch.long, device=dev)
    sched = _schedule(t, n_cb, cfg)
    layer_ids = torch.arange(n_cb, device=dev).view(1, -1, 1)
    use_graph = 0 < c + t <= GRAPH_MAX_TOKENS and cfg.num_step >= GRAPH_MIN_STEPS
    graph = static = None
    _split[0] = c
    try:
        for step in range(cfg.num_step):
            if graph is not None:
                graph.replay()
                h = static
            elif use_graph and step == 1:  # step 0 ran eagerly and warmed the kernels up
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static = targets()
                graph.replay()
                h = static
            else:
                h = targets()
            k = sched[step]
            if k <= 0:
                continue
            logits = model.audio_heads(h).view(2, t, n_cb, -1).permute(0, 2, 1, 3).float()  # [2, C, T, V]
            pred, scores = model._predict_tokens_with_scoring(logits[0:1], logits[1:2], cfg)
            scores = scores - (layer_ids * cfg.layer_penalty_factor)
            if cfg.position_temperature > 0.0:
                scores = _gumbel_sample(scores, cfg.position_temperature)
            scores.masked_fill_(tokens != mask_id, -float("inf"))
            _, top = torch.topk(scores.flatten(), k)
            flat = tokens.flatten()
            flat[top] = pred.flatten()[top]
            tokens.copy_(flat.view_as(tokens))
            ids[0, :, c - t:c] = tokens[0]
            ids[0, :, c:c + t] = tokens[0]
    finally:
        _split[0] = None
        del graph, static
    return tokens[0]

