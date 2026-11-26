"""Faster Chatterbox decoding: the same model and sampling, one token per CUDA-graph replay over a static KV cache with
SDPA attention. The library turns every layer to eager attention to read three heads for its alignment check
(15.6 ms per token on an RTX 4070); here those heads are computed on their own."""

import torch
import torch.nn.functional as F

# (layer, head) the library's alignment check reads (chatterbox.models.t3.inference.alignment_stream_analyzer).
SPY = ((12, 15), (13, 11), (9, 2))
# Prompt + speech tokens. Each step attends over the smallest bucket that holds the tokens so far: a token cost 12.2 ms
# with 1536 cached positions in fp32 and 5.0 ms with 768 in fp16 (RTX 4070).
BUCKETS = (256, 512, 768, 1024, 1536, 2048)
MAX_LEN = BUCKETS[-1]


class Unsupported(Exception):
    """The fast path cannot run this request: use the library's ``inference``."""


class Decoder:
    def __init__(self, t3, max_len=MAX_LEN):
        self.t3, self.tfmr, self.max_len = t3, t3.tfmr, max_len
        cfg = self.tfmr.config
        p = next(self.tfmr.parameters())
        self.heads, self.hd = cfg.num_attention_heads, cfg.head_dim
        if cfg.num_key_value_heads != self.heads:
            raise Unsupported("grouped-query attention")
        shape = (cfg.num_hidden_layers, 2, self.heads, max_len, self.hd)
        self.k = torch.zeros(shape, device=p.device, dtype=p.dtype)
        self.v = torch.zeros_like(self.k)
        self.embed = torch.zeros(2, 1, cfg.hidden_size, device=p.device, dtype=p.dtype)
        self.pos = torch.zeros(1, dtype=torch.long, device=p.device)
        self.ar = torch.arange(max_len, device=p.device)
        self.dtype = p.dtype
        self.spy_at = {layer: i for i, (layer, _h) in enumerate(SPY)}
        self.graphs = {}

    def prefill(self, embeds):
        out = self.tfmr(inputs_embeds=embeds.to(self.dtype), use_cache=True)
        n = embeds.shape[1]
        for i, layer in enumerate(out.past_key_values.layers):
            self.k[i, :, :, :n] = layer.keys
            self.v[i, :, :, :n] = layer.values
        return n

    def _step(self, n):
        h, pos = self.embed, self.pos
        cos, sin = self.tfmr.rotary_emb(h, pos.view(1, 1))
        keep = (self.ar[:n] <= pos).view(1, 1, 1, -1)
        rows = [None] * len(SPY)
        for i, layer in enumerate(self.tfmr.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(2, 1, self.heads, self.hd).transpose(1, 2)
            k = attn.k_proj(x).view(2, 1, self.heads, self.hd).transpose(1, 2)
            v = attn.v_proj(x).view(2, 1, self.heads, self.hd).transpose(1, 2)
            cos_, sin_ = cos.unsqueeze(1), sin.unsqueeze(1)
            q = q * cos_ + _rotate_half(q) * sin_
            k = k * cos_ + _rotate_half(k) * sin_
            self.k[i].index_copy_(2, pos, k)
            self.v[i].index_copy_(2, pos, v)
            keys, values = self.k[i, :, :, :n], self.v[i, :, :, :n]
            o = F.scaled_dot_product_attention(q, keys, values, attn_mask=keep, scale=attn.scaling)
            if i in self.spy_at:
                head = SPY[self.spy_at[i]][1]
                s = (q[0, head] @ keys[0, head].transpose(0, 1)) * attn.scaling
                rows[self.spy_at[i]] = torch.softmax(s.masked_fill(~keep[0, 0], float("-inf")), dim=-1)
            h = h + attn.o_proj(o.transpose(1, 2).reshape(2, 1, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = self.t3.speech_head(self.tfmr.norm(h))[:, -1, :].float()
        rows = torch.stack(rows).float()
        return logits, F.pad(rows, (0, self.max_len - n))

    def step(self, embed, pos):
        """Logits (2, V) and the spied attention rows (3, 1, max_len) of the token ``embed`` at ``pos``."""
        self.embed.copy_(embed.to(self.dtype))
        self.pos.fill_(pos)
        n = next(b for b in BUCKETS if b > pos and b <= self.max_len)
        if n not in self.graphs:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._step(n)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits, rows = self._step(n)
            self.graphs[n] = (graph, logits, rows)
        graph, logits, rows = self.graphs[n]
        graph.replay()
        return logits, rows


def _analyzer(hp, text_slice):
    """The library's alignment check without its attention hooks: the decoder hands it the spied rows."""
    from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer

    a = AlignmentStreamAnalyzer.__new__(AlignmentStreamAnalyzer)
    i, j = text_slice
    a.text_tokens_slice, a.eos_idx, a.alignment = text_slice, hp.stop_speech_token, torch.zeros(0, j - i)
    a.curr_frame_pos, a.text_position, a.started, a.started_at = 0, 0, False, None
    a.complete, a.completed_at, a.generated_tokens = False, None, []
    a.last_aligned_attns = [None] * len(SPY)
    return a


@torch.inference_mode()
def inference(
    t3,
    decoder,
    *,
    t3_cond,
    text_tokens,
    max_new_tokens=1000,
    temperature=0.8,
    top_p=0.95,
    min_p=0.05,
    repetition_penalty=1.2,
    cfg_weight=0.5,
    **_kw,
):
    """``T3.inference`` with the same sampling, alignment check and stops, on ``decoder``."""
    from chatterbox.models.t3.t3 import _ensure_BOT_EOT
    from transformers.generation.logits_process import (
        MinPLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        TopPLogitsWarper,
    )

    hp = t3.hp
    _ensure_BOT_EOT(text_tokens, hp)
    text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=t3.device)
    start = hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start, cfg_weight=cfg_weight
    )
    j = len_cond + text_tokens.size(-1)
    if embeds.shape[0] != 2 or embeds.shape[1] != j + 1 or j + 2 + max_new_tokens > decoder.max_len:
        raise Unsupported(f"prompt {embeds.shape} with {max_new_tokens} new tokens")
    decoder.prefill(embeds[:, :j])
    _logits, rows_start = decoder.step(embeds[:, j : j + 1], j)
    rows_start = rows_start[:, 0].float().cpu()
    bos = torch.tensor([[hp.start_speech_token]], dtype=torch.long, device=t3.device)
    bos_embed = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
    logits_both, rows = decoder.step(torch.cat([bos_embed, bos_embed]), j + 1)
    analyzer = _analyzer(hp, (len_cond, j)) if hp.is_multilingual else None
    if analyzer is not None:
        for b in range(len(SPY)):
            first = torch.zeros(j + 2, decoder.max_len)
            first[j], first[j + 1] = rows_start[b], rows[b, 0].float().cpu()
            analyzer.last_aligned_attns[b] = first
    generated = bos.clone()
    predicted = []
    top_p_warper, min_p_warper = TopPLogitsWarper(top_p=top_p), MinPLogitsWarper(min_p=min_p)
    repetition = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
    for i in range(max_new_tokens):
        cond, uncond = logits_both[0:1], logits_both[1:2]
        logits = cond + cfg_weight * (cond - uncond)
        if analyzer is not None:
            if i > 0:
                spied = rows[:, 0].float().cpu()
                for b in range(len(SPY)):
                    analyzer.last_aligned_attns[b] = spied[b : b + 1]
            logits = analyzer.step(logits, next_token=int(generated[0, -1]))
        ids = generated[:1]
        logits = repetition(ids, logits)
        if temperature != 1.0:
            logits = logits / temperature
        logits = top_p_warper(ids, min_p_warper(ids, logits))
        token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        predicted.append(token)
        generated = torch.cat([generated, token], dim=1)
        if int(token) == hp.stop_speech_token:
            break
        embed = t3.speech_emb(token) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        logits_both, rows = decoder.step(torch.cat([embed, embed]), j + 2 + i)
    return torch.cat(predicted, dim=1)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
