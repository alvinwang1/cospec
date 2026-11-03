"""Shared GPU helpers: model loading, a cached greedy LM wrapper, timing.

Plain PyTorch + Hugging Face transformers so it runs on a V100 (sm70) without
vLLM. Use fp16: the V100 has no bf16.
"""
from __future__ import annotations

import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def load_model(name: str, device: str = "cuda", dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
    return tok, model.to(device).eval()


def kv_shape(model, batch: int, ctx: int):
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return (batch, cfg.num_key_value_heads, ctx, head_dim)


def crop_cache(cache: DynamicCache, length: int):
    """Truncate the cache to `length` tokens.

    crop(-n) removes n tokens in both transformers 4.x and 5.x (5.x rejects
    the older crop(length) form).
    """
    remove = cache.get_seq_length() - length
    if remove > 0:
        cache.crop(-remove)


def zero_cache(model, batch: int, ctx: int) -> DynamicCache:
    """A KV cache of the right shape without running prefill.

    Attention cost doesn't depend on the values, so this is enough for timing.
    """
    cache = DynamicCache()
    shape = kv_shape(model, batch, ctx)
    for layer in range(model.config.num_hidden_layers):
        k = torch.zeros(shape, dtype=model.dtype, device=model.device)
        cache.update(k, torch.zeros_like(k), layer)
    return cache


@torch.inference_mode()
def verify_forward(model, cache: DynamicCache, batch: int, ctx: int, q: int):
    """One verification step: q tokens per sequence on top of ctx cached ones."""
    dev = model.device
    ids = torch.randint(0, model.config.vocab_size, (batch, q), device=dev)
    pos = torch.arange(ctx, ctx + q, device=dev)
    model(input_ids=ids, past_key_values=cache, position_ids=pos.expand(batch, q),
          cache_position=pos, use_cache=True)
    crop_cache(cache, ctx)


def cuda_time_ms(fn, warmup: int = 3, iters: int = 10) -> tuple[float, float]:
    """Median and p90 wall time of fn() on the GPU, in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return statistics.median(times), times[min(len(times) - 1, int(0.9 * len(times)))]


class CachedLM:
    """Batch-1 causal LM with an explicit, croppable KV cache."""

    def __init__(self, model):
        self.model = model
        self.cache = DynamicCache()
        self.len = 0

    @torch.inference_mode()
    def extend(self, ids: list[int]) -> torch.Tensor:
        """Feed tokens; returns float32 logits [len(ids), vocab]."""
        dev = self.model.device
        x = torch.tensor([ids], device=dev)
        pos = torch.arange(self.len, self.len + len(ids), device=dev)
        out = self.model(input_ids=x, past_key_values=self.cache, position_ids=pos[None],
                         cache_position=pos, use_cache=True)
        self.len += len(ids)
        return out.logits[0].float()

    def crop(self, n: int):
        if n < self.len:
            crop_cache(self.cache, n)
            self.len = n


def greedy_draft(drafter: CachedLM, logits: torch.Tensor, k: int):
    """Draft k tokens greedily from the drafter's last logits.

    Returns (tokens, confidences); confidence is the drafter's softmax
    probability of the token it picked. Feeds tokens 0..k-2 into the cache.
    """
    toks, confs = [], []
    for j in range(k):
        p = torch.softmax(logits[-1], dim=-1)
        t = int(p.argmax())
        toks.append(t)
        confs.append(float(p[t]))
        if j < k - 1:
            logits = drafter.extend([t])
    return toks, confs


def eos_ids(tok, model) -> set[int]:
    ids = {tok.eos_token_id}
    gen = getattr(model.generation_config, "eos_token_id", None)
    ids |= set(gen if isinstance(gen, (list, tuple)) else [gen])
    return {i for i in ids if i is not None}


def chat_ids(tok, prompt: str, max_len: int | None = None) -> list[int]:
    msgs = [{"role": "user", "content": prompt}]
    try:  # Qwen3: answer directly instead of emitting a thinking block
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False).input_ids
    return ids[-max_len:] if max_len else ids
