"""Check step timing on the GPU before profiling.

  python gpu/graph_check.py --model Qwen/Qwen3-4B

1. Exactness: verification logits with the engine-style attention (cospec_sdpa) match Hugging
   Face's sdpa on a real cached prefix, and a CUDA-graph replay matches the eager forward.
2. Timing: one step's cost under eager HF sdpa, and as CUDA graphs with HF sdpa, cospec_sdpa
   and the timing-only cospec_sdpa_nocausal. The profile should use the fastest exact path.
"""
from __future__ import annotations

import argparse
import copy

import torch
from transformers import DynamicCache

from common import GraphedStep, cuda_time_ms, load_model, verify_forward, zero_cache


@torch.inference_mode()
def attention_matches(model, prefix: int = 300, q: int = 6) -> float:
    """Max |cospec_sdpa - sdpa| logit difference for q new tokens on a real prefix."""
    torch.manual_seed(0)
    dev = model.device
    ids = torch.randint(0, model.config.vocab_size, (2, prefix + q), device=dev)
    model.set_attn_implementation("sdpa")
    cache = DynamicCache()
    model(input_ids=ids[:, :prefix], past_key_values=cache, use_cache=True)
    pos = torch.arange(prefix, prefix + q, device=dev)
    out = {}
    for attn in ("sdpa", "cospec_sdpa"):
        model.set_attn_implementation(attn)
        out[attn] = model(input_ids=ids[:, prefix:], past_key_values=copy.deepcopy(cache),
                          position_ids=pos.expand(2, -1), cache_position=pos, use_cache=True).logits.float()
    model.set_attn_implementation("sdpa")
    return float((out["sdpa"] - out["cospec_sdpa"]).abs().max())


@torch.inference_mode()
def graph_matches_eager(model, B: int = 4, ctx: int = 256, q: int = 5) -> float:
    step = GraphedStep(model, B, ctx)
    graphed = step.logits(q).float().clone()
    eager = step._forward(*step.inputs[:2]).float()       # same static cache and inputs, run eagerly
    return float((graphed - eager).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    args = ap.parse_args()
    _, model = load_model(args.model)
    print(torch.cuda.get_device_name(), args.model)

    d_attn = attention_matches(model)
    model.set_attn_implementation("cospec_sdpa")
    d_graph = graph_matches_eager(model)
    print(f"cospec_sdpa vs HF sdpa, max logit difference: {d_attn:.4f}")
    print(f"CUDA graph vs eager (cospec_sdpa), max logit difference: {d_graph:.4f}")
    if not (d_attn < 0.5 and d_graph < 0.5):
        raise SystemExit("engine-style attention or graph replay does not match")

    variants = ["sdpa", "cospec_sdpa", "cospec_sdpa_nocausal"]
    print(f"\n{'B':>3} {'q':>2} {'ctx':>5} {'eager sdpa':>10} " + " ".join(f"{'graph ' + v:>27}" for v in variants))
    for B, q, ctx in [(1, 1, 512), (8, 5, 512), (32, 5, 1024), (64, 1, 1024), (64, 9, 1024), (64, 9, 4096)]:
        model.set_attn_implementation("sdpa")
        cache = zero_cache(model, B, ctx)
        row = [cuda_time_ms(lambda: verify_forward(model, cache, B, ctx, q))[0]]
        del cache
        torch.cuda.empty_cache()
        for v in variants:
            model.set_attn_implementation(v)
            step = GraphedStep(model, B, ctx)
            row.append(step.time_ms(q)[0])
            del step
            torch.cuda.empty_cache()
        print(f"{B:>3} {q:>2} {ctx:>5} {row[0]:10.2f} " + " ".join(f"{t:27.2f}" for t in row[1:]), flush=True)


if __name__ == "__main__":
    main()
