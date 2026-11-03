"""Profile verification step time T(batch, tokens, KV) on one GPU.

Writes a CSV the simulator's cost model is fit from:

  python gpu/profile_step_time.py --model Qwen/Qwen3-4B --drafter Qwen/Qwen3-0.6B \
      --out data/profiles/v100_qwen3_4b.csv
  python -m cospec.step_model fit data/profiles/v100_qwen3_4b.csv -o data/profiles/cost_model.json

Rows with kind=verify time one target forward of q tokens per sequence over
ctx cached tokens. Rows with kind=draft time one drafter step (q=1), which is
the per-depth drafting cost.
"""
from __future__ import annotations

import argparse
import csv
import os

import torch

from common import cuda_time_ms, load_model, verify_forward, zero_cache


def ints(s):
    return [int(x) for x in s.split(",")]


def profile(model, kind, batches, qs, ctxs, max_kv, iters, writer, fh):
    for ctx in ctxs:
        for B in batches:
            if B * ctx > max_kv:
                continue
            try:
                cache = zero_cache(model, B, ctx)
                for q in qs:
                    med, p90 = cuda_time_ms(lambda: verify_forward(model, cache, B, ctx, q), iters=iters)
                    row = dict(kind=kind, n_seqs=B, q=q, ctx=ctx, n_tokens=B * q, kv_tokens=B * ctx,
                               ms=round(med, 4), ms_p90=round(p90, 4))
                    writer.writerow(row)
                    fh.flush()
                    print(row, flush=True)
                del cache
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at {kind} B={B} ctx={ctx}; skipping")
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--drafter", default=None, help="also profile a draft model's single-token step")
    ap.add_argument("--batch", type=ints, default=ints("1,2,4,8,16,24,32,48,64"))
    ap.add_argument("--q", type=ints, default=ints("1,2,3,4,5,7,9"), help="tokens per sequence = 1 + k")
    ap.add_argument("--ctx", type=ints, default=ints("128,512,1024,2048,4096"))
    ap.add_argument("--max-kv-tokens", type=int, default=100_000,
                    help="skip B*ctx above this (KV for Qwen3-4B fp16 is ~144 KB/token)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = ["kind", "n_seqs", "q", "ctx", "n_tokens", "kv_tokens", "ms", "ms_p90"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        _, model = load_model(args.model)
        print(f"{torch.cuda.get_device_name()}  {args.model}  layers={model.config.num_hidden_layers}")
        profile(model, "verify", args.batch, args.q, args.ctx, args.max_kv_tokens, args.iters, w, fh)
        if args.drafter:
            del model
            torch.cuda.empty_cache()
            _, drafter = load_model(args.drafter)
            profile(drafter, "draft", args.batch, [1], args.ctx, args.max_kv_tokens * 4, args.iters, w, fh)


if __name__ == "__main__":
    main()
