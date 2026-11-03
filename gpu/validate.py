"""Single-GPU validation: replay a simulated schedule's batch shapes on the GPU.

For a sample of simulated group steps, run each virtual rank's verification
batch back to back, charge the step the max over ranks, and compare with what
the cost model predicted. This checks the step-time model on the batch shapes
the policies actually produce (not just the profiling grid).

  python experiments/run_sim.py --policy cospec --cost data/profiles/cost_model.json \
      --dump-steps results/steps_cospec.npz
  python gpu/validate.py --steps results/steps_cospec.npz --cost data/profiles/cost_model.json

HF forwards need the same q for every sequence in a batch, so a rank verifying
n_tokens over n_seqs runs q = round(n_tokens / n_seqs) per sequence, and each
sequence gets the rank's mean context. Idle ranks run a 1-token dummy pass.
Drafting and MoE time are excluded on both sides.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

from common import cuda_time_ms, load_model, verify_forward, zero_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cospec.step_model import CostModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--steps", required=True, help="npz from run_sim.py --dump-steps")
    ap.add_argument("--cost", required=True)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--ctx-bucket", type=int, default=64)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=None, help="per-step CSV")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.steps)
    n_seqs, n_tok, kv = z["n_seqs"], z["n_tokens"], z["kv_tokens"]
    cost = CostModel.load(args.cost)
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(len(n_seqs), size=min(args.n_steps, len(n_seqs)), replace=False))

    _, model = load_model(args.model)
    cache_key, cache = None, None
    rows = []
    for s in pick:
        pred_r = cost.rank.time(n_seqs[s], n_tok[s], kv[s])
        meas_r = np.zeros(len(pred_r))
        for r in range(len(pred_r)):
            B = int(n_seqs[s, r])
            if B == 0:
                B, q, ctx = 1, 1, args.ctx_bucket
            else:
                q = max(1, int(round(n_tok[s, r] / B)))
                ctx = max(args.ctx_bucket, int(round(kv[s, r] / B / args.ctx_bucket)) * args.ctx_bucket)
            if cache_key != (B, ctx):
                cache = None
                torch.cuda.empty_cache()
                cache, cache_key = zero_cache(model, B, ctx), (B, ctx)
            meas_r[r], _ = cuda_time_ms(lambda: verify_forward(model, cache, B, ctx, q),
                                        warmup=1, iters=args.iters)
        rows.append(dict(step=int(s), pred_max=float(pred_r.max()), meas_max=float(meas_r.max()),
                         pred_sum=float(pred_r.sum()), meas_sum=float(meas_r.sum())))
        print(rows[-1], flush=True)

    pm = np.array([r["pred_max"] for r in rows])
    mm = np.array([r["meas_max"] for r in rows])
    print(f"\n{len(rows)} steps: group step-time MAPE {100 * np.mean(np.abs(pm - mm) / mm):.1f}%, "
          f"total predicted / measured = {pm.sum() / mm.sum():.3f}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
