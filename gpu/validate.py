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
import os
import sys

import numpy as np
import torch

from checkpoint import Progress, open_csv_checkpoint
from common import GraphedStep, cuda_time_ms, load_model, verify_forward, zero_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cospec.step_model import CostModel  # noqa: E402


FIELDS = ["step", "status", "pred_max", "meas_max", "pred_sum", "meas_sum"]


def measure(model, n_seqs, n_tok, kv, args, state):
    """Measured verification time of each rank in one simulated step."""
    meas = np.zeros(len(n_seqs))
    for r in range(len(n_seqs)):
        B = int(n_seqs[r])
        if B == 0:
            B, q, ctx = 1, 1, args.ctx_bucket
        else:
            q = max(1, int(round(n_tok[r] / B)))
            ctx = max(args.ctx_bucket, int(round(kv[r] / B / args.ctx_bucket)) * args.ctx_bucket)
        if state.get("key") != (B, ctx):
            state.clear()
            torch.cuda.empty_cache()
            state["cache"] = GraphedStep(model, B, ctx) if args.timing == "graph" \
                else zero_cache(model, B, ctx)
            state["key"] = (B, ctx)
        cache = state["cache"]
        if args.timing == "graph":
            meas[r], _ = cache.time_ms(q, warmup=1, iters=args.iters)
        else:
            meas[r], _ = cuda_time_ms(lambda: verify_forward(model, cache, B, ctx, q), warmup=1, iters=args.iters)
    return meas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--steps", required=True, help="npz from run_sim.py --dump-steps")
    ap.add_argument("--cost", required=True)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--ctx-bucket", type=int, default=64)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=None, help="per-step CSV; a rerun with the same --out resumes from it")
    ap.add_argument("--fresh", action="store_true", help="discard an existing --out and start over")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="cospec_sdpa", choices=["cospec_sdpa", "cospec_sdpa_nocausal", "sdpa"],
                    help="attention kernel path (see gpu/common.py _engine_sdpa)")
    ap.add_argument("--timing", choices=["graph", "eager"], default="graph",
                    help="must match how the cost model's profile was timed")
    args = ap.parse_args()

    z = np.load(args.steps)
    n_seqs, n_tok, kv = z["n_seqs"], z["n_tokens"], z["kv_tokens"]
    cost = CostModel.load(args.cost)
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(len(n_seqs), size=min(args.n_steps, len(n_seqs)), replace=False))

    rows, fh, w = open_csv_checkpoint(args.out, FIELDS, ["step", "status"], args.fresh) if args.out \
        else ([], None, None)
    done = {int(r["step"]) for r in rows}
    todo = [int(s) for s in pick if int(s) not in done]
    prog = Progress(len(pick), "validate", len(pick) - len(todo))
    prog.start()

    if todo:
        _, model = load_model(args.model, attn=args.attn)
        state = {}
        for s in todo:
            pred = cost.rank.time(n_seqs[s], n_tok[s], kv[s])
            try:
                meas = measure(model, n_seqs[s], n_tok[s], kv[s], args, state)
                row = dict(step=s, status="ok", pred_max=float(pred.max()), meas_max=float(meas.max()),
                           pred_sum=float(pred.sum()), meas_sum=float(meas.sum()))
                msg = f"step {s}: predicted {pred.max():.2f} ms, measured {meas.max():.2f} ms"
            except torch.cuda.OutOfMemoryError:
                # Smaller GPUs (e.g. a 16 GB T4) can't hold the biggest simulated ranks' KV.
                state.clear()
                torch.cuda.empty_cache()
                row = dict(step=s, status="oom", pred_max=float(pred.max()), meas_max="",
                           pred_sum=float(pred.sum()), meas_sum="")
                msg = f"step {s}: out of memory, skipped"
            rows.append(row)
            if w:
                w.writerow(row)
                fh.flush()
            prog.step(msg)
    if fh:
        fh.close()

    ok = [r for r in rows if r["status"] == "ok"]
    n_oom = len(rows) - len(ok)
    if not ok:
        sys.exit(f"every sampled step ran out of GPU memory ({n_oom}); simulate with a smaller --max-seqs")
    if n_oom:
        print(f"skipped {n_oom} of {len(rows)} steps that ran out of GPU memory")
    pm = np.array([float(r["pred_max"]) for r in ok])
    mm = np.array([float(r["meas_max"]) for r in ok])
    print(f"\n{len(ok)} steps: group step-time MAPE {100 * np.mean(np.abs(pm - mm) / mm):.1f}%, "
          f"total predicted / measured = {pm.sum() / mm.sum():.3f}")

if __name__ == "__main__":
    main()
