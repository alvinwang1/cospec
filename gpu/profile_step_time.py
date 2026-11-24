"""Profile verification step time T(batch, tokens, KV) on one GPU.

Writes a CSV the simulator's cost model is fit from:

  python gpu/profile_step_time.py --model Qwen/Qwen3-4B --drafter Qwen/Qwen3-0.6B \
      --out data/profiles/v100_qwen3_4b.csv
  python -m cospec.step_model fit data/profiles/v100_qwen3_4b.csv -o data/profiles/cost_model.json

Rows with kind=verify time one target forward of q tokens per sequence over
ctx cached tokens. Rows with kind=draft time one drafter step (q=1), which is
the per-depth drafting cost.

Steps are timed as CUDA graphs by default (--timing graph), as serving engines run decode, so
the profile measures GPU work rather than Python and kernel-launch overhead; --timing eager
times plain Hugging Face forwards.

Every measurement is appended as it finishes, so rerunning the same command
resumes where it stopped (--fresh starts over). Configs that ran out of memory
are listed in <out>.oom and not retried.
"""
from __future__ import annotations

import argparse

import torch

from checkpoint import Progress, open_csv_checkpoint
from common import GraphedStep, cuda_time_ms, load_model, verify_forward, zero_cache

FIELDS = ["kind", "n_seqs", "q", "ctx", "n_tokens", "kv_tokens", "ms", "ms_p90"]
OOM_FIELDS = ["kind", "n_seqs", "q", "ctx"]


def ints(s):
    return [int(x) for x in s.split(",")]


def key(kind, B, q, ctx):
    return (kind, int(B), int(q), int(ctx))


def grid(kind, batches, qs, ctxs, max_kv):
    return [key(kind, B, q, ctx) for ctx in ctxs for B in batches if B * ctx <= max_kv for q in qs]


def profile(model, kind, batches, qs, ctxs, max_kv, iters, out, oom, done, prog, timing):
    (w, fh), (ow, ofh) = out, oom
    for ctx in ctxs:
        for B in batches:
            if B * ctx > max_kv:
                continue
            todo = [q for q in qs if key(kind, B, q, ctx) not in done]
            if not todo:
                continue
            cache = None
            try:
                if timing == "graph":
                    cache = GraphedStep(model, B, ctx)
                else:
                    cache = zero_cache(model, B, ctx)
                while todo:
                    q = todo[0]
                    if timing == "graph":
                        med, p90 = cache.time_ms(q, iters=iters)
                    else:
                        med, p90 = cuda_time_ms(lambda: verify_forward(model, cache, B, ctx, q), iters=iters)
                    w.writerow(dict(kind=kind, n_seqs=B, q=q, ctx=ctx, n_tokens=B * q, kv_tokens=B * ctx,
                                    ms=round(med, 4), ms_p90=round(p90, 4)))
                    fh.flush()
                    todo.pop(0)
                    prog.step(f"{kind} B={B} q={q} ctx={ctx}: {med:.2f} ms")
            except torch.cuda.OutOfMemoryError:
                for q in todo:
                    ow.writerow(dict(kind=kind, n_seqs=B, q=q, ctx=ctx))
                ofh.flush()
                prog.step(f"{kind} B={B} ctx={ctx}: out of memory, skipping q={todo}", n=len(todo))
            del cache
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
    ap.add_argument("--fresh", action="store_true", help="discard an existing --out and start over")
    ap.add_argument("--attn", default="cospec_sdpa", choices=["cospec_sdpa", "cospec_sdpa_nocausal", "sdpa"],
                    help="attention kernel path (see gpu/common.py _engine_sdpa)")
    ap.add_argument("--timing", choices=["graph", "eager"], default="graph",
                    help="time CUDA-graph replays (like a serving engine) or eager forwards")
    args = ap.parse_args()

    rows, fh, w = open_csv_checkpoint(args.out, FIELDS, FIELDS, args.fresh)
    oom_rows, ofh, ow = open_csv_checkpoint(args.out + ".oom", OOM_FIELDS, OOM_FIELDS, args.fresh)
    done = {key(r["kind"], r["n_seqs"], r["q"], r["ctx"]) for r in rows + oom_rows}

    stages = [("verify", args.model, args.q, args.max_kv_tokens)]
    if args.drafter:
        stages.append(("draft", args.drafter, [1], args.max_kv_tokens * 4))
    jobs = {kind: grid(kind, args.batch, qs, args.ctx, max_kv) for kind, _, qs, max_kv in stages}
    prog = Progress(sum(map(len, jobs.values())), "profile", sum(k in done for j in jobs.values() for k in j))
    prog.start()

    for kind, name, qs, max_kv in stages:
        if all(k in done for k in jobs[kind]):
            print(f"{kind} ({name}): already profiled")
            continue
        _, model = load_model(name, attn=args.attn)
        print(f"{torch.cuda.get_device_name()}  {name}  layers={model.config.num_hidden_layers}", flush=True)
        profile(model, kind, args.batch, qs, args.ctx, max_kv, args.iters, (w, fh), (ow, ofh), done, prog,
                args.timing)
        del model
        torch.cuda.empty_cache()
    fh.close()
    ofh.close()
    print(f"profile complete: {args.out}")


if __name__ == "__main__":
    main()
