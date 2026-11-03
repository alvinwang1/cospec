"""Collect replayable acceptance traces (see cospec/traces.py for the format).

For each prompt: decode the target's greedy output y (with SD, which is
lossless and faster), then for *every* position p draft K tokens from the
drafter given prompt + y[:p], recording the drafter's confidence in each and
how many leading drafts match y[p:p+K]. Any budget policy can then be replayed
exactly in the simulator, so the GPU never has to rerun per policy.

  python gpu/collect_traces.py --prompts data/prompts/code.jsonl \
      --out data/traces/code_qwen3-0.6b.npz --K 8 --max-new-tokens 512

Saves every --save-every prompts and resumes from an existing --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from common import CachedLM, chat_ids, eos_ids, greedy_draft, load_model
from sd_decode import speculative_generate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cospec.traces import TraceSet  # noqa: E402


@torch.inference_mode()
def position_drafts(drafter, prompt: list[int], y: list[int], K: int):
    """conf [L, K] and acc_len [L] for every output position of y."""
    D = CachedLM(drafter)
    ctx = prompt + y
    P, L = len(prompt), len(y)
    D.extend(ctx[: P - 1])
    conf = np.zeros((L, K), dtype=np.float32)
    acc = np.zeros(L, dtype=np.int16)
    for p in range(L):
        # Cache holds ctx[:P+p-1]; feed the last context token, then draft.
        logits = D.extend([ctx[P + p - 1]])
        toks, confs = greedy_draft(D, logits, K)
        D.crop(P + p)
        conf[p] = confs
        ref = y[p:p + K]
        n = 0
        while n < len(ref) and toks[n] == ref[n]:
            n += 1
        acc[p] = n
    return conf, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--drafter", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompts", required=True, help="JSONL from gpu/build_prompts.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    with open(args.prompts) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    prompts = prompts[: args.limit]

    confs, accs, lens, plens, cats, ids = [], [], [], [], [], []
    if os.path.exists(args.out):
        prev = TraceSet.load(args.out)
        if prev.K != args.K:
            sys.exit(f"{args.out} has K={prev.K}, asked for K={args.K}")
        confs, accs = [prev.conf], [prev.acc_len]
        lens, plens, cats = list(prev.output_lens()), list(prev.prompt_len), list(prev.category)
        ids = list(prev.meta.get("ids", []))
        print(f"resuming after {len(ids)} prompts")
    done = set(ids)

    tok, target = load_model(args.target)
    _, drafter = load_model(args.drafter)
    eos = eos_ids(tok, target)

    def save():
        ts = TraceSet(
            conf=np.concatenate(confs) if confs else np.zeros((0, args.K), np.float32),
            acc_len=np.concatenate(accs) if accs else np.zeros(0, np.int16),
            offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            prompt_len=np.array(plens, dtype=np.int32), category=np.array(cats, dtype=str),
            meta=dict(target=args.target, drafter=args.drafter, K=args.K, ids=ids,
                      max_new_tokens=args.max_new_tokens, decoding="greedy"),
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        ts.save(args.out)

    t_start, n_new = time.time(), 0
    for item in prompts:
        if item["id"] in done:
            continue
        p_ids = chat_ids(tok, item["prompt"], args.max_prompt_tokens)
        y, _ = speculative_generate(target, drafter, p_ids, args.max_new_tokens, 4, eos)
        if len(y) < 2:
            continue
        conf, acc = position_drafts(drafter, p_ids, y, args.K)
        confs.append(conf)
        accs.append(acc)
        lens.append(len(y))
        plens.append(len(p_ids))
        cats.append(item["category"])
        ids.append(item["id"])
        n_new += 1
        el = time.time() - t_start
        print(f"[{len(ids)}] {item['id']}: prompt={len(p_ids)} out={len(y)} "
              f"mean_acc={acc[1:].mean():.2f}/{args.K}  ({el / n_new:.1f}s/prompt)", flush=True)
        if n_new % args.save_every == 0:
            save()
    save()
    print(f"saved {len(ids)} requests to {args.out}")


if __name__ == "__main__":
    main()
