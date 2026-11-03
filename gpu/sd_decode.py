"""Minimal lossless speculative decoding (greedy) and a correctness check.

Week-1 milestone: show SD output == plain greedy output on the V100, and
measure acceptance and speedup.

  python gpu/sd_decode.py --target Qwen/Qwen3-4B --drafter Qwen/Qwen3-0.6B -k 4 \
      --prompt "Write a Python function that checks if a number is prime."
"""
from __future__ import annotations

import argparse
import time

import torch

from common import CachedLM, chat_ids, eos_ids, greedy_draft, load_model


@torch.inference_mode()
def speculative_generate(target, drafter, prompt_ids: list[int], max_new: int, k: int, eos: set[int]):
    """Greedy SD. k = 0 is plain greedy decoding.

    Invariants: the target cache holds every committed token except the last;
    the drafter cache holds a prefix of that and catches up lazily.
    """
    T = CachedLM(target)
    D = CachedLM(drafter) if k > 0 else None
    seq = list(prompt_ids)
    out = [int(T.extend(seq)[-1].argmax())]
    seq.append(out[0])
    stats = dict(steps=0, drafted=0, accepted=0, confs=[], acc=[])

    while len(out) < max_new and out[-1] not in eos:
        S = len(seq)
        drafts, confs = [], []
        if k > 0:
            logits = D.extend(seq[D.len:])
            drafts, confs = greedy_draft(D, logits, k)
        pred = T.extend([seq[-1]] + drafts).argmax(-1).tolist()
        L = 0
        while L < len(drafts) and drafts[L] == pred[L]:
            L += 1
        new = drafts[:L] + [pred[L]]
        T.crop(S + L)
        if k > 0:
            D.crop(min(D.len, S + L))
        seq += new
        out += new
        stats["steps"] += 1
        stats["drafted"] += len(drafts)
        stats["accepted"] += L
        stats["confs"].append(confs)
        stats["acc"].append(L)

    for i, t in enumerate(out):
        if t in eos:
            out = out[: i + 1]
            break
    return out[:max_new], stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--drafter", default="Qwen/Qwen3-0.6B")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--prompt", default="Write a Python function that returns the n-th Fibonacci number.")
    args = ap.parse_args()

    tok, target = load_model(args.target)
    _, drafter = load_model(args.drafter)
    ids = chat_ids(tok, args.prompt)
    eos = eos_ids(tok, target)

    results = {}
    for k in (0, args.k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, st = speculative_generate(target, drafter, ids, args.max_new_tokens, k, eos)
        torch.cuda.synchronize()
        results[k] = (out, time.perf_counter() - t0, st)
        print(f"k={k}: {len(out)} tokens in {results[k][1]:.2f}s, {st['steps']} steps, "
              f"{len(out) / max(st['steps'], 1):.2f} tokens/step")

    ref, sd = results[0][0], results[args.k][0]
    if ref == sd:
        print("LOSSLESS: speculative output matches greedy output exactly")
    else:
        i = next((j for j, (a, b) in enumerate(zip(ref, sd)) if a != b), min(len(ref), len(sd)))
        print(f"MISMATCH at token {i} of {len(ref)}. Small fp16 differences between batched verification "
              f"and single-token decoding can flip near-tied argmaxes; check that this is rare across prompts.")
    st = results[args.k][2]
    print(f"acceptance rate {st['accepted'] / max(st['drafted'], 1):.3f}, "
          f"speedup {results[0][1] / results[args.k][1]:.2f}x")
    print("---\n" + tok.decode(sd, skip_special_tokens=True))


if __name__ == "__main__":
    main()
