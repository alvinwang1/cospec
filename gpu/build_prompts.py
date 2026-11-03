"""Build prompt files for trace collection (needs `datasets`, `huggingface_hub`).

  python gpu/build_prompts.py --out-dir data/prompts --per-dataset 150

Writes code.jsonl (HumanEval + MBPP) and chat.jsonl (ShareGPT + MT-Bench),
one {"id", "dataset", "category", "prompt"} object per line.
"""
from __future__ import annotations

import argparse
import json
import os
import random


def humaneval():
    from datasets import load_dataset
    for r in load_dataset("openai/openai_humaneval", split="test"):
        yield r["task_id"], f"Complete the following Python function.\n\n```python\n{r['prompt']}```"


def mbpp():
    from datasets import load_dataset
    for r in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        yield f"mbpp/{r['task_id']}", f"{r['text']}\nYour code should pass this test:\n{r['test_list'][0]}"


def mt_bench():
    from datasets import load_dataset
    for i, r in enumerate(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")):
        yield f"mtbench/{r.get('prompt_id', i)}", r["prompt"][0]


def sharegpt(max_chars=6000):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered",
                           "ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    for conv in data:
        turns = conv.get("conversations") or []
        if turns and turns[0].get("from") == "human" and 16 < len(turns[0]["value"]) < max_chars:
            yield f"sharegpt/{conv['id']}", turns[0]["value"]


SOURCES = {"code": [("humaneval", humaneval), ("mbpp", mbpp)],
           "chat": [("sharegpt", sharegpt), ("mt_bench", mt_bench)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/prompts")
    ap.add_argument("--per-dataset", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    for cat, sources in SOURCES.items():
        rows = []
        for name, fn in sources:
            items = list(fn())
            rng.shuffle(items)
            rows += [dict(id=i, dataset=name, category=cat, prompt=p) for i, p in items[: args.per_dataset]]
            print(f"{name}: {min(len(items), args.per_dataset)} of {len(items)}")
        rng.shuffle(rows)
        path = os.path.join(args.out_dir, f"{cat}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} prompts to {path}")


if __name__ == "__main__":
    main()
