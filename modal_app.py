"""Run CoSpec's GPU steps on Modal.

  pip install modal && modal setup                 # once
  modal run --detach modal_app.py::main                  # check, profile, prompts, traces, validate
  modal run modal_app.py::status                   # progress of every step, any time
  modal run --detach modal_app.py::main --steps profile  # only some steps (comma-separated)
  modal deploy modal_app.py && modal run modal_app.py::submit_traces   # traces (see below)
  modal run --detach modal_app.py::main --steps sweep --grid main   # simulator sweep on a 32-core CPU box
  modal volume get cospec-data data modal_out && modal volume get cospec-data results modal_out
  modal app stop cospec                            # stop everything (progress is kept)

Outputs go to the `cospec-data` Volume (data/ and results/). Every step checkpoints there and
resumes, so rerunning the same command after a failure, a timeout or a stop continues where
it left off. Failed containers are also retried automatically.

Traces run as independent jobs on the deployed app: `submit_traces` starts every unfinished
shard and returns at once, so nothing has to stay connected. Each shard resumes from its
checkpoint; the last one to finish merges the shards and starts validation (once). Redeploy
after changing code. (Waiting on all shards from one long-running coordinator proved fragile:
when that call was cancelled, the whole run stopped with it.)

GPUs. The step-time profile and the validation run on one H100 (pinned with "!", so Modal never
substitutes another type): that is the GPU the cost model, and the paper, describe, and both
steps must run on the same type. Trace collection decodes one prompt at a time, which is bound
by per-step overhead rather than GPU speed, so it runs on cheap GPUs (L4, A10G, T4
or L40S, whichever is free), sharded across many containers in parallel. Greedy traces don't depend on the GPU.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import modal

ROOT = Path(__file__).parent
REMOTE = "/root/cospec"
DATA, RESULTS = "/vol/data", "/vol/results"
TARGET, DRAFTER, K = "Qwen/Qwen3-4B", "Qwen/Qwen3-0.6B", 8
PROFILE_GPU = "H100!"   # profile + validation; "!" = exactly this type
# Trace shards: any of these cheap GPUs, whichever is free (L4s are sometimes scarce).
TRACE_GPU = ["L4", "A10G", "T4", "L40S"]
MAX_TRACE_CONTAINERS = 8    # leaves room for the H100 under a 10-GPU workspace limit
VAL_POLICIES = ("cospec", "fixed4")
N_VAL_STEPS = 200
# Step timing: CUDA-graph replays (as engines run decode) with fused SDPA that handles GQA in the
# kernel. The no-causal variant skips masking among the <= 9 new tokens: same cost as an engine's
# causal verify kernel, without PyTorch's slower masked path (see gpu/graph_check.py).
TIMING_ARGS = ("--timing", "graph", "--attn", "cospec_sdpa_nocausal")
KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2   # Qwen3-4B fp16: layers * kv heads * head dim * (K, V) * 2 bytes
STEPS = ("check", "profile", "prompts", "traces", "validate", "sweep")

app = modal.App("cospec")
data = modal.Volume.from_name("cospec-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("cospec-hf-cache", create_if_missing=True)
locks = modal.Dict.from_name("cospec-locks", create_if_missing=True)   # start-once flags
VOLUMES = {"/vol": data, "/hf": hf_cache}

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Versions the test suite passes with.
    .pip_install("torch==2.15.0", "transformers==5.24.0", "huggingface_hub==2.4.0", "datasets==5.2.0",
                 "numpy==2.5.5", "matplotlib==3.11.2", "pytest==9.2.0")
    .env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(ROOT, REMOTE, ignore=[".git", ".venv", ".pytest_cache", "**/__pycache__", "data", "results",
                                         "figs", "paper", "colab", "modal_out", "*.zip"])
)
RETRIES = modal.Retries(max_retries=3, initial_delay=30.0)


# ---------------------------------------------------------------------------
# Helpers (run inside containers)

def sh(*cmd):
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REMOTE, check=True)


def py(*args):
    sh(sys.executable, *args)


def offline():
    """Models come from the cache Volume; don't hit the Hub from many containers at once."""
    os.environ["HF_HUB_OFFLINE"] = "1"


def gpu_info() -> tuple[str, int]:
    """(tag like 'h100-80gb-hbm3', total memory in bytes) of this container's GPU."""
    import re
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.splitlines()[0]
    name, mib = [x.strip() for x in out.split(",")]
    tag = re.sub(r"[^a-z0-9]+", "-", re.sub(r"nvidia|tesla", "", name.lower())).strip("-")
    return tag, int(mib) * 2**20


def cost_path(tag):
    return f"{DATA}/profiles/cost_model_{tag}.json"


def shard_paths(cat, i):
    return f"{DATA}/prompts/shards/{cat}_{i:02d}.jsonl", f"{DATA}/traces/shards/{cat}_{i:02d}.npz"


def count_lines(path):
    try:
        with open(path) as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


def trace_done(path) -> int:
    """Prompts finished (recorded or skipped) in a trace checkpoint."""
    if not os.path.exists(path):
        return 0
    import numpy as np
    meta = json.loads(str(np.load(path)["meta"]))
    return len(meta.get("ids", [])) + len(meta.get("skipped", []))


def unfinished() -> list[tuple[str, int]]:
    """Shards whose trace checkpoint doesn't yet cover all of their prompts."""
    out = []
    for p in sorted(glob.glob(f"{DATA}/prompts/shards/*_*.jsonl")):
        cat, i = Path(p).stem.rsplit("_", 1)
        if count_lines(p) and trace_done(shard_paths(cat, int(i))[1]) < count_lines(p):
            out.append((cat, int(i)))
    return out


# ---------------------------------------------------------------------------
# Steps

@app.function(image=image, volumes=VOLUMES, timeout=3600)
def fetch_models():
    from huggingface_hub import snapshot_download
    for m in (TARGET, DRAFTER):
        print(m, "->", snapshot_download(m), flush=True)
    hf_cache.commit()


@app.function(image=image, volumes=VOLUMES, gpu=TRACE_GPU, timeout=1800)
def check():
    """Unit tests plus the lossless-SD check on a real GPU."""
    offline()
    py("-m", "pytest", "-q")
    py("gpu/sd_decode.py", "--target", TARGET, "--drafter", DRAFTER, "-k", "4")


@app.function(image=image, volumes=VOLUMES, gpu=PROFILE_GPU, timeout=1800)
def graph_check():
    """CUDA-graph step timing works on the profile GPU and matches eager logits."""
    offline()
    py("gpu/graph_check.py", "--model", TARGET)


@app.function(image=image, volumes=VOLUMES, gpu=PROFILE_GPU, timeout=4 * 3600, retries=RETRIES)
def profile() -> str:
    offline()
    tag, mem = gpu_info()
    cost = cost_path(tag)
    if os.path.exists(cost):
        print(f"already have {cost}")
        return tag
    # Enough KV to cover the whole grid (64 seqs x 4096 ctx) when memory allows.
    max_kv = int(min(270_000, (mem - 8.1e9 - 4e9) / KV_BYTES_PER_TOKEN))
    out = f"{DATA}/profiles/{tag}_qwen3_4b.csv"
    print(f"{tag}: {mem / 2**30:.0f} GiB, profiling up to {max_kv} KV tokens", flush=True)
    py("gpu/profile_step_time.py", "--model", TARGET, "--drafter", DRAFTER, "--max-kv-tokens", max_kv, "--out", out,
       *TIMING_ARGS)
    py("-m", "cospec.step_model", "fit", out, "-o", cost)
    data.commit()
    return tag


@app.function(image=image, volumes=VOLUMES, timeout=3600)
def build_prompts(per_dataset: int):
    d = f"{DATA}/prompts"
    if count_lines(f"{d}/code.jsonl") and count_lines(f"{d}/chat.jsonl"):
        print("prompts already built; delete data/prompts on the Volume to rebuild")
    else:
        py("gpu/build_prompts.py", "--out-dir", d, "--per-dataset", per_dataset)
        data.commit()


@app.function(image=image, volumes=VOLUMES, timeout=600)
def make_shards(n_shards: int) -> list[tuple[str, int]]:
    """Split each category's prompts into shards (by a hash of the prompt id), once."""
    layout = f"{DATA}/traces/shards/layout.json"
    if os.path.exists(layout):
        old = json.load(open(layout))["n_shards"]
        if old != n_shards:
            raise ValueError(f"trace collection was started with --shards {old}; rerun with that value")
    else:
        os.makedirs(os.path.dirname(layout), exist_ok=True)
        os.makedirs(f"{DATA}/prompts/shards", exist_ok=True)
        for cat in ("code", "chat"):
            with open(f"{DATA}/prompts/{cat}.jsonl") as f:
                rows = [line for line in f if line.strip()]
            parts = [[] for _ in range(n_shards)]
            for line in rows:
                parts[zlib.crc32(json.loads(line)["id"].encode()) % n_shards].append(line)
            for i, part in enumerate(parts):
                with open(shard_paths(cat, i)[0], "w") as f:
                    f.writelines(part)
        json.dump({"n_shards": n_shards}, open(layout, "w"))
        data.commit()
    return [(cat, i) for cat in ("code", "chat") for i in range(n_shards) if count_lines(shard_paths(cat, i)[0])]


@app.function(image=image, volumes=VOLUMES, gpu=TRACE_GPU, timeout=8 * 3600, retries=RETRIES,
              max_containers=MAX_TRACE_CONTAINERS)
def collect_shard(cat: str, i: int, max_new_tokens: int):
    offline()
    prompts, out = shard_paths(cat, i)
    py("gpu/collect_traces.py", "--target", TARGET, "--drafter", DRAFTER, "--prompts", prompts, "--out", out,
       "--K", K, "--max-new-tokens", max_new_tokens, "--save-minutes", 3)
    data.commit()
    # Publish this shard, then look at everyone else's. Of shards finishing together, at least
    # the last to commit sees all of them done; merging twice is harmless (atomic, identical).
    data.reload()
    left = unfinished()
    if left:
        print(f"{len(left)} shards still running")
        return
    merge_traces.remote()
    if locks.put(f"validate-started-{max_new_tokens}", True, skip_if_exists=True):
        print("all traces merged; starting validation")
        validate.spawn()


@app.function(image=image, volumes=VOLUMES, timeout=600)
def unfinished_shards() -> list[tuple[str, int]]:
    return unfinished()


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def merge_traces():
    """Concatenate finished shards into data/traces/{code,chat}.npz."""
    sys.path.insert(0, REMOTE)
    from cospec.traces import TraceSet
    for cat in ("code", "chat"):
        prompt_files = sorted(glob.glob(f"{DATA}/prompts/shards/{cat}_*.jsonl"))
        missing = [p for p in prompt_files
                   if count_lines(p) and trace_done(p.replace("/prompts/", "/traces/").replace(".jsonl", ".npz"))
                   < count_lines(p)]
        if missing:
            raise RuntimeError(f"unfinished {cat} shards: {missing}; rerun --steps traces")
        sets = [TraceSet.load(f) for f in sorted(glob.glob(f"{DATA}/traces/shards/{cat}_*.npz"))]
        merged = TraceSet.concat(sets)
        merged.meta = {k: v for k, v in sets[0].meta.items() if k not in ("ids", "skipped")}
        merged.meta.update(ids=[x for s in sets for x in s.meta.get("ids", [])],
                           skipped=[x for s in sets for x in s.meta.get("skipped", [])], shards=len(sets))
        merged.save(f"{DATA}/traces/{cat}.npz")
        print(f"{cat}: {merged.n_requests} requests, {len(merged.acc_len)} positions from {len(sets)} shards")
    data.commit()


@app.function(image=image, volumes=VOLUMES, gpu=PROFILE_GPU, timeout=4 * 3600, retries=RETRIES)
def validate(n_steps: int = N_VAL_STEPS):
    offline()
    tag, _ = gpu_info()
    cost = cost_path(tag)
    if not os.path.exists(cost):
        raise RuntimeError(f"no cost model for {tag}; run the profile step on this GPU type first")
    traces = f"{DATA}/traces/code.npz,{DATA}/traces/chat.npz"
    os.makedirs(RESULTS, exist_ok=True)
    for pol in VAL_POLICIES:
        steps = f"{RESULTS}/steps_{pol}_{tag}.npz"
        if not os.path.exists(steps):
            py("experiments/run_sim.py", "--traces", traces, "--cost", cost, "--policy", pol, "--n-ranks", 8,
               "--requests-per-rank", 60, "--dump-steps", steps, "--out", f"{RESULTS}/run_sim_{tag}.jsonl")
            data.commit()
        py("gpu/validate.py", "--model", TARGET, "--steps", steps, "--cost", cost, "--n-steps", n_steps,
           "--out", f"{RESULTS}/validate_{pol}_{tag}.csv", *TIMING_ARGS)
    data.commit()


@app.function(image=image, volumes=VOLUMES, cpu=32, memory=32768, timeout=24 * 3600, retries=RETRIES)
def sweep(grid: str, gpu_tag: str = ""):
    """Simulator sweep (CPU only) on the real traces and the profiled cost model."""
    costs = sorted(glob.glob(cost_path("*")))
    if gpu_tag:
        costs = [cost_path(gpu_tag)]
    if len(costs) != 1 or not os.path.exists(costs[0]):
        raise RuntimeError(f"pick a cost model with --gpu-tag; found {costs}")
    tag = Path(costs[0]).stem.removeprefix("cost_model_")
    py("experiments/sweep.py", "--grid", grid, "--traces", f"{DATA}/traces/code.npz,{DATA}/traces/chat.npz",
       "--cost", costs[0], "--out", f"{RESULTS}/{grid}_{tag}.jsonl", "-j", 32)
    data.commit()


@app.function(image=image, volumes=VOLUMES, timeout=600)
def status():
    import csv

    def rows(path):
        try:
            with open(path, newline="") as f:
                return sum(1 for _ in csv.DictReader(f))
        except FileNotFoundError:
            return 0

    mark = lambda ok: "✓" if ok else "…"
    print("CoSpec on Modal (volume cospec-data)")
    for prof in sorted(glob.glob(f"{DATA}/profiles/*_qwen3_4b.csv")):
        tag = Path(prof).name.removesuffix("_qwen3_4b.csv")
        done = os.path.exists(cost_path(tag))
        print(f" {mark(done)} profile {tag}: {rows(prof)} measured, {rows(prof + '.oom')} out of memory, "
              f"cost model {'fitted' if done else 'not fitted'}")
    n = {c: count_lines(f"{DATA}/prompts/{c}.jsonl") for c in ("code", "chat")}
    print(f" {mark(all(n.values()))} prompts: {n['code']} code, {n['chat']} chat")
    for cat in ("code", "chat"):
        shards = sorted(glob.glob(f"{DATA}/prompts/shards/{cat}_*.jsonl"))
        tot = sum(count_lines(p) for p in shards)
        done = sum(trace_done(p.replace("/prompts/", "/traces/").replace(".jsonl", ".npz")) for p in shards)
        merged = os.path.exists(f"{DATA}/traces/{cat}.npz")
        print(f" {mark(merged)} traces {cat}: {done}/{tot} prompts over {len(shards)} shards"
              f"{', merged' if merged else ''}")
    for path in sorted(glob.glob(f"{RESULTS}/validate_*.csv")):
        print(f" {mark(rows(path) >= N_VAL_STEPS)} {Path(path).stem}: {rows(path)}/{N_VAL_STEPS} steps")
    for path in sorted(glob.glob(f"{RESULTS}/*.jsonl")):
        print(f"   {Path(path).name}: {count_lines(path)} runs")


# ---------------------------------------------------------------------------

@app.function(image=image, timeout=24 * 3600)
def pipeline(todo: list[str], per_dataset: int, max_new_tokens: int, shards: int, grid: str, gpu_tag: str):
    """Runs the steps in order inside Modal, so the run survives the laptop going to sleep."""
    if set(todo) & {"check", "profile", "traces", "validate"}:
        fetch_models.remote()
    if "check" in todo:
        check.remote()
    prof = profile.spawn() if "profile" in todo else None      # runs alongside trace collection
    if "prompts" in todo:
        build_prompts.remote(per_dataset)
    if "traces" in todo:
        jobs = make_shards.remote(shards)
        print(f"collecting traces in {len(jobs)} shards on {TRACE_GPU}", flush=True)
        results = collect_shard.starmap([(c, i, max_new_tokens) for c, i in jobs], return_exceptions=True)
        failed = [j for j, r in zip(jobs, results) if isinstance(r, BaseException)]
        if failed:
            raise RuntimeError(f"trace shards failed after retries: {failed}; rerun --steps traces to resume")
        merge_traces.remote()
    if prof is not None:
        print("profiled on", prof.get(), flush=True)
    if "validate" in todo:
        validate.remote()
    if "sweep" in todo:
        sweep.remote(grid, gpu_tag)
    status.remote()


@app.local_entrypoint()
def submit_traces(max_new_tokens: int = 512):
    """Start every unfinished trace shard on the deployed app (`modal deploy modal_app.py` first)."""
    pending = modal.Function.from_name("cospec", "unfinished_shards").remote()
    if not pending:
        print("all shards complete; run merge/validate if they haven't happened: "
              "modal run modal_app.py::main --steps validate")
        return
    shard = modal.Function.from_name("cospec", "collect_shard")
    for cat, i in pending:
        shard.spawn(cat, i, max_new_tokens)
    print(f"started {len(pending)} shards: {pending}. They run on the deployed app; check with "
          f"`modal run modal_app.py::status`. The last one merges and starts validation.")


@app.local_entrypoint()
def main(steps: str = "check,profile,prompts,traces,validate", per_dataset: int = 150,
         max_new_tokens: int = 512, shards: int = 8, grid: str = "main", gpu_tag: str = ""):
    todo = [s.strip() for s in steps.split(",") if s.strip()]
    bad = set(todo) - set(STEPS)
    if bad:
        raise SystemExit(f"unknown steps {sorted(bad)}; choose from {STEPS}")
    pipeline.remote(todo, per_dataset, max_new_tokens, shards, grid, gpu_tag)
    print("\nDownload: modal volume get cospec-data data modal_out && modal volume get cospec-data results modal_out")
