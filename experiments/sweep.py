"""Run a grid of simulated experiments in parallel; results go to a JSONL file.

  python experiments/sweep.py --grid main --out results/main.jsonl -j 16
  python experiments/sweep.py --grid skew --traces data/traces/code.npz,data/traces/chat.npz \
      --cost data/profiles/cost_model.json --out results/skew.jsonl

Finished configs already in --out are skipped, so an interrupted sweep resumes.
"""
import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cospec.experiment import run_experiment  # noqa: E402

# system name -> (budget policy, router)
SYSTEMS = {
    "nosd": ("nosd", "least_loaded"),
    "fixed4": ("fixed4", "least_loaded"),
    "perrank": ("perrank", "least_loaded"),
    "cospec_tput": ("cospec_tput", "least_loaded"),
    "cospec_d1": ("cospec", "least_loaded"),
    "cospec_d1d2": ("cospec", "spec_aware"),
    "oracle": ("oracle", "spec_aware"),
}

GRIDS = {
    # The proposal's main grid: systems x workloads x ranks x loads x seeds.
    "main": dict(system=list(SYSTEMS), mix=["code", "chat"], n_ranks=[4, 8, 16, 64],
                 load=[0.6, 1.0, 1.4], seed=[0, 1, 2]),
    # How gains depend on imbalance.
    "skew": dict(system=["perrank", "cospec_d1", "cospec_d1d2", "oracle"], mix=["code", "chat"], n_ranks=[16],
                 load=[1.0], affinity_frac=[0.0, 0.2, 0.4, 0.6, 0.8], seed=[0, 1, 2]),
    # Sensitivity to the MoE / all-to-all term.
    "moe": dict(system=["perrank", "cospec_d1", "cospec_tput"], mix=["code"], n_ranks=[16], load=[1.0],
                moe_per_token_ms=[0.0, 0.05, 0.1, 0.2], seed=[0, 1, 2]),
    "smoke": dict(system=list(SYSTEMS), mix=["code"], n_ranks=[4], load=[1.0], seed=[0],
                  requests_per_rank=[40]),
}


def expand(grid: dict, extra: dict) -> list[dict]:
    keys = list(grid)
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, vals))
        policy, router = SYSTEMS[cfg.pop("system")]
        out.append({**extra, **cfg, "policy": policy, "router": router})
    return out


def key(cfg: dict) -> str:
    return json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True)


def work(cfg):
    try:
        res = run_experiment(cfg)
    except Exception as e:  # keep the sweep going; record the failure
        res = {"error": repr(e)}
    res["config"] = cfg
    res["system"] = next(s for s, pr in SYSTEMS.items() if pr == (cfg["policy"], cfg["router"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=list(GRIDS), default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--traces", default=None, help="comma-separated trace .npz files (default: synthetic)")
    ap.add_argument("--cost", default=None, help="cost model json (default: built-in V100 guess)")
    ap.add_argument("--arrivals", default=None, help="Azure trace CSV for arrival times (default: Poisson)")
    ap.add_argument("--tpot-slo-ms", type=float, default=None)
    args = ap.parse_args()

    extra = {k: v for k, v in dict(traces=args.traces, cost=args.cost, arrivals=args.arrivals,
                                   tpot_slo_ms=args.tpot_slo_ms).items() if v is not None}
    cfgs = expand(GRIDS[args.grid], extra)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {key(json.loads(line)["config"]) for line in f if line.strip()}
    todo = [c for c in cfgs if key(c) not in done]
    print(f"{len(cfgs)} configs, {len(todo)} to run")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Big rank counts first so the long jobs don't straggle at the end.
    todo.sort(key=lambda c: -c.get("n_ranks", 8))
    with Pool(args.jobs) as pool, open(args.out, "a") as f:
        for i, res in enumerate(pool.imap_unordered(work, todo), 1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            tag = res.get("error") or f"goodput={res['goodput_rps']:.1f} rps"
            print(f"[{i}/{len(todo)}] {res['system']} R={res['config'].get('n_ranks')} "
                  f"load={res['config'].get('load')} mix={res['config'].get('mix')}: {tag}", flush=True)


if __name__ == "__main__":
    main()
