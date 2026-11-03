"""One simulated experiment from a flat config dict (shared by CLI and sweeps)."""
from __future__ import annotations

from functools import lru_cache

from .allocator import make_policy
from .router import make_router
from .simulator import SimConfig, Simulator
from .step_model import CostModel, StepModel
from .traces import Calibrator, TraceSet, category_priors, synthetic_traces
from .workload import MIXES, build_requests, capacity_rps

DEFAULTS = dict(
    traces="synthetic",       # comma-separated .npz paths, or "synthetic"
    synthetic_requests=600,
    cost=None,                # cost model json (from `python -m cospec.step_model fit`)
    n_ranks=8,
    policy="cospec",          # nosd | fixedK | perrank | cospec | cospec_tput | oracle | perrank_oracle
    router="least_loaded",    # round_robin | least_loaded | least_kv | spec_aware
    router_prior="category",  # category | global
    load=0.6,                 # offered load as a fraction of no-SD capacity
    mix="even",
    affinity_frac=0.3,
    zipf_s=1.0,
    arrivals="poisson",       # poisson, or path to an Azure trace CSV
    requests_per_rank=150,
    seed=0,
    train_frac=0.3,
    calibrate=True,
    max_seqs=64,
    max_tokens=512,
    K=8,
    tpot_slo_ms=50.0,
    queue_slo_ms=2000.0,
    moe_base_ms=None,         # override cost model MoE terms
    moe_per_token_ms=None,
    draft_per_depth_ms=None,
    dump_steps=None,
)


@lru_cache(maxsize=8)
def _load_traces(spec: str, synthetic_requests: int, seed: int):
    if spec == "synthetic":
        tr = synthetic_traces(synthetic_requests, seed=1000 + seed)
    else:
        tr = TraceSet.concat([TraceSet.load(p) for p in spec.split(",")])
    return tr


@lru_cache(maxsize=8)
def _prepared(spec: str, synthetic_requests: int, seed: int, train_frac: float, calibrate: bool):
    tr = _load_traces(spec, synthetic_requests, seed)
    train, test = tr.split(train_frac, seed=seed)
    calib = Calibrator.fit(train) if calibrate else Calibrator.identity()
    return train, test, calib, category_priors(train)


def load_cost(cfg) -> CostModel:
    cost = CostModel.load(cfg["cost"]) if cfg["cost"] else CostModel(rank=StepModel())
    for k in ("moe_base_ms", "moe_per_token_ms", "draft_per_depth_ms"):
        if cfg.get(k) is not None:
            setattr(cost, k, float(cfg[k]))
    return cost


def run_experiment(overrides: dict) -> dict:
    cfg = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    _, test, calib, priors = _prepared(cfg["traces"], cfg["synthetic_requests"], cfg["seed"],
                                       cfg["train_frac"], cfg["calibrate"])
    cost = load_cost(cfg)
    R = cfg["n_ranks"]
    K = min(cfg["K"], test.K)
    rate = cfg["load"] * capacity_rps(test, cost, R, cfg["max_seqs"])
    reqs = build_requests(test, cfg["requests_per_rank"] * R, rate, R, MIXES[cfg["mix"]],
                          cfg["affinity_frac"], cfg["zipf_s"], cfg["arrivals"], seed=cfg["seed"])

    sim_cfg = SimConfig(n_ranks=R, max_seqs=cfg["max_seqs"], max_tokens=cfg["max_tokens"], K=K,
                        tpot_slo_ms=cfg["tpot_slo_ms"], queue_slo_ms=cfg["queue_slo_ms"],
                        record_steps=bool(cfg["dump_steps"]))
    policy = make_policy(cfg["policy"], K, cfg["max_tokens"])
    router = make_router(cfg["router"], cost, policy)
    sim = Simulator(sim_cfg, cost, policy, router, test, reqs, calib, priors, cfg["router_prior"])
    out = sim.run()
    if cfg["dump_steps"]:
        sim.dump_steps(cfg["dump_steps"])
    out.update({k: cfg[k] for k in ("load", "mix", "affinity_frac", "zipf_s", "seed", "traces")})
    out["target_rps"] = rate
    return out
