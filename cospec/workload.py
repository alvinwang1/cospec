"""Request streams: which traces to replay, when they arrive, where they're pinned.

Skew: production routers can't place every request freely (prefix-cache and
session affinity pin some to a rank). A fraction `affinity_frac` of requests
is pinned to a rank drawn from a Zipf(`zipf_s`) distribution over ranks, which
gives controllable, realistic load imbalance. The router places the rest.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from .step_model import CostModel
from .traces import TraceSet

MIXES = {
    "code": {"code": 0.8, "chat": 0.2},
    "chat": {"code": 0.2, "chat": 0.8},
    "even": {"code": 0.5, "chat": 0.5},
}


@dataclass
class Request:
    rid: int
    trace_idx: int
    arrival_ms: float
    pinned_rank: int = -1


def sample_trace_indices(traces: TraceSet, n: int, mix: dict[str, float], rng) -> np.ndarray:
    """Sample n requests (with replacement) so categories follow `mix`."""
    lens = traces.output_lens()
    eligible = lens >= 2
    present = {c: np.flatnonzero((traces.category == c) & eligible) for c in mix}
    present = {c: ix for c, ix in present.items() if len(ix)}
    if not present:
        raise ValueError(f"no traces match mix categories {list(mix)}")
    cats = list(present)
    p = np.array([mix[c] for c in cats], dtype=float)
    p /= p.sum()
    which = rng.choice(len(cats), size=n, p=p)
    return np.array([rng.choice(present[cats[w]]) for w in which])


def poisson_arrivals(n: int, rate_rps: float, rng) -> np.ndarray:
    return np.cumsum(rng.exponential(1000.0 / rate_rps, size=n))


def azure_arrivals(path: str, n: int, rate_rps: float, rng) -> np.ndarray:
    """Arrival pattern from an Azure LLM inference trace CSV, rescaled to rate.

    Keeps the trace's burstiness; starts at a random offset.
    """
    ts = []
    with open(path) as f:
        reader = csv.DictReader(f)
        col = next(c for c in reader.fieldnames if c.lower().startswith("timestamp"))
        for row in reader:
            ts.append(np.datetime64(row[col].strip()[:26]))
    ts = np.sort(np.array(ts))
    gaps = np.diff(ts).astype("timedelta64[us]").astype(float) / 1000.0  # ms
    if len(gaps) < n:
        gaps = np.tile(gaps, n // len(gaps) + 1)
    start = rng.integers(0, len(gaps) - n + 1)
    gaps = gaps[start:start + n]
    gaps *= (1000.0 / rate_rps) / max(gaps.mean(), 1e-9)
    return np.cumsum(gaps)


def zipf_pins(n: int, n_ranks: int, frac: float, s: float, rng) -> np.ndarray:
    pins = np.full(n, -1)
    if frac <= 0:
        return pins
    w = 1.0 / np.arange(1, n_ranks + 1) ** s
    w /= w.sum()
    pinned = rng.random(n) < frac
    pins[pinned] = rng.choice(n_ranks, size=pinned.sum(), p=w)
    return pins


def capacity_rps(traces: TraceSet, cost: CostModel, n_ranks: int, max_seqs: int) -> float:
    """Rough no-speculation decode capacity of the group, requests/s."""
    L = traces.output_lens()
    mean_L = float(L[L >= 2].mean())
    mean_ctx = float(traces.prompt_len.mean() + mean_L / 2)
    step_ms = float(cost.rank.time(max_seqs, max_seqs, max_seqs * mean_ctx))
    return n_ranks * max_seqs / (mean_L * step_ms) * 1000.0


def build_requests(traces: TraceSet, n: int, rate_rps: float, n_ranks: int, mix: dict[str, float],
                   affinity_frac: float = 0.0, zipf_s: float = 1.0, arrivals: str = "poisson",
                   seed: int = 0) -> list[Request]:
    rng = np.random.default_rng(seed)
    idx = sample_trace_indices(traces, n, mix, rng)
    if arrivals == "poisson":
        t = poisson_arrivals(n, rate_rps, rng)
    else:
        t = azure_arrivals(arrivals, n, rate_rps, rng)
    pins = zipf_pins(n, n_ranks, affinity_frac, zipf_s, rng)
    return [Request(i, int(idx[i]), float(t[i]), int(pins[i])) for i in range(n)]
