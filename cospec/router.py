"""Decode routers: pick a rank for each arriving request.

Baselines balance request counts or KV. The speculation-aware router (D2)
scores each rank by the group goodput the budget policy would reach after the
request joins it, so high-acceptance requests go where there is slack to
speculate and speculation-heavy load is counted as load.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .allocator import Policy, RankBatch, evaluate
from .step_model import CostModel


@dataclass
class RankView:
    """What a router may look at: per-rank state at routing time."""
    surv: list[np.ndarray]     # per rank, [n_r, K] survival estimates of resident requests
    cap: list[np.ndarray]      # per rank, [n_r]
    kv: np.ndarray             # [R] total context on each rank (running + queued)
    n_running: np.ndarray      # [R]
    n_queued: np.ndarray       # [R]
    max_seqs: int


class Router:
    name = "router"

    def route(self, view: RankView, req_surv: np.ndarray, req_cap: int, req_kv: int) -> int:
        raise NotImplementedError


class RoundRobin(Router):
    name = "round_robin"

    def __init__(self):
        self.next = 0

    def route(self, view, req_surv, req_cap, req_kv):
        r = self.next % len(view.kv)
        self.next += 1
        return r


class LeastLoaded(Router):
    """Fewest resident requests, ties broken by KV (the usual production rule)."""
    name = "least_loaded"

    def route(self, view, req_surv, req_cap, req_kv):
        load = view.n_running + view.n_queued
        return int(np.lexsort((view.kv, load))[0])


class LeastKV(Router):
    name = "least_kv"

    def route(self, view, req_surv, req_cap, req_kv):
        return int(np.argmin(view.kv))


class SpecAware(Router):
    """D2: route to the rank that maximizes post-allocation group goodput.

    Only ranks with a free batch slot are considered (falling back to the
    shortest queue). To bound cost, at most `max_candidates` of them, the ones
    with the lowest current step time, are scored exactly.
    """
    name = "spec_aware"

    def __init__(self, cost: CostModel, policy: Policy, max_candidates: int = 8):
        self.cost, self.policy = cost, policy
        self.max_candidates = max_candidates

    def route(self, view, req_surv, req_cap, req_kv):
        R = len(view.kv)
        resident = view.n_running + view.n_queued
        free = np.flatnonzero(resident < view.max_seqs)
        if len(free) == 0:
            return int(np.lexsort((view.kv, view.n_queued))[0])

        K = req_surv.shape[-1]
        batches = [RankBatch(view.surv[r].reshape(-1, K), view.cap[r], float(view.kv[r])) for r in range(R)]
        n = np.array([b.n_seqs for b in batches])
        t_now = self.cost.rank.time(n, n, view.kv)
        cands = free[np.argsort(t_now[free], kind="stable")[: self.max_candidates]]

        best, best_score = int(cands[0]), -np.inf
        for r in cands:
            trial = list(batches)
            b = batches[r]
            trial[r] = RankBatch(
                np.vstack([b.surv, req_surv[None, :]]),
                np.append(b.cap, req_cap),
                b.kv_tokens + req_kv,
            )
            value, time = evaluate(trial, self.policy.allocate(trial, self.cost), self.cost, self.policy.depth)
            score = value / time
            if score > best_score:
                best, best_score = int(r), score
        return best


def make_router(spec: str, cost: CostModel, policy: Policy) -> Router:
    if spec == "round_robin":
        return RoundRobin()
    if spec == "least_loaded":
        return LeastLoaded()
    if spec == "least_kv":
        return LeastKV()
    if spec == "spec_aware":
        return SpecAware(cost, policy)
    raise ValueError(f"unknown router {spec!r}")
