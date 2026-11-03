"""Speculation-budget policies.

Every policy maps a snapshot of all ranks to a budget k_i (draft tokens to
verify) for each running request. Policies only see predicted survival
probabilities computed from drafter confidences, never whether tokens will be
accepted, so decisions are non-anticipating and decoding stays lossless. The
oracle policies are the exception and only serve as upper bounds.

Value of a budget: request i commits 1 + sum_{j<=k_i} s_ij tokens in
expectation, where s_ij is the survival probability of its j-th draft token.
Within one rank the step time depends only on how many tokens it verifies, so
a rank verifying m draft tokens should take its m most likely ones; because
s_ij is non-increasing in j that set is a valid per-request prefix. Across
ranks, lockstep couples everything through G = max_r T_r: given a step-time
level tau, each rank can fill itself up to tau ("water-filling") and ranks
below the slowest one speculate for free.

Two group objectives:

* cospec_tput: maximize sum of expected tokens / G over the whole group. This
  maximizes instantaneous throughput but starves the bottleneck rank: the
  group-wide ratio is large, so speculation that slows the slowest rank never
  pays off, even though that rank is the one with the backlog.
* cospec (default D1): anchor the level at the bottleneck. Each rank finds its
  own goodput-optimal budget, tau* is the slowest resulting step, and every
  rank fills its slack up to tau*. Step time equals the per-rank policy's, and
  every request gets at least its per-rank budget, so it dominates per-rank.

Both are solved exactly by `waterfill`: Dinkelbach iterations for the ratio,
each a single sorted sweep over all candidate tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .step_model import CostModel


@dataclass
class RankBatch:
    surv: np.ndarray    # [n, K] predicted survival per draft token, non-increasing in j
    cap: np.ndarray     # [n] max budget per request (draft depth, tokens left)
    kv_tokens: float    # total context on this rank

    @property
    def n_seqs(self) -> int:
        return len(self.cap)


@dataclass
class Allocation:
    ks: list[np.ndarray]   # budget per request, per rank
    m: np.ndarray          # draft tokens verified per rank
    value: float           # expected committed tokens this step
    time: float            # group step time (ms)


def _rank_candidates(b: RankBatch, max_tokens: int | None):
    """Draft tokens of one rank, most likely first: (gains, owner request)."""
    n, K = b.surv.shape
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    mask = np.arange(K)[None, :] < b.cap[:, None]
    gains = b.surv[mask]                       # row-major: request order, then depth
    owner = np.nonzero(mask)[0]
    order = np.argsort(-gains, kind="stable")  # stable keeps depth order on ties
    gains, owner = gains[order], owner[order]
    keep = gains > 0
    gains, owner = gains[keep], owner[keep]
    if max_tokens is not None:
        room = max(0, max_tokens - n)
        gains, owner = gains[:room], owner[:room]
    return gains, owner


def waterfill(batches: list[RankBatch], cost: CostModel, depth: int, max_tokens: int | None = None,
              moe: bool = True, min_m=None, max_level: float | None = None,
              tol: float = 1e-9, max_iter: int = 50) -> Allocation:
    """Maximize expected tokens / group step time over all ranks jointly.

    min_m[r] forces rank r to verify at least its top min_m[r] tokens;
    max_level caps every rank's step time (tokens that would exceed it are
    not candidates).
    """
    R = len(batches)
    n = np.array([b.n_seqs for b in batches], dtype=float)
    kv = np.array([b.kv_tokens for b in batches], dtype=float)
    cands = [_rank_candidates(b, max_tokens) for b in batches]
    min_m = np.zeros(R, dtype=int) if min_m is None else np.minimum(min_m, [len(g) for g, _ in cands])
    moe_base = cost.moe_base_ms if moe else 0.0
    moe_c = (cost.moe_per_token_ms / R) if moe else 0.0

    # times[r][m]: step time of rank r verifying m draft tokens, m = 0..M_r.
    times = [cost.rank_times(n[r], n[r] + np.arange(len(g) + 1), kv[r], depth) for r, (g, _) in enumerate(cands)]
    tau0 = max(times[r][min_m[r]] for r in range(R))
    base_value = n.sum() + sum(g[: min_m[r]].sum() for r, (g, _) in enumerate(cands))
    base_tok = n.sum() + min_m.sum()
    base_time = tau0 + moe_base + moe_c * base_tok

    # Optional tokens: (r, m) for m > min_m[r], "turning on" at level times[r][m].
    ev_t, ev_g, ev_r = [], [], []
    for r, (g, _) in enumerate(cands):
        t = times[r][min_m[r] + 1:]
        gg = g[min_m[r]:]
        if max_level is not None:
            ok = t <= max_level * (1 + 1e-12)
            t, gg = t[ok], gg[ok]          # times are monotone, so this is a prefix
        ev_t.append(t)
        ev_g.append(gg)
        ev_r.append(np.full(len(t), r, dtype=np.int64))
    ev_t, ev_g, ev_r = np.concatenate(ev_t), np.concatenate(ev_g), np.concatenate(ev_r)
    order = np.argsort(ev_t, kind="stable")    # stable: each rank stays in m order
    ev_t, ev_g, ev_r = ev_t[order], ev_g[order], ev_r[order]
    level = np.maximum(ev_t, tau0)

    # Dinkelbach: for ratio lam, maximize value - lam * time. The best level is
    # one of the event levels, so one cumulative sum evaluates all of them.
    lam = base_value / base_time
    value, time = base_value, base_time
    take = np.zeros(len(ev_g), dtype=bool)
    for _ in range(max_iter):
        h = ev_g - lam * moe_c
        # Tokens with h <= 0 never help at this lam. They sit at the end of
        # their rank's list, so skipping them keeps each rank a prefix.
        F = np.concatenate([[-lam * tau0], np.cumsum(np.maximum(h, 0.0)) - lam * level])
        k = int(np.argmax(F))
        cand = np.zeros(len(ev_g), dtype=bool)
        cand[:k] = h[:k] > 0
        m_r = min_m + np.bincount(ev_r[cand], minlength=R)
        v = base_value + ev_g[cand].sum()
        t = max(times[r][m_r[r]] for r in range(R)) + moe_base + moe_c * (n.sum() + m_r.sum())
        if v / t <= lam * (1 + tol):
            break
        lam, value, time, take = v / t, v, t, cand

    m_r = min_m + np.bincount(ev_r[take], minlength=R)
    ks = [np.bincount(cands[r][1][: m_r[r]], minlength=batches[r].n_seqs) for r in range(R)]
    return Allocation(ks, m_r, float(value), float(time))


def evaluate(batches: list[RankBatch], ks: list[np.ndarray], cost: CostModel, depth: int) -> tuple[float, float]:
    """Expected committed tokens and group step time of a given allocation."""
    value = sum(float(b.n_seqs + b.surv[np.arange(b.surv.shape[1])[None, :] < k[:, None]].sum())
                for b, k in zip(batches, ks))
    n = np.array([b.n_seqs for b in batches])
    n_tok = n + np.array([int(k.sum()) for k in ks])
    kv = np.array([b.kv_tokens for b in batches])
    return value, cost.group_time(n, n_tok, kv, depth)


# ---------------------------------------------------------------------------
# Policies

class Policy:
    name = "policy"
    depth = 0             # sequential draft steps each step pays for
    surv_source = "conf"  # "conf" (non-anticipating) or "oracle"

    def allocate(self, batches: list[RankBatch], cost: CostModel) -> list[np.ndarray]:
        raise NotImplementedError


class NoSD(Policy):
    name = "nosd"

    def allocate(self, batches, cost):
        return [np.zeros(b.n_seqs, dtype=int) for b in batches]


class Fixed(Policy):
    def __init__(self, k: int):
        self.k = self.depth = k
        self.name = f"fixed{k}"

    def allocate(self, batches, cost):
        return [np.minimum(self.k, b.cap).astype(int) for b in batches]


class PerRank(Policy):
    """Each rank maximizes its own goodput, unaware of the lockstep group.

    The per-engine adaptive baseline (SmartSpec / DSpark style): the same
    optimizer as CoSpec, applied to one rank at a time.
    """
    name = "perrank"

    def __init__(self, depth: int, max_tokens: int | None = None):
        self.depth, self.max_tokens = depth, max_tokens

    def local(self, batches, cost) -> list[Allocation]:
        return [waterfill([b], cost, self.depth, self.max_tokens, moe=False) for b in batches]

    def allocate(self, batches, cost):
        return [a.ks[0] for a in self.local(batches, cost)]


class CoSpec(PerRank):
    """D1: bottleneck-anchored water-filling across all lockstep ranks."""
    name = "cospec"

    def allocate(self, batches, cost):
        local = self.local(batches, cost)
        tau = max(a.time for a in local)
        m = np.array([int(a.m[0]) for a in local])
        return waterfill(batches, cost, self.depth, self.max_tokens, min_m=m, max_level=tau).ks


class CoSpecTput(PerRank):
    """D1 ablation: maximize instantaneous group throughput (no anchor)."""
    name = "cospec_tput"

    def allocate(self, batches, cost):
        return waterfill(batches, cost, self.depth, self.max_tokens).ks


class Oracle(CoSpec):
    """CoSpec fed true acceptance instead of predictions: an upper bound."""
    name = "oracle"
    surv_source = "oracle"


class PerRankOracle(PerRank):
    name = "perrank_oracle"
    surv_source = "oracle"


POLICIES = {"nosd", "fixedK", "perrank", "cospec", "cospec_tput", "oracle", "perrank_oracle"}


def make_policy(spec: str, K: int, max_tokens: int | None = None) -> Policy:
    if spec == "nosd":
        return NoSD()
    if spec.startswith("fixed"):
        return Fixed(int(spec.removeprefix("fixed").lstrip(":") or 4))
    table = {"perrank": PerRank, "cospec": CoSpec, "cospec_tput": CoSpecTput,
             "oracle": Oracle, "perrank_oracle": PerRankOracle}
    if spec not in table:
        raise ValueError(f"unknown policy {spec!r}; choose from {sorted(POLICIES)}")
    return table[spec](K, max_tokens)
