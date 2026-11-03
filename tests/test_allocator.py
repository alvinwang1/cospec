import itertools

import numpy as np
import pytest

from cospec.allocator import CoSpec, PerRank, RankBatch, evaluate, waterfill
from cospec.step_model import CostModel, StepModel


def _batch(rng, n, K, kv=None):
    conf = rng.uniform(0.2, 0.99, size=(n, K))
    surv = np.cumprod(conf, axis=1)
    cap = rng.integers(0, K + 1, size=n)
    return RankBatch(surv, cap, float(kv if kv is not None else rng.integers(100, 5000) * n))


def _brute(batches, cost, depth, feasible=lambda ks: True):
    best = (-1.0, None)
    choices = [itertools.product(*[range(c + 1) for c in b.cap]) for b in batches]
    for combo in itertools.product(*[list(c) for c in choices]):
        ks = [np.array(k, dtype=int) for k in combo]
        if not feasible(ks):
            continue
        v, t = evaluate(batches, ks, cost, depth)
        if v / t > best[0]:
            best = (v / t, ks)
    return best


COSTS = [
    CostModel(StepModel(a=5, b=0.3, c=1.0, knee=4, d=1e-3, e=0.1), draft_per_depth_ms=0.5),
    CostModel(StepModel(a=5, b=0.3, c=1.0, knee=4, d=1e-3, e=0.1), moe_base_ms=1.0, moe_per_token_ms=0.8),
]


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("cost", COSTS)
def test_waterfill_matches_brute_force(seed, cost):
    rng = np.random.default_rng(seed)
    K = 3
    batches = [_batch(rng, rng.integers(0, 3), K) for _ in range(rng.integers(1, 4))]
    alloc = waterfill(batches, cost, depth=K)
    v, t = evaluate(batches, alloc.ks, cost, K)
    assert v == pytest.approx(alloc.value) and t == pytest.approx(alloc.time)
    best, _ = _brute(batches, cost, K)
    assert v / t == pytest.approx(best, rel=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_budgets_respect_caps_and_token_limit(seed):
    rng = np.random.default_rng(seed)
    batches = [_batch(rng, 20, 8) for _ in range(4)]
    for b in batches:
        b.cap[:] = rng.integers(0, 9, size=20)
    alloc = waterfill(batches, COSTS[0], depth=8, max_tokens=40)
    for b, k in zip(batches, alloc.ks):
        assert (k <= b.cap).all() and (k >= 0).all()
        assert b.n_seqs + k.sum() <= 40


@pytest.mark.parametrize("seed", range(25))
def test_cospec_dominates_perrank(seed):
    """Same group step time as per-rank, and no request gets a smaller budget."""
    rng = np.random.default_rng(seed)
    cost = CostModel(StepModel(), draft_per_depth_ms=0.8)
    batches = [_batch(rng, int(n), 8) for n in rng.integers(0, 64, size=8)]
    pr = PerRank(8, 512).allocate(batches, cost)
    cs = CoSpec(8, 512).allocate(batches, cost)
    _, t_pr = evaluate(batches, pr, cost, 8)
    v_cs, t_cs = evaluate(batches, cs, cost, 8)
    assert t_cs == pytest.approx(t_pr)
    for a, b in zip(pr, cs):
        assert (b >= a).all()


def test_slack_is_spent_on_light_ranks():
    """A light rank should speculate deeper under CoSpec than on its own."""
    rng = np.random.default_rng(0)
    cost = CostModel(StepModel(), draft_per_depth_ms=0.8)
    heavy = _batch(rng, 64, 8, kv=64 * 2000)
    light = _batch(rng, 4, 8, kv=4 * 500)
    pr = PerRank(8, 512).allocate([heavy, light], cost)
    cs = CoSpec(8, 512).allocate([heavy, light], cost)
    assert cs[1].sum() > pr[1].sum()
