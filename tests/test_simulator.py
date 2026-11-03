import numpy as np
import pytest

from cospec.allocator import make_policy
from cospec.experiment import run_experiment
from cospec.router import LeastLoaded
from cospec.simulator import SimConfig, Simulator
from cospec.step_model import CostModel, StepModel
from cospec.traces import Calibrator, TraceSet, synthetic_traces
from cospec.workload import build_requests


@pytest.fixture(scope="module")
def traces():
    return synthetic_traces(80, K=6, seed=3, mean_output_len=40, mean_prompt_len=100)


def _sim(traces, policy, n_ranks=4, **kw):
    reqs = build_requests(traces, 120, rate_rps=200, n_ranks=n_ranks, mix={"code": 0.5, "chat": 0.5},
                          affinity_frac=0.5, seed=1)
    cost = CostModel(StepModel())
    return Simulator(SimConfig(n_ranks=n_ranks, max_seqs=16, **kw), cost, make_policy(policy, traces.K),
                     LeastLoaded(), traces, reqs, Calibrator.fit(traces))


@pytest.mark.parametrize("policy", ["nosd", "fixed3", "perrank", "cospec", "cospec_tput", "oracle"])
def test_every_request_finishes_with_its_full_output(traces, policy):
    sim = _sim(traces, policy)
    sim.run()
    assert np.isfinite(sim.finish).all()
    assert (sim.pos == sim.L).all()
    assert (sim.finish >= sim.admit).all() and (sim.admit >= sim.arrival).all()


def test_nosd_commits_one_token_per_step(traces):
    sim = _sim(traces, "nosd")
    sim.run()
    assert (sim.steps == sim.L - 1).all()
    assert sim.accepted.sum() == 0


def test_replay_never_accepts_beyond_trace(traces):
    sim = _sim(traces, "fixed6")
    sim.run()
    assert (sim.accepted <= sim.drafted).all()


def test_policies_never_read_acceptance(traces):
    """Scrambling acceptance must not change the first allocation (losslessness)."""
    sims = []
    for scramble in (False, True):
        tr = TraceSet(traces.conf, traces.acc_len.copy(), traces.offsets, traces.prompt_len, traces.category)
        if scramble:
            tr.acc_len[:] = np.random.default_rng(0).integers(0, tr.K + 1, size=len(tr.acc_len))
        sim = _sim(tr, "cospec")
        sim.calib = Calibrator.identity()
        for i in range(sim.N):
            sim._route(i)
        sim._admit()
        sims.append(sim)
    sims[0]._step()
    sims[1]._step()
    assert (sims[0].drafted == sims[1].drafted).all()


def test_max_seqs_respected(traces):
    sim = _sim(traces, "cospec")
    orig = sim._step

    def checked():
        assert all(len(r) <= sim.cfg.max_seqs for r in sim.running)
        return orig()
    sim._step = checked
    sim.run()


def test_run_experiment_smoke():
    m = run_experiment(dict(policy="cospec", router="spec_aware", n_ranks=4, requests_per_rank=20,
                            synthetic_requests=60))
    assert m["n_requests"] == 80 and 0 <= m["slo_attainment"] <= 1
