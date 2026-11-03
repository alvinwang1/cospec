import numpy as np
import pytest

from cospec.step_model import StepModel
from cospec.traces import Calibrator, TraceSet, category_priors, synthetic_traces


def test_step_model_fit_recovers_parameters():
    true = StepModel(a=11, b=0.01, c=0.12, knee=96, d=2e-4, e=0.02)
    rng = np.random.default_rng(0)
    n = rng.integers(1, 65, size=400)
    q = rng.integers(1, 10, size=400)
    tok = n * q
    kv = n * rng.integers(64, 4096, size=400)
    ms = true.time(n, tok, kv) * (1 + 0.01 * rng.normal(size=400))
    fit, err = StepModel.fit(n, tok, kv, ms)
    assert err < 0.03
    grid_n, grid_t, grid_kv = 32, np.arange(32, 300, 16), 32 * 1000
    assert np.allclose(fit.time(grid_n, grid_t, grid_kv), true.time(grid_n, grid_t, grid_kv), rtol=0.05)


def test_trace_roundtrip_and_subset(tmp_path):
    tr = synthetic_traces(20, K=5, seed=0, mean_output_len=30)
    tr.save(tmp_path / "t.npz")
    back = TraceSet.load(tmp_path / "t.npz")
    assert (back.acc_len == tr.acc_len).all() and (back.category == tr.category).all()
    assert np.allclose(back.conf, tr.conf, atol=1e-3)
    sub = back.subset([3, 7])
    assert sub.n_requests == 2
    assert (sub.acc_len[: sub.output_lens()[0]] == tr.acc_len[tr.offsets[3]:tr.offsets[4]]).all()


def test_acc_len_never_runs_past_output():
    tr = synthetic_traces(50, K=8, seed=1, mean_output_len=20)
    for i in range(tr.n_requests):
        a = tr.acc_len[tr.offsets[i]:tr.offsets[i + 1]]
        assert (a <= len(a) - np.arange(len(a))).all()


def test_calibration_is_monotone_and_improves_fit():
    tr = synthetic_traces(300, K=8, seed=2)
    train, test = tr.split(0.5, seed=0)
    cal = Calibrator.fit(train)
    s = cal.survival(test.conf)
    assert (np.diff(s, axis=1) <= 1e-6).all()
    truth = test.true_survival()
    raw = np.cumprod(test.conf, axis=1)
    assert np.mean((s - truth) ** 2) <= np.mean((raw - truth) ** 2)


def test_priors_order_categories():
    pri = category_priors(synthetic_traces(200, K=8, seed=4))
    assert pri["code"][0] > pri["chat"][0]
