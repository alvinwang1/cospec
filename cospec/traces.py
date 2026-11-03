"""Acceptance traces: storage format, synthetic generator, calibration, priors.

A trace records, for every output position p of a request, the K tokens a
drafter proposes given the *true* prefix, the drafter's confidence in each, and
how many of them the target accepts under greedy verification. Under greedy
decoding the output sequence does not depend on the speculation policy, so a
trace replays any budget policy exactly: a request at position p that verifies
k draft tokens commits min(k, acc_len[p]) + 1 tokens.

Row p of a request means "p output tokens are already committed". Row 0 is
drafting right after the prompt; the decode-only simulator starts at p = 1
because prefill produces the first token.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TraceSet:
    conf: np.ndarray        # float32 [P, K]  drafter confidence per drafted token
    acc_len: np.ndarray     # int16   [P]     leading draft tokens the target accepts
    offsets: np.ndarray     # int64   [N+1]   row range of each request
    prompt_len: np.ndarray  # int32   [N]
    category: np.ndarray    # str     [N]     e.g. "code" / "chat"
    meta: dict = field(default_factory=dict)

    @property
    def n_requests(self) -> int:
        return len(self.prompt_len)

    @property
    def K(self) -> int:
        return self.conf.shape[1]

    def output_lens(self) -> np.ndarray:
        return np.diff(self.offsets)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            conf=self.conf.astype(np.float16),
            acc_len=self.acc_len.astype(np.int16),
            offsets=self.offsets.astype(np.int64),
            prompt_len=self.prompt_len.astype(np.int32),
            category=self.category.astype(str),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str) -> "TraceSet":
        z = np.load(path, allow_pickle=False)
        return cls(
            conf=z["conf"].astype(np.float32),
            acc_len=z["acc_len"].astype(np.int16),
            offsets=z["offsets"].astype(np.int64),
            prompt_len=z["prompt_len"].astype(np.int32),
            category=z["category"].astype(str),
            meta=json.loads(str(z["meta"])),
        )

    def subset(self, idx) -> "TraceSet":
        idx = np.asarray(idx, dtype=np.int64)
        rows = [np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx]
        rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        lens = self.output_lens()[idx]
        return TraceSet(
            conf=self.conf[rows],
            acc_len=self.acc_len[rows],
            offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            prompt_len=self.prompt_len[idx],
            category=self.category[idx],
            meta=dict(self.meta),
        )

    @classmethod
    def concat(cls, sets: list["TraceSet"]) -> "TraceSet":
        if len({s.K for s in sets}) != 1:
            raise ValueError("all trace sets must share the same draft depth K")
        lens = np.concatenate([s.output_lens() for s in sets])
        return cls(
            conf=np.concatenate([s.conf for s in sets]),
            acc_len=np.concatenate([s.acc_len for s in sets]),
            offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            prompt_len=np.concatenate([s.prompt_len for s in sets]),
            category=np.concatenate([s.category for s in sets]),
            meta={"sources": [s.meta for s in sets]},
        )

    def split(self, train_frac: float, seed: int = 0) -> tuple["TraceSet", "TraceSet"]:
        """Split by request into a calibration set and a replay set."""
        perm = np.random.default_rng(seed).permutation(self.n_requests)
        n_train = int(round(train_frac * self.n_requests))
        return self.subset(np.sort(perm[:n_train])), self.subset(np.sort(perm[n_train:]))

    def true_survival(self) -> np.ndarray:
        """[P, K] indicator that draft token j (0-based) is accepted."""
        return (np.arange(self.K)[None, :] < self.acc_len[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic traces, used until real GPU traces exist and in tests.

SYNTH_CATEGORIES = {
    # mean per-token acceptance, AR(1) persistence and noise in logit space
    "code": dict(mean=0.82, rho=0.9, sigma=0.5),
    "chat": dict(mean=0.62, rho=0.9, sigma=0.6),
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def synthetic_traces(
    n_requests: int,
    K: int = 8,
    mix: dict[str, float] | None = None,
    seed: int = 0,
    mean_output_len: float = 256,
    mean_prompt_len: float = 512,
    depth_decay: float = 0.97,
    conf_signal: float = 0.8,
    conf_noise: float = 0.7,
) -> TraceSet:
    """Generate traces whose drafter confidences are informative but noisy."""
    mix = mix or {"code": 0.5, "chat": 0.5}
    cats = list(mix)
    probs = np.array([mix[c] for c in cats], dtype=float)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)

    confs, accs, lens, plens, ccat = [], [], [], [], []
    for _ in range(n_requests):
        cat = cats[rng.choice(len(cats), p=probs)]
        prof = SYNTH_CATEGORIES[cat]
        L = int(np.clip(rng.lognormal(np.log(mean_output_len), 0.6), 8, 4 * mean_output_len))
        P = int(np.clip(rng.lognormal(np.log(mean_prompt_len), 0.7), 16, 8 * mean_prompt_len))

        mu = _logit(prof["mean"])
        z = np.empty(L)
        z[0] = mu + rng.normal() * prof["sigma"]
        eps = rng.normal(size=L) * prof["sigma"] * np.sqrt(1 - prof["rho"] ** 2)
        for p in range(1, L):
            z[p] = mu + prof["rho"] * (z[p - 1] - mu) + eps[p]
        q = _sigmoid(z)[:, None] * depth_decay ** np.arange(K)[None, :]  # [L, K]

        ok = rng.random((L, K)) < q
        acc = np.where(ok.all(axis=1), K, np.argmin(ok, axis=1))
        # Drafts past the end of the output cannot match anything.
        acc = np.minimum(acc, L - np.arange(L))

        logit_conf = _logit(q) + conf_signal * (2 * ok - 1) + conf_noise * rng.normal(size=(L, K))
        confs.append(_sigmoid(logit_conf).astype(np.float32))
        accs.append(acc.astype(np.int16))
        lens.append(L)
        plens.append(P)
        ccat.append(cat)

    return TraceSet(
        conf=np.concatenate(confs),
        acc_len=np.concatenate(accs),
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        prompt_len=np.array(plens, dtype=np.int32),
        category=np.array(ccat),
        meta={"source": "synthetic", "K": K, "seed": seed, "mix": mix},
    )


# ---------------------------------------------------------------------------
# Turning confidences into survival probabilities.

def _pav_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: weighted least-squares non-decreasing fit."""
    vals, wts, cnts = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wsum = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wsum
            c = cnts[-2] + cnts[-1]
            vals[-2:], wts[-2:], cnts[-2:] = [v], [wsum], [c]
    return np.repeat(vals, cnts)


class Calibrator:
    """Monotone map from cumulative drafter confidence to survival probability.

    The raw estimate of P(first j draft tokens accepted) is prod_{i<=j} conf_i.
    Drafters are usually miscalibrated, so we fit an isotonic map on a held-out
    split. A monotone map keeps survival non-increasing along the draft.
    """

    def __init__(self, xs: np.ndarray | None = None, ys: np.ndarray | None = None):
        self.xs, self.ys = xs, ys

    @classmethod
    def identity(cls) -> "Calibrator":
        return cls()

    @classmethod
    def fit(cls, traces: TraceSet, n_bins: int = 40) -> "Calibrator":
        pred = np.cumprod(traces.conf, axis=1).ravel()
        true = traces.true_survival().ravel()
        edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(edges) - 2)
        cnt = np.bincount(which, minlength=len(edges) - 1).astype(float)
        keep = cnt > 0
        xs = (np.bincount(which, weights=pred, minlength=len(cnt))[keep] / cnt[keep])
        ys = (np.bincount(which, weights=true, minlength=len(cnt))[keep] / cnt[keep])
        ys = _pav_increasing(ys, cnt[keep])
        return cls(np.concatenate([[0.0], xs, [1.0]]), np.concatenate([[0.0], ys, [ys[-1]]]))

    def survival(self, conf: np.ndarray) -> np.ndarray:
        raw = np.cumprod(conf, axis=-1)
        if self.xs is None:
            return raw
        return np.interp(raw, self.xs, self.ys).astype(np.float32)


def category_priors(traces: TraceSet) -> dict[str, np.ndarray]:
    """Mean survival by draft depth for each category (and "_all").

    Routers use these before a request has drafted anything.
    """
    surv = traces.true_survival()
    row_cat = np.repeat(traces.category, traces.output_lens())
    priors = {"_all": surv.mean(axis=0)}
    for c in np.unique(traces.category):
        priors[str(c)] = surv[row_cat == c].mean(axis=0)
    return priors
