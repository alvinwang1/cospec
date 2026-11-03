"""Step-time models: one rank's verification step, and a lockstep group.

Rank model (ms), fit from GPU profiles:

    T(n_seq, n_tok, kv) = a + b*n_tok + c*max(0, n_tok - knee) + d*kv + e*n_seq

n_tok counts every token in the verification batch (1 + k_i per request), kv is
the total context length attended over. Below the knee the forward pass is
bound by reading weights; above it each extra token costs compute.

Group model: DP-attention ranks advance in lockstep, so

    G = max_r (T_r + D_r) + moe_base + moe_per_token * sum_r n_tok_r / R

where D_r is drafting time and the last terms stand in for expert compute and
all-to-all, which scale with the group's total tokens spread over R ranks.
Set the MoE terms to 0 for pure DP lockstep.

CLI:  python -m cospec.step_model fit profile.csv -o step_model.json
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class StepModel:
    a: float = 10.0       # ms, fixed cost (weight reads, launch overhead)
    b: float = 0.005      # ms per token below the knee
    c: float = 0.13       # extra ms per token above the knee
    knee: float = 64.0    # tokens where the step turns compute bound
    d: float = 1.6e-4     # ms per KV token attended
    e: float = 0.01       # ms per sequence

    def time(self, n_seqs, n_tokens, kv_tokens):
        n_tokens = np.asarray(n_tokens, dtype=float)
        return (self.a + self.b * n_tokens + self.c * np.maximum(0.0, n_tokens - self.knee)
                + self.d * np.asarray(kv_tokens, dtype=float) + self.e * np.asarray(n_seqs, dtype=float))

    @classmethod
    def fit(cls, n_seqs, n_tokens, kv_tokens, ms, knees=None) -> tuple["StepModel", float]:
        """Least squares with non-negative coefficients; grid-searches the knee.

        Returns the model and its mean absolute percentage error on the data.
        """
        n_seqs, n_tokens, kv_tokens, ms = (np.asarray(x, dtype=float) for x in (n_seqs, n_tokens, kv_tokens, ms))
        if knees is None:
            knees = np.unique(np.quantile(n_tokens, np.linspace(0.02, 0.9, 30)).round())
        best = None
        for knee in knees:
            X = np.stack([np.ones_like(n_tokens), n_tokens, np.maximum(0, n_tokens - knee), kv_tokens, n_seqs], 1)
            coef = _nnls(X, ms)
            err = np.mean(np.abs(X @ coef - ms) / ms)
            if best is None or err < best[0]:
                best = (err, knee, coef)
        err, knee, coef = best
        a, b, c, d, e = coef
        return cls(a=a, b=b, c=c, knee=float(knee), d=d, e=e), float(err)

    def to_dict(self):
        return {k: float(v) for k, v in asdict(self).items()}


def _nnls(X, y, iters=50):
    """Tiny active-set NNLS: refit dropping negative coefficients."""
    active = np.ones(X.shape[1], dtype=bool)
    coef = np.zeros(X.shape[1])
    for _ in range(iters):
        sol, *_ = np.linalg.lstsq(X[:, active], y, rcond=None)
        coef[:] = 0
        coef[active] = sol
        if (sol >= 0).all():
            break
        active[np.flatnonzero(active)[sol < 0]] = False
    return coef


@dataclass
class CostModel:
    rank: StepModel
    draft_base_ms: float = 0.0          # per step that drafts at all
    draft_per_depth_ms: float = 0.8     # per sequential draft step (EAGLE-like head)
    draft_per_depth_seq_ms: float = 0.002
    moe_base_ms: float = 0.0
    moe_per_token_ms: float = 0.0

    def draft_time(self, n_seqs, depth):
        n_seqs = np.asarray(n_seqs, dtype=float)
        t = self.draft_base_ms + depth * (self.draft_per_depth_ms + self.draft_per_depth_seq_ms * n_seqs)
        return np.where((n_seqs > 0) & (depth > 0), t, 0.0)

    def rank_times(self, n_seqs, n_tokens, kv_tokens, depth):
        return self.rank.time(n_seqs, n_tokens, kv_tokens) + self.draft_time(n_seqs, depth)

    def group_time(self, n_seqs, n_tokens, kv_tokens, depth) -> float:
        R = len(n_seqs)
        moe = self.moe_base_ms + self.moe_per_token_ms * float(np.sum(n_tokens)) / R
        return float(np.max(self.rank_times(n_seqs, n_tokens, kv_tokens, depth))) + moe

    def to_dict(self):
        d = asdict(self)
        d["rank"] = self.rank.to_dict()
        return d

    @classmethod
    def from_dict(cls, d) -> "CostModel":
        d = dict(d)
        return cls(rank=StepModel(**d.pop("rank")), **d)

    @classmethod
    def load(cls, path) -> "CostModel":
        with open(path) as f:
            d = json.load(f)
        # Accept either a full cost model or a bare fitted rank model.
        return cls.from_dict(d) if "rank" in d else cls(rank=StepModel(**{k: d[k] for k in StepModel().to_dict()}))


def read_profile_csv(path, kind="verify"):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("kind", "verify") == kind:
                rows.append(r)
    cols = {k: np.array([float(r[k]) for r in rows]) for k in ("n_seqs", "n_tokens", "kv_tokens", "ms")}
    return cols


def fit_draft(path):
    """Fit draft_per_depth_ms + draft_per_depth_seq_ms * n_seqs from draft rows."""
    p = read_profile_csv(path, kind="draft")
    if len(p["ms"]) == 0:
        return None
    X = np.stack([np.ones_like(p["n_seqs"]), p["n_seqs"]], 1)
    coef = _nnls(X, p["ms"])
    return {"draft_per_depth_ms": float(coef[0]), "draft_per_depth_seq_ms": float(coef[1])}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit a cost model from gpu/profile_step_time.py output")
    f.add_argument("profile")
    f.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    p = read_profile_csv(args.profile)
    model, err = StepModel.fit(p["n_seqs"], p["n_tokens"], p["kv_tokens"], p["ms"])
    cost = CostModel(rank=model)
    draft = fit_draft(args.profile)
    if draft:
        cost.draft_per_depth_ms = draft["draft_per_depth_ms"]
        cost.draft_per_depth_seq_ms = draft["draft_per_depth_seq_ms"]
    with open(args.out, "w") as fh:
        json.dump(cost.to_dict(), fh, indent=2)
    print(json.dumps(cost.to_dict(), indent=2))
    print(f"verify fit MAPE: {100 * err:.1f}% over {len(p['ms'])} points")


if __name__ == "__main__":
    main()
