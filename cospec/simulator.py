"""Discrete-step simulator of R decode ranks running in lockstep.

Each group step:
  1. route requests that have arrived, admit queued ones into free batch slots,
  2. build each rank's batch: survival estimates of its requests' current drafts,
  3. ask the policy for budgets k_i,
  4. charge the group max_r(T_r + D_r) + MoE time (step_model.CostModel),
  5. replay the acceptance trace: request i commits min(k_i, acc_len) + 1 tokens.

The cluster is decode-only (prefill happens elsewhere, as in PD
disaggregation): a request arrives with its prompt's KV and its first token.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .allocator import Policy, RankBatch
from .router import RankView, Router
from .step_model import CostModel
from .traces import Calibrator, TraceSet
from .workload import Request


@dataclass
class SimConfig:
    n_ranks: int = 8
    max_seqs: int = 64              # batch slots per rank
    max_tokens: int | None = 512    # verification tokens per rank per step
    kv_capacity: int | None = None  # per rank; admission reserves prompt + output
    K: int | None = None            # max draft depth (default: trace depth)
    tpot_slo_ms: float = 50.0
    queue_slo_ms: float = 2000.0
    warmup_frac: float = 0.1        # requests excluded from latency stats at each end
    record_steps: bool = False


class Simulator:
    def __init__(self, cfg: SimConfig, cost: CostModel, policy: Policy, router: Router,
                 traces: TraceSet, requests: list[Request], calibrator: Calibrator | None = None,
                 priors: dict[str, np.ndarray] | None = None, router_prior: str = "category"):
        self.cfg, self.cost, self.policy, self.router = cfg, cost, policy, router
        self.tr = traces
        self.K = min(cfg.K or traces.K, traces.K)
        self.calib = calibrator or Calibrator.identity()
        self.priors = priors or {}
        self.router_prior = router_prior

        reqs = sorted(requests, key=lambda r: r.arrival_ms)
        N = len(reqs)
        self.N = N
        tidx = np.array([r.trace_idx for r in reqs])
        self.row0 = traces.offsets[tidx]
        self.L = traces.output_lens()[tidx]
        self.prompt = traces.prompt_len[tidx].astype(np.int64)
        self.cat = traces.category[tidx]
        self.arrival = np.array([r.arrival_ms for r in reqs])
        self.pinned = np.array([r.pinned_rank for r in reqs])

        self.pos = np.ones(N, dtype=np.int64)      # prefill already produced token 0
        self.rank = np.full(N, -1)
        self.admit = np.full(N, np.nan)
        self.finish = np.full(N, np.nan)
        self.steps = np.zeros(N, dtype=np.int64)
        self.accepted = np.zeros(N, dtype=np.int64)
        self.drafted = np.zeros(N, dtype=np.int64)
        self.last_surv = np.zeros((N, self.K), dtype=np.float32)

        R = cfg.n_ranks
        self.running: list[list[int]] = [[] for _ in range(R)]
        self.queue: list[deque[int]] = [deque() for _ in range(R)]
        self.reserved = np.zeros(R)               # kv_capacity accounting
        self.step_log: list[dict] = []
        self.alloc_ms: list[float] = []
        self.idle_frac: list[float] = []
        self.now = 0.0

    # -- routing ----------------------------------------------------------

    def _prior(self, i: int) -> np.ndarray:
        key = str(self.cat[i]) if self.router_prior == "category" else "_all"
        p = self.priors.get(key, self.priors.get("_all"))
        if p is None:
            return np.zeros(self.K, dtype=np.float32)
        return np.asarray(p[: self.K], dtype=np.float32)

    def _cap(self, ids) -> np.ndarray:
        return np.minimum(self.K, self.L[ids] - self.pos[ids] - 1)

    def _view(self) -> RankView:
        R = self.cfg.n_ranks
        surv, cap, kv = [], [], np.zeros(R)
        for r in range(R):
            run = self.running[r]
            q = list(self.queue[r])
            s = [self.last_surv[run]] if run else []
            s += [self._prior(i)[None, :] for i in q]
            surv.append(np.vstack(s) if s else np.zeros((0, self.K), dtype=np.float32))
            ids = np.array(run + q, dtype=np.int64)
            cap.append(self._cap(ids) if len(ids) else np.zeros(0, dtype=np.int64))
            kv[r] = float((self.prompt[ids] + self.pos[ids]).sum()) if len(ids) else 0.0
        return RankView(surv, cap, kv,
                        np.array([len(x) for x in self.running]),
                        np.array([len(x) for x in self.queue]),
                        self.cfg.max_seqs)

    def _route(self, i: int):
        if self.pinned[i] >= 0:
            r = int(self.pinned[i]) % self.cfg.n_ranks
        else:
            r = self.router.route(self._view(), self._prior(i), int(self._cap([i])[0]),
                                  int(self.prompt[i] + self.pos[i]))
        self.rank[i] = r
        self.queue[r].append(i)

    def _admit(self):
        cap = self.cfg.kv_capacity
        for r in range(self.cfg.n_ranks):
            q = self.queue[r]
            while q and len(self.running[r]) < self.cfg.max_seqs:
                i = q[0]
                need = float(self.prompt[i] + self.L[i])
                if cap is not None and self.running[r] and self.reserved[r] + need > cap:
                    break
                q.popleft()
                self.reserved[r] += need
                self.running[r].append(i)
                self.admit[i] = self.now
                self.last_surv[i] = self._prior(i)

    # -- one lockstep step ------------------------------------------------

    def _step(self) -> float:
        R, K = self.cfg.n_ranks, self.K
        ids_r, rows_r, batches = [], [], []
        for r in range(R):
            ids = np.array(self.running[r], dtype=np.int64)
            rows = self.row0[ids] + self.pos[ids]
            if self.policy.surv_source == "oracle":
                surv = (np.arange(K)[None, :] < self.tr.acc_len[rows][:, None]).astype(np.float32)
            else:
                surv = self.calib.survival(self.tr.conf[rows, :K])
            kv = float((self.prompt[ids] + self.pos[ids]).sum())
            batches.append(RankBatch(surv.reshape(len(ids), K), self._cap(ids), kv))
            ids_r.append(ids)
            rows_r.append(rows)
            if len(ids):
                self.last_surv[ids] = surv

        t0 = time.perf_counter()
        ks = self.policy.allocate(batches, self.cost)
        self.alloc_ms.append((time.perf_counter() - t0) * 1000.0)

        n = np.array([b.n_seqs for b in batches])
        n_tok = n + np.array([int(k.sum()) for k in ks])
        kv = np.array([b.kv_tokens for b in batches])
        rank_ms = self.cost.rank_times(n, n_tok, kv, self.policy.depth)
        moe = self.cost.moe_base_ms + self.cost.moe_per_token_ms * n_tok.sum() / R
        group_ms = float(rank_ms.max() + moe)
        self.idle_frac.append(float(1.0 - rank_ms.mean() / rank_ms.max()))
        if self.cfg.record_steps:
            self.step_log.append(dict(t=self.now, n_seqs=n, n_tokens=n_tok, kv_tokens=kv,
                                      rank_ms=rank_ms, group_ms=group_ms))

        end = self.now + group_ms
        for r in range(R):
            ids = ids_r[r]
            if not len(ids):
                continue
            k = ks[r]
            acc = np.minimum(k, self.tr.acc_len[rows_r[r]])
            self.pos[ids] += acc + 1
            self.steps[ids] += 1
            self.accepted[ids] += acc
            self.drafted[ids] += k
            done = self.pos[ids] >= self.L[ids]
            if done.any():
                fin = ids[done]
                self.finish[fin] = end
                self.reserved[r] -= (self.prompt[fin] + self.L[fin]).sum()
                self.running[r] = ids[~done].tolist()
        return group_ms

    # -- main loop --------------------------------------------------------

    def run(self) -> dict:
        nxt = 0
        while True:
            while nxt < self.N and self.arrival[nxt] <= self.now:
                self._route(nxt)
                nxt += 1
            self._admit()
            if not any(self.running):
                if nxt >= self.N and not any(self.queue):
                    break
                if nxt < self.N:
                    self.now = max(self.now, self.arrival[nxt])
                continue
            self.now += self._step()
        return self.metrics()

    def metrics(self) -> dict:
        lo, hi = int(self.cfg.warmup_frac * self.N), int(np.ceil((1 - self.cfg.warmup_frac) * self.N))
        w = slice(lo, max(hi, lo + 1))
        dec = np.maximum(self.L - 1, 1)
        tpot = (self.finish - self.admit) / dec
        queue = self.admit - self.arrival
        ok = (tpot <= self.cfg.tpot_slo_ms) & (queue <= self.cfg.queue_slo_ms)
        span = self.arrival[w].max() - self.arrival[w].min()
        makespan = self.finish.max() - self.arrival.min()
        alloc = np.array(self.alloc_ms) if self.alloc_ms else np.zeros(1)
        return {
            "policy": self.policy.name,
            "router": self.router.name,
            "n_ranks": self.cfg.n_ranks,
            "n_requests": self.N,
            "offered_rps": float(len(self.arrival[w]) / span * 1000.0) if span > 0 else float("nan"),
            "goodput_rps": float(ok[w].sum() / span * 1000.0) if span > 0 else float("nan"),
            "slo_attainment": float(ok[w].mean()),
            "throughput_tok_s": float(dec.sum() / makespan * 1000.0),
            "tpot_mean_ms": float(np.mean(tpot[w])),
            "tpot_p50_ms": float(np.percentile(tpot[w], 50)),
            "tpot_p99_ms": float(np.percentile(tpot[w], 99)),
            "queue_p99_ms": float(np.percentile(queue[w], 99)),
            "tokens_per_req_step": float((self.accepted.sum() + self.steps.sum()) / max(self.steps.sum(), 1)),
            "accept_rate": float(self.accepted.sum() / max(self.drafted.sum(), 1)),
            "mean_budget": float(self.drafted.sum() / max(self.steps.sum(), 1)),
            "lockstep_idle_frac": float(np.mean(self.idle_frac)) if self.idle_frac else 0.0,
            "n_steps": len(self.alloc_ms),
            "alloc_ms_mean": float(alloc.mean()),
            "alloc_ms_p99": float(np.percentile(alloc, 99)),
        }

    def dump_steps(self, path: str):
        """Save the per-step, per-rank batch shapes for gpu/validate.py."""
        if not self.step_log:
            raise RuntimeError("run with SimConfig(record_steps=True)")
        np.savez_compressed(
            path,
            n_seqs=np.stack([s["n_seqs"] for s in self.step_log]),
            n_tokens=np.stack([s["n_tokens"] for s in self.step_log]),
            kv_tokens=np.stack([s["kv_tokens"] for s in self.step_log]),
            rank_ms=np.stack([s["rank_ms"] for s in self.step_log]),
            group_ms=np.array([s["group_ms"] for s in self.step_log]),
            depth=np.array(self.policy.depth),
        )
