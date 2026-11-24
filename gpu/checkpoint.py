"""Checkpoint and progress helpers for the long GPU runs.

Kept free of torch so the tests can run anywhere.
"""
from __future__ import annotations

import csv
import os
import time


def fmt_secs(s: float) -> str:
    if s != s or s == float("inf"):
        return "?"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Progress:
    """Prints done/total, elapsed time and an ETA from this session's rate."""

    def __init__(self, total: int, label: str, done: int = 0):
        self.total, self.label, self.done = total, label, done
        self._start_done, self._t0 = done, time.time()

    def start(self):
        print(f"[{self.label}] {self.done}/{self.total} already done, {self.total - self.done} to go", flush=True)

    def step(self, msg: str = "", n: int = 1):
        self.done += n
        elapsed = time.time() - self._t0
        new = self.done - self._start_done
        eta = (self.total - self.done) * elapsed / new if new > 0 else float("nan")
        pct = 100.0 * self.done / self.total if self.total else 100.0
        print(f"[{self.label} {self.done}/{self.total} {pct:.0f}% | {fmt_secs(elapsed)} elapsed, "
              f"ETA {fmt_secs(eta)}] {msg}", flush=True)


def _complete_rows(path: str, fields: list[str], required) -> list[dict]:
    """Rows of an incrementally written CSV, minus a torn last line or incomplete rows."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]                     # killed mid-write
    rows = []
    for r in csv.DictReader(lines):
        if None in r or any(r.get(k) in (None, "") for k in required):
            continue
        rows.append({k: r.get(k) or "" for k in fields})
    return rows


def open_csv_checkpoint(path: str, fields: list[str], required=(), fresh: bool = False):
    """Resume a CSV that a run appends to row by row.

    Returns (rows already there, file handle, DictWriter appending to it). The
    file is first rewritten (atomically) with only its complete rows, so a run
    killed mid-line resumes cleanly. fresh=True starts over.
    """
    rows = [] if fresh else _complete_rows(path, fields, required)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    fh = open(path, "a", newline="")
    return rows, fh, csv.DictWriter(fh, fieldnames=fields)
