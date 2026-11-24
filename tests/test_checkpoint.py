"""Resume logic of the GPU scripts' checkpoints (no GPU or torch needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu"))

from checkpoint import Progress, fmt_secs, open_csv_checkpoint  # noqa: E402
from cospec.traces import TraceSet, synthetic_traces  # noqa: E402

FIELDS = ["kind", "n_seqs", "ms"]


def test_csv_checkpoint_appends_and_resumes(tmp_path):
    path = str(tmp_path / "sub" / "p.csv")
    rows, fh, w = open_csv_checkpoint(path, FIELDS, FIELDS)
    assert rows == []
    w.writerow(dict(kind="verify", n_seqs=1, ms=1.5))
    w.writerow(dict(kind="verify", n_seqs=2, ms=2.5))
    fh.close()
    rows, fh, w = open_csv_checkpoint(path, FIELDS, FIELDS)
    assert [r["n_seqs"] for r in rows] == ["1", "2"]
    w.writerow(dict(kind="draft", n_seqs=4, ms=0.5))
    fh.close()
    rows, fh, _ = open_csv_checkpoint(path, FIELDS, FIELDS)
    fh.close()
    assert [r["kind"] for r in rows] == ["verify", "verify", "draft"]
    with open(path) as f:
        assert f.read().count("kind,n_seqs,ms") == 1   # one header after several resumes


def test_csv_checkpoint_drops_torn_and_incomplete_rows(tmp_path):
    path = str(tmp_path / "p.csv")
    with open(path, "w") as f:
        f.write("kind,n_seqs,ms\nverify,1,1.5\nverify,2,\nverify,3,9.")   # missing value, then torn line
    rows, fh, w = open_csv_checkpoint(path, FIELDS, FIELDS)
    w.writerow(dict(kind="verify", n_seqs=3, ms=3.5))
    fh.close()
    rows, fh, _ = open_csv_checkpoint(path, FIELDS, FIELDS)
    fh.close()
    assert [(r["n_seqs"], r["ms"]) for r in rows] == [("1", "1.5"), ("3", "3.5")]


def test_csv_checkpoint_fresh_starts_over(tmp_path):
    path = str(tmp_path / "p.csv")
    _, fh, w = open_csv_checkpoint(path, FIELDS, FIELDS)
    w.writerow(dict(kind="verify", n_seqs=1, ms=1.0))
    fh.close()
    rows, fh, _ = open_csv_checkpoint(path, FIELDS, FIELDS, fresh=True)
    fh.close()
    assert rows == []


def test_progress_reports_counts(capsys):
    p = Progress(4, "x", done=1)
    p.start()
    p.step("a")
    p.step("b", n=2)
    out = capsys.readouterr().out
    assert "1/4 already done, 3 to go" in out
    assert "[x 2/4 50%" in out and "[x 4/4 100%" in out


def test_fmt_secs():
    assert fmt_secs(65) == "1:05"
    assert fmt_secs(3725) == "1:02:05"
    assert fmt_secs(float("nan")) == "?"


def test_trace_save_is_atomic_and_overwrites(tmp_path):
    path = str(tmp_path / "t.npz")
    a, b = synthetic_traces(3, seed=0), synthetic_traces(5, seed=1)
    a.save(path)
    b.save(path)
    assert os.listdir(tmp_path) == ["t.npz"]           # no temp file left behind
    loaded = TraceSet.load(path)
    assert loaded.n_requests == 5
    np.testing.assert_array_equal(loaded.acc_len, b.acc_len)
