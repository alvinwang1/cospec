"""Run one simulated experiment and print its metrics.

  python experiments/run_sim.py --policy cospec --router spec_aware --n-ranks 16 --load 0.8
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cospec.experiment import DEFAULTS, run_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument(flag, type=type(v) if v is not None else str, default=None)
    ap.add_argument("--out", help="append the metrics as a JSON line to this file")
    args = vars(ap.parse_args())
    out_path = args.pop("out")
    res = run_experiment(args)
    print(json.dumps(res, indent=2))
    if out_path:
        with open(out_path, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
