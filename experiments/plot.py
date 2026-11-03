"""Plot sweep results.

  python experiments/plot.py results/main.jsonl --x load --y goodput_rps --out figs/goodput.png
  python experiments/plot.py results/skew.jsonl --x affinity_frac --y goodput_rps --out figs/skew.png
  python experiments/plot.py results/main.jsonl --table   # mean over seeds, as text
"""
import argparse
import json
from collections import defaultdict

import numpy as np

ORDER = ["nosd", "fixed4", "perrank", "cospec_tput", "cospec_d1", "cospec_d1d2", "oracle"]


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                print("skipping failed run:", r["error"])
                continue
            r.update({k: v for k, v in r["config"].items() if k not in r})
            rows.append(r)
    return rows


def aggregate(rows, x, y, facets):
    """{facet: {system: (xs, mean, std)}} averaging over seeds."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        acc[tuple(r[f] for f in facets)][r["system"]][r[x]].append(r[y])
    out = {}
    for fk, systems in acc.items():
        out[fk] = {}
        for s, by_x in systems.items():
            xs = sorted(by_x)
            out[fk][s] = (np.array(xs), np.array([np.mean(by_x[v]) for v in xs]),
                          np.array([np.std(by_x[v]) for v in xs]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--x", default="load")
    ap.add_argument("--y", default="goodput_rps")
    ap.add_argument("--facets", default="mix,n_ranks", help="comma-separated columns, one subplot each")
    ap.add_argument("--out", default=None)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    rows = load(args.results)
    facets = [f for f in args.facets.split(",") if f and f != args.x]
    agg = aggregate(rows, args.x, args.y, facets)
    keys = sorted(agg)
    systems = [s for s in ORDER if any(s in agg[k] for k in keys)]

    if args.table or not args.out:
        for k in keys:
            print("  ".join(f"{f}={v}" for f, v in zip(facets, k)))
            xs = sorted({v for s in agg[k].values() for v in s[0]})
            print(f"  {'system':14s}" + "".join(f"{args.x}={v:<10}" for v in xs))
            for s in systems:
                if s not in agg[k]:
                    continue
                sx, m, _ = agg[k][s]
                vals = dict(zip(sx, m))
                print(f"  {s:14s}" + "".join(f"{vals.get(v, float('nan')):<{len(args.x) + 11}.2f}" for v in xs))
            print()
        if not args.out:
            return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(4, len(keys))
    nrows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.6 * cols, 2.8 * nrows), squeeze=False, sharey=False)
    colors = dict(zip(ORDER, plt.get_cmap("tab10").colors))
    for ax, k in zip(axes.flat, keys):
        for s in systems:
            if s in agg[k]:
                xs, m, sd = agg[k][s]
                ax.errorbar(xs, m, yerr=sd, label=s, color=colors[s], marker="o", ms=3, capsize=2, lw=1.2)
        ax.set_title(", ".join(f"{f}={v}" for f, v in zip(facets, k)), fontsize=9)
        ax.set_xlabel(args.x)
        ax.set_ylabel(args.y)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
