#!/usr/bin/env python3
"""
Compare SIS, SIR, APF, and RPF on the shared vehicle benchmark.

Generate the CSVs first:
    cargo run --release --example sis  > sis.csv
    cargo run --release --example sir  > sir.csv
    cargo run --release --example apf  > apf.csv
    cargo run --release --example rpf  > rpf.csv

Usage:
    python examples/plot_compare.py sis.csv sir.csv apf.csv rpf.csv [--save out.png]

Three panels:
  1. Truth vs estimated tracks in 2D (one estimate line per filter).
  2. Tracking error (Euclidean distance) over time.
  3. Effective sample size over time.

Requires Python 3 + matplotlib. No other dependencies.
"""

from __future__ import annotations

import argparse
import csv
import sys


FILTERS = [
    ("sis", "SIS", "solid"),
    ("sir", "SIR", "dashed"),
    ("apf", "APF", "dashdot"),
    ("rpf", "RPF", "dotted"),
]


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cols = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            cols[k].append(float(v))
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("sis_csv",  help="CSV from the sis example")
    ap.add_argument("sir_csv",  help="CSV from the sir example")
    ap.add_argument("apf_csv",  help="CSV from the apf example")
    ap.add_argument("rpf_csv",  help="CSV from the rpf example")
    ap.add_argument(
        "--save", metavar="PATH",
        help="Write the figure to PATH instead of opening a window.",
    )
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write(
            "matplotlib is required: pip install matplotlib\n"
        )
        sys.exit(1)

    csvs = [args.sis_csv, args.sir_csv, args.apf_csv, args.rpf_csv]
    datasets = {}
    for (key, _, _ls), path in zip(FILTERS, csvs):
        datasets[key] = load_csv(path)

    # All CSVs share the same ground truth — use sis as the reference.
    truth = datasets["sis"]

    fig, (ax_track, ax_err, ax_ess) = plt.subplots(
        3, 1, figsize=(9, 12),
        gridspec_kw={"height_ratios": [2, 1, 1]},
        constrained_layout=True,
    )

    # Panel 1: tracks.
    ax_track.plot(
        truth["truth_x"], truth["truth_y"],
        label="truth", color="black", linewidth=1.5, zorder=5,
    )
    ax_track.scatter(
        [truth["truth_x"][0]], [truth["truth_y"][0]],
        marker="o", s=40, color="black", label="start", zorder=6,
    )
    for key, label, ls in FILTERS:
        d = datasets[key]
        ax_track.plot(d["est_x"], d["est_y"], label=label, linestyle=ls, linewidth=1.0, alpha=0.8)

    ax_track.set_xlabel("x")
    ax_track.set_ylabel("y")
    ax_track.set_title("Truth vs estimated tracks")
    ax_track.set_aspect("equal", adjustable="datalim")
    ax_track.grid(True, alpha=0.3)
    ax_track.legend(loc="best")

    # Panel 2: tracking error.
    steps = truth["step"]
    for key, label, ls in FILTERS:
        d = datasets[key]
        ax_err.plot(steps, d["err"], label=label, linestyle=ls, linewidth=0.9)

    ax_err.set_xlabel("step")
    ax_err.set_ylabel("tracking error  (m)")
    ax_err.set_title("Tracking error  (Euclidean distance between truth and estimate)")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    # Panel 3: ESS.
    for key, label, ls in FILTERS:
        d = datasets[key]
        ax_ess.plot(steps, d["ess"], label=label, linestyle=ls, linewidth=0.9)

    ax_ess.set_xlabel("step")
    ax_ess.set_ylabel("effective sample size")
    ax_ess.set_title("Effective sample size  (smaller = cloud is degenerating)")
    ax_ess.grid(True, alpha=0.3)
    ax_ess.legend(loc="best")

    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"wrote {args.save}", file=sys.stderr)
    else:
        plt.show()


if __name__ == "__main__":
    main()
