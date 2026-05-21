#!/usr/bin/env python3
"""
Plot the CSV produced by `examples/mlnlg.rs`.

Usage:
    cargo run --release --example mlnlg > mlnlg.csv
    python examples/plot_mlnlg.py mlnlg.csv [--save out.png]

Three panels:
  1. Nonlinear state ξ: truth vs estimate over time.
  2. Tracking error (Euclidean distance between full truth and estimate
     state vectors [ξ, z₀, z₁, z₂, z₃]) over time.
  3. Effective sample size over time.

Requires Python 3 + matplotlib. No other dependencies.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys


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
    ap.add_argument("csv", help="CSV produced by the mlnlg example")
    ap.add_argument(
        "--save", metavar="PATH",
        help="Write the figure to PATH instead of opening a window.",
    )
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write(
            "matplotlib is required: pip install matplotlib (or apt install "
            "python3-matplotlib)\n"
        )
        sys.exit(1)

    d = load_csv(args.csv)

    # Full-state Euclidean distance: sqrt(Δξ² + Δz₀² + Δz₁² + Δz₂² + Δz₃²)
    dims = [("truth_xi", "est_xi"),
            ("truth_z0", "est_z0"), ("truth_z1", "est_z1"),
            ("truth_z2", "est_z2"), ("truth_z3", "est_z3")]
    err = [
        math.sqrt(sum((d[tk][i] - d[ek][i]) ** 2 for tk, ek in dims))
        for i in range(len(d["step"]))
    ]

    fig, (ax_xi, ax_err, ax_ess) = plt.subplots(
        3, 1, figsize=(9, 9), constrained_layout=True,
    )

    # Panel 1: ξ truth vs estimate.
    ax_xi.plot(d["step"], d["truth_xi"], label="truth", linewidth=1.5)
    ax_xi.plot(
        d["step"], d["est_xi"], label="weighted-mean estimate",
        linewidth=1.0, linestyle="--", alpha=0.85,
    )
    ax_xi.set_xlabel("step")
    ax_xi.set_ylabel("ξ")
    ax_xi.set_title("MLNLG — nonlinear state ξ: truth vs estimate")
    ax_xi.grid(True, alpha=0.3)
    ax_xi.legend(loc="best")

    # Panel 2: tracking error.
    ax_err.plot(d["step"], err, linewidth=0.9)
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("tracking error")
    ax_err.set_title(
        "Tracking error  (Euclidean distance between truth and estimate state vectors)"
    )
    ax_err.grid(True, alpha=0.3)

    # Panel 3: ESS.
    ax_ess.plot(d["step"], d["ess"], linewidth=0.9)
    ax_ess.set_xlabel("step")
    ax_ess.set_ylabel("effective sample size")
    ax_ess.set_title("Effective sample size  (smaller = cloud is degenerating)")
    ax_ess.grid(True, alpha=0.3)

    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"wrote {args.save}", file=sys.stderr)
    else:
        plt.show()


if __name__ == "__main__":
    main()
