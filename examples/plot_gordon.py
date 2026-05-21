#!/usr/bin/env python3
"""
Plot the CSV produced by `examples/gordon.rs`.

Usage:
    cargo run --release --example gordon > gordon.csv
    python examples/plot_gordon.py gordon.csv [--save out.png]

Three panels:
  1. Truth vs estimated state x over time.
  2. Tracking error (|x_truth - x_est|) over time.
  3. Effective sample size over time.

Requires Python 3 + matplotlib. No other dependencies.
"""

from __future__ import annotations

import argparse
import csv
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
    ap.add_argument("csv", help="CSV produced by the gordon example")
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
    err = [abs(t - e) for t, e in zip(d["truth_x"], d["est_x"])]

    fig, (ax_state, ax_err, ax_ess) = plt.subplots(
        3, 1, figsize=(9, 9), constrained_layout=True,
    )

    # Panel 1: truth vs estimate over time.
    ax_state.plot(d["step"], d["truth_x"], label="truth", linewidth=1.5)
    ax_state.plot(
        d["step"], d["est_x"], label="weighted-mean estimate",
        linewidth=1.0, linestyle="--", alpha=0.85,
    )
    ax_state.set_xlabel("step")
    ax_state.set_ylabel("x")
    ax_state.set_title("Gordon benchmark — truth vs estimate")
    ax_state.grid(True, alpha=0.3)
    ax_state.legend(loc="best")

    # Panel 2: tracking error.
    ax_err.plot(d["step"], err, linewidth=0.9)
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("tracking error")
    ax_err.set_title(
        "Tracking error  (|x_truth − x_est|)"
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
