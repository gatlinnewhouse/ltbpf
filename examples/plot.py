#!/usr/bin/env python3
"""
Plot the CSV produced by `examples/vehicle.rs`.

Usage:
    cargo run --release --example vehicle > out.csv
    python examples/plot.py out.csv [--save out.png]

Three panels:
  1. Truth vs estimated track in 2D.
  2. Tracking error (Euclidean distance) over time.
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
    ap.add_argument("csv", help="CSV produced by the vehicle example")
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
    # Give the track panel twice the height of the other two — the
    # equal-aspect constraint on it tends to squeeze it otherwise.
    fig, (ax_track, ax_err, ax_ess) = plt.subplots(
        3, 1, figsize=(9, 12),
        gridspec_kw={"height_ratios": [2, 1, 1]},
        constrained_layout=True,
    )

    # Panel 1: tracks overlaid.
    ax_track.plot(d["truth_x"], d["truth_y"], label="truth", linewidth=1.5)
    ax_track.plot(
        d["est_x"], d["est_y"], label="weighted-mean estimate",
        linewidth=1.0, alpha=0.85,
    )
    ax_track.scatter([d["truth_x"][0]], [d["truth_y"][0]],
                     marker="o", s=40, color="black", label="start", zorder=5)
    ax_track.set_xlabel("x")
    ax_track.set_ylabel("y")
    ax_track.set_title("Truth vs estimated track")
    ax_track.set_aspect("equal", adjustable="datalim")
    ax_track.grid(True, alpha=0.3)
    ax_track.legend(loc="best")

    # Panel 2: tracking error over time.
    ax_err.plot(d["step"], d["err"], linewidth=0.9)
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("tracking error  (m)")
    ax_err.set_title(
        "Tracking error  (Euclidean distance between truth and estimate)"
    )
    ax_err.grid(True, alpha=0.3)

    # Panel 3: ESS over time.
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
