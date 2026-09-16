#!/usr/bin/env python3
"""Draw a temperature x LR sweep of mnist under the evidence critic: docs/<name>-score.png.

    .venv/bin/python docs/mnist-evidence-report.py --name mnist-evidence

Reads runs/<name>/lr<L>-...-temperature<T>-seed<N>.csv (the reward every
trace_every epochs, one epoch a sample) and .json (the arm's record), and
runs/<name>-chance.json, the learning-off reward at each temperature on the
same network (AUTHORITY.md §8: the softmax reads the rest noise as evidence,
so chance sits under the uniform estimate's ln 0.1 and depends on T). One
panel a temperature, a curve an LR, the rolling mean over --window samples;
chance drawn at the learning-off level and the uniform estimate's -2.30.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK = "#333333"
COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--window", type=int, default=10, help="samples in the rolling mean")
    args = parser.parse_args()
    runs = ROOT / "runs" / args.name
    chance = {float(k): v for k, v in json.loads((ROOT / "runs" / f"{args.name}-chance.json").read_text()).items()}
    arms: dict[float, dict[float, tuple[list[int], list[float], dict]]] = {}
    pattern = re.compile(r"lr(?P<lr>[-0-9.e+]+)-.*temperature(?P<T>[-0-9.e+]+)-seed\d+\.csv$")
    for path in sorted(runs.glob("*.csv")):
        m = pattern.match(path.name)
        if not m:
            continue
        with open(path) as handle:
            rows = [(int(r["epoch"]), float(r["score"])) for r in csv.DictReader(handle)]
        record = json.loads(path.with_suffix(".json").read_text())
        arms.setdefault(float(m["T"]), {})[float(m["lr"])] = ([e for e, _ in rows], [s for _, s in rows], record)
    temperatures = sorted(arms)
    fig, axes = plt.subplots(1, len(temperatures), figsize=(5.2 * len(temperatures), 4.6), sharey=True)
    for ax, T in zip(axes, temperatures):
        for k, lr in enumerate(sorted(arms[T])):
            epochs, scores, record = arms[T][lr]
            rolling = [statistics.fmean(scores[max(0, i - args.window + 1):i + 1]) for i in range(len(scores))]
            ax.plot(epochs, rolling, color=COLOURS[k % len(COLOURS)], linewidth=1.8,
                    label=f"LR {lr:g}: last tenth {record['last_tenth']:.2f}, right {record['accuracy_last_tenth']:.2f}")
        ax.axhline(chance[T]["mean"], color=INK, linestyle="--", linewidth=1.0)
        ax.text(0, chance[T]["mean"] + 0.05, f"chance at T = {T:g}: learning off, {chance[T]['mean']:.2f}", color=INK, fontsize=8, va="bottom")
        ax.axhline(math.log(0.1), color="#999999", linestyle=":", linewidth=1.0)
        ax.text(0, math.log(0.1) + 0.05, "uniform estimate, ln 0.1", color="#999999", fontsize=8, va="bottom")
        ax.set_title(f"temperature {T:g}", color=INK, fontsize=11)
        ax.set_xlabel("epoch", color=INK)
        ax.grid(axis="y", color="#e6e6e6")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].set_ylabel(f"evidence reward ln q_label, rolling mean of {args.window} samples", color=INK)
    fig.suptitle(f"{args.name}: the evidence critic's reward over the run, one seed, by temperature and learning rate",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "docs" / f"{args.name}-score.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
