#!/usr/bin/env python3
"""Overlay a sweep's groups, score over the run, on one axis: docs/<name>-score.png.

    .venv/bin/python docs/sweep-trace-report.py --name goo455-ff-epoch-100k --knob interval

Reads runs/<name>/<arm>-seed<N>.csv, the traces docs/rust-sweep.py writes: one
epoch's score every trace_every epochs, so each point is a single epoch and the
curve is smoothed over --window samples (default 10, so 10,000 epochs at the
usual trace_every of 1,000). Arms are grouped by everything in their name but
the seed -- the same grouping docs/mnist-estimator-report.py uses -- and each
group is drawn as the mean over its seeds with a band of one standard
deviation. Where --chance names a directory of learning-off arms (LR 0), each
group's own resting score is drawn dashed in its colour, since the level a run
is read against is the network's own rest and not a number shared between
epoch lengths (AUTHORITY.md §8, docs/mnist-evidence-h0.md).

Companion to docs/goo-trace-report.py, which draws one arm's seeds in detail;
this one answers "which group climbed further" at a glance.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def group_of(stem: str) -> str:
    return re.sub(r"-seed\d+$", "", stem)


def smooth(xs: list[float], window: int) -> list[float]:
    return [statistics.fmean(xs[max(0, i - window + 1):i + 1]) for i in range(len(xs))]


def traces(directory: Path, window: int) -> dict[str, list[tuple[list[int], list[float]]]]:
    out: dict[str, list[tuple[list[int], list[float]]]] = {}
    for path in sorted(directory.glob("*.csv")):
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        epochs = [int(r["epoch"]) for r in rows]
        scores = smooth([float(r["score"]) for r in rows], window)
        out.setdefault(group_of(path.stem), []).append((epochs, scores))
    return out


def resting(directory: Path, knob: str) -> dict[str, float]:
    """Each learning-off group's score, keyed by its knob value."""
    rest: dict[str, list[float]] = {}
    for path in directory.glob("*.json"):
        if path.name.endswith("-network.json"):
            continue
        record = json.loads(path.read_text())
        if "arm" not in record:
            continue
        found = re.search(rf"{knob}(-?[\d.]+)", record["arm"])
        if found:
            rest.setdefault(found.group(1).rstrip("."), []).append(record["last_tenth"])
    return {k: statistics.fmean(v) for k, v in rest.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--knob", default=None, help="the swept knob, used to label the groups and match --chance")
    parser.add_argument("--window", type=int, default=10, help="samples per rolling mean")
    parser.add_argument("--chance", default=None, metavar="NAME", help="runs/NAME holding the learning-off (LR 0) arms")
    args = parser.parse_args()

    groups = traces(ROOT / "runs" / args.name, args.window)
    if not groups:
        raise SystemExit(f"no traces in runs/{args.name}")
    rest = resting(ROOT / "runs" / args.chance, args.knob) if args.chance and args.knob else {}

    figure, axes = plt.subplots(figsize=(9, 5.5))
    for colour, (name, runs) in zip(plt.rcParams["axes.prop_cycle"].by_key()["color"], sorted(groups.items())):
        length = min(len(s) for _, s in runs)
        epochs = runs[0][0][:length]
        stack = np.array([s[:length] for _, s in runs])
        mean, deviation = stack.mean(axis=0), stack.std(axis=0)
        found = re.search(rf"{args.knob}(-?[\d.]+)", name) if args.knob else None
        value = found.group(1).rstrip(".") if found else name
        label = f"{args.knob} {value}" if args.knob else name
        axes.plot(epochs, mean, color=colour, label=f"{label} ({len(runs)} seeds)")
        axes.fill_between(epochs, mean - deviation, mean + deviation, color=colour, alpha=0.15)
        if value in rest:
            axes.axhline(rest[value], color=colour, linestyle=":", linewidth=1)

    axes.set_xlabel("epoch")
    axes.set_ylabel(f"score, rolling mean over {args.window} samples")
    axes.set_title(f"{args.name}: score over the run" + (", dotted = that group's learning-off rest" if rest else ""))
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    out = ROOT / "docs" / f"{args.name}-score.png"
    figure.savefig(out, dpi=140)
    print(out)


if __name__ == "__main__":
    main()
