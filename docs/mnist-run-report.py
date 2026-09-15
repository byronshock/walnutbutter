#!/usr/bin/env python3
"""Draw the reward of one or more mnist runs (AUTHORITY.md §8) over epochs: docs/<out>.png.

    .venv/bin/python docs/mnist-run-report.py --out mnist-p005-rules runs/mnist-p005 runs/mnist-p005-equal

Each run folder holds one <arm>.csv of (epoch, score) sampled every so often, and
<arm>.json with the run's record. The class critic pays 1 or 0 an epoch, so the
figure shows a rolling mean over `--window` samples, with chance for ten classes
drawn at 0.1 (a tie loses, so chance is a little under it).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INK, GREY = "#2b2b2b", "#9a9a9a"
COLOURS = ["#c0392b", "#1f6fb2", "#2e8b57", "#8e44ad"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", help="run folders under runs/, each with one arm's csv and json")
    parser.add_argument("--out", required=True, help="the figure's name under docs/, without .png")
    parser.add_argument("--window", type=int, default=20, help="samples per rolling mean")
    parser.add_argument("--labels", nargs="*", default=None, help="one legend label per run, in order")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=120)
    lines = []
    for k, folder in enumerate(args.runs):
        folder = ROOT / folder if not Path(folder).is_absolute() else Path(folder)
        csv_path = next(folder.glob("*.csv"))
        record = json.loads(next(folder.glob("*.json")).read_text())
        with open(csv_path) as handle:
            rows = [(int(r["epoch"]), float(r["score"])) for r in csv.DictReader(handle)]
        epochs = [e for e, _ in rows]
        scores = [s for _, s in rows]
        step = epochs[1] - epochs[0] if len(epochs) > 1 else epochs[0]
        rolling = [statistics.fmean(scores[max(0, i - args.window + 1):i + 1]) for i in range(len(scores))]
        label = args.labels[k] if args.labels and k < len(args.labels) else folder.name
        colour = COLOURS[k % len(COLOURS)]
        ax.plot(epochs, rolling, color=colour, linewidth=2.0, label=f"{label}: last tenth {record['last_tenth']:.3f}")
        lines.append(f"{label}: mean {record['mean']:.3f}, last tenth {record['last_tenth']:.3f}, "
                     f"{record['epochs_per_second']} epochs/s, stuck on/off {record['stuck_on']}/{record['stuck_off']}")
    ax.axhline(0.1, color=INK, linestyle="--", linewidth=1.0)
    ax.text(ax.get_xlim()[1], 0.105, "chance, ten classes", ha="right", va="bottom", color=INK, fontsize=9)
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel(f"reward, rolling mean of {args.window} samples ({args.window * step:,} epochs)", color=INK)
    ax.set_ylim(0.0, max(0.5, max(max(l.get_ydata()) for l in ax.get_lines()) + 0.05))
    ax.grid(axis="y", color="#e6e6e6")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("mnist: the class critic's reward over the run", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / f"{args.out}.png")
    print("\n".join(lines))
    print(f"docs/{args.out}.png")


if __name__ == "__main__":
    main()
