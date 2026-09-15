#!/usr/bin/env python3
"""Plot one arm's score over the run, every seed, from a rust-sweep's CSV traces.

    .venv/bin/python docs/goo-trace-report.py --name goo-count-threshold --arm threshold0.5-goo80

Reads runs/<name>/<arm>-seed<N>.csv: one epoch's score every trace_every
epochs, so each point is a single epoch's reward in eighths and the curve is
smoothed over --window samples (default 10, so 10,000 epochs at the usual
trace_every of 1,000). Every seed thin, their mean bold, chance dashed.
Writes docs/<name>-<arm>-trace.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--arm", default="", help="the arm's name without its -seed<N>, e.g. threshold0.5-goo80; empty for a seed-only run")
    parser.add_argument("--window", type=int, default=10, help="samples per rolling mean")
    parser.add_argument("--seed", type=int, default=None, help="draw this one seed, its raw samples under the smoothed line")
    args = parser.parse_args()
    runs = ROOT / "runs" / args.name
    stem = f"{args.arm}-seed" if args.arm else "seed"  # a run that sweeps only the seed names its arms seed<N>
    seeds = sorted(int(p.stem[len(stem):]) for p in runs.glob(f"{stem}*.csv"))
    if args.seed is not None:
        seeds = [s for s in seeds if s == args.seed]
    if not seeds:
        raise SystemExit(f"no {stem}<N>.csv under {runs}" + (f" for seed {args.seed}" if args.seed is not None else ""))
    traces, finals = {}, {}
    for seed in seeds:
        with open(runs / f"{stem}{seed}.csv") as handle:
            rows = [(int(r["epoch"]), float(r["score"])) for r in csv.DictReader(handle)]
        traces[seed] = rows
        summary = runs / f"{stem}{seed}.json"
        finals[seed] = json.loads(summary.read_text())["last_tenth"] if summary.exists() else None
    plot(args, traces, finals)


def smooth(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        out.append(statistics.fmean(chunk))
    return out


def plot(args, traces, finals) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#1a1a19", "#5f5e58", "#e6e5df", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    seeds = sorted(traces)
    xs = [e for e, _ in traces[seeds[0]]]
    smoothed = {s: smooth([v for _, v in traces[s]], args.window) for s in seeds}
    one = len(seeds) == 1
    if one:  # the raw samples, each a single epoch's score in eighths, under the smoothed line
        s = seeds[0]
        ax.scatter(xs, [v for _, v in traces[s]], s=6, color=BLUE, alpha=0.35, linewidths=0, label="one epoch's score, every 1,000")
    for s in seeds:
        ys = smoothed[s]
        colour = ORANGE if finals.get(s) is not None and finals[s] < 0.52 else BLUE
        ax.plot(xs[:len(ys)], ys, color=colour, linewidth=0.8, alpha=0.45)
    n = min(len(v) for v in smoothed.values())
    mean = [statistics.fmean(smoothed[s][i] for s in seeds) for i in range(n)]
    ax.plot(xs[:n], mean, color=INK, linewidth=2.0,
            label=(f"seed {seeds[0]}, rolling mean of {args.window} samples" if one else f"mean of {len(seeds)} seeds"))
    ax.axhline(0.5, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate("chance", (0.995, 0.5), xycoords=("axes fraction", "data"), ha="right", va="bottom", color=INK2, fontsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_xlabel("epoch", color=INK2, fontsize=9)
    step = xs[1] - xs[0] if len(xs) > 1 else 1
    ax.set_ylabel(f"score, rolling mean of {args.window} sampled epochs ({args.window * step:,} epochs)", color=INK2, fontsize=8)
    stuck = [s for s in seeds if finals.get(s) is not None and finals[s] < 0.52]
    if not one:
        note = f"thin: each seed" + (f"; orange: seed{'s' if len(stuck) > 1 else ''} {', '.join(map(str, stuck))} ended at chance" if stuck else "")
        ax.plot([], [], color=BLUE, linewidth=0.8, alpha=0.6, label=note)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=INK2)
    ax.set_ylim(0.35, 1.0)
    who = f"seed {seeds[0]}" if one else (args.arm or "every seed")
    tail = f" -- {finals[seeds[0]]:.3f} over the last tenth" if one and finals.get(seeds[0]) is not None else ""
    fig.suptitle(f"{args.name}: {args.arm + ', ' if one and args.arm else ''}{who}, score over the run{tail}", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "docs" / (f"{args.name}-{args.arm}-trace.png" if args.arm else f"{args.name}-trace.png")
    if one:
        out = out.with_name(out.stem + f"-seed{seeds[0]}.png")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(out)
    print("last-tenth per seed:", " ".join(f"{s}:{finals[s]:.3f}" for s in seeds if finals.get(s) is not None))
    # when does the mean first cross 0.55, and where does each seed's smoothed trace settle?
    first = next((xs[i] for i, m in enumerate(mean) if m >= 0.55), None)
    print("mean first reaches 0.55 at epoch:", first)


if __name__ == "__main__":
    main()
