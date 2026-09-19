#!/usr/bin/env python3
"""Draw an ordered knob's estimator over the run, one hue light to dark: docs/<name>-<knob>.png.

    .venv/bin/python docs/sweep-estimator-report.py --name goo455-ff-delta-100k --knob delta \
        --out goo455-ff-delta-comeapart --subtitle "zero-hidden goo of 455, 100 ms epochs, LR 0.0075"

Reads runs/<name>/*.json, the records docs/rust-sweep.py writes per arm, each
carrying `estimator`: at every trace interval the correlation of the weight
change since the start with the supervised direction d, and the correlation of
the change since the previous interval (AUTHORITY.md §8, §9.12). Arms are
grouped by their value of --knob and each group drawn as the mean over its
seeds.

Two panels, and the lower one is the point of the figure: the cumulative
correlation says how much a run has banked, and only the per-interval
correlation says whether it is still learning *now*. A run whose cumulative
curve has flattened and whose window correlation sits at zero has stopped;
one whose cumulative curve has flattened while the window is still positive
is still climbing and merely slowly.

Where `mnist-estimator-report.py` draws a handful of unordered arms in the
colour cycle with a legend of full arm names, this one is for a knob with an
**order** -- delta, the epoch length, a count. An ordered quantity gets one
hue running light to dark, so the ramp carries the ordering and the reader
does not have to learn which arbitrary colour means more; each line is named
at its own right-hand end, so identity never depends on colour alone and
there is no legend box to look back and forth to. Label text stays in ink,
never in the line's colour: the line end beside it carries the identity.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, RULE = "#1f2328", "#57606a", "#d8dee4"


def value_of(record: dict, knob: str) -> float | None:
    """The arm's value for this knob: the record's own field, else parsed from its name."""
    if knob in record and isinstance(record[knob], (int, float)):
        return float(record[knob])
    found = re.search(rf"{knob}(-?[\d.]+)", record.get("arm", ""))
    return float(found.group(1).rstrip(".")) if found else None


def read(name: str, knob: str) -> tuple[dict, dict]:
    cumulative: dict[float, dict[int, list[float]]] = {}
    window: dict[float, dict[int, list[float]]] = {}
    for path in glob.glob(str(ROOT / "runs" / name / "*.json")):
        if path.endswith("-network.json"):
            continue
        record = json.loads(Path(path).read_text())
        if "arm" not in record:
            continue
        value = value_of(record, knob)
        if value is None:
            continue
        for point in record.get("estimator") or []:
            cumulative.setdefault(value, {}).setdefault(point["epoch"], []).append(point["corr_cum"])
            window.setdefault(value, {}).setdefault(point["epoch"], []).append(point["corr_window"])
    return cumulative, window


def series(store: dict, key: float):
    epochs = sorted(store[key])
    mean = np.array([statistics.fmean(store[key][e]) for e in epochs])
    deviation = np.array([statistics.stdev(store[key][e]) if len(store[key][e]) > 1 else 0.0 for e in epochs])
    return np.array(epochs), mean, deviation


def roll(values: np.ndarray, window: int) -> np.ndarray:
    return np.array([values[max(0, i - window + 1):i + 1].mean() for i in range(len(values))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True, help="the sweep, under runs/")
    parser.add_argument("--knob", required=True, help="the ordered knob to group and shade by, e.g. delta")
    parser.add_argument("--out", default=None, help="the figure's name under docs/ (default: <name>-<knob>)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--subtitle", default="", help="the configuration the arms share")
    parser.add_argument("--window", type=int, default=10, help="trace samples per rolling mean in the lower panel")
    parser.add_argument("--mark", type=float, default=None, help="a value to mark as the shipped one")
    args = parser.parse_args()

    cumulative, window = read(args.name, args.knob)
    if not cumulative:
        raise SystemExit(f"no arms with a {args.knob} in runs/{args.name}")
    values = sorted(cumulative)
    # One hue, more knob = darker, so the ramp itself carries the ordering. Deliberately
    # a hue the surrounding interface does not use: information reads as information when
    # it does not land on the chrome's own colours (Byron, September 19, 2026, of the four
    # Mondrian lines his De Stijl kit paints chrome from -- "you are meant to stay out of
    # it ... it pops better when it doesn't"). Measured: these six sit 40.6 from the
    # nearest of that palette's nineteen points, where a blue ramp's dark end sat 21.1.
    shades = [plt.cm.Purples(x) for x in np.linspace(0.40, 0.97, len(values))]

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8.6), sharex=True,
                                         gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.13})
    ends: list[tuple[float, str, float]] = []  # a line's end, its label: placed after, so crowded ones can be nudged apart
    for value, shade in zip(values, shades):
        epochs, mean, deviation = series(cumulative, value)
        top.plot(epochs, mean, color=shade, linewidth=2)
        top.fill_between(epochs, mean - deviation, mean + deviation, color=shade, alpha=0.12, linewidth=0)
        label = f"{value:g}" + ("  (shipped)" if args.mark is not None and value == args.mark else "")
        ends.append((mean[-1], label, epochs[-1]))

        epochs, mean, _ = series(window, value)
        bottom.plot(epochs, roll(mean, args.window), color=shade, linewidth=2)
        if value in (values[0], values[-1]):  # only the extremes: the middle labels collide, and the top panel names every line
            bottom.annotate(f"{value:g}", (epochs[-1], roll(mean, args.window)[-1]), xytext=(8, 0),
                            textcoords="offset points", color=INK, fontsize=10, va="center")

    # Curves that finish close together would stamp their labels on each other, so walk
    # up the list and push each one clear of the last by a fixed share of the axis.
    ends.sort()
    reach = top.get_ylim()[1] - top.get_ylim()[0]
    placed: list[float] = []
    for height, label, epoch in ends:
        if placed and height - placed[-1] < reach * 0.028:
            height = placed[-1] + reach * 0.028
        placed.append(height)
        top.annotate(label, (epoch, height), xytext=(8, 0), textcoords="offset points",
                     color=INK, fontsize=10, va="center")

    top.set_ylabel("correlation of the whole weight change\nwith the supervised direction", color=INK, fontsize=10)
    top.set_title((args.title or f"{args.name}: the estimator against {args.knob}")
                  + (f"\n{args.subtitle}" if args.subtitle else ""),
                  color=INK, fontsize=13, loc="left", pad=14)
    bottom.set_ylabel(f"correlation of each interval's own change\n(rolling mean, {args.window} trace samples)",
                      color=INK, fontsize=10)
    bottom.set_xlabel("epoch", color=INK, fontsize=10)
    bottom.axhline(0, color=MUTED, linewidth=1)

    last = max(max(v) for v in cumulative.values())
    for axes in (top, bottom):
        axes.grid(color=RULE, linewidth=0.7)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(RULE)
        axes.tick_params(colors=MUTED, labelsize=9)
        axes.set_xlim(0, last * 1.08)
    bottom.annotate("no longer learning", (last * 0.02, 0), xytext=(0, -16), textcoords="offset points",
                    color=MUTED, fontsize=9)
    top.annotate(args.knob, (last, top.get_ylim()[1] * 0.96), xytext=(8, 0), textcoords="offset points",
                 color=MUTED, fontsize=10, va="center", fontweight="bold")

    figure.subplots_adjust(left=0.13, right=0.88, top=0.90, bottom=0.08)
    out = ROOT / "docs" / f"{args.out or f'{args.name}-{args.knob}'}.png"
    figure.savefig(out, dpi=150, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
