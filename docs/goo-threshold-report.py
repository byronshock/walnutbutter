#!/usr/bin/env python3
"""Read a one-knob rust-sweep of goo off disk and write docs/<name>.md and docs/<name>-score.png.

    .venv/bin/python docs/goo-threshold-report.py --name goo-thr-hebb --knob threshold

Reads runs/<name>/<knob><V>-seed<N>.json, the summary docs/rust-sweep.py writes
per arm: the mean over the last tenth of the run, and how many neurons ended
stuck on and off. One row per knob value, averaged over seeds, with the seed
range; a paired-on-the-seed t of each level against the default; and a figure
of last-tenth accuracy against the knob, mean and min-max band. For THRESHOLD
on goo the report also gives the effective theta, since goo scales it by its
fan-in (AUTHORITY.md §5.2), and the floor over theta.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def paired_t(a: list[float], b: list[float]) -> tuple[float, float]:
    d = [x - y for x, y in zip(a, b)]
    mean = statistics.fmean(d)
    if len(d) < 2:
        return mean, float("nan")
    sd = statistics.stdev(d)
    return mean, (mean / (sd / math.sqrt(len(d))) if sd else float("inf"))


def main() -> None:
    from walnutbutter.constants import MINIMUM_POTENTIAL, THRESHOLD, THRESHOLD_FAN_IN
    from walnutbutter.goo import DEFAULT_COUNT

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--knob", default="threshold")
    parser.add_argument("--default", type=float, default=None, help="the level to pair every other level against")
    args = parser.parse_args()
    runs = ROOT / "runs" / args.name
    # the knob's value, then any knobs the sweep held fixed (goo60, say), then the seed
    pattern = re.compile(rf"^{args.knob}(?P<value>[-0-9.e+]+)(?:-[a-z_]+[-0-9.e+]+)*-seed(?P<seed>\d+)\.json$")
    levels: dict[float, dict[int, dict]] = {}
    for path in sorted(runs.glob(f"{args.knob}*seed*.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        levels.setdefault(float(m["value"]), {})[int(m["seed"])] = json.loads(path.read_text())
    if not levels:
        raise SystemExit(f"nothing under {runs} matches {args.knob}<V>-seed<N>.json -- has the sweep finished?")
    seeds = sorted(set.intersection(*(set(v) for v in levels.values())))
    if not seeds:
        raise SystemExit("no seed has finished at every level yet")
    default = args.default if args.default is not None else (THRESHOLD if args.knob == "threshold" else None)
    scale = (DEFAULT_COUNT - 1) / THRESHOLD_FAN_IN  # goo's fan-in over the reference: how far the axis is stretched
    one = next(iter(next(iter(levels.values())).values()))
    epochs_per_second = statistics.fmean(r["epochs_per_second"] for v in levels.values() for r in v.values())

    def last(level: float) -> list[float]:
        return [levels[level][s]["last_tenth"] for s in seeds]

    eligibility = one.get("eligibility", "wrong_hebb")  # what the arms ran (§6.7); the driver records it since September 14, 2026
    if eligibility == "hebb" and "count_memory" not in one:
        eligibility = "wrong_hebb"  # a record from before September 16, 2026: hebb then named the +-1 rule renamed that day
    problem = one.get("problem", "copy")  # and the problem, since September 15; before that every goo sweep was copy
    # chance by critic: the class critic pays a tenth by luck, the evidence critic ln 0.1, the row and graded critics a half
    chance = {"class": 0.1, "evidence": math.log(0.1)}.get(one.get("critic"), 0.5)
    lines = [
        f"# Sweep {args.name} ({date.today().strftime('%B %-d, %Y')})",
        "",
        f"{args.knob.upper()} against {len(seeds)} seeds of goo (AUTHORITY.md §3.4), the {problem} problem (§8), the reinforce "
        f"rule with the {eligibility} eligibility, the Rust wave loop (§6.15), homeostasis and un-sticking at the command line's "
        f"constants, everything else at the defaults in `constants.py`. {one['container']}. Every level at seed *s* "
        f"is given the same input stream (§4.5), so levels pair epoch by epoch. About {epochs_per_second:,.0f} epochs "
        f"a second an arm.",
        "",
        f"Driver: `docs/rust-sweep.py --name {args.name} --problem {problem} --goo ... --eligibility {eligibility} "
        f"--{args.knob.replace('_', '-')} ... "
        f"--seed 1 ... {max(seeds)} --epochs N`; report: `docs/goo-threshold-report.py --name {args.name} --knob "
        f"{args.knob}`; figure: `{args.name}-score.png`. Every arm's trace and summary is under `runs/{args.name}/` "
        f"(not in git).",
        "",
    ]
    if args.knob == "threshold":
        lines += [
            f"Goo scales its potential axis by fan-in (§5.2), so the theta a goo neuron actually starts at is the "
            f"level times {scale:.2f}, and the floor is MINIMUM_POTENTIAL × {scale:.2f} = {MINIMUM_POTENTIAL * scale:.2f} "
            f"throughout: sweeping THRESHOLD alone sweeps the floor-to-threshold ratio with it.",
            "",
            "| THRESHOLD | theta on goo | floor / theta | accuracy, last tenth | over seeds | stuck on | stuck off | vs default |",
            "|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [f"| {args.knob} | accuracy, last tenth | over seeds | stuck on | stuck off | vs default |", "|---|---|---|---|---|---|"]
    best_level, best = None, -1.0
    for level in sorted(levels):
        r = last(level)
        mean = statistics.fmean(r)
        if mean > best:
            best_level, best = level, mean
        on = statistics.fmean(levels[level][s]["stuck_on"] for s in seeds)
        off = statistics.fmean(levels[level][s]["stuck_off"] for s in seeds)
        if default is not None and default in levels and level != default:
            d, t = paired_t(r, last(default))
            versus = f"{d:+.4f} (t {t:+.1f})"
        else:
            versus = "—" if default is None or level != default else "the default"
        mark = "**" if level == best_level else ""
        if args.knob == "threshold":
            theta = level * scale
            lines.append(f"| {level:g} | {theta:.3f} | {MINIMUM_POTENTIAL * scale / theta:.2f} | {mark}{mean:.4f}{mark} | "
                         f"{min(r):.3f} to {max(r):.3f} | {on:.1f} | {off:.1f} | {versus} |")
        else:
            lines.append(f"| {level:g} | {mark}{mean:.4f}{mark} | {min(r):.3f} to {max(r):.3f} | {on:.1f} | {off:.1f} | {versus} |")
    # the best level's mark is only known after the loop: re-mark
    lines = [ln.replace("**", "") if ln.startswith("|") and f"| {best_level:g} |" not in ln else ln for ln in lines]
    lines.append("")
    lines.append(f"Best level: {args.knob} {best_level:g}, {best:.4f} over the last tenth, averaged over {len(seeds)} seeds.")
    lines.append("")
    (ROOT / "docs" / f"{args.name}.md").write_text("\n".join(lines) + "\n")
    plot(args, levels, seeds, scale, default, eligibility, chance)
    print("\n".join(lines))


def plot(args, levels, seeds, scale, default, eligibility: str = "wrong_hebb", chance: float = 0.5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, GRID, BLUE = "#fcfcfb", "#1a1a19", "#5f5e58", "#e6e5df", "#2a78d6"
    xs = sorted(levels)
    series = [[levels[x][s]["last_tenth"] for s in seeds] for x in xs]
    mean = [statistics.fmean(v) for v in series]
    lo, hi = [min(v) for v in series], [max(v) for v in series]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.13, linewidth=0)
    ax.plot(xs, mean, color=BLUE, linewidth=1.8, marker="o", markersize=3.5, label=f"mean over {len(seeds)} seeds; band is the range")
    ax.axhline(chance, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate("chance", (0.995, chance), xycoords=("axes fraction", "data"), ha="right", va="bottom", color=INK2, fontsize=7)
    if default is not None and default in levels:
        ax.axvline(default, color=INK2, linewidth=0.8, linestyle=(0, (2, 3)))
        ax.annotate("default", (default, 1.0), xycoords=("data", "axes fraction"), ha="left", va="top", color=INK2, fontsize=7,
                    xytext=(3, -2), textcoords="offset points")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_xlabel(args.knob.upper() + (" (the constant; goo starts at this × %.2f)" % scale if args.knob == "threshold" else ""),
                  color=INK2, fontsize=9)
    ax.set_ylabel("accuracy over the last tenth of the run", color=INK2, fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f"Goo against {args.knob.upper()}, {eligibility} eligibility", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "docs" / f"{args.name}-score.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
