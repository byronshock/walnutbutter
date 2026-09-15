#!/usr/bin/env python3
"""Read the goo-250k arms off disk and write docs/goo-250k.md and docs/goo-250k-score.png.

    .venv/bin/python docs/goo-250k-report.py

Three arms, ten seeds each, paired on the input stream (AUTHORITY.md §4.5):
the shipped goo (threshold and floor scaled with fan-in, §5.2), the same goo
run flat, and the 8 x 10 hex grid as the baseline. Reads each run's
checkpoint under runs/goo-250k/ -- the Teacher's history is in there, ten
progress reports per run -- so nothing has to be re-run to redraw the figure.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "goo-250k"
SEEDS = range(1, 11)
ARMS = [  # file stem -> how it is named in the report, and its colour in the figure
    ("scaled", "goo, threshold and floor scaled", "#2a78d6"),
    ("flat", "goo, flat threshold and floor", "#eb6834"),
    ("grid", "hex grid 8 x 10 (baseline)", "#8a897f"),
]


def load(stem: str) -> list[dict]:
    """One record per seed: the final figures and the Teacher's history."""
    out = []
    for seed in SEEDS:
        path = RUNS / f"{stem}-seed{seed}.json"
        if not path.exists():
            raise SystemExit(f"missing {path} -- has the sweep finished?")
        data = json.loads(path.read_text())
        learning = data.get("learning") or {}
        history = learning.get("history", [])
        if not history:
            raise SystemExit(f"{path} has no learning history -- was it run without a Teacher?")
        out.append({
            "seed": seed,
            "epochs": data["epoch"],
            "history": history,
            "stuck_on": history[-1]["stuck_on"],
            "stuck_off": history[-1]["stuck_off"],
            "accuracy_to_date": history[-1]["accuracy_to_date"],
            "recent": history[-1]["recent"],
        })
    return out


def paired_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mean difference a - b, paired on the seed, and its t statistic."""
    d = [x - y for x, y in zip(a, b)]
    mean = statistics.fmean(d)
    if len(d) < 2:
        return mean, float("nan")
    sd = statistics.stdev(d)
    return mean, (mean / (sd / math.sqrt(len(d))) if sd else float("inf"))


def main() -> None:
    arms = {stem: load(stem) for stem, _, _ in ARMS}
    recent = {stem: [r["recent"] for r in rows] for stem, rows in arms.items()}
    to_date = {stem: [r["accuracy_to_date"] for r in rows] for stem, rows in arms.items()}
    epochs = arms["scaled"][0]["epochs"]

    lines = [
        f"# Sweep goo-250k (September 14, 2026)",
        "",
        f"Does goo learn once its potential axis is scaled with fan-in (AUTHORITY.md §5.2)? "
        f"Three arms, {len(list(SEEDS))} seeds each, {epochs:,} epochs per run, the reversal problem, "
        f"the array engine, everything else at the defaults in `constants.py`. Every arm at seed *s* is "
        f"given the same input stream (§4.5), so the arms are paired epoch by epoch and not merely on average.",
        "",
        "Driver: `walnutbutter --goo --seeds 10 --seed 1 --input-seed 1 --epochs 250000 --engine arrays`, "
        "once per arm; report: `docs/goo-250k-report.py`; figure: `goo-250k-score.png`. Every run's "
        "checkpoint and log is under `runs/goo-250k/` (not in git).",
        "",
        "| arm | accuracy, last tenth | range | accuracy, to date | stuck on | stuck off |",
        "|---|---|---|---|---|---|",
    ]
    for stem, label, _ in ARMS:
        r, d = recent[stem], to_date[stem]
        rows = arms[stem]
        lines.append(
            f"| {label} | **{statistics.fmean(r):.4f}** | {min(r):.3f} to {max(r):.3f} | "
            f"{statistics.fmean(d):.4f} | {statistics.fmean(x['stuck_on'] for x in rows):.1f} | "
            f"{statistics.fmean(x['stuck_off'] for x in rows):.1f} |"
        )
    lines += ["", "Paired on the seed, over the last tenth of each run:", ""]
    lines += ["| comparison | mean difference | t |", "|---|---|---|"]
    for a, b in (("scaled", "flat"), ("scaled", "grid"), ("flat", "grid")):
        mean, t = paired_t(recent[a], recent[b])
        name = {s: l for s, l, _ in ARMS}
        lines.append(f"| {name[a]} − {name[b]} | {mean:+.4f} | {t:+.2f} |")
    lines += ["", "Per seed, accuracy over the last tenth:", "",
              "| seed | " + " | ".join(l for _, l, _ in ARMS) + " |",
              "|---|" + "---|" * len(ARMS)]
    for i, seed in enumerate(SEEDS):
        lines.append(f"| {seed} | " + " | ".join(f"{recent[s][i]:.3f}" for s, _, _ in ARMS) + " |")
    lines.append("")

    (ROOT / "docs" / "goo-250k.md").write_text("\n".join(lines) + "\n")
    plot(arms)
    print("\n".join(lines))


def plot(arms: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, GRID = "#fcfcfb", "#1a1a19", "#5f5e58", "#e6e5df"
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for stem, label, colour in ARMS:
        rows = arms[stem]
        xs = [h["epoch"] for h in rows[0]["history"]]
        series = [[h["recent"] for h in r["history"]] for r in rows]
        mean = [statistics.fmean(v[i] for v in series) for i in range(len(xs))]
        lo = [min(v[i] for v in series) for i in range(len(xs))]
        hi = [max(v[i] for v in series) for i in range(len(xs))]
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.13, linewidth=0)
        ax.plot(xs, mean, color=colour, linewidth=1.8, label=label)
    ax.axhline(0.5, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate("chance", (0.995, 0.5), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                color=INK2, fontsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_xlabel("epoch", color=INK2, fontsize=9)
    ax.set_ylabel("accuracy over the recent window", color=INK2, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Goo at 250,000 epochs: does the rescaled axis learn?", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "docs" / "goo-250k-score.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
