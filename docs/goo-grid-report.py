#!/usr/bin/env python3
"""Read a two-knob rust-sweep of goo off disk and write docs/<name>.md and docs/<name>-score.png.

    .venv/bin/python docs/goo-grid-report.py --name goo-count-threshold --x threshold --y goo

Reads runs/<name>/<y><V>-<x><W>-seed<N>.json (or the two knobs in the other
order), the summary docs/rust-sweep.py writes per arm. One table of
last-tenth accuracy averaged over seeds, rows by --y and columns by --x; one
of the seed range; one of neurons stuck on / off; and a heatmap figure. Cells
are marked where every seed finished, blank where the sweep has not got to.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--x", default="threshold")
    parser.add_argument("--y", default="goo")
    args = parser.parse_args()
    runs = ROOT / "runs" / args.name
    num = r"[-0-9.e+]+"
    other = r"(?:-[a-z_]+[-0-9.e+]+)*"  # knobs the sweep held fixed (goo60, say), anywhere between the two swept ones
    pattern = re.compile(rf"^(?:{args.y}(?P<y>{num}){other}-{args.x}(?P<x>{num})|{args.x}(?P<x2>{num}){other}-{args.y}(?P<y2>{num}))"
                         rf"{other}-seed(?P<seed>\d+)\.json$")
    cells: dict[tuple[float, float], dict[int, dict]] = {}
    for path in sorted(runs.glob("*.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        x = float(m["x"] if m["x"] is not None else m["x2"])
        y = float(m["y"] if m["y"] is not None else m["y2"])
        cells.setdefault((y, x), {})[int(m["seed"])] = json.loads(path.read_text())
    if not cells:
        raise SystemExit(f"nothing under {runs} matches {args.y}<V>-{args.x}<W>-seed<N>.json -- has the sweep finished?")
    ys = sorted({y for y, _ in cells})
    xs = sorted({x for _, x in cells})
    seeds = sorted({s for v in cells.values() for s in v})
    one = next(iter(next(iter(cells.values())).values()))
    label = {"goo": "goo neurons", "threshold": "THRESHOLD", "projection": "projection P", "lr": "LR"}

    def stat(y, x, key):
        rows = cells.get((y, x), {})
        vals = [r[key] for r in rows.values()]
        return vals

    def table(title, key, fmt, extra=None):
        out = [f"**{title}**", "", "| " + label.get(args.y, args.y) + " \\ " + label.get(args.x, args.x) + " | "
               + " | ".join(f"{x:g}" for x in xs) + " |", "|" + "---|" * (len(xs) + 1)]
        for y in ys:
            row = []
            for x in xs:
                vals = stat(y, x, key)
                if not vals:
                    row.append("—")
                    continue
                cell = fmt(vals)
                if len(vals) < len(seeds):
                    cell += f" ({len(vals)}/{len(seeds)})"
                row.append(cell)
            out.append(f"| {y:g} | " + " | ".join(row) + " |")
        return out + [""]

    best = max(((y, x) for (y, x), v in cells.items() if len(v) == len(seeds)),
               key=lambda k: statistics.fmean(r["last_tenth"] for r in cells[k].values()), default=None)
    lines = [
        f"# Sweep {args.name} ({date.today().strftime('%B %-d, %Y')})",
        "",
        f"{label.get(args.y, args.y)} × {label.get(args.x, args.x)}, {len(seeds)} seeds, the copy problem (AUTHORITY.md §8), "
        f"the reinforce rule with the {one['eligibility']} eligibility, the Rust wave loop (§6.15), homeostasis and "
        f"un-sticking at the command line's constants. Every arm at seed *s* is given the same input stream (§4.5)."
        + (f" The floor follows the threshold at {one['floor_ratio']:g} × THRESHOLD, so floor / theta is {one['floor_ratio']:g} "
           f"in every cell; goo then scales both by its fan-in over {18} (§5.2)." if one.get("floor_ratio") is not None else ""),
        "",
        f"Driver: `docs/rust-sweep.py --name {args.name} --problem copy --goo ... --threshold ... --floor-ratio "
        f"{one.get('floor_ratio', '')} --eligibility {one['eligibility']} --seed 1 ... {max(seeds)} --epochs N`; report: "
        f"`docs/goo-grid-report.py --name {args.name}`; figure: `{args.name}-score.png`. Every arm's trace and summary is "
        f"under `runs/{args.name}/` (not in git).",
        "",
    ]
    lines += table("Accuracy over the last tenth, mean over seeds", "last_tenth", lambda v: f"{statistics.fmean(v):.3f}")
    lines += table("Range over seeds", "last_tenth", lambda v: f"{min(v):.3f}–{max(v):.3f}")
    lines += table("Neurons stuck on / off at the end, mean over seeds", "stuck_on",
                   lambda v: f"{statistics.fmean(v):.0f}")  # on; off follows
    lines[-2] = lines[-2]  # keep
    lines += table("Neurons stuck off at the end, mean over seeds", "stuck_off", lambda v: f"{statistics.fmean(v):.0f}")
    lines += table("What a neuron starts at: theta (floor is the ratio times this)", "theta", lambda v: f"{v[0]:.2f}")
    if best is not None:
        y, x = best
        lines.append(f"Best cell: {label.get(args.y, args.y)} {y:g}, {label.get(args.x, args.x)} {x:g}: "
                     f"{statistics.fmean(r['last_tenth'] for r in cells[best].values()):.4f} over {len(seeds)} seeds.")
        lines.append("")
    # stability, row by row: a plateau is a band, and a stable cell is one where every seed learned
    full = {k: v for k, v in cells.items() if len(v) == len(seeds)}
    lines += [f"**Stability by {label.get(args.y, args.y)}** (full cells only): the mean over every {label.get(args.x, args.x)}, "
              "the five-level band with the highest rolling mean, how many cells had every seed above 0.55, and the cell with "
              "the highest worst seed", "",
              f"| {label.get(args.y, args.y)} | mean | best band | every seed > 0.55 | worst-seed leader |", "|---|---|---|---|---|"]
    for y in ys:
        row = [x for x in xs if (y, x) in full]
        if len(row) < 5:
            continue
        means = {x: statistics.fmean(r["last_tenth"] for r in full[(y, x)].values()) for x in row}
        bands = [(statistics.fmean(means[u] for u in row[i:i + 5]), row[i], row[i + 4]) for i in range(len(row) - 4)]
        bm, lo, hi = max(bands)
        every = sum(1 for x in row if all(r["last_tenth"] > 0.55 for r in full[(y, x)].values()))
        lead = max(row, key=lambda x: (min(r["last_tenth"] for r in full[(y, x)].values()), means[x]))
        worst = min(r["last_tenth"] for r in full[(y, lead)].values())
        lines.append(f"| {y:g} | {statistics.fmean(means.values()):.4f} | {lo:g}–{hi:g}: {bm:.4f} | {every} of {len(row)} | "
                     f"{label.get(args.x, args.x)} {lead:g}: {means[lead]:.4f}, worst {worst:.3f} |")
    lines.append("")
    every_cells = sorted((y, x) for (y, x), v in full.items() if all(r["last_tenth"] > 0.55 for r in v.values()))
    if every_cells:
        lines.append("Cells where every seed learned (above 0.55): " + ", ".join(
            f"{label.get(args.y, args.y)} {y:g} / {label.get(args.x, args.x)} {x:g}" for y, x in every_cells))
        lines.append("")
    (ROOT / "docs" / f"{args.name}.md").write_text("\n".join(lines) + "\n")
    plot(args, cells, ys, xs, seeds, label)
    print("\n".join(lines))


def plot(args, cells, ys, xs, seeds, label) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    SURFACE, INK, INK2 = "#fcfcfb", "#1a1a19", "#5f5e58"
    grid = np.full((len(ys), len(xs)), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            rows = cells.get((y, x), {})
            if rows:
                grid[i, j] = statistics.fmean(r["last_tenth"] for r in rows.values())
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    lo = min(0.5, np.nanmin(grid))
    im = ax.imshow(grid, cmap="Blues", vmin=lo, vmax=max(np.nanmax(grid), lo + 1e-6), aspect="auto", origin="lower")
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_xlabel(label.get(args.x, args.x), color=INK2, fontsize=9)
    ax.set_ylabel(label.get(args.y, args.y), color=INK2, fontsize=9)
    if len(ys) * len(xs) <= 64:  # past that the labels overlap and say nothing
        for i in range(len(ys)):
            for j in range(len(xs)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                            color=INK if v < lo + 0.6 * (np.nanmax(grid) - lo) else SURFACE)
    if len(xs) > 12:  # thin the x ticks so they read
        keep = list(range(0, len(xs), max(1, len(xs) // 10)))
        ax.set_xticks(keep, [f"{xs[k]:g}" for k in keep])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors=INK2, labelsize=7, length=0)
    cb.outline.set_visible(False)
    cb.set_label("accuracy, last tenth (0.5 = chance)", color=INK2, fontsize=8)
    fig.suptitle(f"Goo on copy: {label.get(args.y, args.y)} × {label.get(args.x, args.x)}",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "docs" / f"{args.name}-score.png", facecolor=SURFACE)
    plt.close(fig)
    # one line per row over the x knob: the shape a heatmap hides when the rows are ordered
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    from matplotlib import cm
    for i, y in enumerate(ys):
        ax.plot(xs, grid[i], color=cm.Blues(0.35 + 0.6 * i / max(1, len(ys) - 1)), linewidth=1.4, marker="o",
                markersize=2.5, label=f"{label.get(args.y, args.y)} {y:g}")
    ax.axhline(0.5, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate("chance", (0.995, 0.5), xycoords=("axes fraction", "data"), ha="right", va="bottom", color=INK2, fontsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="#e6e5df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_xlabel(label.get(args.x, args.x), color=INK2, fontsize=9)
    ax.set_ylabel(f"accuracy, last tenth, mean of {len(seeds)} seeds", color=INK2, fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=7, ncol=2, labelcolor=INK2)
    fig.suptitle(f"{label.get(args.y, args.y)} by {label.get(args.x, args.x)}, one line a row", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "docs" / f"{args.name}-rows.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
