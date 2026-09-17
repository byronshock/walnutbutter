#!/usr/bin/env python3
"""Draw the estimator's correlation over time from a rust-sweep of mnist: docs/<name>-estimator.png and .md.

    .venv/bin/python docs/mnist-estimator-report.py --name mnist-ff-eligibility

Reads runs/<name>/*.json, the records the driver writes per arm, each carrying
`estimator`: at every trace interval, over the input-to-output synapses, the
Pearson correlation and the sign agreement of the weight change since the
start with the supervised direction d, and the correlation of the change since
the previous interval (AUTHORITY.md §8). Arms are grouped by everything in
their name but the seed; each group is drawn as the mean over seeds with a band
of one standard deviation. Two panels: the cumulative correlation and the
window correlation. The table gives, per group, the final cumulative
correlation and sign agreement, the mean window correlation over the last
quarter of the run, the reward over the last tenth and the fraction right.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INK = "#333333"
COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    runs = ROOT / "runs" / args.name
    groups: dict[str, list[dict]] = {}
    for path in sorted(runs.glob("*.json")):
        record = json.loads(path.read_text())
        if not record.get("estimator"):
            continue
        key = re.sub(r"-seed\d+$", "", record["arm"])
        groups.setdefault(key, []).append(record)
    if not groups:
        raise SystemExit(f"no records with an estimator trace under {runs}")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    rows = []
    for k, (key, records) in enumerate(sorted(groups.items())):
        epochs = [e["epoch"] for e in records[0]["estimator"]]
        cum = np.array([[e["corr_cum"] if e["corr_cum"] is not None else np.nan for e in r["estimator"]] for r in records])
        win = np.array([[e["corr_window"] if e["corr_window"] is not None else np.nan for e in r["estimator"]] for r in records])
        colour = COLOURS[k % len(COLOURS)]
        for ax, data in ((top, cum), (bottom, win)):
            mean, sd = np.nanmean(data, axis=0), np.nanstd(data, axis=0)
            ax.plot(epochs, mean, color=colour, linewidth=1.8, label=f"{key} ({len(records)} seeds)")
            ax.fill_between(epochs, mean - sd, mean + sd, color=colour, alpha=0.15, linewidth=0)
        quarter = max(1, len(epochs) // 4)
        rows.append({
            "arm": key, "seeds": len(records),
            "corr_cum_final": float(np.nanmean(cum[:, -1])), "corr_cum_final_sd": float(np.nanstd(cum[:, -1])),
            "sign_cum_final": statistics.fmean(r["estimator"][-1]["sign_cum"] for r in records),
            "corr_window_last_quarter": float(np.nanmean(win[:, -quarter:])),
            "reward_last_tenth": statistics.fmean(r["last_tenth"] for r in records),
            "right_last_tenth": statistics.fmean(r["accuracy_last_tenth"] for r in records if r.get("accuracy_last_tenth") is not None) if any(r.get("accuracy_last_tenth") is not None for r in records) else float("nan"),
        })
    for ax, what in ((top, "since the start"), (bottom, "since the previous trace point")):
        ax.axhline(0.0, color="#999999", linewidth=0.8)
        ax.set_ylabel(f"correlation with d, weight change {what}", color=INK, fontsize=9)
        ax.grid(axis="y", color="#e6e6e6")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    top.legend(frameon=False, fontsize=8, loc="upper left")
    bottom.set_xlabel("epoch", color=INK)
    fig.suptitle(f"{args.name}: the estimator's correlation with the supervised direction over the run, mean over seeds ± 1 sd",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = ROOT / "docs" / f"{args.name}-estimator.png"
    fig.savefig(png, dpi=110)
    lines = [f"# {args.name}: the estimator's correlation over time ({date.today().strftime('%B %-d, %Y')})", "",
             "Over the input-to-output synapses, the correlation of the weight change with the supervised direction "
             "d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is "
             f"`{png.name}`.", "",
             "| arm | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['seeds']} | {r['corr_cum_final']:+.3f} ± {r['corr_cum_final_sd']:.3f} | {r['sign_cum_final']:.3f} | "
                     f"{r['corr_window_last_quarter']:+.3f} | {r['reward_last_tenth']:.2f} | {r['right_last_tenth']:.3f} |")
    md = ROOT / "docs" / f"{args.name}-estimator.md"
    md.write_text("\n".join(lines) + "\n")
    print(png)
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
