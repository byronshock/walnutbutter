#!/usr/bin/env python3
"""The epoch-length sweep of the second toy (Byron, September 23, 2026): epoch
length in waves {20, 35, 60, 100} by rest hazard {0, 0.001, 0.003}, ten seeds,
25,000 epochs, on his mechanism with the linear hazard.

    python docs/synaptic-noise-toy2-epoch-report.py <dir> [figure.png]

Reads every toy2_epoch_*.json in <dir>; prints one table per (drive, LR) of
accuracy over the last 1,250 epochs (mean ± sd over seeds), the epochs to 0.8
on the seed-mean curve, output spikes an epoch and outputs stuck off; draws
accuracy against epoch length, one line per rest hazard, one panel per
(drive, LR).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np


def main():
    d = sys.argv[1]
    fig = sys.argv[2] if len(sys.argv) > 2 else None
    results = []
    for name in sorted(os.listdir(d)):
        if name.startswith("toy2_epoch_") and name.endswith(".json"):
            results += json.load(open(os.path.join(d, name)))
    groups = defaultdict(list)
    for r in results:
        groups[(r["drive"], r["lr"], r["h0"], r["waves"])].append(r)
    panels = sorted({(k[0], k[1]) for k in groups})
    cells = {}
    for key, rs in groups.items():
        last = [r["hist"][-1] for r in rs]
        curve = np.mean([[h["acc"] for h in r["hist"]] for r in rs], axis=0)
        eps = [h["epoch"] for h in rs[0]["hist"]]
        cells[key] = dict(acc=np.mean([h["acc"] for h in last]), sd=np.std([h["acc"] for h in last]),
                          t80=next((e for e, a in zip(eps, curve) if a >= 0.8), None),
                          t95=next((e for e, a in zip(eps, curve) if a >= 0.95), None),
                          spikes=np.mean([h["spikes"] for h in last]), stuck=np.mean([h["stuck_off"] for h in last]),
                          corr=np.mean([h["corr"] for h in last]), n=len(rs))
    for drive, lr in panels:
        h0s = sorted({k[2] for k in groups if k[:2] == (drive, lr)})
        waves = sorted({k[3] for k in groups if k[:2] == (drive, lr)})
        n = max(v["n"] for k, v in cells.items() if k[:2] == (drive, lr))
        print(f"\n### {drive} drive, LR {lr:g}, {n} seeds\n")
        print("| waves an epoch | " + " | ".join(f"h0 = {h:g}" for h in h0s) + " |")
        print("|---|" + "---|" * len(h0s))
        for w in waves:
            row = []
            for h in h0s:
                c = cells.get((drive, lr, h, w))
                row.append("—" if c is None else
                           f"{c['acc']:.3f} ± {c['sd']:.3f}, to 0.8 in {c['t80'] or '—'}, to 0.95 in {c['t95'] or '—'}, "
                           f"{c['spikes']:.0f} spikes, stuck {c['stuck']:.1f}")
            print(f"| {w} | " + " | ".join(row) + " |")
    if not fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2), squeeze=False)
    for ax, (drive, lr) in zip(axes[0], panels):
        h0s = sorted({k[2] for k in groups if k[:2] == (drive, lr)})
        waves = sorted({k[3] for k in groups if k[:2] == (drive, lr)})
        for h in h0s:
            ys = [cells[(drive, lr, h, w)]["acc"] if (drive, lr, h, w) in cells else np.nan for w in waves]
            es = [cells[(drive, lr, h, w)]["sd"] if (drive, lr, h, w) in cells else np.nan for w in waves]
            ax.errorbar(waves, ys, yerr=es, marker="o", capsize=3, label=f"rest hazard {h:g}")
        ax.set_xscale("log")
        ax.set_xticks(waves)
        ax.set_xticklabels([str(w) for w in waves])
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.axhline(0.2, color="gray", lw=0.5)
        ax.set_xlabel("waves an epoch (one wave is one hop)")
        ax.set_ylabel("accuracy over the last 1,250 of 25,000 epochs")
        ax.set_title(f"{drive} drive, LR {lr:g}")
        ax.legend(fontsize=8)
    fig_.tight_layout()
    fig_.savefig(fig, dpi=120)
    print(f"\nfigure: {fig}")


if __name__ == "__main__":
    main()
