#!/usr/bin/env python3
"""Tables and the figure for the second toy (docs/synaptic-noise-toy2.py), from
the JSON it writes:

    python docs/synaptic-noise-toy2-report.py <dir with toy2_local.json, toy2_exact_read.json, toy2_none.json> [figure.png]

For every (rule, drive, read, h0) it prints the best learning rate's arm --
accuracy over the last checkpoint, the correlation of the weight change with
the supervised direction, whispers an epoch from on- and off-inputs, output
spikes, stuck-off outputs, and the fraction of weights whose sign flipped --
and draws accuracy and correlation against the rest hazard.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else []


def arms(results):
    groups = defaultdict(list)
    for r in results:
        groups[(r["rule"], r["drive"], r["read"], r["h0"], r["lr"])].append(r)
    rows = {}
    for key, rs in groups.items():
        last = [r["hist"][-1] for r in rs]
        curve = np.mean([[h["acc"] for h in r["hist"]] for r in rs], axis=0)
        eps = [h["epoch"] for h in rs[0]["hist"]]
        rows[key] = dict(
            acc=np.mean([h["acc"] for h in last]), acc_sd=np.std([h["acc"] for h in last]),
            corr=np.mean([h["corr"] for h in last]),
            won=np.mean([h["whisper_on"] for h in last]), woff=np.mean([h["whisper_off"] for h in last]),
            spikes=np.mean([h["spikes"] for h in last]), stuck=np.mean([h["stuck_off"] for h in last]),
            flipped=np.mean([h["flipped"] for h in last]), rails=np.mean([h["rails"] for h in last]),
            t80=next((e for e, a in zip(eps, curve) if a >= 0.8), None), eps=eps, curve=curve, n=len(rs))
    return rows


def best_per_h0(rows):
    best = {}
    for (rule, drive, read, h0, lr), v in rows.items():
        k = (rule, drive, read, h0)
        if k not in best or v["acc"] > best[k][1]["acc"]:
            best[k] = (lr, v)
    return best


def main():
    d = sys.argv[1]
    fig = sys.argv[2] if len(sys.argv) > 2 else None
    results = load(d, "toy2_local.json") + load(d, "toy2_exact_read.json") + load(d, "toy2_none.json")
    rows = arms(results)
    best = best_per_h0(rows)
    print("| rule | drive | read | h0 | best LR | accuracy | corr with d | whispers on / off | spikes/epoch | stuck | flipped | to 0.8 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(best, key=lambda k: (k[0], k[1], k[2], -k[3])):
        lr, v = best[k]
        rule, drive, read, h0 = k
        print(f"| {rule} | {drive} | {read} | {h0:g} | {lr:g} | {v['acc']:.3f} ± {v['acc_sd']:.3f} | {v['corr']:+.2f} | "
              f"{v['won']:.2f} / {v['woff']:.2f} | {v['spikes']:.1f} | {v['stuck']:.1f} | {v['flipped']:.2f} | {v['t80'] or '—'} |")
    print("\nevery arm:\n")
    print("| rule | drive | read | h0 | LR | accuracy | corr | flipped |")
    print("|---|---|---|---|---|---|---|---|")
    for k in sorted(rows, key=lambda k: (k[0], k[1], k[2], -k[3], k[4])):
        v = rows[k]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]:g} | {k[4]:g} | {v['acc']:.3f} ± {v['acc_sd']:.3f} | {v['corr']:+.2f} | {v['flipped']:.2f} |")
    if not fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    series = sorted({(k[0], k[1], k[2]) for k in best})
    styles = {("local", "forced", "spikes"): ("tab:orange", "-", "s"),
              ("local", "potential", "spikes"): ("tab:red", "-", "^"),
              ("exact", "forced", "transmissions"): ("tab:blue", "-", "o"),
              ("none", "forced", "spikes"): ("gray", ":", "x"),
              ("none", "potential", "spikes"): ("gray", "--", "x")}
    for s in series:
        h0s = sorted({k[3] for k in best if k[:3] == s})
        acc = [best[s + (h,)][1]["acc"] for h in h0s]
        corr = [best[s + (h,)][1]["corr"] for h in h0s]
        color, ls, marker = styles.get(s, ("black", "-", "."))
        label = f"{s[0]} rule, {s[1]} drive" + (", read on transmissions" if s[2] == "transmissions" else "")
        ax1.plot(h0s, acc, ls, color=color, marker=marker, label=label)
        ax2.plot(h0s, corr, ls, color=color, marker=marker, label=label)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("h0, the hazard at rest (spikes per hop, before the count's factor)")
    ax1.axhline(0.2, color="gray", lw=0.5)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("accuracy at 15,000 epochs, best LR")
    ax1.set_title("Learning on Byron's mechanism")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_ylabel("correlation of the weight change with d")
    ax2.set_title("Does it learn the pattern?")
    ax1.legend(fontsize=7)
    fig_.tight_layout()
    fig_.savefig(fig, dpi=120)
    print(f"\nfigure: {fig}")


if __name__ == "__main__":
    main()
