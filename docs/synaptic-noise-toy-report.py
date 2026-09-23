#!/usr/bin/env python3
"""Tables and the figure for docs/synaptic-escape-noise-2026-09-22.md, from the
JSON that docs/synaptic-noise-toy.py writes.

    python docs/synaptic-noise-toy-report.py <dir with snr.json, sweep16.json, sweep64b.json, sweep256.json> [figure.png]

Prints the signal-to-noise table, the learning grid at each fan-in and the
best arm per rule, and draws two panels: the signal-to-noise against the
fan-in, and the best arms' accuracy over epochs.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np


def knob(r):
    return ("delta", r["delta"]) if r["rule"] == "neuron" else ("sigma", r["sigma"])


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else []


def snr_table(snr):
    groups = defaultdict(list)
    for r in snr:
        groups[(r["fan_in"], r["rule"], knob(r))].append(r)
    rows = {}
    print("| fan-in | rule | knob | SNR per epoch | corr with d at 4,000 | output spikes/epoch |")
    print("|---|---|---|---|---|---|")
    for key in sorted(groups):
        rs = groups[key]
        s = np.array([r["hist"][-1]["snr"] for r in rs])
        c = np.array([r["hist"][-1]["corr"] for r in rs])
        sp = np.mean([np.mean([h["spikes"] for h in r["hist"]]) for r in rs])
        fan, rule, (kn, kv) = key
        rows[key] = (s.mean(), s.std() / np.sqrt(len(s)), c.mean(), sp)
        print(f"| {fan} | {rule} | {kn} {kv:g} | {s.mean():.1e} ± {s.std() / np.sqrt(len(s)):.0e} | {c.mean():+.2f} | {sp:.1f} |")
    return rows


def sweep_table(sw, fan):
    groups = defaultdict(list)
    for r in sw:
        groups[(r["rule"], knob(r), r["lr"])].append(r)
    rows = []
    print(f"\nfan-in {fan}: accuracy over the last {sw[0]['epochs'] // 20} epochs of {sw[0]['epochs']}, {len(next(iter(groups.values())))} seeds (chance 0.2)\n")
    print("| rule | knob | LR | accuracy | epochs to 0.8 | output spikes/epoch | stuck off |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        rs = groups[key]
        acc = np.array([r["hist"][-1]["acc"] for r in rs])
        sp = np.mean([r["hist"][-1]["spikes"] for r in rs])
        stuck = np.mean([r["hist"][-1]["stuck_off"] for r in rs])
        curve = np.mean([[h["acc"] for h in r["hist"]] for r in rs], axis=0)
        eps = [h["epoch"] for h in rs[0]["hist"]]
        t80 = next((e for e, a in zip(eps, curve) if a >= 0.8), None)
        rule, (kn, kv), lr = key
        rows.append((key, acc.mean(), acc.std(), t80, sp, stuck, eps, curve))
        print(f"| {rule} | {kn} {kv:g} | {lr:g} | {acc.mean():.3f} ± {acc.std():.3f} | {t80 if t80 else '—'} | {sp:.1f} | {stuck:.1f} |")
    best = {}
    for rule in ("neuron", "synapse"):
        cand = [row for row in rows if row[0][0] == rule]
        best[rule] = max(cand, key=lambda row: row[1])
        key, a, sd, t80, sp, stuck, eps, curve = best[rule]
        print(f"\nbest {rule}: {key[1][0]} {key[1][1]:g}, LR {key[2]:g}: accuracy {a:.3f}, to 0.8 in {t80}")
    return best


def main():
    d = sys.argv[1]
    fig = sys.argv[2] if len(sys.argv) > 2 else None
    print("## Signal to noise, no learning\n")
    snr = load(d, "snr.json") + load(d, "snr_small_sigma.json") + load(d, "snr_mid.json")
    snr_rows = snr_table(snr)
    sweeps = {16: load(d, "sweep16.json"), 64: load(d, "sweep64b.json"), 256: load(d, "sweep256.json")}
    bests = {}
    print("\n## Learning\n")
    for fan, sw in sweeps.items():
        if sw:
            bests[fan] = sweep_table(sw, fan)
    if not fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fans = sorted({k[0] for k in snr_rows})
    for rule, marker in (("neuron", "o"), ("synapse", "s")):
        # the best knob per fan-in, and the knob the learning sweep liked
        y = [max(v[0] for k, v in snr_rows.items() if k[0] == f and k[1] == rule) for f in fans]
        ax1.plot(fans, y, marker=marker, label=f"{rule} rule, best knob")
    ratio = [max(v[0] for k, v in snr_rows.items() if k[0] == f and k[1] == "neuron")
             / max(v[0] for k, v in snr_rows.items() if k[0] == f and k[1] == "synapse") for f in fans]
    ax1b = ax1.twinx()
    ax1b.plot(fans, ratio, "k--", marker="x", label="ratio neuron / synapse")
    ax1b.plot(fans, [f / fans[0] * ratio[0] for f in fans], "k:", label="a slope of one")
    ax1b.set_yscale("log")
    ax1b.set_ylabel("ratio")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("fan-in (synapses an output hears)")
    ax1.set_ylabel("signal-to-noise of the update, per epoch")
    ax1.set_title("The estimator's signal-to-noise")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    colors = {16: "tab:green", 64: "tab:blue", 256: "tab:red"}
    for fan, best in bests.items():
        for rule, ls in (("neuron", "-"), ("synapse", "--")):
            key, a, sd, t80, sp, stuck, eps, curve = best[rule]
            ax2.plot(eps, curve, ls, color=colors[fan], label=f"fan-in {fan} {rule} ({key[1][0]} {key[1][1]:g}, LR {key[2]:g})")
    ax2.axhline(0.2, color="gray", lw=0.5)
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("accuracy, per 750 epochs")
    ax2.set_ylim(0, 1)
    ax2.set_title("Learning at each rule's best arm")
    ax2.legend(fontsize=7)
    fig_.tight_layout()
    fig_.savefig(fig, dpi=120)
    print(f"\nfigure: {fig}")


if __name__ == "__main__":
    main()
