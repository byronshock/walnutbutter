"""Summarise instrument.py's json files: the numbers the answer quotes."""
import json, math, sys, statistics as st
import numpy as np

OUT = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/excitation"
HOP = 2.55
labels = sys.argv[1:] or ["syn-rate-fresh", "syn-rate-25k", "syn-charged-fresh", "syn-charged-25k", "neu-fresh", "neu-25k",
                          "syn-rate-fresh-sup", "syn-rate-25k-sup", "syn-charged-fresh-sup", "syn-charged-25k-sup"]


def a(d, k):
    return np.array(d[k], dtype=float)


for label in labels:
    try:
        d = json.load(open(f"{OUT}/{label}.json"))
    except FileNotFoundError:
        print(f"\n### {label}: not there"); continue
    th = a(d, "theta")
    E = d["epochs"]
    fired_frac = a(d, "fired_frac")
    always = fired_frac >= 1.0
    ck = a(d, "rates_ckpt") > 0.99 if d["source"] != "fresh" else None
    print(f"\n### {label} ({E} epochs, mismatches {d['mismatch']}, spike check {d['spikes_check_ok']}): "
          f"waves/epoch {d['waves']:.0f}")
    print(f"  output spikes per epoch (sum of 60) {a(d, 'spikes').sum():.1f}, read escapes {a(d, 'reads').sum():.1f}; "
          f"outputs firing in every one of the {E} epochs: {int(always.sum())}; counted (spike or read) every epoch: "
          f"{int((a(d, 'counted_frac') >= 1).sum())}; never fired: {int((fired_frac == 0).sum())}"
          + (f"; checkpoint rate memory > 0.99: {int(ck.sum())} (fire-if-one {int(ck[:30].sum())}, fire-if-zero {int(ck[30:].sum())})"
             if ck is not None else ""))
    print(f"  inputs: spikes per epoch per bit-1 input {d['in_spikes_per_bit1']:.2f}, per bit-0 input {d['in_spikes_per_bit0']:.3f}, "
          f"per clock input {d['in_clock_spikes']:.2f}; ventured arrivals at outputs per epoch from bit-1 sources "
          f"{d['in_ven']['in_ven_bit1']:.1f} ({d['in_ven']['in_ven_bit1'] / max(1e-9, d['edge_counts']['bit1']):.4f} per synapse), "
          f"bit-0 {d['in_ven']['in_ven_bit0']:.1f} ({d['in_ven']['in_ven_bit0'] / max(1e-9, d['edge_counts']['bit0']):.4f} per synapse), "
          f"clock {d['in_ven']['in_ven_clock']:.1f} ({d['in_ven']['in_ven_clock'] / max(1e-9, d['edge_counts']['clock']):.4f} per synapse); "
          f"synapses to outputs: bit-1 {d['edge_counts']['bit1']:.0f}, bit-0 {d['edge_counts']['bit0']:.0f}, clock {d['edge_counts']['clock']}")
    groups = [("all 60", np.ones(60, bool))]
    if ck is not None:
        groups += [("ckpt stuck-on", ck), ("ckpt rest", ~ck)]
    groups += [("fired every epoch", always), ("not every epoch", ~always)]
    for name, m in groups:
        if not m.any():
            continue
        f = lambda k: a(d, k)[m]
        relE, relI, venE, venI = f("rel_E"), f("rel_I"), f("ven_E"), f("ven_I")
        t = th[m]
        net_rel, net_ven = relE + relI, venE + venI
        spikes = f("spikes")
        sp = spikes.sum()
        print(f"  [{name}, {int(m.sum())}] per output per epoch, integrated weight in units of its theta: relayed +{np.mean(relE / t):.2f} "
              f"{np.mean(relI / t):+.2f} = {np.mean(net_rel / t):+.2f}; ventured +{np.mean(venE / t):.3f} {np.mean(venI / t):+.3f} "
              f"= {np.mean(net_ven / t):+.3f}; arrivals relayed {np.mean(f('n_rel')):.1f}, ventured {np.mean(f('n_ven')):.2f}; "
              f"dropped at refractory: relayed {np.mean(f('n_drop_rel')):.1f} (net {np.mean(f('drop_rel_w') / t):+.2f}), "
              f"ventured {np.mean(f('n_drop_ven')):.2f}; floor clamps {np.mean(f('clamps')):.1f} erasing {np.mean(f('clamp_loss') / t):.2f}")
        ven_share_E = venE.sum() / max(1e-12, (relE + venE).sum())
        # spikes
        n_ven_rate = (f("n_ven") + f("n_drop_ven")) / d.get("interval", 100.0)
        chance = 1 - np.exp(-n_ven_rate * HOP)
        chance_w = float((chance * spikes).sum() / sp) if sp else float("nan")
        print(f"      ventured share of the excitatory weight {ven_share_E:.3f}; spikes {np.mean(spikes):.2f}/output/epoch, "
              f"reads {np.mean(f('reads')):.2f}; of {sp * E:.0f} spikes: ventured arrival in the same wave "
              f"{f('sp_ven_same_wave').sum() / max(1, sp):.3f}, within a hop before {f('sp_ven_within_hop').sum() / max(1, sp):.3f} "
              f"(chance at the output's ventured arrival rate {chance_w:.3f}); needed this wave's ventured charge "
              f"{f('sp_pivotal_wave').sum() / max(1, sp):.3f}; needed the ventured charge since the last reset "
              f"{f('sp_needs_ven_since').sum() / max(1, sp):.3f}; ventured charge since reset at a spike, mean "
              f"{(f('sp_ven_since_sum') / t).sum() / max(1, sp):+.3f} theta")
        print(f"      V: mean u {np.mean(a(d, 'u_mean')[m]):.3f}, time with V > 0 {np.mean(a(d, 'V_pos_frac')[m]):.3f}, "
              f"time at the floor {np.mean(a(d, 'V_floor_frac')[m]):.3f}, mean V/theta {np.mean(a(d, 'V_mean')[m] / t):+.3f}")
        course = np.array(d["V_course"])[m].mean(0)
        print(f"      V/theta by 10 ms of the epoch: {' '.join(f'{x:+.2f}' for x in course)}")
    ru = np.array(d["read_u"])
    if len(ru):
        print(f"  read escapes: {len(ru)} over {E} epochs; u at the escape: 0 {np.mean(ru == 0):.3f}, (0,0.5) "
              f"{np.mean((ru > 0) & (ru < 0.5)):.3f}, [0.5,0.9) {np.mean((ru >= 0.5) & (ru < 0.9)):.3f}, >=0.9 {np.mean(ru >= 0.9):.3f}")
    spv = np.array(d["sp_V"])
    if len(spv):
        print(f"  V at the fire / theta: below 1 {np.mean(spv < 1):.3f} (the neuron rule's escapes), median {np.median(spv):.3f}, "
              f"90th pct {np.percentile(spv, 90):.3f}")
    # across patterns: spread over epochs of each output's relayed and ventured net drive, and what the spikes track
    pe = d["per_epoch"]
    rel = np.array([p["rel"] for p in pe]) / th
    ven = np.array([p["ven"] for p in pe]) / th
    spk = np.array([p["spikes"] for p in pe], dtype=float)
    def corr(x, y):
        out = []
        for j in range(x.shape[1]):
            if np.std(x[:, j]) > 0 and np.std(y[:, j]) > 0:
                out.append(np.corrcoef(x[:, j], y[:, j])[0, 1])
        return (np.mean(out), len(out)) if out else (float("nan"), 0)
    cr, nr = corr(rel, spk)
    cv, nv = corr(ven, spk)
    print(f"  across the {E} epochs, per output: SD of the relayed net drive {np.mean(rel.std(0)):.3f} theta, of the ventured "
          f"{np.mean(ven.std(0)):.3f} theta; corr(epoch spikes, relayed net) {cr:+.3f} over {nr} outputs, "
          f"corr(epoch spikes, ventured net) {cv:+.3f} over {nv}")
