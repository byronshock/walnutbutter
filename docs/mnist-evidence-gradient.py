#!/usr/bin/env python3
"""A gradient check of the estimator on the feedforward mnist network: does the REINFORCE update point where a linear classifier's would?

With no hidden neurons and the outputs apart, the 1,152 input-to-output synapses are
a linear classifier's weights. Learning off, every epoch's hazard scores (§6.7) are
accumulated, weighted by the advantage under three critics computed from the same
counts -- the evidence log score at T, the plain probability q_label at T, and the
class critic -- to give each critic's estimated gradient G_ij per synapse. Beside it,
the supervised direction for a pixel-to-class weight: d_ij = P(pixel i on | class of
output j) - P(pixel i on), the sign a linear classifier's gradient has on average.
Reported per critic: the sign agreement of G with d over the synapses where d is not
tiny, the correlation of G with d, and how much of G is the per-output mean (a
loudness push the label cannot see) against the residual (the discriminative part).

    .venv/bin/python docs/mnist-evidence-gradient.py [epochs] [temperature]
"""
import json, math, statistics as st, sys
import numpy as np
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
T = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
spec = importlib.util.spec_from_file_location("rs", ROOT / "docs" / "rust-sweep.py"); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from walnutbutter import fast, mnist
from walnutbutter.learning import evidence_reward
from walnutbutter.problems import dataset_stream

grid, args = rs.grid_of("mnist", {"hidden_neurons": 0.0, "seed": 1, "threshold": 0.6, "temperature": T}, "hazard", True, -4.0)
patterns, labels = dataset_stream("mnist", 1)
edges = [c for n in grid.all_neurons() for c in n.outgoing]  # the engine's edge order (see fast.build)
neurons = grid.all_neurons(); index = {n: i for i, n in enumerate(neurons)}
src = np.array([index[c.source] for c in edges]); tgt = np.array([index[c.target] for c in edges])
G = {"evidence": np.zeros(len(edges)), "probability": np.zeros(len(edges)), "class": np.zeros(len(edges))}
baseline = {k: None for k in G}
rate = 0.05


def probe(epoch, engine, grid, out, book):
    counts = fast._counts(engine)  # spikes, plus under exploration at the synapse the read synapses' escapes (§5.10)
    groups = [sum(counts[i] for i in out[c * 5:(c + 1) * 5]) for c in range(10)]
    y = grid.input_label
    top = max(groups); z = [math.exp((g - top) / T) for g in groups]; q = [x / sum(z) for x in z]
    rewards = {"evidence": evidence_reward(groups, y, T), "probability": q[y],
               "class": 1.0 if all(groups[y] > g for c, g in enumerate(groups) if c != y) else 0.0}
    scores = np.array(engine.scores())
    for k, r in rewards.items():
        if baseline[k] is None:
            baseline[k] = r
        G[k] += (r - baseline[k]) * scores
        baseline[k] += rate * (r - baseline[k])


fast.train(grid, EPOCHS, lr=0.0, target="label", trace_every=0, patterns=patterns, labels=labels, eligibility="hazard",
           seed=1, homeostasis=0.0, unstick=0.0, critic="evidence", probe=probe, probe_every=1)

# the supervised direction: P(input i on | class of output j) - P(input i on), over the whole training split
bits, y = mnist.bits_of("train")
coded = np.concatenate([np.ones((len(bits), 3), dtype=bool), bits, ~bits], axis=1)  # 3 clocks, 196 on, 196 complements
p_on = coded.mean(0)
p_on_given = np.stack([coded[y == k].mean(0) for k in range(10)])  # (10, 395)
out_index = {index[n]: k for k, n in enumerate(grid.output_row())}
io = np.array([s < grid.across and t in out_index for s, t in zip(src, tgt)])  # the input-to-output synapses
d = np.array([p_on_given[out_index[t] // 5, s] - p_on[s] if io[e] else 0.0 for e, (s, t) in enumerate(zip(src, tgt))])
big = io & (np.abs(d) > 0.02)
report = {"epochs": EPOCHS, "temperature": T, "synapses": int(io.sum()), "with_direction": int(big.sum())}
print(f"{int(io.sum())} input-to-output synapses, {int(big.sum())} with a supervised direction of |d| > 0.02; {EPOCHS} epochs, T = {T}")
for k, g in G.items():
    gi = g[io]
    per_target = {t: gi[tgt[io] == t].mean() for t in set(tgt[io])}
    loud = np.array([per_target[t] for t in tgt[io]])
    resid = gi - loud
    agree = float((np.sign(g[big]) == np.sign(d[big])).mean())
    corr = float(np.corrcoef(g[big], d[big])[0, 1])
    report[k] = {"sign_agreement": agree, "correlation": corr, "loudness_share": float(np.abs(loud).mean() / (np.abs(gi).mean() + 1e-12)),
                 "mean_per_target": float(np.mean(list(per_target.values()))), "scale": float(np.abs(gi).mean())}
    print(f"  {k:>11}: sign agreement with d {agree:.3f} (chance 0.5), correlation {corr:+.3f}, "
          f"|loudness part| / |G| {report[k]['loudness_share']:.2f}, mean per-output push {report[k]['mean_per_target']:+.3g}, |G| {report[k]['scale']:.3g}")
(ROOT / "runs" / f"mnist-evidence-gradient-T{T:g}.json").write_text(json.dumps(report, indent=1))
