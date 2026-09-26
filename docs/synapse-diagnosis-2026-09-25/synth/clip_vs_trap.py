"""Reconcile two reports: critic-sign (pushes biggest near w=+1, ceiling clips ~1000 times under the synapse rule in
2000 fresh epochs) and weight-drift (synapses with w >= theta - floor from often-on inputs are frozen under the synapse
rule). Reads the critic's per-edge npz (fresh run, lr 0.002, 2000 epochs) and classes every input->output synapse by its
time-mean weight over its output's threshold and by how often its input is on. No epochs are run; the grid is rebuilt
only to get targets, thresholds, floors and input statistics."""
import importlib.util, sys
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("rs", "docs/rust-sweep.py"); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from walnutbutter.mnist import pixel_statistics

CRIT = Path("/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/critic-sign")
SYN_FIX = ("--exploration", "synapse", "--hazard-family", "loglinear", "--synapse-scaling", "count", "--trace-counts", "all")
NEU_FIX = ("--exploration", "neuron", "--drive", "rate")

for rule in ("synapse", "neuron"):
    for seed in (1, 2):
        if rule == "synapse":
            arm = {"interval": 100.0, "lr": 0.002, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0,
                   "temperature": 2.0, "synapse_hazard": 0.01, "drive": "rate", "seed": seed}; fix = SYN_FIX
        else:
            arm = {"interval": 100.0, "lr": 0.002, "delta": 0.3125, "threshold": 0.6, "minimum_potential": -0.2,
                   "hidden_neurons": 0.0, "temperature": 2.0, "seed": seed}; fix = NEU_FIX
        g, cli = rs.grid_of("mnist", arm, "hazard", True, None, None, fix)
        conns = [g.connections[i] for i in range(1, len(g.connections) + 1)]
        wg = np.array([c.weight for c in conns])
        z = np.load(CRIT / f"edges-{rule}-seed{seed}-2000.npz")
        T = int(z["epochs"])
        assert np.abs(z["w0"] - wg).max() < 0.05, np.abs(z["w0"] - wg).max()  # same order (w0 is after epoch 1's update)
        on, _ = pixel_statistics(g.clock)
        inrow = {n: i for i, n in enumerate(g.input_row())}
        outs = list(dict.fromkeys(g.output_row())); oidx = {n: k for k, n in enumerate(outs)}
        p = np.array([on[inrow[c.source]] for c in conns]); t = np.array([oidx[c.target] for c in conns])
        th = np.array([n.threshold for n in outs])[t]; fl = np.array([n.minimum_potential for n in outs])[t]
        wm = z["w_sum"] / T
        r = wm / th; line = (th - fl) / th
        print(f"{rule} seed {seed}: {T} epochs; floor/theta {fl[0]/th[0]:+.3f}, fire-from-anywhere line (theta-floor)/theta {line[0]:.3f}; "
              f"total top clips {int(z['n_top'].sum())}, bottom {int(z['n_bot'].sum())}")
        classes = [("w/th < 0.5", r < 0.5), ("0.5..1", (r >= 0.5) & (r < 1)), ("1..line", (r >= 1) & (r < line)), (">= line", r >= line)]
        for often in (True, False):
            sel = p > 0.5 if often else p <= 0.5
            print(f"   inputs on {'> 0.5' if often else '<= 0.5'} of the time")
            for name, m in classes:
                m = m & sel
                if not m.any():
                    print(f"      {name:10s} n=   0"); continue
                ea = z["e_abs"][m] / T; top = z["n_top"][m]
                print(f"      {name:10s} n={m.sum():4d}  mean|e|/epoch {ea.mean():.4f}  median {np.median(ea):.4f}  "
                      f"share with |e| < 1e-3/epoch {(ea < 1e-3).mean():.2f}  top-clip events {int(top.sum()):5d} "
                      f"(on {int((top > 0).sum())} synapses)  |w1-w0| mean {np.abs(z['w1'][m] - z['w0'][m]).mean():.4f}")
        # where do the top clips land: weight near +1 AND under the line?
        near1 = wm > 0.9
        print(f"   synapses with mean w > 0.9: {near1.sum()}, of which >= line: {(near1 & (r >= line)).sum()}; "
              f"their top clips: below line {int(z['n_top'][near1 & (r < line)].sum())}, at/above line {int(z['n_top'][near1 & (r >= line)].sum())}")
