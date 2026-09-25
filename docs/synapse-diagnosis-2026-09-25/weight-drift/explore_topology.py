"""Topology of the pinned point: which zones the 1425 synapses join, the starting weight distribution, the weight range."""
import importlib.util, collections, json, statistics as st
import numpy as np
spec = importlib.util.spec_from_file_location("rs", "docs/rust-sweep.py"); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from walnutbutter.persistence import wiring_digest
FIX = ("--exploration", "synapse", "--hazard-family", "loglinear", "--synapse-scaling", "count", "--trace-counts", "all")
arm = {"interval": 100.0, "lr": 0.002, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0, "temperature": 2.0,
       "synapse_hazard": 0.01, "drive": "rate", "seed": 1}
g, cli = rs.grid_of("mnist", arm, "hazard", True, None, None, FIX)
ins = set(g.input_row()); outs = list(dict.fromkeys(g.output_row()))
print("neurons", len(g.all_neurons()), "inputs", len(ins), "outputs", len(outs), "output_row len", len(g.output_row()))
print("weight_range", g.weight_range, "threshold of output", outs[0].threshold, "floor", outs[0].minimum_potential)
kinds = collections.Counter()
for cid in range(1, len(g.connections) + 1):
    c = g.connections[cid]
    z = lambda n: "in" if n in ins else ("out" if n in outs else "hid")
    kinds[(z(c.source), z(c.target), c.kind)] += 1
print(kinds)
w = np.array([g.connections[i].weight for i in range(1, len(g.connections) + 1)])
print("start weights: mean %.4f sd %.4f min %.4f max %.4f frac>0 %.3f" % (w.mean(), w.std(), w.min(), w.max(), (w > 0).mean()))
fan = collections.Counter(g.connections[i].target for i in range(1, len(g.connections) + 1))
print("fan-in of outputs:", sorted(fan[o] for o in outs))
thr = sorted(set(round(o.threshold, 6) for o in outs)); print("output thresholds", thr[:10])
d = json.load(open("runs/synapse-lr-25k/" + rs.arm_name(arm) + "-network.json"))
print("digest match", d["wiring_digest"] == wiring_digest(g))
