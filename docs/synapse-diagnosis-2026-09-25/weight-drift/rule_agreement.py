"""Do the two rules move the same synapses the same way? Same seed = same wiring, same starting weights, same image order.
Correlation of dw (start -> 25k) between arms, per seed; and between seeds of one arm (different wiring: not comparable,
so only same-seed pairs). Also dw restricted to synapses below theta - floor at the start and at 25k (outside the blind spot)."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
R = {(r["rule"], r["drive"], r["lr"], r["seed"]): r for r in json.load(open(HERE / "drift_static.json"))}
import importlib.util
spec = importlib.util.spec_from_file_location("ds", str(HERE / "drift_static.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
W = {}
for rule, folder, arm, fix in ds.arms():
    d = json.load(open(f"runs/{folder}/{ds.rs.arm_name(arm)}-network.json")); W[(rule, arm.get("drive", "rate"), arm["lr"], arm["seed"])] = np.array(d["weights"])
W0 = {}
for seed in (1, 2, 3):
    g, _ = ds.rs.grid_of("mnist", {"interval": 100.0, "lr": 0.002, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0,
                                   "temperature": 2.0, "synapse_hazard": 0.01, "drive": "rate", "seed": seed}, "hazard", True, None, None, ds.SYN_FIX)
    W0[seed] = np.array([g.connections[i].weight for i in range(1, len(g.connections) + 1)])
pairs = [(("synapse", "rate", 0.002), ("neuron", "rate", 0.002)), (("synapse", "rate", 0.002), ("neuron", "rate", 0.0005)),
         (("synapse", "rate", 0.01), ("neuron", "rate", 0.002)), (("synapse", "charged", 0.002), ("neuron", "rate", 0.002)),
         (("synapse", "rate", 0.002), ("synapse", "charged", 0.002)), (("synapse", "rate", 0.002), ("synapse", "rate", 0.01)),
         (("neuron", "rate", 0.0005), ("neuron", "rate", 0.002)), (("synapse", "rate", 0.0005), ("synapse", "rate", 0.002))]
print("correlation of the change start -> 25k between two arms of the same seed (seeds 1, 2, 3)")
for a, b in pairs:
    cs = [float(np.corrcoef(W[a + (s,)] - W0[s], W[b + (s,)] - W0[s])[0, 1]) for s in (1, 2, 3)]
    print(f"  {a[0]} {a[1]} lr {a[2]:g}  vs  {b[0]} {b[1]} lr {b[2]:g}: " + ", ".join(f"{c:+.2f}" for c in cs))
