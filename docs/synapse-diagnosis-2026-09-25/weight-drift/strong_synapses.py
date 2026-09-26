"""The strongest excitatory synapses from often-on inputs: per output, the largest w/theta over its synapses from inputs on
more than half the time (P(on) > 0.5), at start and at 25k, and the share of outputs busy at 25k by that largest weight.
A single arrival of w >= theta - floor (4/3 theta at floor -0.2, threshold 0.6) fires the output from anywhere."""
import importlib.util, json, collections
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("ds", str(Path(__file__).resolve().parent / "drift_static.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
from walnutbutter.mnist import pixel_statistics
acc = collections.OrderedDict()
for rule, folder, arm, fix in ds.arms():
    g, cli = ds.rs.grid_of("mnist", arm, "hazard", True, None, None, fix)
    d = json.load(open(f"runs/{folder}/{ds.rs.arm_name(arm)}-network.json"))
    conns = [g.connections[i] for i in range(1, len(g.connections) + 1)]
    w1 = np.array(d["weights"]); w0 = np.array([c.weight for c in conns])
    on, _ = pixel_statistics(g.clock)
    inrow = {n: i for i, n in enumerate(g.input_row())}; allidx = {n: i for i, n in enumerate(g.all_neurons())}
    outs = list(dict.fromkeys(g.output_row())); oidx = {n: k for k, n in enumerate(outs)}
    p = np.array([on[inrow[c.source]] for c in conns]); t = np.array([oidx[c.target] for c in conns])
    th = np.array([n.threshold for n in outs]); fl = np.array([n.minimum_potential for n in outs])
    rate = np.array([d["rates"][allidx[n]] for n in outs])
    M0 = np.full(len(outs), -9.0); M1 = np.full(len(outs), -9.0)
    for k in range(len(outs)):
        m = (t == k) & (p > 0.5)
        if m.any():
            M0[k] = w0[m].max() / th[k]; M1[k] = w1[m].max() / th[k]
    strong = (p > 0.5) & (w0 > 0.5 * th[t])  # synapses from often-on inputs starting above half a threshold
    key = (rule, arm.get("drive", "rate"), arm["lr"])
    a = acc.setdefault(key, collections.defaultdict(list))
    a["M0"] += M0.tolist(); a["M1"] += M1.tolist(); a["rate"] += rate.tolist()
    a["strong_dw"] += (w1[strong] - w0[strong]).tolist()
    a["n_over0"].append(int(((p > 0.5) & (w0 >= th[t] - fl[t])).sum())); a["n_over1"].append(int(((p > 0.5) & (w1 >= th[t] - fl[t])).sum()))
    a["floor_over_th"] = [float(fl[0] / th[0])]
edges = [-9, 0.5, 1.0, 1.333, 9]; lab = ["<0.5", "0.5..1", "1..4/3", ">=4/3"]
print("share of outputs busy at 25k by the largest w/theta from an input on >50% of the time (n outputs, 3 seeds pooled);")
print("synapses from often-on inputs with w >= theta - floor (single-arrival firers), start -> 25k, per seed;")
print("mean change of synapses from often-on inputs that started above theta/2")
print(f"{'rule':8s}{'drive':8s}{'lr':>7s} | " + " | ".join(f"25k {l:>7s}" for l in lab) + " | " + " | ".join(f"start {l:>7s}" for l in lab) + " | firers start->25k | dw strong (n)")
for key, a in acc.items():
    rate = np.array(a["rate"]); cells = []
    for K in ("M1", "M0"):
        M = np.array(a[K])
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (M >= lo) & (M < hi)
            cells.append(f"{(rate[m] > 0.99).mean():.2f} ({m.sum():3d})" if m.any() else "    -     ")
    print(f"{key[0]:8s}{key[1]:8s}{key[2]:7g} | " + " | ".join(cells) + f" | {a['n_over0']} -> {a['n_over1']} | {np.mean(a['strong_dw']):+.3f} ({len(a['strong_dw'])})")
