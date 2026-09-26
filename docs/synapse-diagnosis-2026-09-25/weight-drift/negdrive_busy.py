"""Outputs whose expected drive a presentation at 25k is below -2 thresholds: what distinguishes the busy ones from the quiet?
Per output: fan-in, largest single weight / theta, sum of positive weight / theta from rarely-on inputs (p<0.1), from
often-on inputs (p>0.9), sum of negative weight / theta from often-on inputs, and the drive at start."""
import importlib.util, json, collections
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("ds", str(Path(__file__).resolve().parent / "drift_static.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
from walnutbutter.mnist import pixel_statistics
rows = collections.defaultdict(list)
for rule, folder, arm, fix in ds.arms():
    if arm["lr"] not in (0.002, 0.01, 0.0005):
        continue
    g, cli = ds.rs.grid_of("mnist", arm, "hazard", True, None, None, fix)
    d = json.load(open(f"runs/{folder}/{ds.rs.arm_name(arm)}-network.json"))
    conns = [g.connections[i] for i in range(1, len(g.connections) + 1)]
    w1 = np.array(d["weights"]); w0 = np.array([c.weight for c in conns])
    on, _ = pixel_statistics(g.clock)
    inrow = {n: i for i, n in enumerate(g.input_row())}; allidx = {n: i for i, n in enumerate(g.all_neurons())}
    outs = list(dict.fromkeys(g.output_row())); oidx = {n: k for k, n in enumerate(outs)}
    p = np.array([on[inrow[c.source]] for c in conns]); t = np.array([oidx[c.target] for c in conns])
    th = np.array([n.threshold for n in outs]); rate = np.array([d["rates"][allidx[n]] for n in outs])
    for k in range(len(outs)):
        m = t == k
        D1 = (p[m] * w1[m]).sum() / th[k]
        if D1 >= -2:
            continue
        rows[(rule, arm.get("drive", "rate"), arm["lr"], rate[k] > 0.99)].append((
            m.sum(), w1[m].max() / th[k], np.clip(w1[m & (p < 0.1)], 0, None).sum() / th[k],
            np.clip(w1[m & (p > 0.9)], 0, None).sum() / th[k], np.clip(w1[m & (p > 0.9)], None, 0).sum() / th[k],
            np.clip(w1[m & (p >= 0.1) & (p <= 0.9)], 0, None).sum() / th[k], D1, (p[m] * w0[m]).sum() / th[k],
            np.clip(w0[m & (p < 0.1)], 0, None).sum() / th[k]))
print("outputs with expected drive at 25k below -2 thresholds; mean per output (n outputs, 3 seeds pooled)")
print(f"{'rule':8s}{'drive':8s}{'lr':>7s} {'busy':>5s} | {'n':>3s} | fan-in | max w/th | pos w/th p<0.1 (start) | pos w/th 0.1<=p<=0.9 | pos w/th p>0.9 | neg w/th p>0.9 | D25k/th | D0/th")
for key in sorted(rows, key=lambda k: (k[0], k[1], k[2], k[3])):
    a = np.array(rows[key]); mu = a.mean(0)
    print(f"{key[0]:8s}{key[1]:8s}{key[2]:7g} {str(key[3]):>5s} | {len(a):3d} | {mu[0]:6.1f} | {mu[1]:8.2f} | {mu[2]:6.2f} ({mu[8]:5.2f})        | {mu[5]:6.2f}              | {mu[3]:6.2f}        | {mu[4]:7.2f}        | {mu[6]:7.2f} | {mu[7]:6.2f}")
