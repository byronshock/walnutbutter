"""Change start -> 25k by the synapse's starting weight in its target's threshold units (w0/theta), for synapses from
inputs on more than half the time (P(on) > 0.5): mean dw, mean |dw|, and the share with |dw| < 0.01 (nearly unmoved).
Also every synapse ending at or above theta - floor (fires its output alone from the floor): where it started."""
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
    inrow = {n: i for i, n in enumerate(g.input_row())}
    p = np.array([on[inrow[c.source]] for c in conns])
    th = np.array([c.target.threshold for c in conns]); fl = np.array([c.target.minimum_potential for c in conns])
    a = acc.setdefault((rule, arm.get("drive", "rate"), arm["lr"]), collections.defaultdict(list))
    a["x0"] += (w0 / th).tolist(); a["dw"] += (w1 - w0).tolist(); a["p"] += p.tolist(); a["x1"] += (w1 / th).tolist()
    a["reach"] += ((th - fl) <= 1.0).tolist()
edges = [-3, -1, -0.5, 0, 0.5, 1.0, 4 / 3, 3]; lab = ["<-1", "-1..-.5", "-.5..0", "0..0.5", "0.5..1", "1..4/3", ">=4/3"]
print("synapses from inputs with P(on) > 0.5, by starting weight / threshold: mean dw | mean |dw| | share |dw|<0.01  (n, 3 seeds pooled)")
print(f"{'rule':8s}{'drive':8s}{'lr':>7s} | " + " | ".join(f"{l:>22s}" for l in lab))
for key, a in acc.items():
    x0, dw, p = np.array(a["x0"]), np.array(a["dw"]), np.array(a["p"])
    cells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x0 >= lo) & (x0 < hi) & (p > 0.5)
        cells.append(f"{dw[m].mean():+.3f} {np.abs(dw[m]).mean():.3f} {(np.abs(dw[m]) < 0.01).mean():.2f} ({m.sum():3d})")
    print(f"{key[0]:8s}{key[1]:8s}{key[2]:7g} | " + " | ".join(f"{c:>22s}" for c in cells))
print()
print("same for inputs with P(on) < 0.1 (rarely on)")
for key, a in acc.items():
    x0, dw, p = np.array(a["x0"]), np.array(a["dw"]), np.array(a["p"])
    cells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x0 >= lo) & (x0 < hi) & (p < 0.1)
        cells.append(f"{dw[m].mean():+.3f} {np.abs(dw[m]).mean():.3f} {(np.abs(dw[m]) < 0.01).mean():.2f} ({m.sum():3d})")
    print(f"{key[0]:8s}{key[1]:8s}{key[2]:7g} | " + " | ".join(f"{c:>22s}" for c in cells))
print()
print("synapses from inputs with P(on) > 0.5 at or above 4/3 theta: count at start, at 25k, and of those at 25k how many started below 1 theta")
for key, a in acc.items():
    x0, x1, p = np.array(a["x0"]), np.array(a["x1"]), np.array(a["p"])
    m0 = (x0 >= 4 / 3) & (p > 0.5); m1 = (x1 >= 4 / 3) & (p > 0.5)
    print(f"{key[0]:8s}{key[1]:8s}{key[2]:7g} | start {m0.sum():3d}  25k {m1.sum():3d}  (of which started < 1 theta: {(m1 & (x0 < 1)).sum():3d}, "
          f"< 0.5 theta: {(m1 & (x0 < 0.5)).sum():3d}); of the start's, still >= 4/3 at 25k: {(m0 & m1).sum():3d}")
