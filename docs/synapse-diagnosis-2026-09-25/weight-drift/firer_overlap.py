"""Per arm at 25k: outputs whose threshold lets one synapse fire them from the floor within the weight cap (4/3 theta <= 1),
outputs holding such a synapse (w >= theta - floor) from an input on >50% of the time ("has a firer"), busy outputs
(rate memory > 0.99), and the overlap; and the same at the start (rebuild)."""
import importlib.util, json
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("ds", str(Path(__file__).resolve().parent / "drift_static.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
from walnutbutter.mnist import pixel_statistics
print(f"{'rule':8s}{'drive':8s}{'lr':>7s} s | reachable | firer start | firer 25k | busy 25k | busy&firer | busy w/o firer | firer not busy")
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
    rate = np.array([d["rates"][allidx[n]] for n in outs]); busy = rate > 0.99
    bar = (th - fl)[t]
    f0 = np.bincount(t[(p > 0.5) & (w0 >= bar)], minlength=60) > 0
    f1 = np.bincount(t[(p > 0.5) & (w1 >= bar)], minlength=60) > 0
    reach = (th - fl) <= 1.0
    print(f"{rule:8s}{arm.get('drive','rate'):8s}{arm['lr']:7g} {arm['seed']} | {reach.sum():9d} | {f0.sum():11d} | {f1.sum():9d} | {busy.sum():8d} | "
          f"{(busy & f1).sum():10d} | {(busy & ~f1).sum():14d} | {(f1 & ~busy).sum():14d}")
