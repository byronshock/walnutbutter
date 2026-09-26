"""Read the ft-*.npz of floor_test.py. For every arm: the synapses from often-on inputs (on > 50% of the time) at or
above 4 theta/3 (the control's line) and at or above the arm's own line theta - floor, over time; per 250-epoch window,
how far such synapses move and how much eligibility they collect, by where they stood at the window's start (w/theta,
and w over the arm's own line); busy outputs (own spike in > 99% of the last 250 epochs) and spikes; and at the end,
busy against holding a synapse at or above the arm's own line."""
import importlib.util, glob, re, sys
from pathlib import Path
import numpy as np
M = "/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure"
spec = importlib.util.spec_from_file_location("rs", f"{M}/docs/rust-sweep.py"); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from walnutbutter.mnist import pixel_statistics
HERE = Path(__file__).resolve().parent
SYN_FIX = ("--exploration", "synapse", "--hazard-family", "loglinear", "--synapse-scaling", "count", "--trace-counts", "all")
_meta = {}


def meta(seed):
    if seed not in _meta:
        arm = {"interval": 100.0, "lr": 0.002, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0, "temperature": 2.0,
               "synapse_hazard": 0.01, "drive": "rate", "seed": seed}
        g, cli = rs.grid_of("mnist", arm, "hazard", True, None, None, SYN_FIX)
        edges = [c for n in g.all_neurons() for c in n.outgoing]
        on, _ = pixel_statistics(g.clock)
        inrow = {n: i for i, n in enumerate(g.input_row())}
        outs = list(g.output_row()); oidx = {n: k for k, n in enumerate(outs)}
        p = np.array([on[inrow[c.source]] for c in edges]); t = np.array([oidx[c.target] for c in edges])
        _meta[seed] = (p, t)
    return _meta[seed]


def busy_at(sp, e, win=250):
    s = sp[max(0, e - win):e]
    return int(((s > 0).mean(0) > 0.99).sum()), float(s.sum(1).mean())


files = sorted(glob.glob(str(HERE / "ft-*.npz")))
rows = []
print("== lr 0 controls (1,000 epochs): what the floor alone does to activity ==")
for f in files:
    m = re.match(r".*ft-floor([-\d.]+)-lr([\d.e-]+)-s(\d)-e(\d+)\.npz", f)
    floor, lr, seed, E = float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))
    if lr != 0:
        continue
    z = np.load(f); sp = z["spikes"]; rd = z["reads"]
    b, s = busy_at(sp, E, E)
    print(f"floor {floor:+.1f} seed {seed}: busy (own spike in >99% of epochs) {b:2d}/60, spikes/epoch {s:6.1f}, read escapes/epoch {rd.sum(1).mean():5.1f}, "
          f"outputs silent in >99% of epochs {int(((sp > 0).mean(0) < 0.01).sum())}")
print()
for f in files:
    m = re.match(r".*ft-floor([-\d.]+)-lr([\d.e-]+)-s(\d)-e(\d+)\.npz", f)
    floor, lr, seed, E = float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))
    if lr == 0:
        continue
    z = np.load(f)
    p, t = meta(seed)
    th, fl = z["th"], z["fl"]
    W, ep, sp, rd = z["w"], z["epochs"], z["spikes"], z["reads"]
    tt, line = th[t], (th - fl)[t]
    often = p > 0.5
    print(f"-- floor {floor:+.1f} (line = {line[0] / tt[0]:.3f} theta) lr {lr:g} seed {seed} -- score trace {np.round(z['trace'], 2).tolist()}")
    print("   epoch | often-on synapses >= 4th/3 | >= own line | outputs holding one >= own line | busy (last 250) | spikes/ep | reads/ep")
    for k, e in enumerate(ep):
        if e not in (0, 250, 1000, 2500, 5000):
            continue
        w = W[k]
        a43 = int((often & (w >= 4 / 3 * tt)).sum()); aown = int((often & (w >= line)).sum()); nout = len(set(t[often & (w >= line)]))
        if e == 0:
            print(f"   {e:5d} | {a43:26d} | {aown:11d} | {nout:31d} |               - |         - |        -")
        else:
            b, s = busy_at(sp, e)
            print(f"   {e:5d} | {a43:26d} | {aown:11d} | {nout:31d} | {b:15d} | {s:9.1f} | {rd[max(0, e - 250):e].sum(1).mean():8.1f}")
    # windows: synapse class at window start -> movement and eligibility inside the window
    X = W[:-1] / tt  # w/theta at each window's start
    R = W[:-1] / line  # w over the arm's own line
    mv = np.abs(np.diff(W, axis=0)); ea = z["eabs"] / 250.0  # mean |e| per epoch in the window
    oft = np.broadcast_to(often, X.shape)
    cells = []
    for lo, hi, nm in ((-9, 0, "w<0"), (0, 0.5, "0..th/2"), (0.5, 1, "th/2..th"), (1, 4 / 3, "th..4th/3"), (4 / 3, 2, "4th/3..2th"),
                       (2, 3, "2th..3th"), (3, 9, ">=3th")):
        mk = oft & (X >= lo) & (X < hi)
        if mk.sum() == 0:
            cells.append(f"      {nm:11s} n     0"); continue
        cells.append(f"      {nm:11s} n {mk.sum():5d}  mean |dw| {mv[mk].mean():.4f}  unmoved {(mv[mk] == 0).mean():.3f}  "
                     f"mean |e|/epoch {ea[mk].mean():.4f}  share |e|<1e-3 {(ea[mk] < 1e-3).mean():.3f}")
    print("   per 250-epoch window, often-on inputs, by w/theta at the window's start:")
    print("\n".join(cells))
    cells = []
    for lo, hi, nm in ((-9, 0.5, "w/line<0.5"), (0.5, 0.75, "0.5..0.75"), (0.75, 0.9, "0.75..0.9"), (0.9, 1.0, "0.9..1"), (1.0, 9, ">=1 (own line)")):
        mk = oft & (R >= lo) & (R < hi)
        if mk.sum() == 0:
            cells.append(f"      {nm:15s} n     0"); continue
        cells.append(f"      {nm:15s} n {mk.sum():5d}  mean |dw| {mv[mk].mean():.4f}  unmoved {(mv[mk] == 0).mean():.3f}  "
                     f"mean |e|/epoch {ea[mk].mean():.4f}  share |e|<1e-3 {(ea[mk] < 1e-3).mean():.3f}")
    print("   same windows, by w over the arm's own line (theta - floor) at the window's start:")
    print("\n".join(cells))
    # the class the synthesis names for -0.6: between 4th/3 and 2th, split by whether the output's line 2th is under the cap
    if floor != -0.2:
        for nm, sel in (("outputs with theta <= 0.5", th[t] <= 0.5 + 1e-9), ("outputs with theta > 0.5", th[t] > 0.5 + 1e-9)):
            mk = oft & (X >= 4 / 3) & (R < 1) & np.broadcast_to(sel, X.shape)
            if mk.sum():
                print(f"   4th/3 <= w < own line, {nm}: n {mk.sum()}, mean |dw| {mv[mk].mean():.4f}, unmoved {(mv[mk] == 0).mean():.3f}, "
                      f"mean |e|/epoch {ea[mk].mean():.4f}")
    # at the end: busy against holding a synapse >= own line
    w = W[-1]
    hold = np.zeros(60, bool); hold[list(set(t[often & (w >= line)]))] = True
    fired = (sp[-250:] > 0).mean(0) > 0.99
    rate_busy = z["rates"][-1] > 0.99
    cap_ok = (th - fl) <= 1.0
    print(f"   end: outputs whose line is under the weight cap {cap_ok.sum()}; holding a synapse >= own line {hold.sum()}, of them busy {int((hold & fired).sum())}; "
          f"not holding one {(~hold).sum()}, of them busy {int((~hold & fired).sum())}; busy by rate memory > 0.99: {int(rate_busy.sum())}")
    print()
