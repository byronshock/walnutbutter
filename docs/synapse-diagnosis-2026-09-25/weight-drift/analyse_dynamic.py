"""Read the dyn-*.npz of drift_dynamic.py: drift over the first 5,000 epochs, the growth of synapses that fire their
output alone (w >= theta - floor, from inputs on >50% of the time), how much a synapse moves in a 250-epoch window by
where it stood at the window's start, busy outputs over time; and from the lr-0 controls, which outputs were busy before
any learning, against the 25k checkpoints."""
import importlib.util, json, glob, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ds", str(HERE / "drift_static.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
from walnutbutter.mnist import pixel_statistics


def meta(rule, drive, lr, seed):
    arm = ({"interval": 100.0, "lr": lr, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0, "temperature": 2.0,
            "synapse_hazard": 0.01, "drive": drive, "seed": seed} if rule == "synapse" else
           {"interval": 100.0, "lr": lr, "delta": 0.3125, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0,
            "temperature": 2.0, "seed": seed})
    fix, folder = (ds.SYN_FIX, "synapse-lr-25k") if rule == "synapse" else (ds.NEU_FIX, "neuron-lr-25k")
    g, cli = ds.rs.grid_of("mnist", arm, "hazard", True, None, None, fix)
    edges = [c for n in g.all_neurons() for c in n.outgoing]
    on, _ = pixel_statistics(g.clock)
    inrow = {n: i for i, n in enumerate(g.input_row())}
    outs = list(g.output_row()); oidx = {n: k for k, n in enumerate(outs)}
    p = np.array([on[inrow[c.source]] for c in edges]); t = np.array([oidx[c.target] for c in edges])
    th = np.array([n.threshold for n in outs]); fl = np.array([n.minimum_potential for n in outs])
    d = json.load(open(f"runs/{folder}/{ds.rs.arm_name(arm)}-network.json"))
    allidx = {n: i for i, n in enumerate(g.all_neurons())}
    rate25 = np.array([d["rates"][allidx[n]] for n in outs])
    # the 25k weights in engine order: the checkpoint's are in connection-id order
    cid = {id(g.connections[i]): i - 1 for i in range(1, len(g.connections) + 1)}
    w25 = np.array(d["weights"])[[cid[id(c)] for c in edges]]
    return p, t, th, fl, rate25, w25


files = sorted(glob.glob(str(HERE / "dyn-*.npz")))
controls = {}
for f in files:
    m = re.match(r".*dyn-(\w+)-(\w+)-lr([\d.e-]+)-s(\d)-run([\d.e-]+)-e(\d+)\.npz", f)
    rule, drive, lr, seed, run, E = m.group(1), m.group(2), float(m.group(3)), int(m.group(4)), float(m.group(5)), int(m.group(6))
    if run == 0:
        controls[(rule, drive, seed)] = np.load(f)
print("== no-learning controls (lr 0, 1,000 epochs): outputs firing in >99% of epochs before any learning, against busy at 25k ==")
for (rule, drive, seed), z in sorted(controls.items()):
    fired = (z["spikes"] > 0).mean(0)
    busy0 = fired > 0.99
    lines = []
    for lr in (0.0005, 0.001, 0.002, 0.005, 0.01) if rule == "synapse" else (0.0005, 0.002):
        p, t, th, fl, rate25, w25 = meta(rule, drive, lr, seed)
        b25 = rate25 > 0.99
        lines.append(f"lr {lr:g}: busy25k {b25.sum():2d}, of them busy at start {(b25 & busy0).sum():2d}, new {(b25 & ~busy0).sum():2d}; start-busy silenced {(busy0 & ~b25).sum():2d}")
    print(f"{rule:7s} {drive:7s} s{seed}: busy at start {busy0.sum():2d}/60 (spikes/epoch {z['spikes'].sum(1).mean():6.1f}, read escapes/epoch {z['reads'].sum(1).mean():5.1f}, "
          f"outputs firing in <1% of epochs {(fired < 0.01).sum():2d})")
    for l in lines:
        print("      " + l)
print()
print("== learning runs: every 1,000 epochs (snapshot at the end of the epoch) ==")
for f in files:
    m = re.match(r".*dyn-(\w+)-(\w+)-lr([\d.e-]+)-s(\d)-run([\d.e-]+)-e(\d+)\.npz", f)
    rule, drive, lr, seed, run, E = m.group(1), m.group(2), float(m.group(3)), int(m.group(4)), float(m.group(5)), int(m.group(6))
    if run == 0 or E < 5000:
        continue
    z = np.load(f)
    p, t, th, fl, rate25, w25 = meta(rule, drive, lr, seed)
    W = z["w"]; ep = z["epochs"]; w0 = W[0]; bar = (th - fl)[t]
    print(f"-- {rule} {drive} lr {lr:g} seed {seed} --")
    print("   epoch | mean dw | dw w0>0 | dw w0<0 | sd dw | firer synapses (p>.5) | outputs w/ firer | busy (fired >99% of last 250 ep) | spikes/ep | reads/ep")
    for k, e in enumerate(ep):
        if e % 1000 and e != 250:
            continue
        w = W[k]; dw = w - w0
        firer = (p > 0.5) & (w >= bar)
        if e == 0:
            busy = "   -"; sp = rd = float("nan")
        else:
            win = z["spikes"][max(0, e - 250):e]; busy = f"{((win > 0).mean(0) > 0.99).sum():4d}"
            sp = z["spikes"][max(0, e - 250):e].sum(1).mean(); rd = z["reads"][max(0, e - 250):e].sum(1).mean()
        print(f"   {e:5d} | {dw.mean():+.4f} | {dw[w0 > 0].mean():+.4f} | {dw[w0 < 0].mean():+.4f} | {dw.std():.3f} | {firer.sum():21d} | "
              f"{len(set(t[firer])):16d} | {busy:>32s} | {sp:9.1f} | {rd:8.1f}")
    dw25 = w25 - w0; firer25 = (p > 0.5) & (w25 >= bar)
    print(f"   25000 | {dw25.mean():+.4f} | {dw25[w0 > 0].mean():+.4f} | {dw25[w0 < 0].mean():+.4f} | {dw25.std():.3f} | {firer25.sum():21d} | "
          f"{len(set(t[firer25])):16d} | {'(rate memory) ' + str((rate25 > 0.99).sum()):>32s} |")
    # how far a synapse from an often-on input moves in a 250-epoch window, by where it stood at the window's start
    x = W[:-1] / th[t]; mv = np.abs(np.diff(W, axis=0)); often = (p > 0.5)[None, :].repeat(len(x), 0)
    cells = []
    for lo, hi, nm in ((-9, 0, "w<0"), (0, 0.5, "0..th/2"), (0.5, 1, "th/2..th"), (1, 4 / 3, "th..4/3th"), (4 / 3, 9, ">=4/3th")):
        mk = often & (x >= lo) & (x < hi)
        cells.append(f"{nm} {mv[mk].mean():.4f} ({(mv[mk] == 0).mean():.2f} unmoved, n {mk.sum()})")
    print("   |dw| per 250-epoch window, inputs on >50%, by w/theta at window start: " + "; ".join(cells))
