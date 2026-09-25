"""The weights into the outputs, fresh vs 25k, and the drift they give an output per epoch in expectation.

For each network: per output, the rate memory (fires-every-epoch share), the sums of the weights into it (all, clock,
pixel-on inputs, complement inputs), and the expected integrated drive per epoch -- relayed: spikes per bit-1 input per
epoch times the weights of the inputs on, averaged over the training split class by class (pixel_statistics); ventured:
escapes per synapse per epoch (by the source's bit) times the weights. The per-epoch constants are the ones the
instrumented runs measured (instrument.py), read from their json. Also: which half (fire-if-one / fire-if-zero) the
stuck-on outputs sit in, and how the expected drift moved from the fresh network to the 25k one, by input group.
"""
import importlib.util, json, sys, statistics as st
import numpy as np
from walnutbutter.mnist import pixel_statistics
from walnutbutter.neuron import Neuron

ROOT = "/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure"
OUT = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/excitation"
spec = importlib.util.spec_from_file_location("rs", f"{ROOT}/docs/rust-sweep.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
SYN_FIXED = ["--exploration", "synapse", "--hazard-family", "loglinear", "--synapse-scaling", "count", "--trace-counts", "all"]
NEU_FIXED = ["--exploration", "neuron", "--drive", "rate"]


def network(kind, drive, seed, source, lr=0.002):
    if kind == "synapse":
        arm = dict(interval=100.0, lr=lr, threshold=0.6, minimum_potential=-0.2, hidden_neurons=0.0, temperature=2.0,
                   synapse_hazard=0.01, drive=drive, seed=seed)
        fixed = SYN_FIXED
    else:
        arm = dict(interval=100.0, lr=lr, delta=0.3125, threshold=0.6, minimum_potential=-0.2, hidden_neurons=0.0,
                   temperature=2.0, seed=seed)
        fixed = NEU_FIXED
    g, _ = rs.grid_of("mnist", arm, "hazard", True, None, None, fixed)
    if source is not None:
        g = rs.resume_grid(g, f"{ROOT}/{source}")[0]
    return g


def matrix(g):
    outs, ins = list(g.output_row()), list(g.input_row())
    oi = {id(n): k for k, n in enumerate(outs)}
    W = np.zeros((len(ins), len(outs)))
    for s, n in enumerate(ins):
        for c in n.outgoing:
            W[s, oi[id(c.target)]] += c.weight
    global FLOORS
    FLOORS = np.array([n.minimum_potential for n in outs])
    return W, np.array([n.rate for n in outs]), np.array([n.threshold for n in outs])


def arm_path(kind, drive, seed, lr=0.002):
    if kind == "synapse":
        return (f"runs/synapse-lr-25k/interval100-lr{lr:g}-threshold0.6-minimum_potential-0.2-hidden_neurons0-"
                f"temperature2-synapse_hazard0.01-drive{drive}-seed{seed}-network.json")
    return (f"runs/neuron-lr-25k/interval100-lr{lr:g}-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-"
            f"temperature2-seed{seed}-network.json")


def constants(label):
    d = json.load(open(f"{OUT}/{label}.json"))
    ec = d["edge_counts"]
    k1 = d["in_ven"]["in_ven_bit1"] / ec["bit1"] if ec["bit1"] else 0.0
    k0 = d["in_ven"]["in_ven_bit0"] / ec["bit0"] if ec["bit0"] else 0.0
    kc = d["in_ven"]["in_ven_clock"] / ec["clock"] if ec["clock"] else 0.0
    return dict(s1=d["in_spikes_per_bit1"], s0=d["in_spikes_per_bit0"], sc=d["in_clock_spikes"], k1=k1, k0=k0, kc=kc)


on, on_given = pixel_statistics(clock=3)
CLOCK = 3
prior = np.full(10, 0.1)


def drift(W, c, synaptic, cls=None):
    """Expected integrated drive per epoch per output, relayed and ventured, averaged over the split or given a class."""
    p = on if cls is None else on_given[cls]
    p = p.copy()
    spikes = np.where(np.arange(len(p)) < CLOCK, c["sc"], c["s1"] * p + c["s0"] * (1 - p))
    rel = spikes @ W
    if synaptic:
        ven_rate = np.where(np.arange(len(p)) < CLOCK, c["kc"], c["k1"] * p + c["k0"] * (1 - p))
        ven = ven_rate @ W
    else:
        ven = np.zeros(W.shape[1])
    return rel, ven


def report(label, W, rates, thetas, c, synaptic, W0=None):
    rel, ven = drift(W, c, synaptic)
    on_mask = rates > 0.99
    half = np.array([0] * 30 + [1] * 30)
    print(f"\n== {label}: stuck on (rate > 0.99) {on_mask.sum()} of 60 -- fire-if-one {on_mask[:30].sum()}, "
          f"fire-if-zero {on_mask[30:].sum()}; rate < 0.01: {(rates < 0.01).sum()}")
    print(f"   weights into outputs: mean {W[W != 0].mean():+.4f}; per output sum: all {W.sum(0).mean():+.3f}, "
          f"clock {W[:CLOCK].sum(0).mean():+.3f}, pixel {W[CLOCK:CLOCK + 196].sum(0).mean():+.3f}, "
          f"complement {W[CLOCK + 196:].sum(0).mean():+.3f}")
    print(f"   expected drive per epoch, split average (mean over outputs, in units of the output's theta): "
          f"relayed {np.mean(rel / thetas):+.3f}, ventured {np.mean(ven / thetas):+.3f}; outputs with relayed+ventured > 0: "
          f"{((rel + ven) > 0).sum()} of 60, relayed alone > 0: {(rel > 0).sum()}")
    for name, m in (("stuck-on", on_mask), ("rest", ~on_mask)):
        if m.sum():
            print(f"   {name:8s} ({m.sum():2d}): relayed/theta mean {np.mean(rel[m] / thetas[m]):+.3f} "
                  f"[min {np.min(rel[m] / thetas[m]):+.3f}], ventured/theta mean {np.mean(ven[m] / thetas[m]):+.3f}, "
                  f"relayed>0 {int((rel[m] > 0).sum())}, (rel+ven)>0 {int(((rel + ven)[m] > 0).sum())}")
    # single-arrival triggers: a synapse whose weight alone lifts the output from rest (w >= theta) or from the floor
    idx = np.arange(W.shape[0])
    spikes_in = np.where(idx < CLOCK, c["sc"], c["s1"] * on + c["s0"] * (1 - on))
    ven_in = np.where(idx < CLOCK, c["kc"], c["k1"] * on + c["k0"] * (1 - on)) if synaptic else np.zeros(W.shape[0])
    trig = W >= thetas[None, :]
    trig_f = W >= (thetas - FLOORS)[None, :]
    rel_trig, ven_trig = spikes_in @ trig, ven_in @ trig
    print(f"   trigger synapses (w >= theta; w >= theta - floor): per output {trig.sum(0).mean():.2f}; {trig_f.sum(0).mean():.2f}; "
          f"outputs with at least one {int((trig.sum(0) > 0).sum())}; {int((trig_f.sum(0) > 0).sum())}; "
          f"trigger arrivals per epoch per output, relayed {rel_trig.mean():.2f}, ventured {ven_trig.mean():.3f}")
    for name, m in (("stuck-on", on_mask), ("rest", ~on_mask)):
        if m.sum():
            print(f"     {name:8s}: trigger synapses {trig.sum(0)[m].mean():.2f} (from floor {trig_f.sum(0)[m].mean():.2f}); "
                  f"relayed trigger arrivals/epoch {rel_trig[m].mean():.2f} [share of outputs with >= 1: {np.mean(rel_trig[m] >= 1):.2f}], "
                  f"ventured {ven_trig[m].mean():.3f}; positive weights {int((W[:, m] > 0).sum())}, mean positive w/theta "
                  f"{np.mean((W / thetas[None, :])[:, m][W[:, m] > 0]):.3f}, mean negative w/theta {np.mean((W / thetas[None, :])[:, m][W[:, m] < 0]):+.3f}")
    # class-conditional: how many of the 10 classes give a positive expected drive, per output
    pos = np.zeros(60, int)
    sel = []
    for k in range(10):
        r, v = drift(W, c, synaptic, k)
        pos += ((r + v) > 0)
    print(f"   classes (of 10) with a positive expected drive, per output: stuck-on mean {pos[on_mask].mean() if on_mask.any() else float('nan'):.2f}, "
          f"rest mean {pos[~on_mask].mean() if (~on_mask).any() else float('nan'):.2f}; outputs positive for all 10 classes: {(pos == 10).sum()}")
    # the output's own label selectivity: drive when it should be on minus when it should be off
    good = []
    for j in range(60):
        k = (j % 30) // 3
        r_k, v_k = drift(W, c, synaptic, k)
        others = [drift(W, c, synaptic, q) for q in range(10) if q != k]
        r_o = np.mean([o[0][j] for o in others]); v_o = np.mean([o[1][j] for o in others])
        want_on_given_k = j < 30  # fire-if-one: on for its class; fire-if-zero: off for its class
        diff = (r_k[j] - r_o) if want_on_given_k else (r_o - r_k[j])
        good.append(diff / thetas[j])
    good = np.array(good)
    print(f"   relayed selectivity (drive when it should be on minus when it should be off, /theta): mean {good.mean():+.3f}, "
          f"positive for {int((good > 0).sum())} of 60")
    if W0 is not None:
        dW = W - W0
        r0, v0 = drift(W0, c, synaptic)
        pgroups = {"clock": slice(0, CLOCK), "pixel": slice(CLOCK, CLOCK + 196), "complement": slice(CLOCK + 196, None)}
        spikes = np.where(np.arange(len(on)) < CLOCK, c["sc"], c["s1"] * on + c["s0"] * (1 - on))
        parts = {name: float(np.mean((spikes[sl] @ dW[sl]) / thetas)) for name, sl in pgroups.items()}
        print(f"   from fresh: relayed drive/theta moved {np.mean((rel - r0) / thetas):+.3f} (clock {parts['clock']:+.3f}, "
              f"pixel {parts['pixel']:+.3f}, complement {parts['complement']:+.3f}); ventured/theta moved "
              f"{np.mean((ven - v0) / thetas):+.3f}; |dW| mean {np.abs(dW[W != 0]).mean():.3f}, dW mean {dW[W != 0].mean():+.4f}, "
              f"weights at +-1: {int((np.abs(W[W != 0]) > 0.999).sum())} of {int((W != 0).sum())} (fresh {int((np.abs(W0[W0 != 0]) > 0.999).sum())})")
        for name, m in (("stuck-on", on_mask), ("rest", ~on_mask)):
            if m.sum():
                print(f"     {name:8s}: relayed/theta fresh {np.mean(r0[m] / thetas[m]):+.3f} -> {np.mean(rel[m] / thetas[m]):+.3f}")
    return rel, ven


if __name__ == "__main__":
    which = sys.argv[1:] or ["main"]
    c_rate = constants("syn-rate-25k")
    c_charged = constants("syn-charged-25k")
    c_neu = constants("neu-25k")
    print("per-epoch constants measured (instrument.py, 40 epochs):")
    for name, c in (("synapse rate 25k", c_rate), ("synapse charged 25k", c_charged), ("neuron 25k", c_neu),
                    ("synapse rate fresh", constants("syn-rate-fresh")), ("synapse charged fresh", constants("syn-charged-fresh"))):
        print(f"  {name:22s}: spikes per bit-1 input {c['s1']:.3f}, per bit-0 input {c['s0']:.3f}, per clock input {c['sc']:.3f}; ventured escapes per "
              f"synapse per epoch: bit-1 source {c['k1']:.4f}, bit-0 source {c['k0']:.4f}, clock {c['kc']:.4f}")
    fresh = network("synapse", "rate", 1, None)
    W0, r0, th0 = matrix(fresh)
    fresh_n = network("neuron", "rate", 1, None)
    assert np.array_equal(matrix(fresh_n)[0], W0), "fresh weights differ between the rules"
    report("fresh seed 1, synapse constants (rate)", W0, np.zeros(60), th0, c_rate, True)
    report("fresh seed 1, synapse constants (charged)", W0, np.zeros(60), th0, c_charged, True)
    for seed in (1, 2, 3):
        f = network("synapse", "rate", seed, None); W0s, _, th = matrix(f)
        for drive, c in (("rate", c_rate), ("charged", c_charged)):
            W, rates, th = matrix(network("synapse", drive, seed, arm_path("synapse", drive, seed)))
            report(f"synapse {drive} lr 0.002 seed {seed} at 25k", W, rates, th, c, True, W0s)
        W, rates, th = matrix(network("neuron", "rate", seed, arm_path("neuron", "rate", seed)))
        report(f"neuron lr 0.002 seed {seed} at 25k", W, rates, th, c_neu, False, W0s)
