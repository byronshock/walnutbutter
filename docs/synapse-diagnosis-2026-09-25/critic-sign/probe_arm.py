"""Per-epoch record of the critic and the update at mnist's pinned point, both rules, run by fast.train itself.

usage: probe_arm.py RULE SEED EPOCHS [resume]
  RULE    synapse | neuron   (synapse: loglinear, h0 0.01, count scaling, trace all; neuron: width 0.3125, hazard)
  resume  continue from the arm's 25k final checkpoint in runs/synapse-lr-25k or runs/neuron-lr-25k (read only)

The run is fast.train with the sweep's own grid_of / resume_grid, lr 0.002, rate drive, evidence critic, T 2.
A probe every epoch (after the update) reads: the reward R (recomputed from the engine's counts by the same
function), the baseline b the update used and the advantage A = R - b (tracked exactly as train does), each
output's spikes and read escapes, each input->output synapse's settled score e (engine.scores(), cleared at the
next reset), the weight change the update applied, and the critic's marginal for one more count on each class's
evidence. Writes <out>/<tag>.npz.
"""
import importlib.util, json, math, sys, time
from pathlib import Path

import numpy as np

from walnutbutter import fast
from walnutbutter.learning import class_evidence, evidence_reward
from walnutbutter.problems import PROBLEMS, dataset_stream

ROOT = Path("/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure")
OUT = Path("/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/critic-sign")
spec = importlib.util.spec_from_file_location("rs", str(ROOT / "docs/rust-sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)

rule, seed, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
resume = len(sys.argv) > 4 and sys.argv[4] == "resume"
LR = 0.002
common = ["--problem", "mnist", "--interval", "100", "--threshold", "0.6", "--minimum-potential", "-0.2",
          "--hidden-neurons", "0", "--temperature", "2", "--eligibility", "hazard", "--seed", str(seed),
          "--epochs", str(epochs), "--lr", str(LR), "--name", "diag-critic-sign-unused"]
if rule == "synapse":
    words = common + ["--exploration", "synapse", "--synapse-hazard", "0.01", "--hazard-family", "loglinear",
                      "--synapse-scaling", "count", "--trace-counts", "all", "--drive", "rate", "charged"]  # swept there: in the arm name
    sweep = "synapse-lr-25k"
else:
    words = common + ["--exploration", "neuron", "--delta", "0.3125", "--drive", "rate"]
    sweep = "neuron-lr-25k"
sys.argv = ["rust-sweep.py"] + words
args = rs.parse()
swept, arms = rs.grid_and_arms(args)
arms = [a for a in arms if a.get("drive", "rate") == "rate"]
assert len(arms) == 1, arms
arm = arms[0]
fixed = tuple(rs.fixed_words(args))
grid, cli = rs.grid_of("mnist", arm, args.eligibility[0], args.scale, args.floor_ratio, args.wiring, fixed)
name = rs.arm_name(arm)
print("arm", name, "fixed", fixed, flush=True)

patterns, labels = dataset_stream(PROBLEMS["mnist"].data, seed)
explore_state, baseline, offset = None, None, 0
if resume:
    source = ROOT / "runs" / sweep / f"{name}-network.json"
    grid, offset, reference, baseline, explore_state = rs.resume_grid(grid, source)
    grid.use_input_stream(patterns, labels)
    grid.input_at = offset
    patterns = labels = None
    print("resumed at", offset, "baseline", baseline, flush=True)

row = grid.output_row()
T = grid.temperature
P = grid.population
state = {"b": baseline, "w_prev": None}
rec = {k: [] for k in ("R", "b", "A", "label", "spk", "rd", "e_j", "epos_j", "eneg_j", "npos_j", "nneg_j", "dw_j",
                       "wsum_j", "dRp", "dRm", "V", "atlow", "athigh")}
mask = None


def probe(epoch, engine, network, out, book):
    global mask
    if mask is None:
        src, tgt = [], []
        neurons = list(network.all_neurons())
        for n in neurons:
            for c in n.outgoing:
                src.append(c.source); tgt.append(c.target)
        idx = {n: i for i, n in enumerate(neurons)}
        outs = set(row)
        ins = set(network.input_row())
        tgt_i = np.array([idx[t] for t in tgt]); src_i = np.array([idx[s] for s in src])
        mask = np.array([(t in outs) and (s in ins) for s, t in zip(src, tgt)])
        pos = {o: k for k, o in enumerate(out)}
        probe.col = np.array([pos.get(t, -1) for t in tgt_i])
        kinds = {"in->out": int(mask.sum()), "total": len(src),
                 "from outputs": int(sum(s in outs for s in src)), "into inputs": int(sum(t in ins for t in tgt))}
        print("edges", kinds, flush=True)
        probe.low, probe.high = network.weight_range
        if state["w_prev"] is None:
            state["w_prev"] = np.array(engine.weights())  # epoch 1: A = 0 moved nothing
    spk = np.array(engine.epoch_spike_counts(), dtype=np.int64)[out]
    rd = np.array(engine.read_counts(), dtype=np.int64)[out]
    counts = [int(x) for x in spk + rd]
    groups = class_evidence(counts, P)
    y = network.input_label
    R = evidence_reward(groups, y, T)
    if state["b"] is None:
        state["b"] = R
    b = state["b"]
    A = R - b
    state["b"] = b + 0.05 * (R - b)
    e = np.array(engine.scores())
    w = np.array(engine.weights())
    dw = w - state["w_prev"]; state["w_prev"] = w
    col = probe.col
    m = mask
    ej = np.bincount(col[m], weights=e[m], minlength=60)
    epos = np.bincount(col[m], weights=np.where(e[m] > 0, e[m], 0.0), minlength=60)
    eneg = np.bincount(col[m], weights=np.where(e[m] < 0, e[m], 0.0), minlength=60)
    npos = np.bincount(col[m], weights=(e[m] > 0).astype(float), minlength=60)
    nneg = np.bincount(col[m], weights=(e[m] < 0).astype(float), minlength=60)
    dwj = np.bincount(col[m], weights=dw[m], minlength=60)
    wsj = np.bincount(col[m], weights=w[m], minlength=60)
    g = np.array(groups, dtype=float)
    base = evidence_reward(groups, y, T)
    dRp = np.array([evidence_reward([gg + (1 if k == c else 0) for c, gg in enumerate(groups)], y, T) - base for k in range(len(groups))])
    dRm = np.array([evidence_reward([gg - (1 if k == c else 0) for c, gg in enumerate(groups)], y, T) - base for k in range(len(groups))])
    V = np.array(engine.potentials())[out]
    for k, v in (("R", R), ("b", b), ("A", A), ("label", y), ("spk", spk), ("rd", rd), ("e_j", ej), ("epos_j", epos),
                 ("eneg_j", eneg), ("npos_j", npos), ("nneg_j", nneg), ("dw_j", dwj), ("wsum_j", wsj), ("dRp", dRp),
                 ("dRm", dRm), ("V", V), ("atlow", int((w[m] <= probe.low).sum())), ("athigh", int((w[m] >= probe.high).sum()))):
        rec[k].append(v)
    if epoch % 500 == 0:
        print(f"epoch {offset + epoch}: R {R:.3f} b {b:.3f} spikes {spk.sum()} reads {rd.sum()} "
              f"mean w in->out {w[m].mean():.5f} at bounds {int((w[m] <= probe.low).sum())}/{int((w[m] >= probe.high).sum())}",
              flush=True)


t0 = time.time()
mean, trace, engine, report = fast.train(
    grid, epochs, lr=cli.lr, target=cli.target, trace_every=1000, patterns=patterns, labels=labels,
    eligibility=cli.eligibility, seed=seed, explore_state=explore_state, pending_events=getattr(grid, "engine_pending", None),
    homeostasis=cli.homeostasis, target_rate=cli.target_rate, unstick=cli.unstick, unstick_target=cli.unstick_target,
    critic=cli.critic, probe=probe, probe_every=1, epoch_offset=offset, baseline=baseline)
assert abs(state["b"] - report["baseline"]) < 1e-12, (state["b"], report["baseline"])
R = np.array(rec["R"])
check = [float(R[999 + 1000 * k]) for k in range(len(trace))]
assert all(abs(a - t) < 1e-9 for a, t in zip(check, trace)), (check, trace)
tag = f"{rule}-seed{seed}-{'resume25k' if resume else 'fresh'}-{epochs}"
np.savez_compressed(OUT / f"{tag}.npz", **{k: np.array(v) for k, v in rec.items()}, lr=cli.lr, offset=offset,
                    weight_range=np.array(grid.weight_range), trace=np.array(trace))
print(json.dumps({"tag": tag, "arm": name, "lr": cli.lr, "mean": mean, "last_tenth": report["last_tenth"],
                  "stuck_on": report["stuck_on"], "trace": trace, "seconds": round(time.time() - t0)}), flush=True)
