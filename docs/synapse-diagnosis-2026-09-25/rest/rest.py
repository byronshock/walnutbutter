"""Output activity against the rest hazard and learning, over time, at mnist's pinned point on the Rust loop.

usage: rest.py <rule: synapse|neuron> <h0 or -> <lr> <drive: rate|charged> <seed> <epochs>

Builds the arm exactly as docs/rust-sweep.py's grid_of does (goo 455, 0 hidden, interval 100, threshold 0.6,
floor -0.2, TAU inf, count read, evidence critic T 2; synapse: loglinear, count scaling, trace all; neuron: width
0.3125), then runs fast.train with a probe after every epoch's update that records, read only:
  per epoch: each output's spikes and read escapes, its potential at the read, the label, the reward (evidence) and
  class-right, the inputs' total spikes, and each output's summed eligibility (engine.scores() over its fan-in);
  every 250 epochs: the outputs' rate memories and all weights.
Writes <out>/<arm>.npz and <out>/<arm>.log. Nothing is written anywhere else.
"""
import importlib.util, json, math, sys, time
from pathlib import Path

import numpy as np

OUT = Path("/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/rest")
rule, h0s, lr, drive, seed, epochs = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])

spec = importlib.util.spec_from_file_location("rs", "docs/rust-sweep.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from walnutbutter import fast
from walnutbutter.learning import TARGETS, class_evidence, evidence_reward
from walnutbutter.mnist import supervised_direction
from walnutbutter.problems import dataset_stream

GRID_LR = 0.002  # the arm dict's lr as the sweeps built it; the lr actually run is `lr`, passed to fast.train
if rule == "synapse":
    arm = {"interval": 100.0, "lr": GRID_LR, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0,
           "temperature": 2.0, "synapse_hazard": float(h0s), "drive": drive, "seed": seed}
    fixed = ("--exploration", "synapse")
    name = f"syn-h{float(h0s):g}-lr{lr:g}-{drive}-s{seed}"
else:
    arm = {"interval": 100.0, "lr": GRID_LR, "delta": 0.3125, "threshold": 0.6, "minimum_potential": -0.2,
           "hidden_neurons": 0.0, "temperature": 2.0, "seed": seed}
    fixed = ()
    name = f"neu-lr{lr:g}-{drive}-s{seed}"
grid, args = rs.grid_of("mnist", arm, "hazard", True, None, None, fixed)
assert args.homeostasis == 0.0 and args.unstick == 0.0, (args.homeostasis, args.unstick)
patterns, labels = dataset_stream("mnist", seed)
direction = supervised_direction(grid)

n_all = len(grid.all_neurons())
log = open(OUT / f"{name}.log", "w")
rec = {k: [] for k in ("spk", "read", "V", "label", "reward", "right", "in_spk", "score_out")}
snap = {"epoch": [], "rates": [], "weights": []}
state = {"edge_target": None, "baseline": None, "t0": time.time(), "thr": None}
WIN = 250


def probe(epoch, engine, network, out, book):
    spk = engine.epoch_spike_counts(); rd = engine.read_counts()
    counts = [spk[i] + rd[i] for i in out]
    groups = class_evidence([int(c) for c in counts], network.population)
    label = network.input_label
    reward = evidence_reward(groups, label, network.temperature)
    right = 1.0 if all(groups[label] > g for c, g in enumerate(groups) if c != label) else 0.0
    V = engine.potentials()
    if state["edge_target"] is None:
        # the engine's edge order is flatten's: each neuron's outgoing in all_neurons order
        neurons = network.all_neurons()
        idx = {id(n): k for k, n in enumerate(neurons)}
        state["edge_target"] = np.array([idx[id(c.target)] for n in neurons for c in n.outgoing])
        state["outpos"] = {o: p for p, o in enumerate(out)}
        state["edge_outpos"] = np.array([state["outpos"].get(t, -1) for t in state["edge_target"]])
        state["ins"] = [i for i in range(len(neurons)) if i not in state["outpos"]]
        state["thr"] = [book.thresholds[i] for i in out]
    sc = np.asarray(engine.scores())
    score_out = np.bincount(state["edge_outpos"][state["edge_outpos"] >= 0], weights=sc[state["edge_outpos"] >= 0],
                            minlength=len(out))
    rec["spk"].append([spk[i] for i in out]); rec["read"].append([rd[i] for i in out])
    rec["V"].append([V[i] for i in out]); rec["label"].append(label); rec["reward"].append(reward)
    rec["right"].append(right); rec["in_spk"].append(sum(spk[i] for i in state["ins"])); rec["score_out"].append(score_out)
    if epoch % WIN == 0 or epoch == 1:
        rates = [book.rates[i] for i in out]
        snap["epoch"].append(epoch); snap["rates"].append(rates); snap["weights"].append(list(engine.weights()))
        if epoch % WIN == 0:
            s = np.array(rec["spk"][-WIN:]); r = np.array(rec["read"][-WIN:])
            c = s + r
            print(f"{epoch:5d} count/out {c.mean():6.3f} (spk {s.mean():6.3f} read {r.mean():6.3f}) "
                  f"outs>0/epoch {(c > 0).sum(1).mean():5.1f} rate>0.99 {sum(x > 0.99 for x in rates):2d} "
                  f"rate<0.01 {sum(x < 0.01 for x in rates):2d} score {np.mean(rec['reward'][-WIN:]):7.3f} "
                  f"right {np.mean(rec['right'][-WIN:]):.3f} ({time.time() - state['t0']:.0f}s)", file=log, flush=True)
    if epoch % 1000 == 0:
        dump(partial=True)


def dump(partial=False):
    np.savez_compressed(OUT / f"{name}.npz", spk=np.array(rec["spk"], dtype=np.uint16), read=np.array(rec["read"], dtype=np.uint16),
                        V=np.array(rec["V"], dtype=np.float32), label=np.array(rec["label"], dtype=np.int8),
                        reward=np.array(rec["reward"]), right=np.array(rec["right"], dtype=np.int8),
                        in_spk=np.array(rec["in_spk"], dtype=np.int32), score_out=np.array(rec["score_out"], dtype=np.float32),
                        snap_epoch=np.array(snap["epoch"]), rates=np.array(snap["rates"], dtype=np.float32),
                        weights=np.array(snap["weights"], dtype=np.float32), edge_outpos=state["edge_outpos"],
                        thresholds=np.array(state["thr"]), partial=partial,
                        meta=json.dumps({"rule": rule, "h0": h0s, "lr": lr, "drive": drive, "seed": seed, "epochs": epochs,
                                         "arm": arm, "fixed": fixed, "args_lr": args.lr, "delta": grid.escape_delta,
                                         "exploration": grid.exploration}))


print(f"{name}: exploration {grid.exploration} h0 {getattr(grid, 'synapse_hazard_rest', None)} drive {grid.drive} "
      f"delta {grid.escape_delta} lr {lr} critic {args.critic} read {grid.read} pickiness {grid.pickiness}", file=log, flush=True)
mean, trace, engine, report = fast.train(
    grid, epochs, lr=lr, target=args.target, trace_every=1000, patterns=patterns, labels=labels,
    eligibility=args.eligibility, seed=seed, homeostasis=args.homeostasis, target_rate=args.target_rate,
    unstick=args.unstick, unstick_target=args.unstick_target, critic=args.critic, direction=direction,
    probe=probe, probe_every=1)
dump()
print("TRACE " + json.dumps({"trace": trace, "right": report["right"], "estimator": report["estimator"],
                             "stuck_on": report["stuck_on"], "stuck_off": report["stuck_off"]}), file=log, flush=True)
print(f"done {time.time() - state['t0']:.0f}s", file=log, flush=True)
