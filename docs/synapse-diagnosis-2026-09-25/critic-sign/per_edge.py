"""Per input->output synapse, over a fresh run of the pinned arm: where the clip to [-1, 1] takes weight off the update
LR*A*e (at the top or the bottom), and how big the pushes are against the synapse's weight.  usage: per_edge.py RULE SEED EPOCHS"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from walnutbutter import fast
from walnutbutter.learning import class_evidence, evidence_reward
from walnutbutter.problems import PROBLEMS, dataset_stream
ROOT = Path("/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure")
OUT = Path("/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/critic-sign")
spec = importlib.util.spec_from_file_location("rs", str(ROOT / "docs/rust-sweep.py")); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
rule, seed, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
w = ["--problem", "mnist", "--interval", "100", "--threshold", "0.6", "--minimum-potential", "-0.2", "--hidden-neurons", "0",
     "--temperature", "2", "--eligibility", "hazard", "--seed", str(seed), "--lr", "0.002", "--name", "x"]
w += (["--exploration", "neuron", "--delta", "0.3125", "--drive", "rate"] if rule == "neuron" else
      ["--exploration", "synapse", "--synapse-hazard", "0.01", "--hazard-family", "loglinear", "--synapse-scaling", "count",
       "--trace-counts", "all", "--drive", "rate"])
sys.argv = ["x"] + w
args = rs.parse(); swept, arms = rs.grid_and_arms(args)
grid, cli = rs.grid_of("mnist", arms[0], args.eligibility[0], args.scale, args.floor_ratio, args.wiring, tuple(rs.fixed_words(args)))
patterns, labels = dataset_stream(PROBLEMS["mnist"].data, seed)
n = sum(len(x.outgoing) for x in grid.all_neurons())
acc = {k: np.zeros(n) for k in ("clip_top", "clip_bot", "n_top", "n_bot", "push_abs", "push_sq", "e_abs", "e_sum", "w_sum")}
st = {"b": None, "w": None, "w0": None}
def probe(epoch, engine, network, out, book):
    counts = [int(a + b) for a, b in zip(np.array(engine.epoch_spike_counts())[out], np.array(engine.read_counts())[out])]
    R = evidence_reward(class_evidence(counts, network.population), network.input_label, network.temperature)
    if st["b"] is None: st["b"] = R
    A = R - st["b"]; st["b"] += 0.05 * (R - st["b"])
    e = np.array(engine.scores()); wn = np.array(engine.weights())
    if st["w"] is None:
        st["w"] = st["w0"] = wn; return
    push = cli.lr * A * e
    clip = (wn - st["w"]) - push
    acc["clip_top"] += np.where(clip < 0, clip, 0); acc["n_top"] += clip < -1e-15
    acc["clip_bot"] += np.where(clip > 0, clip, 0); acc["n_bot"] += clip > 1e-15
    acc["push_abs"] += np.abs(push); acc["push_sq"] += push ** 2; acc["e_abs"] += np.abs(e); acc["e_sum"] += e
    acc["w_sum"] += st["w"]
    st["w"] = wn
fast.train(grid, epochs, lr=cli.lr, target=cli.target, patterns=patterns, labels=labels, eligibility=cli.eligibility, seed=seed,
           homeostasis=cli.homeostasis, target_rate=cli.target_rate, unstick=cli.unstick, unstick_target=cli.unstick_target,
           critic=cli.critic, probe=probe, probe_every=1)
np.savez_compressed(OUT / f"edges-{rule}-seed{seed}-{epochs}.npz", w0=st["w0"], w1=st["w"], epochs=epochs - 1, **acc)
T = epochs - 1
wm = acc["w_sum"] / T
print(f"{rule} seed {seed}, {T} paid epochs, {n} synapses: sum of weights {st['w0'].sum():+.3f} -> {st['w'].sum():+.3f}")
print(f"  clip at the top {acc['clip_top'].sum():+.3f} ({int(acc['n_top'].sum())} events), at the bottom {acc['clip_bot'].sum():+.3f} ({int(acc['n_bot'].sum())} events); unclipped pushes summed {(st['w'].sum() - st['w0'].sum()) - acc['clip_top'].sum() - acc['clip_bot'].sum():+.3f}")
bins = [-1.0001, -0.75, -0.25, 0.25, 0.75, 1.0001]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (wm >= lo) & (wm < hi)
    print(f"  mean weight in [{lo:+.2f},{hi:+.2f}) {m.sum():4d} synapses: rms push/epoch {np.sqrt(acc['push_sq'][m].sum() / (m.sum() * T)):.5f}, "
          f"mean |e|/epoch {acc['e_abs'][m].sum() / (m.sum() * T):.4f}, mean e/epoch {acc['e_sum'][m].sum() / (m.sum() * T):+.5f}")
print(f"  corr over synapses of mean |e| with mean weight: {np.corrcoef(acc['e_abs'], wm)[0, 1]:+.3f}")
