"""Per edge, for 60 epochs of the neuron-rule arm: is the weight change LR*A*score, with score read after the pay?"""
import importlib.util, sys
from pathlib import Path
import numpy as np
from walnutbutter import fast
from walnutbutter.learning import class_evidence, evidence_reward
from walnutbutter.problems import PROBLEMS, dataset_stream
ROOT = Path("/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure")
spec = importlib.util.spec_from_file_location("rs", str(ROOT / "docs/rust-sweep.py")); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
rule = sys.argv[1]
w = ["--problem", "mnist", "--interval", "100", "--threshold", "0.6", "--minimum-potential", "-0.2", "--hidden-neurons", "0",
     "--temperature", "2", "--eligibility", "hazard", "--seed", "1", "--lr", "0.002", "--name", "x"]
w += (["--exploration", "neuron", "--delta", "0.3125", "--drive", "rate"] if rule == "neuron" else
      ["--exploration", "synapse", "--synapse-hazard", "0.01", "--drive", "rate"])
sys.argv = ["x"] + w
args = rs.parse(); swept, arms = rs.grid_and_arms(args)
grid, cli = rs.grid_of("mnist", arms[0], args.eligibility[0], args.scale, args.floor_ratio, args.wiring, tuple(rs.fixed_words(args)))
print("quash", grid.quash_rate, "weight_range", grid.weight_range)
patterns, labels = dataset_stream(PROBLEMS["mnist"].data, 1)
st = {"b": None, "w": None}
def probe(epoch, engine, network, out, book):
    counts = [int(a + b) for a, b in zip(np.array(engine.epoch_spike_counts())[out], np.array(engine.read_counts())[out])]
    R = evidence_reward(class_evidence(counts, network.population), network.input_label, network.temperature)
    if st["b"] is None: st["b"] = R
    A = R - st["b"]; st["b"] += 0.05 * (R - st["b"])
    e = np.array(engine.scores()); wn = np.array(engine.weights())
    if st["w"] is not None and epoch > 1:
        dw = wn - st["w"]; pred = np.clip(st["w"] + 0.002 * A * e, -1, 1) - st["w"]
        bad = np.abs(dw - pred) > 1e-12
        if epoch % 10 == 0 or bad.sum():
            print(f"epoch {epoch} A {A:+.3f} sum dw {dw.sum():+.5f} sum pred {pred.sum():+.5f} edges off {bad.sum()}"
                  + (f"  e.g. edge {np.argmax(bad)}: w {st['w'][np.argmax(bad)]:+.5f} dw {dw[np.argmax(bad)]:+.6f} pred {pred[np.argmax(bad)]:+.6f} e {e[np.argmax(bad)]:+.4f}" if bad.sum() else ""))
    st["w"] = wn
fast.train(grid, 40, lr=cli.lr, target=cli.target, patterns=patterns, labels=labels, eligibility=cli.eligibility, seed=1,
           homeostasis=cli.homeostasis, target_rate=cli.target_rate, unstick=cli.unstick, unstick_target=cli.unstick_target,
           critic=cli.critic, probe=probe, probe_every=1)
