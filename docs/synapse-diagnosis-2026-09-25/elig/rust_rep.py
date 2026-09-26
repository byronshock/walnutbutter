"""The critic investigator's measurement, repeated on the Rust loop at mnist's point on fresh seeds: fast.train as
probe_arm.py ran it (the sweep's grid_of, lr 0.002, rate drive, evidence critic), a probe every epoch after the update
reading engine.scores() -- the epoch's settled scores, cleared at the next reset -- summed over every input->output
synapse. Also records the read escapes and spikes. usage: rust_rep.py SEED EPOCHS [STREAM]"""
import importlib.util, json, sys, time
import numpy as np
from walnutbutter import fast
from walnutbutter.problems import PROBLEMS, dataset_stream

ROOT = "/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure"
OUT = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/elig"
spec = importlib.util.spec_from_file_location("rs", ROOT + "/docs/rust-sweep.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
seed, epochs = int(sys.argv[1]), int(sys.argv[2])
stream = int(sys.argv[3]) if len(sys.argv) > 3 else seed  # the exploration stream's seed; the network and the patterns keep SEED
words = ["--problem", "mnist", "--interval", "100", "--threshold", "0.6", "--minimum-potential", "-0.2",
         "--hidden-neurons", "0", "--temperature", "2", "--eligibility", "hazard", "--seed", str(seed),
         "--epochs", str(epochs), "--lr", "0.002", "--name", "diag-elig-unused",
         "--exploration", "synapse", "--synapse-hazard", "0.01", "--hazard-family", "loglinear",
         "--synapse-scaling", "count", "--trace-counts", "all", "--drive", "rate", "charged"]
sys.argv = ["rust-sweep.py"] + words
args = rs.parse()
swept, arms = rs.grid_and_arms(args)
arms = [a for a in arms if a.get("drive", "rate") == "rate"]
assert len(arms) == 1, arms
arm = arms[0]
grid, cli = rs.grid_of("mnist", arm, args.eligibility[0], args.scale, args.floor_ratio, args.wiring, tuple(rs.fixed_words(args)))
patterns, labels = dataset_stream(PROBLEMS["mnist"].data, seed)
rec = {"S": [], "S_inout": [], "rd": [], "spk": []}
mask = None


def probe(epoch, engine, network, out, book):
    global mask
    if mask is None:
        neurons = list(network.all_neurons())
        idx = {n: i for i, n in enumerate(neurons)}
        ins, outs = set(network.input_row()), set(network.output_row())
        mask = np.array([(c.source in ins) and (c.target in outs) for n in neurons for c in n.outgoing])
    e = np.array(engine.scores())
    rec["S"].append(e.sum())
    rec["S_inout"].append(e[mask].sum())
    rec["rd"].append(int(np.array(engine.read_counts())[out].sum()))
    rec["spk"].append(int(np.array(engine.epoch_spike_counts())[out].sum()))


t0 = time.time()
mean, trace, engine, report = fast.train(
    grid, epochs, lr=cli.lr, target=cli.target, trace_every=1000, patterns=patterns, labels=labels,
    eligibility=cli.eligibility, seed=stream, homeostasis=cli.homeostasis, target_rate=cli.target_rate,
    unstick=cli.unstick, unstick_target=cli.unstick_target, critic=cli.critic, probe=probe, probe_every=1)
S = np.array(rec["S_inout"])
np.savez_compressed(f"{OUT}/rust-seed{seed}{'' if stream == seed else f'-stream{stream}'}-{epochs}.npz", **{k: np.array(v) for k, v in rec.items()})
print(json.dumps({"seed": seed, "stream": stream, "arm": rs.arm_name(arm), "epochs": epochs, "mean_S": float(S.mean()),
                  "se": float(S.std(ddof=1) / len(S) ** 0.5), "edges": int(mask.sum()), "all_edges": len(mask),
                  "trace": trace, "seconds": round(time.time() - t0)}), flush=True)
