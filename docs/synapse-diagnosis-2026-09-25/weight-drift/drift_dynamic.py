"""Weight drift over time: rerun an arm of the 25k sweeps from its start for a few thousand epochs on the Rust loop,
exactly as docs/rust-sweep.py's run_arm calls fast.train (same grid_of, same streams, same seed), with a read-only probe
every epoch: each output's spikes and read escapes, and every 250 epochs the weights and the rate memories.
lr_run = 0 is a no-learning control on the same network and inputs (the starting activity).
Usage: drift_dynamic.py rule drive lr seed epochs lr_run   ->  writes dyn-<tag>.npz beside this file."""
import importlib.util, json, sys, csv, time
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("rs", "docs/rust-sweep.py"); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from walnutbutter import fast
from walnutbutter.problems import PROBLEMS, dataset_stream

HERE = Path(__file__).resolve().parent
SYN_FIX = ("--exploration", "synapse", "--hazard-family", "loglinear", "--synapse-scaling", "count", "--trace-counts", "all")
NEU_FIX = ("--exploration", "neuron", "--drive", "rate")
rule, drive, lr, seed, epochs, lr_run = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
if rule == "synapse":
    arm = {"interval": 100.0, "lr": lr, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0, "temperature": 2.0,
           "synapse_hazard": 0.01, "drive": drive, "seed": seed}; fix = SYN_FIX; folder = "synapse-lr-25k"
else:
    arm = {"interval": 100.0, "lr": lr, "delta": 0.3125, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0,
           "temperature": 2.0, "seed": seed}; fix = NEU_FIX; folder = "neuron-lr-25k"
tag = f"{rule}-{drive}-lr{lr:g}-s{seed}-run{lr_run:g}-e{epochs}"
grid, args = rs.grid_of("mnist", arm, "hazard", True, None, None, fix)
edges = [c for n in grid.all_neurons() for c in n.outgoing]  # the engine's order (fast.build)
w0 = np.array([c.weight for c in edges])
patterns, labels = dataset_stream(PROBLEMS["mnist"].data, seed)
EVERY = 250
spikes, reads, wsnap, rsnap, snap_epochs = [], [], [w0.copy()], [], [0]


def probe(epoch, engine, network, out, book):
    s = engine.epoch_spike_counts(); r = engine.read_counts()
    spikes.append([s[i] for i in out]); reads.append([r[i] for i in out])
    if epoch % EVERY == 0:
        wsnap.append(np.array(engine.weights())); rsnap.append([book.rates[i] for i in out]); snap_epochs.append(epoch)


t0 = time.time()
mean, trace, engine, report = fast.train(
    grid, epochs, lr=lr_run, target=args.target, trace_every=1000, patterns=patterns, labels=labels,
    eligibility=args.eligibility, seed=seed, explore_state=None, pending_events=None,
    homeostasis=args.homeostasis, target_rate=args.target_rate, unstick=args.unstick,
    unstick_target=args.unstick_target, critic=args.critic, direction=None, probe=probe, probe_every=1)
# the sweep's own trace for the same epochs: the rerun is the same run iff the scores agree
same = None
if lr_run == lr:
    rows = list(csv.DictReader(open(f"runs/{folder}/{rs.arm_name(arm)}.csv")))
    ref = {int(r["epoch"]): r["score"] for r in rows}
    mine = {1000 * (k + 1): f"{s:.6g}" for k, s in enumerate(trace)}
    same = all(ref.get(e) == v for e, v in mine.items())
out_row = grid.output_row()
np.savez_compressed(HERE / f"dyn-{tag}.npz", w=np.array(wsnap), epochs=np.array(snap_epochs), rates=np.array(rsnap),
                    spikes=np.array(spikes, dtype=np.int32), reads=np.array(reads, dtype=np.int32),
                    thresholds=np.array(report["thresholds"]), trace=np.array(trace))
print(json.dumps({"tag": tag, "seconds": round(time.time() - t0), "matches_sweep_trace": same, "trace": [round(t, 4) for t in trace],
                  "homeostasis": args.homeostasis, "unstick": args.unstick}), flush=True)
