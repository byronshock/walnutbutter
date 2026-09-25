"""Floor test of the synthesis's trap claim: rerun the synapse rule (rate drive, mnist's pinned point) from the start
with minimum_potential -0.2 (control), -0.6 (the fire-from-anywhere line moves to 2 theta) or -1.2 (3 theta).
Everything else is the sweep's arm (grid_of, streams, seed) -- the same wiring and starting weights at every floor.
A read-only probe every epoch: each output's spikes and read escapes, each synapse's settled eligibility e
(engine.scores()) summed as |e| per 250-epoch window, and every 250 epochs the weights and the rate memories.
Usage: floor_test.py floor lr seed epochs   ->  writes ft-<tag>.npz beside this file."""
import importlib.util, json, sys, csv, time
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("rs", "docs/rust-sweep.py"); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from walnutbutter import fast
from walnutbutter.problems import PROBLEMS, dataset_stream

HERE = Path(__file__).resolve().parent
SYN_FIX = ("--exploration", "synapse", "--hazard-family", "loglinear", "--synapse-scaling", "count", "--trace-counts", "all")
floor, lr, seed, epochs = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
arm = {"interval": 100.0, "lr": lr, "threshold": 0.6, "minimum_potential": floor, "hidden_neurons": 0.0, "temperature": 2.0,
       "synapse_hazard": 0.01, "drive": "rate", "seed": seed}
tag = f"floor{floor:g}-lr{lr:g}-s{seed}-e{epochs}"
grid, args = rs.grid_of("mnist", arm, "hazard", True, None, None, SYN_FIX)
edges = [c for n in grid.all_neurons() for c in n.outgoing]  # the engine's order (fast.build)
w0 = np.array([c.weight for c in edges])
outs = list(grid.output_row())
th = np.array([n.threshold for n in outs]); fl = np.array([n.minimum_potential for n in outs])
patterns, labels = dataset_stream(PROBLEMS["mnist"].data, seed)
EVERY = 250
nE = len(edges)
spikes, reads, wsnap, rsnap, snap_epochs = [], [], [w0.copy()], [], [0]
eabs_win, ezero_win, esum_win = [], [], []
acc = {"abs": np.zeros(nE), "zero": np.zeros(nE), "sum": np.zeros(nE)}


def probe(epoch, engine, network, out, book):
    s = engine.epoch_spike_counts(); r = engine.read_counts()
    spikes.append([s[i] for i in out]); reads.append([r[i] for i in out])
    e = np.array(engine.scores())
    acc["abs"] += np.abs(e); acc["zero"] += (e == 0.0); acc["sum"] += e
    if epoch % EVERY == 0:
        wsnap.append(np.array(engine.weights())); rsnap.append([book.rates[i] for i in out]); snap_epochs.append(epoch)
        eabs_win.append(acc["abs"].copy()); ezero_win.append(acc["zero"].copy()); esum_win.append(acc["sum"].copy())
        for k in acc:
            acc[k][:] = 0


t0 = time.time()
mean, trace, engine, report = fast.train(
    grid, epochs, lr=lr, target=args.target, trace_every=1000, patterns=patterns, labels=labels,
    eligibility=args.eligibility, seed=seed, explore_state=None, pending_events=None,
    homeostasis=args.homeostasis, target_rate=args.target_rate, unstick=args.unstick,
    unstick_target=args.unstick_target, critic=args.critic, direction=None, probe=probe, probe_every=1)
same = None
if floor == -0.2 and lr > 0:  # the control is the sweep's own arm: its trace must match the sweep's csv
    rows = list(csv.DictReader(open(f"runs/synapse-lr-25k/{rs.arm_name(arm)}.csv")))
    ref = {int(r["epoch"]): r["score"] for r in rows}
    mine = {1000 * (k + 1): f"{s:.6g}" for k, s in enumerate(trace)}
    same = all(ref.get(e) == v for e, v in mine.items())
np.savez_compressed(HERE / f"ft-{tag}.npz", w=np.array(wsnap), epochs=np.array(snap_epochs), rates=np.array(rsnap),
                    spikes=np.array(spikes, dtype=np.int32), reads=np.array(reads, dtype=np.int32),
                    eabs=np.array(eabs_win), ezero=np.array(ezero_win), esum=np.array(esum_win),
                    th=th, fl=fl, trace=np.array(trace))
print(json.dumps({"tag": tag, "seconds": round(time.time() - t0), "matches_sweep_trace": same, "trace": [round(t, 4) for t in trace],
                  "theta_range": [float(th.min()), float(th.max())], "floor_over_theta": float((fl / th).mean()),
                  "homeostasis": args.homeostasis, "unstick": args.unstick}), flush=True)
