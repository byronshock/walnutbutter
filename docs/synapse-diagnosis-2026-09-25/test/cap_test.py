"""Cap test at the pinned point (floor -0.2): take the synapse rule's lr 0.01 25k checkpoint (rate drive), pull every
input->output synapse at or above 4 theta/3 (the fire-from-anywhere line) down to 0.9 theta, and continue for a few
thousand epochs. Arms:
  synapse cap|nocap   resume the synapse arm itself (lr 0.01), weights capped or as saved
  neuron  cap|nocap   resume the neuron rule's lr 0.002 25k checkpoint of the same seed (same wiring) and overwrite all
                      of its weights with the synapse checkpoint's (capped or not); run the neuron rule at lr 0.002
Every 100 epochs the weights are read; every epoch each output's spikes and read escapes.
Usage: cap_test.py rule cap|nocap seed epochs   ->  writes cap-<tag>.npz beside this file."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
M = "/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure"
spec = importlib.util.spec_from_file_location("rs", f"{M}/docs/rust-sweep.py"); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from walnutbutter import fast
from walnutbutter.mnist import pixel_statistics
from walnutbutter.problems import PROBLEMS, dataset_stream

HERE = Path(__file__).resolve().parent
SYN_FIX = ("--exploration", "synapse", "--hazard-family", "loglinear", "--synapse-scaling", "count", "--trace-counts", "all")
NEU_FIX = ("--exploration", "neuron", "--drive", "rate")
rule, capmode, seed, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
syn_arm = {"interval": 100.0, "lr": 0.01, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0, "temperature": 2.0,
           "synapse_hazard": 0.01, "drive": "rate", "seed": seed}
neu_arm = {"interval": 100.0, "lr": 0.002, "delta": 0.3125, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0,
           "temperature": 2.0, "seed": seed}
syn_src = Path(M) / "runs/synapse-lr-25k" / f"{rs.arm_name(syn_arm)}-network.json"
syn_weights = json.load(open(syn_src))["weights"]  # connection-id order
if rule == "synapse":
    arm, fix, src = syn_arm, SYN_FIX, syn_src
else:
    arm, fix, src = neu_arm, NEU_FIX, Path(M) / "runs/neuron-lr-25k" / f"{rs.arm_name(neu_arm)}-network.json"
fresh, cli = rs.grid_of("mnist", arm, "hazard", True, None, None, fix)
grid, offset, reference, baseline, explore_state = rs.resume_grid(fresh, src)
conns = [grid.connections[i] for i in range(1, len(grid.connections) + 1)]
# the synapse checkpoint's wiring must be this grid's: check against a synapse grid of the same seed
sg, _ = rs.grid_of("mnist", syn_arm, "hazard", True, None, None, SYN_FIX)
sidx = {n: k for k, n in enumerate(sg.all_neurons())}; gidx = {n: k for k, n in enumerate(grid.all_neurons())}
sconns = [sg.connections[i] for i in range(1, len(sg.connections) + 1)]
assert [(sidx[c.source], sidx[c.target]) for c in sconns] == [(gidx[c.source], gidx[c.target]) for c in conns]
for c, w in zip(conns, syn_weights):
    c.weight = float(w)  # a no-op for the synapse arm (resume already holds them); the neuron arm takes the synapse's weights
on, _ = pixel_statistics(grid.clock)
inrow = {n: i for i, n in enumerate(grid.input_row())}
outs = list(grid.output_row()); oidx = {n: k for k, n in enumerate(outs)}
edges = [c for n in grid.all_neurons() for c in n.outgoing]  # engine order
p = np.array([on[inrow[c.source]] for c in edges]); t = np.array([oidx[c.target] for c in edges])
th = np.array([n.threshold for n in outs]); fl = np.array([n.minimum_potential for n in outs])
line = (th - fl)[t]
w_start = np.array([c.weight for c in edges])
trapped = w_start >= line
if capmode == "cap":
    for c, tr in zip(edges, trapped):
        if tr:
            c.weight = 0.9 * c.target.threshold
w_start = np.array([c.weight for c in edges])
patterns, labels = dataset_stream(PROBLEMS["mnist"].data, seed)
grid.use_input_stream(patterns, labels); grid.input_at = offset
EVERY = 100
spikes, reads, wsnap, snap_epochs = [], [], [w_start.copy()], [0]


def probe(epoch, engine, network, out, book):
    s = engine.epoch_spike_counts(); r = engine.read_counts()
    spikes.append([s[i] for i in out]); reads.append([r[i] for i in out])
    if (epoch - offset) % EVERY == 0:
        wsnap.append(np.array(engine.weights())); snap_epochs.append(epoch - offset)


t0 = time.time()
mean, trace, engine, report = fast.train(
    grid, epochs, lr=cli.lr, target=cli.target, trace_every=500, patterns=None, labels=None,
    eligibility=cli.eligibility, seed=seed, explore_state=explore_state, pending_events=getattr(grid, "engine_pending", None),
    homeostasis=cli.homeostasis, target_rate=cli.target_rate, unstick=cli.unstick, unstick_target=cli.unstick_target,
    critic=cli.critic, probe=probe, probe_every=1, epoch_offset=offset, baseline=baseline)
tag = f"{rule}-{capmode}-s{seed}-e{epochs}"
np.savez_compressed(HERE / f"cap-{tag}.npz", w=np.array(wsnap), epochs=np.array(snap_epochs), spikes=np.array(spikes, dtype=np.int32),
                    reads=np.array(reads, dtype=np.int32), trapped=trapped, p=p, t=t, th=th, fl=fl, trace=np.array(trace))
print(json.dumps({"tag": tag, "offset": offset, "lr": cli.lr, "seconds": round(time.time() - t0), "trapped_at_25k": int(trapped.sum()),
                  "trapped_often_on": int((trapped & (p > 0.5)).sum()), "trace": [round(x, 4) for x in trace]}), flush=True)
