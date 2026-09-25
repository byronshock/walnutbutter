"""Causal check with the weights frozen (lr 0): load an arm's 25k weights into a fresh build (same seed, same wiring,
checked by digest), optionally cap every synapse from an often-on input (P(on) > 0.5) that fires its output alone
(w >= theta - floor) at 0.9 theta, and count busy outputs (fired in >99% of epochs) over 400 epochs; run under either
rule's dynamics. Usage: ablate.py src_rule src_drive src_lr seed dyn_rule cap(0/1)"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ds", str(HERE / "drift_static.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
from walnutbutter import fast
from walnutbutter.mnist import pixel_statistics
from walnutbutter.persistence import wiring_digest
from walnutbutter.problems import PROBLEMS, dataset_stream
src_rule, src_drive, src_lr, seed, dyn_rule, cap = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
E = 400
def arm_of(rule, drive, lr):
    return ({"interval": 100.0, "lr": lr, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0, "temperature": 2.0,
             "synapse_hazard": 0.01, "drive": drive, "seed": seed}, ds.SYN_FIX, "synapse-lr-25k") if rule == "synapse" else \
           ({"interval": 100.0, "lr": lr, "delta": 0.3125, "threshold": 0.6, "minimum_potential": -0.2, "hidden_neurons": 0.0,
             "temperature": 2.0, "seed": seed}, ds.NEU_FIX, "neuron-lr-25k")
sarm, sfix, sfolder = arm_of(src_rule, src_drive, src_lr)
d = json.load(open(f"runs/{sfolder}/{ds.rs.arm_name(sarm)}-network.json"))
darm, dfix, _ = arm_of(dyn_rule, src_drive if dyn_rule == "synapse" else "rate", 0.002)
g, args = ds.rs.grid_of("mnist", darm, "hazard", True, None, None, dfix)
assert d["wiring_digest"] == wiring_digest(g)
on, _ = pixel_statistics(g.clock); inrow = {n: i for i, n in enumerate(g.input_row())}
capped = 0
for i, w in enumerate(d["weights"], start=1):
    c = g.connections[i]; c.weight = float(w)
    if cap and on[inrow[c.source]] > 0.5 and w >= c.target.threshold - c.target.minimum_potential:
        c.weight = 0.9 * c.target.threshold; capped += 1
patterns, labels = dataset_stream(PROBLEMS["mnist"].data, seed)
spikes = []
def probe(epoch, engine, network, out, book):
    s = engine.epoch_spike_counts(); spikes.append([s[i] for i in out])
t0 = time.time()
fast.train(g, E, lr=0.0, target=args.target, trace_every=0, patterns=patterns, labels=labels, eligibility=args.eligibility,
           seed=seed, homeostasis=args.homeostasis, target_rate=args.target_rate, unstick=args.unstick,
           unstick_target=args.unstick_target, critic=args.critic, probe=probe, probe_every=1)
S = np.array(spikes); fired = (S > 0).mean(0)
print(json.dumps({"weights_from": f"{src_rule} {src_drive} lr {src_lr:g} s{seed} @25k", "dynamics": dyn_rule, "capped_firers": capped,
                  "busy": int((fired > 0.99).sum()), "silent(<1%)": int((fired < 0.01).sum()), "spikes_per_epoch": round(float(S.sum(1).mean()), 1),
                  "busy_list": np.flatnonzero(fired > 0.99).tolist(), "seconds": round(time.time() - t0)}), flush=True)
