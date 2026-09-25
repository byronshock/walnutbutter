"""Where the outputs' excitation comes from, measured wave by wave on the object engine (weights frozen, no Teacher).

usage: instrument.py LABEL KIND DRIVE SOURCE EPOCHS [suppress]
  KIND   synapse | neuron
  DRIVE  rate | charged      (neuron: rate)
  SOURCE fresh | path to an arm's -network.json
  suppress: under exploration at the synapse, the ventured signals are decided (draws, gains, read escapes as ever)
            but not pushed, so nothing ventured is delivered -- the counterfactual without the ventured input.

Per output per epoch: integrated weight from relayed and from ventured arrivals, split excitatory / inhibitory;
arrivals dropped at a refractory target; floor clamps and the charge they erased; spikes, and for each spike whether a
ventured arrival was integrated in the same wave / within one hop before it, and whether the spike needed the ventured
charge (V at the fire minus the ventured charge integrated since the last reset is below theta); read escapes and the
u they escaped at; V sampled every 1 ms. Inputs: relayed spikes and ventured escapes per synapse, by the bit.
"""
import importlib.util, json, math, random, sys, time
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.problems import dataset_stream

ROOT = "/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure"
OUT = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/excitation"
spec = importlib.util.spec_from_file_location("rs", f"{ROOT}/docs/rust-sweep.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)

label, kind, drive, source, epochs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
suppress = len(sys.argv) > 6 and sys.argv[6] == "suppress"
SYN_FIXED = ["--exploration", "synapse", "--hazard-family", "loglinear", "--synapse-scaling", "count", "--trace-counts", "all"]
NEU_FIXED = ["--exploration", "neuron", "--drive", "rate"]
if kind == "synapse":
    arm = dict(interval=100.0, lr=0.002, threshold=0.6, minimum_potential=-0.2, hidden_neurons=0.0, temperature=2.0,
               synapse_hazard=0.01, drive=drive, seed=1)
    fixed = SYN_FIXED
else:
    arm = dict(interval=100.0, lr=0.002, delta=0.3125, threshold=0.6, minimum_potential=-0.2, hidden_neurons=0.0,
               temperature=2.0, seed=1)
    fixed = NEU_FIXED
g, args = rs.grid_of("mnist", arm, "hazard", True, None, None, fixed)
offset = 0
if source != "fresh":
    g, offset, _, _, _ = rs.resume_grid(g, f"{ROOT}/{source}")
patterns, labels = dataset_stream("mnist", 1)
g.use_input_stream(patterns, labels)
g.input_at = offset
synaptic = g.exploration == "synapse"
assert (kind == "synapse") == synaptic
theta_of = {}
outs = list(g.output_row())
ins = list(g.input_row())
oi = {id(n): k for k, n in enumerate(outs)}
ii = {id(n): k for k, n in enumerate(ins)}
O = len(outs)
theta = [n.threshold for n in outs]
floor = [n.minimum_potential for n in outs]
hop = Neuron.hop
rates_ckpt = [n.rate for n in outs]
print(f"{label}: epoch offset {offset}, tau {Neuron.tau}, hop {hop}, drive {g.drive}, exploration {g.exploration}, "
      f"output theta {min(theta):.4g}-{max(theta):.4g}, floor {min(floor):.4g}-{max(floor):.4g}, "
      f"fan-in {min(len(n.incoming) for n in outs)}-{max(len(n.incoming) for n in outs)}, "
      f"escape_scale {g.escape_scale:.5g}, synapse_scale(in0) {ins[0].synapse_scale:.5g}, suppress {suppress}", flush=True)

if suppress:
    assert synaptic
    original = g._decide_synapses
    g.decider = lambda: (lambda wave: (original(wave), [])[1])

# ---- state carried across waves and epochs
Vcur = [n.potential for n in outs]
ven_since = [0.0] * O  # ventured charge integrated since the last reset (spike or floor clamp)
rel_since = [0.0] * O
last_ven_t = [-1e18] * O
state = {"epoch": None}
records = []  # one per epoch
SAMPLES = int(round(g.interval))  # V every 1 ms


def new_epoch():
    e = {
        "label": g.input_label, "start": g.time, "bits": list(g.input_pattern),
        "rel_E": [0.0] * O, "rel_I": [0.0] * O, "ven_E": [0.0] * O, "ven_I": [0.0] * O,
        "n_rel": [0] * O, "n_ven": [0] * O, "drop_rel_w": [0.0] * O, "drop_ven_w": [0.0] * O,
        "n_drop_rel": [0] * O, "n_drop_ven": [0] * O,
        "clamps": [0] * O, "clamp_loss": [0.0] * O,
        "spikes": [0] * O, "sp_ven_same_wave": [0] * O, "sp_ven_within_hop": [0] * O,
        "sp_pivotal_wave": [0] * O, "sp_needs_ven_since": [0] * O, "sp_ven_since_sum": [0.0] * O,
        "sp_V": [],  # V at the fire, over theta, every spike (for the neuron rule's sub-threshold spikes)
        "reads": [0] * O, "read_u": [], "last_read": [0] * O,
        "V": [[None] * SAMPLES for _ in range(O)], "k": 0,
        "in_ven_bit1": 0, "in_ven_bit0": 0, "in_ven_clock": 0,
        "waves": 0, "waves_ven": 0,
    }
    return e


def fill(e, t):
    """Samples of V strictly before t take the value before this wave."""
    k = e["k"]
    start = e["start"]
    while k < SAMPLES and start + k < t:
        for j in range(O):
            e["V"][j][k] = Vcur[j]
        k += 1
    e["k"] = k


original_on_wave = g._on_wave


def on_wave(wave):
    original_on_wave(wave)
    e = state["epoch"]
    if e is None or state.get("epoch_no") != g.epoch:
        e = new_epoch(); state["epoch"] = e; state["epoch_no"] = g.epoch; records.append(e)
    t = wave.time
    fill(e, t)
    e["waves"] += 1
    ventured = set(wave.ventured)
    if ventured:
        e["waves_ven"] += 1
    wave_sum = [0.0] * O
    wave_ven = [0.0] * O
    touched = set()
    bits = e["bits"]
    for k, c in enumerate(wave.delivered):
        j = oi.get(id(c.target))
        if j is None:
            continue
        v = k in ventured
        w = c.weight
        if v:
            s = ii.get(id(c.source))
            if s is not None:
                if s < g.clock:
                    e["in_ven_clock"] += 1
                elif bits[s]:
                    e["in_ven_bit1"] += 1
                else:
                    e["in_ven_bit0"] += 1
        if c.last_signal == t:  # integrated
            touched.add(j)
            wave_sum[j] += w
            if v:
                wave_ven[j] += w
                e["n_ven"][j] += 1
                if w >= 0: e["ven_E"][j] += w
                else: e["ven_I"][j] += w
                last_ven_t[j] = t
            else:
                e["n_rel"][j] += 1
                if w >= 0: e["rel_E"][j] += w
                else: e["rel_I"][j] += w
        else:
            if v:
                e["n_drop_ven"][j] += 1; e["drop_ven_w"][j] += w
            else:
                e["n_drop_rel"][j] += 1; e["drop_rel_w"][j] += w
    fired = {oi[id(n)] for n in wave.fired if id(n) in oi}
    for j in touched | fired:
        n = outs[j]
        raw = Vcur[j] + wave_sum[j]
        ven_since[j] += wave_ven[j]
        rel_since[j] += wave_sum[j] - wave_ven[j]
        if j in fired:
            e["spikes"][j] += 1
            e["sp_V"].append(raw / theta[j])
            if wave_ven[j] != 0.0:
                e["sp_ven_same_wave"][j] += 1
            if last_ven_t[j] >= t - hop - 1e-9:
                e["sp_ven_within_hop"][j] += 1
            if raw - wave_ven[j] < theta[j]:
                e["sp_pivotal_wave"][j] += 1
            if raw - ven_since[j] < theta[j]:
                e["sp_needs_ven_since"][j] += 1
            e["sp_ven_since_sum"][j] += ven_since[j]
            ven_since[j] = rel_since[j] = 0.0
            if abs(n.potential) > 1e-12:
                state["mismatch"] = state.get("mismatch", 0) + 1
        else:
            expect = raw
            if raw < floor[j]:
                e["clamps"][j] += 1
                e["clamp_loss"][j] += floor[j] - raw
                expect = floor[j]
                ven_since[j] = rel_since[j] = 0.0
            if abs(n.potential - expect) > 1e-9:
                state["mismatch"] = state.get("mismatch", 0) + 1
        Vcur[j] = n.potential
    if synaptic:
        for j, n in enumerate(outs):
            r = n.read_count
            if r != e["last_read"][j]:
                u = min(max(n.potential, 0.0), theta[j]) / theta[j]
                for _ in range(r - e["last_read"][j]):
                    e["read_u"].append(u)
                e["last_read"][j] = r


g._on_wave = on_wave
rng = random.Random(7)
t0 = time.time()
in_spikes_bit1, in_spikes_bit0, n_bit1, n_bit0 = 0, 0, 0, 0
for epoch in range(epochs):
    run_epoch(g, verbose=False, rng=rng)
    e = records[-1]
    assert e["start"] == g.time
    fill(e, g.time + g.interval + 1.0)
    e["reads"] = [n.read_count for n in outs] if synaptic else [0] * O
    e["spikes_check"] = [n.epoch_spikes for n in outs]
    for s, n in enumerate(ins):
        if s < g.clock:
            continue
        if e["bits"][s]:
            in_spikes_bit1 += n.epoch_spikes; n_bit1 += 1
        else:
            in_spikes_bit0 += n.epoch_spikes; n_bit0 += 1
    e["in_clock_spikes"] = [ins[s].epoch_spikes for s in range(g.clock)]
print(f"{label}: {epochs} epochs in {time.time() - t0:.0f} s, bookkeeping mismatches {state.get('mismatch', 0)}", flush=True)

# ---- per-output summaries over the epochs
import statistics as st


def per_out(key):
    return [sum(e[key][j] for e in records) / len(records) for j in range(O)]


def per_edge_count(src_bit):
    pass


summary = {"label": label, "kind": kind, "drive": g.drive, "source": source, "epochs": epochs, "suppress": suppress,
           "offset": offset, "theta": theta, "floor": floor, "rates_ckpt": rates_ckpt,
           "spikes_check_ok": all(e["spikes_check"] == e["spikes"] for e in records),
           "mismatch": state.get("mismatch", 0)}
for key in ("rel_E", "rel_I", "ven_E", "ven_I", "n_rel", "n_ven", "n_drop_rel", "n_drop_ven", "drop_rel_w", "drop_ven_w",
            "clamps", "clamp_loss", "spikes", "sp_ven_same_wave", "sp_ven_within_hop", "sp_pivotal_wave",
            "sp_needs_ven_since", "sp_ven_since_sum", "reads"):
    summary[key] = per_out(key)
summary["fired_frac"] = [sum(1 for e in records if e["spikes"][j] > 0) / len(records) for j in range(O)]
summary["counted_frac"] = [sum(1 for e in records if e["spikes"][j] + e["reads"][j] > 0) / len(records) for j in range(O)]
V = [[v for e in records for v in e["V"][j]] for j in range(O)]
summary["V_mean"] = [st.fmean(v) for v in V]
summary["V_pos_frac"] = [sum(1 for x in v if x > 0) / len(v) for v in V]
summary["V_floor_frac"] = [sum(1 for x in v if x <= floor[j] + 1e-12) / len(v) for j, v in enumerate(V)]
summary["u_mean"] = [st.fmean(min(max(x, 0.0), theta[j]) / theta[j] for x in v) for j, v in enumerate(V)]
# the time course: mean V/theta per 10 ms bin, per output
bins = SAMPLES // 10
summary["V_course"] = [[st.fmean(e["V"][j][k] / theta[j] for e in records for k in range(b * 10, (b + 1) * 10)) for b in range(bins)]
                       for j in range(O)]
summary["V_epoch_traces_first3"] = [[[round(e["V"][j][k] / theta[j], 3) for k in range(0, SAMPLES, 5)] for j in range(O)]
                                    for e in records[:3]]
summary["read_u"] = [u for e in records for u in e["read_u"]]
summary["sp_V"] = [x for e in records for x in e["sp_V"]]
summary["in_spikes_per_bit1"] = in_spikes_bit1 / max(1, n_bit1)
summary["in_spikes_per_bit0"] = in_spikes_bit0 / max(1, n_bit0)
summary["in_clock_spikes"] = st.fmean(x for e in records for x in e["in_clock_spikes"])
summary["in_ven"] = {k: st.fmean(e[k] for e in records) for k in ("in_ven_bit1", "in_ven_bit0", "in_ven_clock")}
summary["waves"] = st.fmean(e["waves"] for e in records)
summary["labels"] = [e["label"] for e in records]
# per-epoch per-output net drives, for the across-pattern spread
summary["per_epoch"] = [{"label": e["label"], "rel": [e["rel_E"][j] + e["rel_I"][j] for j in range(O)],
                         "ven": [e["ven_E"][j] + e["ven_I"][j] for j in range(O)], "spikes": e["spikes"],
                         "reads": e["reads"]} for e in records]
# the synapses from inputs, by the bit, for the per-synapse ventured rate
edges = [(ii[id(c.source)], oi[id(c.target)], c.weight) for n in ins for c in n.outgoing if id(c.target) in oi]
summary["edges"] = edges
bits_epochs = [e["bits"] for e in records]
n1 = st.fmean(sum(1 for (s, j, w) in edges if s >= g.clock and b[s]) for b in bits_epochs)
n0 = st.fmean(sum(1 for (s, j, w) in edges if s >= g.clock and not b[s]) for b in bits_epochs)
nc = sum(1 for (s, j, w) in edges if s < g.clock)
summary["edge_counts"] = {"bit1": n1, "bit0": n0, "clock": nc}
json.dump(summary, open(f"{OUT}/{label}.json", "w"))
print(f"{label}: wrote {OUT}/{label}.json", flush=True)
