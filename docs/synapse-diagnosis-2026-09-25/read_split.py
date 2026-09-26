"""What the count read counts under exploration at the synapse at mnist's operating point: each output's own spikes
against its read synapse's escapes, per epoch, and where the outputs' potentials sit (u = V/theta) at the read."""
import importlib.util, random, statistics as st
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.problems import dataset_stream
spec = importlib.util.spec_from_file_location("rs", "docs/rust-sweep.py"); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
for drive in ("rate", "charged"):
    patterns, labels = dataset_stream("mnist", 1)
    g, cli = rs.grid_of("mnist", {"hidden_neurons": 0.0, "seed": 1, "interval": 100.0, "threshold": 0.6, "minimum_potential": -0.2}, "hazard", True, None)
    Neuron.tau = float("inf"); g.set_delta(0.0); g.set_exploration("synapse"); g.drive = drive
    g.use_input_stream(patterns[:50], labels[:50])
    rng = random.Random(7)
    outs = list(dict.fromkeys(g.output_row()))
    spikes, reads, us, active = [], [], [], []
    for epoch in range(12):
        run_epoch(g, verbose=False, rng=rng)
        spikes.append(sum(n.epoch_spikes for n in outs)); reads.append(sum(n.read_count for n in outs))
        us += [min(max(n.potential_at(g.time), 0.0), n.threshold) / n.threshold for n in outs]
        active.append(sum(1 for n in outs if n.epoch_spikes + n.read_count > 0))
    ins = [n for n in g.all_neurons() if n not in outs]
    print(f"{drive:8s} per epoch over 60 outputs: spikes {st.mean(spikes):6.1f}  read escapes {st.mean(reads):6.1f}  "
          f"outputs counting >0 {st.mean(active):4.1f}/60 | u at the read: mean {st.mean(us):.3f}, "
          f"share above 0.9 {sum(u > 0.9 for u in us)/len(us):.2f}, share at 0 {sum(u == 0 for u in us)/len(us):.2f} | "
          f"output thresholds {min(n.threshold for n in outs):.3f}-{max(n.threshold for n in outs):.3f}, fan-in {min(len(n.incoming) for n in outs)}-{max(len(n.incoming) for n in outs)}")
