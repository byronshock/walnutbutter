"""§12.8 / plan §5.1: objects vs Rust with == on mnist's operating point under exploration at the synapse.

goo 455 (--hidden-neurons 0), interval 100, threshold 0.6, floor -0.2, hazard eligibility, TAU inf, count read,
evidence critic at the problem's rate; synapse, loglinear, h0 0.01, count scaling, full trace; rate and charged drives."""
import importlib.util, random, sys, time
from walnutbutter import fast
from walnutbutter.learning import Teacher
from walnutbutter.neuron import Neuron
from walnutbutter.problems import dataset_stream

spec = importlib.util.spec_from_file_location("rs", "docs/rust-sweep.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
for drive in ("rate", "charged"):
    for seed in (1, 2):
        t0 = time.time()
        patterns, labels = dataset_stream("mnist", seed)
        g, cli = rs.grid_of("mnist", {"hidden_neurons": 0.0, "seed": seed, "interval": 100.0, "threshold": 0.6,
                                      "minimum_potential": -0.2}, "hazard", True, None)
        Neuron.tau = float("inf")
        g.set_delta(0.0)
        g.set_exploration("synapse")
        g.drive = drive
        g.use_input_stream(patterns[:200], labels[:200])
        assert len(g.all_neurons()) == 455
        teacher = Teacher(g, seed=seed + 100, rule="reinforce", eligibility="hazard", target="label", critic="evidence", lr=cli.lr)
        parted = fast.compare(g, epochs=EPOCHS, teacher=teacher)
        print(f"{drive:8s} seed {seed}: lr {cli.lr} interval {g.interval} theta0 {g.all_neurons()[0].threshold:.4g} "
              f"-> {'AGREE' if parted == [] else 'PARTED ' + str(parted)[:300]}  ({time.time() - t0:.0f} s)", flush=True)
