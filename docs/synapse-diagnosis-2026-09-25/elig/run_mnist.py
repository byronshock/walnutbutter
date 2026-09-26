"""mnist's point as the critic investigator's probe_arm.py built it (docs/rust-sweep.py's own grid_of: interval 100,
threshold 0.6, floor -0.2, hidden 0, temperature 2, hazard eligibility, lr 0.002, exploration synapse, h0 0.01,
loglinear, count scaling, trace all, rate drive), a Teacher on the object engine with the evidence critic, run under
fast.compare (objects against Rust with == after every epoch), _decide_synapses instrumented.
usage: run_mnist.py SEED EPOCHS"""
import importlib.util, json, sys, time
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/elig")
from instrument import Instrument
from walnutbutter import fast
from walnutbutter.learning import Teacher
from walnutbutter.problems import PROBLEMS, dataset_stream

ROOT = "/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure"
OUT = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/elig"
spec = importlib.util.spec_from_file_location("rs", ROOT + "/docs/rust-sweep.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
seed, epochs = int(sys.argv[1]), int(sys.argv[2])
words = ["--problem", "mnist", "--interval", "100", "--threshold", "0.6", "--minimum-potential", "-0.2",
         "--hidden-neurons", "0", "--temperature", "2", "--eligibility", "hazard", "--seed", str(seed),
         "--epochs", str(epochs), "--lr", "0.002", "--name", "diag-elig-unused",
         "--exploration", "synapse", "--synapse-hazard", "0.01", "--hazard-family", "loglinear",
         "--synapse-scaling", "count", "--trace-counts", "all", "--drive", "rate"]
sys.argv = ["rust-sweep.py"] + words
args = rs.parse()
swept, arms = rs.grid_and_arms(args)
assert len(arms) == 1, arms
arm = arms[0]
grid, cli = rs.grid_of("mnist", arm, args.eligibility[0], args.scale, args.floor_ratio, args.wiring, tuple(rs.fixed_words(args)))
patterns, labels = dataset_stream(PROBLEMS["mnist"].data, seed)
grid.use_input_stream(patterns, labels)
ins = Instrument(grid)
teacher = Teacher(grid, seed=seed, rule="reinforce", eligibility="hazard", target="label", critic=cli.critic, lr=cli.lr,
                  homeostasis=cli.homeostasis, target_rate=cli.target_rate, unstick=cli.unstick,
                  unstick_target=cli.unstick_target)
print("arm", rs.arm_name(arm), "critic", cli.critic, "lr", cli.lr, "homeostasis", cli.homeostasis, "unstick", cli.unstick,
      "interval", grid.interval, "synapses", len(ins.conns), flush=True)
t0 = time.time()
parted = fast.compare(grid, epochs=epochs, teacher=teacher)
ins.finish()
W = ins.wave_array()
rows = np.array(ins.rows)
tag = f"mnist-seed{seed}-{epochs}"
np.savez_compressed(f"{OUT}/{tag}.npz", waves=W, rows=rows, xb=ins.xb, mb=ins.mb, cal=ins.cal, cal_u=ins.cal_u)
print(json.dumps({"tag": tag, "parted": parted[:3], "bad": ins.bad[:5], "n_bad": len(ins.bad), "epochs": len(rows),
                  "waves": len(W), "seconds": round(time.time() - t0)}), flush=True)
