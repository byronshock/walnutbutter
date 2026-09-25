#!/usr/bin/env python3
"""Probe four arms of the mnist-evidence sweep over their first epochs: does the reward's early rise mean learning?

Every 100 epochs, over that window: the evidence reward, the class critic's
fraction right, the mean output spike count, the fraction of outputs at the
refractory ceiling (7 spikes an epoch), the interior's mean count, and the
spread of the class sums. Written to runs/mnist-evidence-diag/<arm>.json.
"""
import json, statistics as st, sys
from multiprocessing import Pool
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500


def run(arm):
    T, lr = arm
    spec = importlib.util.spec_from_file_location("rs", ROOT / "docs" / "rust-sweep.py"); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    from walnutbutter import fast
    from walnutbutter.problems import dataset_stream
    grid, args = rs.grid_of("mnist", {"goo": 644.0, "seed": 1, "threshold": 0.6, "temperature": T, "lr": lr}, "hazard", True, -4.0)
    patterns, labels = dataset_stream("mnist", 1)
    window = {"reward": [], "right": [], "out": [], "ceiling": [], "interior": [], "spread": []}
    rows = []
    pop, out_n = grid.population, grid.outputs

    def probe(epoch, engine, grid, out, book):
        counts = fast._counts(engine)  # spikes, plus under exploration at the synapse the read synapses' escapes (§5.10)
        outs = [counts[i] for i in out]
        groups = [sum(outs[c * pop:(c + 1) * pop]) for c in range(len(outs) // pop)]
        label = grid.input_label
        window["reward"].append(fast._reward(engine, grid, out, "evidence", None))
        window["right"].append(1.0 if all(groups[label] > g for c, g in enumerate(groups) if c != label) else 0.0)
        window["out"].append(st.mean(outs))
        window["ceiling"].append(sum(1 for c in outs if c >= 7) / len(outs))
        interior = [counts[i] for i in range(grid.across, grid.count - out_n)]
        window["interior"].append(st.mean(interior))
        window["spread"].append(st.pstdev(groups))
        if epoch % 100 == 0:
            rows.append({"epoch": epoch, **{k: st.mean(v) for k, v in window.items()}})
            for v in window.values():
                v.clear()

    fast.train(grid, EPOCHS, lr=lr, target="label", trace_every=0, patterns=patterns, labels=labels, eligibility="hazard",
               seed=1, homeostasis=0.0, unstick=0.0, critic="evidence", probe=probe, probe_every=1)
    out = ROOT / "runs" / "mnist-evidence-diag"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"T{T:g}-lr{lr:g}.json").write_text(json.dumps(rows))
    return arm


if __name__ == "__main__":
    arms = [(1.0, 0.001), (1.0, 0.03), (2.0, 0.006), (4.0, 0.01)]
    with Pool(len(arms)) as pool:
        for arm in pool.imap_unordered(run, arms):
            print("done", arm, flush=True)
