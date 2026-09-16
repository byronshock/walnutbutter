#!/usr/bin/env python3
"""Watch one arm of a mnist sweep run on past its end, the estimator's correlation printed every thousand epochs, until Ctrl-C.

    .venv/bin/python docs/mnist-watch.py --seed 4            # sweep 1's hazard arm at seed 4, from the start, without end
    .venv/bin/python docs/mnist-watch.py --seed 4 --epochs 30000 --every 500

The sweeps run on the Rust driver, which records traces but saves no
checkpoint, so there is nothing to resume; but a seed reruns bit for bit, so
the arm is rebuilt as the sweep built it (the same knobs through the same
`grid_of`, the same stream, the same exploration seed) and run from epoch 1:
by the sweep's last epoch it is the sweep's arm to the bit, and then it goes
on. Every `--every` epochs one line: the epoch, the reward and the fraction
right over the window, the cumulative correlation and sign agreement of the
weight change with the supervised direction (AUTHORITY.md §8), the window's
correlation, and the outputs' spikes a neuron. Every `--checkpoint-every`
epochs the weights and thresholds are written back to the mesh and a
checkpoint saved under runs/watch/, and one more at Ctrl-C; a run restored
from it continues with those weights but not to the bit (the stream's
position, the exploration stream's state and the baseline are not in it).
"""
import argparse, json, math, signal, statistics as st, sys, time
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--eligibility", choices=("hazard", "hebb", "wrong_hebb", "perturb"), default="hazard")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hidden-neurons", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--floor-ratio", type=float, default=-4.0)
    parser.add_argument("--every", type=int, default=1000, help="epochs between lines")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="epochs between checkpoints; 0 for none")
    parser.add_argument("--epochs", type=int, default=10 ** 9, help="stop after this many; the default is until Ctrl-C")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("rs", ROOT / "docs" / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    from walnutbutter import fast, mnist
    from walnutbutter.persistence import checkpoint
    from walnutbutter.problems import dataset_stream

    arm = {"hidden_neurons": float(args.hidden_neurons), "seed": args.seed, "threshold": args.threshold,
           "temperature": args.temperature, "lr": args.lr}
    grid, cli = rs.grid_of("mnist", arm, args.eligibility, True, args.floor_ratio)
    patterns, labels = dataset_stream("mnist", args.seed)
    edges = [c for n in grid.all_neurons() for c in n.outgoing]  # the engine's order
    d, mask = mnist.supervised_direction(grid)(edges)
    d_masked = d[mask]
    w0 = np.array([c.weight for c in edges])
    state = {"w_prev": w0.copy(), "rewards": [], "hits": [], "outs": [], "started": time.perf_counter(), "epoch": 0, "engine": None, "book": None}
    out_dir = ROOT / "runs" / "watch"
    print(f"{grid!r}; {args.eligibility} eligibility, LR {args.lr:g}, T {args.temperature:g}, threshold {args.threshold:g}, "
          f"floor {cli.minimum_potential:g}, delta {cli.delta:g}, seed {args.seed}; {int(mask.sum())} input-to-output synapses; "
          f"a line every {args.every:,} epochs, Ctrl-C to stop", flush=True)
    print(f"{'epoch':>8} {'reward':>8} {'right':>6} {'corr cum':>9} {'sign':>6} {'corr win':>9} {'out/neuron':>10} {'epochs/s':>8}", flush=True)

    def corr(x):
        return float("nan") if x.std() == 0.0 or d_masked.std() == 0.0 else float(np.corrcoef(x, d_masked)[0, 1])

    def save(engine, book, epoch):
        if engine is None:
            return
        for c, w in zip(edges, engine.weights()):
            c.weight = w
        for n, theta in zip(grid.all_neurons(), book.thresholds):
            n.threshold = theta
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.eligibility}-seed{args.seed}-epoch{epoch}.json"
        checkpoint(grid, path)
        print(f"checkpoint {path.relative_to(ROOT)} (weights and thresholds; not a resume to the bit)", flush=True)

    def probe(epoch, engine, grid, out, book):
        state["engine"], state["book"], state["epoch"] = engine, book, epoch
        counts = engine.epoch_spike_counts()
        groups = [sum(counts[i] for i in out[c * 5:(c + 1) * 5]) for c in range(10)]
        y = grid.input_label
        state["rewards"].append(fast._reward(engine, grid, out, "evidence", None))
        state["hits"].append(1.0 if all(groups[y] > g for c, g in enumerate(groups) if c != y) else 0.0)
        state["outs"].append(st.mean(counts[i] for i in out))
        if epoch % args.every == 0:
            w = np.array(engine.weights())
            cum, win = (w - w0)[mask], (w - state["w_prev"])[mask]
            state["w_prev"] = w
            rate = args.every / (time.perf_counter() - state["started"]); state["started"] = time.perf_counter()
            print(f"{epoch:>8,} {st.mean(state['rewards']):>8.3f} {st.mean(state['hits']):>6.3f} {corr(cum):>+9.3f} "
                  f"{float((np.sign(cum) == np.sign(d_masked)).mean()):>6.3f} {corr(win):>+9.3f} {st.mean(state['outs']):>10.2f} {rate:>8.0f}", flush=True)
            state["rewards"].clear(); state["hits"].clear(); state["outs"].clear()
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            save(engine, book, epoch)

    try:
        fast.train(grid, args.epochs, lr=args.lr, target="label", trace_every=0, patterns=patterns, labels=labels,
                   eligibility=args.eligibility, seed=args.seed, homeostasis=0.0, unstick=0.0, critic="evidence",
                   probe=probe, probe_every=1)
    except KeyboardInterrupt:
        print(f"\nstopped at epoch {state['epoch']:,}", flush=True)
        save(state["engine"], state["book"], state["epoch"])
        return 130
    save(state["engine"], state["book"], state["epoch"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
