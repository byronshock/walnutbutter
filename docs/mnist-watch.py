#!/usr/bin/env python3
"""Watch one arm of a mnist sweep run on, the estimator's correlation printed every thousand epochs, until Ctrl-C.

    .venv/bin/python docs/mnist-watch.py --seed 4            # sweep 1's hazard arm at seed 4, from the start, without end
    .venv/bin/python docs/mnist-watch.py --seed 4 --epochs 30000 --every 500
    .venv/bin/python docs/mnist-watch.py --wiring ff2 --seed 1 --interval 100 --every 100   # one fully connected goo, from the start
    .venv/bin/python docs/mnist-watch.py --resume runs/mnist-ff-interval/interval100-threshold0.6-hidden_neurons0-temperature2-seed1-network.json \\
        --seed 1 --interval 100 --output-coding population --population 5 --outputs 50   # on from that arm's saved network

From the start: a seed reruns bit for bit, so the arm is rebuilt as the
sweep built it (the same knobs through the same `grid_of`, the same stream,
the same exploration seed) and run from epoch 1; by the sweep's last epoch it
is the sweep's arm to the bit, and then it goes on. With `--resume` the arm
is continued from the network a sweep saved at its end (`docs/rust-sweep.py`
saves one per arm since the morning of September 16, 2026): the checkpoint's
state whole, the stream advanced to the epoch reached, the estimator still
measured from the first start's weights, the epochs counted on -- not to the
bit, the exploration stream starting afresh (seed + 1,000,000). The knobs
given must rebuild the arm's layout (`--interval`, `--output-coding`,
`--population`, `--outputs` for a layout the problem no longer defaults to).

Every `--every` epochs one line: the epoch, the reward and the fraction right
over the window, the cumulative correlation and sign agreement of the weight
change with the supervised direction (AUTHORITY.md §8), the window's
correlation, and the outputs' spikes a neuron. Every `--checkpoint-every`
epochs the network is written back and checkpointed under runs/watch/, and
once more at Ctrl-C, in the form `--resume` takes.
"""
import argparse, json, math, signal, statistics as st, sys, time
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--eligibility", choices=("hazard", "hebb", "count_hebb", "wrong_hebb", "perturb"), default="hazard")
    parser.add_argument("--lr", type=float, default=None, help="the problem's unless given")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hidden-neurons", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--floor-ratio", type=float, default=-4.0)
    parser.add_argument("--interval", type=float, default=None, help="the epoch's length in ms; the problem's unless given")
    parser.add_argument("--tau", type=float, default=None, help="the potential's leak in ms; the constant unless given")
    parser.add_argument("--output-coding", choices=("population", "complement"), default=None, help="the problem's unless given")
    parser.add_argument("--population", type=int, default=None, help="the problem's unless given")
    parser.add_argument("--outputs", type=int, default=None, help="the problem's unless given")
    parser.add_argument("--wiring", choices=("scaled", "ff2", "ff2-partial", "scaled-open", "zones-equal", "zones", "uniform"), default=None,
                        help="goo's wiring (§3.4): the command line's default, scaled, unless given; ff2 is fully connected feedforward")
    parser.add_argument("--projection", type=float, default=None, help="P for --wiring ff2-partial (and the three older wirings)")
    parser.add_argument("--resume", default=None, metavar="NETWORK.json", help="continue from a sweep's saved network (see above)")
    parser.add_argument("--isi-factor", choices=("on", "off"), default=None,
                        help="the ISI factor of §0.2: on for a fresh run; a resumed network keeps its checkpoint's setting unless given")
    parser.add_argument("--every", type=int, default=1000, help="epochs between lines")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="epochs between checkpoints; 0 for none")
    parser.add_argument("--epochs", type=int, default=10 ** 9, help="stop after this many more; the default is until Ctrl-C")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("rs", ROOT / "docs" / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    from walnutbutter import fast, mnist
    from walnutbutter.neuron import Neuron
    from walnutbutter.learning import class_evidence
    from walnutbutter.problems import dataset_stream

    arm = {"hidden_neurons": float(args.hidden_neurons), "seed": args.seed, "threshold": args.threshold, "temperature": args.temperature}
    parser.set_defaults(projection=None)
    for knob in ("lr", "interval", "tau", "projection"):
        if getattr(args, knob) is not None:
            arm[knob] = getattr(args, knob)
    fixed = []
    for flag in ("output_coding", "population", "outputs"):
        if getattr(args, flag) is not None:
            fixed += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
    if args.isi_factor == "off":
        fixed.append("--no-isi-factor")
    grid, cli = rs.grid_of("mnist", arm, args.eligibility, True, args.floor_ratio, args.wiring, tuple(fixed))
    patterns, labels = dataset_stream("mnist", args.seed)
    offset, reference, explore_seed, baseline = 0, None, args.seed, None
    if args.resume:
        grid, offset, reference, baseline = rs.resume_grid(grid, Path(args.resume), None if args.isi_factor is None else args.isi_factor == "on")
        grid.use_input_stream(patterns, labels)
        grid.input_at = offset
        patterns = labels = None
        explore_seed += 1_000_000
    edges = [c for n in grid.all_neurons() for c in n.outgoing]  # the engine's order
    d, mask = mnist.supervised_direction(grid)(edges)
    d_masked = d[mask]
    w0 = np.array(reference if reference is not None else [c.weight for c in edges])
    state = {"w_prev": np.array([c.weight for c in edges]), "rewards": [], "hits": [], "outs": [], "started": time.perf_counter(),
             "epoch": 0, "engine": None, "book": None}
    out_dir = ROOT / "runs" / "watch"
    print(f"{grid!r}; {args.eligibility} eligibility, LR {cli.lr:g}, T {args.temperature:g}, threshold {args.threshold:g}, "
          f"floor {cli.minimum_potential:g}, delta {cli.delta:g} (hazard x {grid.escape_scale:.3f}), tau {cli.tau:g} ms, epoch {grid.interval:g} ms, "
          f"ISI factor {'on at ' + format(Neuron.target_isi, 'g') + ' ms' if Neuron.isi_factor else 'off'}, "
          f"{grid.output_coding} outputs, seed {args.seed}; {int(mask.sum())} input-to-output synapses"
          f"{f'; resumed at epoch {offset:,} from {args.resume}' if args.resume else ''}; a line every {args.every:,} epochs, Ctrl-C to stop",
          flush=True)
    print(f"{'epoch':>8} {'reward':>8} {'right':>6} {'corr cum':>9} {'sign':>6} {'corr win':>9} {'out/neuron':>10} {'epochs/s':>8}", flush=True)

    def corr(x):
        return float("nan") if x.std() == 0.0 or d_masked.std() == 0.0 else float(np.corrcoef(x, d_masked)[0, 1])

    def save(engine, book, epoch):
        if engine is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.eligibility}-seed{args.seed}-epoch{offset + epoch}-network.json"
        rs._save_network(engine, grid, {"thresholds": book.thresholds, "expected_counts": book.expected, "rates": book.rates,
                                        "expectations": [None if e != e else e for e in engine.expectations()],  # p_hat_j (§6.7)
                                        "decisions": list(engine.decision_counts())}, path)
        print(f"checkpoint {path.relative_to(ROOT)} (the network whole; --resume takes it, not to the bit)", flush=True)

    def probe(epoch, engine, grid, out, book):
        state["engine"], state["book"], state["epoch"] = engine, book, epoch
        counts = engine.epoch_spike_counts()
        groups = class_evidence([int(counts[i]) for i in out], grid.population, grid.output_coding)
        y = grid.input_label
        state["rewards"].append(fast._reward(engine, grid, out, "evidence", None))
        state["hits"].append(1.0 if all(groups[y] > g for c, g in enumerate(groups) if c != y) else 0.0)
        state["outs"].append(st.mean(counts[i] for i in out))
        if epoch % args.every == 0:
            w = np.array(engine.weights())
            cum, win = (w - w0)[mask], (w - state["w_prev"])[mask]
            state["w_prev"] = w
            rate = args.every / (time.perf_counter() - state["started"]); state["started"] = time.perf_counter()
            print(f"{offset + epoch:>8,} {st.mean(state['rewards']):>8.3f} {st.mean(state['hits']):>6.3f} {corr(cum):>+9.3f} "
                  f"{float((np.sign(cum) == np.sign(d_masked)).mean()):>6.3f} {corr(win):>+9.3f} {st.mean(state['outs']):>10.2f} {rate:>8.0f}", flush=True)
            state["rewards"].clear(); state["hits"].clear(); state["outs"].clear()
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            save(engine, book, epoch)

    try:
        fast.train(grid, args.epochs, lr=cli.lr, target="label", trace_every=0, patterns=patterns, labels=labels,
                   eligibility=args.eligibility, seed=explore_seed, homeostasis=0.0, unstick=0.0, critic="evidence",
                   probe=probe, probe_every=1, reference_weights=reference, epoch_offset=offset)
    except KeyboardInterrupt:
        print(f"\nstopped at epoch {offset + state['epoch']:,}", flush=True)
        save(state["engine"], state["book"], state["epoch"])
        return 130
    save(state["engine"], state["book"], state["epoch"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
