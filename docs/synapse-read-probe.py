#!/usr/bin/env python3
"""Continue each arm of a finished synapse sweep from its checkpoint and count what the read and the drive do.

For every arm of the named sweep, the network runs on from `runs/<sweep>/<arm>-network.json`
through the sweep driver's own resume (docs/rust-sweep.py, resume_grid), under the settings
the checkpoint was saved with -- the same run, continued, learning as it learned -- for
EPOCHS more epochs. Nothing is written back to the sweep: the continuation is thrown away
and only its counts are kept, in runs/synapse-read-probe/<sweep>/<arm>.json.

Every epoch, after the run and before nothing else changes:

- each output's count as the read takes it (§5.10, §7.9): its own spikes plus its read
  synapse's ventured transmissions, kept apart, and grouped by what the label code (11.10)
  wants of the output -- the label's fire-if-one population and every other class's
  fire-if-zero population want a 1, the rest a 0;
- the input zone's spikes, the driven ones apart from the rest, beside the drive's arrivals
  (rate drive) or deliveries (charged drive) that epoch, which input_schedule drew;
- whether the label's class strictly out-spiked every other (§9.7), to set beside the
  sweep's own last-tenth fraction as a check that the continuation is the run it continues.

Beside the summary, runs/synapse-read-probe/<sweep>/<arm>.npz keeps the raw material for a
matched filter at the outputs: every epoch's label, each output's own spikes in BIN_MS bins
across the epoch (from the waves the engine's run returns, which fast.train discards), and
each output's read-synapse escapes that epoch (counted at their decisions, untimed).

It was run on September 26, 2026 on the code at 301978a, where the arms were paid by the evidence critic, as they had
been; that is what "the same run, continued" meant then, and runs/synapse-read-probe/ holds what it read. Under the code
since, the evidence critic is gone (AUTHORITY.md §9.4) and a continuation is paid by the critic the arguments name --
mnist's matched filter, starting empty since those checkpoints carry none -- and is the network continued under another
critic, not the same run.

Usage: docs/synapse-read-probe.py SWEEP EPOCHS -- <the sweep's own rust-sweep.py arguments>
"""
import json
import subprocess
import sys
import importlib.util
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = Path(subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=HERE,
                           capture_output=True, text=True, check=True).stdout.strip()).parent  # runs/ lives in the main checkout
BIN_MS = 5.0  # the time bins an output's spikes are kept in: one refractory period each


class Keeping:
    """The engine, unchanged, except that it keeps what its last run returned -- (time, fired indices) per wave."""

    def __init__(self, engine):
        self._engine, self.waves = engine, []

    def run(self, until):
        self.waves = self._engine.run(until)
        return self.waves

    def __getattr__(self, name):
        return getattr(self._engine, name)


def driver():
    spec = importlib.util.spec_from_file_location("rs", HERE / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    return rs


def run(job):
    label, sweep, epochs, arm, fixed, scale, floor_ratio, wiring, eligibility = job
    rs = driver()
    from walnutbutter import fast
    from walnutbutter.problems import dataset_stream

    name = rs.arm_name(arm)
    grid, args = rs.grid_of("mnist", arm, eligibility, scale, floor_ratio, wiring, tuple(fixed))
    fresh = sweep is None  # a control: the arm's network as its seed builds it, never learning (lr 0)
    offset, baseline, explore_state = 0, None, None
    if not fresh:
        source = MAIN / "runs" / sweep / f"{name}-network.json"
        grid, offset, _reference, baseline, explore_state = rs.resume_grid(grid, source)
    patterns, labels = dataset_stream("mnist", int(arm["seed"]))
    grid.use_input_stream(patterns, labels)  # §12.11: the stream continues where the checkpoint left it
    grid.input_at = offset

    drawn = []  # the drive's arrivals (or deliveries) each epoch, as input_schedule hands them to the engine
    schedule = grid.input_schedule

    def counted_schedule():
        events = schedule()
        drawn.append(len(events))
        return events
    grid.input_schedule = counted_schedule
    build = fast.build

    def keeping_build(*a, **k):
        engine, neurons, index = build(*a, **k)
        return Keeping(engine), neurons, index
    fast.build = keeping_build
    bins = int(round(grid.interval / BIN_MS))
    kept_labels, kept_bins, kept_reads = [], [], []

    groups = ("label+", "label-", "other+", "other-")  # the label's fire-if-one, its fire-if-zero, the other classes' two
    want = {"label+": 1, "label-": 0, "other+": 0, "other-": 1}
    tally = {g: {"neuron_epochs": 0, "spikes": 0, "reads": 0} for g in groups}
    histogram = {1: Counter(), 0: Counter()}  # the count read per output per epoch, by what the label code wants
    inputs = {"spikes_driven": 0, "spikes_undriven": 0, "driven_neurons": 0, "drawn": 0}
    per_epoch_input_spikes = []
    right = 0
    population, across = grid.population, grid.across

    def probe(epoch, engine, net, out, book):
        nonlocal right
        spikes, reads = engine.epoch_spike_counts(), engine.read_counts()
        label = net.input_label
        start = net.horizon - net.interval
        where = {i: k for k, i in enumerate(out)}
        timed = np.zeros((len(out), bins), dtype=np.uint8)
        for when, fired in engine.waves:
            b = min(int((when - start) // BIN_MS), bins - 1)
            for i in fired:
                if i in where:
                    timed[where[i], b] += 1
        if (timed.sum(axis=1) != [spikes[i] for i in out]).any():
            raise RuntimeError(f"epoch {epoch}: the waves' output spikes are not the engine's counts")
        kept_labels.append(label)
        kept_bins.append(timed)
        kept_reads.append([reads[i] for i in out])
        classes = len(out) // (2 * population)
        for k, i in enumerate(out):
            half, cls = divmod(k, classes * population)
            cls //= population
            group = ("label" if cls == label else "other") + ("+" if half == 0 else "-")
            t = tally[group]
            t["neuron_epochs"] += 1
            t["spikes"] += spikes[i]
            t["reads"] += reads[i]
            histogram[want[group]][min(spikes[i] + reads[i], 20)] += 1
        forced = engine.forced_flags()
        driven = sum(spikes[i] for i in range(across) if forced[i])
        inputs["spikes_driven"] += driven
        inputs["spikes_undriven"] += sum(spikes[i] for i in range(across) if not forced[i])
        inputs["driven_neurons"] += sum(1 for i in range(across) if forced[i])
        inputs["drawn"] += drawn[-1]
        per_epoch_input_spikes.append(sum(spikes[i] for i in range(across)))
        evidence = [0] * classes
        for k, i in enumerate(out):
            half, cls = divmod(k, classes * population)
            evidence[cls // population] += (spikes[i] + reads[i]) * (1 if half == 0 else -1)
        right += all(evidence[label] > e for c, e in enumerate(evidence) if c != label)

    lr = 0.0 if fresh else args.lr
    fast.train(grid, epochs, lr=lr, target=args.target, trace_every=0, patterns=None, labels=None,
               eligibility=args.eligibility, seed=int(arm["seed"]), explore_state=explore_state,
               pending_events=getattr(grid, "engine_pending", None), homeostasis=args.homeostasis,
               target_rate=args.target_rate, unstick=args.unstick, unstick_target=args.unstick_target,
               critic=args.critic, probe=probe, probe_every=1, epoch_offset=offset, baseline=baseline)

    per = sorted(per_epoch_input_spikes)
    result = {
        "arm": name, "sweep": sweep, "fresh": fresh, "from_epoch": offset, "epochs": epochs,
        "interval_ms": grid.interval, "drive": grid.drive, "drive_steps": grid.drive_steps,
        "input_rate_per_ms": grid.input_rate, "lr": lr, "seed": int(arm["seed"]),
        "right": right / epochs, "groups": tally,
        "histogram": {str(k): dict(sorted(v.items())) for k, v in histogram.items()},
        "inputs": {**inputs, "epochs": epochs,
                   "input_spikes_per_epoch_min_median_max": [per[0], per[len(per) // 2], per[-1]]},
    }
    out_dir = MAIN / "runs" / "synapse-read-probe" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(out_dir / f"{name}.npz", labels=np.array(kept_labels, dtype=np.uint8),
                        spikes=np.array(kept_bins, dtype=np.uint8), reads=np.array(kept_reads, dtype=np.uint16),
                        bin_ms=BIN_MS)
    return name


def jobs_of(label, sweep, epochs, words):
    """The jobs for one sweep's arms, as its own rust-sweep.py arguments name them: continuations of its checkpoints,
    or with `sweep` None, fresh networks of the same arms that never learn -- each seed's untrained read."""
    rs = driver()
    argv = sys.argv
    sys.argv = ["rust-sweep.py"] + list(words)
    try:
        args = rs.parse()
    finally:
        sys.argv = argv
    _swept, arms = rs.grid_and_arms(args)
    fixed = rs.fixed_words(args)
    return [(label, sweep, epochs, arm, fixed, args.scale, args.floor_ratio, args.wiring, args.eligibility[0])
            for arm in arms]


def main():
    """docs/synapse-read-probe.py SWEEP EPOCHS -- <the sweep's rust-sweep.py arguments>, one sweep on as many workers as
    it has arms; or docs/synapse-read-probe.py --batch SPEC.json --workers N, every entry of SPEC -- {"label", "sweep"
    (null for fresh learning-off networks), "epochs", "args"} -- in one pool of N, the biggest networks first."""
    if sys.argv[1] == "--batch":
        spec = json.loads(Path(sys.argv[2]).read_text())
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
        jobs = [job for entry in spec for job in jobs_of(entry["label"], entry["sweep"], entry["epochs"], entry["args"])]
        jobs.sort(key=lambda job: -job[3].get("hidden_neurons", 0.0))
    else:
        sweep, epochs = sys.argv[1], int(sys.argv[2])
        jobs = jobs_of(sweep, sweep, epochs, sys.argv[sys.argv.index("--") + 1:])
        workers = len(jobs)
    print(f"{len(jobs)} jobs on {min(workers, len(jobs))} workers", flush=True)
    with Pool(min(workers, len(jobs))) as pool:
        for name in pool.imap_unordered(run, jobs):
            print("done", name, flush=True)


if __name__ == "__main__":
    main()
