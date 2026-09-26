#!/usr/bin/env python3
"""Sweep a problem's knobs with the Rust wave loop (AUTHORITY.md §6.15), a million epochs an arm.

    docs/rust-sweep.py --name rate-interval-1m --problem doubled_copy \
        --interval 25 30 35 --input-rate 0.1 0.2 0.35 0.5 0.75 1 --seed 1 2 3 4 5 6 --epochs 1000000

The arms are the product of every list given. One process per arm; each builds the network the
command line would build, then runs `fast.train`, which keeps the input and the Poisson drive
in Python's seeded stream and does the epoch and the weight update in Rust. Writes
runs/<name>/<arm>.csv (a downsampled trace) and docs/<name>.md.

`--goo N [N ...]` sets the goo's count (AUTHORITY.md §4.1), a knob like any other so
its count can be swept, its potential axis scaled with fan-in unless `--no-scale-with-fan-in`.
`--floor-ratio R` ties the floor to the threshold arm by arm, MINIMUM_POTENTIAL = R × THRESHOLD
(the grid's own ratio is -4), so a threshold sweep moves the whole axis and not the ratio (§5.2).
The Teacher's homeostasis and un-sticking run at the constants the command line uses, so an arm
here is the run `walnutbutter --seeds` would do.

The reinforce rule with the row critic and late = count. `--eligibility hazard` (the default
under escape noise), `hebb` (the default when the threshold decides, sigma 0: the
single-spike rule of §6.7, charged at every decision against the neuron's own expectation of
its spike), `count_hebb` (its epoch form: what each synapse delivered times its target's count
minus the target's own expectation, hebb until September 17, 2026), `wrong_hebb` (the ±1 rule
hebb replaced on September 16, 2026) or `perturb`, which draws its noise per wave (§6.1) from the arm's seed -- the same stream
a Teacher with that seed would use -- and `--sigma` is then a knob like any other.

Each arm's inputs are drawn up front from the input stream of §4.5, keyed to the arm's
seed alone, so every arm at seed s sees the same epochs in the same order however the
network differs. Comparisons across arms are therefore paired epoch by epoch.

`--resume-from OTHER` continues each arm from the network `runs/OTHER/<arm>-network.json`
saved at the end of that sweep's run of it: the checkpoint's weights, thresholds, clock,
spike times, widths and expectations, the input stream advanced to the epoch it reached,
`--epochs` more epochs from there, the estimator still measured against the first
start's weights, and the trace and record continued from the earlier ones. The reinforcement
baseline is carried over (§9.3), and so is the state of every stream, so the continuation is
the same run to the bit, as §12.11 requires (September 21, 2026; before that the exploration
stream was reseeded and the engine's state was left behind, and a resume parted from the
uninterrupted run at its first epoch). `--population` and `--outputs`, fixed for the
sweep, rebuild an arm under a layout the problem no longer defaults to.

`--checkpoint-every N` (25,000 epochs by default; 0 turns it off) writes each arm's
checkpoint as it goes rather than only at the end, with the arm's record so far inside it.
An arm that has a checkpoint and no `.csv` was cut short, so rerunning the sweep's own
command picks each one up where it stopped and finishes the epochs asked for -- a machine
going down costs at most N epochs an arm instead of the whole run. The checkpoint is moved
into place once written, so the file on disk is always one a run can go on from.

Exploration at the synapse (AUTHORITY.md §7.5-§7.9, §8.16, §8.17, 5.4b): `--exploration synapse`,
with `--synapse-hazard H0` and `--drive-steps N` numeric knobs like any other, and
`--hazard-family`, `--synapse-scaling`, `--trace-counts` and `--drive` string knobs handled as
`--eligibility` is -- one value runs the whole sweep under it, more than one sweeps it, an arm per
value. Under it no arm is given the default width (§6.13), and a positive `--delta` is refused. A
resumed arm keeps the settings its checkpoint was saved under -- the exploration, h0, the family,
the scaling, the trace, the drive and DRIVE_STEPS, the width, and the clock (TAU, the refractory
period, the hop, bored-after and the rate read's window) -- except those the sweep names, and one
that names the other exploration is refused (§12.9); the drive and the clock are kept too, where
until September 25, 2026 every resume took the arm's. Each arm's json names them all as the arm
ran them, and under `--trace-counts ventured` it and the sweep's docs/<name>.md say the estimator
is biased (§8.17). An arm the command line or the engine refuses (§12.2) says so in its line,
{"arm", "refused"}, and the sweep goes on with the others.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import statistics
import sys
import time

from walnutbutter.constants import DECISION_MEMORY, ESCAPE_DELTA  # the driver's default eligibility follows the neuron (§1.3)
from walnutbutter.network import (DRIVES, EXPLORATIONS, SYNAPSE_HAZARD_FAMILIES, SYNAPSE_HAZARD_SCALINGS, TRACE_VENTURED_BIAS,
                                  TRACES)
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOBS = {  # knob -> command-line flag on the simulator, for the record in the report
    "interval": "--interval",
    "input_rate": "--input-rate",
    "cv": "--cv",  # the input train's coefficient of variation; the drive's own coordinate (§4.3)
    "rows": "--rows",
    "lr": "--lr",
    "quash": "--quash",
    "rate_tau": "--rate-tau",
    "tau": "--tau",  # the potential's leak, ms (§5.1): set on the Neuron class by grid_of, arm by arm, each arm its own process
    "sigma": "--sigma",  # exploration noise, only felt under --eligibility perturb (§6.1)
    "delta": "--delta",  # escape noise (§5.2): the decision's width in starting thresholds; --eligibility hazard learns by it
    "threshold": "--threshold",  # THRESHOLD, quoted per THRESHOLD_FAN_IN incoming synapses; goo scales it (§5.2)
    "minimum_potential": "--minimum-potential",  # the floor; or derive it from the threshold with --floor-ratio
    "pickiness": "--pickiness",  # the count read's line, in spikes (§5.10, §9.5)
    "goo": "--goo",  # goo (§3.4) in place of the grid, with this many neurons
    "hidden_neurons": "--hidden-neurons",  # goo's hidden count, the goo being inputs + hidden + outputs (§8, mnist)
    "temperature": "--temperature",  # the evidence critic's temperature: the class sums as log-odds at this scale (§8)
    "projection": "--projection",  # the probability of goo's three earlier wirings, under --wiring (§3.4)
    "scaling_factor": "--scaling-factor",  # goo's scaled rule: every neuron hears N times this in expectation (§3.4)
    "synapse_hazard": "--synapse-hazard",  # h0, the rest hazard, under --exploration synapse (§7.6)
    "drive_steps": "--drive-steps",  # DRIVE_STEPS, the charged drive's deliveries from rest to threshold (5.4b)
    "seed": "--seed",
}
DERIVED = ("cv",)  # knobs that are a reparametrisation of another, handled by hand in grid_of
STRING_KNOBS = {  # knob -> the names it takes: handled as --eligibility is, one value fixed for the sweep, more swept
    "exploration": EXPLORATIONS,  # what explores (§7.1)
    "hazard_family": SYNAPSE_HAZARD_FAMILIES,  # the synapse hazard's family (§7.6)
    "synapse_scaling": SYNAPSE_HAZARD_SCALINGS,  # count or fan-out (§7.7)
    "trace_counts": TRACES,  # what a trace counts (§8.17)
    "drive": DRIVES,  # how a bit becomes spikes (§4.3, 5.4b)
}


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--problem", required=True)
    for knob in KNOBS:
        if knob != "seed":
            parser.add_argument(f"--{knob.replace('_', '-')}", type=float, nargs="*" if knob == "goo" else "+",
                                default=None, metavar="V")
    parser.add_argument("--seed", type=int, nargs="+", default=[1])
    for knob, names in STRING_KNOBS.items():
        parser.add_argument(f"--{knob.replace('_', '-')}", nargs="+", choices=names, default=None,
                            help="more than one value sweeps it, an arm per value; one runs the whole sweep under it")
    parser.add_argument("--floor-ratio", type=float, default=None, metavar="R",
                        help="tie the floor to the threshold, arm by arm: MINIMUM_POTENTIAL = R * THRESHOLD (goo's ratio is -4)")
    parser.add_argument("--wiring", choices=("scaled", "ff2", "ff2-partial", "scaled-open", "zones-equal", "zones", "uniform"), default=None,
                        help="which rule wires goo (§3.4): the command line's default, scaled, unless given")
    parser.add_argument("--no-scale-with-fan-in", dest="scale", action="store_false",
                        help="run goo at a flat threshold and floor instead of the §5.2 rescaling")
    parser.add_argument("--eligibility", nargs="+", choices=("hazard", "hebb"),
                        default=["hazard" if ESCAPE_DELTA > 0 else "hebb"],
                        help="what the reward acts on (§6.7): the centred Hebbian term charged per decision (hebb, the single-spike "
                             "rule of September 17, 2026), its epoch form (count_hebb, hebb until that day), the uncentred +-1 it replaced "
                             "(wrong_hebb), the perturbation the neuron decided under (perturb), or the score of the "
                             "escape-noise decision on each synapse's trace (hazard; needs --delta). More than one value "
                             "sweeps it, an arm per value")
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--resume-from", default=None, metavar="NAME",
                        help="continue every arm from runs/NAME/<arm>-network.json for --epochs more epochs (see above)")
    parser.add_argument("--population", type=int, default=None, help="neurons per population, fixed for the sweep (the problem's unless given)")
    parser.add_argument("--outputs", type=int, default=None, help="the output zone's width, fixed for the sweep (the problem's unless given)")
    parser.add_argument("--trace-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=25_000, metavar="N",
                        help="write each arm's checkpoint every N epochs, so a run cut short resumes from there "
                             "rather than starting over (0 saves only at the end, as before September 22, 2026)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--summary", action="store_true", help="summarise what is on disk; run nothing")
    args = parser.parse_args()
    if args.drive_steps and any(not math.isfinite(steps) or steps != int(steps) or steps < 1 for steps in args.drive_steps):
        parser.error(f"DRIVE_STEPS is a count of deliveries, a whole number of at least 1; got "
                     f"{' '.join(f'{steps:g}' for steps in args.drive_steps)} (5.4b)")
    if args.goo == []:  # bare --goo: the working network's count
        from walnutbutter.constants import GOO_COUNT
        args.goo = [float(GOO_COUNT)]
    return args


def grid_of(problem: str, arm: dict, eligibility: str | None = None, scale: bool = True, floor_ratio: float | None = None,
            wiring: str | None = None, fixed: tuple = ()):
    """The network the command line would build for this problem, with the arm's knobs applied.

    An arm carrying `eligibility` (a swept one) overrides the argument; `fixed`
    is extra command-line words the whole sweep runs with (--population,
    --outputs, and a string knob given one value). With no eligibility named
    the command line's rule applies: hazard, the one §8.3 keeps under
    exploration at the synapse. What explores, and the width, are settled as
    the command line settles them (cli.apply_exploration): the default width
    under the neuron rule only, none under exploration at the synapse (§6.13).
    A goo whose size is not its zones and its hidden count together is
    refused as the command line refuses it (§11.7). The settings the arm
    and `fixed` name are kept on the network as `named_settings`, for
    resume_grid (§12.9).
    """
    eligibility = arm.get("eligibility", eligibility)
    from walnutbutter.cli import apply_exploration, apply_problem, build_parser, refuse_goo_size
    from walnutbutter.goo import Goo
    from walnutbutter.neuron import Neuron

    from walnutbutter.constants import GOO_MINIMUM_POTENTIAL, GOO_THRESHOLD

    from walnutbutter.problems import PROBLEMS
    argv = ["--problem", problem] + ([] if eligibility is None else ["--eligibility", eligibility]) + \
        ([] if wiring is None else ["--wiring", wiring]) + list(fixed)
    base_threshold = GOO_THRESHOLD  # goo is the only container and has its own (§4.11)
    if "threshold" not in arm:
        argv += ["--threshold", f"{GOO_THRESHOLD:g}"]
    if floor_ratio is not None:
        if "minimum_potential" in arm:
            raise ValueError("--floor-ratio derives the floor from the threshold; do not also sweep --minimum-potential")
        argv += ["--minimum-potential", f"{floor_ratio * float(arm.get('threshold', base_threshold)):g}"]
    elif "minimum_potential" not in arm:
        argv += ["--minimum-potential", f"{GOO_MINIMUM_POTENTIAL:g}"]
    for knob, value in arm.items():
        if knob in ("seed", "rows", "eligibility") or knob in DERIVED:
            continue
        if knob in STRING_KNOBS:  # a swept string knob, a word like the eligibility's
            argv += [f"--{knob.replace('_', '-')}", value]
            continue
        if knob == "drive_steps":  # a whole number, which parse checked: as the command line takes it, 1e6 not "1e+06"
            argv += [KNOBS[knob], str(int(value))]
            continue
        argv += [KNOBS[knob], f"{value:g}"]
    if "cv" in arm:
        argv += ["--cv", f"{arm['cv']:.12g}"]
    args = build_parser().parse_args(argv)
    apply_problem(args)
    refuse_goo_size(args)  # a goo that is not the zones and the hidden count together, refused as the command line refuses it (§11.7)
    apply_exploration(args)  # the exploration and the width, refused where the command line refuses them (§6.13, 5.4b)
    Neuron.refractory, Neuron.hop = args.refractory, args.hop
    Neuron.tau, Neuron.bored_after, Neuron.rate_tau = args.tau, args.bored_after, args.rate_tau
    # goo is the only container (AUTHORITY.md §4.1); apply_problem sized it: --goo, or the problem's hidden count and zones
    grid = Goo(count=int(args.goo), across=args.across, weight=None, seed=int(arm["seed"]), threshold=args.threshold, minimum_potential=args.minimum_potential,
               scale_with_fan_in=scale, projection=args.projection, outputs=args.outputs, wiring=args.wiring,
               scaling_factor=args.scaling_factor)
    grid.population, grid.clock = args.population, args.clock
    grid.temperature = args.temperature  # the evidence critic's (§8)
    grid.readout, grid.read, grid.read_window = args.readout, args.read, args.read_window
    grid.pickiness = args.pickiness  # the count read's line, in spikes (§5.10, §9.5)
    grid.interval, grid.drive = args.interval, args.drive
    grid.drive_steps = args.drive_steps  # 5.4b
    grid.input_rate, grid.input_rate_off = args.input_rate, args.input_rate_off
    grid.quash_rate, grid.quash_k = args.quash, args.quash_k
    grid.rule = "reinforce"
    if args.exploration == "synapse":  # §7.5, once the thresholds are the container's; no width (§6.13)
        grid.set_exploration("synapse", h0=args.synapse_hazard, family=args.hazard_family, scaling=args.synapse_scaling,
                             trace=args.trace_counts)
    else:
        grid.set_delta(args.delta)  # escape noise (§5.2), once the thresholds are the container's
    grid.named_settings = args.named  # what the sweep named, which a resume applies over its checkpoint (§12.9)
    grid.drive_given = args.drive_given  # the drive as the arm gave it, for resume_grid to derive at the clock it resumes on
    return grid, args


def _save_network(engine, grid, report: dict, path, progress: dict | None = None) -> None:
    """The arm's network as the run has it, the engine's state written back to the mesh and checkpointed.

    Called at the end of a run and, with `--checkpoint-every`, as it goes. A run
    restored from it is the same run continued, to the bit (§12.11): the stream
    positions and the exploration stream's state go in with the weights, and so
    does the reinforcement baseline (§9.3), written beside the checkpoint's own
    fields. `progress` is the arm's record so far -- its trace rows and its
    estimator -- so that a continuation's record is whole and not just the leg
    it ran.

    Under exploration at the synapse the gains, the read counts and every event
    in flight with its ventured mark come with it (§8.14, §12.9) -- from the
    `report` where fast.train put them there, else from the engine -- and the
    settings from the network, which is what `persistence.checkpoint` writes
    them from. The engine's queue is written into the network's own schedule
    as well, so the file's `pending` is the queue the engine left and not the
    one the network was restored with; `engine_pending` is the same queue in
    the engine's terms, with marks under exploration at the synapse.

    The file appears whole or not at all. A checkpoint is written for the sake of
    a run that gets cut short, so it must not be the thing a crash catches
    half-written: the state goes to a temporary name and is moved over the old
    checkpoint, which stands until the new one is complete.
    """
    import json as _json
    from walnutbutter.persistence import checkpoint
    edges = [c for n in grid.all_neurons() for c in n.outgoing]  # the engine's order (fast.build)
    for c, w in zip(edges, engine.weights()):
        c.weight = w
    for n, theta in zip(grid.all_neurons(), report["thresholds"]):
        n.threshold = theta
    for n, expectation in zip(grid.all_neurons(), report.get("expectations") or []):
        n.expectation = expectation  # the single-spike rule's per-decision expectation (§6.7)
    for n, decisions in zip(grid.all_neurons(), report.get("decisions") or []):
        n.decisions = decisions
    for n, rate in zip(grid.all_neurons(), report.get("rates") or []):
        n.rate = rate  # the Teacher's rate memory, so a continuation's stuck counts and homeostasis start where they were
    for n, potential in zip(grid.all_neurons(), engine.potentials()):
        n.potential = potential  # §12.11: what reset(false) carried across epochs, so the checkpoint holds the run's state
    for c, trace in zip(edges, engine.traces()):
        c.trace = trace  # the per-synapse traces of §6.7, in the engine's edge order
    for c, note in zip(edges, engine.notes()):
        c.noted = note  # and each one's note, B_ij (§6.7, §12.9), the engine's: the network's own was rebuilt at every
        # reset from the traces it was restored with, and until September 25, 2026 a checkpoint carried that instead
    for n, expected in zip(grid.all_neurons(), engine.expected_since_spike()):
        n.expected = expected  # and the expected counts the traces are debited against
    import math as _math
    _none = lambda t: None if _math.isinf(t) and t < 0 else t  # the engine's -inf is the objects' None
    for n, at in zip(grid.all_neurons(), engine.fired_times()):
        n.fired_at = _none(at)  # the refractory test and the hazard run from absolute times that outlive an epoch
    for n, at in zip(grid.all_neurons(), engine.previous_fired_times()):
        n.previous_fired_at = _none(at)
    for n, since in zip(grid.all_neurons(), engine.exposed_since()):
        n.exposed_since = since
    for c, last in zip(edges, engine.last_signals()):
        c.last_signal = _none(last)  # when each synapse last carried a signal, in the engine's edge order
    for n, at in zip(grid.all_neurons(), engine.last_updates()):
        n.last_update = at  # when the potential was last brought up to date: the leak's decay runs from it (§2.3)
    for c, at in zip(edges, engine.trace_ats()):
        c.trace_at = at  # and each trace's
    for n, count in zip(grid.all_neurons(), engine.spike_counts()):
        n.spikes = count  # the run's spike counts to date
    synaptic = grid.exploration == "synapse"
    if synaptic:  # §8.14: G_j, open across the reads, and each output's read count this epoch
        for n, gain in zip(grid.all_neurons(), _of(report, "gains", engine.gains)):
            n.gain = gain
        for n, count in zip(grid.all_neurons(), _of(report, "read_counts", engine.read_counts)):
            n.read_count = count
    # §12.11: every event in flight at the boundary, in delivery order -- under exploration at the synapse each signal
    # with its ventured mark (§7.9), which the unmarked form refuses to drop
    pending = [tuple(e) for e in _of(report, "pending", lambda: engine.pending_events(synaptic))]
    _queue_into(grid, edges, pending)
    import random as _random
    from walnutbutter.fast import sync_explore
    grid.explore_rng = _random.Random()  # §12.11: the checkpoint carries the exploration stream's state. The Rust
    sync_explore(engine, grid.explore_rng)  # engine advanced it; without this it is saved null and a resume reseeds
    writing = path.with_name(path.name + ".writing")
    data = checkpoint(grid, writing)
    data["engine_pending"] = pending  # §12.11: the queue in the engine's terms, for fast.train(pending_events=...)
    data["baseline"] = report.get("baseline")  # §9.3: b as the run left it, so a resume is paid against it
    data["progress"] = progress  # the trace rows and the estimator to date, so a continuation carries the whole record
    writing.write_text(_json.dumps(data))
    os.replace(writing, path)  # atomic: the checkpoint on disk is always one the run can be continued from


def _of(report: dict, key: str, engine_read):
    """What `report` holds under `key`, else the engine's own reading: a report fast.train made carries the state of
    exploration at the synapse, and one a driver built by hand (docs/mnist-watch.py) need not."""
    held = report.get(key)
    return engine_read() if held is None else held


def _queue_into(grid, edges, pending) -> None:
    """The engine's queue, (time, kind, payload[, ventured]) in delivery order, as the network's own schedule: a signal by
    its edge in the engine's order, with its mark; a stimulus by its neuron; a charge (the engine's EXTERNAL, 5.4b) by
    its neuron, of theta / DRIVE_STEPS at delivery."""
    from walnutbutter.propagation import SIGNAL, STIMULUS
    neurons = grid.all_neurons()
    grid.schedule.clear()
    for event in pending:
        time, kind, payload = event[0], event[1], event[2]
        if kind == SIGNAL:
            grid.schedule.signal(edges[payload], time, ventured=len(event) > 3 and bool(event[3]))
        elif kind == STIMULUS:
            grid.schedule.stimulus(neurons[payload], time)
        else:
            grid.schedule.charge(neurons[payload], grid.drive_steps, time)


RESUMED_SETTINGS = ("readout", "read", "read_window", "pickiness", "interval", "presentation", "input_rate", "input_rate_off",
                    "temperature", "population", "clock", "quash_rate", "quash_k",
                    )  # what the arm's settings decide, applied to a restored network over its checkpoint. Not the drive:
# a checkpoint resumes under the drive it was saved under unless the sweep names one (§12.9, resume_grid)


def resume_grid(fresh, source):
    """The arm's network as `runs/<other>/<arm>-network.json` left it, ready to run on: (grid, epochs already run, first-start weights, baseline).

    `fresh` is the arm's network as grid_of builds it from the seed -- the same
    wiring, so its weights are the first start's, the reference the estimator
    keeps measuring from. The checkpoint's state is taken whole (weights,
    thresholds, rate memories, expectations, the clock, the widths, the signals
    in flight, and under exploration at the synapse the gains, read counts,
    ventured marks and its settings); the arm's settings are applied over it,
    and the caller advances the input stream to the epoch reached. The
    reinforcement baseline comes back with it (§9.3): a resumed run is the same
    run continued and is paid against the baseline the run had reached, not a
    fresh one.

    §12.9: a network saved under a setting resumes under it unless the resuming
    run overrides it explicitly. So the drive, DRIVE_STEPS, the settings of
    §7.6, §7.7 and §8.17, the width and the clock -- TAU, the refractory
    period, the hop, bored-after and the rate read's window, set on the Neuron
    class as grid_of sets them -- are the checkpoint's unless the arm or the
    sweep names them (grid_of's `named_settings`), and a width named is applied
    only where it is not the checkpoint's own, whose per-neuron widths stand.
    The drive's rate is derived from the arm's CV at the refractory period the
    network resumes on (§4.3), as the command line derives it. The exploration
    is the exception: a sweep that names the other one is refused. A
    checkpoint written by the object engine carries no `engine_pending`, and
    its queue is taken from the network it restores to, and no top-level
    baseline, which is then its Teacher's, in its learning record.
    """
    from walnutbutter.cli import derive_rate
    from walnutbutter.neuron import Neuron
    from walnutbutter.persistence import CLOCK, clock_of, restore
    reference = [c.weight for n in fresh.all_neurons() for c in n.outgoing]
    arm_clock = {name: getattr(Neuron, name) for name in CLOCK}  # before restore, which sets the rate window from the file
    restored, data = restore(source)
    if data.get("seed") != fresh.seed:  # the reference weights would be another network's, and the correlation meaningless
        raise ValueError(f"{source} was built from seed {data.get('seed')}; the arm is seed {fresh.seed} -- resume a network under its own seed")
    edges = sum(len(n.outgoing) for n in restored.all_neurons())
    if edges != len(reference) or len(restored.all_neurons()) != len(fresh.all_neurons()):
        raise ValueError(f"{source} holds a network of {len(restored.all_neurons())} neurons and {edges} synapses; the arm builds "
                         f"{len(fresh.all_neurons())} and {len(reference)} -- the same seed and layout are needed to resume")
    named = getattr(fresh, "named_settings", frozenset())
    if "exploration" in named and fresh.exploration != restored.exploration:
        raise ValueError(f"{source} was saved under the {restored.exploration} exploration and the sweep names "
                         f"--exploration {fresh.exploration}: a run is not resumed under the other exploration, nothing "
                         "mapping the neuron rule's per-decision bookkeeping onto the gain or back (§12.9)")
    for attr in RESUMED_SETTINGS:
        setattr(restored, attr, getattr(fresh, attr))
    saved_clock = clock_of(data)  # §12.9: the clock the checkpoint ran on, where the sweep names none of it
    for name in CLOCK:
        setattr(Neuron, name, arm_clock[name] if name in named else saved_clock.get(name, arm_clock[name]))
    if getattr(fresh, "drive_given", None) is not None:  # §4.3: the arm's drive at the refractory period resumed on
        restored.input_rate = derive_rate(fresh.drive_given, Neuron.refractory)[0]
    if "drive" in named:
        restored.drive = fresh.drive
    if "drive_steps" in named:
        restored.drive_steps = fresh.drive_steps
    if restored.exploration == "synapse" and named & {"synapse_hazard", "hazard_family", "synapse_scaling", "trace_counts"}:
        restored.set_exploration(  # the settings named over the checkpoint's; kappa_i recomputed from them (§7.7)
            "synapse",
            h0=fresh.synapse_hazard_rest if "synapse_hazard" in named else restored.synapse_hazard_rest,
            family=fresh.synapse_hazard_family if "hazard_family" in named else restored.synapse_hazard_family,
            scaling=fresh.synapse_hazard_scaling if "synapse_scaling" in named else restored.synapse_hazard_scaling,
            trace=fresh.trace_mode if "trace_counts" in named else restored.trace_mode)
    if "delta" in named and fresh.escape_delta != restored.escape_delta:
        restored.set_delta(fresh.escape_delta)  # a width named that is not the checkpoint's (§5.2, §12.9)
    restored.rule = "reinforce"
    restored.engine_pending = data.get("engine_pending")  # §12.11: for fast.train(pending_events=...)
    if restored.engine_pending is None and len(restored.schedule):  # an object engine's checkpoint: its own queue
        from walnutbutter.fast import _queue, flatten
        neurons, index = flatten(restored)[:2]
        edge_index = {id(c): k for k, c in enumerate(c for n in neurons for c in n.outgoing)}
        restored.engine_pending = _queue(restored, index, edge_index)
    restored.engine_progress = data.get("progress")  # the record the checkpoint carried, for the caller to continue
    baseline = data.get("baseline")  # §9.3: the driver's, beside the checkpoint's fields, or an object checkpoint's
    if baseline is None:  # Teacher's, in its learning record; until September 25, 2026 that one was dropped
        baseline = (data.get("learning") or {}).get("baseline")
    return restored, int(data["epoch"]), reference, baseline, data.get("explore_state")


def _rate_by_zone(grid, rates: list[float]) -> dict:
    """The final rate memories averaged over the input zone, the interior and the output zone (goo), or the whole grid."""
    import statistics as st
    if hasattr(grid, "across") and hasattr(grid, "outputs") and hasattr(grid, "count"):
        i, o = grid.across, grid.outputs
        return {"inputs": st.mean(rates[:i]), "interior": st.mean(rates[i:len(rates) - o]) if len(rates) > i + o else None,
                "outputs": st.mean(rates[len(rates) - o:])}
    return {"all": st.mean(rates)}


def _output_counts(engine, grid) -> list[int]:
    """The output zone's counts in the last epoch, from the engine, which holds them (the mesh's neurons do not): each
    output's spikes, plus under exploration at the synapse its read synapse's escapes, the count read (§5.10, §7.9).

    Goo's outputs are its last `outputs` neurons, in the engine's order; for
    any other container nothing is recorded.
    """
    if not hasattr(grid, "outputs"):
        return []
    from walnutbutter.fast import _counts
    counts = _counts(engine)
    return [int(c) for c in counts[len(counts) - grid.outputs:]]


def _trace_rows(earlier, trace, right, offset: int, trace_every: int) -> list:
    """The arm's trace as its csv holds it: whatever record came before, then this leg's points at their own epochs.

    One place builds it, so the record a continued arm writes is the record the
    uninterrupted run would have written, row for row.
    """
    rows = [tuple(row) for row in earlier]
    for k, score in enumerate(trace, start=1):
        was_right = right[k - 1] if k <= len(right) else None
        rows.append((offset + k * trace_every, f"{score:.6g}", "" if was_right is None else f"{was_right:.6g}"))
    return rows


def arm_name(arm: dict) -> str:
    return "-".join(f"{knob}{value}" if isinstance(value, str) else f"{knob}{value:g}" for knob, value in arm.items())


def run_arm(job: tuple) -> dict:
    arm, problem, epochs, trace_every, name, eligibility, scale, floor_ratio, wiring, resume_from, fixed = job[:11]
    checkpoint_every = job[11] if len(job) > 11 else 0  # a worker the Pool respawns under a driver older than the flag
    from walnutbutter import fast
    from walnutbutter.network import input_stream
    from walnutbutter.problems import PROBLEMS, dataset_stream

    out = ROOT / "runs" / name
    path = out / f"{arm_name(arm)}.csv"
    if path.exists():
        return {"arm": arm_name(arm), "skipped": True}
    resumed_self = False
    if resume_from is None and (out / f"{arm_name(arm)}-network.json").exists():
        # the arm has a checkpoint and no record: it was cut short. Rerunning the sweep's own command finishes it,
        # so recovering from a machine going down is the same line again and costs at most --checkpoint-every epochs.
        resume_from, resumed_self = name, True
    try:  # an arm the command line refuses, or a resume §12.9 refuses, says so in its line rather than killing the sweep
        grid, args = grid_of(problem, arm, eligibility, scale, floor_ratio, wiring, fixed)
    except ValueError as exc:
        return {"arm": arm_name(arm), "refused": str(exc)}
    eligibility = arm.get("eligibility", eligibility)
    offset, reference, earlier_trace, earlier_estimator, baseline = 0, None, [], [], None
    if resume_from:  # continue from the saved network; the earlier trace and estimator are carried over so the record is whole
        source = ROOT / "runs" / resume_from / f"{arm_name(arm)}-network.json"
        if not source.exists():
            return {"arm": arm_name(arm), "missing": str(source)}
        try:
            grid, offset, reference, baseline, resumed_explore_state = resume_grid(grid, source)
        except ValueError as exc:
            return {"arm": arm_name(arm), "refused": str(exc)}
        if resumed_self:  # --epochs is the whole run a recovery finishes, not a leg added on to it
            asked, epochs = epochs, epochs - offset
            if epochs <= 0:  # the checkpoint is already at the end: ask for more epochs, or move it aside
                return {"arm": arm_name(arm), "checkpoint_at": offset, "epochs_asked": asked}
        carried = getattr(grid, "engine_progress", None) or {}  # the record the checkpoint itself carried
        earlier_trace = [tuple(row) for row in carried.get("rows") or []]
        earlier_estimator = carried.get("estimator") or []
        earlier = source.with_name(f"{arm_name(arm)}.json")  # a sweep that finished wrote its record beside the
        if not earlier_estimator and earlier.exists():  # checkpoint; a run cut short has only what the checkpoint holds
            earlier_estimator = json.loads(earlier.read_text()).get("estimator") or []
        earlier_csv = source.with_name(f"{arm_name(arm)}.csv")
        if not earlier_trace and earlier_csv.exists():
            earlier_trace = [(int(r["epoch"]), r["score"], r.get("right", "")) for r in csv.DictReader(open(earlier_csv))]
    direction = None
    if PROBLEMS[problem].data == "mnist" and hasattr(grid, "outputs"):  # the estimator's correlation over time (§8)
        from walnutbutter.mnist import supervised_direction
        direction = supervised_direction(grid)
    first = grid.all_neurons()[0]
    started_at = {"theta": first.threshold, "floor": first.minimum_potential}  # what a neuron starts at, scaling applied
    data = dataset_stream(PROBLEMS[problem].data, int(arm["seed"]))  # a dataset's images with their labels (§8), or
    if data is None:  # random bits drawn up front, §4.5: every arm at this seed sees the same epochs in the same order
        patterns, labels = input_stream(offset + epochs, grid.raw_bit_count(), int(arm["seed"])), None
    else:
        patterns, labels = data
    explore_seed, explore_state = int(arm["seed"]), None
    if resume_from:  # §12.11: the streams continue where the checkpoint left them -- a resume is the same run, continued
        grid.use_input_stream(patterns, labels)
        grid.input_at = offset
        patterns = labels = None
        explore_state = resumed_explore_state
    network_path = path.with_name(path.stem + "-network.json")

    def write_checkpoint(epoch: int, engine, grid_now, report_now: dict) -> None:
        """The arm as it stands, at every --checkpoint-every epochs: the state to go on from and the record so far."""
        rows = _trace_rows(earlier_trace, report_now["trace"], report_now["right"], offset, trace_every)
        _save_network(engine, grid_now, report_now, network_path,
                      {"rows": rows, "estimator": earlier_estimator + (report_now.get("estimator") or [])})

    started = time.perf_counter()
    try:  # what the engine refuses (§12.2) -- at the start, or a threshold §7.5 refuses mid-run -- is the arm's line too
        mean, trace, engine, report = fast.train(
            grid, epochs, lr=args.lr, target=args.target, trace_every=trace_every, patterns=patterns, labels=labels,
            eligibility=args.eligibility, seed=explore_seed, explore_state=explore_state, pending_events=getattr(grid, "engine_pending", None),
            homeostasis=args.homeostasis, target_rate=args.target_rate, unstick=args.unstick,
            unstick_target=args.unstick_target, critic=args.critic, direction=direction,
            checkpoint=write_checkpoint if checkpoint_every else None, checkpoint_every=checkpoint_every,
            reference_weights=reference, epoch_offset=offset, baseline=baseline,
        )
    except ValueError as exc:  # and not the sweep's death, with every sibling arm's work
        return {"arm": arm_name(arm), "refused": str(exc)}
    elapsed = time.perf_counter() - started
    from walnutbutter.neuron import Neuron
    if report.get("estimator") is not None:
        report["estimator"] = earlier_estimator + report["estimator"]
    rows = _trace_rows(earlier_trace, trace, report.get("right") or [], offset, trace_every)
    _save_network(engine, grid, report, network_path,  # so an arm can be resumed, not rerun
                  {"rows": rows, "estimator": report.get("estimator") or []})
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "score", "right"])
        writer.writerows(rows)
    result = {"arm": arm_name(arm), "mean": mean, "last_tenth": report["last_tenth"], "stuck_on": report["stuck_on"],
              "stuck_off": report["stuck_off"], "unstuck": report["unstuck"], "seconds": round(elapsed),
              "epochs_per_second": round(epochs / elapsed), "eligibility": eligibility, **started_at,
              "read": grid.read, "pickiness": grid.pickiness,  # what "on" meant at the read (§5.10, §9.5)
              "critic": args.critic, "problem": problem, "lr": args.lr,  # the rate the arm ran at, swept or the problem's own
              "population": grid.population, "outputs": getattr(grid, "outputs", None),
              "resumed_from": resume_from, "epoch_offset": offset, "epochs_run": epochs,  # a continuation: from where, and how far
              # the clock the arm ran on, a resumed arm's being its checkpoint's where the sweep named none (§12.9)
              "tau": Neuron.tau, "refractory": Neuron.refractory, "hop": Neuron.hop,
              "explore_seed": explore_seed,
              "threshold": args.threshold, "minimum_potential": args.minimum_potential, "floor_ratio": floor_ratio,
              # escape noise (§5.2), 0 when the threshold decided or the synapses explore (§6.13): the width of the network
              # the arm ran, which a resumed arm takes from its checkpoint unless the sweep names one (§12.9)
              "delta": grid.escape_delta,
              "escape_scale": grid.escape_scale,  # and the count's scaling of every hazard, sqrt(60 / N) (§5.2)
              **_exploration_record(grid),  # what explored, with what, and the drive (§7.1, §7.6, §7.7, §8.17, 5.4b)
              "decision_memory": DECISION_MEMORY if eligibility == "hebb" else None,  # the single-spike rule's memory (§6.7,
              # September 17, 2026); a record naming hebb with a count_memory that is not null ran the epoch form, count_hebb since
              # naming hebb without it is from before September 16, 2026, when hebb named the +-1 rule now called wrong_hebb
              "wiring": getattr(grid, "wiring", None),  # goo's rule (§3.4), and its knobs
              "projection": getattr(grid, "projection", None),
              "scaling_factor": getattr(grid, "scaling_factor", None),
              "temperature": grid.temperature if args.critic == "evidence" else None,  # the evidence critic's (§8), and
              "accuracy_last_tenth": report.get("accuracy_last_tenth"),  # the class critic's fraction right beside it
              "last_tenth_over": "engine",  # every epoch of the last tenth, as the engine counted them
              "rate_by_zone": _rate_by_zone(grid, report["rates"]),  # the final rate memories, averaged over each zone
              "estimator": report.get("estimator"),  # the estimator's correlation over time (§8), for a dataset on goo
              "output_counts_last": _output_counts(engine, grid),  # the last epoch's output spikes, in order
              "container": repr(grid)}  # goo is the only container (§4.1)
    if offset:
        # A continued arm's leg is not the run. fast.train measures the last tenth of what it was asked to run, so an
        # arm that lost its last 25,000 epochs to a reboot would report a tenth of those and not a tenth of the
        # million -- a noisier number than its siblings' and, read beside them, a wrong one. Take it from the record
        # instead, over the last tenth of the whole run, and say that is where it came from. The fraction right is
        # exact either way (each traced point is the mean over its own interval); the score is those points' mean,
        # sampled every --trace-every epochs rather than counted over all of them.
        tenth = rows[-max(1, len(rows) // 10):]
        result["last_tenth"] = statistics.fmean(float(row[1]) for row in tenth)
        right_of = [float(row[2]) for row in tenth if row[2] != ""]
        result["accuracy_last_tenth"] = statistics.fmean(right_of) if right_of else None
        result["last_tenth_over"] = f"the record's last tenth, {len(tenth)} points from epoch {tenth[0][0]:,}"
    path.with_suffix(".json").write_text(json.dumps(result))  # the summary the trace cannot give: the mean over the last tenth
    return result


def _exploration_record(grid) -> dict:
    """What an arm's json says of what explored (§7.1): the exploration, and under exploration at the synapse h0, the
    family, the scaling and the trace -- null under the neuron rule, where none of them is in force -- with the bias
    §8.17 asks a TRACE ventured run to state; and the drive, with DRIVE_STEPS where the drive is charged (5.4b). Read
    from the network the arm ran, so a resumed arm names what it resumed under."""
    synaptic = grid.exploration == "synapse"
    return {"exploration": grid.exploration,
            "synapse_hazard": grid.synapse_hazard_rest if synaptic else None,
            "hazard_family": grid.synapse_hazard_family if synaptic else None,
            "synapse_scaling": grid.synapse_hazard_scaling if synaptic else None,
            "trace_counts": grid.trace_mode if synaptic else None,
            "estimator_bias": TRACE_VENTURED_BIAS if synaptic and grid.trace_mode == "ventured" else None,
            "drive": grid.drive, "drive_steps": grid.drive_steps if grid.drive == "charged" else None}


def string_words(args) -> list[str]:
    """Each string knob given one value, as the command-line words that run the whole sweep under it, as a single
    --eligibility does; one given more than once is swept (grid_and_arms)."""
    words = []
    for knob in STRING_KNOBS:
        values = getattr(args, knob)
        if values is not None and len(values) == 1:
            words += [f"--{knob.replace('_', '-')}", values[0]]
    return words


def fixed_words(args) -> list[str]:
    """The command-line words every arm of the sweep is built with: --population, --outputs and the string knobs given
    one value each."""
    return ([] if args.population is None else ["--population", str(args.population)]) + \
           ([] if args.outputs is None else ["--outputs", str(args.outputs)]) + string_words(args)


def grid_and_arms(args):
    given = {knob: getattr(args, knob) for knob in KNOBS if knob != "seed" and getattr(args, knob) is not None}
    if len(args.eligibility) > 1:
        given["eligibility"] = list(args.eligibility)  # a swept eligibility, an arm per value
    for knob in STRING_KNOBS:  # a swept string knob, the same way; one given once is a fixed word (fixed_words)
        values = getattr(args, knob)
        if values is not None and len(values) > 1:
            given[knob] = list(values)
    given["seed"] = list(args.seed)
    swept = [knob for knob, values in given.items() if len(values) > 1]
    return swept, [dict(zip(given, values)) for values in itertools.product(*given.values())]


def summarise(args) -> None:
    swept, arms = grid_and_arms(args)
    out = ROOT / "runs" / args.name
    rows, records = [], []
    for arm in arms:
        path = out / f"{arm_name(arm)}.csv"
        if not path.exists():
            continue
        if path.with_suffix(".json").exists():
            records.append(json.loads(path.with_suffix(".json").read_text()))
        scores = [float(r["score"]) for r in csv.DictReader(open(path))]
        tenth = max(1, len(scores) // 10)
        rows.append({"arm": arm, "last_tenth": statistics.fmean(scores[-tenth:]),
                     "first_tenth": statistics.fmean(scores[:tenth]), "max": max(scores)})
    if not rows:
        print("nothing on disk yet")
        return
    axes = [k for k in swept if k != "seed"]
    from walnutbutter.problems import PROBLEMS
    sizes = (" " + " ".join(f"{g:g}" for g in args.goo)) if args.goo else ""
    container = "goo" + sizes + ("" if args.scale else ", flat")
    words = " ".join(string_words(args))  # what the whole sweep explored by and was driven by, where it named them
    ran = sorted({r.get("exploration") or "neuron" for r in records})  # and as the arms ran: a resume keeps its checkpoint's
    if "synapse" in ran and "--exploration" not in words:  # exploration though the sweep names none (§12.9)
        said = (f"exploration at the synapse{' and the neuron rule' if len(ran) > 1 else ''}, as the arms' checkpoints "
                "were saved")
        words = f"{words}, {said}" if words else said
    biased = any(r.get("estimator_bias") for r in records)
    lines = [f"# {args.name}: {args.problem} on the {container}, {args.epochs:,} epochs an arm, the Rust wave loop (§6.15), "
             f"{', '.join(args.eligibility)} eligibility{f', {words}' if words else ''}"
             f"{f' -- {TRACE_VENTURED_BIAS}' if biased else ''}", ""]  # §8.17: a TRACE ventured run says so in its record
    if len(axes) == 2:
        x, y = axes
        xs = sorted({r["arm"][x] for r in rows})
        ys = sorted({r["arm"][y] for r in rows})
        fmt = lambda v: v if isinstance(v, str) else f"{v:g}"  # the eligibility and the string knobs are words
        lines += ["Mean score over the last tenth of each run, averaged over seeds.", "",
                  "| " + y + " \\ " + x + " | " + " | ".join(fmt(v) for v in xs) + " |",
                  "|" + "---|" * (len(xs) + 1)]
        for v in ys:
            cells = []
            for u in xs:
                got = [r["last_tenth"] for r in rows if r["arm"][x] == u and r["arm"][y] == v]
                cells.append(f"{statistics.fmean(got):.4f}" if got else "—")
            lines.append(f"| {fmt(v)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["| arm | first tenth | last tenth | best |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["last_tenth"]):
        lines.append(f"| {arm_name(r['arm'])} | {r['first_tenth']:.4f} | {r['last_tenth']:.4f} | {r['max']:.4f} |")
    (ROOT / "docs" / f"{args.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:24]))


def main() -> int:
    venv = ROOT / ".venv"
    if (venv / "bin" / "python").exists() and Path(sys.prefix).resolve() != venv.resolve():
        python = str(venv / "bin" / "python")
        os.execv(python, [python, __file__] + sys.argv[1:])
    args = parse()
    (ROOT / "runs" / args.name).mkdir(parents=True, exist_ok=True)
    swept, arms = grid_and_arms(args)
    if not args.summary:
        # two cores short of the machine, not one: the driver takes one and Byron keeps one for a single run he can
        # watch beside the sweep (September 16, 2026: "default to 30 workers" on the 32-core machine)
        workers = args.workers or min(len(arms), max(1, (os.cpu_count() or 3) - 2))
        cadence = f"checkpointing every {args.checkpoint_every:,}" if args.checkpoint_every else "no checkpoints until the end"
        print(f"{len(arms)} arms on {workers} workers, {args.epochs:,} epochs each, {cadence}, sweeping {swept}", flush=True)
        started = time.perf_counter()
        fixed = fixed_words(args)
        jobs = [(arm, args.problem, args.epochs, args.trace_every, args.name, args.eligibility[0], args.scale,
                 args.floor_ratio, args.wiring, args.resume_from, tuple(fixed), args.checkpoint_every) for arm in arms]
        with Pool(workers) as pool:
            for result in pool.imap_unordered(run_arm, jobs):
                print(f"[{time.perf_counter() - started:6.0f}s] {json.dumps(result)}", flush=True)
    summarise(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
