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
baseline is carried over (§9.3). Still not a resume to the bit, which §12.11 requires: the
exploration stream starts afresh (seed + 1,000,000) and no checkpoint carries a generator's
state. `--population` and `--outputs`, fixed for the
sweep, rebuild an arm under a layout the problem no longer defaults to.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import statistics
import sys
import time

from walnutbutter.constants import DECISION_MEMORY, ESCAPE_DELTA  # the driver's default eligibility follows the neuron (§1.3)
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
    "seed": "--seed",
}
DERIVED = ("cv",)  # knobs that are a reparametrisation of another, handled by hand in grid_of


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--problem", required=True)
    for knob in KNOBS:
        if knob != "seed":
            parser.add_argument(f"--{knob.replace('_', '-')}", type=float, nargs="*" if knob == "goo" else "+",
                                default=None, metavar="V")
    parser.add_argument("--seed", type=int, nargs="+", default=[1])
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
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--summary", action="store_true", help="summarise what is on disk; run nothing")
    args = parser.parse_args()
    if args.goo == []:  # bare --goo: the working network's count
        from walnutbutter.constants import GOO_COUNT
        args.goo = [float(GOO_COUNT)]
    return args


def grid_of(problem: str, arm: dict, eligibility: str = "hebb", scale: bool = True, floor_ratio: float | None = None,
            wiring: str | None = None, fixed: tuple = ()):
    """The network the command line would build for this problem, with the arm's knobs applied.

    An arm carrying `eligibility` (a swept one) overrides the argument; `fixed`
    is extra command-line words the whole sweep runs with (--population, --outputs).
    """
    eligibility = arm.get("eligibility", eligibility)
    from walnutbutter.cli import apply_problem, build_parser
    from walnutbutter.goo import Goo
    from walnutbutter.neuron import Neuron

    from walnutbutter.constants import GOO_MINIMUM_POTENTIAL, GOO_THRESHOLD

    from walnutbutter.problems import PROBLEMS
    argv = ["--problem", problem, "--eligibility", eligibility] + ([] if wiring is None else ["--wiring", wiring]) + list(fixed)
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
        argv += [KNOBS[knob], f"{value:g}"]
    if "cv" in arm:
        argv += ["--cv", f"{arm['cv']:.12g}"]
    args = build_parser().parse_args(argv)
    apply_problem(args)
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
    grid.input_rate, grid.input_rate_off = args.input_rate, args.input_rate_off
    grid.quash_rate, grid.quash_k = args.quash, args.quash_k
    grid.rule = "reinforce"
    grid.set_delta(args.delta)  # escape noise (§5.2), once the thresholds are the container's
    return grid, args


def _save_network(engine, grid, report: dict, path) -> None:
    """The arm's network at the end of the run: the engine's weights and thresholds written back to the mesh and checkpointed.

    A run restored from it continues with those weights (the CLI's --load-weights, or
    docs/mnist-watch.py); it is not a resume to the bit, since the stream's position
    and the exploration stream's state are not in a checkpoint. The reinforcement
    baseline is (§9.3), written beside the checkpoint's own fields.
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
    data = checkpoint(grid, path)
    data["baseline"] = report.get("baseline")  # §9.3: b as the run left it, so a resume is paid against it
    path.write_text(_json.dumps(data))


RESUMED_SETTINGS = ("readout", "read", "read_window", "pickiness", "interval", "drive", "input_rate", "input_rate_off",
                    "temperature", "population", "clock", "quash_rate", "quash_k",
                    )  # what the arm's settings decide, applied to a restored network over its checkpoint


def resume_grid(fresh, source):
    """The arm's network as `runs/<other>/<arm>-network.json` left it, ready to run on: (grid, epochs already run, first-start weights, baseline).

    `fresh` is the arm's network as grid_of builds it from the seed -- the same
    wiring, so its weights are the first start's, the reference the estimator
    keeps measuring from. The checkpoint's state is taken whole (weights,
    thresholds, rate memories, expectations, the clock, the widths, the signals
    in flight); the arm's settings are applied over it, and the caller advances
    the input stream to the epoch reached. The reinforcement baseline comes back
    with it (§9.3): a resumed run is the same run continued and is paid against
    the baseline the run had reached, not a fresh one.
    """
    from walnutbutter.neuron import Neuron
    from walnutbutter.persistence import restore
    reference = [c.weight for n in fresh.all_neurons() for c in n.outgoing]
    restored, data = restore(source)
    if data.get("seed") != fresh.seed:  # the reference weights would be another network's, and the correlation meaningless
        raise ValueError(f"{source} was built from seed {data.get('seed')}; the arm is seed {fresh.seed} -- resume a network under its own seed")
    edges = sum(len(n.outgoing) for n in restored.all_neurons())
    if edges != len(reference) or len(restored.all_neurons()) != len(fresh.all_neurons()):
        raise ValueError(f"{source} holds a network of {len(restored.all_neurons())} neurons and {edges} synapses; the arm builds "
                         f"{len(fresh.all_neurons())} and {len(reference)} -- the same seed and layout are needed to resume")
    for attr in RESUMED_SETTINGS:
        setattr(restored, attr, getattr(fresh, attr))
    restored.rule = "reinforce"
    return restored, int(data["epoch"]), reference, data.get("baseline")


def _rate_by_zone(grid, rates: list[float]) -> dict:
    """The final rate memories averaged over the input zone, the interior and the output zone (goo), or the whole grid."""
    import statistics as st
    if hasattr(grid, "across") and hasattr(grid, "outputs") and hasattr(grid, "count"):
        i, o = grid.across, grid.outputs
        return {"inputs": st.mean(rates[:i]), "interior": st.mean(rates[i:len(rates) - o]) if len(rates) > i + o else None,
                "outputs": st.mean(rates[len(rates) - o:])}
    return {"all": st.mean(rates)}


def _output_counts(engine, grid) -> list[int]:
    """The output zone's spike counts in the last epoch, from the engine, which holds them (the mesh's neurons do not).

    Goo's outputs are its last `outputs` neurons, in the engine's order; for
    any other container nothing is recorded.
    """
    if not hasattr(grid, "outputs"):
        return []
    counts = engine.epoch_spike_counts()
    return [int(c) for c in counts[len(counts) - grid.outputs:]]


def arm_name(arm: dict) -> str:
    return "-".join(f"{knob}{value}" if isinstance(value, str) else f"{knob}{value:g}" for knob, value in arm.items())


def run_arm(job: tuple) -> dict:
    arm, problem, epochs, trace_every, name, eligibility, scale, floor_ratio, wiring, resume_from, fixed = job
    from walnutbutter import fast
    from walnutbutter.network import input_stream
    from walnutbutter.problems import PROBLEMS, dataset_stream

    out = ROOT / "runs" / name
    path = out / f"{arm_name(arm)}.csv"
    if path.exists():
        return {"arm": arm_name(arm), "skipped": True}
    grid, args = grid_of(problem, arm, eligibility, scale, floor_ratio, wiring, fixed)
    eligibility = arm.get("eligibility", eligibility)
    offset, reference, earlier_trace, earlier_estimator, baseline = 0, None, [], [], None
    if resume_from:  # continue from the saved network; the earlier trace and estimator are carried over so the record is whole
        source = ROOT / "runs" / resume_from / f"{arm_name(arm)}-network.json"
        if not source.exists():
            return {"arm": arm_name(arm), "missing": str(source)}
        grid, offset, reference, baseline = resume_grid(grid, source)
        earlier = source.with_name(f"{arm_name(arm)}.json")
        if earlier.exists():
            earlier_estimator = json.loads(earlier.read_text()).get("estimator") or []
        earlier_csv = source.with_name(f"{arm_name(arm)}.csv")
        if earlier_csv.exists():
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
    explore_seed = int(arm["seed"])
    if resume_from:  # the stream is attached here and advanced to the epoch reached; the exploration stream starts afresh
        grid.use_input_stream(patterns, labels)
        grid.input_at = offset
        patterns = labels = None
        explore_seed += 1_000_000
    started = time.perf_counter()
    mean, trace, engine, report = fast.train(
        grid, epochs, lr=args.lr, target=args.target, trace_every=trace_every, patterns=patterns, labels=labels,
        eligibility=args.eligibility, seed=explore_seed,
        homeostasis=args.homeostasis, target_rate=args.target_rate, unstick=args.unstick,
        unstick_target=args.unstick_target, critic=args.critic, direction=direction,
        reference_weights=reference, epoch_offset=offset, baseline=baseline,
    )
    elapsed = time.perf_counter() - started
    _save_network(engine, grid, report, path.with_name(path.stem + "-network.json"))  # so an arm can be resumed, not rerun
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "score", "right"])
        for epoch, score, was_right in earlier_trace:
            writer.writerow([epoch, score, was_right])
        traced_right = report.get("right") or []
        for k, score in enumerate(trace, start=1):
            was_right = traced_right[k - 1] if k <= len(traced_right) else None
            writer.writerow([offset + k * trace_every, f"{score:.6g}",
                             "" if was_right is None else f"{was_right:.6g}"])
    from walnutbutter.neuron import Neuron
    if report.get("estimator") is not None:
        report["estimator"] = earlier_estimator + report["estimator"]
    result = {"arm": arm_name(arm), "mean": mean, "last_tenth": report["last_tenth"], "stuck_on": report["stuck_on"],
              "stuck_off": report["stuck_off"], "unstuck": report["unstuck"], "seconds": round(elapsed),
              "epochs_per_second": round(epochs / elapsed), "eligibility": eligibility, **started_at,
              "read": grid.read, "pickiness": grid.pickiness,  # what "on" meant at the read (§5.10, §9.5)
              "critic": args.critic, "problem": problem, "lr": args.lr,  # the rate the arm ran at, swept or the problem's own
              "population": grid.population, "outputs": getattr(grid, "outputs", None),
              "resumed_from": resume_from, "epoch_offset": offset, "epochs_run": epochs,  # a continuation: from where, and how far
              "tau": args.tau, "refractory": args.refractory, "hop": args.hop,  # the clock the arm ran on
              "explore_seed": explore_seed,
              "threshold": args.threshold, "minimum_potential": args.minimum_potential, "floor_ratio": floor_ratio,
              "delta": args.delta,  # escape noise (§5.2), 0 when the threshold decided
              "escape_scale": grid.escape_scale,  # and the count's scaling of every hazard, sqrt(60 / N) (§5.2)
              "decision_memory": DECISION_MEMORY if eligibility == "hebb" else None,  # the single-spike rule's memory (§6.7,
              # September 17, 2026); a record naming hebb with a count_memory that is not null ran the epoch form, count_hebb since
              # naming hebb without it is from before September 16, 2026, when hebb named the +-1 rule now called wrong_hebb
              "wiring": getattr(grid, "wiring", None),  # goo's rule (§3.4), and its knobs
              "projection": getattr(grid, "projection", None),
              "scaling_factor": getattr(grid, "scaling_factor", None),
              "temperature": grid.temperature if args.critic == "evidence" else None,  # the evidence critic's (§8), and
              "accuracy_last_tenth": report.get("accuracy_last_tenth"),  # the class critic's fraction right beside it
              "rate_by_zone": _rate_by_zone(grid, report["rates"]),  # the final rate memories, averaged over each zone
              "estimator": report.get("estimator"),  # the estimator's correlation over time (§8), for a dataset on goo
              "output_counts_last": _output_counts(engine, grid),  # the last epoch's output spikes, in order
              "container": repr(grid)}  # goo is the only container (§4.1)
    path.with_suffix(".json").write_text(json.dumps(result))  # the summary the trace cannot give: the mean over the last tenth
    return result


def grid_and_arms(args):
    given = {knob: getattr(args, knob) for knob in KNOBS if knob != "seed" and getattr(args, knob) is not None}
    if len(args.eligibility) > 1:
        given["eligibility"] = list(args.eligibility)  # a swept eligibility, an arm per value (the only string-valued knob)
    given["seed"] = list(args.seed)
    swept = [knob for knob, values in given.items() if len(values) > 1]
    return swept, [dict(zip(given, values)) for values in itertools.product(*given.values())]


def summarise(args) -> None:
    swept, arms = grid_and_arms(args)
    out = ROOT / "runs" / args.name
    rows = []
    for arm in arms:
        path = out / f"{arm_name(arm)}.csv"
        if not path.exists():
            continue
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
    lines = [f"# {args.name}: {args.problem} on the {container}, {args.epochs:,} epochs an arm, the Rust wave loop (§6.15), "
             f"{', '.join(args.eligibility)} eligibility", ""]
    if len(axes) == 2:
        x, y = axes
        xs = sorted({r["arm"][x] for r in rows})
        ys = sorted({r["arm"][y] for r in rows})
        fmt = lambda v: v if isinstance(v, str) else f"{v:g}"  # eligibility is the one string-valued knob
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
        print(f"{len(arms)} arms on {workers} workers, {args.epochs:,} epochs each, sweeping {swept}", flush=True)
        started = time.perf_counter()
        fixed = ([] if args.population is None else ["--population", str(args.population)]) + \
                ([] if args.outputs is None else ["--outputs", str(args.outputs)])
        jobs = [(arm, args.problem, args.epochs, args.trace_every, args.name, args.eligibility[0], args.scale,
                 args.floor_ratio, args.wiring, args.resume_from, tuple(fixed)) for arm in arms]
        with Pool(workers) as pool:
            for result in pool.imap_unordered(run_arm, jobs):
                print(f"[{time.perf_counter() - started:6.0f}s] {json.dumps(result)}", flush=True)
    summarise(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
