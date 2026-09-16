#!/usr/bin/env python3
"""Sweep a problem's knobs with the Rust wave loop (AUTHORITY.md §6.15), a million epochs an arm.

    docs/rust-sweep.py --name rate-interval-1m --problem doubled_copy \
        --interval 25 30 35 --input-rate 0.1 0.2 0.35 0.5 0.75 1 --seed 1 2 3 4 5 6 --epochs 1000000

The arms are the product of every list given. One process per arm; each builds the grid the
command line would build, then runs `fast.train`, which keeps the input and the Poisson drive
in Python's seeded stream and does the epoch and the weight update in Rust. Writes
runs/<name>/<arm>.csv (a downsampled trace) and docs/<name>.md.

`--goo N [N ...]` runs goo (AUTHORITY.md §3.4) in place of the grid, a knob like any other so
its count can be swept, its potential axis scaled with fan-in unless `--no-scale-with-fan-in`.
`--floor-ratio R` ties the floor to the threshold arm by arm, MINIMUM_POTENTIAL = R × THRESHOLD
(the grid's own ratio is -4), so a threshold sweep moves the whole axis and not the ratio (§5.2).
The Teacher's homeostasis and un-sticking run at the constants the command line uses, so an arm
here is the run `walnutbutter --seeds` would do.

The reinforce rule with the row critic and late = count. `--eligibility hazard` (the default
under escape noise), `hebb` (the default when the threshold decides, sigma 0: the centred
Hebbian rule of §6.7, what each synapse delivered times its target's count minus the
target's own expectation), `wrong_hebb` (the ±1 rule hebb replaced on September 16, 2026)
or `perturb`, which draws its noise per wave (§6.1) from the arm's seed -- the same stream
a Teacher with that seed would use -- and `--sigma` is then a knob like any other.

Each arm's inputs are drawn up front from the input stream of §4.5, keyed to the arm's
seed alone, so every arm at seed s sees the same epochs in the same order however the
network differs. Comparisons across arms are therefore paired epoch by epoch.
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

from walnutbutter.constants import COUNT_MEMORY, ESCAPE_DELTA  # the driver's default eligibility follows the neuron (§1.3)
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
    "sigma": "--sigma",  # exploration noise, only felt under --eligibility perturb (§6.1)
    "delta": "--delta",  # escape noise (§5.2): the decision's width in starting thresholds; --eligibility hazard learns by it
    "threshold": "--threshold",  # THRESHOLD, quoted per THRESHOLD_FAN_IN incoming synapses; goo scales it (§5.2)
    "minimum_potential": "--minimum-potential",  # the floor; or derive it from the threshold with --floor-ratio
    "teacher_threshold": "--teacher-threshold",  # the count read's line, in Hz (§4.3)
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
                        help="tie the floor to the threshold, arm by arm: MINIMUM_POTENTIAL = R * THRESHOLD (the grid's is -4)")
    parser.add_argument("--wiring", choices=("scaled", "zones-equal", "zones", "uniform"), default=None,
                        help="which rule wires goo (§3.4): the command line's default, scaled, unless given")
    parser.add_argument("--no-scale-with-fan-in", dest="scale", action="store_false",
                        help="run goo at a flat threshold and floor instead of the §5.2 rescaling")
    parser.add_argument("--eligibility", nargs="+", choices=("hebb", "wrong_hebb", "perturb", "hazard"),
                        default=["hazard" if ESCAPE_DELTA > 0 else "hebb"],
                        help="what the reward acts on (§6.7): the centred Hebbian term (hebb), the uncentred +-1 it replaced "
                             "(wrong_hebb), the perturbation the neuron decided under (perturb), or the score of the "
                             "escape-noise decision on each synapse's trace (hazard; needs --delta). More than one value "
                             "sweeps it, an arm per value")
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--trace-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--summary", action="store_true", help="summarise what is on disk; run nothing")
    args = parser.parse_args()
    if args.goo == []:  # bare --goo: the working network's count
        from walnutbutter.constants import GOO_COUNT
        args.goo = [float(GOO_COUNT)]
    return args


def grid_of(problem: str, arm: dict, eligibility: str = "hebb", scale: bool = True, floor_ratio: float | None = None,
            wiring: str | None = None):
    """The network the command line would build for this problem, with the arm's knobs applied.

    An arm carrying `eligibility` (a swept one) overrides the argument.
    """
    eligibility = arm.get("eligibility", eligibility)
    from walnutbutter.cli import apply_problem, build_parser
    from walnutbutter.goo import Goo
    from walnutbutter.grid import GridOfNeurons
    from walnutbutter.neuron import Neuron

    from walnutbutter.constants import GOO_MINIMUM_POTENTIAL, GOO_THRESHOLD, THRESHOLD

    from walnutbutter.problems import PROBLEMS
    on_goo = ("goo" in arm or "hidden_neurons" in arm or PROBLEMS[problem].goo is not None
              or PROBLEMS[problem].hidden_neurons is not None)  # posed on goo: by the arm, or by the problem
    argv = ["--problem", problem, "--eligibility", eligibility] + ([] if wiring is None else ["--wiring", wiring])
    base_threshold = GOO_THRESHOLD if on_goo else THRESHOLD  # goo has its own (§1.2)
    if on_goo and "threshold" not in arm:
        argv += ["--threshold", f"{GOO_THRESHOLD:g}"]
    if floor_ratio is not None:
        if "minimum_potential" in arm:
            raise ValueError("--floor-ratio derives the floor from the threshold; do not also sweep --minimum-potential")
        argv += ["--minimum-potential", f"{floor_ratio * float(arm.get('threshold', base_threshold)):g}"]
    elif on_goo and "minimum_potential" not in arm:
        argv += ["--minimum-potential", f"{GOO_MINIMUM_POTENTIAL:g}"]
    for knob, value in arm.items():
        if knob in ("seed", "rows", "eligibility") or knob in DERIVED:
            continue
        argv += [KNOBS[knob], f"{value:g}"]
    if "rows" in arm:
        argv += ["--rows", f"{arm['rows']:g}"]
    if "cv" in arm:
        argv += ["--cv", f"{arm['cv']:.12g}"]
    args = build_parser().parse_args(argv)
    apply_problem(args)
    Neuron.refractory, Neuron.refractory_hops = args.refractory, args.refractory_hops
    Neuron.tau, Neuron.bored_after, Neuron.rate_tau = args.tau, args.bored_after, args.rate_tau
    if on_goo:  # apply_problem sized the goo: --goo, or the problem's hidden count and zones
        grid = Goo(count=int(args.goo), across=args.across, weight=None, seed=int(arm["seed"]),
                   permute=not args.no_permute, threshold=args.threshold, minimum_potential=args.minimum_potential,
                   scale_with_fan_in=scale, projection=args.projection, outputs=args.outputs, wiring=args.wiring,
                   scaling_factor=args.scaling_factor)
    else:
        grid = GridOfNeurons(across=args.across, rows=args.rows, weight=None, seed=int(arm["seed"]),
                             omega=args.omega, reach=args.grid_reach, permute=not args.no_permute,
                             threshold=args.threshold, minimum_potential=args.minimum_potential)
    grid.coding, grid.population, grid.clock = args.coding, args.population, args.clock
    grid.temperature = args.temperature  # the evidence critic's (§8)
    grid.readout, grid.read, grid.read_window = args.readout, args.read, args.read_window
    grid.teacher_threshold = args.teacher_threshold  # the count read's line (§4.3)
    grid.interval, grid.drive = args.interval, args.drive
    grid.input_rate, grid.input_rate_off = args.input_rate, args.input_rate_off
    grid.quash_rate, grid.quash_k = args.quash, args.quash_k
    grid.hebb_rate, grid.synapse_tau, grid.flip = args.hebb, args.synapse_tau, args.flip
    grid.rule = "reinforce"
    grid.set_delta(args.delta)  # escape noise (§5.2), once the thresholds are the container's
    return grid, args


def _save_network(engine, grid, report: dict, path) -> None:
    """The arm's network at the end of the run: the engine's weights and thresholds written back to the mesh and checkpointed.

    A run restored from it continues with those weights (the CLI's --load-weights, or
    docs/mnist-watch.py); it is not a resume to the bit, since the stream's position,
    the exploration stream's state and the baseline are not in a checkpoint.
    """
    from walnutbutter.persistence import checkpoint
    edges = [c for n in grid.all_neurons() for c in n.outgoing]  # the engine's order (fast.build)
    for c, w in zip(edges, engine.weights()):
        c.weight = w
    for n, theta in zip(grid.all_neurons(), report["thresholds"]):
        n.threshold = theta
    for n, expected in zip(grid.all_neurons(), report.get("expected_counts") or []):
        n.expected_count = expected  # the hebb eligibility's expectation (§6.7), so a continuation starts centred
    checkpoint(grid, path)


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
    arm, problem, epochs, trace_every, name, eligibility, scale, floor_ratio, wiring = job
    from walnutbutter import fast
    from walnutbutter.network import input_stream
    from walnutbutter.problems import PROBLEMS, dataset_stream

    out = ROOT / "runs" / name
    path = out / f"{arm_name(arm)}.csv"
    if path.exists():
        return {"arm": arm_name(arm), "skipped": True}
    grid, args = grid_of(problem, arm, eligibility, scale, floor_ratio, wiring)
    eligibility = arm.get("eligibility", eligibility)
    direction = None
    if PROBLEMS[problem].data == "mnist" and hasattr(grid, "outputs"):  # the estimator's correlation over time (§8)
        from walnutbutter.mnist import supervised_direction
        direction = supervised_direction(grid)
    first = grid.all_neurons()[0]
    started_at = {"theta": first.threshold, "floor": first.minimum_potential}  # what a neuron starts at, scaling applied
    data = dataset_stream(PROBLEMS[problem].data, int(arm["seed"]))  # a dataset's images with their labels (§8), or
    if data is None:  # random bits drawn up front, §4.5: every arm at this seed sees the same epochs in the same order
        patterns, labels = input_stream(epochs, grid.raw_bit_count(), int(arm["seed"])), None
    else:
        patterns, labels = data
    started = time.perf_counter()
    mean, trace, engine, report = fast.train(
        grid, epochs, lr=args.lr, target=args.target, trace_every=trace_every, patterns=patterns, labels=labels,
        eligibility=args.eligibility, sigma=args.sigma, seed=int(arm["seed"]),
        homeostasis=args.homeostasis, target_rate=args.target_rate, unstick=args.unstick,
        unstick_target=args.unstick_target, critic=args.critic, direction=direction,
    )
    elapsed = time.perf_counter() - started
    _save_network(engine, grid, report, path.with_name(path.stem + "-network.json"))  # so an arm can be resumed, not rerun
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "score"])
        for k, score in enumerate(trace, start=1):
            writer.writerow([k * trace_every, f"{score:.6g}"])
    result = {"arm": arm_name(arm), "mean": mean, "last_tenth": report["last_tenth"], "stuck_on": report["stuck_on"],
              "stuck_off": report["stuck_off"], "unstuck": report["unstuck"], "seconds": round(elapsed),
              "epochs_per_second": round(epochs / elapsed), "eligibility": eligibility, **started_at,
              "read": grid.read, "teacher_threshold": grid.teacher_threshold,  # what "on" meant at the read (§4.3)
              "critic": args.critic, "problem": problem,
              "threshold": args.threshold, "minimum_potential": args.minimum_potential, "floor_ratio": floor_ratio,
              "delta": args.delta,  # escape noise (§5.2), 0 when the threshold decided
              "escape_scale": grid.escape_scale,  # and the count's scaling of every hazard, sqrt(60 / N) (§5.2)
              "count_memory": COUNT_MEMORY if eligibility == "hebb" else None,  # the centred rule's memory (§6.7); a record
              # naming hebb without it is from before September 16, 2026, when hebb named the +-1 rule now called wrong_hebb
              "wiring": getattr(grid, "wiring", None),  # goo's rule (§3.4), and its knob
              "scaling_factor": getattr(grid, "scaling_factor", None),
              "temperature": grid.temperature if args.critic == "evidence" else None,  # the evidence critic's (§8), and
              "accuracy_last_tenth": report.get("accuracy_last_tenth"),  # the class critic's fraction right beside it
              "rate_by_zone": _rate_by_zone(grid, report["rates"]),  # the final rate memories, averaged over each zone
              "estimator": report.get("estimator"),  # the estimator's correlation over time (§8), for a dataset on goo
              "output_counts_last": _output_counts(engine, grid),  # the last epoch's output spikes, in order
              "container": repr(grid) if hasattr(grid, "count") else f"{args.across}x{args.rows} hex grid, omega {args.omega:g}"}
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
    on_goo = (bool(args.goo) or args.hidden_neurons is not None or PROBLEMS[args.problem].goo is not None
              or PROBLEMS[args.problem].hidden_neurons is not None)  # posed on goo: by the arms, or by the problem
    sizes = (" " + " ".join(f"{g:g}" for g in args.goo)) if args.goo else ""
    container = ("goo" + sizes + ("" if args.scale else ", flat")) if on_goo else "hex grid"
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
        workers = args.workers or min(len(arms), max(1, (os.cpu_count() or 2) - 1))
        print(f"{len(arms)} arms on {workers} workers, {args.epochs:,} epochs each, sweeping {swept}", flush=True)
        started = time.perf_counter()
        jobs = [(arm, args.problem, args.epochs, args.trace_every, args.name, args.eligibility[0], args.scale,
                 args.floor_ratio, args.wiring) for arm in arms]
        with Pool(workers) as pool:
            for result in pool.imap_unordered(run_arm, jobs):
                print(f"[{time.perf_counter() - started:6.0f}s] {json.dumps(result)}", flush=True)
    summarise(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
