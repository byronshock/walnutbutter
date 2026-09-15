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

The reinforce rule with the row critic and late = count. `--eligibility hebb` (the default,
sigma 0) or `--eligibility perturb`, which draws its noise per wave (§6.1) from the arm's
seed -- the same stream a Teacher with that seed would use -- and `--sigma` is then a knob
like any other.

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
    "threshold": "--threshold",  # THRESHOLD, quoted per THRESHOLD_FAN_IN incoming synapses; goo scales it (§5.2)
    "minimum_potential": "--minimum-potential",  # the floor; or derive it from the threshold with --floor-ratio
    "teacher_threshold": "--teacher-threshold",  # the count read's line, in Hz (§4.3)
    "goo": "--goo",  # goo (§3.4) in place of the grid, with this many neurons
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
    parser.add_argument("--no-scale-with-fan-in", dest="scale", action="store_false",
                        help="run goo at a flat threshold and floor instead of the §5.2 rescaling")
    parser.add_argument("--eligibility", choices=("hebb", "perturb"), default="hebb",
                        help="what the reward acts on (§6.7): a Hebbian +-1, or the perturbation the neuron decided under")
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--trace-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--summary", action="store_true", help="summarise what is on disk; run nothing")
    args = parser.parse_args()
    if args.goo == []:  # bare --goo: the working network's count
        from walnutbutter.constants import GOO_COUNT
        args.goo = [float(GOO_COUNT)]
    return args


def grid_of(problem: str, arm: dict, eligibility: str = "hebb", scale: bool = True, floor_ratio: float | None = None):
    """The network the command line would build for this problem, with the arm's knobs applied."""
    from walnutbutter.cli import apply_problem, build_parser
    from walnutbutter.goo import Goo
    from walnutbutter.grid import GridOfNeurons
    from walnutbutter.neuron import Neuron

    from walnutbutter.constants import GOO_MINIMUM_POTENTIAL, GOO_THRESHOLD, THRESHOLD

    argv = ["--problem", problem, "--eligibility", eligibility]
    base_threshold = GOO_THRESHOLD if "goo" in arm else THRESHOLD  # goo has its own (§1.2)
    if "goo" in arm and "threshold" not in arm:
        argv += ["--threshold", f"{GOO_THRESHOLD:g}"]
    if floor_ratio is not None:
        if "minimum_potential" in arm:
            raise ValueError("--floor-ratio derives the floor from the threshold; do not also sweep --minimum-potential")
        argv += ["--minimum-potential", f"{floor_ratio * float(arm.get('threshold', base_threshold)):g}"]
    elif "goo" in arm and "minimum_potential" not in arm:
        argv += ["--minimum-potential", f"{GOO_MINIMUM_POTENTIAL:g}"]
    for knob, value in arm.items():
        if knob in ("seed", "rows") or knob in DERIVED:
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
    if "goo" in arm:
        grid = Goo(count=int(arm["goo"]), across=args.across, weight=None, seed=int(arm["seed"]),
                   permute=not args.no_permute, threshold=args.threshold, minimum_potential=args.minimum_potential,
                   scale_with_fan_in=scale)
    else:
        grid = GridOfNeurons(across=args.across, rows=args.rows, weight=None, seed=int(arm["seed"]),
                             omega=args.omega, reach=args.grid_reach, permute=not args.no_permute,
                             threshold=args.threshold, minimum_potential=args.minimum_potential)
    grid.coding, grid.population = args.coding, args.population
    grid.readout, grid.read, grid.read_window = args.readout, args.read, args.read_window
    grid.teacher_threshold = args.teacher_threshold  # the count read's line (§4.3)
    grid.interval, grid.drive = args.interval, args.drive
    grid.input_rate, grid.input_rate_off = args.input_rate, args.input_rate_off
    grid.quash_rate, grid.quash_k = args.quash, args.quash_k
    grid.hebb_rate, grid.synapse_tau, grid.flip = args.hebb, args.synapse_tau, args.flip
    grid.rule = "reinforce"
    return grid, args


def arm_name(arm: dict) -> str:
    return "-".join(f"{knob}{value:g}" for knob, value in arm.items())


def run_arm(job: tuple) -> dict:
    arm, problem, epochs, trace_every, name, eligibility, scale, floor_ratio = job
    from walnutbutter import fast
    from walnutbutter.network import input_stream

    out = ROOT / "runs" / name
    path = out / f"{arm_name(arm)}.csv"
    if path.exists():
        return {"arm": arm_name(arm), "skipped": True}
    grid, args = grid_of(problem, arm, eligibility, scale, floor_ratio)
    first = grid.all_neurons()[0]
    started_at = {"theta": first.threshold, "floor": first.minimum_potential}  # what a neuron starts at, scaling applied
    patterns = input_stream(epochs, grid.raw_bit_count(), int(arm["seed"]))  # drawn up front, §4.5: every arm at this
    started = time.perf_counter()                                           # seed sees the same epochs in the same order
    mean, trace, _, report = fast.train(
        grid, epochs, lr=args.lr, target=args.target, trace_every=trace_every, patterns=patterns,
        eligibility=args.eligibility, sigma=args.sigma, seed=int(arm["seed"]),
        homeostasis=args.homeostasis, target_rate=args.target_rate, unstick=args.unstick,
        unstick_target=args.unstick_target,
    )
    elapsed = time.perf_counter() - started
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "score"])
        for k, score in enumerate(trace, start=1):
            writer.writerow([k * trace_every, f"{score:.6g}"])
    result = {"arm": arm_name(arm), "mean": mean, "last_tenth": report["last_tenth"], "stuck_on": report["stuck_on"],
              "stuck_off": report["stuck_off"], "unstuck": report["unstuck"], "seconds": round(elapsed),
              "epochs_per_second": round(epochs / elapsed), "eligibility": eligibility, **started_at,
              "read": grid.read, "teacher_threshold": grid.teacher_threshold,  # what "on" meant at the read (§4.3)
              "threshold": args.threshold, "minimum_potential": args.minimum_potential, "floor_ratio": floor_ratio,
              "container": repr(grid) if "goo" in arm else f"{args.across}x{args.rows} hex grid, omega {args.omega:g}"}
    path.with_suffix(".json").write_text(json.dumps(result))  # the summary the trace cannot give: the mean over the last tenth
    return result


def grid_and_arms(args):
    given = {knob: getattr(args, knob) for knob in KNOBS if knob != "seed" and getattr(args, knob) is not None}
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
    container = ("goo " + " ".join(f"{g:g}" for g in args.goo) + ("" if args.scale else ", flat")) if args.goo else "hex grid"
    lines = [f"# {args.name}: {args.problem} on the {container}, {args.epochs:,} epochs an arm, the Rust wave loop (§6.15), "
             f"{args.eligibility} eligibility", ""]
    if len(axes) == 2:
        x, y = axes
        xs = sorted({r["arm"][x] for r in rows})
        ys = sorted({r["arm"][y] for r in rows})
        lines += ["Mean score over the last tenth of each run, averaged over seeds.", "",
                  "| " + y + " \\ " + x + " | " + " | ".join(f"{v:g}" for v in xs) + " |",
                  "|" + "---|" * (len(xs) + 1)]
        for v in ys:
            cells = []
            for u in xs:
                got = [r["last_tenth"] for r in rows if r["arm"][x] == u and r["arm"][y] == v]
                cells.append(f"{statistics.fmean(got):.4f}" if got else "—")
            lines.append(f"| {v:g} | " + " | ".join(cells) + " |")
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
        jobs = [(arm, args.problem, args.epochs, args.trace_every, args.name, args.eligibility, args.scale,
                 args.floor_ratio) for arm in arms]
        with Pool(workers) as pool:
            for result in pool.imap_unordered(run_arm, jobs):
                print(f"[{time.perf_counter() - started:6.0f}s] {json.dumps(result)}", flush=True)
    summarise(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
