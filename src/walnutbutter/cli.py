"""Command-line entry point. Parses arguments, then hands off to the monitor."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from .butter import CELL_AREA
from .cartesian import CartesianNodes
from .columns import HexColumns
from .goo import DEFAULT_COUNT as GOO_COUNT, Goo
from .constants import GOO_MINIMUM_POTENTIAL, GOO_THRESHOLD
from .grid import GridOfNeurons
from .inputs import CODES, DEFAULT_CODE, parse_bits
from .constants import (
    ACROSS, BORED_AFTER, CRITIC, EXPLORE, FLIP, HEBB_RATE, INPUT_CV, INPUT_DRIVE, INPUT_RATE, INPUT_RATE_OFF, POPULATION,
    LEAKY_ELIGIBILITY, RATE_ON, RATE_TAU, READ_WINDOW, TEACHER_THRESHOLD,
    SYNAPSE_TAU, QUASH_K, QUASH_RATE, TAU, DOPAMINE_EXPECTATION_START, DOPAMINE_EXPECTATION_TAU, DOPAMINE_ORDER, DOPAMINE_PUNISH_GAIN, DOPAMINE_RELEASE_ALPHA, DOPAMINE_RELEASE_THETA,
    DOPAMINE_TAU, ELIGIBILITY, WEIGHT_DECAY, HOMEOSTASIS, INTERVAL, LATE, LR,
    MINIMUM_POTENTIAL, OMEGA, PROBLEM, REACH, REFRACTORY, REFRACTORY_HOPS, ROWS, RULE, SIGMA, TARGET, TARGET_RATE,
    THRESHOLD_FAN_IN,
    THRESHOLD, UNSTICK, UNSTICK_TARGET, WEIGHT_EPSILON, WEIGHT_RANGE,
    cv_for_rate, rate_for_cv,
)
from .dopamine import ORDERS, Dopamine
from .learning import CRITICS, ELIGIBILITIES, LATE_RULES, RULES, TARGETS, Teacher
from .monitor import main, run_epoch
from .network import input_stream
from .neuron import Neuron
from .persistence import across_of, checkpoint, restore, resume_teacher
from .problems import PROBLEMS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="walnutbutter",
        description=(
            "A hexagonal mesh of neurons that learns to reproduce its input on its output row. "
            "By default it opens a window, free-runs, learns, and reports accuracy until you close it."
        ),
    )
    parser.add_argument(
        "--problem",
        choices=sorted(PROBLEMS),
        default=PROBLEM,
        help="what the network is asked to do: "
        + "; ".join(f"{p.name} ({p.description})" for p in PROBLEMS.values())
        + f". Default: {PROBLEM}",
    )
    parser.add_argument(
        "-a",
        "--across",
        type=int,
        default=None,
        help=f"number of hexagons across (default: the problem's, {ACROSS} for {PROBLEM}; or twice the code length with --ecc)",
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        default=ROWS,
        help=f"number of hexagon rows (default: {ROWS})",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        metavar="N",
        nargs="?",
        const=0,
        default=None,
        help="Cartesian neurons instead of the hex grid: a --across x --rows hexagonal lattice at unit spacing, "
        "wired by distance (--receptive-field-sigma) and learning like the grid; or with N, that many neurons at "
        "random in a --across x --rows unit region, just shown",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        metavar="N",
        help="hexagonal columns in R3: the --across x --rows field of cells extruded into N layers, one neuron "
        "per cell per layer, half a unit apart in the plane and an eighth between layers. Butter within one unit "
        "horizontally always connects, at any "
        "height; the rest is --omega shortcuts. The bottom layer is the input and the top layer the output "
        "(with one layer: bottom row in, top row out, and the stack is the hex grid exactly)",
    )
    parser.add_argument(
        "--goo",
        type=int,
        nargs="?",
        const=GOO_COUNT,
        default=None,
        metavar="N",
        help=f"goo: N neurons with no positions at all, every ordered pair connected, no neighbourhood and no "
        f"shortcuts (default: {GOO_COUNT}, the working network since September 14, 2026). Goo has its own "
        f"threshold and floor, {GOO_THRESHOLD:g} and {GOO_MINIMUM_POTENTIAL:g} before its fan-in scaling, which "
        f"--threshold and --minimum-potential override (a value equal to the grid's default is taken as not given). "
        f"The first --across neurons are the input zone and the last --across the output, because with no rows "
        f"there is nowhere else to put them. --omega, --reach and --rows do not reach it, and there is no geometry "
        f"to draw, so it cannot be shown",
    )
    parser.add_argument(
        "--scale-with-fan-in",
        "--scale_with_fan_in",
        dest="scale_with_fan_in",
        action="store_true",
        default=None,
        help="rescale each neuron's potential axis by its in-degree over THRESHOLD_FAN_IN (AUTHORITY.md §5.2): "
        "--threshold and --minimum-potential are quoted at an interior hex cell's 18 incoming synapses, so a "
        "neuron with four times the fan-in starts four times as far from zero in both directions. On by default "
        "for goo and off for every other container, because turning it on for the grid would move every "
        "threshold every result to date was measured at",
    )
    parser.add_argument(
        "--direct-projection",
        "--direct_projection",
        dest="direct_projection",
        action="store_true",
        default=None,
        help="wire the input zone straight onto the output zone, as a fully connected goo does (AUTHORITY.md §3.4): "
        "the default unless the problem says otherwise -- copy does, so that a copy has to go through the interior",
    )
    parser.add_argument(
        "--no-direct-projection",
        "--no_direct_projection",
        dest="direct_projection",
        action="store_false",
        help="leave out every connection from an input neuron to an output neuron; only goo can be built so",
    )
    parser.add_argument(
        "--no-scale-with-fan-in",
        "--no_scale_with_fan_in",
        dest="scale_with_fan_in",
        action="store_false",
        help="run goo at a flat --threshold and --minimum-potential, as every other container does",
    )
    parser.add_argument(
        "--engine",
        choices=("objects", "arrays"),
        default=None,
        help="how the network is run: objects (default; every neuron and connection is an object and signals "
        "queue wave by wave) or arrays (the same network as numpy vectors and a scipy sparse matrix, much "
        "faster, needs numpy and scipy). A loaded checkpoint keeps its engine unless this is given.",
    )
    parser.add_argument(
        "--reach",
        type=float,
        default=REACH,
        metavar="UNITS",
        help=f"with --nodes: a neuron connects to every neuron within this many unit distances "
        f"(default: {REACH:g}, the two hex rings at unit density)",
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(800, 600),
        help="window or image size in pixels (default: 800 600)",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=float,
        default=None,
        help=f"fixed weight for every connection (default: random, uniform between {WEIGHT_RANGE[0]:g} and {WEIGHT_RANGE[1]:g})",
    )
    parser.add_argument(
        "-o",
        "--omega",
        type=float,
        default=OMEGA,
        help=f"proportion of connections that are small-world shortcuts, 0 to <1 (default: {OMEGA:g})",
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="BITS",
        default=None,
        help="raw input bits, one per half place, e.g. 1011 for 8 across (default: random)",
    )
    parser.add_argument(
        "--ecc",
        nargs="?",
        const=DEFAULT_CODE,
        default=None,
        choices=sorted(CODES),
        metavar="CODE",
        help="encode the 4 data bits with an error-correcting code before complement coding: hamming74 "
        "(the default with bare --ecc; corrects single errors; 14 across) or parity64 (detects only; "
        "12 across). Sets --across to fit unless given.",
    )
    parser.add_argument(
        "--no-permute",
        action="store_true",
        help="lay the complement-coded bits on the bottom row in order instead of scrambling them",
    )
    parser.add_argument(
        "--positive-weights",
        "--positive_weights",
        action="store_true",
        help="keep every weight between epsilon and 1: no inhibitory connections, before or after learning",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=WEIGHT_EPSILON,
        help=f"the smallest weight allowed under --positive-weights (default: {WEIGHT_EPSILON:g})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        metavar="N",
        default=None,
        help="run N seeds in parallel (consecutive from --seed, or from a random base), headless, for --epochs "
        "each; print a table sorted best first and checkpoint every run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for the random weights, so a run can be repeated (default: chosen and printed)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"input a neuron needs before it fires (default: {THRESHOLD:g})",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="window mode where nothing happens until you press Space for the next epoch (instead of free-running)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="no window: run --epochs epochs and exit",
    )
    parser.add_argument(
        "--report",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="while free-running, print a progress line (and record it in the checkpoint history) this often (default: 1)",
    )
    parser.add_argument(
        "--no-learn",
        action="store_true",
        help="do not teach the network (learning is on by default)",
    )
    parser.add_argument(
        "--rule",
        choices=RULES,
        default=None,
        help=f"the learning rule (AUTHORITY.md §6): teacher (an external teacher scores the read and pays the epoch's "
        f"eligibility), adaline (Widrow-Hoff: each scored neuron's own error times what each synapse delivered), "
        f"dopamine (the neurons pay themselves as they refire), or reinforce (the pre-alpha's rule). "
        f"Default: the problem's, else {RULE}",
    )
    parser.add_argument(
        "--dopamine-tau",
        type=float,
        default=DOPAMINE_TAU,
        metavar="MS",
        help=f"decay time constant of the global dopamine value (default: {DOPAMINE_TAU:g})",
    )
    parser.add_argument(
        "--expectation-tau",
        type=float,
        default=DOPAMINE_EXPECTATION_TAU,
        metavar="MS",
        help=f"exponential window of the expected dopamine trace, which starts at 0 (default: {DOPAMINE_EXPECTATION_TAU:g} ms, 10 minutes)",
    )
    parser.add_argument(
        "--expectation-start",
        type=float,
        default=DOPAMINE_EXPECTATION_START,
        metavar="UNITS",
        help=f"where the expected dopamine trace starts (default: {DOPAMINE_EXPECTATION_START:g}); a high start holds early learning back",
    )
    parser.add_argument(
        "--release-alpha",
        type=float,
        default=DOPAMINE_RELEASE_ALPHA,
        metavar="SHAPE",
        help=f"shape of the gamma density that gives the amount a refire releases against its delay past the refractory "
        f"period (default: {DOPAMINE_RELEASE_ALPHA:g})",
    )
    parser.add_argument(
        "--release-theta",
        type=float,
        default=DOPAMINE_RELEASE_THETA,
        metavar="MS",
        help=f"scale of that gamma; the release peaks (alpha - 1) * theta past the end of the refractory period "
        f"(default: {DOPAMINE_RELEASE_THETA:g})",
    )
    parser.add_argument(
        "--order",
        choices=ORDERS,
        default=DOPAMINE_ORDER,
        help=f"at a refire, release the dopamine before the weight update or after it (default: {DOPAMINE_ORDER})",
    )
    parser.add_argument(
        "--quash",
        type=float,
        default=None,
        metavar="RATE",
        help=f"cycles are quashed (AUTHORITY.md §6.11): a refire weakens each contributing synapse by this fraction of "
        f"its weight, falling off with the delay since its previous spike (default: the problem's, {QUASH_RATE:g} where "
        f"it quashes; 0 = off)",
    )
    parser.add_argument(
        "--leaky",
        action="store_true",
        default=LEAKY_ELIGIBILITY,
        help="append the leaky trace of AUTHORITY.md §6.12 to the reinforce rule's chain, so the global reward reaches "
        "each synapse in proportion to the charge it still had in its target when that target's state was read",
    )
    parser.add_argument(
        "--synapse-tau",
        type=float,
        default=SYNAPSE_TAU,
        metavar="MS",
        help=f"the leak of the eligibility trace on a synapse (AUTHORITY.md §6.12), the synapse's own and no longer the "
        f"neuron's: it governs leaky Hebb and the reinforce rule's leaky eligibility alike (default: {SYNAPSE_TAU:g} ms)"
    )
    parser.add_argument(
        "--hebb",
        type=float,
        default=None,
        metavar="RATE",
        help=f"leaky Hebb (AUTHORITY.md §6.12): a firing neuron potentiates each synapse that still had charge in it by "
        f"this much times its leaky trace. Composes with whatever else runs (default: the problem's, {HEBB_RATE:g} where "
        f"it runs it; 0 = off)",
    )
    parser.add_argument(
        "--read",
        choices=("fired", "again", "window", "rate", "count"),
        default=None,
        help="what the teacher reads at the end of an epoch (AUTHORITY.md §4.3): fired this epoch, spiked again after "
        "the input's moment, fired within the read window, rate (the exponential-window firing-rate estimate scored "
        "against RATE_ON and RATE_OFF), or count: the epoch's spikes counted, a rate estimated from the count, and the "
        "neuron on if that exceeds --teacher-threshold (default: the problem's)",
    )
    parser.add_argument(
        "--teacher-threshold",
        "--teacher_threshold",
        dest="teacher_threshold",
        type=float,
        default=TEACHER_THRESHOLD,
        metavar="HZ",
        help=f"the count read's line between off and on, in Hz (default: {TEACHER_THRESHOLD:g}, the middle of the "
        f"one-spike band: at a 35 ms epoch one spike is 28.6 Hz and reads on, none reads off; 42.9 would mean two)",
    )
    parser.add_argument(
        "--read-window",
        type=float,
        default=None,
        metavar="MS",
        help=f"the window of the window read: a neuron is on if it spiked this recently before the epoch's end "
        f"(default: the problem's, else {READ_WINDOW:g} ms)",
    )
    parser.add_argument(
        "--rate-tau",
        type=float,
        default=RATE_TAU,
        metavar="MS",
        help=f"the exponential window the rate read estimates over (default: {RATE_TAU:g} ms)",
    )
    parser.add_argument(
        "--rate-on",
        type=float,
        default=RATE_ON,
        metavar="HZ",
        help=f"the rate an output the target says should be on is driven to; off is silence (default: {RATE_ON:g} Hz, "
        f"which is 1/REFRACTORY: saturation)",
    )
    parser.add_argument(
        "--explore",
        choices=("wave", "epoch"),
        default=EXPLORE,
        help=f"when the exploration draw is taken (AUTHORITY.md §6.1): wave, afresh before every firing decision so a "
        f"neuron's noise is what it decided under, or epoch, once at the input's moment, which is the pre-alpha's "
        f"(default: {EXPLORE})",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        metavar="N",
        help=f"neurons per raw bit where the coding repeats it (AUTHORITY.md §4.3): population coding uses this many, and "
        f"population-complement uses this many and then complement-codes the lot (default: the problem's, else {POPULATION})",
    )
    parser.add_argument(
        "--drive",
        choices=("forced", "rate"),
        default=None,
        help=f"how a bit becomes spikes (AUTHORITY.md §4.3): forced, one spike at the epoch's moment, or rate, an "
        f"independent Poisson process across the epoch so a bit is a firing rate and a bit-1 neuron may produce no "
        f"spike at all (default: the problem's, else {INPUT_DRIVE})",
    )
    parser.add_argument(
        "--cv",
        type=float,
        default=INPUT_CV,
        metavar="CV",
        help=f"the coefficient of variation of the input spike train, which is how the drive is specified "
        f"(AUTHORITY.md §4.3): the rate follows as (1 - CV)/REFRACTORY, so {INPUT_CV:g} is "
        f"{1000.0 * (1.0 - INPUT_CV) / REFRACTORY:.0f} Hz. 0 would need infinite drive and 1 none "
        f"(default: {INPUT_CV:g})",
    )
    parser.add_argument(
        "--input-rate",
        type=float,
        default=None,
        metavar="PER_MS",
        help=f"the drive in its other coordinate: the rate of the Poisson process that DRIVES a bit-1 neuron -- not "
        f"the rate it fires at, since arrivals inside the refractory period are dropped and it fires at the first one "
        f"after, giving a mean interspike interval of REFRACTORY + 1/rate. Overrides --cv "
        f"(default: whatever --cv works out to, {INPUT_RATE:g}/ms)",
    )
    parser.add_argument(
        "--input-rate-off",
        type=float,
        default=INPUT_RATE_OFF,
        metavar="PER_MS",
        help=f"the driving rate for a bit-0 neuron (default: {INPUT_RATE_OFF:g}/ms, silence)",
    )
    parser.add_argument(
        "--flip",
        type=float,
        default=None,
        metavar="P",
        help=f"corrupt the input (AUTHORITY.md §4.3): flip each coded bit with this probability before the row is forced, "
        f"and score the read against the clean pattern (default: the problem's, {FLIP:g} where it corrupts; 0 = off)",
    )
    parser.add_argument(
        "--quash-k",
        type=float,
        default=QUASH_K,
        metavar="PER_MS",
        help=f"how fast the quash falls off with that delay, exp(-k * delay) (default: {QUASH_K:g})",
    )
    parser.add_argument(
        "--punish",
        action="store_true",
        help="reverse a bit-0 input neuron's update even under the teacher rule, whose score already knows about them",
    )
    parser.add_argument(
        "--no-punish",
        action="store_true",
        help="do not reverse the update of an input neuron that should not fire (by default, above-expected dopamine "
        "punishes a bit-0 input neuron for refiring instead of rewarding it)",
    )
    parser.add_argument(
        "--punish-gain",
        type=float,
        default=DOPAMINE_PUNISH_GAIN,
        metavar="GAIN",
        help=f"a should-not-fire refire is punished this many times as hard as a refire is rewarded (default: {DOPAMINE_PUNISH_GAIN:g})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        metavar="FRACTION",
        help=f"every weight moves toward 0 by this fraction each epoch: synapses that forget (default: {WEIGHT_DECAY:g}; 0 = off)",
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default=TARGET,
        help=f"what the top row should show, derived from the input row (default: {TARGET})",
    )
    parser.add_argument(
        "--late",
        choices=LATE_RULES,
        default=LATE,
        help=f"what a signal arriving after its target has already fired earns: count (the same update "
        "as one that landed, a local Hebbian term under the global reward), ignore (nothing: the "
        "node-perturbation estimator proper) or depress (the opposite update, the shape of spike-timing-dependent "
        f"plasticity). Default: {LATE}",
    )
    parser.add_argument(
        "--critic",
        choices=sorted(CRITICS),
        default=None,
        help=f"how the reward is judged: row (fraction of output neurons matching the target), sustained (of the "
        f"neurons the target says should be on, the fraction on: did the forced neurons sustain?), decoded "
        f"(read the row as a word, error-correct it, fraction of data bits right), or decoded-exact "
        f"(all data bits right or nothing). Default: {CRITIC}, or the problem's",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"learning rate, either rule (default: {LR:g})",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA,
        help=f"exploration noise: std dev added to each neuron's potential at every input (default: {SIGMA:g}; 0 = off)",
    )
    parser.add_argument(
        "--eligibility",
        choices=ELIGIBILITIES,
        default=ELIGIBILITY,
        help=f"what the global reward acts on: the neuron's exploration noise (perturb) or plain Hebbian (default: {ELIGIBILITY})",
    )
    parser.add_argument(
        "--homeostasis",
        type=float,
        default=HOMEOSTASIS,
        metavar="RATE",
        help=f"per-epoch rate at which each neuron's threshold moves toward its target firing rate (default: {HOMEOSTASIS:g}; 0 = off)",
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=TARGET_RATE,
        help=f"firing rate homeostasis aims for, 0 to 1 (default: {TARGET_RATE:g})",
    )
    parser.add_argument(
        "--unstick",
        type=float,
        default=UNSTICK,
        metavar="RATE",
        help=f"per-epoch rate at which a stuck output neuron's threshold moves toward --unstick-target; only "
        f"output neurons firing >99%% or <1%% of the time are touched, only while stuck (default: {UNSTICK:g}; 0 = off)",
    )
    parser.add_argument(
        "--unstick-target",
        type=float,
        default=UNSTICK_TARGET,
        help=f"firing rate the output un-sticking aims for (default: {UNSTICK_TARGET:g})",
    )
    parser.add_argument(
        "--minimum-potential",
        type=float,
        default=MINIMUM_POTENTIAL,
        help=f"floor on a neuron's potential: inhibition and carried-over charge can go no lower (default: {MINIMUM_POTENTIAL:g})",
    )
    parser.add_argument(
        "--discharge",
        action="store_true",
        help="zero every potential between inputs (the old epoch-by-epoch behaviour) instead of keeping it",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        metavar="MS",
        help=f"nominal milliseconds between inputs: the epoch's length (default: the problem's, else {INTERVAL:g}). "
        "Each epoch runs the schedule up to the next input's time; signals still in flight then join the next epoch",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=TAU,
        metavar="MS",
        help=f"leak time constant of every neuron, nominal milliseconds (default: {TAU:g}; inf switches the leak off). "
        "The leak is computed only when a signal reaches a neuron",
    )
    parser.add_argument(
        "--refractory",
        type=float,
        default=REFRACTORY,
        metavar="MS",
        help=f"absolute refractory period of every neuron, nominal milliseconds (default: {REFRACTORY:g})",
    )
    parser.add_argument(
        "--bored-after",
        type=float,
        default=BORED_AFTER,
        metavar="MS",
        help=f"threshold homeostasis: a neuron's threshold falls with its silence, reaching zero after this many ms "
        f"without a spike, so a bored neuron fires on its own (default: {BORED_AFTER:g}; 0 = off)",
    )
    parser.add_argument(
        "--refractory-hops",
        type=float,
        default=REFRACTORY_HOPS,
        metavar="RATIO",
        help=f"the refractory period divided by the time a signal takes to travel one hop; not an integer "
        f"(default: {REFRACTORY_HOPS:g}, so a hop is {REFRACTORY / REFRACTORY_HOPS:g} ms)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="with --headless: how many epochs to run, printing accuracy along the way (default: 1)",
    )
    parser.add_argument(
        "--input-seed",
        type=int,
        default=None,
        help="draw the run's inputs up front from a stream of their own, keyed to this seed (AUTHORITY.md §4.5), so "
             "two runs that differ in anything else still see the same epochs in the same order; without it the bits "
             "come from the network's own stream, which a differently built network consumes differently",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print a line for every epoch's input and for every neuron that fires (slow; off by default)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="accepted for compatibility: runs are quiet unless --verbose",
    )
    parser.add_argument(
        "--trace",
        metavar="FILE",
        help="CSV the Teacher appends one line per epoch to: epoch, time, dopamine, expected, score "
        "(default: next to the checkpoint, runs/<date>-<time>-seed<seed>.csv; --no-save skips it)",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="do not write the per-epoch trace",
    )
    parser.add_argument(
        "--save-weights",
        metavar="FILE",
        help="checkpoint file, written at every progress report and on exit "
        "(default: runs/<date>-<time>-seed<seed>.json)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="do not write a checkpoint",
    )
    parser.add_argument(
        "--load-weights",
        metavar="FILE",
        help="start from a checkpoint: rebuilds its mesh (size, omega, seed, permutation) and loads its weights",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="write a picture of the grid to PATH (e.g. grid.png)",
    )
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code (0 = success)."""
    args = build_parser().parse_args(argv)
    args.given_rule = args.rule  # what --rule said, if anything: it outlives a checkpoint's problem
    # a seed batch is headless by definition, and so is goo: it has no positions, so there is nothing to draw
    args.show = not args.headless and args.seeds is None and args.goo is None
    args.fast = args.show and not args.step
    if args.goo is not None and args.save:
        print("error: goo has no geometry to draw, so --save has no picture to write", file=sys.stderr)
        return 2
    if args.goo is not None and (args.nodes is not None or args.layers is not None):
        other = "--nodes" if args.nodes is not None else "--layers"
        print(f"error: --goo is a container of its own; it cannot be combined with {other}", file=sys.stderr)
        return 2
    try:
        apply_problem(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    apply_container(args)
    was_verbose, was_refractory, was_hops, was_bored, was_tau = Neuron.verbose, Neuron.refractory, Neuron.refractory_hops, Neuron.bored_after, Neuron.tau
    Neuron.verbose = bool(args.verbose) and not args.fast and not args.quiet
    if args.refractory <= 0 or args.refractory_hops <= 0 or args.interval <= 0 or args.tau <= 0:
        print("error: --tau, --refractory, --refractory-hops and --interval must be positive", file=sys.stderr)
        return 2
    if args.bored_after < 0 or (args.quash is not None and args.quash < 0) or args.quash_k < 0:
        print("error: --bored-after, --quash and --quash-k must not be negative", file=sys.stderr)
        return 2
    if args.dopamine_tau <= 0 or args.release_alpha <= 0 or args.release_theta <= 0 or args.expectation_tau <= 0:
        print("error: --dopamine-tau, --release-alpha, --release-theta and --expectation-tau must be positive", file=sys.stderr)
        return 2
    if args.punish_gain < 0 or not 0 <= args.weight_decay < 1:
        print("error: --punish-gain must not be negative and --weight-decay must be in [0, 1)", file=sys.stderr)
        return 2
    Neuron.refractory, Neuron.refractory_hops, Neuron.bored_after, Neuron.tau = args.refractory, args.refractory_hops, args.bored_after, args.tau
    try:
        return _run(args)
    finally:
        Neuron.verbose, Neuron.refractory, Neuron.refractory_hops, Neuron.bored_after, Neuron.tau = was_verbose, was_refractory, was_hops, was_bored, was_tau


def READ_MEANS(args) -> str:
    """How the banner says what "on" means at the read (AUTHORITY.md §4.3)."""
    if args.read == "again":
        return "spiked again after the input"
    if args.read == "fired":
        return "fired this epoch"
    if args.read == "rate":
        return (f"a firing rate over a {args.rate_tau:g} ms window, scored against {args.rate_on:g} Hz on and 0 Hz off "
                f"(§6.9: the targets are saturation and silence)")
    return f"fired in the last {args.read_window:g} ms"


def apply_problem(args: argparse.Namespace) -> None:
    """Settle what the problem decides: whether a Teacher scores or trains, the epoch's length, the target, the readout."""
    problem = PROBLEMS[args.problem]
    if args.input_rate is None:  # the drive is given as a CV; lambda is the coordinate the schedule wants (§4.3)
        # With no refractory period there is no dead time, so the train is Poisson and its CV is 1 whatever the rate:
        # the CV coordinate does not reach it. Such a run is rejected further down; leave the default and let it be.
        args.input_rate = rate_for_cv(args.cv, args.refractory) if args.refractory > 0 else INPUT_RATE
    args.cv = cv_for_rate(args.input_rate, args.refractory)  # and --input-rate, if given, sets the CV it implies
    if args.across is None:
        args.across = 2 * CODES[args.ecc].code_bits if args.ecc else problem.across
    args.learn = not args.no_learn  # a Teacher scores every problem; whether it may train is the problem's
    if not problem.trained:
        args.homeostasis, args.unstick = 0.0, 0.0  # nothing outside the network moves a threshold, whichever rule runs (§6.7)
    if problem.target is not None:
        args.target = problem.target
    if args.critic is None:
        args.critic = problem.critic if problem.critic is not None else CRITIC  # --critic overrides the problem's
    if args.rule is None:
        args.rule = problem.rule or RULE
    if args.rule in ("teacher", "adaline"):
        args.no_punish = not args.punish  # the score already knows which neurons should not have fired (§6.9, §6.10)
    if args.quash is None:
        args.quash = QUASH_RATE if problem.quash else 0.0
    if args.hebb is None:
        args.hebb = HEBB_RATE if problem.hebb else 0.0
    if args.drive is None:
        args.drive = problem.drive if problem.drive is not None else INPUT_DRIVE
    if args.population is None:
        args.population = problem.population if problem.population is not None else POPULATION
    if args.flip is None:
        args.flip = problem.flip if problem.flip is not None else 0.0
    if args.interval is None:
        args.interval = problem.interval if problem.interval is not None else INTERVAL
    args.readout, args.coding = problem.readout, problem.coding
    if args.read_window is None:
        args.read_window = problem.read_window
    if args.read is None:
        args.read = problem.read  # --read overrides what the problem asks for
    if args.direct_projection is None:
        args.direct_projection = problem.direct_projection  # --direct-projection / --no- override the problem
    if args.read == "window" and args.read_window is None:
        args.read_window = READ_WINDOW
    args.grid_reach, args.input_cells = problem.reach, problem.input_cells
    args.no_permute = args.no_permute or not problem.permute
    if args.rows == ROWS and problem.rows != ROWS:
        args.rows = problem.rows  # the problem's rows, unless --rows was given (a value equal to the default is taken as not given)


def apply_container(args: argparse.Namespace) -> None:
    """Goo's own threshold and floor (AUTHORITY.md §1.2) unless --threshold or --minimum-potential was given.

    A value equal to the grid's default is taken as not given, as --rows is;
    the sweep driver, which knows what an arm swept, decides for itself.
    """
    if args.goo is None:
        return
    if args.threshold == THRESHOLD:
        args.threshold = GOO_THRESHOLD
    if args.minimum_potential == MINIMUM_POTENTIAL:
        args.minimum_potential = GOO_MINIMUM_POTENTIAL


def _run(args: argparse.Namespace) -> int:
    try:
        if args.show or args.save:
            try:
                from . import visualizer
            except ImportError:
                print(
                    "error: the visualizer needs pygame; install it with: pip install -e '.[viz]'",
                    file=sys.stderr,
                )
                return 2
        width, height = args.window
        if args.nodes is not None and args.nodes < 0:
            print(f"error: --nodes cannot be negative, got {args.nodes}", file=sys.stderr)
            return 2
        if args.nodes is not None and args.nodes > 0:
            return _run_nodes(args, width, height)  # a random scatter: shown, not learnable (no rows)
        if args.seeds is not None:
            return _run_seeds(args)  # grid, or the lattice with bare --nodes
        loaded = None
        if args.load_weights:
            try:
                loaded = restore(args.load_weights)
            except (OSError, ValueError, KeyError) as exc:
                print(f"error: cannot load {args.load_weights}: {exc}", file=sys.stderr)
                return 2
            grid_from_file, data = loaded
            args.seed = data["seed"]
            args.across, args.rows, args.omega = across_of(data), data["rows"], data["omega"]
            if data.get("problem") and data["problem"] != args.problem:
                args.problem = data["problem"]
                args.interval = None if PROBLEMS[args.problem].interval is not None else args.interval
                args.rule = args.given_rule  # --rule if it was given, else the problem's
                apply_problem(args)
                print(f"problem: {args.problem} (from the checkpoint)", file=sys.stderr)
            if data.get("container") == "lattice":
                args.nodes = 0
            args.ecc = _ecc_from_checkpoint(data)
            args.threshold = data["threshold"]
            args.minimum_potential = data.get("minimum_potential", -1.0)
            args.weight = None if data["random_weights"] else data["weight"]
            low, high = data.get("weight_range", (-1.0, 1.0))
            args.positive_weights, args.epsilon = low > 0, low
            print(
                f"loaded {args.load_weights}: {args.across}x{args.rows}, seed {args.seed}, "
                f"{data['epoch']:,} epochs so far",
                file=sys.stderr,
            )
        seed = args.seed if args.seed is not None else random.randrange(2**31)
        if args.save_weights is None and not args.no_save:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            args.save_weights = str(Path("runs") / f"{stamp}-seed{seed}.json")
        if args.save_weights:
            Path(args.save_weights).parent.mkdir(parents=True, exist_ok=True)
            print(f"checkpointing to {args.save_weights}", file=sys.stderr)
        settings = dict(
            across=args.across,
            rows=args.rows,
            weight=args.weight,
            threshold=args.threshold,
            seed=seed,
            omega=args.omega,
            permute=not args.no_permute,
            weight_range=(args.epsilon, 1.0) if args.positive_weights else WEIGHT_RANGE,
            minimum_potential=args.minimum_potential,
        )
        if args.minimum_potential >= args.threshold:
            print(f"error: minimum potential ({args.minimum_potential}) must be below the threshold ({args.threshold})", file=sys.stderr)
            return 2
        if args.positive_weights and not 0 < args.epsilon < 1:
            print(f"error: epsilon must be between 0 and 1, got {args.epsilon}", file=sys.stderr)
            return 2
        if args.positive_weights:
            print(f"positive weights: every weight kept between {args.epsilon:g} and 1", file=sys.stderr)
        if args.weight is None or args.omega > 0 or args.input is None:
            print(f"seed {seed}", file=sys.stderr)

        try:
            input_bits = parse_bits(args.input) if args.input is not None else None
            if loaded:
                grid, data = loaded
            elif args.goo is not None:
                if args.goo < args.across:
                    print(
                        f"error: --goo needs at least as many neurons as the zones are wide, got {args.goo} "
                        f"for {args.across} across",
                        file=sys.stderr,
                    )
                    return 2
                grid = Goo(
                    count=args.goo, across=args.across, weight=args.weight, threshold=args.threshold, seed=seed,
                    permute=not args.no_permute, weight_range=settings["weight_range"],
                    minimum_potential=args.minimum_potential,
                    scale_with_fan_in=args.scale_with_fan_in is not False,
                    direct=args.direct_projection is not False,
                )
                print(f"{grid!r}: {grid.mean_out_degree():.1f} outgoing per neuron"
                      + ("" if grid.direct else f" -- {len(grid.connections)} connections, none from the input zone to the output zone"),
                      file=sys.stderr)
                first = grid.all_neurons()[0]
                if grid.scale_with_fan_in_on:
                    print(
                        f"fan-in scaling (§5.2): x{grid.fan_in_scale():.2f} on the potential axis, so threshold "
                        f"{first.threshold:.3f} and floor {first.minimum_potential:.3f}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"fan-in scaling off: a flat threshold {first.threshold:g} and floor "
                        f"{first.minimum_potential:g} at {grid.count - 1} incoming synapses",
                        file=sys.stderr,
                    )
                if grid.zones_overlap():
                    print("goo: the input and output zones overlap -- it is reading what it writes", file=sys.stderr)
            elif args.layers is not None:
                if args.layers < 1:
                    print(f"error: --layers needs at least 1, got {args.layers}", file=sys.stderr)
                    return 2
                grid = HexColumns(layers=args.layers, **settings)
                print(f"{grid!r}: input {len(grid.input_row())} neurons, output {len(grid.output_row())}", file=sys.stderr)
            elif args.nodes is not None:
                if args.reach < 0:
                    print(f"error: --reach must not be negative, got {args.reach}", file=sys.stderr)
                    return 2
                grid = CartesianNodes(
                    across=args.across, rows=args.rows, seed=seed, threshold=args.threshold,
                    minimum_potential=args.minimum_potential, permute=not args.no_permute,
                    weight_range=settings["weight_range"],
                )
                grid.connect_within(reach=args.reach, weight=args.weight)
            else:
                grid = GridOfNeurons(**settings, reach=args.grid_reach)
            if args.direct_projection is False and args.goo is None and not loaded:
                print("error: only goo can be built with no direct projection from its inputs to its outputs; use --goo, "
                      "or --direct-projection to run this problem on the grid as wired", file=sys.stderr)
                return 2
            if args.scale_with_fan_in and args.goo is None and not loaded:
                grid.scale_with_fan_in(args.threshold, args.minimum_potential)
                print(
                    f"fan-in scaling (§5.2): every threshold and floor rescaled by in-degree / "
                    f"{THRESHOLD_FAN_IN:g}",
                    file=sys.stderr,
                )
            if args.input_cells and not loaded:
                grid.set_input_cells(args.input_cells)
            grid.interval = args.interval
            grid.problem = args.problem
            grid.readout, grid.read, grid.read_window, grid.coding = args.readout, args.read, args.read_window, args.coding
            grid.quash_rate, grid.quash_k = args.quash, args.quash_k
            grid.hebb_rate, grid.synapse_tau = args.hebb, args.synapse_tau
            grid.drive, grid.input_rate, grid.input_rate_off = args.drive, args.input_rate, args.input_rate_off
            grid.explore, grid.rate_on = args.explore, args.rate_on
            grid.teacher_threshold = args.teacher_threshold
            grid.population = args.population
            Neuron.rate_tau = args.rate_tau
            if args.input_seed is not None:
                grid.use_input_stream(input_stream(max(1, args.epochs), grid.raw_bit_count(), args.input_seed))
                print(f"inputs: {max(1, args.epochs):,} patterns drawn up front from input seed {args.input_seed} "
                      f"(§4.5), the same for any network run at this input seed", file=sys.stderr)
            if args.drive == "rate":
                isi = args.refractory + 1.0 / args.input_rate if args.input_rate else float("inf")
                print(f"input drive: rate at CV {args.cv:.3g} (AUTHORITY.md §4.3) -- a Poisson process DRIVES each "
                      f"input neuron across the {args.interval:g} ms epoch at {args.input_rate:g}/ms of arrivals where "
                      f"its bit is 1 and {args.input_rate_off:g}/ms where it is 0. Arrivals inside the refractory "
                      f"period are dropped, so a bit-1 neuron fires every {isi:.2f} ms on average "
                      f"({1000.0 / isi:.0f} Hz, {args.interval / isi:.1f} spikes an epoch), at a coefficient of "
                      f"variation of {args.cv:.2f} against a Poisson train's 1", file=sys.stderr)
            grid.flip = args.flip
            if not PROBLEMS[args.problem].trained:
                print(
                    f"problem {args.problem}: {PROBLEMS[args.problem].description}. Epochs {args.interval:g} ms apart; "
                    f"scored on the {args.readout} row against {args.target} by the {args.critic} critic, on meaning "
                    f"{READ_MEANS(args)}; "
                    "nothing outside the network trains it",
                    file=sys.stderr,
                )
            if args.ecc:
                try:
                    grid.use_ecc(args.ecc)
                except ValueError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2
                code = CODES[args.ecc]
                print(
                    f"input: {code.data_bits} data bits -> {code.name} ({code.code_bits}, {code.data_bits}) code, "
                    f"{'corrects' if code.corrects_single_errors else 'detects'} single errors -> complement code "
                    f"-> {2 * code.code_bits} across",
                    file=sys.stderr,
                )
            if isinstance(grid, CartesianNodes):
                print(
                    f"{grid!r}, wired: {len(grid.connections)} one-way connections, every pair within "
                    f"{grid.reach:g} units ({grid.mean_out_degree():.1f} per neuron)",
                    file=sys.stderr,
                )
            if not args.no_permute or loaded:
                laid = "coded" if grid.coding == "complement" else "raw"
                where = "the input zone" if isinstance(getattr(grid, "mesh", grid), Goo) else "the bottom row"
                print(f"input permutation: place i along {where} shows {laid} bit {grid.permutation}", file=sys.stderr)
            if args.rule != "reinforce":
                grid.rule = args.rule
                if grid.dopamine is None:  # a loaded checkpoint brings its own pool
                    grid.dopamine = Dopamine(tau=args.dopamine_tau, release_alpha=args.release_alpha, release_theta=args.release_theta,
                                             order=args.order, lr=args.lr, expectation_tau=args.expectation_tau,
                                             punish=not args.no_punish, punish_gain=args.punish_gain, decay=args.weight_decay,
                                             expectation_start=args.expectation_start)
                print(f"rule: {args.rule}, {grid.dopamine.order}, tau {grid.dopamine.tau:g} ms, release gamma(alpha "
                      f"{grid.dopamine.release_alpha:g}, theta {grid.dopamine.release_theta:g} ms), expectation tau "
                      f"{grid.dopamine.expectation_tau:g} ms from {grid.dopamine.expectation:g}, lr {grid.dopamine.lr:g}, "
                      f"{f'bit-0 input neurons punished {grid.dopamine.punish_gain:g}x for refiring' if grid.dopamine.punish else 'no punishment'}, "
                      f"weight decay {grid.dopamine.decay:g} per epoch; hop {Neuron.hop():g} ms, tau {Neuron.tau:g} ms, "
                      f"bored after {Neuron.bored_after:g} ms, "
                      f"{f'quash {args.quash:g} falling off at {args.quash_k:g}/ms' if args.quash else 'no quash'}"
                      f"{f', leaky Hebb {args.hebb:g}' if args.hebb else ''}", file=sys.stderr)
            else:
                grid.dopamine = None  # the reinforce rule keeps no pool, so §6.8's weight decay does not run under it
                # the Teacher zeroes sigma for the hebb eligibility, so the network is deterministic: report what runs
                effective_sigma = args.sigma if args.eligibility == "perturb" else 0.0
                print(f"rule: reinforce ({args.eligibility} eligibility"
                      f"{' + leaky trace' if args.leaky else ''}, late signals {args.late}), lr {args.lr:g}, "
                      f"sigma {effective_sigma:g} ({args.explore})"
                      f"{' (no exploration: reward-modulated Hebb, not a policy gradient)' if not effective_sigma else ''}"
                      f", no weight decay; hop {Neuron.hop():g} ms, tau {Neuron.tau:g} ms, "
                      f"bored after {Neuron.bored_after:g} ms, "
                      f"{f'quash {args.quash:g} falling off at {args.quash_k:g}/ms' if args.quash else 'no quash'}"
                      f"{f', leaky Hebb {args.hebb:g}' if args.hebb else ''}", file=sys.stderr)
            engine = args.engine or (data.get("engine", "objects") if loaded else "objects")
            if engine == "arrays":
                try:
                    from .arrays import ArrayNetwork
                except ImportError:
                    print("error: the array engine needs numpy and scipy; install them with: pip install -e '.[arrays]'", file=sys.stderr)
                    return 2
                grid = ArrayNetwork(grid)
                print(f"engine: arrays ({len(grid)} neurons, {len(grid.weight)} connections as vectors)", file=sys.stderr)
            teacher = None
            if args.learn:
                teacher = Teacher(
                    grid,
                    target=args.target,
                    lr=args.lr,
                    sigma=args.sigma,
                    eligibility=args.eligibility,
                    seed=seed,
                    homeostasis=args.homeostasis,
                    target_rate=args.target_rate,
                    discharge=args.discharge,
                    unstick=args.unstick,
                    unstick_target=args.unstick_target,
                    critic=args.critic,
                    late=args.late,
                    leaky=args.leaky,
                    rule=args.rule,
                )
                if loaded:
                    resume_teacher(teacher, data)
                if args.trace is None and args.save_weights and not args.no_trace:
                    args.trace = str(Path(args.save_weights).with_suffix(".csv"))
                if args.trace and not args.no_trace:
                    Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
                    teacher.trace_to(args.trace)
                    print(f"tracing every epoch to {args.trace}", file=sys.stderr)
                teacher.epoch(input_bits, verbose=Neuron.verbose)  # the first epoch, with exploration, like every other
            else:
                explore = random.Random(seed)  # the exploration noise of an untrained run
                run_epoch(grid, input_bits, verbose=Neuron.verbose, noise=args.sigma, rng=explore, discharge=args.discharge)

            def save_checkpoint():
                if args.save_weights:
                    checkpoint(grid, args.save_weights, teacher)
            if args.show:
                # Free-running: the system runs and learns on its own and the window monitors it.
                # --step: nothing happens until Space is pressed.
                visualizer.show(
                    grid, width, height, fast=args.fast, teacher=teacher, report_seconds=args.report,
                    on_report=save_checkpoint, noise=None if teacher else args.sigma, rng=None if teacher else explore,
                )
            else:
                report_every = max(1, args.epochs // 10)
                started = time.perf_counter()
                for epoch in range(2, args.epochs + 1):
                    if teacher:
                        teacher.epoch(verbose=Neuron.verbose)  # silent unless --verbose: printing is slower than learning
                        if epoch % report_every == 0 or epoch == args.epochs:
                            elapsed = time.perf_counter() - started
                            teacher.record(elapsed, (epoch - 1) / elapsed if elapsed else None)
                            print(f"epoch {epoch}: {teacher.status()}", file=sys.stderr)
                            save_checkpoint()
                    else:
                        run_epoch(grid, verbose=Neuron.verbose, noise=args.sigma, rng=explore, discharge=args.discharge)
                        if epoch % report_every == 0 or epoch == args.epochs:
                            print(f"epoch {epoch}: {health(grid)}", file=sys.stderr)
                            save_checkpoint()
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        mesh = getattr(grid, "mesh", grid)
        if args.omega > 0 and not isinstance(mesh, (CartesianNodes, Goo)):
            print(
                f"omega {args.omega:g}: {len(mesh.small_world_connections())} small-world "
                f"connections among {len(mesh.connections)}",
                file=sys.stderr,
            )
        if teacher:
            teacher.close_trace()
            print(f"after {teacher.epochs} epochs: {teacher.status()}", file=sys.stderr)
        elif grid.epoch > 1:
            print(f"after {grid.epoch} epochs: {health(grid)}", file=sys.stderr)
        if args.save_weights:
            save_checkpoint()
            print(f"saved weights to {args.save_weights}", file=sys.stderr)

        print(
            f"{len(grid.fired_neurons())} of {len(grid.neurons)} neurons fired in {len(grid.waves)} waves",
            file=sys.stderr,
        )
        if args.save:
            visualizer.save(grid, args.save, width, height)
            print(f"Saved {args.save}", file=sys.stderr)
    except KeyboardInterrupt:
        # Ctrl+C is the normal way to stop, so exit cleanly rather than with a traceback.
        print("\nStopped.", file=sys.stderr)
    return 0


def health(grid) -> str:
    """One line on an untrained run: what fired this epoch, spikes to date, and the dopamine."""
    fired = len(grid.fired_neurons())
    line = f"{fired} of {len(grid.neurons)} fired this epoch, {grid.total_spikes():,} spikes to date"
    if grid.dopamine is not None:
        line += f", {grid.dopamine.status()}"
    return line


def _run_nodes(args: argparse.Namespace, width: int, height: int) -> int:
    """--nodes [N]: place a Cartesian population and show it. No wiring, input or learning yet."""
    if args.nodes < 0:
        print(f"error: --nodes cannot be negative, got {args.nodes}", file=sys.stderr)
        return 2
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    common = dict(across=args.across, rows=args.rows, seed=seed, threshold=args.threshold,
                  minimum_potential=args.minimum_potential)
    print(f"seed {seed}", file=sys.stderr)
    if args.nodes == 0:
        nodes = CartesianNodes(layout="hex", **common)
        print(f"{nodes!r}", file=sys.stderr)
    else:
        nodes = CartesianNodes(layout="random", count=args.nodes, **common)
        print(f"{nodes!r} ({len(nodes) * CELL_AREA / (nodes.width * nodes.height):.2f} per unit cell)", file=sys.stderr)
    if args.reach < 0:
        print(f"error: --reach must not be negative, got {args.reach}", file=sys.stderr)
        return 2
    made = nodes.connect_within(reach=args.reach, weight=args.weight)
    print(
        f"wired: {made} one-way connections, {nodes.mean_out_degree():.1f} outgoing per neuron, every pair within "
        f"{args.reach:g} units; input and learning are not defined for a scatter (it has no rows)",
        file=sys.stderr,
    )
    if args.save or args.show:
        try:
            from . import visualizer
        except ImportError:
            print("error: the visualizer needs pygame; install it with: pip install -e '.[viz]'", file=sys.stderr)
            return 2
        if args.save:
            visualizer.save_nodes(nodes, args.save, width, height)
            print(f"Saved {args.save}", file=sys.stderr)
        if args.show:
            visualizer.show_nodes(nodes, width, height)
    else:
        for neuron in nodes:
            x, y = neuron.position
            print(f"{neuron.name}: ({x:+.3f}, {y:+.3f})")
    return 0


def _seed_worker(job: dict) -> dict:
    """One seed's headless run, in its own process. Returns a summary row."""
    Neuron.verbose = False
    seed, epochs = job["seed"], job["epochs"]
    if job.get("goo") is not None:
        settings = job["settings"]
        grid = Goo(
            count=job["goo"], across=settings["across"], weight=settings["weight"], threshold=settings["threshold"],
            seed=seed, permute=settings["permute"], weight_range=settings["weight_range"],
            minimum_potential=settings["minimum_potential"],
            scale_with_fan_in=job.get("scale_with_fan_in") is not False,
            direct=job.get("direct_projection") is not False,
        )
    elif job.get("lattice"):
        settings = job["settings"]
        grid = CartesianNodes(
            across=settings["across"], rows=settings["rows"], seed=seed, threshold=settings["threshold"],
            minimum_potential=settings["minimum_potential"], permute=settings["permute"],
            weight_range=settings["weight_range"],
        )
        grid.connect_within(reach=job["lattice"]["reach"], weight=settings["weight"])
    elif not job.get("layers"):
        grid = GridOfNeurons(**job["settings"], seed=seed, reach=job.get("grid_reach", 2))
        if job.get("input_cells"):
            grid.set_input_cells(job["input_cells"])
    if job.get("layers") and job.get("goo") is None:
        grid = HexColumns(layers=job["layers"], **job["settings"], seed=seed)
    if job.get("scale_with_fan_in") and job.get("goo") is None:
        grid.scale_with_fan_in(job["settings"]["threshold"], job["settings"]["minimum_potential"])
    if job.get("ecc"):
        grid.use_ecc(job["ecc"])
    Neuron.refractory, Neuron.refractory_hops = job.get("refractory", Neuron.refractory), job.get("refractory_hops", Neuron.refractory_hops)
    Neuron.bored_after = job.get("bored_after", Neuron.bored_after)
    Neuron.tau = job.get("tau", Neuron.tau)
    grid.interval = job.get("interval", grid.interval)
    grid.problem = job.get("problem")
    grid.readout, grid.read, grid.read_window = job.get("readout", "top"), job.get("read", "fired"), job.get("read_window")
    grid.rule = job["teacher"].get("rule", RULE)
    grid.coding = job.get("coding", "complement")
    grid.quash_rate, grid.quash_k = job.get("quash", (0.0, QUASH_K))
    grid.flip = job.get("flip", 0.0)
    grid.hebb_rate = job.get("hebb", 0.0)
    grid.synapse_tau = job.get("synapse_tau", SYNAPSE_TAU)
    grid.drive = job.get("drive", INPUT_DRIVE)
    grid.input_rate, grid.input_rate_off = job.get("input_rate", INPUT_RATE), job.get("input_rate_off", INPUT_RATE_OFF)
    grid.explore, grid.rate_on = job.get("explore", EXPLORE), job.get("rate_on", RATE_ON)
    grid.teacher_threshold = job.get("teacher_threshold", TEACHER_THRESHOLD)
    Neuron.rate_tau = job.get("rate_tau", RATE_TAU)
    if job.get("input_seed") is not None:
        grid.use_input_stream(input_stream(epochs, grid.raw_bit_count(), job["input_seed"]))
    if job["teacher"].get("rule", RULE) == "dopamine":
        grid.dopamine = Dopamine(**job["dopamine"])
    if job.get("engine") == "arrays":
        from .arrays import ArrayNetwork
        grid = ArrayNetwork(grid)
    teacher = Teacher(grid, seed=seed, **job["teacher"])
    report_every = max(1, epochs // 10)
    started = time.perf_counter()
    rewards = [teacher.epoch(verbose=False)]
    for epoch in range(2, epochs + 1):
        rewards.append(teacher.epoch(verbose=False))
        if epoch % report_every == 0 or epoch == epochs:
            elapsed = time.perf_counter() - started
            teacher.record(elapsed, (epoch - 1) / elapsed if elapsed else None)
    if job["save"]:
        checkpoint(grid, job["save"], teacher)
    tail = rewards[-max(1, len(rewards) // 10):]  # the last tenth of the run
    return {
        "seed": seed,
        "to_date": teacher.accuracy_to_date,
        "recent": sum(tail) / len(tail),
        "epochs": teacher.epochs,
        "save": job["save"],
    }


def _run_seeds(args: argparse.Namespace) -> int:
    """--seeds N: N headless runs in parallel, one table at the end, a checkpoint per run."""
    if args.seeds < 1:
        print(f"error: --seeds needs at least 1, got {args.seeds}", file=sys.stderr)
        return 2
    if args.epochs < 2:
        print("error: --seeds needs --epochs of at least 2", file=sys.stderr)
        return 2
    base = args.seed if args.seed is not None else random.randrange(2**31 - args.seeds)
    seeds = list(range(base, base + args.seeds))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    settings = dict(
        across=args.across,
        rows=args.rows,
        weight=args.weight,
        threshold=args.threshold,
        omega=args.omega,
        permute=not args.no_permute,
        weight_range=(args.epsilon, 1.0) if args.positive_weights else WEIGHT_RANGE,
        minimum_potential=args.minimum_potential,
    )
    teacher_kwargs = dict(
        target=args.target,
        lr=args.lr,
        sigma=args.sigma,
        eligibility=args.eligibility,
        homeostasis=args.homeostasis,
        target_rate=args.target_rate,
        discharge=args.discharge,
        unstick=args.unstick,
        unstick_target=args.unstick_target,
        critic=args.critic,
        late=args.late,
        leaky=args.leaky,
        rule=args.rule,
    )
    dopamine = dict(tau=args.dopamine_tau, release_alpha=args.release_alpha, release_theta=args.release_theta,
                    order=args.order, lr=args.lr, expectation_tau=args.expectation_tau, punish=not args.no_punish,
                    punish_gain=args.punish_gain, decay=args.weight_decay, expectation_start=args.expectation_start)
    if args.no_learn:
        print("error: --seeds is for comparing learning runs; drop --no-learn", file=sys.stderr)
        return 2
    jobs = []
    for seed in seeds:
        save = None
        if not args.no_save:
            save = args.save_weights or str(Path("runs") / f"{stamp}-seed{seed}.json")
            if args.save_weights and args.seeds > 1:
                save = str(Path(args.save_weights).with_suffix("")) + f"-seed{seed}.json"
            Path(save).parent.mkdir(parents=True, exist_ok=True)
        lattice = {"reach": args.reach} if args.nodes is not None else None
        jobs.append({"seed": seed, "epochs": args.epochs, "settings": settings, "teacher": teacher_kwargs,
                     "save": save, "lattice": lattice, "goo": args.goo, "ecc": args.ecc, "engine": args.engine or "objects",
                     "scale_with_fan_in": args.scale_with_fan_in, "direct_projection": args.direct_projection,
                     "layers": args.layers, "refractory": args.refractory, "refractory_hops": args.refractory_hops,
                     "interval": args.interval, "dopamine": dopamine, "problem": args.problem, "bored_after": args.bored_after,
                     "tau": args.tau, "grid_reach": args.grid_reach, "input_cells": args.input_cells,
                     "readout": args.readout, "read": args.read, "read_window": args.read_window, "coding": args.coding,
                     "quash": (args.quash, args.quash_k), "flip": args.flip, "hebb": args.hebb,
                     "synapse_tau": args.synapse_tau,
                     "drive": args.drive, "input_rate": args.input_rate, "input_rate_off": args.input_rate_off,
                     "explore": args.explore, "rate_on": args.rate_on, "rate_tau": args.rate_tau,
                     "teacher_threshold": args.teacher_threshold,
                     "input_seed": None if args.input_seed is None else args.input_seed + (seed - base)})
    if args.engine == "arrays":
        try:
            import numpy, scipy  # noqa: F401
        except ImportError:
            print("error: the array engine needs numpy and scipy; install them with: pip install -e '.[arrays]'", file=sys.stderr)
            return 2
    workers = max(1, min(args.seeds, (os.cpu_count() or 2) - 1))
    wiring = (f"lattice, reach {args.reach:g}" if args.nodes is not None
              else f"{args.layers} layers of hexagonal columns, omega {args.omega:g}" if args.layers
              else f"hex grid, omega {args.omega:g}")
    shape = f"{args.across}x{args.rows} {wiring}"
    if args.goo is not None:
        shape = f"{args.goo} neurons of fully connected goo, {args.across} in and {args.across} out"
    # say which axis the arm ran on, so a sweep's own log identifies it (§5.2)
    scaled = args.scale_with_fan_in is not False if args.goo is not None else bool(args.scale_with_fan_in)
    shape += ", fan-in scaled" if scaled else ", flat threshold and floor"
    shape += f", {args.rule} rule" + (f" with the {args.eligibility} eligibility" if args.rule == "reinforce" else "")
    print(
        f"{args.seeds} seeds from {base} on {workers} cores, {args.epochs:,} epochs each, {shape}",
        file=sys.stderr,
    )
    started = time.perf_counter()
    with Pool(workers) as pool:
        rows = pool.map(_seed_worker, jobs)
    rows.sort(key=lambda r: (r["recent"], r["to_date"]), reverse=True)
    print(f"{'seed':>11}  {'last tenth':>10}  {'to date':>8}  checkpoint")
    for r in rows:
        print(f"{r['seed']:>11}  {r['recent']:>10.1%}  {r['to_date']:>8.1%}  {r['save'] or '-'}")
    best = rows[0]
    print(
        f"best seed {best['seed']}: {best['recent']:.1%} over its last tenth "
        f"({time.perf_counter() - started:.0f}s wall)",
        file=sys.stderr,
    )
    return 0


def _ecc_from_checkpoint(data: dict) -> str | None:
    """The code a checkpoint used: a name, or the (6, 4) code for files written when ecc was a flag."""
    value = data.get("ecc")
    if value is True:
        return "parity64"
    return value or None
