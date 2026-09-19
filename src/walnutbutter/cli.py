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

from .goo import DEFAULT_COUNT as GOO_COUNT, WIRINGS, ZONE_WIRINGS, Goo
from .constants import GOO_MINIMUM_POTENTIAL, GOO_PROJECTION, GOO_SCALING_FACTOR, GOO_THRESHOLD
from .inputs import parse_bits
from .constants import (
    ACROSS, BORED_AFTER, CRITIC, ESCAPE_DELTA, FLIP, INPUT_CV, INPUT_DRIVE, INPUT_RATE, INPUT_RATE_OFF, POPULATION, TEMPERATURE,
    RATE_ON, RATE_TAU, READ_WINDOW, ROW_CRITIC_PICKINESS_IN_SPIKES,
    QUASH_K, QUASH_RATE, TAU, LEAK_TAU, PRESENTATION_TIME,
    ELIGIBILITY, HOMEOSTASIS, INTERVAL, LR,
    MINIMUM_POTENTIAL, PROBLEM, REFRACTORY, HOP, LAG, RULE, TARGET, TARGET_RATE,
    THRESHOLD_FAN_IN,
    THRESHOLD, UNSTICK, UNSTICK_TARGET, WEIGHT_EPSILON, WEIGHT_RANGE,
    cv_for_rate, rate_for_cv,
)
from .learning import CRITICS, ELIGIBILITIES, RULES, TARGETS, Teacher
from .problems import dataset_stream
from .monitor import main, run_epoch
from .network import input_stream
from .neuron import Neuron
from .persistence import across_of, checkpoint, restore, resume_teacher
from .problems import PROBLEMS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="walnutbutter",
        description=(
            "A goo of neurons that learns to reproduce its input on its output zone (AUTHORITY.md §4.1). "
            "By default it free-runs, learns, and reports accuracy until interrupted."
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
        help=f"width of the input zone (default: the problem's, {ACROSS} for {PROBLEM}; or twice the code length with --ecc)",
    )
    parser.add_argument(
        "--goo",
        type=int,
        nargs="?",
        const=GOO_COUNT,
        default=None,
        metavar="N",
        help=f"how many neurons the goo has: no positions at all, wired by the scaled rule at --scaling-factor "
        f"(default: the problem's, else {GOO_COUNT}). Goo has its own "
        f"threshold and floor, {GOO_THRESHOLD:g} and {GOO_MINIMUM_POTENTIAL:g} before its fan-in scaling, which "
        f"--threshold and --minimum-potential override. "
        f"The first --across neurons are the input zone and the last --across (or the problem's outputs) the "
        f"output (§4.3). Goo is the only container (§4.1), so this only sets the count",
    )
    parser.add_argument(
        "--hidden-neurons",
        "--hidden_neurons",
        dest="hidden_neurons",
        type=int,
        default=None,
        metavar="H",
        help="goo's hidden count (AUTHORITY.md §8; Byron, September 16, 2026: 'How will we know if they are buying us anything "
        "if they are always part of the economy?'): the goo is the input zone, H hidden neurons and the output zone, so "
        "--hidden-neurons sizes it instead of --goo; 0 is allowed under the scaled rule, the outputs then hearing the inputs "
        "directly (default: the problem's, 199 for mnist; otherwise --goo sizes the goo and the hidden count follows)",
    )
    parser.add_argument(
        "--scale-with-fan-in",
        "--scale_with_fan_in",
        dest="scale_with_fan_in",
        action="store_true",
        default=None,
        help="rescale each neuron's potential axis by its in-degree over THRESHOLD_FAN_IN (AUTHORITY.md §5.2): "
        "--threshold and --minimum-potential are quoted at THRESHOLD_FAN_IN incoming synapses, so a "
        "neuron with four times the fan-in starts four times as far from zero in both directions. On by default "
        "for goo and off for every other container, because turning it on for the grid would move every "
        "threshold every result to date was measured at",
    )
    parser.add_argument(
        "--projection",
        type=float,
        default=GOO_PROJECTION,
        metavar="P",
        help=f"the probability of goo's three earlier wirings, under --wiring zones-equal, zones or uniform (AUTHORITY.md "
        f"§3.4); the scaled rule does not read it. Under the zone rule: the probability an ordered pair with an interior end "
        f"projects, one way, each direction its own draw; pairs with both ends in a zone never project, and under "
        f"zones-equal an interior-to-zone projection is scaled up so that every neuron hears the same number in expectation "
        f"(default: {GOO_PROJECTION:g})",
    )
    parser.add_argument(
        "--scaling-factor",
        "--scaling_factor",
        dest="scaling_factor",
        type=float,
        default=GOO_SCALING_FACTOR,
        metavar="S",
        help=f"goo's wiring, the scaled rule (AUTHORITY.md §3.4; Byron, September 16, 2026): every neuron hears N times S "
        f"synapses in expectation, each ordered pair projecting, one way, at that fan-in over the sources the target may "
        f"hear, stopped at 1: the hidden neurons for an input, the inputs and hidden neurons for an output, everyone else "
        f"for a hidden neuron. No neuron projects onto itself, no input onto another input, and no output onto any zone "
        f"(the outputs apart, Byron, September 16, 2026): an output hears the inputs directly and projects onto the hidden "
        f"alone (default: {GOO_SCALING_FACTOR:g}: 32.2 synapses on the mnist goo of 644, 3 on goo 60)",
    )
    parser.add_argument(
        "--wiring",
        choices=WIRINGS,
        default="scaled",
        help="which rule wires goo (AUTHORITY.md §3.4): scaled, the rule since September 16, 2026, at --scaling-factor, the "
        "outputs apart since 03:05 that night; ff2, fully connected feedforward -- every input onto every output and nothing "
        "else, two layers and no hidden neurons (Byron, the same afternoon); ff2-partial, the same two layers with each "
        "input-to-output pair drawn at --projection (Byron, September 17); scaled-open, the night's first version, the outputs open to every zone; "
        "or one of the three before them at --projection -- zones-equal (the zone rule with equal fan-in, the rule until "
        "the 16th: the zones never project onto each other and an interior-to-zone projection is scaled up so every "
        "neuron hears the same number), zones (that rule without the equal fan-in) or uniform (one probability over "
        "every ordered pair). The two zone wirings need an interior (default: scaled)",
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
        "-w",
        "--weight",
        type=float,
        default=None,
        help=f"fixed weight for every connection (default: random, uniform between {WEIGHT_RANGE[0]:g} and {WEIGHT_RANGE[1]:g})",
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="BITS",
        default=None,
        help="raw input bits, one per half place, e.g. 1011 for 8 across (default: random)",
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
        help=f"which rule pays at the read (AUTHORITY.md §9.1): reinforce, the one rule the specification carries, "
        f"or local for none, in which case the local rules of §10 are the whole of the learning. "
        f"Default: the problem's, else {RULE}",
    )
    parser.add_argument(
        "--quash",
        type=float,
        nargs="?",
        const=QUASH_RATE,
        default=None,
        metavar="RATE",
        help=f"ask for the quash (AUTHORITY.md §10.1), which is off unless a run does: a refire weakens each "
        f"contributing synapse by this fraction of its weight, falling off with the delay since its previous spike. "
        f"Bare, it runs at {QUASH_RATE:g}, the value the constant holds for it; with a number, at that instead",
    )
    parser.add_argument(
        "--read",
        choices=("fired", "again", "window", "rate", "count"),
        default=None,
        help="what the teacher reads at the end of an epoch (AUTHORITY.md §4.3): fired this epoch, spiked again after "
        "the input's moment, fired within the read window, rate (the exponential-window firing-rate estimate scored "
        "against RATE_ON and RATE_OFF), or count: the epoch's spikes counted and the neuron on if that count is at "
        "least --pickiness (default: the problem's)",
    )
    parser.add_argument(
        "--pickiness",
        "--row-critic-pickiness-in-spikes",
        dest="pickiness",
        type=int,
        default=ROW_CRITIC_PICKINESS_IN_SPIKES,
        metavar="SPIKES",
        help=f"the count read's line between off and on, in spikes (default: {ROW_CRITIC_PICKINESS_IN_SPIKES}, §9.5). "
        f"An integer, and a count rather than a rate, so the read does not change meaning with the epoch's length; it "
        f"is chosen against the escape hazard's rest rate and travels with INTERVAL rather than alone",
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
        "--population",
        type=int,
        default=None,
        metavar="N",
        help=f"neurons per raw bit where the coding repeats it (AUTHORITY.md §4.3): population coding uses this many, and "
        f"population-complement uses this many and then complement-codes the lot (default: the problem's, else {POPULATION})",
    )
    parser.add_argument(
        "--outputs",
        type=int,
        default=None,
        metavar="N",
        help="the output zone's width on goo (AUTHORITY.md §3.4, §8): the problem's unless given -- mnist's 60 since "
        "September 16, 2026; --outputs 50 --population 5 --output-coding population is its earlier layout",
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
        "--quash-k",
        type=float,
        default=QUASH_K,
        metavar="PER_MS",
        help=f"how fast the quash falls off with that delay, exp(-k * delay) (default: {QUASH_K:g})",
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default=TARGET,
        help=f"what the output zone should show, derived from the input zone (default: {TARGET})",
    )
    parser.add_argument(
        "--critic",
        choices=sorted(CRITICS),
        default=None,
        help=f"how the reward is judged: row (fraction of output neurons matching the target), sustained (of the "
        f"neurons the target says should be on, the fraction on: did the forced neurons sustain?), decoded "
        f"(read the row as a word, error-correct it, fraction of data bits right), or decoded-exact "
        f"(all data bits right or nothing); class, graded and evidence score a dataset's label (§8): class pays 1 when the "
        f"label's group of outputs out-spikes every other, graded the fraction of the other groups it out-spikes, and "
        f"evidence reads the group sums as log-odds at --temperature and pays the softmax cross-entropy ln q_label, "
        f"chance ln 0.1. Default: {CRITIC}, or the problem's",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        metavar="T",
        help=f"the evidence critic's temperature (AUTHORITY.md §8): a lead of T spikes makes a class e times as likely; "
        f"T -> 0 is the class critic, T -> infinity pays every epoch ln 0.1 (default: {TEMPERATURE:g})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"learning rate, either rule (default: the problem's own, else {LR:g})",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=ESCAPE_DELTA,
        metavar="D",
        help=f"escape noise (AUTHORITY.md §5.2): the firing decision is a draw, D times the neuron's starting threshold "
        f"wide -- one expected spike per hop at threshold, e times more per D of margin above it, so a neuron nobody talks "
        f"to fires on its own at a rate its margin sets (default: {ESCAPE_DELTA:g}; 0 = the deterministic threshold)",
    )
    parser.add_argument(
        "--eligibility",
        choices=ELIGIBILITIES,
        default=None,
        help=f"what the global reward acts on: the neuron's exploration noise (perturb); the centred Hebbian term charged at "
        f"every decision, the target's spike or silence minus its own expectation of it times what the synapse has in its "
        f"potential (hebb, the single-spike rule of September 17, 2026); its epoch form, what each synapse delivered this "
        f"epoch times the target's count minus its expectation (count_hebb, hebb until that day); the uncentred "
        f"+-1 by whether the target fired (wrong_hebb, the rule hebb replaced on September 16, 2026); or the score of the "
        f"escape-noise decision on each synapse's trace (hazard, needs --delta; AUTHORITY.md §6.7) (default: hazard "
        f"when the network has escape noise, else {ELIGIBILITY})",
    )
    parser.add_argument(
        "--homeostasis",
        type=float,
        nargs="?",
        const=HOMEOSTASIS,
        default=None,
        metavar="RATE",
        help=f"ask for homeostasis (§9.9), which is off unless a run does: each neuron's threshold drifts toward its "
        f"target firing rate at this rate per epoch. Bare, it runs at {HOMEOSTASIS:g}, the value the constant holds "
        f"for it; with a number, at that instead",
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
        nargs="?",
        const=UNSTICK,
        default=None,
        metavar="RATE",
        help=f"ask for un-sticking (§9.10), which is off unless a run does: a stuck neuron's threshold moves toward "
        f"--unstick-target at this rate per epoch -- every neuron, not the outputs only; only neurons firing >99%% "
        f"or <1%% of the time are touched, only while stuck, and never one forced this epoch. Bare, it runs at "
        f"{UNSTICK:g}, the value the constant holds for it; with a number, at that instead",
    )
    parser.add_argument(
        "--unstick-target",
        type=float,
        default=UNSTICK_TARGET,
        help=f"firing rate the un-sticking aims for (default: {UNSTICK_TARGET:g})",
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
        "--presentation-time",
        type=float,
        default=PRESENTATION_TIME,
        metavar="MS",
        help="how far into each epoch the drive runs (AUTHORITY.md §5.4a; default: the whole epoch). Past it the "
        "input zone is undriven, which is not silent: every neuron still decides at every wave. Longer than the "
        "epoch is refused",
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
        help=f"leak time constant of every neuron, nominal milliseconds (default: {TAU:g} -- the potential does not "
        f"leak, which is the evidence accumulator of AUTHORITY.md §2; {LEAK_TAU:g} is the leak as it was swept). "
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
        "--hop",
        type=float,
        default=HOP,
        metavar="MS",
        help=f"the time a signal takes to travel one connection, nominal milliseconds (default: {HOP:g}, which is "
        f"(REFRACTORY + LAG) / 2 at LAG {LAG:g}, so two hops clear the refractory period by the LAG)",
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
        help="CSV the Teacher appends one line per epoch to: epoch, time, baseline, score "
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
        help="start from a checkpoint: rebuilds its network (count, wiring, seed) and loads its weights",
    )
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code (0 = success)."""
    args = build_parser().parse_args(argv)
    args.given_rule = args.rule  # what --rule said, if anything: it outlives a checkpoint's problem
    # goo has no positions, so there is nothing to draw: every run is headless (§4.1)
    args.show = args.fast = False
    try:
        apply_problem(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    apply_container(args)
    was_verbose, was_refractory, was_hop, was_bored, was_tau = Neuron.verbose, Neuron.refractory, Neuron.hop, Neuron.bored_after, Neuron.tau
    Neuron.verbose = bool(args.verbose) and not args.fast and not args.quiet
    if args.refractory <= 0 or args.hop <= 0 or args.interval <= 0 or args.tau <= 0:
        print("error: --tau, --refractory, --hop and --interval must be positive", file=sys.stderr)
        return 2
    if args.bored_after < 0 or (args.quash is not None and args.quash < 0) or args.quash_k < 0:
        print("error: --bored-after, --quash and --quash-k must not be negative", file=sys.stderr)
        return 2
    Neuron.refractory, Neuron.hop, Neuron.bored_after, Neuron.tau = args.refractory, args.hop, args.bored_after, args.tau
    try:
        return _run(args)
    finally:
        Neuron.verbose, Neuron.refractory, Neuron.hop, Neuron.bored_after, Neuron.tau = was_verbose, was_refractory, was_hop, was_bored, was_tau


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
        args.across = problem.across
    args.learn = not args.no_learn  # a Teacher scores every problem; whether it may train is the problem's
    if not problem.trained:
        args.homeostasis, args.unstick = 0.0, 0.0  # nothing outside the network moves a threshold, whichever rule runs (§6.7)
    if args.homeostasis is None:  # §9.9: off unless a run asks -- the flag, else the problem's, else zero
        args.homeostasis = problem.homeostasis if problem.homeostasis is not None else 0.0
    if args.unstick is None:  # §9.10, the same
        args.unstick = problem.unstick if problem.unstick is not None else 0.0
    if args.lr is None:  # the problem's own rate, unless --lr was given: a sentinel, so every rate given on the command line survives
        args.lr = problem.lr if problem.lr is not None else LR
    if problem.target is not None:
        args.target = problem.target
    if args.critic is None:
        args.critic = problem.critic if problem.critic is not None else CRITIC  # --critic overrides the problem's
    if args.rule is None:
        args.rule = problem.rule or RULE
    if args.quash is None:  # §10.1, the same
        args.quash = problem.quash if problem.quash is not None else 0.0
    if args.drive is None:
        args.drive = problem.drive if problem.drive is not None else INPUT_DRIVE
    if args.population is None:
        args.population = problem.population if problem.population is not None else POPULATION
    if args.interval is None:
        args.interval = problem.interval if problem.interval is not None else INTERVAL
    args.readout = problem.readout
    if args.read_window is None:
        args.read_window = problem.read_window
    if args.read is None:
        args.read = problem.read  # --read overrides what the problem asks for
    if args.read == "window" and args.read_window is None:
        args.read_window = READ_WINDOW
    if args.outputs is None:
        args.outputs = problem.outputs  # the output zone's width when it differs from the input's (goo, §8), unless --outputs
    args.clock = problem.clock  # clock neurons at the front of the input zone, always driven (§4.3)
    args.data = problem.data  # a dataset the inputs and their labels come from (§4.5, §8)
    if args.hidden_neurons is None:
        args.hidden_neurons = problem.hidden_neurons  # the problem's hidden count, unless --hidden-neurons was given (§8)
    if args.goo is None and args.hidden_neurons is not None:
        args.goo = args.across + args.hidden_neurons + (args.outputs or args.across)  # inputs + hidden + outputs
    if args.goo is None:
        args.goo = problem.goo if problem.goo is not None else GOO_COUNT  # goo is the only container (§4.1)


def apply_container(args: argparse.Namespace) -> None:
    """Goo's own threshold and floor (AUTHORITY.md §1.2) unless --threshold or --minimum-potential was given.

    A value equal to the default is taken as not given;
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
        if args.seeds is not None:
            return _run_seeds(args)
        loaded = None
        if args.load_weights:
            try:
                loaded = restore(args.load_weights)
            except (OSError, ValueError, KeyError) as exc:
                print(f"error: cannot load {args.load_weights}: {exc}", file=sys.stderr)
                return 2
            grid_from_file, data = loaded
            args.seed = data["seed"]
            args.across = across_of(data)
            if data.get("problem") and data["problem"] != args.problem:
                args.problem = data["problem"]
                args.interval = None if PROBLEMS[args.problem].interval is not None else args.interval
                args.rule = args.given_rule  # --rule if it was given, else the problem's
                apply_problem(args)
                print(f"problem: {args.problem} (from the checkpoint)", file=sys.stderr)
            args.threshold = data["threshold"]
            args.minimum_potential = data.get("minimum_potential", -1.0)
            args.weight = None if data["random_weights"] else data["weight"]
            low, high = data.get("weight_range", (-1.0, 1.0))
            args.positive_weights, args.epsilon = low > 0, low
            print(
                f"loaded {args.load_weights}: {args.across} across, seed {args.seed}, "
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
            weight=args.weight,
            threshold=args.threshold,
            seed=seed,
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
        if args.weight is None or args.input is None:
            print(f"seed {seed}", file=sys.stderr)

        try:
            input_bits = parse_bits(args.input) if args.input is not None else None
            if loaded:
                grid, data = loaded
            else:
                if args.hidden_neurons is not None and (args.hidden_neurons < 0 or args.goo != args.across + args.hidden_neurons + (args.outputs or args.across)):
                    print(f"error: --hidden-neurons {args.hidden_neurons} with {args.across} in and {args.outputs or args.across} "
                          f"out is a goo of {args.across + args.hidden_neurons + (args.outputs or args.across)}, not --goo {args.goo}",
                          file=sys.stderr)
                    return 2
                if args.wiring in ZONE_WIRINGS and args.goo <= args.across + (args.outputs or args.across):
                    print(
                        f"error: the {args.wiring} wiring needs an interior for its zones to talk through: --goo must "
                        f"exceed the zones together, got {args.goo} for {args.across} in and {args.outputs or args.across} out",
                        file=sys.stderr,
                    )
                    return 2
                if args.goo < args.across + (args.outputs or args.across):
                    print(f"error: the zones would overlap: --goo must be at least the zones together, got {args.goo} for "
                          f"{args.across} in and {args.outputs or args.across} out", file=sys.stderr)
                    return 2
                if not 0.0 < args.projection <= 1.0:
                    print(f"error: --projection must be in (0, 1], got {args.projection}", file=sys.stderr)
                    return 2
                if args.scaling_factor <= 0.0:
                    print(f"error: --scaling-factor must be positive, got {args.scaling_factor}", file=sys.stderr)
                    return 2
                grid = Goo(
                    count=args.goo, across=args.across, weight=args.weight, threshold=args.threshold, seed=seed,
                    weight_range=settings["weight_range"],
                    minimum_potential=args.minimum_potential,
                    scale_with_fan_in=args.scale_with_fan_in is not False,
                    projection=args.projection, outputs=args.outputs, wiring=args.wiring,
                    scaling_factor=args.scaling_factor,
                )
                print(f"{grid!r}: {grid.mean_out_degree():.1f} projections per neuron; {_wiring_summary(grid)}", file=sys.stderr)
                inner, edge = (grid.interior() or grid.all_neurons())[0], grid.all_neurons()[0]
                if grid.scale_with_fan_in_on:
                    print(
                        f"fan-in scaling (§5.2): an interior neuron hears {len(inner.incoming)} synapses and starts at "
                        f"threshold {inner.threshold:.3f}, floor {inner.minimum_potential:.3f}; a zone neuron hears "
                        f"{len(edge.incoming)} and starts at {edge.threshold:.3f}, {edge.minimum_potential:.3f}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"fan-in scaling off: a flat threshold {edge.threshold:g} and floor {edge.minimum_potential:g}",
                        file=sys.stderr,
                    )
            grid.interval = args.interval
            grid.presentation = args.presentation_time  # §5.4a; None is the whole epoch
            grid.presentation_time  # refuse a window past the horizon now, not at the first draw
            if not loaded or args.delta != ESCAPE_DELTA:
                grid.set_delta(args.delta)  # escape noise (§5.2), from the thresholds the container gave; a checkpoint keeps its own
            if args.eligibility is None:  # the eligibility follows the neuron (§1.3)
                args.eligibility = "hazard" if grid.hazard else ELIGIBILITY
            grid.problem = args.problem
            grid.readout, grid.read, grid.read_window = args.readout, args.read, args.read_window
            grid.clock = args.clock
            if args.clock:
                print(f"clock neurons: the first {args.clock} input neurons are driven every epoch whatever the pattern "
                      f"(§4.3), and take no raw bits", file=sys.stderr)
            grid.quash_rate, grid.quash_k = args.quash, args.quash_k
            grid.drive, grid.input_rate, grid.input_rate_off = args.drive, args.input_rate, args.input_rate_off
            grid.rate_on = args.rate_on
            grid.pickiness = args.pickiness
            grid.population = args.population
            grid.temperature = args.temperature  # the evidence critic's (§8)
            Neuron.rate_tau = args.rate_tau
            if args.data is not None:
                patterns, labels = dataset_stream(args.data, args.input_seed if args.input_seed is not None else seed)
                grid.use_input_stream(patterns, labels)
                print(f"inputs: the {len(patterns):,} images of {args.data} in the shuffle of seed "
                      f"{args.input_seed if args.input_seed is not None else seed}, with their labels, going round "
                      f"again when the run outlasts them (§4.5, §8)", file=sys.stderr)
            elif args.input_seed is not None:
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
            if not PROBLEMS[args.problem].trained:
                print(
                    f"problem {args.problem}: {PROBLEMS[args.problem].description}. Epochs {args.interval:g} ms apart; "
                    f"scored on the {args.readout} row against {args.target} by the {args.critic} critic, on meaning "
                    f"{READ_MEANS(args)}; "
                    "nothing outside the network trains it",
                    file=sys.stderr,
                )
            if args.rule == "local":
                grid.rule = args.rule
                print(f"rule: local — nothing pays at the read (§9.1); "
                      f"{f'quash {args.quash:g} falling off at {args.quash_k:g}/ms' if args.quash else 'no quash'}"
                      f"; hop {Neuron.hop:g} ms, tau {Neuron.tau:g} ms", file=sys.stderr)
            else:
                # the Teacher zeroes sigma for the Hebbian eligibilities, so the network is deterministic: report what runs
                print(f"rule: reinforce ({args.eligibility} eligibility), lr {args.lr:g}, "
                      f"escape delta {args.delta:g}"
                      f"{f' (every hazard times {grid.escape_scale:.3g} at {len(grid.all_neurons())} neurons)' if args.delta else ''}"
                      f"; hop {Neuron.hop:g} ms, tau {Neuron.tau:g} ms, "
                      f"bored after {Neuron.bored_after:g} ms, "
                      f"{f'quash {args.quash:g} falling off at {args.quash_k:g}/ms' if args.quash else 'no quash'}",
                      file=sys.stderr)
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
                    eligibility=args.eligibility,
                    seed=seed,
                    homeostasis=args.homeostasis,
                    target_rate=args.target_rate,
                    discharge=args.discharge,
                    unstick=args.unstick,
                    unstick_target=args.unstick_target,
                    critic=args.critic,
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
                run_epoch(grid, input_bits, verbose=Neuron.verbose, rng=explore, discharge=args.discharge)

            def save_checkpoint():
                if args.save_weights:
                    checkpoint(grid, args.save_weights, teacher)

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
                    run_epoch(grid, verbose=Neuron.verbose, rng=explore, discharge=args.discharge)
                    if epoch % report_every == 0 or epoch == args.epochs:
                        print(f"epoch {epoch}: {health(grid)}", file=sys.stderr)
                        save_checkpoint()
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

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
    except KeyboardInterrupt:
        # Ctrl+C is the normal way to stop, so exit cleanly rather than with a traceback.
        print("\nStopped.", file=sys.stderr)
    return 0


def health(grid) -> str:
    """One line on an untrained run: what fired this epoch and the spikes to date."""
    fired = len(grid.fired_neurons())
    line = f"{fired} of {len(grid.neurons)} fired this epoch, {grid.total_spikes():,} spikes to date"
    return line



def _wiring_summary(grid: Goo) -> str:
    """One line on what goo's wiring lets talk to what, for the run's header."""
    if grid.wiring == "scaled":
        hidden = grid.interior_count()
        heard = (f"an input from the {hidden} hidden at P {grid.input_projection():.3g}" if hidden
                 else "an input from no one, left at the container's threshold (§5.2)")
        return (f"every neuron hears {grid.expected_fan_in():g} synapses in expectation where it has sources: {heard}, an "
                f"output from the {grid.count - grid.outputs} inputs and hidden at {grid.output_projection():.3g}"
                + (f", a hidden neuron from the {grid.count - 1} others at {grid.hidden_projection():.3g}" if hidden else "")
                + "; the outputs project onto the hidden alone")
    if grid.wiring == "scaled-open":
        return (f"every neuron hears {grid.expected_fan_in():g} synapses in expectation, an input neuron from the "
                f"{grid.count - grid.across} outside its zone at P {grid.input_projection():.3g} and every other neuron "
                f"from the {grid.count - 1} others at {grid.hidden_projection():.3g}")
    if grid.wiring == "uniform":
        return f"every ordered pair at P {grid.projection:g}, the zones included"
    return f"the zones talk only through the {len(grid.interior())} interior neurons"


def _seed_worker(job: dict) -> dict:
    """One seed's headless run, in its own process. Returns a summary row."""
    Neuron.verbose = False
    seed, epochs = job["seed"], job["epochs"]
    settings = job["settings"]
    grid = Goo(
        count=job["goo"], across=settings["across"], weight=settings["weight"], threshold=settings["threshold"],
        seed=seed, weight_range=settings["weight_range"],
        minimum_potential=settings["minimum_potential"],
        scale_with_fan_in=job.get("scale_with_fan_in") is not False,
        projection=job.get("projection", GOO_PROJECTION), outputs=job.get("outputs"),
        wiring=job.get("wiring", "scaled"), scaling_factor=job.get("scaling_factor", GOO_SCALING_FACTOR),
    )
    Neuron.refractory, Neuron.hop = job.get("refractory", Neuron.refractory), job.get("hop", Neuron.hop)
    Neuron.bored_after = job.get("bored_after", Neuron.bored_after)
    Neuron.tau = job.get("tau", Neuron.tau)
    grid.interval = job.get("interval", grid.interval)
    grid.presentation = job.get("presentation_time", grid.presentation)  # §5.4a, in each worker
    grid.set_delta(job.get("delta", ESCAPE_DELTA))  # escape noise (§5.2), once the thresholds are the container's
    grid.problem = job.get("problem")
    grid.readout, grid.read, grid.read_window = job.get("readout", "top"), job.get("read", "fired"), job.get("read_window")
    grid.rule = job["teacher"].get("rule", RULE)
    grid.clock = job.get("clock", 0)
    grid.quash_rate, grid.quash_k = job.get("quash", (0.0, QUASH_K))
    grid.drive = job.get("drive", INPUT_DRIVE)
    grid.input_rate, grid.input_rate_off = job.get("input_rate", INPUT_RATE), job.get("input_rate_off", INPUT_RATE_OFF)
    grid.rate_on = job.get("rate_on", RATE_ON)
    grid.pickiness = job.get("pickiness", ROW_CRITIC_PICKINESS_IN_SPIKES)
    grid.population = job.get("population", POPULATION)  # the seed worker left it at the constant until September 16, 2026
    grid.temperature = job.get("temperature", TEMPERATURE)  # the evidence critic's (§8)
    Neuron.rate_tau = job.get("rate_tau", RATE_TAU)
    if job.get("data") is not None:
        grid.use_input_stream(*dataset_stream(job["data"], job["input_seed"] if job.get("input_seed") is not None else seed))
    elif job.get("input_seed") is not None:
        grid.use_input_stream(input_stream(epochs, grid.raw_bit_count(), job["input_seed"]))
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
        weight=args.weight,
        threshold=args.threshold,
        weight_range=(args.epsilon, 1.0) if args.positive_weights else WEIGHT_RANGE,
        minimum_potential=args.minimum_potential,
    )
    if args.eligibility is None:  # the eligibility follows the neuron (§1.3): every seed's network gets args.delta
        args.eligibility = "hazard" if args.delta > 0 else ELIGIBILITY
    teacher_kwargs = dict(
        target=args.target,
        lr=args.lr,
        eligibility=args.eligibility,
        homeostasis=args.homeostasis,
        target_rate=args.target_rate,
        discharge=args.discharge,
        unstick=args.unstick,
        unstick_target=args.unstick_target,
        critic=args.critic,
        rule=args.rule,
    )
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
        jobs.append({"seed": seed, "epochs": args.epochs, "settings": settings, "teacher": teacher_kwargs,
                     "save": save, "goo": args.goo, "engine": args.engine or "objects",
                     "scale_with_fan_in": args.scale_with_fan_in, "projection": args.projection,
                     "wiring": args.wiring, "scaling_factor": args.scaling_factor,
                     "refractory": args.refractory, "hop": args.hop,
                     "interval": args.interval, "presentation_time": args.presentation_time,
                     "problem": args.problem, "bored_after": args.bored_after,
                     "tau": args.tau,
                     "readout": args.readout, "read": args.read, "read_window": args.read_window,
                     "quash": (args.quash, args.quash_k),
                     "drive": args.drive, "input_rate": args.input_rate, "input_rate_off": args.input_rate_off,
                     "rate_on": args.rate_on, "rate_tau": args.rate_tau,
                     "pickiness": args.pickiness, "delta": args.delta,
                     "population": args.population, "temperature": args.temperature,
                     "outputs": args.outputs, "data": args.data, "clock": args.clock,
                     "input_seed": None if args.input_seed is None else args.input_seed + (seed - base)})
    if args.engine == "arrays":
        try:
            import numpy, scipy  # noqa: F401
        except ImportError:
            print("error: the array engine needs numpy and scipy; install them with: pip install -e '.[arrays]'", file=sys.stderr)
            return 2
    workers = max(1, min(args.seeds, (os.cpu_count() or 2) - 1))
    density = (f"at scaling factor {args.scaling_factor:g}" if args.wiring == "scaled"
               else f"at projection {args.projection:g} under the {args.wiring} wiring")
    hidden = args.goo - args.across - (args.outputs or args.across)
    shape = f"{args.goo} neurons of goo {density}, {args.across} in, {hidden} hidden and {args.outputs or args.across} out"
    # say which axis the arm ran on, so a sweep's own log identifies it (§5.2)
    scaled = args.scale_with_fan_in is not False
    shape += ", fan-in scaled" if scaled else ", flat threshold and floor"
    shape += f", {args.rule} rule" + (f" with the {args.eligibility} eligibility" if args.rule == "reinforce" else "")
    shape += f", escape delta {args.delta:g}" if args.delta else ""
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
