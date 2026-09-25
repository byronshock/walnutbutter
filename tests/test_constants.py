"""constants.py is the one place a default lives: the constructors, the clock, the Teacher and the CLI all read it."""

import inspect

import pytest

from walnutbutter import constants as C
from walnutbutter.cli import build_parser
from walnutbutter.goo import DEFAULT_COUNT, Goo
from walnutbutter.network import Network
from walnutbutter.goo import Goo
from walnutbutter.learning import Teacher, reinforce
from walnutbutter.monitor import main
from walnutbutter.neuron import Neuron


def defaults_of(function) -> dict:
    return {name: p.default for name, p in inspect.signature(function).parameters.items() if p.default is not p.empty}


def test_the_command_line_defaults_are_the_constants():
    args = build_parser().parse_args([])
    assert args.epsilon == C.WEIGHT_EPSILON
    assert (args.threshold, args.minimum_potential) == (C.THRESHOLD, C.MINIMUM_POTENTIAL)
    assert (args.interval, args.refractory, args.hop) == (None, C.REFRACTORY, C.HOP)  # the problem's, else INTERVAL
    assert args.rule is None and C.RULE == "local"  # the problem's rule, else the constant (§9.1)
    assert args.problem == C.PROBLEM
    assert args.target == C.TARGET
    assert args.eligibility is None and C.ELIGIBILITY == "hazard"  # §8.3: where a run names none, the eligibility is hazard
    assert args.delta == C.ESCAPE_DELTA == 0.455  # escape noise on by default (§5.2, Byron, September 15, 2026)
    assert args.critic is None and C.CRITIC == "row"  # the problem's critic, else the constant
    assert (args.lr, args.target_rate) == (None, C.TARGET_RATE)  # the problem's rate, else LR
    # §9.9, §9.10, §10.1: the three are latent knobs -- off unless a run asks, and the constant is what asking gets
    assert (args.homeostasis, args.unstick, args.quash) == (None, None, None)
    for argv, want in ((['--homeostasis'], C.HOMEOSTASIS), (['--unstick'], C.UNSTICK), (['--quash'], C.QUASH_RATE)):
        asked = build_parser().parse_args(argv)
        assert getattr(asked, argv[0][2:]) == want, argv
    assert args.unstick_target == C.UNSTICK_TARGET
    assert args.pickiness == C.ROW_CRITIC_PICKINESS_IN_SPIKES == 2 and not hasattr(C, "THRESHOLD_RANGE")  # the clamp is gone



def test_a_rate_given_on_the_command_line_survives_the_problem():
    """--lr names the rate, whatever its value (September 18, 2026).

    The rate used to be taken from the problem whenever `args.lr` still equalled
    LR, so `--lr 0.03` -- exactly the constant -- was indistinguishable from not
    passing it, and ran at mnist's 0.002 under an arm named lr0.03. The default
    is a sentinel now, so only an absent flag defers to the problem.
    """
    from walnutbutter.cli import apply_problem

    for problem, given, want in (("mnist", None, 0.002), ("mnist", 0.03, 0.03), ("mnist", 0.0075, 0.0075),
                                 ("reversal", None, C.LR), ("reversal", 0.03, C.LR)):
        argv = ["--problem", problem] + ([] if given is None else ["--lr", str(given)])
        args = build_parser().parse_args(argv)
        apply_problem(args)
        assert args.lr == want, (problem, given, args.lr)


def test_the_neuron_and_its_clock_read_the_constants():
    assert (Neuron.refractory, Neuron.hop, Neuron.bored_after, Neuron.tau) == (C.REFRACTORY, C.HOP, C.BORED_AFTER, C.TAU)
    assert build_parser().parse_args([]).tau == C.TAU
    assert build_parser().parse_args([]).bored_after == C.BORED_AFTER
    # §3.2: the hop is specified directly, and two of them clear the refractory wall by the LAG
    assert Neuron.hop == C.HOP == 2.55 == (C.REFRACTORY + C.LAG) / 2.0
    assert 2 * C.HOP == pytest.approx(C.REFRACTORY + C.LAG) and C.LAG == 0.1
    assert not hasattr(C, "REFRACTORY_HOPS")  # retired: REFRACTORY / 2.55 is 1.9607843137..., not a register value
    neuron = Neuron()
    assert (neuron.threshold, neuron.minimum_potential) == (C.THRESHOLD, C.MINIMUM_POTENTIAL)
    assert Goo(count=4, across=2).interval == C.INTERVAL


def test_the_container_builds_the_default_network_from_the_constants():
    """§4.1: goo is the only container, so its defaults are the network's."""
    for build in (Goo, main):
        d = defaults_of(build)
        assert (d["across"], d["weight_range"]) == (C.ACROSS, C.WEIGHT_RANGE), build.__name__
        assert (d["count"], d["threshold"], d["minimum_potential"]) == (
            C.GOO_COUNT, C.GOO_THRESHOLD, C.GOO_MINIMUM_POTENTIAL), build.__name__


def test_goo_has_its_own_count_threshold_and_floor():
    """§1.2: the threshold belongs to the container (Byron, September 14, 2026: 60 units of goo, THRESHOLD=1)."""
    from walnutbutter.cli import apply_container
    d = defaults_of(Goo)
    assert (d["across"], d["weight_range"]) == (C.ACROSS, C.WEIGHT_RANGE)
    assert (d["count"], d["threshold"], d["minimum_potential"]) == (C.GOO_COUNT, C.GOO_THRESHOLD, C.GOO_MINIMUM_POTENTIAL)
    assert DEFAULT_COUNT == C.GOO_COUNT == 60 and C.GOO_THRESHOLD == 0.2  # from the fine sweep (§3.4)
    assert C.GOO_MINIMUM_POTENTIAL == C.GOO_THRESHOLD * C.MINIMUM_POTENTIAL / C.THRESHOLD == pytest.approx(-0.8)  # the grid's ratio, kept
    assert C.THRESHOLD == 0.25 and C.MINIMUM_POTENTIAL == -1.0  # the grid, the columns and the lattice keep theirs
    assert build_parser().parse_args(["--goo"]).goo == C.GOO_COUNT
    assert build_parser().parse_args([]).goo is None  # no goo unless asked for
    # the command line gives goo its own threshold and floor unless told otherwise
    args = build_parser().parse_args(["--goo"]); apply_container(args)
    assert (args.threshold, args.minimum_potential) == (C.GOO_THRESHOLD, C.GOO_MINIMUM_POTENTIAL)
    args = build_parser().parse_args(["--goo", "--threshold", "0.5", "--minimum-potential", "-2"]); apply_container(args)
    assert (args.threshold, args.minimum_potential) == (0.5, -2.0)
    args = build_parser().parse_args([]); apply_container(args)
    assert (args.threshold, args.minimum_potential) == (C.THRESHOLD, C.MINIMUM_POTENTIAL)  # the grid is untouched


def test_the_fan_in_the_threshold_is_quoted_at_has_one_home():
    """§5.2: THRESHOLD and MINIMUM_POTENTIAL are quoted per THRESHOLD_FAN_IN incoming synapses."""
    assert C.THRESHOLD_FAN_IN == 18.0  # the archived hex grid's interior in-degree, kept as the unit (§4.11)
    assert defaults_of(Network.scale_with_fan_in)["reference"] == C.THRESHOLD_FAN_IN
    assert defaults_of(Goo)["scale_with_fan_in"] is True  # goo scales; nothing else does yet
    args = build_parser().parse_args([])
    assert args.scale_with_fan_in is None  # unset: goo scales, every other container does not
    assert build_parser().parse_args(["--scale-with-fan-in"]).scale_with_fan_in is True
    assert build_parser().parse_args(["--no-scale-with-fan-in"]).scale_with_fan_in is False


def test_exploration_at_the_synapse_reads_the_constants():
    """Appendix A's synapse rows (§7.1, §7.6, §7.7, §8.17, 5.4b): the register's values, and the setter's and the network's
    defaults are those constants. EXPLORATION stays neuron until §7.6's conditions are met."""
    assert (C.EXPLORATION, C.SYNAPSE_HAZARD_REST, C.SYNAPSE_HAZARD_FAMILY, C.SYNAPSE_HAZARD_SCALING, C.TRACE,
            C.DRIVE_STEPS) == ("neuron", 0.01, "loglinear", "count", "all", 3)
    d = defaults_of(Network.set_exploration)
    assert (d["h0"], d["family"], d["scaling"], d["trace"]) == (
        C.SYNAPSE_HAZARD_REST, C.SYNAPSE_HAZARD_FAMILY, C.SYNAPSE_HAZARD_SCALING, C.TRACE)
    g = Goo(count=4, across=2)  # built under the neuron rule, as it is built deterministic until set_delta
    assert (g.exploration, g.drive_steps) == ("neuron", C.DRIVE_STEPS)


def test_the_teacher_and_the_rule_read_the_constants():
    d = defaults_of(Teacher)
    assert (d["target"], d["critic"]) == (C.TARGET, C.CRITIC)
    assert d["eligibility"] is None  # §8.3: a run that names none gets ELIGIBILITY
    assert (d["lr"], d["baseline_rate"], d["window"]) == (C.LR, C.BASELINE_RATE, C.WINDOW)
    assert (d["homeostasis"], d["unstick"]) == (0.0, 0.0)  # §9.9, §9.10: the library default is off
    assert d["target_rate"] == C.TARGET_RATE and "threshold_range" not in d
    assert d["unstick_target"] == C.UNSTICK_TARGET
    r = defaults_of(reinforce)
    assert (r["lr"], r["eligibility"]) == (C.LR, C.ELIGIBILITY)
    assert d["rule"] == C.RULE
    # §9.1 carries one rule that pays at the read and one local rule; the eight dopamine constants,
    # TEACHER_CREDIT, HEBB_RATE and WEIGHT_DECAY all left with the rules they belonged to
    for gone in ("DOPAMINE_TAU", "DOPAMINE_RELEASE_ALPHA", "DOPAMINE_RELEASE_THETA", "DOPAMINE_ORDER",
                 "DOPAMINE_EXPECTATION_TAU", "DOPAMINE_EXPECTATION_START", "DOPAMINE_PUNISH",
                 "DOPAMINE_PUNISH_GAIN", "WEIGHT_DECAY", "HEBB_RATE", "TEACHER_CREDIT"):
        assert not hasattr(C, gone), gone
