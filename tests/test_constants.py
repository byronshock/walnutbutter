"""constants.py is the one place a default lives: the constructors, the clock, the Teacher and the CLI all read it."""

import inspect

import pytest

from walnutbutter import constants as C
from walnutbutter.cartesian import CartesianNodes
from walnutbutter.cli import build_parser
from walnutbutter.columns import HexColumns
from walnutbutter.dopamine import Dopamine
from walnutbutter.goo import DEFAULT_COUNT, Goo
from walnutbutter.network import Network
from walnutbutter.grid import GridOfNeurons
from walnutbutter.learning import Teacher, reinforce
from walnutbutter.monitor import main
from walnutbutter.neuron import Neuron


def defaults_of(function) -> dict:
    return {name: p.default for name, p in inspect.signature(function).parameters.items() if p.default is not p.empty}


def test_the_command_line_defaults_are_the_constants():
    args = build_parser().parse_args([])
    assert (args.rows, args.omega, args.reach, args.epsilon) == (C.ROWS, C.OMEGA, C.REACH, C.WEIGHT_EPSILON)
    assert (args.threshold, args.minimum_potential) == (C.THRESHOLD, C.MINIMUM_POTENTIAL)
    assert (args.interval, args.refractory, args.refractory_hops) == (None, C.REFRACTORY, C.REFRACTORY_HOPS)  # the problem's, else INTERVAL
    assert args.rule is None and C.RULE == "teacher"  # the problem's rule, else the constant
    assert (args.dopamine_tau, args.release_alpha, args.release_theta, args.order) == (
        C.DOPAMINE_TAU, C.DOPAMINE_RELEASE_ALPHA, C.DOPAMINE_RELEASE_THETA, C.DOPAMINE_ORDER)
    assert args.problem == C.PROBLEM
    assert (args.target, args.late) == (C.TARGET, C.LATE)
    assert args.eligibility is None and C.ELIGIBILITY == "perturb"  # follows the neuron: hazard under escape noise (§1.3)
    assert args.delta == C.ESCAPE_DELTA == 0.455  # escape noise on by default (§5.2, Byron, September 15, 2026)
    assert args.critic is None and C.CRITIC == "row"  # the problem's critic, else the constant
    assert (args.lr, args.sigma, args.homeostasis, args.target_rate) == (C.LR, C.SIGMA, C.HOMEOSTASIS, C.TARGET_RATE)
    assert (args.unstick, args.unstick_target) == (C.UNSTICK, C.UNSTICK_TARGET)
    assert args.teacher_threshold == C.TEACHER_THRESHOLD and not hasattr(C, "THRESHOLD_RANGE")  # the clamp is gone


def test_the_neuron_and_its_clock_read_the_constants():
    assert (Neuron.refractory, Neuron.refractory_hops, Neuron.bored_after, Neuron.tau) == (C.REFRACTORY, C.REFRACTORY_HOPS, C.BORED_AFTER, C.TAU)
    assert build_parser().parse_args([]).tau == C.TAU
    assert build_parser().parse_args([]).bored_after == C.BORED_AFTER
    assert Neuron.hop() == C.REFRACTORY / C.REFRACTORY_HOPS
    neuron = Neuron()
    assert (neuron.threshold, neuron.minimum_potential) == (C.THRESHOLD, C.MINIMUM_POTENTIAL)
    assert GridOfNeurons(across=2, rows=2, omega=0).interval == C.INTERVAL


def test_every_container_builds_the_default_network_from_the_constants():
    for build in (GridOfNeurons, HexColumns, CartesianNodes, main):
        d = defaults_of(build)
        assert (d["across"], d["rows"], d["threshold"], d["minimum_potential"], d["weight_range"]) == (
            C.ACROSS, C.ROWS, C.THRESHOLD, C.MINIMUM_POTENTIAL, C.WEIGHT_RANGE), build.__name__
    assert defaults_of(GridOfNeurons)["omega"] == defaults_of(HexColumns)["omega"] == C.OMEGA
    assert defaults_of(CartesianNodes.connect_within)["reach"] == C.REACH


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
    assert C.THRESHOLD_FAN_IN == 18.0  # an interior hex cell's two rings at REACH 2
    assert defaults_of(Network.scale_with_fan_in)["reference"] == C.THRESHOLD_FAN_IN
    assert defaults_of(Goo)["scale_with_fan_in"] is True  # goo scales; nothing else does yet
    args = build_parser().parse_args([])
    assert args.scale_with_fan_in is None  # unset: goo scales, every other container does not
    assert build_parser().parse_args(["--scale-with-fan-in"]).scale_with_fan_in is True
    assert build_parser().parse_args(["--no-scale-with-fan-in"]).scale_with_fan_in is False


def test_the_teacher_and_the_rule_read_the_constants():
    d = defaults_of(Teacher)
    assert (d["target"], d["critic"], d["late"]) == (C.TARGET, C.CRITIC, C.LATE)
    assert d["eligibility"] is None  # resolved from the network: hazard under escape noise, else ELIGIBILITY (§1.3)
    assert (d["lr"], d["sigma"], d["baseline_rate"], d["window"]) == (C.LR, C.SIGMA, C.BASELINE_RATE, C.WINDOW)
    assert (d["homeostasis"], d["target_rate"]) == (C.HOMEOSTASIS, C.TARGET_RATE) and "threshold_range" not in d
    assert (d["unstick"], d["unstick_target"]) == (C.UNSTICK, C.UNSTICK_TARGET)
    r = defaults_of(reinforce)
    assert (r["lr"], r["sigma"], r["eligibility"], r["late"]) == (C.LR, C.SIGMA, C.ELIGIBILITY, C.LATE)
    assert d["rule"] == C.RULE
    p = defaults_of(Dopamine)
    assert (p["tau"], p["release_alpha"], p["release_theta"], p["order"], p["lr"]) == (
        C.DOPAMINE_TAU, C.DOPAMINE_RELEASE_ALPHA, C.DOPAMINE_RELEASE_THETA, C.DOPAMINE_ORDER, C.LR)
    assert p["expectation_tau"] == C.DOPAMINE_EXPECTATION_TAU == build_parser().parse_args([]).expectation_tau
    assert p["punish"] == C.DOPAMINE_PUNISH and not build_parser().parse_args([]).no_punish
    assert (p["punish_gain"], p["decay"]) == (C.DOPAMINE_PUNISH_GAIN, C.WEIGHT_DECAY)
    assert p["expectation_start"] == C.DOPAMINE_EXPECTATION_START == build_parser().parse_args([]).expectation_start
    assert (build_parser().parse_args([]).punish_gain, build_parser().parse_args([]).weight_decay) == (C.DOPAMINE_PUNISH_GAIN, C.WEIGHT_DECAY)
