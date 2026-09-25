"""Checkpoints, resume, the command line and the sweep driver under exploration at the synapse (AUTHORITY.md §8.14,
§12.9-§12.11, §6.13, §7.1, §7.6, §7.7, §8.17, 5.4b): step 5 of docs/synapse-build-map-2026-09-24.md.

The object engine's round trips run on goo 24 and a few epochs; the Rust loop's are in tests/test_resume_exact_rust.py.
The live sweep checkpoints are exercised on copies, against the code before this step as well as this one, and the test
that does it skips where runs/ is not on disk or that code is not in the repository's history.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from walnutbutter import constants as C
from walnutbutter import fast
from walnutbutter.cli import apply_exploration, apply_problem, build_parser, cli_main
from walnutbutter.clock import slack
from walnutbutter.goo import Goo
from walnutbutter.learning import Teacher
from walnutbutter.network import TRACE_VENTURED_BIAS, input_stream
from walnutbutter.neuron import Neuron
from walnutbutter import persistence
from walnutbutter.persistence import checkpoint, load_weights, read_checkpoint, restore, resume_teacher

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def quiet_and_the_clock_restored(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)
    monkeypatch.setattr(Neuron, "tau", Neuron.tau)  # restored after each test, whatever a test or the CLI set it to
    monkeypatch.setattr(Neuron, "rate_tau", Neuron.rate_tau)


def rs():
    spec = importlib.util.spec_from_file_location("rs", ROOT / "docs" / "rust-sweep.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def goo(drive="rate", threshold=C.GOO_THRESHOLD, **settings) -> Goo:
    g = Goo(count=24, across=4, seed=3, weight=None, threshold=threshold, minimum_potential=-4.0 * threshold)
    g.readout, g.read, g.drive = "top", "count", drive
    g.set_exploration("synapse", **settings)
    return g


def teach(network, seed=3):
    return Teacher(network, seed=seed, rule="reinforce", eligibility="hazard", target="copy")


def state(network) -> dict:
    """Everything a resume must continue exactly, the queue with its marks among it."""
    neurons, edges = network.all_neurons(), [c for n in network.all_neurons() for c in n.outgoing]
    return {
        "weights": [c.weight for c in edges], "traces": [(c.trace, c.trace_at) for c in edges],
        "stamps": [c.last_signal for c in edges], "spikes": [n.spikes for n in neurons],
        "potentials": [n.potential for n in neurons], "gains": [n.gain for n in neurons],
        "clocks": [n.exposed_since for n in neurons], "thresholds": [n.threshold for n in neurons],
        "queue": [(t, c.id, v) for t, c, v in network.schedule.pending(marks=True)],
        "waiting": persistence._waiting(network),
    }


# --- persistence: what a checkpoint carries, and what it refuses -------------------------------------------------------

@pytest.mark.parametrize("family, scaling, trace", [("loglinear", "count", "all"), ("linear", "fan-out", "ventured")])
def test_a_checkpoint_carries_the_gains_the_read_counts_the_marks_and_the_settings(tmp_path, family, scaling, trace):
    """§8.14, §12.9: under exploration at the synapse a checkpoint is format 3 and carries each neuron's gain, each
    output's read count, the ventured mark on every signal in flight and the settings of §7.1, §7.6, §7.7, §8.17 and
    5.4b; kappa_i is recomputed from them at the restore and stored nowhere (§7.7, §12.10), and every trace is kept."""
    g = goo("charged", threshold=2.0, h0=0.3, family=family, scaling=scaling, trace=trace)
    g.drive_steps = 4
    t = teach(g)
    for _ in range(4):
        t.epoch(verbose=False)
    for _ in range(12):  # on to a read with a gain standing open across it (§8.16), for the file to carry
        t.epoch(verbose=False)
        if any(n.gain for n in g.all_neurons()):
            break
    assert any(n.gain for n in g.all_neurons())
    data = checkpoint(g, tmp_path / "s.json", t)
    assert (data["format"], data["exploration"]) == (persistence.SYNAPSE_FORMAT, "synapse") == (3, "synapse")
    assert (data["synapse_hazard_rest"], data["synapse_hazard_family"], data["synapse_hazard_scaling"],
            data["trace_mode"], data["drive"], data["drive_steps"]) == (0.3, family, scaling, trace, "charged", 4)
    assert data["gains"] == [n.gain for n in g.all_neurons()]
    assert data["read_counts"] == [n.read_count for n in g.all_neurons()] and sum(data["read_counts"]) > 0
    assert all(len(entry) == 3 for entry in data["pending"]) and any(entry[2] for entry in data["pending"])
    assert not any("kappa" in key or "synapse_scale" in key for key in data), "a rule, not a state (§12.10)"
    assert data.get("estimator_bias") == (TRACE_VENTURED_BIAS if trace == "ventured" else None)

    back, again = restore(tmp_path / "s.json")
    assert again == data
    assert (back.exploration, back.synapse_hazard_rest, back.synapse_hazard_family, back.synapse_hazard_scaling,
            back.trace_mode, back.drive, back.drive_steps) == ("synapse", 0.3, family, scaling, trace, "charged", 4)
    assert [n.synapse_scale for n in back.all_neurons()] == [n.synapse_scale for n in g.all_neurons()]
    assert back.traced and all(n.traced and n.synaptic and n.delta == 0.0 for n in back.all_neurons())
    assert all(n.trace_ventured == (trace == "ventured") for n in back.all_neurons())
    assert state(back) == state(g)
    assert [n.read_count for n in back.all_neurons()] == data["read_counts"]


def test_a_checkpoint_of_the_neuron_rule_stays_format_two_and_names_its_exploration(tmp_path):
    """A neuron-rule checkpoint is format 2 with two-element pending entries and none of the synapse keys, so a reader
    that knows only format 2 -- the trunk's -- reads it as before; it names its exploration, whichever it is (§8.14), and
    such a reader refuses a format-3 file rather than resume it under the neuron rule (§12.9)."""
    g = Goo(count=24, across=4, seed=3, weight=None)
    g.readout, g.read, g.drive = "top", "count", "rate"
    g.set_delta(C.ESCAPE_DELTA)
    t = teach(g)
    for _ in range(6):
        t.epoch(verbose=False)
    data = checkpoint(g, tmp_path / "n.json", t)
    assert (data["format"], data["exploration"]) == (2, "neuron")
    assert all(len(entry) == 2 for entry in data["pending"])
    for key in ("gains", "read_counts", "synapse_hazard_rest", "synapse_hazard_family", "synapse_hazard_scaling",
                "trace_mode", "drive_steps", "estimator_bias"):
        assert key not in data, key
    g3 = goo()
    checkpoint(g3, tmp_path / "s.json")
    trunk = pytest.MonkeyPatch()
    trunk.setattr(persistence, "FORMATS", (persistence.FORMAT,))  # a reader of format 2 alone, as the trunk's is
    try:
        assert read_checkpoint(tmp_path / "n.json")["format"] == 2
        with pytest.raises(ValueError, match="unknown checkpoint format 3"):
            read_checkpoint(tmp_path / "s.json")
    finally:
        trunk.undo()


def test_read_checkpoint_accepts_formats_two_and_three_and_refuses_the_rest(tmp_path):
    """read_checkpoint takes the 796 format-2 files on disk and format 3, and nothing else; a file whose format and
    exploration disagree was written by no build of this code and is refused at the restore (§12.2)."""
    for fmt in (1, 4, 99, None):
        path = tmp_path / f"f{fmt}.json"
        path.write_text(json.dumps({"format": fmt}))
        with pytest.raises(ValueError, match="unknown checkpoint format"):
            read_checkpoint(path)
    g = goo()
    data = checkpoint(g, tmp_path / "s.json")
    for fmt, mode in ((2, "synapse"), (3, "neuron")):
        wrong = dict(data, format=fmt, exploration=mode)
        (tmp_path / "wrong.json").write_text(json.dumps(wrong))
        with pytest.raises(ValueError, match="§8.14"):
            restore(tmp_path / "wrong.json")
    (tmp_path / "odd.json").write_text(json.dumps(dict(data, exploration="dendrite")))
    with pytest.raises(ValueError, match="unknown exploration 'dendrite'.*§7.1"):
        restore(tmp_path / "odd.json")


def test_a_file_written_before_the_exploration_was_carried_restores_as_the_neuron_rule(tmp_path):
    """§8.14: a checkpoint that carries no §7.1 setting was saved under the neuron rule, its two-element pending entries
    are relayed signals, and it has no waiting list; it restores as it did and continues as the run it was."""
    g = Goo(count=24, across=4, seed=3, weight=None)
    g.readout, g.read, g.drive = "top", "count", "rate"
    g.set_delta(C.ESCAPE_DELTA)
    t = teach(g)
    for _ in range(4):
        t.epoch(verbose=False)
    data = checkpoint(g, tmp_path / "n.json", t)
    assert data["pending"], "something in flight to carry"
    old = {k: v for k, v in data.items() if k not in ("exploration", "waiting")}  # as every file before today
    (tmp_path / "old.json").write_text(json.dumps(old))
    new_back, _ = restore(tmp_path / "n.json")
    old_back, _ = restore(tmp_path / "old.json")
    assert old_back.exploration == "neuron" and not any(n.synaptic for n in old_back.all_neurons())
    assert all(not ventured for _, _, ventured in old_back.schedule.pending(marks=True))
    assert state(old_back) == state(new_back) == state(g)


@pytest.mark.parametrize("drive, tau, trace", [
    ("rate", math.inf, "all"), ("charged", math.inf, "all"), ("charged", 2.0, "ventured"), ("rate", 2.0, "all"),
])
def test_an_object_resume_under_exploration_at_the_synapse_is_the_same_run_continued(tmp_path, drive, tau, trace):
    """§12.11 on the object engine under exploration at the synapse: sixteen epochs uninterrupted against eight, a
    checkpoint with ventured signals in flight, and eight more. Every weight, trace, spike, gain, exposure clock and
    the queue with its marks must match (§8.14, §12.9)."""
    Neuron.tau = tau

    def build():
        g = goo(drive, h0=0.2, trace=trace)
        return g, teach(g)

    whole, straight = build()
    for _ in range(16):
        straight.epoch(verbose=False)

    part, teacher = build()
    for _ in range(8):
        teacher.epoch(verbose=False)
    assert any(ventured for _, _, ventured in part.schedule.pending(marks=True)), "ventured signals in flight at the cut"
    data = checkpoint(part, tmp_path / "half.json", teacher)

    back, data = restore(tmp_path / "half.json")
    back.readout, back.read = "top", "count"
    resumed = teach(back, seed=999)
    resumed.rng.setstate((3, tuple(data["explore_state"]), None))  # the seed 999 is overwritten by the state
    resume_teacher(resumed, data)
    for _ in range(8):
        resumed.epoch(verbose=False)
    assert state(back) == state(whole)
    assert [n.read_count for n in back.all_neurons()] == [n.read_count for n in whole.all_neurons()]
    assert resumed.baseline == straight.baseline


@pytest.mark.parametrize("synaptic", [True, False])
def test_an_arrival_waiting_past_the_horizon_survives_a_checkpoint(tmp_path, synaptic):
    """§3.4, §12.11: the drive draws an epoch's arrivals up to the horizon and the schedule runs what falls before it by
    more than the clock's slack, so an arrival drawn within that slack waits for the next epoch. A checkpoint carries
    it -- a charge (5.4b) or, under the neuron rule, a stimulus -- in the queue's order, and the run continued from the
    file is the run continued from memory. Until September 25, 2026 an object checkpoint dropped it."""
    def build():
        if synaptic:
            g = goo("charged")
        else:
            g = Goo(count=24, across=4, seed=3, weight=None)
            g.readout, g.read, g.drive = "top", "count", "rate"
            g.set_delta(C.ESCAPE_DELTA)
        t = teach(g)
        for _ in range(3):
            t.epoch(verbose=False)
        late = g.horizon - slack(g.horizon) / 2  # inside the slack: it waits (§3.4)
        if synaptic:
            g.schedule.charge(g.input_row()[1], g.drive_steps, late)
        else:
            g.schedule.stimulus(g.input_row()[1], late)
        return g, t

    kept, t_kept = build()
    waiting = persistence._waiting(kept)
    assert [entry[:3] for entry in waiting if entry[0] < kept.horizon] == [
        [kept.horizon - slack(kept.horizon) / 2, "charge" if synaptic else "stimulus", 1]]
    for _ in range(3):
        t_kept.epoch(verbose=False)

    saved, t_saved = build()
    data = checkpoint(saved, tmp_path / "w.json", t_saved)
    assert data["waiting"] == waiting
    back, data = restore(tmp_path / "w.json")
    back.readout, back.read = "top", "count"
    assert persistence._waiting(back) == waiting
    resumed = teach(back)
    resumed.rng.setstate((3, tuple(data["explore_state"]), None))
    resume_teacher(resumed, data)
    for _ in range(3):
        resumed.epoch(verbose=False)
    assert state(back) == state(kept)


def test_a_resume_under_the_other_exploration_is_refused(tmp_path):
    """§12.9 (Q17): a checkpoint saved under one exploration resumes under that one only. A network exploring at the
    synapse is not given a neuron-rule checkpoint, one that has run under the neuron rule is not given a synapse one,
    and a fresh network takes the checkpoint's exploration with its settings."""
    synapse_file, neuron_file = tmp_path / "s.json", tmp_path / "n.json"
    s = goo(h0=0.05)
    t = teach(s)
    t.epoch(verbose=False)
    checkpoint(s, synapse_file, t)
    n = Goo(count=24, across=4, seed=3, weight=None)
    n.readout, n.read = "top", "count"
    n.set_delta(C.ESCAPE_DELTA)
    t = teach(n)
    t.epoch(verbose=False)
    checkpoint(n, neuron_file, t)

    with pytest.raises(ValueError, match="§12.9"):
        load_weights(goo(), read_checkpoint(neuron_file))
    ran = Goo(count=24, across=4, seed=3, weight=None)
    ran.readout, ran.read = "top", "count"
    ran.set_delta(C.ESCAPE_DELTA)
    teach(ran).epoch(verbose=False)
    with pytest.raises(ValueError, match="§12.9"):
        load_weights(ran, read_checkpoint(synapse_file))
    fresh = Goo(count=24, across=4, seed=3, weight=None)
    load_weights(fresh, read_checkpoint(synapse_file))
    assert (fresh.exploration, fresh.synapse_hazard_rest) == ("synapse", 0.05)


# --- the command line ------------------------------------------------------------------------------------------------

def settled(argv, saved=None):
    """The command line's settings for `argv`, on the copy problem unless it names another: the default problem reads
    'fired', which exploration at the synapse refuses (§5.10), and these tests are about what else is settled."""
    args = build_parser().parse_args(["--problem", "copy"] + list(argv))
    apply_problem(args)
    apply_exploration(args, saved)
    return args


def test_the_command_line_names_exploration_at_the_synapse_and_its_settings():
    """The flags of §7.1, §7.6, §7.7, §8.17 and 5.4b: each a sentinel until named, settled to the register's value or
    the one named; --delta settles to ESCAPE_DELTA under the neuron rule and to no width under the synapse's (§6.13).
    TRACE is --trace-counts: --trace names the per-epoch CSV and still does."""
    args = settled(["--exploration", "synapse", "--synapse-hazard", "0.05", "--hazard-family", "linear",
                    "--synapse-scaling", "fan-out", "--trace-counts", "ventured", "--drive", "charged", "--drive-steps", "4"])
    assert (args.exploration, args.synapse_hazard, args.hazard_family, args.synapse_scaling, args.trace_counts,
            args.drive, args.drive_steps, args.delta) == ("synapse", 0.05, "linear", "fan-out", "ventured", "charged", 4, 0.0)
    assert args.named == {"exploration", "synapse_hazard", "hazard_family", "synapse_scaling", "trace_counts", "drive",
                          "drive_steps"}
    args = settled(["--exploration", "synapse", "--delta", "0"])  # the width §6.13 gives every neuron: asks for nothing
    assert (args.delta, args.synapse_hazard, args.trace_counts) == (0.0, C.SYNAPSE_HAZARD_REST, C.TRACE)
    assert settled([]).delta == C.ESCAPE_DELTA and settled(["--delta", "0.3"]).delta == 0.3
    assert settled(["--delta", str(C.ESCAPE_DELTA)]).named == {"delta"}  # named, though it is the default
    traced = build_parser().parse_args(["--trace", "ventured.csv", "--trace-counts", "ventured"])
    assert (traced.trace, traced.trace_counts) == ("ventured.csv", "ventured")


@pytest.mark.parametrize("argv, clause", [
    (["--exploration", "synapse", "--delta", "0.3"], "§6.13"),
    (["--synapse-hazard", "0.05"], "--exploration synapse"),
    (["--trace-counts", "ventured", "--hazard-family", "linear"], "§8.17"),
    (["--exploration", "synapse", "--synapse-hazard", "1"], "§7.6"),
    (["--exploration", "synapse", "--synapse-hazard", "-0.1"], "§7.6"),
    (["--drive", "charged"], "5.4b"),
    (["--exploration", "synapse", "--drive-steps", "4"], "5.4b"),
    # a width that is not one, under either rule: until September 25, 2026 exploration at the synapse settled it to 0
    (["--exploration", "synapse", "--delta", "-1"], "must not be negative, and must be a number"),
    (["--exploration", "synapse", "--delta", "nan"], "must not be negative, and must be a number"),
    (["--delta", "-1"], "must not be negative, and must be a number"),
    (["--delta", "nan"], "must not be negative, and must be a number"),
    # what the network and the engines refuse under exploration at the synapse, refused here before anything is built
    (["--exploration", "synapse", "--eligibility", "hebb"], "§8.3"),
    (["--exploration", "synapse", "--drive", "forced"], "§5.7"),
    (["--exploration", "synapse", "--read", "fired"], "§5.10"),
])
def test_the_command_line_refuses_what_the_file_refuses(argv, clause, capsys):
    """Each refusal says its clause (§12.2), and the run exits 2 before anything is built."""
    with pytest.raises(ValueError, match=re.escape(clause)):
        settled(argv)
    assert cli_main(argv + ["--problem", "copy", "--goo", "24", "--headless", "--no-save"]) == 2
    err = capsys.readouterr().err
    assert clause in err and "projections per neuron" not in err, "refused before the network is built"


@pytest.mark.parametrize("steps", ["0", "-1", "2.5", "three"])
def test_drive_steps_is_a_whole_number_of_at_least_one(steps, capsys):
    """5.4b: DRIVE_STEPS is a count of deliveries; any other value is refused with the clause, not rounded."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--drive-steps", steps])
    assert "5.4b" in capsys.readouterr().err


def test_the_width_under_the_neuron_rule_is_what_it_was(tmp_path, capsys):
    """--delta's sentinel moves nothing under the neuron rule: a run that names no width, and one that names
    ESCAPE_DELTA, are the run the command line made before the sentinel, bit for bit, and the banner says so."""
    common = ["--problem", "copy", "--goo", "24", "--headless", "--epochs", "4", "--seed", "3", "--no-trace"]
    assert cli_main(common + ["--save-weights", str(tmp_path / "none.json")]) == 0
    banner = capsys.readouterr().err
    assert cli_main(common + ["--save-weights", str(tmp_path / "named.json"), "--delta", str(C.ESCAPE_DELTA)]) == 0
    assert f"escape delta {C.ESCAPE_DELTA:g} (every hazard" in banner
    none, named = (json.loads((tmp_path / f"{x}.json").read_text()) for x in ("none", "named"))
    g = Goo(count=24, across=8, seed=3, weight=None, threshold=C.GOO_THRESHOLD, minimum_potential=C.GOO_MINIMUM_POTENTIAL)
    g.set_delta(C.ESCAPE_DELTA)
    assert none["weights"] == named["weights"] and none["escape_delta"] == named["escape_delta"] == C.ESCAPE_DELTA
    assert none["deltas"] == [n.delta for n in g.all_neurons()]
    assert (none["format"], none["exploration"]) == (2, "neuron")


def test_a_loaded_checkpoint_keeps_its_widths_unless_the_run_names_another(tmp_path):
    """§12.9 for the width: a neuron-rule checkpoint loaded with no --delta, or with its own, keeps each neuron's width
    as the file holds it -- set from the thresholds it started at, which homeostasis has since moved -- and one named
    that is not the checkpoint's is set afresh from the thresholds as they stand."""
    common = ["--problem", "copy", "--headless", "--epochs", "3", "--no-trace"]
    assert cli_main(common + ["--goo", "24", "--seed", "3", "--delta", "0.3", "--homeostasis", "0.01",
                              "--save-weights", str(tmp_path / "a.json")]) == 0
    a = json.loads((tmp_path / "a.json").read_text())
    moved = [theta * 0.3 for theta in a["thresholds"]]
    assert moved != a["deltas"], "homeostasis must have moved a threshold for the test to mean anything"
    for extra, want in (([], a["deltas"]), (["--delta", "0.3"], a["deltas"]), (["--delta", "0.455"], None)):
        out = tmp_path / "b.json"
        assert cli_main(common + ["--load-weights", str(tmp_path / "a.json"), "--save-weights", str(out)] + extra) == 0
        b = json.loads(out.read_text())
        if want is None:
            assert b["escape_delta"] == 0.455 and b["deltas"] != a["deltas"]
        else:
            assert b["escape_delta"] == 0.3 and b["deltas"] == want


def test_a_synapse_run_on_the_command_line_resumes_under_its_own_settings_and_its_drive(tmp_path, capsys):
    """§12.9 on the command line: a run saved under exploration at the synapse, the charged drive, TRACE ventured and a
    rest hazard of its own, loaded with no flags, resumes under all of them -- the drive included, which until
    September 25, 2026 the problem's replaced on every load. A flag named overrides; the other exploration is refused.
    A TRACE ventured run says in its banner and its checkpoints that its estimator is biased (§8.17)."""
    first = tmp_path / "first.json"
    assert cli_main(["--problem", "copy", "--goo", "24", "--headless", "--epochs", "3", "--seed", "3", "--no-trace",
                     "--exploration", "synapse", "--drive", "charged", "--drive-steps", "4", "--trace-counts", "ventured",
                     "--synapse-hazard", "0.05", "--save-weights", str(first)]) == 0
    assert TRACE_VENTURED_BIAS in capsys.readouterr().err
    saved = json.loads(first.read_text())
    assert (saved["format"], saved["drive"], saved["drive_steps"], saved["estimator_bias"]) == (
        3, "charged", 4, TRACE_VENTURED_BIAS)

    def load(*extra):
        out = tmp_path / "again.json"
        code = cli_main(["--problem", "copy", "--headless", "--epochs", "2", "--no-trace", "--load-weights", str(first),
                         "--save-weights", str(out), *extra])
        return code, (json.loads(out.read_text()) if code == 0 else None), capsys.readouterr().err

    code, again, err = load()
    assert code == 0 and (again["exploration"], again["drive"], again["drive_steps"], again["trace_mode"],
                          again["synapse_hazard_rest"]) == ("synapse", "charged", 4, "ventured", 0.05)
    assert TRACE_VENTURED_BIAS in err
    code, again, _ = load("--drive", "rate", "--synapse-hazard", "0.02", "--trace-counts", "all")
    assert code == 0 and (again["drive"], again["synapse_hazard_rest"], again["trace_mode"]) == ("rate", 0.02, "all")
    code, _, err = load("--exploration", "neuron")
    assert code == 2 and "§12.9" in err
    code, _, err = load("--delta", "0.2")
    assert code == 2 and "§6.13" in err


def test_the_drive_of_a_neuron_rule_checkpoint_is_kept_on_load(tmp_path):
    """§12.9: under the neuron rule too, a loaded network keeps the drive it was saved under unless --drive is given;
    until September 25, 2026 the problem's drive replaced it on every load."""
    first = tmp_path / "forced.json"
    assert cli_main(["--problem", "copy", "--goo", "24", "--headless", "--epochs", "2", "--seed", "3", "--no-trace",
                     "--drive", "forced", "--save-weights", str(first)]) == 0
    for extra, want in (([], "forced"), (["--drive", "rate"], "rate")):
        out = tmp_path / "out.json"
        assert cli_main(["--problem", "copy", "--headless", "--epochs", "2", "--no-trace", "--load-weights", str(first),
                         "--save-weights", str(out)] + extra) == 0
        assert json.loads(out.read_text())["drive"] == want


@pytest.mark.parametrize("flags", [
    [],  # the neuron rule at the register's width
    ["--exploration", "synapse", "--drive", "charged", "--synapse-hazard", "0.2"],
    ["--exploration", "synapse", "--drive", "charged", "--drive-steps", "5", "--tau", "2", "--threshold", "2",
     "--hazard-family", "linear", "--synapse-scaling", "fan-out", "--trace-counts", "ventured"],
    ["--tau", "2", "--hop", "2.4", "--rate-tau", "4"],  # the clock of §12.9, under the neuron rule
])
def test_a_run_loaded_on_the_command_line_is_the_same_run_continued(tmp_path, flags):
    """§12.11 on the command line: twelve epochs straight against five, saved, and seven more by --load-weights naming
    none of the run's settings. Every key of the two final checkpoints is the same -- weights, traces, spikes, the queue,
    the streams -- but the Teacher's progress log, whose entries carry the wall clock and follow each invocation's
    report cadence; its statistics are compared. Until September 25, 2026 a load reseeded the exploration stream, so
    every run parted from its uninterrupted self at the first resumed wave, and it ran on the constants' clock, not the
    checkpoint's (§12.9)."""
    common = ["--problem", "copy", "--goo", "24", "--headless", "--seed", "3", "--no-trace"] + flags
    assert cli_main(common + ["--epochs", "12", "--save-weights", str(tmp_path / "straight.json")]) == 0
    assert cli_main(common + ["--epochs", "5", "--save-weights", str(tmp_path / "half.json")]) == 0
    Neuron.tau, Neuron.rate_tau = C.TAU, C.RATE_TAU  # the constants' clock: what a load that named none used to run on
    assert cli_main(["--problem", "copy", "--headless", "--no-trace", "--epochs", "7", "--load-weights",
                     str(tmp_path / "half.json"), "--save-weights", str(tmp_path / "resumed.json")]) == 0
    straight, half, resumed = (json.loads((tmp_path / f"{x}.json").read_text()) for x in ("straight", "half", "resumed"))
    assert straight["spikes"] != half["spikes"] and straight["explore_state"] != half["explore_state"], \
        "the run must go on past the cut for the comparison to mean anything"
    assert set(straight) == set(resumed) and [k for k in straight if k != "learning" and straight[k] != resumed[k]] == []
    history = lambda data: {k: v for k, v in data["learning"].items() if k != "history"}
    assert history(straight) == history(resumed)


def test_a_driver_checkpoint_loaded_on_the_command_line_brings_its_baseline_and_its_stream(tmp_path, monkeypatch):
    """A checkpoint the sweep driver wrote keeps the baseline beside its fields and has no learning record (§9.3), and
    carries the exploration stream's state (§12.11): a Teacher resumed from it takes both, on the command line as through
    `resume_teacher`. Until September 25, 2026 the command line's Teacher started a fresh baseline and a reseeded stream."""
    import random
    import walnutbutter.cli as cli
    stream = random.Random(7)
    [stream.random() for _ in range(5)]
    data = {"baseline": 0.25, "explore_state": list(stream.getstate()[1])}
    teacher = teach(goo(), seed=3)
    resume_teacher(teacher, data)
    assert teacher.baseline == 0.25 and teacher.rng.getstate() == stream.getstate()
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    r = rs()
    grid, args = r.grid_of("copy", {"threshold": 0.6, "goo": 40.0, "seed": 3}, "hazard", True, -4.0, None,
                           ("--exploration", "synapse"))
    _, _, engine, report = fast.train(grid, 6, lr=args.lr, patterns=input_stream(6, grid.raw_bit_count(), 3), seed=3,
                                      eligibility="hazard", target="copy")
    r._save_network(engine, grid, report, tmp_path / "driver.json")
    saved = json.loads((tmp_path / "driver.json").read_text())
    assert "learning" not in saved and saved["baseline"] is not None
    seen = {}
    real = cli.resume_teacher
    def spy(teacher, data):
        real(teacher, data)
        seen.update(baseline=teacher.baseline, stream=list(teacher.rng.getstate()[1]))
    monkeypatch.setattr(cli, "resume_teacher", spy)
    assert cli_main(["--problem", "copy", "--headless", "--epochs", "1", "--no-trace", "--no-save", "--load-weights",
                     str(tmp_path / "driver.json")]) == 0
    assert seen == {"baseline": saved["baseline"], "stream": saved["explore_state"]}


def test_seeds_run_under_exploration_at_the_synapse(tmp_path, capsys):
    """--seeds carries the exploration and its settings to every worker, names them in the header, and says under TRACE
    ventured that the estimator is biased (§8.17); each seed's checkpoint is format 3."""
    base = tmp_path / "s.json"
    assert cli_main(["--problem", "copy", "--goo", "24", "--seeds", "2", "--seed", "3", "--epochs", "2",
                     "--exploration", "synapse", "--trace-counts", "ventured", "--hazard-family", "linear",
                     "--save-weights", str(base)]) == 0
    err = capsys.readouterr().err
    assert "exploration at the synapse, h0 0.01, the linear family" in err and TRACE_VENTURED_BIAS in err
    assert "escape delta" not in err
    for seed in (3, 4):
        data = json.loads((tmp_path / f"s-seed{seed}.json").read_text())
        assert (data["format"], data["synapse_hazard_family"], data["trace_mode"]) == (3, "linear", "ventured")
    assert cli_main(["--problem", "copy", "--goo", "24", "--seeds", "2", "--epochs", "2", "--no-save",
                     "--exploration", "synapse", "--delta", "0.4"]) == 2


# --- the sweep driver --------------------------------------------------------------------------------------------------

def test_grid_of_gives_the_default_width_under_the_neuron_rule_only():
    """docs/rust-sweep.py builds as the command line does: the default width under the neuron rule, none under
    exploration at the synapse (§6.13), a positive one named with it refused, and a swept string knob's value applied."""
    r = rs()
    arm = {"goo": 24.0, "seed": 3}
    neuron, args = r.grid_of("copy", arm, "hazard", True, -4.0, None, ())
    assert (neuron.exploration, neuron.escape_delta, args.delta) == ("neuron", C.ESCAPE_DELTA, C.ESCAPE_DELTA)
    synapse, args = r.grid_of("copy", arm, None, True, -4.0, None, ("--exploration", "synapse", "--drive", "charged"))
    assert (synapse.exploration, synapse.escape_delta, synapse.drive, args.eligibility) == ("synapse", 0.0, "charged", None)
    assert all(n.delta == 0.0 and n.synaptic for n in synapse.all_neurons())
    assert synapse.named_settings == {"exploration", "drive"}
    swept, _ = r.grid_of("copy", dict(arm, synapse_hazard=0.05, trace_counts="ventured", drive_steps=2.0), "hazard", True,
                         -4.0, None, ("--exploration", "synapse", "--drive", "charged"))
    assert (swept.synapse_hazard_rest, swept.trace_mode, swept.drive_steps) == (0.05, "ventured", 2)
    with pytest.raises(ValueError, match="§6.13"):
        r.grid_of("copy", dict(arm, delta=0.3), "hazard", True, -4.0, None, ("--exploration", "synapse"))


def test_the_string_knobs_are_fixed_by_one_value_and_swept_by_more(monkeypatch):
    """The exploration, the family, the scaling, the trace and the drive, handled as the eligibility is: one value
    runs the whole sweep under it (a fixed word, out of the arm's name), more than one sweeps it; the rest hazard and
    DRIVE_STEPS are numeric knobs, and a DRIVE_STEPS that is not a whole number is refused (5.4b)."""
    r = rs()
    monkeypatch.setattr("sys.argv", ["rust-sweep", "--name", "x", "--problem", "copy", "--exploration", "synapse",
                                     "--trace-counts", "all", "ventured", "--synapse-hazard", "0.01", "0.05",
                                     "--drive", "charged", "--drive-steps", "3", "--seed", "1", "2"])
    args = r.parse()
    swept, arms = r.grid_and_arms(args)
    assert swept == ["synapse_hazard", "trace_counts", "seed"] and len(arms) == 8
    assert r.fixed_words(args) == ["--exploration", "synapse", "--drive", "charged"]
    assert r.arm_name(arms[1]) == "synapse_hazard0.01-drive_steps3-trace_countsall-seed2"
    monkeypatch.setattr("sys.argv", ["rust-sweep", "--name", "x", "--problem", "copy", "--drive-steps", "2.5"])
    with pytest.raises(SystemExit):
        r.parse()


def test_the_arm_json_names_the_exploration_and_the_drive(tmp_path, monkeypatch):
    """Every arm's json names the exploration, h0, the family, the scaling, the trace, the drive and DRIVE_STEPS --
    null where the neuron rule leaves them out of force -- and a TRACE ventured arm says its estimator is biased (§8.17).
    An arm the command line refuses says so in its line instead of stopping the sweep."""
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    r = rs()
    monkeypatch.setattr(r, "ROOT", tmp_path)
    (tmp_path / "runs" / "arms").mkdir(parents=True)
    arm = {"goo": 24.0, "seed": 3}
    job = lambda fixed: (arm, "copy", 3, 1, "arms", "hazard", True, -4.0, None, None, fixed, 0)
    synapse = r.run_arm(job(("--exploration", "synapse", "--drive", "charged", "--trace-counts", "ventured",
                             "--hazard-family", "linear", "--synapse-scaling", "fan-out", "--drive-steps", "4",
                             "--synapse-hazard", "0.3")))
    assert {k: synapse[k] for k in ("exploration", "synapse_hazard", "hazard_family", "synapse_scaling", "trace_counts",
                                    "drive", "drive_steps", "estimator_bias", "delta")} == {
        "exploration": "synapse", "synapse_hazard": 0.3, "hazard_family": "linear",
        "synapse_scaling": "fan-out", "trace_counts": "ventured", "drive": "charged", "drive_steps": 4,
        "estimator_bias": TRACE_VENTURED_BIAS, "delta": 0.0}
    assert json.loads((tmp_path / "runs" / "arms" / f"{r.arm_name(arm)}.json").read_text()) == synapse
    counts = synapse["output_counts_last"]
    saved = json.loads((tmp_path / "runs" / "arms" / f"{r.arm_name(arm)}-network.json").read_text())
    outputs = saved["read_counts"][-len(counts):]
    assert sum(outputs) > 0 and all(c >= r_ for c, r_ in zip(counts, outputs)), "the read counts are in the count"
    (tmp_path / "runs" / "arms" / f"{r.arm_name(arm)}.csv").unlink()
    (tmp_path / "runs" / "arms" / f"{r.arm_name(arm)}-network.json").unlink()
    neuron = r.run_arm(job(()))
    assert (neuron["exploration"], neuron["synapse_hazard"], neuron["hazard_family"], neuron["synapse_scaling"],
            neuron["trace_counts"], neuron["estimator_bias"], neuron["drive"], neuron["drive_steps"],
            neuron["delta"]) == ("neuron", None, None, None, None, None, "rate", None, C.ESCAPE_DELTA)
    refused = r.run_arm((dict(arm, seed=4), "copy", 3, 1, "arms", "hazard", True, -4.0, None, None, ("--drive", "charged"), 0))
    assert "5.4b" in refused["refused"]


def test_a_resumed_arm_keeps_its_checkpoint_s_settings_unless_the_sweep_names_them(tmp_path):
    """§12.9 at the driver: `resume_grid` no longer writes the arm's drive over the checkpoint's, and keeps the settings
    of §7.6, §7.7, §8.17 and 5.4b and the width the checkpoint was saved under, except those the sweep names; one that
    names the other exploration is refused. A width named that is the checkpoint's own leaves its widths as saved."""
    r = rs()
    arm = {"goo": 24.0, "seed": 3}
    saved, _ = r.grid_of("copy", arm, "hazard", True, -4.0, None,
                         ("--exploration", "synapse", "--drive", "charged", "--synapse-hazard", "0.05"))
    checkpoint(saved, tmp_path / "s.json")

    plain, _ = r.grid_of("copy", arm, "hazard", True, -4.0, None, ())  # names nothing: the checkpoint's stand
    back = r.resume_grid(plain, tmp_path / "s.json")[0]
    assert (back.exploration, back.drive, back.drive_steps, back.synapse_hazard_rest) == ("synapse", "charged", 3, 0.05)
    named, _ = r.grid_of("copy", arm, "hazard", True, -4.0, None,
                         ("--exploration", "synapse", "--drive", "rate", "--trace-counts", "ventured"))
    back = r.resume_grid(named, tmp_path / "s.json")[0]
    assert (back.drive, back.trace_mode, back.synapse_hazard_rest) == ("rate", "ventured", 0.05)
    assert all(n.trace_ventured for n in back.all_neurons())
    steps, _ = r.grid_of("copy", dict(arm, drive_steps=5.0), "hazard", True, -4.0, None,
                         ("--exploration", "synapse", "--drive", "charged"))
    back = r.resume_grid(steps, tmp_path / "s.json")[0]
    assert (back.drive, back.drive_steps) == ("charged", 5)
    other, _ = r.grid_of("copy", arm, "hazard", True, -4.0, None, ("--exploration", "neuron"))
    with pytest.raises(ValueError, match="§12.9"):
        r.resume_grid(other, tmp_path / "s.json")

    width, _ = r.grid_of("copy", dict(arm, delta=0.3), "hazard", True, -4.0, None, ())
    for n in width.all_neurons():
        n.threshold *= 1.5  # a threshold the run has moved, which a width set afresh would be taken from
    width.drive = "forced"
    checkpoint(width, tmp_path / "n.json")
    same, _ = r.grid_of("copy", dict(arm, delta=0.3), "hazard", True, -4.0, None, ())
    back = r.resume_grid(same, tmp_path / "n.json")[0]
    assert [n.delta for n in back.all_neurons()] == [n.delta for n in width.all_neurons()] and back.drive == "forced"
    wider, _ = r.grid_of("copy", dict(arm, delta=0.4), "hazard", True, -4.0, None, ())
    back = r.resume_grid(wider, tmp_path / "n.json")[0]
    assert back.escape_delta == 0.4 and [n.delta for n in back.all_neurons()] == [0.4 * n.threshold for n in back.all_neurons()]


def test_a_resumed_arm_records_the_width_and_the_exploration_it_ran(tmp_path, monkeypatch):
    """An arm's json records the network the arm ran, a resumed one's being its checkpoint's where the sweep names
    nothing (§12.9): a synapse checkpoint resumed by a sweep that names no exploration records exploration at the synapse
    and width 0 (§6.13), and a neuron-rule checkpoint of width 0.3 records 0.3. Until September 25, 2026 both recorded
    the fresh arm's --delta, ESCAPE_DELTA. The sweep's docs/<name>.md names the exploration the arms ran and, under TRACE
    ventured, the estimator's bias (§8.17)."""
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    r = rs()
    monkeypatch.setattr(r, "ROOT", tmp_path)
    arm = {"threshold": 0.6, "goo": 24.0, "seed": 3}
    job = lambda sweep, fixed, source=None: (arm, "copy", 3, 1, sweep, "hazard", True, -4.0, None, source, fixed, 0)
    for sweep in ("syn", "resumed", "wide", "kept"):
        (tmp_path / "runs" / sweep).mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    r.run_arm(job("syn", ("--exploration", "synapse", "--trace-counts", "ventured", "--hazard-family", "linear")))
    plain = r.run_arm(job("resumed", (), "syn"))
    saved = json.loads((tmp_path / "runs" / "resumed" / f"{r.arm_name(arm)}-network.json").read_text())
    assert (plain["exploration"], plain["delta"]) == (saved["exploration"], saved["escape_delta"]) == ("synapse", 0.0)
    assert (plain["hazard_family"], plain["trace_counts"], plain["estimator_bias"]) == ("linear", "ventured", TRACE_VENTURED_BIAS)

    wide, _ = r.grid_of("copy", dict(arm, delta=0.3), "hazard", True, -4.0, None, ())
    checkpoint(wide, tmp_path / "runs" / "wide" / f"{r.arm_name(arm)}-network.json")  # the arm's name, a width of its own
    kept = r.run_arm(job("kept", (), "wide"))
    assert kept["delta"] == 0.3 and kept["exploration"] == "neuron"

    monkeypatch.setattr("sys.argv", ["rust-sweep", "--name", "resumed", "--problem", "copy", "--goo", "24", "--threshold",
                                     "0.6", "--seed", "3", "--epochs", "3", "--summary"])
    r.summarise(r.parse())
    header = (tmp_path / "docs" / "resumed.md").read_text().splitlines()[0]
    assert "exploration at the synapse, as the arms' checkpoints were saved" in header and TRACE_VENTURED_BIAS in header


@pytest.mark.parametrize("problem, eligibility, fixed, extra, clause", [
    ("copy", "hebb", ("--exploration", "synapse"), {}, "§8.3"),
    ("copy", "hazard", ("--exploration", "synapse", "--drive", "forced"), {}, "§5.7"),
    ("reversal", "hazard", ("--exploration", "synapse"), {}, "§5.10"),
    ("copy", "hazard", (), {"delta": 0.0}, "§8.3"),  # the neuron rule at width 0: refused by fast.train itself
])
def test_an_arm_the_mode_or_the_engine_refuses_says_so_and_the_sweep_goes_on(tmp_path, monkeypatch, problem, eligibility,
                                                                              fixed, extra, clause):
    """A refused arm returns {"arm", "refused"} with its clause (§12.2) and writes nothing, rather than raising in its
    worker, which kills the sweep and every sibling arm's work with it. The first three are refused where the command
    line refuses them (apply_exploration); until September 25, 2026 they, and the last, were left to fast.train."""
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    r = rs()
    monkeypatch.setattr(r, "ROOT", tmp_path)
    (tmp_path / "runs" / "arms").mkdir(parents=True)
    arm = dict({"goo": 24.0}, **extra, seed=3)
    result = r.run_arm((arm, problem, 3, 1, "arms", eligibility, True, -4.0, None, None, fixed, 0))
    assert set(result) == {"arm", "refused"} and clause in result["refused"]
    assert list((tmp_path / "runs" / "arms").iterdir()) == []


@pytest.mark.parametrize("steps", ["nan", "inf", "2.5", "0"])
def test_the_sweep_refuses_a_drive_steps_that_is_not_a_whole_number(steps, monkeypatch, capsys):
    """5.4b at the sweep's parser: nan and inf are refused with the clause as 2.5 is, not left to crash the check."""
    r = rs()
    monkeypatch.setattr("sys.argv", ["rust-sweep", "--name", "x", "--problem", "copy", "--drive-steps", steps])
    with pytest.raises(SystemExit):
        r.parse()
    assert "5.4b" in capsys.readouterr().err


def test_the_sweep_hands_drive_steps_on_as_the_whole_number_it_is():
    """A DRIVE_STEPS the sweep took as a whole number reaches the command line as one: 1e6 as 1000000, not "1e+06",
    which the command line's parser refuses (5.4b)."""
    r = rs()
    grid, args = r.grid_of("copy", {"goo": 24.0, "drive_steps": 1e6, "seed": 3}, "hazard", True, -4.0, None,
                           ("--exploration", "synapse", "--drive", "charged"))
    assert grid.drive_steps == args.drive_steps == 1_000_000


def test_the_watcher_names_what_the_network_runs():
    """docs/mnist-watch.py's banner names the width and TAU the network runs -- a resumed one's being its checkpoint's
    (§12.9) -- and, under TRACE ventured, the estimator's bias (§8.17)."""
    source = (ROOT / "docs" / "mnist-watch.py").read_text()
    assert "TRACE_VENTURED_BIAS" in source and "grid.escape_delta" in source and "Neuron.tau" in source
    assert "cli.delta" not in source and "cli.tau" not in source


def test_the_mnist_drivers_read_the_count_the_read_takes():
    """docs/mnist-watch.py, docs/mnist-evidence-diag.py and docs/mnist-evidence-gradient.py read each output's count
    as the read does, spikes plus the read synapse's escapes (§5.10, §7.9), through fast._counts."""
    for name in ("mnist-watch.py", "mnist-evidence-diag.py", "mnist-evidence-gradient.py"):
        source = (ROOT / "docs" / name).read_text()
        assert "fast._counts(engine)" in source and "epoch_spike_counts" not in source, name


# --- the live sweep checkpoints, on copies ---------------------------------------------------------------------------

def _runs() -> Path | None:
    """The repository's runs/, or where this is a git worktree without one, the main checkout's -- read only."""
    if (ROOT / "runs").is_dir():
        return ROOT / "runs"
    try:
        common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    runs = (ROOT / common).resolve().parent / "runs"
    return runs if runs.is_dir() else None


LIVE = ("mnist-1m-lowlr", "mnist-1m-hidden", "mnist-1m-leaky-10seed")
BEFORE = "ca4fe3e"  # the code before this step: the merge of PR #21's branch under the synapse engines
BEFORE_RUN = """
import importlib.util, json, sys
from pathlib import Path
code, root, job = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
import walnutbutter
assert Path(walnutbutter.__file__).resolve().is_relative_to(code.resolve()), walnutbutter.__file__
spec = importlib.util.spec_from_file_location("rs", code / "docs" / "rust-sweep.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from walnutbutter.neuron import Neuron
Neuron.verbose = False
rs.ROOT = root
print(json.dumps(rs.run_arm(tuple(job))))
"""
# what this step changed on purpose in what a resumed arm writes: the checkpoint names its exploration and the stimuli and
# charges waiting in the queue, its signals in flight are the engine's queue and not the one the network was restored
# with, and its notes are the engine's (§12.9); the arm's json names the exploration and the drive (§7.1, 5.4b)
ADDED = {"exploration", "waiting", "pending", "notes"}
ARM_ADDED = {"exploration", "synapse_hazard", "hazard_family", "synapse_scaling", "trace_counts", "estimator_bias", "drive",
             "drive_steps"}
WALL_CLOCK = {"seconds", "epochs_per_second"}


@pytest.mark.parametrize("sweep", LIVE)
def test_a_live_sweep_checkpoint_resumes_through_the_sweep_as_the_code_before_this_step_resumed_it(tmp_path, sweep):
    """The format-2 checkpoints on disk still resume exactly: copies of an arm's three files, continued by `run_arm
    --resume-from` -- the path a sweep takes, through grid_of, the arm's knobs from its name, what the arm names
    and the checkpoint's own settings (§12.9) -- for six epochs with a checkpoint written mid-run, under this code and
    under the code before this step (git archive of BEFORE, the same engine). The two land on the same weights, stream,
    thresholds, queue in the engine's terms and record; they differ only in what the step added on purpose, and the
    file the resume writes is still format 2, its queue unmarked. runs/ is read and nothing under it is written."""
    runs = _runs()
    if runs is None or not (runs / sweep).is_dir():
        pytest.skip("runs/ is not on disk here")
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    source = next((p for p in sorted((runs / sweep).glob("*-network.json"))
                   if json.loads(p.read_text()).get("engine_pending")), None)
    if source is None:
        pytest.skip(f"no checkpoint of {sweep} carries engine_pending")
    name = source.name[:-len("-network.json")]
    if not all((runs / sweep / f"{name}{suffix}").exists() for suffix in (".json", ".csv")):
        pytest.skip(f"{name} has no record beside its checkpoint")
    before_code = tmp_path / "before"
    before_code.mkdir()
    try:
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", BEFORE, "src", "docs/rust-sweep.py"], capture_output=True,
                                 check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(before_code)], input=archive, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"the code before this step ({BEFORE}) is not in this repository's history")
    from walnutbutter import mnist
    (before_code / "mnist").symlink_to(mnist.FOLDER)  # the data beside its src/, where mnist.py looks for it
    original = {suffix: (runs / sweep / f"{name}{suffix}").read_bytes() for suffix in ("-network.json", ".json", ".csv")}
    r = rs()
    arm = {knob: float(value) for knob, value in re.findall(r"([a-z_]+?)(-?[0-9.]+)(?:-|$)", name)}
    assert r.arm_name(arm) == name
    job = (arm, "mnist", 6, 2, "cont", "hazard", True, None, None, sweep, (), 4)

    for side in ("now", "before"):
        (tmp_path / side / "runs" / "cont").mkdir(parents=True)
        (tmp_path / side / "runs" / sweep).mkdir()
        for suffix, content in original.items():  # copies: the resume reads them and writes under runs/cont
            (tmp_path / side / "runs" / sweep / f"{name}{suffix}").write_bytes(content)
    import sys
    was = {attr: getattr(Neuron, attr) for attr in persistence.CLOCK}
    r.ROOT = tmp_path / "now"
    try:
        now = r.run_arm(job)
    finally:
        for attr, value in was.items():
            setattr(Neuron, attr, value)
    path = [str(before_code / "src")] + [p for p in sys.path if p and Path(p).resolve() != (ROOT / "src").resolve()]
    ran = subprocess.run([sys.executable, "-c", BEFORE_RUN, str(before_code), str(tmp_path / "before"), json.dumps(job)],
                         capture_output=True, text=True, env={"PYTHONPATH": ":".join(path), "PATH": "/usr/bin:/bin"})
    assert ran.returncode == 0, ran.stderr[-2000:]
    before = json.loads(ran.stdout.strip().splitlines()[-1])
    assert now["epoch_offset"] == before["epoch_offset"] > 0 and now["epochs_run"] == before["epochs_run"] == 6

    files = lambda side, suffix: (tmp_path / side / "runs" / "cont" / f"{name}{suffix}").read_text()
    new, old = json.loads(files("now", "-network.json")), json.loads(files("before", "-network.json"))
    assert set(new) - set(old) <= ADDED and set(old) <= set(new)
    assert [key for key in old if key not in ADDED and old[key] != new[key]] == []
    assert files("now", ".csv") == files("before", ".csv")
    assert set(now) - set(before) == ARM_ADDED
    assert [key for key in before if key not in WALL_CLOCK and before[key] != now[key]] == []
    assert (now["exploration"], now["drive"]) == ("neuron", "rate")

    assert (new["format"], new["exploration"]) == (2, "neuron")
    assert all(len(e) == 3 for e in new["engine_pending"]) and all(len(e) == 2 for e in new["pending"])
    from walnutbutter.propagation import SIGNAL
    back, _ = restore(tmp_path / "now" / "runs" / "cont" / f"{name}-network.json")
    ids = [c.id for n in back.all_neurons() for c in n.outgoing]  # the engine's edge order, as connection ids
    assert sorted(map(tuple, new["pending"])) == sorted((t, ids[k]) for t, kind, k in new["engine_pending"] if kind == SIGNAL)
    assert {suffix: (runs / sweep / f"{name}{suffix}").read_bytes() for suffix in original} == original, "runs/ untouched"
