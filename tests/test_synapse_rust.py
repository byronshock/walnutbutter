"""Exploration at the synapse on the Rust loop (AUTHORITY.md §7.5-§7.9, §8.16-§8.17, 5.4b): the object engine and Rust agree
to the bit (§8.15, §12.4).

fast.compare holds the two to each other with == after every epoch: every wave's time and what it fired in what order,
the spikes, the read counts, the potentials and the times they were last brought up to date, the exposure clocks, the
spike times and the driven marks, every trace and the moment it was brought up to, every synapse's stamp of its last
signal, the scores, notes and gains, the weights, the queue left in flight with each signal's ventured mark, the settings
the engine holds against the network's, and the exploration stream's state. The arms run goo 60 on the copy problem and
goo 455 on mnist's configuration (--hidden-neurons 0) over {rate, charged} x {all, ventured} x {loglinear, linear}, with
the fan-out scaling and a finite TAU among them, learning off and on, and the run options that move under the mode:
thresholds moved by homeostasis and un-sticking, the quash, a rest hazard other than the default and DRIVE_STEPS other
than 3. Waves built by hand hold what no drawn run is sure to: a charge and a signal in one wave, a threshold moved
while a charge is in flight, an output zone out of index order with a neuron at two places. Skipped when the extension
is not built, like every Rust test (CI does not build it).
"""

from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path

import pytest

from walnutbutter import fast
from walnutbutter.constants import TARGET_RATE, UNSTICK_TARGET
from walnutbutter.goo import Goo
from walnutbutter.learning import Teacher
from walnutbutter.monitor import run_epoch
from walnutbutter.network import input_stream
from walnutbutter.neuron import Neuron
from walnutbutter.propagation import EXTERNAL, SIGNAL

pytestmark = pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")

COMBOS = [(drive, trace, family) for drive in ("rate", "charged") for trace in ("all", "ventured")
          for family in ("loglinear", "linear")]


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    """Every Neuron class setting an arm or grid_of moves comes back after the test."""
    monkeypatch.setattr(Neuron, "verbose", False)
    for name in ("tau", "refractory", "hop", "bored_after", "rate_tau"):
        monkeypatch.setattr(Neuron, name, getattr(Neuron, name))


def goo(seed=3, weight=None, drive="rate", **settings) -> Goo:
    """Goo 60 on the copy problem's read (§11), exploring at the synapse. A weight of 0.02 on every synapse keeps many
    sources above zero potential, so §8.16's entry posts at most waves and most weights move under the rule."""
    g = Goo(count=60, across=8, seed=seed, weight=weight)
    g.rule, g.drive, g.read = "reinforce", "rate", "count"
    g.set_exploration("synapse", **settings)
    g.drive = drive
    return g


def _rust_sweep():
    spec = importlib.util.spec_from_file_location("rs", Path(__file__).resolve().parent.parent / "docs" / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    return rs


# --- the arithmetic ------------------------------------------------------------------------------------------------


def test_the_engine_s_hazard_is_the_object_engine_s_to_the_bit():
    """§7.5's engine note: u, h and m in the same forms and order -- the loglinear family through the platform's pow, which
    CPython's float ** and Rust's powf both reach -- over potentials below zero, at zero, inside and past the threshold,
    exposures negative and positive, h0 = 0 among the rest hazards, and the goo thresholds the fan-in gives (§4.10)."""
    import walnutbutter_schedule as engine
    rng = random.Random(11)
    neuron = Neuron(threshold=1.0)
    rests = (0.0, 0.01, 0.001, 0.5, 0.25, 0.999)
    for trial in range(20000):
        rest = rng.choice(rests) if trial % 2 else rng.random()
        theta = rng.choice((1.0, 0.2 * (rng.randint(1, 400) / 18.0), rng.random() * 3 + 1e-3))
        v = rng.choice((0.0, -rng.random(), rng.random() * theta, theta * rng.randint(0, 6) / rng.randint(1, 6)))
        since = rng.random() * 10
        now = since + rng.choice((0.0, Neuron.hop, rng.random() * 20, -1.0, 1e9))
        scale = rng.choice((1.0, math.sqrt(60 / 455), math.sqrt(60 / 455) / rng.randint(1, 40)))
        neuron.threshold, neuron.potential, neuron.last_update = theta, v, now
        neuron.exposed_since, neuron.synapse_scale = since, scale
        for family in ("loglinear", "linear"):
            assert engine.synapse_hazard(v, theta, now - since, Neuron.hop, scale, rest, family) == \
                neuron.synapse_expected(now, rest, family == "linear")


# --- agreement ------------------------------------------------------------------------------------------------------

GOO_VARIANTS = [  # (scaling, TAU, seed, weight, learning, options): each combination runs all four
    ("count", math.inf, 1, None, False, {}),
    ("count", math.inf, 2, 0.02, True, {}),
    ("fan-out", math.inf, 3, 0.02, False, {}),
    ("fan-out", 2.0, 4, 0.02, True, {}),
    # the run options that move under the mode. Thresholds moved between epochs by homeostasis and un-sticking (§9.9,
    # §9.10), which the charge (5.4b) and u (§7.5) read as they stand when they are read, at rates that keep every one
    # above zero; the quash (§10.2), which reads the stamps relayed and ventured arrivals leave (§1.7, §7.9), under TRACE
    # ventured too, where a relayed arrival moves no trace but is still taken in (§8.17); a rest hazard other than the
    # default, 0 among them (§7.6); DRIVE_STEPS other than 3 (5.4b) under the charged drive
    ("count", math.inf, 2, 0.02, True, {"homeostasis": 1e-3, "unstick": 1e-3}),
    ("fan-out", 2.0, 4, 0.02, True, {"homeostasis": 1e-4, "unstick": 1e-3, "quash": 0.2, "h0": 0.05, "steps": 5}),
    ("count", math.inf, 3, 0.02, False, {"quash": 0.02, "h0": 0.0, "steps": 2}),
    ("count", math.inf, 1, 0.02, False, {"quash": 0.3, "h0": 0.1}),
]


@pytest.mark.parametrize("drive,trace,family", COMBOS)
def test_goo_60_copy_agrees_to_the_bit(drive, trace, family):
    """The copy problem on goo 60, 20 epochs an arm: pure dynamics from a stream of its own, and the reinforce rule
    posted at the synapses' decisions (§8.16) under a Teacher at a rate that moves the weights, the evidence accumulator
    and the leak, the count's scaling and the fan-out's (§7.7), and the run options that move under the mode."""
    ventured = moved = scored = shifted = quashed = 0
    for scaling, tau, seed, weight, learning, options in GOO_VARIANTS:
        Neuron.tau = tau
        rest = {"h0": options["h0"]} if "h0" in options else {}  # the default otherwise
        g = goo(seed=seed, weight=weight, drive=drive, trace=trace, family=family, scaling=scaling, **rest)
        g.quash_rate = options.get("quash", 0.0)
        g.drive_steps = options.get("steps", g.drive_steps)
        before = [c.weight for c in g.connections.values()]
        thresholds = [n.threshold for n in g.all_neurons()]
        if learning:
            teacher = Teacher(g, seed=seed + 100, rule="reinforce", eligibility="hazard", target="copy", lr=0.5,
                              homeostasis=options.get("homeostasis", 0.0), unstick=options.get("unstick", 0.0))
            parted = fast.compare(g, epochs=20, teacher=teacher)
        else:
            parted = fast.compare(g, epochs=20, rng=random.Random(seed + 200))
        assert parted == [], (scaling, tau, seed, weight, learning, options, parted)
        assert sum(n.spikes for n in g.all_neurons()) > 0
        if learning:
            scored += sum(1 for c in g.connections.values() if c.score)
            moved += sum(a != c.weight for a, c in zip(before, g.connections.values()))
        elif g.quash_rate:
            quashed += sum(a != c.weight for a, c in zip(before, g.connections.values()))  # the quash alone moved them
        if "homeostasis" in options:
            shifted += sum(a != n.threshold for a, n in zip(thresholds, g.all_neurons()))
        ventured += sum(len(wave.ventured) for wave in g.waves)
    assert scored > 0  # the last epochs' entries posted and were settled into the scores (§8.16), paid or not
    assert moved > 0  # and an advantage paid them into the weights (a reward that never moves pays nothing)
    assert ventured > 0  # the last epochs delivered escapes, so the marks were exercised
    assert shifted > 0 and quashed > 0  # the thresholds moved between epochs, and the quash weakened synapses


MNIST_ARMS = [  # (drive, trace, family, scaling, TAU, seed, learning, options): each drive, trace and family both ways
    ("rate", "all", "loglinear", "count", math.inf, 1, True, {}),
    ("rate", "all", "linear", "fan-out", math.inf, 2, True, {}),
    ("rate", "ventured", "loglinear", "count", 2.0, 3, True, {}),
    ("rate", "ventured", "linear", "count", math.inf, 4, False, {}),
    ("charged", "all", "loglinear", "fan-out", math.inf, 5, True, {}),
    ("charged", "all", "linear", "count", 2.0, 6, True, {}),
    ("charged", "ventured", "loglinear", "count", math.inf, 7, False, {}),
    ("charged", "ventured", "linear", "fan-out", 20.0, 8, True, {}),
    # thresholds moved by homeostasis and un-sticking (§9.9, §9.10), a rest hazard other than the default (§7.6),
    # DRIVE_STEPS other than 3 (5.4b)
    ("rate", "all", "loglinear", "count", math.inf, 9, True, {"homeostasis": 1e-4, "unstick": 1e-3, "h0": 0.05}),
    ("charged", "ventured", "loglinear", "fan-out", 2.0, 10, True, {"homeostasis": 1e-4, "h0": 0.05, "steps": 2}),
]


@pytest.mark.parametrize("drive,trace,family,scaling,tau,seed,learning,options", MNIST_ARMS)
def test_the_mnist_configuration_agrees_to_the_bit(drive, trace, family, scaling, tau, seed, learning, options):
    """mnist's configuration as the sweep driver builds it (problems.py; goo 455, --hidden-neurons 0: 395 inputs, 60
    outputs, some 1,400 synapses as the seed wires them, 60 read synapses), on the digits, paid by the evidence critic at
    the problem's rate, 20 epochs an arm. The epoch is cut to 5 ms, about two hops, for time: the object engine's loop
    takes E + O, some 1,460 draws, a wave, and mnist's own 35 ms interval is compared in the step's record rather than
    here."""
    from walnutbutter.problems import dataset_stream
    try:
        patterns, labels = dataset_stream("mnist", seed)
    except FileNotFoundError:
        pytest.skip("the MNIST files are not fetched")
    rs = _rust_sweep()
    g, cli = rs.grid_of("mnist", {"hidden_neurons": 0.0, "seed": seed, "interval": 5.0}, "hazard", True, -4.0)
    Neuron.tau = tau
    g.set_delta(0.0)  # grid_of applies the command line's width; the synapse takes every width to 0 (§6.13)
    rest = {"h0": options["h0"]} if "h0" in options else {}  # the default otherwise
    g.set_exploration("synapse", family=family, scaling=scaling, trace=trace, **rest)
    g.drive = drive
    g.drive_steps = options.get("steps", g.drive_steps)
    g.use_input_stream(patterns[:100], labels[:100])
    assert len(g.all_neurons()) == 455 and len(set(g.output_row())) == 60
    if learning:
        teacher = Teacher(g, seed=seed + 100, rule="reinforce", eligibility="hazard", target="label", critic="evidence",
                          lr=cli.lr, homeostasis=options.get("homeostasis", 0.0), unstick=options.get("unstick", 0.0))
        parted = fast.compare(g, epochs=20, teacher=teacher)
    else:
        parted = fast.compare(g, epochs=20, rng=random.Random(seed + 200))
    assert parted == [], parted


@pytest.mark.parametrize("drive,trace,family,homeostasis,unstick", [
    ("rate", "all", "loglinear", 0.0, 0.0), ("charged", "ventured", "linear", 0.0, 0.0),
    # thresholds moved between epochs, which the charge and u read as they stand (5.4b, §7.5)
    ("rate", "ventured", "linear", 1e-4, 1e-3), ("charged", "all", "loglinear", 1e-4, 1e-3)])
def test_fast_train_is_the_teacher_s_run(drive, trace, family, homeostasis, unstick):
    """fast.train carries the mode and the charged drive: the same patterns and the same seed give the weights, the read,
    the rate memories, the thresholds homeostasis and un-sticking move and the stream a Teacher on the object engine
    reaches, to the bit. Its eligibility, named as None, is hazard here (§8.3), and the engine keeps every trace through
    the centre call train makes."""
    patterns = input_stream(15, goo().raw_bit_count(), 3)
    mesh = goo(weight=0.02, drive=drive, trace=trace, family=family)
    mesh.use_input_stream(patterns, None)
    start = [n.threshold for n in mesh.all_neurons()]
    moves = dict(homeostasis=homeostasis, target_rate=TARGET_RATE, unstick=unstick, unstick_target=UNSTICK_TARGET)
    teacher = Teacher(mesh, seed=7, rule="reinforce", eligibility="hazard", target="copy", lr=0.5, **moves)
    rewards = [teacher.epoch(verbose=False) for _ in range(15)]
    twin = goo(weight=0.02, drive=drive, trace=trace, family=family)
    mean, trace_, engine, report = fast.train(twin, 15, lr=0.5, target="copy", patterns=patterns, seed=7, trace_every=1,
                                              **moves)
    assert trace_ == rewards and engine.traced() and engine.exploration() == "synapse"
    assert list(engine.weights()) == [c.weight for n in mesh.all_neurons() for c in n.outgoing]
    assert list(engine.gains()) == [n.gain for n in mesh.all_neurons()]
    assert list(engine.read_counts()) == [n.read_count for n in mesh.all_neurons()]
    assert report["rates"] == [n.rate for n in mesh.all_neurons()]
    assert report["thresholds"] == [n.threshold for n in mesh.all_neurons()] == list(engine.thresholds())
    assert (report["thresholds"] != start) == bool(homeostasis or unstick)
    assert list(engine.explore_state()) == list(teacher.rng.getstate()[1])
    assert [c.weight for c in twin.connections.values()] != [c.weight for c in mesh.connections.values()]  # the mesh
    # is not written back by train; the engine holds the run


@pytest.mark.parametrize("drive,tau", [("rate", math.inf), ("charged", math.inf), ("charged", 2.0)])
def test_a_network_that_ran_on_the_objects_continues_on_rust(drive, tau):
    """§12.9, §12.11 through fast.build: a network that has run on the objects -- its gains and read counts (§8.16,
    §7.9), its traces, stamps and clocks under way, signals in flight with their ventured marks -- is built into the
    engine, which holds what the network holds straight after the build (the read counts among it, which the next reset
    zeroes before a run could show them) and then runs on beside the objects to the bit, paid against the baseline the
    Teacher had reached (§9.3)."""
    Neuron.tau = tau
    g = goo(seed=2, weight=0.02, drive=drive, trace="ventured")
    teacher = Teacher(g, seed=5, rule="reinforce", eligibility="hazard", target="copy", lr=0.5, homeostasis=1e-4,
                      unstick=1e-3)
    neurons, index = fast.flatten(g)[:2]
    edges = [c for n in neurons for c in n.outgoing]
    edge_index = {id(c): k for k, c in enumerate(edges)}
    for _ in range(40):  # on until the objects hold everything the build hands over
        teacher.epoch(verbose=False)
        queue = fast._queue(g, index, edge_index)
        if (any(e[3] for e in queue) and not all(e[3] for e in queue) and any(n.read_count for n in neurons)
                and (tau != math.inf or any(n.gain for n in neurons))):
            break
    assert any(e[3] for e in queue) and not all(e[3] for e in queue) and any(n.read_count for n in neurons)
    assert tau != math.inf or any(n.gain for n in neurons)
    never = float("-inf")
    engine, _, _ = fast.build(g, explore_rng=random.Random(0), pending_events=queue)
    assert list(engine.gains()) == [n.gain for n in neurons]
    assert list(engine.read_counts()) == [n.read_count for n in neurons]
    assert engine.pending_events(True) == queue
    assert list(engine.potentials()) == [n.potential for n in neurons]
    assert list(engine.thresholds()) == [n.threshold for n in neurons]
    assert list(engine.exposed_since()) == [n.exposed_since for n in neurons]
    assert list(engine.fired_times()) == [never if n.fired_at is None else n.fired_at for n in neurons]
    assert list(engine.traces()) == [c.trace for c in edges] and list(engine.trace_ats()) == [c.trace_at for c in edges]
    assert list(engine.last_signals()) == [never if c.last_signal is None else c.last_signal for c in edges]
    assert fast.compare(g, epochs=10, teacher=teacher) == []  # built with the objects' queue, marks and all


@pytest.mark.parametrize("drive", ("rate", "charged"))
@pytest.mark.parametrize("scaling", ("count", "fan-out"))
def test_an_output_zone_out_of_index_order_with_a_neuron_at_two_places(scaling, drive):
    """§3.8, §7.9: after E, one draw per output neuron for its read synapse, in output order -- one per neuron, whatever
    places it holds. Goo lays its zone out in index order with no neuron twice, so here the zone is reversed and a neuron
    put at a second place, on the instance and before set_exploration, so kappa_i's F_i counts that read synapse once
    (§7.7); the engine lays the read slots out as the file says and the objects do, and the two agree to the bit."""
    g = Goo(count=60, across=8, seed=2, weight=0.02)
    g.rule, g.drive, g.read = "reinforce", "rate", "count"
    zone = type(g).output_row(g)[::-1]
    zone.insert(3, zone[5])  # the sixth place's neuron at the fourth as well
    g.output_row = lambda: list(zone)
    g.set_exploration("synapse", scaling=scaling)
    g.drive = drive
    assert fast.compare(g, epochs=10, rng=random.Random(1)) == []
    engine, neurons, index = fast.build(g, explore_rng=random.Random(1))
    slots, first = engine.exploration_settings()[5], list(dict.fromkeys(zone))  # each neuron at its first place
    assert [slots[index[n]] for n in first] == [len(g.connections) + k for k in range(len(first))]
    assert sum(1 for s in slots if s >= 0) == len(first) == len(zone) - 1


# --- the stream and the queue ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("drive", ("rate", "charged"))
def test_the_stream_stands_where_the_object_engine_leaves_it(drive):
    """§3.8, §7.3, §12.6: an epoch takes E + O uniforms a wave, all of them whatever fires, in both engines, and Rust hands
    the stream back where the objects leave theirs: a fresh stream stepped (E + O) x waves draws on."""
    g, twin = goo(drive=drive), goo(drive=drive)
    rng = random.Random(9)
    engine, neurons, index = fast.build(twin, explore_rng=random.Random(9))
    run_epoch(g, verbose=False, rng=rng)
    engine.reset(False)
    row = [index[n] for n in twin.input_row()]
    times = [when for _, when in g.input_events]
    (engine.charge_many if drive == "charged" else engine.stimulate_many)([row[p] for p, _ in g.input_events], times)
    waves = engine.run(g.horizon)
    assert len(waves) == len(g.waves) and [len(w[1]) for w in waves] == [len(w.fired) for w in g.waves]
    draws = len(g.connections) + len(set(g.output_row()))
    stepped = random.Random(9)
    for _ in range(draws * len(g.waves)):
        stepped.random()
    assert list(engine.explore_state()) == list(rng.getstate()[1]) == list(stepped.getstate()[1])


def _carry(engine, other) -> None:
    """Everything reset(False) keeps across an epoch boundary that the run moved, from one engine to another built from
    the same network (the thresholds, which nothing here moves, it has already)."""
    other.set_weights(engine.weights())
    other.set_potentials(engine.potentials())
    other.set_last_updates(engine.last_updates())
    other.set_traces(engine.traces())
    other.set_trace_ats(engine.trace_ats())
    other.set_notes(engine.notes())
    other.set_fired_at(engine.fired_times())
    other.set_previous_fired_at(engine.previous_fired_times())
    other.set_exposed_since(engine.exposed_since())
    other.set_last_signals(engine.last_signals())
    other.set_spike_counts(engine.spike_counts())
    other.set_gains(engine.gains())
    other.set_read_counts(engine.read_counts())
    other.set_explore_state(engine.explore_state())


def test_ventured_signals_in_flight_round_trip_through_the_queue():
    """§7.9, §12.9 at the engine: `pending_events(marks=True)` gives every event in flight with its mark, in delivery
    order; `push_events` takes them back into a fresh engine, which then runs on to the bit as the first does. Without
    marks the queue is refused while a ventured signal is in it, rather than handed over as relayed; fast.build takes
    quadruples, and a triple loads relayed."""
    g = goo(weight=0.02, drive="charged", trace="ventured")
    engine, neurons, index = fast.build(g, explore_rng=random.Random(4))
    at = [index[n] for n in g.input_row()]
    events = []
    for _ in range(6):
        engine.reset(False)
        g.new_random_input()
        g.time = g.input_time
        g.epoch += 1
        schedule = g.input_schedule()
        engine.charge_many([at[p] for p, _ in schedule], [when for _, when in schedule])
        engine.run(g.time + g.interval)
        events = engine.pending_events(True)
        if any(e[3] for e in events):
            break
    assert any(e[3] for e in events) and not all(e[3] for e in events)  # ventured and relayed signals both in flight
    with pytest.raises(ValueError, match="§12.9"):
        engine.pending_events()
    fresh = goo(weight=0.02, drive="charged", trace="ventured")
    other, _, _ = fast.build(fresh, explore_rng=random.Random(4))
    _carry(engine, other)
    other.push_events([e[0] for e in events], [e[1] for e in events], [e[2] for e in events], [e[3] for e in events])
    assert other.pending_events(True) == events
    # both run on: the same drive, and they land on the same bits
    g.new_random_input()
    g.time = g.input_time
    schedule = g.input_schedule()
    for e in (engine, other):
        e.reset(False)
        e.charge_many([at[p] for p, _ in schedule], [when for _, when in schedule])
        e.run(g.time + g.interval)
    for getter in ("spike_counts", "potentials", "gains", "scores", "notes", "traces", "trace_ats", "exposed_since",
                   "read_counts", "explore_state"):
        assert getattr(engine, getter)() == getattr(other, getter)(), getter
    assert engine.pending_events(True) == other.pending_events(True)
    assert engine.exploration_settings() == other.exploration_settings()  # set by the build from the same network
    # through fast.build: quadruples keep their marks, triples load relayed
    built, _, _ = fast.build(goo(weight=0.02, drive="charged", trace="ventured"), explore_rng=random.Random(4),
                             pending_events=events)
    assert built.pending_events(True) == events
    relayed = [e for e in events if not e[3]]
    built, _, _ = fast.build(goo(weight=0.02, drive="charged", trace="ventured"), explore_rng=random.Random(4),
                             pending_events=[e[:3] for e in relayed])
    assert built.pending_events(True) == relayed and built.pending_events() == [e[:3] for e in relayed]


# --- waves built by hand ---------------------------------------------------------------------------------------------
# What no drawn run is sure to hold: the drive's times are continuous, so a charge and a signal meet in one wave only
# inside the slack, and a charge never outlives its epoch while the thresholds move only between them. Each is built
# the same way in both engines -- the same preset potentials, the same events queued, the same stream -- and run.


def _queued(potentials, events, seed=2, weight=None, stream=5):
    """goo 60 under the charged drive twice, the objects' and a twin for the engine, each neuron at `potentials` (by
    engine index), and `events` -- (time, kind, payload), a signal's payload its edge and a charge's its neuron -- in
    the objects' schedule and the engine's queue, both given the same stream. Returns (g, engine, neurons, edges)."""
    g, twin = goo(seed=seed, weight=weight, drive="charged"), goo(seed=seed, weight=weight, drive="charged")
    neurons, index = fast.flatten(g)[:2]
    edges = [c for n in neurons for c in n.outgoing]
    for mesh in (g, twin):
        for i, v in potentials.items():
            list(mesh.all_neurons())[i].potential = v
    engine, _, _ = fast.build(twin, explore_rng=random.Random(stream),
                              pending_events=[(t, k, p, False) for t, k, p in events])
    for t, k, p in events:
        if k == EXTERNAL:
            g.schedule.charge(neurons[p], g._drive_steps(), t)
        else:
            g.schedule.signal(edges[p], t)
    g.explore_rng = random.Random(stream)
    return g, engine, neurons, edges


def _run(g, engine, until, neurons):
    """Both run to `until`; the objects' and the engine's waves as (time, fired indices), each side's potentials."""
    index = {n: i for i, n in enumerate(neurons)}
    g.schedule.run(until, g.waves, g._on_wave, g._everyone(), explore=g.explorer(), synapses=g.decider())
    theirs = [(t, list(f)) for t, f in engine.run(until)]
    mine = [(w.time, [index[n] for n in w.fired]) for w in g.waves]
    assert list(engine.explore_state()) == list(g.explore_rng.getstate()[1])
    return mine, theirs, [n.potential for n in neurons], list(engine.potentials())


def test_a_charge_is_integrated_before_a_signal_of_its_wave():
    """5.4b, §3.5: a wave takes its external deliveries first, then its signals, so a charge due at the moment of a signal
    is integrated before it. Here the signal into an input is popped first, 1e-12 earlier and inside the wave's slack,
    the charge anchoring the wave; at the potential picked the two orders of the sum differ in the last bit, and both
    engines land on the charge first's."""
    Neuron.tau = math.inf  # no decay, so the sum is the whole of the arithmetic
    g = goo(seed=1, drive="charged")
    neurons, index = fast.flatten(g)[:2]
    edges = [c for n in neurons for c in n.outgoing]
    inputs, steps, found = {index[n] for n in g.input_row()}, g._drive_steps(), 0
    for e, c in enumerate(edges):
        a, theta, w = index[c.target], c.target.threshold, c.weight
        if a not in inputs:
            continue
        for k in range(1, 400):
            p = theta * k / 400
            charged_first = (p + theta / steps) + w
            if charged_first != (p + w) + theta / steps and c.target.minimum_potential < charged_first < theta:
                break
        else:
            continue
        g2, engine, ns, _ = _queued({a: p}, [(3.0 - 1e-12, SIGNAL, e), (3.0, EXTERNAL, a)], seed=1)
        mine, theirs, objects, rust = _run(g2, engine, 3.5, ns)
        assert mine == theirs and objects == rust and rust[a] == charged_first, (a, e, p)
        found += 1
        if found == 3:
            break
    assert found


def test_a_charged_input_decides_among_the_touched():
    """5.4b, §3.5: a delivery taken touches its input as a signal touches its target, and the input decides among the
    wave's touched neurons in the order they were touched -- before a neuron a signal touched in the same wave -- not
    among everyone else after them. The order they fire in is the order their signals sum in downstream (§3.7), so both
    project onto one neuron holding potential, and the engines agree on the order and the sum."""
    Neuron.tau = math.inf
    g = goo(seed=1, drive="charged")
    neurons, index = fast.flatten(g)[:2]
    edges = [c for n in neurons for c in n.outgoing]
    inputs, found = {index[n] for n in g.input_row()}, 0
    for c1 in edges:
        a, t = index[c1.source], index[c1.target]
        if a not in inputs:
            continue
        bs = [index[c2.source] for c2 in neurons[t].incoming
              if index[c2.source] not in inputs and index[c2.source] != t and neurons[index[c2.source]].incoming]
        if not bs:
            continue
        b = bs[0]
        e_in = edges.index(neurons[b].incoming[0])
        presets = {a: neurons[a].threshold, b: neurons[b].threshold * 2 + 1.0, t: 0.1 * neurons[t].threshold}
        g2, engine, ns, _ = _queued(presets, [(3.0, SIGNAL, e_in), (3.0, EXTERNAL, a)], seed=1)
        mine, theirs, objects, rust = _run(g2, engine, 3.0 + Neuron.hop + 0.1, ns)
        assert mine == theirs and objects == rust, (a, b, t)
        assert mine[0][1][:2] == [a, b]  # the charged input first: charges are taken before the wave's signals
        found += 1
        if found == 3:
            break
    assert found


def test_a_charge_takes_the_threshold_its_input_holds_when_it_lands():
    """5.4b: theta / DRIVE_STEPS on the threshold the input holds when the delivery lands, not the one it held when it was
    queued: the charge is queued, the input's threshold then moved (as homeostasis moves it between epochs, §9.9), and
    the charge lands on the moved one in both engines -- queued in Rust as a checkpoint's queue is, and as the drive's own
    charge_many queues it."""
    Neuron.tau = math.inf
    g = goo(seed=2, weight=0.02, drive="charged")
    a = fast.flatten(g)[1][g.input_row()[0]]
    g, engine, neurons, _ = _queued({}, [(3.0, EXTERNAL, a)], weight=0.02)
    other, _, _ = fast.build(goo(seed=2, weight=0.02, drive="charged"), explore_rng=random.Random(5))
    other.charge_many([a], [3.0])
    moved = [n.threshold for n in neurons]
    moved[a] = moved[a] * 1.1
    for e in (engine, other):
        e.set_thresholds(moved)
    neurons[a].threshold = moved[a]
    mine, theirs, objects, rust = _run(g, engine, 3.5, neurons)
    other.run(3.5)
    assert mine == theirs and objects == rust == list(other.potentials()) and rust[a] == moved[a] / g._drive_steps()


# --- what the engine refuses ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("homeostasis,unstick,threshold", [
    (0.1, 0.01, 0.0499),  # homeostasis takes it to -1e-6, and un-sticking would lift it back above zero
    (0.0, 0.1, 0.0005),  # un-sticking takes it there itself
])
def test_a_threshold_taken_to_zero_is_refused_at_the_move_that_takes_it(homeostasis, unstick, threshold):
    """§7.5: under exploration at the synapse a threshold at or below zero is refused at the moment homeostasis or
    un-sticking (§9.9, §9.10) takes it there -- homeostasis's moment included, where the un-sticking of the same step
    would lift it back. A neuron that hears nothing, and so never fires, at a rate of 0.001; homeostasis toward 0.5,
    un-sticking toward 1e-4 or 0.5. The Teacher on the objects and fast.train both refuse in the first epoch."""
    patterns = input_stream(3, goo().raw_bit_count(), 3)
    kw = dict(homeostasis=homeostasis, target_rate=0.5, unstick=unstick, unstick_target=1e-4 if homeostasis else 0.5)

    def mesh():
        g = goo(seed=2)
        deaf = next(n for n in g.all_neurons() if not n.incoming and n not in g.input_row())
        deaf.rate, deaf.threshold = 0.001, threshold
        g.use_input_stream(patterns, None)
        return g, deaf

    g, deaf = mesh()
    teacher = Teacher(g, seed=5, rule="reinforce", eligibility="hazard", target="copy", lr=0.5, **kw)
    with pytest.raises(ValueError, match="§7.5"):
        teacher.epoch(verbose=False)
    assert deaf.threshold <= 0.0 and g.epoch == 1
    if homeostasis:
        assert deaf.threshold + unstick * (deaf.rate - kw["unstick_target"]) > 0.0  # what un-sticking would have made it
    twin, _ = mesh()
    with pytest.raises(ValueError, match="§7.5"):
        fast.train(twin, 3, lr=0.5, target="copy", patterns=patterns, seed=5, **kw)
    assert twin.epoch == 1




def test_the_engine_refuses_what_the_file_refuses():
    """§12.2 at the engine's own setters, each citing its clause, the engine left as it was; and it keeps every trace under
    the mode through the setters that recompute it."""
    import walnutbutter_schedule as rust
    g = goo()
    engine, neurons, index = fast.build(g, explore_rng=random.Random(1))
    n = len(neurons)
    assert engine.traced() and engine.exploration() == "synapse"
    engine.set_deltas([0.0] * n)
    engine.set_centred(False, 1e-4)
    assert engine.traced()
    with pytest.raises(ValueError, match="§6.13"):
        engine.set_deltas([0.1] * n)
    with pytest.raises(ValueError, match="§8.3"):
        engine.set_centred(True, 1e-4)
    with pytest.raises(ValueError, match="§7.5"):
        engine.set_thresholds([0.0] * n)
    with pytest.raises(ValueError, match="5.4b"):
        engine.charge_many([0], [1.0])  # no charged drive set
    for steps in (0, -1, True, False, 3.0, "3", None, -2**70, 2**64):  # a bool is an int to Python, and no count
        with pytest.raises(ValueError, match="5.4b"):
            engine.set_charged_drive(steps)
    assert engine.exploration_settings()[6] == 0  # none set: the engine left as it was
    with pytest.raises(ValueError, match="5.4b"):
        engine.push_events([1.0], [0], [0])  # an EXTERNAL event is a charge
    with pytest.raises(ValueError, match="unknown event kind"):
        engine.push_events([1.0], [3], [0])
    with pytest.raises(ValueError, match="§7.9"):
        engine.push_events([1.0], [1], [0], [True])  # a ventured stimulus
    with pytest.raises(ValueError, match="§7.9"):
        engine.set_read_counts([1] * n)  # only an output has a read synapse
    assert engine.reinforce_hazard(0.0, 0.1) == 0  # re-keyed: the synapses' decisions are draws (§8.3)
    scales, outputs = [1.0] * n, [index[x] for x in g.output_row()]
    held = engine.exploration_settings()
    for mode, rest, family, trace, clause in (("both", 0.01, "loglinear", "all", "§7.1"), ("synapse", 1.0, "loglinear", "all", "§7.6"),
                                              ("synapse", -0.1, "loglinear", "all", "§7.6"), ("synapse", 0.01, "cubic", "all", "§7.6"),
                                              ("synapse", 0.01, "loglinear", "some", "§8.17"),
                                              # h0 as the objects take it: an int or a float, never a bool (§7.6)
                                              ("synapse", False, "loglinear", "all", "§7.6"),
                                              ("synapse", True, "loglinear", "all", "§7.6"),
                                              ("synapse", "0.01", "loglinear", "all", "§7.6"),
                                              ("synapse", 10**400, "loglinear", "all", "§7.6")):
        with pytest.raises(ValueError, match=clause):
            engine.set_exploration(mode, rest, family, trace, scales, outputs)
    assert engine.exploration_settings() == held  # refused, and left as it was

    def fresh(sources=(0, 1), targets=(1, 0), active=(True, True), thresholds=(1.0, 1.0), bored=0.0):
        return rust.Engine(2, list(sources), list(targets), [0.5] * len(sources), list(active), list(thresholds), [-4.0, -4.0],
                           math.inf, Neuron.refractory, Neuron.hop, bored, Neuron.rate_tau)

    def explore(e):
        e.set_explore_state(list(random.Random(1).getstate()[1]))
        e.set_exploration("synapse", 0.01, "loglinear", "all", [1.0, 1.0], [1])

    with pytest.raises(ValueError, match="§7.3"):  # no stream
        fresh().set_exploration("synapse", 0.01, "loglinear", "all", [1.0, 1.0], [1])
    for e, clause in ((fresh(sources=(1, 0), targets=(0, 1)), "§3.8"),  # edges not numbered source-major
                      (fresh(active=(True, False)), "§3.8"), (fresh(thresholds=(1.0, 0.0)), "§7.5"), (fresh(bored=5.0), "§7.5")):
        with pytest.raises(ValueError, match=clause):
            explore(e)
        assert e.exploration() == "neuron" and not e.traced()  # refused, and left as it was
    e = fresh()
    e.set_explore_state(list(random.Random(1).getstate()[1]))
    e.set_deltas([0.1, 0.1])
    with pytest.raises(ValueError, match="§6.13"):
        e.set_exploration("synapse", 0.01, "loglinear", "all", [1.0, 1.0], [1])
    e = fresh()
    with pytest.raises(ValueError, match="5.4b"):
        e.set_charged_drive(3)  # the charged drive under the neuron rule
    with pytest.raises(ValueError, match="§12.9"):
        e.set_gains([0.5, 0.0])
    with pytest.raises(ValueError, match="§12.9"):
        e.push_events([1.0], [2], [0], [True])  # a ventured signal on an engine under the neuron rule
    with pytest.raises(ValueError, match="§5.2|§8.3"):
        e.reinforce_hazard(0.1, 0.1)  # nothing draws
