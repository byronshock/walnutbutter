import pytest

from walnutbutter.goo import Goo
from walnutbutter.neuron import Neuron
from walnutbutter.propagation import Signal, Wave, propagate


@pytest.fixture(autouse=True)
def _deterministic_stimulus(forced_input):
    """This file is about the schedule and the plumbing, not the input process (see conftest.forced_input)."""


def chain(n, weight=1.0, threshold=1.0):
    """a0 -> a1 -> ... -> a(n-1), one-way."""
    neurons = [Neuron(f"n{i}", threshold=threshold) for i in range(n)]
    for i in range(n - 1):
        neurons[i].connect(neurons[i + 1], connection_id=i + 1, weight=weight)
    return neurons


def test_forced_fire_travels_down_a_chain_one_wave_per_hop(capsys):
    neurons = chain(4)
    waves = propagate(fire=[neurons[0]])
    assert [w.number for w in waves] == [0, 1, 2, 3]  # the last neuron has nothing to send
    assert [w.fired for w in waves] == [[n] for n in neurons]
    assert [n.fired_in_wave for n in neurons] == [0, 1, 2, 3]


def test_waves_record_the_signals_delivered(capsys):
    a, b = chain(2)
    waves = propagate(fire=[a])
    assert waves[1].delivered == [a.connection_to(b)]  # the queue is the connections themselves
    (signal,) = waves[1].signals()
    assert isinstance(signal, Signal)
    assert signal.connection is a.connection_to(b)
    assert signal.target is b and signal.amount == 1.0 and signal.wave == 1


def test_two_way_pair_fires_once_each(capsys, monkeypatch):
    monkeypatch.setattr(Neuron, "hop", 5.0 / 3.0)  # a third of the period: a two-hop loop cannot refire, where HOP 2.55 lets it
    monkeypatch.setattr(Neuron, "verbose", True)
    a, b = Neuron("a"), Neuron("b")
    a.connect(b)
    b.connect(a)
    waves = propagate(fire=[a])
    assert (a.fired_in_wave, b.fired_in_wave) == (0, 1)
    assert capsys.readouterr().out.count("fired") == 2
    assert len(waves) == 3  # wave 2 delivers b's signal back to a, which ignores it


def test_at_two_hops_a_two_way_pair_reverberates(capsys):
    """HOP = 2.55 ms (§3.2, September 19, 2026): a spike sent around a two-way pair comes back after two hops, LAG past
    the moment the refractory period ends, so the pair refires every hop in turn until the clock stops it -- where at a
    third of the period each fired once. Every reciprocal pair strong enough to fire its partner is a two-hop
    oscillator, and the LAG is what keeps the return off the wall rather than on it."""
    assert Neuron.hop == 2.55 and 2 * Neuron.hop == pytest.approx(Neuron.refractory + 0.1)
    a, b = Neuron("a"), Neuron("b")
    a.connect(b)
    b.connect(a)
    waves = propagate(fire=[a], until=20.0)
    times = [w.time for w in waves if w.fired]
    assert times == pytest.approx([k * 2.55 for k in range(8)])  # 2.55 is not representable: the clock's slack is §3.4's
    assert times[2] == pytest.approx(5.1) and times[2] > Neuron.refractory  # two hops land past the wall, not on it
    assert [w.fired for w in waves if w.fired] == [[a], [b]] * 4


def test_all_signals_in_a_wave_are_delivered_before_anyone_fires(capsys):
    # Two half-strength inputs arriving in the same wave add up and fire the target in that wave.
    a, b, c = Neuron("a"), Neuron("b"), Neuron("c", threshold=1.0)
    a.connect(c, weight=0.5)
    b.connect(c, weight=0.5)
    propagate(fire=[a, b])
    assert c.fired_in_wave == 1


def test_inputs_below_threshold_do_not_fire_and_leave_potential(capsys):
    a, b = chain(2, weight=0.75)
    propagate(fire=[a])
    assert not b.has_fired and b.potential == 0.75


@pytest.mark.parametrize("order", ["excite first", "inhibit first"])
def test_inhibition_result_does_not_depend_on_order(order, capsys):
    # With recursion, c could fire on the excitatory input before the inhibitory
    # one arrived. With wave delivery the net input is 0 either way.
    excite, inhibit, c = Neuron("e"), Neuron("i"), Neuron("c", threshold=1.0)
    excite.connect(c, weight=1.0)
    inhibit.connect(c, weight=-1.0)
    stimulus = [excite, inhibit] if order == "excite first" else [inhibit, excite]
    propagate(fire=stimulus)
    assert not c.has_fired and c.potential == 0.0


def test_external_inputs_fire_only_when_they_reach_threshold(capsys):
    a, b = Neuron("a", threshold=1.0), Neuron("b", threshold=1.0)
    waves = propagate(inputs={a: 1.0, b: 0.5})
    assert a.fired_in_wave == 0 and not b.has_fired and b.potential == 0.5
    assert waves[0].fired == [a]


def test_forced_neuron_ignores_threshold_and_is_not_fired_twice(capsys, monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", True)
    a = Neuron("a", threshold=100.0)
    propagate(fire=[a, a])
    assert a.fired_in_wave == 0
    assert capsys.readouterr().out.count("fired") == 1


def test_a_refractory_neuron_is_not_forced(capsys):
    a = Neuron("a")
    a.fire(now=0.0)
    waves = propagate(fire=[a], now=0.0)
    assert waves[0].fired == [] and waves[0].time == 0.0
    assert propagate(fire=[a], now=5.0)[0].fired == [a]


def test_empty_stimulus_gives_no_waves():
    assert propagate() == []




def first_wave(grid, neuron) -> int:
    """The first wave of the epoch a neuron fired in (it may refire later)."""
    return next(w.number for w in grid.waves if neuron in w.fired)




