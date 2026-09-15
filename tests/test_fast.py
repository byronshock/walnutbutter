"""The Rust wave loop, when it is built. Every test here skips cleanly when it is not.

The contract is the project's existing one: a third engine must land on the same bits as
the first two, or it is not a second opinion about the specification, only a faster guess.
"""

import random

import pytest

from walnutbutter import fast
from walnutbutter.grid import GridOfNeurons
from walnutbutter.neuron import Neuron


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def mesh(rows=5, seed=1, quash=0.0, hebb=0.0, rule="teacher"):
    grid = GridOfNeurons(across=12, rows=rows, weight=None, seed=seed, permute=False, omega=0)
    grid.coding, grid.readout, grid.read = "population", "top", "fired"
    grid.drive, grid.quash_rate, grid.hebb_rate, grid.rule = "rate", quash, hebb, rule
    return grid


def test_it_says_plainly_when_it_is_not_built():
    """The project runs without the extension; asking for it without it gives a real message."""
    if fast.available():
        pytest.skip("the extension is built, so there is nothing to refuse")
    with pytest.raises(ImportError, match="rust/README"):
        fast.build(mesh())


def test_the_edge_order_is_the_object_engine_s_push_order():
    """Not cosmetic: signals due at one moment are summed in push order, so this sets the bits."""
    grid = mesh()
    neurons, index, source, target, weight, active = fast.flatten(grid)
    assert len(weight) == len(grid.connections)
    at = 0
    for neuron in neurons:
        for connection in neuron.outgoing:
            assert source[at] == index[connection.source]
            assert target[at] == index[connection.target]
            assert weight[at] == connection.weight
            at += 1
    assert at == len(weight)


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_it_agrees_with_the_object_engine_bit_for_bit():
    for quash, hebb in ((0.0, 0.0), (0.02, 0.0), (0.0, 0.01), (0.02, 0.01)):
        grid = mesh(quash=quash, hebb=hebb)
        parted = fast.compare(grid, epochs=40)
        assert parted == [], f"quash {quash}, hebb {hebb}: {parted}"


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_it_agrees_on_the_shallow_grid_too():
    grid = mesh(rows=2, seed=5, quash=0.02)
    assert fast.compare(grid, epochs=40) == []


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_it_draws_pythons_own_stream_bit_for_bit_and_hands_it_back():
    """§6.1: every engine draws from the same Box-Muller stream. Rust runs Python's MT19937, from Python's state."""
    import random
    from walnutbutter.exploration import gaussians
    grid = mesh(rows=2, seed=3)
    rng, mirror = random.Random(99), random.Random(99)
    engine, neurons, index = fast.build(grid, sigma=0.1, explore_rng=rng)
    engine.reset(False, False)
    engine.stimulus(0, 0.0)
    engine.run(1.0)  # one wave, one draw per neuron
    assert list(engine.noises()) == gaussians(mirror, len(neurons), 0.1)  # equal, not approximately
    fast.sync_explore(engine, rng)
    assert rng.getstate() == mirror.getstate()  # Python's stream carries on from where Rust left it
    with pytest.raises(ValueError, match="needs a stream"):
        fast.build(mesh(rows=2), sigma=0.1)


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
@pytest.mark.parametrize("eligibility", ["perturb", "hebb"])
def test_it_agrees_with_the_object_engine_under_the_reinforce_rule(eligibility):
    """§7: a rule is implemented in every engine and passes the same tests. The perturb rule, per-wave draws and all."""
    from walnutbutter.learning import Teacher
    grid = mesh(rows=3, seed=11, rule="reinforce")
    # homeostasis and un-sticking on, fast enough to act inside the test (as tests/test_arrays.py runs them)
    teacher = Teacher(grid, seed=7, rule="reinforce", eligibility=eligibility, sigma=0.1, homeostasis=0.01, unstick=0.1)
    assert (teacher.sigma > 0) == (eligibility == "perturb")  # the Teacher zeroes sigma for hebb
    before = [n.threshold for n in grid.all_neurons()]
    parted = fast.compare(grid, epochs=60, teacher=teacher)
    assert parted == [], parted
    assert [n.threshold for n in grid.all_neurons()] != before  # the thresholds really moved, and Rust followed
    if eligibility == "perturb":
        assert any(n.noise != 0.0 for n in grid.all_neurons())  # the draws really were taken
        assert teacher.rng.getstate() != random.Random(7).getstate()  # and the stream really moved


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_goo_runs_on_the_rust_engine_and_agrees_with_the_object_engine():
    """§3.4 on §6.15: the container the sweep is about, on the engine the sweep runs on, bit for bit."""
    from walnutbutter.goo import Goo
    from walnutbutter.learning import Teacher
    goo = Goo(count=40, across=8, seed=3, weight=None)
    goo.rule, goo.drive = "reinforce", "rate"
    teacher = Teacher(goo, seed=7, rule="reinforce", eligibility="hebb", homeostasis=0.01, unstick=0.1)
    parted = fast.compare(goo, epochs=60, teacher=teacher)
    assert parted == [], parted
    from walnutbutter.constants import GOO_THRESHOLD
    assert goo.all_neurons()[0].threshold != goo.fan_in_scale() * GOO_THRESHOLD  # moved by homeostasis, on both engines alike


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_compare_refuses_what_it_cannot_mirror():
    from walnutbutter.learning import Teacher
    grid = mesh(rows=2, seed=1)
    with pytest.raises(ValueError, match="row critic"):
        fast.compare(grid, epochs=1, teacher=Teacher(grid, seed=1, rule="reinforce", late="ignore", homeostasis=0.0, unstick=0.0))


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_train_runs_the_perturb_rule_from_a_seeded_stream():
    a = fast.train(mesh(rows=2, seed=4), 30, eligibility="perturb", sigma=0.1, seed=5)
    b = fast.train(mesh(rows=2, seed=4), 30, eligibility="perturb", sigma=0.1, seed=5)
    c = fast.train(mesh(rows=2, seed=4), 30, eligibility="perturb", sigma=0.1, seed=6)
    assert list(a[2].weights()) == list(b[2].weights())  # the same seed, the same run
    assert list(a[2].weights()) != list(c[2].weights())  # a different stream, a different one
    assert set(a[3]) >= {"last_tenth", "rates", "thresholds", "stuck_on", "stuck_off"} and 0 <= a[3]["last_tenth"] <= 1
    flat = fast.train(mesh(rows=2, seed=4), 30, homeostasis=0.0, unstick=0.0)
    assert flat[3]["thresholds"] == [0.25] * len(flat[3]["thresholds"])  # nothing moved them
    moved = fast.train(mesh(rows=2, seed=4), 30, homeostasis=0.01, unstick=0.1)
    assert moved[3]["thresholds"] != flat[3]["thresholds"]
    with pytest.raises(ValueError, match="positive sigma"):
        fast.train(mesh(rows=2), 1, eligibility="perturb", sigma=0.0)


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_signals_in_flight_outlive_the_epoch():
    """AUTHORITY.md §4.2: the schedule is not drained at the boundary."""
    grid = mesh()
    engine, neurons, index = fast.build(grid)
    row = grid.input_row()
    grid.new_random_input()
    for place, when in grid.input_schedule():
        engine.stimulus(index[row[place]], when)
    engine.run(grid.time + grid.interval)
    assert engine.pending() > 0
    engine.reset(False, True)
    assert engine.pending() > 0  # a reset clears the epoch's state, not the schedule


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_it_is_faster_than_the_engine_it_replaces():
    grid = mesh()
    timings = fast.benchmark(grid, epochs=200)
    assert timings["rust"] < timings["objects"] / 5  # the profile says 30-100x; hold it to 5
