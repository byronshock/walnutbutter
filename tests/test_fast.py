"""The Rust wave loop, when it is built. Every test here skips cleanly when it is not.

The contract is the project's existing one: a third engine must land on the same bits as
the first two, or it is not a second opinion about the specification, only a faster guess.
"""

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
def test_it_refuses_exploration_noise_rather_than_approximating_it():
    grid = mesh()
    with pytest.raises(ValueError, match="exploration noise"):
        fast.build(grid, sigma=0.1)


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
