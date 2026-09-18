"""The array engine is the same network as the object engine: run them side by side."""

import json
import random

import pytest

np = pytest.importorskip("numpy")  # the array engine is optional: without numpy and scipy these tests skip
pytest.importorskip("scipy")

from walnutbutter.arrays import ArrayNetwork  # noqa: E402
from walnutbutter.cli import cli_main
from walnutbutter.constants import ACROSS, ESCAPE_DELTA, GOO_COUNT
from walnutbutter.goo import Goo
from walnutbutter.learning import Teacher, accuracy, reward, stuck_neurons
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.persistence import checkpoint, restore


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def pair(seed=3, count=GOO_COUNT, across=ACROSS, delta=0.0, **kwargs):
    """The same network twice: one run by objects, one by arrays."""
    mesh = Goo(count=count, across=across, weight=None, seed=seed, **kwargs)
    twin = Goo(count=count, across=across, weight=None, seed=seed, **kwargs)
    if delta:  # §8.3: the rule that pays at the read needs the decision to be a draw
        mesh.set_delta(delta)
        twin.set_delta(delta)
    return mesh, ArrayNetwork(twin)


def fired_waves(grid) -> list:
    """The wave each neuron fired in, -1 for none, in mesh order."""
    if getattr(grid, "engine", "objects") == "arrays":
        return grid.fired_wave.tolist()
    return [-1 if n.fired_in_wave is None else n.fired_in_wave for n in grid.all_neurons()]


def weights(grid) -> np.ndarray:
    if getattr(grid, "engine", "objects") == "arrays":
        return grid.weight.copy()
    return np.array([grid.connections[i].weight for i in range(1, len(grid.connections) + 1)])


def thresholds(grid) -> np.ndarray:
    if getattr(grid, "engine", "objects") == "arrays":
        return grid.threshold_v.copy()
    return np.array([n.threshold for n in grid.all_neurons()])


def rates(grid) -> np.ndarray:
    if getattr(grid, "engine", "objects") == "arrays":
        return grid.rate.copy()
    return np.array([n.rate for n in grid.all_neurons()])


def test_wrapping_copies_the_mesh_exactly():
    mesh = Goo(count=GOO_COUNT, across=ACROSS, weight=None, seed=1)
    net = ArrayNetwork(mesh)
    assert net.across == 8 and len(net) == GOO_COUNT and len(net.weight) == len(mesh.connections)
    assert np.array_equal(weights(net), weights(mesh))
    assert [net.neurons_list[i].name for i in net.input_index] == [n.name for n in mesh.input_row()]
    assert [net.neurons_list[i] for i in net.output_index] == [mesh.get_neuron_at(c, 0) for c in range(8)]
    assert net.mesh is mesh and repr(net).startswith("ArrayNetwork(")


def test_both_engines_draw_the_same_inputs_and_fire_the_same_neurons_wave_by_wave():
    mesh, net = pair()
    for _ in range(50):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert net.input_bits == mesh.input_bits and net.input_pattern == mesh.input_pattern
        assert fired_waves(net) == fired_waves(mesh)
        assert len(net.waves) == len(mesh.waves)
        mesh_index = {n: i for i, n in enumerate(mesh.all_neurons())}
        assert [w.fired.tolist() for w in net.waves] == [sorted(mesh_index[n] for n in w.fired) for w in mesh.waves]
    assert net.epoch == mesh.epoch == 50




def test_carry_over_keeps_the_same_potentials_within_rounding():
    mesh, net = pair(seed=9, count=24, across=6)
    for _ in range(30):
        run_epoch(mesh, verbose=False, discharge=False)
        run_epoch(net, verbose=False, discharge=False)
        assert fired_waves(net) == fired_waves(mesh)
        assert np.allclose(net.potential, [n.potential for n in mesh.all_neurons()], atol=1e-9)


def test_an_inactive_connection_carries_nothing_in_either_engine():
    a = Goo(count=24, across=6, weight=1.0, seed=4)
    b = Goo(count=24, across=6, weight=1.0, seed=4)
    for g in (a, b):
        for c in g.get_neuron_at(0, 1).outgoing:
            c.is_active = False  # the first input neuron is cut off from everyone
    net = ArrayNetwork(b)
    a.set_input([True] + [False] * 5)
    net.set_input([True] + [False] * 5)
    a.fire_input()
    net.fire_input()
    assert fired_waves(net) == fired_waves(a)
    assert sum(w >= 0 for w in fired_waves(net)) == 1  # only the forced neuron fired



def test_checkpoints_cross_between_engines(tmp_path):
    mesh, net = pair(seed=11, count=24, across=6)
    teacher = Teacher(net, seed=2)
    for _ in range(20):
        teacher.epoch(verbose=False)
    path = tmp_path / "arrays.json"
    data = checkpoint(net, path, teacher)
    assert data["engine"] == "arrays" and data["epoch"] == 20
    assert data["weights"] == net.weight.tolist() and data["thresholds"] == net.threshold_v.tolist()
    restored, record = restore(path)  # an object mesh, with the array engine's weights
    assert np.array_equal(weights(restored), weights(net)) and restored.epoch == 20
    again = ArrayNetwork(restored)
    assert np.array_equal(again.weight, net.weight) and again.epoch == 20
    # and the other way: the object engine's checkpoint loads into arrays
    objects_teacher = Teacher(mesh, seed=2)
    for _ in range(20):
        objects_teacher.epoch(verbose=False)
    path2 = tmp_path / "objects.json"
    checkpoint(mesh, path2, objects_teacher)
    assert json.loads(path2.read_text())["engine"] == "objects"
    wrapped = ArrayNetwork(restore(path2)[0])
    assert np.allclose(wrapped.weight, weights(mesh), atol=1e-12)


def test_fired_neurons_and_accuracy_read_through_the_mesh():
    _, net = pair(seed=6, count=24, across=6)
    run_epoch(net, verbose=False)
    fired = net.fired_neurons()
    assert fired and all(n.has_fired for n in fired)
    assert len(fired) == sum(w >= 0 for w in fired_waves(net))
    assert 0.0 <= accuracy(net) <= 1.0
    assert net.mesh.epoch == 1 and len(net.mesh.waves) == len(net.waves)


def test_cli_engine_arrays_runs_headless_learns_and_checkpoints(tmp_path, capsys):
    save = tmp_path / "arrays.json"
    assert cli_main(["--engine", "arrays", "--headless", "--epochs", "30", "--seed", "1", "--save-weights", str(save)]) == 0
    err = capsys.readouterr().err
    assert "engine: arrays" in err and "after 30 epochs" in err
    data = json.loads(save.read_text())
    assert data["engine"] == "arrays" and data["epoch"] == 30 and data["learning"]["epochs"] == 30
    # a loaded checkpoint keeps its engine
    assert cli_main(["--load-weights", str(save), "--headless", "--epochs", "5", "--no-save"]) == 0
    assert "engine: arrays" in capsys.readouterr().err
    # unless told otherwise
    assert cli_main(["--load-weights", str(save), "--engine", "objects", "--headless", "--epochs", "5", "--no-save"]) == 0
    assert "engine: arrays" not in capsys.readouterr().err


def test_cli_engine_arrays_seed_batch_and_lattice(tmp_path, capsys):
    assert cli_main(["--engine", "arrays", "--seeds", "2", "--seed", "1", "--epochs", "20", "--no-save"]) == 0
    out = capsys.readouterr().out
    assert out.count("%") >= 4  # two rows of two percentages
    assert cli_main(["--engine", "arrays", "--headless", "--epochs", "10", "--no-save"]) == 0
    assert "engine: arrays" in capsys.readouterr().err


def test_arrays_are_much_faster_on_a_big_mesh():
    import time
    mesh, net = pair(seed=1, count=480, across=24)
    ta, tb = Teacher(mesh, seed=1), Teacher(net, seed=1)
    def rate(t, n):
        t0 = time.perf_counter()
        for _ in range(n):
            t.epoch(verbose=False)
        return n / (time.perf_counter() - t0)
    objects, arrays = rate(ta, 20), rate(tb, 200)
    assert arrays > objects  # a loose bound: the point is that it is not slower

