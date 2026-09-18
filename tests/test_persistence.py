import json

import pytest

from walnutbutter.constants import ESCAPE_DELTA
from walnutbutter.goo import Goo
from walnutbutter.learning import Teacher
from walnutbutter.monitor import main, run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.persistence import checkpoint, load_weights, read_checkpoint, restore, resume_teacher


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def test_checkpoint_round_trips_weights_settings_and_permutation(tmp_path):
    grid = main(count=48, across=8, seed=5)
    teacher = Teacher(grid, seed=5)
    for _ in range(20):
        teacher.epoch(verbose=False)
    path = tmp_path / "w.json"
    written = checkpoint(grid, path, teacher)

    restored, data = restore(path)
    assert data == written
    assert (restored.across, restored.count, restored.seed) == (8, 48, 5)
    assert restored.epoch == grid.epoch == 21
    assert [c.weight for c in restored.connections.values()] == [c.weight for c in grid.connections.values()]
    # the mesh itself was rebuilt identically: same shortcuts, same neighbours
    assert [(c.source.name, c.target.name, c.kind) for c in restored.connections.values()] == [
        (c.source.name, c.target.name, c.kind) for c in grid.connections.values()
    ]



def test_a_checkpoint_keeps_the_single_spike_rules_state(tmp_path):
    """§6.7 (September 17, 2026): p_hat_j, the decisions to date, E_j and each synapse's note round-trip beside the traces."""
    import math
    from walnutbutter.neuron import Neuron
    was = Neuron.tau
    Neuron.tau = math.inf
    try:
        grid = Goo(count=32, across=8, seed=3, scaling_factor=0.5)
        grid.set_delta(0.7)
        teacher = Teacher(grid, seed=3, rule="reinforce", eligibility="hebb")
        for _ in range(5):
            teacher.epoch(verbose=False)
        neurons = list(grid.all_neurons())
        assert any(n.expectation is not None for n in neurons) and any(n.decisions for n in neurons)
        # the notes and traces are settled and cleared at each spike (§8.11), so at the epoch boundary a network
        # this active carries none open; what the checkpoint must keep is the per-neuron state, and every field
        assert all(n.decisions for n in neurons)
        data = checkpoint(grid, tmp_path / "s.json", teacher)
        assert data["learning"]["eligibility"] == "hebb" and len(data["notes"]) == len(grid.connections)
        back, _ = restore(tmp_path / "s.json")
        assert [n.expectation for n in back.all_neurons()] == [n.expectation for n in neurons]
        assert [n.decisions for n in back.all_neurons()] == [n.decisions for n in neurons]
        assert [n.expected for n in back.all_neurons()] == [n.expected for n in neurons]
        assert [c.noted for c in back.connections.values()] == [c.noted for c in grid.connections.values()]
        assert all(n.traced for n in back.all_neurons())  # escape noise keeps the trace, restored with the widths
    finally:
        Neuron.tau = was


def test_checkpoint_is_written_atomically_and_is_json(tmp_path):
    grid = Goo(count=12, across=4, seed=1)
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    assert json.loads(path.read_text())["connections"] == len(grid.connections)
    assert not (tmp_path / "w.json.tmp").exists()


def test_resume_teacher_continues_the_statistics(tmp_path):
    grid = main(count=32, across=8, seed=2)
    teacher = Teacher(grid, target="all-off", seed=2)
    for _ in range(30):
        teacher.epoch(verbose=False)
    path = tmp_path / "w.json"
    checkpoint(grid, path, teacher)

    restored, data = restore(path)
    fresh = Teacher(restored, target="all-off", seed=3)
    resume_teacher(fresh, data)
    assert fresh.epochs == 30
    assert fresh.accuracy_to_date == pytest.approx(teacher.accuracy_to_date)
    assert fresh.average == teacher.average and fresh.baseline == teacher.baseline
    assert "to date over 30 epochs" in fresh.status()


def test_load_weights_rejects_a_different_mesh(tmp_path):
    grid = Goo(count=24, across=6, seed=1)
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    data = read_checkpoint(path)
    with pytest.raises(ValueError):
        load_weights(Goo(count=30, across=6, seed=1), data)
    with pytest.raises(ValueError, match="seed"):
        load_weights(Goo(count=24, across=6, seed=2), data)  # same size, different wiring


def test_restore_refuses_an_unseeded_mesh(tmp_path):
    grid = Goo(count=12, across=4)  # no seed: the drawn wiring cannot be rebuilt
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    with pytest.raises(ValueError, match="seed"):
        restore(path)


def test_unknown_format_is_rejected(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"format": 99}))
    with pytest.raises(ValueError):
        read_checkpoint(path)


def test_checkpoint_without_teacher_has_no_learning_record(tmp_path):
    grid = Goo(count=12, across=4, seed=1)
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    data = read_checkpoint(path)
    assert "learning" not in data
    teacher = Teacher(restore(path)[0])
    resume_teacher(teacher, data)  # a no-op
    assert teacher.epochs == 0


def test_checkpoint_keeps_the_weight_range(tmp_path):
    grid = Goo(count=24, across=6, weight=None, seed=1, weight_range=(0.01, 1.0))
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    restored, data = restore(path)
    assert data["weight_range"] == [0.01, 1.0]
    assert restored.weight_range == (0.01, 1.0)
    assert [c.weight for c in restored.connections.values()] == [c.weight for c in grid.connections.values()]


def test_checkpoint_keeps_per_neuron_thresholds_and_rates(tmp_path):
    grid = main(count=32, across=8, seed=1)
    teacher = Teacher(grid, seed=1, homeostasis=0.05)
    for _ in range(200):
        teacher.epoch(verbose=False)
    thresholds = [n.threshold for n in grid.all_neurons()]
    assert len(set(thresholds)) > 1  # homeostasis has spread them out
    path = tmp_path / "w.json"
    checkpoint(grid, path, teacher)
    restored, data = restore(path)
    assert [n.threshold for n in restored.all_neurons()] == thresholds
    assert [n.rate for n in restored.all_neurons()] == [n.rate for n in grid.all_neurons()]
    assert data["learning"]["homeostasis"] == 0.05


def test_checkpoint_keeps_the_minimum_potential(tmp_path):
    grid = Goo(count=24, across=6, seed=1, minimum_potential=-0.4, scale_with_fan_in=False)
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    restored, data = restore(path)
    assert data["minimum_potential"] == -0.4
    assert all(n.minimum_potential == -0.4 for n in restored.all_neurons())


def test_checkpoint_without_a_floor_restores_without_one(tmp_path):
    grid = Goo(count=24, across=6, seed=1, scale_with_fan_in=False)
    path = tmp_path / "w.json"
    data = checkpoint(grid, path)
    del data["minimum_potential"], data["floors"]  # a file old enough to lack the scalar lacks the per-neuron list too (§5.2)
    path.write_text(json.dumps(data))
    restored, _ = restore(path)
    assert all(n.minimum_potential == -1.0 for n in restored.all_neurons())  # the fallback _restore_goo names


def test_a_checkpoint_carries_a_floor_per_neuron_once_a_container_scales_with_fan_in(tmp_path):
    """§5.2: the floor stopped being one scalar when goo started rescaling its potential axis."""
    from walnutbutter.goo import Goo
    goo = Goo(count=24, across=6, seed=1, weight=None, projection=1.0, wiring="zones-equal")  # under the zone rule at P 1 an input of 24 hears exactly the 12 interior
    path = tmp_path / "goo.json"
    data = checkpoint(goo, path)
    assert data["scale_with_fan_in"] is True
    assert data["floors"] == [n.minimum_potential for n in goo.all_neurons()]
    from walnutbutter.constants import GOO_MINIMUM_POTENTIAL
    assert data["floors"][0] == pytest.approx(GOO_MINIMUM_POTENTIAL * 12 / 18)  # an input of 24 hears the 12 interior (§3.4)
    restored, _ = restore(path)
    assert [n.minimum_potential for n in restored.all_neurons()] == data["floors"]
    assert [n.threshold for n in restored.all_neurons()] == data["thresholds"]


def test_checkpoint_records_the_unstick_settings(tmp_path):
    grid = main(count=32, across=8, seed=1)
    teacher = Teacher(grid, seed=1, unstick=0.005, unstick_target=0.45)
    teacher.step()
    path = tmp_path / "w.json"
    data = checkpoint(grid, path, teacher)
    assert data["learning"]["unstick"] == 0.005 and data["learning"]["unstick_target"] == 0.45


def test_checkpoint_carries_the_accuracy_history_and_resume_continues_it(tmp_path):
    grid = main(count=32, across=8, seed=1)
    teacher = Teacher(grid, seed=1)
    teacher.step()
    for i in range(3):
        teacher.epoch(verbose=False)
        teacher.record(elapsed=float(i), epochs_per_second=100.0)
    path = tmp_path / "w.json"
    data = checkpoint(grid, path, teacher)
    assert len(data["learning"]["history"]) == 3
    restored, data = restore(path)
    resumed = Teacher(restored, seed=2)
    resume_teacher(resumed, data)
    assert resumed.history == teacher.history
    resumed.epoch(verbose=False)
    resumed.record(elapsed=0.5)
    assert len(resumed.history) == 4 and resumed.history[-1]["epoch"] == 5
