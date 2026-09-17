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
@pytest.mark.parametrize("eligibility", ["perturb", "wrong_hebb", "hebb"])
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
@pytest.mark.parametrize("eligibility", ["wrong_hebb", "hebb", "count_hebb"])
def test_goo_runs_on_the_rust_engine_and_agrees_with_the_object_engine(eligibility):
    """§3.4 on §6.15: the container the sweep is about, on the engine the sweep runs on, bit for bit."""
    from walnutbutter.goo import Goo
    from walnutbutter.learning import Teacher
    goo = Goo(count=40, across=8, seed=3, weight=None)
    goo.rule, goo.drive = "reinforce", "rate"
    teacher = Teacher(goo, seed=7, rule="reinforce", eligibility=eligibility, homeostasis=0.01, unstick=0.1)
    parted = fast.compare(goo, epochs=60, teacher=teacher)
    assert parted == [], parted
    from walnutbutter.constants import GOO_THRESHOLD
    assert goo.all_neurons()[0].threshold != goo.fan_in_scale() * GOO_THRESHOLD  # moved by homeostasis, on both engines alike


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_compare_refuses_what_it_cannot_mirror():
    from walnutbutter.learning import Teacher
    grid = mesh(rows=2, seed=1)
    with pytest.raises(ValueError, match="row, class, graded or evidence critic"):
        fast.compare(grid, epochs=1, teacher=Teacher(grid, seed=1, rule="reinforce", late="ignore", homeostasis=0.0, unstick=0.0))


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_train_runs_the_perturb_rule_from_a_seeded_stream():
    a = fast.train(mesh(rows=2, seed=4), 30, eligibility="perturb", sigma=0.1, seed=5)
    b = fast.train(mesh(rows=2, seed=4), 30, eligibility="perturb", sigma=0.1, seed=5)
    c = fast.train(mesh(rows=2, seed=4), 30, eligibility="perturb", sigma=0.1, seed=6)
    assert list(a[2].weights()) == list(b[2].weights())  # the same seed, the same run
    assert list(a[2].weights()) != list(c[2].weights())  # a different stream, a different one
    assert set(a[3]) >= {"last_tenth", "rates", "thresholds", "stuck_on", "stuck_off"} and 0 <= a[3]["last_tenth"] <= 1
    flat = fast.train(mesh(rows=2, seed=4), 30, homeostasis=0.0, unstick=0.0)  # the default eligibility, hebb (§6.7)
    assert flat[3]["thresholds"] == [0.25] * len(flat[3]["thresholds"])  # nothing moved them
    assert len(flat[3]["expected_counts"]) == len(flat[3]["rates"]) and any(e is not None for e in flat[3]["expected_counts"])
    old = fast.train(mesh(rows=2, seed=4), 30, eligibility="wrong_hebb", homeostasis=0.0, unstick=0.0)
    assert list(old[2].weights()) != list(flat[2].weights())  # the +-1 rule and the centred one are not the same rule
    with pytest.raises(ValueError, match="eligibility must be"):
        fast.train(mesh(rows=2), 1, eligibility="magic")
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


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_a_resumed_run_measures_from_its_first_start_and_counts_its_epochs_on(tmp_path):
    """docs/rust-sweep.py --resume-from: the saved network continued, the estimator against the first start's weights, the
    trace's epochs offset, the stream advanced to the epoch reached (September 16, 2026)."""
    import importlib.util
    import numpy as np
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rs", Path(__file__).resolve().parent.parent / "docs" / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    from walnutbutter.network import input_stream
    arm = {"goo": 24.0, "seed": 3}
    fresh, cli = rs.grid_of("copy", arm, "hazard", True, -4.0)
    first = [c.weight for n in fresh.all_neurons() for c in n.outgoing]
    everyone = lambda edges: (np.arange(len(edges)) % 2 * 2.0 - 1.0, np.ones(len(edges), dtype=bool))  # a direction with spread
    patterns = input_stream(12, fresh.raw_bit_count(), 3)
    mean, trace, engine, report = fast.train(fresh, 6, lr=cli.lr, target="copy", trace_every=3, patterns=patterns, eligibility="hazard",
                                             seed=3, homeostasis=0.0, unstick=0.0, direction=everyone)
    assert [e["epoch"] for e in report["estimator"]] == [3, 6] and fresh.input_at == 6
    rs._save_network(engine, fresh, report, tmp_path / "arm-network.json")
    again, _ = rs.grid_of("copy", arm, "hazard", True, -4.0)
    grid, offset, reference = rs.resume_grid(again, tmp_path / "arm-network.json")
    assert offset == 6 and reference == first and grid is not again
    assert [c.weight for n in grid.all_neurons() for c in n.outgoing] == list(engine.weights())  # the saved state, whole
    assert [n.rate for n in grid.all_neurons()] == report["rates"] and [n.expected_count for n in grid.all_neurons()] == report["expected_counts"]
    grid.use_input_stream(patterns, None)
    grid.input_at = offset
    w_saved = np.array(engine.weights())
    mean, trace, engine2, report2 = fast.train(grid, 6, lr=cli.lr, target="copy", trace_every=3, eligibility="hazard", seed=3 + 1_000_000,
                                               homeostasis=0.0, unstick=0.0, direction=everyone, reference_weights=reference, epoch_offset=offset)
    assert [e["epoch"] for e in report2["estimator"]] == [9, 12] and grid.input_at == 12 and grid.epoch == 12
    w_end = np.array(engine2.weights())
    d = np.arange(len(w_end)) % 2 * 2.0 - 1.0
    want = float(np.corrcoef(w_end - np.array(first), d)[0, 1]) if (w_end - np.array(first)).std() > 0 else None
    assert report2["estimator"][-1]["corr_cum"] == want  # against the first start, not the resumed one
    with pytest.raises(ValueError, match="reference weights cover"):
        fast.train(rs.grid_of("copy", arm, "hazard", True, -4.0)[0], 1, direction=everyone, reference_weights=[0.0], eligibility="hazard", seed=1)
    other, _ = rs.grid_of("copy", {"goo": 30.0, "seed": 3}, "hazard", True, -4.0)
    with pytest.raises(ValueError, match="same seed and layout"):
        rs.resume_grid(other, tmp_path / "arm-network.json")
    wrong_seed, _ = rs.grid_of("copy", {"goo": 24.0, "seed": 4}, "hazard", True, -4.0)  # the same layout, another seed's weights
    with pytest.raises(ValueError, match="own seed"):
        rs.resume_grid(wrong_seed, tmp_path / "arm-network.json")


@pytest.mark.skipif(not fast.available(), reason="the Rust schedule is not built")
def test_a_resumed_network_keeps_the_isi_factor_it_was_saved_under(tmp_path):
    """§0.2 (September 17, 2026): the factor is on for a fresh arm, and a resume keeps its checkpoint's setting -- off for
    one saved before the factor existed -- unless told otherwise, so an arm runs on under the rule it started with."""
    import importlib.util
    import json
    from pathlib import Path
    from walnutbutter.neuron import Neuron
    spec = importlib.util.spec_from_file_location("rs", Path(__file__).resolve().parent.parent / "docs" / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    was = Neuron.isi_factor
    try:
        arm = {"goo": 24.0, "seed": 3}
        fresh, cli = rs.grid_of("copy", arm, "hazard", True, -4.0)
        assert Neuron.isi_factor is True and cli.isi_factor is True
        _, off = rs.grid_of("copy", arm, "hazard", True, -4.0, fixed=("--no-isi-factor",))
        assert Neuron.isi_factor is False and off.isi_factor is False
        mean, trace, engine, report = fast.train(fresh, 2, lr=cli.lr, target="copy", eligibility="hazard", seed=3,
                                                 homeostasis=0.0, unstick=0.0)
        path = tmp_path / "arm-network.json"
        rs._save_network(engine, fresh, report, path)  # saved while the factor was off (the last grid_of set it)
        assert json.loads(path.read_text())["isi_factor"] is False
        again, _ = rs.grid_of("copy", arm, "hazard", True, -4.0)
        assert Neuron.isi_factor is True
        rs.resume_grid(again, path)
        assert Neuron.isi_factor is False  # the checkpoint's
        rs.resume_grid(rs.grid_of("copy", arm, "hazard", True, -4.0)[0], path, True)
        assert Neuron.isi_factor is True  # unless told otherwise
        on_grid, _ = rs.grid_of("copy", arm, "hazard", True, -4.0)  # and a network saved with the factor on
        mean, trace, engine, report = fast.train(on_grid, 2, lr=cli.lr, target="copy", eligibility="hazard", seed=3,
                                                 homeostasis=0.0, unstick=0.0)
        on_path = tmp_path / "on-network.json"
        rs._save_network(engine, on_grid, report, on_path)
        assert json.loads(on_path.read_text())["isi_factor"] is True
        rs.resume_grid(rs.grid_of("copy", arm, "hazard", True, -4.0, fixed=("--no-isi-factor",))[0], on_path)
        assert Neuron.isi_factor is True  # the checkpoint's, over the resuming grid's own
        rs.resume_grid(rs.grid_of("copy", arm, "hazard", True, -4.0)[0], on_path, False)
        assert Neuron.isi_factor is False  # unless told otherwise
        data = json.loads(path.read_text())
        del data["isi_factor"]
        path.write_text(json.dumps(data))
        rs.resume_grid(rs.grid_of("copy", arm, "hazard", True, -4.0)[0], path)
        assert Neuron.isi_factor is False  # a network saved before the factor existed ran without it
    finally:
        Neuron.isi_factor = was
