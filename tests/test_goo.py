"""Goo: the fourth container, with the plane taken away (AUTHORITY.md §3.4)."""

import json
import pathlib

import pytest

from walnutbutter.cli import build_parser, cli_main
from walnutbutter.constants import ACROSS, GOO_COUNT, GOO_MINIMUM_POTENTIAL, GOO_THRESHOLD, THRESHOLD_FAN_IN, WEIGHT_RANGE
from walnutbutter.goo import DEFAULT_COUNT, Goo
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.persistence import checkpoint, restore


@pytest.fixture(autouse=True)
def _deterministic_stimulus(forced_input):
    """This file is about the container, not the input process (see conftest.forced_input)."""


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def wiring(goo) -> list[tuple[str, str]]:
    """Every connection as (source, target), in id order."""
    return [(goo.connections[i].source.name, goo.connections[i].target.name) for i in range(1, len(goo.connections) + 1)]


# --- what goo is ----------------------------------------------------------






# --- the zones ------------------------------------------------------------


def test_nothing_between_the_zones_is_addressable_which_is_what_makes_it_goo():
    goo = Goo(count=20, across=4, seed=1)
    assert goo.get_neuron_at(0, 2) is None and goo.get_neuron_at(0, -1) is None
    assert goo.get_neuron_at(4, 0) is None and goo.get_neuron_at(-1, 1) is None




def test_the_permutation_scrambles_the_coded_bits_across_the_input_zone():
    goo = Goo(count=20, across=4, seed=1)
    assert sorted(goo.permutation) == [0, 1, 2, 3] and goo.permutation != [0, 1, 2, 3]
    assert Goo(count=20, across=4, seed=1, permute=False).permutation == [0, 1, 2, 3]


# --- what it is for -------------------------------------------------------


# --- running it -----------------------------------------------------------

def test_goo_presents_an_input_and_is_read_like_any_other_container():
    goo = Goo(seed=1, weight=None)
    run_epoch(goo, bits=[True, False, True, False], verbose=False)
    assert goo.epoch == 1 and goo.input_bits == [True, False, True, False]
    assert len(goo.input_pattern) == 8 and len(goo.output_fired()) == 8
    assert goo.waves and goo.total_spikes() > 0


def test_both_engines_run_goo_the_same_way_wave_by_wave():
    """§7: the object engine and the array engine are the same network, in goo as anywhere else."""
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork

    mesh, net = Goo(count=30, seed=5, weight=None), ArrayNetwork(Goo(count=30, seed=5, weight=None))
    index = {n: i for i, n in enumerate(mesh.all_neurons())}
    for _ in range(20):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert net.input_bits == mesh.input_bits and net.input_pattern == mesh.input_pattern
        assert net.output_fired() == mesh.output_fired()
        assert [w.fired.tolist() for w in net.waves] == [sorted(index[n] for n in w.fired) for w in mesh.waves]
    assert net.epoch == mesh.epoch == 20


# --- keeping it -----------------------------------------------------------

def test_a_goo_checkpoint_round_trips_from_its_count_alone():
    goo = Goo(count=24, across=6, seed=4, weight=None)
    for _ in range(5):
        run_epoch(goo, verbose=False)
    data = checkpoint(goo, "goo.json")
    assert data["container"] == "goo" and data["count"] == 24 and data["shortcuts"] == []
    back, _ = restore("goo.json")
    assert isinstance(back, Goo) and wiring(back) == wiring(goo)
    assert [back.connections[i].weight for i in range(1, 25)] == [goo.connections[i].weight for i in range(1, 25)]
    assert back.permutation == goo.permutation and back.epoch == 5 and back.time == goo.time
    assert [n.potential for n in back.all_neurons()] == [n.potential for n in goo.all_neurons()]


def test_goo_restores_without_a_seed_because_no_draw_chose_its_wiring():
    """Unlike a grid, whose shortcuts cannot be rebuilt without the stream that picked them."""
    goo = Goo(count=12, across=4, seed=None, weight=None, projection=1.0)  # at P 1 no draw chose the wiring
    run_epoch(goo, verbose=False)
    checkpoint(goo, "seedless.json")
    back, _ = restore("seedless.json")
    assert [c.weight for c in back.connections.values()] == [c.weight for c in goo.connections.values()]


# --- the command line -----------------------------------------------------



def test_goo_is_headless_and_has_no_picture_to_save(capsys):
    assert cli_main(["--goo", "--epochs", "2", "--seed", "1", "--no-save"]) == 0  # no --headless, and no window opens
    assert cli_main(["--goo", "--save", "goo.png", "--epochs", "2", "--no-save"]) == 2
    assert "no geometry to draw" in capsys.readouterr().err




def test_goo_is_a_container_of_its_own_and_will_not_be_mixed_with_another(capsys):
    assert cli_main(["--goo", "--nodes", "--headless", "--epochs", "2", "--no-save"]) == 2
    assert "cannot be combined with --nodes" in capsys.readouterr().err
    assert cli_main(["--goo", "--layers", "3", "--headless", "--epochs", "2", "--no-save"]) == 2
    assert "cannot be combined with --layers" in capsys.readouterr().err


# --- the potential axis scales with fan-in (AUTHORITY.md §5.2) ------------


def test_the_scaling_can_be_switched_off_and_then_goo_is_flat():
    goo = Goo(seed=1, scale_with_fan_in=False)
    assert not goo.scale_with_fan_in_on
    assert {n.threshold for n in goo.all_neurons()} == {GOO_THRESHOLD} == {0.2}  # goo's own, not the grid's 0.25
    assert {n.minimum_potential for n in goo.all_neurons()} == {GOO_MINIMUM_POTENTIAL} and GOO_MINIMUM_POTENTIAL == pytest.approx(-0.8)



def test_the_rule_is_available_to_any_container_and_reads_each_neurons_own_fan_in():
    """Off by default everywhere but goo, because turning it on moves every threshold results were measured at."""
    from walnutbutter.grid import GridOfNeurons
    grid = GridOfNeurons(seed=1, weight=None)
    assert {n.threshold for n in grid.all_neurons()} == {0.25}  # untouched until asked
    grid.scale_with_fan_in(0.25, -1.0)
    degrees = {len(n.incoming) for n in grid.all_neurons()}
    assert len(degrees) > 1  # a grid is not homogeneous: corners have fewer neighbours than the interior
    assert len({round(n.threshold, 9) for n in grid.all_neurons()}) == len(degrees)
    for n in grid.all_neurons():
        assert n.threshold == pytest.approx(0.25 * len(n.incoming) / THRESHOLD_FAN_IN)
        assert n.minimum_potential / n.threshold == pytest.approx(-4.0)
    with pytest.raises(ValueError, match="reference fan-in must be positive"):
        grid.scale_with_fan_in(0.25, -1.0, reference=0)


def test_both_engines_see_the_scaled_axis():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork
    mesh = Goo(count=30, across=6, seed=2, weight=None)
    net = ArrayNetwork(mesh)
    assert net.threshold_v.tolist() == [n.threshold for n in mesh.all_neurons()]
    assert net.floor.tolist() == [n.minimum_potential for n in mesh.all_neurons()]



def test_any_container_can_be_asked_to_scale_from_the_command_line(capsys):
    assert cli_main(["--scale-with-fan-in", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "every threshold and floor rescaled by in-degree / 18" in capsys.readouterr().err



# --- the count read (AUTHORITY.md §4.3): count, estimate a rate, threshold it ----------

def test_the_count_read_counts_the_epochs_spikes_and_thresholds_the_rate():
    from walnutbutter.constants import TEACHER_THRESHOLD
    goo = Goo(count=20, across=4, seed=1, permute=False)
    goo.read = "count"
    assert goo.teacher_threshold == TEACHER_THRESHOLD == 14.3  # the middle of the one-spike band at 35 ms
    run_epoch(goo, bits=[True, False], verbose=False)
    per_ms = 1000.0 / goo.interval
    for neuron, hz, on in zip(goo.output_row(), goo.output_counts_hz(), goo.output_fired()):
        assert hz == neuron.epoch_spikes * per_ms and on == (hz >= 14.3)
    # at 35 ms one spike is 28.6 Hz and reads on; none reads off: the line sits halfway between them
    a, b = goo.output_row()[0], goo.output_row()[1]
    a.spikes_at_reset, a.spikes = 10, 10
    b.spikes_at_reset, b.spikes = 10, 11
    assert goo.output_fired()[:2] == [False, True]
    goo.teacher_threshold = 42.9  # the middle of the two-spike band: one spike now reads off, two on
    b.spikes = 12
    assert goo.output_fired()[:2] == [False, True]
    a.spikes = 11
    assert goo.output_fired()[0] is False
    goo.teacher_threshold = TEACHER_THRESHOLD
    a.spikes = 10  # back to none for a and one for b: the one-spike read the levels below are asserted against
    assert goo.output_levels()[:2] == [0.0, 1.0]  # a bit, as the row critic wants it
    goo.reset()
    assert a.epoch_spikes == b.epoch_spikes == 0  # the next epoch starts its count afresh


def test_the_count_read_is_the_same_in_all_three_engines():
    pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter import fast
    from walnutbutter.learning import Teacher
    mesh, twin = Goo(count=30, across=6, seed=5, weight=None, permute=False), Goo(count=30, across=6, seed=5, weight=None, permute=False)
    for g in (mesh, twin):
        g.read, g.rule, g.drive = "count", "reinforce", "rate"
    net = ArrayNetwork(twin)
    for _ in range(20):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert net.output_counts_hz() == mesh.output_counts_hz() and net.output_fired() == mesh.output_fired()
    if fast.available():
        goo = Goo(count=30, across=6, seed=5, weight=None, permute=False)
        goo.read, goo.rule, goo.drive = "count", "reinforce", "rate"
        teacher = Teacher(goo, seed=7, rule="reinforce", eligibility="hebb", target="copy", homeostasis=0.01, unstick=0.1)
        parted = fast.compare(goo, epochs=40, teacher=teacher)  # the reward is the count read on both sides
        assert parted == [], parted
        goo.read = "window"
        with pytest.raises(ValueError, match="reads 'fired' or 'count'"):
            fast.compare(goo, epochs=1, teacher=Teacher(goo, seed=7, rule="reinforce", target="copy", homeostasis=0.0, unstick=0.0))


def test_copy_is_read_by_count_and_the_threshold_reaches_the_command_line(tmp_path, capsys):
    from walnutbutter.problems import PROBLEMS
    assert PROBLEMS["copy"].read == "count"
    save = tmp_path / "count.json"
    assert cli_main(["--problem", "copy", "--goo", "--teacher-threshold", "60", "--headless", "--epochs", "3", "--seed", "1",
                     "--save-weights", str(save)]) == 0
    assert json.loads(save.read_text())["teacher_threshold"] == 60.0
    assert cli_main(["--load-weights", str(save), "--headless", "--epochs", "2", "--no-save"]) == 0



# --- no direct projection (AUTHORITY.md §3.4, Byron, September 14, 2026) -------------


# --- the zone rule (AUTHORITY.md §3.4, Byron, September 14, 2026) ---------------------

def test_the_default_goo_is_sixty_neurons_at_projection_a_fifth_by_one_probability():
    """§3.4, Byron, September 15, 2026: 'Please eliminate the zone rule: P(i projects onto j) = 0 if i == j, P otherwise.'"""
    from walnutbutter.constants import GOO_PROJECTION
    goo = Goo(seed=1)
    assert DEFAULT_COUNT == GOO_COUNT == 60 and len(goo) == goo.count == 60
    assert goo.projection == GOO_PROJECTION == 0.2 and goo.wiring == "uniform"
    assert 600 < len(goo.connections) < 820  # the seed's fifth of the 3,540 ordered pairs
    assert len(goo.interior()) == 44 and len(goo.zone_indices()) == 16
    assert repr(goo).startswith("Goo(60 neurons, ") and repr(goo).endswith(" projections at P 0.2; 8 in, 8 out)")
    assert Goo(count=ACROSS * 10, seed=1).count == 80  # the grid's eighty is still one --goo away
    full = Goo(seed=1, projection=1.0)  # the fully connected goo: every ordered pair
    assert len(full.connections) == 60 * 59 == 3540 and full.mean_out_degree() == pytest.approx(59)
    assert all(len(n.incoming) == 59 for n in full.all_neurons())  # every neuron hears everyone
    assert repr(full) == "Goo(60 neurons, 3540 projections at P 1; 8 in, 8 out)"


def test_every_ordered_pair_projects_once_each_way():
    goo = Goo(count=5, across=2, seed=1, projection=1.0)  # zones {0, 1} and {3, 4}; the interior is neuron 2
    pairs = wiring(goo)
    assert len(pairs) == len(set(pairs)) == 20
    assert set(pairs) == {(f"Goo_{i}", f"Goo_{j}") for i in range(5) for j in range(5) if i != j}
    assert [c.id for c in goo.connections.values()] == list(range(1, 21))  # ids contiguous, source-major


def test_at_projection_one_the_count_fixes_the_topology_and_below_it_the_seed_does():
    one, two = Goo(count=6, across=2, seed=1, weight=None, projection=1.0), Goo(count=6, across=2, seed=2, weight=None, projection=1.0)
    assert wiring(one) == wiring(two)  # P 1: nothing drawn for the topology
    weights = lambda g: [g.connections[i].weight for i in range(1, len(g.connections) + 1)]
    assert weights(one) != weights(two) and weights(one) == weights(Goo(count=6, across=2, seed=1, weight=None, projection=1.0))
    a, b, c = (Goo(count=30, across=6, seed=s, weight=None, projection=0.5) for s in (1, 1, 2))
    assert wiring(a) == wiring(b) and wiring(a) != wiring(c)  # below 1 the seed decides, and repeats itself
    possible = 30 * 29  # every ordered pair
    assert 0.4 * possible < len(a.connections) < 0.6 * possible
    assert "at P 0.5" in repr(a)


def test_there_are_never_any_shortcuts_because_there_is_no_neighbourhood_to_get_past():
    goo = Goo(count=12, across=4, seed=1, projection=1.0)
    assert goo.small_world_connections() == [] and goo.omega == 0.0
    assert goo.local_connections() == list(goo.connections.values())
    assert len(goo.connections_of_kind("goo")) == 12 * 11 == 132
    assert goo.connections_of_kind("small_world") == []


def test_a_weight_is_fixed_or_drawn_in_connection_id_order_inside_the_range():
    assert {c.weight for c in Goo(count=5, across=2, weight=0.3).connections.values()} == {0.3}
    low, high = WEIGHT_RANGE
    drawn = Goo(count=24, seed=3, weight=None)
    assert all(low <= c.weight <= high for c in drawn.connections.values())
    narrow = Goo(count=24, seed=3, weight=None, weight_range=(0.2, 0.3))
    assert all(0.2 <= c.weight <= 0.3 for c in narrow.connections.values())
    assert narrow.clip_weight(5.0) == 0.3 and narrow.clip_weight(-5.0) == 0.2


def test_the_zones_go_by_index_because_there_are_no_places_to_go_by():
    goo = Goo(count=20, across=4, seed=1, permute=False)
    assert goo.rows == 2  # it is counting the two zones, not any depth
    assert [n.name for n in goo.input_row()] == ["Goo_0", "Goo_1", "Goo_2", "Goo_3"]
    assert [n.name for n in goo.output_row()] == ["Goo_16", "Goo_17", "Goo_18", "Goo_19"]
    assert goo.get_neuron_at(0, 1) is goo[0] and goo.get_neuron_at(0, 0) is goo[16]
    assert goo.input_width() == 4 and len(goo.interior()) == 12
    assert goo.zone_indices() == {0, 1, 2, 3, 16, 17, 18, 19}


def test_goo_refuses_overlapping_zones_and_a_projection_off_the_unit_interval():
    """Under one probability the zones need no interior between them, but they may not overlap."""
    for count in (15, 12, 8, 7):
        with pytest.raises(ValueError, match="would overlap"):
            Goo(count=count, across=8)
    assert Goo(count=16, across=8, seed=1).interior() == []  # the zones abut: no interior, and that is allowed
    assert Goo(count=17, across=8, seed=1).interior()[0].name == "Goo_8"
    with pytest.raises(ValueError, match="needs an interior"):
        Goo(count=16, across=8, wiring="zones")  # the earlier wirings did need one
    with pytest.raises(ValueError, match="unknown wiring"):
        Goo(count=20, across=4, wiring="mesh")
    with pytest.raises(ValueError, match="at least one neuron wide"):
        Goo(count=10, across=0)
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="projection probability"):
            Goo(count=20, across=4, projection=bad)


def test_at_projection_one_every_neuron_reaches_everyone_and_the_zones_are_not_apart():
    goo = Goo(count=40, across=8, seed=1, projection=1.0)
    everyone = set(goo.all_neurons())
    for n in goo.all_neurons():
        assert {c.target for c in n.outgoing} == everyone - {n} and {c.source for c in n.incoming} == everyone - {n}
    assert not goo.zones_are_apart()  # an input projects straight onto an output again
    assert any(c.source in goo.input_row() and c.target in goo.output_row() for c in goo.connections.values())


def test_goo_rescales_its_potential_axis_by_each_neurons_own_fan_in():
    goo = Goo(seed=1, projection=1.0)  # at P 1 the fan-in is exact: 59 for everyone
    inner, edge = goo.interior()[0], goo.input_row()[0]
    assert inner.threshold == pytest.approx(GOO_THRESHOLD * 59 / THRESHOLD_FAN_IN) == pytest.approx(0.656, abs=1e-3)
    assert edge.threshold == inner.threshold
    assert all(n.minimum_potential / n.threshold == pytest.approx(-4.0) for n in goo.all_neurons())  # one axis
    assert goo.fan_in_scale() == pytest.approx(59 / THRESHOLD_FAN_IN)  # the interior's stretch at projection 1
    sparse = Goo(seed=1)  # at the default P the fan-in is the seed's, and each neuron is scaled by its own
    assert all(n.threshold == pytest.approx(GOO_THRESHOLD * len(n.incoming) / THRESHOLD_FAN_IN) for n in sparse.all_neurons())
    flat = Goo(seed=1, scale_with_fan_in=False)
    assert {n.threshold for n in flat.all_neurons()} == {GOO_THRESHOLD} == {0.2}
    assert {n.minimum_potential for n in flat.all_neurons()} == {GOO_MINIMUM_POTENTIAL}


def test_a_neuron_hearing_exactly_the_reference_fan_in_is_left_where_the_grid_starts():
    """§5.2: THRESHOLD and MINIMUM_POTENTIAL are quoted at THRESHOLD_FAN_IN incoming synapses, so that fan-in is a fixed point."""
    goo = Goo(count=int(THRESHOLD_FAN_IN) + 1, across=4, seed=1, threshold=0.25, minimum_potential=-1.0, projection=1.0)
    for n in goo.all_neurons():  # every neuron of 19 hears 18
        assert n.threshold == pytest.approx(0.25) and n.minimum_potential == pytest.approx(-1.0)


def test_below_projection_one_a_goo_round_trips_with_its_seed_and_refuses_without():
    pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter import fast
    from walnutbutter.learning import Teacher
    goo = Goo(count=30, across=6, seed=3, weight=None, projection=0.5)
    run_epoch(goo, verbose=False)
    data = checkpoint(goo, "half.json")
    assert data["projection"] == 0.5 and data["connections"] == len(goo.connections)
    back, _ = restore("half.json")
    assert wiring(back) == wiring(goo) and back.projection == 0.5 and back.wiring == "uniform"
    with pytest.raises(ValueError, match="without a seed"):
        checkpoint(Goo(count=30, across=6, seed=None, weight=None, projection=0.5), "seedless.json")
        restore("seedless.json")
    old = json.loads(pathlib.Path("half.json").read_text()); old["direct_projection"] = False
    pathlib.Path("old.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="before the zone rule"):
        restore("old.json")
    # every engine runs the drawn wiring the same way
    mesh, twin = (Goo(count=30, across=6, seed=3, weight=None, projection=0.5) for _ in range(2))
    net = ArrayNetwork(twin)
    for _ in range(10):
        run_epoch(mesh, verbose=False); run_epoch(net, verbose=False)
        assert net.output_fired() == mesh.output_fired()
    if fast.available():
        g = Goo(count=30, across=6, seed=3, weight=None, projection=0.5)
        g.rule, g.drive, g.read = "reinforce", "rate", "count"
        assert fast.compare(g, epochs=40, teacher=Teacher(g, seed=7, rule="reinforce", eligibility="hebb", target="copy",
                                                          homeostasis=0.01, unstick=0.1)) == []


def test_copy_runs_on_goo_and_on_the_grid_and_the_direct_flag_is_gone(capsys):
    from walnutbutter.problems import PROBLEMS
    assert not hasattr(PROBLEMS["copy"], "direct_projection")
    assert cli_main(["--problem", "copy", "--goo", "--headless", "--epochs", "2", "--seed", "1", "--no-save"]) == 0
    assert "44 neurons in neither zone" in capsys.readouterr().err
    assert cli_main(["--problem", "copy", "--headless", "--epochs", "2", "--seed", "1", "--no-save"]) == 0  # the grid may
    with pytest.raises(SystemExit):
        cli_main(["--problem", "copy", "--goo", "--direct-projection", "--headless", "--epochs", "2", "--no-save"])


# --- the command line, under the rule ---------------------------------------------

def test_the_command_line_builds_goo_learns_and_checkpoints(tmp_path, capsys):
    save = tmp_path / "goo.json"
    assert cli_main(["--goo", "--headless", "--epochs", "20", "--seed", "1", "--save-weights", str(save)]) == 0
    err = capsys.readouterr().err
    assert "projections at P 0.2; 8 in, 8 out)" in err
    assert "input permutation: place i along the input zone" in err  # not "the bottom row": goo has none
    assert "omega" not in err  # omega does not reach goo, so it is not reported as if it had
    data = json.loads(save.read_text())
    assert data["container"] == "goo" and data["count"] == 60 and data["projection"] == 0.2 and data["epoch"] == 20
    assert cli_main(["--load-weights", str(save), "--headless", "--epochs", "5", "--no-save"]) == 0


def test_the_command_line_sizes_goo_sets_its_projection_and_refuses_one_with_no_interior(capsys):
    assert cli_main(["--goo", "24", "--projection", "1", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "24 neurons, 552 projections at P 1" in capsys.readouterr().err  # 24 x 23: every ordered pair
    assert cli_main(["--goo", "24", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "projections at P 0.2" in capsys.readouterr().err  # the default
    assert cli_main(["--goo", "24", "--projection", "0.5", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "at P 0.5" in capsys.readouterr().err
    for bad in ("15", "8", "4"):
        assert cli_main(["--goo", bad, "--headless", "--epochs", "2", "--no-save"]) == 2
        assert "would overlap" in capsys.readouterr().err
    assert cli_main(["--goo", "--projection", "1.5", "--headless", "--epochs", "2", "--no-save"]) == 2
    assert "--projection must be in (0, 1]" in capsys.readouterr().err


def test_the_command_line_reports_the_scaling_and_can_turn_it_off(capsys):
    assert cli_main(["--goo", "--projection", "1", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    err = capsys.readouterr().err
    assert ("fan-in scaling (§5.2): an interior neuron hears 59 synapses and starts at threshold 0.656, floor -2.622; "
            "a zone neuron hears 59 and starts at 0.656, -2.622") in err
    assert cli_main(["--goo", "--no-scale-with-fan-in", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "fan-in scaling off: a flat threshold 0.2 and floor -0.8" in capsys.readouterr().err


def test_a_seed_batch_runs_goo_and_its_header_says_what_ran(capsys):
    assert cli_main(["--goo", "20", "--seeds", "2", "--seed", "1", "--epochs", "5", "--no-save"]) == 0
    err = capsys.readouterr().err
    assert ("20 neurons of goo at projection 0.2, 8 in and 8 out, fan-in scaled, reinforce rule with the hazard eligibility, "
            "escape delta 0.455") in err  # the default eligibility follows the neuron (§1.3)
    assert cli_main(["--goo", "20", "--projection", "0.5", "--no-scale-with-fan-in", "--eligibility", "hebb", "--seeds", "2",
                     "--seed", "1", "--epochs", "5", "--no-save"]) == 0
    assert "goo at projection 0.5, 8 in and 8 out, flat threshold and floor, reinforce rule with the hebb eligibility" in capsys.readouterr().err


def test_the_two_earlier_wirings_are_kept_and_the_uniform_rule_hears_equally_for_free():
    """§3.4: the zone rule of September 14 and its equal fan-in of September 15 are kept for their checkpoints only."""
    import statistics as st
    uniform = Goo(count=300, across=60, outputs=20, seed=5, weight=None, projection=0.2)
    hears = lambda g, ns: st.mean(len(n.incoming) for n in ns)
    assert abs(hears(uniform, uniform.interior()) - 0.2 * 299) / (0.2 * 299) < 0.1
    assert abs(hears(uniform, uniform.input_row() + uniform.output_row()) - 0.2 * 299) / (0.2 * 299) < 0.1
    assert not uniform.zones_are_apart() and uniform.zone_projection() == 0.2
    equal = Goo(count=300, across=60, outputs=20, seed=5, weight=None, projection=0.2, wiring="zones-equal")
    assert equal.zones_are_apart() and equal.zone_projection() == pytest.approx(0.2 * 299 / 220)
    assert abs(hears(equal, equal.input_row() + equal.output_row()) - hears(equal, equal.interior())) / hears(equal, equal.interior()) < 0.1
    zones = Goo(count=300, across=60, outputs=20, seed=5, weight=None, projection=0.2, wiring="zones")
    assert zones.zones_are_apart() and zones.zone_projection() == 0.2
    assert abs(hears(zones, zones.output_row()) - 0.2 * 220) / (0.2 * 220) < 0.1  # P times the interior
    assert len({len(g.connections) for g in (uniform, equal, zones)}) == 3  # three wirings of one seed


def test_a_checkpoint_keeps_its_wiring_and_older_ones_restore_under_theirs(tmp_path):
    import json
    for wiring_name in ("uniform", "zones", "zones-equal"):
        goo = Goo(count=40, across=8, outputs=4, seed=3, weight=None, projection=0.3, wiring=wiring_name)
        run_epoch(goo, verbose=False)
        data = checkpoint(goo, tmp_path / f"{wiring_name}.json")
        assert data["wiring"] == wiring_name
        back, _ = restore(tmp_path / f"{wiring_name}.json")
        assert back.wiring == wiring_name and wiring(back) == wiring(goo)
    # before the uniform rule a checkpoint carried no wiring: equal_fan_in told the two zone wirings apart
    for wiring_name, field in (("zones-equal", {"equal_fan_in": True}), ("zones", {})):
        old = json.loads((tmp_path / f"{wiring_name}.json").read_text()); del old["wiring"]; old.update(field)
        (tmp_path / "before.json").write_text(json.dumps(old))
        before, _ = restore(tmp_path / "before.json")
        assert before.wiring == wiring_name
        assert wiring(before) == wiring(Goo(count=40, across=8, outputs=4, seed=3, weight=None, projection=0.3, wiring=wiring_name))
