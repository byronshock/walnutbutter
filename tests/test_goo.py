"""Goo: the fourth container, with the plane taken away (AUTHORITY.md §3.4)."""

import json

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

def test_the_default_goo_is_sixty_neurons_fully_connected():
    goo = Goo(seed=1)
    assert DEFAULT_COUNT == GOO_COUNT == 60  # the working network (§1.2, Byron, September 14, 2026)
    assert len(goo) == goo.count == 60 and len(goo.connections) == 60 * 59 == 3540
    assert goo.mean_out_degree() == 59.0  # every neuron to every other, and nothing left over
    assert all(len(n.incoming) == 59 for n in goo.all_neurons())
    assert repr(goo) == "Goo(60 neurons fully connected, 3540 connections; 8 in, 8 out)"
    assert Goo(count=ACROSS * 10, seed=1).count == 80  # the grid's eighty is still one --goo away"


def test_every_ordered_pair_connects_once_each_way_and_nothing_connects_to_itself():
    goo = Goo(count=5, across=2, seed=1)
    pairs = wiring(goo)
    assert len(pairs) == len(set(pairs)) == 20  # 5 x 4: no pair made twice
    assert all(source != target for source, target in pairs)
    assert all(goo[i].connection_to(goo[j]) is not None for i in range(5) for j in range(5) if i != j)
    assert all(goo[i].connection_to(goo[i]) is None for i in range(5))


def test_the_count_alone_fixes_the_topology_and_the_seed_only_moves_the_weights():
    """§3.4: nothing is drawn to decide which pairs connect, so two seeds wire identically."""
    one, two = Goo(count=6, across=2, seed=1, weight=None), Goo(count=6, across=2, seed=2, weight=None)
    assert wiring(one) == wiring(two)
    assert wiring(one) == [(f"Goo_{s}", f"Goo_{t}") for s in range(6) for t in range(6) if s != t]  # source then target
    weights = lambda g: [g.connections[i].weight for i in range(1, len(g.connections) + 1)]
    assert weights(one) != weights(two)
    assert weights(one) == weights(Goo(count=6, across=2, seed=1, weight=None))  # and a seed repeats itself


def test_there_are_never_any_shortcuts_because_there_is_no_neighbourhood_to_get_past():
    goo = Goo(count=12, across=4, seed=1)
    assert goo.small_world_connections() == [] and goo.omega == 0.0
    assert goo.local_connections() == list(goo.connections.values())
    assert len(goo.connections_of_kind("goo")) == 12 * 11
    assert goo.connections_of_kind("small_world") == []


def test_a_weight_is_fixed_or_drawn_in_connection_id_order_inside_the_range():
    assert {c.weight for c in Goo(count=5, across=2, weight=0.3).connections.values()} == {0.3}
    low, high = WEIGHT_RANGE
    drawn = Goo(count=8, seed=3, weight=None)
    assert all(low <= c.weight <= high for c in drawn.connections.values())
    narrow = Goo(count=8, seed=3, weight=None, weight_range=(0.2, 0.3))
    assert all(0.2 <= c.weight <= 0.3 for c in narrow.connections.values())
    assert narrow.clip_weight(5.0) == 0.3 and narrow.clip_weight(-5.0) == 0.2


# --- the zones ------------------------------------------------------------

def test_the_zones_go_by_index_because_there_are_no_places_to_go_by():
    goo = Goo(count=20, across=4, seed=1, permute=False)
    assert goo.rows == 2  # it is counting the two zones, not any depth
    assert [n.name for n in goo.input_row()] == ["Goo_0", "Goo_1", "Goo_2", "Goo_3"]
    assert [n.name for n in goo.output_row()] == ["Goo_16", "Goo_17", "Goo_18", "Goo_19"]
    assert goo.get_neuron_at(0, 1) is goo[0] and goo.get_neuron_at(0, 0) is goo[16]
    assert goo.input_width() == 4 and not goo.zones_overlap()


def test_nothing_between_the_zones_is_addressable_which_is_what_makes_it_goo():
    goo = Goo(count=20, across=4, seed=1)
    assert goo.get_neuron_at(0, 2) is None and goo.get_neuron_at(0, -1) is None
    assert goo.get_neuron_at(4, 0) is None and goo.get_neuron_at(-1, 1) is None


def test_small_goo_overlaps_its_zones_on_purpose_rather_than_refusing():
    """§2: a neuron may sit in several zones at once; structures are permissive."""
    tight = Goo(count=ACROSS, across=ACROSS, seed=1, permute=False)
    assert tight.zones_overlap() and tight.input_row() == tight.output_row()  # it reads what it writes
    assert "overlapping" in repr(tight)
    partial = Goo(count=12, across=8, seed=1, permute=False)
    assert partial.zones_overlap() and partial.input_row()[4] is partial.output_row()[0]


def test_goo_refuses_only_the_zone_it_could_not_fit():
    with pytest.raises(ValueError, match="at least as many neurons"):
        Goo(count=7, across=8)
    with pytest.raises(ValueError, match="at least one neuron wide"):
        Goo(count=10, across=0)


def test_the_permutation_scrambles_the_coded_bits_across_the_input_zone():
    goo = Goo(count=20, across=4, seed=1)
    assert sorted(goo.permutation) == [0, 1, 2, 3] and goo.permutation != [0, 1, 2, 3]
    assert Goo(count=20, across=4, seed=1, permute=False).permutation == [0, 1, 2, 3]


# --- what it is for -------------------------------------------------------

def test_the_output_zone_is_one_hop_from_the_input_zone():
    """§3.4: goo has no far. Every input neuron projects straight onto every output neuron."""
    goo = Goo(count=40, across=8, seed=1)
    assert all(source.connection_to(target) is not None
               for source in goo.input_row() for target in goo.output_row())


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
    goo = Goo(count=12, across=4, seed=None, weight=None)
    run_epoch(goo, verbose=False)
    checkpoint(goo, "seedless.json")
    back, _ = restore("seedless.json")
    assert [c.weight for c in back.connections.values()] == [c.weight for c in goo.connections.values()]


# --- the command line -----------------------------------------------------

def test_the_command_line_builds_goo_learns_and_checkpoints(tmp_path, capsys):
    save = tmp_path / "goo.json"
    assert cli_main(["--goo", "--headless", "--epochs", "20", "--seed", "1", "--save-weights", str(save)]) == 0
    err = capsys.readouterr().err
    assert "60 neurons fully connected, 3540 connections" in err
    assert "input permutation: place i along the input zone" in err  # not "the bottom row": goo has none
    assert "omega" not in err  # omega does not reach goo, so it is not reported as if it had
    data = json.loads(save.read_text())
    assert data["container"] == "goo" and data["count"] == 60 and data["epoch"] == 20
    assert cli_main(["--load-weights", str(save), "--headless", "--epochs", "5", "--no-save"]) == 0


def test_the_command_line_sizes_goo_and_says_when_the_zones_overlap(capsys):
    assert cli_main(["--goo", "16", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "16 neurons fully connected, 240 connections" in capsys.readouterr().err
    assert cli_main(["--goo", "8", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "the input and output zones overlap" in capsys.readouterr().err


def test_goo_is_headless_and_has_no_picture_to_save(capsys):
    assert cli_main(["--goo", "--epochs", "2", "--seed", "1", "--no-save"]) == 0  # no --headless, and no window opens
    assert cli_main(["--goo", "--save", "goo.png", "--epochs", "2", "--no-save"]) == 2
    assert "no geometry to draw" in capsys.readouterr().err


def test_the_command_line_refuses_goo_smaller_than_its_zones(capsys):
    assert cli_main(["--goo", "4", "--headless", "--epochs", "2", "--no-save"]) == 2
    assert "at least as many neurons as the zones are wide" in capsys.readouterr().err


def test_a_seed_batch_runs_goo(capsys):
    assert cli_main(["--goo", "20", "--seeds", "2", "--seed", "1", "--epochs", "10", "--no-save"]) == 0
    assert "20 neurons of fully connected goo, 8 in and 8 out" in capsys.readouterr().err


def test_goo_is_a_container_of_its_own_and_will_not_be_mixed_with_another(capsys):
    assert cli_main(["--goo", "--nodes", "--headless", "--epochs", "2", "--no-save"]) == 2
    assert "cannot be combined with --nodes" in capsys.readouterr().err
    assert cli_main(["--goo", "--layers", "3", "--headless", "--epochs", "2", "--no-save"]) == 2
    assert "cannot be combined with --layers" in capsys.readouterr().err


# --- the potential axis scales with fan-in (AUTHORITY.md §5.2) ------------

def test_goo_rescales_its_potential_axis_by_fan_in_and_keeps_the_grids_ratio():
    goo = Goo(seed=1)
    scale = 59 / THRESHOLD_FAN_IN  # in-degree 59 over the in-degree the thresholds are quoted at
    assert goo.fan_in_scale() == pytest.approx(scale) and goo.scale_with_fan_in_on
    thetas = {n.threshold for n in goo.all_neurons()}
    floors = {n.minimum_potential for n in goo.all_neurons()}
    assert len(thetas) == len(floors) == 1  # homogeneous: every goo neuron has the same fan-in
    assert thetas.pop() == pytest.approx(GOO_THRESHOLD * scale) and floors.pop() == pytest.approx(GOO_MINIMUM_POTENTIAL * scale)
    assert GOO_THRESHOLD * scale == pytest.approx(3.278, abs=1e-3)  # what a goo neuron starts at since September 14, 2026
    # the floor is not a second decision: it is the same axis, so the ratio is the grid's -4 exactly
    assert all(n.minimum_potential / n.threshold == pytest.approx(-4.0) for n in goo.all_neurons())


def test_the_scaling_can_be_switched_off_and_then_goo_is_flat():
    goo = Goo(seed=1, scale_with_fan_in=False)
    assert not goo.scale_with_fan_in_on
    assert {n.threshold for n in goo.all_neurons()} == {GOO_THRESHOLD} == {1.0}  # goo's own, not the grid's 0.25
    assert {n.minimum_potential for n in goo.all_neurons()} == {GOO_MINIMUM_POTENTIAL} == {-4.0}


def test_a_neuron_wired_like_an_interior_grid_cell_is_left_exactly_where_it_was():
    """§5.2: THRESHOLD and MINIMUM_POTENTIAL are quoted at THRESHOLD_FAN_IN, so that fan-in is a fixed point."""
    goo = Goo(count=int(THRESHOLD_FAN_IN) + 1, across=4, seed=1, threshold=0.25, minimum_potential=-1.0)  # at the grid's constants
    assert goo.fan_in_scale() == pytest.approx(1.0)  # in-degree exactly THRESHOLD_FAN_IN
    assert all(n.threshold == pytest.approx(0.25) for n in goo.all_neurons())
    assert all(n.minimum_potential == pytest.approx(-1.0) for n in goo.all_neurons())


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


def test_the_command_line_reports_the_scaling_and_can_turn_it_off(capsys):
    assert cli_main(["--goo", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "fan-in scaling (§5.2): x3.28 on the potential axis, so threshold 3.278 and floor -13.111" in capsys.readouterr().err
    assert cli_main(["--goo", "--no-scale-with-fan-in", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "fan-in scaling off: a flat threshold 1 and floor -4" in capsys.readouterr().err  # goo's own constants


def test_any_container_can_be_asked_to_scale_from_the_command_line(capsys):
    assert cli_main(["--scale-with-fan-in", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "every threshold and floor rescaled by in-degree / 18" in capsys.readouterr().err


def test_a_seed_batchs_header_says_which_axis_the_arm_ran_on(capsys):
    """A sweep's own log should identify its arm without cross-referencing the command that made it."""
    assert cli_main(["--goo", "20", "--seeds", "2", "--seed", "1", "--epochs", "5", "--no-save"]) == 0
    err = capsys.readouterr().err
    assert "fully connected goo, 8 in and 8 out, fan-in scaled, reinforce rule with the perturb eligibility" in err
    assert cli_main(["--goo", "20", "--no-scale-with-fan-in", "--eligibility", "hebb", "--seeds", "2", "--seed", "1",
                     "--epochs", "5", "--no-save"]) == 0
    assert "flat threshold and floor, reinforce rule with the hebb eligibility" in capsys.readouterr().err
    assert cli_main(["--seeds", "2", "--seed", "1", "--epochs", "5", "--no-save"]) == 0
    assert "hex grid, omega 0.2, flat threshold and floor" in capsys.readouterr().err



# --- the count read (AUTHORITY.md §4.3): count, estimate a rate, threshold it ----------

def test_the_count_read_counts_the_epochs_spikes_and_thresholds_the_rate():
    from walnutbutter.constants import TEACHER_THRESHOLD
    goo = Goo(count=20, across=4, seed=1, permute=False)
    goo.read = "count"
    assert goo.teacher_threshold == TEACHER_THRESHOLD == 40.0
    run_epoch(goo, bits=[True, False], verbose=False)
    per_ms = 1000.0 / goo.interval
    for neuron, hz, on in zip(goo.output_row(), goo.output_counts_hz(), goo.output_fired()):
        assert hz == neuron.epoch_spikes * per_ms and on == (hz >= 40.0)
    # at 35 ms one spike is 28.6 Hz and reads off; two are 57 and read on: a stray background spike is not a one
    a, b = goo.output_row()[0], goo.output_row()[1]
    a.spikes_at_reset, a.spikes = 10, 11
    b.spikes_at_reset, b.spikes = 10, 12
    assert goo.output_fired()[:2] == [False, True]
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

def test_goo_can_be_built_with_no_synapse_from_the_input_zone_to_the_output_zone():
    """A copy then has to go through the interior: at least two hops, not the one synapse from input i to output i."""
    direct, cut = Goo(seed=1), Goo(seed=1, direct=False)
    assert direct.projects_directly() and not cut.projects_directly()
    assert len(cut.connections) == len(direct.connections) - 8 * 8 == 3540 - 64  # exactly the input x output block
    outputs = set(cut.output_row())
    for n in cut.input_row():
        assert not any(c.target in outputs for c in n.outgoing)
    assert all(any(c.target in set(cut.input_row()) for c in n.outgoing) for n in cut.output_row())  # the way back stays
    assert all(len(n.incoming) == 59 - 8 for n in cut.output_row())  # an output hears 51, not 59
    assert all(len(n.incoming) == 59 for n in cut.all_neurons() if n not in outputs)  # nobody else is touched
    assert [c.id for c in cut.connections.values()] == list(range(1, len(cut.connections) + 1))  # ids stay contiguous
    assert "no direct projection" in repr(cut) and "no direct projection" not in repr(direct)
    # §5.2 scales each neuron by its own in-degree: the outputs, hearing 8 fewer synapses, start lower; nobody else moves
    assert all(n.threshold == pytest.approx(GOO_THRESHOLD * 51 / THRESHOLD_FAN_IN) for n in cut.output_row())
    assert all(n.threshold == pytest.approx(GOO_THRESHOLD * 59 / THRESHOLD_FAN_IN) for n in cut.all_neurons() if n not in outputs)
    assert all(n.minimum_potential / n.threshold == pytest.approx(-4.0) for n in cut.all_neurons())  # the ratio holds throughout


def test_no_direct_projection_round_trips_and_runs_on_every_engine(tmp_path):
    pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter import fast
    from walnutbutter.learning import Teacher
    goo = Goo(count=30, across=6, seed=3, weight=None, direct=False)
    run_epoch(goo, verbose=False)
    data = checkpoint(goo, tmp_path / "cut.json")
    assert data["direct_projection"] is False and data["connections"] == 30 * 29 - 36
    back, _ = restore(tmp_path / "cut.json")
    assert not back.direct and not back.projects_directly() and len(back.connections) == data["connections"]
    mesh, twin = Goo(count=30, across=6, seed=3, weight=None, direct=False), Goo(count=30, across=6, seed=3, weight=None, direct=False)
    net = ArrayNetwork(twin)
    for _ in range(10):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert net.output_fired() == mesh.output_fired()
    if fast.available():
        g = Goo(count=30, across=6, seed=3, weight=None, direct=False)
        g.rule, g.drive, g.read = "reinforce", "rate", "count"
        assert fast.compare(g, epochs=30, teacher=Teacher(g, seed=7, rule="reinforce", eligibility="hebb", target="copy",
                                                          homeostasis=0.01, unstick=0.1)) == []


def test_copy_asks_for_no_direct_projection_and_the_flag_overrides(capsys):
    from walnutbutter.problems import PROBLEMS
    from walnutbutter.cli import apply_problem
    assert PROBLEMS["copy"].direct_projection is False and PROBLEMS["reversal"].direct_projection is True
    args = build_parser().parse_args(["--problem", "copy"]); apply_problem(args)
    assert args.direct_projection is False
    args = build_parser().parse_args(["--problem", "copy", "--direct-projection"]); apply_problem(args)
    assert args.direct_projection is True
    assert cli_main(["--problem", "copy", "--goo", "--headless", "--epochs", "2", "--seed", "1", "--no-save"]) == 0
    assert "none from the input zone to the output zone" in capsys.readouterr().err
    assert cli_main(["--problem", "copy", "--headless", "--epochs", "2", "--seed", "1", "--no-save"]) == 2  # the grid cannot
    assert "only goo can be built with no direct projection" in capsys.readouterr().err
    assert cli_main(["--problem", "copy", "--direct-projection", "--headless", "--epochs", "2", "--seed", "1", "--no-save"]) == 0
