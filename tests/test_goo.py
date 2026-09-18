"""Goo: the only container, with the plane taken away (AUTHORITY.md §4.1)."""

import json
import pathlib

import pytest

from walnutbutter.cli import build_parser, cli_main
from walnutbutter.constants import ACROSS, ESCAPE_DELTA, GOO_COUNT, GOO_MINIMUM_POTENTIAL, GOO_THRESHOLD, THRESHOLD_FAN_IN, WEIGHT_RANGE
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
    assert data["container"] == "goo" and data["count"] == 24
    back, _ = restore("goo.json")
    assert isinstance(back, Goo) and wiring(back) == wiring(goo)
    assert [back.connections[i].weight for i in range(1, 25)] == [goo.connections[i].weight for i in range(1, 25)]
    assert [n.potential for n in back.all_neurons()] == [n.potential for n in goo.all_neurons()]


def test_goo_restores_without_a_seed_because_no_draw_chose_its_wiring():
    """Unlike a grid, whose shortcuts cannot be rebuilt without the stream that picked them."""
    goo = Goo(count=12, across=4, seed=None, weight=None, projection=1.0, wiring="zones-equal")  # at P 1 no draw chose the wiring
    run_epoch(goo, verbose=False)
    checkpoint(goo, "seedless.json")
    back, _ = restore("seedless.json")
    assert [c.weight for c in back.connections.values()] == [c.weight for c in goo.connections.values()]
    checkpoint(Goo(count=12, across=4, seed=None, weight=None), "drawn.json")  # the scaled rule draws, so it needs its seed
    with pytest.raises(ValueError, match="at scaling factor 0.05 without a seed"):
        restore("drawn.json")


# --- the command line -----------------------------------------------------



def test_goo_is_headless(capsys):
    """§4.1: goo has no positions, so there is nothing to draw and every run is headless."""
    assert cli_main(["--goo", "--epochs", "2", "--seed", "1", "--no-save"]) == 0  # no --headless, and no window opens





def test_the_scaling_can_be_switched_off_and_then_goo_is_flat():
    goo = Goo(seed=1, scale_with_fan_in=False)
    assert not goo.scale_with_fan_in_on
    assert {n.threshold for n in goo.all_neurons()} == {GOO_THRESHOLD} == {0.2}  # goo's own, not the grid's 0.25
    assert {n.minimum_potential for n in goo.all_neurons()} == {GOO_MINIMUM_POTENTIAL} and GOO_MINIMUM_POTENTIAL == pytest.approx(-0.8)




def test_both_engines_see_the_scaled_axis():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork
    mesh = Goo(count=30, across=6, seed=2, weight=None)
    net = ArrayNetwork(mesh)
    assert net.threshold_v.tolist() == [n.threshold for n in mesh.all_neurons()]
    assert net.floor.tolist() == [n.minimum_potential for n in mesh.all_neurons()]




def test_the_count_read_counts_the_epochs_spikes_and_takes_the_pickiness_as_its_line():
    """§5.10: the read is the count itself; §9.5 puts the line at ROW_CRITIC_PICKINESS_IN_SPIKES, an integer."""
    from walnutbutter.constants import ROW_CRITIC_PICKINESS_IN_SPIKES
    goo = Goo(count=20, across=4, seed=1)
    goo.read = "count"
    assert goo.pickiness == ROW_CRITIC_PICKINESS_IN_SPIKES == 2  # spikes, not hertz: the epoch's length does not enter
    run_epoch(goo, bits=[True, False], verbose=False)
    for neuron, n, on in zip(goo.output_row(), goo.output_counts(), goo.output_fired()):
        assert n == neuron.epoch_spikes and on == (n >= 2)
    a, b = goo.output_row()[0], goo.output_row()[1]
    a.spikes_at_reset, a.spikes = 10, 11  # one spike: below the line
    b.spikes_at_reset, b.spikes = 10, 12  # two: on it
    assert goo.output_fired()[:2] == [False, True]
    goo.pickiness = 1  # one spike now reads on
    assert goo.output_fired()[:2] == [True, True]
    goo.pickiness = ROW_CRITIC_PICKINESS_IN_SPIKES
    a.spikes = 10  # back to none for a and two for b: the read the levels below are asserted against
    assert goo.output_levels()[:2] == [0.0, 1.0]  # a bit, as the row critic wants it
    goo.reset()
    assert a.epoch_spikes == b.epoch_spikes == 0  # the next epoch starts its count afresh


def test_the_count_read_is_the_same_in_all_three_engines():
    pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter import fast
    from walnutbutter.learning import Teacher
    mesh, twin = Goo(count=30, across=6, seed=5, weight=None), Goo(count=30, across=6, seed=5, weight=None)
    for g in (mesh, twin):
        g.read, g.rule, g.drive = "count", "reinforce", "rate"
    net = ArrayNetwork(twin)
    for _ in range(20):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert net.output_counts_hz() == mesh.output_counts_hz() and net.output_fired() == mesh.output_fired()
    if fast.available():
        goo = Goo(count=30, across=6, seed=5, weight=None)
        goo.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
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
    assert cli_main(["--problem", "copy", "--goo", "--pickiness", "3", "--headless", "--epochs", "3", "--seed", "1",
                     "--save-weights", str(save)]) == 0
    assert json.loads(save.read_text())["pickiness"] == 3
    assert cli_main(["--load-weights", str(save), "--headless", "--epochs", "2", "--no-save"]) == 0



# --- no direct projection (AUTHORITY.md §3.4, Byron, September 14, 2026) -------------


# --- the zone rule (AUTHORITY.md §3.4, Byron, September 14, 2026) ---------------------

def test_the_default_goo_is_sixty_neurons_at_scaling_factor_a_twentieth_by_the_scaled_rule():
    """§3.4: the scaled rule (Byron, September 16, 2026): every neuron hears N s synapses in expectation, no input another input."""
    from walnutbutter.constants import GOO_SCALING_FACTOR
    goo = Goo(seed=1)
    assert DEFAULT_COUNT == GOO_COUNT == 60 and len(goo) == goo.count == 60
    assert goo.scaling_factor == GOO_SCALING_FACTOR == 0.05 and goo.wiring == "scaled" and goo.expected_fan_in() == 3.0
    assert goo.input_projection() == pytest.approx(3 / 44) and goo.output_projection() == pytest.approx(3 / 52)
    assert goo.hidden_projection() == pytest.approx(3 / 59)  # an input hears the hidden, an output the inputs and hidden
    assert 130 < len(goo.connections) < 240  # three synapses a neuron in expectation, sixty neurons: 180, give or take
    assert goo.input_zone_is_apart() and goo.outputs_are_apart() and not goo.zones_are_apart()  # inputs onto outputs is allowed
    assert any(c.source in goo.input_row() for n in goo.output_row() for c in n.incoming)  # an output hears the inputs directly
    assert not any(c.source in goo.output_row() for n in goo.output_row() for c in n.incoming)  # and never the other outputs
    assert not any(c.source in goo.output_row() for n in goo.input_row() for c in n.incoming)  # and no input hears an output
    assert all(goo.zone_of(goo.neurons.index(c.target)) == "hidden" for n in goo.output_row() for c in n.outgoing)
    assert len(goo.interior()) == 44 and len(goo.zone_indices()) == 16
    assert repr(goo).startswith("Goo(60 neurons, ") and repr(goo).endswith(" projections at scaling factor 0.05; 8 in, 44 hidden, 8 out, zones apart, inputs onto outputs)")
    open_ = Goo(seed=1, wiring="scaled-open")  # the night's first scaled rule: the outputs open to every zone
    assert open_.input_projection() == pytest.approx(3 / 52) and open_.hidden_projection() == open_.output_projection() == pytest.approx(3 / 59)
    assert open_.input_zone_is_apart() and not open_.outputs_are_apart() and repr(open_).endswith("8 out, input zone apart)")
    assert any(c.source in open_.output_row() for n in open_.input_row() for c in n.incoming)
    feedforward = Goo(count=16, across=8, seed=1, weight=None)  # no hidden: inputs -> outputs and nothing else
    assert feedforward.interior() == [] and feedforward.input_projection() == 0.0 and feedforward.output_projection() == pytest.approx(0.8 / 8)
    assert all(not n.incoming for n in feedforward.input_row()) and all(c.source in feedforward.input_row() for n in feedforward.output_row() for c in n.incoming)
    assert all(n.threshold == GOO_THRESHOLD for n in feedforward.input_row())  # hearing nothing, left at the container's threshold (§5.2)
    with pytest.raises(ValueError, match="no one zone probability"):
        goo.zone_projection()
    assert Goo(count=ACROSS * 10, seed=1).count == 80  # the grid's eighty is still one --goo away
    full = Goo(seed=1, projection=1.0, wiring="zones-equal")  # the fully connected goo of the zone rule: every pair with an interior end
    assert len(full.connections) == 60 * 59 - 16 * 15 == 3300 and full.mean_out_degree() == pytest.approx(3300 / 60)
    assert all(len(n.incoming) == 59 for n in full.interior())  # an interior neuron hears everyone
    assert all(len(n.incoming) == 44 for n in full.input_row() + full.output_row())  # a zone neuron hears the interior
    assert repr(full) == "Goo(60 neurons, 3300 projections at P 1; 8 in, 44 hidden, 8 out, zones apart)"


def test_every_pair_with_an_interior_end_projects_once_each_way_and_the_zones_never_do():
    goo = Goo(count=5, across=2, seed=1, projection=1.0, wiring="zones-equal")  # zones {0, 1} and {3, 4}; the interior is neuron 2
    pairs = wiring(goo)
    assert len(pairs) == len(set(pairs)) == 8
    assert set(pairs) == {(f"Goo_2", f"Goo_{k}") for k in (0, 1, 3, 4)} | {(f"Goo_{k}", "Goo_2") for k in (0, 1, 3, 4)}
    assert [c.id for c in goo.connections.values()] == list(range(1, 9))  # ids contiguous, source-major
    uniform = Goo(count=5, across=2, seed=1, projection=1.0, wiring="uniform")  # the evening's rule: every ordered pair
    assert len(wiring(uniform)) == 20 and not uniform.zones_are_apart()


def test_at_projection_one_the_count_fixes_the_topology_and_below_it_the_seed_does():
    zone = dict(wiring="zones-equal")
    one, two = Goo(count=6, across=2, seed=1, weight=None, projection=1.0, **zone), Goo(count=6, across=2, seed=2, weight=None, projection=1.0, **zone)
    assert wiring(one) == wiring(two)  # P 1: nothing drawn for the topology
    weights = lambda g: [g.connections[i].weight for i in range(1, len(g.connections) + 1)]
    assert weights(one) != weights(two) and weights(one) == weights(Goo(count=6, across=2, seed=1, weight=None, projection=1.0, **zone))
    a, b, c = (Goo(count=30, across=6, seed=s, weight=None, projection=0.5, **zone) for s in (1, 1, 2))
    assert wiring(a) == wiring(b) and wiring(a) != wiring(c)  # below 1 the seed decides, and repeats itself
    assert a.zones_are_apart() and c.zones_are_apart()
    assert "at P 0.5" in repr(a)
    x, y, z = (Goo(count=30, across=6, seed=s, weight=None) for s in (1, 1, 2))  # the scaled rule always draws
    assert wiring(x) == wiring(y) and wiring(x) != wiring(z)
    assert Goo(count=30, across=6, seed=1, weight=None, scaling_factor=0.1).expected_fan_in() == 3.0  # and its knob is the fan-in


def test_there_are_never_any_shortcuts_because_there_is_no_neighbourhood_to_get_past():
    goo = Goo(count=12, across=4, seed=1, projection=1.0, wiring="zones-equal")
    assert goo.small_world_connections() == [] and goo.omega == 0.0
    assert goo.local_connections() == list(goo.connections.values())
    assert len(goo.connections_of_kind("goo")) == 12 * 11 - 8 * 7 == 76
    assert goo.connections_of_kind("small_world") == []


def test_a_weight_is_fixed_or_drawn_in_connection_id_order_inside_the_range():
    assert {c.weight for c in Goo(count=5, across=2, weight=0.3, projection=1.0, wiring="zones-equal").connections.values()} == {0.3}  # 8 projections, always
    low, high = WEIGHT_RANGE
    drawn = Goo(count=24, seed=3, weight=None)
    assert all(low <= c.weight <= high for c in drawn.connections.values())
    narrow = Goo(count=24, seed=3, weight=None, weight_range=(0.2, 0.3))
    assert all(0.2 <= c.weight <= 0.3 for c in narrow.connections.values())
    assert narrow.clip_weight(5.0) == 0.3 and narrow.clip_weight(-5.0) == 0.2


def test_the_zones_go_by_index_because_there_are_no_places_to_go_by():
    goo = Goo(count=20, across=4, seed=1)
    assert goo.rows == 2  # it is counting the two zones, not any depth
    assert [n.name for n in goo.input_row()] == ["Goo_0", "Goo_1", "Goo_2", "Goo_3"]
    assert [n.name for n in goo.output_row()] == ["Goo_16", "Goo_17", "Goo_18", "Goo_19"]
    assert goo.get_neuron_at(0, 1) is goo[0] and goo.get_neuron_at(0, 0) is goo[16]
    assert goo.input_width() == 4 and len(goo.interior()) == 12
    assert goo.zone_indices() == {0, 1, 2, 3, 16, 17, 18, 19}


def test_goo_refuses_overlapping_zones_a_zone_wiring_with_no_interior_and_knobs_off_their_ranges():
    """Under the zone wirings the zones talk only through the interior, so there has to be one; the scaled rule needs none."""
    for wiring_name in ("zones-equal", "zones"):
        for count in (16, 12, 8, 7):
            with pytest.raises(ValueError, match="needs an interior"):
                Goo(count=count, across=8, wiring=wiring_name)
        assert Goo(count=17, across=8, seed=1, wiring=wiring_name).interior()[0].name == "Goo_8"  # one interior neuron is enough
    assert Goo(count=16, across=8, seed=1).interior() == []  # the scaled rule: the outputs hear the inputs directly
    assert Goo(count=16, across=8, seed=1, wiring="uniform").interior() == []  # as the evening's rule did
    for wiring_name in ("scaled", "uniform"):
        with pytest.raises(ValueError, match="would overlap"):
            Goo(count=15, across=8, wiring=wiring_name)
    with pytest.raises(ValueError, match="unknown wiring"):
        Goo(count=20, across=4, wiring="mesh")
    with pytest.raises(ValueError, match="at least one neuron wide"):
        Goo(count=10, across=0)
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="projection probability"):
            Goo(count=20, across=4, projection=bad)
    for bad in (0.0, -0.05):
        with pytest.raises(ValueError, match="scaling factor must be positive"):
            Goo(count=20, across=4, scaling_factor=bad)


def test_the_zones_never_project_onto_each_other_and_at_projection_one_everything_else_does():
    goo = Goo(count=40, across=8, seed=1, projection=1.0, wiring="zones-equal")
    ins, outs, inner = goo.input_row(), goo.output_row(), goo.interior()
    zone = set(ins) | set(outs)
    for n in zone:
        assert not any(c.target in zone for c in n.outgoing)  # not in -> out, not out -> in, not within a zone
    for n in inner:
        assert {c.target for c in n.outgoing} == (set(goo.all_neurons()) - {n})  # an interior neuron reaches everyone
        assert {c.source for c in n.incoming} == (set(goo.all_neurons()) - {n})
    assert goo.zones_are_apart()


def test_goo_rescales_its_potential_axis_by_each_neurons_own_fan_in():
    goo = Goo(seed=1, projection=1.0, wiring="zones-equal")  # under the zone rule at P 1 the fan-in is exact: 59 and 44
    inner, edge = goo.interior()[0], goo.input_row()[0]
    assert inner.threshold == pytest.approx(GOO_THRESHOLD * 59 / THRESHOLD_FAN_IN) == pytest.approx(0.656, abs=1e-3)
    assert edge.threshold == pytest.approx(GOO_THRESHOLD * 44 / THRESHOLD_FAN_IN) == pytest.approx(0.489, abs=1e-3)
    assert all(n.minimum_potential / n.threshold == pytest.approx(-4.0) for n in goo.all_neurons())  # one axis
    assert goo.fan_in_scale() == pytest.approx(59 / THRESHOLD_FAN_IN)  # the interior's stretch at projection 1
    sparse = Goo(seed=1)  # at the default scaling the fan-in is the seed's, and each neuron is scaled by its own
    assert all(n.threshold == pytest.approx(GOO_THRESHOLD * len(n.incoming) / THRESHOLD_FAN_IN) for n in sparse.all_neurons() if n.incoming)
    deaf = [n for n in sparse.all_neurons() if not n.incoming]  # and one that hears nothing is left at the quoted axis (§5.2)
    assert deaf and all(n.threshold == GOO_THRESHOLD and n.minimum_potential == GOO_MINIMUM_POTENTIAL for n in deaf)
    flat = Goo(seed=1, scale_with_fan_in=False)
    assert {n.threshold for n in flat.all_neurons()} == {GOO_THRESHOLD} == {0.2}
    assert {n.minimum_potential for n in flat.all_neurons()} == {GOO_MINIMUM_POTENTIAL}


def test_a_neuron_hearing_exactly_the_reference_fan_in_is_left_where_the_grid_starts():
    """§5.2: THRESHOLD and MINIMUM_POTENTIAL are quoted at THRESHOLD_FAN_IN incoming synapses, so that fan-in is a fixed point."""
    goo = Goo(count=int(THRESHOLD_FAN_IN) + 1, across=4, seed=1, threshold=0.25, minimum_potential=-1.0, projection=1.0, wiring="zones-equal")
    for n in goo.interior():  # an interior neuron of 19 hears 18
        assert n.threshold == pytest.approx(0.25) and n.minimum_potential == pytest.approx(-1.0)
    for n in goo.input_row():  # a zone neuron hears the 11 of the interior
        assert n.threshold == pytest.approx(0.25 * 11 / THRESHOLD_FAN_IN)


def test_below_projection_one_a_goo_round_trips_with_its_seed_and_refuses_without():
    pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter import fast
    from walnutbutter.learning import Teacher
    zone = dict(wiring="zones-equal")
    goo = Goo(count=30, across=6, seed=3, weight=None, projection=0.5, **zone)
    goo.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
    run_epoch(goo, verbose=False)
    data = checkpoint(goo, "half.json")
    assert data["projection"] == 0.5 and data["connections"] == len(goo.connections)
    back, _ = restore("half.json")
    assert wiring(back) == wiring(goo) and back.projection == 0.5 and back.wiring == "zones-equal" and back.zones_are_apart()
    with pytest.raises(ValueError, match="at projection 0.5 without a seed"):
        checkpoint(Goo(count=30, across=6, seed=None, weight=None, projection=0.5, **zone), "seedless.json")
        restore("seedless.json")
    old = json.loads(pathlib.Path("half.json").read_text()); old["direct_projection"] = False
    pathlib.Path("old.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="before the zone rule"):
        restore("old.json")
    # every engine runs the drawn wiring the same way
    mesh, twin = (Goo(count=30, across=6, seed=3, weight=None, projection=0.5, **zone) for _ in range(2))
    net = ArrayNetwork(twin)
    for _ in range(10):
        run_epoch(mesh, verbose=False); run_epoch(net, verbose=False)
        assert net.output_fired() == mesh.output_fired()
    if fast.available():
        g = Goo(count=30, across=6, seed=3, weight=None, projection=0.5, **zone)
        g.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
        g.rule, g.drive, g.read = "reinforce", "rate", "count"
        assert fast.compare(g, epochs=40, teacher=Teacher(g, seed=7, rule="reinforce", eligibility="hebb", target="copy",
                                                          homeostasis=0.01, unstick=0.1)) == []


def test_copy_runs_on_goo_and_on_the_grid_and_the_direct_flag_is_gone(capsys):
    from walnutbutter.problems import PROBLEMS
    assert not hasattr(PROBLEMS["copy"], "direct_projection")
    assert cli_main(["--problem", "copy", "--goo", "--headless", "--epochs", "2", "--seed", "1", "--no-save"]) == 0
    assert ("every neuron hears 3 synapses in expectation where it has sources: an input from the 44 hidden at P 0.0682, an "
            "output from the 52 inputs and hidden at 0.0577, a hidden neuron from the 59 others at 0.0508; the outputs "
            "project onto the hidden alone") in capsys.readouterr().err
    assert cli_main(["--problem", "copy", "--headless", "--epochs", "2", "--seed", "1", "--no-save"]) == 0  # the grid may
    with pytest.raises(SystemExit):
        cli_main(["--problem", "copy", "--goo", "--direct-projection", "--headless", "--epochs", "2", "--no-save"])


# --- the command line, under the rule ---------------------------------------------

def test_the_command_line_builds_goo_learns_and_checkpoints(tmp_path, capsys):
    save = tmp_path / "goo.json"
    assert cli_main(["--goo", "--headless", "--epochs", "20", "--seed", "1", "--save-weights", str(save)]) == 0
    err = capsys.readouterr().err
    assert "projections at scaling factor 0.05; 8 in, 44 hidden, 8 out, zones apart, inputs onto outputs)" in err
    assert "omega" not in err  # omega does not reach goo, so it is not reported as if it had
    data = json.loads(save.read_text())
    assert data["container"] == "goo" and data["count"] == 60 and data["epoch"] == 20
    assert data["wiring"] == "scaled" and data["scaling_factor"] == 0.05 and data["projection"] == 0.2  # the earlier wirings' knob, recorded still
    assert cli_main(["--load-weights", str(save), "--headless", "--epochs", "5", "--no-save"]) == 0


def test_the_command_line_sizes_goo_sets_its_wiring_and_refuses_what_a_wiring_cannot_build(capsys):
    assert cli_main(["--goo", "24", "--wiring", "zones-equal", "--projection", "1", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "24 neurons, 312 projections at P 1" in capsys.readouterr().err  # 24 x 23 less 16 x 15
    assert cli_main(["--goo", "24", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "projections at scaling factor 0.05" in capsys.readouterr().err  # the default
    assert cli_main(["--goo", "24", "--scaling-factor", "0.2", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "at scaling factor 0.2; 8 in, 8 hidden, 8 out, zones apart, inputs onto outputs): " in capsys.readouterr().err
    assert cli_main(["--goo", "24", "--wiring", "zones", "--projection", "0.5", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "at P 0.5; 8 in, 8 hidden, 8 out, zones apart): " in capsys.readouterr().err
    assert cli_main(["--goo", "16", "--headless", "--epochs", "2", "--seed", "1", "--no-save"]) == 0  # the scaled rule needs no interior
    assert "16 neurons" in capsys.readouterr().err
    for bad in ("16", "8", "4"):
        assert cli_main(["--goo", bad, "--wiring", "zones-equal", "--headless", "--epochs", "2", "--no-save"]) == 2
        assert "needs an interior" in capsys.readouterr().err
    for bad in ("8", "4"):
        assert cli_main(["--goo", bad, "--headless", "--epochs", "2", "--no-save"]) == 2
        assert "would overlap" in capsys.readouterr().err
    assert cli_main(["--goo", "--projection", "1.5", "--headless", "--epochs", "2", "--no-save"]) == 2
    assert "--projection must be in (0, 1]" in capsys.readouterr().err
    assert cli_main(["--goo", "--scaling-factor", "0", "--headless", "--epochs", "2", "--no-save"]) == 2
    assert "--scaling-factor must be positive" in capsys.readouterr().err
    # --hidden-neurons sizes the goo as inputs + hidden + outputs (§8), and 0 builds under the scaled rule
    assert cli_main(["--hidden-neurons", "4", "--headless", "--epochs", "2", "--seed", "1", "--no-save"]) == 0
    assert "Goo(20 neurons" in capsys.readouterr().err
    assert cli_main(["--hidden-neurons", "0", "--headless", "--epochs", "2", "--seed", "1", "--no-save"]) == 0
    assert "8 in, 0 hidden, 8 out, zones apart, inputs onto outputs)" in capsys.readouterr().err
    assert cli_main(["--goo", "24", "--hidden-neurons", "4", "--headless", "--epochs", "2", "--no-save"]) == 2
    assert "is a goo of 20, not --goo 24" in capsys.readouterr().err


def test_the_command_line_reports_the_scaling_and_can_turn_it_off(capsys):
    assert cli_main(["--goo", "--wiring", "zones-equal", "--projection", "1", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    err = capsys.readouterr().err
    assert ("fan-in scaling (§5.2): an interior neuron hears 59 synapses and starts at threshold 0.656, floor -2.622; "
            "a zone neuron hears 44 and starts at 0.489, -1.956") in err
    assert cli_main(["--goo", "--no-scale-with-fan-in", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "fan-in scaling off: a flat threshold 0.2 and floor -0.8" in capsys.readouterr().err


def test_a_seed_batch_runs_goo_and_its_header_says_what_ran(capsys):
    assert cli_main(["--goo", "20", "--seeds", "2", "--seed", "1", "--epochs", "5", "--no-save"]) == 0
    err = capsys.readouterr().err
    assert ("20 neurons of goo at scaling factor 0.05, 8 in, 4 hidden and 8 out, fan-in scaled, reinforce rule with the hazard "
            "eligibility, escape delta 0.455") in err  # the default eligibility follows the neuron (§1.3)
    assert cli_main(["--goo", "20", "--wiring", "zones-equal", "--projection", "0.5", "--no-scale-with-fan-in", "--eligibility", "hebb",
                     "--seeds", "2", "--seed", "1", "--epochs", "5", "--no-save"]) == 0
    assert ("goo at projection 0.5 under the zones-equal wiring, 8 in, 4 hidden and 8 out, flat threshold and floor, reinforce "
            "rule with the hebb eligibility") in capsys.readouterr().err


def test_the_rule_hears_n_s_everywhere_and_the_three_earlier_wirings_are_kept():
    """§3.4: the scaled rule is the rule; the zone rule with and without equal fan-in and the uniform one restore checkpoints."""
    import statistics as st
    hears = lambda g, ns: st.mean(len(n.incoming) for n in ns)
    scaled = Goo(count=300, across=60, outputs=20, seed=5, weight=None)
    assert scaled.wiring == "scaled" and scaled.expected_fan_in() == 15.0 and scaled.input_projection() == pytest.approx(15 / 220)
    assert scaled.output_projection() == pytest.approx(15 / 280) and scaled.hidden_projection() == pytest.approx(15 / 299)
    assert scaled.input_zone_is_apart() and scaled.outputs_are_apart() and not scaled.zones_are_apart()
    for group in (scaled.input_row(), scaled.output_row(), scaled.interior()):
        assert abs(hears(scaled, group) - 15.0) / 15.0 < 0.1  # every zone hears N s
    assert any(c.source in scaled.input_row() for n in scaled.output_row() for c in n.incoming)  # the outputs hear the inputs
    uniform = Goo(count=300, across=60, outputs=20, seed=5, weight=None, projection=0.2, wiring="uniform")
    assert abs(hears(uniform, uniform.interior()) - 0.2 * 299) / (0.2 * 299) < 0.1
    assert abs(hears(uniform, uniform.input_row() + uniform.output_row()) - 0.2 * 299) / (0.2 * 299) < 0.1
    assert not uniform.zones_are_apart() and uniform.zone_projection() == 0.2
    equal = Goo(count=300, across=60, outputs=20, seed=5, weight=None, projection=0.2, wiring="zones-equal")
    assert equal.wiring == "zones-equal" and equal.zones_are_apart() and equal.zone_projection() == pytest.approx(0.2 * 299 / 220)
    assert abs(hears(equal, equal.input_row() + equal.output_row()) - hears(equal, equal.interior())) / hears(equal, equal.interior()) < 0.1
    zones = Goo(count=300, across=60, outputs=20, seed=5, weight=None, projection=0.2, wiring="zones")
    assert zones.zones_are_apart() and zones.zone_projection() == 0.2
    assert abs(hears(zones, zones.output_row()) - 0.2 * 220) / (0.2 * 220) < 0.1  # P times the interior
    opened = Goo(count=300, across=60, outputs=20, seed=5, weight=None, wiring="scaled-open")
    assert len({len(g.connections) for g in (scaled, opened, uniform, equal, zones)}) == 5  # five wirings of one seed


def test_a_checkpoint_keeps_its_wiring_and_older_ones_restore_under_theirs(tmp_path):
    import json
    for wiring_name in ("scaled", "scaled-open", "zones-equal", "zones", "uniform"):
        goo = Goo(count=40, across=8, outputs=4, seed=3, weight=None, projection=0.3, scaling_factor=0.1, wiring=wiring_name)
        run_epoch(goo, verbose=False)
        data = checkpoint(goo, tmp_path / f"{wiring_name}.json")
        assert data["wiring"] == wiring_name and data["scaling_factor"] == 0.1 and data["projection"] == 0.3
        back, _ = restore(tmp_path / f"{wiring_name}.json")
        assert back.wiring == wiring_name and back.scaling_factor == 0.1 and wiring(back) == wiring(goo)
    # before the uniform rule a checkpoint carried no wiring: equal_fan_in told the two zone wirings apart
    for wiring_name, field in (("zones-equal", {"equal_fan_in": True}), ("zones", {})):
        old = json.loads((tmp_path / f"{wiring_name}.json").read_text()); del old["wiring"]; old.update(field)
        (tmp_path / "before.json").write_text(json.dumps(old))
        before, _ = restore(tmp_path / "before.json")
        assert before.wiring == wiring_name
        assert wiring(before) == wiring(Goo(count=40, across=8, outputs=4, seed=3, weight=None, projection=0.3, wiring=wiring_name))


def test_ff2_wires_every_input_onto_every_output_and_nothing_else():
    """§3.4 (Byron, September 16, 2026: "P_ij = P(i is in the input layer and j is in the output layer) ... for the feedforward
    problem I want everything to connect fully"): two layers, fully connected, one way; no hidden neurons under it."""
    g = Goo(count=10, across=6, outputs=4, seed=2, weight=None, wiring="ff2")
    ins, outs = set(g.input_row()), set(g.output_row())
    assert len(g.connections) == 6 * 4 and all(c.source in ins and c.target in outs for c in g.connections.values())
    assert {(c.source, c.target) for c in g.connections.values()} == {(i, o) for i in ins for o in outs}  # each pair exactly once
    assert all(len(n.incoming) == 6 for n in outs) and all(len(n.incoming) == 0 for n in ins)
    assert g.expected_fan_in() == 6.0 and g.output_projection() == 1.0 and g.input_projection() == 0.0 and g.hidden_projection() == 0.0
    assert g.input_zone_is_apart() and g.outputs_are_apart() and g.interior() == []
    assert repr(g) == "Goo(10 neurons, 24 projections, fully connected feedforward; 6 in, 0 hidden, 4 out)"
    from walnutbutter.constants import GOO_THRESHOLD, THRESHOLD_FAN_IN
    assert all(n.threshold == pytest.approx(GOO_THRESHOLD * 6 / THRESHOLD_FAN_IN) for n in outs)  # the axis scales with the whole input zone
    assert all(n.threshold == GOO_THRESHOLD for n in ins)  # hearing nothing: the quoted threshold (§5.2)
    with pytest.raises(ValueError, match="two layers"):
        Goo(count=12, across=6, outputs=4, wiring="ff2")
    with pytest.raises(ValueError, match="no one zone probability"):
        g.zone_projection()
    same = Goo(count=10, across=6, outputs=4, seed=2, weight=None, wiring="ff2")
    assert [c.weight for c in same.connections.values()] == [c.weight for c in g.connections.values()]  # the seed's weights
    # the checkpoint restores under its own wiring, and the engines agree, learning on, escape noise on
    from walnutbutter import fast
    from walnutbutter.learning import Teacher
    from walnutbutter.persistence import checkpoint, restore
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "ff2.json"
        checkpoint(g, path)
        back, data = restore(path)
        assert data["wiring"] == "ff2" and back.wiring == "ff2" and len(back.connections) == 24
        assert [c.weight for c in back.connections.values()] == [c.weight for c in g.connections.values()]
    mesh, twin = (Goo(count=16, across=10, outputs=6, seed=5, weight=None, wiring="ff2") for _ in range(2))
    for x in (mesh, twin):
        x.rule, x.drive, x.read = "reinforce", "rate", "count"
        x.set_delta(0.455)
    from walnutbutter.arrays import ArrayNetwork
    net = ArrayNetwork(twin)
    a = Teacher(mesh, seed=7, rule="reinforce", eligibility="hebb", target="copy", homeostasis=0.0, unstick=0.0)
    b = Teacher(net, seed=7, rule="reinforce", eligibility="hebb", target="copy", homeostasis=0.0, unstick=0.0)
    for _ in range(20):
        assert a.epoch(verbose=False) == b.epoch(verbose=False)
        assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
    if fast.available():
        g3 = Goo(count=16, across=10, outputs=6, seed=5, weight=None, wiring="ff2")
        g3.rule, g3.drive, g3.read = "reinforce", "rate", "count"
        g3.set_delta(0.455)
        assert fast.compare(g3, epochs=40, teacher=Teacher(g3, seed=7, rule="reinforce", eligibility="hazard", target="copy",
                                                          homeostasis=0.0, unstick=0.0)) == []
    assert cli_main(["--goo", "16", "--wiring", "ff2", "--seeds", "1", "--seed", "1", "--epochs", "2", "--no-save"]) == 0  # 8 in, 8 out
    with pytest.raises(ValueError, match="two layers"):
        cli_main(["--goo", "20", "--wiring", "ff2", "--seeds", "1", "--seed", "1", "--epochs", "2", "--no-save"])  # a hidden zone of 4: refused


def test_ff2_partial_draws_each_input_output_pair_at_p_and_nothing_else():
    """§3.4 (Byron, September 17, 2026: "P(neuron i projects onto neuron j) = P iff i in inputs, j in outputs"): ff2 with P,
    every pair its own draw; P = 1 is ff2 to the bit; no hidden neurons under it."""
    from walnutbutter import fast
    from walnutbutter.learning import Teacher
    g = Goo(count=60, across=40, outputs=20, seed=2, weight=None, wiring="ff2-partial", projection=0.5)
    ins, outs = set(g.input_row()), set(g.output_row())
    assert all(c.source in ins and c.target in outs for c in g.connections.values())
    pairs = {(c.source, c.target) for c in g.connections.values()}
    assert len(pairs) == len(g.connections) and 0.35 * 800 < len(pairs) < 0.65 * 800  # about half of the 800 pairs, each at most once
    assert g.expected_fan_in() == 20.0 and g.output_projection() == 0.5 and g.input_projection() == 0.0 and g.hidden_projection() == 0.0
    assert g.projection_probability(0, 59) == 0.5 and g.projection_probability(59, 0) == 0.0 and g.projection_probability(0, 1) == 0.0
    assert repr(g) == f"Goo(60 neurons, {len(g.connections)} projections at P 0.5, two layers; 40 in, 0 hidden, 20 out)"
    full, one = (Goo(count=20, across=12, outputs=8, seed=3, weight=None, wiring=w, projection=1.0) for w in ("ff2", "ff2-partial"))
    assert [(c.source.name, c.target.name, c.weight) for c in one.connections.values()] == \
           [(c.source.name, c.target.name, c.weight) for c in full.connections.values()]  # P = 1 is ff2 to the bit
    with pytest.raises(ValueError, match="two layers"):
        Goo(count=24, across=12, outputs=8, wiring="ff2-partial", projection=0.5)
    with pytest.raises(ValueError, match="no one zone probability"):
        g.zone_projection()
    import tempfile, pathlib
    from walnutbutter.persistence import checkpoint, restore
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "p.json"
        checkpoint(g, path)
        back, data = restore(path)
        assert data["wiring"] == "ff2-partial" and data["projection"] == 0.5 and back.wiring == "ff2-partial" and back.projection == 0.5
        assert [c.weight for c in back.connections.values()] == [c.weight for c in g.connections.values()]
    if fast.available():
        g3 = Goo(count=24, across=16, outputs=8, seed=5, weight=None, wiring="ff2-partial", projection=0.5)
        g3.rule, g3.drive, g3.read = "reinforce", "rate", "count"
        g3.set_delta(0.455)
        assert fast.compare(g3, epochs=40, teacher=Teacher(g3, seed=7, rule="reinforce", eligibility="hazard", target="copy",
                                                          homeostasis=0.0, unstick=0.0)) == []
    assert cli_main(["--goo", "16", "--wiring", "ff2-partial", "--projection", "0.5", "--seeds", "1", "--seed", "1", "--epochs", "2", "--no-save"]) == 0
