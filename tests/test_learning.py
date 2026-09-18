import random
import statistics

import pytest

from walnutbutter.constants import ESCAPE_DELTA
from walnutbutter import learning
from walnutbutter.goo import Goo
from walnutbutter.learning import (
    Teacher,
    accuracy,
    delivered_connections,
    delivered_signals,
    landed,
    expected_outputs,
    output_errors,
    output_row,
    reinforce,
)
from walnutbutter.monitor import main, run_epoch
from walnutbutter.neuron import Neuron


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def test_output_row_is_the_output_zone_by_place():
    """§4.3: the output zone is the last `outputs` neurons in index order, read by place."""
    grid = Goo(count=24, across=6)
    row = output_row(grid)
    assert row == [grid.get_neuron_at(place, 0) for place in range(6)]
    assert row == list(grid.all_neurons())[-6:]


def test_expected_outputs_for_each_target():
    grid = Goo(count=24, across=6)
    grid.set_input([True, True, False, False, False, True])
    assert expected_outputs(grid, "reversed") == [True, False, False, False, True, True]
    assert expected_outputs(grid, "copy") == [True, True, False, False, False, True]
    assert expected_outputs(grid, "all-off") == [False] * 6
    assert expected_outputs(grid, "all-on") == [True] * 6
    with pytest.raises(ValueError):
        expected_outputs(Goo(count=24, across=6))


def test_errors_and_accuracy_against_the_top_row():
    grid = Goo(count=12, across=4)
    grid.set_input([True, False, False, False])  # reversed target: only place 3 should fire
    top = output_row(grid)
    top[3].fire()  # correct
    top[0].fire()  # should not have
    errors = output_errors(grid, "reversed")
    assert [errors[n] for n in top] == [-1, 0, 0, 0]
    assert accuracy(grid, "reversed") == 0.75


def test_run_epoch_noise_gives_every_neuron_a_remembered_starting_potential():
    grid = Goo(count=24, across=6, weight=None, seed=1)
    run_epoch(grid, verbose=False, noise=0.1, rng=random.Random(1))
    noises = [n.noise for n in grid.all_neurons()]
    # 24 draws of standard deviation 0.1 have a standard error of 0.02 on their mean, so the bound has to be
    # several of those or it fails whenever an unrelated change shifts which draws are taken
    assert len(set(noises)) > 1 and abs(statistics.mean(noises)) < 0.10
    assert 0.05 < statistics.pstdev(noises) < 0.15
    forced = grid.waves[0].fired[0]  # the input neurons whose coded bit is 1 are the forced ones
    assert forced.has_fired and forced.forced
    run_epoch(grid, verbose=False)  # without noise everything starts from 0 again
    assert all(n.noise == 0.0 for n in grid.all_neurons())


def test_delivered_connections_are_those_whose_source_fired():
    grid = main(count=32, across=8, weight=None, seed=1)
    delivered = delivered_connections(grid)
    assert delivered
    assert all(c.source.has_fired for c in delivered)
    in_flight = {c for _, c in grid.schedule.pending()}  # a source that fired just before the horizon: its signals wait for the next epoch
    assert all(c in delivered or c in in_flight for c in grid.connections.values() if c.source.has_fired and c.is_active)


def test_reinforce_moves_delivered_weights_by_advantage_times_noise():
    grid = Goo(count=32, across=8, weight=None, seed=2)
    grid.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
    run_epoch(grid, verbose=False, noise=0.1, rng=random.Random(2))
    before = {c.id: c.weight for c in grid.connections.values()}
    changed = reinforce(grid, advantage=0.5, lr=0.01, sigma=0.1)
    assert changed > 0
    for c in grid.connections.values():
        if c.target.forced or c not in delivered_connections(grid):
            assert c.weight == before[c.id]  # forced inputs and idle connections untouched
        else:
            expected = max(-1.0, min(1.0, before[c.id] + 0.01 * 0.5 * c.target.noise / 0.1))
            assert c.weight == pytest.approx(expected)


def test_ignoring_late_signals_skips_those_that_arrived_after_their_target_fired():
    grid = Goo(count=32, across=8, weight=None, seed=2)
    grid.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
    run_epoch(grid, verbose=False, noise=0.1, rng=random.Random(2))
    before = {c.id: c.weight for c in grid.connections.values()}
    last = {s.connection: s for s in delivered_signals(grid)}  # the rule judges a connection by its last delivery of the epoch
    counted = {c for c, s in last.items() if landed(s) and not s.target.forced}
    late = {c for c, s in last.items() if not landed(s)}
    assert counted and late  # some signals arrived too late to count
    reinforce(grid, advantage=0.5, lr=0.01, sigma=0.1, late="ignore")
    for c in grid.connections.values():
        if c not in counted:
            assert c.weight == before[c.id]  # forced inputs, idle connections and late signals untouched
        else:
            expected = max(-1.0, min(1.0, before[c.id] + 0.01 * 0.5 * c.target.noise / 0.1))
            assert c.weight == pytest.approx(expected)


def test_reinforce_with_zero_advantage_changes_nothing():
    grid = main(count=32, across=8, weight=None, seed=3)
    grid.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
    before = [c.weight for c in grid.connections.values()]
    assert reinforce(grid, advantage=0.0) == 0
    assert [c.weight for c in grid.connections.values()] == before


def test_the_wrong_hebb_eligibility_uses_target_firing_and_keeps_weights_in_range():
    grid = main(count=32, across=8, weight=None, seed=4)
    grid.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
    before = {c.id: c.weight for c in grid.connections.values()}
    reinforce(grid, advantage=-1.0, lr=0.5, eligibility="wrong_hebb")
    for c in delivered_connections(grid):
        if c.target.forced:
            continue
        if c.target.has_fired:
            assert c.weight <= before[c.id]  # negative advantage x positive eligibility
        else:
            assert c.weight >= before[c.id]
        assert -1.0 <= c.weight <= 1.0
    with pytest.raises(ValueError):
        reinforce(grid, 1.0, eligibility="magic")


def test_the_reinforce_rule_still_runs_and_moves_weights():
    """The pre-alpha's rule, factored out behind rule="reinforce": it runs, scores and moves weights. Under the
    schedule the mesh reverberates on its own, so this is kept for comparison, not as a performance claim."""
    grid = Goo(count=32, across=8, weight=None, seed=2)
    grid.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
    before = [c.weight for c in grid.connections.values()]
    teacher = Teacher(grid, target="all-off", lr=0.1, seed=2, rule="reinforce")
    rewards = [teacher.epoch(verbose=False) for _ in range(200)]
    assert all(0.0 <= r <= 1.0 for r in rewards) and grid.dopamine is None
    assert [c.weight for c in grid.connections.values()] != before
    assert all(-1.0 <= c.weight <= 1.0 for c in grid.connections.values())


def test_teacher_validates_tracks_and_reports():
    grid = main(count=32, across=8, seed=1)
    grid.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
    with pytest.raises(ValueError):
        Teacher(grid, target="upside-down")
    with pytest.raises(ValueError):
        Teacher(grid, eligibility="magic")
    with pytest.raises(ValueError):
        Teacher(grid, lr=-1)
    teacher = Teacher(grid, target="reversed", lr=0.01, window=10, seed=1, rule="reinforce")
    assert "no epochs yet" in teacher.status()
    first = teacher.step()
    assert teacher.epochs == 1 and teacher.last_reward == first == teacher.average == teacher.baseline
    second = teacher.epoch(verbose=False)
    assert teacher.epochs == 2 and 0 <= teacher.average <= 1 and 0 <= second <= 1
    assert grid.epoch == 2
    assert "learning reversed (hazard, lr 0.01, sigma 0, homeostasis 1e-06 toward 0.5, unstick 0.001): accuracy" in teacher.status()
    for hebbian in ("wrong_hebb", "hebb", "count_hebb"):
        assert Teacher(grid, eligibility=hebbian).sigma == 0.0  # no injected noise for any Hebbian variant
    assert grid.tally and not grid.centred  # the last Teacher, count_hebb's, switched the tally of what each synapse delivers on (§6.7)
    Teacher(grid, eligibility="hebb")
    assert grid.centred and not grid.tally and all(n.centred and n.traced for n in grid.all_neurons())  # the single-spike rule


def test_teacher_with_seed_is_reproducible():
    def run(seed):
        grid = Goo(count=32, across=8, weight=None, seed=1)
        teacher = Teacher(grid, target="copy", seed=seed)
        for _ in range(30):
            teacher.epoch(verbose=False)
        return [c.weight for c in grid.connections.values()]

    assert run(5) == run(5)
    assert run(5) != run(6)


def test_targets_registry_has_the_builtin_targets():
    assert set(learning.TARGETS) == {"reversed", "copy", "complement", "all-off", "all-on", "label"}
    assert learning.TARGETS["complement"]([True, False]) == [False, True]  # shallow_not (§8)


def test_accuracy_to_date_is_the_mean_over_all_epochs():
    grid = main(count=32, across=8, seed=1)
    teacher = Teacher(grid, seed=1)
    assert teacher.accuracy_to_date is None
    rewards = [teacher.step()] + [teacher.epoch(verbose=False) for _ in range(9)]
    assert teacher.accuracy_to_date == pytest.approx(sum(rewards) / 10)
    assert "to date over 10 epochs" in teacher.status()


def test_reinforce_clips_to_the_grid_weight_range():
    grid = Goo(count=32, across=8, weight=None, seed=2, weight_range=(0.001, 1.0), threshold=2.0)
    grid.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
    run_epoch(grid, verbose=False, noise=0.1, rng=random.Random(2))
    reinforce(grid, advantage=-1.0, lr=50.0)  # a huge negative push: everything touched should hit the floor, not go negative
    touched = [c for c in delivered_connections(grid) if not c.target.forced and c.target.noise > 0]
    assert touched
    assert all(c.weight == 0.001 for c in touched)
    assert all(c.weight >= 0.001 for c in grid.connections.values())


def test_rates_track_firing_and_stuck_neurons_are_counted():
    from walnutbutter.learning import stuck_neurons, update_rates, RATE_MEMORY
    grid = Goo(count=12, across=4)
    always, never = grid.get_neuron_at(0, 0), grid.get_neuron_at(1, 0)
    for _ in range(600):
        grid.reset()
        always.fire(wave=1)  # fired by the network, not forced
        update_rates(grid)
    assert always.rate > 0.99 and never.rate < 0.01
    on, off = stuck_neurons(grid)
    assert always in on and never in off
    assert abs((0.5 + RATE_MEMORY * 0.5) - 0.505) < 1e-12  # one step from the initial 0.5 toward 1


def test_forced_neurons_are_left_out_of_rates_and_homeostasis_but_unforced_inputs_are_not():
    from walnutbutter.learning import forced, homeostasis, update_rates
    grid = Goo(count=12, across=4, seed=1)
    # the forced drive: one stimulus at the input's moment, so a bit-1 input is forced for certain. Under the Poisson
    # drive an input that keeps itself firing is refractory whenever its own stimuli land and is never marked forced --
    # which is the rule (§4.3, §5.3), and made this test fail one run in five (found September 16, 2026)
    grid.drive = "forced"
    grid.set_input([True, False, True, False])
    grid.fire_input()
    row = grid.input_row()
    forced_input, unforced_input = row[0], row[1]
    assert forced(forced_input) and not forced(unforced_input)
    rates_before = {n: n.rate for n in grid.all_neurons()}
    update_rates(grid)
    assert forced_input.rate == rates_before[forced_input]  # its firing was not the network's doing
    assert unforced_input.rate != rates_before[unforced_input]  # an ordinary neuron: its rate moved
    forced_input.rate, unforced_input.rate = 1.0, 1.0
    thresholds = {n: n.threshold for n in grid.all_neurons()}
    moved = homeostasis(grid, rate=0.1, target=0.5)
    assert moved == len(grid.neurons) - len(grid.input_neurons())  # everyone except the two forced this epoch
    assert forced_input.threshold == thresholds[forced_input]
    assert unforced_input.threshold == pytest.approx(thresholds[unforced_input] + 0.05)


def test_homeostasis_moves_thresholds_toward_the_target_rate_and_nothing_clips_them():
    from walnutbutter.learning import homeostasis
    grid = Goo(count=12, across=4)
    hot, cold = grid.get_neuron_at(0, 0), grid.get_neuron_at(1, 0)
    hot.rate, cold.rate = 1.0, 0.0
    before = {n: n.threshold for n in grid.all_neurons()}
    assert homeostasis(grid, rate=0.1, target=0.5) == 12  # nothing has been forced: every neuron is eligible
    assert hot.threshold == pytest.approx(before[hot] + 0.05)
    assert cold.threshold == pytest.approx(before[cold] - 0.05)
    hot.threshold, cold.threshold = before[hot], before[cold]
    homeostasis(grid, rate=0.1)  # the default target is 0.5
    assert hot.threshold == pytest.approx(before[hot] + 0.05)
    assert cold.threshold == pytest.approx(before[cold] - 0.05)
    hot.threshold, cold.threshold = before[hot], before[cold]
    homeostasis(grid, rate=0.1, target=0.5)
    assert hot.threshold == pytest.approx(before[hot] + 0.05)
    assert cold.threshold == pytest.approx(before[cold] - 0.05)
    assert homeostasis(grid, rate=0.0) == 0
    for _ in range(500):
        homeostasis(grid, rate=1.0)
    # no range: the [-5, 5] clamp was eliminated as artificial (Byron, September 14, 2026)
    # (the last 0.1 step above was not reset, so the loop began 0.05 from `before`)
    assert hot.threshold == pytest.approx(before[hot] + 0.05 + 250.0) and cold.threshold == pytest.approx(before[cold] - 0.05 - 250.0)


def test_teacher_homeostasis_reduces_stuck_neurons():
    from walnutbutter.learning import stuck_neurons
    def run(homeostasis):
        grid = Goo(count=48, across=8, weight=None, seed=1)
        # un-sticking off: since September 14, 2026 it reaches every neuron and would do this job on its own
        teacher = Teacher(grid, seed=1, homeostasis=homeostasis, target_rate=0.5, unstick=0.0)
        for _ in range(3000):
            teacher.epoch(verbose=False)
        on, off = stuck_neurons(grid)
        return len(on) + len(off)
    assert run(0.0) > run(0.01)


def test_teacher_validates_homeostasis_and_reports_it():
    grid = main(count=32, across=8, seed=1)
    with pytest.raises(ValueError):
        Teacher(grid, homeostasis=-0.1)
    with pytest.raises(ValueError):
        Teacher(grid, target_rate=1.5)
    teacher = Teacher(grid, homeostasis=0.01, target_rate=0.4, seed=1)
    teacher.step()
    assert "homeostasis 0.01 toward 0.4" in teacher.status() and "stuck" not in teacher.status()
    default = Teacher(grid, seed=1)
    default.step()
    assert "homeostasis 1e-06 toward 0.5" in default.status() and "sigma 0.1" in default.status()
    assert default.homeostasis == 1e-6 and default.target_rate == 0.5
    off = Teacher(grid, seed=1, homeostasis=0)
    off.step()
    assert "homeostasis" not in off.status()


def test_there_is_no_threshold_range_anywhere():
    """Byron, September 14, 2026: eliminate threshold clipping, it's artificial. A threshold goes where the rules take it."""
    from walnutbutter.learning import homeostasis
    grid = Goo(count=12, across=4)
    hot = grid.get_neuron_at(0, 0)
    started = hot.threshold  # goo starts each neuron at its own fan-in-scaled threshold (§5.2)
    hot.rate = 1.0
    for _ in range(200):
        homeostasis(grid, rate=1.0)
    assert hot.threshold == pytest.approx(started + 100.0)  # far past the 5 that once stopped it
    with pytest.raises(TypeError):
        homeostasis(grid, rate=1.0, threshold_range=(0.0, 1.5))
    with pytest.raises(TypeError):
        Teacher(grid, threshold_range=(0, 2), seed=1)
    teacher = Teacher(grid, seed=1, homeostasis=0.01)
    teacher.epoch(verbose=False)
    assert not hasattr(teacher, "threshold_range") and "in [" not in teacher.status()


def test_the_leak_is_the_default_and_discharge_is_available():
    grid = Goo(count=12, across=4, weight=None, seed=1)
    default = Teacher(grid, seed=1)
    assert default.discharge is False
    default.epoch(verbose=False)
    assert "discharge" not in default.status()
    discharging = Teacher(grid, seed=1, discharge=True)
    discharging.epoch(verbose=False)
    assert "discharge" in discharging.status()


def test_unstick_touches_every_stuck_neuron_wherever_it_sits():
    """It was the output row only until September 14, 2026 (AUTHORITY.md §6.7): now a stuck interior neuron is nudged too."""
    from walnutbutter.learning import unstick
    grid = Goo(count=24, across=6)
    outputs = output_row(grid)
    hot, cold, fine = outputs[0], outputs[1], outputs[2]
    hot.rate, cold.rate, fine.rate = 1.0, 0.0, 0.5
    interior = grid.get_neuron_at(3, 1)
    interior.rate = 1.0  # stuck, and not an output: nudged all the same
    before = {n: n.threshold for n in grid.all_neurons()}
    nudged = unstick(grid, rate=0.1)
    assert set(nudged) == {hot, cold, interior}
    assert hot.threshold == pytest.approx(before[hot] + 0.05)
    assert cold.threshold == pytest.approx(before[cold] - 0.05)
    assert interior.threshold == pytest.approx(before[interior] + 0.05)
    assert fine.threshold == before[fine]
    assert unstick(grid, rate=0.0) == []


def test_unstick_stops_once_the_neuron_is_no_longer_stuck_and_nothing_clips_it():
    from walnutbutter.learning import unstick
    grid = Goo(count=24, across=6)
    hot = output_row(grid)[0]
    started = hot.threshold
    hot.rate = 1.0
    for _ in range(300):
        unstick(grid, rate=1.0)
    assert hot.threshold == pytest.approx(started + 150.0)  # no range to stop at
    hot.rate = 0.6  # out of the stuck band: nothing more happens
    assert unstick(grid, rate=1.0) == []
    assert hot.threshold == pytest.approx(started + 150.0)  # and stays there: nothing more happened


def test_teacher_applies_unsticking_and_reports_it():
    grid = main(count=32, across=8, weight=1.0, seed=1)  # weight 1: every output fires every epoch
    teacher = Teacher(grid, seed=1, unstick=0.01, homeostasis=0)
    for _ in range(600):
        teacher.epoch(verbose=False)
    assert teacher.unstuck_count > 0
    assert any(n.threshold > 0.25 for n in output_row(grid))
    assert "unstick 0.01" in teacher.status()
    off = Teacher(grid, seed=1, unstick=0)
    off.step()
    assert "unstick" not in off.status() and off.unstuck_count == 0
    with pytest.raises(ValueError):
        Teacher(grid, unstick=-1)
    with pytest.raises(ValueError):
        Teacher(grid, unstick_target=0)


def test_unstick_defaults_on_at_one_thousandth():
    teacher = Teacher(main(count=32, across=8, seed=1))
    assert teacher.unstick == 1e-3 and teacher.unstick_target == 0.5



def test_record_appends_the_current_figures_to_the_history():
    grid = main(count=32, across=8, seed=1)
    teacher = Teacher(grid, seed=1)
    assert teacher.history == []
    teacher.step()
    entry = teacher.record(elapsed=12.34, epochs_per_second=2500.6)
    assert teacher.history == [entry]
    assert entry["epoch"] == 1 and entry["elapsed"] == 12.3 and entry["epochs_per_second"] == 2501
    assert entry["accuracy_to_date"] == round(teacher.accuracy_to_date, 4) and entry["recent"] == round(teacher.average, 4)
    assert entry["stuck_on"] + entry["stuck_off"] <= len(grid.neurons) - grid.across
    teacher.epoch(verbose=False)
    teacher.record()
    assert len(teacher.history) == 2 and teacher.history[1]["epoch"] == 2 and teacher.history[1]["elapsed"] is None


def test_a_late_signal_counts_by_default_and_is_ignored_or_depressed_on_request():
    """a fires in wave 0 and drives both b and c in wave 1; c's signal reaches b in wave 2, after b fired."""
    from walnutbutter.propagation import propagate

    a, b, c = Neuron("a"), Neuron("b"), Neuron("c")
    a_b = a.connect(b, 1, weight=1.0)
    a_c = a.connect(c, 2, weight=1.0)
    c_b = c.connect(b, 3, weight=0.5)
    b.noise = c.noise = 0.1  # both perturbed, so both would be eligible if timing were ignored

    class Tiny:
        hazard = True  # §8.3: the reinforce rule runs only where the decision is a draw
        waves = propagate(fire=[a])
        weight_range = (-1.0, 1.0)

    assert b.fired_in_wave == 1 and c.fired_in_wave == 1
    late = [s for s in delivered_signals(Tiny) if s.connection is c_b]
    assert late and late[0].wave == 2 and not landed(late[0])
    assert all(landed(s) for s in delivered_signals(Tiny) if s.connection in (a_b, a_c))
    changed = reinforce(Tiny, advantage=1.0, lr=0.1, sigma=0.1, late="ignore")
    assert changed == 2
    assert a_b.weight == pytest.approx(1.0) and a_c.weight == pytest.approx(1.0)  # clipped at the top
    assert c_b.weight == 0.5  # untouched: it changed nothing in this epoch
    reinforce(Tiny, advantage=-1.0, lr=0.1, eligibility="wrong_hebb", late="ignore")
    assert c_b.weight == 0.5  # the +-1 Hebbian variant respects timing too, when late signals are ignored
    assert a_b.weight < 1.0
    assert reinforce(Tiny, advantage=1.0, lr=0.1, sigma=0.1) == 3  # by default the late signal counts
    assert c_b.weight == pytest.approx(0.6)  # pre fired, post fired, reward was good: local rule, global signal
    assert reinforce(Tiny, advantage=1.0, lr=0.1, sigma=0.1, late="depress") == 3
    assert c_b.weight == pytest.approx(0.5)  # post before pre: the same good epoch now weakens it
    with pytest.raises(ValueError):
        reinforce(Tiny, advantage=1.0, late="whenever")
    ignoring = Teacher(Goo(count=12, across=4, weight=None, seed=1), late="ignore", seed=1)
    ignoring.epoch(verbose=False)
    assert ignoring.late == "ignore" and "late signals ignored" in ignoring.status()
    assert Teacher(Goo(count=12, across=4)).late == "count"
    with pytest.raises(ValueError):
        Teacher(Goo(count=12, across=4), late="whenever")



def test_unstick_reaches_every_stuck_neuron_and_leaves_a_forced_one_alone():
    """§6.7, September 14, 2026: un-sticking is for every neuron, not the output row -- the interior of a goo with no
    direct projection was dead for want of it. A neuron forced this epoch is left alone, as homeostasis leaves it."""
    from walnutbutter.learning import unstick
    grid = Goo(count=24, across=6)
    interior, out, forced_in = list(grid.all_neurons())[8], output_row(grid)[3], grid.input_row()[0]  # 8 is hidden: in neither zone
    for n in grid.all_neurons():
        n.rate = 0.5  # nobody stuck
    interior.rate, out.rate, forced_in.rate = 0.0, 1.0, 1.0
    forced_in.forced = True
    before = {n: n.threshold for n in grid.all_neurons()}
    nudged = unstick(grid, rate=0.1)
    assert set(nudged) == {interior, out}  # the interior neuron as much as the output; the forced input not at all
    assert interior.threshold == pytest.approx(before[interior] - 0.05) and out.threshold == pytest.approx(before[out] + 0.05)
    assert forced_in.threshold == before[forced_in]
    assert all(n.threshold == before[n] for n in grid.all_neurons() if n not in (interior, out))


def test_the_count_hebb_eligibility_is_what_the_synapse_delivered_times_the_centred_count():
    """§6.7's epoch form (September 16, 2026; count_hebb since the single-spike rule of September 17): e_ij = x_ij (n_j -
    n_bar_j) -- the signals the target integrated along the synapse this
    epoch, times its spike count minus its own running expectation. A forced target is skipped, a target with no
    expectation yet moves nothing, and the rule carries its own tally, so late = count and no leaky trace."""
    from walnutbutter.goo import Goo
    grid = Goo(count=32, across=8, weight=None, seed=4)
    grid.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
    grid.rule, grid.drive = "reinforce", "rate"
    teacher = Teacher(grid, eligibility="count_hebb", seed=3, rule="reinforce")
    assert grid.tally and teacher.sigma == 0.0
    run_epoch(grid, verbose=False)
    tallies = {c.id: c.eligibility for c in grid.connections.values()}
    assert any(tallies.values())  # signals were delivered, and counted
    assert all(n.expected_count is None for n in grid.all_neurons())  # nothing has been seen yet
    before = {c.id: c.weight for c in grid.connections.values()}
    assert reinforce(grid, advantage=0.7, lr=0.1, eligibility="count_hebb") == 0  # a first epoch centres on itself
    assert {c.id: c.weight for c in grid.connections.values()} == before
    for k, n in enumerate(grid.all_neurons()):
        n.expected_count = 0.5 * (k % 5)  # set by hand, so the arithmetic can be checked to the bit
    changed = reinforce(grid, advantage=0.7, lr=0.1, eligibility="count_hebb")
    moved = 0
    for c in grid.connections.values():
        t = c.target
        e = tallies[c.id] * (t.epoch_spikes - t.expected_count)
        if t.forced or not e:
            assert c.weight == before[c.id]
        else:
            assert c.weight == max(-1.0, min(1.0, before[c.id] + 0.1 * 0.7 * e))
            moved += 1
    assert changed == moved > 0
    with pytest.raises(ValueError, match="late = count"):
        reinforce(grid, advantage=1.0, eligibility="hebb", late="ignore")
    with pytest.raises(ValueError, match="no leaky trace"):
        reinforce(grid, advantage=1.0, eligibility="hebb", leaky=True)


def test_update_rates_keeps_the_expected_count_from_the_first_epoch_seen():
    """§6.7: n_bar_j starts at the first count seen in an unforced epoch and then moves by COUNT_MEMORY an epoch."""
    from walnutbutter.constants import COUNT_MEMORY
    from walnutbutter.goo import Goo
    from walnutbutter.learning import update_rates
    grid = Goo(count=18, across=6, weight=None, seed=2)
    grid.drive = "rate"
    run_epoch(grid, verbose=False)
    update_rates(grid)
    assert any(n.forced for n in grid.all_neurons()) and any(not n.forced for n in grid.all_neurons())
    for n in grid.all_neurons():
        assert n.expected_count == (None if n.forced else float(n.epoch_spikes))
    seen = {n: n.expected_count for n in grid.all_neurons()}
    run_epoch(grid, verbose=False)
    update_rates(grid)
    for n in grid.all_neurons():
        if n.forced:
            assert n.expected_count == seen[n]  # a forced epoch says nothing about the network
        elif seen[n] is None:
            assert n.expected_count == float(n.epoch_spikes)
        else:
            assert n.expected_count == seen[n] + COUNT_MEMORY * (float(n.epoch_spikes) - seen[n])
