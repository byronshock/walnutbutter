"""--problem: what the network is asked to do, whether anything outside it trains it, and how it is read."""

import json

import pytest

from walnutbutter import constants as C
from walnutbutter.cli import apply_problem, build_parser, cli_main
from walnutbutter.goo import Goo
from walnutbutter.learning import Teacher, accuracy
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.problems import PROBLEMS


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def test_the_problems_and_the_default():
    # §9.1 carries one rule that pays at the read and §5.2 one coding, so the seven problems posed on the
    # external teacher and on a dropped coding went with them
    assert set(PROBLEMS) == {"reversal", "copy", "mnist"}
    assert build_parser().parse_args([]).problem == C.PROBLEM == "reversal"
    # copy (Byron, September 14, 2026): eight complement-coded input neurons, eight outputs, place for place, unpermuted
    copy = PROBLEMS["copy"]
    assert PROBLEMS["reversal"].trained
    assert all(p.rule == "reinforce" for p in PROBLEMS.values())  # §9.1: the one rule that pays at the read
    assert all(p.interval is None for p in PROBLEMS.values())  # every problem takes INTERVAL (§4.2)


def test_apply_problem_settles_interval_target_readout_and_training():
    args = build_parser().parse_args(["--problem", "copy"])
    apply_problem(args)
    assert args.interval == C.INTERVAL == 35.0 and args.target == "copy" and args.readout == "top"
    args = build_parser().parse_args(["--problem", "copy", "--interval", "7"])
    apply_problem(args)
    assert args.interval == 7.0  # an explicit interval wins
    args = build_parser().parse_args([])
    apply_problem(args)
    assert args.interval == C.INTERVAL and args.readout == "top" and args.read == "fired" and args.target == "reversed"
    args = build_parser().parse_args(["--problem", "copy", "--rule", "local"])
    apply_problem(args)  # §9.1: a run may have none, and then the local rules are the whole of the learning
    assert args.rule == "local"


def test_the_inputs_are_the_outputs_and_on_means_spiked_again():
    grid = Goo(count=12, across=4, weight=1.0)
    grid.readout, grid.read, grid.interval = "input", "again", 20.0
    assert grid.output_row() == grid.input_row()
    run_epoch(grid, bits=[True, False], verbose=False)
    fired_at = [n.fired_at for n in grid.input_row()]
    assert grid.output_fired() == [t is not None and t > grid.time for t in fired_at]  # strictly after the forced moment
    forced = [n for n in grid.input_row() if n.forced]
    assert forced and all(n.spikes >= 1 for n in forced)
    for n in forced:  # a forced neuron that only fired the once, at the input's moment, is not on
        if n.fired_at == grid.time:
            assert not grid.output_fired()[grid.input_row().index(n)]
    assert 0.0 <= accuracy(grid, "copy") <= 1.0
    grid.read, grid.read_window = "window", 5.0
    since = grid.horizon - 5.0
    assert grid.output_fired() == [t is not None and t >= since - 1e-9 for t in fired_at]  # the window read, still available
    grid.read = "fired"
    assert grid.output_fired() == [n.has_fired for n in grid.input_row()]  # the plain read: fired this epoch




def test_the_sustained_critic_scores_only_the_forced_neurons():
    from walnutbutter.learning import CRITICS, sustained
    grid = Goo(count=12, across=4, weight=1.0)
    grid.readout, grid.read = "input", "fired"  # §5.10: this test drives has_fired, so it names the fired read
    grid.set_input_bits([True, False])  # pattern 1 0 0 1 after complement coding, before the permutation
    row = grid.input_row()
    forced = [n for n, bit in zip(row, grid.input_pattern) if bit]
    unforced = [n for n, bit in zip(row, grid.input_pattern) if not bit]
    assert len(forced) == len(unforced) == 2
    for n in row:
        n.has_fired = False
    assert sustained(grid) == 0.0
    forced[0].has_fired = True
    assert sustained(grid) == 0.5  # one of the two forced neurons sustained
    unforced[0].has_fired = unforced[1].has_fired = True
    assert sustained(grid) == 0.5  # the unforced neurons are not scored, on or off
    forced[1].has_fired = True
    assert sustained(grid) == 1.0
    assert CRITICS["sustained"] is sustained and "sustained" in CRITICS
    grid.set_input_bits([False, False])  # still two forced: complement coding always forces half the row
    assert sustained(grid) in (0.0, 0.5, 1.0)


def test_both_engines_read_the_same_outputs():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        grid = Goo(count=32, across=8, weight=None, seed=4)
        grid.readout, grid.read, grid.interval = "input", "again", 20.0
        return grid

    mesh, net = make(), ArrayNetwork(make())
    assert net.output_index.tolist() == [net.index[n] for n in net.mesh.input_row()]
    a, b = Teacher(mesh, seed=1, target="copy", homeostasis=0, unstick=0), Teacher(net, seed=1, target="copy", homeostasis=0, unstick=0)
    for _ in range(40):
        assert a.epoch(verbose=False) == b.epoch(verbose=False)
        assert mesh.output_fired() == net.output_fired()



def test_reversal_is_unchanged(tmp_path, capsys):
    save = tmp_path / "r.json"
    assert cli_main(["--headless", "--seed", "3", "--epochs", "3", "--save-weights", str(save), "--no-trace"]) == 0
    data = json.loads(save.read_text())
    assert data["problem"] == "reversal" and "learning" in data and data["across"] == 8 and data["interval"] == 35.0
    assert not save.with_suffix(".csv").exists()







def test_quashing_weakens_the_synapses_that_carried_a_cycle():
    import math
    from walnutbutter.local import quash
    from walnutbutter.propagation import Wave

    a, b, idle = Neuron("a"), Neuron("b"), Neuron("idle")
    looped = a.connect(b, 1, weight=0.8)
    stale = idle.connect(b, 2, weight=0.8)
    b.previous_fired_at, b.fired_at = 10.0, 16.0  # it refired six milliseconds later
    looped.last_signal, stale.last_signal = 15.0, 9.0  # one carried a signal since that spike, one did not
    first = Neuron("first")
    first.connect(b, 3, weight=0.8).last_signal = 15.0

    assert quash(Wave(0, 16.0, fired=[b, a]), rate=0.02, k=0.2, weight_range=(-1.0, 1.0)) == 2
    factor = 0.02 * math.exp(-0.2 * 6.0)
    assert looped.weight == pytest.approx(0.8 * (1 - factor))  # pulled toward zero, hardest for the tightest loop
    assert stale.weight == 0.8  # it carried nothing since the previous spike
    assert a.outgoing[0] is looped and a.previous_fired_at is None  # a's first spike: no cycle of its own to quash
    tight, loose = Neuron("t"), Neuron("l")
    for neuron, delay in ((tight, 5.0), (loose, 30.0)):
        neuron.previous_fired_at, neuron.fired_at = 0.0, delay
        Neuron("f").connect(neuron, 4, weight=1.0).last_signal = delay - 0.1
    quash(Wave(0, 5.0, fired=[tight]), 0.02, 0.2, (-1.0, 1.0))
    quash(Wave(0, 30.0, fired=[loose]), 0.02, 0.2, (-1.0, 1.0))
    assert tight.incoming[0].weight < loose.incoming[0].weight < 1.0


def test_population_copy_runs_and_both_engines_quash_identically():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        grid = Goo(count=120, across=12, weight=None, seed=5)
        grid.readout, grid.read, grid.interval = "top", "fired", 20.0
        grid.quash_rate, grid.quash_k = 0.02, 0.2
        return grid

    mesh, net = make(), ArrayNetwork(make())
    assert net.quash_rate == 0.02 and net.population == 3
    before = np.array([c.weight for c in mesh.connections.values()])
    for _ in range(30):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert mesh.input_pattern == net.input_pattern and len(mesh.input_pattern) == 12
        assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
        assert np.allclose([c.weight for c in mesh.connections.values()], net.weight, atol=1e-12)
    after = np.array([c.weight for c in mesh.connections.values()])
    assert np.abs(after).sum() < np.abs(before).sum()  # quashing only ever pulls a weight toward zero





def test_rate_drive_makes_a_bit_a_firing_rate_not_a_mandated_spike():
    """AUTHORITY.md §4.3: each input neuron is a Poisson process across the epoch."""
    import statistics

    def grid_with(drive, rate=0.1, off=0.0, seed=4):
        grid = Goo(count=24, across=12, weight=1.0, seed=seed)
        grid.interval = 20.0
        grid.drive, grid.input_rate, grid.input_rate_off = drive, rate, off
        grid.set_input_bits([True, False, False, True, True, False])
        return grid

    forced = grid_with("forced")
    assert forced.input_schedule() == [(p, 0.0) for p in (0, 3, 4, 7, 8, 11)]  # one spike each, all at t_e
    assert forced.input_schedule() == forced.input_schedule()  # and no randomness to consume

    rated = grid_with("rate")
    counts, silent, runs = [], 0, 400
    for _ in range(runs):
        events = rated.input_schedule()
        assert all(0.0 <= when < rated.interval for _, when in events)
        assert [w for _, w in events] == sorted(w for _, w in events)  # in time order
        assert all(rated.input_pattern[place] for place, _ in events)  # a zero bit fires at rate 0: silence
        counts.append(len(events))
        silent += int(not any(place == 0 for place, _ in events))
    assert 1.6 < statistics.fmean(counts) / 6 < 2.4  # six bit-1 neurons, two spikes each on average over 20 ms
    assert 0.08 < silent / runs < 0.20  # exp(-0.1 * 20) = 0.135: a bit-1 neuron often produces nothing at all

    background = grid_with("rate", off=0.05)
    zeros = sum(1 for _ in range(200) for place, _ in background.input_schedule() if not background.input_pattern[place])
    assert zeros  # with an off-rate a zero bit is a low rate rather than silence


def test_rate_drive_runs_and_both_engines_draw_the_same_train():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        grid = Goo(count=24, across=12, weight=None, seed=5)
        grid.readout, grid.read, grid.interval = "top", "fired", 20.0
        grid.drive, grid.quash_rate = "rate", 0.02
        return grid

    mesh, net = make(), ArrayNetwork(make())
    assert net.drive == "rate" and net.input_rate == mesh.input_rate
    spread = set()
    for _ in range(30):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert mesh.input_events == net.input_events  # the same Poisson train, drawn from the same stream
        assert mesh.input_pattern == net.input_pattern
        assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
        assert np.allclose([c.weight for c in mesh.connections.values()], net.weight, atol=1e-12)
        spread.update(round(when - mesh.time, 6) for _, when in mesh.input_events)
    assert len(spread) > 30  # the stimuli land all over the epoch, not in one wave
    assert len({round(w, 6) for w in spread}) > 1 and max(spread) > 5.0


def test_a_problem_and_the_command_line_choose_the_drive():
    """The real default, deliberately not under conftest's forced_input: no problem overrides it."""
    args = build_parser().parse_args(["--problem", "copy"])
    apply_problem(args)
    assert args.drive == C.INPUT_DRIVE == "rate"  # no unified wave front at time zero (§4.3)
    assert all(p.drive is None for p in PROBLEMS.values())
    args = build_parser().parse_args(["--problem", "copy", "--drive", "forced", "--input-rate", "0.2"])
    apply_problem(args)
    assert args.drive == "forced" and args.input_rate == 0.2


def test_the_default_drive_leaves_no_wave_front_at_time_zero():
    """§4.3: forced drive locks every spike onto a hop grid; rate drive has no such anchor.

    The grid is the run's clock and not the epoch's. Until September 19, 2026 the two were the same
    thing, because INTERVAL was exactly 14 hops at HOP 2.5; at the specified 2.55 it is 13.7255, so
    each epoch's forced inputs start a lattice of their own and the per-epoch alignment falls to 11%.
    What survives is the run-wide one: every spike a cascade causes sits on a multiple of the hop,
    and the 9% that do not are the driven spikes themselves, anchored at multiples of INTERVAL.
    That the epoch no longer holds a whole number of hops is a consequence of A1's fix and is
    recorded with it -- it weakens the very synchrony Byron struck `forced` for on September 14.
    """
    def spikes_on_the_grid(drive):
        grid = Goo(count=60, across=12, weight=None, seed=1)
        grid.readout, grid.read, grid.drive = "top", "fired", drive
        hop, on_grid, total, at_zero = Neuron.hop, 0, 0, 0
        row = set(grid.input_row())
        for _ in range(40):
            run_epoch(grid, [True, False, True, False, True, False], verbose=False)
            for wave in grid.waves:
                phase = wave.time % hop  # the run's clock, not the epoch's: INTERVAL is not a whole number of hops
                aligned = min(phase, hop - phase) < 1e-9
                for neuron in wave.fired:
                    total += 1
                    on_grid += aligned
                    at_zero += (abs(wave.time - grid.time) < 1e-9 and neuron in row)
        return on_grid / total, at_zero

    aligned, at_zero = spikes_on_the_grid("forced")
    assert aligned > 0.9 and at_zero > 0  # the cascades on one lattice, and the input row starts together
    aligned, at_zero = spikes_on_the_grid("rate")
    assert aligned == 0.0 and at_zero == 0  # no common lattice, and nothing fires at the epoch's moment


def test_the_driving_process_is_not_the_spike_train():
    """AUTHORITY.md §4.3: the Poisson process drives, and the refractory period makes the spike train a renewal one."""
    import statistics
    from walnutbutter.constants import INPUT_CV, INPUT_RATE, REFRACTORY, cv_for_rate, rate_for_cv

    assert INPUT_CV == 0.6  # the drive is specified by the CV it produces, and lambda follows (§4.3)
    assert INPUT_RATE == pytest.approx(rate_for_cv(0.6)) == pytest.approx(2 / 15)
    assert cv_for_rate(INPUT_RATE) == pytest.approx(INPUT_CV)  # the two coordinates round-trip
    assert 1000.0 * (1 - INPUT_CV) / REFRACTORY == pytest.approx(80.0)  # rate = (1 - CV) / REFRACTORY, exactly

    def train(rate, epochs=400):
        grid = Goo(count=60, across=12, weight=0.0, seed=1)  # weight 0: no mesh drive
        grid.drive, grid.input_rate = "rate", rate
        row, times, arrivals = grid.input_row(), {p: [] for p in range(12)}, 0
        for _ in range(epochs):
            run_epoch(grid, [True] * 6, verbose=False)
            arrivals += len(grid.input_events)
            for wave in grid.waves:
                for neuron in wave.fired:
                    if neuron in row:
                        times[row.index(neuron)].append(wave.time)
        isis = [b - a for v in times.values() for a, b in zip(v, v[1:])]
        spikes = sum(len(v) for v in times.values())
        return arrivals, spikes, isis

    arrivals, spikes, isis = train(INPUT_RATE)
    kept = 1.0 / (INPUT_RATE * REFRACTORY + 1.0)  # the rest land inside a refractory period and are dropped
    assert spikes == pytest.approx(arrivals * kept, rel=0.1) and kept < 0.65
    assert all(gap >= REFRACTORY - 1e-9 for gap in isis)  # nothing fires sooner than the refractory period allows
    mean = statistics.fmean(isis)
    assert mean == pytest.approx(REFRACTORY + 1.0 / INPUT_RATE, rel=0.05)  # dead time plus an exponential wait

    for rate in (3.8, INPUT_RATE, 0.05):  # the CV the drive produces is the one the identity predicts, across the range
        measured = (lambda i: statistics.stdev(i) / statistics.fmean(i))(train(rate)[2])
        assert measured == pytest.approx(cv_for_rate(rate), rel=0.12), rate
        assert measured < 1.0  # always below a Poisson train's 1.0: this is what "not a Poisson process" buys


# --- the same problem, only NOT (AUTHORITY.md §8) ------------------------------------

def test_a_problem_can_carry_its_own_learning_rate():
    """§8, mnist (Byron, September 16, 2026: "default LR to 0.002 for this task"): the constant unless the problem or --lr says."""
    from walnutbutter.cli import apply_problem, build_parser
    from walnutbutter.constants import LR
    assert PROBLEMS["mnist"].lr == 0.002 and PROBLEMS["copy"].lr is None
    args = build_parser().parse_args(["--problem", "mnist"]); apply_problem(args)
    assert args.lr == 0.002
    args = build_parser().parse_args(["--problem", "mnist", "--lr", "0.01"]); apply_problem(args)
    assert args.lr == 0.01  # given, so kept
    args = build_parser().parse_args(["--problem", "copy"]); apply_problem(args)
    assert args.lr == LR



def test_the_complement_is_what_excitation_alone_cannot_reach():
    """§8: an output whose whole input group is silent receives no signal, so no weight can drive it."""
    from walnutbutter.learning import accuracy

    grid = Goo(count=24, across=12, weight=None, seed=4)
    grid.readout, grid.read, grid.drive = "top", "fired", "rate"
    quiet_but_wanted = quiet_and_wanted_quiet = 0
    for _ in range(60):
        run_epoch(grid, [True, False, True, False, True, False], verbose=False)
        want = [not b for b in grid.input_pattern]
        for fired, w in zip(grid.output_fired(), want):
            if w and not fired:
                quiet_but_wanted += 1
            elif not w and not fired:
                quiet_and_wanted_quiet += 1
    assert quiet_but_wanted > 0  # the half the task cannot reach by excitation
    assert accuracy(grid, "complement") <= 1.0


# --- doubled, complement-coded and scrambled (AUTHORITY.md §8) -----------------------


def test_the_input_stream_is_the_same_whatever_the_network_is():
    """AUTHORITY.md §4.5: inputs drawn up front, from a stream of their own, so arms are comparable."""
    from walnutbutter.network import input_stream

    patterns = input_stream(50, 4, seed=11)
    assert patterns == input_stream(50, 4, seed=11) != input_stream(50, 4, seed=12)
    assert len(patterns) == 50 and all(len(p) == 4 for p in patterns)

    seen = []
    for reach, omega, rows in ((2, 0.0, 5), (5, 0.2, 8), (3, 0.4, 2)):  # every build consumes randomness differently
        grid = Goo(count=8 * rows, across=8, weight=None, seed=1)
        grid.use_input_stream(patterns)
        seen.append([grid.new_random_input() for _ in range(50)])
    assert seen[0] == seen[1] == seen[2] == patterns  # the same epochs in the same order, whatever the network

    drawn = []
    for reach, omega, rows in ((2, 0.0, 5), (5, 0.2, 8)):  # and without a stream they diverge, as they always did
        grid = Goo(count=8 * rows, across=8, weight=None, seed=1)
        drawn.append([grid.new_random_input() for _ in range(50)])
    assert drawn[0] != drawn[1]


def test_a_run_longer_than_its_stream_cycles_it():
    grid = Goo(count=16, across=8, weight=None, seed=1)
    grid.use_input_stream([[True, False, True, False], [False, False, True, True]])
    got = [grid.new_random_input() for _ in range(5)]
    assert got == [got[0], got[1], got[0], got[1], got[0]]
    grid.use_input_stream(None)  # and handing back None goes to drawing again
    assert grid.input_stream is None and len(set(tuple(grid.new_random_input()) for _ in range(40))) > 1


def test_the_stream_is_checked_when_it_is_attached():
    grid = Goo(count=16, across=8, weight=None, seed=1)
    with pytest.raises(ValueError, match="pattern 1 has 3 bits, expected 4"):
        grid.use_input_stream([[True, False, True, False], [True, False, True]])
    assert grid.input_stream is None  # and nothing is half-attached


def test_the_array_engine_carries_the_stream_across_the_wrap():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter.network import input_stream

    patterns = input_stream(30, 4, seed=5)
    mesh = Goo(count=24, across=8, weight=None, seed=2)
    mesh.use_input_stream(patterns)
    net = ArrayNetwork(mesh)
    assert net.input_stream == patterns and net.input_at == 0
    assert [net.new_random_input() for _ in range(10)] == patterns[:10]
    net.sync_to_mesh()
    assert mesh.input_at == 10  # the mesh picks up where the arrays left off
    assert mesh.new_random_input() == patterns[10]


def test_copy_asks_each_output_for_exactly_its_input_place(capsys):
    """§8 copy: output place i is taught to show coded bit i, unpermuted, on the grid and on goo alike."""
    from walnutbutter.cli import apply_problem
    from walnutbutter.goo import Goo
    from walnutbutter.learning import TARGETS
    args = build_parser().parse_args(["--problem", "copy"])
    apply_problem(args)
    goo = Goo(seed=1)
    goo.set_input_bits([True, False, False, True])
    assert goo.input_pattern == [True, False, False, True, False, True, True, False]  # the bits, then their negations
    assert TARGETS["copy"](goo.target_pattern) == goo.input_pattern  # what the eight outputs are asked to show
    assert len(goo.input_row()) == len(goo.output_row()) == 8
    assert cli_main(["--problem", "copy", "--goo", "--headless", "--epochs", "3", "--seed", "1", "--no-save"]) == 0
    assert "learning copy (" in capsys.readouterr().err  # the Teacher's status names the target it teaches


def test_the_drive_stops_at_the_presentation_time_and_the_tail_is_not_silent():
    """§5.4a (Byron, September 18, 2026): the drive runs to t_e + PRESENTATION_TIME and not past it.

    Three things are checked because each can fail on its own: no arrival lands past the window --
    which is the clause, and the only test that can catch a wrong window, since all three engines
    share `input_schedule` and a wrong one would be wrong in all three at once; the default draws the
    whole epoch, so nothing measured moves until a run shortens it; and the tail is *undriven*, not
    silent -- every neuron still decides at every wave (§6.8) and carries the hazard's rest (§7.2).
    """
    from walnutbutter.goo import Goo
    from walnutbutter.neuron import Neuron

    Neuron.verbose = False
    grid = Goo(count=60, across=12, weight=None, seed=1)
    grid.readout, grid.read, grid.drive = "top", "fired", "rate"
    grid.set_input_bits([True] * 6)

    assert grid.presentation is None and grid.presentation_time == grid.interval == 35.0
    whole = grid.input_schedule()
    assert whole and max(t for _, t in whole) < grid.time + 35.0

    grid.presentation = 10.0
    short = grid.input_schedule()
    assert short, "a ten-millisecond window still draws arrivals"
    assert max(t for _, t in short) < grid.time + 10.0  # the clause: nothing past the window
    assert len(short) < len(whole)  # and fewer draws, which is what shifts the stream (§12.6)

    grid.presentation = 99.0  # §0.5: refused, not clipped
    with pytest.raises(ValueError, match="longer than the epoch"):
        grid.input_schedule()


def test_the_presentation_window_round_trips_and_defaults_to_the_whole_epoch(tmp_path):
    """§12.9: the window is a run's setting, so a checkpoint carries it or §12.11's exact resume breaks."""
    from walnutbutter.goo import Goo
    from walnutbutter.persistence import checkpoint, read_checkpoint

    grid = Goo(count=12, across=4, seed=1)
    data = checkpoint(grid, tmp_path / "whole.json")
    assert data["presentation_time"] is None  # the default is the whole epoch, written as such

    grid.presentation = 12.5
    data = checkpoint(grid, tmp_path / "short.json")
    assert read_checkpoint(tmp_path / "short.json")["presentation_time"] == 12.5
