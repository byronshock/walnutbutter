"""--problem: what the network is asked to do, whether anything outside it trains it, and how it is read."""

import json

import pytest

from walnutbutter import constants as C
from walnutbutter.cli import apply_problem, build_parser, cli_main
from walnutbutter.grid import GridOfNeurons
from walnutbutter.learning import Teacher, accuracy
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.problems import PROBLEMS


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def test_the_problems_and_the_default():
    assert set(PROBLEMS) == {"reversal", "sustain_inputs", "improved_sustain", "population_copy",
                             "population_denoise", "shallow_copy"}
    assert build_parser().parse_args([]).problem == C.PROBLEM == "reversal"
    assert PROBLEMS["reversal"].trained and not PROBLEMS["sustain_inputs"].trained
    sustain = PROBLEMS["sustain_inputs"]
    assert sustain.across == 4 and sustain.coding == "raw"  # the 16 four-bit inputs as they are on 4 neurons
    assert (sustain.readout, sustain.read, sustain.target, sustain.critic) == ("input", "again", "copy", "row")
    assert sustain.interval is None  # every problem takes INTERVAL now, swept to 35 ms (§4.2)


def test_apply_problem_settles_interval_target_readout_and_training():
    args = build_parser().parse_args(["--problem", "sustain_inputs"])
    apply_problem(args)
    assert args.interval == C.INTERVAL == 35.0 and args.target == "copy" and args.readout == "input" and args.read == "again"
    assert args.critic == "row" and args.coding == "raw" and args.across == 4
    assert args.learn and args.homeostasis == 0 and args.unstick == 0  # scored, never trained from outside
    args = build_parser().parse_args(["--problem", "sustain_inputs", "--interval", "7"])
    apply_problem(args)
    assert args.interval == 7.0  # an explicit interval wins
    args = build_parser().parse_args([])
    apply_problem(args)
    assert args.interval == C.INTERVAL and args.readout == "top" and args.read == "fired" and args.target == "reversed"
    assert args.coding == "complement"
    args = build_parser().parse_args(["--problem", "sustain_inputs", "--rule", "reinforce"])
    apply_problem(args)  # the reinforce rule may be asked for on any problem (§6.7, Byron, September 13, 2026)
    assert args.rule == "reinforce" and args.homeostasis == 0 and args.unstick == 0


def test_the_inputs_are_the_outputs_and_on_means_spiked_again():
    grid = GridOfNeurons(across=4, rows=3, weight=1.0, omega=0)
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


def test_raw_coding_lays_the_bits_down_as_they_are():
    from walnutbutter.persistence import checkpoint, restore
    grid = GridOfNeurons(across=4, rows=3, weight=1.0, omega=0, seed=2, permute=False)
    grid.coding = "raw"
    assert grid.raw_bit_count() == 4
    grid.set_input_bits([True, False, False, True])
    assert grid.input_pattern == [True, False, False, True] and grid.input_coded == grid.input_bits == [True, False, False, True]
    grid.set_input_bits([False, False, False, False])  # nothing forced is a legal input now
    assert grid.input_neurons() == [] and sum(grid.input_pattern) == 0
    seen = {tuple(grid.new_random_input()) for _ in range(300)}
    assert len(seen) == 16 and all(len(bits) == 4 for bits in seen)  # all sixteen four-bit patterns
    odd = GridOfNeurons(across=3, rows=2, omega=0)
    odd.coding = "raw"
    assert odd.raw_bit_count() == 3  # raw coding has no evenness requirement
    grid.coding = "complement"
    assert grid.raw_bit_count() == 2
    grid.coding = "raw"
    grid.use_ecc(False)
    with pytest.raises(ValueError):
        grid.use_ecc()  # a code needs complement coding
    grid.ecc = None
    import tempfile, os
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "raw.json")
        assert checkpoint(grid, path)["coding"] == "raw"
        assert restore(path)[0].coding == "raw"


def test_both_engines_agree_on_raw_inputs():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        grid = GridOfNeurons(across=4, rows=6, weight=None, seed=9)
        grid.coding, grid.readout, grid.read, grid.interval = "raw", "input", "again", 20.0
        return grid

    mesh, net = make(), ArrayNetwork(make())
    assert net.coding == "raw"
    for _ in range(40):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert mesh.input_pattern == net.input_pattern and len(mesh.input_pattern) == 4
        assert mesh.output_fired() == net.output_fired()
        assert accuracy(mesh, "copy") == accuracy(net, "copy")


def test_the_sustained_critic_scores_only_the_forced_neurons():
    from walnutbutter.learning import CRITICS, sustained
    grid = GridOfNeurons(across=4, rows=3, weight=1.0, omega=0)
    grid.readout = "input"
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
        grid = GridOfNeurons(across=8, rows=4, weight=None, seed=4)
        grid.readout, grid.read, grid.interval = "input", "again", 20.0
        return grid

    mesh, net = make(), ArrayNetwork(make())
    assert net.output_index.tolist() == [net.index[n] for n in net.mesh.input_row()]
    a, b = Teacher(mesh, seed=1, target="copy", homeostasis=0, unstick=0), Teacher(net, seed=1, target="copy", homeostasis=0, unstick=0)
    for _ in range(40):
        assert a.epoch(verbose=False) == b.epoch(verbose=False)
        assert mesh.output_fired() == net.output_fired()


def test_sustain_inputs_is_scored_traced_and_checkpointed(tmp_path, capsys):
    save, trace = tmp_path / "s.json", tmp_path / "s.csv"
    assert cli_main(["--headless", "--problem", "sustain_inputs", "--seed", "3", "--epochs", "12", "-r", "4",
                     "--save-weights", str(save)]) == 0
    err = capsys.readouterr().err
    assert "-> coded" not in err
    assert "problem sustain_inputs" in err and "Epochs 35 ms apart" in err and "by the row critic, on meaning spiked again after the input" in err
    assert "shows raw bit" in err
    assert "teaching copy" in err and "rule: teacher" in err and f"tracing every epoch to {trace}" in err
    data = json.loads(save.read_text())
    assert data["across"] == 4 and data["epoch"] == 12 and data["problem"] == "sustain_inputs" and data["interval"] == 35.0
    assert data["coding"] == "raw" and data["learning"]["target"] == "copy" and data["learning"]["critic"] == "row"
    assert data["learning"]["rule"] == "teacher"
    lines = trace.read_text().splitlines()
    assert lines[0] == "epoch,time_ms,dopamine,expected,score" and len(lines) == 13
    epoch, time_ms, dopamine, expected, score = lines[-1].split(",")
    assert epoch == "12" and float(time_ms) == 11 * 35.0 and float(dopamine) >= 0 and float(expected) >= 0 and 0 <= float(score) <= 1
    assert cli_main(["--headless", "--load-weights", str(save), "--epochs", "3", "--no-save"]) == 0
    err = capsys.readouterr().err
    assert "problem: sustain_inputs (from the checkpoint)" in err and "teaching copy" in err and "tracing" not in err
    assert cli_main(["--headless", "--problem", "sustain_inputs", "--seeds", "2", "--seed", "1", "--epochs", "3", "--no-save"]) == 0
    assert cli_main(["--headless", "--problem", "sustain_inputs", "--rule", "reinforce", "--epochs", "3", "--no-save"]) == 0
    err = capsys.readouterr().err  # §6.7: the rule runs on an untrained problem, keeping no dopamine pool and so no decay
    assert "rule: reinforce (perturb eligibility" in err and "no weight decay" in err


def test_reversal_is_unchanged(tmp_path, capsys):
    save = tmp_path / "r.json"
    assert cli_main(["--headless", "--seed", "3", "--epochs", "3", "-r", "4", "--save-weights", str(save), "--no-trace"]) == 0
    data = json.loads(save.read_text())
    assert data["problem"] == "reversal" and "learning" in data and data["across"] == 8 and data["interval"] == 35.0
    assert not save.with_suffix(".csv").exists()


def test_the_external_teacher_scores_the_read_and_pays_the_eligibility():
    from walnutbutter.dopamine import Dopamine, apply_teacher
    from walnutbutter.learning import RULES, teacher_score

    assert RULES[0] == "teacher" and C.RULE == "teacher" and C.TEACHER_CREDIT is None  # None: 1/n, so the score spans [-1, 1]
    grid = GridOfNeurons(across=4, rows=3, weight=1.0, omega=0, permute=False)  # the bits land where they are
    grid.coding, grid.readout, grid.read, grid.rule = "raw", "input", "again", "teacher"
    grid.dopamine = Dopamine(lr=0.1)
    grid.set_input_bits([True, False, True, False])
    row = grid.input_row()

    def score_with(on):  # what the teacher would say if these input neurons read as on
        for neuron, is_on in zip(row, on):
            neuron.fired_at = grid.time + 1.0 if is_on else None
        return teacher_score(grid)

    assert score_with([True, False, True, False]) == 1.0  # all four right
    assert score_with([False, True, False, True]) == -1.0  # all four wrong
    assert score_with([True, False, True, True]) == 0.5  # three right, one wrong
    assert score_with([True, True, True, True]) == 0.0  # two and two
    assert score_with([False, False, True, False]) == 0.5
    assert sorted({score_with([a, b, c, d]) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)}) == [-1.0, -0.5, 0.0, 0.5, 1.0]

    # the trace: a refire earns, the teacher pays, the trace clears
    into = next(c for c in grid.connections.values() if c.target is row[0])
    into.weight, into.eligibility = 0.5, 2.0
    assert apply_teacher(grid, 0.5, 0.1) == 1 and into.weight == pytest.approx(0.5 + 0.1 * 0.5 * 2.0)
    assert into.eligibility == 0.0  # cleared, so an epoch's credit never carries into the next
    into.eligibility = 2.0
    assert apply_teacher(grid, 0.0, 0.1) == 0 and into.eligibility == 0.0  # a zero score still clears the trace


def test_the_teacher_rule_runs_end_to_end_and_both_engines_agree():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter.dopamine import Dopamine

    def make():
        grid = GridOfNeurons(across=4, rows=6, weight=None, seed=7)
        grid.coding, grid.readout, grid.read, grid.interval, grid.rule = "raw", "input", "again", 20.0, "teacher"
        grid.dopamine = Dopamine(lr=0.05)
        return grid

    mesh, net = make(), ArrayNetwork(make())
    a, b = Teacher(mesh, seed=1, target="copy", critic="row", rule="teacher"), Teacher(net, seed=1, target="copy", critic="row", rule="teacher")
    before = [c.weight for c in mesh.connections.values()]
    for _ in range(60):
        assert a.epoch(verbose=False) == b.epoch(verbose=False)
        assert a.last_signal == b.last_signal and a.last_signal in (-1.0, -0.5, 0.0, 0.5, 1.0)
        assert np.allclose([c.weight for c in mesh.connections.values()], net.weight, atol=1e-12)
    assert a.moved == b.moved > 0 and [c.weight for c in mesh.connections.values()] != before
    assert all(c.eligibility == 0.0 for c in mesh.connections.values()) and not net.eligibility.any()
    assert "teaching copy" in a.status() and "synapses moved" in a.status()


def test_adaline_corrects_mistakes_only_and_scales_by_what_each_synapse_delivered():
    from walnutbutter.dopamine import Dopamine
    from walnutbutter.learning import RULES, adaline_errors, apply_adaline

    assert "adaline" in RULES
    grid = GridOfNeurons(across=4, rows=3, weight=1.0, omega=0, permute=False)
    grid.coding, grid.readout, grid.read, grid.rule = "raw", "input", "again", "adaline"
    grid.dopamine = Dopamine(lr=0.1)
    grid.set_input_bits([True, False, True, False])
    row = grid.input_row()
    for neuron, on in zip(row, [True, True, False, False]):  # 0 right, 1 wrong (on, should be off), 2 wrong (off, should be on), 3 right
        neuron.fired_at = grid.time + 1.0 if on else None
    assert adaline_errors(grid) == [0.0, -1.0, 1.0, 0.0]

    into = {}
    for place, neuron in enumerate(row):
        c = next(c for c in grid.connections.values() if c.target is neuron)
        c.weight, c.eligibility = 0.5, 2.0
        into[place] = c
    elsewhere = next(c for c in grid.connections.values() if c.target not in row)
    elsewhere.weight, elsewhere.eligibility = 0.5, 2.0

    assert apply_adaline(grid, adaline_errors(grid), 0.1) == 2  # only the two neurons read wrongly
    assert into[0].weight == into[3].weight == 0.5  # read correctly: nothing moves
    assert into[1].weight == pytest.approx(0.5 + 0.1 * -1.0 * 2.0)  # it fired and should not have: weakened
    assert into[2].weight == pytest.approx(0.5 + 0.1 * 1.0 * 2.0)  # it should have fired: strengthened
    assert elsewhere.weight == 0.5  # not a scored neuron: the mesh behind them is an untrained reservoir
    assert all(c.eligibility == 0.0 for c in grid.connections.values())  # the trace is always cleared


def test_adaline_runs_end_to_end_and_both_engines_agree():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter.dopamine import Dopamine

    def make():
        grid = GridOfNeurons(across=4, rows=6, weight=None, seed=11)
        grid.coding, grid.readout, grid.read, grid.interval, grid.rule = "raw", "input", "again", 20.0, "adaline"
        grid.dopamine = Dopamine(lr=0.05, decay=0.0)  # no forgetting: only ADALINE moves a weight here
        return grid

    mesh, net = make(), ArrayNetwork(make())
    kw = dict(seed=2, target="copy", critic="row", rule="adaline", homeostasis=0, unstick=0)  # only ADALINE moves anything
    a, b = Teacher(mesh, **kw), Teacher(net, **kw)
    before = [c.weight for c in mesh.connections.values()]
    for _ in range(40):  # the engines part company at epoch 58 on this seed, when a potential lands within rounding of a threshold
        assert a.epoch(verbose=False) == b.epoch(verbose=False)
        assert a.mistakes == b.mistakes and 0 <= a.mistakes <= 4
        assert np.allclose([c.weight for c in mesh.connections.values()], net.weight, atol=1e-12)
    assert a.moved == b.moved > 0 and [c.weight for c in mesh.connections.values()] != before
    assert all(c.eligibility == 0.0 for c in mesh.connections.values()) and not net.eligibility.any()
    assert "correcting copy" in a.status() and "wrong" in a.status()
    row = set(mesh.input_row())  # only the scored neurons' incoming weights moved
    for connection, was in zip(mesh.connections.values(), before):
        if connection.target not in row:
            assert connection.weight == was


def test_population_coding_fills_three_neurons_per_bit():
    grid = GridOfNeurons(across=12, rows=10, weight=1.0, omega=0, permute=False)
    grid.coding = "population"
    assert grid.population == C.POPULATION == 3 and grid.raw_bit_count() == 4
    grid.set_input_bits([True, False, False, True])
    assert grid.input_pattern == [True] * 3 + [False] * 6 + [True] * 3  # 1001 -> 111000000111
    grid.set_input_bits([False, False, False, True])
    assert grid.input_pattern == [False] * 9 + [True] * 3  # 0001 -> 000000000111
    seen = {tuple(grid.new_random_input()) for _ in range(300)}
    assert len(seen) == 16 and all(len(bits) == 4 for bits in seen)
    odd = GridOfNeurons(across=10, rows=3, omega=0)  # 10 is not a multiple of three
    odd.coding = "population"
    with pytest.raises(ValueError):
        odd.raw_bit_count()


def test_the_teacher_score_spans_minus_one_to_one_whatever_the_zone():
    from walnutbutter.learning import teacher_score
    grid = GridOfNeurons(across=12, rows=10, weight=1.0, omega=0, permute=False)
    grid.coding, grid.read = "population", "fired"
    grid.set_input_bits([True, False, False, True])
    want = list(grid.input_pattern)

    def score_with(correct: int) -> float:
        for place, (neuron, bit) in enumerate(zip(grid.output_row(), want)):
            neuron.has_fired = bit if place < correct else not bit
        return teacher_score(grid)

    assert score_with(12) == pytest.approx(1.0) and score_with(0) == pytest.approx(-1.0)
    assert score_with(6) == pytest.approx(0.0)  # six of twelve right is a zero
    assert score_with(9) == pytest.approx(0.5) and score_with(3) == pytest.approx(-0.5)


def test_quashing_weakens_the_synapses_that_carried_a_cycle():
    import math
    from walnutbutter.dopamine import quash
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
        grid = GridOfNeurons(across=12, rows=10, weight=None, seed=5)
        grid.coding, grid.readout, grid.read, grid.interval = "population", "top", "fired", 20.0
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


def test_population_denoise_reads_the_input_zone_against_the_clean_code():
    denoise, copy = PROBLEMS["population_denoise"], PROBLEMS["population_copy"]
    assert denoise.flip == C.FLIP == pytest.approx(1 / 12) and copy.flip is None
    for same in ("across", "rows", "interval", "coding", "critic", "target", "rule", "quash", "permute", "reach", "trained"):
        assert getattr(denoise, same) == getattr(copy, same)  # the same network, read somewhere else
    assert (denoise.readout, denoise.read) == ("input", "again") and (copy.readout, copy.read) == ("top", "fired")
    args = build_parser().parse_args(["--problem", "population_denoise"])
    apply_problem(args)
    assert args.flip == pytest.approx(1 / 12) and args.readout == "input" and args.read == "again"
    assert args.across == 12 and args.rows == 10 and args.quash == C.QUASH_RATE
    args = build_parser().parse_args(["--problem", "population_denoise", "--flip", "0"])
    apply_problem(args)
    assert args.flip == 0.0  # an explicit flip wins
    args = build_parser().parse_args(["--problem", "population_copy"])
    apply_problem(args)
    assert args.flip == 0.0  # every other problem presents its input uncorrupted


def test_a_flip_corrupts_what_is_forced_but_not_what_is_scored():
    from walnutbutter.learning import expected_outputs, teacher_score

    def grid_with(flip):
        grid = GridOfNeurons(across=12, rows=10, weight=1.0, omega=0, permute=False, seed=3)
        grid.coding, grid.readout, grid.read, grid.flip = "population", "input", "again", flip
        return grid

    clean = grid_with(0.0)
    clean.set_input_bits([True, False, False, True])
    assert clean.input_pattern == clean.target_pattern == [True] * 3 + [False] * 6 + [True] * 3
    assert [n.should_fire for n in clean.input_row()] == clean.target_pattern

    every = grid_with(1.0)  # every bit flips: the forced row is the complement of the target
    every.set_input_bits([True, False, False, True])
    assert every.input_pattern == [not bit for bit in every.target_pattern]
    assert [n.should_fire for n in every.input_row()] == every.target_pattern  # what it should do, not what it was forced with
    assert expected_outputs(every, "copy") == every.target_pattern
    for neuron in every.input_row():  # a network that carries the corruption through reads the presented pattern
        neuron.has_fired, neuron.fired_at = True, 1.0
    every.input_time, every.time = 0.0, 0.0
    assert teacher_score(every, "copy") == pytest.approx(0.0)  # ...and every one of the twelve is then wrong

    some, wrong = grid_with(1 / 12), 0  # about one neuron in twelve
    for _ in range(2000):
        some.new_random_input()
        wrong += sum(a != b for a, b in zip(some.input_pattern, some.target_pattern))
    assert 0.06 < wrong / (2000 * 12) < 0.11


def test_no_flip_draws_nothing_from_the_stream():
    def bits(flip):
        grid = GridOfNeurons(across=12, rows=10, weight=None, seed=7, permute=False)
        grid.coding, grid.flip = "population", flip
        return [grid.new_random_input() for _ in range(20)]

    assert bits(0.0) == bits(0.0)
    assert bits(0.0) != bits(1 / 12)  # the flips come from the same seeded stream, so they do move it
    grid = GridOfNeurons(across=12, rows=10, weight=None, seed=7, permute=False)
    grid.coding = "population"
    assert [grid.new_random_input() for _ in range(20)] == bits(0.0)  # flip 0 leaves an older run bit for bit


def test_population_denoise_runs_and_both_engines_flip_identically():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        grid = GridOfNeurons(across=12, rows=10, weight=None, seed=5, permute=False)
        grid.coding, grid.readout, grid.read, grid.interval = "population", "input", "again", 20.0
        grid.quash_rate, grid.quash_k, grid.flip = 0.02, 0.2, 1 / 12
        return grid

    mesh, net = make(), ArrayNetwork(make())
    assert net.flip == mesh.flip
    corrupted = 0
    for _ in range(30):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert mesh.input_pattern == net.input_pattern and mesh.target_pattern == net.target_pattern
        assert net.sign[net.input_index].tolist() == [1.0 if bit else -1.0 for bit in mesh.target_pattern]
        assert np.allclose([c.weight for c in mesh.connections.values()], net.weight, atol=1e-12)
        corrupted += sum(a != b for a, b in zip(mesh.input_pattern, mesh.target_pattern))
    assert corrupted  # thirty epochs of twelve bits at one in twelve: some of them flipped


def test_shallow_copy_is_population_copy_one_hop_wide():
    shallow, deep = PROBLEMS["shallow_copy"], PROBLEMS["population_copy"]
    assert shallow.rows == 2 and deep.rows == 10  # the only thing that shrinks
    for same in ("across", "interval", "coding", "critic", "target", "rule", "quash", "permute", "reach",
                 "readout", "read", "trained", "flip"):
        assert getattr(shallow, same) == getattr(deep, same)
    args = build_parser().parse_args(["--problem", "shallow_copy"])
    apply_problem(args)
    assert args.rows == 2 and args.across == 12 and args.readout == "top" and args.read == "fired"
    args = build_parser().parse_args(["--problem", "shallow_copy", "--rows", "4"])
    apply_problem(args)
    assert args.rows == 4  # an explicit row count still wins

    grid = GridOfNeurons(across=12, rows=2, weight=None, seed=5, permute=False)
    top, bottom = set(grid.output_row()), set(grid.input_row())
    assert len(top) == len(bottom) == 12 and not (top & bottom)  # two rows, and they are different rows
    assert len(list(grid.all_neurons())) == 24
    reaches = sum(1 for c in grid.connections.values() if c.source in bottom and c.target in top)
    feeds_back = sum(1 for c in grid.connections.values() if c.source in top and c.target in bottom)
    assert reaches and feeds_back  # one hop from input to output, and the output feeds back: the only cycle here


def test_rate_drive_makes_a_bit_a_firing_rate_not_a_mandated_spike():
    """AUTHORITY.md §4.3: each input neuron is a Poisson process across the epoch."""
    import statistics

    def grid_with(drive, rate=0.1, off=0.0, seed=4):
        grid = GridOfNeurons(across=12, rows=2, weight=1.0, omega=0, permute=False, seed=seed)
        grid.coding, grid.interval = "population", 20.0
        grid.drive, grid.input_rate, grid.input_rate_off = drive, rate, off
        grid.set_input_bits([True, False, False, True])
        return grid

    forced = grid_with("forced")
    assert forced.input_schedule() == [(p, 0.0) for p in (0, 1, 2, 9, 10, 11)]  # one spike each, all at t_e
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
        grid = GridOfNeurons(across=12, rows=2, weight=None, seed=5, permute=False)
        grid.coding, grid.readout, grid.read, grid.interval = "population", "top", "fired", 20.0
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
    args = build_parser().parse_args(["--problem", "shallow_copy"])
    apply_problem(args)
    assert args.drive == C.INPUT_DRIVE == "forced"  # every problem today presents its input as it always has
    args = build_parser().parse_args(["--problem", "shallow_copy", "--drive", "rate", "--input-rate", "0.2"])
    apply_problem(args)
    assert args.drive == "rate" and args.input_rate == 0.2
