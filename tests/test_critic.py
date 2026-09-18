import pytest

from walnutbutter.goo import Goo
from walnutbutter.inputs import HAMMING74, complement_code
from walnutbutter.learning import (
    CRITICS, Teacher, decoded_accuracy, decoded_exact, decoded_output, expected_data, output_row, read_output_word, reward,
)
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def hamming_grid(seed=1):
    grid = Goo(count=56, across=14, seed=seed)
    grid.use_ecc()
    grid.set_input_bits([True, False, True, True])
    return grid


def show_on_output(grid, pattern):
    """Force the output row to display `pattern` (as fired flags)."""
    for neuron, bit in zip(output_row(grid), pattern):
        neuron.reset()
        if bit:
            neuron.fire(wave=3)


def test_reading_the_perfect_output_recovers_the_data():
    grid = hamming_grid()
    target = grid.input_pattern[::-1]  # the reversed target, exactly
    show_on_output(grid, target)
    assert read_output_word(grid) == HAMMING74.encode([True, False, True, True])
    assert decoded_output(grid) == [True, False, True, True] == expected_data(grid)
    assert decoded_accuracy(grid) == 1.0 and decoded_exact(grid) == 1.0
    assert reward(grid, critic="decoded") == 1.0 and reward(grid, critic="row") == 1.0


def test_one_wrong_output_neuron_still_decodes_correctly_under_hamming():
    grid = hamming_grid()
    target = grid.input_pattern[::-1]
    for j in range(14):
        wrong = list(target)
        wrong[j] = not wrong[j]
        show_on_output(grid, wrong)
        assert reward(grid, critic="row") == pytest.approx(13 / 14)  # the row critic docks it
        assert decoded_exact(grid) == 1.0  # the decoding critic does not: the code absorbs one error
    # a pair whose two neurons contradict each other is an unreadable bit, and one of those is absorbed too
    word = read_output_word(grid)
    assert None in word or all(b is not None for b in word)


def test_two_wrong_outputs_can_defeat_the_code_and_the_critics_disagree():
    grid = hamming_grid()
    target = grid.input_pattern[::-1]
    worst = None
    for a in range(14):
        for b in range(a + 1, 14):
            wrong = list(target)
            wrong[a], wrong[b] = not wrong[a], not wrong[b]
            show_on_output(grid, wrong)
            worst = min(worst if worst is not None else 1.0, decoded_accuracy(grid))
    assert worst < 1.0  # some double errors get through
    assert reward(grid, critic="row") == pytest.approx(12 / 14)


def test_decoding_critic_without_a_code_compares_raw_bits():
    grid = Goo(count=32, across=8, seed=1)
    grid.set_input_bits([True, True, False, False])
    show_on_output(grid, grid.input_pattern[::-1])
    assert decoded_output(grid) == [True, True, False, False] and decoded_exact(grid) == 1.0
    assert expected_data(grid) == grid.input_bits


def test_critics_registry_and_teacher_validation():
    assert set(CRITICS) == {"row", "sustained", "decoded", "decoded-exact", "population", "class", "graded", "evidence"}
    grid = hamming_grid()
    run_epoch(grid, verbose=False)
    teacher = Teacher(grid, critic="decoded", seed=1)
    teacher.step()
    assert "critic decoded" in teacher.status()
    assert "critic" not in Teacher(grid, seed=1).status()
    with pytest.raises(ValueError):
        Teacher(grid, critic="oracle")
    with pytest.raises(ValueError):
        Teacher(grid, critic="decoded", target="all-off")


def test_teacher_with_decoding_critic_learns_something(capsys):
    import statistics
    grid = Goo(count=56, across=14, weight=None, seed=2)
    grid.use_ecc()
    teacher = Teacher(grid, critic="decoded", lr=0.1, seed=2)
    rewards = [teacher.epoch(verbose=False) for _ in range(1500)]
    assert all(0 <= r <= 1 for r in rewards) and set(rewards) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    assert statistics.mean(rewards[-300:]) >= statistics.mean(rewards[:300]) - 0.05  # at least not getting worse


def test_cli_critic_option_and_checkpoint(tmp_path, capsys):
    from pathlib import Path
    from walnutbutter.cli import cli_main
    from walnutbutter.persistence import read_checkpoint
    assert cli_main(["--headless", "--ecc", "--critic", "decoded", "--seed", "1", "-q", "--epochs", "5"]) == 0
    err = capsys.readouterr().err
    assert "critic decoded" in err
    f = next(Path("runs").glob("*-seed1.json"))
    assert read_checkpoint(f)["learning"]["critic"] == "decoded"
    with pytest.raises(SystemExit):
        cli_main(["--headless", "--critic", "oracle"])


# --- the rate read (AUTHORITY.md §4.3, §6.9) -----------------------------------------

def test_the_rate_read_estimates_hz_over_an_exponential_window():
    import math
    from walnutbutter.constants import RATE_ON, RATE_TAU, REFRACTORY
    from walnutbutter.neuron import Neuron

    assert RATE_TAU == REFRACTORY == 5.0 and RATE_ON == 1000.0 / REFRACTORY == 200.0  # saturation is 1/REFRACTORY
    n = Neuron("n")
    assert n.firing_rate(20.0) == 0.0  # no spikes is a rate of zero
    n.fire(0, 20.0)
    assert n.firing_rate(20.0) == pytest.approx(RATE_ON)  # one spike at the read reads exactly RATE_ON
    assert n.firing_rate(25.0) == pytest.approx(RATE_ON * math.exp(-1.0))
    n.fire(0, 25.0)  # a second spike stacks on what is left of the first
    assert n.firing_rate(25.0) == pytest.approx(RATE_ON * (1 + math.exp(-1.0)))

    sat = Neuron("sat")  # the fastest train the refractory period allows
    for k in range(40):
        sat.fire(0, k * REFRACTORY)
    t0 = 39 * REFRACTORY
    across = [sat.firing_rate(t0 + f * REFRACTORY / 200) for f in range(200)]
    assert sum(across) / len(across) == pytest.approx(RATE_ON, rel=0.01)  # unbiased in time-average
    assert min(across) < 120 and max(across) > 300  # but it swings widely within one interval


def test_the_rate_read_grades_the_score_and_leaves_a_boolean_read_alone():
    from walnutbutter.goo import Goo
    from walnutbutter.learning import accuracy

    grid = Goo(count=12, across=4, weight=1.0, permute=False)
    grid.coding, grid.readout = "raw", "top"
    grid.set_input_bits([True, True, False, False])
    grid.horizon = 20.0

    grid.read = "fired"  # unchanged: levels are 0.0 and 1.0 and the scores are what they always were
    for neuron, on in zip(grid.output_row(), [True, True, False, False]):
        neuron.has_fired = on
    assert grid.output_levels() == [1.0, 1.0, 0.0, 0.0]
    assert accuracy(grid, "copy") == pytest.approx(1.0)

    grid.read = "rate"
    for neuron, rate in zip(grid.output_row(), [200.0, 100.0, 0.0, 400.0]):
        neuron.rate_level, neuron.rate_at = rate, 20.0
    assert grid.output_levels() == [1.0, 0.5, 0.0, 1.0]  # clipped at saturation
    assert grid.output_fired() == [True, True, False, True]  # half of saturation reads as on
    assert accuracy(grid, "copy") == pytest.approx((1.0 + 0.5 + 1.0 + 0.0) / 4)


def test_the_rate_read_is_bit_identical_across_the_engines():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter.goo import Goo
    from walnutbutter.learning import accuracy
    from walnutbutter.monitor import run_epoch
    from walnutbutter.neuron import Neuron

    Neuron.verbose = False

    def make():
        grid = Goo(count=60, across=12, weight=None, seed=5, permute=False)
        grid.coding, grid.readout, grid.read, grid.interval = "population", "top", "rate", 20.0
        grid.drive, grid.quash_rate = "rate", 0.02
        return grid

    mesh, net = make(), ArrayNetwork(make())
    seen = set()
    for _ in range(30):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert mesh.output_rates() == net.output_rates()  # exactly, not approximately
        assert accuracy(mesh, "copy") == accuracy(net, "copy")
        seen.update(round(level, 6) for level in mesh.output_levels())
    assert len(seen - {0.0, 1.0}) > 3  # the read is genuinely graded, not a bit in disguise


# --- the kinder teacher (AUTHORITY.md §6.13) -----------------------------------------

def test_the_majority_vote_reads_a_population_code():
    from walnutbutter.learning import population_vote

    assert population_vote([True] * 3, 3) == [True]     # three is a 1
    assert population_vote([True, True, False], 3) == [True]   # two of three is a 1
    assert population_vote([True, False, False], 3) == [False]  # one of three is a 0
    assert population_vote([False] * 3, 3) == [False]   # none is a 0
    assert population_vote([True, False, True, False, False, False], 3) == [True, False]


