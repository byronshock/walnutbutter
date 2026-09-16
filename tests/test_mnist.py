"""The mnist problem (AUTHORITY.md §8): the loader, the label-carrying stream, goo's two zone widths and the class critic."""

from __future__ import annotations

import gzip
import random
import struct

import numpy as np
import pytest

from walnutbutter import fast, mnist
from walnutbutter.goo import Goo
from walnutbutter.learning import Teacher, class_accuracy, graded_accuracy, label_code
from walnutbutter.monitor import run_epoch
from walnutbutter.problems import PROBLEMS, dataset_stream


def idx(path, array: np.ndarray) -> None:
    """Write `array` as a gzipped IDX file of unsigned bytes, the way MNIST is distributed."""
    header = struct.pack(">HBB", 0, 0x08, array.ndim) + b"".join(struct.pack(">I", n) for n in array.shape)
    with gzip.open(path, "wb") as handle:
        handle.write(header + array.astype(np.uint8).tobytes())


def tiny(tmp_path):
    """Three 28 x 28 images with labels 7, 0, 7 under the distributed names: image k has its top-left 2 x 2 block at 255 * k / 2."""
    images = np.zeros((3, 28, 28), dtype=np.uint8)
    images[1, 0:2, 0:2] = 127   # a mean just under half: off
    images[2, 0:2, 0:2] = 255   # full: on
    images[2, 26:28, 26:28] = 200  # and the bottom-right block on too
    labels = np.array([7, 0, 7], dtype=np.uint8)
    (names_i, _, _), (names_l, _, _) = mnist.FILES["train"]
    idx(tmp_path / names_i, images)
    idx(tmp_path / names_l, labels)
    return images, labels


def test_the_idx_files_are_read_and_the_bits_are_the_blocks_at_half(tmp_path):
    images, labels = tiny(tmp_path)
    x, y = mnist.load("train", tmp_path)
    assert x.shape == (3, 28, 28) and y.tolist() == [7, 0, 7] and (x == images).all()
    bits, labels = mnist.bits_of("train", tmp_path)
    assert bits.shape == (3, 196) and bits.dtype == bool and labels.tolist() == [7, 0, 7]
    assert bits[0].sum() == 0
    assert bits[1].sum() == 0  # 127 / 255 is under half: off
    assert bits[2].sum() == 2 and bits[2][0] and bits[2][195]  # row-major: the first block and the last
    with pytest.raises(FileNotFoundError, match="not in"):
        mnist.load("train", tmp_path / "nowhere")
    with pytest.raises(ValueError, match="split"):
        mnist.load("validation", tmp_path)
    with pytest.raises(ValueError, match="split"):
        mnist.load("test", tmp_path)  # not a split this module knows: "we do not dare touch the test split"
    assert set(mnist.FILES) == {"train"}


def test_downsample_and_binarize_average_each_block():
    image = np.zeros((1, 28, 28), dtype=np.uint8)
    image[0, 4:6, 6:8] = [[255, 255], [0, 0]]  # a block whose mean is exactly half
    means = mnist.downsample(image)
    assert means.shape == (1, 14, 14) and means[0, 2, 3] == pytest.approx(0.5) and means.sum() == pytest.approx(0.5)
    bits = mnist.binarize(means)
    assert bits.shape == (1, 196) and bits[0, 2 * 14 + 3] and bits.sum() == 1  # at half is on


def test_the_stream_is_a_seeded_shuffle_with_the_labels_beside_it(tmp_path):
    tiny(tmp_path)
    patterns, labels = mnist.stream("train", 1, tmp_path)
    again, labels_again = mnist.stream("train", 1, tmp_path)
    other, _ = mnist.stream("train", 2, tmp_path)
    assert len(patterns) == 3 and len(labels) == 3 and sorted(labels) == [0, 7, 7]
    assert [patterns[k] for k in range(3)] == [again[k] for k in range(3)] and labels == labels_again
    assert any(patterns[k] != other[k] for k in range(3)) or labels != _  # a different seed, a different order
    bits, raw = mnist.bits_of("train", tmp_path)
    for k in range(3):  # each pattern is the image its label says
        assert patterns[k] in [row.tolist() for row in bits[raw == labels[k]]]
    assert all(isinstance(b, bool) for b in patterns[0]) and len(patterns[0]) == 196


def test_the_network_carries_the_label_and_the_class_critic_scores_it():
    goo = Goo(count=20, across=4, outputs=6, seed=3, weight=None)
    goo.coding, goo.population, goo.read, goo.rule = "raw", 3, "count", "reinforce"
    assert goo.output_width() == 6 and len(goo.output_row()) == 6 and len(goo.input_row()) == 4
    assert len(goo.interior()) == 10 and goo.zones_are_apart() and repr(goo).endswith("4 in, 6 out, zones apart)")
    patterns = [[True, False, True, False], [False, True, False, True]]
    goo.use_input_stream(patterns, [1, 0])
    run_epoch(goo, verbose=False, rng=random.Random(1))
    assert goo.input_label == 1
    assert label_code(goo) == [False, False, False, True, True, True]
    run_epoch(goo, verbose=False, rng=random.Random(1))
    assert goo.input_label == 0 and label_code(goo) == [True, True, True, False, False, False]
    run_epoch(goo, verbose=False, rng=random.Random(1))
    assert goo.input_label == 1  # round again
    # the critic, on counts set by hand: the label's group must out-spike every other, ties and silence lose
    outputs = goo.output_row()
    for counts, label, want in (([1, 1, 0, 3, 0, 0], 1, 1.0), ([1, 1, 0, 3, 0, 0], 0, 0.0), ([1, 1, 1, 3, 0, 0], 1, 0.0),
                                ([0, 0, 0, 0, 0, 0], 1, 0.0), ([0, 0, 0, 0, 0, 1], 1, 1.0)):
        for neuron, c in zip(outputs, counts):
            neuron.spikes_at_reset, neuron.spikes = 0, c
        goo.input_label = label
        assert class_accuracy(goo) == want
    # the graded critic (§8): the fraction of the other classes the label's class out-spikes; a tie is not beaten
    for counts, label, want in (([1, 1, 0, 3, 0, 0], 1, 1.0), ([1, 1, 0, 3, 0, 0], 0, 0.0), ([1, 1, 1, 3, 0, 0], 1, 0.0),
                                ([0, 0, 0, 0, 0, 0], 1, 0.0)):
        for neuron, c in zip(outputs, counts):
            neuron.spikes_at_reset, neuron.spikes = 0, c
        goo.input_label = label
        assert graded_accuracy(goo) == want
    goo.population = 2  # three classes of two: the label's class beats one of the other two
    for neuron, c in zip(outputs, [5, 0, 2, 2, 9, 0]):
        neuron.spikes_at_reset, neuron.spikes = 0, c
    goo.input_label = 0
    assert graded_accuracy(goo) == pytest.approx(1 / 2) and class_accuracy(goo) == 0.0
    goo.input_label = 2
    assert graded_accuracy(goo) == 1.0 and class_accuracy(goo) == 1.0
    goo.population = 3
    goo.input_label = None
    with pytest.raises(ValueError, match="labels"):
        class_accuracy(goo)
    with pytest.raises(ValueError, match="labels"):
        graded_accuracy(goo)
    with pytest.raises(ValueError, match="interior"):
        Goo(count=10, across=4, outputs=6)  # the zones talk only through an interior, so there has to be one
    with pytest.raises(ValueError, match="labels for"):
        goo.use_input_stream(patterns, [1])


@pytest.mark.parametrize("critic", ["class", "graded"])
def test_the_three_engines_agree_under_the_class_critic(critic):
    pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    rng = random.Random(4)
    patterns = [[rng.random() < 0.5 for _ in range(8)] for _ in range(50)]
    labels = [rng.randrange(2) for _ in range(50)]

    def make():
        g = Goo(count=40, across=8, outputs=6, seed=3, weight=None)
        g.coding, g.population, g.read, g.rule, g.drive = "raw", 3, "count", "reinforce", "rate"
        g.set_delta(0.455)
        g.use_input_stream(patterns, labels)
        return g

    mesh, twin = make(), make()
    net = ArrayNetwork(twin)
    teachers = [Teacher(x, seed=7, rule="reinforce", target="label", critic=critic, homeostasis=0.01, unstick=0.1)
                for x in (mesh, net)]
    rewards = []
    for _ in range(40):
        rewards.append([t.epoch(verbose=False) for t in teachers])
        assert rewards[-1][0] == rewards[-1][1] and 0.0 <= rewards[-1][0] <= 1.0
        assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
        assert net.input_label == mesh.input_label
    assert any(r[0] == 1.0 for r in rewards) and any(r[0] == 0.0 for r in rewards)
    if fast.available():
        g = make()
        teacher = Teacher(g, seed=7, rule="reinforce", target="label", critic=critic, homeostasis=0.01, unstick=0.1)
        assert fast.compare(g, epochs=40, teacher=teacher) == []
        g = make()
        mean, trace, engine, report = fast.train(g, 60, target="label", critic=critic, patterns=patterns, labels=labels,
                                                 eligibility="hazard", seed=7, trace_every=0)
        assert 0.0 <= mean <= 1.0 and g.input_at == 60  # the stream of 50 went round again


def test_the_mnist_problem_is_posed_on_goo_with_two_zone_widths():
    problem = PROBLEMS["mnist"]
    assert (problem.across, problem.outputs, problem.goo, problem.population, problem.clock) == (395, 50, 644, 5, 3)
    assert (problem.homeostasis, problem.unstick) == (0.0, 0.0)  # the hazard keeps nothing stuck; the un-sticking overshot
    from walnutbutter.cli import apply_problem, build_parser
    args = build_parser().parse_args(["--problem", "mnist"]); apply_problem(args)
    assert (args.homeostasis, args.unstick, args.goo, args.outputs, args.population) == (0.0, 0.0, 644, 50, 5)
    args = build_parser().parse_args(["--problem", "mnist", "--unstick", "0.01"]); apply_problem(args)
    assert args.unstick == 0.01 and args.homeostasis == 0.0  # given on the command line, it is kept
    assert (problem.target, problem.critic, problem.coding, problem.read, problem.data) == ("label", "graded", "complement", "count", "mnist")
    assert not problem.permute and problem.trained and problem.rule == "reinforce"
    assert dataset_stream(None, 1) is None
    with pytest.raises(ValueError, match="no dataset"):
        dataset_stream("cifar", 1)


@pytest.mark.skipif(not mnist.available(), reason="MNIST is not in mnist/")
def test_the_real_data_streams_and_a_short_run_scores(capsys):
    """With the folder filled: the shuffle covers the split, every label is a digit, and the command line runs the problem."""
    patterns, labels = mnist.stream("train", 1)
    assert len(patterns) == 60_000 and len(labels) == 60_000 and set(labels) == set(range(10))
    assert len(patterns[0]) == 196 and 0 < sum(patterns[0]) < 196
    from walnutbutter.cli import cli_main
    assert cli_main(["--headless", "--problem", "mnist", "--seed", "1", "--epochs", "5", "--no-save", "-q"]) == 0
    err = capsys.readouterr().err
    assert "Goo(644 neurons" in err and "395 in, 50 out" in err and "images of mnist" in err and "learning label (hazard" in err
    assert "clock neurons: the first 3" in err


def test_clock_neurons_lead_the_input_zone_and_fire_every_epoch():
    """§4.3, Byron: 'Clock neurons can be created for a task as input neurons always driven by 1.'"""
    pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        g = Goo(count=30, across=8, outputs=4, seed=3, weight=None, permute=False)
        g.coding, g.clock, g.drive, g.read = "complement", 2, "rate", "count"
        return g

    goo = make()
    assert goo.raw_bit_count() == 3  # eight places: two clocks, then three bits and their complements
    goo.set_input_bits([True, False, True])
    assert goo.input_coded == [True, True, True, False, True, False, True, False]
    assert [n.should_fire for n in goo.input_row()][:2] == [True, True]
    twin = make()
    net = ArrayNetwork(twin)
    rng_a, rng_b = random.Random(1), random.Random(1)
    for _ in range(6):
        run_epoch(goo, verbose=False, rng=rng_a)
        run_epoch(net, verbose=False, rng=rng_b)
        clocks = goo.input_row()[:2]
        assert all(n.epoch_spikes >= 1 for n in clocks)  # driven every epoch (a clock the mesh fired first still fired)
        assert [n.spikes for n in goo.all_neurons()] == net.spikes.tolist()
    from walnutbutter.persistence import checkpoint, restore
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as folder:
        data = checkpoint(goo, pathlib.Path(folder) / "clock.json")
        assert data["clock"] == 2
        back, _ = restore(pathlib.Path(folder) / "clock.json")
        assert back.clock == 2 and back.raw_bit_count() == 3


def test_a_checkpoint_keeps_the_output_zone(tmp_path):
    from walnutbutter.persistence import checkpoint, restore
    goo = Goo(count=20, across=4, outputs=6, seed=3, weight=None, projection=0.5)
    run_epoch(goo, verbose=False)
    data = checkpoint(goo, tmp_path / "zones.json")
    assert data["outputs"] == 6
    back, _ = restore(tmp_path / "zones.json")
    assert back.outputs == 6 and len(back.output_row()) == 6 and back.zones_are_apart()
    assert [c.weight for c in back.connections.values()] == [c.weight for c in goo.connections.values()]
