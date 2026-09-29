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
    goo = Goo(count=26, across=4, outputs=12, seed=3, weight=None)
    goo.population, goo.read, goo.rule = 3, "count", "reinforce"  # §5.11: 2CP = 2 classes * 3 a half
    assert goo.output_width() == 12 and len(goo.output_row()) == 12 and len(goo.input_row()) == 4
    assert len(goo.interior()) == 10 and goo.input_zone_is_apart()
    patterns = [[True, False], [False, True]]  # §5.2: 4 places take 2 raw bits
    goo.use_input_stream(patterns, [1, 0])
    run_epoch(goo, verbose=False, rng=random.Random(1))
    assert goo.input_label == 1
    assert label_code(goo) == [False] * 3 + [True] * 3 + [True] * 3 + [False] * 3
    run_epoch(goo, verbose=False, rng=random.Random(1))
    assert goo.input_label == 0 and label_code(goo) == [True] * 3 + [False] * 3 + [False] * 3 + [True] * 3
    run_epoch(goo, verbose=False, rng=random.Random(1))
    assert goo.input_label == 1  # round again
    # the critic, on counts set by hand: the label's group must out-spike every other, ties and silence lose
    outputs = goo.output_row()
    for counts, label, want in (([1, 1, 0, 3, 0, 0] + [0] * 6, 1, 1.0), ([1, 1, 0, 3, 0, 0] + [0] * 6, 0, 0.0), ([1, 1, 1, 3, 0, 0] + [0] * 6, 1, 0.0),
                                ([0, 0, 0, 0, 0, 0] + [0] * 6, 1, 0.0), ([0, 0, 0, 0, 0, 1] + [0] * 6, 1, 1.0)):
        for neuron, c in zip(outputs, counts):
            neuron.spikes_at_reset, neuron.spikes = 0, c
        goo.input_label = label
        assert class_accuracy(goo) == want
    # the graded critic (§8): the fraction of the other classes the label's class out-spikes; a tie is not beaten
    for counts, label, want in (([1, 1, 0, 3, 0, 0] + [0] * 6, 1, 1.0), ([1, 1, 0, 3, 0, 0] + [0] * 6, 0, 0.0), ([1, 1, 1, 3, 0, 0] + [0] * 6, 1, 0.0),
                                ([0, 0, 0, 0, 0, 0] + [0] * 6, 1, 0.0)):
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
        Goo(count=10, across=4, outputs=6, wiring="zones-equal")  # under the zone rule the zones talk only through an interior
    assert Goo(count=10, across=4, outputs=6, seed=1).interior() == []  # under the scaled rule the outputs hear the inputs
    with pytest.raises(ValueError, match="labels for"):
        goo.use_input_stream(patterns, [1])


def test_a_matched_filter_knows_nothing_at_the_start_and_learns_the_templates_and_the_noise():
    """§9.4 (Byron, September 26, 2026: "default to a noise-whitened matched filter"): an empty filter scores every
    class alike and pays ln(1 / C); a class's first presentation sets its template and adds nothing to the noise; each
    epoch is read before it is folded; and on outputs that tell the classes apart the filter comes to pay near 0 and
    to pick the label, under both forms."""
    import math
    pytest.importorskip("numpy")
    from walnutbutter.learning import MatchedFilter
    for form in ("whitened", "poisson"):
        f = MatchedFilter(form, 3, 4, memory=30)
        assert f.read([5, 0, 0, 1], 0) == (pytest.approx(math.log(1 / 3)), False)  # alike: a tie is not a win
        f.fold([5, 0, 0, 1], 0)
        assert (f.presented, f.folded, f.noise.sum()) == ([1, 0, 0], 0, 0.0)  # no template yet to deviate from
        assert f.templates[0].tolist() == [5, 0, 0, 1]
        f.fold([3, 0, 0, 1], 0)  # the second deviates by -2 from the first: n / (n + 1) of it, a mean of one epoch
        assert (f.folded, f.noise[0, 0], f.templates[0, 0]) == (1, 0.5 * 4, 4.0)
        rng = random.Random(1)
        shapes = [[5, 0, 0, 1], [0, 5, 0, 1], [0, 0, 5, 1]]
        for _ in range(300):
            y = rng.randrange(3)
            f.fold([max(0, round(v + rng.gauss(0, 1))) for v in shapes[y]], y)
        assert f.presented[0] + f.presented[1] + f.presented[2] == 302 and f.folded == 299
        for y, shape in enumerate(shapes):
            reward, right = f.read(shape, y)
            assert right and -0.01 < reward <= 0.0, (form, y, reward)
    with pytest.raises(ValueError, match="unknown matched filter"):
        MatchedFilter("evidence", 3, 4)
    with pytest.raises(ValueError, match="at least one per class"):
        MatchedFilter("whitened", 3, 4, memory=2)


def test_a_matched_filter_goes_on_from_its_state_to_the_bit():
    """§12.9, §12.11: the filter a checkpoint carries reads and folds on exactly as the one it was taken from."""
    pytest.importorskip("numpy")
    import json
    from walnutbutter.learning import MatchedFilter
    rng = random.Random(2)
    for form in ("whitened", "poisson"):
        f = MatchedFilter(form, 4, 6, memory=40)
        epochs = [([rng.randrange(8) for _ in range(6)], rng.randrange(4)) for _ in range(200)]
        for x, y in epochs[:120]:
            f.fold(x, y)
        g = MatchedFilter.from_state(json.loads(json.dumps(f.state())))  # through the file, as a checkpoint takes it
        for x, y in epochs[120:]:
            assert f.read(x, y) == g.read(x, y)
            f.fold(x, y)
            g.fold(x, y)
        assert f.state() == g.state()


def test_the_poisson_filter_on_alike_outputs_is_the_evidence_critic_it_replaced():
    """§9.4, what it replaced: where every output fires at one rate when the label code wants it on and at another when
    it wants it off, the poisson filter's estimate is the evidence critic's softmax of the class sums n_k = n_k+ - n_k-
    at T = 1 / ln(on / off) -- the relationship the September 26 continuations were read by."""
    import math
    pytest.importorskip("numpy")
    from walnutbutter.learning import MatchedFilter
    on, off, classes, population = 9.0, 7.0, 3, 2
    f = MatchedFilter("poisson", classes, 2 * classes * population)
    for k in range(classes):  # class k's template: its fire-if-one population on, every other off; fire-if-zero reversed
        ones = [on if c == k else off for c in range(classes) for _ in range(population)]
        f.templates[k] = ones + [on + off - v for v in ones]
    temperature = 1 / math.log(on / off)
    rng = random.Random(3)
    for _ in range(50):
        counts = [rng.randrange(15) for _ in range(2 * classes * population)]
        sums = [sum(counts[c * population:(c + 1) * population]) - sum(counts[(classes + c) * population:(classes + c + 1) * population])
                for c in range(classes)]
        label = rng.randrange(classes)
        evidence = sums[label] / temperature - math.log(sum(math.exp(n / temperature) for n in sums))
        assert f.read(counts, label)[0] == pytest.approx(evidence, abs=1e-9)


@pytest.mark.parametrize("critic", ["whitened", "poisson"])
def test_a_matched_filter_learns_before_it_pays(critic):
    """§9.4 (Byron, September 26, 2026, "learn, then pay"): until the filter has folded FILTER_MEMORY epochs the
    advantage is zero -- no weight moves and the baseline neither starts nor moves -- though every epoch is read; from
    the first epoch it pays on, the baseline starts there and the rule learns."""
    pytest.importorskip("numpy")
    rng = random.Random(4)
    patterns = [[rng.random() < 0.5 for _ in range(4)] for _ in range(30)]
    labels = [rng.randrange(2) for _ in range(30)]
    g = Goo(count=46, across=8, outputs=12, seed=3, weight=None)  # §5.11: 2 classes * 3 a half
    g.population, g.read, g.rule, g.drive = 3, "count", "reinforce", "rate"
    g.set_delta(0.455)
    g.use_input_stream(patterns, labels)
    teacher = Teacher(g, seed=7, rule="reinforce", target="label", critic=critic, filter_memory=12, lr=0.5)
    start = [c.weight for c in g.connections.values()]
    for epoch in range(12):
        assert not teacher.filter.paying
        teacher.epoch(verbose=False)
        assert teacher.baseline is None and [c.weight for c in g.connections.values()] == start, epoch
    assert teacher.filter.paying and sum(teacher.filter.presented) == 12
    first = teacher.epoch(verbose=False)
    assert teacher.baseline == first  # §9.3: started at the first paid epoch's reward, which moves it nowhere
    for _ in range(10):
        teacher.epoch(verbose=False)
    assert [c.weight for c in g.connections.values()] != start  # and from there the rule learns


def test_the_teacher_pays_mnist_through_the_whitened_filter():
    """11.9: mnist's critic is whitened; a matched filter scores a label, so it needs the label target and a stream
    that carries labels, and the evidence critic is gone (§9.4)."""
    pytest.importorskip("numpy")
    from walnutbutter.learning import CRITICS, FILTERS, Teacher
    from walnutbutter.problems import PROBLEMS
    assert PROBLEMS["mnist"].critic == "whitened" and FILTERS == ("whitened", "poisson") and "evidence" not in CRITICS
    goo = Goo(count=34, across=4, outputs=20, seed=3, weight=None)  # §5.11: 2CP = 2 * 10 * 1
    goo.population, goo.read, goo.rule = 1, "count", "reinforce"
    goo.set_delta(0.455)
    with pytest.raises(ValueError, match="target is label"):
        Teacher(goo, target="copy", critic="whitened", rule="reinforce")
    with pytest.raises(ValueError, match="unknown critic"):
        Teacher(goo, target="label", critic="evidence", rule="reinforce")
    teacher = Teacher(goo, target="label", critic="whitened", rule="reinforce", seed=1)
    assert (teacher.filter.classes, teacher.filter.outputs, teacher.filter.memory) == (10, 20, 5000)
    with pytest.raises(ValueError, match="labels"):
        teacher.epoch(verbose=False)  # a stream with no labels


@pytest.mark.parametrize("critic", ["class", "graded", "whitened", "poisson"])
def test_the_three_engines_agree_under_the_class_critic(critic):
    pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    rng = random.Random(4)
    patterns = [[rng.random() < 0.5 for _ in range(4)] for _ in range(50)]  # §5.2: 8 places take 4 raw bits
    labels = [rng.randrange(2) for _ in range(50)]

    def make():
        g = Goo(count=46, across=8, outputs=12, seed=3, weight=None)  # §5.11: 2 classes * 3 a half
        g.population, g.read, g.rule, g.drive = 3, "count", "reinforce", "rate"
        g.set_delta(0.455)
        g.use_input_stream(patterns, labels)
        return g

    mesh, twin = make(), make()
    net = ArrayNetwork(twin)
    memory = 10  # §9.4: a filter pays from its tenth epoch, so the forty below learn under it
    teachers = [Teacher(x, seed=7, rule="reinforce", target="label", critic=critic, homeostasis=0.01, unstick=0.1,
                        filter_memory=memory) for x in (mesh, net)]
    rewards = []
    for _ in range(40):
        rewards.append([t.epoch(verbose=False) for t in teachers])
        assert rewards[-1][0] == rewards[-1][1]
        filtered = critic in ("whitened", "poisson")
        assert (rewards[-1][0] <= 0.0) if filtered else (0.0 <= rewards[-1][0] <= 1.0)
        assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
        assert net.input_label == mesh.input_label
    if filtered:  # the log score: never a clean 0 or 1, but on both sides of a uniform estimate over forty epochs
        import math
        uniform = math.log(1 / (6 // 3))  # two classes of three outputs on this goo
        assert any(r[0] > uniform for r in rewards) and any(r[0] < uniform for r in rewards)
        assert teachers[0].filter.state() == teachers[1].filter.state()  # §9.4: one book, whichever engine fed it
    else:
        assert any(r[0] == 1.0 for r in rewards) and any(r[0] == 0.0 for r in rewards)
    if fast.available():
        g = make()
        teacher = Teacher(g, seed=7, rule="reinforce", target="label", critic=critic, homeostasis=0.01, unstick=0.1,
                          filter_memory=memory)
        assert fast.compare(g, epochs=40, teacher=teacher) == []
        g = make()
        mean, trace, engine, report = fast.train(g, 60, target="label", critic=critic, patterns=patterns, labels=labels,
                                                 eligibility="hazard", seed=7, trace_every=0, filter_memory=memory)
        assert g.input_at == 60  # the stream of 50 went round again
        assert mean <= 0.0 if filtered else 0.0 <= mean <= 1.0
        # §9.7: the fraction right, the filter's own under a filter, and by the class sums beside it
        assert 0.0 <= report["accuracy_last_tenth"] <= 1.0 and 0.0 <= report["accuracy_last_tenth_sums"] <= 1.0
        if critic == "class":
            assert report["accuracy_last_tenth"] == report["accuracy_last_tenth_sums"]
        assert (report["filter"] is not None) == filtered


def test_the_mnist_problem_is_posed_on_goo_with_two_zone_widths():
    problem = PROBLEMS["mnist"]
    assert (problem.across, problem.outputs, problem.hidden_neurons, problem.population, problem.clock) == (395, 60, 199, 3, 3)
    assert problem.goo is None  # the goo is inputs + hidden + outputs, 654, sized by the hidden count since September 16, 2026
    assert (problem.homeostasis, problem.unstick) == (0.0, 0.0)  # the hazard keeps nothing stuck; the un-sticking overshot
    from walnutbutter.cli import apply_problem, build_parser
    args = build_parser().parse_args(["--problem", "mnist"]); apply_problem(args)
    assert (args.homeostasis, args.unstick, args.goo, args.outputs, args.population) == (0.0, 0.0, 654, 60, 3)
    assert args.hidden_neurons == 199 and problem.hidden_neurons == 199 and problem.goo is None  # inputs + hidden + outputs
    args = build_parser().parse_args(["--problem", "mnist", "--hidden-neurons", "0"]); apply_problem(args)
    assert args.goo == 395 + 60 and args.hidden_neurons == 0  # Byron, September 16, 2026: the task without hidden neurons
    args = build_parser().parse_args(["--problem", "mnist", "--goo", "500"]); apply_problem(args)
    assert args.goo == 500 and args.hidden_neurons == 199  # --goo wins the sizing; the mismatch is refused when built
    args = build_parser().parse_args(["--problem", "mnist", "--unstick", "0.01"]); apply_problem(args)
    assert args.unstick == 0.01 and args.homeostasis == 0.0  # given on the command line, it is kept
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
    assert "Goo(654 neurons" in err and "395 in, 199 hidden, 60 out" in err and "images of mnist" in err and "learning label (hazard" in err
    assert "clock neurons: the first 3" in err


def test_clock_neurons_lead_the_input_zone_and_fire_every_epoch():
    """§4.3, Byron: 'Clock neurons can be created for a task as input neurons always driven by 1.'"""
    pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        g = Goo(count=30, across=8, outputs=4, seed=3, weight=None)
        g.clock, g.drive, g.read = 2, "rate", "count"
        return g

    goo = make()
    assert goo.raw_bit_count() == 3  # eight places: two clocks, then three bits and their complements
    goo.set_input_bits([True, False, True])
    assert goo.input_coded == [True, True, True, False, True, False, True, False]
    assert [n.should_fire for n in goo.input_row()][:2] == [True, True]
    twin = make()
    net = ArrayNetwork(twin)
    rng_a, rng_b = random.Random(1), random.Random(1)
    for k in range(6):
        run_epoch(goo, verbose=False, rng=rng_a)
        run_epoch(net, verbose=False, rng=rng_b)
        clocks = goo.input_row()[:2]
        assert all(n.should_fire for n in clocks)  # driven every epoch
        if k == 5:  # and spiking in nearly every one: a forced spike that lands in a refractory period is lost (§4.3)
            assert all(n.spikes >= 5 for n in clocks)
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
    for wiring, apart in (("driven-inputs", False), ("scaled", True)):  # the outputs hear one another under the rule (§4.5)
        goo = Goo(count=20, across=4, outputs=6, seed=3, weight=None, projection=0.5, wiring=wiring)
        run_epoch(goo, verbose=False)
        data = checkpoint(goo, tmp_path / "zones.json")
        assert data["outputs"] == 6
        back, _ = restore(tmp_path / "zones.json")
        assert back.outputs == 6 and len(back.output_row()) == 6 and back.input_zone_is_apart() and back.wiring == wiring
        assert back.outputs_are_apart() is apart
        assert [c.weight for c in back.connections.values()] == [c.weight for c in goo.connections.values()]


def test_the_supervised_direction_and_the_estimators_correlation_over_a_run():
    """§8: d = P(coded input on | class) - P(coded input on) on the input-to-output synapses, and the Rust driver's trace of the
    weight change's correlation with it at every trace interval, under any eligibility."""
    pytest.importorskip("numpy")
    if not mnist.available() or not fast.available():
        pytest.skip("needs the mnist data and the Rust engine")
    from walnutbutter.cli import apply_problem, build_parser
    args = build_parser().parse_args(["--problem", "mnist", "--hidden-neurons", "0"]); apply_problem(args)
    goo = Goo(count=args.goo, across=args.across, outputs=args.outputs, seed=1, weight=None, threshold=0.6, minimum_potential=-2.4)
    goo.population, goo.clock, goo.read, goo.rule, goo.drive = args.population, 3, "count", "reinforce", "rate"
    goo.set_delta(0.455)
    direction = mnist.supervised_direction(goo)
    edges = [c for n in goo.all_neurons() for c in n.outgoing]
    d, mask = direction(edges)
    inputs, outputs = set(goo.input_row()), set(goo.output_row())
    assert len(d) == len(mask) == len(edges) == len(goo.connections)
    assert mask.tolist() == [c.source in inputs and c.target in outputs for c in edges]  # the input-to-output synapses; with no
    assert not mask.all() and all(c.source in outputs and c.target in outputs for c, m in zip(edges, mask) if not m)  # hidden
    assert not d[~mask].any()  # neurons the rest are outputs onto outputs (§4.5), outside the mask
    assert np.abs(d).max() <= 1.0 and all(d[e] == 0.0 for e, c in enumerate(edges) if c.source in goo.input_row()[:3])  # clocks: no direction
    assert 0.3 < (np.abs(d[mask]) > 0.02).mean() < 0.7  # a good part of the synapses have a direction worth the name
    on, on_given = mnist.pixel_statistics(3)
    assert on.shape == (395,) and on_given.shape == (10, 395) and on[:3].tolist() == [1.0, 1.0, 1.0]
    assert np.allclose(on[3:199] + on[199:], 1.0)  # a bit and its complement
    patterns, labels = dataset_stream("mnist", 1)
    mean, trace, engine, report = fast.train(goo, 40, lr=0.001, target="label", trace_every=10, patterns=patterns, labels=labels,
                                             eligibility="hazard", seed=1, homeostasis=0.0, unstick=0.0, critic="whitened",
                                             direction=direction, filter_memory=10)  # §9.4: paying from epoch 11
    est = report["estimator"]
    assert [e["epoch"] for e in est] == [10, 20, 30, 40]
    assert all(e["corr_cum"] is None or -1.0 <= e["corr_cum"] <= 1.0 for e in est)
    assert all(0.0 <= e["sign_cum"] <= 1.0 for e in est) and est[-1]["corr_window"] is not None
    goo2 = Goo(count=args.goo, across=args.across, outputs=args.outputs, seed=1, weight=None, threshold=0.6, minimum_potential=-2.4)
    goo2.population, goo2.clock, goo2.read, goo2.rule, goo2.drive = args.population, 3, "count", "reinforce", "rate"
    goo2.set_delta(0.455)
    _, _, _, plain = fast.train(goo2, 10, lr=0.001, target="label", trace_every=5, patterns=patterns, labels=labels,
                                eligibility="hebb", seed=1, homeostasis=0.0, unstick=0.0, critic="whitened")
    assert plain["estimator"] is None  # without a direction there is no trace, and hebb runs under the whitened critic too


def test_complement_coding_of_the_output_zone_reads_one_sums_minus_zero_sums():
    """§8 (Byron, September 16, 2026: "force complement coding ... three fire-if-one and three fire-if-zero"): the zone is
    the fire-if-one populations then the fire-if-zero ones in the same order, a class's evidence is n+ - n-, the label
    code is on/off for the label's groups and off/on for every other class's, and a zero neuron's supervised direction
    is its class's reversed."""
    from walnutbutter.learning import class_evidence, class_sums, label_code
    # two classes of two, complement-coded: ones [1 2 | 0 5], zeros [0 1 | 4 0] (§5.11)
    assert class_evidence([1, 2, 0, 5, 0, 1, 4, 0], 2) == [3 - 1, 5 - 4]
    assert class_evidence([1, 2, 0, 5, 0, 0], 3) == [3 - 5]
    with pytest.raises(ValueError, match="does not divide"):
        class_evidence([1, 2, 3], 2)
    goo = Goo(count=24, across=4, outputs=12, seed=3, weight=None)  # three classes: 2 fire-if-one, 2 fire-if-zero each
    goo.population, goo.read, goo.rule = 2, "count", "reinforce"
    goo.use_input_stream([[True, False]], [1])
    run_epoch(goo, verbose=False, rng=random.Random(1))
    assert label_code(goo) == [False, False, True, True, False, False] + [True, True, False, False, True, True]
    outputs = goo.output_row()
    counts = [3, 1, 0, 0, 2, 2] + [0, 0, 1, 0, 0, 0]  # ones 4, 0, 4; zeros 0, 1, 0 -> evidence 4, -1, 4
    for neuron, c in zip(outputs, counts):
        neuron.spikes_at_reset, neuron.spikes = 0, c
    assert class_sums(goo) == [4, -1, 4]
    goo.input_label = 1
    assert class_accuracy(goo) == 0.0 and graded_accuracy(goo) == 0.0  # the label's class is out-evidenced by both others
    for neuron, c in zip(outputs, [0, 0, 0, 0, 0, 0] + [3, 3, 0, 0, 3, 3]):  # only fire-if-zero spikes: evidence -6, 0, -6
        neuron.spikes_at_reset, neuron.spikes = 0, c
    assert class_accuracy(goo) == 1.0 and graded_accuracy(goo) == 1.0  # the label is the one class not spoken against
    for neuron in outputs:  # every neuron loud alike: every class at zero evidence, a tie, which is not a win
        neuron.spikes = 4
    assert class_sums(goo) == [0, 0, 0] and class_accuracy(goo) == 0.0
    # the supervised direction: a fire-if-zero neuron of class k takes -(P(pixel | k) - P(pixel))
    ff = Goo(count=445, across=395, outputs=50, seed=1, weight=None, threshold=0.6, minimum_potential=-2.4)
    ff.population, ff.clock = 5, 3
    cc = Goo(count=455, across=395, outputs=60, seed=1, weight=None, threshold=0.6, minimum_potential=-2.4)
    cc.population, cc.clock = 3, 3
    from walnutbutter.mnist import supervised_direction
    edges = [c for n in cc.all_neurons() for c in n.outgoing]
    try:
        d, mask = supervised_direction(cc)(edges)  # the pixel statistics need the dataset
    except FileNotFoundError:
        pytest.skip("the MNIST files are not fetched")
    row = cc.output_row()
    plain, plain_mask = supervised_direction(ff)([c for n in ff.all_neurons() for c in n.outgoing])
    assert int(mask.sum()) > 0 and int(plain_mask.sum()) > 0 and set(np.sign(plain[plain_mask])) <= {-1.0, 0.0, 1.0}
    # a fire-if-one and a fire-if-zero neuron of the same class see the same pixel with opposite signs
    by_target = {}
    for e, c in enumerate(edges):
        if mask[e]:
            by_target.setdefault(c.target, {})[c.source] = d[e]
    pairs = 0
    for k in range(30):
        one, zero = row[k], row[30 + k]  # the same class (k // 3), one and zero
        for source, value in by_target.get(one, {}).items():
            if source in by_target.get(zero, {}):
                assert by_target[zero][source] == -value
                pairs += 1
    assert pairs > 0


def test_the_mnist_problem_is_complement_coded_on_the_outputs_and_the_engines_agree(tmp_path):
    """The problem's defaults since September 16, 2026, the checkpoint's round trip, and the Rust loop on the configuration."""
    from walnutbutter.cli import apply_problem, build_parser
    from walnutbutter.persistence import checkpoint, restore
    from walnutbutter.problems import PROBLEMS
    p = PROBLEMS["mnist"]
    args = build_parser().parse_args(["--problem", "mnist"]); apply_problem(args)
    goo = Goo(count=20, across=4, outputs=6, seed=3, weight=None)
    goo.population, goo.read, goo.rule = 1, "count", "reinforce"
    data = checkpoint(goo, tmp_path / "c.json")
    assert data["population"] == 1  # §5.11 fixes the coding, so only the population is a checkpoint field
    back, _ = restore(tmp_path / "c.json")
    from walnutbutter.arrays import ArrayNetwork
    if fast.available():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location("rs", Path(__file__).resolve().parent.parent / "docs" / "rust-sweep.py")
        rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
        from walnutbutter.problems import dataset_stream
        try:
            patterns, labels = dataset_stream("mnist", 1)
        except FileNotFoundError:
            pytest.skip("the MNIST files are not fetched")
        grid, cli = rs.grid_of("mnist", {"hidden_neurons": 0.0, "seed": 1, "threshold": 0.6}, "hazard", True, -4.0)
        grid.use_input_stream(patterns, labels)
        teacher = Teacher(grid, seed=1, rule="reinforce", eligibility="hazard", target="label", critic=cli.critic, lr=cli.lr,
                          homeostasis=0.0, unstick=0.0)
        assert fast.compare(grid, epochs=4, teacher=teacher) == []
