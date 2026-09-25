"""§12.11 on the Rust path: a run resumed from a rust-sweep checkpoint is the uninterrupted run, continued."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from walnutbutter import fast
from walnutbutter.network import input_stream
from walnutbutter.neuron import Neuron


@pytest.mark.parametrize("tau, homeostasis, unstick", [
    ("inf", 0.0, 0.0),   # the accumulator (§2.2), the default
    ("2.0", 0.0, 0.0),   # the leak (§2.3): potentials and traces decay lazily, against times a resume has to carry
    ("inf", 0.01, 0.1),  # with the threshold moves of §9.9 and §9.10, which run on the Teacher's rate memory
    ("2.0", 0.01, 0.1),
])
def test_a_rust_sweep_resume_reproduces_the_uninterrupted_trace(tmp_path, tau, homeostasis, unstick):
    """Twelve epochs straight through against six, `_save_network`, `resume_grid`, six more: every traced score and every
    weight must match. Before September 21, 2026 the checkpoint carried none of what the engine keeps across epochs --
    the exploration stream (saved null: `fast.sync_explore` had no call site), potentials, traces and the §6.7 expected
    counts, the firing times the refractory test and the hazard run from, the event heap, and under the leak the times
    the lazy decays run from -- and a resumed run parted from the uninterrupted one at the first resumed epoch. The
    engine's spike-triggered rate trace (§4.3) is not carried: nothing on this path reads it, and the checkpoint
    format does not hold it."""
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    spec = importlib.util.spec_from_file_location("rs", Path(__file__).resolve().parent.parent / "docs" / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    arm, fixed = {"goo": 24.0, "seed": 3}, ("--tau", tau)
    everyone = lambda edges: (np.arange(len(edges)) % 2 * 2.0 - 1.0, np.ones(len(edges), dtype=bool))
    common = dict(target="copy", trace_every=1, eligibility="hazard", homeostasis=homeostasis, unstick=unstick, direction=everyone)
    was = Neuron.tau  # grid_of sets the class clock from --tau; leave it as we found it
    try:
        whole, cli = rs.grid_of("copy", arm, "hazard", True, -4.0, None, fixed)
        patterns = input_stream(12, whole.raw_bit_count(), 3)
        _, straight, engine_w, _ = fast.train(whole, 12, lr=cli.lr, patterns=patterns, seed=3, **common)

        part, _ = rs.grid_of("copy", arm, "hazard", True, -4.0, None, fixed)
        _, first_half, engine, report = fast.train(part, 6, lr=cli.lr, patterns=patterns, seed=3, **common)
        assert first_half == straight[:6]
        rs._save_network(engine, part, report, tmp_path / "arm-network.json")

        again, _ = rs.grid_of("copy", arm, "hazard", True, -4.0, None, fixed)
        grid, offset, reference, baseline, explore_state = rs.resume_grid(again, tmp_path / "arm-network.json")
        assert offset == 6 and explore_state, "the checkpoint must carry the exploration stream's state (§12.11)"
        grid.use_input_stream(patterns, None)
        grid.input_at = offset
        _, second_half, engine2, _ = fast.train(grid, 6, lr=cli.lr, seed=3, explore_state=explore_state, pending_events=grid.engine_pending,
                                                reference_weights=reference, epoch_offset=offset, baseline=baseline, **common)
        assert second_half == straight[6:], "the resumed run must be the same run, continued"
        assert list(engine2.weights()) == list(engine_w.weights())
    finally:
        Neuron.tau = was


@pytest.mark.parametrize("tau", ["inf", "2.0"])
def test_checkpointing_as_the_run_goes_does_not_change_the_run(tmp_path, tau):
    """§12.9 written mid-run: twelve epochs checkpointed every three against twelve that were not.

    A checkpoint is a read of the engine, so a run that writes them must be the run that does
    not -- score for score and weight for weight. `_save_network` writes the engine's state back
    onto the mesh's neurons and synapses to build the file, which is the one thing here that
    touches the network the loop is running on; if the loop ever read those objects back, this
    is the test that would fail.
    """
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    spec = importlib.util.spec_from_file_location("rs", Path(__file__).resolve().parent.parent / "docs" / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    arm, fixed = {"goo": 24.0, "seed": 3}, ("--tau", tau)
    everyone = lambda edges: (np.arange(len(edges)) % 2 * 2.0 - 1.0, np.ones(len(edges), dtype=bool))
    common = dict(target="copy", trace_every=1, eligibility="hazard", homeostasis=0.01, unstick=0.1, direction=everyone)
    was = Neuron.tau
    try:
        plain, cli = rs.grid_of("copy", arm, "hazard", True, -4.0, None, fixed)
        patterns = input_stream(12, plain.raw_bit_count(), 3)
        _, quiet, quiet_engine, _ = fast.train(plain, 12, lr=cli.lr, patterns=patterns, seed=3, **common)

        watched, _ = rs.grid_of("copy", arm, "hazard", True, -4.0, None, fixed)
        seen = []
        def write(epoch, engine, grid, report):
            seen.append(epoch)
            rs._save_network(engine, grid, report, tmp_path / "arm-network.json",
                             {"rows": rs._trace_rows([], report["trace"], report["right"], 0, 1), "estimator": report["estimator"]})
        _, traced, traced_engine, _ = fast.train(watched, 12, lr=cli.lr, patterns=patterns, seed=3,
                                                 checkpoint=write, checkpoint_every=3, **common)
        assert seen == [3, 6, 9], "a checkpoint at every third epoch, and never on the epoch the run ends at"
        assert traced == quiet, "a checkpointed run must be the run that was not checkpointed"
        assert list(traced_engine.weights()) == list(quiet_engine.weights())
    finally:
        Neuron.tau = was


@pytest.mark.parametrize("tau", ["inf", "2.0"])
def test_a_run_goes_on_from_a_checkpoint_written_mid_run(tmp_path, tau):
    """The checkpoint a run writes at epoch six is the one it would have written by stopping there.

    Twelve epochs straight through; then twelve more that are cut off after six, continued from
    the checkpoint the sixth epoch wrote from inside the loop. This is the guarantee the cadence
    is for: what is on disk when a machine goes down is a point the run continues from exactly
    (§12.11), and the record the checkpoint carries is the record the uninterrupted run wrote.
    """
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    spec = importlib.util.spec_from_file_location("rs", Path(__file__).resolve().parent.parent / "docs" / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    arm, fixed = {"goo": 24.0, "seed": 3}, ("--tau", tau)
    everyone = lambda edges: (np.arange(len(edges)) % 2 * 2.0 - 1.0, np.ones(len(edges), dtype=bool))
    common = dict(target="copy", trace_every=1, eligibility="hazard", homeostasis=0.01, unstick=0.1, direction=everyone)
    was = Neuron.tau
    try:
        whole, cli = rs.grid_of("copy", arm, "hazard", True, -4.0, None, fixed)
        patterns = input_stream(12, whole.raw_bit_count(), 3)
        _, straight, engine_w, report_w = fast.train(whole, 12, lr=cli.lr, patterns=patterns, seed=3, **common)
        rows_w = rs._trace_rows([], straight, report_w["right"], 0, 1)

        cut, _ = rs.grid_of("copy", arm, "hazard", True, -4.0, None, fixed)
        def write(epoch, engine, grid, report):  # the run is cut short right after this, as a reboot would cut it
            rs._save_network(engine, grid, report, tmp_path / "arm-network.json",
                             {"rows": rs._trace_rows([], report["trace"], report["right"], 0, 1),
                              "estimator": report["estimator"]})
        fast.train(cut, 12, lr=cli.lr, patterns=patterns, seed=3, checkpoint=write, checkpoint_every=6, **common)

        again, _ = rs.grid_of("copy", arm, "hazard", True, -4.0, None, fixed)
        grid, offset, reference, baseline, explore_state = rs.resume_grid(again, tmp_path / "arm-network.json")
        assert offset == 6
        carried = grid.engine_progress
        assert [tuple(row) for row in carried["rows"]] == rows_w[:6], "the checkpoint carries the record to that point"
        grid.use_input_stream(patterns, None)
        grid.input_at = offset
        _, rest, engine2, report2 = fast.train(grid, 6, lr=cli.lr, seed=3, explore_state=explore_state,
                                               pending_events=grid.engine_pending, reference_weights=reference,
                                               epoch_offset=offset, baseline=baseline, **common)
        assert rest == straight[6:], "a run continued from a mid-run checkpoint is the uninterrupted run"
        assert list(engine2.weights()) == list(engine_w.weights())
        assert rs._trace_rows(carried["rows"], rest, report2["right"], offset, 1) == rows_w
    finally:
        Neuron.tau = was


def test_an_arm_cut_short_finishes_when_the_sweep_is_run_again(tmp_path, monkeypatch):
    """The recovery the cadence is for, at the driver: an arm with a checkpoint and no record finishes the epochs asked for.

    Twelve epochs as one arm, against an arm stopped at six whose sweep is simply run again. The
    second one must end with the record the first wrote, row for row -- which is what makes losing
    a machine mid-sweep cost the epochs since the last checkpoint instead of the whole run. On
    September 22, 2026 a ten-arm run of a million epochs was lost at 98 per cent to a reboot,
    because an arm wrote nothing until it was done.
    """
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    spec = importlib.util.spec_from_file_location("rs", Path(__file__).resolve().parent.parent / "docs" / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    monkeypatch.setattr(rs, "ROOT", tmp_path)  # the arms land under the test's own runs/, not the repository's
    arm, fixed = {"goo": 24.0, "seed": 3}, ("--tau", "inf")
    job = lambda name, epochs: (arm, "copy", epochs, 1, name, "hazard", True, -4.0, None, None, fixed, 4)
    was = Neuron.tau
    try:
        for name in ("whole", "cut"):
            (tmp_path / "runs" / name).mkdir(parents=True)
        straight = rs.run_arm(job("whole", 12))

        stopped = rs.run_arm(job("cut", 6))  # the arm as a reboot would leave it: a checkpoint, and no record
        assert stopped["epochs_run"] == 6
        for leftover in ("cut/%s.csv" % rs.arm_name(arm), "cut/%s.json" % rs.arm_name(arm)):
            (tmp_path / "runs" / leftover).unlink()
        finished = rs.run_arm(job("cut", 12))  # the same command again: it picks the arm up where it stopped
        assert finished["resumed_from"] == "cut" and finished["epoch_offset"] == 6 and finished["epochs_run"] == 6

        rows = lambda name: (tmp_path / "runs" / name / f"{rs.arm_name(arm)}.csv").read_text()
        assert rows("cut") == rows("whole"), "a continued arm's record is the uninterrupted arm's record"
        assert finished["last_tenth"] == straight["last_tenth"]
        # the leg a recovery runs is not the run, so its summary is taken over the record's last tenth and says so;
        # otherwise an arm that lost its last epochs would report a tenth of those beside its siblings' whole tenth
        assert straight["last_tenth_over"] == "engine"
        assert finished["last_tenth_over"].startswith("the record's last tenth")
        carried = json.loads((tmp_path / "runs" / "whole" / f"{rs.arm_name(arm)}-network.json").read_text())["progress"]
        assert len(carried["rows"]) == 12, "an arm that finished carries its record too, for --resume-from"
    finally:
        Neuron.tau = was


def test_an_arm_killed_between_checkpoints_resumes_from_the_last_one(tmp_path, monkeypatch):
    """The same recovery from a checkpoint written inside the loop rather than at the end of a run.

    The arm is killed just after its epoch-eight checkpoint, as a machine going down would kill it,
    and the sweep is run again. What it finishes with must be what the uninterrupted arm wrote, and
    the epochs between the checkpoint and the kill are the only ones done twice.
    """
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    spec = importlib.util.spec_from_file_location("rs", Path(__file__).resolve().parent.parent / "docs" / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    arm, fixed = {"goo": 24.0, "seed": 3}, ("--tau", "2.0")
    job = lambda name, epochs: (arm, "copy", epochs, 1, name, "hazard", True, -4.0, None, None, fixed, 8)
    was = Neuron.tau
    try:
        for name in ("whole", "killed"):
            (tmp_path / "runs" / name).mkdir(parents=True)
        rs.run_arm(job("whole", 20))

        saved, write = [], rs._save_network
        def die_after_the_checkpoint(engine, grid, report, path, progress=None):
            write(engine, grid, report, path, progress)
            saved.append(int(progress["rows"][-1][0]))
            raise RuntimeError("the machine goes down here")
        monkeypatch.setattr(rs, "_save_network", die_after_the_checkpoint)
        with pytest.raises(RuntimeError):
            rs.run_arm(job("killed", 20))
        monkeypatch.setattr(rs, "_save_network", write)
        assert saved == [8], "the checkpoint at epoch eight, written from inside the run"
        assert not (tmp_path / "runs" / "killed" / f"{rs.arm_name(arm)}.csv").exists()

        finished = rs.run_arm(job("killed", 20))
        assert finished["epoch_offset"] == 8 and finished["epochs_run"] == 12
        rows = lambda name: (tmp_path / "runs" / name / f"{rs.arm_name(arm)}.csv").read_text()
        assert rows("killed") == rows("whole"), "what the arm finishes with is the uninterrupted run's record"
    finally:
        Neuron.tau = was


# --- exploration at the synapse (AUTHORITY.md §7.5-§7.9, §8.16, §8.17, 5.4b): §12.9's checkpoint and §12.11's resume ---

SYNAPSE_ARM = {"threshold": 0.6, "synapse_hazard": 0.1, "goo": 24.0, "seed": 3}  # a rest hazard that leaves escapes in flight


def _rs():
    spec = importlib.util.spec_from_file_location("rs", Path(__file__).resolve().parent.parent / "docs" / "rust-sweep.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    return rs


@pytest.mark.parametrize("tau, drive, settings", [
    ("inf", "rate", ()),
    ("inf", "charged", ()),  # the accumulator under the charged drive, where the gains stand open at the cut (§8.16)
    ("2.0", "rate", ("--hazard-family", "linear", "--synapse-scaling", "fan-out")),
    ("2.0", "charged", ("--trace-counts", "ventured", "--drive-steps", "4")),
    ("inf", "charged", ("--hazard-family", "linear", "--trace-counts", "ventured")),
])
def test_a_rust_sweep_resume_under_exploration_at_the_synapse_reproduces_the_uninterrupted_trace(tmp_path, tau, drive, settings):
    """§12.11 under exploration at the synapse: twelve epochs straight through against six, `_save_network`,
    `resume_grid`, six more, on both drives. The cut has ventured signals in flight (§7.9), and under the accumulator
    with the charged drive gains standing open across it (§8.16); the checkpoint carries them, the read counts of the
    epoch just finished and the settings (§8.14, §12.9), and the resumed run is the uninterrupted one to the bit."""
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    rs = _rs()
    fixed = ("--tau", tau, "--exploration", "synapse", "--drive", drive) + settings
    everyone = lambda edges: (np.arange(len(edges)) % 2 * 2.0 - 1.0, np.ones(len(edges), dtype=bool))
    common = dict(target="copy", trace_every=1, eligibility="hazard", direction=everyone)
    was = Neuron.tau
    try:
        whole, cli = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, fixed)
        patterns = input_stream(12, whole.raw_bit_count(), 3)
        _, straight, engine_w, _ = fast.train(whole, 12, lr=cli.lr, patterns=patterns, seed=3, **common)

        part, _ = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, fixed)
        _, first_half, engine, report = fast.train(part, 6, lr=cli.lr, patterns=patterns, seed=3, **common)
        assert first_half == straight[:6]
        assert any(e[3] for e in report["pending"]), "the cut must have ventured signals in flight"
        if tau == "inf" and drive == "charged":
            assert any(report["gains"]), "and, under the accumulator, gains open across it"
        rs._save_network(engine, part, report, tmp_path / "arm-network.json")
        data = json.loads((tmp_path / "arm-network.json").read_text())
        assert (data["format"], data["exploration"]) == (3, "synapse")
        assert data["gains"] == list(engine.gains()) and data["read_counts"] == list(engine.read_counts())
        assert [list(e) for e in engine.pending_events(True)] == data["engine_pending"]
        assert sorted((t, i, v) for t, i, v in data["pending"]) == sorted(  # the network's own queue is the engine's
            (t, [c for n in part.all_neurons() for c in n.outgoing][k].id, v) for t, kind, k, v in data["engine_pending"] if kind == 2)
        assert (data["drive"], data["drive_steps"]) == (drive, 4 if "4" in settings else 3)

        again, _ = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, fixed)
        grid, offset, reference, baseline, explore_state = rs.resume_grid(again, tmp_path / "arm-network.json")
        assert offset == 6 and grid.exploration == "synapse" and grid.drive == drive
        assert [n.synapse_scale for n in grid.all_neurons()] == [n.synapse_scale for n in part.all_neurons()]  # recomputed
        grid.use_input_stream(patterns, None)
        grid.input_at = offset
        _, second_half, engine2, _ = fast.train(grid, 6, lr=cli.lr, seed=3, explore_state=explore_state,
                                                pending_events=grid.engine_pending, reference_weights=reference,
                                                epoch_offset=offset, baseline=baseline, **common)
        assert second_half == straight[6:], "the resumed run must be the same run, continued"
        assert list(engine2.weights()) == list(engine_w.weights())
        assert list(engine2.gains()) == list(engine_w.gains()) and list(engine2.traces()) == list(engine_w.traces())
        assert list(engine2.pending_events(True)) == list(engine_w.pending_events(True))
    finally:
        Neuron.tau = was


@pytest.mark.parametrize("drive", ["rate", "charged"])
def test_a_run_under_exploration_at_the_synapse_goes_on_from_a_checkpoint_written_mid_run(tmp_path, drive):
    """#21's callback under the mode: the checkpoint the run writes at epoch six, after that epoch's update and before
    the next reset, carries the report's gains, read counts and marked queue, and the run continued from it is the
    uninterrupted run. The finished epoch's read counts are in the file, and the resumed run's first reset zeroes them."""
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    rs = _rs()
    fixed = ("--tau", "inf", "--exploration", "synapse", "--drive", drive)
    common = dict(target="copy", trace_every=1, eligibility="hazard", homeostasis=0.01, unstick=0.1)
    whole, cli = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, fixed)
    patterns = input_stream(12, whole.raw_bit_count(), 3)
    _, straight, engine_w, _ = fast.train(whole, 12, lr=cli.lr, patterns=patterns, seed=3, **common)

    cut, _ = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, fixed)
    reports = []
    def write(epoch, engine, grid, report):
        reports.append(report)
        rs._save_network(engine, grid, report, tmp_path / "arm-network.json",
                         {"rows": rs._trace_rows([], report["trace"], report["right"], 0, 1), "estimator": None})
    fast.train(cut, 12, lr=cli.lr, patterns=patterns, seed=3, checkpoint=write, checkpoint_every=6, **common)
    data = json.loads((tmp_path / "arm-network.json").read_text())
    report = reports[0]
    assert (data["gains"], data["read_counts"]) == (report["gains"], report["read_counts"])
    assert data["engine_pending"] == [list(e) for e in report["pending"]] and any(e[3] for e in report["pending"])
    assert report["settings"] == {key: data[key] for key in report["settings"]}  # the settings, as the file holds them
    assert sum(data["read_counts"]) > 0, "the finished epoch's read counts are carried"

    again, _ = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, fixed)
    grid, offset, reference, baseline, explore_state = rs.resume_grid(again, tmp_path / "arm-network.json")
    assert [n.read_count for n in grid.all_neurons()] == data["read_counts"]
    grid.use_input_stream(patterns, None)
    grid.input_at = offset
    _, rest, engine2, _ = fast.train(grid, 6, lr=cli.lr, seed=3, explore_state=explore_state,
                                     pending_events=grid.engine_pending, reference_weights=reference,
                                     epoch_offset=offset, baseline=baseline, **common)
    assert rest == straight[6:] and list(engine2.weights()) == list(engine_w.weights())


def test_an_unmarked_engine_queue_loads_as_relayed():
    """A (time, kind, payload) triple, as every engine_pending on disk before September 25, 2026 holds it, is a relayed
    signal (§7.9): fast.build hands it over unmarked beside a marked quadruple."""
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    rs = _rs()
    grid, cli = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, ("--exploration", "synapse"))
    patterns = input_stream(6, grid.raw_bit_count(), 3)
    _, _, engine, report = fast.train(grid, 6, lr=cli.lr, target="copy", patterns=patterns, seed=3, eligibility="hazard")
    queue = list(report["pending"])
    assert any(e[3] for e in queue)
    import random
    rebuilt = fast.build(grid, explore_rng=random.Random(3), pending_events=[e[:3] for e in queue])
    assert [tuple(e) for e in rebuilt[0].pending_events(True)] == [(t, k, p, False) for t, k, p, _ in queue]


@pytest.mark.parametrize("drive", ["rate", "charged"])
def test_a_rust_checkpoint_resumes_on_the_objects_and_an_object_checkpoint_on_rust(tmp_path, drive):
    """§12.9: a checkpoint round-trips in every engine that carries the rule. Six epochs on the Rust loop, saved by the
    sweep driver and continued for six on the object engine under a Teacher, land on the Rust loop's twelve to the bit;
    and six on the objects under a Teacher, checkpointed and continued for six on the Rust loop through `resume_grid` --
    which takes the queue of a file with no engine_pending from the network it restores -- land on the objects' twelve.
    Ventured signals are in flight at both cuts."""
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    from walnutbutter.learning import Teacher
    from walnutbutter.persistence import checkpoint, restore, resume_teacher
    rs = _rs()
    fixed = ("--tau", "inf", "--exploration", "synapse", "--drive", drive)
    common = dict(target="copy", eligibility="hazard")
    whole, cli = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, fixed)
    patterns = input_stream(12, whole.raw_bit_count(), 3)
    _, _, rust_whole, _ = fast.train(whole, 12, lr=cli.lr, patterns=patterns, seed=3, **common)

    part, _ = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, fixed)
    _, _, engine, report = fast.train(part, 6, lr=cli.lr, patterns=patterns, seed=3, **common)
    assert any(e[3] for e in report["pending"])
    rs._save_network(engine, part, report, tmp_path / "rust.json")
    objects, data = restore(tmp_path / "rust.json")
    objects.readout, objects.read, objects.rule = "top", "count", "reinforce"
    objects.use_input_stream(patterns)
    objects.input_at = data["input_at"]
    teacher = Teacher(objects, seed=0, lr=cli.lr, rule="reinforce", **common)
    teacher.rng.setstate((3, tuple(data["explore_state"]), None))
    teacher.baseline = data["baseline"]
    for _ in range(6):
        teacher.epoch(verbose=False)
    edges = [c for n in objects.all_neurons() for c in n.outgoing]
    assert [c.weight for c in edges] == list(rust_whole.weights()), "Rust six, objects six: the Rust loop's twelve"

    straight, _ = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, fixed)
    straight.use_input_stream(patterns)
    t = Teacher(straight, seed=3, lr=cli.lr, rule="reinforce", **common)
    for _ in range(12):
        t.epoch(verbose=False)
    halfway, _ = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, fixed)
    halfway.use_input_stream(patterns)
    t = Teacher(halfway, seed=3, lr=cli.lr, rule="reinforce", **common)
    for _ in range(6):
        t.epoch(verbose=False)
    assert any(ventured for _, _, ventured in halfway.schedule.pending(marks=True))
    saved = checkpoint(halfway, tmp_path / "objects.json", t)
    assert "engine_pending" not in saved
    fresh, _ = rs.grid_of("copy", SYNAPSE_ARM, "hazard", True, -4.0, None, fixed)
    grid, offset, reference, baseline, explore_state = rs.resume_grid(fresh, tmp_path / "objects.json")
    assert any(e[3] for e in grid.engine_pending), "the queue, marks and all, from the network the file restores"
    assert baseline == saved["learning"]["baseline"] == t.baseline, "and the Teacher's baseline from its learning record"
    grid.use_input_stream(patterns, None)
    grid.input_at = offset
    _, _, engine2, _ = fast.train(grid, 6, lr=cli.lr, seed=3, explore_state=explore_state, pending_events=grid.engine_pending,
                                  baseline=baseline, **common)
    assert list(engine2.weights()) == [c.weight for n in straight.all_neurons() for c in n.outgoing], \
        "objects six, Rust six: the objects' twelve"


def _arm_files(root, sweep, arm, rs):
    """An arm's three records as a sweep leaves them: its checkpoint, its json and its csv."""
    out, name = root / "runs" / sweep, rs.arm_name(arm)  # an arm's name holds dots, so no with_suffix
    return (json.loads((out / f"{name}-network.json").read_text()), json.loads((out / f"{name}.json").read_text()),
            (out / f"{name}.csv").read_text())


def test_an_object_checkpoint_resumed_by_the_sweep_is_the_uninterrupted_arm(tmp_path, monkeypatch):
    """§9.3, §12.11 at the driver: six epochs on the object engine under a Teacher, checkpointed by persistence, and six
    more by `run_arm --resume-from`, the path a sweep takes, land on the arm run whole for twelve on the Rust loop -- every
    key of the checkpoint the same but the record, which an object file does not carry. The reinforcement baseline is the
    Teacher's, in the file's learning record; until September 25, 2026 `resume_grid` read only the driver's own and the
    resumed arm started a fresh baseline, a different run."""
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    from walnutbutter.learning import Teacher
    from walnutbutter.persistence import checkpoint
    rs = _rs()
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    monkeypatch.setattr(Neuron, "tau", Neuron.tau)
    arm = {"threshold": 0.6, "goo": 40.0, "seed": 3}  # a setup that learns, so the baseline moves the weights
    fixed = ("--tau", "inf", "--exploration", "synapse", "--synapse-hazard", "0.01")
    for sweep in ("whole", "objects", "resumed"):
        (tmp_path / "runs" / sweep).mkdir(parents=True)
    rs.run_arm((arm, "copy", 12, 1, "whole", "hazard", True, -4.0, None, None, fixed, 0))

    grid, cli = rs.grid_of("copy", arm, "hazard", True, -4.0, None, fixed)
    grid.use_input_stream(input_stream(12, grid.raw_bit_count(), 3))
    teacher = Teacher(grid, seed=3, lr=cli.lr, rule="reinforce", target="copy", eligibility="hazard",
                      homeostasis=cli.homeostasis, target_rate=cli.target_rate, unstick=cli.unstick,
                      unstick_target=cli.unstick_target)
    for _ in range(6):
        teacher.epoch(verbose=False)
    saved = checkpoint(grid, tmp_path / "runs" / "objects" / f"{rs.arm_name(arm)}-network.json", teacher)
    assert "baseline" not in saved and saved["learning"]["baseline"] is not None
    result = rs.run_arm((arm, "copy", 6, 1, "resumed", "hazard", True, -4.0, None, "objects", fixed, 0))
    assert result["epoch_offset"] == 6 and result["epochs_run"] == 6
    whole, resumed = (_arm_files(tmp_path, sweep, arm, rs)[0] for sweep in ("whole", "resumed"))
    fresh = rs.grid_of("copy", arm, "hazard", True, -4.0, None, fixed)[0]
    assert whole["weights"] != [fresh.connections[i].weight for i in range(1, len(fresh.connections) + 1)], "the arm must learn"
    assert [key for key in whole if whole[key] != resumed.get(key)] == ["progress"]


@pytest.mark.parametrize("fixed, arm", [
    (("--tau", "inf"), {"threshold": 2.0, "goo": 24.0, "seed": 3}),  # the accumulator: B_ij rebuilt at every reset
    (("--tau", "inf", "--exploration", "synapse", "--drive", "charged"), {"threshold": 2.0, "goo": 24.0, "seed": 3}),
    (("--tau", "2.0", "--exploration", "synapse"), {"threshold": 0.6, "synapse_hazard": 0.1, "goo": 24.0, "seed": 3}),
])
def test_a_resumed_arm_writes_the_uninterrupted_arm_s_checkpoint_key_for_key(tmp_path, monkeypatch, fixed, arm):
    """§12.9, §12.11: an arm cut short at epoch five and finished by running the sweep again writes, at the end, the
    checkpoint the uninterrupted arm writes -- every key, the notes B_ij among them -- and the same record. Until September
    25, 2026 `_save_network` left the notes to the network's own reset, which rebuilt them from the traces it had been
    restored with, so under the accumulator a resumed arm's file carried notes that were neither run's."""
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    rs = _rs()
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    monkeypatch.setattr(Neuron, "tau", Neuron.tau)
    for sweep in ("whole", "cut"):
        (tmp_path / "runs" / sweep).mkdir(parents=True)
    job = lambda sweep, epochs: (arm, "copy", epochs, 1, sweep, "hazard", True, -4.0, None, None, fixed, 0)
    rs.run_arm(job("whole", 12))
    rs.run_arm(job("cut", 5))
    for leftover in (".csv", ".json"):  # as a reboot leaves it: the checkpoint, and no record
        (tmp_path / "runs" / "cut" / f"{rs.arm_name(arm)}{leftover}").unlink()
    finished = rs.run_arm(job("cut", 12))
    assert (finished["epoch_offset"], finished["epochs_run"]) == (5, 7)
    (whole, whole_json, whole_csv), (cut, _, cut_csv) = (_arm_files(tmp_path, sweep, arm, rs) for sweep in ("whole", "cut"))
    assert [key for key in whole if whole[key] != cut.get(key)] == [] and set(whole) == set(cut)
    assert cut_csv == whole_csv


@pytest.mark.parametrize("mode", [(), ("--exploration", "synapse")])
def test_a_resume_that_names_no_clock_runs_on_the_checkpoint_s(tmp_path, monkeypatch, mode):
    """§12.9 for the clock: a TAU 2 arm resumed by a sweep that does not name --tau runs at TAU 2, its checkpoint's, and is
    the uninterrupted TAU 2 arm key for key; its json says so. Until September 25, 2026 it ran at the constant's TAU inf
    and parted from the run at the first resumed epoch. A sweep that names --tau overrides the checkpoint's."""
    if not fast.available():
        pytest.skip("the Rust loop is not built")
    rs = _rs()
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    monkeypatch.setattr(Neuron, "tau", Neuron.tau)
    arm = {"threshold": 0.6, "goo": 24.0, "seed": 3}
    job = lambda sweep, epochs, fixed, source=None: (arm, "copy", epochs, 1, sweep, "hazard", True, -4.0, None, source, fixed, 0)
    for sweep in ("whole", "first", "plain", "named"):
        (tmp_path / "runs" / sweep).mkdir(parents=True)
    rs.run_arm(job("whole", 6, ("--tau", "2.0") + mode))
    rs.run_arm(job("first", 3, ("--tau", "2.0") + mode))
    plain = rs.run_arm(job("plain", 3, mode, "first"))
    assert (plain["tau"], plain["refractory"], plain["hop"]) == (2.0, Neuron.refractory, Neuron.hop)
    (whole, _, whole_csv), (resumed, _, resumed_csv) = (_arm_files(tmp_path, sweep, arm, rs) for sweep in ("whole", "plain"))
    assert resumed["tau"] == 2.0 and [key for key in whole if whole[key] != resumed.get(key)] == []
    assert resumed_csv == whole_csv
    named = rs.run_arm(job("named", 3, ("--tau", "inf") + mode, "first"))
    assert named["tau"] == math.inf and _arm_files(tmp_path, "named", arm, rs)[0]["tau"] == math.inf
