"""§12.11 on the Rust path: a run resumed from a rust-sweep checkpoint is the uninterrupted run, continued."""
import importlib.util
import json
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
