"""§12.11 on the Rust path: a run resumed from a rust-sweep checkpoint is the uninterrupted run, continued."""
import importlib.util
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
