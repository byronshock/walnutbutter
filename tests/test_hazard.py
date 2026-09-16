"""Escape noise (AUTHORITY.md §5.2) and the hazard eligibility (§6.7): Williams's unit, in every engine."""

from __future__ import annotations

import math
import random

import pytest

from walnutbutter import fast
from walnutbutter.goo import Goo
from walnutbutter.learning import Teacher
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron


def goo(seed: int = 3, **kw) -> Goo:
    g = Goo(count=40, across=8, seed=seed, weight=None, **kw)
    g.rule, g.drive, g.read = "reinforce", "rate", "count"
    return g


def test_the_score_is_zero_in_expectation():
    """§6.7: P * m/expm1(m) - (1 - P) * m = 0 at every margin, so a neuron far from threshold contributes nothing, not noise."""
    for m in (1e-4, 0.05, 0.3, 1.0, 3.0, 20.0):
        p = -math.expm1(-m)
        assert abs(p * (m * math.exp(-m) / -math.expm1(-m)) - (1 - p) * m) < 1e-12 * max(1.0, m)


def test_delta_zero_is_the_deterministic_rule_word_for_word():
    a, b = goo(), goo()
    b.set_delta(0.0)
    assert not b.hazard
    for _ in range(10):
        run_epoch(a, verbose=False)
        run_epoch(b, verbose=False)
        assert [n.spikes for n in a.all_neurons()] == [n.spikes for n in b.all_neurons()]


def test_rest_is_loud_and_a_narrow_decision_is_quiet():
    """§5.2: with every weight 0 nothing is delivered, every neuron sits at rest (s = -theta), and only the hazard fires it."""
    loud = Goo(count=20, across=4, seed=1, weight=0.0, projection=1.0, wiring="zones-equal")  # fully connected: every axis has width
    quiet = Goo(count=20, across=4, seed=1, weight=0.0, projection=1.0, wiring="zones-equal")
    loud.set_delta(0.7)   # exp(-1/0.7) = 0.24 spikes per hop at rest: about five an epoch
    quiet.set_delta(0.1)  # exp(-10): one spike in a thousand epochs
    a, b = random.Random(1), random.Random(1)
    for _ in range(20):
        run_epoch(loud, verbose=False, rng=a)
        run_epoch(quiet, verbose=False, rng=b)
    assert sum(n.spikes for n in loud.interior()) / (20 * len(loud.interior())) > 2.0
    assert sum(n.spikes for n in quiet.interior()) < 3


def test_the_trace_is_what_the_synapse_has_in_the_potential():
    """§6.7: with no inhibition the floor never bites, so at every decision p_j(t) = sum_i w_ij x_ij(t) exactly."""
    g = Goo(count=20, across=4, seed=3, weight=None, weight_range=(0.1, 1.0), projection=1.0, wiring="zones-equal")
    g.rule, g.drive = "reinforce", "rate"
    g.set_delta(0.7)
    seen = []
    original = Neuron.decide

    def watched(self, now):
        if self.delta > 0.0 and not self.refractory_at(now):
            from_traces = sum(c.weight * c.trace * math.exp(-(now - c.trace_at) / Neuron.tau) for c in self.incoming)
            seen.append((self.potential_at(now), from_traces))
        return original(self, now)

    Neuron.decide = watched
    try:
        rng = random.Random(5)
        for _ in range(5):
            run_epoch(g, verbose=False, rng=rng)
    finally:
        Neuron.decide = original
    assert len(seen) > 100 and any(p > 0.0 for p, _ in seen)
    for p, from_traces in seen:
        assert p == pytest.approx(from_traces, abs=1e-9)


def test_a_neuron_that_hears_nothing_is_left_at_the_containers_threshold_in_every_engine():
    """§5.2 (September 16, 2026): no incoming synapses under the fan-in scaling leaves a neuron at the quoted threshold
    and floor, scale 1, with its width -- it fires at the hazard's rest like any neuron and never otherwise. At threshold
    0 it was a pacemaker, seven spikes an epoch whatever its drive, which the feedforward network of §8 (every input
    hearing nothing) made untenable; and the array engine had divided by the zero width and silenced it (NaN)."""
    np = pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter.constants import GOO_MINIMUM_POTENTIAL, GOO_THRESHOLD
    mesh, twin = goo(), goo()  # the scaled rule at 0.05 on forty neurons: two synapses a neuron in expectation
    deaf = [i for i, n in enumerate(mesh.all_neurons()) if not n.incoming]
    assert len(deaf) >= 3
    assert all(mesh.all_neurons()[i].threshold == GOO_THRESHOLD and mesh.all_neurons()[i].minimum_potential == GOO_MINIMUM_POTENTIAL for i in deaf)
    mesh.set_delta(0.455)
    twin.set_delta(0.455)
    assert all(mesh.all_neurons()[i].delta == pytest.approx(0.455 * GOO_THRESHOLD) for i in deaf)  # a width to quote
    net = ArrayNetwork(twin)
    for _ in range(20):
        run_epoch(mesh, verbose=False, rng=random.Random(1))
        run_epoch(net, verbose=False, rng=random.Random(1))
        assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
    rest = sum(mesh.all_neurons()[i].spikes for i in deaf) / (20 * len(deaf))  # the hazard's rest: 2.3 an epoch at 0.455
    assert 1.0 < rest < 4.0 and all(mesh.all_neurons()[i].spikes < 7 * 20 for i in deaf)  # not a pacemaker
    if fast.available():
        g = goo()
        g.set_delta(0.455)
        assert fast.compare(g, epochs=20, teacher=Teacher(g, seed=7, rule="reinforce", eligibility="hazard", target="copy",
                                                          homeostasis=0.01, unstick=0.1)) == []
    # a threshold given as 0 outright is the deterministic comparison, and the engines agree on it too (no NaN)
    a, b = (Goo(count=20, across=4, seed=1, weight=None, threshold=0.0, minimum_potential=0.0, scale_with_fan_in=False) for _ in range(2))
    a.set_delta(0.455); b.set_delta(0.455)
    assert all(n.delta == 0.0 for n in a.all_neurons())
    net = ArrayNetwork(b)
    for _ in range(3):
        run_epoch(a, verbose=False, rng=random.Random(1)); run_epoch(net, verbose=False, rng=random.Random(1))
        assert [n.spikes for n in a.all_neurons()] == net.spikes.tolist()
    assert sum(n.spikes for n in a.interior()) / (3 * len(a.interior())) > 3.0  # a pacemaker, by the letter, where the
    # threshold is 0 itself: most waves it is not refractory (a wave is not always there the moment one ends; 4.7 an epoch here)


def test_the_hazard_eligibility_needs_escape_noise():
    g = goo()
    with pytest.raises(ValueError, match="escape noise"):
        Teacher(g, rule="reinforce", eligibility="hazard")
    with pytest.raises(ValueError, match="not be negative"):
        g.set_delta(-1.0)
    if fast.available():
        with pytest.raises(ValueError, match="escape noise"):
            fast.train(g, 2, eligibility="hazard", seed=1)


def test_the_three_engines_agree_under_the_hazard_eligibility():
    """§7 on §5.2 and §6.7: every spike in every engine; the scores and weights bit for bit in Rust, learning on.

    The array engine sums a wave's inputs in the matrix's order and takes numpy's
    exponentials, which differ from libm's in the last bit on one argument in
    twenty (§6.15); under every earlier rule that could only show as a spike within
    an ulp of threshold, but the hazard eligibility is a continuous function of the
    potential, so there it shows in the twelfth digit of a score. Spikes exactly,
    then, and scores and weights to a part in a billion.
    """
    np = pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    mesh, twin = goo(), goo()
    mesh.set_delta(0.7)
    twin.set_delta(0.7)
    net = ArrayNetwork(twin)
    on_mesh = Teacher(mesh, seed=7, rule="reinforce", eligibility="hazard", target="copy", homeostasis=0.01, unstick=0.1)
    on_net = Teacher(net, seed=7, rule="reinforce", eligibility="hazard", target="copy", homeostasis=0.01, unstick=0.1)
    assert on_mesh.sigma == 0.0
    for _ in range(40):
        on_mesh.epoch(verbose=False)
        on_net.epoch(verbose=False)
        assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
        scores = [mesh.connections[i].score for i in range(1, len(mesh.connections) + 1)]
        weights = [mesh.connections[i].weight for i in range(1, len(mesh.connections) + 1)]
        assert np.allclose(scores, net.score, rtol=1e-9, atol=0.0)
        assert np.allclose(weights, net.weight, rtol=1e-9, atol=0.0)
    assert any(c.score != 0.0 for c in mesh.connections.values())
    if fast.available():
        g = goo()
        g.set_delta(0.7)
        teacher = Teacher(g, seed=7, rule="reinforce", eligibility="hazard", target="copy", homeostasis=0.01, unstick=0.1)
        parted = fast.compare(g, epochs=60, teacher=teacher)
        assert parted == [], parted
        assert teacher.rng.getstate() != random.Random(7).getstate()  # the stream really moved, on both sides alike


def test_the_three_engines_agree_under_the_hebb_eligibility_with_escape_noise():
    """§6.7's centred Hebbian rule (September 16, 2026) on the network the sweeps run, goo under escape noise: every
    spike, every tally, every expected count and every weight in every engine. The rule is integer counts and one
    moving average, so the array engine agrees to the bit here, where the hazard's continuous score could not."""
    np = pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    mesh, twin = goo(), goo()
    mesh.set_delta(0.7)
    twin.set_delta(0.7)
    net = ArrayNetwork(twin)
    on_mesh = Teacher(mesh, seed=7, rule="reinforce", eligibility="hebb", target="copy", homeostasis=0.01, unstick=0.1)
    on_net = Teacher(net, seed=7, rule="reinforce", eligibility="hebb", target="copy", homeostasis=0.01, unstick=0.1)
    assert on_mesh.sigma == 0.0 and mesh.tally and net.tally
    ids = range(1, len(mesh.connections) + 1)
    before = [mesh.connections[i].weight for i in ids]
    for _ in range(40):
        assert on_mesh.epoch(verbose=False) == on_net.epoch(verbose=False)
        assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
        assert [mesh.connections[i].eligibility for i in ids] == net.eligibility.tolist()
        assert [n.expected_count for n in mesh.all_neurons()] == [None if np.isnan(e) else e for e in net.expected_count.tolist()]
        assert [mesh.connections[i].weight for i in ids] == net.weight.tolist()
    assert any(mesh.connections[i].eligibility != 0.0 for i in ids)
    assert [mesh.connections[i].weight for i in ids] != before  # the rule moved something
    if fast.available():
        g = goo()
        g.set_delta(0.7)
        teacher = Teacher(g, seed=7, rule="reinforce", eligibility="hebb", target="copy", homeostasis=0.01, unstick=0.1)
        parted = fast.compare(g, epochs=60, teacher=teacher)
        assert parted == [], parted


def test_a_checkpoint_keeps_the_widths(tmp_path):
    from walnutbutter.persistence import checkpoint, restore
    g = goo()
    g.set_delta(1.05)
    run_epoch(g, verbose=False, rng=random.Random(2))
    data = checkpoint(g, tmp_path / "hazard.json")
    assert data["escape_delta"] == 1.05
    back, _ = restore(tmp_path / "hazard.json")
    assert back.hazard and back.escape_delta == 1.05
    assert [n.delta for n in back.all_neurons()] == [n.delta for n in g.all_neurons()]
    assert [n.exposed_since for n in back.all_neurons()] == [n.exposed_since for n in g.all_neurons()]
    assert [(c.trace, c.trace_at) for c in back.connections.values()] == [(c.trace, c.trace_at) for c in g.connections.values()]
