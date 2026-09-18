"""Escape noise (AUTHORITY.md §5.2) and the hazard eligibility (§6.7): Williams's unit, in every engine."""

from __future__ import annotations

import json
import math
import random

import pytest

from walnutbutter.constants import ESCAPE_DELTA
from walnutbutter import fast
from walnutbutter.goo import Goo
from walnutbutter.learning import Teacher
from walnutbutter.monitor import run_epoch
from walnutbutter.constants import DECISION_MEMORY
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


def test_the_reinforce_rule_refuses_where_the_threshold_decides():
    """§8.3: no eligibility runs where the threshold decides -- REINFORCE has no randomness to estimate from."""
    g = goo()
    assert not g.hazard
    for eligibility in ("hazard", "hebb"):
        with pytest.raises(ValueError, match="refuses to learn"):
            Teacher(g, rule="reinforce", eligibility=eligibility)
    with pytest.raises(ValueError, match="not be negative"):
        g.set_delta(-1.0)
    if fast.available():
        with pytest.raises(ValueError, match="refuses to learn"):
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


def test_the_three_engines_agree_under_the_count_hebb_eligibility_with_escape_noise():
    """§6.7's epoch form of the centred Hebbian rule (September 16, 2026; count_hebb since the single-spike rule of
    September 17) on the network the sweeps run, goo under escape noise: every
    spike, every tally, every expected count and every weight in every engine. The rule is integer counts and one
    moving average, so the array engine agrees to the bit here, where the hazard's continuous score could not."""
    np = pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    mesh, twin = goo(), goo()
    mesh.set_delta(0.7)
    twin.set_delta(0.7)
    net = ArrayNetwork(twin)
    on_mesh = Teacher(mesh, seed=7, rule="reinforce", eligibility="count_hebb", target="copy", homeostasis=0.01, unstick=0.1)
    on_net = Teacher(net, seed=7, rule="reinforce", eligibility="count_hebb", target="copy", homeostasis=0.01, unstick=0.1)
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
        teacher = Teacher(g, seed=7, rule="reinforce", eligibility="count_hebb", target="copy", homeostasis=0.01, unstick=0.1)
        parted = fast.compare(g, epochs=60, teacher=teacher)
        assert parted == [], parted


def test_the_count_scales_every_hazard_down_as_its_square_root():
    """§5.2 (Byron, September 16, 2026: "scaling the network MUST reduce the probability of escape noise at each neuron
    by sqrt(N)"): m = (dt / hop) sqrt(N0 / N) e^(s / Delta_j), N0 = ESCAPE_REFERENCE_COUNT = 60, the goo the width was
    set on -- so at 60 nothing moves, a smaller network is louder and a larger one quieter, at every margin alike."""
    from walnutbutter.constants import ESCAPE_REFERENCE_COUNT
    from walnutbutter.goo import Goo
    assert ESCAPE_REFERENCE_COUNT == 60
    for count, scale in ((20, math.sqrt(3.0)), (60, 1.0), (240, 0.5)):
        g = Goo(count=count, across=4, seed=1, weight=0.0, projection=1.0, wiring="zones-equal")
        assert g.escape_scale == 1.0 and all(n.escape_scale == 1.0 for n in g.all_neurons())  # deterministic until set_delta
        g.set_delta(0.455)
        assert g.escape_scale == scale and all(n.escape_scale == scale for n in g.all_neurons())
        neuron = next(iter(g.all_neurons()))
        neuron.exposed_since = 0.0
        # at rest, one hop after its exposure began: the width alone gives e^(-1/Delta); the count scales it, at rest as anywhere
        assert neuron.expected_spikes(Neuron.hop()) == scale * math.exp((0.0 - neuron.threshold) / neuron.delta)
        neuron.potential, neuron.last_update = neuron.threshold, Neuron.hop()  # at threshold: sqrt(N0 / N) a hop, not one
        assert neuron.expected_spikes(Neuron.hop()) == scale * math.exp(0.0)
    grid = Goo(count=16, across=4)
    grid.set_delta(0.455)
    assert grid.escape_scale == math.sqrt(60.0 / 16.0)  # any container: its count
    # the array engine and the Rust loop carry the factor: the agreement tests above run at 40, where it is 1.22


def test_a_checkpoint_keeps_the_widths(tmp_path):
    from walnutbutter.persistence import checkpoint, restore
    g = goo()
    g.set_delta(1.05)
    run_epoch(g, verbose=False, rng=random.Random(2))
    data = checkpoint(g, tmp_path / "hazard.json")
    assert data["escape_delta"] == 1.05 and "escape_scale" not in data  # the count's factor is a rule, recomputed on restore
    back, _ = restore(tmp_path / "hazard.json")
    assert back.hazard and back.escape_delta == 1.05
    assert back.escape_scale == g.escape_scale == math.sqrt(60.0 / 40.0) and all(n.escape_scale == g.escape_scale for n in back.all_neurons())
    assert [n.delta for n in back.all_neurons()] == [n.delta for n in g.all_neurons()]
    assert [n.exposed_since for n in back.all_neurons()] == [n.exposed_since for n in g.all_neurons()]
    assert [(c.trace, c.trace_at) for c in back.connections.values()] == [(c.trace, c.trace_at) for c in g.connections.values()]


def test_the_evidence_accumulator_is_the_neuron_at_tau_infinity_in_every_engine(tmp_path):
    """§5.1 (Byron, September 17, 2026): the evidence accumulator is TAU = inf, nothing leaks, and no engine evaluates
    the decay. The hazard's trace is then the count of the arrivals the target integrated along the synapse since its
    last spike, the potential is the weighted count of that evidence wherever the floor has not bitten, and with
    learning on the three engines agree as they do at TAU 2: every spike, scores and weights to a part in a billion in
    the arrays and to the bit in Rust. A checkpoint carries the infinity."""
    np = pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter.persistence import checkpoint
    was = Neuron.tau
    Neuron.tau = math.inf
    try:
        # the trace is a count and the potential a weighted count: no inhibition, so the floor never bites
        g = Goo(count=20, across=4, seed=3, weight=None, weight_range=(0.1, 1.0), projection=1.0, wiring="zones-equal")
        g.rule, g.drive = "reinforce", "rate"
        g.set_delta(0.7)
        seen = []
        original = Neuron.decide

        def watched(self, now):
            if self.delta > 0.0 and not self.refractory_at(now):
                seen.append((self.potential_at(now), [(c.weight, c.trace) for c in self.incoming]))
            return original(self, now)

        Neuron.decide = watched
        try:
            rng = random.Random(5)
            for _ in range(5):
                run_epoch(g, verbose=False, rng=rng)
        finally:
            Neuron.decide = original
        assert len(seen) > 100 and any(p > 0.0 for p, _ in seen)
        for p, synapses in seen:
            assert p == pytest.approx(sum(w * x for w, x in synapses), abs=1e-9)
            assert all(x == int(x) for _, x in synapses)
        # two arrivals down one synapse into a target that cannot fire: the trace counts both, the potential is twice
        # the weight and does not decay, and the decision's score sum took the trace without a decay factor
        from walnutbutter.propagation import Schedule
        c = next(iter(g.connections.values()))
        j = c.target
        j.threshold = 50.0 * c.weight  # the width was set from the starting threshold, so the hazard is tiny but positive
        j.potential, c.trace, c.score = 0.0, 0.0, 0.0
        for other in j.incoming:
            other.trace = 0.0
        before, later = j.spikes, g.time + 1000.0  # well past the epoch the goo has run, and every refractory period
        s = Schedule()
        s.signal(c, later)
        s.signal(c, later + 20.0)
        s.run(later + 100.0)
        assert j.spikes == before and c.trace == 2.0 and j.potential == 2.0 * c.weight
        assert j.potential_at(later + 5000.0) == j.potential  # nothing leaks
        assert c.score == 0.0 and c.noted != 0.0  # the two silent decisions were noted, not charged: the settle is lazy
        g.settle_scores()  # as the read does first
        assert c.score != 0.0 and math.isfinite(c.score) and c.trace == 2.0  # the debit is in, the arrivals stay open
        # the three engines, learning on
        mesh, twin = goo(), goo()
        mesh.set_delta(0.7)
        twin.set_delta(0.7)
        net = ArrayNetwork(twin)
        on_mesh = Teacher(mesh, seed=7, rule="reinforce", eligibility="hazard", target="copy", homeostasis=0.01, unstick=0.1)
        on_net = Teacher(net, seed=7, rule="reinforce", eligibility="hazard", target="copy", homeostasis=0.01, unstick=0.1)
        ids = range(1, len(mesh.connections) + 1)
        for _ in range(40):
            on_mesh.epoch(verbose=False)
            on_net.epoch(verbose=False)
            assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
            assert np.allclose([mesh.connections[i].score for i in ids], net.score, rtol=1e-9, atol=0.0)
            assert np.allclose([mesh.connections[i].weight for i in ids], net.weight, rtol=1e-9, atol=0.0)
        assert any(c.score != 0.0 for c in mesh.connections.values())
        assert all(c.trace == int(c.trace) for c in mesh.connections.values())
        data = checkpoint(mesh, tmp_path / "accumulator.json")
        assert data["tau"] == math.inf
        with open(tmp_path / "accumulator.json") as f:
            assert json.load(f)["tau"] == math.inf
        if fast.available():
            g = goo()
            g.set_delta(0.7)
            teacher = Teacher(g, seed=7, rule="reinforce", eligibility="hazard", target="copy", homeostasis=0.01, unstick=0.1)
            parted = fast.compare(g, epochs=60, teacher=teacher)
            assert parted == [], parted
    finally:
        Neuron.tau = was


@pytest.mark.parametrize("isi_factor", [True, False])
@pytest.mark.parametrize("eligibility", ["hazard", "hebb"])
def test_the_single_spike_rule_settles_per_arrival_what_the_decisions_charge(eligibility, isi_factor):
    """§6.7 (Byron, September 17, 2026): under the evidence accumulator every synapse's score equals the per-decision
    charge f (c - q) x summed over its target's decisions -- kept here independently, decision by decision, with the
    rule's own (c, q), the ISI factor f of §0.2 computed here from the neuron's last spike (1 when the factor is off),
    and x counted here too: every signal the target integrated along the synapse, cleared when the target spikes and
    when the floor bites, and held against the engine's trace at every decision -- so a floor or a spike that failed to
    close the arrivals parts the two. The engine settles lazily per arrival, at the spike, the floor and the read, with
    no loop over a fan-in at a decision. The arrivals stay open across reads ("Let them run!"), each read paying what
    was charged since the last; under hebb the neuron's expectation is followed alongside."""
    from walnutbutter.connection import Connection
    was, was_factor = Neuron.tau, Neuron.isi_factor
    Neuron.tau, Neuron.isi_factor = math.inf, isi_factor
    try:
        g = goo()
        g.set_delta(0.7)
        if eligibility == "hebb":
            g.centre(True)
        direct = {c: 0.0 for c in g.connections.values()}
        arrivals = {c: 0 for c in g.connections.values()}  # x_ij, counted here and not read from the engine
        mirror = {n: None for n in g.all_neurons()}
        count = {n: 0 for n in g.all_neurons()}
        charged, open_at_a_read, floored, weighed = [0], [False], [0], [False]
        original_decide, original_settle, original_fire = Neuron.decide, Neuron.settle, Neuron.fire

        def factor(now, fired_at):
            if not isi_factor:
                return 1.0
            if fired_at is None:
                return 0.0
            x = (now - fired_at) / 5.1
            return (3 * x - 1) / (1 + x ** 3)

        def integrated(self):
            return self.__dict__.get("last_signal")

        def integrate(self, value):
            self.__dict__["last_signal"] = value
            if value is not None and self in arrivals:
                arrivals[self] += 1  # the propagation stamps a synapse exactly when its target integrates a signal

        def settle(self):
            bites = self.potential < self.minimum_potential
            original_settle(self)
            if bites:
                floored[0] += 1
                for c in self.incoming:
                    arrivals[c] = 0

        def fire(self, wave=0, now=None):
            out = original_fire(self, wave, now)
            for c in self.incoming:
                arrivals[c] = 0
            return out

        def decide(self, now):
            if self.refractory_at(now):
                return original_decide(self, now)
            for c in self.incoming:
                assert c.trace == arrivals[c]  # the engine's open arrivals are the ones counted here
            traces = [(c, arrivals[c]) for c in self.incoming if arrivals[c]]
            m = self.expected_spikes(now) if self.delta > 0.0 else 0.0
            w = factor(now, self.fired_at)
            fired = original_decide(self, now)
            if eligibility == "hebb":
                y = 1.0 if fired else 0.0
                count[self] += 1
                p = mirror[self]
                if p is None:
                    mirror[self] = y
                    assert self.expectation == y
                    return fired  # the first decision charges nothing
                c_, q = y, p
                mirror[self] = p + max(DECISION_MEMORY, 1.0 / count[self]) * (y - p)
                assert self.expectation == mirror[self]
            elif m > 0.0:
                c_, q = (m * math.exp(-m) / -math.expm1(-m), 0.0) if fired else (0.0, m)
            else:
                return fired
            weighed[0] |= w not in (0.0, 1.0)
            for c, x in traces:
                direct[c] += w * (c_ - q) * x
                charged[0] += 1
            return fired

        Neuron.decide, Neuron.settle, Neuron.fire = decide, settle, fire
        Connection.last_signal = property(integrated, integrate)
        try:
            rng = random.Random(5)
            for _ in range(30):
                for c in direct:
                    direct[c] = 0.0  # the read pays and clears the score; what was charged before is not charged again
                run_epoch(g, verbose=False, rng=rng)
                g.settle_scores()  # as the pay does first: the open arrivals' debit into the score
                for c in g.connections.values():
                    assert c.score == pytest.approx(direct[c], rel=1e-9, abs=1e-12)
                open_at_a_read[0] |= any(c.trace != 0.0 for c in g.connections.values())
        finally:
            Neuron.decide, Neuron.settle, Neuron.fire = original_decide, original_settle, original_fire
            del Connection.last_signal
        assert charged[0] > 1000 and open_at_a_read[0] and floored[0] > 100  # "Let them run!", and the floor did bite
        assert weighed[0] == isi_factor  # the factor took values other than 0 and 1 exactly when it was on
        if eligibility == "hebb":
            assert all(n.expectation is not None for n in g.all_neurons())
    finally:
        Neuron.tau, Neuron.isi_factor = was, was_factor


def test_the_isi_factor_has_the_shape_byron_asked_for():
    """§0.2 (Byron, September 17, 2026): f(t - ISI) with f(0) = 1, f(infinity) = 0 and f(-ISI) = -1 -- Fable's cubic
    rational, (3x - 1) / (1 + x^3) at x = t / ISI -- zero at a third of the target, largest at it, falling as 3 (ISI/t)^2;
    0 for a neuron that has never fired; on by default at 5.1 ms, and --no-isi-factor switches it off."""
    from walnutbutter import constants as C
    from walnutbutter.cli import build_parser
    from walnutbutter.neuron import isi_factor
    assert C.TARGET_ISI == 5.1 and C.ISI_FACTOR is True and Neuron.target_isi == C.TARGET_ISI
    assert build_parser().parse_args([]).isi_factor is True and build_parser().parse_args(["--no-isi-factor"]).isi_factor is False
    at = lambda t: isi_factor(100.0 + t, 100.0, 5.1)  # t ms after a spike at 100 ms
    assert at(0.0) == -1.0 and at(5.1) == 1.0 and abs(at(5.1 / 3)) < 1e-12
    assert at(5.0) < 1.0 and at(5.2) < 1.0 and at(5.0) == pytest.approx(0.99941, abs=1e-5)  # the peak is the target
    assert at(10.0) == pytest.approx(0.5718, abs=1e-4) and at(15.0) == pytest.approx(0.2959, abs=1e-4)
    assert at(5.1e4) == pytest.approx(3 * (5.1 / 5.1e4) ** 2, rel=1e-3) and at(1e300) == 0.0
    assert isi_factor(12.0, None, 5.1) == 0.0  # a neuron that has never fired is at t = infinity


@pytest.mark.parametrize("eligibility", ["hazard", "hebb"])
def test_the_three_engines_agree_with_the_isi_factor_off(eligibility):
    """§0.2's --no-isi-factor: the single-spike rule unweighed, as it ran before September 17, 2026, in every engine."""
    np = pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    was, was_factor = Neuron.tau, Neuron.isi_factor
    Neuron.tau, Neuron.isi_factor = math.inf, False
    try:
        mesh, twin = goo(), goo()
        mesh.set_delta(0.7)
        twin.set_delta(0.7)
        net = ArrayNetwork(twin)
        teachers = [Teacher(x, seed=7, rule="reinforce", eligibility=eligibility, target="copy", homeostasis=0.01, unstick=0.1)
                    for x in (mesh, net)]
        ids = range(1, len(mesh.connections) + 1)
        for _ in range(30):
            for t in teachers:
                t.epoch(verbose=False)
            assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
            assert np.allclose([mesh.connections[i].weight for i in ids], net.weight, rtol=1e-9, atol=0.0)
        if fast.available():
            g = goo()
            g.set_delta(0.7)
            parted = fast.compare(g, epochs=30, teacher=Teacher(g, seed=7, rule="reinforce", eligibility=eligibility, target="copy",
                                                                 homeostasis=0.01, unstick=0.1))
            assert parted == [], parted
    finally:
        Neuron.tau, Neuron.isi_factor = was, was_factor


@pytest.mark.parametrize("tau", [2.0, math.inf])
def test_the_three_engines_agree_under_the_hebb_eligibility(tau):
    """§6.7's single-spike rule (September 17, 2026) on goo under escape noise, per decision under the leak and per
    arrival under the evidence accumulator: every spike, decision count and expectation in every engine, scores and
    weights to a part in a billion in the arrays and to the bit in Rust, learning on."""
    np = pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    was = Neuron.tau
    Neuron.tau = tau
    try:
        mesh, twin = goo(), goo()
        mesh.set_delta(0.7)
        twin.set_delta(0.7)
        net = ArrayNetwork(twin)
        on_mesh = Teacher(mesh, seed=7, rule="reinforce", eligibility="hebb", target="copy", homeostasis=0.01, unstick=0.1)
        on_net = Teacher(net, seed=7, rule="reinforce", eligibility="hebb", target="copy", homeostasis=0.01, unstick=0.1)
        assert on_mesh.sigma == 0.0 and mesh.centred and net.centred and not mesh.tally
        ids = range(1, len(mesh.connections) + 1)
        for _ in range(40):
            on_mesh.epoch(verbose=False)
            on_net.epoch(verbose=False)
            assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
            assert [n.decisions for n in mesh.all_neurons()] == net.decisions.tolist()
            mine = [-1.0 if n.expectation is None else n.expectation for n in mesh.all_neurons()]
            assert np.allclose(mine, np.where(np.isnan(net.expectation), -1.0, net.expectation), rtol=1e-12, atol=0.0)
            assert np.allclose([mesh.connections[i].score for i in ids], net.score, rtol=1e-9, atol=1e-12)
            assert np.allclose([mesh.connections[i].weight for i in ids], net.weight, rtol=1e-9, atol=0.0)
        assert any(c.score != 0.0 for c in mesh.connections.values())
        assert all(n.expectation is not None for n in mesh.all_neurons())
        if fast.available():
            g = goo()
            g.set_delta(0.7)
            teacher = Teacher(g, seed=7, rule="reinforce", eligibility="hebb", target="copy", homeostasis=0.01, unstick=0.1)
            parted = fast.compare(g, epochs=60, teacher=teacher)
            assert parted == [], parted
    finally:
        Neuron.tau = was


def test_the_hebb_eligibility_charges_the_decisions_too():
    """§6.7: the centred rule charges the threshold's decisions too -- the trace is kept for it -- and the objects and Rust agree."""
    was = Neuron.tau
    Neuron.tau = math.inf
    try:
        g = goo()
        g.set_delta(ESCAPE_DELTA)  # §8.3: the reinforce rule refuses where the threshold decides
        teacher = Teacher(g, seed=7, rule="reinforce", eligibility="hebb", target="copy", homeostasis=0.0, unstick=0.0)
        assert g.traced and all(n.traced and n.centred for n in g.all_neurons())
        if fast.available():
            parted = fast.compare(g, epochs=30, teacher=teacher)
            assert parted == [], parted
        else:
            for _ in range(30):
                teacher.epoch(verbose=False)
        assert any(c.score != 0.0 for c in g.connections.values()) and any(n.expectation is not None for n in g.all_neurons())
    finally:
        Neuron.tau = was
