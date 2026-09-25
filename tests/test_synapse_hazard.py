"""Exploration at the synapse (AUTHORITY.md §6.13, §7.1, §7.5-§7.9, §8.16-§8.17, 5.4b): the object engine, and every
engine's refusal of what it does not carry yet (§12.2).

The hand-checked cases run on `Tiny`, a network wired edge by edge, with `Stream`, an exploration stream of chosen
uniforms, standing in for the run's stream (§7.3); the rest run on goo of 24 or 60 and a few epochs.
"""

from __future__ import annotations

import math
import random

import pytest

from walnutbutter import constants as C
from walnutbutter import fast
from walnutbutter.goo import Goo
from walnutbutter.learning import Teacher, class_sums, evidence_score, homeostasis, reinforce, unstick, update_rates
from walnutbutter.monitor import run_epoch
from walnutbutter.network import EXPLORATIONS, Network, escape_scale
from walnutbutter.neuron import Neuron
from walnutbutter.propagation import propagate

T0 = 10.0  # the clock at the hand-checked cases' first wave, ms


class Tiny(Network):
    """`count` neurons wired by hand: the first `inputs` are the input zone and the last `outputs` the output zone
    (§4.3), and the connections are made from `edges`, (source, target[, weight]), in the order given, ids from 1."""

    def __init__(self, count, edges, inputs=1, outputs=1, threshold=1.0, weight=0.5):
        self.across, self.rows, self.outputs, self.count = inputs, 2, outputs, count
        self.neurons = [Neuron(f"n{i}", threshold=threshold, minimum_potential=-4.0 * threshold) for i in range(count)]
        self.connections = {}
        for source, target, *given in edges:
            k = len(self.connections) + 1
            self.connections[k] = self.neurons[source].connect(self.neurons[target], k, given[0] if given else weight)
        self._rng = random.Random(0)
        self._init_network(inputs, C.WEIGHT_RANGE)

    def get_neuron_at(self, place, row):
        if row == 1 and 0 <= place < self.across:
            return self.neurons[place]
        if row == 0 and 0 <= place < self.outputs:
            return self.neurons[self.count - self.outputs + place]
        return None

    def output_width(self):
        return self.outputs


class Stream:
    """An exploration stream of chosen uniforms, then `fill` for ever, counting how many were taken (§3.8)."""

    def __init__(self, values=(), fill=1.0):
        self.values, self.fill, self.taken = list(values), fill, 0

    def random(self):
        self.taken += 1
        return self.values.pop(0) if self.values else self.fill


def chain(**settings) -> Tiny:
    """A -> B -> C, weights 0.25 and 0.5, thresholds 1: A the input, B hidden, C the output and its read synapse (§7.9).
    Every clock one hop back, so the first wave's decisions cover one hop of exposure."""
    net = Tiny(3, [(0, 1, 0.25), (1, 2, 0.5)])
    net.set_exploration("synapse", **settings)
    for neuron in net.neurons:
        neuron.exposed_since = T0 - Neuron.hop
    return net


def step(net, now, until, **kw) -> list:
    """The waves one `Network.propagate` runs, which appends them to the epoch's (and returns all of them)."""
    before = len(net.waves)
    net.propagate(now=now, until=until, **kw)
    return net.waves[before:]


def goo(count: int = 24, seed: int = 3, **kw) -> Goo:
    g = Goo(count=count, across=8 if count >= 60 else 4, seed=seed, weight=None, **kw)
    g.rule, g.drive, g.read = "reinforce", "rate", "count"
    return g


def entry(m: float, escapes: int, synapses: int, h: float, family: str, rest: float = C.SYNAPSE_HAZARD_REST) -> float:
    """§8.16's entry in its engine note's form: c only where m > 0, a c - (F - a) m, and rho~ on the finished entry."""
    credit = m * math.exp(-m) / -math.expm1(-m)
    posted = escapes * credit - (synapses - escapes) * m
    return (1.0 - rest) / h * posted if family == "linear" else posted


# --- the settings -------------------------------------------------------------------------------------------------


def test_the_default_that_does_not_move():
    """EXPLORATION is neuron until §7.6's conditions are met, so a network is built under the neuron rule, carrying the
    synapse settings a run that asks would take (the register's values are test_constants')."""
    g = goo()
    assert g.exploration == "neuron" and g.drive_steps == C.DRIVE_STEPS
    assert not any(n.synaptic for n in g.all_neurons())
    assert (g.synapse_hazard_rest, g.synapse_hazard_family, g.synapse_hazard_scaling, g.trace_mode) == (
        C.SYNAPSE_HAZARD_REST, C.SYNAPSE_HAZARD_FAMILY, C.SYNAPSE_HAZARD_SCALING, C.TRACE)


def test_setting_the_exploration_zeroes_every_width_traces_everyone_and_scales_each_source():
    """§6.13: every width 0 and ESCAPE_DELTA not consulted; §7.7: kappa_i the count's, or that over the fan-out, F_i
    counting an output's read synapse (§7.9); every neuron traced, and kept so through what used to recompute it."""
    count = goo()
    count.set_exploration("synapse")
    assert count.exploration == "synapse" and count.escape_delta == 0.0 and not count.hazard and count.explores
    assert all(n.delta == 0.0 and n.traced and n.synaptic for n in count.all_neurons())
    assert all(n.synapse_scale == escape_scale(24) for n in count.all_neurons())
    fan = goo()
    fan.set_exploration("synapse", scaling="fan-out")
    outputs = set(fan.output_row())
    for neuron in fan.all_neurons():
        synapses = len(neuron.outgoing) + (neuron in outputs)
        assert neuron.synapse_scale == (escape_scale(24) / synapses if synapses else escape_scale(24))
    assert any(not n.outgoing and n not in outputs for n in fan.all_neurons())  # a source with no synapse at all
    # the read synapse is not a connection of §1: no id, no place in the topology or the edge order (§7.9)
    assert len(fan.connections) == len(goo().connections) == len(fast.flatten(fan)[2])
    # traced survives the Teacher, set_delta(0) and centre(False), which recompute it under the neuron rule
    Teacher(count, rule="reinforce", eligibility="hazard", target="copy", seed=1)
    count.set_delta(0.0)
    count.centre(False)
    assert count.traced and all(n.traced and n.delta == 0.0 for n in count.all_neurons())


# --- the synapse's decision (§7.5, §7.6) ----------------------------------------------------------------------------


@pytest.mark.parametrize("scaling", ("count", "fan-out"))
@pytest.mark.parametrize("family", ("loglinear", "linear"))
def test_m_and_the_hazard_in_the_engine_note_s_forms(family, scaling):
    """§7.5's engine note: u = min(max(V, 0), theta) / theta; h0 ** (1 - u) or h0 + (1 - h0) * u; m = dt / hop, times
    kappa_i, times h, left to right, capped at 1e3 -- to the bit, and at u = 0, 1/2 and just below 1 what the families say.
    kappa_i is §7.7's, kappa(N) or kappa(N) / F_i, on two sources of two synapses each: A's two, and the output C's one and
    its read synapse (§7.9). At u = 0.1343... the linear form parts from h0 (1 - u) + u by an ulp: the note's is the one."""
    net = Tiny(3, [(0, 1, 0.25), (0, 2, 0.25), (1, 2, 0.5), (2, 1, 0.5)])
    net.set_exploration("synapse", family=family, scaling=scaling)
    a, _, c = net.neurons
    h0, linear = C.SYNAPSE_HAZARD_REST, family == "linear"
    uneven = 0.13436424411240122
    assert h0 + (1.0 - h0) * uneven != h0 * (1.0 - uneven) + uneven  # the premise
    for source, synapses in ((a, 2), (c, 1 + 1)):
        kappa = escape_scale(3) / synapses if scaling == "fan-out" else escape_scale(3)
        source.exposed_since = T0 - Neuron.hop
        for potential, u, want in ((0.0, 0.0, h0), (-0.3, 0.0, h0), (0.5, 0.5, math.sqrt(h0) if not linear else (1 + h0) / 2),
                                   (uneven, uneven, None), (math.nextafter(1.0, 0.0), math.nextafter(1.0, 0.0), 1.0)):
            source.potential = potential
            m, h = source.synapse_expected(T0, h0, linear)
            form = h0 + (1.0 - h0) * u if linear else h0 ** (1.0 - u)
            assert h == form and (want is None or h == pytest.approx(want, rel=1e-12))
            assert m == min((T0 - (T0 - Neuron.hop)) / Neuron.hop * kappa * form, 1e3)
    a.potential = 0.0
    a.exposed_since = T0 + 1.0  # within the slack of a recovery: dt is never negative (§6.5)
    assert a.synapse_expected(T0, h0, linear) == (0.0, h0)
    a.exposed_since = -1e9
    assert a.synapse_expected(T0, h0, linear)[0] == 1e3  # capped
    # under the leak u reads V_i(t), the potential decayed for the time since it was last brought up to date (§2.3)
    was = Neuron.tau
    Neuron.tau = 2.0
    try:
        a.exposed_since = T0 - Neuron.hop
        a.potential, a.last_update = 0.5, T0 - 2.0
        u = min(max(0.5 * math.exp(-(T0 - (T0 - 2.0)) / 2.0), 0.0), a.threshold) / a.threshold
        form = h0 + (1.0 - h0) * u if linear else h0 ** (1.0 - u)
        kappa = escape_scale(3) / 2 if scaling == "fan-out" else escape_scale(3)
        assert a.synapse_expected(T0, h0, linear) == (min((T0 - (T0 - Neuron.hop)) / Neuron.hop * kappa * form, 1e3), form)
    finally:
        Neuron.tau = was


def test_escape_is_strictly_below_the_chance():
    """§7.5: a synapse escapes iff its uniform is strictly below P = -expm1(-m); at P itself it does not -- an output's read
    synapse as any synapse does (§7.9)."""
    for draw, escapes in ((None, False), ("under", True)):
        net = chain()
        a = net.neurons[0]
        m, _ = a.synapse_expected(T0, C.SYNAPSE_HAZARD_REST, False)
        chance = -math.expm1(-m)
        net.explore_rng = Stream([chance if draw is None else math.nextafter(chance, 0.0), 1.0, 1.0])
        net.propagate(inputs={net.neurons[2]: 0.0}, now=T0, until=T0 + 1.0)
        assert [ventured for _, _, ventured in net.schedule.pending(marks=True)] == ([True] if escapes else [])
    for draw, counted in ((None, 0), ("under", 1)):
        net = chain()
        c = net.neurons[2]
        m, _ = c.synapse_expected(T0, C.SYNAPSE_HAZARD_REST, False)
        chance = -math.expm1(-m)
        net.explore_rng = Stream([1.0, 1.0, chance if draw is None else math.nextafter(chance, 0.0)])
        net.propagate(inputs={net.neurons[1]: 0.0}, now=T0, until=T0 + 1.0)
        assert c.read_count == counted and not net.schedule.pending()


def test_a_rest_hazard_of_zero_under_the_loglinear_family_is_the_deterministic_network_still_drawing():
    """§7.6: h0 = 0 under the loglinear family spikes as §6.9's deterministic network does, word for word, and its
    synapses still take their E + O draws every wave (§3.8, §7.1); it posts nothing, and no neuron's own expectation
    moves (§8.6)."""
    plain, synaptic = goo(60), goo(60)
    synaptic.set_exploration("synapse", h0=0.0)
    draws = len(synaptic.connections) + synaptic.outputs
    a, b = random.Random(1), random.Random(1)
    waves = 0
    for _ in range(4):
        run_epoch(plain, verbose=False, rng=a)
        waves += len(run_epoch(synaptic, verbose=False, rng=b))
        assert [n.spikes for n in plain.all_neurons()] == [n.spikes for n in synaptic.all_neurons()]
    assert sum(n.spikes for n in synaptic.all_neurons()) > 0 and waves > 0
    fresh = random.Random(1)
    for _ in range(draws * waves):
        fresh.random()
    assert b.getstate() == fresh.getstate()
    assert a.getstate() == random.Random(1).getstate()  # the deterministic network draws nothing
    synaptic.settle_scores()
    assert all(n.gain == 0.0 and n.read_count == 0 for n in synaptic.all_neurons())
    assert all(c.score == 0.0 for c in synaptic.connections.values())
    assert all(n.expectation is None and n.decisions == 0 for n in synaptic.all_neurons())


def test_every_wave_takes_e_plus_o_draws_whatever_fires():
    """§3.8: one uniform per synapse in edge order, then one per output for its read synapse, at every wave, whether or
    not each can be used -- so the stream's position depends only on the counts."""
    net = goo(60)
    net.set_exploration("synapse")
    stream = Stream(fill=0.5)
    counts = []
    original = net._explore_synapses

    def counted(time):
        before = stream.taken
        original(time)
        counts.append(stream.taken - before)

    net._explore_synapses = counted
    fired = 0
    for _ in range(3):
        run_epoch(net, verbose=False, rng=stream)
        fired += sum(1 for w in net.waves if w.fired)
    assert fired > 0 and counts and set(counts) == {len(net.connections) + net.outputs}


def test_an_output_neuron_at_two_places_has_one_read_synapse(monkeypatch):
    """§7.9, §3.8: every output neuron has one read synapse, whatever places it holds (§0.5 lets a neuron sit in
    several zones); the draws are E, then one per output neuron in the order the output row first names it."""
    net = goo(60)
    net.set_exploration("synapse")
    row = net.output_row()
    monkeypatch.setattr(net, "output_row", lambda: row + [row[0]])
    stream = Stream(fill=0.5)
    counts = []
    original = net._explore_synapses

    def counted(time):
        before = stream.taken
        original(time)
        counts.append(stream.taken - before)

    net._explore_synapses = counted
    run_epoch(net, verbose=False, rng=stream)
    assert counts and set(counts) == {len(net.connections) + len(row)}


def test_the_draws_are_laid_out_in_edge_order_then_output_order():
    """§3.8, §3.6: synapse k of the flattened order takes uniform k and output j's read synapse uniform E + j; the wave's
    escapes are pushed in edge order, sources in index order and each source's synapses as the topology was built."""
    net = goo()
    net.set_exploration("synapse", h0=0.5)
    edges = [c for n in net.all_neurons() for c in n.outgoing]
    total = len(edges) + net.outputs
    chosen = set(range(0, len(edges), 3)) | {len(edges), len(edges) + 2}
    net.explore_rng = Stream([0.0 if k in chosen else 1.0 for k in range(total)])
    for neuron in net.all_neurons():
        neuron.exposed_since = T0 - Neuron.hop
    net.propagate(inputs={net.neurons[5]: 0.0}, now=T0, until=T0 + 1.0)
    assert not net.waves[-1].fired  # a quiet moment: every source decides
    assert net.schedule.pending(marks=True) == [(T0 + Neuron.hop, edges[k], True) for k in sorted(chosen) if k < len(edges)]
    assert [n.read_count for n in net.output_row()] == [1 if len(edges) + j in chosen else 0 for j in range(net.outputs)]
    # a source that spikes leaves its synapses' uniforms unused, and the next source takes its own: A fires, A -> B's
    # uniform is the one below every chance, and B -> C, whose uniform is 1, does not escape
    spiked = chain()
    a, _, c = spiked.neurons
    spiked.explore_rng = Stream([0.0, 1.0, 1.0])  # A -> B's, B -> C's, then C's read synapse's
    spiked.propagate(fire=[a], now=T0, until=T0 + 1.0)
    assert spiked.schedule.pending(marks=True) == [(T0 + Neuron.hop, spiked.connections[1], False)]
    assert c.read_count == 0


# --- the ventured signal and the read synapse (§7.9) --------------------------------------------------------------------


@pytest.mark.parametrize("family", ("loglinear", "linear"))
def test_a_hand_checked_chain(family):
    """A at half its threshold, B at it, C at rest; every draw of the wave below P. B spikes by the comparison (§6.13), so
    its synapse transmits with the spike and does not decide; A's synapse escapes, and so does C's read synapse. The
    spike's signal is pushed before the escape (§3.6), the escape carries its mark (§7.9), the read escape counts at its
    decision (§5.10), and only A posts: C decided at V = 0 (§8.16)."""
    net = chain(family=family)
    a, b, c = net.neurons
    ab, bc = net.connections[1], net.connections[2]
    stream = Stream([0.0, 0.0, 0.0])
    net.explore_rng = stream
    a.potential, b.potential = 0.5, 1.0
    m, h = a.synapse_expected(T0, C.SYNAPSE_HAZARD_REST, family == "linear")
    waves = step(net, inputs={b: 0.0}, now=T0, until=T0 + Neuron.hop / 2)
    assert len(waves) == 1 and waves[0].fired == [b] and stream.taken == 3  # E + O, B's own draw taken though unused
    assert net.schedule.pending(marks=True) == [(T0 + Neuron.hop, bc, False), (T0 + Neuron.hop, ab, True)]
    assert (c.read_count, c.spikes, net.output_counts()) == (1, 0, [1])
    assert (a.potential, a.spikes, a.fired_at, a.has_fired, a.rate, a.rate_level) == (0.5, 0, None, False, 0.5, 0.0)  # an
    # escape moves nothing of its source's (§7.5), and a read escape nothing of the output's: it is not a spike (§2.6)
    assert (c.has_fired, c.fired_at, c.rate_level) == (False, None, 0.0)
    rate = c.rate
    update_rates(net)
    assert c.rate < rate  # the rate memory reads the output's own spikes, and it has none (§2.6, §5.10)
    assert a.exposed_since == b.exposed_since == c.exposed_since == T0  # one clock a source; B's from its spike (§7.5)
    assert a.gain == entry(m, 1, 1, h, family) != 0.0 and b.gain == c.gain == 0.0
    assert ab.score == 0.0  # the escape of A -> B credits the synapses into A, not A -> B itself (§8.16)
    # one hop on: the relayed signal is summed first; B is refractory, so the ventured one is dropped, recorded as delivered
    (wave,) = step(net, now=T0 + Neuron.hop, until=T0 + 1.5 * Neuron.hop)
    assert wave.delivered == [bc, ab] and wave.ventured == [1]
    assert [s.ventured for s in wave.signals()] == [False, True]
    assert (b.potential, ab.trace, ab.last_signal) == (0.0, 0.0, None)
    assert (c.potential, bc.trace, bc.last_signal) == (0.5, 1.0, T0 + Neuron.hop)
    # the read count is the epoch's: it reaches every count reader and is zeroed at the reset (§5.10)
    assert net.output_counts_hz() == [1000.0 / net.interval]
    net.reset()
    assert c.read_count == 0 and net.output_counts() == [0]


@pytest.mark.parametrize("family", ("loglinear", "linear"))
def test_an_output_posts_with_its_read_synapse_among_its_decisions(family):
    """§8.16, §7.9: an output above zero posts for its F_i synapses, its read synapse among them, a_i of them escaping, a
    read escape among those. C holds B's arrival (V = 1/2, the arrival open on B -> C), sends to A, and has its read
    synapse: F = 2. Its entry is §8.16's form at a = 1 when the read synapse escapes and C -> A does not, at a = 0 when
    neither does and at a = 2 when both do; the read settles it into B -> C's score as x G - B (§1.9)."""
    for toward_a, read, escapes in ((1.0, 0.0, 1), (1.0, 1.0, 0), (0.0, 0.0, 2)):
        net = Tiny(3, [(0, 1, 0.25), (1, 2, 0.5), (2, 0, 0.5)])
        net.set_exploration("synapse", family=family)
        a, b, c = net.neurons
        bc = net.connections[2]
        for neuron in net.neurons:
            neuron.exposed_since = T0 - Neuron.hop
        c.potential, bc.trace = 0.5, 1.0  # B's signal integrated when C's gain was 0, so it noted 0
        m, h = c.synapse_expected(T0, C.SYNAPSE_HAZARD_REST, family == "linear")
        net.explore_rng = Stream([1.0, 1.0, toward_a, read])  # A -> B, B -> C, C -> A in edge order, then C's read synapse
        net.propagate(inputs={b: 0.0}, now=T0, until=T0 + 1.0)
        assert (c.read_count, c.spikes) == (1 if read == 0.0 else 0, 0)
        assert [ventured for _, _, ventured in net.schedule.pending(marks=True)] == ([True] if toward_a == 0.0 else [])
        posted = entry(m, escapes, 2, h, family)
        assert c.gain == posted != 0.0 and a.gain == b.gain == 0.0  # A and B decided at V = 0 and post nothing
        net.settle_scores()
        assert (bc.score, bc.noted, bc.trace, c.gain) == (posted, posted, 1.0, posted)


@pytest.mark.parametrize("family", ("loglinear", "linear"))
def test_a_whisper_s_credit_keeps_its_precision(family):
    """§8.16's engine note: c = m * exp(-m) / -expm1(-m), in that form. B sits just above rest, V = 0.01, and its one
    synapse escapes: at so small an m the form 1 - exp(-m) would lose the low bits the note's keeps."""
    net = chain(family=family)
    a, b, c = net.neurons
    b.potential = 0.01
    m, h = b.synapse_expected(T0, C.SYNAPSE_HAZARD_REST, family == "linear")
    net.explore_rng = Stream([1.0, 0.0, 1.0])
    net.propagate(inputs={c: 0.0}, now=T0, until=T0 + 1.0)
    assert [ventured for _, _, ventured in net.schedule.pending(marks=True)] == [True]
    assert b.gain == entry(m, 1, 1, h, family) and a.gain == c.gain == 0.0


def test_a_ventured_arrival_is_a_signal_in_every_other_way():
    """§7.9: integrated into the potential and, under TRACE all, the trace (§8.17), noting the gain as it stood before the
    wave's own entries (§8.16), and recorded on the stamp (§1.7)."""
    net = chain()
    a, b, c = net.neurons
    ab = net.connections[1]
    a.potential, b.gain = 0.5, 0.3
    net.explore_rng = Stream([0.0, 1.0, 1.0])
    net.propagate(inputs={c: 0.0}, now=T0, until=T0 + 1.0)
    net.propagate(now=T0 + Neuron.hop, until=T0 + 1.5 * Neuron.hop)
    assert (b.potential, ab.trace, ab.noted, ab.last_signal, ab.trace_at) == (0.25, 1.0, 0.3, T0 + Neuron.hop, T0 + Neuron.hop)
    assert b.gain != 0.3  # B then decided on its new potential and posted: after the arrival had noted


def test_a_source_that_hears_no_one_and_sends_nowhere_keeps_its_clock():
    """§7.5: one exposure clock per source, brought to t at every wave for every source that did not spike -- F_i = 0
    included, which takes no draw and makes no decision."""
    net = Tiny(4, [(0, 1), (0, 3), (1, 3)])  # neuron 2 is hidden, with no synapse in or out
    net.set_exploration("synapse")
    stream = Stream()
    net.explore_rng = stream
    for t in (T0, T0 + 1.0):
        net.propagate(inputs={net.neurons[0]: 0.0}, now=t, until=t + 0.5)
        assert net.neurons[2].exposed_since == t
    assert stream.taken == 2 * (3 + 1)


def test_a_refractory_source_whispers_at_rest_on_a_clock_run_from_its_spike():
    """§7.8: a spike resets its source and its synapses go on deciding at u = 0, the rest hazard, over the exposure since
    the spike -- not suspended, and not resumed at the period's end."""
    net = chain()
    a, _, c = net.neurons
    ab = net.connections[1]
    net.explore_rng = Stream()
    net.propagate(fire=[a], now=T0, until=T0 + 0.5)
    assert (a.fired_at, a.exposed_since, a.potential) == (T0, T0, 0.0)
    for t, below in ((T0 + 1.0, False), (T0 + 2.0, True)):
        assert a.refractory_at(t)
        m = min((t - a.exposed_since) / Neuron.hop * a.synapse_scale * C.SYNAPSE_HAZARD_REST ** (1.0 - 0.0), 1e3)
        chance = -math.expm1(-m)
        net.explore_rng = Stream([math.nextafter(chance, 0.0) if below else chance, 1.0, 1.0])
        net.propagate(inputs={c: 0.0}, now=t, until=t + 0.5)
        assert a.exposed_since == t
    assert net.schedule.pending(marks=True) == [(T0 + Neuron.hop, ab, False), (T0 + 2.0 + Neuron.hop, ab, True)]


# --- the rule at the synapses' decisions (§8.16, §8.17) -------------------------------------------------------------


@pytest.mark.parametrize("tau", (math.inf, 2.0))
def test_a_source_driven_below_zero_posts_nothing(tau):
    """§8.16 (Byron, September 25, 2026, Q4): the wave posts for a source only while V_i > 0 -- at or below zero the clip
    holds its hazard at h0 whatever the synapses into it delivered. K inhibits S below zero, S's synapse escapes, and S
    posts nothing; excited instead, the same wave posts."""
    was = Neuron.tau
    Neuron.tau = tau
    try:
        for weight, posts in ((-0.5, False), (0.5, True)):
            net = Tiny(3, [(0, 1, weight), (1, 2)])
            net.set_exploration("synapse")
            k, s, _ = net.neurons
            ks = net.connections[1]
            net.explore_rng = Stream([1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
            net.propagate(fire=[k], now=T0, until=T0 + 1.0)
            net.propagate(now=T0 + Neuron.hop, until=T0 + 1.5 * Neuron.hop)
            assert s.potential == weight and ks.trace == 1.0
            assert [v for _, _, v in net.schedule.pending(marks=True)] == [True]  # S's synapse escaped either way
            net.settle_scores()
            assert (s.gain != 0.0 if tau == math.inf else ks.score != 0.0) is posts
            if not posts:
                assert s.gain == 0.0 and ks.score == 0.0
    finally:
        Neuron.tau = was


@pytest.mark.parametrize("family", ("loglinear", "linear"))
def test_under_the_leak_the_walk_posts_on_the_leaked_trace_and_the_decayed_potential(family):
    """§8.16, §8.12 at TAU 2: the walk posts the wave's entry times each trace leaked since it was brought up to date,
    x e^{-(t - t_x)/TAU} (8.5); and the entry's m reads V_i(t), the potential decayed for the time since it was last
    brought up to date (§7.5, §2.3). B heard A 2 ms before it decides and an external input 1 ms before, and is not
    touched at the wave: its trace is 2 ms old and its potential 1 ms old. Checked against the forms computed here."""
    was, h0 = Neuron.tau, C.SYNAPSE_HAZARD_REST
    Neuron.tau = 2.0
    try:
        for draw, escapes in ((1.0, 0), (0.0, 1)):
            net = chain(family=family)
            a, b, c = net.neurons
            ab = net.connections[1]
            b.potential, b.last_update = 0.5, T0 - 1.0
            ab.trace, ab.trace_at = 1.0, T0 - 2.0
            u = min(max(0.5 * math.exp(-(T0 - (T0 - 1.0)) / 2.0), 0.0), b.threshold) / b.threshold
            h = h0 + (1.0 - h0) * u if family == "linear" else h0 ** (1.0 - u)
            m = min((T0 - (T0 - Neuron.hop)) / Neuron.hop * escape_scale(3) * h, 1e3)
            net.explore_rng = Stream([1.0, draw, 1.0])
            net.propagate(inputs={c: 0.0}, now=T0, until=T0 + 1.0)
            assert [ventured for _, _, ventured in net.schedule.pending(marks=True)] == [True] * escapes
            assert ab.score == entry(m, escapes, 1, h, family) * 1.0 * math.exp(-(T0 - (T0 - 2.0)) / 2.0) != 0.0
            assert (ab.trace, ab.trace_at, b.gain) == (1.0, T0 - 2.0, 0.0)  # the walk moves no trace, and there is no gain
    finally:
        Neuron.tau = was


@pytest.mark.parametrize("family", ("loglinear", "linear"))
def test_the_gain_settles_what_posting_every_wave_would(family):
    """§8.16's bookkeeping is its entry regrouped, as §8.11's is §8.4's. At a TAU so long that exp(-dt / TAU) == 1.0 the
    leak's path runs the accumulator's dynamics to the bit and posts every wave's entry directly, by the fan-in walk; the
    accumulator's gain settles the same sums at spikes, the floor, forced spikes, a discharge and the reads -- within
    rounding, since a regrouping is not required to reproduce the sum bit for bit (§8.11)."""
    grouped, direct = goo(60, seed=5), goo(60, seed=5)
    for net in (grouped, direct):
        net.set_exploration("synapse", h0=0.05, family=family)
    a, b = random.Random(9), random.Random(9)
    was, posted, gained = Neuron.tau, 0, False
    try:
        for epoch in range(6):
            Neuron.tau = math.inf
            run_epoch(grouped, verbose=False, rng=a, discharge=epoch == 3)
            grouped.settle_scores()
            Neuron.tau = 1e300
            assert math.exp(-1e6 / Neuron.tau) == 1.0
            run_epoch(direct, verbose=False, rng=b, discharge=epoch == 3)
            direct.settle_scores()  # nothing to settle: the walk posted as it went
            assert [n.spikes for n in grouped.all_neurons()] == [n.spikes for n in direct.all_neurons()]
            assert grouped.output_counts() == direct.output_counts()
            mine = [c.score for c in grouped.connections.values()]
            theirs = [c.score for c in direct.connections.values()]
            assert mine == pytest.approx(theirs, rel=1e-9, abs=1e-12)
            posted += sum(1 for s in theirs if s)
            gained = gained or any(n.gain != 0.0 for n in grouped.all_neurons())  # a gain left open across a read
    finally:
        Neuron.tau = was
    assert posted > 20 and gained
    assert any(n.forced for n in grouped.all_neurons())  # the rate drive forced spikes


def test_the_settles_the_read_and_the_reset():
    """§8.16: a settle -- spike, floor, forced spike, discharge -- posts x G - B, clears x and B and restarts G, per
    neuron and whether or not an arrival is open; the read posts the same and re-bases B = x G, leaving x and G (§1.9);
    the epoch's reset re-bases as the read does."""
    net = chain()
    a, b, c = net.neurons
    ab = net.connections[1]

    def open_(trace=2.0, noted=0.3, gain=0.5, score=0.1):
        ab.trace, ab.noted, b.gain, ab.score = trace, noted, gain, score

    open_()
    net.settle_scores()
    assert (ab.score, ab.noted, ab.trace, b.gain) == (0.1 + (2.0 * 0.5 - 0.3), 2.0 * 0.5, 2.0, 0.5)
    open_()
    net.reset()
    assert (ab.score, ab.noted, ab.trace, b.gain) == (0.0, 2.0 * 0.5, 2.0, 0.5)
    open_()
    b.fire(0, T0)
    assert (ab.score, ab.noted, ab.trace, b.gain, b.exposed_since) == (0.1 + (2.0 * 0.5 - 0.3), 0.0, 0.0, 0.0, T0)
    open_()
    b.potential = -10.0
    b.settle()  # the floor bites
    assert (ab.score, ab.noted, ab.trace, b.gain) == (0.1 + (2.0 * 0.5 - 0.3), 0.0, 0.0, 0.0)
    open_()
    net.reset(discharge=True)
    assert (ab.noted, ab.trace, b.gain) == (0.0, 0.0, 0.0)
    a.gain = c.gain = 0.7  # no open arrival on A (it hears no one) nor on C: G restarts all the same
    a.potential = c.potential = -10.0
    a.settle()
    c.reset(discharge=True)
    assert a.gain == c.gain == 0.0


@pytest.mark.parametrize("tau", (math.inf, 2.0))
def test_under_trace_ventured_a_relayed_arrival_is_not_there_for_the_synapse(tau):
    """§8.17 (Byron, September 25, 2026, Q5): under TRACE ventured a relayed arrival notes nothing, leaves x as it is, and
    neither decays it nor moves the moment it was brought up to -- while the potential takes it. A ventured one counts."""
    was = Neuron.tau
    Neuron.tau = tau
    try:
        net = chain(trace="ventured")
        a, b, c = net.neurons
        bc = net.connections[2]
        bc.trace, bc.trace_at, bc.noted = 1.0, 3.0, 0.2
        c.threshold = 2.0  # so that C takes both arrivals without spiking
        for neuron in net.neurons:
            neuron.last_update = T0  # under the leak, the potentials set here stand at T0
        net.explore_rng = Stream()
        b.potential = 1.0
        net.propagate(inputs={b: 0.0}, now=T0, until=T0 + 1.0)
        c.gain = 0.7
        net.propagate(now=T0 + Neuron.hop, until=T0 + 1.5 * Neuron.hop)
        assert c.potential == 0.5 and bc.last_signal == T0 + Neuron.hop
        assert (bc.trace, bc.trace_at, bc.noted) == (1.0, 3.0, 0.2)
        t = T0 + 20.0
        b.potential = 0.6
        net.explore_rng = Stream([1.0, 0.0, 1.0])
        net.propagate(inputs={a: 0.0}, now=t, until=t + 1.0)
        gain = c.gain
        net.propagate(now=t + Neuron.hop, until=t + 1.5 * Neuron.hop)
        arrived = t + Neuron.hop
        if tau == math.inf:
            assert (bc.trace, bc.noted) == (2.0, 0.2 + gain)
        else:
            assert (bc.trace, bc.noted) == (1.0 * math.exp(-(arrived - 3.0) / tau) + 1.0, 0.2)
        assert bc.trace_at == arrived
    finally:
        Neuron.tau = was


def test_the_read_count_reaches_every_count_reader_and_nothing_that_reads_a_spike():
    """§5.10: the count is the output's spikes plus its read synapse's escapes this epoch, and every clause that reads a
    count reads the sum -- the class evidence and the evidence critic among them; the rate memory reads spikes (§2.6)."""
    g = Goo(count=24, across=4, outputs=12, seed=1, weight=None)
    g.population, g.read = 3, "count"
    g.set_exploration("synapse")
    outputs = g.output_row()
    for neuron, k in zip(outputs, (0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 5, 0)):
        neuron.read_count = k
    g.input_label = 0
    assert g.output_counts() == [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 5, 0]
    assert g.output_counts_hz() == [k * (1000.0 / g.interval) for k in g.output_counts()]
    assert class_sums(g) == [0 + 1 + 2 - (0 + 0 + 0), 3 - (0 + 5 + 0)]
    assert evidence_score(g) == pytest.approx(math.log(math.exp(3 / 2) / (math.exp(3 / 2) + math.exp(-2 / 2))))
    assert g.output_fired() == [k >= g.pickiness for k in g.output_counts()]
    rates = [n.rate for n in g.all_neurons()]
    update_rates(g)
    assert all(n.rate < r for n, r in zip(g.all_neurons(), rates))  # no neuron spiked: the escapes are not firing
    g.reset()
    assert g.output_counts() == [0] * 12


def test_a_run_learns_under_the_reinforce_rule_and_stays_finite():
    """The reinforce rule runs under exploration at the synapse, posted at the synapses' decisions (§8.3, §8.16)."""
    g = goo(60)
    g.set_exploration("synapse")
    before = [c.weight for c in g.connections.values()]
    teacher = Teacher(g, seed=7, rule="reinforce", eligibility="hazard", target="copy")
    for _ in range(4):
        teacher.epoch(verbose=False)
    after = [c.weight for c in g.connections.values()]
    assert after != before and all(math.isfinite(w) for w in after)
    assert all(math.isfinite(n.gain) for n in g.all_neurons())


# --- the charged drive (5.4b) -------------------------------------------------------------------------------------


def charged(count=3, edges=((0, 1, 0.25), (1, 2, 0.5)), **settings) -> Tiny:
    net = Tiny(count, list(edges))
    net.set_exploration("synapse", **settings)
    net.drive = "charged"
    net.explore_rng = Stream()
    return net


def test_the_charged_arrivals_are_the_rate_drive_s_at_three_times_the_rate():
    """5.4b: drawn as 5.4 draws them, from the network's stream and in the same order, at DRIVE_STEPS times the rate; and
    nothing at the epoch's moment (§5.7)."""
    g, twin = goo(), goo()
    g.set_exploration("synapse")
    g.drive = "charged"
    twin.input_rate = C.INPUT_RATE * C.DRIVE_STEPS
    bits = [True, False]
    g.set_input_bits(bits, 0.0)
    twin.set_input_bits(bits, 0.0)
    events = g.input_schedule()
    assert events == twin.input_schedule() and events and all(when > g.time for _, when in events)


def test_a_charge_delivers_the_threshold_it_finds_over_drive_steps():
    """5.4b (Byron, September 25, 2026, Q10): theta_i / DRIVE_STEPS, a division, on the threshold the input holds when the
    delivery lands; it touches the input, comes along no synapse -- no trace, note, stamp or score moves -- and forces no
    spike."""
    net = charged(edges=((1, 0, 0.5), (0, 1, 0.25), (1, 2, 0.5)))  # A hears B, so A has a synapse a charge must not move
    a = net.neurons[0]
    into_a = net.connections[1]
    into_a.trace, into_a.noted, into_a.last_signal, into_a.score = 1.0, 0.2, 5.0, 0.1
    stamp = a.touched_stamp
    net.schedule.charge(a, C.DRIVE_STEPS, T0)
    threshold = C.GOO_THRESHOLD * (13 / C.THRESHOLD_FAN_IN)  # goo's at an in-degree of 13 (§4.10)
    assert threshold / C.DRIVE_STEPS != threshold * (1.0 / C.DRIVE_STEPS)  # the premise: the two forms part here
    a.threshold = threshold  # moved after the charge was scheduled: the charge reads the threshold at delivery
    (wave,) = step(net, now=T0, until=T0 + 1.0)
    assert a.potential == threshold / C.DRIVE_STEPS and a.forced and not wave.fired and wave.time == T0
    assert a.touched_stamp != stamp  # touched, as a signal touches its target (§3.5)
    assert (into_a.trace, into_a.noted, into_a.last_signal, into_a.score) == (1.0, 0.2, 5.0, 0.1)


def test_a_wave_takes_its_charges_before_its_signals(monkeypatch):
    """§3.5, 5.4b: a charge due at the moment of a signal is integrated before it, in whichever order the two fell inside
    the clock's slack, and the charge's exact time anchors the wave (§3.4)."""
    net = charged()
    a, b, _ = net.neurons
    ab = net.connections[1]
    taken = []
    original = Neuron.receive

    def recorded(self, amount, now=None):
        taken.append((self.name, amount))
        return original(self, amount, now)

    monkeypatch.setattr(Neuron, "receive", recorded)
    net.schedule.signal(ab, T0 - 1e-13)  # a hair earlier, within the slack: the heap pops it first
    net.schedule.charge(b, C.DRIVE_STEPS, T0)
    (wave,) = step(net, now=T0, until=T0 + 1.0)
    assert taken == [("n1", 1.0 / C.DRIVE_STEPS), ("n1", 0.25)] and wave.time == T0


def test_a_charged_input_is_asked_before_the_neurons_the_wave_s_signals_touched():
    """§3.5, §3.6: a charge touches its input as a signal touches its target, and before the wave's signals, so the fire
    phase asks the input first and its spike's signals are pushed first (§3.7). A sits two thirds of the way to its
    threshold and takes its last charge while S's signal takes B over its threshold in the same wave: A fires, then B."""
    net = charged(count=4, edges=((2, 1, 1.0), (0, 3, 0.5), (1, 3, 0.5)), h0=0.0)  # A = n0, B = n1, S = n2, the output n3
    a, b = net.neurons[:2]
    sb, a_out, b_out = net.connections[1], net.connections[2], net.connections[3]
    a.potential = 1.0 - 1.0 / C.DRIVE_STEPS
    assert a.potential + 1.0 / C.DRIVE_STEPS >= a.threshold  # the premise: this charge takes A to its threshold
    net.schedule.signal(sb, T0)
    net.schedule.charge(a, C.DRIVE_STEPS, T0)
    (wave,) = step(net, now=T0, until=T0 + 1.0)
    assert wave.fired == [a, b] and wave.delivered == [sb]
    assert net.schedule.pending(marks=True) == [(T0 + Neuron.hop, a_out, False), (T0 + Neuron.hop, b_out, False)]


def test_under_the_neuron_rule_an_external_input_is_taken_where_the_wave_holds_it():
    """§3.5 takes the charged drive's deliveries before the wave's signals, and nothing else: an external input of a given
    amount (Network.propagate's `inputs`) is taken in the wave's own order, as it was before the charged drive existed, so
    the neuron rule's sums do not move. V = 0.3, a signal of 0.6 a hair before an external 0.1 inside the slack: summed in
    that order they are 0.9999999999999999, short of the threshold of 1, where the other order reaches it."""
    net = Tiny(2, [(0, 1, 0.6)])
    d = net.neurons[1]
    d.potential = 0.3
    assert 0.3 + 0.1 + 0.6 >= d.threshold > 0.3 + 0.6 + 0.1  # the premise
    net.schedule.signal(net.connections[1], T0 - 1e-13)
    (wave,) = step(net, inputs={d: 0.1}, now=T0, until=T0 + 1.0)
    assert d.potential == 0.3 + 0.6 + 0.1 and not wave.fired and d.spikes == 0 and wave.time == T0


def test_a_fourth_charge_where_three_fall_a_rounding_short():
    """5.4b: from rest after DRIVE_STEPS deliveries, or after one more where their sum falls a rounding short of the
    threshold, accepted as it falls. At a fan-in of 35, 0.2 * 35/18 over three, summed three times, is one ulp short."""
    net = charged(h0=0.0)
    a = net.neurons[0]
    a.threshold = 0.2 * (35 / 18.0)
    total = 0.0
    for _ in range(C.DRIVE_STEPS):
        total += a.threshold / C.DRIVE_STEPS
    assert total < a.threshold  # the premise
    for k in range(C.DRIVE_STEPS + 1):
        net.schedule.charge(a, C.DRIVE_STEPS, T0 + k)
        net.propagate(now=T0 + k, until=T0 + k + 0.5)
        assert a.spikes == (1 if k == C.DRIVE_STEPS else 0)


def test_a_charge_dropped_at_a_refractory_input_still_marks_it():
    """5.4b, 5.8 (Byron, September 25, 2026, Q8): a delivery dropped at a refractory input is still delivered (§1.7), and
    any delivery sets the driven mark."""
    net = charged()
    a = net.neurons[0]
    a.fired_at = T0 - 1.0
    stamp = a.touched_stamp
    net.schedule.charge(a, C.DRIVE_STEPS, T0)
    (wave,) = step(net, now=T0, until=T0 + 1.0)
    assert a.potential == 0.0 and a.forced and not wave.fired and a.touched_stamp == stamp


def test_the_driven_mark_is_read_as_forced_by_the_pay_the_rate_memory_and_the_thresholds():
    """5.8, §8.1, §2.6, §9.2 (Byron, September 25, 2026, Q9): an input the charged drive delivered to this epoch is
    passed over by the pay, the rate memory, homeostasis and un-sticking, as a forced one is."""
    net = charged(edges=((1, 0, 0.5), (0, 1, 0.25), (1, 2, 0.5)))
    a, b, _ = net.neurons
    net.set_input([True])
    net.fire_input()
    assert a.forced and not b.forced and net.input_events
    into_a, into_b = net.connections[1], net.connections[2]
    into_a.score = into_b.score = 1.0
    rates = (a.rate, b.rate)
    update_rates(net)
    assert a.rate == rates[0] and b.rate != rates[1]
    thresholds = (a.threshold, b.threshold)
    homeostasis(net, 0.1, 0.4)
    assert a.threshold == thresholds[0] and b.threshold != thresholds[1]
    a.rate = b.rate = 0.0  # stuck off, both
    thresholds = (a.threshold, b.threshold)
    unstick(net, 0.1, 0.5)
    assert a.threshold == thresholds[0] and b.threshold != thresholds[1]
    weights = (into_a.weight, into_b.weight)
    reinforce(net, 1.0, lr=0.1, eligibility="hazard")
    assert into_a.weight == weights[0] and into_b.weight != weights[1]


def test_a_charged_input_spikes_only_by_the_comparison():
    """5.4b: the deliveries force no spike; an input spikes when its potential reaches its threshold (§6.13)."""
    g = goo(60)
    g.set_exploration("synapse")
    g.drive = "charged"
    stream = random.Random(4)
    seen = []
    original = Neuron.fire

    def watched(self, wave=0, now=None):
        seen.append(self.potential_at(now) >= self.threshold)
        return original(self, wave, now)

    Neuron.fire = watched
    try:
        for _ in range(3):
            run_epoch(g, verbose=False, rng=stream)
    finally:
        Neuron.fire = original
    inputs = g.input_row()
    assert seen and all(seen) and any(n.spikes for n in inputs)
    assert all(n.forced == bit for n, bit in zip(inputs, g.input_pattern))


# --- refusals (§12.2) -------------------------------------------------------------------------------------------


def test_the_settings_are_refused_where_they_are_not_one_of_the_file_s():
    g = goo()
    with pytest.raises(ValueError, match="exploration"):
        g.set_exploration("both")
    for name, value in (("family", "cubic"), ("scaling", "fan-in"), ("trace", "relayed")):
        with pytest.raises(ValueError, match=name):
            g.set_exploration("synapse", **{name: value})
    for h0 in (-0.01, 1.0, 1.5, math.nan, math.inf, "0.01"):
        with pytest.raises(ValueError, match="§7.6"):
            g.set_exploration("synapse", h0=h0)
    assert g.exploration == "neuron"  # a refused setting leaves the network as it was


def test_a_width_and_exploration_at_the_synapse_together_are_refused():
    """§6.13: the system explores by one thing (§7.1)."""
    g = goo()
    g.set_delta(C.ESCAPE_DELTA)
    with pytest.raises(ValueError, match="§6.13"):
        g.set_exploration("synapse")
    h = goo()
    h.set_exploration("synapse")
    with pytest.raises(ValueError, match="§6.13"):
        h.set_delta(C.ESCAPE_DELTA)
    h.set_delta(0.0)  # a width of 0 is the width §6.13 gives every neuron
    assert h.exploration == "synapse" and h.traced


def test_a_network_carrying_the_other_exploration_s_bookkeeping_is_refused():
    """§12.9 (Byron, September 25, 2026, Q17): nothing maps the neuron rule's per-decision bookkeeping onto the gain or
    back, so a network that ran under one is not switched to the other."""
    g = goo()
    g.set_delta(C.ESCAPE_DELTA)
    teacher = Teacher(g, seed=1, rule="reinforce", eligibility="hazard", target="copy")
    for _ in range(2):
        teacher.epoch(verbose=False)
    g.set_delta(0.0)
    with pytest.raises(ValueError, match="§12.9"):
        g.set_exploration("synapse")
    h = goo()
    h.set_exploration("synapse")
    h.neurons[3].gain = 0.25
    with pytest.raises(ValueError, match="§12.9"):
        h.set_exploration("neuron")


def test_a_network_whose_neurons_have_spiked_is_not_switched():
    """§12.9: the exposure clock a neuron holds (§2.1) runs, after a spike, from the refractory period's end under the neuron
    rule (§6.12) and from the spike itself under exploration at the synapse (§7.8), and nothing maps one onto the other --
    so a network whose neurons have spiked is not switched, though it has no gain, E_j or note open."""
    plain = Tiny(3, [(0, 1, 0.25), (1, 2, 0.5)])
    a = plain.neurons[0]
    plain.propagate(fire=[a], now=T0, until=T0 + 0.5)  # the deterministic neuron rule: A spikes, and nothing is open
    assert (a.fired_at, a.exposed_since, a.expected, a.credit) == (T0, T0 + Neuron.refractory, 0.0, 0.0)
    with pytest.raises(ValueError, match="§12.9"):
        plain.set_exploration("synapse")
    assert plain.exploration == "neuron" and not a.synaptic
    synaptic = chain()
    a = synaptic.neurons[0]
    synaptic.explore_rng = Stream()
    synaptic.propagate(fire=[a], now=T0, until=T0 + 0.5)
    assert (a.fired_at, a.exposed_since) == (T0, T0) and not any(n.gain for n in synaptic.neurons)
    with pytest.raises(ValueError, match="§12.9"):
        synaptic.set_exploration("neuron")
    assert synaptic.exploration == "synapse" and a.synaptic


def test_the_settings_are_refused_again_where_the_network_runs():
    """§12.2: the settings stay writable attributes. h0 and the family are read at every wave, and kappa_i and every
    trace were computed from the scaling and TRACE where set_exploration ran (§7.5, §7.7, §8.17), so a setting assigned
    after it -- one the file does not name, or one it names that the network was not set up under -- is refused where the
    network runs."""
    for name, value, clause in (("synapse_hazard_rest", 1.5, "§7.6"), ("synapse_hazard_rest", -0.5, "§7.6"),
                                ("synapse_hazard_rest", math.nan, "§7.6"), ("synapse_hazard_family", "cubic", "§7.6"),
                                ("synapse_hazard_scaling", "fan-in", "§7.7"), ("synapse_hazard_scaling", "fan-out", "§7.7"),
                                ("trace_mode", "relayed", "§8.17"), ("trace_mode", "ventured", "§8.17")):
        g = goo()
        g.set_exploration("synapse")
        setattr(g, name, value)
        with pytest.raises(ValueError, match=clause):
            run_epoch(g, verbose=False, rng=random.Random(1))
        assert g.epoch == 0 and not g.waves  # refused before anything moved
    g = goo()
    g.set_exploration("synapse")
    g.synapse_hazard_rest, g.synapse_hazard_family = 0.05, "linear"  # the file's own: read at every wave, and taken
    assert run_epoch(g, verbose=False, rng=random.Random(1))


def test_a_width_that_is_not_a_number_is_refused():
    """§6.13: under exploration at the synapse every width is 0 and the neuron fires iff V >= theta; a width that is not a
    number is neither 0 nor positive, would take the hazard's branch and silence the neuron, and is refused -- by
    set_delta under either rule, and where the network runs if a neuron is given one directly."""
    for mode in EXPLORATIONS:
        g = goo()
        g.set_exploration(mode)
        with pytest.raises(ValueError, match="ESCAPE_DELTA"):
            g.set_delta(math.nan)
        assert g.escape_delta == 0.0 and all(n.delta == 0.0 for n in g.all_neurons())
    h = goo()
    h.set_exploration("synapse")
    h.neurons[5].delta = math.nan
    with pytest.raises(ValueError, match="§6.13"):
        run_epoch(h, verbose=False, rng=random.Random(1))


def test_a_threshold_at_or_below_zero_is_refused():
    """§7.5 (Byron, September 25, 2026, Q13): u is undefined there -- at the build, and when homeostasis, un-sticking or
    anything else takes a threshold there."""
    flat = Goo(count=24, across=4, seed=1, weight=None, threshold=0.0, minimum_potential=0.0, scale_with_fan_in=False)
    with pytest.raises(ValueError, match="§7.5"):
        flat.set_exploration("synapse")
    g = goo()
    g.set_exploration("synapse")
    with pytest.raises(ValueError, match="§7.5"):
        homeostasis(g, 10.0, 0.9)
    h = goo()
    h.set_exploration("synapse")
    for neuron in h.all_neurons():
        neuron.rate = 0.0
    with pytest.raises(ValueError, match="§7.5"):
        unstick(h, 10.0, 0.9)
    k = goo()
    k.set_exploration("synapse")
    k.neurons[4].threshold = -0.1
    with pytest.raises(ValueError, match="§7.5"):
        run_epoch(k, verbose=False, rng=random.Random(1))


def test_an_inactive_connection_is_refused():
    """§3.8 (Byron, September 25, 2026, Q12): an inactive synapse keeps its draw, and what it does with it is not
    specified; a network holding one is refused -- when it is set, and when it runs."""
    g = goo()
    g.connections[2].is_active = False
    with pytest.raises(ValueError, match="§3.8"):
        g.set_exploration("synapse")
    h = goo()
    h.set_exploration("synapse")
    h.connections[2].is_active = False
    with pytest.raises(ValueError, match="§3.8"):
        run_epoch(h, verbose=False, rng=random.Random(1))


def test_the_reads_that_are_not_the_count_are_refused():
    """§5.10 (Byron, September 25, 2026, Q7): until a clause says what each reads under exploration at the synapse."""
    for read in ("fired", "again", "window", "rate"):
        g = goo()
        g.set_exploration("synapse")
        g.read = read
        with pytest.raises(ValueError, match="§5.10"):
            g.output_fired()
        with pytest.raises(ValueError, match="§5.10"):
            g.output_levels()
        with pytest.raises(ValueError, match="§5.10"):
            run_epoch(g, verbose=False, rng=random.Random(1))


def test_hebb_is_refused():
    """§8.3: under exploration at the synapse the decisions are the synapses', and hebb has no neuron decision to centre."""
    g = goo()
    g.set_exploration("synapse")
    with pytest.raises(ValueError, match="§8.3"):
        Teacher(g, rule="reinforce", eligibility="hebb")
    with pytest.raises(ValueError, match="§8.3"):
        g.centre(True)
    with pytest.raises(ValueError, match="§8.3"):
        reinforce(g, 1.0, eligibility="hebb")
    h = goo()
    h.centre(True)
    with pytest.raises(ValueError, match="§8.3"):
        h.set_exploration("synapse")


def test_the_drive_is_refused_where_the_file_does_not_name_it():
    """§5.7: a name the code does not know is refused, where it used to become one spike at the epoch's moment; 5.4b: the
    charged drive is refused under the neuron rule, and DRIVE_STEPS is a whole number of at least 1."""
    g = goo()
    g.set_input_bits([True, False], 0.0)
    g.drive = "poisson"
    with pytest.raises(ValueError, match="§5.7"):
        g.input_schedule()
    g.drive = "charged"
    with pytest.raises(ValueError, match="5.4b"):
        g.input_schedule()
    with pytest.raises(ValueError, match="5.4b"):
        run_epoch(g, verbose=False, rng=random.Random(1))
    h = goo()
    h.set_exploration("synapse")
    h.drive = "charged"
    h.set_input_bits([True, False], 0.0)
    for steps in (0, -1, 2.5, 3.0, "3", True, None):
        h.drive_steps = steps
        with pytest.raises(ValueError, match="5.4b"):
            h.input_schedule()
    h.drive_steps = 1
    assert h.input_schedule()
    # §5.7: the code's forced drive is one spike at the epoch's moment, kept for the neuron rule's plumbing tests and
    # refused under exploration at the synapse -- before the epoch moves anything
    k = goo()
    k.set_exploration("synapse")
    k.drive = "forced"
    k.set_input_bits([True, False], 0.0)
    with pytest.raises(ValueError, match="§5.7"):
        k.input_schedule()
    with pytest.raises(ValueError, match="§5.7"):
        run_epoch(k, verbose=False, rng=random.Random(1))
    assert k.epoch == 0 and not k.waves
    plain = goo()
    plain.drive = "forced"
    plain.set_input_bits([True, False], 0.0)
    events = plain.input_schedule()  # the neuron rule's, as it always was
    assert events and all(when == plain.time for _, when in events)


def test_a_count_of_deliveries_may_be_any_whole_number_type():
    """5.4b: DRIVE_STEPS is a whole number of at least 1, and a numpy integer is one -- a sweep may build it that way. It is
    handed on as an int, so the arrivals and the charges are what 3 gives, to the bit."""
    np = pytest.importorskip("numpy")
    nets = []
    for steps in (np.int64(3), 3):
        g = goo()
        g.set_exploration("synapse")
        g.drive, g.drive_steps = "charged", steps
        g.set_input_bits([True, False], 0.0)
        g.explore_rng = random.Random(1)
        g.fire_input()
        nets.append(g)
    wide, plain = nets
    assert wide.input_events == plain.input_events and wide.input_events
    assert all(type(when) is float for _, when in wide.input_events)
    assert [n.potential for n in wide.all_neurons()] == [n.potential for n in plain.all_neurons()]
    assert [n.spikes for n in wide.all_neurons()] == [n.spikes for n in plain.all_neurons()]


def test_a_run_without_a_stream_is_refused():
    """§7.3: a network exploring at the synapse draws whatever its rest hazard, h0 = 0 included, and must be run with a
    stream of its own -- run_epoch no longer falls back to the module's."""
    g = goo()
    g.set_exploration("synapse", h0=0.0)
    with pytest.raises(ValueError, match="§7.3"):
        run_epoch(g, verbose=False)
    g.set_input_bits([True, False])
    with pytest.raises(ValueError, match="§7.3"):
        g.fire_input()


def test_what_has_no_clause_under_the_mechanism_is_refused(monkeypatch):
    """The inputs read as the outputs have no read synapse (§5.1, §7.9 give one to every output); a bored threshold
    moves the theta u is read on (§7.5); the module's propagate has no network to explore at (§7.5, §12.2)."""
    g = goo()
    g.readout = "input"
    with pytest.raises(ValueError, match="§7.9"):
        g.set_exploration("synapse")
    h = goo()
    h.set_exploration("synapse")
    monkeypatch.setattr(Neuron, "bored_after", 10.0)
    with pytest.raises(ValueError, match="§7.5"):
        run_epoch(h, verbose=False, rng=random.Random(1))
    monkeypatch.setattr(Neuron, "bored_after", 0.0)
    with pytest.raises(ValueError, match="§7.5"):
        propagate(fire=[h.neurons[0]])


def test_the_rust_loop_refuses_what_it_does_not_carry_yet():
    """§12.2: fast.build, train and compare refuse exploration at the synapse and the charged drive, each where it runs,
    whether or not the extension is built."""
    g = goo()
    g.set_exploration("synapse")
    for run in (lambda: fast.build(g, explore_rng=random.Random(1)), lambda: fast.train(g, 1, eligibility="hazard", seed=1),
                lambda: fast.compare(g, epochs=1)):
        with pytest.raises(ValueError, match="§7.5"):
            run()
    h = goo()
    h.drive = "charged"
    for run in (lambda: fast.build(h), lambda: fast.train(h, 1, eligibility="hazard", seed=1),
                lambda: fast.compare(h, epochs=1)):
        with pytest.raises(ValueError, match="5.4b"):
            run()


def test_the_array_engine_refuses_what_it_does_not_carry_yet():
    """§12.2: at the wrap, and again where it runs, since its settings stay writable after it."""
    pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork
    g = goo()
    g.set_exploration("synapse")
    with pytest.raises(ValueError, match="§7.5"):
        ArrayNetwork(g)
    h = goo()
    h.drive = "charged"
    with pytest.raises(ValueError, match="5.4b"):
        ArrayNetwork(h)
    net = ArrayNetwork(goo())
    net.set_exploration("synapse")
    net.set_input_bits([True, False])
    with pytest.raises(ValueError, match="§7.5"):
        net.fire_input()
    with pytest.raises(ValueError, match="§7.5"):
        net.propagate(fire=[net.neurons[0]])
    net = ArrayNetwork(goo())
    net.drive = "charged"
    net.set_input_bits([True, False])
    with pytest.raises(ValueError, match="5.4b"):
        net.fire_input()
    # the wrapped mesh set after the wrap: its neurons are the ones the array engine runs, and sync_to_mesh writes back
    mesh = goo()
    net = ArrayNetwork(mesh)
    mesh.set_exploration("synapse")
    net.explore_rng = random.Random(1)
    net.set_input_bits([True, False])
    for run in (net.fire_input, lambda: net.propagate(fire=[net.neurons[0]]), lambda: net._run(10.0)):
        with pytest.raises(ValueError, match="§7.5"):
            run()
    mesh = goo()
    net = ArrayNetwork(mesh)
    mesh.drive = "charged"
    net.set_input_bits([True, False])
    with pytest.raises(ValueError, match="5.4b"):
        net.fire_input()


def test_a_checkpoint_refuses_what_it_does_not_carry_yet(tmp_path):
    """§12.9, §8.14: a checkpoint does not yet carry the gains, the read counts, the ventured marks or the settings of
    §7.1, §7.6, §7.7, §8.17 and 5.4b, and refuses rather than write a file that would resume as another run (§12.2)."""
    from walnutbutter.persistence import checkpoint
    g = goo()
    g.set_exploration("synapse")
    with pytest.raises(ValueError, match="§12.9"):
        checkpoint(g, tmp_path / "synapse.json")
    h = goo()
    h.drive = "charged"
    with pytest.raises(ValueError, match="§12.9"):
        checkpoint(h, tmp_path / "charged.json")
