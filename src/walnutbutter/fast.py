"""The Rust wave loop, when it is built: a third engine, checked against the other two.

Profiling the array engine found the arithmetic to be 1.3% of the runtime — the sparse
matrix-vector product that performs every signal delivery costs 42 ms out of 3,271 — and
the rest to be Python interpreter overhead. The network is small (60 neurons, 766
connections) and numpy's per-call overhead dwarfs sixty doubles of work, so vectorising
cannot pay at this scale; a compiled loop that owns the state across a run avoids both.

The extension is optional. `available()` says whether it is there, and everything else
raises a clear error when it is not, so the project runs unchanged without it. Build it
with the instructions in rust/README.md.

The contract is the one the object and array engines already keep: the same run must give
the same bits. `compare()` is the harness for that.
"""

from __future__ import annotations

from .constants import (
    DECISION_MEMORY, QUASH_K, RATE_MEMORY, STUCK_ABOVE, STUCK_BELOW, TARGET_RATE,
    UNSTICK_TARGET, WEIGHT_RANGE,
)
from .neuron import Neuron

try:  # pragma: no cover - exercised only where the extension is built
    import walnutbutter_schedule as _rust
except ImportError:  # pragma: no cover
    _rust = None


def available() -> bool:
    """True if the Rust extension is importable."""
    return _rust is not None


def _require():
    if _rust is None:
        raise ImportError(
            "the Rust schedule is not built; see rust/README.md "
            "(sudo apt install rustc cargo; pip install maturin; cd rust && maturin develop --release)"
        )
    return _rust


def flatten(network):
    """The network as parallel arrays: neuron order and, within each neuron, `outgoing` order.

    The edge order matters and is not arbitrary. Signals due at the same moment are taken
    in the order they were pushed, and the object engine pushes a firing neuron's
    outgoing connections in `neuron.outgoing` order, so an engine that pushes them in a
    different order sums a wave's input in a different order and lands on different bits.
    """
    neurons = list(network.all_neurons())
    index = {neuron: i for i, neuron in enumerate(neurons)}
    source, target, weight, active = [], [], [], []
    for neuron in neurons:
        for connection in neuron.outgoing:
            source.append(index[connection.source])
            target.append(index[connection.target])
            weight.append(connection.weight)
            active.append(bool(connection.is_active))
    return neurons, index, source, target, weight, active


def _synaptic(network) -> bool:
    """Whether the network explores at the synapse (AUTHORITY.md §7.1)."""
    return getattr(network, "exploration", "neuron") == "synapse"


def _refuse(network) -> None:
    """What the Rust loop refuses, by build, train and compare alike (AUTHORITY.md §12.2) -- before the extension is
    looked for, so the refusal is the same whether or not it is built. It carries exploration at the synapse and the
    charged drive (§7.5-§7.9, §8.16-§8.17, 5.4b), and refuses there what the object engine refuses where a network runs:
    the settings the file does not name or the network was not set up under, what the mechanism has no rule for, a read
    other than the count, any width; and a drive the file does not name, or the charged drive under the neuron rule."""
    if _synaptic(network):
        network._refuse_synapse_run(list(network.all_neurons()))
    if hasattr(network, "_drive_steps"):
        network._drive_steps()  # the drive, checked as fire_input checks it


def _drive(network) -> str:
    return getattr(network, "drive", "rate")


def build(network, *, quash_rate=0.0, quash_k=QUASH_K, weight_range=WEIGHT_RANGE, explore_rng=None, pending_events=None):
    """An Engine carrying this network's topology and state, ready to run epochs.

    `explore_rng` -- a `random.Random` -- is the exploration stream the firing
    decisions draw from (§7.3): its state is handed over here and `sync_explore`
    hands it back, so the other two engines given the same stream take the same
    draws in the same order. A network with a width needs one, and so does one
    exploring at the synapse, whose synapses draw at every wave (§7.3).

    Under exploration at the synapse the engine takes the network's settings --
    h0, the family and the trace -- with kappa_i as set_exploration computed it
    (§7.5, §7.7) and the output zone by place, whose distinct neurons each have a
    read synapse (§7.9); each neuron's gain and read count come with it, and
    under the charged drive DRIVE_STEPS (5.4b). `pending_events` are (time,
    kind, payload) triples or (time, kind, payload, ventured) quadruples, as
    `pending_events()` gives them without and with marks; a triple is relayed.
    """
    _refuse(network)
    synaptic = _synaptic(network)
    deltas = [n.delta for n in network.all_neurons()]  # escape noise (§5.2): each neuron's decision width, 0 when off
    drawn = any(d > 0.0 for d in deltas) or synaptic
    if drawn and explore_rng is None:
        what = "exploration at the synapse draws at every wave" if synaptic else "escape noise needs a stream"
        raise ValueError(f"{what}: pass explore_rng, the random.Random the other engines use (§7.3)")
    rust = _require()
    neurons, index, source, target, weight, active = flatten(network)
    engine = rust.Engine(
        len(neurons), source, target, weight, active,
        [n.threshold for n in neurons], [n.minimum_potential for n in neurons],
        Neuron.tau, Neuron.refractory, Neuron.hop, Neuron.bored_after, Neuron.rate_tau,
    )
    if drawn:
        engine.set_explore_state(list(explore_rng.getstate()[1]))
    low, high = weight_range
    engine.set_rules(quash_rate, quash_k, low, high)
    engine.set_deltas(deltas)
    engine.set_escape_scales([n.escape_scale for n in neurons])  # §5.2: the count's scaling of every hazard
    # the single-spike rule (§6.7): the centred rule when the network runs it, and each neuron's expectation and decisions
    # to date, so a resumed run charges from where it was, and its expected counts with them (§12.11); the notes are
    # rebuilt from the traces and the expected counts at the first reset
    engine.set_centred(bool(getattr(network, "centred", False)), DECISION_MEMORY)
    if synaptic:  # §7.5-§7.9: the synapses decide, and every width is 0 (§6.13)
        engine.set_exploration("synapse", network.synapse_hazard_rest, network.synapse_hazard_family, network.trace_mode,
                               [n.synapse_scale for n in neurons], [index[n] for n in network.output_row()])
        engine.set_gains([n.gain for n in neurons])  # §8.16: G_j, open across the reads as the arrivals it settles are
        engine.set_read_counts([n.read_count for n in neurons])  # §7.9: this epoch's read-synapse escapes
        if _drive(network) == "charged":
            engine.set_charged_drive(network._drive_steps())  # 5.4b: theta / DRIVE_STEPS a delivery
    engine.set_centred_state([float("nan") if n.expectation is None else n.expectation for n in neurons],
                             [n.decisions for n in neurons], [n.expected for n in neurons])
    # §12.11: what reset(false) carries across epochs comes with a restored network -- the potentials, the per-synapse
    # traces and the expected counts above; all zero on a fresh build, so a fresh run is unchanged
    engine.set_potentials([n.potential for n in neurons])
    engine.set_traces([c.trace for n in neurons for c in n.outgoing])
    engine.set_last_updates([n.last_update for n in neurons])  # and the times the lazy decays run from (§2.3, the leak)
    engine.set_trace_ats([c.trace_at for n in neurons for c in n.outgoing])
    engine.set_spike_counts([n.spikes for n in neurons])  # cumulative, so the run's counts carry on
    never = float("-inf")  # the engine's "never fired"; the objects say None
    engine.set_fired_at([never if n.fired_at is None else n.fired_at for n in neurons])
    engine.set_previous_fired_at([never if n.previous_fired_at is None else n.previous_fired_at for n in neurons])
    engine.set_exposed_since([n.exposed_since for n in neurons])  # since when each hazard has run (§6.5)
    engine.set_last_signals([never if c.last_signal is None else c.last_signal for n in neurons for c in n.outgoing])
    if pending_events:  # §12.11: the signals a checkpoint found in flight, back in the queue in delivery order
        engine.push_events([e[0] for e in pending_events], [e[1] for e in pending_events], [e[2] for e in pending_events],
                           [bool(e[3]) if len(e) > 3 else False for e in pending_events])  # a triple is relayed (§7.9)
    return engine, neurons, index


def sync_explore(engine, explore_rng) -> None:
    """Carry Python's stream on from where the engine left it: the draws it took are now taken in Python too."""
    state = engine.explore_state()
    if state is not None:
        explore_rng.setstate((3, tuple(state), None))


def _counts(engine) -> list[int]:
    """Each neuron's count this epoch as the count read takes it (§5.10): its spikes, plus under exploration at the synapse
    its read synapse's escapes (§7.9), which are 0 for every neuron under the neuron rule."""
    return [spikes + reads for spikes, reads in zip(engine.epoch_spike_counts(), engine.read_counts())]


def _outputs_on(engine, network, out) -> list[bool]:
    """The read (§5.10) from the engine's arrays: fired this epoch, or the count against the pickiness (§9.5)."""
    if network.read == "count":
        counts = _counts(engine)
        return [counts[i] >= network.pickiness for i in out]
    if network.read == "fired":
        fired = engine.fired_this_epoch()
        return [fired[i] for i in out]
    raise ValueError(f"the Rust loop reads 'fired' or 'count'; {network.read!r} is read in Python -- use the array engine")


def _reward(engine, network, out, critic, want_of) -> float:
    """The epoch's reward from the engine's arrays: the row critic against the target, or a label critic (§8)."""
    if critic in ("class", "graded", "evidence"):
        from .learning import class_evidence  # the one function every engine reads the zone through (§8)
        counts = _counts(engine)
        groups = class_evidence([int(counts[i]) for i in out], network.population)
        label = network.input_label
        if label is None:
            raise ValueError(f"the {critic} critic needs a stream that carries labels (a dataset, §8)")
        if critic == "evidence":
            from .learning import evidence_reward  # the one function every engine pays it through
            return evidence_reward(groups, label, network.temperature)
        others = [g for c, g in enumerate(groups) if c != label]
        if critic == "class":
            return 1.0 if all(groups[label] > g for g in others) else 0.0
        return sum(1 for g in others if groups[label] > g) / len(others)
    on = _outputs_on(engine, network, out)
    want = want_of(network.target_pattern)
    return sum(1 for o, w in zip(on, want) if o == w) / len(want)


class _Thresholds:
    """Teacher.step's bookkeeping after the update, mirrored: learning.update_rates, homeostasis and unstick.

    The Rust engine owns the potentials; the Teacher owns each neuron's running
    firing rate and moves its threshold from that once an epoch, in Python, which
    is cheap and keeps the arithmetic in the same order as the object engine's.
    """

    def __init__(self, engine, neurons, out, *, homeostasis, target_rate, unstick, unstick_target, rate_memory=RATE_MEMORY):
        self.engine, self.out = engine, out
        self.rates = [n.rate for n in neurons]
        self.thresholds = [n.threshold for n in neurons]
        self.homeostasis, self.target_rate = homeostasis, target_rate
        self.unstick, self.unstick_target = unstick, unstick_target
        self.rate_memory = rate_memory
        self.unstuck = 0

    def step(self) -> None:
        fired, forced = self.engine.fired_this_epoch(), self.engine.forced_flags()
        rates, thresholds = self.rates, self.thresholds
        for i in range(len(rates)):
            if not forced[i]:
                rates[i] += self.rate_memory * ((1.0 if fired[i] else 0.0) - rates[i])
        # Each move is handed to the engine where it is made: under exploration at the synapse set_thresholds refuses a
        # threshold at or below zero, and §7.5 refuses it at the moment homeostasis or un-sticking takes it there -- so a
        # threshold homeostasis takes to zero is refused before un-sticking could lift it back, as learning.py refuses it
        if self.homeostasis > 0:
            for i in range(len(rates)):
                if not forced[i]:
                    thresholds[i] = thresholds[i] + self.homeostasis * (rates[i] - self.target_rate)  # nothing clips it
            self.engine.set_thresholds(thresholds)
        if self.unstick > 0:
            moved = False
            for i in range(len(rates)):  # every neuron, not the outputs only (§6.7, September 14, 2026)
                if not forced[i] and (rates[i] > STUCK_ABOVE or rates[i] < STUCK_BELOW):
                    thresholds[i] = thresholds[i] + self.unstick * (rates[i] - self.unstick_target)
                    self.unstuck += 1
                    moved = True
            if moved:
                self.engine.set_thresholds(thresholds)

    def stuck(self) -> tuple[int, int]:
        """learning.stuck_neurons: how many are (almost) always on, and always off."""
        return sum(1 for r in self.rates if r > STUCK_ABOVE), sum(1 for r in self.rates if r < STUCK_BELOW)


def _queue(network, index, edge_index) -> list[tuple]:
    """The object engine's schedule as the Rust engine's `pending_events(marks=True)` gives its own: every event still
    in the queue, in delivery order, as (time, kind, payload, ventured) -- a signal's payload its edge in engine order
    and its mark (§7.9), a stimulus's or a charge's its neuron."""
    from .propagation import SIGNAL, STIMULUS
    out = []
    for time, kind, _, payload in sorted(network.schedule._heap, key=lambda e: e[:3]):
        if kind == SIGNAL:
            connection, ventured = payload
            out.append((time, kind, edge_index[id(connection)], ventured))
        elif kind == STIMULUS:
            out.append((time, kind, index[payload], False))
        elif payload[1] is None:  # a charge of theta / steps (5.4b), the engine's EXTERNAL
            out.append((time, kind, index[payload[0]], False))
        else:  # the engine's EXTERNAL is a charge, and it has no event for a given amount: refused, not relabelled
            raise ValueError("an external input of a given amount is in flight, and the Rust loop carries none: it would "
                             "be handed over as a charge of theta / DRIVE_STEPS (5.4b, §12.2)")
    return out


def compare(network, epochs=20, bits=None, *, teacher=None, rng=None):
    """Run `epochs` on the object engine and on the Rust one, and report where they part.

    Returns a list of (epoch, what) for every disagreement, empty when the two agree. The
    network is left as the object engine ran it; the Rust engine is built from its starting
    state and stepped alongside -- the signals it has in flight, with their ventured marks,
    among it, so a network that has already run on the objects is continued on both (§12.11).

    With a `teacher` (a Teacher on this network, RULE = reinforce, the row critic, late =
    count, no trace) each epoch is `teacher.epoch()` on the object side and the same
    reward, advantage, §6.7 update, firing-rate memory, homeostasis and un-sticking
    mirrored on the Rust side, so the per-decision entries of §8.4 and the Teacher's
    threshold moves are checked as well as the dynamics. The Rust engine is handed the teacher's own stream at the start and the
    two then draw independently; that they land on the same bits, and on the same stream
    state after every epoch, is the test. Without a teacher, `rng` is the stream the epochs
    draw from (run_epoch's), handed over the same way: a network exploring at the synapse
    needs one, since its synapses draw at every wave (§7.3).

    Checked after every epoch, each with ==: every wave's time and the neurons it fired, in
    the order they fired (§3.7: that order is the order their signals sum in); the spikes;
    the read counts; the potentials, the times they were last brought up to date and the
    exposure clocks; the spike times the refractory test and the quash read, and the
    driven marks (5.8); every synapse's trace and the moment it was brought up to, and its
    stamp of the last signal its target took in (§1.7), which the quash reads (§10.2); the
    weights; the queue left in flight, each signal with its ventured mark (§7.9); under the
    rule, the scores and notes and E_j, or the gain G_j under exploration at the synapse
    (§8.16), which posts whether or not a teacher pays it; under exploration at the synapse
    the settings the engine holds against the network's as they stand (§12.9): h0, the
    family, the trace, kappa_i, the read synapses' draws as the objects laid them out, and
    DRIVE_STEPS; and wherever something draws, the stream's state. A teacher's baseline is
    taken as it stands, so a teacher that has already paid is mirrored from where it is.
    """
    from .learning import TARGETS
    from .monitor import run_epoch

    _refuse(network)
    if teacher is not None:
        if teacher.rule != "reinforce" or teacher.critic not in ("row", "class", "graded", "evidence"):
            raise ValueError("compare mirrors the reinforce rule with the row, class, graded or evidence critic only")
        if teacher.network is not network:
            raise ValueError("the teacher must be teaching this network")
        if teacher.discharge:
            raise ValueError("compare mirrors an epoch that keeps its potentials (reset(False)); a discharging teacher is "
                             "not mirrored, and is refused rather than compared as something else (§12.2)")
        if rng is not None:
            raise ValueError("the teacher's stream is the run's; pass rng only without a teacher")
    stream = teacher.rng if teacher is not None else rng
    synaptic = _synaptic(network)
    if synaptic and stream is None:
        raise ValueError("exploration at the synapse draws for every synapse at every wave and needs a stream of its own "
                         "(§7.3): pass rng, or a teacher")
    neurons, index = flatten(network)[:2]
    edges = [c for neuron in neurons for c in neuron.outgoing]
    edge_index = {id(c): k for k, c in enumerate(edges)}
    engine, neurons, index = build(
        network,
        quash_rate=network.quash_rate, quash_k=network.quash_k,
        weight_range=network.weight_range,
        explore_rng=stream,
        pending_events=_queue(network, index, edge_index),  # what the objects have in flight, marks and all
    )
    out = [index[neuron] for neuron in network.output_row()]
    charged = _drive(network) == "charged"
    book = None
    if teacher is not None:
        book = _Thresholds(engine, neurons, out, homeostasis=teacher.homeostasis, target_rate=teacher.target_rate,
                           unstick=teacher.unstick, unstick_target=teacher.unstick_target)
    scored = synaptic or (teacher is not None and teacher.eligibility in ("hazard", "hebb"))
    baseline = None if teacher is None else teacher.baseline  # None until the teacher's first read (§9.3)
    never = float("-inf")  # the engine's "never"; the objects say None
    parted = []

    def differ(what, mine, theirs):
        """Record where two per-neuron or per-synapse lists part, with ==."""
        if mine != theirs:
            parted.append((epoch, f"{what} differ on {sum(1 for a, b in zip(mine, theirs) if a != b)} of {len(mine)}"))

    for epoch in range(epochs):
        if teacher is None:
            run_epoch(network, bits, verbose=False, rng=rng)
        else:
            reward = teacher.epoch(bits, verbose=False)
        engine.reset(False)
        row = network.input_row()
        events = network.input_events or []
        if charged:  # 5.4b: deliveries of theta / DRIVE_STEPS, not forced spikes
            engine.charge_many([index[row[place]] for place, _ in events], [when for _, when in events])
        else:
            for place, when in events:
                engine.stimulus(index[row[place]], when)
        waves = engine.run(network.horizon)
        mine = [(wave.time, [index[n] for n in wave.fired]) for wave in network.waves]
        theirs = [(time, list(fired)) for time, fired in waves]
        if mine != theirs:
            where = next((k for k, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
            parted.append((epoch, f"the waves differ, their times or what they fired in what order: objects {len(mine)} "
                                  f"waves, rust {len(theirs)}, first parting at wave {where}"))

        if teacher is not None:
            mine = _reward(engine, network, out, teacher.critic, TARGETS[teacher.target])
            if mine != reward:
                parted.append((epoch, f"rewards differ: objects {reward:g}, rust {mine:g}"))
            if baseline is None:
                baseline = reward
            engine.reinforce_scores(reward - baseline, teacher.lr)  # §8.4, under either eligibility
            book.step()
            baseline += teacher.baseline_rate * (reward - baseline)
            if [n.rate for n in neurons] != book.rates:
                parted.append((epoch, "firing-rate memories differ"))
            if [n.threshold for n in neurons] != book.thresholds:
                parted.append((epoch, "thresholds differ after homeostasis / un-sticking"))
        differ("the thresholds the engine reads", [n.threshold for n in neurons], list(engine.thresholds()))

        mine = [n.spikes for n in neurons]
        theirs = list(engine.spike_counts())
        if mine != theirs:
            parted.append((epoch, f"spikes differ: {sum(abs(a - b) for a, b in zip(mine, theirs))} in total"))
        differ("read counts", [n.read_count for n in neurons], list(engine.read_counts()))
        differ("potentials", [n.potential for n in neurons], list(engine.potentials()))
        differ("the potentials' last updates", [n.last_update for n in neurons], list(engine.last_updates()))
        differ("exposure clocks", [n.exposed_since for n in neurons], list(engine.exposed_since()))
        differ("spike times", [never if n.fired_at is None else n.fired_at for n in neurons], list(engine.fired_times()))
        differ("previous spike times", [never if n.previous_fired_at is None else n.previous_fired_at for n in neurons],
               list(engine.previous_fired_times()))
        differ("driven marks", [bool(n.forced) for n in neurons], list(engine.forced_flags()))
        differ("traces", [c.trace for c in edges], list(engine.traces()))
        differ("the traces' moments", [c.trace_at for c in edges], list(engine.trace_ats()))
        differ("last signals", [never if c.last_signal is None else c.last_signal for c in edges],
               list(engine.last_signals()))
        if scored:
            mine = [c.score for c in edges]
            theirs = list(engine.scores())
            if mine != theirs:
                parted.append((epoch, f"scores differ on {sum(1 for a, b in zip(mine, theirs) if a != b)} synapses"))
            mine = [c.noted for c in edges]
            theirs = list(engine.notes())
            if mine != theirs:
                parted.append((epoch, f"notes differ on {sum(1 for a, b in zip(mine, theirs) if a != b)} synapses"))
            if synaptic:
                differ("gains", [n.gain for n in neurons], list(engine.gains()))
            else:
                mine = [n.expected for n in neurons]
                theirs = list(engine.expected_since_spike())
                if mine != theirs:
                    parted.append((epoch, f"expected counts since the last spike differ on {sum(1 for a, b in zip(mine, theirs) if a != b)} neurons"))
        if teacher is not None and teacher.eligibility == "hebb":
            mine = [n.expectation for n in neurons]
            theirs = [None if e != e else e for e in engine.expectations()]
            if mine != theirs:
                parted.append((epoch, f"expectations differ on {sum(1 for a, b in zip(mine, theirs) if a != b)} neurons"))
            if [n.decisions for n in neurons] != list(engine.decision_counts()):
                parted.append((epoch, "decision counts differ"))
        mine = [c.weight for c in edges]
        theirs = list(engine.weights())
        worst = max((abs(a - b) for a, b in zip(mine, theirs)), default=0.0)
        if worst:
            parted.append((epoch, f"weights differ by up to {worst:g}"))
        mine, theirs = _queue(network, index, edge_index), list(engine.pending_events(True))
        if mine != theirs:
            ventured = (sum(1 for e in mine if e[3]), sum(1 for e in theirs if e[3]))
            parted.append((epoch, f"the queues in flight differ: objects {len(mine)} events ({ventured[0]} ventured), "
                                  f"rust {len(theirs)} ({ventured[1]} ventured)"))
        if synaptic:  # the settings as the network holds them now, against the engine's (§12.9)
            steps = network._drive_steps() if charged else 0
            settings = ("synapse", network.synapse_hazard_rest, network.synapse_hazard_family, network.trace_mode,
                        [n.synapse_scale for n in neurons], [slot for _, slot in network._layout], steps)
            held = engine.exploration_settings()
            if settings != (held[0], held[1], held[2], held[3], list(held[4]), list(held[5]), held[6]):
                names = ("exploration", "h0", "family", "trace", "kappa", "read slots", "DRIVE_STEPS")
                parted.append((epoch, "the settings differ: " + ", ".join(
                    name for name, a, b in zip(names, settings, held) if a != (list(b) if isinstance(b, (list, tuple)) else b))))
        if stream is not None and network.explores:
            if list(engine.explore_state()) != list(stream.getstate()[1]):
                parted.append((epoch, "the two streams are in different states: a different number of draws was taken"))
        if parted:
            break
    return parted


def benchmark(network, epochs=200, bits=None):
    """Time the same epochs on whichever engines are available. Returns {engine: seconds}."""
    import time
    from .monitor import run_epoch

    timings = {}
    started = time.perf_counter()
    for _ in range(epochs):
        run_epoch(network, bits, verbose=False)
    timings["objects"] = time.perf_counter() - started

    if available():
        engine, neurons, index = build(
            network, quash_rate=network.quash_rate, quash_k=network.quash_k,
            weight_range=network.weight_range,
        )
        row = network.input_row()
        started = time.perf_counter()
        for _ in range(epochs):
            engine.reset(False)
            network.new_random_input()
            for place, when in network.input_schedule():
                engine.stimulus(index[row[place]], when)
            engine.run(network.time + network.interval)
        timings["rust"] = time.perf_counter() - started
    return timings


def train(network, epochs, *, lr=0.03, target="copy", baseline_rate=0.05, trace_every=0, patterns=None,
          eligibility=None, seed=None, explore_state=None, pending_events=None, homeostasis=0.0, target_rate=TARGET_RATE,
          unstick=0.0, unstick_target=UNSTICK_TARGET, critic="row", labels=None, probe=None, probe_every=0,
          direction=None, reference_weights=None, epoch_offset=0, baseline=None):
    """Run `epochs` of the §8.4 rule, the whole wave loop in Rust.

    Python keeps what must stay reproducible — the input bits and the Poisson drive come
    from the network's own seeded stream, in the same order the other engines draw them — and
    Rust does the epoch and the weight update. Returns (mean score, trace), where the trace
    is every `trace_every` epochs' score, or empty when that is 0.

    `patterns` is a run's inputs, drawn up front by `network.input_stream` and attached
    here (§4.5), so two arms that differ in anything see the same epochs in the same
    order. Without it the bits come from the network's own stream, which a differently built
    network consumes differently.

    `eligibility` is one of the two §8.3 keeps: hebb, which charges every decision against
    the neuron's own expectation (§8.6), or hazard, which takes the escape decision's own
    score (§8.7). Either way the network needs a positive ESCAPE_DELTA
    (`network.set_delta`), since §8.3 refuses the rule where the threshold decides, and the
    decisions draw from `random.Random(seed)` -- the stream a Teacher with that seed
    would use, so a Rust run and a Python run of the same seed take the same draws. A
    network exploring at the synapse (`network.set_exploration("synapse")`) needs no
    width: its synapses' decisions are the draws, the hazard's row is posted for them
    (§8.16), and hebb, having no neuron decision to centre, is refused there (§8.3).
    Named as None, the eligibility is hebb under the neuron rule, as it always was, and
    hazard under exploration at the synapse, the one §8.3 keeps there. Under the charged
    drive (5.4b) the epoch's arrivals are delivered as charges.

    `critic` is row (the fraction of outputs matching the target), class (§8: the
    label's group of outputs out-spikes every other group, or nothing), graded (the
    fraction of the other groups it out-spikes) or evidence (the group sums as
    log-odds at `network.temperature`, paid the softmax cross-entropy); the label critics
    need `labels` beside `patterns`, one per pattern, as `mnist.stream` gives them.
    Under the evidence critic the report also carries the class critic's fraction
    over the last tenth, `accuracy_last_tenth`, so a fraction right stays readable.

    `probe(epoch, engine, network, out, book)` is called after every `probe_every`-th
    epoch's update, for a diagnostic to read the engine's counts and the rate
    memories as the run goes; it must not change anything.

    `direction(edges)`, given the engine's edges in order, returns a per-edge
    supervised direction `d` and a mask of the edges it is defined on (§8: for
    mnist, P(pixel on | the output's class) - P(pixel on) on the input-to-output
    synapses). With it the report carries `estimator`: at every `trace_every`
    epochs, over the masked edges, the Pearson correlation and the sign agreement
    of the weight change since the start with `d`, and the correlation of the
    change since the previous trace point -- the estimator integrated, which is
    what the weights are, under any eligibility. A run resumed from a saved
    network (docs/rust-sweep.py --resume-from) passes `reference_weights`, the
    weights of its first start, so the change is still measured from there, and
    `epoch_offset`, the epochs already run, so the trace's epochs count on.

    `homeostasis`, `unstick` and their targets are the Teacher's threshold moves
    (§1.3), mirrored here at the constants the command line runs them at, so a
    Rust run is the run `walnutbutter --seeds` would do; 0 switches either off,
    which is what every rust-sweep before September 14, 2026 ran with.

    Returns (mean score, trace, engine, report): the trace is every
    `trace_every` epochs' score, or empty when that is 0, and the report holds
    the mean over the last tenth of the run, the final firing-rate memories and
    thresholds, and how many neurons ended stuck on and off.
    """
    import random

    from .learning import TARGETS

    _refuse(network)
    synaptic = _synaptic(network)
    if eligibility is None:
        eligibility = "hazard" if synaptic else "hebb"
    if eligibility not in ("hazard", "hebb"):
        raise ValueError(f"§8.3 keeps two eligibilities, hazard and hebb; got {eligibility!r}")
    if eligibility == "hebb" and synaptic:
        raise ValueError("hebb is refused under exploration at the synapse: the decisions are the synapses', and it has no "
                         "neuron decision to centre (§8.3)")
    if not network.explores:  # §8.3: no eligibility runs where the threshold decides
        raise ValueError("the reinforce rule refuses to learn where the threshold decides: REINFORCE estimates a gradient "
                         "from the randomness of the decision, and with no width there is no randomness to estimate from. "
                         "network.set_delta(ESCAPE_DELTA > 0), or network.set_exploration(\"synapse\"), first (§8.3)")
    explore_rng = random.Random(seed)  # §7.3: the exploration stream the firing decisions draw from
    if explore_state:  # §12.11: a resume continues that stream where the checkpoint left it, rather than seeding afresh
        explore_rng.setstate((3, tuple(int(x) for x in explore_state), None))

    if critic not in ("row", "class", "graded", "evidence"):
        raise ValueError(f"the Rust loop is paid by the row, class, graded or evidence critic, got {critic!r}")
    if patterns is not None:
        network.use_input_stream(patterns, labels)  # a run longer than the stream goes round again (§4.5)
    network.centre(eligibility == "hebb")  # §8.6: under hebb every neuron charges its decisions against its own expectation

    engine, neurons, index = build(
        network, quash_rate=network.quash_rate, quash_k=network.quash_k,
        weight_range=network.weight_range, explore_rng=explore_rng, pending_events=pending_events,
    )
    row = network.output_row()
    out = [index[neuron] for neuron in row]
    places = network.input_row()
    at = [index[neuron] for neuron in places]
    deliver = engine.charge_many if _drive(network) == "charged" else engine.stimulate_many  # 5.4b, or forced spikes
    want_of = TARGETS[target]
    book = _Thresholds(engine, neurons, out, homeostasis=homeostasis, target_rate=target_rate, unstick=unstick,
                       unstick_target=unstick_target)

    # §9.3: a resumed run is the same run continued, so it is paid against the baseline the run had reached
    total = tail = hits = 0.0
    tenth = max(1, epochs // 10)
    # The fraction right over each trace interval, beside the score (§9.11). One epoch's
    # class reward is 0 or 1, so a traced point is the mean over the interval and not a
    # single epoch's: a curve rather than a coin sequence.
    right, interval_hits = [], 0.0
    trace = []
    estimator = None
    if direction is not None:
        import numpy as np
        edges = [c for neuron in neurons for c in neuron.outgoing]  # the engine's order (see compare)
        d, mask = direction(edges)
        d, mask = np.asarray(d, dtype=float), np.asarray(mask, dtype=bool)
        if len(d) != len(edges) or len(mask) != len(edges):
            raise ValueError(f"the direction covers {len(d)} edges, the engine has {len(edges)}")
        d_masked = d[mask]
        w_prev = np.array(engine.weights())
        w0 = w_prev if reference_weights is None else np.asarray(reference_weights, dtype=float)
        if len(w0) != len(edges):
            raise ValueError(f"the reference weights cover {len(w0)} edges, the engine has {len(edges)}")
        estimator = []

        def _corr(x):
            if x.std() == 0.0 or d_masked.std() == 0.0:
                return None
            return float(np.corrcoef(x, d_masked)[0, 1])
    for epoch in range(epochs):
        network.reset()
        network.new_random_input()
        network.time = network.input_time
        network.epoch += 1
        events = network.input_schedule()
        network.horizon = network.time + network.interval
        engine.reset(False)
        if events:
            deliver([at[place] for place, _ in events], [when for _, when in events])
        engine.run(network.horizon)

        reward = _reward(engine, network, out, critic, want_of)
        if baseline is None:
            baseline = reward
        engine.reinforce_scores(reward - baseline, lr)  # §8.4, under either eligibility
        book.step()
        baseline += baseline_rate * (reward - baseline)
        total += reward
        won = _reward(engine, network, out, "class", want_of) if (
            critic == "evidence" and (trace_every or epoch >= epochs - tenth)) else None
        if epoch >= epochs - tenth:
            tail += reward
            if critic == "evidence":  # the fraction right beside the log score: did the label's class win outright?
                hits += won
        if won is not None:
            interval_hits += won
        if trace_every and (epoch + 1) % trace_every == 0:
            trace.append(reward)
            right.append(interval_hits / trace_every if critic == "evidence" else None)
            interval_hits = 0.0
            if estimator is not None:
                w = np.array(engine.weights())
                cum, window = (w - w0)[mask], (w - w_prev)[mask]
                estimator.append({"epoch": epoch_offset + epoch + 1, "corr_cum": _corr(cum), "corr_window": _corr(window),
                                  "sign_cum": float((np.sign(cum) == np.sign(d_masked)).mean())})
                w_prev = w
        if probe is not None and probe_every and (epoch + 1) % probe_every == 0:
            probe(epoch + 1, engine, network, out, book)
    on, off = book.stuck()
    report = {"last_tenth": tail / tenth, "rates": book.rates, "thresholds": book.thresholds,
              # the single-spike rule's expectations and decisions to date (§6.7), so a continuation charges from where it was
              "expectations": [None if e != e else e for e in engine.expectations()], "decisions": list(engine.decision_counts()),
              "stuck_on": on, "stuck_off": off, "unstuck": book.unstuck,
              "accuracy_last_tenth": hits / tenth if critic == "evidence" else None,
              "right": right,  # the fraction right at every trace_every, beside `trace`'s score
              "baseline": baseline,  # §9.3: what b had reached, so a resume is paid against it and not a fresh one
              "estimator": estimator}  # the estimator's correlation over time (§8), when a direction was given
    return total / epochs, trace, engine, report
