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


def build(network, *, quash_rate=0.0, quash_k=QUASH_K, weight_range=WEIGHT_RANGE, explore_rng=None):
    """An Engine carrying this network's topology and state, ready to run epochs.

    `explore_rng` -- a `random.Random` -- is the exploration stream the firing
    decisions draw from (§7.3): its state is handed over here and `sync_explore`
    hands it back, so the other two engines given the same stream take the same
    draws in the same order.
    """
    rust = _require()
    neurons, index, source, target, weight, active = flatten(network)
    engine = rust.Engine(
        len(neurons), source, target, weight, active,
        [n.threshold for n in neurons], [n.minimum_potential for n in neurons],
        Neuron.tau, Neuron.refractory, Neuron.hop, Neuron.bored_after, Neuron.rate_tau,
    )
    deltas = [n.delta for n in neurons]  # escape noise (§5.2): each neuron's decision width, 0 when off
    if any(d > 0.0 for d in deltas):
        if explore_rng is None:
            raise ValueError("escape noise needs a stream: pass explore_rng, the random.Random the other engines use (§7.3)")
        engine.set_explore_state(list(explore_rng.getstate()[1]))
    low, high = weight_range
    engine.set_rules(quash_rate, quash_k, low, high)
    engine.set_deltas(deltas)
    engine.set_escape_scales([n.escape_scale for n in neurons])  # §5.2: the count's scaling of every hazard
    # the single-spike rule (§6.7): the centred rule when the network runs it, and each neuron's expectation and decisions
    # to date, so a resumed run charges from where it was; the traces, notes and expected counts start at zero with
    # the traces, as the engine holds no signal in a potential yet
    engine.set_centred(bool(getattr(network, "centred", False)), DECISION_MEMORY)
    engine.set_centred_state([float("nan") if n.expectation is None else n.expectation for n in neurons],
                             [n.decisions for n in neurons], [0.0] * len(neurons))
    return engine, neurons, index


def sync_explore(engine, explore_rng) -> None:
    """Carry Python's stream on from where the engine left it: the draws it took are now taken in Python too."""
    state = engine.explore_state()
    if state is not None:
        explore_rng.setstate((3, tuple(state), None))


def _outputs_on(engine, network, out) -> list[bool]:
    """The read (§5.10) from the engine's arrays: fired this epoch, or the count against the pickiness (§9.5)."""
    if network.read == "count":
        counts = engine.epoch_spike_counts()
        return [counts[i] >= network.pickiness for i in out]
    if network.read == "fired":
        fired = engine.fired_this_epoch()
        return [fired[i] for i in out]
    raise ValueError(f"the Rust loop reads 'fired' or 'count'; {network.read!r} is read in Python -- use the array engine")


def _reward(engine, network, out, critic, want_of) -> float:
    """The epoch's reward from the engine's arrays: the row critic against the target, or a label critic (§8)."""
    if critic in ("class", "graded", "evidence"):
        from .learning import class_evidence  # the one function every engine reads the zone through (§8)
        counts = engine.epoch_spike_counts()
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
        moved = False
        if self.homeostasis > 0:
            for i in range(len(rates)):
                if not forced[i]:
                    thresholds[i] = thresholds[i] + self.homeostasis * (rates[i] - self.target_rate)  # nothing clips it
            moved = True
        if self.unstick > 0:
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


def compare(network, epochs=20, bits=None, *, teacher=None):
    """Run `epochs` on the object engine and on the Rust one, and report where they part.

    Returns a list of (epoch, what) for every disagreement, empty when the two agree. The
    network is left as the object engine ran it; the Rust engine is built from its starting
    state and stepped alongside.

    With a `teacher` (a Teacher on this network, RULE = reinforce, the row critic, late =
    count, no trace) each epoch is `teacher.epoch()` on the object side and the same
    reward, advantage, §6.7 update, firing-rate memory, homeostasis and un-sticking
    mirrored on the Rust side, so the per-decision entries of §8.4 and the Teacher's
    threshold moves are checked as well as the dynamics. The Rust engine is handed the teacher's own stream at the start and the
    two then draw independently; that they land on the same bits, and on the same stream
    state at the end, is the test.
    """
    from .learning import TARGETS
    from .monitor import run_epoch

    if teacher is not None:
        if teacher.rule != "reinforce" or teacher.critic not in ("row", "class", "graded", "evidence"):
            raise ValueError("compare mirrors the reinforce rule with the row, class, graded or evidence critic only")
        if teacher.network is not network:
            raise ValueError("the teacher must be teaching this network")
    engine, neurons, index = build(
        network,
        quash_rate=network.quash_rate, quash_k=network.quash_k,
        weight_range=network.weight_range,
        explore_rng=teacher.rng if teacher is not None else None,
    )
    edges = [c for neuron in neurons for c in neuron.outgoing]
    out = [index[neuron] for neuron in network.output_row()]
    book = None
    if teacher is not None:
        book = _Thresholds(engine, neurons, out, homeostasis=teacher.homeostasis, target_rate=teacher.target_rate,
                           unstick=teacher.unstick, unstick_target=teacher.unstick_target)
    baseline = None
    parted = []
    for epoch in range(epochs):
        if teacher is None:
            run_epoch(network, bits, verbose=False)
        else:
            reward = teacher.epoch(bits, verbose=False)
        engine.reset(False)
        for place, when in (network.input_events or []):
            engine.stimulus(index[network.input_row()[place]], when)
        engine.run(network.horizon)

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

        mine = [n.spikes for n in neurons]
        theirs = list(engine.spike_counts())
        if mine != theirs:
            parted.append((epoch, f"spikes differ: {sum(abs(a - b) for a, b in zip(mine, theirs))} in total"))
        if teacher is not None and teacher.eligibility in ("hazard", "hebb"):
            mine = [c.score for c in edges]
            theirs = list(engine.scores())
            if mine != theirs:
                parted.append((epoch, f"scores differ on {sum(1 for a, b in zip(mine, theirs) if a != b)} synapses"))
            mine = [c.noted for c in edges]
            theirs = list(engine.notes())
            if mine != theirs:
                parted.append((epoch, f"notes differ on {sum(1 for a, b in zip(mine, theirs) if a != b)} synapses"))
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
        if parted:
            break
    if not parted and teacher is not None and network.hazard:
        if list(engine.explore_state()) != list(teacher.rng.getstate()[1]):
            parted.append((epochs, "the two streams ended in different states: a different number of draws was taken"))
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
          eligibility="hebb", seed=None, explore_state=None, homeostasis=0.0, target_rate=TARGET_RATE,
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
    would use, so a Rust run and a Python run of the same seed take the same draws.

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

    if eligibility not in ("hazard", "hebb"):
        raise ValueError(f"§8.3 keeps two eligibilities, hazard and hebb; got {eligibility!r}")
    if not network.hazard:  # §8.3: no eligibility runs where the threshold decides
        raise ValueError("the reinforce rule refuses to learn where the threshold decides: REINFORCE estimates a gradient "
                         "from the randomness of the decision, and with no width there is no randomness to estimate from. "
                         "network.set_delta(ESCAPE_DELTA > 0) first (§8.3)")
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
        weight_range=network.weight_range, explore_rng=explore_rng,
    )
    row = network.output_row()
    out = [index[neuron] for neuron in row]
    places = network.input_row()
    at = [index[neuron] for neuron in places]
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
            engine.stimulate_many([at[place] for place, _ in events], [when for _, when in events])
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
