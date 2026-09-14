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

from .constants import QUASH_K, SYNAPSE_TAU, WEIGHT_RANGE
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


def flatten(grid):
    """The grid as parallel arrays: neuron order and, within each neuron, `outgoing` order.

    The edge order matters and is not arbitrary. Signals due at the same moment are taken
    in the order they were pushed, and the object engine pushes a firing neuron's
    outgoing connections in `neuron.outgoing` order, so an engine that pushes them in a
    different order sums a wave's input in a different order and lands on different bits.
    """
    neurons = list(grid.all_neurons())
    index = {neuron: i for i, neuron in enumerate(neurons)}
    source, target, weight, active = [], [], [], []
    for neuron in neurons:
        for connection in neuron.outgoing:
            source.append(index[connection.source])
            target.append(index[connection.target])
            weight.append(connection.weight)
            active.append(bool(connection.is_active))
    return neurons, index, source, target, weight, active


def build(grid, *, quash_rate=0.0, quash_k=QUASH_K, hebb_rate=0.0, synapse_tau=SYNAPSE_TAU,
          weight_range=WEIGHT_RANGE, earn=False, sigma=0.0):
    """An Engine carrying this grid's topology and state, ready to run epochs."""
    rust = _require()
    neurons, index, source, target, weight, active = flatten(grid)
    engine = rust.Engine(
        len(neurons), source, target, weight, active,
        [n.threshold for n in neurons], [n.minimum_potential for n in neurons],
        Neuron.tau, Neuron.refractory, Neuron.hop(), Neuron.bored_after, Neuron.rate_tau,
    )
    low, high = weight_range
    engine.set_rules(quash_rate, quash_k, hebb_rate, synapse_tau, low, high, earn, sigma)
    return engine, neurons, index


def compare(grid, epochs=20, bits=None):
    """Run `epochs` on the object engine and on the Rust one, and report where they part.

    Returns a list of (epoch, what) for every disagreement, empty when the two agree. The
    grid is left as the object engine ran it; the Rust engine is built from its starting
    state and stepped alongside.
    """
    from .monitor import run_epoch

    engine, neurons, index = build(
        grid,
        quash_rate=grid.quash_rate, quash_k=grid.quash_k,
        hebb_rate=grid.hebb_rate, synapse_tau=grid.synapse_tau,
        weight_range=grid.weight_range, earn=grid.rule in ("teacher", "adaline"),
    )
    edges = [c for neuron in neurons for c in neuron.outgoing]
    parted = []
    for epoch in range(epochs):
        run_epoch(grid, bits, verbose=False)
        engine.reset(False, grid.rule in ("teacher", "adaline"))
        for place, when in (grid.input_events or []):
            engine.stimulus(index[grid.input_row()[place]], when)
        engine.run(grid.horizon)

        mine = [n.spikes for n in neurons]
        theirs = list(engine.spike_counts())
        if mine != theirs:
            parted.append((epoch, f"spikes differ: {sum(abs(a - b) for a, b in zip(mine, theirs))} in total"))
        mine = [c.weight for c in edges]
        theirs = list(engine.weights())
        worst = max((abs(a - b) for a, b in zip(mine, theirs)), default=0.0)
        if worst:
            parted.append((epoch, f"weights differ by up to {worst:g}"))
        if parted:
            break
    return parted


def benchmark(grid, epochs=200, bits=None):
    """Time the same epochs on whichever engines are available. Returns {engine: seconds}."""
    import time
    from .monitor import run_epoch

    timings = {}
    started = time.perf_counter()
    for _ in range(epochs):
        run_epoch(grid, bits, verbose=False)
    timings["objects"] = time.perf_counter() - started

    if available():
        engine, neurons, index = build(
            grid, quash_rate=grid.quash_rate, quash_k=grid.quash_k,
            hebb_rate=grid.hebb_rate, synapse_tau=grid.synapse_tau,
            weight_range=grid.weight_range, earn=grid.rule in ("teacher", "adaline"),
        )
        row = grid.input_row()
        started = time.perf_counter()
        for _ in range(epochs):
            engine.reset(False, grid.rule in ("teacher", "adaline"))
            grid.new_random_input()
            for place, when in grid.input_schedule():
                engine.stimulus(index[row[place]], when)
            engine.run(grid.time + grid.interval)
        timings["rust"] = time.perf_counter() - started
    return timings


def train(grid, epochs, *, lr=0.03, target="copy", baseline_rate=0.05, trace_every=0, patterns=None):
    """Run `epochs` of the §6.7 rule with the hebb eligibility, the whole wave loop in Rust.

    Python keeps what must stay reproducible — the input bits and the Poisson drive come
    from the grid's own seeded stream, in the same order the other engines draw them — and
    Rust does the epoch and the weight update. Returns (mean score, trace), where the trace
    is every `trace_every` epochs' score, or empty when that is 0.

    `patterns` is a run's inputs, drawn up front by `network.input_stream` and attached
    here (§4.5), so two arms that differ in anything see the same epochs in the same
    order. Without it the bits come from the grid's own stream, which a differently built
    network consumes differently.

    The row critic and sigma 0 only: this is the configuration every recent result was
    measured on, and the one the engine supports (see rust/README.md).
    """
    from .learning import TARGETS

    if patterns is not None:
        if len(patterns) < epochs:
            raise ValueError(f"the input stream holds {len(patterns)} patterns for a run of {epochs} epochs")
        grid.use_input_stream(patterns)

    engine, neurons, index = build(
        grid, quash_rate=grid.quash_rate, quash_k=grid.quash_k,
        hebb_rate=grid.hebb_rate, synapse_tau=grid.synapse_tau, weight_range=grid.weight_range,
    )
    row = grid.output_row()
    out = [index[neuron] for neuron in row]
    places = grid.input_row()
    at = [index[neuron] for neuron in places]
    want_of = TARGETS[target]

    baseline = None
    total = 0.0
    trace = []
    for epoch in range(epochs):
        grid.reset()
        grid.new_random_input()
        grid.time = grid.input_time
        grid.epoch += 1
        events = grid.input_schedule()
        grid.horizon = grid.time + grid.interval
        engine.reset(False, False)
        if events:
            engine.stimulate_many([at[place] for place, _ in events], [when for _, when in events])
        engine.run(grid.horizon)

        fired = engine.fired_this_epoch()
        want = want_of(grid.target_pattern)
        reward = sum(1 for i, w in zip(out, want) if fired[i] == w) / len(want)
        if baseline is None:
            baseline = reward
        engine.reinforce_hebb(reward - baseline, lr)
        baseline += baseline_rate * (reward - baseline)
        total += reward
        if trace_every and (epoch + 1) % trace_every == 0:
            trace.append(reward)
    engine.set_weights(engine.weights())  # no-op, but leaves the engine addressable by the caller
    return total / epochs, trace, engine
