"""The array engine: the same network as vectors and a sparse matrix.

`ArrayNetwork` wraps a mesh built the usual way (a `Goo`) and runs it with
numpy and scipy instead of neuron objects. Neurons become vectors of length N (potential, threshold,
noise, firing rate, spike times, the wave each fired in this epoch);
connections become vectors of length E in connection-id order (source,
target, weight, active, the stamp of the last signal integrated) and a
sparse N x N matrix for delivery. A wave is one sparse matrix-vector
product: the neurons whose signals arrive now, as a 0/1 vector, times the
weights, gives every neuron the sum of its incoming signals; those not
refractory take it, the floor is applied, and whoever has reached threshold
fires, its signals scheduled one hop later.

The schedule (AUTHORITY.md §4) is a heap of distinct times, each with the
0/1 vector of sources whose signals arrive then and, when an input lands
then, the vector of neurons forced. The object engine (`propagation.py`)
and this one are meant to be the same network: same topology, same ids,
same rules, interchangeable checkpoints, and `tests/test_arrays.py` runs
them side by side. They differ only in the order floating-point additions
happen, so on the rare wave where a potential sits within rounding of a
threshold the two can decide differently and their runs diverge from
there, like two seeds. Exponentials are taken with `math.exp` on both
sides so that the two engines agree to the last bit.

The mesh stays attached as `mesh`: `sync_to_mesh()` copies the arrays back
into its neuron and connection objects, which is how checkpoints are
written. Nothing prints per
neuron in this engine.

Under exploration at the synapse (AUTHORITY.md §7.5-§7.9, §8.16-§8.17) a
wave takes E + O uniforms after the floor, from numpy's MT19937 loaded with
the exploration stream's state and handed back where the objects leave it
(`SynapseDraws`); after the fire phase every source that did not spike
decides on its synapses at once, one m and one P per source, and the
escapes are scheduled in edge order (§3.6) on a per-edge row of their own
beside the per-source rows, delivered with their wave (`_decide_synapses`). u, h, m, P and the credit
are the engine notes' forms (§7.5, §8.16) in numpy's arithmetic, whose pow,
exp and expm1 part from libm's in the last bit on about one argument in
twenty -- the other source of §12.4's tolerance. The gain is a vector
restarted on neuron masks, and each output's read synapse counts into a
vector the count readers add. The charged drive's deliveries (5.4b) are kept
per time as the stimuli are, and added one at a time before the wave's
matrix sum.
"""

from __future__ import annotations

import heapq
import math
import random

import numpy as np
from scipy.sparse import csr_array

from .constants import DECISION_MEMORY, RATE_MEMORY, STUCK_ABOVE, STUCK_BELOW
from .exploration import hazard_draws
from .network import Network
from .neuron import Neuron
from .clock import TOLERANCE, before, slack
from .propagation import Wave


def slack_v(times: np.ndarray) -> np.ndarray:
    """clock.slack, as a vector."""
    return TOLERANCE * np.maximum(1.0, np.abs(times))


class SynapseDraws:
    """The exploration stream's uniforms for the synapses' decisions, `count` = E + O a wave (AUTHORITY.md §3.8, §7.3).

    From a `random.Random`, numpy's MT19937 is loaded with its state and makes
    the uniforms `random()` makes -- two 32-bit words a uniform, (a >> 5) 2^26
    + (b >> 6) over 2^53, CPython's genrand_res53, equal and not approximately
    equal (§12.6) -- and `close` hands the state back, so the stream stands
    where the object engine leaves it. Any other stream (a test's chosen
    values, a subclass) is drawn one `random()` at a time, as the objects draw it.
    """

    def __init__(self, rng, count: int):
        self.rng, self.count = rng, count
        self._held = None  # the Python stream's state as loaded, while numpy draws for it
        if type(rng) is random.Random:
            self._held = rng.getstate()
            self._generator = np.random.MT19937()
            key, pos = self._held[1][:-1], self._held[1][-1]
            self._generator.state = {"bit_generator": "MT19937", "state": {"key": np.array(key, dtype=np.uint32), "pos": pos}}

    def take(self) -> np.ndarray:
        """This wave's uniforms, E then O."""
        if self._held is None:
            return np.array(hazard_draws(self.rng, self.count), dtype=float)
        raw = self._generator.random_raw(2 * self.count)
        return ((raw[0::2] >> 5) * 67108864.0 + (raw[1::2] >> 6)) * (1.0 / 9007199254740992.0)

    def close(self) -> None:
        """Hand the state back to the Python stream: the next `random()` is the one after the last uniform taken here."""
        if self._held is not None:
            state = self._generator.state["state"]
            self.rng.setstate((self._held[0], tuple(int(k) for k in state["key"]) + (int(state["pos"]),), self._held[2]))
            self._held = None


class ArrayWave:
    """What happened in one wave of the array engine: its time and the indices of the neurons that fired."""

    __slots__ = ("number", "time", "fired")

    def __init__(self, number: int, time: float, fired: np.ndarray):
        self.number = number
        self.time = time
        self.fired = fired

    def __repr__(self) -> str:
        return f"ArrayWave(number={self.number}, time={self.time:g}, fired={self.fired.tolist()})"


class ArrayNetwork(Network):
    """A mesh run as arrays. Build the mesh first, then wrap it: `ArrayNetwork(Goo(seed=1))`."""

    engine = "arrays"

    def _explores_as(self, network) -> tuple:
        """How `network` explores: the mode, and under exploration at the synapse the rest hazard, the family, the
        scaling and the trace as they stand, and the scaling and trace kappa_i and every trace were computed from."""
        if network.exploration != "synapse":
            return (network.exploration,)
        return (network.exploration, network.synapse_hazard_rest, network.synapse_hazard_family,
                network.synapse_hazard_scaling, network.trace_mode, network._computed_from)

    def _refuse_run(self) -> bool:
        """What a run refuses where it runs (AUTHORITY.md §12.2), at every fire_input, propagate and _run; returns whether
        it explores at the synapse. Under exploration at the synapse, what the object engine refuses where a network
        runs (Network._refuse_synapse_run), read on the arrays' own settings, thresholds, active flags, widths and hebb,
        and a stream. The settings stay writable after the wrap, the wrapper's and the wrapped mesh's alike, and the
        arrays run the mesh's neurons and sync_to_mesh writes back into them, so the two must explore alike -- the mode,
        the rest hazard and the family the arrays read at every wave, and the scaling and trace kappa_i and every trace
        were computed from -- which setting it on the ArrayNetwork keeps them."""
        mesh = self.mesh
        if mesh.exploration != self._exploration:
            self._refuse_apart()
        if self._exploration != "synapse":
            return False
        if self.explore_rng is None:
            raise ValueError("exploration at the synapse draws for every synapse at every wave, whatever its rest "
                             "hazard, and needs a stream of its own (§7.3): run the epoch with an rng")
        self._refuse_synapse_run(self.neurons_list, self.threshold_v.tolist(), self.active.tolist(), self.delta_v.tolist())
        if self._explores_as(mesh) != self._explores_as(self):
            self._refuse_apart()
        return True

    def _refuse_apart(self) -> None:
        raise ValueError(f"the wrapped mesh explores as {self._explores_as(self.mesh)} and the array engine as "
                         f"{self._explores_as(self)}: the arrays run the mesh's neurons under the exploration taken at "
                         "the wrap, and the two are not run apart -- set it on the ArrayNetwork, which sets both "
                         "(§7.1, §12.2)")

    def _take_exploration(self, network) -> None:
        """The exploration as `network` holds it (§7.1, §7.6, §7.7, §8.17), and kappa_i as set_exploration computed it
        there: a rule, computed once (§7.5), taken rather than recomputed."""
        self._exploration = network.exploration
        self.synapse_hazard_rest, self.synapse_hazard_family = network.synapse_hazard_rest, network.synapse_hazard_family
        self.synapse_hazard_scaling, self.trace_mode = network.synapse_hazard_scaling, network.trace_mode
        self._computed_from = network._computed_from
        self.synapse_scale_v = np.array([x.synapse_scale for x in self.neurons_list], dtype=float)  # kappa_i

    def __init__(self, mesh):
        self.mesh = mesh
        self.across, self.rows = mesh.across, mesh.rows
        self._init_network(mesh.across, mesh.weight_range)
        self.epoch = mesh.epoch
        self.time, self.interval, self.input_time = mesh.time, mesh.interval, mesh.input_time
        self.horizon = mesh.horizon
        self.readout, self.read, self.read_window = mesh.readout, mesh.read, mesh.read_window
        self.population, self.quash_rate, self.quash_k = mesh.population, mesh.quash_rate, mesh.quash_k
        self.temperature = mesh.temperature  # the evidence critic's temperature (§8)
        self.clock = mesh.clock  # clock neurons at the front of the input zone (§4.3)
        self.drive, self.input_rate, self.input_rate_off = mesh.drive, mesh.input_rate, mesh.input_rate_off
        self.drive_steps = mesh.drive_steps  # DRIVE_STEPS under the charged drive (5.4b)
        self.presentation = mesh.presentation  # the window the drive runs in (§5.4a), the mesh's
        self.explore_rng = mesh.explore_rng
        self.rate_on = mesh.rate_on
        self.pickiness = mesh.pickiness  # spikes: the count read's line (§5.10, §9.5)
        self.rule = mesh.rule
        self.seed = mesh.seed
        self.threshold = mesh.threshold
        self.minimum_potential = mesh.minimum_potential
        self._rng = random.Random()
        self._rng.setstate(mesh._rng.getstate())  # the same input sequence as the mesh would draw
        self.input_stream, self.input_at = mesh.input_stream, mesh.input_at  # and the same attached stream (§4.5)
        self.input_labels, self.input_label = mesh.input_labels, mesh.input_label  # with its labels, when a dataset gave them (§8)
        for name in ("input_pattern", "target_pattern", "input_bits", "input_coded", "input_events"):
            setattr(self, name, getattr(mesh, name))

        neurons = list(mesh.all_neurons())
        self.neurons_list = neurons
        self.index = {neuron: i for i, neuron in enumerate(neurons)}
        n = len(neurons)
        self.potential = np.array([x.potential for x in neurons], dtype=float)
        self.threshold_v = np.array([x.threshold for x in neurons], dtype=float)
        self.floor = np.array([x.minimum_potential for x in neurons], dtype=float)
        self.escape_delta = mesh.escape_delta  # escape noise (§5.2): ESCAPE_DELTA on this network
        self.delta_v = np.array([x.delta for x in neurons], dtype=float)  # each neuron's decision width, in potential units
        self.escape_scale = getattr(mesh, "escape_scale", 1.0)  # sqrt(N0 / N), the count's scaling of every hazard (§5.2)
        self.escape_scale_v = np.array([x.escape_scale for x in neurons], dtype=float)
        self.exposed_since = np.array([x.exposed_since for x in neurons], dtype=float)  # since when each hazard has run
        self.draw = np.ones(n)  # this wave's uniforms for the decisions
        self.rate = np.array([x.rate for x in neurons], dtype=float)
        self.centred = getattr(mesh, "centred", False)  # the hebb eligibility charges every decision (§6.7, the single-spike
        # rule); `traced`, whether each synapse's trace is kept, is the base class's property: escape noise, or this
        self.expectation = np.array([np.nan if x.expectation is None else x.expectation for x in neurons], dtype=float)  # p_hat_j
        self.decisions = np.array([x.decisions for x in neurons], dtype=np.int64)  # decisions to date, for the warm start
        self.expected = np.array([x.expected for x in neurons], dtype=float)  # E_j: the spikes expected since the last spike
        self.gain = np.array([x.gain for x in neurons], dtype=float)  # G_j (§8.16), under exploration at the synapse
        self.read_count = np.array([x.read_count for x in neurons], dtype=np.int64)  # an output's read synapse's escapes
        # this epoch (§7.9), added to its spikes in the count read (§5.10) and nowhere else; zeroed at the reset
        self._take_exploration(mesh)
        self.fired_wave = np.full(n, -1, dtype=np.int64)  # -1: has not fired this epoch
        for x in neurons:
            if x.has_fired:
                self.fired_wave[self.index[x]] = x.fired_in_wave
        self.forced = np.array([x.forced for x in neurons], dtype=bool)
        # the clock: when each neuron last spiked (-inf: never), the spike before that, and how many spikes ever
        self.fired_at = np.array([-np.inf if x.fired_at is None else x.fired_at for x in neurons], dtype=float)
        self.previous_fired_at = np.array([-np.inf if x.previous_fired_at is None else x.previous_fired_at for x in neurons], dtype=float)
        self.rate_level = np.array([x.rate_level for x in neurons], dtype=float)  # Hz, brought up to rate_at (§4.3)
        self.rate_at = np.array([x.rate_at for x in neurons], dtype=float)
        self.spikes = np.array([x.spikes for x in neurons], dtype=np.int64)
        self.spikes_at_reset = np.array([x.spikes_at_reset for x in neurons], dtype=np.int64)  # the count read's start (§4.3)
        self.last_update = np.array([x.last_update for x in neurons], dtype=float)  # the lazy leak's bookkeeping

        e = len(mesh.connections)
        connections = [mesh.connections[i] for i in range(1, e + 1)]
        self._edges = connections  # edge id (0-based) -> its Connection, for sync_to_mesh
        self.source = np.array([self.index[c.source] for c in connections], dtype=np.int64)
        self.target = np.array([self.index[c.target] for c in connections], dtype=np.int64)
        self.weight = np.array([c.weight for c in connections], dtype=float)
        self.active = np.array([c.is_active for c in connections], dtype=bool)
        self.last_signal = np.array([-np.inf if c.last_signal is None else c.last_signal for c in connections], dtype=float)
        self.delivered_wave = np.full(e, -1, dtype=np.int64)  # the wave of this epoch each connection delivered in
        self.trace = np.array([c.trace for c in connections], dtype=float)  # its charge in its target's potential (§6.7)
        self.trace_at = np.array([c.trace_at for c in connections], dtype=float)
        self.score = np.array([c.score for c in connections], dtype=float)  # the eligibility this epoch (§6.7)
        self.noted = np.array([c.noted for c in connections], dtype=float)  # B_ij: each open arrival's note (§6.7)
        self._active_edges = np.flatnonzero(self.active)
        self._build_matrix()

        self.input_index = np.array([self.index[x] for x in mesh.input_row()], dtype=np.int64)
        self.output_index = np.array([self.index[x] for x in mesh.output_row()], dtype=np.int64)
        # exploration at the synapse's layout (§3.8, §7.9): the draw each edge takes -- its place in edge order, sources
        # in index order and each source's synapses as the topology built them, which connection-id order need not be --
        # then one read synapse per output neuron, in the order the output row first names it, drawing after every edge
        flat = {id(c): k for k, c in enumerate(c for x in neurons for c in x.outgoing)}
        self._edge_draw = np.array([flat[id(c)] for c in connections], dtype=np.int64)
        self._readers = np.array(list(dict.fromkeys(self.output_index.tolist())), dtype=np.int64)
        self._synapses = np.bincount(self.source, minlength=n)  # F_i, every synapse out of i, active or not (§3.8, §7.7)
        self._synapses[self._readers] += 1  # and an output's read synapse (§7.9)
        self.waves: list[ArrayWave] = []
        # the schedule: distinct times, each with the sources whose signals arrive then, the neurons forced then, the
        # edges a ventured signal arrives along then (§7.9) and the charged drive's deliveries due then (5.4b)
        self._times: list[float] = []
        self._arrivals: dict[float, np.ndarray] = {}
        self._stimuli: dict[float, list[int]] = {}  # indices, not a mask: rate drive lands one arrival per moment
        self._ventured: dict[float, np.ndarray] = {}  # edge ids, one entry per signal
        self._charges: dict[float, list[tuple[int, int]]] = {}  # (neuron index, DRIVE_STEPS), in the order scheduled
        for time, connection, ventured in mesh.schedule.pending(marks=True):
            if ventured:  # an escape of its one synapse, not a relay of its source's spike (§7.9)
                self._venture(np.array([connection.id - 1], dtype=np.int64), time)
            else:
                self._arrive(self.index[connection.source], time)
        for time, neuron, steps in mesh.schedule.charges():
            self._charge_at(self.index[neuron], steps, time)

    # --- the matrix -------------------------------------------------------

    def _build_matrix(self) -> None:
        """Delivery matrix W[target, source] = weight over active edges, and its 0/1 shadow for "touched"."""
        n = len(self.neurons_list)
        edges = self._active_edges
        order = np.lexsort((self.source[edges], self.target[edges]))  # CSR order: by target, then source
        self._order = edges[order]  # edge id (0-based) at each data position
        rows, cols = self.target[self._order], self.source[self._order]
        counts = np.bincount(rows, minlength=n)
        indptr = np.concatenate(([0], np.cumsum(counts)))
        # Rows 0..n-1 carry the weights (W[target, source]); rows n..2n-1 carry a 1 per edge, so one
        # product gives both the summed input and whether a neuron was touched at all.
        data = np.concatenate((self.weight[self._order], np.ones(len(cols))))
        indices = np.concatenate((cols, cols)).astype(np.int32)
        indptr2 = np.concatenate((indptr, indptr[1:] + indptr[-1])).astype(np.int32)
        self._matrix = csr_array((data, indices, indptr2), shape=(2 * n, n))
        self._edge_count = len(cols)
        self._out_degree = np.bincount(self.source[edges], minlength=n).astype(float)  # active edges out of each neuron
        self._matrix_dirty = False

    def _refresh_matrix(self) -> None:
        if self._matrix_dirty:
            self._matrix.data[: self._edge_count] = self.weight[self._order]
            self._matrix_dirty = False

    # --- the schedule ----------------------------------------------------------

    def _key(self, time: float) -> float:
        time = float(time)
        if (time not in self._arrivals and time not in self._stimuli and time not in self._ventured
                and time not in self._charges):
            heapq.heappush(self._times, time)
        return time

    def _venture(self, edges: np.ndarray, time: float) -> None:
        """Ventured signals along `edges` (edge ids, 0-based, one entry per signal), arriving at `time` (§7.9)."""
        time = self._key(time)
        held = self._ventured.get(time)
        self._ventured[time] = edges if held is None else np.concatenate((held, edges))

    def _charge_at(self, index: int, steps: int, time: float) -> None:
        """A delivery of the charged drive to the neuron at `index`, at `time`: theta / `steps` on the threshold it holds
        when the delivery lands (5.4b), as Schedule.charge."""
        time = self._key(time)
        at = self._charges.get(time)
        if at is None:
            self._charges[time] = [(index, steps)]
        else:
            at.append((index, steps))

    def _arrive(self, source: int, time: float) -> None:
        time = self._key(time)
        if time not in self._arrivals:
            self._arrivals[time] = np.zeros(len(self.neurons_list))
        self._arrivals[time][source] = 1.0

    def _arrive_all(self, firing: np.ndarray, time: float) -> None:
        time = self._key(time)
        if time not in self._arrivals:
            self._arrivals[time] = np.zeros(len(self.neurons_list))
        self._arrivals[time] += firing

    def _stimulate(self, forced: np.ndarray, time: float) -> None:
        """Force every neuron the mask names, at `time`."""
        self._stimulate_at(np.flatnonzero(forced).tolist(), time)

    def _stimulate_at(self, indices, time: float) -> None:
        """Force the neurons at `indices`, at `time`.

        The stimuli are kept as index lists rather than masks because rate drive (§4.3)
        delivers each arrival at its own continuous moment, so a mask per arrival was one
        60-wide allocation per stimulus -- over a thousand an epoch, and the largest
        single line in the profile after the wave loop itself.
        """
        time = self._key(time)
        at = self._stimuli.get(time)
        if at is None:
            self._stimuli[time] = list(indices)
        else:
            at.extend(indices)

    def pending(self) -> list[tuple[float, int]]:
        """Relayed signals in flight as (time, source index), by time: what sync_to_mesh puts back on the mesh's schedule,
        with the ventured ones of `pending_ventured`."""
        return [(time, int(i)) for time in sorted(self._arrivals) for i in np.flatnonzero(self._arrivals[time] > 0)]

    def pending_ventured(self) -> list[tuple[float, int]]:
        """Ventured signals in flight as (time, edge id, 0-based), by time and then as scheduled (§7.9)."""
        return [(time, int(e)) for time in sorted(self._ventured) for e in self._ventured[time].tolist()]

    def pending_charges(self) -> list[tuple[float, int, int]]:
        """The charged drive's deliveries still to come as (time, neuron index, DRIVE_STEPS), by time and then as
        scheduled (5.4b)."""
        return [(time, i, steps) for time in sorted(self._charges) for i, steps in self._charges[time]]

    def _run(self, until: float = math.inf) -> list[ArrayWave]:
        """Process every wave due before `until` (see propagation.Schedule.run), appending to this epoch's waves. Under
        exploration at the synapse the wave's uniforms come from `SynapseDraws`, which hands the stream back however the
        run ends (§7.3, §12.6)."""
        synaptic = self._refuse_run()
        draws = SynapseDraws(self.explore_rng, len(self.source) + len(self._readers)) if synaptic else None
        try:
            return self._waves(until, synaptic, draws)
        finally:
            if draws is not None:
                draws.close()

    def _waves(self, until: float, synaptic: bool, draws) -> list[ArrayWave]:
        """_run's loop. Under exploration at the synapse a wave takes its charges before its signals (§3.5, 5.4b),
        delivers its ventured signals with its relayed ones (§7.9), and after the fire phase has its synapses decide
        (§7.5, `_decide_synapses`); its neurons take the comparison (§6.13) and draw nothing."""
        potential, threshold, floor, fired_wave = self.potential, self.threshold_v, self.floor, self.fired_wave
        n = len(potential)
        hop = Neuron.hop
        out_degree = self._out_degree
        hazard = self.hazard and not synaptic  # the neurons' own draws (§5.2); under the synapses', every width is 0
        while self._times and before(self._times[0], until):
            self._refresh_matrix()
            first = heapq.heappop(self._times)
            limit = first + slack(first)
            firing = self._arrivals.pop(first, None)
            stimulus = self._stimuli.pop(first, None)
            ventured = self._ventured.pop(first, None)
            charges = self._charges.pop(first, None)
            time = first
            anchored = stimulus is not None or charges is not None
            while self._times and self._times[0] <= limit:  # the same moment, within the clock's slack
                other = heapq.heappop(self._times)
                more = self._arrivals.pop(other, None)
                if more is not None:
                    firing = more if firing is None else firing + more
                along = self._ventured.pop(other, None)
                if along is not None:
                    ventured = along if ventured is None else np.concatenate((ventured, along))
                forced_too = self._stimuli.pop(other, None)
                if forced_too is not None:
                    stimulus = forced_too if stimulus is None else stimulus + forced_too
                charged_too = self._charges.pop(other, None)
                if charged_too is not None:
                    charges = charged_too if charges is None else charges + charged_too
                if not anchored and (forced_too is not None or charged_too is not None):
                    time = other  # an input's exact time anchors the wave: the earliest, as the objects take it (§3.4)
                    anchored = True
            number = len(self.waves)
            refractory = self.fired_at + Neuron.refractory > time + slack(time)  # fired within the refractory period before now
            take = np.zeros(n, dtype=bool)
            if charges is not None:  # the charged drive's deliveries first, one at a time in the wave's order (§3.5, 5.4b),
                # each touching its input -- brought up to now as a signal brings its target; the floor has nothing to do
                # there, a charge being positive (§7.5 refuses a threshold at or below zero) on a potential at or above it
                for i, steps in charges:
                    self.forced[i] = True  # 5.8's driven mark, set by any delivery, one dropped at a refractory input included
                    if not refractory[i]:
                        self._leak_one(i, time)
                        potential[i] += threshold[i] / steps  # theta / DRIVE_STEPS, a division, on the threshold it holds now
            if firing is not None or ventured is not None:
                if firing is not None:
                    both = self._matrix @ firing
                    incoming, touched = both[:n], both[n:] > 0
                else:
                    incoming, touched = np.zeros(n), np.zeros(n, dtype=bool)
                if ventured is not None:  # each along its one edge, beside the per-source rows (§7.9)
                    reached = self.target[ventured]
                    incoming = incoming + np.bincount(reached, weights=self.weight[ventured], minlength=n)
                    touched[reached] = True
                take = touched & ~refractory
                self.leak(time, take)  # the lazy leak: a neuron is brought up to date when a signal reaches it
                summed = potential[take] + incoming[take]
                potential[take] = np.maximum(summed, floor[take])
                if firing is not None:
                    delivering = self.active & (firing[self.source] > 0)
                    self.delivered_wave[delivering] = number
                    integrated = delivering & take[self.target]
                    self.last_signal[integrated] = time
                else:
                    integrated = np.zeros(len(self.source), dtype=bool)
                landed = None
                if ventured is not None:  # delivered, and taken in where its target is not refractory (§7.9, §1.7)
                    self.delivered_wave[ventured] = number
                    landed = ventured[take[reached]]
                    self.last_signal[landed] = time
                if self.traced:  # what each synapse now has in its target's potential (§6.7)
                    if synaptic:
                        self._note_arrivals(integrated, landed, time)
                    else:
                        if Neuron.tau == math.inf:  # the evidence accumulator (§5.1): a count, no decay evaluated
                            self.trace[integrated] += 1.0
                            self.noted[integrated] += self.expected[self.target[integrated]]  # the debit counts from here
                        else:
                            self.trace[integrated] = self.trace[integrated] * np.exp(-(time - self.trace_at[integrated]) / Neuron.tau) + 1.0
                        self.trace_at[integrated] = time
                    clipped = np.zeros(n, dtype=bool)
                    clipped[np.flatnonzero(take)[summed < floor[take]]] = True  # the floor bit: the weights are not in it
                    if synaptic:
                        self._settle_gain(clipped)  # and G restarts on the neuron, an arrival open on it or none (§8.16)
                    else:
                        self._close_arrivals(clipped[self.target])
            if synaptic:
                uniforms = draws.take()  # §3.8: one per synapse in edge order, then one per read synapse, whatever fires
            elif hazard:
                self.draw = np.array(hazard_draws(self.explore_rng, n))  # §7.3: one uniform per neuron, in neuron order
            if stimulus is None:
                fire_forced = None
            else:
                fire_forced = np.zeros(n, dtype=bool)
                fire_forced[stimulus] = True
                fire_forced &= ~refractory
            if Neuron.bored_after > 0.0:  # threshold homeostasis: the threshold falls with the silence since the last spike
                since = np.where(self.fired_at == -np.inf, 0.0, self.fired_at)
                facing = threshold - threshold * (time - since) / Neuron.bored_after
            else:
                facing = threshold
            deciding = ~refractory if fire_forced is None else ~refractory & ~fire_forced  # a stimulus decides nothing
            m = np.zeros(n)
            if hazard:  # the decision is a draw (§5.2)
                standing = self.potential_at(time)  # a decayed copy: `potential` stays the state, which a spike resets below
                # a collapsed axis -- no incoming synapses under §5.2's scaling, so no width -- keeps the deterministic rule,
                # as the objects and Rust do: it is not divided by its zero width (which gave NaN, and silence, until
                # September 16, 2026)
                soft = deciding & (self.delta_v > 0.0)
                elapsed = np.maximum(time - self.exposed_since, 0.0)
                m[soft] = np.minimum(elapsed[soft] / hop * self.escape_scale_v[soft]
                                     * np.exp((standing[soft] - facing[soft]) / self.delta_v[soft]), 1e3)
                ready = np.where(soft, self.draw < -np.expm1(-m), deciding & (standing >= facing))
                self.exposed_since[soft] = time
            else:
                ready = deciding & (self.potential_at(time) >= facing)  # everyone, touched or not, like the object engine
            # the decision charges the eligibility (§6.7, the single-spike rule): a credit c and an expectation q per neuron
            credit_v = np.zeros(n)
            if self.centred:
                y = ready.astype(float)
                unset = deciding & np.isnan(self.expectation)
                self.decisions[deciding] += 1
                charged = deciding & ~unset
                credit_v[charged] = y[charged]
                q = np.where(charged, self.expectation, 0.0)
                self.expectation[unset] = y[unset]  # the first decision sets the expectation and charges nothing
                rate = np.maximum(DECISION_MEMORY, 1.0 / self.decisions[charged])
                self.expectation[charged] = self.expectation[charged] + rate * (y[charged] - self.expectation[charged])
            else:
                charged = m > 0.0
                hit = charged & ready
                credit_v[hit] = m[hit] * np.exp(-m[hit]) / -np.expm1(-m[hit])  # m e^-m / (1 - e^-m), as the objects compute it
                q = np.where(charged & ~ready, m, 0.0)
            if charged.any():
                if Neuron.tau == math.inf:  # the evidence accumulator (§5.1): the debit settles per arrival, the credit at the spike
                    self.expected[charged] += q[charged]
                else:
                    e = credit_v - q
                    edges = charged[self.target] & (self.trace != 0.0)
                    if edges.any():
                        self.score[edges] += e[self.target[edges]] * self.trace[edges] * np.exp(-(time - self.trace_at[edges]) / Neuron.tau)
            fired = ready if fire_forced is None else (fire_forced | ready)
            idx = np.flatnonzero(fired)
            if len(idx):
                self.previous_fired_at[idx] = self.fired_at[idx]
                self.fired_at[idx] = time
                # the rate trace: one spike's worth on, decaying with Neuron.rate_tau (§4.3). math.exp, as the
                # object engine uses, so the two agree to the last bit; a level of 0 short-circuits there too.
                self.rate_level[idx] = [0.0 if not lv else lv * math.exp(-(time - at) / Neuron.rate_tau)
                                        for lv, at in zip(self.rate_level[idx].tolist(), self.rate_at[idx].tolist())]
                self.rate_level[idx] += 1000.0 / Neuron.rate_tau
                self.rate_at[idx] = time
                self.last_update[idx] = time
                potential[idx] = 0.0  # the spike resets the potential
                if synaptic:
                    self.exposed_since[idx] = time  # the synapses' exposure runs from the spike, and nothing suspends it (§7.8)
                else:
                    self.exposed_since[idx] = time + Neuron.refractory  # the hazard resumes after the refractory period (§5.2)
                if self.traced:
                    if synaptic:
                        self._settle_gain(fired)  # the spike settles x G - B into every open arrival and restarts G (§8.16)
                    else:
                        self._close_arrivals(fired[self.target], credit_v)  # the potential reset: the spike settles every arrival (§6.7)
                        self.expected[idx] = 0.0
                fired_wave[idx] = number
                self.spikes[idx] += 1
                if fire_forced is not None:
                    self.forced |= fire_forced
                if out_degree @ fired > 0:  # someone has active connections out: signals in flight
                    self._arrive_all(fired.astype(float), time + hop)
            if synaptic:  # §7.5: the synapses decide after the spikes
                self._decide_synapses(uniforms, fired, time)
            self.waves.append(ArrayWave(number, time, idx))
            if self.quash_rate and len(idx):
                self._quash(time, idx)
        return self.waves

    def _close_arrivals(self, edges: np.ndarray, credit=None) -> None:
        """Close the open arrivals on `edges` (a mask over connections): under the evidence accumulator settle their credit
        and debit into the score first (§6.7), under the leak zero the traces. `credit` is per neuron, indexed by target."""
        if Neuron.tau == math.inf:
            open_ = edges & (self.trace != 0.0)
            if open_.any():
                t = self.target[open_]
                c = 0.0 if credit is None else credit[t]
                self.score[open_] += c * self.trace[open_] - (self.trace[open_] * self.expected[t] - self.noted[open_])
                self.trace[open_] = 0.0
                self.noted[open_] = 0.0
        else:
            self.trace[edges] = 0.0

    # --- exploration at the synapse (AUTHORITY.md §7.5-§7.9, §8.16-§8.17) ----------------

    def _leak_one(self, i: int, now: float) -> None:
        """Neuron.leak for the neuron at `i` alone, with math.exp as the objects take it: a charge's input (5.4b)."""
        elapsed = now - self.last_update[i]
        if elapsed > 0.0:
            if Neuron.tau != math.inf:
                self.potential[i] *= math.exp(-elapsed / Neuron.tau)
            self.last_update[i] = now

    def _note_arrivals(self, integrated: np.ndarray, landed, time: float) -> None:
        """What the wave's arrivals do to the traces under exploration at the synapse: `integrated` the relayed edges
        whose targets took them in (a mask), `landed` the ventured ones (edge ids, one per signal, or None). Each raises
        its trace by one -- decayed first under the leak -- and under the evidence accumulator notes the gain as it
        stands before the wave's decisions post (§8.16); under TRACE ventured a relayed arrival is, for the synapse's
        learning, not there: no count, no note, no decay, no moment moved (§8.17)."""
        counted = np.zeros(len(self.trace)) if self.trace_mode == "ventured" else integrated.astype(float)
        if landed is not None and len(landed):
            counted += np.bincount(landed, minlength=len(counted))
        moved = counted > 0.0
        if not moved.any():
            return
        if Neuron.tau == math.inf:
            self.trace[moved] += counted[moved]
            self.noted[moved] += counted[moved] * self.gain[self.target[moved]]
        else:
            self.trace[moved] = self.trace[moved] * np.exp(-(time - self.trace_at[moved]) / Neuron.tau) + counted[moved]
        self.trace_at[moved] = time

    def _settle_gain(self, neurons: np.ndarray) -> None:
        """A settle under exploration at the synapse, on the neurons of the mask `neurons` -- their spike, the floor, a
        forced spike or a discharge (§8.16): under the evidence accumulator every open arrival on their incoming synapses
        posts x G - B and is cleared, under the leak the traces are zeroed, and either way G restarts at 0 on every one
        of them, whether or not an arrival was open (restarted here per neuron, not through _close_arrivals' edges)."""
        edges = neurons[self.target]
        if Neuron.tau == math.inf:
            open_ = edges & (self.trace != 0.0)
            if open_.any():
                self.score[open_] += self.trace[open_] * self.gain[self.target[open_]] - self.noted[open_]
                self.trace[open_] = 0.0
                self.noted[open_] = 0.0
        else:
            self.trace[edges] = 0.0
        self.gain[neurons] = 0.0

    def _decide_synapses(self, uniforms: np.ndarray, fired: np.ndarray, time: float) -> None:
        """Network._decide_synapses as vectors: after the fire phase every synapse of every source that did not spike this
        wave decides (AUTHORITY.md §7.5), refractory or not (§7.8), touched or not, on one m and one P = -expm1(-m) per
        source, in §7.5's engine note's forms -- u = min(max(V, 0), theta) / theta on the potential decayed to now, h0 **
        (1 - u) or h0 + (1 - h0) u, m = dt / hop, times kappa_i, times h, capped at 1e3. A synapse escapes iff its
        uniform is strictly below P, and its escape is a ventured signal one hop later along its one edge (§7.9), the
        wave's held in edge order (§3.6); an output's read synapse decides with them, its uniform after every edge's, and
        its escape counts toward the output's read at this wave. Where V > 0 and m > 0 the wave posts §8.16's entry,
        a c - (F - a) m with c = m e^-m / -expm1(-m) and F - a an integer, times (1 - h0) / h under the linear family:
        into the gain under the evidence accumulator, and under the leak by one walk over the source's fan-in; an entry
        that is not finite, rho~ overflowing where h has underflowed, is refused as the objects refuse it (§12.2). Every
        other source's one exposure clock is brought to now, F_i = 0 included (§7.5): a spike's own was, at the spike
        (§7.8)."""
        n, e = len(self.potential), len(self.source)
        deciding = ~fired
        standing = self.potential_at(time)
        threshold = self.threshold_v
        rest = self.synapse_hazard_rest
        linear = self.synapse_hazard_family == "linear"
        u = np.minimum(np.maximum(standing, 0.0), threshold) / threshold
        h = rest + (1.0 - rest) * u if linear else rest ** (1.0 - u)
        m = np.minimum(np.maximum(time - self.exposed_since, 0.0) / Neuron.hop * self.synapse_scale_v * h, 1e3)
        chance = -np.expm1(-m)
        escaped = deciding[self.source] & (uniforms[self._edge_draw] < chance[self.source])
        readers = self._readers
        read = deciding[readers] & (uniforms[e:] < chance[readers])
        escapes = np.bincount(self.source[escaped], minlength=n)
        escapes[readers[read]] += 1  # a read escape is one of F_i's decisions (§8.16)
        self.read_count[readers[read]] += 1  # it delivers nothing, so nothing of it is in flight (§7.9)
        posting = deciding & (self._synapses > 0) & (m > 0.0) & (standing > 0.0)  # §8.16: only while V_i > 0, c where m > 0
        if posting.any():
            mp, a = m[posting], escapes[posting]
            entry = a * (mp * np.exp(-mp) / -np.expm1(-mp)) - (self._synapses[posting] - a) * mp
            if linear:
                with np.errstate(over="ignore", invalid="ignore"):  # refused below, as the objects refuse it
                    entry = (1.0 - rest) / h[posting] * entry  # rho~, the log-derivative the linear family leaves unfolded
                if not np.isfinite(entry).all():
                    raise ValueError("under the linear family rho~ = (1 - h0) / h multiplies the entry, and where h has "
                                     "underflowed -- h0 = 0, and a source's potential leaked below the normal range -- "
                                     "it overflows and the entry is not finite: the run is refused rather than post it "
                                     "(§8.16, §12.2)")
            if Neuron.tau == math.inf:
                self.gain[posting] += entry  # after this wave's arrivals have noted (§8.16)
            else:  # the leak (§8.12): the walk, once a wave per posting source
                posted = np.zeros(n)
                posted[posting] = entry
                walk = posting[self.target] & (self.trace != 0.0)
                if walk.any():
                    self.score[walk] += posted[self.target[walk]] * self.trace[walk] * np.exp(-(time - self.trace_at[walk]) / Neuron.tau)
        self.exposed_since[deciding] = time  # one clock per source, F_i = 0 included; a spike's was brought to now by it
        if escaped.any():  # held in edge order, as the objects push them (§3.6), not in connection-id order
            k = np.flatnonzero(escaped)
            self._venture(k[np.argsort(self._edge_draw[k])], time + Neuron.hop)

    def settle_scores(self) -> None:
        """Network.settle_scores as vectors: under the evidence accumulator, every open arrival's debit into its score (§6.7)
        -- or under exploration at the synapse its net credit, x G - B, re-basing B = x G and leaving x and G (§8.16)."""
        if Neuron.tau != math.inf or not self.traced:
            return
        open_ = self.trace != 0.0
        if open_.any():
            if self._exploration == "synapse":
                owed = self.trace[open_] * self.gain[self.target[open_]]
                self.score[open_] += owed - self.noted[open_]
                self.noted[open_] = owed
                return
            e = self.trace[open_] * self.expected[self.target[open_]]
            self.score[open_] -= e - self.noted[open_]
            self.noted[open_] = e

    def centre(self, on: bool) -> None:
        """Network.centre: the hebb eligibility charges every decision (§6.7); the mesh's neurons follow, for the checkpoint."""
        if on and self._exploration == "synapse":
            raise ValueError("hebb is refused under exploration at the synapse: the decisions are the synapses', and it has "
                             "no neuron decision to centre (§8.3)")
        self.centred = bool(on)
        for neuron in self.neurons_list:
            neuron.centred = self.centred
            neuron.traced = self.centred or neuron.delta > 0.0 or neuron.synaptic

    def set_exploration(self, mode: str, **settings) -> None:
        """Network.set_exploration, made on the mesh -- whose neurons the arrays run, and which holds every check and
        computes kappa_i -- with what the arrays have reached written into it first, then taken back: the settings,
        kappa_i and the widths, every one 0 under exploration at the synapse (§6.13)."""
        if mode == "synapse" and self.centred:  # the arrays' own hebb, which centre does not give the mesh
            raise ValueError("hebb is refused under exploration at the synapse: the decisions are the synapses', and it has "
                             "no neuron decision to centre (§8.3)")
        self.sync_to_mesh()
        self.mesh.set_exploration(mode, **settings)
        self._take_exploration(self.mesh)
        if mode == "synapse":
            self.escape_delta, self.escape_scale = self.mesh.escape_delta, self.mesh.escape_scale
        self._take_widths()

    def set_delta(self, delta: float) -> None:
        """Network.set_delta, which sets the widths on the mesh's neurons, and the arrays' own taken from them: the
        decision reads delta_v and escape_scale_v, not the neurons (§6.5, §6.13)."""
        super().set_delta(delta)
        self._take_widths()

    def _take_widths(self) -> None:
        self.delta_v = np.array([x.delta for x in self.neurons_list], dtype=float)
        self.escape_scale_v = np.array([x.escape_scale for x in self.neurons_list], dtype=float)

    # --- the epoch ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.neurons_list)

    @property
    def neurons(self):
        """The mesh's neuron container, for code that only counts or lists them (see sync_to_mesh)."""
        return self.mesh.neurons

    @property
    def connections(self):
        return self.mesh.connections

    def all_neurons(self):
        return self.neurons_list

    def get_neuron_at(self, place: int, row: int, *args):
        return self.mesh.get_neuron_at(place, row, *args)

    def input_row(self):
        return self.mesh.input_row()

    def output_row(self):
        return self.mesh.output_row()

    def input_width(self) -> int:
        return self.mesh.input_width()

    def has_fired(self) -> np.ndarray:
        return self.fired_wave >= 0

    def total_spikes(self) -> int:
        return int(self.spikes.sum())

    def reset(self, discharge: bool = False) -> None:
        synaptic = self._exploration == "synapse"
        if discharge:
            self.potential[:] = 0.0
            if self.traced:  # nothing is left in a zeroed potential (§6.7)
                if synaptic:
                    self._settle_gain(np.ones(len(self.potential), dtype=bool))  # and every G restarts (§8.16)
                else:
                    self._close_arrivals(np.ones(len(self.trace), dtype=bool))
        if self.traced:
            self.score[:] = 0.0  # the score is the epoch's (§6.7)
            if Neuron.tau == math.inf:  # an open arrival's debit counts from here: B = x E, or x G at the synapses (§8.16)
                self.noted = self.trace * (self.gain if synaptic else self.expected)[self.target]
        self.read_count[:] = 0  # the read synapse's escapes are the epoch's (§7.9)
        self.fired_wave[:] = -1
        self.forced[:] = False
        self.delivered_wave[:] = -1
        self.spikes_at_reset = self.spikes.copy()  # Neuron.reset snapshots the count
        self.waves = []

    def leak(self, now: float, which=None) -> None:
        """Bring potentials up to `now` (all, or the boolean mask `which`): one vector operation, the same arithmetic as the objects."""
        mask = np.ones(len(self.potential), dtype=bool) if which is None else which
        elapsed = now - self.last_update
        moving = mask & (elapsed > 0.0)
        if moving.any():
            if Neuron.tau != math.inf:
                self.potential[moving] *= np.exp(-elapsed[moving] / Neuron.tau)
            self.last_update[moving] = now

    def potential_at(self, now: float) -> np.ndarray:
        """Every potential as it stands at `now`, decayed for the time since its last update; changes nothing."""
        elapsed = np.maximum(now - self.last_update, 0.0)
        if Neuron.tau == math.inf:
            return self.potential
        return self.potential * np.exp(-elapsed / Neuron.tau)

    def set_input(self, pattern, time: float | None = None) -> None:
        super().set_input(pattern, time)

    def fire_input(self, until: float | None = None) -> list[ArrayWave]:
        if self.input_pattern is None:
            raise ValueError("no input pattern set; call set_input() first")
        self._refuse_run()  # before anything is drawn or scheduled: a run it refuses leaves the network as it was
        steps = self._drive_steps()  # and so is a drive it refuses
        self.time = self.input_time if self.input_time is not None else self.next_time()
        self.epoch += 1
        self.input_events = self.input_schedule()  # the same draws as the mesh, from the same stream (§4.3)
        for place, when in self.input_events:
            if self.drive == "charged":
                self._charge_at(int(self.input_index[place]), steps, when)  # a delivery, not a forced spike (5.4b)
            else:
                self._stimulate_at((int(self.input_index[place]),), when)
        self.horizon = self.time + self.interval if until is None else float(until)
        return self._run(self.horizon)

    def propagate(self, fire=(), inputs=None, now: float | None = None, until: float | None = None) -> list[ArrayWave]:
        """Run a cascade from the mesh neurons `fire`, forced at `now` (default: the clock), up to `until` (default: one interval)."""
        self._refuse_run()
        if inputs:
            raise NotImplementedError("external input amounts are not vectorised; use the object engine")
        now = self.time if now is None else now
        fire = list(fire)
        if fire:  # as the objects push a stimulus per neuron: none makes no wave of its own at `now`
            forced = np.zeros(len(self.neurons_list), dtype=bool)
            forced[[self.index[neuron] for neuron in fire]] = True
            self._stimulate(forced, now)
        self.horizon = now + self.interval if until is None else float(until)
        return self._run(self.horizon)

    def output_width(self) -> int:
        return len(self.output_index)

    def output_counts(self) -> list[int]:
        """Each output neuron's spikes this epoch, plus its read synapse's escapes this epoch, as Network.output_counts
        (§5.10, §7.9): the read counts are 0 but under exploration at the synapse."""
        out = self.output_index
        return (self.spikes[out] - self.spikes_at_reset[out] + self.read_count[out]).tolist()

    def output_rates(self) -> list[float]:
        """Each output neuron's firing rate at the read, in Hz (see Network.output_rates)."""
        now, out = self.horizon, self.output_index
        return [0.0 if not lv else lv * math.exp(-(now - at) / Neuron.rate_tau)
                for lv, at in zip(self.rate_level[out].tolist(), self.rate_at[out].tolist())]

    def output_times(self) -> list[float | None]:
        return [float(self.fired_at[i]) if self.fired_wave[i] >= 0 else None for i in self.output_index]

    def fired_neurons(self):
        self.sync_to_mesh()
        return self.mesh.fired_neurons()

    def output_fired(self) -> list[bool]:
        self._refuse_read()
        fired_at = self.fired_at[self.output_index]
        if self.read == "again":
            return (fired_at > self.time + slack(self.time)).tolist()
        if self.read == "window" and self.read_window is not None:
            return (fired_at + slack_v(fired_at) >= self.horizon - self.read_window).tolist()
        if self.read == "count":  # §5.10: the count itself, against the row critic's pickiness (§9.5)
            return [n >= self.pickiness for n in self.output_counts()]
        return (self.fired_wave[self.output_index] >= 0).tolist()

    def output_counts_hz(self) -> list[float]:
        idx = self.output_index
        return ((self.spikes[idx] - self.spikes_at_reset[idx] + self.read_count[idx]) * (1000.0 / self.interval)).tolist()  # the count's (§5.10)

    # --- learning: the local rules (AUTHORITY.md §10) --------------------------------

    def _quash(self, time: float, idx: np.ndarray) -> int:
        """A refire is a cycle: weaken its contributing synapses (see local.quash), as one vector operation."""
        previous = self.previous_fired_at[idx]
        ridx = idx[previous > -np.inf]
        if not len(ridx):
            return 0
        n = len(self.potential)
        factor = np.zeros(n)
        # math.exp on the object engine's side too, so the two agree to the last bit
        factor[ridx] = [self.quash_rate * math.exp(-self.quash_k * (time - t)) for t in self.previous_fired_at[ridx].tolist()]
        refired = np.zeros(n, dtype=bool)
        refired[ridx] = True
        mask = self.active & refired[self.target] & (self.last_signal > self.previous_fired_at[self.target])
        if mask.any():
            low, high = self.weight_range
            self.weight[mask] = np.clip(self.weight[mask] * (1.0 - factor[self.target[mask]]), low, high)
            self._matrix_dirty = True
        return int(mask.sum())

    # --- learning: the reinforce rule, factored out ---------------------------------

    def reinforce(self, advantage: float, lr: float, eligibility: str) -> int:
        """The update of AUTHORITY.md §8.4, edge by edge, all at once. Returns connections changed."""
        if not self.explores:  # §8.3: no eligibility runs where the threshold decides
            raise ValueError("the reinforce rule refuses to learn where the threshold decides: REINFORCE estimates a gradient from the randomness of the decision, and with no width there is no randomness to estimate from. Give the network a positive ESCAPE_DELTA (--delta) (§8.3)")
        self.settle_scores()  # under the evidence accumulator the open arrivals' debit comes into the score first (§8.11)
        if not advantage:
            return 0
        # §8.4: every synapse into an unforced neuron moves by what its per-decision entries summed to
        mask = (self.score != 0.0) & ~self.forced[self.target]
        low, high = self.weight_range
        self.weight[mask] = np.clip(self.weight[mask] + lr * advantage * self.score[mask], low, high)
        self._matrix_dirty = True
        return int(mask.sum())

    def update_rates(self) -> None:
        unforced = ~self.forced
        fired = (self.fired_wave[unforced] >= 0).astype(float)
        self.rate[unforced] += RATE_MEMORY * (fired - self.rate[unforced])

    def homeostasis(self, rate: float, target: float) -> int:
        if rate <= 0:
            return 0
        unforced = ~self.forced
        self.threshold_v[unforced] = self.threshold_v[unforced] + rate * (self.rate[unforced] - target)  # nothing clips it
        self._refuse_thresholds(unforced)
        return int(unforced.sum())

    def unstick(self, rate: float, target: float) -> list[int]:
        """learning.unstick as one vector operation: every stuck neuron not forced this epoch."""
        if rate <= 0:
            return []
        stuck = np.flatnonzero(((self.rate > STUCK_ABOVE) | (self.rate < STUCK_BELOW)) & ~self.forced)
        self.threshold_v[stuck] = self.threshold_v[stuck] + rate * (self.rate[stuck] - target)
        self._refuse_thresholds(stuck)
        return stuck.tolist()

    def _refuse_thresholds(self, moved) -> None:
        """§7.5: under exploration at the synapse a threshold at or below zero leaves u undefined, and is refused at the
        moment homeostasis or un-sticking takes it there, as learning refuses it on the objects."""
        if self._exploration == "synapse" and (self.threshold_v[moved] <= 0.0).any():
            raise ValueError("a threshold moved to or below zero leaves u = clip(V, 0, theta) / theta undefined, and under "
                             "exploration at the synapse it is refused (§7.5)")

    def stuck(self) -> tuple[list[int], list[int]]:
        return np.flatnonzero(self.rate > STUCK_ABOVE).tolist(), np.flatnonzero(self.rate < STUCK_BELOW).tolist()

    # --- back to objects ------------------------------------------------------

    def sync_to_mesh(self) -> None:
        """Copy the arrays into the mesh's neurons, connections and schedule (for checkpoints and drawing)."""
        for i, neuron in enumerate(self.neurons_list):
            wave = int(self.fired_wave[i])
            neuron.potential = float(self.potential[i])
            neuron.threshold = float(self.threshold_v[i])
            neuron.rate = float(self.rate[i])
            neuron.expectation = None if np.isnan(self.expectation[i]) else float(self.expectation[i])
            neuron.decisions = int(self.decisions[i])
            neuron.expected = float(self.expected[i])
            neuron.gain = float(self.gain[i])
            neuron.read_count = int(self.read_count[i])
            neuron.has_fired = wave >= 0
            neuron.fired_in_wave = wave if wave >= 0 else None
            neuron.forced = bool(self.forced[i])
            neuron.fired_at = None if self.fired_at[i] == -np.inf else float(self.fired_at[i])
            neuron.previous_fired_at = None if self.previous_fired_at[i] == -np.inf else float(self.previous_fired_at[i])
            neuron.last_update = float(self.last_update[i])
            neuron.spikes = int(self.spikes[i])
            neuron.spikes_at_reset = int(self.spikes_at_reset[i])
            neuron.exposed_since = float(self.exposed_since[i])
        connections = self.mesh.connections
        for i, (weight, last) in enumerate(zip(self.weight.tolist(), self.last_signal.tolist()), start=1):
            connections[i].weight = weight
            connections[i].last_signal = None if last == -np.inf else last
        for i, (trace, at, score, noted) in enumerate(zip(self.trace.tolist(), self.trace_at.tolist(), self.score.tolist(),
                                                          self.noted.tolist()), start=1):
            connections[i].trace, connections[i].trace_at, connections[i].score, connections[i].noted = trace, at, score, noted
        mesh = self.mesh
        mesh.epoch = self.epoch
        mesh.time, mesh.interval, mesh.input_time, mesh.horizon = self.time, self.interval, self.input_time, self.horizon
        mesh.presentation = self.presentation
        mesh.rule = self.rule
        mesh.waves = [Wave(w.number, w.time, [], [self.neurons_list[i] for i in w.fired.tolist()]) for w in self.waves]
        mesh.schedule.clear()
        for time in sorted(set(self._arrivals) | set(self._ventured) | set(self._charges)):
            row = self._arrivals.get(time)
            if row is not None:
                for i in np.flatnonzero(row > 0).tolist():
                    for connection in self.neurons_list[i].outgoing:
                        if connection.is_active:
                            mesh.schedule.signal(connection, time)
            along = self._ventured.get(time)
            if along is not None:  # a ventured signal is its one edge, and keeps its mark (§7.9)
                for k in along.tolist():
                    mesh.schedule.signal(self._edges[k], time, ventured=True)
            for i, steps in self._charges.get(time, ()):  # the charged drive's deliveries still to come (5.4b)
                mesh.schedule.charge(self.neurons_list[i], steps, time)
        for name in ("input_pattern", "target_pattern", "input_bits", "input_coded", "input_events"):
            setattr(mesh, name, getattr(self, name))
        mesh._rng.setstate(self._rng.getstate())
        mesh.input_stream, mesh.input_at = self.input_stream, self.input_at
        mesh.input_labels, mesh.input_label = self.input_labels, self.input_label

    def __repr__(self) -> str:
        return f"ArrayNetwork({self.mesh!r}, {len(self.neurons_list)} neurons, {len(self.weight)} connections)"
