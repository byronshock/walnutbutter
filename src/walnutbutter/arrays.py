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

    def __init__(self, mesh):
        self.mesh = mesh
        self.across, self.rows = mesh.across, mesh.rows
        self._init_network(mesh.across, mesh.weight_range)
        self.permutation = list(mesh.permutation)
        self.ecc = mesh.ecc
        self.epoch = mesh.epoch
        self.time, self.interval, self.input_time = mesh.time, mesh.interval, mesh.input_time
        self.horizon = mesh.horizon
        self.readout, self.read, self.read_window, self.coding = mesh.readout, mesh.read, mesh.read_window, mesh.coding
        self.population, self.quash_rate, self.quash_k = mesh.population, mesh.quash_rate, mesh.quash_k
        self.output_coding = getattr(mesh, "output_coding", "population")  # how the output zone codes the classes (§8)
        self.temperature = mesh.temperature  # the evidence critic's temperature (§8)
        self.clock = mesh.clock  # clock neurons at the front of the input zone (§4.3)
        self.drive, self.input_rate, self.input_rate_off = mesh.drive, mesh.input_rate, mesh.input_rate_off
        self.explore_rng = mesh.explore_rng
        self.rate_on = mesh.rate_on
        self.pickiness = mesh.pickiness  # spikes: the count read's line (§5.10, §9.5)
        self.flip = mesh.flip
        self.rule = mesh.rule
        self.seed = mesh.seed
        self.threshold = mesh.threshold
        self.minimum_potential = mesh.minimum_potential
        self._rng = random.Random()
        self._rng.setstate(mesh._rng.getstate())  # the same input sequence as the mesh would draw
        self.input_stream, self.input_at = mesh.input_stream, mesh.input_at  # and the same attached stream (§4.5)
        self.input_labels, self.input_label = mesh.input_labels, mesh.input_label  # with its labels, when a dataset gave them (§8)
        for name in ("input_pattern", "target_pattern", "input_bits", "input_coded", "input_data", "input_events"):
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
        self.waves: list[ArrayWave] = []
        # the schedule: distinct times, each with the sources whose signals arrive then and the neurons forced then
        self._times: list[float] = []
        self._arrivals: dict[float, np.ndarray] = {}
        self._stimuli: dict[float, list[int]] = {}  # indices, not a mask: rate drive lands one arrival per moment
        for time, connection in mesh.schedule.pending():
            self._arrive(self.index[connection.source], time)

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
        if time not in self._arrivals and time not in self._stimuli:
            heapq.heappush(self._times, time)
        return time

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
        """Signals in flight as (time, source index), by time: what sync_to_mesh puts back on the mesh's schedule."""
        return [(time, int(i)) for time in sorted(self._arrivals) for i in np.flatnonzero(self._arrivals[time] > 0)]

    def _run(self, until: float = math.inf) -> list[ArrayWave]:
        """Process every wave due before `until` (see propagation.Schedule.run), appending to this epoch's waves."""
        potential, threshold, floor, fired_wave = self.potential, self.threshold_v, self.floor, self.fired_wave
        n = len(potential)
        hop = Neuron.hop()
        out_degree = self._out_degree
        while self._times and before(self._times[0], until):
            self._refresh_matrix()
            first = heapq.heappop(self._times)
            limit = first + slack(first)
            firing = self._arrivals.pop(first, None)
            stimulus = self._stimuli.pop(first, None)
            time = first
            while self._times and self._times[0] <= limit:  # the same moment, within the clock's slack
                other = heapq.heappop(self._times)
                more = self._arrivals.pop(other, None)
                if more is not None:
                    firing = more if firing is None else firing + more
                forced_too = self._stimuli.pop(other, None)
                if forced_too is not None:
                    stimulus = forced_too if stimulus is None else stimulus + forced_too
                    time = other  # an input's exact time anchors the wave
            number = len(self.waves)
            refractory = self.fired_at + Neuron.refractory > time + slack(time)  # fired within the refractory period before now
            take = np.zeros(n, dtype=bool)
            if firing is not None:
                both = self._matrix @ firing
                incoming, touched = both[:n], both[n:] > 0
                take = touched & ~refractory
                self.leak(time, take)  # the lazy leak: a neuron is brought up to date when a signal reaches it
                summed = potential[take] + incoming[take]
                potential[take] = np.maximum(summed, floor[take])
                delivering = self.active & (firing[self.source] > 0)
                self.delivered_wave[delivering] = number
                integrated = delivering & take[self.target]
                self.last_signal[integrated] = time
                if self.traced:  # what each synapse now has in its target's potential (§6.7)
                    if Neuron.tau == math.inf:  # the evidence accumulator (§5.1): a count, no decay evaluated
                        self.trace[integrated] += 1.0
                        self.noted[integrated] += self.expected[self.target[integrated]]  # the debit counts from here
                    else:
                        self.trace[integrated] = self.trace[integrated] * np.exp(-(time - self.trace_at[integrated]) / Neuron.tau) + 1.0
                    self.trace_at[integrated] = time
                    clipped = np.zeros(n, dtype=bool)
                    clipped[np.flatnonzero(take)[summed < floor[take]]] = True  # the floor bit: the weights are not in it
                    self._close_arrivals(clipped[self.target])
            if self.hazard:
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
            if self.hazard:  # the decision is a draw (§5.2)
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
            if Neuron.isi_factor and charged.any():  # §0.2: each charge weighed by the time since the neuron's last spike
                w = self._isi_factor(time)
                credit_v = w * credit_v
                q = w * q
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
                self.exposed_since[idx] = time + Neuron.refractory  # the hazard resumes after the refractory period (§5.2)
                if self.traced:
                    self._close_arrivals(fired[self.target], credit_v)  # the potential reset: the spike settles every arrival (§6.7)
                    self.expected[idx] = 0.0
                fired_wave[idx] = number
                self.spikes[idx] += 1
                if fire_forced is not None:
                    self.forced |= fire_forced
                if out_degree @ fired > 0:  # someone has active connections out: signals in flight
                    self._arrive_all(fired.astype(float), time + hop)
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

    def _isi_factor(self, time: float) -> np.ndarray:
        """neuron.isi_factor per neuron (AUTHORITY.md §0.2), in the same order of operations; 0 where a neuron never fired."""
        fired = self.fired_at > -np.inf
        x = np.where(fired, (time - np.where(fired, self.fired_at, 0.0)) / Neuron.target_isi, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            f = (3.0 * x - 1.0) / (1.0 + x * x * x)
        return np.where(fired, f, 0.0)

    def settle_scores(self) -> None:
        """Network.settle_scores as vectors: under the evidence accumulator, every open arrival's debit into its score (§6.7)."""
        if Neuron.tau != math.inf or not self.traced:
            return
        open_ = self.trace != 0.0
        if open_.any():
            e = self.trace[open_] * self.expected[self.target[open_]]
            self.score[open_] -= e - self.noted[open_]
            self.noted[open_] = e

    def centre(self, on: bool) -> None:
        """Network.centre: the hebb eligibility charges every decision (§6.7); the mesh's neurons follow, for the checkpoint."""
        self.centred = bool(on)
        for neuron in self.neurons_list:
            neuron.centred = self.centred
            neuron.traced = self.centred or neuron.delta > 0.0

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
        if discharge:
            self.potential[:] = 0.0
            if self.traced:
                self._close_arrivals(np.ones(len(self.trace), dtype=bool))  # nothing is left in a zeroed potential (§6.7)
        if self.traced:
            self.score[:] = 0.0  # the score is the epoch's (§6.7)
            if Neuron.tau == math.inf:
                self.noted = self.trace * self.expected[self.target]  # an open arrival's debit counts from here
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
        self.time = self.input_time if self.input_time is not None else self.next_time()
        self.epoch += 1
        self.input_events = self.input_schedule()  # the same draws as the mesh, from the same stream (§4.3)
        for place, when in self.input_events:
            self._stimulate_at((int(self.input_index[place]),), when)
        self.horizon = self.time + self.interval if until is None else float(until)
        return self._run(self.horizon)

    def propagate(self, fire=(), inputs=None, now: float | None = None, until: float | None = None) -> list[ArrayWave]:
        """Run a cascade from the mesh neurons `fire`, forced at `now` (default: the clock), up to `until` (default: one interval)."""
        if inputs:
            raise NotImplementedError("external input amounts are not vectorised; use the object engine")
        now = self.time if now is None else now
        forced = np.zeros(len(self.neurons_list), dtype=bool)
        forced[[self.index[neuron] for neuron in fire]] = True
        self._stimulate(forced, now)
        self.horizon = now + self.interval if until is None else float(until)
        return self._run(self.horizon)

    def output_width(self) -> int:
        return len(self.output_index)

    def output_counts(self) -> list[int]:
        """Each output neuron's spikes this epoch, as Network.output_counts."""
        return (self.spikes[self.output_index] - self.spikes_at_reset[self.output_index]).tolist()

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
        return ((self.spikes[idx] - self.spikes_at_reset[idx]) * (1000.0 / self.interval)).tolist()

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
        if not self.hazard:  # §8.3: no eligibility runs where the threshold decides
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
        return int(unforced.sum())

    def unstick(self, rate: float, target: float) -> list[int]:
        """learning.unstick as one vector operation: every stuck neuron not forced this epoch."""
        if rate <= 0:
            return []
        stuck = np.flatnonzero(((self.rate > STUCK_ABOVE) | (self.rate < STUCK_BELOW)) & ~self.forced)
        self.threshold_v[stuck] = self.threshold_v[stuck] + rate * (self.rate[stuck] - target)
        return stuck.tolist()

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
        mesh.rule = self.rule
        mesh.waves = [Wave(w.number, w.time, [], [self.neurons_list[i] for i in w.fired.tolist()]) for w in self.waves]
        mesh.schedule.clear()
        for time, i in self.pending():
            for connection in self.neurons_list[i].outgoing:
                if connection.is_active:
                    mesh.schedule.signal(connection, time)
        for name in ("input_pattern", "target_pattern", "input_bits", "input_coded", "input_data", "input_events"):
            setattr(mesh, name, getattr(self, name))
        mesh._rng.setstate(self._rng.getstate())
        mesh.input_stream, mesh.input_at = self.input_stream, self.input_at
        mesh.input_labels, mesh.input_label = self.input_labels, self.input_label

    def __repr__(self) -> str:
        return f"ArrayNetwork({self.mesh!r}, {len(self.neurons_list)} neurons, {len(self.weight)} connections)"
