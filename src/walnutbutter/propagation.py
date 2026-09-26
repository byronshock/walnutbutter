"""Signalling on a schedule: a time-ordered queue of signals, processed a wave at a time.

AUTHORITY.md §4. A signal takes one hop (`Neuron.hop`, specified directly
at 2.55 ms) to travel a connection. Firing a neuron does
not call its neighbours: each active outgoing connection is scheduled to
deliver one hop later, and the schedule is a heap of events by time. A
**wave** is everything scheduled for one moment: the signals arriving, and
the stimulus if an input lands then. Two signals computed to arrive within
the clock's tolerance of each other are the same moment (clock.py), and an
input's exact time anchors the wave it joins.

A wave is processed in two phases:

1. deliver every signal for that time, adding weights to the potentials of
   the targets that are not refractory, then let each touched neuron settle
   (apply its floor to the wave's total);
2. fire every forced neuron that is not refractory, then every touched
   neuron whose potential now meets its threshold and is not refractory.

Because all deliveries finish before any firing decision is made, the result
cannot depend on the order in which neurons or connections are stored. A
neuron may fire as often as its refractory period allows; nothing else
limits it. Each connection stamps the time of the last signal its target
integrated (`Connection.last_signal`): the synapse's only trace.

The schedule outlives an epoch: signals due at or after the horizon the
network runs to wait for the next epoch, where they join the next input's
waves. `propagate()` runs a standalone cascade from a stimulus to
exhaustion, for small experiments and tests.

Under exploration at the synapse (AUTHORITY.md §7.5) a wave has a third
step, after the fire phase: every synapse of every neuron that did not spike
decides, and each escape is a **ventured** signal one hop later (§7.9) -- a
SIGNAL like any other, pushed after the wave's spikes in edge order (§3.6),
its mark riding in the payload and never in the kind or the sort key, so it
is summed in push order with the relayed ones (§3.7). The network supplies
that step (`Network.decider`), as it supplies the draws before the fire phase.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Callable, Iterable, Mapping

from .clock import before, slack
from .connection import Connection
from .constants import INTERVAL
from .neuron import Neuron

_wave_stamps = itertools.count(1)  # a fresh stamp per wave, across all schedules
EXTERNAL, STIMULUS, SIGNAL = 0, 1, 2  # event kinds, in the order they are taken within a wave


class Signal:
    """A record of one delivery: the connection, the wave it was delivered in, its target and amount, and whether it was
    ventured -- an escape of its synapse rather than a relay of its source's spike (AUTHORITY.md §7.9).

    Built on request by `Wave.signals()` for inspection and the reinforce
    rule; the schedule itself queues the connections, each with its mark.
    """

    __slots__ = ("connection", "wave", "target", "amount", "ventured")

    def __init__(self, connection: Connection, wave: int, ventured: bool = False):
        self.connection = connection
        self.wave = wave
        self.target = connection.target
        self.amount = connection.weight
        self.ventured = ventured

    def __repr__(self) -> str:
        mark = ", ventured" if self.ventured else ""
        return f"Signal({self.connection!r}, wave {self.wave}{mark})"


class Wave:
    """What happened at one moment: its time, the connections whose signals were delivered, and the neurons that fired.

    `ventured` holds the positions in `delivered` of the ventured signals
    (§7.9), so under the neuron rule, where nothing is ventured, it is empty.
    """

    __slots__ = ("number", "time", "delivered", "fired", "ventured")

    def __init__(
        self, number: int, time: float = 0.0, delivered: list[Connection] | None = None, fired: list[Neuron] | None = None,
        ventured: list[int] | None = None,
    ):
        self.number = number
        self.time = time
        self.delivered: list[Connection] = [] if delivered is None else delivered
        self.fired: list[Neuron] = [] if fired is None else fired
        self.ventured: list[int] = [] if ventured is None else ventured

    def signals(self) -> list[Signal]:
        """The deliveries of this wave as Signal records, each with its mark."""
        ventured = set(self.ventured)
        return [Signal(connection, self.number, k in ventured) for k, connection in enumerate(self.delivered)]

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Wave)
            and (self.number, self.time, self.delivered, self.fired, self.ventured)
            == (other.number, other.time, other.delivered, other.fired, other.ventured)
        )

    def __repr__(self) -> str:
        return f"Wave(number={self.number}, time={self.time:g}, delivered={self.delivered!r}, fired={self.fired!r})"


class Schedule:
    """The time-ordered queue of everything still to happen: signals in flight, stimuli and external inputs due."""

    def __init__(self):
        self._heap: list[tuple] = []  # (time, kind, sequence, payload): a signal's payload is (connection, ventured),
        # an external's (neuron, amount, steps) -- the amount given, or None for a charge of theta / steps (5.4b)
        self._sequence = itertools.count()

    def __len__(self) -> int:
        return len(self._heap)

    def signal(self, connection: Connection, time: float, ventured: bool = False) -> None:
        """A signal along `connection` arriving at `time`: relayed with its source's spike, or `ventured` by an escape of
        its synapse (§7.9). The mark is the payload's: the kind and the sort key are a signal's either way (§3.7)."""
        heapq.heappush(self._heap, (float(time), SIGNAL, next(self._sequence), (connection, bool(ventured))))

    def stimulus(self, neuron: Neuron, time: float) -> None:
        """`neuron` is forced to fire at `time`, refractory period permitting."""
        heapq.heappush(self._heap, (float(time), STIMULUS, next(self._sequence), neuron))

    def external(self, neuron: Neuron, amount: float, time: float) -> None:
        """An external input of `amount` delivered to `neuron` at `time`, which fires it only if it reaches threshold."""
        heapq.heappush(self._heap, (float(time), EXTERNAL, next(self._sequence), (neuron, amount, 0)))

    def charge(self, neuron: Neuron, steps: int, time: float) -> None:
        """A delivery of the charged drive (AUTHORITY.md 5.4b) to `neuron` at `time`: theta / steps, a division, on the
        threshold the neuron holds when it lands -- computed at delivery, not here. It is an external input: taken
        before the wave's signals (§3.5), anchoring the wave (§3.4), dropped at a refractory neuron, and setting the
        neuron's driven mark (5.8) whether it was taken or dropped."""
        heapq.heappush(self._heap, (float(time), EXTERNAL, next(self._sequence), (neuron, None, steps)))

    def next_time(self) -> float | None:
        return self._heap[0][0] if self._heap else None

    def pending(self, marks: bool = False) -> list[tuple]:
        """The signals in flight, by time then order: what a checkpoint records. (time, connection) pairs, or with
        `marks` (time, connection, ventured) triples (§7.9)."""
        signals = [(time, payload) for time, kind, _, payload in sorted(self._heap, key=lambda e: e[:3]) if kind == SIGNAL]
        if marks:
            return [(time, connection, ventured) for time, (connection, ventured) in signals]
        return [(time, connection) for time, (connection, _) in signals]

    def charges(self) -> list[tuple[float, Neuron, int]]:
        """The charged drive's deliveries still to come (5.4b), by time then order, as (time, neuron, steps): what the
        array engine takes over from a mesh it wraps."""
        return [(time, payload[0], payload[2]) for time, kind, _, payload in sorted(self._heap, key=lambda e: e[:3])
                if kind == EXTERNAL and payload[1] is None]

    def clear(self) -> None:
        self._heap.clear()

    def run(
        self,
        until: float = math.inf,
        waves: list[Wave] | None = None,
        on_wave: Callable[[Wave], None] | None = None,
        everyone: list[Neuron] | None = None,
        explore: Callable[[float], None] | None = None,
        synapses: Callable[[Wave], list[Connection]] | None = None,
    ) -> list[Wave]:
        """Process every wave due before `until`, appending to and returning `waves`.

        `on_wave` is called after each wave has fired, with the wave: this is
        where the local rules of §10 run. Waves are numbered on from the
        length of `waves`. With `everyone`, the network's neurons, every wave
        also fires any neuron that has become ready without being touched:
        one whose threshold has fallen with its silence (Neuron.threshold_at),
        or one recovered from a refractory period with enough potential.
        `explore` takes the wave's draws after the floor and before the fire
        phase (§3.8); `synapses`, under exploration at the synapse, makes the
        synapses' decisions after it and returns the escapes, in edge order,
        which are pushed as ventured signals one hop later (§7.5, §3.6).
        """
        waves = [] if waves is None else waves
        hop = Neuron.hop
        accumulating = Neuron.tau == math.inf  # the evidence accumulator (§5.1): nothing leaks, no decay is evaluated
        heap = self._heap
        while heap and before(heap[0][0], until):
            first = heap[0][0]
            limit = first + slack(first)
            batch = []
            time = None  # an input's exact time anchors the wave; otherwise the earliest signal's
            while heap and heap[0][0] <= limit:
                event = heapq.heappop(heap)
                batch.append(event)
                if event[1] != SIGNAL and time is None:
                    time = event[0]
            if time is None:
                time = first
            wave = Wave(len(waves), time)
            mark = next(_wave_stamps)
            touched: list[Neuron] = []
            forced: list[Neuron] = []
            for _, kind, _, payload in batch:  # the charged drive's deliveries first, in the order the wave holds them (§3.5)
                if kind == EXTERNAL and payload[1] is None:
                    neuron, _, steps = payload
                    amount = neuron.threshold / steps  # 5.4b: theta / steps, a division, on the threshold it holds now
                    neuron.forced = True  # 5.8's driven mark, set by any delivery, one dropped at a refractory input included
                    if neuron.receive(amount, time) and neuron.touched_stamp != mark:
                        neuron.touched_stamp = mark
                        touched.append(neuron)
            for _, kind, _, payload in batch:  # then the signals in push order (§3.7), and every other event as it falls
                if kind == SIGNAL:
                    connection, ventured = payload
                    if ventured:
                        wave.ventured.append(len(wave.delivered))
                    wave.delivered.append(connection)
                    target = connection.target
                    if target.receive(connection.weight, time):
                        connection.last_signal = time
                        # what this synapse now has in the potential (§6.7); under TRACE ventured a relayed arrival is,
                        # for the synapse's learning, not there: no count, no note, no decay, no moment moved (§8.17)
                        if target.traced and (ventured or not target.trace_ventured):
                            if accumulating:
                                connection.trace += 1.0  # the count of arrivals since the target's last spike (§5.1)
                                # the debit counts from here: the spikes expected so far, or under exploration at the
                                # synapse the gain as it stands before this wave's decisions post (§8.16)
                                connection.noted += target.gain if target.synaptic else target.expected
                            else:
                                connection.trace = connection.trace * math.exp(-(time - connection.trace_at) / Neuron.tau) + 1.0
                            connection.trace_at = time
                        if target.touched_stamp != mark:  # each neuron once per wave, without a set
                            target.touched_stamp = mark
                            touched.append(target)
                elif kind == STIMULUS:
                    forced.append(payload)
                elif payload[1] is not None:  # an external input of a given amount, taken in heap order as it always was:
                    # §3.5 puts the charged drive's deliveries first and nothing else
                    neuron, amount, _ = payload
                    if neuron.receive(amount, time) and neuron.touched_stamp != mark:
                        neuron.touched_stamp = mark
                        touched.append(neuron)
            for neuron in touched:
                neuron.settle()  # the floor applies to the wave's total, whatever order it arrived in
            if explore is not None:
                explore(time)  # §6.1: the exploration draw comes before the decision it is meant to explain
            for neuron in forced:
                if not neuron.refractory_at(time):  # a stimulus listed twice fires once: the first spike makes it refractory
                    self._fire(neuron, wave, hop)
                    neuron.forced = True
            for neuron in touched:
                if neuron.decide(time):  # the threshold, or the escape-noise draw (§5.2)
                    self._fire(neuron, wave, hop)
            if everyone is not None:
                for neuron in everyone:
                    if neuron.touched_stamp != mark and neuron.decide(time):  # the touched were checked above
                        self._fire(neuron, wave, hop)
            if synapses is not None:  # §7.5: the synapses decide after the spikes, and their escapes follow them (§3.6)
                for connection in synapses(wave):
                    self.signal(connection, time + hop, ventured=True)
            waves.append(wave)
            if on_wave is not None:
                on_wave(wave)
        return waves

    def _fire(self, neuron: Neuron, wave: Wave, hop: float) -> None:
        """Fire one neuron and schedule its outgoing signals one hop later."""
        for connection in neuron.fire(wave.number, wave.time):
            self.signal(connection, wave.time + hop)
        wave.fired.append(neuron)


def propagate(
    fire: Iterable[Neuron] = (),
    inputs: Mapping[Neuron, float] | None = None,
    now: float = 0.0,
    until: float | None = None,
) -> list[Wave]:
    """Run one cascade from a stimulus at clock time `now` and return its waves.

    It has no network, and so no stream and no synapses' decisions: a neuron
    set to explore at the synapse is refused here (§7.5, §12.2) and runs by
    its network's `propagate`.

    `fire` lists neurons forced to fire at `now` regardless of threshold (an
    external stimulus); a neuron still refractory at `now` ignores it.
    `inputs` maps neurons to external input amounts delivered at `now`,
    which fire the neuron only if it reaches threshold. Later waves follow
    one hop apart until nothing is left in flight or the clock reaches
    `until` (default: one INTERVAL after `now`). A bound is needed because
    activity can sustain itself: with a refractory period of a few hops, a
    loop brings a neuron's own spike back to refire it, forever.
    """
    fire, inputs = list(fire), dict(inputs or {})
    if any(neuron.synaptic for neuron in [*fire, *inputs]):
        raise ValueError("the module's propagate has no network, so no stream for the synapses' draws and no synapses' "
                         "decisions after the fire phase: under exploration at the synapse run the cascade by the "
                         "network's own propagate (§7.5, §12.2)")
    schedule = Schedule()
    for neuron in fire:
        schedule.stimulus(neuron, now)
    for neuron, amount in inputs.items():
        schedule.external(neuron, amount, now)
    return schedule.run(now + INTERVAL if until is None else until)
