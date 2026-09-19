from __future__ import annotations

import math

from .clock import slack
from .connection import Connection
from .constants import (BORED_AFTER, DECISION_MEMORY, HOP, MINIMUM_POTENTIAL, RATE_TAU, REFRACTORY,
                        TAU, THRESHOLD)


class Neuron:
    """A leaky integrate-and-fire neuron with an absolute refractory period, on a clock in nominal milliseconds.

    AUTHORITY.md §0 and §5: the neuron integrates delta functions (a signal
    adds its weight to the potential), the potential leaks with time
    constant `tau` (lazily: nothing happens to a quiet neuron, and when a
    signal arrives the potential is first decayed for the time since it was
    last brought up to date), and it fires when the potential reaches its
    threshold, and resets. A neuron that fired within the last `refractory`
    milliseconds ignores every signal, forced stimulus included. It
    remembers its last two spike times: the gap between them is what its
    dopamine release depends on (dopamine.py). `refractory` and `hop` are
    global properties of neurons, one value for the whole network; a signal
    takes `hop` milliseconds to travel a connection.
    """

    verbose = False  # class-wide: print a line each time any neuron fires (off unless asked: walnutbutter -v)
    tau = TAU  # leak time constant, nominal milliseconds (see constants.py); math.inf switches the leak off
    refractory = REFRACTORY  # absolute refractory period, nominal milliseconds (see constants.py)
    hop = HOP  # ms a signal takes to travel one connection, specified directly (see constants.py, §3.2)
    rate_tau = RATE_TAU  # ms: the exponential window the read estimates a firing rate over (see constants.py, §4.3)
    bored_after = BORED_AFTER  # ms of silence after which the threshold has fallen to zero (see constants.py); 0 = off

    def __init__(self, name: str = "Neuron", threshold: float = THRESHOLD, minimum_potential: float = MINIMUM_POTENTIAL):
        self.name = name
        self.outgoing: list[Connection] = []  # connections this neuron sends signals along
        self.incoming: list[Connection] = []  # connections that deliver signals to this neuron
        self.threshold = float(threshold)  # total weighted input needed to fire
        self.minimum_potential = float(minimum_potential)  # inhibition can push the potential no lower than this
        self.potential = 0.0  # weighted input received since the last spike
        self.delta = 0.0  # the width of this neuron's firing decision, in potential units (AUTHORITY.md §5.2, escape
        # noise): ESCAPE_DELTA times its starting threshold, set by Network.set_delta; 0 is the deterministic threshold
        self.exposed_since = 0.0  # clock time the hazard has run from: the previous decision, or the refractory period's end
        self.draw = 1.0  # this wave's uniform for the decision, set by the network's explorer hook; 1 never fires
        self.escape_scale = 1.0  # sqrt(ESCAPE_REFERENCE_COUNT / N): the network's count scales every hazard down as its
        # square root (AUTHORITY.md §5.2, September 16, 2026); Network.set_delta sets it, and a restore recomputes it
        self.traced = False  # keep the trace of what each incoming synapse has in the potential (§6.7): under escape
        # noise, or under the centred rule; Network.set_delta and Network.centre set it
        self.centred = False  # the hebb eligibility charges this neuron's decisions (§6.7, the single-spike rule)
        self.expectation: float | None = None  # p_hat_j: the neuron's per-decision expectation of its own spike, which
        # the hebb eligibility charges (§6.7); None until its first decision, which sets it
        self.decisions = 0  # decisions to date, for the expectation's warm start (DECISION_MEMORY)
        self.expected = 0.0  # E_j: the spikes expected of this neuron over its decisions since its last spike -- the
        # hazard's m or the centred rule's p_hat, summed -- so the accumulator's debit settles per arrival (§6.7)
        self.credit = 0.0  # what the decision that fired credits each open arrival with, for fire() to settle (§6.7)
        self.touched_stamp = 0  # last wave (a global stamp) in which a signal reached this neuron
        self.rate = 0.5  # running estimate of how often this neuron fires per epoch (the reinforce rule)
        self.has_fired = False  # fired in the current epoch
        self.fired_in_wave: int | None = None  # the wave of the current epoch it (last) fired in; None until it fires
        self.forced = False  # forced to fire by the stimulus in the current epoch
        self.rate_level = 0.0  # Hz: the exponential-window estimate of this neuron's firing rate, brought up to rate_at
        self.rate_at = 0.0  # when that estimate was last brought up to date; lazy, like the potential's leak
        self.should_fire: bool | None = None  # an input neuron's bit this epoch (True forced, False should not fire); None: not an input
        self.fired_at: float | None = None  # clock time of the last spike, across epochs
        self.previous_fired_at: float | None = None  # clock time of the spike before that
        self.last_update = 0.0  # clock time the potential was last brought up to date (the lazy leak)
        self.spikes = 0  # how many times this neuron has ever fired
        self.spikes_at_reset = 0  # and how many it had when the epoch began: the difference is the count read (§4.3)

    def connect(
        self, target: Neuron, connection_id: int = 0, weight: float = 1.0, kind: str = "local"
    ) -> Connection:
        """Create a one-way connection from this neuron to `target`.

        The Connection is stored in this neuron's outgoing list and the target's
        incoming list. Returns it so a registry can record it by ID.
        """
        connection = Connection(connection_id, self, target, weight, kind=kind)
        self.outgoing.append(connection)
        target.incoming.append(connection)
        return connection

    def connection_to(self, target: Neuron) -> Connection | None:
        """The outgoing connection to `target`, or None if there is none."""
        for connection in self.outgoing:
            if connection.target is target:
                return connection
        return None

    def targets(self) -> list[Neuron]:
        """The neurons this neuron can send a signal to."""
        return [connection.target for connection in self.outgoing]

    def sources(self) -> list[Neuron]:
        """The neurons that can send a signal to this neuron."""
        return [connection.source for connection in self.incoming]

    def refractory_at(self, now: float) -> bool:
        """True if the neuron fired within the refractory period before `now` (a neuron that fired *at* now included).

        The clock's slack applies: a spike returning the instant the period
        ends, give or take a rounding error, finds the neuron recovered.
        """
        return self.fired_at is not None and now + slack(now) < self.fired_at + Neuron.refractory

    def potential_at(self, now: float) -> float:
        """The potential as it stands at `now`, decayed for the time since it was last brought up to date; changes nothing."""
        elapsed = now - self.last_update
        if elapsed > 0.0 and Neuron.tau != math.inf:
            return self.potential * math.exp(-elapsed / Neuron.tau)
        return self.potential

    def firing_rate(self, now: float) -> float:
        """This neuron's firing rate at `now` in Hz, an exponential window of width `Neuron.rate_tau` (AUTHORITY.md §4.3).

        Each spike puts 1/rate_tau on the trace and it decays with the same
        constant, so a neuron that has never fired, or whose spikes are long
        past, reads zero. The trace is not reset between epochs: it is the
        neuron's own state, and at a 35 ms epoch a spike from the epoch before
        contributes 0.09% of one.
        """
        if not self.rate_level:
            return 0.0
        return self.rate_level * math.exp(-(now - self.rate_at) / Neuron.rate_tau)

    def leak(self, now: float) -> None:
        """Bring the potential up to `now`: decay it for the time since the last update. Lazy, so call it on arrival."""
        elapsed = now - self.last_update
        if elapsed > 0.0:
            if Neuron.tau != math.inf:
                self.potential *= math.exp(-elapsed / Neuron.tau)
            self.last_update = now

    def receive(self, amount: float, now: float | None = None) -> bool:
        """Take in weighted input at time `now`: leak first, then integrate, unless refractory. Returns whether it was integrated.

        Receiving never fires the neuron by itself; the schedule checks
        readiness once every signal in the wave has been delivered, after
        `settle()` has applied the floor to the wave's total. Negative weights
        push the potential down (an inhibitory connection). Without `now` the
        clock is not consulted: no leak, and nothing blocks.
        """
        if now is not None:
            if self.refractory_at(now):
                return False
            self.leak(now)
        self.potential += amount
        return True

    def settle(self) -> None:
        """Apply the floor: inhibition saturates at `minimum_potential`.

        Called once per wave on every neuron that received a signal, so the
        floor acts on the wave's summed input and the result does not depend
        on the order the signals arrived in.
        """
        if self.potential < self.minimum_potential:
            self.potential = self.minimum_potential
            if self.traced:
                self.clear_arrivals()  # at the floor the potential is the floor whatever the weights (§6.7)

    def settle_arrivals(self, credit: float = 0.0) -> None:
        """Under the evidence accumulator (§5.1), close every open arrival on this neuron's incoming synapses (§6.7).

        Each synapse's score takes the credit for its count and the debit
        accrued since each arrival -- the spikes expected of this neuron from
        then until now -- and its count and its note are cleared. A spike
        settles with its decision's credit; the floor, a forced spike and a
        discharge with none.
        """
        expected = self.expected
        for connection in self.incoming:
            if connection.trace != 0.0:
                connection.score += credit * connection.trace - (connection.trace * expected - connection.noted)
                connection.trace = 0.0
                connection.noted = 0.0

    def clear_arrivals(self) -> None:
        """Nothing any synapse delivered is in the potential: the floor bit, or it was discharged (§6.7).

        Under the evidence accumulator the open arrivals are closed without
        credit, their debit so far settled; under the leak the traces are zeroed.
        """
        if Neuron.tau == math.inf:
            self.settle_arrivals(0.0)
        else:
            for connection in self.incoming:
                connection.trace = 0.0

    @property
    def ready(self) -> bool:
        """True if this neuron has enough input to fire, against its resting threshold."""
        return self.potential >= self.threshold

    def threshold_at(self, now: float) -> float:
        """The threshold a bored neuron faces at `now`: its own, falling linearly with the silence since its last spike.

        Threshold homeostasis (AUTHORITY.md §5.4): the threshold reaches zero
        after `bored_after` ms without a spike and keeps falling at the same
        rate, so a neuron that nobody talks to fires on its own, and resets
        its threshold by firing. A neuron that has never fired has been
        silent since the clock started.
        """
        if Neuron.bored_after <= 0.0:
            return self.threshold
        since = 0.0 if self.fired_at is None else self.fired_at
        return self.threshold - self.threshold * (now - since) / Neuron.bored_after

    def can_fire(self, now: float | None) -> bool:
        """Enough potential for the threshold it faces at `now`, and not refractory (the clock is ignored when `now` is None)."""
        if now is None:
            return self.ready
        if self.refractory_at(now):
            return False
        return self.potential_at(now) >= self.threshold_at(now)

    def expected_spikes(self, now: float) -> float:
        """m_j(now): the spikes the hazard expects of this neuron since its exposure began (AUTHORITY.md §5.2).

        (dt / hop) * sqrt(N0 / N) * exp(s / delta), with s the margin the
        decision is made on, dt the time since the previous decision or the
        refractory period's end, and sqrt(N0 / N) the count's scaling of the
        hazard (`escape_scale`, §5.2); capped at 1e3, beyond which the chance
        of a spike is already 1 to the last bit. Needs delta > 0.
        """
        s = self.potential_at(now) - self.threshold_at(now)
        elapsed = now - self.exposed_since
        if elapsed < 0.0:
            elapsed = 0.0
        return min(elapsed / Neuron.hop * self.escape_scale * math.exp(s / self.delta), 1e3)

    def decide(self, now: float) -> bool:
        """The firing decision at `now`: can_fire when delta is 0, else the escape-noise draw (AUTHORITY.md §5.2).

        The decision also charges the eligibility of every incoming synapse
        (§6.7, the single-spike rule): each score moves by (c - q) * trace, the
        credit c and the expectation q of the decision. Under the hazard
        eligibility c = m e^-m / (1 - e^-m) and q = 0 when the neuron fires,
        c = 0 and q = m when it does not; under the centred rule (hebb) c is
        the outcome, 1 or 0, and q the neuron's own expectation of it, moved
        after the charge. Under the evidence accumulator (§5.1) the charge is
        lazy: q is added to the neuron's expected count and a spike's credit is
        left for fire() to settle per arrival. The schedule calls it exactly
        once per neuron per wave.
        """
        if self.refractory_at(now):
            return False
        if self.delta <= 0.0:
            fired = self.can_fire(now)
            m = 0.0
        else:
            m = self.expected_spikes(now)
            fired = self.draw < -math.expm1(-m)
            self.exposed_since = now
        if self.centred:
            y = 1.0 if fired else 0.0
            p = self.expectation
            self.decisions += 1
            if p is None:
                self.expectation = y  # the first decision sets the expectation and charges nothing
                return fired
            credit, q = y, p
            self.expectation = p + max(DECISION_MEMORY, 1.0 / self.decisions) * (y - p)  # moved after the charge
        elif m > 0.0:
            if fired:
                credit, q = m * math.exp(-m) / -math.expm1(-m), 0.0  # m e^-m / (1 - e^-m): finite at any m
            else:
                credit, q = 0.0, m
        else:
            return fired
        tau = Neuron.tau
        if tau == math.inf:  # the evidence accumulator (§5.1): the debit settles per arrival, the credit at the spike
            self.expected += q
            if fired:
                self.credit = credit
        else:
            e = credit - q
            for connection in self.incoming:
                if connection.trace != 0.0:
                    connection.score += e * connection.trace * math.exp(-(now - connection.trace_at) / tau)
        return fired

    def fire(self, wave: int = 0, now: float | None = None) -> list[Connection]:
        """Spike: mark this neuron as fired in `wave` at time `now` and return the connections to signal along.

        This does not deliver anything: the caller (see propagation.py)
        schedules the returned connections one hop later, so that all of a
        wave's signals are delivered before any neuron in the next wave
        decides whether to fire.
        """
        self.has_fired = True
        self.fired_in_wave = wave
        self.potential = 0.0  # the spike resets the potential
        self.spikes += 1
        if now is not None:
            self.previous_fired_at = self.fired_at
            self.fired_at = now
            self.last_update = now
            self.rate_level = self.firing_rate(now) + 1000.0 / Neuron.rate_tau  # Hz: one spike's worth (§4.3)
            self.rate_at = now
            self.exposed_since = now + Neuron.refractory  # the hazard resumes when the refractory period ends (§5.2)
        if self.traced:
            if Neuron.tau == math.inf:
                self.settle_arrivals(self.credit)  # the evidence accumulator: the spike credits and closes every open arrival (§6.7)
            else:
                for connection in self.incoming:
                    connection.trace = 0.0  # the spike reset the potential: nothing any synapse delivered is still in it (§6.7)
            self.expected = 0.0
            self.credit = 0.0
        if Neuron.verbose:
            print(f"{self.name} fired in wave {wave}.")
        return [connection for connection in self.outgoing if connection.is_active]

    def reset(self, discharge: bool = False) -> None:
        """Start a new epoch: clear the fired-this-epoch state.

        The potential is kept: a neuron that did not fire keeps its
        sub-threshold charge, which leaks as the clock moves on, and a spike
        has already reset the potential of one that fired. With
        `discharge=True` every potential is zeroed instead. Spike times are
        kept: the refractory period outlives the epoch.
        """
        if discharge:
            self.potential = 0.0
            if self.traced:
                self.clear_arrivals()  # nothing is left in a zeroed potential (§6.7)
        self.has_fired = False
        self.fired_in_wave = None
        self.forced = False
        self.spikes_at_reset = self.spikes

    @property
    def epoch_spikes(self) -> int:
        """How many times this neuron has fired since the epoch began: what the count read thresholds (§4.3)."""
        return self.spikes - self.spikes_at_reset

    def list_connections(self) -> None:
        print(f"I am {self.name}, and I send signals to:")
        for connection in self.outgoing:
            state = "" if connection.is_active else " (inactive)"
            print(f"    #{connection.id} {connection.target.name}, weight {connection.weight:g}{state}")
