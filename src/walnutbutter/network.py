"""What a container of neurons provides: an input zone, an output zone, epochs and the schedule.

A network addresses its two zones by place (`get_neuron_at(place, 1)` is the
input zone and `get_neuron_at(place, 0)` the output zone, AUTHORITY.md §4.3):
the input zone receives the complement-coded and permuted input pattern, the
output zone is read as the output, and the epoch machinery resets, presents an
input at a time and runs the schedule to the next input's time (§4.2). Goo is
the only container (§4.1) and builds on this.
"""

from __future__ import annotations

import math
import numbers

import random
from typing import Iterable

from .constants import (
    DRIVE_STEPS, ESCAPE_REFERENCE_COUNT, INPUT_DRIVE, INPUT_RATE, INPUT_RATE_OFF, INTERVAL, POPULATION, PRESENTATION_TIME,
    SYNAPSE_HAZARD_FAMILY, SYNAPSE_HAZARD_REST, SYNAPSE_HAZARD_SCALING, TEMPERATURE, TRACE,
    QUASH_K, QUASH_RATE, RATE_ON, ROW_CRITIC_PICKINESS_IN_SPIKES, THRESHOLD_FAN_IN,
)

EXPLORATIONS = ("neuron", "synapse")  # what explores (AUTHORITY.md §7.1): the neuron's escape noise, or its synapses'
SYNAPSE_HAZARD_FAMILIES = ("loglinear", "linear")  # the synapse hazard's two families (§7.6), both carried
SYNAPSE_HAZARD_SCALINGS = ("count", "fan-out")  # kappa(N) on every synapse, or that over the source's fan-out (§7.7)
TRACES = ("all", "ventured")  # what a trace counts under exploration at the synapse (§8.17)
DRIVES = ("rate", "forced", "charged")  # how a bit becomes spikes: the Poisson forced drive of §5.4 (the code's "rate"),
# one spike at the epoch's moment (the code's "forced", which §5.7 does not carry and the plumbing tests pin), and the
# charged drive of 5.4b; any other name is refused rather than taken for the second (§5.7)
TRACE_VENTURED_BIAS = ("TRACE ventured: each trace counts the ventured arrivals alone, so it is not the potential's "
                       "derivative by the weight and the estimator is biased toward what was ventured (§8.17)")
# what a run under TRACE ventured says in its record -- its banner, its checkpoints, a sweep arm's json -- as §8.17 asks


def refuse_width(delta) -> None:
    """A decision width that is negative or not a number, refused under either exploration (§12.2): it is neither the
    comparison of §6.9 nor a draw, and would silence every neuron. Checked where the width is set (set_delta) and where
    a run names it on the command line, where under exploration at the synapse it would otherwise be settled to 0."""
    if not delta >= 0.0:  # nan fails this as a negative width does
        raise ValueError(f"ESCAPE_DELTA must not be negative, and must be a number, got {delta}")


def refuse_synapse_settings(h0, family, scaling, trace) -> None:
    """The settings of §7.6, §7.7 and §8.17, refused (§12.2) where they are not the file's: checked where they are set,
    where a run names them on the command line, and again where the network runs, since they stay writable attributes
    and h0 and the family are read at every wave -- an unknown family would otherwise run as the loglinear."""
    if isinstance(h0, bool) or not isinstance(h0, (int, float)) or not 0.0 <= h0 < 1.0:
        raise ValueError(f"the rest hazard h0 must lie in [0, 1), got {h0!r}: at 1 the hazard is flat and scores "
                         "nothing, above 1 the family turns over, below 0 it is undefined (§7.6)")
    if family not in SYNAPSE_HAZARD_FAMILIES:
        raise ValueError(f"unknown synapse hazard family {family!r}; choose from {', '.join(SYNAPSE_HAZARD_FAMILIES)} (§7.6)")
    if scaling not in SYNAPSE_HAZARD_SCALINGS:
        raise ValueError(f"unknown synapse hazard scaling {scaling!r}; choose from {', '.join(SYNAPSE_HAZARD_SCALINGS)} (§7.7)")
    if trace not in TRACES:
        raise ValueError(f"unknown trace {trace!r}; choose from {', '.join(TRACES)} (§8.17)")


def escape_scale(count: int) -> float:
    """sqrt(ESCAPE_REFERENCE_COUNT / count): what a network of `count` neurons multiplies every hazard by (AUTHORITY.md §5.2).

    Byron, September 16, 2026: "scaling the network MUST reduce the probability
    of escape noise at each neuron by sqrt(N)." The reference count is the goo
    of 60 the width was set on, so at 60 the factor is exactly 1 and nothing
    measured there moves; a network of 445 runs its hazards at 0.37 of what
    the width alone gives. A count of 0 has nothing to scale and gets 1.
    """
    import math
    return math.sqrt(ESCAPE_REFERENCE_COUNT / count) if count > 0 else 1.0
from .local import quash
from .exploration import hazard_draws
from .inputs import complement_code
from .neuron import Neuron
from .clock import before, slack
from .propagation import Schedule, Wave


def input_stream(count: int, raw_bits: int, seed: int) -> list[list[bool]]:
    """`count` epochs' worth of raw input bits, from a stream of their own (AUTHORITY.md §4.5).

    Drawn from `random.Random(f"walnutbutter inputs {seed}")`, which is not the
    network's stream and is touched by nothing else, so the same seed gives the
    same inputs whatever the network's size, reach, omega, weights or rule. That
    is what makes two arms of a sweep comparable epoch by epoch rather than only
    on average; drawing from the network's own stream, as every run before
    September 14, 2026 did, gave different arms different inputs at the same
    seed, because building a different network consumes the stream differently.
    """
    rng = random.Random(f"walnutbutter inputs {seed}")
    return [[rng.random() < 0.5 for _ in range(raw_bits)] for _ in range(count)]


class Network:
    """Mixin with the input/epoch machinery. Subclasses provide across (the count per row), rows, get_neuron_at, neurons and _rng."""

    across: int
    rows: int
    weight_range: tuple[float, float]

    def _init_network(self, across: int, weight_range: tuple[float, float]) -> None:
        low, high = weight_range
        if not low < high:
            raise ValueError(f"weight range must run from low to high, got {weight_range}")
        self.weight_range = (float(low), float(high))
        self.waves: list[Wave] = []  # Waves of the most recent propagation
        self.input_pattern: list[bool] | None = None  # one bit per place in the input zone: what is forced
        self.target_pattern: list[bool] | None = None  # the clean pattern the read is scored against, before any flips (§4.3)
        self.input_bits: list[bool] | None = None  # the raw bits before complement coding
        self.input_coded: list[bool] | None = None  # the complement-coded bits, which are the pattern itself (§5.2)
        self.input_stream: list[list[bool]] | None = None  # raw-bit patterns to present in order, in place of fresh
        # draws (§4.5); None draws from the network's own stream, as it always did
        self.input_at = 0  # how far through that stream the run has got
        self.input_labels: list[int] | None = None  # the label of each pattern of the stream, when a dataset gave one (§8, mnist)
        self.input_label: int | None = None  # this epoch's label, when the stream carries labels
        self.epoch = 0  # how many inputs have been presented
        self.time = 0.0  # the clock, nominal milliseconds: the time of the last input
        self.interval = INTERVAL  # default spacing of inputs when no time is given
        self.presentation = PRESENTATION_TIME  # §5.4a: ms into the epoch the drive runs; None is the whole epoch
        self.input_time: float | None = None  # when the pending input arrives
        self.horizon = 0.0  # the time the schedule has run to: the next input may not come before it
        self.schedule = Schedule()  # signals in flight, across epochs
        self.quash_rate = 0.0  # a refire weakens its contributing synapses by this fraction of their weight (§6.11); off until asked
        self.quash_k = QUASH_K  # per ms: the quash falls off with the delay since the previous spike
        self.explore_rng = None  # the exploration stream the firing decisions draw from (§7.3)
        self.escape_delta = 0.0  # ESCAPE_DELTA as set on this network (§5.2): 0 keeps the deterministic threshold
        self.escape_scale = 1.0  # sqrt(ESCAPE_REFERENCE_COUNT / N), the count's scaling of every hazard (§5.2), once set_delta ran
        self._exploration = "neuron"  # what explores (§7.1): a network is built under the neuron rule, deterministic
        # until set_delta gives it a width, and explores at the synapse once set_exploration asks (see `exploration`)
        self.synapse_hazard_rest = SYNAPSE_HAZARD_REST  # h0 (§7.6), under exploration at the synapse
        self.synapse_hazard_family = SYNAPSE_HAZARD_FAMILY  # loglinear or linear (§7.6)
        self.synapse_hazard_scaling = SYNAPSE_HAZARD_SCALING  # count or fan-out (§7.7)
        self.trace_mode = TRACE  # what a trace counts (§8.17): all, or ventured
        self._computed_from = (SYNAPSE_HAZARD_SCALING, TRACE)  # the scaling and the trace set_exploration last computed
        # kappa_i and every neuron's trace_ventured from: a run refuses the attributes where they no longer agree
        self._draws: list[float] = []  # this wave's uniforms for the synapses' decisions, E then O (§3.8)
        self._layout: list[tuple[Neuron, int]] = []  # each neuron, in index order, with its read synapse's draw or -1
        self._draw_count = 0  # E + O: every synapse, then every output's read synapse (§3.8, §7.9)
        self.rule = "local"  # which rule pays at the read: "reinforce", or "local" for none, in which case the local
        # rules of §10 are the whole of the learning (§9.1: "a run may have none")
        self.centred = False  # the hebb eligibility charges every neuron's decisions against its own expectation (§6.7,
        # the single-spike rule). A Teacher with that eligibility switches it on, in whichever engine (centre)
        self.readout = "top"  # what is read as the output: the output zone, or "input" (the inputs are the outputs)
        # (as they are), "population" (each bit repeated `population` times) or "population-complement" (both, in that
        # order: repeated, then the whole run followed by its negation)
        self.population = POPULATION  # neurons per raw bit under population coding
        # -- fire-if-one populations then, in the same order, fire-if-zero ones (Byron, September 16, 2026; learning.OUTPUT_CODINGS)
        self.temperature = TEMPERATURE  # the evidence critic's temperature: the class sums as log-odds at this scale (§8)
        self.clock = 0  # clock neurons (§4.3, Byron, September 15, 2026): this many input neurons at the front of the input
        # zone whose bit is always 1, so the drive fires them every epoch whatever the pattern; they take no raw bits
        self.drive = INPUT_DRIVE  # how a bit becomes spikes (§4.3): "forced", one spike at the epoch's moment, or "rate"
        # -- or "charged" (5.4b), theta / drive_steps a delivery, under exploration at the synapse only (DRIVES)
        self.drive_steps = DRIVE_STEPS  # under the charged drive, the deliveries from rest to threshold (5.4b)
        self.input_rate = INPUT_RATE  # per ms: what a bit-1 neuron fires at under rate drive
        self.input_rate_off = INPUT_RATE_OFF  # per ms: what a bit-0 neuron fires at
        self.input_events: list[tuple[int, float]] | None = None  # the (place, time) stimuli this epoch actually used;
        # input_schedule() draws afresh under rate drive, so this is what to read to see what happened
        self.read = "count"  # §5.10's default read: the epoch's spikes counted. "fired", "again", "window" and "rate" are its non-defaults
        # (a forced neuron must have refired); "window", within read_window ms before the horizon
        self.read_window: float | None = None  # the window for read == "window"
        self.rate_on = RATE_ON  # Hz: the rate a target-on output is driven to, and what output_levels divides by (§4.3)
        self.pickiness = ROW_CRITIC_PICKINESS_IN_SPIKES  # spikes: the count read's line between off and on (§5.10, §9.5)

    def all_neurons(self) -> Iterable[Neuron]:
        """Every neuron, in a stable order. Subclasses override if `neurons` is not a list."""
        return self.neurons

    def get_neuron_at(self, place: int, row: int) -> Neuron | None:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def hazard(self) -> bool:
        """True when the firing decision is a draw (AUTHORITY.md §5.2): set_delta gave this network a positive width."""
        return self.escape_delta > 0.0

    @property
    def exploration(self) -> str:
        """What explores (AUTHORITY.md §7.1): "neuron", the neuron's own escape noise of §6.5 -- deterministic where no
        width was given -- or "synapse", its synapses' (§7.5). Set by set_exploration."""
        return self._exploration

    @property
    def explores(self) -> bool:
        """True when something draws: a positive width (§6.5), or exploration at the synapse whatever its rest hazard,
        h0 = 0 included (§7.1, §7.6). What the reinforce rule needs to have something to estimate from (§8.3)."""
        return self.hazard or self._exploration == "synapse"

    @property
    def traced(self) -> bool:
        """Whether each synapse keeps its trace of what it has in its target's potential (§6.7): escape noise, the centred
        rule, or exploration at the synapse."""
        return self.hazard or self.centred or self._exploration == "synapse"

    def centre(self, on: bool) -> None:
        """The hebb eligibility (§6.7, the single-spike rule): every neuron charges its decisions against its own expectation."""
        if on and self._exploration == "synapse":
            raise ValueError("hebb is refused under exploration at the synapse: the decisions are the synapses', and it has "
                             "no neuron decision to centre (§8.3)")
        self.centred = bool(on)
        for neuron in self._everyone():
            neuron.centred = self.centred
            neuron.traced = self.centred or neuron.delta > 0.0 or neuron.synaptic

    def settle_scores(self) -> None:
        """Under the evidence accumulator (§5.1), bring every open arrival's debit into its synapse's score (§6.7).

        The pay at the read calls it first, so the score holds what the epoch's
        decisions charged; the arrivals stay open, their debit counting from
        now. Under the leak the scores are charged per decision and there is
        nothing to settle. Under exploration at the synapse the read posts
        x G - B, the net credit, and re-bases B = x G, leaving x and G as they
        are (§8.16, §1.9).
        """
        if Neuron.tau != math.inf or not self.traced:
            return
        if self._exploration == "synapse":
            for connection in self.connections.values():
                if connection.trace != 0.0:
                    owed = connection.trace * connection.target.gain
                    connection.score += owed - connection.noted
                    connection.noted = owed
            return
        for connection in self.connections.values():
            if connection.trace != 0.0:
                expected = connection.target.expected
                connection.score -= connection.trace * expected - connection.noted
                connection.noted = connection.trace * expected

    def set_delta(self, delta: float) -> None:
        """Escape noise (AUTHORITY.md §5.2): every neuron's decision is `delta` times its starting threshold wide; 0 turns it off.

        Call it once the thresholds are what the container gave them -- after
        fan-in scaling -- and before anything moves them: a neuron's width is
        set from the threshold it starts at and stays put when homeostasis
        later moves the threshold. The draws come from the exploration stream
        (§6.1), so a run with a positive width needs one. A positive width is
        refused while the network explores at the synapse (§6.13); a width of
        0 is the one §6.13 gives every neuron there, and asks for nothing. A
        width that is not a number is refused under either rule: it is neither
        the comparison of §6.9 nor a draw, and would silence every neuron.
        """
        refuse_width(delta)
        if delta != 0.0 and self._exploration == "synapse":
            raise ValueError(f"a neuron width ({delta:g}) and exploration at the synapse together are refused: the system "
                             "explores by one thing, and under exploration at the synapse every width is 0 (§6.13, §7.1)")
        self.escape_delta = float(delta)
        everyone = self._everyone()
        self.escape_scale = escape_scale(len(everyone))  # the count's scaling of every hazard (§5.2, September 16, 2026)
        for neuron in everyone:
            # a neuron whose starting threshold is not positive -- no incoming synapses under fan-in scaling (§5.2) --
            # has a collapsed axis with no width to quote on it, and keeps the deterministic rule
            neuron.delta = delta * neuron.threshold if neuron.threshold > 0.0 else 0.0
            neuron.escape_scale = self.escape_scale
            neuron.traced = neuron.delta > 0.0 or neuron.centred or neuron.synaptic  # the trace of §6.7 is kept under escape noise

    def set_exploration(self, mode: str, *, h0: float = SYNAPSE_HAZARD_REST, family: str = SYNAPSE_HAZARD_FAMILY,
                        scaling: str = SYNAPSE_HAZARD_SCALING, trace: str = TRACE) -> None:
        """What explores (AUTHORITY.md §7.1): "neuron", the neuron's own escape noise (§6.5), or "synapse", its synapses'.

        Under "synapse" every neuron takes the comparison of §6.13 -- every
        width 0, ESCAPE_DELTA not consulted -- and every synapse decides at
        every wave on its source's potential (§7.5), at the rest hazard `h0`
        in [0, 1) and in the `family` of §7.6, scaled by the `scaling` of §7.7:
        kappa_i is kappa(N) under "count" and kappa(N) / F_i under "fan-out",
        F_i counting an output's read synapse (§7.9) -- a rule and not a state,
        computed here, once, as §7.5's engine note says. `trace` is §8.17's.
        Every neuron keeps its trace, and the reinforce rule is posted at the
        synapses' decisions (§8.16).

        Refused (§12.2), and the network left as it was: a setting that is not
        the file's (§7.6, §7.7, §8.17), a positive width already set (§6.13),
        hebb (§8.3), a threshold at or below zero, where u is undefined (§7.5),
        an inactive connection, whose draw has no rule (§3.8), the inputs read
        as the outputs, which have no read synapse (§7.9), and a network
        carrying the other exploration's bookkeeping, which nothing maps across
        (§12.9) -- a neuron's exposure clock among it, once the neuron has
        spiked. Call it once the thresholds are what the container gave them,
        as set_delta is called.
        """
        if mode not in EXPLORATIONS:
            raise ValueError(f"unknown exploration {mode!r}; choose from {', '.join(EXPLORATIONS)} (§7.1)")
        self._refuse_settings(h0, family, scaling, trace)
        everyone = self._everyone()
        if mode != self._exploration:  # E_j and the credit, or the gain, and the notes taken on them (§12.9)
            if self._exploration == "neuron":
                carried = any(n.expected or n.credit for n in everyone)
            else:
                carried = any(n.gain for n in everyone)
            if carried or any(c.noted for c in self.connections.values()):
                raise ValueError(f"this network carries the {self._exploration} rule's open bookkeeping, and nothing maps "
                                 f"it onto the {mode} rule's: a run is not switched between the two explorations (§12.9)")
            if any(n.fired_at is not None for n in everyone):  # the one clock a neuron holds (§2.1) reads its spike two ways
                now = "the neuron rule" if self._exploration == "neuron" else "exploration at the synapse"
                raise ValueError(f"a neuron of this network has spiked, and the exposure clock it holds runs as {now} runs "
                                 "it -- from the refractory period's end under the neuron rule (§6.12), from the spike "
                                 "itself under exploration at the synapse (§7.8) -- and nothing maps one onto the other: a "
                                 "run is not switched between the two explorations (§12.9)")
        if mode == "synapse":
            if self.escape_delta > 0.0 or any(n.delta > 0.0 for n in everyone):
                raise ValueError("a neuron width and exploration at the synapse together are refused: the system explores "
                                 "by one thing, and under exploration at the synapse every width is 0 (§6.13, §7.1)")
            if self.centred:
                raise ValueError("hebb is refused under exploration at the synapse: the decisions are the synapses', and "
                                 "it has no neuron decision to centre (§8.3)")
            self._refuse_unsupported(everyone)
        self.synapse_hazard_rest, self.synapse_hazard_family = float(h0), family
        self.synapse_hazard_scaling, self.trace_mode = scaling, trace
        self._computed_from = (scaling, trace)
        self._exploration = mode
        if mode == "neuron":
            for neuron in everyone:
                neuron.synaptic = neuron.trace_ventured = False
                neuron.synapse_scale = 1.0
                neuron.traced = neuron.delta > 0.0 or neuron.centred
            return
        self.escape_delta = 0.0
        self.escape_scale = escape_scale(len(everyone))  # kappa(N): every hazard is scaled by the count (§6.6, §7.7)
        readers = {id(neuron) for neuron in self.output_row()}
        for neuron in everyone:
            neuron.delta = 0.0  # §6.13: the neuron's decision is the comparison
            neuron.escape_scale = self.escape_scale
            synapses = len(neuron.outgoing) + (id(neuron) in readers)  # F_i, the read synapse among them (§7.9)
            # a source with no synapse makes no decision; it keeps kappa(N), there being nothing to divide by (cf. §4.11)
            neuron.synapse_scale = self.escape_scale / synapses if scaling == "fan-out" and synapses else self.escape_scale
            neuron.synaptic = neuron.traced = True
            neuron.trace_ventured = trace == "ventured"

    def _refuse_settings(self, h0, family, scaling, trace) -> None:
        """The settings of §7.6, §7.7 and §8.17, refused where they are not the file's (refuse_synapse_settings)."""
        refuse_synapse_settings(h0, family, scaling, trace)

    def _refuse_unsupported(self, everyone: list[Neuron], thresholds=None, actives=None) -> None:
        """What exploration at the synapse has no rule for, refused where it is set and again where it runs (§12.2).
        `thresholds` and `actives` are what the network runs on where that is not its neurons' and connections' own:
        the array engine's vectors."""
        if thresholds is None:
            thresholds = [n.threshold for n in everyone]
        if actives is None:
            actives = [c.is_active for c in self.connections.values()]
        if any(threshold <= 0.0 for threshold in thresholds):
            raise ValueError("a threshold at or below zero leaves u = clip(V, 0, theta) / theta undefined, and under "
                             "exploration at the synapse a network holding one is refused (§7.5)")
        if not all(actives):
            raise ValueError("an inactive connection keeps its draw under exploration at the synapse, and what it does with "
                             "it is not specified: a network holding one is refused (§3.8)")
        if self.readout != "top":
            raise ValueError(f"the {self.readout!r} readout has no read synapse: under exploration at the synapse every "
                             "output neuron has one and the output zone is what is read (§7.9, §5.1)")
        if Neuron.bored_after > 0.0:
            raise ValueError("a bored threshold moves the theta the synapses' u is read on, and no clause gives it a place "
                             "under exploration at the synapse: refused (§7.5)")

    def scale_with_fan_in(
        self, threshold: float, minimum_potential: float, reference: float = THRESHOLD_FAN_IN
    ) -> None:
        """Rescale each neuron's potential axis by its in-degree (AUTHORITY.md §5.2).

        THRESHOLD and MINIMUM_POTENTIAL are quoted at `reference` incoming
        synapses (§4.11), so a neuron wired at that in-degree keeps 0.25 and -1
        exactly and one with four times the fan-in starts four times as far
        from zero in both directions.

        The floor moves with the threshold because it is not a second
        decision: the two are points on one axis and it is the axis being
        rescaled. Scaling the threshold alone squeezes the usable negative
        range from four times the threshold to 0.91 times it, so inhibition
        hits its cap while excitation keeps piling up -- which is most of what
        saturates goo (§3.4).

        Call it after the wiring, because it reads the in-degree. These are
        starting values only: homeostasis and un-sticking move a threshold
        from here.
        """
        if reference <= 0:
            raise ValueError(f"the reference fan-in must be positive, got {reference}")
        for neuron in self.all_neurons():
            # a neuron that hears nothing is left at the quoted threshold and floor (§5.2, September 16, 2026): there is
            # nothing to scale by, and at 0 it was a pacemaker whatever its drive
            scale = len(neuron.incoming) / reference if neuron.incoming else 1.0
            neuron.threshold = threshold * scale
            neuron.minimum_potential = minimum_potential * scale

    def clip_weight(self, weight: float) -> float:
        """Keep a weight inside the network's weight_range."""
        low, high = self.weight_range
        return max(low, min(high, weight))

    # --- input ------------------------------------------------------------

    def input_row(self) -> list[Neuron]:
        """The network's input neurons in order: the input zone, by place (§4.3)."""
        return [self.get_neuron_at(place, self.rows - 1) for place in range(self.across)]

    def output_width(self) -> int:
        """How many neurons are read as the output: the count across, unless the output zone is another width (§4.3)."""
        return self.across

    def output_row(self) -> list[Neuron]:
        """The network's output, by place: the output zone, or the input zone when the inputs are the outputs."""
        if self.readout == "input":
            return self.input_row()
        return [self.get_neuron_at(place, 0) for place in range(self.output_width())]

    def output_counts(self) -> list[int]:
        """Each output neuron's spikes this epoch: what the count read (§4.3) and the class critic (§8) work from --
        plus, under exploration at the synapse, its read synapse's escapes this epoch, the one place they are summed in
        (§5.10, §7.9)."""
        return [neuron.epoch_spikes + neuron.read_count for neuron in self.output_row()]

    def _refuse_read(self) -> None:
        """§5.10: under exploration at the synapse only the count is read, until a clause says what the others read."""
        if self.read != "count" and self._exploration == "synapse":
            raise ValueError(f"the {self.read!r} read is refused under exploration at the synapse until a clause says what "
                             "it reads there; the count read is the output's spikes plus its read synapse's escapes (§5.10)")

    def output_fired(self) -> list[bool]:
        """Whether each output neuron is on at the read: see `read`."""
        self._refuse_read()
        if self.read == "again":
            after = self.time + slack(self.time)  # strictly after the input's moment: a forced neuron must have spiked again
            return [neuron.fired_at is not None and neuron.fired_at > after for neuron in self.output_row()]
        if self.read == "rate":
            return [level >= 0.5 for level in self.output_levels()]  # half of saturation: for reporting and decoding
        if self.read == "window" and self.read_window is not None:
            since = self.horizon - self.read_window
            return [neuron.fired_at is not None and neuron.fired_at + slack(neuron.fired_at) >= since for neuron in self.output_row()]
        if self.read == "count":  # §5.10: the read is the count; §9.5 puts the line at ROW_CRITIC_PICKINESS_IN_SPIKES
            return [n >= self.pickiness for n in self.output_counts()]
        return [neuron.has_fired for neuron in self.output_row()]

    def output_counts_hz(self) -> list[float]:
        """Each output neuron's firing rate estimated from its count this epoch, in Hz: count over the epoch's length (§4.3)."""
        per_ms = 1000.0 / self.interval
        return [(neuron.epoch_spikes + neuron.read_count) * per_ms for neuron in self.output_row()]  # the count's (§5.10)

    def output_rates(self) -> list[float]:
        """Each output neuron's firing rate at the read, in Hz (AUTHORITY.md §4.3): the exponential window of Neuron.firing_rate."""
        return [neuron.firing_rate(self.horizon) for neuron in self.output_row()]

    def output_levels(self) -> list[float]:
        """What the teacher reads, one number in [0, 1] per output neuron (AUTHORITY.md §6.9).

        Under `read = "rate"` it is the measured rate over `rate_on`, clipped:
        0 is silence and 1 is saturation. Under every other read it is the
        boolean of `output_fired` as 0.0 or 1.0, so a rule scored this way is
        scored exactly as it was before the rate read existed.
        """
        self._refuse_read()
        if self.read == "rate":
            return [min(1.0, max(0.0, rate / self.rate_on)) for rate in self.output_rates()]
        return [1.0 if fired else 0.0 for fired in self.output_fired()]

    def input_width(self) -> int:
        """How many neurons the input covers: one bit of the (coded, permuted) pattern each (§4.3)."""
        return self.across

    def next_time(self) -> float:
        """When the next input arrives if no time is given: the interval after the last one (the first at 0)."""
        return self.time + self.interval if self.epoch else 0.0

    def set_input(self, pattern, time: float | None = None) -> None:
        """Store the input pattern, one boolean per input neuron, and the time it arrives (default: next_time()).

        Under `flip` the pattern is corrupted on the way in (§4.3): each bit is
        flipped independently with that probability, drawn from the network's own
        seeded stream, so `input_pattern` is what the row is forced with and
        `target_pattern` is the clean pattern the read is scored against. With
        flip at 0 nothing is drawn and the two are the same.
        """
        pattern = [bool(b) for b in pattern]
        if len(pattern) != self.input_width():
            raise ValueError(f"input pattern has {len(pattern)} bits but the input covers {self.input_width()} neurons")
        time = self.next_time() if time is None else float(time)
        if self.epoch and before(time, self.horizon):
            raise ValueError(f"input time {time} is before the schedule has already run to, {self.horizon}")
        # §5.2: the pattern presented is the pattern the read is scored against -- nothing corrupts an input on the way in
        self.target_pattern = pattern
        self.input_pattern = list(pattern)
        self.input_time = time
        for neuron, bit in zip(self.input_row(), pattern):
            neuron.should_fire = bit  # what the neuron should do, not what it was forced with; the learning rule
            # reverses its sign for a neuron that should not fire

    def raw_bit_count(self) -> int:
        """How many raw bits an input takes: half the input zone, less the clock neurons (AUTHORITY.md §5.2).

        Clock neurons (§5.3) are input neurons too, but their bit is always 1
        and no raw bit reaches them: they come off the width first. An odd
        remainder is refused rather than rounded.
        """
        width = self.input_width() - self.clock
        if width % 2:
            raise ValueError(f"complement coding needs an even number of input neurons, got {width}")
        return width // 2


    def set_input_bits(self, bits, time: float | None = None) -> None:
        """Set the input from raw bits: complement-code them onto the input zone (AUTHORITY.md §5.2).

        k raw bits become 2k coded bits on 2k input neurons, so exactly half
        the zone is driven whatever the raw bits are. The clock neurons of
        §5.3 lead the zone, their bit always 1. Place i of the zone shows
        place i of the coded pattern: there is no permutation.
        """
        bits = [bool(b) for b in bits]
        wanted = self.raw_bit_count()
        if len(bits) != wanted:
            raise ValueError(f"expected {wanted} input bits for {self.input_width()} input neurons, got {len(bits)}")
        coded = [True] * self.clock + complement_code(bits)
        self.set_input(coded, time)
        self.input_bits = bits
        self.input_coded = coded


    def new_random_input(self, time: float | None = None) -> list[bool]:
        """Set the next input: the next pattern of an attached stream, or fresh raw bits.

        Without a stream the bits come from the network's own seeded stream,
        which is also the one used to build it, so a seed reproduces the whole
        sequence of inputs but only for a network built exactly the same way.
        With a stream attached by `use_input_stream` the patterns are the ones
        given, in order, so two networks that differ in anything at all still
        see the same inputs (§4.5). A run longer than the stream cycles it.
        """
        if self.input_stream:
            k = self.input_at % len(self.input_stream)
            bits = self.input_stream[k]
            self.input_label = self.input_labels[k] if self.input_labels is not None else None
            self.input_at += 1
        else:
            bits = [self._rng.random() < 0.5 for _ in range(self.raw_bit_count())]
            self.input_label = None
        self.set_input_bits(bits, time)
        return list(bits)

    def use_input_stream(self, patterns, labels=None) -> None:
        """Present these raw-bit patterns in order, instead of drawing fresh ones (AUTHORITY.md §4.5).

        Each pattern is one epoch's raw bits, `raw_bit_count()` of them. Pass
        None to go back to drawing. A list of lists is checked pattern by
        pattern here rather than at the epoch that trips over it; anything
        else indexable (a dataset's `mnist.Patterns`, say) is checked at its
        two ends. `labels`, one per pattern, ride with a dataset's images
        (§8): `new_random_input` sets `input_label` from them each epoch.
        """
        if patterns is None:
            self.input_stream, self.input_labels, self.input_at = None, None, 0
            return
        wanted = self.raw_bit_count()
        if isinstance(patterns, list):
            patterns = [[bool(b) for b in pattern] for pattern in patterns]
            wrong = next((k for k, pattern in enumerate(patterns) if len(pattern) != wanted), None)
            if wrong is not None:
                raise ValueError(f"input stream pattern {wrong} has {len(patterns[wrong])} bits, expected {wanted}")
        elif len(patterns) and (len(patterns[0]) != wanted or len(patterns[-1]) != wanted):
            raise ValueError(f"input stream patterns have {len(patterns[0])} bits, expected {wanted}")
        if labels is not None and len(labels) != len(patterns):
            raise ValueError(f"{len(labels)} labels for {len(patterns)} patterns")
        self.input_stream, self.input_labels, self.input_at = patterns, None if labels is None else list(labels), 0

    def input_neurons(self) -> list[Neuron]:
        """The bottom-row neurons whose input bit is 1 (empty if no pattern is set)."""
        if self.input_pattern is None:
            return []
        return [neuron for neuron, bit in zip(self.input_row(), self.input_pattern) if bit]

    @property
    def presentation_time(self) -> float:
        """How far into the epoch the drive runs (AUTHORITY.md §5.4a): `presentation`, or the whole epoch when None.

        More than the epoch is **refused** and not clipped (§0.5): a window past
        the horizon would schedule arrivals into an epoch already read (§3.10).
        """
        if self.presentation is None:
            return self.interval
        if self.presentation > self.interval:
            raise ValueError(
                f"presentation time {self.presentation:g} ms is longer than the epoch's {self.interval:g} ms; "
                "§5.4a refuses a window past the horizon rather than clipping it"
            )
        return float(self.presentation)

    def input_schedule(self) -> list[tuple[int, float]]:
        """When each input place is stimulated this epoch, as (place, time) in time order (AUTHORITY.md §4.3).

        Under `drive = "forced"` every place whose bit is 1 is stimulated once
        at the epoch's moment: the behaviour this has always had. Under
        `drive = "rate"` an independent Poisson process **drives** each place
        across the presentation window (§5.4a -- the whole epoch by default), at
        `input_rate` where its bit is 1 and `input_rate_off` where it is 0. Past
        the window there are no arrivals and what the input zone does is its own
        (§7.2): undriven is not the same as silent.

        These are arrivals, not spikes. An arrival that lands while the neuron
        is refractory is dropped by the schedule, so the neuron fires at the
        first arrival after its refractory period ends, and its spike train is
        a renewal process with dead time rather than a Poisson one: mean ISI =
        REFRACTORY + 1/rate, and a coefficient of variation below 1 where a
        Poisson train's is exactly 1. The refractory period that matters is
        the neuron's own, so a spike the mesh drove also silences the drive,
        which is why the arrivals are emitted in full rather than thinned here
        (Byron, September 14, 2026: real neural signalling is not a Poisson
        process).

        The draws come from the network's own seeded stream, in place order,
        so a seed reproduces them and both engines draw the same train.

        Under `drive = "charged"` (5.4b) the arrivals are drawn the same way at
        `drive_steps` times the rate, each a delivery of theta / drive_steps
        rather than a forced spike; it runs under exploration at the synapse
        only, and `drive_steps` is a whole number of at least 1.
        """
        pattern = self.input_pattern
        if pattern is None:
            raise ValueError("no input pattern set; call set_input() first")
        steps = self._drive_steps()
        if self.drive == "forced":
            return [(place, self.time) for place, bit in enumerate(pattern) if bit]
        end = self.time + self.presentation_time
        events: list[tuple[int, float]] = []
        for place, bit in enumerate(pattern):
            rate = self.input_rate if bit else self.input_rate_off
            if steps != 1:
                rate = rate * steps  # 5.4b: at DRIVE_STEPS times the rate
            if rate <= 0.0:
                continue
            when = self.time
            while True:
                when += self._rng.expovariate(rate)
                if when >= end:
                    break
                events.append((place, when))
        events.sort(key=lambda event: event[1])
        return events

    def _drive_steps(self) -> int:
        """The drive checked where the epoch's arrivals are drawn and again before an epoch moves anything (§12.2), and
        the deliveries each arrival stands for: 1 under the rate and forced drives, DRIVE_STEPS under the charged (5.4b).

        Refused: a drive the code does not know, which used to be taken for
        one spike at the epoch's moment (§5.7); the code's forced drive, which
        is that spike, under exploration at the synapse -- kept for the neuron
        rule's plumbing tests, and not carried into a mechanism §5.7 already
        governs; the charged drive under the neuron rule (5.4b); and a
        DRIVE_STEPS that is not a whole number of at least 1 (5.4b). Any
        integral type is a whole number, a numpy integer among them, and is
        handed on as an int; a float is not, whatever its value.
        """
        if self.drive not in DRIVES:
            raise ValueError(f"unknown drive {self.drive!r}; choose from {', '.join(DRIVES)}. A drive the code does not know "
                             "is refused rather than taken for one spike at the epoch's moment (§5.7)")
        if self.drive == "forced" and self._exploration == "synapse":
            raise ValueError("the code's forced drive makes every bit-1 input spike at the epoch's moment, which §5.7 rules "
                             "out; it is kept for the neuron rule's plumbing tests and refused under exploration at the "
                             "synapse (§5.7, §12.2): drive it by the rate or the charged drive")
        if self.drive != "charged":
            return 1  # the rate drive's arrivals are the rate's own
        if self._exploration != "synapse":
            raise ValueError("the charged drive runs under exploration at the synapse only: under the neuron rule a "
                             "charged input would fire by its own hazard, not by the comparison its arithmetic is "
                             "written on (5.4b)")
        steps = self.drive_steps
        if isinstance(steps, bool) or not isinstance(steps, numbers.Integral) or steps < 1:
            raise ValueError(f"DRIVE_STEPS is a count of deliveries, a whole number of at least 1; got {steps!r} (5.4b)")
        return int(steps)

    def fire_input(self, until: float | None = None) -> list[Wave]:
        """Present the input: schedule the stimulus at its time and run the schedule to the horizon.

        The horizon is `until`, by default the interval after the input: the
        next input's time. Signals due at or after it wait for the next epoch.
        Returns this epoch's waves.
        """
        if self.input_pattern is None:
            raise ValueError("no input pattern set; call set_input() first")
        explore = self.explorer()  # before anything is drawn or scheduled: a run it refuses leaves the network as it was
        steps = self._drive_steps()  # and so is a drive it refuses
        self.time = self.input_time if self.input_time is not None else self.next_time()
        self.epoch += 1
        row = self.input_row()
        self.input_events = self.input_schedule()
        for place, when in self.input_events:
            if self.drive == "charged":
                self.schedule.charge(row[place], steps, when)  # a delivery, not a forced spike (5.4b)
            else:
                self.schedule.stimulus(row[place], when)
        self.horizon = self.time + self.interval if until is None else float(until)
        waves = self.schedule.run(self.horizon, self.waves, self._on_wave, self._everyone(),
                                  explore=explore, synapses=self.decider())
        return waves

    def _everyone(self) -> list[Neuron]:
        neurons = self.all_neurons()
        return neurons if isinstance(neurons, list) else list(neurons)

    def _on_wave(self, wave: Wave) -> None:
        """After a wave has fired, the local rules run (AUTHORITY.md §10).

        The specification carries the quash alone (§10.2), so there is one
        of them. It is local — the neuron's own spikes and the stamps on its
        own synapses — lazy, and off until a run asks for it.
        """
        if self.quash_rate:
            quash(wave, self.quash_rate, self.quash_k, self.weight_range)

    def total_spikes(self) -> int:
        return sum(neuron.spikes for neuron in self.all_neurons())

    def output_times(self) -> list[float | None]:
        """When each output neuron fired in the last cascade (the cascade's time), or None if it did not."""
        return [neuron.fired_at if neuron.has_fired else None for neuron in self.output_row()]

    # --- running ----------------------------------------------------------

    def propagate(self, fire=(), inputs=None, now: float | None = None, until: float | None = None) -> list[Wave]:
        """Run a cascade from the given stimulus at `now` (default: the clock) on the network's schedule.

        Anything already in flight runs with it, up to `until` (default: one
        interval after `now`; activity can sustain itself, so a bound is
        needed). The waves are appended to this epoch's and returned.
        """
        now = self.time if now is None else now
        explore = self.explorer()
        for neuron in fire:
            self.schedule.stimulus(neuron, now)
        for neuron, amount in (inputs or {}).items():
            self.schedule.external(neuron, amount, now)
        self.horizon = now + self.interval if until is None else float(until)
        return self.schedule.run(self.horizon, self.waves, self._on_wave, self._everyone(),
                                 explore=explore, synapses=self.decider())

    def reset(self, discharge: bool = False) -> None:
        """Start a new epoch: clear every neuron's fired-this-epoch state and this epoch's waves.

        Potentials are kept (there is no leak); with `discharge=True` every
        potential is zeroed instead. Signals in flight stay scheduled.
        """
        for neuron in self.all_neurons():
            neuron.reset(discharge)
        if self.traced:
            accumulating = Neuron.tau == math.inf
            synaptic = self._exploration == "synapse"
            for connection in self.connections.values():
                connection.score = 0.0  # the score is the epoch's (§6.7); the trace is the potential's and stays
                if accumulating:  # an open arrival's debit counts from here: B = x E, or x G at the synapses (§8.16)
                    target = connection.target
                    connection.noted = connection.trace * (target.gain if synaptic else target.expected)
        self.waves = []

    def _explore(self, time: float) -> None:
        """Before each wave's firing decision: one uniform per neuron, in neuron order (AUTHORITY.md §6.7, §7.3)."""
        neurons = self._everyone()
        for neuron, draw in zip(neurons, hazard_draws(self.explore_rng, len(neurons))):
            neuron.draw = draw

    def _explore_synapses(self, time: float) -> None:
        """Before each wave's fire phase under exploration at the synapse: one uniform per synapse in edge order, then one
        per output neuron for its read synapse, in output order -- all of them, whatever fires (AUTHORITY.md §3.8)."""
        self._draws = hazard_draws(self.explore_rng, self._draw_count)

    def explorer(self):
        """The hook Schedule.run calls before each wave fires, or None where the threshold decides.

        Under exploration at the synapse it is the synapses' draws, and this is
        where a run is checked for what that exploration refuses (§12.2) and
        its draws are laid out: E in edge order -- sources in index order, each
        source's synapses in the order the topology was built (§12.5) -- then O
        in output order (§3.8).
        """
        if self._exploration == "synapse":
            if self.explore_rng is None:
                raise ValueError("exploration at the synapse draws for every synapse at every wave, whatever its rest "
                                 "hazard, and needs a stream of its own (§7.3): run the epoch with an rng")
            everyone = self._everyone()
            self._refuse_synapse_run(everyone)
            edges = sum(len(neuron.outgoing) for neuron in everyone)
            slot = {}  # one read synapse per output neuron, in output order: a neuron at two places has one (§7.9)
            for neuron in self.output_row():
                slot.setdefault(id(neuron), edges + len(slot))
            self._layout = [(neuron, slot.get(id(neuron), -1)) for neuron in everyone]
            self._draw_count = edges + len(slot)
            return self._explore_synapses
        if self.hazard and self.explore_rng is None:
            raise ValueError("escape noise needs a stream for its draws (§7.3): run the epoch with an rng, as run_epoch does")
        return self._explore if self.hazard else None

    def _refuse_synapse_run(self, everyone: list[Neuron], thresholds=None, actives=None, deltas=None) -> None:
        """What a run under exploration at the synapse refuses where it runs (§12.2), its stream aside: the settings as
        they stand, since they stay writable attributes, against the ones set_exploration computed from; what the
        mechanism has no rule for; a read other than the count; any neuron width; and hebb (§8.3), the network's or a
        neuron's. The object engine checks it at every fire_input and propagate, the Rust loop's driver (fast.py) where
        it builds, trains and compares, and the array engine where it runs, on its own thresholds, active flags, widths
        and hebb."""
        scaling, trace = self.synapse_hazard_scaling, self.trace_mode
        self._refuse_settings(self.synapse_hazard_rest, self.synapse_hazard_family, scaling, trace)
        if (scaling, trace) != self._computed_from:
            raise ValueError(f"the scaling {scaling!r} and the trace {trace!r} are not the ones set_exploration computed "
                             f"kappa_i and every trace from, {self._computed_from}: kappa_i is computed once, where the "
                             "network is built or resumed (§7.5, §7.7), and TRACE sets what each trace counts (§8.17); "
                             "name them to set_exploration")
        self._refuse_unsupported(everyone, thresholds, actives)
        self._refuse_read()
        if deltas is None:
            deltas = [n.delta for n in everyone]
        if any(delta != 0.0 for delta in deltas):  # a width that is not a number included
            raise ValueError("a neuron width and exploration at the synapse together are refused: under exploration "
                             "at the synapse every width is 0 (§6.13)")
        if self.centred or any(n.centred for n in everyone):  # writable too, the network's and each neuron's
            raise ValueError("hebb is refused under exploration at the synapse: the decisions are the synapses', and it has "
                             "no neuron decision to centre (§8.3)")

    def decider(self):
        """The hook Schedule.run calls after each wave has fired: the synapses' decisions (§7.5), or None under the
        neuron rule. Call explorer() first, which lays the draws out."""
        return self._decide_synapses if self._exploration == "synapse" else None

    def _decide_synapses(self, wave: Wave) -> list:
        """After the fire phase: every synapse of every source that did not spike this wave decides (AUTHORITY.md §7.5).

        A source that spiked transmitted on every synapse with its spike, and
        its draws go unused (§3.8). Every other source -- refractory or not
        (§7.8), touched or not -- has its synapses decide together on one m and
        one P = -expm1(-m) (Neuron.synapse_expected), each escaping iff its
        uniform is strictly below P; an output's read synapse decides with them,
        and its escape counts toward the output's read at this wave (§7.9). An
        escape leaves the source's potential where it was. The source's one
        exposure clock is brought to now, F_i = 0 included.

        Where the source's potential is above zero and m > 0, the wave posts
        §8.16's entry for it: a c - (F - a) m, c = m e^-m / (1 - e^-m), times
        rho~ = (1 - h0) / h(u) under the linear family -- into the gain under
        the evidence accumulator, the arrivals settling it (x G - B), and under
        the leak by one walk over the source's fan-in. Where rho~ overflows --
        h0 = 0 and a potential leaked below the normal range, h underflowing
        with it -- the entry in the engine note's form is not finite, and the
        run is refused (§12.2). The escapes are returned in edge order for the
        schedule to push one hop later, after the wave's spikes (§3.6).
        """
        draws, number, time = self._draws, wave.number, wave.time
        rest = self.synapse_hazard_rest
        linear = self.synapse_hazard_family == "linear"
        tau = Neuron.tau
        escaped = []
        k = 0  # the next synapse's draw, in edge order
        for neuron, slot in self._layout:
            outgoing = neuron.outgoing
            if neuron.fired_in_wave == number:  # it spiked: its synapses transmitted, and decide nothing (§6.13)
                k += len(outgoing)
                continue
            synapses = len(outgoing) + (slot >= 0)  # F_i, the read synapse among them
            if not synapses:
                neuron.exposed_since = time  # one clock per source, brought to now though nothing decides on it
                continue
            m, h = neuron.synapse_expected(time, rest, linear)
            chance = -math.expm1(-m)
            escapes = 0
            for connection in outgoing:
                if draws[k] < chance:
                    escaped.append(connection)
                    escapes += 1
                k += 1
            if slot >= 0 and draws[slot] < chance:
                neuron.read_count += 1  # it delivers nothing, so nothing of it is in flight (§7.9)
                escapes += 1
            neuron.exposed_since = time
            if m > 0.0 and neuron.potential_at(time) > 0.0:  # §8.16: posted only while V_i > 0, and c only where m > 0
                entry = escapes * (m * math.exp(-m) / -math.expm1(-m)) - (synapses - escapes) * m
                if linear:
                    entry = (1.0 - rest) / h * entry  # rho~, the log-derivative the linear family leaves unfolded
                    if not math.isfinite(entry):  # h > 0 wherever m > 0, so the division itself is always defined
                        raise ValueError("under the linear family rho~ = (1 - h0) / h multiplies the entry, and where h "
                                         "has underflowed -- h0 = 0, and a source's potential leaked below the normal "
                                         "range -- it overflows and the entry is not finite: the run is refused rather "
                                         "than post it (§8.16, §12.2)")
                if tau == math.inf:
                    neuron.gain += entry  # after this wave's arrivals have noted (§8.16)
                else:
                    for connection in neuron.incoming:  # the leak (§8.12): the walk, once a wave per posting source
                        if connection.trace != 0.0:
                            connection.score += entry * connection.trace * math.exp(-(time - connection.trace_at) / tau)
        return escaped

    def fired_neurons(self) -> list[Neuron]:
        """Return the neurons that have fired since the last reset."""
        return [n for n in self.all_neurons() if n.has_fired]
