"""What every container of neurons shares: an input row, an output row, epochs and the schedule.

A network has `across` x `rows` addressable positions (`get_neuron_at(place,
row)`, row 0 at the top), a bottom row that receives the complement-coded and
permuted input pattern, a top row that is read as the output, and the epoch
machinery: reset, present an input at a time, run the schedule to the next
input's time (AUTHORITY.md §4.2). The hex mesh and the Cartesian lattice both
build on this; the learning code works on either.
"""

from __future__ import annotations

import math

import random
from typing import Iterable

from .constants import (
    ESCAPE_REFERENCE_COUNT, EXPLORE, HEBB_RATE, INPUT_DRIVE, INPUT_RATE, INPUT_RATE_OFF, INTERVAL, POPULATION, TEMPERATURE,
    QUASH_K, QUASH_RATE, RATE_ON, SYNAPSE_TAU, TEACHER_THRESHOLD, THRESHOLD_FAN_IN,
)


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
from .dopamine import MODES, apply_teacher, leaky_hebb, learn, quash
from .exploration import gaussians, hazard_draws
from .inputs import CODES, DEFAULT_CODE, Code, complement_code
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
        self.input_pattern: list[bool] | None = None  # one bit per place along the bottom row: what is forced
        self.target_pattern: list[bool] | None = None  # the clean pattern the read is scored against, before any flips (§4.3)
        self.input_bits: list[bool] | None = None  # the raw bits before complement coding
        self.input_coded: list[bool] | None = None  # the complement-coded bits before permutation
        self.permutation: list[int] = list(range(across))  # place i along the bottom row shows coded bit permutation[i]
        self.input_stream: list[list[bool]] | None = None  # raw-bit patterns to present in order, in place of fresh
        # draws (§4.5); None draws from the network's own stream, as it always did
        self.input_at = 0  # how far through that stream the run has got
        self.input_labels: list[int] | None = None  # the label of each pattern of the stream, when a dataset gave one (§8, mnist)
        self.input_label: int | None = None  # this epoch's label, when the stream carries labels
        self.epoch = 0  # how many inputs have been presented
        self.time = 0.0  # the clock, nominal milliseconds: the time of the last input
        self.interval = INTERVAL  # default spacing of inputs when no time is given
        self.input_time: float | None = None  # when the pending input arrives
        self.horizon = 0.0  # the time the schedule has run to: the next input may not come before it
        self.schedule = Schedule()  # signals in flight, across epochs
        self.dopamine = None  # a dopamine.Dopamine when the dopamine or teacher rule runs (set by whoever builds the run)
        self.quash_rate = 0.0  # a refire weakens its contributing synapses by this fraction of their weight (§6.11); off until asked
        self.quash_k = QUASH_K  # per ms: the quash falls off with the delay since the previous spike
        self.hebb_rate = 0.0  # leaky Hebb (§6.12): a firing neuron potentiates its gated synapses by this much times
        # their leaky trace. Like the quash it composes with whatever rule pays the read; off until asked.
        self.synapse_tau = SYNAPSE_TAU  # the leak of that trace, the synapse's own and no longer the neuron's (§6.12)
        self.explore = EXPLORE  # when the exploration draw is taken (§6.1): every wave, or once per epoch
        self.sigma = 0.0  # the standard deviation of that draw; whoever runs the epoch sets it
        self.explore_rng = None  # the stream it comes from
        self.escape_delta = 0.0  # ESCAPE_DELTA as set on this network (§5.2): 0 keeps the deterministic threshold
        self.escape_scale = 1.0  # sqrt(ESCAPE_REFERENCE_COUNT / N), the count's scaling of every hazard (§5.2), once set_delta ran
        self.rule = "dopamine"  # how the schedule's hook serves learning: "dopamine" (the refires move the weights), "teacher"
        # (they earn eligibility for the read) or "adaline" (every synapse counts what it delivered, §6.10)
        self.tally = False  # every synapse counts the signals its target integrated this epoch (Connection.eligibility):
        # the x_ij of the count_hebb eligibility (§6.7). A Teacher with that eligibility switches it on, in whichever engine
        self.centred = False  # the hebb eligibility charges every neuron's decisions against its own expectation (§6.7,
        # the single-spike rule). A Teacher with that eligibility switches it on, in whichever engine (centre)
        self.readout = "top"  # what is read as the output: the top row, or "input" (the inputs are the outputs)
        self.coding = "complement"  # how raw bits reach the input row: "complement" (bits then their negations), "raw"
        # (as they are), "population" (each bit repeated `population` times) or "population-complement" (both, in that
        # order: repeated, then the whole run followed by its negation)
        self.population = POPULATION  # neurons per raw bit under population coding
        self.output_coding = "population"  # how the output zone codes the classes (§8): a population a class, or "complement"
        # -- fire-if-one populations then, in the same order, fire-if-zero ones (Byron, September 16, 2026; learning.OUTPUT_CODINGS)
        self.temperature = TEMPERATURE  # the evidence critic's temperature: the class sums as log-odds at this scale (§8)
        self.clock = 0  # clock neurons (§4.3, Byron, September 15, 2026): this many input neurons at the front of the input
        # zone whose bit is always 1, so the drive fires them every epoch whatever the pattern; they take no raw bits
        self.flip = 0.0  # probability each bit of the coded, permuted pattern is flipped before the row is forced (§4.3); off until asked
        self.drive = INPUT_DRIVE  # how a bit becomes spikes (§4.3): "forced", one spike at the epoch's moment, or "rate"
        self.input_rate = INPUT_RATE  # per ms: what a bit-1 neuron fires at under rate drive
        self.input_rate_off = INPUT_RATE_OFF  # per ms: what a bit-0 neuron fires at
        self.input_events: list[tuple[int, float]] | None = None  # the (place, time) stimuli this epoch actually used;
        # input_schedule() draws afresh under rate drive, so this is what to read to see what happened
        self.input_cells: list[tuple[int, int]] | None = None  # an input zone, (place, row) cells, in place of the bottom row
        self.read = "fired"  # what "on" means at the read: "fired" this epoch; "again", spiked after the epoch's input moment
        # (a forced neuron must have refired); "window", within read_window ms before the horizon
        self.read_window: float | None = None  # the window for read == "window"
        self.rate_on = RATE_ON  # Hz: the rate a target-on output is driven to, and what output_levels divides by (§4.3)
        self.teacher_threshold = TEACHER_THRESHOLD  # Hz: the count read's line between off and on (§4.3)
        self.ecc: str | None = None  # name of the error-correcting code applied before complement coding, if any
        self.input_data: list[bool] | None = None  # the raw data bits when ecc is on

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
    def traced(self) -> bool:
        """Whether each synapse keeps its trace of what it has in its target's potential (§6.7): escape noise, or the centred rule."""
        return self.hazard or self.centred

    def centre(self, on: bool) -> None:
        """The hebb eligibility (§6.7, the single-spike rule): every neuron charges its decisions against its own expectation."""
        self.centred = bool(on)
        for neuron in self._everyone():
            neuron.centred = self.centred
            neuron.traced = self.centred or neuron.delta > 0.0

    def settle_scores(self) -> None:
        """Under the evidence accumulator (§5.1), bring every open arrival's debit into its synapse's score (§6.7).

        The pay at the read calls it first, so the score holds what the epoch's
        decisions charged; the arrivals stay open, their debit counting from
        now. Under the leak the scores are charged per decision and there is
        nothing to settle.
        """
        if Neuron.tau != math.inf or not self.traced:
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
        (§6.1), so a run with a positive width needs one.
        """
        if delta < 0.0:
            raise ValueError(f"ESCAPE_DELTA must not be negative, got {delta}")
        self.escape_delta = float(delta)
        everyone = self._everyone()
        self.escape_scale = escape_scale(len(everyone))  # the count's scaling of every hazard (§5.2, September 16, 2026)
        for neuron in everyone:
            # a neuron whose starting threshold is not positive -- no incoming synapses under fan-in scaling (§5.2) --
            # has a collapsed axis with no width to quote on it, and keeps the deterministic rule
            neuron.delta = delta * neuron.threshold if neuron.threshold > 0.0 else 0.0
            neuron.escape_scale = self.escape_scale
            neuron.traced = neuron.delta > 0.0 or neuron.centred  # the trace of §6.7 is kept under escape noise

    def scale_with_fan_in(
        self, threshold: float, minimum_potential: float, reference: float = THRESHOLD_FAN_IN
    ) -> None:
        """Rescale each neuron's potential axis by its in-degree (AUTHORITY.md §5.2).

        THRESHOLD and MINIMUM_POTENTIAL are quoted at `reference` incoming
        synapses -- an interior hex cell's two rings -- so a neuron wired like
        that cell keeps 0.25 and -1 exactly and one with four times the fan-in
        starts four times as far from zero in both directions.

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
        """The network's input neurons in order: the bottom row, left to right, or the input zone if one is set."""
        if self.input_cells is not None:
            return [self.get_neuron_at(place, row) for place, row in self.input_cells]
        return [self.get_neuron_at(place, self.rows - 1) for place in range(self.across)]

    def set_input_cells(self, cells) -> None:
        """Put the input on these (place, row) cells instead of the bottom row; the permutation resets to the identity."""
        cells = [(int(place), int(row)) for place, row in cells]
        for place, row in cells:
            if self.get_neuron_at(place, row) is None:
                raise ValueError(f"no neuron at place {place}, row {row}")
        self.input_cells = cells
        self.permutation = list(range(len(cells)))

    def output_width(self) -> int:
        """How many neurons are read as the output: the count across, unless a container's output zone is another width (goo, §3.4)."""
        return self.across

    def output_row(self) -> list[Neuron]:
        """The network's output, left to right: the top row, or the input row when the inputs are the outputs."""
        if self.readout == "input":
            return self.input_row()
        return [self.get_neuron_at(place, 0) for place in range(self.output_width())]

    def output_counts(self) -> list[int]:
        """Each output neuron's spikes this epoch: what the count read (§4.3) and the class critic (§8) work from."""
        return [neuron.epoch_spikes for neuron in self.output_row()]

    def output_fired(self) -> list[bool]:
        """Whether each output neuron is on at the read: see `read`."""
        if self.read == "again":
            after = self.time + slack(self.time)  # strictly after the input's moment: a forced neuron must have spiked again
            return [neuron.fired_at is not None and neuron.fired_at > after for neuron in self.output_row()]
        if self.read == "rate":
            return [level >= 0.5 for level in self.output_levels()]  # half of saturation: for reporting and decoding
        if self.read == "window" and self.read_window is not None:
            since = self.horizon - self.read_window
            return [neuron.fired_at is not None and neuron.fired_at + slack(neuron.fired_at) >= since for neuron in self.output_row()]
        if self.read == "count":  # §4.3: count the epoch's spikes, estimate the rate, and threshold it
            return [level >= self.teacher_threshold for level in self.output_counts_hz()]
        return [neuron.has_fired for neuron in self.output_row()]

    def output_counts_hz(self) -> list[float]:
        """Each output neuron's firing rate estimated from its count this epoch, in Hz: count over the epoch's length (§4.3)."""
        per_ms = 1000.0 / self.interval
        return [neuron.epoch_spikes * per_ms for neuron in self.output_row()]

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
        if self.read == "rate":
            return [min(1.0, max(0.0, rate / self.rate_on)) for rate in self.output_rates()]
        return [1.0 if fired else 0.0 for fired in self.output_fired()]

    def input_width(self) -> int:
        """How many neurons the input covers: one bit of the (coded, permuted) pattern each."""
        return len(self.input_cells) if self.input_cells is not None else self.across

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
        self.target_pattern = pattern
        self.input_pattern = [bit != (self._rng.random() < self.flip) for bit in pattern] if self.flip else list(pattern)
        self.input_time = time
        for neuron, bit in zip(self.input_row(), pattern):
            neuron.should_fire = bit  # what the neuron should do, not what it was forced with; the learning rule
            # reverses its sign for a neuron that should not fire (dopamine.py)

    @property
    def code(self) -> Code | None:
        return CODES[self.ecc] if self.ecc else None

    def raw_bit_count(self) -> int:
        """How many raw bits an input takes: half the count across (complement coding), all of it (raw), or the code's data bits.

        Clock neurons (§4.3) are input neurons too, but their bit is always 1
        and no raw bit reaches them: they come off the width first.
        """
        width = self.input_width() - self.clock
        if self.coding in ("raw", "population", "population-complement"):
            if self.code:
                raise ValueError(f"an error-correcting code needs complement coding, not {self.coding}")
            if self.coding == "raw":
                return width
            if self.coding == "population-complement":
                if width % 2:
                    raise ValueError(f"complement coding needs an even number of input neurons, got {width}")
                half = width // 2
                if half % self.population:
                    raise ValueError(
                        f"doubling into complement coding needs {2 * self.population} input neurons per raw bit, got {width}"
                    )
                return half // self.population
            if width % self.population:
                raise ValueError(f"population coding needs a multiple of {self.population} input neurons, got {width}")
            return width // self.population
        if width % 2:
            raise ValueError(f"complement coding needs an even number of input neurons, got {width}")
        if self.code:
            if width // 2 != self.code.code_bits:
                raise ValueError(f"{self.code.name} needs {2 * self.code.code_bits} input neurons, got {width}")
            return self.code.data_bits
        return width // 2

    def use_ecc(self, code: str | bool | None = DEFAULT_CODE) -> None:
        """Encode raw data bits with a named code before complement coding (True means the default, Hamming (7, 4))."""
        if code is True:
            code = DEFAULT_CODE
        if code is False:
            code = None
        if code is not None and code not in CODES:
            raise ValueError(f"unknown code {code!r}; choose from {', '.join(CODES)}")
        self.ecc = code
        self.raw_bit_count()  # validates the count across

    def set_input_bits(self, bits, time: float | None = None) -> None:
        """Set the input from raw bits: (ecc-encode them,) complement-code them, then permute.

        Without ecc the raw bits number half the count across. With ecc they are
        the 4 data bits, encoded to 7 before complement coding fills 14 places.
        Place i along the bottom row receives coded bit permutation[i].
        """
        bits = [bool(b) for b in bits]
        wanted = self.raw_bit_count()
        if len(bits) != wanted:
            raise ValueError(f"expected {wanted} input bits for {self.input_width()} input neurons, got {len(bits)}")
        word = self.code.encode(bits) if self.code else bits
        if self.coding == "raw":
            coded = list(word)
        elif self.coding == "population":
            coded = [bit for bit in word for _ in range(self.population)]  # each bit fills its own patch of the row
        elif self.coding == "population-complement":
            # repeated first, then the whole run followed by its negation (§4.3): 1001 -> 11000011 -> 1100001100111100
            coded = complement_code([bit for bit in word for _ in range(self.population)])
        else:
            coded = complement_code(word)
        coded = [True] * self.clock + coded  # the clock neurons lead the input zone, always on (§4.3)
        self.set_input([coded[i] for i in self.permutation], time)
        self.input_data = bits if self.code else None
        self.input_bits = word  # the bits that were complement-coded: the codeword with ecc, the raw bits without
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

    def input_schedule(self) -> list[tuple[int, float]]:
        """When each input place is stimulated this epoch, as (place, time) in time order (AUTHORITY.md §4.3).

        Under `drive = "forced"` every place whose bit is 1 is stimulated once
        at the epoch's moment: the behaviour this has always had. Under
        `drive = "rate"` an independent Poisson process **drives** each place
        across the epoch, at `input_rate` where its bit is 1 and
        `input_rate_off` where it is 0.

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
        """
        pattern = self.input_pattern
        if pattern is None:
            raise ValueError("no input pattern set; call set_input() first")
        if self.drive != "rate":
            return [(place, self.time) for place, bit in enumerate(pattern) if bit]
        end = self.time + self.interval
        events: list[tuple[int, float]] = []
        for place, bit in enumerate(pattern):
            rate = self.input_rate if bit else self.input_rate_off
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

    def fire_input(self, until: float | None = None) -> list[Wave]:
        """Present the input: schedule the stimulus at its time and run the schedule to the horizon.

        The horizon is `until`, by default the interval after the input: the
        next input's time. Signals due at or after it wait for the next epoch.
        Returns this epoch's waves.
        """
        if self.input_pattern is None:
            raise ValueError("no input pattern set; call set_input() first")
        self.time = self.input_time if self.input_time is not None else self.next_time()
        self.epoch += 1
        row = self.input_row()
        self.input_events = self.input_schedule()
        for place, when in self.input_events:
            self.schedule.stimulus(row[place], when)
        self.horizon = self.time + self.interval if until is None else float(until)
        waves = self.schedule.run(self.horizon, self.waves, self._on_wave, self._everyone(),
                                  trace=self.rule == "adaline" or self.tally, explore=self.explorer())
        self.forget()
        return waves

    def _everyone(self) -> list[Neuron]:
        neurons = self.all_neurons()
        return neurons if isinstance(neurons, list) else list(neurons)

    def forget(self) -> None:
        """Synapses that forget on their own: every weight moves toward zero by the dopamine rule's decay, once per epoch."""
        if self.dopamine is not None and self.dopamine.decay > 0.0:
            keep = 1.0 - self.dopamine.decay
            for connection in self.connections.values():
                connection.weight *= keep

    def _on_wave(self, wave: Wave) -> None:
        """After a wave has fired, the local rules run in order and then whatever pays the read.

        They compose (Byron, September 13, 2026: "there is not a neuron
        training rule. There are multiple compatible training rules"): the
        quash weakens what carried a cycle, leaky Hebb potentiates what
        carried the spike, and the rule of §6.9-6.10 pays at the read. The
        quash goes first, so a potentiation this wave is not discounted the
        moment it is made; only the quash depends on the weight, so no other
        order matters.
        """
        if self.quash_rate:
            quash(wave, self.quash_rate, self.quash_k, self.weight_range)
        if self.hebb_rate:
            leaky_hebb(wave, self.hebb_rate, self.synapse_tau, Neuron.hop(), self.weight_range)
        if self.dopamine is not None:
            learn(self.dopamine, wave, self.weight_range, mode=MODES.get(self.rule, "apply"))

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
        for neuron in fire:
            self.schedule.stimulus(neuron, now)
        for neuron, amount in (inputs or {}).items():
            self.schedule.external(neuron, amount, now)
        self.horizon = now + self.interval if until is None else float(until)
        return self.schedule.run(self.horizon, self.waves, self._on_wave, self._everyone(),
                                 trace=self.rule == "adaline" or self.tally, explore=self.explorer())

    def reset(self, discharge: bool = False) -> None:
        """Start a new epoch: clear every neuron's fired-this-epoch state and this epoch's waves.

        Potentials are kept (there is no leak); with `discharge=True` every
        potential is zeroed instead. Signals in flight stay scheduled.
        """
        for neuron in self.all_neurons():
            neuron.reset(discharge)
        if self.rule in ("teacher", "adaline") or self.tally:
            for connection in self.connections.values():
                connection.eligibility = 0.0  # a new epoch earns its own credit, or its own tally (§6.7)
        if self.traced:
            accumulating = Neuron.tau == math.inf
            for connection in self.connections.values():
                connection.score = 0.0  # the score is the epoch's (§6.7); the trace is the potential's and stays
                if accumulating:
                    connection.noted = connection.trace * connection.target.expected  # an open arrival's debit counts from here
        self.waves = []

    def perturb(self, sigma: float, rng, now: float | None = None, hold_fired: bool = False) -> None:
        """Exploration: add Gaussian noise of standard deviation `sigma` to every potential, floored.

        The potential is first leaked to `now` (default: the pending input's
        time), so the noise sits on top of what survived the gap. Each neuron
        remembers its draw as `noise` (the reinforce rule's eligibility). The
        draws come from `exploration.gaussians`, shared with the array engine.

        With `hold_fired` a neuron that has already fired this epoch keeps the
        `noise` it decided under instead of taking the new draw: under wave
        exploration (§6.1) the eligibility must refer to the perturbation that
        produced the spike, not to a later one that explains nothing.
        """
        now = self.input_time if now is None else now
        neurons = self.all_neurons() if isinstance(self.all_neurons(), list) else list(self.all_neurons())
        for neuron, draw in zip(neurons, gaussians(rng, len(neurons), sigma)):
            if now is not None:
                neuron.leak(now)
            if not (hold_fired and neuron.has_fired):
                neuron.noise = draw
            summed = neuron.potential + draw
            if summed < neuron.minimum_potential:
                summed = neuron.minimum_potential
                if neuron.traced:
                    neuron.clear_arrivals()  # the floor bit: the potential is the floor whatever the weights (§6.7)
            neuron.potential = summed

    def _explore(self, time: float) -> None:
        """Before each wave's firing decision: the additive draw of §6.1, then the hazard's uniforms (§5.2), in that order."""
        if self.explore == "wave" and self.sigma > 0.0:
            self.perturb(self.sigma, self.explore_rng, time, hold_fired=True)
        if self.hazard:
            neurons = self._everyone()
            for neuron, draw in zip(neurons, hazard_draws(self.explore_rng, len(neurons))):
                neuron.draw = draw

    def explorer(self):
        """The hook Schedule.run calls before each wave fires, or None when nothing is exploring."""
        additive = self.explore == "wave" and self.sigma > 0.0 and self.explore_rng is not None
        if self.hazard and self.explore_rng is None:
            raise ValueError("escape noise needs a stream for its draws (§5.2): run the epoch with an rng, as run_epoch does")
        return self._explore if (additive or self.hazard) else None

    def fired_neurons(self) -> list[Neuron]:
        """Return the neurons that have fired since the last reset."""
        return [n for n in self.all_neurons() if n.has_fired]
