"""What every container of neurons shares: an input row, an output row, epochs and the schedule.

A network has `across` x `rows` addressable positions (`get_neuron_at(place,
row)`, row 0 at the top), a bottom row that receives the complement-coded and
permuted input pattern, a top row that is read as the output, and the epoch
machinery: reset, present an input at a time, run the schedule to the next
input's time (AUTHORITY.md §4.2). The hex mesh and the Cartesian lattice both
build on this; the learning code works on either.
"""

from __future__ import annotations

from typing import Iterable

from .constants import (
    EXPLORE, HEBB_RATE, INPUT_DRIVE, INPUT_RATE, INPUT_RATE_OFF, INTERVAL, POPULATION, QUASH_K, QUASH_RATE,
    RATE_ON, SYNAPSE_TAU,
)
from .dopamine import MODES, apply_teacher, leaky_hebb, learn, quash
from .exploration import gaussians
from .inputs import CODES, DEFAULT_CODE, Code, complement_code
from .neuron import Neuron
from .clock import before, slack
from .propagation import Schedule, Wave


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
        self.rule = "dopamine"  # how the schedule's hook serves learning: "dopamine" (the refires move the weights), "teacher"
        # (they earn eligibility for the read) or "adaline" (every synapse counts what it delivered, §6.10)
        self.readout = "top"  # what is read as the output: the top row, or "input" (the inputs are the outputs)
        self.coding = "complement"  # how raw bits reach the input row: "complement" (bits then their negations), "raw" (as they
        # are) or "population" (each bit repeated `population` times)
        self.population = POPULATION  # neurons per raw bit under population coding
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
        self.ecc: str | None = None  # name of the error-correcting code applied before complement coding, if any
        self.input_data: list[bool] | None = None  # the raw data bits when ecc is on

    def all_neurons(self) -> Iterable[Neuron]:
        """Every neuron, in a stable order. Subclasses override if `neurons` is not a list."""
        return self.neurons

    def get_neuron_at(self, place: int, row: int) -> Neuron | None:  # pragma: no cover - overridden
        raise NotImplementedError

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

    def output_row(self) -> list[Neuron]:
        """The network's output, left to right: the top row, or the input row when the inputs are the outputs."""
        if self.readout == "input":
            return self.input_row()
        return [self.get_neuron_at(place, 0) for place in range(self.across)]

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
        return [neuron.has_fired for neuron in self.output_row()]

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
        """How many raw bits an input takes: half the count across (complement coding), all of it (raw), or the code's data bits."""
        width = self.input_width()
        if self.coding in ("raw", "population"):
            if self.code:
                raise ValueError(f"an error-correcting code needs complement coding, not {self.coding}")
            if self.coding == "raw":
                return width
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
        else:
            coded = complement_code(word)
        self.set_input([coded[i] for i in self.permutation], time)
        self.input_data = bits if self.code else None
        self.input_bits = word  # the bits that were complement-coded: the codeword with ecc, the raw bits without
        self.input_coded = coded

    def new_random_input(self, time: float | None = None) -> list[bool]:
        """Draw fresh raw bits from the network's seeded stream and set them as the input.

        Because the stream is the same one used to build the network, a seed
        reproduces the whole sequence of inputs, not just the first.
        """
        bits = [self._rng.random() < 0.5 for _ in range(self.raw_bit_count())]
        self.set_input_bits(bits, time)
        return bits

    def input_neurons(self) -> list[Neuron]:
        """The bottom-row neurons whose input bit is 1 (empty if no pattern is set)."""
        if self.input_pattern is None:
            return []
        return [neuron for neuron, bit in zip(self.input_row(), self.input_pattern) if bit]

    def input_schedule(self) -> list[tuple[int, float]]:
        """When each input place is stimulated this epoch, as (place, time) in time order (AUTHORITY.md §4.3).

        Under `drive = "forced"` every place whose bit is 1 is stimulated once
        at the epoch's moment: the behaviour this has always had. Under
        `drive = "rate"` each place is an independent Poisson process across
        the epoch, at `input_rate` where its bit is 1 and `input_rate_off`
        where it is 0, so a bit is a firing rate rather than a mandated spike
        and a bit-1 neuron may produce no spike at all. The draws come from
        the network's own seeded stream, in place order, so a seed reproduces
        them and both engines draw the same train. The refractory period drops
        whatever it drops when a stimulus arrives, exactly as before.
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
                                  trace=self.rule == "adaline", explore=self.explorer())
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
                                 trace=self.rule == "adaline", explore=self.explorer())

    def reset(self, discharge: bool = False) -> None:
        """Start a new epoch: clear every neuron's fired-this-epoch state and this epoch's waves.

        Potentials are kept (there is no leak); with `discharge=True` every
        potential is zeroed instead. Signals in flight stay scheduled.
        """
        for neuron in self.all_neurons():
            neuron.reset(discharge)
        if self.rule in ("teacher", "adaline"):
            for connection in self.connections.values():
                connection.eligibility = 0.0  # a new epoch earns its own credit
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
            neuron.potential = max(neuron.minimum_potential, neuron.potential + draw)

    def _explore(self, time: float) -> None:
        """The per-wave exploration draw (§6.1), called before each wave's firing decision."""
        self.perturb(self.sigma, self.explore_rng, time, hold_fired=True)

    def explorer(self):
        """The hook Schedule.run calls before each wave fires, or None when nothing is exploring."""
        return self._explore if (self.explore == "wave" and self.sigma > 0.0 and self.explore_rng is not None) else None

    def fired_neurons(self) -> list[Neuron]:
        """Return the neurons that have fired since the last reset."""
        return [n for n in self.all_neurons() if n.has_fired]
