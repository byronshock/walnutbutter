"""Goo: butter with the plane taken away (AUTHORITY.md §3.4).

Every other container spends its structure on geometry. The hex grid wires
each cell to its two rings of neighbours, the columns extrude those cells
into R3, the lattice measures unit distances -- and all three then have to
say what "near" means before they can say what connects. Goo has no
positions at all, so there is no distance to measure and nothing to be near.

Its wiring is the scaled rule (Byron, September 16, 2026: "P(i projects
onto j) = 0 if i == j; 0 if i and j are both in the input zone; P_ij
necessary to give j an average of N * scaling_factor inputs"), with the
outputs kept apart (Byron, the same night: "I would like the outputs kept
apart from one another again. With hidden=0 we have no cycles ... a
two-layer feedforward network"):

    P(neuron i projects onto neuron j) = 0                   if i == j
                                       = 0                   if i and j are both in the input zone
                                       = 0                   if i is an output and j is in either zone
                                       = min(1, N s / A_j)   otherwise

where N is the count, s is GOO_SCALING_FACTOR, and A_j is the sources j may
hear: the N - I - O hidden neurons for an input, the N - O inputs and
hidden neurons for an output, the N - 1 others for a hidden neuron, I and
O being the zones' widths. Every projection is one-way and each direction
is its own draw, as every connection in this system is (§3). So every
neuron with sources enough hears N s synapses in expectation -- 32.2 on
the mnist goo of 644, 3 on goo 60 -- and the density is the same at any
count. An output hears the inputs directly and the hidden neurons, and
projects onto the hidden neurons alone; an input hears the hidden neurons
alone; with no hidden neurons the goo is inputs -> outputs and nothing
else, a two-layer feedforward network, its inputs hearing nothing (and
left at the container's threshold, §5.2). No interior is needed, only
that the zones not overlap. Nothing is drawn where the probability is 0
or 1; otherwise the seed's stream decides, pair by pair in (i, j) order,
the projection draw and then the weight.

With no rows there is no bottom row to be the input, so the zones go by
index and are addressed the way every other container's rows are --
`get_neuron_at(place, 1)` in, `get_neuron_at(place, 0)` out -- so `rows` is
2 because it is counting those two zones, not any depth. The input zone is
the first `across` neurons and the output zone the last `outputs`.

A sixth wiring, "ff2" -- fully-connected-feedforward-2 (Byron, September 16,
2026, 16:35 MDT: "There are two 'layers', the input 'layer' and the output
'layer'. P_ij = P(i is in the input layer and j is in the output layer)
... for the feedforward problem I want everything to connect fully") --
is exactly that: P(i -> j) = 1 when i is an input and j an output, 0
otherwise, so every input projects onto every output and nothing else
projects at all. It wires two layers and no more: a goo with hidden
neurons is refused under it (how the layers generalise to the zones of a
recurrent goo is Byron's to say). The potential axis still scales with
the fan-in, which under it is the whole input zone.

Four earlier wirings are kept so that their checkpoints restore and their
runs can be repeated. "scaled-open" is the night's first scaled rule, the
outputs open to every zone (an output heard the other outputs and the
inputs heard the outputs), which the two temperature sweeps of §8 ran
under. The three before it have GOO_PROJECTION as their knob: "zones-equal",
the zone rule of September 14 with the equal fan-in of the 15th on top (the
zones apart -- no projection with both ends in a zone -- and an interior-
to-zone projection scaled up to min(1, P (N - 1) / H) so a zone neuron
hears what an interior neuron does), which was the rule until the 16th;
"zones", that rule without the equal fan-in; and "uniform", one probability
over every ordered pair, an evening's rule on the 15th. The two zone
wirings need an interior for their zones to talk through.

Goo rescales its potential axis by each neuron's own fan-in (§5.2).
"""

from __future__ import annotations

import random
from typing import Iterator

from .constants import (
    ACROSS, GOO_COUNT, GOO_MINIMUM_POTENTIAL, GOO_PROJECTION, GOO_SCALING_FACTOR, GOO_THRESHOLD, THRESHOLD_FAN_IN,
    WEIGHT_RANGE,
)
from .connection import Connection
from .network import Network
from .neuron import Neuron

DEFAULT_COUNT = GOO_COUNT  # sixty since September 14, 2026 (§1.2); it was the grid's eighty while the two were compared
ZONES = 2  # what `rows` counts in goo: the input zone and the output zone, and no depth between them
WIRINGS = ("scaled", "ff2", "scaled-open", "zones-equal", "zones", "uniform")  # the rule (§3.4), the fully connected feedforward
# rule beside it (September 16, 2026), and the four before them, kept so their checkpoints restore
SCALED_WIRINGS = ("scaled", "scaled-open")  # the rule and the night's open version of it: N s heard in expectation
ZONE_WIRINGS = ("zones-equal", "zones")  # the wirings whose zones talk only through an interior, and so need one


def scaled_projection(count: int, across: int, outputs: int, scaling_factor: float, zone: str, open: bool = False) -> float:
    """The scaled rule's P(i -> j) (§3.4): N s over the sources j may hear, stopped at 1; 0 where it has none.

    `zone` is j's: "input", "hidden" or "output". Under the rule an input may
    hear the N - I - O hidden neurons, an output the N - O inputs and hidden
    neurons, a hidden neuron the N - 1 others. Under the night's open rule
    (`open`) an input may hear the N - I outside its zone and any other
    neuron the N - 1 others.
    """
    if open:
        sources = count - across if zone == "input" else count - 1
    else:
        sources = {"input": count - across - outputs, "output": count - outputs, "hidden": count - 1}[zone]
    if sources <= 0:
        return 0.0
    return min(1.0, count * scaling_factor / sources)


class Goo(Network):
    """`count` neurons with no positions, wired by the scaled rule of AUTHORITY.md §3.4. Zones by index."""

    def __init__(
        self,
        count: int = DEFAULT_COUNT,
        across: int = ACROSS,
        weight: float | None = 1.0,
        threshold: float = GOO_THRESHOLD,
        seed: int | None = None,
        permute: bool = True,
        weight_range: tuple[float, float] = WEIGHT_RANGE,
        minimum_potential: float = GOO_MINIMUM_POTENTIAL,
        scale_with_fan_in: bool = True,
        projection: float = GOO_PROJECTION,
        outputs: int | None = None,
        wiring: str = "scaled",
        scaling_factor: float = GOO_SCALING_FACTOR,
    ):
        """Make the goo and wire it by the rule (§3.4): the scaled rule with the outputs apart.

        `count` is how many neurons (GOO_COUNT, sixty); `across` the width of
        the input zone, the first `across` neurons, and `outputs` the width of
        the output zone, the last `outputs` (the same as `across` unless
        given: a task with more inputs than classes, §8's mnist, asks for
        different widths). The zones may not overlap. `scaling_factor` is the
        rule's knob: every neuron hears `count` times it in expectation, from
        everyone but itself and, for an input neuron, but the rest of its
        zone. `wiring` is "scaled", the rule; "scaled-open" (the night's first
        scaled rule, the outputs open to every zone), "zones-equal" (the zone
        rule with equal fan-in, the rule until September 16, 2026), "zones"
        (the plain zone rule) and "uniform" (one probability over every pair)
        are the four before it (see the module docstring), kept so their
        checkpoints restore; `projection` is the last three's probability, and
        the two zone wirings need an interior, so under them `count` must
        exceed the zones together.

        `threshold` and `minimum_potential` default to goo's own constants
        (§1.2), and `scale_with_fan_in` rescales each neuron's potential axis
        by its own in-degree over THRESHOLD_FAN_IN (§5.2). `weight` is given
        to every projection; None draws each uniformly from `weight_range`.
        `permute` shuffles which coded bit each place of the input zone
        shows. The seed's stream is spent, in order, on the projection draws
        and weights pair by pair, then the permutation, then the inputs.
        """
        outputs = across if outputs is None else outputs
        if across < 1 or outputs < 1:
            raise ValueError(f"goo needs zones at least one neuron wide, got {across} in and {outputs} out")
        if wiring not in WIRINGS:
            raise ValueError(f"unknown wiring {wiring!r}; choose from {', '.join(WIRINGS)}")
        if wiring in ZONE_WIRINGS and count <= across + outputs:
            raise ValueError(
                f"the {wiring} wiring needs an interior for its zones to talk through: count must exceed the zones "
                f"together, got {count} for {across} in and {outputs} out"
            )
        if wiring == "ff2" and count != across + outputs:
            raise ValueError(f"the ff2 wiring is two layers, every input onto every output and nothing else: count must be the "
                             f"zones together, got {count} for {across} in and {outputs} out (no hidden neurons under it yet)")
        if count < across + outputs:
            raise ValueError(f"the zones would overlap: count must be at least the zones together, got {count} for "
                             f"{across} in and {outputs} out")
        if not 0.0 < projection <= 1.0:
            raise ValueError(f"the projection probability must be in (0, 1], got {projection}")
        if scaling_factor <= 0.0:
            raise ValueError(f"the scaling factor must be positive, got {scaling_factor}")
        self.across = across  # the width of the input zone, not a count of cells: goo has no cells
        self.outputs = outputs  # the width of the output zone
        self.rows = ZONES  # row 1 is the input zone and row 0 the output zone; there is nothing in between
        self.count = count
        self.scaling_factor = scaling_factor  # the rule's knob: every neuron hears count times this in expectation
        self.projection = projection  # the three earlier wirings' probability; the rule does not read it
        self.wiring = wiring  # the rule, or one of the three before it
        self.weight = weight  # fixed weight for every projection, or None for random
        self.threshold = threshold
        self.minimum_potential = minimum_potential
        self.omega = 0.0  # no shortcuts: goo has no neighbourhood for a shortcut to get past
        self.seed = seed
        self._rng = random.Random(seed)  # the projection draws and weights, then the permutation
        self.neurons: list[Neuron] = []
        self.connections: dict[int, Connection] = {}  # by id from 1, like every other container
        self._init_network(across, weight_range)
        self._make(count)
        self._wire()
        self.scale_with_fan_in_on = scale_with_fan_in
        if scale_with_fan_in:
            self.scale_with_fan_in(threshold, minimum_potential)  # after the wiring: it reads the in-degree
        if permute:
            self._rng.shuffle(self.permutation)

    # --- building ---------------------------------------------------------

    def _make(self, count: int) -> None:
        for index in range(count):
            self.neurons.append(
                Neuron(f"Goo_{index}", threshold=self.threshold, minimum_potential=self.minimum_potential)
            )

    def zone_indices(self) -> set[int]:
        """The indices of every neuron in the input zone or the output zone."""
        return set(range(self.across)) | set(range(self.count - self.outputs, self.count))

    def interior_count(self) -> int:
        return self.count - self.across - self.outputs

    def expected_fan_in(self) -> float:
        """What a neuron hears in expectation: N s under the scaled rules; the whole input zone for an output under ff2; P (N - 1)
        for an interior neuron under the zone rules."""
        if self.wiring in SCALED_WIRINGS:
            return self.count * self.scaling_factor
        if self.wiring == "ff2":
            return float(self.across)
        return self.projection * (self.count - 1)

    def zone_of(self, k: int) -> str:
        """Which zone neuron k is in by index: "input", "output" or "hidden"."""
        if k < self.across:
            return "input"
        return "output" if k >= self.count - self.outputs else "hidden"

    def input_projection(self) -> float:
        """The rule's P(i -> j) for j in the input zone: N s over the hidden neurons (over N - I under the open rule), stopped at 1; 0 under ff2."""
        if self.wiring == "ff2":
            return 0.0
        return scaled_projection(self.count, self.across, self.outputs, self.scaling_factor, "input", self.wiring == "scaled-open")

    def output_projection(self) -> float:
        """The rule's P(i -> j) for j in the output zone: N s over the inputs and hidden neurons (over N - 1 under the open rule); 1 under ff2."""
        if self.wiring == "ff2":
            return 1.0
        return scaled_projection(self.count, self.across, self.outputs, self.scaling_factor, "output", self.wiring == "scaled-open")

    def hidden_projection(self) -> float:
        """The rule's P(i -> j) for a hidden j: N s / (N - 1), stopped at 1; there are none under ff2."""
        if self.wiring == "ff2":
            return 0.0
        return scaled_projection(self.count, self.across, self.outputs, self.scaling_factor, "hidden", self.wiring == "scaled-open")

    def zone_projection(self) -> float:
        """P(an interior neuron projects onto a zone neuron) under the three earlier wirings.

        "zones-equal": `projection` scaled up by (count - 1) / interior and
        stopped at 1, so that a zone neuron, hearing only the interior, hears
        what an interior neuron does. "zones" and "uniform": `projection`
        itself. Under the rule an interior neuron projects onto an input
        neuron at `input_projection()` and onto an output at `other_projection()`.
        """
        if self.wiring in SCALED_WIRINGS or self.wiring == "ff2":
            raise ValueError(f"the {self.wiring} rule has no one zone probability: see input_projection(), output_projection() and "
                             "hidden_projection()")
        if self.wiring != "zones-equal":
            return self.projection
        return min(1.0, self.projection * (self.count - 1) / self.interior_count())

    def projection_probability(self, i: int, j: int) -> float:
        """P(neuron i projects onto neuron j) under this goo's wiring (§3.4)."""
        if i == j:
            return 0.0  # no neuron projects onto itself (§3)
        if self.wiring == "ff2":  # two layers, fully connected, one way: every input onto every output and nothing else
            return 1.0 if self.zone_of(i) == "input" and self.zone_of(j) == "output" else 0.0
        if self.wiring in SCALED_WIRINGS:
            source, target = self.zone_of(i), self.zone_of(j)
            if source == "input" and target == "input":
                return 0.0  # the input zone does not talk to itself
            if self.wiring == "scaled" and source == "output" and target != "hidden":
                return 0.0  # an output projects onto no zone: not onto another output, not onto an input (the outputs apart)
            return scaled_projection(self.count, self.across, self.outputs, self.scaling_factor, target, self.wiring == "scaled-open")
        if self.wiring == "uniform":
            return self.projection
        zone = self._zone
        if i in zone:
            return 0.0 if j in zone else self.projection  # the zones never project onto each other
        return self.zone_projection() if j in zone else self.projection

    def _wire(self) -> None:
        """The rule, pair by pair in (i, j) order: a draw where one is needed, then the weight."""
        low, high = self.weight_range
        self._zone = self.zone_indices()
        for i, source in enumerate(self.neurons):
            for j, target in enumerate(self.neurons):
                p = self.projection_probability(i, j)
                if p <= 0.0:
                    continue
                if p < 1.0 and self._rng.random() >= p:
                    continue
                weight = self._rng.uniform(low, high) if self.weight is None else self.weight
                connection_id = len(self.connections) + 1
                self.connections[connection_id] = source.connect(target, connection_id, weight, kind="goo")

    # --- the zones --------------------------------------------------------

    def get_neuron_at(self, place: int, row: int) -> Neuron | None:
        """Row 1 is the input zone (the first `across` neurons), row 0 the output zone (the last `outputs`).

        Nothing else is addressable: the goo between the zones has no place
        and no row, which is the whole of what makes it goo.
        """
        if row == 1 and 0 <= place < self.across:
            return self.neurons[place]
        if row == 0 and 0 <= place < self.outputs:
            return self.neurons[self.count - self.outputs + place]
        return None

    def output_width(self) -> int:
        return self.outputs

    def interior(self) -> list[Neuron]:
        """Every neuron in neither zone: under the zone wirings, the only route from the input zone to the output zone."""
        zone = self.zone_indices()
        return [n for i, n in enumerate(self.neurons) if i not in zone]

    def input_zone_is_apart(self) -> bool:
        """True when no projection has both ends in the input zone, as the rule requires."""
        inputs = set(self.neurons[:self.across])
        return not any(c.source in inputs and c.target in inputs for c in self.connections.values())

    def outputs_are_apart(self) -> bool:
        """True when no output projects onto an output or an input, as the rule requires since the outputs were kept apart."""
        outs, ins = set(self.output_row()), set(self.input_row())
        return not any(c.source in outs and (c.target in outs or c.target in ins) for c in self.connections.values())

    def zones_are_apart(self) -> bool:
        """True when no projection has both ends in a zone, as the two zone wirings require (and the rule does not)."""
        zone = {self.neurons[i] for i in self.zone_indices()}
        return not any(c.source in zone and c.target in zone for c in self.connections.values())

    # --- looking at the wiring ---------------------------------------------

    def all_neurons(self) -> list[Neuron]:
        return self.neurons

    def small_world_connections(self) -> list[Connection]:
        """None, ever: a shortcut is a way past the neighbourhood and goo has no neighbourhood to get past."""
        return []

    def local_connections(self) -> list[Connection]:
        return list(self.connections.values())

    def connections_of_kind(self, kind: str) -> list[Connection]:
        return [c for c in self.connections.values() if c.kind == kind]

    def get_connection(self, connection_id: int) -> Connection | None:
        return self.connections.get(connection_id)

    def mean_out_degree(self) -> float:
        return sum(len(n.outgoing) for n in self.neurons) / len(self.neurons) if self.neurons else 0.0

    def fan_in_scale(self) -> float:
        """How far a neuron hearing everyone else has its potential axis stretched: (count - 1) / THRESHOLD_FAN_IN."""
        return (self.count - 1) / THRESHOLD_FAN_IN

    # --- container protocol -----------------------------------------------

    def __len__(self) -> int:
        return len(self.neurons)

    def __iter__(self) -> Iterator[Neuron]:
        return iter(self.neurons)

    def __getitem__(self, index: int) -> Neuron:
        return self.neurons[index]

    def __repr__(self) -> str:
        zones = f"{self.across} in, {self.interior_count()} hidden, {self.outputs} out"
        if self.wiring in SCALED_WIRINGS:
            apart = "zones apart, inputs onto outputs" if self.wiring == "scaled" else "input zone apart"
            return (f"Goo({self.count} neurons, {len(self.connections)} projections at scaling factor "
                    f"{self.scaling_factor:g}; {zones}, {apart})")
        if self.wiring == "ff2":
            return f"Goo({self.count} neurons, {len(self.connections)} projections, fully connected feedforward; {zones})"
        apart = ", zones apart" if self.wiring != "uniform" else ""
        return f"Goo({self.count} neurons, {len(self.connections)} projections at P {self.projection:g}; {zones}{apart})"
