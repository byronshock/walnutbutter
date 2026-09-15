"""Goo: butter with the plane taken away (AUTHORITY.md §3.4).

Every other container spends its structure on geometry. The hex grid wires
each cell to its two rings of neighbours, the columns extrude those cells
into R3, the lattice measures unit distances -- and all three then have to
say what "near" means before they can say what connects. Goo has no
positions at all, so there is no distance to measure and nothing to be near.

Its wiring is a rule about zones, and it is probabilistic (Byron, September
14, 2026):

    P(neuron i projects onto neuron j) = 0             if i == j
                                       = 0             if i and j are both in a zone
                                       = GOO_PROJECTION otherwise

where the zones are the input zone, the first `across` neurons, and the
output zone, the last `across`. Every projection is one-way and each
direction is its own draw, as every connection in this system is (§3). So
the zones never talk to each other directly -- not input to output, not
output to input, not within a zone -- and everything else, every ordered
pair with an interior end, projects with probability GOO_PROJECTION. A copy
has to go through the interior. At GOO_PROJECTION = 1 nothing is drawn to
decide the topology and `count` fixes it; below 1 the seed's stream decides,
pair by pair in (i, j) order, the projection draw and then the weight.

With no rows there is no bottom row to be the input, so the zones go by
index and are addressed the way every other container's rows are --
`get_neuron_at(place, 1)` in, `get_neuron_at(place, 0)` out -- so `rows` is
2 because it is counting those two zones, not any depth. A goo needs an
interior for its zones to talk through, so `count` must exceed the two zones
together; the output zone is `across` wide unless `outputs` says otherwise
(196 in and 30 out for the digits of §8).

Goo rescales its potential axis by each neuron's own fan-in (§5.2): an
interior neuron hears every other neuron, a zone neuron only the interior,
so the two start at different thresholds and the same threshold per synapse.
"""

from __future__ import annotations

import random
from typing import Iterator

from .constants import (
    ACROSS, GOO_COUNT, GOO_MINIMUM_POTENTIAL, GOO_PROJECTION, GOO_THRESHOLD, THRESHOLD_FAN_IN, WEIGHT_RANGE,
)
from .connection import Connection
from .network import Network
from .neuron import Neuron

DEFAULT_COUNT = GOO_COUNT  # sixty since September 14, 2026 (§1.2); it was the grid's eighty while the two were compared
ZONES = 2  # what `rows` counts in goo: the input zone and the output zone, and no depth between them


class Goo(Network):
    """`count` neurons with no positions, wired by the zone rule of AUTHORITY.md §3.4. Zones by index."""

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
    ):
        """Make the goo and wire it by the zone rule.

        `count` is how many neurons (GOO_COUNT, sixty); `across` the width of
        the input zone, the first `across` neurons, and `outputs` the width of
        the output zone, the last `outputs` (the same as `across` unless
        given: a task with more inputs than classes, §8's mnist, asks for
        different widths). `count` must exceed the two together so there is
        an interior for the zones to talk through. `projection` is the probability an ordered pair with an
        interior end projects; pairs with both ends in a zone never do. At 1
        the topology is fixed by `count`; below 1 the seed decides it.

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
        if count <= across + outputs:
            raise ValueError(
                f"goo needs an interior for its zones to talk through: count must exceed the zones together, "
                f"got {count} for {across} in and {outputs} out"
            )
        if not 0.0 < projection <= 1.0:
            raise ValueError(f"the projection probability must be in (0, 1], got {projection}")
        self.across = across  # the width of the input zone, not a count of cells: goo has no cells
        self.outputs = outputs  # the width of the output zone
        self.rows = ZONES  # row 1 is the input zone and row 0 the output zone; there is nothing in between
        self.count = count
        self.projection = projection  # P(i projects onto j) for a pair with an interior end
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

    def _wire(self) -> None:
        """The zone rule, pair by pair in (i, j) order: a draw where one is needed, then the weight."""
        low, high = self.weight_range
        zone = self.zone_indices()
        for i, source in enumerate(self.neurons):
            in_zone = i in zone
            for j, target in enumerate(self.neurons):
                if i == j:
                    continue  # no neuron projects onto itself (§3)
                if in_zone and j in zone:
                    continue  # the zones never project onto each other
                if self.projection < 1.0 and self._rng.random() >= self.projection:
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
        """Every neuron in neither zone: the only route from the input zone to the output zone."""
        zone = self.zone_indices()
        return [n for i, n in enumerate(self.neurons) if i not in zone]

    def zones_are_apart(self) -> bool:
        """True when no projection has both ends in a zone, as the rule requires."""
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
        """How far an interior neuron's potential axis is stretched at projection 1: (count - 1) / THRESHOLD_FAN_IN."""
        return (self.count - 1) / THRESHOLD_FAN_IN

    # --- container protocol -----------------------------------------------

    def __len__(self) -> int:
        return len(self.neurons)

    def __iter__(self) -> Iterator[Neuron]:
        return iter(self.neurons)

    def __getitem__(self, index: int) -> Neuron:
        return self.neurons[index]

    def __repr__(self) -> str:
        return (f"Goo({self.count} neurons, {len(self.connections)} projections at P {self.projection:g}; "
                f"{self.across} in, {self.outputs} out, zones apart)")
