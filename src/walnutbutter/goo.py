"""Goo: butter with the plane taken away (AUTHORITY.md §3.4).

Every other container spends its structure on geometry. The hex grid wires
each cell to its two rings of neighbours, the columns extrude those cells
into R3, the lattice measures unit distances -- and all three then have to
say what "near" means before they can say what connects. Goo has no
positions at all, so there is no distance to measure and nothing to be near:
what is left is the only wiring that needs no ruler, every ordered pair.

That makes it the control the others are measured against. On the 8 x 10
grid the input row is nine hex steps from the output row, five hops of the
guaranteed neighbourhood, and the small-world shortcuts are the only thing
that shortens the trip. Goo has no far: the output zone is one hop from the
input zone and from every other neuron. So goo and the grid at the same
neuron count differ in exactly two things, locality and depth, and whatever
the grid scores above goo is what those two are worth.

With no rows there is no bottom row to be the input, so the zones go by
index: the first `across` neurons are the input zone and the last `across`
the output zone. Both are addressed the way every other container's rows
are -- `get_neuron_at(place, 1)` in, `get_neuron_at(place, 0)` out -- so
`rows` is 2 because it is counting those two zones, not any depth, and
nothing above the container has to know the difference. Small goo overlaps
its zones and `Goo(count=across)` reads the neurons it writes, which §2
permits rather than prevents.

Nothing is drawn to decide the topology: `count` fixes it completely, and
the seeded stream is spent only on the weights and the permutation.
"""

from __future__ import annotations

import random
from typing import Iterator

from .constants import ACROSS, MINIMUM_POTENTIAL, ROWS, THRESHOLD, WEIGHT_RANGE
from .connection import Connection
from .network import Network
from .neuron import Neuron

DEFAULT_COUNT = ACROSS * ROWS  # the default network's eighty, so goo and the grid compare at equal size
ZONES = 2  # what `rows` counts in goo: the input zone and the output zone, and no depth between them


class Goo(Network):
    """`count` neurons with no positions, every ordered pair connected. Zones by index (AUTHORITY.md §3.4)."""

    def __init__(
        self,
        count: int = DEFAULT_COUNT,
        across: int = ACROSS,
        weight: float | None = 1.0,
        threshold: float = THRESHOLD,
        seed: int | None = None,
        permute: bool = True,
        weight_range: tuple[float, float] = WEIGHT_RANGE,
        minimum_potential: float = MINIMUM_POTENTIAL,
    ):
        """Make the goo and wire it.

        `count` is how many neurons; the default is ACROSS x ROWS, the same
        eighty the default grid has. `across` is the width of the input and
        output zones: the first `across` neurons read the input and the last
        `across` are read as the output, so anything from 2 x `across`
        neurons up keeps the zones apart and smaller goo overlaps them on
        purpose.

        `weight` is given to every connection; None draws each independently
        and uniformly from `weight_range`, in connection-id order from the
        seeded stream (§3.3). `permute` shuffles which coded bit each place
        of the input zone shows, once and for the life of the goo. There are
        no shortcuts to draw and no positions to place, so `seed` reaches
        only those weights, that permutation and the inputs.
        """
        if across < 1:
            raise ValueError(f"goo needs an input zone at least one neuron wide, got {across}")
        if count < across:
            raise ValueError(f"goo needs at least as many neurons as its zones are wide, got {count} for {across} across")
        self.across = across  # the width of each zone, not a count of cells: goo has no cells
        self.rows = ZONES  # row 1 is the input zone and row 0 the output zone; there is nothing in between
        self.count = count
        self.weight = weight  # fixed weight for every connection, or None for random
        self.threshold = threshold
        self.minimum_potential = minimum_potential
        self.omega = 0.0  # no shortcuts: with everything already connected there is nowhere for one to go
        self.seed = seed
        self._rng = random.Random(seed)  # weights, then the permutation
        self.neurons: list[Neuron] = []
        self.connections: dict[int, Connection] = {}  # by id from 1, like every other container
        self._init_network(across, weight_range)
        self._make(count)
        self._wire()
        if permute:
            self._rng.shuffle(self.permutation)

    # --- building ---------------------------------------------------------

    def _make(self, count: int) -> None:
        for index in range(count):
            self.neurons.append(
                Neuron(f"Goo_{index}", threshold=self.threshold, minimum_potential=self.minimum_potential)
            )

    def _wire(self) -> None:
        """Every ordered pair, source index then target index, so the count alone fixes the ids."""
        low, high = self.weight_range
        for source in self.neurons:
            for target in self.neurons:
                if target is source:
                    continue  # no neuron connects to itself (§3)
                weight = self._rng.uniform(low, high) if self.weight is None else self.weight
                connection_id = len(self.connections) + 1
                self.connections[connection_id] = source.connect(target, connection_id, weight, kind="goo")

    # --- the zones --------------------------------------------------------

    def get_neuron_at(self, place: int, row: int) -> Neuron | None:
        """Row 1 is the input zone (the first `across` neurons), row 0 the output zone (the last `across`).

        Nothing else is addressable: the goo between the zones has no place
        and no row, which is the whole of what makes it goo.
        """
        if not 0 <= place < self.across:
            return None
        if row == 1:
            return self.neurons[place]
        if row == 0:
            return self.neurons[self.count - self.across + place]
        return None

    def zones_overlap(self) -> bool:
        """True when the goo is too small to keep its input and output zones apart."""
        return self.count < 2 * self.across

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
        """Exactly count - 1, and worth asserting rather than assuming."""
        return sum(len(n.outgoing) for n in self.neurons) / len(self.neurons) if self.neurons else 0.0

    # --- container protocol -----------------------------------------------

    def __len__(self) -> int:
        return len(self.neurons)

    def __iter__(self) -> Iterator[Neuron]:
        return iter(self.neurons)

    def __getitem__(self, index: int) -> Neuron:
        return self.neurons[index]

    def __repr__(self) -> str:
        zones = f"{self.across} in, {self.across} out"
        if self.zones_overlap():
            zones += ", overlapping"
        return f"Goo({self.count} neurons fully connected, {len(self.connections)} connections; {zones})"
