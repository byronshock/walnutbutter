"""The local rules: what runs inside the wave loop, under whatever pays at the read.

AUTHORITY.md §10.1 and §10.2. A local rule needs nothing but the neuron's own
spikes and the stamps on its own synapses; it runs inside the wave loop, it has
its own rate, and it is off until a run asks for it. **This specification carries
the quash alone** (§10.2), so this module holds one rule.
"""

from __future__ import annotations

import math


def quash(wave, rate: float, k: float, weight_range: tuple[float, float]) -> int:
    """A refire is a cycle: weaken the synapses that contributed to it (AUTHORITY.md §6.11). Returns synapses weakened.

    Byron, September 13, 2026: cycles need to be quashed, and the quash is
    proportional to the synaptic gating, the weight, and an exponential
    decay in the delay since the neuron's previous spike. For a neuron
    refiring at $t$ whose previous spike was at $t_{prev}$, every incoming
    synapse that carried a signal it integrated since that spike moves by

        w <- w - rate * w * exp(-k * (t - t_prev))

    which pulls the weight toward zero, hardest for the tightest loop. It is
    local (the neuron's own spikes and the stamps on its synapses), lazy
    (nothing happens until a refire) and needs no external signal, so it
    runs under every rule; `rate` 0 switches it off.
    """
    low, high = weight_range
    weakened = 0
    for neuron in wave.fired:
        previous = neuron.previous_fired_at
        if previous is None:
            continue  # its first spike: no cycle to quash
        factor = rate * math.exp(-k * (wave.time - previous))
        if not factor:
            continue
        for connection in neuron.incoming:
            if connection.is_active and connection.last_signal is not None and connection.last_signal > previous:
                weight = connection.weight * (1.0 - factor)
                connection.weight = low if weight < low else high if weight > high else weight
                weakened += 1
    return weakened
