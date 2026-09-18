"""Core logic: build a network, present input patterns on its input zone, and propagate.

This module knows nothing about command-line arguments. Keeping the logic
separate from the CLI makes it easy to test and to reuse from other code.
"""

from __future__ import annotations

import random
from typing import Sequence

from .constants import ACROSS, GOO_COUNT, GOO_MINIMUM_POTENTIAL, GOO_THRESHOLD, WEIGHT_RANGE
from .goo import Goo
from .inputs import format_bits
from .network import Network


def run_epoch(
    network: Network,
    bits: Sequence[bool] | None = None,
    verbose: bool = True,
    noise: float = 0.0,
    rng: random.Random | None = None,
    discharge: bool = False,
    time: float | None = None,
) -> list:
    """One epoch: present an input (random unless `bits` is given) at `time`, and run the schedule to the next input's time.

    Weights and thresholds are untouched by this function (the dopamine
    rule, if the network has one, learns as it runs; the teacher rule
    accumulates eligibility for the Teacher to pay at the read). Every
    neuron's fired-this-epoch state is cleared; potentials are kept (there
    is no leak), or zeroed with `discharge=True` (AUTHORITY.md §3.9). `time`
    defaults to the network's interval after the last input. With `noise` > 0
    a Gaussian draw of that standard deviation (the exploration) is added to
    every neuron's potential; each neuron remembers it as `noise`. Prints the
    input unless `verbose` is False. Returns the epoch's waves.
    """
    network.reset(discharge)
    if bits is None:
        network.new_random_input(time)
    else:
        network.set_input_bits(bits, time)
    network.sigma, network.explore_rng = noise, rng or random  # §6.1: wave exploration draws inside the schedule
    if noise > 0 and getattr(network, "explore", "wave") == "epoch":
        network.perturb(noise, rng or random)
    if verbose:
        stages = f"input {format_bits(network.input_bits)}"
        print(
            f"epoch {network.epoch + 1} at {network.input_time:g} ms: {stages}"
            f" -> coded {format_bits(network.input_coded)} -> input zone {format_bits(network.input_pattern)}"
        )
    return network.fire_input()


def main(
    count: int = GOO_COUNT,
    across: int = ACROSS,
    weight: float | None = None,
    threshold: float = GOO_THRESHOLD,
    seed: int | None = None,
    input_bits: Sequence[bool] | None = None,
    permute: bool = True,
    weight_range: tuple[float, float] = WEIGHT_RANGE,
    minimum_potential: float = GOO_MINIMUM_POTENTIAL,
) -> Goo:
    """Build a goo of `count` neurons, run one epoch on its input zone, and return it.

    `weight` is given to every connection, or None (the default) for random
    weights uniform between -1 and 1. `threshold` is given to every neuron.
    `input_bits` are the raw input bits (across / 2 of them); if None they
    are drawn at random. They are complement-coded and, with `permute`,
    scrambled by a permutation fixed for the run. `seed` makes the wiring,
    the weights, the permutation and the random inputs all reproducible.
    Returning the network lets callers (and tests) inspect which neurons
    fired.
    """
    network = Goo(
        count=count,
        across=across,
        weight=weight,
        threshold=threshold,
        seed=seed,
        permute=permute,
        weight_range=weight_range,
        minimum_potential=minimum_potential,
    )
    run_epoch(network, input_bits)
    return network
