"""The problems: what a network is asked to do, and how it is watched doing it.

A problem names the layout, the inputs, the spacing of inputs, what is read
as the output and when, and whether the Teacher only scores it or may also
train it (the reinforce rule, homeostasis, un-sticking). The command line
picks one with --problem; every other option still applies on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import ACROSS, FLIP, ROWS


@dataclass(frozen=True)
class Problem:
    name: str
    description: str
    across: int  # cells across, unless --across (or --ecc) says otherwise
    rows: int
    trained: bool  # True: the Teacher may train (reinforce rule, homeostasis, un-sticking); False: it only scores
    interval: float | None = None  # ms between inputs: the epoch's length (None: INTERVAL)
    readout: str = "top"  # which neurons are read as the output: "top" (the top row) or "input" (the inputs are the outputs)
    read: str = "fired"  # what "on" means at the read: "fired" this epoch, "again" (spiked after the input's moment), or
    # "window" (within read_window ms before the epoch's end)
    read_window: float | None = None  # the window for read == "window"
    target: str | None = None  # what the output should show (None: --target)
    critic: str | None = None  # how the read is scored (None: --critic)
    coding: str = "complement"  # how raw bits reach the input row (§4.3): complement, raw, population, population-complement
    population: int | None = None  # neurons per raw bit where the coding repeats it (None: constants.POPULATION)
    reach: int = 2  # hex steps the grid's local wiring covers
    input_cells: tuple | None = None  # an input zone, (place, row) cells counted from 0, in place of the bottom row
    permute: bool = True  # scramble the coded bits over the input neurons with a fixed permutation
    rule: str | None = None  # the learning rule this problem is posed for (None: --rule, else constants.RULE)
    quash: bool = True  # quash cycles (§6.11); False switches it off for this problem
    hebb: bool = False  # run leaky Hebb (§6.12) alongside whatever else this problem runs
    drive: str | None = None  # how a bit becomes spikes (§4.3): "forced" or "rate" (None: --drive, else constants.INPUT_DRIVE)
    flip: float | None = None  # corrupt the input: flip each coded bit with this probability (§4.3); None means no corruption
    direct_projection: bool = True  # False: no connection runs from an input neuron to an output neuron (§3.4); goo builds it so


PROBLEMS: dict[str, Problem] = {
    "reversal": Problem(
        "reversal",
        "the top row learns to show the bottom row reversed, taught by a Teacher with a target and a critic",
        ACROSS, ROWS, trained=True, rule="reinforce", quash=False,
    ),
    "copy": Problem(
        "copy",
        "an input is complement-coded and presented on the input neurons, and the desired output is exactly the "
        "input expressed across the output neurons (Byron, September 14, 2026): eight in, eight out, output place i "
        "taught to show coded bit i. Unpermuted: 'permuting patterns should no longer matter, all neurons are "
        "first-class citizens of the population' (Byron, same day), and on goo a permutation of the input zone is only "
        "a relabelling of identically wired neurons. Reversal with the target set to copy and the permutation off; the "
        "task the goo comparisons of AUTHORITY.md §3.4 are posed on",
        ACROSS, ROWS, trained=True, target="copy", critic="row", rule="reinforce", quash=False, permute=False,
        read="count",  # Byron, September 14, 2026: count the epoch's spikes, estimate the rate, threshold it (§4.3)
        direct_projection=False,  # and the same day: "INPUT NEURONS DO NOT PROJECT DIRECTLY ONTO OUTPUT NEURONS" (§3.4)
    ),
    "sustain_inputs": Problem(
        "sustain_inputs",
        "the 16 four-bit inputs laid down as they are on 4 neurons (no complement coding, so 0000 forces nothing and "
        "1111 forces all four); an input is forced, the mesh reverberates for one epoch, and the same neurons are read "
        "(the inputs are the outputs): on if they spiked again after the input's moment. The score "
        "is the fraction of the four whose read state matches the pattern: the forced ones on, the others off. "
        "Scored by the Teacher, not trained by it: the neurons learn by dopamine (AUTHORITY.md §6, §8)",
        4, ROWS, trained=False, readout="input", read="again", target="copy", critic="row", coding="raw",
        rule="teacher", quash=False,
    ),
    "improved_sustain": Problem(
        "improved_sustain",
        "sustain_inputs on a different topology (Byron, September 12, 2026): a 10-across, 7-row hex grid wired to reach 3, "
        "the 4 raw input bits presented in the middle, row 4 places 4 to 7 counted from 1 (row 3, places 3 to 6 from 0), "
        "no permutation; the same neurons read back (spiked again), the row critic, learning by dopamine",
        10, 7, trained=False, readout="input", read="again", target="copy", critic="row", coding="raw",
        reach=3, input_cells=((3, 3), (4, 3), (5, 3), (6, 3)), permute=False, rule="teacher", quash=False,
    ),
    "population_copy": Problem(
        "population_copy",
        "population coding (Byron, September 13, 2026): the 4 raw bits each fill three neurons of a 12-wide bottom row, "
        "so 1001 lands as 111000000111, and the top row of a 10-row grid should show the same code. The teacher scores "
        "the top row in [-1, 1], six of twelve right being zero. Cycles are quashed (§6.11): a refire weakens the "
        "synapses that contributed to it",
        12, 10, trained=False, readout="top", read="fired", target="copy", critic="row",
        coding="population", permute=False, rule="teacher", quash=True,
    ),
    "shallow_copy": Problem(
        "shallow_copy",
        "population_copy shrunk to the smallest network that still has an input row and an output row (Byron, "
        "September 13, 2026): twelve across, TWO rows, the four raw bits population-coded onto the bottom and the top "
        "read as fired this epoch. 24 neurons and 215 connections against 120 and 2,195, and the task is one hop wide: "
        "a rule that can learn anything should learn this, and one that cannot will not be rescued by depth. The floor "
        "the rules are measured against (AUTHORITY.md §8)",
        12, 2, trained=False, readout="top", read="fired", target="copy", critic="row",
        coding="population", permute=False, rule="teacher", quash=True,
    ),
    "shallow_not": Problem(
        "shallow_not",
        "shallow_copy, only NOT (Byron, September 14, 2026): the same 12-wide population-coded input on the same grid, "
        "read the same way, but the top row should show the COMPLEMENT of the code. Where copy can be had by excitation "
        "alone, this asks a neuron to fire because nothing told it to: silence propagates no signal, so no weight on any "
        "incoming connection can drive an output whose whole input group is quiet. The only thing in the system that "
        "turns silence into a spike is the bored-neuron threshold of §5.4, calibrated at 200 ms against a 35 ms epoch "
        "(AUTHORITY.md §8)",
        12, 2, trained=False, readout="top", read="fired", target="complement", critic="row",
        coding="population", permute=False, rule="teacher", quash=True,
    ),
    "doubled_copy": Problem(
        "doubled_copy",
        "the four raw bits doubled, complement-coded and scrambled (Byron, September 14, 2026): 1001 becomes 11000011, "
        "then 1100001100111100, then a consistent random permutation spreads those 16 bits over a 16-wide input row, and "
        "the top row of an 8-row grid should show the same code. Every input fires exactly half the row whatever the "
        "bits, and the permutation leaves adjacency carrying nothing, so neither total activity nor position is a clue "
        "(AUTHORITY.md §8)",
        16, 8, trained=False, readout="top", read="fired", target="copy", critic="row",
        coding="population-complement", population=2, permute=True, rule="teacher", quash=True,
    ),
    "reaching_copy": Problem(
        "reaching_copy",
        "doubled_copy's inputs on a shallower, denser grid (Byron, September 14, 2026): the same sixteen coded and "
        "permuted bits, copied to the top of a FIVE-row grid wired to REACH 5. The task is a copy — output place i "
        "against input place i, the permutation scrambling what the input row is shown rather than what the top row "
        "must answer. Four hex steps separate the bottom row from the top, which is inside the reach, so every input "
        "neuron synapses directly onto the neuron above it and all sixteen routes are one hop: the mesh stops being a "
        "depth to relay through. What it costs is density (80 neurons, 3,122 local connections, mean out-degree 39 "
        "against 13.1 at reach 2) (AUTHORITY.md §8)",
        16, 5, trained=False, readout="top", read="fired", target="copy", critic="row",
        coding="population-complement", population=2, reach=5, permute=True, rule="teacher", quash=True,
    ),
    "population_denoise": Problem(
        "population_denoise",
        "population_copy's network read somewhere else (Byron, September 13, 2026): the same 12-wide, 10-row grid, the "
        "same population coding, the same teacher and the same quash, but each of the twelve coded bits is flipped with "
        "probability 1/12 on the way in, and it is the INPUT zone that is read back, on meaning spiked again after the "
        "input's moment. The score is against the CLEAN code, so the network is asked to repair its input: 0 for silence, "
        "0.833 for carrying the corruption through faithfully, 1 only for correcting it (AUTHORITY.md §4.3, §8)",
        12, 10, trained=False, readout="input", read="again", target="copy", critic="row",
        coding="population", permute=False, rule="teacher", quash=True, flip=FLIP,
    ),
}
