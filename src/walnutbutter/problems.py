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
    coding: str = "complement"  # how raw bits reach the input row: "complement" (bits then their negations) or "raw"
    reach: int = 2  # hex steps the grid's local wiring covers
    input_cells: tuple | None = None  # an input zone, (place, row) cells counted from 0, in place of the bottom row
    permute: bool = True  # scramble the coded bits over the input neurons with a fixed permutation
    rule: str | None = None  # the learning rule this problem is posed for (None: --rule, else constants.RULE)
    quash: bool = True  # quash cycles (§6.11); False switches it off for this problem
    hebb: bool = False  # run leaky Hebb (§6.12) alongside whatever else this problem runs
    flip: float | None = None  # corrupt the input: flip each coded bit with this probability (§4.3); None means no corruption


PROBLEMS: dict[str, Problem] = {
    "reversal": Problem(
        "reversal",
        "the top row learns to show the bottom row reversed, taught by a Teacher with a target and a critic",
        ACROSS, ROWS, trained=True, rule="reinforce", quash=False,
    ),
    "sustain_inputs": Problem(
        "sustain_inputs",
        "the 16 four-bit inputs laid down as they are on 4 neurons (no complement coding, so 0000 forces nothing and "
        "1111 forces all four); an input is forced, the mesh reverberates for 20 ms, and the same neurons are read "
        "(the inputs are the outputs): on if they spiked again after the input's moment. The score "
        "is the fraction of the four whose read state matches the pattern: the forced ones on, the others off. "
        "Scored by the Teacher, not trained by it: the neurons learn by dopamine (AUTHORITY.md §6, §8)",
        4, ROWS, trained=False, interval=20.0, readout="input", read="again", target="copy", critic="row", coding="raw",
        rule="teacher", quash=False,
    ),
    "improved_sustain": Problem(
        "improved_sustain",
        "sustain_inputs on a different topology (Byron, September 12, 2026): a 10-across, 7-row hex grid wired to reach 3, "
        "the 4 raw input bits presented in the middle, row 4 places 4 to 7 counted from 1 (row 3, places 3 to 6 from 0), "
        "no permutation; 20 ms epochs, the same neurons read back (spiked again), the row critic, learning by dopamine",
        10, 7, trained=False, interval=20.0, readout="input", read="again", target="copy", critic="row", coding="raw",
        reach=3, input_cells=((3, 3), (4, 3), (5, 3), (6, 3)), permute=False, rule="teacher", quash=False,
    ),
    "population_copy": Problem(
        "population_copy",
        "population coding (Byron, September 13, 2026): the 4 raw bits each fill three neurons of a 12-wide bottom row, "
        "so 1001 lands as 111000000111, and the top row of a 10-row grid should show the same code. The teacher scores "
        "the top row in [-1, 1], six of twelve right being zero. Cycles are quashed (§6.11): a refire weakens the "
        "synapses that contributed to it",
        12, 10, trained=False, interval=20.0, readout="top", read="fired", target="copy", critic="row",
        coding="population", permute=False, rule="teacher", quash=True,
    ),
    "shallow_copy": Problem(
        "shallow_copy",
        "population_copy shrunk to the smallest network that still has an input row and an output row (Byron, "
        "September 13, 2026): twelve across, TWO rows, the four raw bits population-coded onto the bottom and the top "
        "read as fired this epoch. 24 neurons and 215 connections against 120 and 2,195, and the task is one hop wide: "
        "a rule that can learn anything should learn this, and one that cannot will not be rescued by depth. The floor "
        "the rules are measured against (AUTHORITY.md §8)",
        12, 2, trained=False, interval=20.0, readout="top", read="fired", target="copy", critic="row",
        coding="population", permute=False, rule="teacher", quash=True,
    ),
    "population_denoise": Problem(
        "population_denoise",
        "population_copy's network read somewhere else (Byron, September 13, 2026): the same 12-wide, 10-row grid, the "
        "same population coding, the same teacher and the same quash, but each of the twelve coded bits is flipped with "
        "probability 1/12 on the way in, and it is the INPUT zone that is read back, on meaning spiked again after the "
        "input's moment. The score is against the CLEAN code, so the network is asked to repair its input: 0 for silence, "
        "0.833 for carrying the corruption through faithfully, 1 only for correcting it (AUTHORITY.md §4.3, §8)",
        12, 10, trained=False, interval=20.0, readout="input", read="again", target="copy", critic="row",
        coding="population", permute=False, rule="teacher", quash=True, flip=FLIP,
    ),
}
