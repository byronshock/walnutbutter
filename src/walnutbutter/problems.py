"""The problems: what a network is asked to do, and how it is watched doing it.

A problem names the layout, the inputs, the spacing of inputs, what is read
as the output and when, and whether the Teacher only scores it or may also
train it (the reinforce rule, homeostasis, un-sticking). The command line
picks one with --problem; every other option still applies on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import ACROSS, FLIP


def dataset_stream(name: str | None, seed: int):
    """A problem's dataset as an input stream with labels, or None when the inputs are random bits (§4.5)."""
    if name is None:
        return None
    if name == "mnist":
        from . import mnist
        return mnist.stream("train", seed)
    raise ValueError(f"no dataset called {name!r}")


@dataclass(frozen=True)
class Problem:
    name: str
    description: str
    across: int  # the input zone's width, unless --across (or --ecc) says otherwise
    trained: bool  # True: the Teacher may train (reinforce rule, homeostasis, un-sticking); False: it only scores
    interval: float | None = None  # ms between inputs: the epoch's length (None: INTERVAL)
    readout: str = "top"  # which neurons are read as the output: "top" (the output zone) or "input" (the inputs are the outputs)
    read: str = "fired"  # what "on" means at the read: "fired" this epoch, "again" (spiked after the input's moment), or
    # "window" (within read_window ms before the epoch's end)
    read_window: float | None = None  # the window for read == "window"
    target: str | None = None  # what the output should show (None: --target)
    critic: str | None = None  # how the read is scored (None: --critic)
    coding: str = "complement"  # how raw bits reach the input zone (§4.3): complement, raw, population, population-complement
    population: int | None = None  # neurons per raw bit where the coding repeats it (None: constants.POPULATION)
    output_coding: str = "population"  # how the output zone codes the classes (§8, learning.OUTPUT_CODINGS): a population a
    # class, or "complement" -- fire-if-one populations then fire-if-zero ones (mnist; Byron, September 16, 2026)
    permute: bool = True  # scramble the coded bits over the input neurons with a fixed permutation
    rule: str | None = None  # the learning rule this problem is posed for (None: --rule, else constants.RULE)
    quash: bool = True  # quash cycles (§6.11); False switches it off for this problem
    hebb: bool = False  # run leaky Hebb (§6.12) alongside whatever else this problem runs
    drive: str | None = None  # how a bit becomes spikes (§4.3): "forced" or "rate" (None: --drive, else constants.INPUT_DRIVE)
    flip: float | None = None  # corrupt the input: flip each coded bit with this probability (§4.3); None means no corruption
    outputs: int | None = None  # the width of the output zone when it differs from the input's (goo, §3.4); None: across
    clock: int = 0  # clock neurons (§4.3): input neurons at the front of the input zone whose bit is always 1
    goo: int | None = None  # the goo this problem is posed on: --goo defaults to this many neurons (None: the grid unless --goo)
    hidden_neurons: int | None = None  # goo's hidden count, inputs + hidden + outputs being the goo (Byron, September 16, 2026;
    # mnist, §8); --hidden-neurons overrides it, and 0 is a network the scaled rule can build
    data: str | None = None  # a dataset the inputs come from, with their labels, in place of random bits (§4.5): "mnist"
    homeostasis: float | None = None  # the Teacher's threshold drift for this problem (§1.3); None: the constant, unless given
    unstick: float | None = None  # and its un-sticking; 0 switches either off for the problem (mnist, §8: Byron, September 15, 2026)
    lr: float | None = None  # the problem's own learning rate; None: the constant, unless --lr is given (mnist 0.002, §8: Byron,
    # September 16, 2026, "default LR to 0.002 for this task")


PROBLEMS: dict[str, Problem] = {
    "reversal": Problem(
        "reversal",
        "the output zone learns to show the input zone reversed, taught by a Teacher with a target and a critic",
        ACROSS, trained=True, rule="reinforce", quash=False,
    ),
    "copy": Problem(
        "copy",
        "an input is complement-coded and presented on the input neurons, and the desired output is exactly the "
        "input expressed across the output neurons (Byron, September 14, 2026): eight in, eight out, output place i "
        "taught to show coded bit i. Unpermuted: 'permuting patterns should no longer matter, all neurons are "
        "first-class citizens of the population' (Byron, same day), and on goo a permutation of the input zone is only "
        "a relabelling of identically wired neurons. Reversal with the target set to copy and the permutation off; the "
        "task the goo comparisons of AUTHORITY.md §3.4 are posed on",
        ACROSS, trained=True, target="copy", critic="row", rule="reinforce", quash=False, permute=False,
        read="count",  # Byron, September 14, 2026: count the epoch's spikes, estimate the rate, threshold it (§4.3). On
        # goo the zones never project onto each other (§3.4), so the copy has to cross the interior
    ),
    "mnist": Problem(
        "mnist",
        "the MNIST digits (Byron, September 15, 2026: 'a new task, with its own data folder: mnist'): each 28 x 28 image "
        "averaged over 2 x 2 blocks to 14 x 14 and each block on iff its mean is at least half; the input zone is three "
        "clock neurons always driven, the 196 on-off pixels and their 196 complements (395 in all), no permutation; ten "
        "classes on 60 output neurons, complement-coded -- three fire-if-one neurons a class and then three fire-if-zero "
        "(Byron, September 16, 'force complement coding'; five a class, uncoded, from the evening of September 15 until "
        "then) -- read by count, and the evidence critic (Byron, September 16: 'the spikes are EVIDENCE'): each class's "
        "evidence is its fire-if-one sum minus its fire-if-zero sum, as log-odds at --temperature, paid the softmax "
        "cross-entropy ln q_label, chance ln 0.1. --critic graded is the night before's fraction of the other classes "
        "out-spiked and --critic class the earlier 1-or-0, on the same evidence. And "
        "no homeostasis or un-sticking: the hazard keeps nothing stuck and the un-sticking carried the network into "
        "silence. Posed on goo with 199 hidden neurons unless --hidden-neurons says otherwise (644 neurons; Byron, September "
        "16: 'How will we know if they are buying us anything if they are always part of the economy?'), by the reinforce rule "
        "at LR 0.002 (Byron, September 16, 'default LR to 0.002 for this task'); the train split in a seeded shuffle, "
        "cycling (mnist.stream)",
        3 + 2 * 196, trained=True, target="label", critic="evidence", rule="reinforce", quash=False, permute=False,
        read="count", coding="complement", population=3, outputs=60, output_coding="complement", clock=3, hidden_neurons=199,
        data="mnist", homeostasis=0.0, unstick=0.0, lr=0.002,
    ),
}
