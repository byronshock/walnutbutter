"""The Teacher: scores a trained problem's output every epoch, and runs the reinforce rule when asked.

Two rules exist (AUTHORITY.md §6). The default, **dopamine**, lives in
dopamine.py and runs inside the schedule as neurons refire; under it the
Teacher only scores and reports. The **reinforce** rule of the pre-alpha,
factored out behind `rule="reinforce"`, is the rest of this module.

The output zone -- the last `outputs` neurons in index order (AUTHORITY.md
§4.3) -- is the network's output. For each epoch a target pattern is
derived from the input pattern and the reward is the fraction of output
neurons that match it.

Nothing is traced back through the network. Instead each epoch:

1. **Explore.** Every neuron starts the epoch with a small random potential
   (Gaussian, standard deviation `sigma`), so the same input produces
   slightly different cascades from one epoch to the next.
2. **Run** the epoch and score it. The **advantage** is the reward minus a
   running average of recent rewards: how much better or worse than usual.
3. **Reinforce.** Every connection that carried a signal (its source fired)
   into a neuron that was not a forced input is moved by
   `lr * advantage * eligibility`, where the eligibility is the target's
   exploration noise (normalised).

**Late signals.** A signal that arrives after its target has already fired
is dropped on delivery (Neuron.receive) and has no effect on the epoch, yet
by default its connection is still updated: the rule is local, pre fired
and post fired in the same epoch, and the global reward says whether that
coincidence was good. Strictly that is a reward-modulated Hebbian term
riding on the perturbation estimator, a bias with respect to the reward
gradient, but it is the biological shape of the rule (local eligibility,
global signal) and on the 8x10 reversed task it learned faster on every
seed tried (100k epochs: last-tenth 0.76-0.91 with late signals against
0.61-0.75 without). `late` chooses what a late signal earns: "count" (the
default, the same update as one that landed), "ignore" (nothing: the
node-perturbation estimator proper, only the signals that landed) or
"depress" (the opposite update, the shape of spike-timing-dependent
plasticity, where a presynaptic spike after the postsynaptic one weakens
the synapse). A neuron nudged towards firing in an
   epoch that turned out better than usual gets stronger inputs from the
   neurons that fed it; in a worse epoch, weaker.

This is the REINFORCE / node-perturbation estimator of the reward gradient,
a three-factor rule: presynaptic activity x postsynaptic perturbation x
global reward. With `eligibility="wrong_hebb"` the perturbation is replaced
by a plain Hebbian term (+1 if the target fired, -1 if not) and no noise is
injected: the classic reward-modulated Hebbian rule, uncentred, which is why
it points nowhere on a network whose neurons nearly all fire every epoch,
and why it was so named on September 16, 2026. With `eligibility="hebb"` the
term is centred and charged at every decision (AUTHORITY.md §6.7, the
single-spike rule of September 17, 2026): each synapse's score moves by
(y - p_hat_j) x_ij, the outcome of the decision against the neuron's own
per-decision expectation of its spike (moved by DECISION_MEMORY, a plain
mean until then, the first decision charging nothing), times what the
synapse has in the potential; weighed by the ISI factor of §0.2 when it is
on. With `eligibility="count_hebb"`, the epoch form hebb was until that day,
each synapse's eligibility is the signals it delivered this epoch, x_ij,
times its target's spike count minus the target's own running expectation
of that count, n_j - n_bar_j (Williams's y - y_bar, [1] §8.4; the
expectation moves by COUNT_MEMORY an epoch and starts at the first count
seen). Both are the reward-modulated
covariance rule, the Bernoulli-logistic REINFORCE term (y - p) x with the
probability estimated rather than known; no noise is injected either, and
whatever varies the counts is its exploration. With
`eligibility="hazard"` the firing decision itself is the draw (escape noise,
AUTHORITY.md §5.2: the network was given a positive ESCAPE_DELTA) and the
eligibility is Williams's own, the score of each decision summed over the
epoch on each synapse's trace of what it still had in the potential (§6.7);
every engine accumulates it as it runs and this module only pays it.

Forced inputs are never adjusted and weights are kept within the network's
weight_range, [-1, 1] by default.

**Homeostasis.** A neuron whose input sits far from its threshold is never
flipped by the exploration noise, gets no learning signal, and stays "stuck"
on or off. Every neuron therefore tracks its own firing rate and nudges its
threshold toward a target rate each epoch: firing too often raises the
threshold, too rarely lowers it. Neurons forced in wave 0 this epoch are
left out of both, exactly as reinforcement leaves their incoming weights
alone: their firing was not the network's doing. An unforced input neuron
is an ordinary neuron and is treated as one. The default
rate of 1e-6 toward a target of 0.5 is a very slow drift (a fully stuck
neuron moves its threshold by about 0.0006 per thousand epochs, so the
effect belongs to runs of millions of epochs); a rate of 0 switches it off. Thresholds may go negative, and nothing clips them:
the [-5, 5] range that once did was eliminated on September 14, 2026 as artificial (AUTHORITY.md §1.3).
"""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence

from .constants import TEACHER_CREDIT  # noqa: F401  (the teacher's credit per input neuron)
from .constants import (
    BASELINE_RATE, COUNT_MEMORY, LEAKY_ELIGIBILITY, SYNAPSE_TAU, CRITIC, ELIGIBILITY, HOMEOSTASIS, LATE, LR, RATE_MEMORY, RULE, SIGMA,
    STUCK_ABOVE, STUCK_BELOW,
    TARGET, TARGET_RATE, UNSTICK, UNSTICK_TARGET, WINDOW,
)
from .dopamine import Dopamine, apply_teacher
from .monitor import run_epoch
from .network import Network
from .neuron import Neuron

Target = Callable[[Sequence[bool]], list[bool]]

TARGETS: dict[str, Target] = {
    "reversed": lambda pattern: list(pattern)[::-1],
    "copy": lambda pattern: list(pattern),
    "complement": lambda pattern: [not b for b in pattern],  # the same problem, only NOT (§8, Byron, September 14, 2026)
    "all-off": lambda pattern: [False] * len(pattern),
    "all-on": lambda pattern: [True] * len(pattern),
    "label": None,  # the label's population code over the output zone (§8, mnist): expected_outputs reads it off the network
}

RULES = ("teacher", "adaline", "dopamine", "reinforce", "local")  # which rule pays at the read (AUTHORITY.md §6); "local"
# means none of them does, and the local rules that compose -- the quash (§6.11), leaky Hebb (§6.12), the decay (§6.8) --
# are the whole of the learning (Byron, September 13, 2026: "no teacher for now")
ELIGIBILITIES = ("perturb", "wrong_hebb", "hebb", "count_hebb", "hazard")  # wrong_hebb: the +-1 by whether the target fired;
# hebb: the single-spike rule (§6.7, September 17, 2026), charged at every decision of the target; count_hebb, the epoch form: what the
# synapse delivered times the target's count minus its expectation (§6.7); hazard: the score of the escape-noise decision
# (§5.2) on each synapse's trace (§6.7)
LATE_RULES = ("count", "ignore", "depress")  # what a signal that arrived after its target fired earns


def arrays(network) -> bool:
    """True for the array engine (arrays.ArrayNetwork), which does its own vector updates."""
    return getattr(network, "engine", "objects") == "arrays"


def output_row(network: Network) -> list[Neuron]:
    """The network's output neurons, in word order: the output zone (§4.3)."""
    return network.output_row()


def expected_outputs(network: Network, target: str = "reversed") -> list[bool]:
    """What the read should show for the network's current input: the clean pattern, which flips (§4.3) may differ from."""
    if target == "label":
        return label_code(network)
    pattern = getattr(network, "target_pattern", None)
    if pattern is None:
        pattern = network.input_pattern
    if pattern is None:
        raise ValueError("no input pattern set")
    return TARGETS[target](pattern)


OUTPUT_CODINGS = ("population", "complement")  # how the output zone codes the classes (§8): a population of `population`
# neurons a class, or that and then, in the same order, a population that should fire when the class is NOT the label --
# fire-if-one groups then fire-if-zero groups, the input zone's complement coding turned around (Byron, September 16, 2026)


def class_evidence(counts: Sequence[int], population: int, output_coding: str = "population") -> list[int]:
    """The evidence for each class from the output zone's counts (§8): the sum over its population, and under complement
    coding that sum minus the sum over its fire-if-zero population -- a spike from a zero neuron is a spike against.

    Under "population" the zone is C * population wide, class c owning group
    c; under "complement" it is 2 * C * population wide, the C fire-if-one
    groups first and then the C fire-if-zero groups in the same order, so
    n_k = n_k^+ - n_k^-. Every engine reads the zone through this function.
    """
    if output_coding not in OUTPUT_CODINGS:
        raise ValueError(f"unknown output coding {output_coding!r}; choose from {', '.join(OUTPUT_CODINGS)}")
    per = population * (2 if output_coding == "complement" else 1)
    if population < 1 or len(counts) % per:
        raise ValueError(f"an output zone of {len(counts)} does not divide into classes of {population} under {output_coding} coding")
    classes = len(counts) // per
    ones = [sum(counts[c * population:(c + 1) * population]) for c in range(classes)]
    if output_coding == "population":
        return ones
    zeros = [sum(counts[(classes + c) * population:(classes + c + 1) * population]) for c in range(classes)]
    return [one - zero for one, zero in zip(ones, zeros)]


def label_code(network: Network) -> list[bool]:
    """The label as the output zone should show it: the label's fire-if-one group on and every other off; under complement
    coding the fire-if-zero groups the other way round (§8)."""
    if network.input_label is None:
        raise ValueError("the label target needs a stream that carries labels (a dataset, §8)")
    coding = getattr(network, "output_coding", "population")
    per = network.population * (2 if coding == "complement" else 1)
    classes = network.output_width() // per
    ones = [c == network.input_label for c in range(classes) for _ in range(network.population)]
    return ones + [not on for on in ones] if coding == "complement" else ones


def class_sums(network: Network) -> list[int]:
    """The output zone's evidence per class this epoch, from the counts, under the network's output coding."""
    return class_evidence(network.output_counts(), network.population, getattr(network, "output_coding", "population"))


def class_accuracy(network: Network, target: str = "label") -> float:
    """The class critic (§8, mnist; Byron, September 15, 2026: population per class, the most active wins).

    Each class owns `population` output neurons in a row. Their spikes this
    epoch are summed, and the reward is 1 when the label's class out-spikes
    every other class, else 0: a tie loses, and so does silence.
    """
    if network.input_label is None:
        raise ValueError("the class critic needs a stream that carries labels (a dataset, §8)")
    groups = class_sums(network)
    mine = groups[network.input_label]
    return 1.0 if all(mine > g for c, g in enumerate(groups) if c != network.input_label) else 0.0


def graded_accuracy(network: Network, target: str = "label") -> float:
    """The graded critic (§8, mnist; Byron, September 15, 2026: "a graded critic it is").

    The class sums as the class critic takes them; the reward is the fraction
    of the other classes the label's class strictly out-spikes: 1 when it
    out-spikes every one (the class critic's 1), 0 when it out-spikes none,
    and a near miss is paid for what it beat. A tie is not beaten, so
    silence scores 0. Chance is a half.
    """
    if network.input_label is None:
        raise ValueError("the graded critic needs a stream that carries labels (a dataset, §8)")
    groups = class_sums(network)
    mine = groups[network.input_label]
    others = [g for c, g in enumerate(groups) if c != network.input_label]
    return sum(1 for g in others if mine > g) / len(others)


def evidence_reward(groups: Sequence[int], label: int, temperature: float) -> float:
    """The evidence critic's reward from the class sums (§8): ln q_y, with q_k = exp(n_k / T) / sum_j exp(n_j / T).

    Computed from the largest sum, so the exponentials cannot overflow at
    any temperature; every engine pays the reward through this one function,
    so they agree to the bit.
    """
    if temperature <= 0.0:
        raise ValueError(f"the evidence critic needs a positive temperature, got {temperature}")
    top = max(groups)
    return (groups[label] - top) / temperature - math.log(sum(math.exp((g - top) / temperature) for g in groups))


def evidence_score(network: Network, target: str = "label") -> float:
    """The evidence critic (§8, mnist; Byron, September 16, 2026: "the spikes are EVIDENCE for now, not a proper
    maximum-likelihood estimator. We will have to sweep for temperature eventually").

    The class sums as the class critic takes them, read as log-odds at the
    network's `temperature` (TEMPERATURE, §1.3): the estimate is the softmax
    of the sums over T and the reward is the log of the estimate the label
    was given -- the softmax cross-entropy of Bridle (1990) and Bishop (1995,
    §6.9). Chance, a uniform estimate, is ln 0.1 = -2.30; a perfect epoch is
    0; a silent label population is weak evidence, not -infinity. T -> 0 is
    the class critic in log form, T -> infinity pays ln 0.1 whatever the counts.
    """
    if network.input_label is None:
        raise ValueError("the evidence critic needs a stream that carries labels (a dataset, §8)")
    return evidence_reward(class_sums(network), network.input_label, network.temperature)


def output_fired(network: Network) -> list[bool]:
    """Whether each output neuron is on, left to right, whichever engine runs the network (see Network.output_fired)."""
    return network.output_fired()


def output_levels(network: Network) -> list[float]:
    """What the teacher reads: one number in [0, 1] per output neuron (see Network.output_levels, AUTHORITY.md §6.9).

    Under a boolean read these are 0.0 and 1.0 and every score below is what
    it always was; under `read = "rate"` they are the measured firing rate
    over RATE_ON, so silence reads 0 and saturation reads 1.
    """
    return network.output_levels()


def output_errors(network: Network, target: str = "reversed") -> dict[Neuron, int]:
    """Error per output neuron: +1 should have fired, -1 should not have, 0 correct."""
    return {
        neuron: float(want) - level
        for neuron, level, want in zip(output_row(network), output_levels(network), expected_outputs(network, target))
    }


def accuracy(network: Network, target: str = "reversed") -> float:
    """Fraction of the output row that matches the target, 0 to 1. This is the reward."""
    levels = output_levels(network)
    want = expected_outputs(network, target)
    return sum(1.0 - abs(level - float(w)) for level, w in zip(levels, want)) / len(want)


# --- reading the output row as a receiver would ------------------------------------


def read_output_word(network: Network, target: str = "reversed") -> list[bool | None]:
    """Undo the target's arrangement and the permutation, then resolve each complement pair.

    The output row is what the network produced; the target says where each
    coded bit was meant to land (reversed: place j shows coded bit
    permutation[across-1-j]). Coded bit k and its complement k + half form a
    pair: if exactly one of them fired, the bit is read; if both or neither
    did, the bit is unreadable (None) and counts as an error for the code.
    """
    fired = output_fired(network)
    width = len(fired)
    if target == "reversed":
        placed = fired[::-1]  # placed[i] is what place i of the input arrangement would show
    elif target == "copy":
        placed = fired
    else:
        raise ValueError(f"the output word can only be read for the reversed or copy target, not {target!r}")
    coded = [False] * width
    for i, k in enumerate(network.permutation):
        coded[k] = placed[i]
    half = width // 2
    word: list[bool | None] = []
    for k in range(half):
        bit, complement = coded[k], coded[k + half]
        word.append(bit if bit != complement else None)
    return word


def decoded_output(network: Network, target: str = "reversed") -> list[bool]:
    """The data bits a receiver would decode from the output row, after error correction if a code is on.

    Unreadable bits are taken as 0 before correction, so a single unreadable
    or wrong bit is repaired by a correcting code.
    """
    word = [False if b is None else b for b in read_output_word(network, target)]
    if network.code:
        return network.code.decode(word)
    return word


def expected_data(network: Network) -> list[bool]:
    """What the receiver should decode: the data bits when a code is on, else the raw input bits."""
    if network.input_bits is None:
        raise ValueError("no input pattern set")
    return list(network.input_data if network.code else network.input_bits)


def decoded_accuracy(network: Network, target: str = "reversed") -> float:
    """Fraction of the corrected, decoded data bits that are right, 0 to 1."""
    want = expected_data(network)
    got = decoded_output(network, target)
    return sum(a == b for a, b in zip(got, want)) / len(want)


def decoded_exact(network: Network, target: str = "reversed") -> float:
    """1 if the corrected, decoded data bits are all right, else 0."""
    return 1.0 if decoded_output(network, target) == expected_data(network) else 0.0


def population_vote(fired: Sequence[bool], population: int) -> list[bool]:
    """Majority vote within each group of `population` neurons: a raw bit is 1 if most of its group fired.

    Byron, September 14, 2026: "If two of the three neurons fired, that is a
    1. If one of the three fired, that is a 0. Three is a 1, and none is a 0."
    """
    groups = [fired[i:i + population] for i in range(0, len(fired), population)]
    return [sum(1 for f in group if f) * 2 > population for group in groups]


def population_output(network: Network, target: str = "copy") -> list[bool]:
    """The raw bits a receiver would decode from the output row by majority vote (AUTHORITY.md §6.13)."""
    return population_vote(output_fired(network), network.population)


def population_accuracy(network: Network, target: str = "copy") -> float:
    """The kinder teacher: a quarter of a point per raw bit the majority vote gets right (AUTHORITY.md §6.13).

    Byron, September 14, 2026: "Output patterns will be scored against the
    desired input as follows: for each bit, award 0.25 points if the output
    bit matches the desired input." With four raw bits a quarter each, that
    is the fraction of raw bits right, in [0, 1] like every other critic, and
    0.25 a bit falls out of there being four of them. Both the read and the
    target are decoded the same way, so a flipped input (§4.3) is scored
    against the clean code and any target arrangement works.
    """
    want = population_vote(expected_outputs(network, target), network.population)
    got = population_output(network, target)
    return sum(1 for a, b in zip(got, want) if a == b) / len(want)


def teacher_score(network: Network, target: str = "copy", critic: str = "row") -> float:
    """The external teacher's score for the read (AUTHORITY.md §6.10), in [-1, 1] for a four-neuron input zone.

    Byron, September 12, 2026: +0.25 for a forced-input neuron that
    sustains, +0.25 for an input neuron whose input is zero and does not
    fire, and -0.25 for each of those read wrongly, so four inputs give
    -1, -0.5, 0, 0.5 or 1. That 0.25 is 1/4, so the credit is TEACHER_CREDIT
    when one is set and 1/n otherwise: the score spans [-1, 1] for a zone of
    any size, and twelve outputs score 0 when six are right (Byron,
    September 13, 2026).
    """
    if TEACHER_CREDIT is None:
        return 2.0 * CRITICS[critic](network, target) - 1.0
    levels = output_levels(network)  # a credit fixed by hand only makes sense neuron by neuron, so the row form stands
    want = expected_outputs(network, target)
    return TEACHER_CREDIT * sum(1.0 - 2.0 * abs(level - float(w)) for level, w in zip(levels, want))


def adaline_errors(network: Network, target: str = "copy") -> list[float]:
    """The error at each output neuron, desired minus actual (AUTHORITY.md §6.10).

    +1 for a neuron that should have been on and was not, -1 for one that
    was on and should not have been, 0 for one read correctly. A correct
    epoch therefore moves nothing: ADALINE corrects mistakes only.
    """
    levels = output_levels(network)
    want = expected_outputs(network, target)
    return [float(w) - level for level, w in zip(levels, want)]


def apply_adaline(network: Network, errors, lr: float) -> int:
    """Widrow-Hoff at the read: every synapse into output neuron j moves by lr * error_j * what it delivered. Returns synapses moved.

    The eligibility trace is the presynaptic activity the target integrated
    this epoch (propagation.Schedule.run with `trace`), so a synapse that
    delivered nothing moves by nothing. Only the scored neurons' incoming
    weights learn; the rest of the network is an untrained reservoir. The trace
    is cleared either way, so an epoch's activity never counts twice.
    """
    if arrays(network):
        return network.apply_adaline(errors, lr)
    error = {neuron: e for neuron, e in zip(output_row(network), errors)}
    low, high = network.weight_range
    moved = 0
    for connection in network.connections.values():
        if connection.eligibility:
            step = lr * error.get(connection.target, 0.0) * connection.eligibility
            if step:
                weight = connection.weight + step
                connection.weight = low if weight < low else high if weight > high else weight
                moved += 1
            connection.eligibility = 0.0
    return moved


def sustained(network: Network, target: str = "copy") -> float:
    """Of the output neurons the target says should be on, the fraction that are on: did the forced neurons sustain?

    Byron, September 12, 2026: the neurons that were not forced are not
    scored at all. With no neuron to sustain the score is 0.
    """
    levels = output_levels(network)
    want = expected_outputs(network, target)
    on = [level for level, w in zip(levels, want) if w]
    return sum(on) / len(on) if on else 0.0


CRITICS = {
    "row": accuracy,  # fraction of the output row matching the target, neuron by neuron
    "sustained": sustained,  # of the neurons the target says should be on, the fraction on: the forced neurons that sustained
    "decoded": decoded_accuracy,  # fraction of data bits right after reading and error-correcting the row
    "decoded-exact": decoded_exact,  # all data bits right after correction, or nothing
    "population": population_accuracy,  # the kinder teacher: raw bits right after a majority vote per group (§6.13)
    "class": class_accuracy,  # a dataset's label: 1 when the label's group of outputs out-spikes every other group (§8)
    "graded": graded_accuracy,  # the fraction of the other groups the label's group out-spikes (§8)
    "evidence": evidence_score,  # the group sums as evidence at a temperature: the softmax cross-entropy ln q_label (§8)
}


def reward(network: Network, target: str = "reversed", critic: str = "row") -> float:
    """The scalar the network is judged by, according to the chosen critic."""
    return CRITICS[critic](network, target)


def forced(neuron: Neuron) -> bool:
    """True if the neuron was forced to fire by the stimulus this epoch."""
    return neuron.forced


def update_rates(network: Network) -> None:
    """Move every neuron's running firing-rate estimate, and its expected spike count, toward what it did this epoch.

    A neuron forced this epoch is skipped: that firing says nothing about the
    network. The expected count, n_bar_j, is what the count_hebb eligibility centres
    on (AUTHORITY.md §6.7); it starts at the first count seen and then moves
    by COUNT_MEMORY an epoch, after the update has used it.
    """
    if arrays(network):
        return network.update_rates()
    for neuron in network.all_neurons():
        if not forced(neuron):
            neuron.rate += RATE_MEMORY * ((1.0 if neuron.has_fired else 0.0) - neuron.rate)
            count = float(neuron.epoch_spikes)
            if neuron.expected_count is None:
                neuron.expected_count = count  # its first unforced epoch sets the expectation
            else:
                neuron.expected_count += COUNT_MEMORY * (count - neuron.expected_count)


def stuck_neurons(network: Network) -> tuple[list[Neuron], list[Neuron]]:
    """Neurons whose running rate is (almost) always on, and always off (indices, for the array engine)."""
    if arrays(network):
        return network.stuck()
    on = [n for n in network.all_neurons() if n.rate > STUCK_ABOVE]
    off = [n for n in network.all_neurons() if n.rate < STUCK_BELOW]
    return on, off


def homeostasis(
    network: Network, rate: float, target: float = TARGET_RATE
) -> int:
    """Nudge each neuron's threshold toward its target firing rate, except those forced this epoch.

    Returns the number of neurons moved.
    """
    if arrays(network):
        return network.homeostasis(rate, target)
    if rate <= 0:
        return 0
    moved = 0
    for neuron in network.all_neurons():
        if forced(neuron):
            continue
        neuron.threshold = neuron.threshold + rate * (neuron.rate - target)  # nothing clips it
        moved += 1
    return moved


def unstick(
    network: Network,
    rate: float,
    target: float = 0.5,
) -> list[Neuron]:
    """Nudge the threshold of every *stuck* neuron toward a target firing rate.

    Every neuron whose running rate is beyond the stuck band (almost always
    on, or almost always off) is touched, and only while it is; a neuron
    forced this epoch is left alone, as homeostasis leaves it. A saturated
    neuron gets no learning signal because nothing changes whether it fires;
    moving its threshold back toward the region where the rule has a gradient
    gives it one, and nothing else in the network is disturbed.

    It was the output row only until September 14, 2026, when the interior of
    a goo with no direct projection turned out to be dead for want of exactly
    this (AUTHORITY.md §3.4): "all neurons are first-class citizens" (Byron),
    and a restriction to the outputs was an artificial one (§2). Returns the
    neurons nudged (indices, for the array engine).
    """
    if arrays(network):
        return network.unstick(rate, target)
    if rate <= 0:
        return []
    nudged = []
    for neuron in network.all_neurons():
        if forced(neuron):
            continue
        if neuron.rate > STUCK_ABOVE or neuron.rate < STUCK_BELOW:
            neuron.threshold = neuron.threshold + rate * (neuron.rate - target)
            nudged.append(neuron)
    return nudged


def delivered_signals(network: Network) -> list:
    """Every signal delivered in the last epoch, in wave order, each exactly once.

    No deduplication is needed: a neuron fires at most once per epoch, so each
    of its active outgoing connections carries at most one signal. This
    includes signals that arrived after their target had fired; see `landed`.
    """
    return [signal for wave in network.waves for signal in wave.signals()]


def landed(signal) -> bool:
    """True if the signal was taken in: its target had not fired before the wave it arrived in.

    A neuron ignores input once it has fired (Neuron.receive), but propagation
    still records the delivery. A signal from a later wave than the target's
    firing wave changed nothing and must earn no credit or blame.
    """
    fired_in = signal.target.fired_in_wave
    return fired_in is None or signal.wave <= fired_in


def delivered_connections(network: Network) -> list:
    """Every connection that carried a signal in the last epoch, landed or not (see delivered_signals)."""
    return [connection for wave in network.waves for connection in wave.delivered]


def reinforce(
    network: Network,
    advantage: float,
    lr: float = LR,
    sigma: float = SIGMA,
    eligibility: str = ELIGIBILITY,
    late: str = LATE,
    leaky: bool = LEAKY_ELIGIBILITY,
) -> int:
    """Apply the global-reward update for the epoch that has just run. Returns connections changed.

    `late` says what a signal that arrived after its target fired (see `landed`)
    earns: "count" the same update as one that landed, "ignore" none, or
    "depress" the opposite.

    `leaky` appends the trace of §6.12 to the chain, so the update becomes

        w_ij <- clip(w_ij + LR * A * e_j * exp(-(t_read_j - t_fired_i) / TAU)),

    with t_read_j the moment j's answer was fixed: its spike if it fired, and
    the arrival of the signal itself if it did not, since a target that never
    fired offers no moment at which its synapses can be told apart. Without the
    trace every synapse that delivered into j gets the same update whatever it
    delivered; with it, each synapse of a firing neuron is weighted by the
    charge it still had in it when it fired.
    """
    if eligibility not in ELIGIBILITIES:
        raise ValueError(f"unknown eligibility {eligibility!r}; choose from {', '.join(ELIGIBILITIES)}")
    if late not in LATE_RULES:
        raise ValueError(f"unknown late-signal rule {late!r}; choose from {', '.join(LATE_RULES)}")
    if not getattr(network, "hazard", False):  # §8.3: no eligibility runs where the threshold decides
        raise ValueError("the reinforce rule refuses to learn where the threshold decides: REINFORCE estimates a gradient from the randomness of the decision, and with no width there is no randomness to estimate from. Give the network a positive ESCAPE_DELTA (--delta) (§8.3)")
    if eligibility == "hazard":
        if late != "count" or leaky:
            raise ValueError("the hazard eligibility carries its own trace: late = count and no leaky trace (§6.7)")
    if eligibility in ("hebb", "count_hebb") and (late != "count" or leaky):
        raise ValueError(f"the {eligibility} eligibility carries its own trace of what each synapse delivered: late = count "
                         "and no leaky trace (§6.7)")
    if arrays(network):
        return network.reinforce(advantage, lr, sigma, eligibility, late, leaky)
    if eligibility in ("hazard", "hebb"):
        network.settle_scores()  # under the evidence accumulator the open arrivals' debit is brought into the score first (§6.7)
    if not advantage:
        return 0
    low, high = network.weight_range
    step = lr * advantage
    if eligibility in ("hazard", "hebb"):  # §6.7: every synapse into an unforced neuron moves by what its scores summed to
        changed = 0
        for connection in network.connections.values():
            if not connection.score or connection.target.forced:
                continue  # nothing accumulated, or a forced input: its firing was not the network's doing
            weight = connection.weight + step * connection.score
            if weight < low:
                weight = low
            elif weight > high:
                weight = high
            connection.weight = weight
            changed += 1
        return changed
    if eligibility == "count_hebb":  # §6.7's epoch form: what each synapse delivered times its target's count minus its expectation
        changed = 0
        for connection in network.connections.values():
            target = connection.target
            if not connection.eligibility or target.forced:
                continue  # delivered nothing this epoch, or a forced input: its firing was not the network's doing
            if target.expected_count is None:
                continue  # the target's first unforced epoch: the expectation is this count, and the eligibility zero
            e = connection.eligibility * (target.epoch_spikes - target.expected_count)
            if not e:
                continue  # exactly as many spikes as expected: nothing to credit or blame
            weight = connection.weight + step * e
            if weight < low:
                weight = low
            elif weight > high:
                weight = high
            connection.weight = weight
            changed += 1
        return changed
    perturb = eligibility == "perturb"
    tau, hop = getattr(network, "synapse_tau", SYNAPSE_TAU), Neuron.hop()  # the synapse's leak, not the neuron's (§6.12)
    changed = 0
    last_delivery: dict = {}  # once per connection per epoch, by its last delivery (a source may fire more than once)
    for wave in network.waves:
        for connection in wave.delivered:
            last_delivery[connection] = wave.number
    for connection, arrived in last_delivery.items():
        if True:
            target = connection.target
            if target.forced:
                continue  # a forced input: its firing was not the network's doing
            fired_in = target.fired_in_wave
            if perturb:
                e = target.noise / sigma if sigma else 0.0
            else:
                e = 1.0 if target.has_fired else -1.0
            if late != "count" and fired_in is not None and arrived > fired_in:
                if late == "ignore":
                    continue  # dropped on arrival: it changed nothing this epoch
                e = -e  # arrived after the firing: weakened where an early one would be strengthened
            if not e:
                continue
            if leaky:
                if connection.last_signal is None:
                    continue  # nothing was ever integrated here, so it was contributing nothing
                when = target.fired_at if target.has_fired else connection.last_signal
                # A target that never fired has no moment at which its synapses can be told apart, so
                # the trace is exp(-hop/TAU) for all of them: the scale changes, the resolution does not.
                # Reading it at the horizon instead annihilates the whole non-firing half (5e-5 at TAU 2,
                # a 35 ms epoch), which is the depressive half of the wrong_hebb eligibility.
                e = e * math.exp(-(when - connection.last_signal + hop) / tau)
            weight = connection.weight + step * e
            if weight < low:
                weight = low
            elif weight > high:
                weight = high
            connection.weight = weight
            changed += 1
    return changed


class Teacher:
    """Runs epochs with exploration noise and scores them; under the reinforce rule, also reinforces every connection.

    Use `teacher.epoch()` in place of `run_epoch(network)`: it injects the
    exploration noise before the epoch and, with `rule="reinforce"`, applies
    the update after it. With the default `rule="dopamine"` the network
    learns by itself as it runs (a Dopamine is attached to the network if it
    has none) and the Teacher scores, keeps the firing rates, homeostasis
    and un-sticking, and reports.
    """

    def __init__(
        self,
        network: Network,
        target: str = TARGET,
        lr: float = LR,
        sigma: float = SIGMA,
        eligibility: str | None = None,
        baseline_rate: float = BASELINE_RATE,
        window: int = WINDOW,
        seed: int | None = None,
        homeostasis: float = HOMEOSTASIS,
        target_rate: float = TARGET_RATE,
        discharge: bool = False,
        unstick: float = UNSTICK,
        unstick_target: float = UNSTICK_TARGET,
        critic: str = CRITIC,
        late: str = LATE,
        rule: str = RULE,
        leaky: bool = LEAKY_ELIGIBILITY,
    ):
        if rule not in RULES:
            raise ValueError(f"unknown learning rule {rule!r}; choose from {', '.join(RULES)}")
        self.rule = rule
        if rule != "reinforce" and getattr(network, "dopamine", None) is None:
            network.dopamine = Dopamine(lr=lr)
        network.rule = rule if rule != "reinforce" else "dopamine"  # the reinforce rule leaves the schedule's hook alone
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
        if critic not in CRITICS:
            raise ValueError(f"unknown critic {critic!r}; choose from {', '.join(CRITICS)}")
        if critic not in ("row", "population", "class", "graded", "evidence") and target not in ("reversed", "copy"):
            raise ValueError(f"the {critic} critic reads the output as a word, which needs the reversed or copy target")
        if critic in ("class", "graded", "evidence") and target != "label":
            raise ValueError(f"the {critic} critic scores a dataset's label (§8): its target is label")
        self.critic = critic
        if late not in LATE_RULES:
            raise ValueError(f"unknown late-signal rule {late!r}; choose from {', '.join(LATE_RULES)}")
        self.late = late  # what a signal arriving after its target fired earns
        self.leaky = bool(leaky)  # append the leaky trace of §6.12 to the reinforce rule's chain
        if eligibility is None:  # the eligibility follows the neuron (§1.3): the hazard's under escape noise, else the constant's
            eligibility = "hazard" if getattr(network, "hazard", False) else ELIGIBILITY
        if eligibility not in ELIGIBILITIES:
            raise ValueError(f"unknown eligibility {eligibility!r}; choose from {', '.join(ELIGIBILITIES)}")
        if rule == "reinforce" and not getattr(network, "hazard", False):  # §8.3
            raise ValueError("the reinforce rule refuses to learn where the threshold decides: REINFORCE estimates a gradient from the randomness of the decision, and with no width there is no randomness to estimate from. Give the network a positive ESCAPE_DELTA (--delta) (§8.3) before the Teacher")
        if lr < 0 or sigma < 0 or homeostasis < 0 or unstick < 0:
            raise ValueError("learning rate, sigma, homeostasis and unstick rates must not be negative")
        if not 0.0 < unstick_target < 1.0:
            raise ValueError(f"unstick target firing rate must be between 0 and 1, got {unstick_target}")
        self.unstick = unstick
        self.unstick_target = unstick_target
        self.unstuck_count = 0  # how many epoch-nudges the un-sticking has applied
        self.moved = 0  # synapses the external teacher has moved
        self.last_signal: float | None = None  # the teacher's score for the last epoch, in [-1, 1]
        self.mistakes = 0.0  # output neurons read wrongly in the last epoch (the ADALINE rule)
        self.history: list[dict] = []  # one entry per progress report; saved in checkpoints
        if not 0.0 < target_rate < 1.0:
            raise ValueError(f"target firing rate must be between 0 and 1, got {target_rate}")
        self.homeostasis = homeostasis
        self.target_rate = target_rate
        self.discharge = discharge  # zero every potential between inputs instead of letting it leak
        self.network = network
        self.target = target
        self.lr = lr
        self.sigma = sigma if eligibility == "perturb" else 0.0
        self.eligibility = eligibility
        network.tally = eligibility == "count_hebb"  # the tally of what each synapse delivered, the x_ij of §6.7's epoch form, in whichever engine
        network.centre(eligibility == "hebb")  # the single-spike rule (§6.7): every neuron charges its decisions against its own expectation
        self.baseline_rate = baseline_rate
        self.window = window
        self.rng = random.Random(seed)
        self.epochs = 0
        self.total_reward = 0.0  # sum of every epoch's reward, for accuracy to date
        self.baseline: float | None = None  # running average reward: what "usual" looks like
        self.last_reward: float | None = None
        self.average: float | None = None  # exponential moving average over about `window` epochs
        self._trace = None  # a text file the per-epoch trace is written to (see trace_to)

    def trace_to(self, path) -> None:
        """Write one line per epoch to `path` (CSV, appended): epoch, time, dopamine, expected, score."""
        import os
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._trace = open(path, "a", buffering=1)
        if new:
            self._trace.write("epoch,time_ms,dopamine,expected,score\n")

    def close_trace(self) -> None:
        if self._trace is not None:
            self._trace.close()
            self._trace = None

    def epoch(self, bits: Sequence[bool] | None = None, verbose: bool = True) -> float:
        """Run one epoch with exploration noise, then learn from it. Returns its reward."""
        run_epoch(self.network, bits, verbose=verbose, noise=self.sigma, rng=self.rng, discharge=self.discharge)
        return self.step()

    def step(self) -> float:
        """Score the epoch that has just run and reinforce. Returns its reward (accuracy)."""
        reward = CRITICS[self.critic](self.network, self.target)
        if self.baseline is None:
            self.baseline = reward
        advantage = reward - self.baseline
        self.last_signal = teacher_score(self.network, self.target, self.critic) if self.rule == "teacher" else None
        if self.rule == "teacher":
            self.moved += apply_teacher(self.network, self.last_signal, self.lr)  # the teacher pays the epoch's eligibility
        elif self.rule == "adaline":
            errors = adaline_errors(self.network, self.target)
            self.mistakes = sum(abs(e) for e in errors)  # how many output neurons were read wrongly this epoch
            self.moved += apply_adaline(self.network, errors, self.lr)
        elif self.rule == "reinforce":
            reinforce(self.network, advantage, self.lr, self.sigma, self.eligibility, self.late, self.leaky)
        update_rates(self.network)
        homeostasis(self.network, self.homeostasis, self.target_rate)
        self.unstuck_count += len(unstick(self.network, self.unstick, self.unstick_target))
        self.baseline += self.baseline_rate * (reward - self.baseline)
        self.epochs += 1
        if self._trace is not None:
            pool = self.network.dopamine
            level, expected = ("", "") if pool is None else (f"{pool.peek(self.network.horizon):.6g}", f"{pool.expected():.6g}")
            if self.rule == "teacher":
                level, expected = f"{self.last_signal:+g}", f"{self.moved}"  # the teacher's signal, and synapses moved to date
            elif self.rule == "adaline":
                level, expected = f"{self.mistakes:g}", f"{self.moved}"  # output neurons read wrongly, and synapses moved to date
            self._trace.write(f"{self.network.epoch},{self.network.time:g},{level},{expected},{reward:.6g}\n")
        self.total_reward += reward
        self.last_reward = reward
        if self.average is None:
            self.average = reward
        else:
            alpha = 2.0 / (self.window + 1)
            self.average = (1 - alpha) * self.average + alpha * reward
        return reward

    @property
    def accuracy_to_date(self) -> float | None:
        """Mean reward over every epoch taught so far."""
        return self.total_reward / self.epochs if self.epochs else None

    def record(self, elapsed: float | None = None, epochs_per_second: float | None = None) -> dict:
        """Append the current figures to the history (called at each progress report). Returns the entry."""
        on, off = stuck_neurons(self.network)
        entry = {
            "epoch": self.network.epoch,
            "elapsed": None if elapsed is None else round(elapsed, 1),
            "accuracy_to_date": None if self.accuracy_to_date is None else round(self.accuracy_to_date, 4),
            "recent": None if self.average is None else round(self.average, 4),
            "stuck_on": len(on),
            "stuck_off": len(off),
            "epochs_per_second": None if epochs_per_second is None else round(epochs_per_second),
        }
        self.history.append(entry)
        return entry

    def status(self) -> str:
        verb = {"dopamine": "scoring", "teacher": "teaching", "adaline": "correcting"}.get(self.rule, "learning")
        if self.average is None:
            return f"{verb} {self.target}: no epochs yet"
        if self.rule == "teacher":
            signal = "none yet" if self.last_signal is None else f"{self.last_signal:+g}"
            settings = f"signal {signal}, {self.moved:,} synapses moved, lr {self.lr:g}, sigma {self.sigma:g}"
        elif self.rule == "adaline":
            settings = f"{self.mistakes:g} wrong, {self.moved:,} synapses moved, lr {self.lr:g}, sigma {self.sigma:g}"
        elif self.rule == "dopamine":
            settings = f"{self.network.dopamine.status()}, sigma {self.sigma:g}"
        else:
            settings = f"{self.eligibility}{' + leaky trace' if self.leaky else ''}, lr {self.lr:g}, sigma {self.sigma:g}"
        if self.critic != "row":
            settings += f", critic {self.critic}"
        if self.late != "count":
            settings += f", late signals {self.late}d"
        if self.homeostasis:
            settings += f", homeostasis {self.homeostasis:g} toward {self.target_rate:g}"
        if self.unstick:
            settings += f", unstick {self.unstick:g}"
        if self.discharge:
            settings += ", discharge"
        return (
            f"{verb} {self.target} ({settings}): "
            f"accuracy {self.accuracy_to_date:.1%} to date over {self.epochs:,} epochs, "
            f"{self.average:.0%} recent"
        )
