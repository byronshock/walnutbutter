"""The Teacher: scores the output zone every epoch, and pays the one rule that pays at the read.

AUTHORITY.md §9.1: at most one rule pays at the read, and the specification
carries one — the **reinforce** rule, which is the rest of this module. A run
may have none (`rule="local"`), in which case the local rules of §10 are the
whole of the learning.

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
synapse has in the potential. With `eligibility="count_hebb"`, the epoch form hebb was until that day,
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

from .constants import (
    BASELINE_RATE, CRITIC, ELIGIBILITY, LR, RATE_MEMORY, RULE,
    STUCK_ABOVE, STUCK_BELOW,
    TARGET, TARGET_RATE, UNSTICK_TARGET, WINDOW,
)
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

RULES = ("reinforce", "local")  # which rule pays at the read (AUTHORITY.md §9.1); "local"
# means none of them does, and the local rules of §10 -- the quash alone -- are the whole of the learning
# (Byron, September 13, 2026: "no teacher for now")
ELIGIBILITIES = ("hazard", "hebb")  # the two §8.3 keeps. Both are the single-spike rule of §8.4 and differ in what a
# decision's credit and expectation are: hazard takes the escape decision's own score (§8.7), hebb the neuron's own
# estimate of its spike (§8.6). Neither takes a late-signal rule or an eligibility trace of its own (§8.13)


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


def class_evidence(counts: Sequence[int], population: int) -> list[int]:
    """The evidence for each class from the output zone's counts (AUTHORITY.md §5.11, §5.12).

    The zone is complement-coded and coded no other way: for C classes and a
    population of P in each half it is 2CP neurons, the C fire-if-one groups
    first and then, in the same order, the C fire-if-zero groups. So
    n_k = n_k^+ - n_k^-, and a spike from a fire-if-zero neuron is a spike
    against. Every engine reads the zone through this function.
    """
    per = population * 2
    if population < 1 or len(counts) % per:
        raise ValueError(f"an output zone of {len(counts)} does not divide into classes of {population} a half (§5.11)")
    classes = len(counts) // per
    ones = [sum(counts[c * population:(c + 1) * population]) for c in range(classes)]
    zeros = [sum(counts[(classes + c) * population:(classes + c + 1) * population]) for c in range(classes)]
    return [one - zero for one, zero in zip(ones, zeros)]


def label_code(network: Network) -> list[bool]:
    """The label as the output zone should show it: its fire-if-one group on, every other off, and the
    fire-if-zero groups the other way round (§5.11)."""
    if network.input_label is None:
        raise ValueError("the label target needs a stream that carries labels (a dataset, §5.9)")
    classes = network.output_width() // (network.population * 2)
    ones = [c == network.input_label for c in range(classes) for _ in range(network.population)]
    return ones + [not on for on in ones]


def class_sums(network: Network) -> list[int]:
    """The output zone's evidence per class this epoch, from the counts (§5.11)."""
    return class_evidence(network.output_counts(), network.population)


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
    0.25 a bit falls out of there being four of them.
    """
    want = population_vote(expected_outputs(network, target), network.population)
    got = population_output(network, target)
    return sum(1 for a, b in zip(got, want) if a == b) / len(want)






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
    "population": population_accuracy,  # the kinder teacher: raw bits right after a majority vote per group (§6.13)
    "class": class_accuracy,  # a dataset's label: 1 when the label's group of outputs out-spikes every other group (§8)
    "graded": graded_accuracy,  # the fraction of the other groups the label's group out-spikes (§8)
    "evidence": evidence_score,  # the group sums as evidence at a temperature: the softmax cross-entropy ln q_label (§8)
}


def reward(network: Network, target: str = "reversed", critic: str = "row") -> float:
    """The scalar the network is judged by, according to the chosen critic."""
    return CRITICS[critic](network, target)


def forced(neuron: Neuron) -> bool:
    """True if the neuron carries 5.8's driven mark this epoch: forced to fire by the stimulus, or delivered to by the
    charged drive (5.4b) -- a delivery dropped at a refractory input included. Every reader of the mark reads it here or
    as `neuron.forced`: the pay (§8.1), the rate memory (§2.6), homeostasis and un-sticking (§9.2)."""
    return neuron.forced


def _refuse_thresholds_at_or_below_zero(network: Network, moved) -> None:
    """§7.5: under exploration at the synapse a threshold at or below zero leaves u undefined, and is refused at the
    moment homeostasis or un-sticking takes it there."""
    if getattr(network, "exploration", "neuron") == "synapse" and any(neuron.threshold <= 0.0 for neuron in moved):
        raise ValueError("a threshold moved to or below zero leaves u = clip(V, 0, theta) / theta undefined, and under "
                         "exploration at the synapse it is refused (§7.5)")


def update_rates(network: Network) -> None:
    """Move every neuron's running firing-rate estimate toward what it did this epoch (AUTHORITY.md §9.8).

    A neuron forced this epoch is skipped: that firing says nothing about the
    network.
    """
    if arrays(network):
        return network.update_rates()
    for neuron in network.all_neurons():
        if not forced(neuron):
            neuron.rate += RATE_MEMORY * ((1.0 if neuron.has_fired else 0.0) - neuron.rate)


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
    moved = []
    for neuron in network.all_neurons():
        if forced(neuron):
            continue
        neuron.threshold = neuron.threshold + rate * (neuron.rate - target)  # nothing clips it
        moved.append(neuron)
    _refuse_thresholds_at_or_below_zero(network, moved)
    return len(moved)


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
    _refuse_thresholds_at_or_below_zero(network, nudged)
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
    eligibility: str = ELIGIBILITY,
) -> int:
    """Apply the epoch's update to every synapse, by the single-spike rule of AUTHORITY.md §8.4.

    Two eligibilities survive (§8.3), hazard and hebb. Both are the same rule
    and differ only in what a decision's credit and expectation are, so both
    carry their own trace: neither takes a late-signal rule or an eligibility
    trace of its own (§8.13). Under exploration at the synapse the decisions are
    the synapses', the hazard's row is posted for them (§8.16) and the read
    settles it into the score the same way; hebb is refused there (§8.3).
    """
    if eligibility not in ELIGIBILITIES:
        raise ValueError(f"unknown eligibility {eligibility!r}; choose from {', '.join(ELIGIBILITIES)}")
    if eligibility == "hebb" and getattr(network, "exploration", "neuron") == "synapse":
        raise ValueError("hebb is refused under exploration at the synapse: the decisions are the synapses', and it has no "
                         "neuron decision to centre (§8.3)")
    if not getattr(network, "explores", False):  # §8.3: no eligibility runs where the threshold decides
        raise ValueError("the reinforce rule refuses to learn where the threshold decides: REINFORCE estimates a gradient from the randomness of the decision, and with no width there is no randomness to estimate from. Give the network a positive ESCAPE_DELTA (--delta) (§8.3)")
    if arrays(network):
        return network.reinforce(advantage, lr, eligibility)
    network.settle_scores()  # under the evidence accumulator the open arrivals' debit is brought into the score first (§8.11)
    if not advantage:
        return 0
    low, high = network.weight_range
    step = lr * advantage
    # §8.4: every synapse into an unforced neuron moves by what its per-decision entries summed to
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


class Teacher:
    """The teacher of AUTHORITY.md §9: it stands outside the network, reads the output zone once an epoch,
    scores it, pays the one rule that pays at the read, and keeps its book.

    Use `teacher.epoch()` in place of `run_epoch(network)`. With
    `rule="reinforce"` it applies the update of §8 after the epoch; with
    `rule="local"` it pays nothing and the local rules of §10 are the whole
    of the learning (§9.1: "a run may have none"). Either way it keeps the
    firing rates, homeostasis and un-sticking in §9.2's order, and reports.
    """

    def __init__(
        self,
        network: Network,
        target: str = TARGET,
        lr: float = LR,
        eligibility: str | None = None,
        baseline_rate: float = BASELINE_RATE,
        window: int = WINDOW,
        seed: int | None = None,
        homeostasis: float = 0.0,  # §9.9: off unless a run asks; HOMEOSTASIS is the value it would run at
        target_rate: float = TARGET_RATE,
        discharge: bool = False,
        unstick: float = 0.0,  # §9.10, the same
        unstick_target: float = UNSTICK_TARGET,
        critic: str = CRITIC,
        rule: str = RULE,
    ):
        if rule not in RULES:
            raise ValueError(f"unknown learning rule {rule!r}; choose from {', '.join(RULES)}")
        self.rule = rule
        network.rule = rule
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
        if critic not in CRITICS:
            raise ValueError(f"unknown critic {critic!r}; choose from {', '.join(CRITICS)}")
        if critic not in ("row", "population", "class", "graded", "evidence") and target not in ("reversed", "copy"):
            raise ValueError(f"the {critic} critic reads the output as a word, which needs the reversed or copy target")
        if critic in ("class", "graded", "evidence") and target != "label":
            raise ValueError(f"the {critic} critic scores a dataset's label (§8): its target is label")
        self.critic = critic
        if eligibility is None:  # §8.3: where a run names none and the decision is a draw, the eligibility is hazard
            eligibility = ELIGIBILITY
        if eligibility not in ELIGIBILITIES:
            raise ValueError(f"unknown eligibility {eligibility!r}; choose from {', '.join(ELIGIBILITIES)}")
        if eligibility == "hebb" and getattr(network, "exploration", "neuron") == "synapse":
            raise ValueError("hebb is refused under exploration at the synapse: the decisions are the synapses', and it "
                             "has no neuron decision to centre (§8.3)")
        if rule == "reinforce" and not getattr(network, "explores", False):  # §8.3
            raise ValueError("the reinforce rule refuses to learn where the threshold decides: REINFORCE estimates a gradient from the randomness of the decision, and with no width there is no randomness to estimate from. Give the network a positive ESCAPE_DELTA (--delta) (§8.3) before the Teacher")
        if lr < 0 or homeostasis < 0 or unstick < 0:
            raise ValueError("learning rate, homeostasis and unstick rates must not be negative")
        if not 0.0 < unstick_target < 1.0:
            raise ValueError(f"unstick target firing rate must be between 0 and 1, got {unstick_target}")
        self.unstick = unstick
        self.unstick_target = unstick_target
        self.unstuck_count = 0  # how many epoch-nudges the un-sticking has applied
        self.history: list[dict] = []  # one entry per progress report; saved in checkpoints
        if not 0.0 < target_rate < 1.0:
            raise ValueError(f"target firing rate must be between 0 and 1, got {target_rate}")
        self.homeostasis = homeostasis
        self.target_rate = target_rate
        self.discharge = discharge  # zero every potential between inputs instead of letting it leak
        self.network = network
        self.target = target
        self.lr = lr
        self.eligibility = eligibility
        network.centre(eligibility == "hebb")  # §8.6: under hebb every neuron charges its decisions against its own expectation
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
        """Write one line per epoch to `path` (CSV, appended): epoch, time, baseline, score."""
        import os
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._trace = open(path, "a", buffering=1)
        if new:
            self._trace.write("epoch,time_ms,baseline,score\n")

    def close_trace(self) -> None:
        if self._trace is not None:
            self._trace.close()
            self._trace = None

    def epoch(self, bits: Sequence[bool] | None = None, verbose: bool = True) -> float:
        """Run one epoch with exploration noise, then learn from it. Returns its reward."""
        run_epoch(self.network, bits, rng=self.rng, verbose=verbose, discharge=self.discharge)
        return self.step()

    def step(self) -> float:
        """Score the epoch that has just run and reinforce. Returns its reward (accuracy)."""
        reward = CRITICS[self.critic](self.network, self.target)
        if self.baseline is None:
            self.baseline = reward
        advantage = reward - self.baseline
        if self.rule == "reinforce":
            reinforce(self.network, advantage, self.lr, self.eligibility)
        update_rates(self.network)
        homeostasis(self.network, self.homeostasis, self.target_rate)
        self.unstuck_count += len(unstick(self.network, self.unstick, self.unstick_target))
        self.baseline += self.baseline_rate * (reward - self.baseline)
        self.epochs += 1
        if self._trace is not None:
            self._trace.write(f"{self.network.epoch},{self.network.time:g},{self.baseline:.6g},{reward:.6g}\n")
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
        verb = "scoring" if self.rule == "local" else "learning"
        if self.average is None:
            return f"{verb} {self.target}: no epochs yet"
        if self.rule == "local":
            settings = "nothing pays at the read (§9.1); the local rules are the whole of it"
        else:
            settings = f"{self.eligibility}, lr {self.lr:g}"
        if self.critic != "row":
            settings += f", critic {self.critic}"
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
