# Sweep mnist-evidence-ff (September 16, 2026): the two-layer feedforward network

Byron, 03:05 MDT: "I would like the outputs kept apart from one another
again. With hidden=0 we have no cycles, eliminate interference from other
output neurons, a two-layer feedforward network. Please do a comparable
sweep to the one you just did with no_hidden so we can narrow down the
source of unlearning." So the scaled rule with the outputs apart (AUTHORITY.md
§3.4, decision 5 of §8) and `--hidden-neurons 0`: a goo of 445 in which the
395 inputs project onto the 50 outputs and nothing else, 1,152 synapses, each
output hearing 22 or 23 inputs at 0.056, the inputs hearing nothing and left
at the container's threshold (§5.2). The same fifteen arms as the two sweeps
before it: T in {1, 2, 4} x LR in {0.001, 0.003, 0.006, 0.01, 0.03}, one
seed, 10,000 epochs, threshold 0.6 with the floor at -2.4, Delta 0.455;
fifteen arms on fifteen workers, about 50 epochs a second each, landed 03:25
MDT. Chance is the network's learning-off reward at each temperature (seed 1,
500 epochs; `runs/mnist-evidence-ff-chance.json`). The figure
(`mnist-evidence-ff-score.png`) draws the traces against it.

Each cell gives this network first, then in parentheses the no-hidden and
the 199-hidden sweeps under the open rule (outputs onto every zone).

**Reward over the last tenth: feedforward (no hidden, open; 199 hidden, open)**

| T \ LR | 0.001 | 0.003 | 0.006 | 0.01 | 0.03 | learning off |
|---|---|---|---|---|---|---|
| 1 | -4.24 (-4.26; -3.94) | -4.14 (-4.01; -4.06) | -3.75 (-3.62; -3.65) | -3.60 (-3.34; -3.38) | -2.55 (-2.62; -2.98) | -5.51 (-6.61; -5.60) |
| 2 | -3.02 (-2.89; -2.76) | -2.97 (-2.88; -2.83) | -2.80 (-2.80; -2.83) | -2.76 (-2.72; -2.75) | -2.48 (-2.67; -2.62) | -3.40 (-3.84; -3.39) |
| 4 | -2.54 (-2.58; -2.47) | -2.53 (-2.50; -2.43) | -2.49 (-2.46; -2.46) | -2.49 (-2.41; -2.41) | -2.40 (-2.44; -2.42) | -2.63 (-2.76; -2.60) |

**Fraction right over the last tenth, the class critic: feedforward (no hidden, open; 199 hidden, open)**

| T \ LR | 0.001 | 0.003 | 0.006 | 0.01 | 0.03 | learning off |
|---|---|---|---|---|---|---|
| 1 | 0.08 (0.07; 0.09) | 0.08 (0.07; 0.09) | 0.07 (0.07; 0.07) | 0.07 (0.06; 0.08) | 0.06 (0.05; 0.06) | 0.06 |
| 2 | 0.07 (0.07; 0.07) | 0.07 (0.07; 0.07) | 0.08 (0.07; 0.08) | 0.08 (0.08; 0.09) | 0.07 (0.05; 0.08) | 0.06 |
| 4 | 0.07 (0.07; 0.07) | 0.07 (0.08; 0.09) | 0.07 (0.08; 0.08) | 0.08 (0.08; 0.08) | 0.09 (0.06; 0.07) | 0.06 |

**The output zone's final rate memory, feedforward: the fraction of epochs an output spiked in**

| T \ LR | 0.001 | 0.003 | 0.006 | 0.01 | 0.03 | learning off |
|---|---|---|---|---|---|---|
| 1 | 0.92 | 0.80 | 0.59 | 0.34 | 0.07 | 0.92 |
| 2 | 0.93 | 0.90 | 0.77 | 0.74 | 0.23 | 0.92 |
| 4 | 0.93 | 0.93 | 0.87 | 0.84 | 0.49 | 0.92 |

**The last epoch's class sums, feedforward** (ten classes, five outputs each):

| T \ LR | 0.001 | 0.01 | 0.03 |
|---|---|---|---|
| 1 | [13, 12, 14, 15, 9, 12, 13, 12, 18, 10] | [4, 3, 3, 5, 4, 3, 4, 3, 2, 3] | [1, 0, 0, 1, 1, 0, 1, 0, 0, 0] |
| 2 | [15, 11, 14, 16, 12, 12, 12, 12, 17, 12] | [12, 10, 5, 12, 11, 9, 9, 6, 12, 10] | [2, 2, 0, 2, 1, 3, 2, 1, 0, 1] |
| 4 | [16, 11, 15, 18, 13, 8, 13, 12, 16, 12] | [11, 10, 8, 9, 7, 10, 9, 14, 13, 11] | [8, 6, 10, 7, 4, 10, 6, 7, 9, 5] |

## What it narrows down

**The same picture a third time.** Every arm ends above its chance, the
fraction right is 0.06 to 0.09 in every cell against a learning-off 0.06,
and the reward's rise is once more the output zone going quiet and even:
its rate memory falls with the learning rate from 0.92 to 0.07 at T = 1, and
the last epoch's class sums go from twelve to eighteen a class at LR 0.001
(the rest, 2.5 spikes a neuron, even across the ten) to one or none at LR
0.03. There is no cycle here, no lateral projection among the outputs, no
hidden neuron: whatever the estimator does, it does on the 1,152
input-to-output synapses alone, a linear classifier's weights, which carry
80 to 87% on these bits (`mnist-by-eye.md`). So the source of unlearning is
not the wiring. It is the estimator and its critic on a single layer.

**A gradient check** (`mnist-evidence-gradient.py`; learning off, 2,000
epochs, T = 2): the hazard scores of every epoch, weighted by the
advantage under three critics computed from the same counts, accumulate to
each critic's estimated gradient per synapse, G. Beside it the supervised
direction d for a pixel-to-class weight, P(pixel on | the output's class) -
P(pixel on), the sign a linear classifier's gradient has on average; 501
of the 1,152 synapses have |d| > 0.02.

| critic | sign agreement of G with d (chance 0.5) | correlation | per-output mean push, share of |G| |
|---|---|---|---|
| evidence, ln q | 0.547 | +0.067 | 0.61 |
| probability, q | 0.521 | +0.044 | 0.54 |
| class, 1 or 0 | 0.511 | +0.054 | 0.53 |

After two thousand epochs the estimate points the supervised way on barely
more than half of the synapses, under any of the three critics, and more
than half of its size is a per-output mean that the label cannot see. The
discriminative signal is there -- the agreement is above a half -- but it is
the small term; the estimator's noise is the large one, and while a run
accumulates the small term its weights walk under the large one, which is
what the sweeps show as quieting and evening. Over 10,000 epochs the same
check gives sign agreements of 0.555, 0.543 and 0.547 and correlations of
+0.19, +0.17 and +0.11 (evidence, probability, class): the correlation grows
with the epochs about as the square root of their number, which is how a
real signal accumulates under noise, while the sign agreement does not move,
because the per-output push still decides most signs -- 0.73 of |G| under the
evidence critic at 10,000 epochs, and now a push toward quiet (-0.69 a
synapse), the drift the sweeps show.

**The input code the outputs receive.** With learning off, an on-pixel
input fires 3.5 spikes an epoch and an off-pixel input 1.75, each with a
spread of a spike -- the off pixel is not silent, it fires at the hazard's
rest, since escape noise applies to every neuron at the same width (§5.2).
The clocks fire 3.5 like an on pixel. So each of an output's 22 synapses
carries a pixel signal of 1.8 spikes on a rest of 1.8, and a synapse from an
off pixel earns eligibility as if its pixel were half on. That halves the
contrast the linear classifier gets and blurs the credit, epoch by epoch.
Whether the inputs should carry the hazard's rest at all -- a driven neuron
firing by its drive alone would give the outputs the bits as they are -- is
a rule about the neuron, and Byron's to call.
