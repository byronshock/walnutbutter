# Sweep mnist-evidence (September 16, 2026)

Temperature x learning rate under the evidence critic (AUTHORITY.md §8,
decision 4), on mnist's goo of 644 under the scaled rule (§3.4): T in
{1, 2, 4} x LR in {0.001, 0.003, 0.006, 0.01, 0.03}, one seed, 10,000
epochs an arm, GOO_THRESHOLD 0.6 with the floor at -2.4, Delta 0.455, the
hazard eligibility, the Rust loop; fifteen arms on fifteen workers, launched
02:07 MDT and landed 02:26. Byron: "Let's start small." The figure
(`mnist-evidence-score.png`) draws each arm's reward as a rolling mean of ten
single-epoch samples, against its temperature's chance.

Chance is not the uniform estimate's ln 0.1 = -2.30: the softmax reads the
hazard's rest noise as evidence, so with learning off the reward on this
network averages -5.60 at T = 1, -3.39 at T = 2 and -2.60 at T = 4 (seed 1,
500 epochs; `runs/mnist-evidence-chance.json`), with a spread of 3.2, 1.6
and 0.8 an epoch. Each arm is read against its own column here.

**Reward over the last tenth (ln q_label; 0 is perfect)**

| T \ LR | 0.001 | 0.003 | 0.006 | 0.01 | 0.03 | learning off |
|---|---|---|---|---|---|---|
| 1 | -3.94 | -4.06 | -3.65 | -3.38 | -2.98 | -5.60 |
| 2 | -2.76 | -2.83 | -2.83 | -2.75 | -2.62 | -3.39 |
| 4 | -2.47 | -2.43 | -2.46 | -2.41 | -2.42 | -2.60 |

**Fraction right over the last tenth (the class critic: the label's class loudest outright)**

| T \ LR | 0.001 | 0.003 | 0.006 | 0.01 | 0.03 | learning off |
|---|---|---|---|---|---|---|
| 1 | 0.09 | 0.09 | 0.07 | 0.08 | 0.06 | 0.06 |
| 2 | 0.07 | 0.07 | 0.08 | 0.09 | 0.08 | 0.06 |
| 4 | 0.07 | 0.09 | 0.08 | 0.08 | 0.07 | 0.06 |

**Neurons of 644 stuck on at the end (a rate memory above 0.99: a spike in every epoch)**

| T \ LR | 0.001 | 0.003 | 0.006 | 0.01 | 0.03 | learning off |
|---|---|---|---|---|---|---|
| 1 | 317 | 226 | 148 | 83 | 18 | 0 |
| 2 | 340 | 283 | 258 | 191 | 54 | 0 |
| 4 | 346 | 335 | 306 | 265 | 117 | 0 |

## What the reward's rise is, and is not

The traces begin with single-epoch samples far above chance -- at T = 1 and
LR 0.001, -0.2, -0.7 and -0.9 in the first five hundred epochs against a
resting mean of -5.6 -- which looked like learning followed by collapse. It is
the log score's skew: at T = 1 the label's class is near the top by luck in a
tenth of epochs and scores near -1 there, and far below in the rest and
scores -8 to -15, so the mean is -5.6 while single samples near -1 are
ordinary. A probe of four arms (`mnist-evidence-diag.py`: every epoch's class
critic, output and interior counts, the fraction of outputs at the refractory
ceiling and the spread of the ten class sums, averaged over windows of a
hundred epochs, the first 2,500 epochs) reads:

| arm | epochs to | reward | right | output spikes a neuron | interior spikes a neuron | spread of the class sums |
|---|---|---|---|---|---|---|
| T = 1, LR 0.001 | 100 | -5.37 | 0.06 | 2.97 | 2.84 | 3.10 |
|  | 1,000 | -4.92 | 0.03 | 2.95 | 2.80 | 2.46 |
|  | 2,500 | -4.29 | 0.10 | 2.93 | 2.80 | 2.28 |
| T = 1, LR 0.03 | 100 | -6.05 | 0.09 | 2.24 | 2.19 | 3.68 |
|  | 1,000 | -4.15 | 0.03 | 1.06 | 0.96 | 1.85 |
|  | 2,500 | -3.36 | 0.03 | 0.64 | 0.74 | 1.51 |
| T = 2, LR 0.006 | 100 | -3.19 | 0.10 | 2.90 | 2.82 | 3.09 |
|  | 1,000 | -3.00 | 0.06 | 2.70 | 2.69 | 2.58 |
|  | 2,500 | -2.78 | 0.12 | 2.87 | 2.57 | 2.21 |
| T = 4, LR 0.01 | 100 | -2.50 | 0.14 | 2.93 | 2.81 | 3.21 |
|  | 1,000 | -2.57 | 0.06 | 2.84 | 2.69 | 2.89 |
|  | 2,500 | -2.33 | 0.10 | 2.71 | 2.52 | 2.27 |

**The class critic's fraction right is at chance in every window of every
arm, from the first hundred epochs on** (0.03 to 0.14 over windows of a
hundred; chance 0.06 to 0.10). No output ever reaches the ceiling. What
rises is the reward, and it rises as the spread of the class sums falls:
from 3.1 to 2.3 at T = 1, LR 0.001, with the counts held at three spikes a
neuron; from 3.7 to 1.5 at T = 1, LR 0.03, where the outputs and the interior
go quiet, 2.2 to 0.6 spikes a neuron -- the drift toward silence of the
Delta sweep again. Either way the class sums become more even and steadier,
the estimate more uniform, and ln q_label climbs toward the uniform estimate's
-2.30 without the label's class winning any more often. That is the log
score's own gradient: for a label-blind network E[ln q_y] rises whenever the
noise in q falls (Jensen), and the network has two cheap ways to lower that
noise, evening out and quieting, both of which the estimator finds. The
digit's gradient is there too, but it is the small term beside a systematic
one. The "stuck on" counts of the sweep's table are not the ceiling: a rate
memory above 0.99 is a spike in every epoch, which three spikes an epoch at
rest gives to a good part of the network over ten thousand epochs, and the
quieter high-LR arms have fewer of them for that reason alone.

**What would take the decoy gradient away.** The expected reward of a
label-blind network under the plain probability, r = q_label at the same
temperature, is exactly a tenth whatever the noise, since the label is drawn
independently of the counts and E[q_y] = (1/10) sum_k E[q_k] = 1/10. Its
only gradient is the discriminative one: (1 - q_y) q_y / T per spike on the
label's population, -q_y q_k / T on class k, largest where the label's class
is already near the top. The Brier score keeps a variance term (-sum q_k^2)
and so the same decoy. Byron's to call.
