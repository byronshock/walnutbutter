# Exploration at the synapse on mnist: the first measurements, and why its outputs stay busy

September 25, 2026. The engines carry exploration at the synapse (AUTHORITY.md §7.5–§7.9, §8.16–§8.17, 5.4b; PR #25).
This note records what the first measurements on mnist found, the diagnosis Byron asked for when they came back
at chance, and what is running now. Everything below was run from a frozen copy of the code at 94dd249 with its
own engine build, at niceness 19. The scripts, their summary tables and the investigators' reports are in
`docs/synapse-diagnosis-2026-09-25/`; the raw data (about 136 MB) is in the measurement worktree's
`runs/synapse-diagnosis-2026-09-25/`, which git ignores.

## The operating point

The standing accumulator cell of the September 24 sweeps, with no width: mnist, `--hidden-neurons 0` (goo 455:
395 inputs, 60 outputs, 1,425 input-to-output synapses), interval 100, threshold 0.6, floor −0.2, hazard
eligibility, TAU inf, the count read, the evidence critic, temperature 2, pickiness 2. Under the synapse: h0 0.01,
loglinear, count scaling, the full trace.

## 1. The engines agree there (plan §5.1)

The object engine and the Rust loop, compared with `==` on everything (`fast.compare`) at that point, both drives,
seeds 1–2, learning on at lr 0.002: all four arms agree over 40 epochs (`compare_config.py`).

## 2. The smoke, and a resume (plan §5.2)

3,000 epochs, seeds 1–3, both drives. A copy cut at epoch 1,500 and resumed from its checkpoint file finishes with
every checkpoint key equal to the uninterrupted arm's except `progress`, which records the invocation. The trace
CSVs differ only in where the points fall: each invocation traces every 1,000 epochs from its own start.

Every arm ends far below the uniform line (ln 0.1 = −2.30): last tenths −4.6 to −6.3, accuracy 0.05–0.11. So does
the neuron rule at width 0.3125 over the same 3,000 epochs (−3.9 to −5.3, accuracy 0.09–0.11): 3,000 epochs is
too early at this point for either rule.

## 3. The learning rate at 25,000 epochs (plan §5.3)

Seeds 1–3; the neuron rule beside it at the same horizon. Accuracy is the fraction right over the last 2,500
epochs; chance is 0.100.

| rule, drive | lr | accuracy | last-tenth score | estimator correlation | outputs firing every epoch |
|---|---|---|---|---|---|
| neuron | 0.0005 | 0.108 | −3.77 | 0.10 | 34 |
| neuron | 0.002 | 0.106 | −3.06 | 0.08 | 10 |
| synapse, charged | 0.0005 | 0.094 | −4.41 | 0.08 | 47 |
| synapse, charged | 0.001 | 0.097 | −4.19 | 0.11 | 44 |
| synapse, charged | 0.002 | 0.114 | −3.92 | 0.13 | 45 |
| synapse, charged | 0.005 | 0.114 | −5.01 | 0.09 | 32 |
| synapse, charged | 0.01 | 0.101 | −6.47 | 0.05 | 27 |
| synapse, rate | 0.0005 | 0.098 | −4.82 | 0.09 | 44 |
| synapse, rate | 0.001 | 0.115 | −4.58 | 0.12 | 42 |
| synapse, rate | 0.002 | 0.103 | −4.50 | 0.08 | 39 |
| synapse, rate | 0.005 | 0.088 | −5.98 | 0.05 | 34 |
| synapse, rate | 0.01 | 0.098 | −8.44 | 0.04 | 26 |

Nothing has learned by 25,000 epochs under either rule, so the rate cannot be chosen on accuracy yet. Rates of
0.005 and above hurt the synapse rule; the charged drive scores better than the rate drive at every rate up to
0.002; the estimator's correlation with the supervised direction is as good as the neuron rule's. The synapse
arms' 298 "stuck off" neurons are all inputs of rarely-on pixels, which under the synapse rule never fire on their
own; no output is among them. Speed under the full sweep: synapse 11–14 epochs/s on the rate drive, 5 on the
charged; the neuron rule 18.

## 4. The diagnosis

Asked: why do the synapse rule's outputs fire nearly every epoch (42 of 60 against 10)? Four investigations, a
synthesis and an intervention test (`reports/`).

**They are busy from the start, not made busy.** With learning off, 41–50 of the 60 outputs fire every epoch under
both rules, flat over 5,000 epochs. The point makes it so: accumulators with no leak, a floor at −θ/3 that
discards most inhibition, about 96 relayed arrivals per output per epoch, weights drawn on [−1, 1]. Removing the
floor with the weights frozen takes one seed's busy outputs from 49 to 10. The neuron rule quiets the outputs over
training (seed 1: 40 busy at epoch 250, 11 at 5,000, 7 at 25,000); the synapse rule does not (47, 47, 42). There is
no net upward drift: the mean weight change is negative in 27 of 30 synapse arms.

**Its steps are about five times smaller.** An output's incoming weights are credited only through the output's
own synapses' decisions (§8.16), and on a goo with no hidden neurons its only synapse is the read synapse: about
half an escape per epoch, posted only while its potential is above zero. The eligibility summed over an output's
fan-in is 0.8–1.8 under the synapse rule against 5.7–8.5 under the neuron rule. In weight movement, synapse lr
0.002 matches neuron lr 0.0005 (mean change −0.027 against −0.025, spread 0.21 against 0.19), and synapse 0.01
matches neuron 0.002.

**The rule cannot see its outputs' own spikes.** About 90% of an output's count is its own spikes, which are
deterministic crossings of its threshold; their randomness comes from the inputs' whispers, whose odds no weight
sets. The rule credits the fan-in only with the reward effects of the read escapes, the other 4–11%. Measured: an
output's eligibility correlates −0.004 to +0.003 with its own spikes and +0.63 to +0.78 with its read escapes
(under the neuron rule, +0.18 to +0.25 with its spikes). This is inferred from those numbers and §8.16's text; it
has not been tested by intervention.

**A trap, its extreme case.** A synapse strong enough that one arrival fires its output from anywhere at or above
the floor — weight at least θ − floor, 4θ/3 at this point — is credited exactly nothing, for good. Its arrival notes
the output's gain, the output spikes in the same wave, and the settle posts x·G − B = G − G = 0 (§8.16; the
deterministic spike posts nothing, §6.13). A random walk carries synapses over that line and they stay: from 10–24
per seed at the start to 45–65 at lr 0.002 and 77–94 at lr 0.01 by 25,000 epochs; 87–97% of the outputs holding one
fire every epoch. Eligibility drops 75–215 times at exactly that line; the neuron rule has no such drop. Tested by
intervention (`test/`):

- Moving the floor moves the line: at −0.6 the drop sits at 2θ, and the class between 4θ/3 and 2θ, frozen at −0.2
  (0.0002–0.0008 eligibility per epoch), learns at 0.11–0.17.
- Capping the trapped synapses at 0.9θ in two 25,000-epoch checkpoints quiets their outputs at once (busy 29 → 18,
  25 → 12); under the synapse rule the capped synapses walk back over the line (35–42 within 1,000 epochs), under
  the neuron rule they do not (15 and 7 by 3,000).

**A second blind spot at the silent end.** An output held below zero posts nothing (the V > 0 gate of §8.16) and
stops learning. A deep floor with a high rate shows it: at floor −1.2 and lr 0.01, 37 of 60 outputs fired no spike
of their own in the last 250 epochs, and 93% of the frozen windows were synapses into them.

**Ruled out.** The critic rewarding activity (its correlation with the total count stays within ±0.065, and one
more count always costs, −0.005 to −0.024); an upward bias in the rule (its expected entry is exactly zero, and the
measured sums straddle zero); the ventured input as the source (removing its delivery changes spikes by −1.6% on
the rate drive); the rest hazard setting the level (a thirty-fold change moves spikes by 5%).

**A loose end, settled: noise.** One investigator measured the synapse rule's summed eligibility at +0.35 ± 0.12
per epoch, where §8.16 makes its mean exactly zero (the gate, the trace, m and F are fixed before a wave's draws, and
the entry's mean over the draws is F(1 − e^−m)c − F e^−m m = 0). Checked by two agents independently
(`elig/`, `reports/eligibility-*.txt`): the object engine's escapes, read counts and gain moves match the clause
at every one of about 7 million waves; the settled scores equal the per-wave direct posting to about 1e-13;
objects and Rust agree with `==` on every run, resumed 25,000-epoch networks included; twelve fresh Rust records
give −0.009 ± 0.081 per epoch (n 40,000). The statistic is skewed, its mean riding on the top 1% of epochs; the
original four records were about a one-in-three-hundred draw.

## 5. Running now

Byron's choice: the synapse rule at 200,000 epochs with the trap out of reach and at the pinned floor. Synapse,
rate drive, lr 0.002 and 0.005, floor −0.2 and −1.2 (at −1.2 the line sits at 3θ, above the weight cap for every
output), seeds 1–3; the neuron rule at floor −1.2, lr 0.0005, seeds 1–3 as the reference; a trace point every
5,000 epochs, to be judged on accuracy against the neuron rule's own windows (0.15–0.18 over epochs 50,000–100,000,
0.16–0.22 over 100,000–200,000 at floor −0.2). `floorsweep.sh`, launched 17:26 MDT.

What the diagnosis predicts: the deep floor removes the trap and lowers the starting activity (busy at lr 0:
46 → 14), but it does not give the rule sight of its outputs' spikes, and at the higher rate it may silence outputs
at the other end. If the arms at −1.2 still sit at chance by 200,000 epochs while the neuron rule leaves it, the
blindness to deterministic spikes, not the trap, is what stands in the way.

## Open for Byron and Cedric

Whether §8.16 should see a deterministic spike — the trap and the blindness both follow from the rule as written
and from §6.13's "the deterministic spike posts nothing: it is not a draw" — or whether the synapse is meant to
learn only through its own chances, as Byron and Cedric's idea of §1's closing note has it.
