# The rewrite: what Byron has decided, in order

*Answers to `rewrite-decisions.md`, in Byron's words, as they are given. A
clause is written into `AUTHORITY.md` only from an answer here.*

## 1. What fires a neuron, once θ is deprecated?

**Answered September 17, 2026, 14:50 MDT.** *"We will eventually get to A!
This is exciting. But first we must tackle neural spike escape noise as a
potential mechanism. I will argue later that it's not the mechanism, but we
have a learning rule derived for it right now that is working right now, at
least in a feedforward fashion, and not well."*

So: **escape noise fires the neuron, for now** — the firing decision is a
draw whose rate rises with the potential (option B of the three offered) —
and the eligibility derived from it, the single-spike rule's hazard row,
survives with it. Option A, exploration alone with the potential driving
nothing, is where the project is going, not where the rewrite starts.

*What this leaves open, and is asked next:* the escape hazard's rate is
quoted today as a margin above θ in units of ESCAPE_DELTA × θ, so with θ
deprecated the rule needs a scale from somewhere else.

**The scale, answered 14:53 MDT.** *"We are going to keep the level. The
system must implement what I was just using. We can make changes later to
this."* So θ stays exactly as the working runs had it — the level the margin
is measured from and the unit the width is quoted in — and "deprecate θ" is
a later change, not the rewrite's starting point. The specification's first
job is to state the system Byron has been running, not an improved one.

## 2. The baseline the specification states

**Answered September 17, 2026, 14:56 MDT,** correcting the table of the last
working sweep: *"neuron: NOT leaky, but please leave the leak option with its
analysis. 2 hops."*

So the specified neuron is the **evidence accumulator**: the potential is the
sum of the arrivals since the last spike and nothing decays. The leak stays
as an option — TAU, selectable per run — and the analysis of it stays with
it: the half-life against the hop, what a steady per-hop input settles at,
and the leak sweep of September 16 that found the fast leak best on every
seed while confounding the leak with the learning rate (`RECORD.md` §1.2,
§5.1, §8).

The clock is **two hops**: a hop of 2.5 ms against the 5 ms refractory
period, so a spike sent around a two-way pair returns exactly as the period
ends.

*To be recorded with the numbers:* every measurement in the record, the
0.217 right of the partly connected goo included, was made at TAU 2 ms and
three hops. The specified defaults are now the accumulator and two hops, so
the record's numbers belong to a configuration the specification no longer
starts from, and the clauses must say so rather than let the two be read as
one.

## 3. The drive

**Answered September 17, 2026, 14:58 MDT:** *"Please include the Poisson
drive as it ran."* So the specified drive is the rate drive of `RECORD.md`
§4.3 exactly as the sweeps ran it: an independent Poisson process drives each
input neuron across the epoch, arrivals drawn Exp(λ) apart from the network's
own seeded stream, an arrival landing while the neuron is refractory dropped,
so the spike train is a renewal process with dead time; specified by its
coefficient of variation, INPUT_CV 0.6, from which λ = 0.133/ms follows, 80
Hz, 2.8 spikes across a 35 ms epoch. The deterministic spike every TARGET_ISI
is a later change, not the rewrite's starting point.

## 4. What the specification drops, and what it keeps

**Answered September 17, 2026, 15:02 MDT**, item by item against the list in
`rewrite-outline.md`. In his words where they decide something:

**Kept.**
- **The quash (§6.11), as a non-default.** It is the only rule that ever
  closed a short loop, and the record has it as the difference between 0.978
  and chance on a two-hop copy; it is off for a feedforward network and
  matters again when recurrence returns.
- **The array engine, with tolerances, as a foundational rule** — *"keep,
  with tolerances, as a foundational rule; all platforms are built from one
  clear authority."* So the specification states, as a rule about the
  system rather than about any engine, that every platform is built from
  this file alone, and names the tolerance where floating-point
  associativity forbids bit agreement (the arrays sum a wave in matrix
  order and use numpy's exponentials; spikes agree exactly, continuous
  quantities to a part in a billion).
- **The critics** other than evidence: row, class, graded, population,
  sustained and the decoded critics.
- **Homeostasis and un-sticking, as non-defaults.**

**Dropped from the specification, kept in the record.** The dopamine rule
and its eight constants; the external teacher of §6.9 with RATE_ON/RATE_OFF
and TEACHER_CREDIT; ADALINE; leaky Hebb with HEBB_RATE and SYNAPSE_TAU; the
bored clock and BORED_AFTER; the reads fired, again, window and rate; the
perturb, wrong_hebb and count_hebb eligibilities with SIGMA, EXPLORE and
COUNT_MEMORY; every problem but mnist; the codings and the error-correcting
codes and the input flips, keeping complement coding and the clock neurons
that mnist uses; the weight decay and the bit-0 punishment; the visualizer;
the goo wirings zones, zones-equal, uniform and scaled-open and the direct
projection; and the input permutation.

**Dropped as code, kept as a direction.** *"(1) drop but keep the idea of
dopamine reinforcement around. It's a very important direction for Cedric's
work, but the implementation was mine and I lacked understanding."* So
dopamine leaves the specification as a mechanism and stays in it as an
intention: a reward produced locally at a spike and consumed globally,
Cedric's to take up, with the record's §6.2-§6.6 as the first attempt and
its own account of why it did not learn.

**The containers, answered 15:08 MDT:** *"archive the hex containers,
lattice, spread, the whole shebang. We'll bring back small world shortcuts
later."* So the specification carries goo alone. The hex grid, the hexagonal
columns and the lattice or spread leave it, with the hex metric, the
guaranteed neighbourhood, the positions and OMEGA, REACH and ROWS; they stay
in `RECORD.md` §2-§3 and in the code at the tag `lab-notebook-2026-09-17`,
which is what archived means here — nothing is deleted, and the rebuild does
not carry them. Small-world shortcuts are named as an intention to return,
not as a clause: they were the only thing that shortened a trip across the
grid, and goo has no far.

**The critics, answered 15:09 MDT:** *"evidence, row, graded"*. Three
critics survive: evidence (the class sums read as log-odds at TEMPERATURE,
mnist's), row (the fraction of outputs matching a target pattern) and graded
(the fraction of the other classes the label's class out-spikes). The
sustained, population and decoded critics go with the problems they scored.

*One consequence to settle:* the fraction right that every mnist run reports
-- 0.217 on the last sweep, the number the project is judged by -- is the
class critic's, computed beside the evidence reward. With the class critic
dropped, that reading has to be kept as a **reported measure** rather than a
critic, or the runs stop reporting the thing they are read by.

**The codings, confirmed 15:10 MDT.** Kept: complement coding on the inputs,
complement coding on the outputs, and the clock neurons -- what mnist runs.
Dropped: raw coding, population coding, population-complement, the Hamming
and parity error-correcting codes, and the input bit flips (FLIP).

**The fraction right, settled the same minute.** Kept as a **reported
measure** and not a critic: every run reports the fraction of epochs in which
the label's class out-spikes every other, beside the evidence reward it is
paid on. *Claude's proposal, accepted in "confirm and proceed"; to be
corrected in a word.*

**Proceed.** The drafting of the new `AUTHORITY.md` begins from here: the
baseline of answers 1-3, the scope of answer 4.

**The class critic, kept (15:12 MDT):** *"keep the class critic"*. So four
critics survive: evidence, row, graded and class -- the 1-or-0 that pays when
the label's class out-spikes every other, a tie losing. The fraction right
reported beside the evidence reward is that critic's own number, so the
reported measure and the critic are one rule again rather than two.
