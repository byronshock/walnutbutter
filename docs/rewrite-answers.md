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
