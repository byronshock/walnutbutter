# The second round: Byron's answers

*Answers to `docs/rewrite-decisions-2.md`, in Byron's words, as they are
given. A clause is changed in `AUTHORITY.md` only from an answer here.*

## Part 2a — the confirmations

**1. §0.4 — the refractory period as a computational feature. Confirmed**
(September 17, 2026): *"confirmed"*. No rule may work around it; the reading
stands as a rule and stops being Claude's.

**2. §1.5 and §8.14 — what a checkpoint holds of a synapse. Corrected**
(September 17, 2026): *"a checkpoint should hold a synapse's score for
analysis purposes."*

So the score $e_{ij}$ is checkpointed, and both clauses change. §1.5 said the
checkpoint carries the stamp, the trace and the note but not the score; §8.14
said the same of the rule's own state. Two consequences follow and are not
separate decisions:

- The constraint goes with it. Both clauses said a checkpoint is therefore
  *taken at a read*, where the score has just been paid. That was a
  consequence of not carrying the score, not a rule of its own, so a
  checkpoint may now be written anywhere and still round-trip.
- `persistence.py` gains a field, and §12.9's round-trip covers it.

The reason is Byron's and is worth keeping in the clause: the score is what a
synapse earned since the last read, and a checkpoint that drops it cannot be
read after the fact for what the network was learning at the moment it was
saved.

**3. §1.9 — the read re-bases the note. Deferred** (September 17, 2026):
*"Can be deferred."* The clause stands as Claude's reading; $B_{ij}
\leftarrow x_{ij}E_j$ is forced once arrivals stay open and no debit is paid
twice, but it is not yet Byron's.

**4. §3.4 — the clock's slack. Confirmed** (September 17, 2026):
*"4 confirmed"*. The built rule is the relative slack, and the record's "times
are rounded to a nanosecond" is a mis-description of it: the clock is in
nominal milliseconds, so TOLERANCE $10^{-9}$ is a picosecond under a
millisecond of clock time and grows in proportion after.

*Raised with it and not yet answered:* the slack is 4.5 million ulps at every
$t$, and because it grows with the clock while §0.8 says the network never
stops, it reaches a whole hop at about $7\times10^{7}$ epochs — a month of
clock time — past which two events a hop apart share a wave. TOLERANCE
$10^{-12}$ would still be 4,500 ulps and push that to 79 years. Changing it
changes which events share a wave, so it is Byron's.

**5. §3.6 — the order the fire phase asks in. Delegated** (September 17,
2026): *"your choice"*. **Kept as a rule.** The order neurons fire in is the
order their signals are pushed onto the schedule, and §3.7 makes push order
normative for how a later wave sums them; if the fire order is an
implementation's choice then push order is too, and the bit agreement of
§12.4 cannot be met. So: forced neurons first in the order the wave holds
them, then the neurons the wave touched in the order they were touched, then
every other neuron in index order.

**6. §3.7 — push order is part of the rule. Delegated** (September 17, 2026):
*"your choice"*. **Kept as a rule**, for the same reason as 5: §12.4 compares
the object engine and the Rust loop with `==`, and that is undecidable unless
the summation order is fixed. The array engine's matrix order is the named
exception and costs it the tolerance of §12.4.

**7. §4.11 — a neuron that hears nothing. Confirmed** (September 17, 2026):
*"confirmed"*. It keeps the container's quoted threshold and floor at scale 1,
so it keeps a width and fires at the hazard's rest like any other neuron.

**8. §5.8 — the driven mark. Deferred** (September 17, 2026): *"deferred"*.
That the mark is stated in the drive and used by §8.1's forced skip remains
Claude's reading.

**9. §6.2 — the floor applies once a wave. Confirmed** (September 17, 2026):
*"confirmed"*. On the wave's summed input, before anyone fires, which is what
makes a wave's result independent of the order its signals arrived in.

**10. §6.3 — what settles with no credit. Confirmed** (September 17, 2026):
*"confirmed"*. The floor, a forced spike and a discharge each close every open
arrival with no credit, their accrued debit settled.

**11. §6.5 — the hazard's engine requirements. Confirmed, with the clamp
promoted** (September 17, 2026): *"Yes, please. 11 can be confirmed
otherwise."*

The cap at $10^3$ and the $-\mathrm{expm1}(-m)$ form stay as engine
requirements: both exist so the three engines agree, and expm1 also keeps the
precision of a small $m$, which is the ordinary case — a quiet neuron on the
mnist goo carries $m \approx 4\times10^{-4}$ over one decision interval.

The clamp on $\Delta t$ is now stated as a rule of the decision rather than
an engine note. It is not an implementation's convenience: $\Delta t$ goes
negative in exactly one case, which the file itself creates — §1.5's
refractory test allows the clock's slack, so a neuron counts as recovered up
to that slack before its exposure begins — and a negative $\Delta t$ gives a
negative $m$, which is also the expectation posted to every incoming synapse
at a silent decision. Firing would be unaffected, since a uniform is never
below a negative probability; the learning rule would take a wrong-signed
entry across the whole fan-in, once per spike, always the same way.

**12-19. Confirmed** (September 17, 2026): *"Please confirm up through 19."*
§6.7 the draw's position and order; §6.8 every neuron decides at every wave;
§7.6 a resumed network keeps the shaping-function setting; §8.4 the
single-spike rule as derived; §8.5 the trace; §8.13 the late-signal question;
§9.2 the order at the read. (18, §8.14, was already settled by answer 2.)

**20. §11.14 — what mnist does not fix. Confirmed as written, with a caution**
(September 17, 2026): *"Option 3 above, and then warn that the container may
be a little rough around the edges. For now."*

The epoch's length, the container's threshold and its floor stay out of the
problem: they come from the constants of Appendix A or from the run. But the
file now says what that means, because the record's mnist numbers were taken
at a 100 ms epoch and a threshold of 0.6 with the floor at −2.4, where the
defaults are 35 ms and 0.2 with −0.8. Two changes:

- The preamble's caution is widened from TAU and hops to name the epoch
  length, the threshold and the floor as well, and says plainly that a run
  naming none of them is a configuration the record has not measured.
- §4.10 now says the container's numbers are rough for now, in Byron's words,
  and that a run taking them is taking a starting point and not a decision.

Not done, and not asked for: changing GOO_THRESHOLD itself. That 0.2 is the
wrong default is a separate question from where the constant lives.

**21. §12.8 — agreement proven on the configuration. Confirmed** (September
17, 2026): *"21. yes, confirm. We don't even have an engine built right now
and we are working on the system."*

The second half settles more than the clause. The engines of the rebuild do
not exist: what is in the tree is the lab notebook's code, frozen at
`lab-notebook-2026-09-17`, and the rebuild is from this file. So every "what
it requires of the engines" paragraph is a requirement on an engine yet to be
written, not a description of one that runs, and the file's own rule — every
clause implemented in every engine that runs it, and tested before it is
built — applies from the first line of the rebuild rather than as a
retrofit.

Two consequences:

- Where this file and the frozen code differ, as they now do on the shaping
  function (§7.4), there is no bug to fix. The frozen code belongs to the
  record; the file is what the rebuild is written from.
- K1 (§9.7, every run reports the fraction right) and K2 (§9.12, the
  estimator's correlation in every engine) were posed as work the object and
  array engines do not do. There are no such engines to burden. Both are
  ordinary requirements on the engines to be built, and the §12.1 exception
  §9.12 was going to need is not needed.

*So that Part 2c is answered by this entry rather than separately.*

**4b. TOLERANCE. Answered** (September 17, 2026): *"4b. 1e-12."* The relative
form stays and the constant drops from $10^{-9}$ to $10^{-12}$ — still about
4,500 units in the last place of a double at every $t$, where $10^{-9}$ was
4.5 million and would have grown to a whole hop after about a month of clock
time. TOLERANCE also enters the constant register, which no row had claimed.
A requirement goes with it: the slack must exceed the rounding accumulated
along the longest chain of times an engine builds, to be checked when the
clock is built.

**§7.5 — the zero-mean argument. Corrected** (September 17, 2026): *"7.5
please fix."* The argument was stated as though it held identically for both
eligibilities. It does not. Under the hazard $(c_j - q_j)$ is exactly the
score function and both branches have mean $m e^{-m}$, so it is exactly
zero-mean at every decision whatever the state. Under hebb it is $y - \hat
p_j$, a centred Hebbian rule rather than a likelihood derivative, whose mean
is $p_j - \hat p_j$ — the lag of the neuron's own estimate — and vanishes only
where that estimate has caught up. The conclusion survives on both: weighting
the credit alone shifts the mean by $(f-1)E[c_j]$, weighting both leaves it
where it was. §8.3 now states the difference, since a run under hebb posts a
systematic component wherever a neuron's rate is moving, and DECISION_MEMORY
sets how long that lasts.

---

## Part 1 — the decisions

**D1. §0.6 — what the substance is. Deferred, and the premise corrected**
(September 17, 2026): *"Defer. I can use the containers we have. But
eventually people will need a way of interacting with them. The substance
will still be spread on the plane. It's much easier to work with that way
once it's stable."*

The question was posed wrongly. The record's definition is not in doubt: walnut
butter is a substance spread on the plane. What was archived at 15:08 MDT was
the plane's **containers as code**, not the idea, so §0.6 needs no new
definition — goo is what the substance is built on for now, and the plane
returns later as the way it is worked with and looked at. The Open marker on
§0.6 is replaced by that, and §0.12 gains a bullet for the plane as an
interface.

*Corrected the same minute, against Claude's reading that the plane was "the
idea":* **"It is not THE idea, but AN idea of how to make neural networks
useful to makers."** So the plane is not constitutive of the substance. It is
one candidate interface — a form in which a maker could work with the
material — and §0.6 and §0.12 now say that rather than treating it as the
definition.

**D2. §5.10 — TEACHER_THRESHOLD. Replaced** (September 17, 2026): *"The row
critic needs a parameter, ROW_CRITIC_PICKINESS_IN_SPIKES. It should be an
integer, probably 1 or 2."* — after *"please rename TEACHER_THRESHOLD
ROW_CRITIC_TEACHER_THRESHOLD"* a minute earlier.

The rate-or-line question is dissolved rather than answered: an integer count
cannot change meaning with the epoch's length. Three changes follow. §5.10 is
the count and nothing else — a rate is derivable where a rule wants one, but
what counts as **on** now belongs to the critic that asks. §9.5 carries the
pickiness. The register row is replaced.

Left **open**: the value. It has to be chosen against the rest rate of the
escape hazard, since a neuron that hears nothing still fires at it — measured
at 37.6% of epochs reading as on at pickiness 1 on the mnist goo at 35 ms,
8.2% at pickiness 2, and 74.0% and 38.9% at 100 ms. So the integer removes the
epoch-length reinterpretation of the constant but not of the background, and a
value fixed once does not hold across epoch lengths or network sizes.

**D3. §7.4 — TARGET_ISI, one constant or three. Answered: one** (September 17,
2026): *"TARGET_ISI is a universal time constant, really the universal time
constant of the system. For now."*

So the three jobs are the same interval by intent and a sweep is meant to move
all three. The warning is withdrawn from §7.4 and the register row states the
status. Its value stays open.

**D4 and W5. §9.3, §12.9, §12.11 — the resume. Answered: exact** (September
17, 2026): *"Exact against the run it continues: the resumed run reproduces,
bit for bit, what the uninterrupted run would have done ... It's worth
having."*

§12.11 stops being "a continuation, not the same run" and becomes "a resume is
exact": the same spikes at the same waves, the same weights, the same
reinforcement, epoch after epoch. §12.9 carries what that rests on — the state
of all three streams (exploration, the network's own, the input stream), each
as a generator state rather than a seed and a count of draws, plus the
reinforcement baseline. §9.3's Open closes with it: a resumed run is paid
against the baseline the run had reached.

This replaces the lab notebook's resume, under which a resumed sweep arm was
not comparable to an uninterrupted one — the exploration stream restarted at
the driver's seed + 1,000,000, the baseline from the first resumed epoch, the
input stream advanced rather than restored.

**Also settled in passing** (September 17, 2026): *"An epoch is going to be 35
ms at pickiness 2."* ROW_CRITIC_PICKINESS_IN_SPIKES is 2 and INTERVAL stays
35 ms, and §9.5 says the two are chosen together, since a longer epoch at the
same pickiness lets more background through.

**D5. §9.9, §9.10, §10.1 — the three non-defaults. Answered: keep the values**
(September 17, 2026): *"Keep the values. They are latent knobs."* HOMEOSTASIS
$10^{-6}$, UNSTICK $10^{-3}$, QUASH_RATE 0.02 and QUASH_K 0.2/ms stay at their
values and a run switches each off. A rule that is off carries the value it
would run at, so turning it on is asking for it rather than inventing it.

**D6. §3.9 — the discharge. Answered: it stays** (September 17, 2026):
*"--discharge stays as a non-default."* So §8.11's arm that settles a
discharge with no credit stands, and §1.6's third clearing event with it.

**D7. §8.3 — the eligibility where the threshold decides. Answered: refuse**
(September 17, 2026): *"Refuse to learn."* No eligibility runs there and the
reinforce rule refuses the configuration. REINFORCE estimates a gradient from
the randomness of the decision; with no width there is no randomness and
nothing to estimate, so a rule with no derivation behind it is refused rather
than substituted (§0.9).

**D8. §8.8 — hebb's credit at the spike. Open, defaulting to a full unit**
(September 17, 2026): *"Open. Defaults to 1? 1 is the best number to default
to. 0 is the second-best."* The question stays open; the default is 1, which
is also what keeps hebb as near zero-mean as it comes — any discount widens
the gap between $E[c_j]$ and $E[q_j]$ rather than closing it.

**§3.2 — the hop is fixed directly** (September 17, 2026): *"I no longer want
to specify REFRACTORY_HOPS. I want to specify $h$ directly as
TIME_CONSTANT_OF_TRANSMISSION."* And on why the form went rather than the
name: *"refractory_hops was a nice convenience when we were working on an
integer hex grid."*

$h$ = TIME_CONSTANT_OF_TRANSMISSION = 2.5 ms is now a constant of its own and
REFRACTORY_HOPS is retired. One consequence is named in the clause: REFRACTORY
and $h$ are independent now, so "the refractory period is exactly two hops" is
a fact about the two values rather than a rule relating them. The shaping
function's punishment is anchored at REFRACTORY and so always lands at the
wall; *which* return lands there is what the pair decides. At $h$ = 2 ms a
two-hop return would arrive inside the refractory period and be dropped, and
the tightest loop the rule could see would be three hops, already past the
zero crossing and rewarded.

**§3.2 and §7.4 — the hop and the target, reparameterised** (September 17,
2026): *"For numerical consistency, a signal generated at time t is delivered
at time t + REFRACTORY + LAG, never at time t + REFRACTORY ... TARGET_ISI =
HOPS*(REFRACTORY + LAG). Let LAG = 0.1 ms. Then for REFRACTORY=5 ms,
TARGET_ISI=10.2."* And: *"I need to specify HOPS, REFRACTORY, and LAG.
TARGET_ISI follows."*

Specified: HOPS 2, REFRACTORY 5 ms, LAG 0.1 ms. Derived: the hop
$h = \text{REFRACTORY} + \text{LAG} = 5.1$ ms, and
$\text{TARGET\_ISI} = \text{HOPS} \times h = 10.2$ ms.

The LAG is structural rather than cosmetic: a signal sent to a neuron that
fired in the same wave arrives LAG after that neuron recovers, so no delivery
is ever settled by which side of the clock's slack it fell on. That removes
the knife-edge at its source, where an epsilon on the hop would have patched
it.

*And it settles where 5.1 ms came from.* TARGET_ISI was 5.1 ms from September
17, 04:31 MDT, hardcoded and "not known". It is $h$ — one hop, REFRACTORY +
LAG — so the original constant named a one-hop loop, which needs a neuron to
project onto itself and is forbidden (§1.1). That is why the curve collapsed
when it was anchored at the wall: at HOPS = 1 the whole of it lives inside the
LAG. HOPS = 2 is the shortest loop the network can have, and it puts the
two-hop return exactly on the peak, with a refire at the wall at $-1$. The
rule reads: a refire is paid for when the neuron's own signal coming back
explains it, and punished when nothing could have returned yet.

The preamble's caution is corrected with it — the record's hop was 1.667 ms
against this file's 5.1 ms, three times as long and longer than the refractory
period rather than a fraction of it.

**W1. §4.5, §4.6, §4.7 — output projections. Answered for the feedforward
wirings** (September 17, 2026): *"For the feedforward mnist problem, output
neurons do not project. They have the required fan-in at their inputs, but we
read the spikes directly and do nothing further with them. This is a desirable
optimization for the feedforward problem only."*

Stated in §4.6 and §4.7 as the point of those wirings rather than a
consequence, with the engine requirement it implies: an output neuron holds no
outgoing connections at all and its spike schedules nothing.

*Still Claude's reading, and narrower than it was:* §4.5's third line, under
which an output projects onto no **input** neuron in the scaled wiring. Byron's
answer is expressly for the feedforward problem only, so it does not carry to
the scaled rule, where an output does project onto hidden neurons. What an
output may do to an input there is unsettled.

**W2. §5.12, §11.6 — the signed class evidence. Confirmed** (September 17,
2026): *"W2 is fine as is."*

**W3. §7.5, §8.9 — what the shaping function weighs. Kept as built, not
settled** (September 17, 2026): *"I still don't understand, so we're sticking
with what's built."* The function weighs what every decision adds to a
synapse's score, rather than the single number the read pays. Recorded as
Claude's reading kept rather than as Byron's decision, since he said plainly
the argument had not landed; both clauses still stand to be corrected in a
word. Worth noting for whoever returns to it: the two readings differ
measurably, so a run under each would settle it without anyone having to
follow an argument.

**W4. §8.6 — the warm start. Kept** (September 17, 2026): *"We don't worry
about what happens in the neuron's first ten thousand decisions as it's just
waking up and hasn't had its coffee yet. Seriously, I'm more interested in
what the neuron is doing after an hour or two."*

The estimate is the plain mean of a neuron's first $1/\text{DECISION\_MEMORY}$
decisions and an exponential average after, as built. The scale settles it: at
a 35 ms epoch a neuron makes about 1,155 decisions an epoch, so ten thousand
of them is 8.7 epochs — 303 ms of clock — against roughly 103,000 epochs in an
hour. The warm start is 0.008% of the first hour, and "after an hour or two"
is 103,000 to 206,000 epochs, which is where the record's runs already sit.

DECISION_MEMORY's own value stays open, a starting value to be swept.

**W6. §0.11 — where the synapse-as-learner direction belongs. Answered: §0**
(September 17, 2026): *"This is the direction we are going with our next
research spurt. Please open an issue to implement the synaptic escape noise
learning rule."*

It stays among the values rather than moving to §0.12's list of intended
changes, because §0.12 is for changes that are named and parked and this is
the one being taken up. The clause says so and cites the issue.

Opened as byronshock/walnutbutter#15, carrying both quotations, what the
inversion re-asks (§0.10, §7.4, §8 entire, the learning rate against the noise
dimension, what a synapse holds), what it does not re-ask (what a neuron is),
the sparse-matrix form, and the two costs — one draw per synapse per wave
under §12.6's one-stream invariant, and a clock of irregular times where a
parallel form wants regular ones.

---

*That closes the second round: 21 confirmations, 8 decisions, 6 readings, and
4b. What remains is the attribution pass — the clauses Byron has now confirmed
still say "Claude's reading, to be corrected in a word".*
