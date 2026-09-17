# The authority

This file specifies walnutbutter. The code follows it: where the two
disagree, this file is right and the code has a bug. A change to the system
is made here first, then carried into the code and its tests.

Byron and Cedric own the substance of it. Claude keeps it and the code in
step. This file is copyright (C) 2026 Byron Shock and Cedric Shock and
licensed CC BY-SA 4.0 (`LICENSE-CC-BY-SA-4.0.txt`); the code it specifies
is under the AGPL, version 3 or later (`LICENSE`).

**Draft, September 17, 2026.** Every section below was drafted from `RECORD.md` and from Byron's answers in `docs/rewrite-answers.md`, and checked against the record; the checkers' findings are in `docs/authority-draft-checks.md` and are not yet applied. Nothing here is in force until Byron says so.

## 0. Standing values — non-negotiable

*These are values, not mechanisms: no clause of this file may contradict one, and where a clause and a value disagree the value is right and the clause is to be rewritten.*

**0.1 What a value binds, and what a direction binds.** Every clause of this file states a rule the system implements today. A value marked **a direction** states where Byron and Cedric intend the system to go: it is not implemented, no clause is written from it, and it overrides no clause in force. Where the system as it runs and a direction disagree, this file states the system as it runs.
*Byron, September 17, 2026, 14:53 MDT, keeping θ rather than deprecating it in the rewrite: "We are going to keep the level. The system must implement what I was just using. We can make changes later to this." [`docs/rewrite-answers.md` §1]*

**0.2 The inputs are the outputs.** An input neuron, a hidden neuron and an output neuron are defined only with respect to where external connections are; the neurons themselves operate identically. No rule may branch on a neuron's index, its zone or its label. *Engines:* a neuron's place in a list, an array or a matrix row is an ordering and never a role.
*Byron and Cedric, from the beginning [RECORD §0]. Restated by Byron, September 14, 2026: "All neurons are first-class citizens of the population." [RECORD §2, §8]*

**0.3 A neuron integrates delta functions to determine its potential.** A signal is an instant: it adds its weight to the potential of the neuron it reaches and has no width and no shape. The mechanism is §1.2.
*Byron and Cedric, from the beginning [RECORD §0].*

**0.4 A neuron that fires is then absolutely refractory, and that is a computational feature.** While refractory it ignores its inputs and does not integrate them. This is a feedback control mechanism and a computational feature of the system, not a limitation to be worked around, and no rule may work around it. The mechanism is §1.5.
*Byron and Cedric, from the beginning [RECORD §0, §5.3].*

**0.5 Structures are permissive; nothing is restricted artificially.** A neuron may sit in several input and output zones at once. Where the system refuses a configuration, the refusal is a clause of this file with a reason, never a convenience of the code; and no quantity is clipped to keep it tidy.
*Byron and Cedric [RECORD §2]. Byron, September 14, 2026, striking the threshold clamp: "Please eliminate threshold clipping. It's artificial." [RECORD §1.3]*

**0.6 The goal is the substance, not a task.** A problem exists only as a way of watching whether the substance is alive and learning; no rule is chosen because it scores well on one.
*Byron and Cedric, standing since the project began [RECORD §2]. **Open:** the record defines walnut butter as a substance spread on the plane, and the plane and its containers are archived — Byron, September 17, 2026, 15:08 MDT: "archive the hex containers, lattice, spread, the whole shebang." What the substance is with no plane is Byron's to say.*

**0.7 As few knobs as possible, and a default regime that is stable.** Where a quantity must change with the size or the shape of a network, the change is built into the rule rather than left to a sweep.
*Byron, September 16, 2026: "A design principle of this project is that it should have as few knobs as possible and by default operate in or near a stable regime. Therefore scaling the network MUST reduce the probability of escape noise at each neuron by sqrt(N). I want to build this into the rule." [RECORD §5.2] The instance he gave it for is the firing hazard's scaling by the network's count.*

**0.8 The network keeps living.** There is no training run and no evaluation run, only one run that keeps going. No rule may assume an end, and no result is read as convergence: a run is read as health, and drift is normal.
*Byron and Cedric, standing [RECORD §6.14, §7: "The network runs forever: these are read as health, not convergence, and drift is normal."]*

**0.9 One authority, every platform.** Every engine is built from this file alone. A rule stated here is implemented in every engine that runs it, and an engine that lacks a rule says so and refuses — it never approximates. Engines are compared with `==`, with one tolerance, named here and hidden nowhere: the object engine and the Rust loop agree to the bit, and the array engine, which sums a wave in matrix order and takes its exponentials from numpy, agrees exactly on which neurons spike in which wave and to about a part in a billion on continuous quantities.
*Byron, September 14, 2026: "rules in authority.md must be implemented cross-platform" [RECORD §7]. Byron, September 17, 2026, 15:02 MDT, keeping the array engine: "keep, with tolerances, as a foundational rule; all platforms are built from one clear authority."*

**0.10 Learning is paid by one global scalar.** One number scores what the network did, a baseline is subtracted from it, and every synapse's update is proportional to what is left. No rule in force hands a neuron an error of its own.
*The frame is Byron and Cedric's, from the beginning [RECORD §0]. **A direction:** that the scalar be produced locally at a spike and consumed globally — dopamine — leaves this file as a mechanism and stays as an intention, Cedric's to take up. Byron, September 17, 2026, 15:02 MDT: "(1) drop but keep the idea of dopamine reinforcement around. It's a very important direction for Cedric's work, but the implementation was mine and I lacked understanding." [RECORD §6.2–§6.6 is the attempt, and stays there.]*

**0.11 Exploration belongs to the synapse — a direction.** Today exploration is the neuron's: one firing decision per neuron per wave, and a synapse is credited by what it happened to have in the potential when its neuron took the chance (the firing and learning clauses). That the exploration and the credit it earns belong to the same object — the synapse — is where the project is going, and nothing in this file is written from it until Byron specifies it.
*Byron, September 17, 2026, 06:05 MDT: "Computationally exploration noise can be generated at the synapses, and semantically this is clean: a SYNAPSE explores its own impulse response, rather than a NEURON exploring its impulse response!" [RECORD §0.3, "Not specified or built yet"]. The record places this at the top at Byron's instruction and expressly not among §0's rules; it stands here as a direction and not as a rule — Claude's reading, to be corrected in a word.*

**0.12 The changes intended, and not made.** These are named here and nowhere else in this file; no clause is written from one, and when one is taken up it is specified whole, from §0 down.
- Exploration generated at the synapse (§0.11), and with it the deprecation of θ and a hazard set by a neuron's fan-in. *Byron, September 17, 2026: "Theta should also be deprecated" — and, the same afternoon, 14:53 MDT, that the rewrite states the system as he has been running it, θ and its fan-in scaling included. [RECORD §0.3, §0.4; `docs/rewrite-answers.md` §1]*
- A dopamine reward produced locally at a spike (§0.10).
- A deterministic drive: one spike every TARGET_ISI on each driven neuron in place of the Poisson rate drive. *Byron, September 17, 2026, 14:58 MDT: "Please include the Poisson drive as it ran." [`docs/rewrite-answers.md` §3]*
- Small-world shortcuts, to return. *Byron, September 17, 2026, 15:08 MDT: "We'll bring back small world shortcuts later."*

---

## 1. The neuron

**1.1 What a neuron holds.** One of each, per neuron:

- its potential $p_j$, and the clock time that potential was last brought up to date;
- its threshold $\theta_j$ and its floor $p^{\min}_j$ — the starting values its container gave it, and wherever homeostasis or un-sticking have since moved them when a run turns those on (the firing and learning clauses);
- the width of its firing decision $\Delta_j$, the hazard's scaling by the network's count, and the moment its hazard has run from (the firing clauses);
- the time of its last spike $t^{\text{fired}}_j$ and of the spike before it; the number of spikes it has ever fired, and the number it had fired when the epoch began, whose difference is the count read (the drive-and-read clauses);
- whether it has fired in the epoch so far, and whether the drive forced it in this epoch;
- its rate memory $r_j$ (§1.7);
- its per-decision expectation $\hat p_j$, the number of decisions it has made to date, and $E_j$ (§1.8).

Its refractory state is not held: it is read from the time of its last spike (§1.5). *Engines:* every engine carries all of these per neuron, in the same neuron order, and a checkpoint round-trips every one of them.
*Claude's inventory of the state the clauses of this file require; the pieces are Byron's and Cedric's where the clauses that use them are [RECORD §0 notation, §5.1, §5.2, §6.7, §7].*

**1.2 The potential is an accumulator of evidence.** The potential is the sum of the weights of the signals the neuron has integrated since its last spike, undiminished: nothing decays. With $x_{ij}$ the number of arrivals $j$ integrated along the synapse $i \to j$ since its last spike,

$$p_j = \sum_i w_{ij}\,x_{ij}, \qquad \frac{\partial p_j}{\partial w_{ij}} = x_{ij},$$

exactly while the floor has not bitten; once inhibition has taken the potential to $p^{\min}_j$ the potential is the floor whatever the weights, and the identity holds again from the next arrival (the firing clauses set the floor). A signal that reaches a refractory neuron is not integrated and is not counted (§1.5).
*Byron, September 17, 2026, 14:56 MDT, choosing it for the rewrite: "neuron: NOT leaky, but please leave the leak option with its analysis. 2 hops." [`docs/rewrite-answers.md` §2] And his statement of it the same morning: "since we are keeping weighted synapses the potential is a weighted count of evidence, and its derivative delta functions of spike arrivals weighted by the evidence each carries." [RECORD §5.1]*

**1.3 The leak, as a per-run option.** A run may select a leaky potential instead of the accumulator: with the leak on, a signal of weight $w$ arriving at time $t$ first decays the potential for the time since it was last brought up to date, and is then added,

$$p \leftarrow p\,e^{-(t - t_{\text{last}})/\tau}, \qquad t_{\text{last}} \leftarrow t, \qquad p \leftarrow p + w,$$

with $\tau$ = TAU = 2 ms. The accumulator of §1.2 is the default and is $\tau = \infty$; a checkpoint carries which of the two its network ran under, and a resumed network keeps it. *Engines:* under the accumulator no engine evaluates a decay anywhere — Byron, September 17, 2026: "we DO need to skip the exponential decay calculation when we select evidence-accumulator since it's just going to slow things down" — and the skip changes no bit, those factors being exactly 1.
*Byron, September 17, 2026, 14:56 MDT: "please leave the leak option with its analysis." The analysis stays with it in the record and no part of it is a clause [RECORD §1.2, §5.1, §8].*

**1.4 What a spike does to the potential.** A spike resets the potential to zero and is remembered, along with the spike before it:

$$p \leftarrow 0, \qquad t_{\text{prev}} \leftarrow t^{\text{fired}}, \qquad t^{\text{fired}} \leftarrow t,$$

and the neuron's spike count rises by one. Nothing any synapse delivered is still in the potential, so every incoming synapse's trace $x_{ij}$ goes to zero with it, and the spike is the moment the learning rule settles what those arrivals earned (the learning clauses). The neuron is then refractory (§1.5).
*Byron and Cedric, from the beginning [RECORD §0, §5.2].*

**1.5 The refractory period.** A neuron that fired at $t^{\text{fired}}$ is refractory while $t < t^{\text{fired}} + \text{REFRACTORY}$, REFRACTORY = 5 ms. While refractory it ignores every signal, does not integrate it, cannot be forced by the drive, and makes no firing decision. *Engines:* the comparison allows the clock's slack (the clock clauses), so a spike returning the instant the period ends finds the neuron recovered rather than a hair short of it, and the three engines recover the same neuron at the same wave.
*Byron and Cedric, from the beginning [RECORD §0, §5.3].*

**1.6 The refractory period is two hops.** REFRACTORY_HOPS = 2, so a signal takes $h = \text{REFRACTORY}/\text{REFRACTORY\_HOPS} = 2.5$ ms to travel one connection and a spike sent around a two-way pair returns exactly as the refractory period ends.
*Byron, September 17, 2026: "Please set hops=2 by default" [RECORD §1.2], restated for the rewrite the same day, 14:56 MDT: "2 hops." The hop's use by the schedule is the clock's [RECORD §4.1].*

**1.7 The rate memory.** After every epoch, each neuron the drive did not force in that epoch moves its rate memory toward what it did:

$$r_j \leftarrow r_j + \text{RATE\_MEMORY}\,(y - r_j), \qquad y = 1 \text{ if it fired in the epoch, } 0 \text{ if it did not},$$

RATE_MEMORY = 0.01, about the last hundred epochs; $r_j$ starts at 0.5. A neuron forced in the epoch is left alone, its firing that epoch saying nothing about the network. $r_j$ is what homeostasis and un-sticking read when a run turns them on, and what a report means by a stuck neuron (the learning and reporting clauses).
*From the pre-alpha, carried [RECORD §1.3, §6.7]; the exemption of a forced neuron is the one homeostasis and un-sticking already follow.*

**1.8 The per-decision expectation.** $\hat p_j$ is the neuron's own estimate of the chance that it spikes at a decision. It is undefined until the neuron's first decision, which sets it to that decision's outcome and charges nothing; after every decision, charged first, it moves by

$$\hat p_j \leftarrow \hat p_j + \max\!\big(\text{DECISION\_MEMORY},\, 1/n\big)\,(y - \hat p_j), \qquad y \in \{0, 1\},$$

$n$ the decisions the neuron has made to date, so the estimate is the plain mean of its first $1/\text{DECISION\_MEMORY}$ decisions and an exponential average after; DECISION_MEMORY = $10^{-4}$. The estimate's own move takes the outcome unweighed: what the learning rule weighs is what it charges, not what the neuron remembers. Under the accumulator the neuron also carries $E_j$, the expectation accumulated over its decisions since its last spike as the rule charged it, restarted at zero by the spike, so that a debit can be settled per arrival (the learning clauses). *Engines:* $\hat p_j$, $n$ and $E_j$ are per-neuron state in every engine and are checkpointed; a neuron whose decisions the rule in force does not charge keeps them untouched.
*Byron, September 17, 2026: "Expectation is changed per decision in this architecture" [RECORD §6.7]. The warm start and the window are Claude's reading of "changed per decision", to be corrected in a word. **Open:** DECISION_MEMORY is "a starting value, to be swept", and its window is not settled [RECORD §1.3].*

## 2. The synapse

*A synapse is a one-way connection between two neurons. It holds a weight,
which is what it delivers, and four pieces of state, which are what the
learning rule reads. Where a clause here says a target "integrated" a
signal, §3.5 says what that means.*

**2.1 One-way, with its own weight.** Every connection runs one way, from a
source neuron $i$ to a target neuron $j$, and carries its own weight
$w_{ij}$: the amount delivered to $j$ when a signal travels it. Between two
neurons there are two connections, $i \to j$ and $j \to i$, with independent
weights; a pair is never one object. No neuron connects to itself.
*Byron and Cedric, from the beginning [RECORD §3]. Byron's correction of the
verb, September 14, 2026: "Instead of 'connects' I should have said
'projects'. The connections are all one-way." Restated by him in the scaled
wiring rule, September 16, 2026, 01:18 MDT: "P(i projects onto j) = 0 if i
== j".*

**2.2 Ids from 1, in the order the connections are made.** Connections take
ids from 1 in the order they are made, so one seed rebuilds the same
network, connection for connection. The order the pairs are offered in
belongs to the wiring, not to the synapse.
*Byron and Cedric [RECORD §3, §3.4].*

**2.3 The weight at build.** Every weight is drawn independently and
uniformly from WEIGHT_RANGE = $[-1, 1]$ from the network's seeded stream, at
the moment its connection is made — so the weights are drawn in
connection-id order, interleaved with whatever draws the wiring itself makes.
A network may instead be built with one fixed weight given to every
connection; that is a test's network, not a run's.
*Byron and Cedric [RECORD §1.1, §3.3].*

**2.4 Learning keeps a weight inside WEIGHT_RANGE.** Every rule that moves a
weight clips the result back into WEIGHT_RANGE:
$w_{ij} \leftarrow \operatorname{clip}(w_{ij} + \Delta, -1, +1)$. The range
is both the prior the weights are drawn from and the only bound on them; a
weight that reaches an end stays there until a rule moves it back.
*Byron and Cedric [RECORD §3.3, §6.7]. Engines: the clip is part of each
update, not a tidying pass over the weights afterwards.*

**2.5 What a synapse carries.** Beside its weight a synapse carries exactly
four pieces of state: its trace $x_{ij}$ (§2.6), its stamp (§2.7), its note
$B_{ij}$ (§2.8) and its score $e_{ij}$ (§2.9). Nothing else per synapse is a
rule of this file. A checkpoint carries the stamp, the trace (with the
moment it was last brought up to date) and the note; it does not carry the
score, so a checkpoint is taken at a read, where the score has just been
paid.
*The four are the single-spike rule's and the schedule's (Byron, September
17, 2026) [RECORD §4.4, §6.7, §7]. The checkpoint sentence is Claude's
reading of what the checkpoint holds, to be corrected in a word.*

**2.6 The trace $x_{ij}$: what the synapse has in its target's potential.**
Under the evidence accumulator the trace is the count of the arrivals the
target integrated along this synapse since the target's last spike — an
integer, raised by one at each arrival integrated. It is the derivative of
the target's potential with respect to this weight, exactly while the floor
has not bitten: $p_j = \sum_i w_{ij} x_{ij}$, so $\partial p_j / \partial
w_{ij} = x_{ij}$. It is cleared when the target spikes and when the floor
bites, the two events after which what the weights delivered is no longer
the potential. Under the leak, which stays as a per-run option (TAU finite),
the same quantity is the leaked charge $\sum_a e^{-(t - t_a)/\text{TAU}}$
brought up to date lazily, which is why the synapse also keeps the moment
its trace was last brought up to date.
*Byron, September 17, 2026, stating the accumulator: "since we are keeping
weighted synapses the potential is a weighted count of evidence, and its
derivative delta functions of spike arrivals weighted by the evidence each
carries" [RECORD §5.1, §6.7].*

**2.7 The stamp: the last signal its target integrated.** Each connection
records the clock time of the last signal its target actually integrated
along it, and nothing if it has carried none. A signal delivered while the
target is refractory is integrated by nothing and does not move the stamp,
though it is still recorded as delivered. This is the only trace of activity
a target has of its own inputs, and it is what the quash reads (a
non-default) to find the synapses that carried a cycle.
*Byron and Cedric [RECORD §4.4, §6.4, §6.11].*

**2.8 The note $B_{ij}$: the debit its open arrivals carry.** An arrival
stays open until its target spikes, so as each arrival is integrated the
synapse adds $E_j$, the target's expected spikes since its last spike, to
its note: $B_{ij} \mathrel{+}= E_j$. The note is what lets the spike settle
every open arrival in one operation,
$e_{ij} \mathrel{+}= c_j x_{ij} - (x_{ij} E_j - B_{ij})$, and it is cleared
with the trace at that spike.
*Byron, September 17, 2026, on whether a debit may run past a read: "Let
them run! Epochs are for the convenience of teaching the network, and to
some extent an emergent property of the time constants, but they don't
really exist in nature." [RECORD §6.7.] Engines: an arrival is one operation
on the synapse and a decision one operation on the neuron — no loop over a
fan-in may run at a decision.*

**2.9 The score $e_{ij}$: what the synapse has earned since the last read.**
Each synapse carries the score charged to it by the learning rule since the
last read. At the read the debit accrued so far on the open arrivals is
settled into the score, the update is paid on it, and the score is cleared
for the next read; the arrivals stay open, and their debit counts from that
moment, which the engines do by re-basing the note to $B_{ij} = x_{ij} E_j$
as the score is cleared.
*Byron, September 17, 2026 [RECORD §6.7]. The re-basing is Claude's
statement of the record's "the open arrivals counting from now", to be
corrected in a word.*

## 3. The clock

*One run has one clock, and every time in this file is a time on it.*

**3.1 Time is in nominal milliseconds.** The clock runs on from the first
input and is never restarted; an epoch is a boundary on it, not a reset of
it.
*Byron and Cedric [RECORD §4.1].*

**3.2 The hop.** A signal takes one **hop**,
$h = \text{REFRACTORY} / \text{REFRACTORY\_HOPS}$, to travel any connection,
and there is no other delay: the time in signalling is carried by the hop
and the refractory period alone. With REFRACTORY 5 ms and REFRACTORY_HOPS 2,
$h$ = 2.5 ms and the refractory period is exactly two hops, so a spike sent
around a two-way pair of connections returns the moment the period ends. The
ratio need not be an integer.
*The hop is Byron's, September 11, 2026; the value his, September 17, 2026:
"Please set hops=2 by default" [RECORD §1.2, §4.1].*

**3.3 A wave is everything at one time.** The queue is a time-ordered
schedule of signals and stimuli, not a per-hop loop, and a wave is the batch
at the front of it sharing one time — the signals arriving, and an input
stamped for that moment. A cascade from one input may still be running when
the next input's signals join the schedule: the epoch is not a barrier, and
"the queue empties, then the clock advances" does not hold.
*Decided September 11, 2026 [RECORD §4.1].*

**3.4 Two moments within the clock's tolerance are one moment.** With
$\text{slack}(t) = \text{TOLERANCE} \times \max(1, |t|)$ and TOLERANCE =
$10^{-9}$, two events whose times differ by no more than the slack of the
earlier are in the same wave, and every comparison of two moments — the end
of a refractory period included — allows the same slack. Where a wave holds
an input or any other external event, that event's exact time is the wave's
time; otherwise the earliest event's time is. A hop is rarely representable
exactly, so without that anchor a chain of hops drifts off the input clock.
*[RECORD §4.1, `clock.py`.] The record states this as "times are rounded to
a nanosecond"; the rule as built is the relative slack above, a nanosecond
up to a millisecond of clock time and growing with it. Claude's reading, to
be corrected in a word. Engines: same tolerance and same anchor, or two
engines put the same two signals in different waves.*

**3.5 A wave has two phases: deliver, then fire.** First **deliver**: every
signal of the wave delivers its weight to its target; a target that is not
refractory integrates it, which stamps the synapse (§2.7) and moves its
trace (§2.6), and a refractory target integrates nothing. Then every neuron
the wave touched settles — its floor applies to the wave's total, so no
result depends on the order the signals arrived in. Then **fire**: every
forced neuron fires unless it is refractory, then every neuron that is not
refractory decides, whether the wave touched it or not, and each neuron that
fires schedules its outgoing signals for $t + h$. No decision in a wave sees
a spike of that wave, since a wave's spikes arrive one hop later.
*Byron and Cedric [RECORD §4.4]. What a decision is belongs to the firing
clauses (the escape noise, RECORD §5.2); that every neuron is asked, and not
only the touched, is what lets a neuron nobody talks to spike.*

**3.6 The order the fire phase asks in.** The forced neurons first, in the
order the wave holds them; then the neurons the wave touched, in the order
they were touched; then every other neuron, in index order. The order is a
rule, not an implementation's choice: the order neurons fire in is the order
their signals are pushed onto the schedule, and push order is the order a
later wave sums them in (§3.7).
*Claude's reading of what the three engines do and of what §3.7 requires of
them, to be corrected in a word [RECORD §4.4, §6.15].*

**3.7 A wave's deliveries are summed in push order.** Signals due at one
moment are summed in the order they were pushed onto the schedule, so an
engine that flattens the topology differently sums a wave differently and
lands on different bits. The object engine and the Rust loop sum in push
order and agree to the bit; the array engine sums a wave in matrix order, and
its exception — every spike equal, continuous quantities to a part in $10^9$
— is named in the platform clause and nowhere else.
*Byron, September 14, 2026: "rules in authority.md must be implemented
cross-platform" [RECORD §6.15].*

**3.8 The wave's draws are taken between the phases.** One uniform per
neuron, in neuron order, from the network's exploration stream, after the
floor has settled and before any neuron fires. Every neuron takes a draw
whether it can use it or not, so the stream's position depends only on the
count of neurons and the count of waves.
*Byron, September 15 and 16, 2026, with the escape noise [RECORD §5.2,
§6.15: "one uniform per neuron, in neuron order"]. What the draw decides is
the firing clauses'. Engines: a draw that is not used is still taken.*

**3.9 An epoch.** An epoch is one input and the schedule run to the
**horizon**, $t_e + \text{INTERVAL}$. In order: (1) **reset** — every
neuron's fired-this-epoch state is cleared, and each synapse's score with it
(§2.9), while potentials, spike times, traces, notes, stamps and signals in
flight are all kept; (2) **input** — the epoch's pattern is placed on the
input neurons at the times the drive gives it; (3) **run** — the schedule
runs wave by wave to the horizon; (4) **report** — the output is read and,
under the reinforce rule, scored and paid.
*Byron and Cedric [RECORD §4.2]. The record's third step, a per-epoch
exploration nudge of every potential, goes with the perturb eligibility;
what a wave draws is §3.8.*

**3.10 Signals due at or after the horizon wait.** They stay on the schedule
and join the next epoch's waves. An input may not be given a time before the
horizon the schedule has already run to.
*Byron and Cedric [RECORD §4.2].*

**3.11 INTERVAL.** INTERVAL is 35 ms: the epoch's length, and the spacing of
inputs when no time is given — the first input at 0 and each next one
INTERVAL after the last. No problem sets it, so the epoch's length lives in
one place; a run may set it for the whole of that run.
*Byron, September 14, 2026, deciding it on the epoch sweep — the record's
words for his decision, not a quotation of it: "INTERVAL is 35 ms, and no
problem overrides it" [RECORD §1.2, §4.2]. **Open:** the mnist runs this
specification states ran at 100 ms, set per run, so whether 35 ms is still
the default is Byron's to say.*

## 4. The container and its wiring

One container survives, goo, with three wirings of it; a wiring decides only what projects onto what, and the container hands each neuron the potential axis its fan-in earns.

**4.1 Goo is the only container.** The system builds one kind of network: goo. The hex grid, the hexagonal columns, the lattice and a spread leave the specification, and with them the positions they needed, the hex metric, the guaranteed neighbourhood, the small-world shortcuts and the constants OMEGA, REACH and ROWS. Nothing is deleted: they stay in `RECORD.md` §2–§3 and in the code at the tag `lab-notebook-2026-09-17`, which is what archived means here, and the rebuild does not carry them. Small-world shortcuts are named here as an intention to return — they were the only thing that shortened a trip across a grid, and goo has no far — not as a clause, and no rule below may assume them.
*Byron, September 17, 2026, 15:08 MDT: "archive the hex containers, lattice, spread, the whole shebang. We'll bring back small world shortcuts later." [RECORD §2, §3.4]*

**4.2 What goo is.** Goo is $N$ neurons with no positions at all. No distance between two of them is defined, so there is no neighbourhood, no near and no far, and nothing for a shortcut to get past. What connects to what is decided by a probabilistic rule over ordered pairs (4.5–4.7) and by nothing else. The count is GOO_COUNT = 60 unless a run gives another, and the zone widths (4.3) set the smallest count a goo can be built at.
*Byron, September 14, 2026, asking for a fourth container and saying what it was — fully connected goo — and, the same day, setting its count: "We will speed everything up by selecting 60 units of goo". [RECORD §2, §3.4]*

**4.3 Zones by index.** With no positions there is no bottom row to be the input, so the zones go by index: the **first** `across` neurons in index order are the input zone and the **last** `outputs` are the output zone, `outputs` being `across` unless the problem gives its own two widths. A neuron in neither zone is a hidden neuron. Both widths must be at least one, and the count must leave room for both (4.5–4.7). The zones are addressed the way every other container's rows were, so the drive and the read need not know what container they are on: `get_neuron_at(place, 1)` is the input zone and `get_neuron_at(place, 0)` the output zone, and goo reports `rows` = 2 because it is counting those two zones and not any depth. Being in a zone says where a neuron is driven and where it is read, and nothing else follows from it: no rule branches on a neuron's index or its zone. Structures are permissive — a neuron may sit in several input and output zones at once — but each of the three wirings below needs its zones disjoint and refuses what it cannot wire; whether the permissive reading returns is open and Byron's.
*Zones by index are Claude's reading of what "fully connected goo" has to mean to be buildable, September 14, 2026, written down here so that changing it is editing a working system; the two widths are from mnist, September 15, 2026. The permissive rule is the standing value of §0 ("A neuron may sit in several input and output zones at once; structures are permissive, never artificially restricted"). [RECORD §2, §3.4]*

**4.4 A projection is an ordered pair.** Every projection is one-way, from a source to a target, with its own weight; between two neurons there may be two, $i \to j$ and $j \to i$, each with an independent weight, and no neuron projects onto itself. A wiring is a rule that gives $P(i \to j)$ for every ordered pair of neurons. The pairs are taken in $(i, j)$ order — source index, then target index — and the projections made are numbered from 1 in that order, which is the order every engine reads them in.
*Byron, September 14, 2026: "I want goo to be constructed according to a probabilistic rule: P(Neuron i connects to Neuron j) = 0 if i == j; 0 if i in inputs or outputs AND j in inputs or outputs; P_connection otherwise" — and, correcting the verb: "Instead of 'connects' I should have said 'projects'. The connections are all one-way." [RECORD §3, §3.4]*

**4.5 The scaled rule — the default wiring.** With $s$ = GOO_SCALING_FACTOR, $I$ and $O$ the zones' widths:

$$P(i \to j) = \begin{cases}
0 & i = j \\
0 & i,\ j \text{ both in the input zone} \\
0 & i \text{ in the output zone},\ j \text{ in either zone} \\
\min\!\big(1,\ Ns / A_j\big) & \text{otherwise}
\end{cases}$$

where $A_j$ is the sources $j$ may hear under the three zeros above: the $N - I - O$ hidden neurons for an input neuron, the $N - O$ inputs and hidden neurons for an output neuron, the $N - 1$ others for a hidden neuron. So every neuron with sources enough hears $Ns$ synapses in expectation and the density is the same at any count; where $Ns$ exceeds $A_j$ the probability stops at 1 and the neuron hears $A_j$ of them. An output projects onto hidden neurons alone; an input hears hidden neurons alone; with no hidden neurons the goo is inputs onto outputs and nothing else, and every input hears nothing (4.11). The zones may not overlap — the count must be at least the two widths together — and a goo that would overlap them is refused. GOO_SCALING_FACTOR = 0.05.
*Byron, September 16, 2026, 01:18 MDT: "P(i projects onto j) = 0 if i == j; 0 if i and j are both in the input zone; P_ij necessary to give j an average of N * scaling_factor inputs. Please default scaling_factor to 0.05." The third line is Byron the same night, 03:05 MDT: "I would like the outputs kept apart from one another again. With hidden=0 we have no cycles, eliminate interference from other output neurons, a two-layer feedforward network" — that an output also projects onto no input neuron is Claude's reading of "no cycles" and "feedforward", to be corrected in a word. [RECORD §3.4]*

**4.6 ff2 — fully connected, feedforward.**

$$P(i \to j) = \begin{cases} 1 & i \text{ in the input zone},\ j \text{ in the output zone} \\ 0 & \text{otherwise} \end{cases}$$

Every input projects onto every output and nothing else projects at all. Nothing about the topology is drawn: the count and the two widths fix it. It wires two layers and no more — a goo with hidden neurons is refused under it, so the count must be exactly the two widths together. How the two layers generalise to the zones of a recurrent goo is Byron's to say and is not built.
*Byron, September 16, 2026, 16:35 MDT: "Please make a fully-connected-feedforward-2 rule: There are two 'layers', the input 'layer' and the output 'layer'. P_ij = P (i is in the input layer and j is in the output layer). These 'layers' generalize to zones in a recurrent goo, but for the feedforward problem I want everything to connect fully." [RECORD §3.4]*

**4.7 ff2-partial — the same two layers at a probability.** With $P$ = GOO_PROJECTION, which must be in $(0, 1]$:

$$P(i \to j) = \begin{cases} P & i \text{ in the input zone},\ j \text{ in the output zone} \\ 0 & \text{otherwise} \end{cases}$$

each input-to-output pair its own draw, nothing else projecting, two layers and no hidden neurons as under ff2, of which it is the general form: at $P = 1$ it is ff2 to the bit — the same projections, the same weights, the same stream — and every engine must satisfy that exactly. An output hears $IP$ inputs in expectation. GOO_PROJECTION = 0.2.
*Byron, September 17, 2026, about 02:40 MDT: "I'd like to define a partly connected topology that is otherwise identical (2 layers), where P(neuron i projects onto neuron j) = P iff i in inputs, j in outputs". [RECORD §3.4]*

**4.8 The draw on the seed's stream.** Nothing is drawn where the probability is 0: no projection is made and no number is taken. Where it is 1, the projection is made without a draw. Where it is strictly between, one uniform from the network's seeded stream decides it, and the projection is made if the uniform is below the probability. A weight follows a projection that is made — one uniform on WEIGHT_RANGE, unless the run fixes every weight instead — so a pair is taken projection first and then weight, pair by pair in $(i, j)$ order (4.4), and nothing else of the topology is drawn. *What this requires of the engines:* the wiring is drawn once, in the container, and the array engine and the Rust loop take the projections and weights already made, in the container's order; no engine draws a topology of its own, so all three carry the same wiring and the same starting weights to the bit, with none of the tolerance the array engine is allowed on continuous quantities elsewhere.
*Byron, September 14, 2026, on the first probabilistic wiring; the order within a pair and the stream's ownership are Claude's reading of it, unchanged since. [RECORD §3.3, §3.4]*

**4.9 A wiring that was drawn needs its seed to be rebuilt.** Where any pair's probability was strictly between 0 and 1, the seed that drew the wiring is part of the network's identity: without it the goo cannot be rebuilt, and a checkpoint of one that has no seed is refused rather than rebuilt differently. Where no probability was strictly between 0 and 1 — ff2, or a scaled goo whose every probability is 0 or 1 — the count, the two widths and the wiring fix the topology, and it rebuilds with no seed at all. A checkpoint carries the count, the two widths, the wiring, its knob ($s$ or $P$) and the seed, and restores under the wiring it was built with, never under whatever the default has since become.
*Byron, September 14, 2026 ("a goo wired below 1 needs its seed to be rebuilt, as a grid needs its seed for its shortcuts"); the restore-under-its-own-wiring rule is Claude's, September 15–16, 2026, as each new wiring was added. [RECORD §3.4]*

**4.10 The potential axis scales with fan-in.** The container gives neuron $j$, of in-degree $d_j$, its starting threshold and its floor:

$$\theta_j = \text{GOO\_THRESHOLD} \cdot \frac{d_j}{F}, \qquad
p^{\min}_j = \text{GOO\_MINIMUM\_POTENTIAL} \cdot \frac{d_j}{F}, \qquad
F = \text{THRESHOLD\_FAN\_IN} = 18,$$

with GOO_THRESHOLD = 0.2 and GOO_MINIMUM_POTENTIAL = −0.8, so the floor sits four times the threshold below zero at every fan-in. The floor moves with the threshold because it is not a second decision: the two are points on one axis and it is the axis being rescaled. The scaling is applied after the wiring, since it reads the in-degree. These are **starting** values only: homeostasis and un-sticking (non-defaults) move a threshold from where it starts, the firing section quotes the escape width in units of the starting threshold, and a checkpoint stores the threshold and floor a run actually reached rather than recomputing them.
*Byron, September 14, 2026, choosing the threshold out of the three readings the record named, and the floor with it; goo's own two numbers date from the same day. That the rule is linear is Claude's reading — that is what "scales with fan-in" says without further instruction — decided for now, and a different slope, or a threshold quoted against something other than the fan-in, is Byron's to call. $F = 18$ is the in-degree the numbers were first quoted at, an interior hex cell's two rings, and stays the unit although that container has left the specification (4.1). [RECORD §5.2]*

**4.11 A neuron that hears nothing.** A neuron with no incoming synapses is left at the container's quoted threshold and floor, scale 1 — there is nothing to scale by — so it keeps a positive threshold and the width that is quoted against it: it fires at the rest rate of the escape hazard like any other neuron, and by its drive if it is an input neuron, and never otherwise. This is a rule about the scaling alone; a threshold given as 0 outright still means the deterministic comparison (the firing section). All three engines are held to it alike.
*Claude's resolution, September 16, 2026, of the question the outputs-apart rule (4.5) opened by leaving every input neuron of a two-layer goo hearing nothing; to be corrected in a word. [RECORD §5.2]*

## 5. The drive and the read

*What reaches the network from outside, and what is read back out of it.
Everything between the two belongs to the neuron (§1), the clock (§3) and
the rule (§8).*

**5.1 The input zone is the input; the output zone is the output.** The
drive of 5.4 reaches the neurons of the input zone and no others; the read
of 5.10 counts the neurons of the output zone and no others. A zone is
where the external connections are and nothing more: per §0.1 the neurons
themselves are identical, and no rule may give a neuron a role beyond
membership of a zone. A problem names both widths (§10); a container says
which neurons a zone holds (§4).
*Byron and Cedric, from the beginning [RECORD §4.3].*

**5.2 An input is complement-coded, and coded no other way.** A pattern is
$k$ raw bits followed by their negations, $2k$ coded bits onto $2k$ input
neurons, so exactly half the coded input is driven whatever the raw bits
are. Place $i$ of the input zone shows coded bit $i$: there is no
permutation. The pattern presented is the pattern the read is scored
against — nothing corrupts an input on the way in.
*Byron, September 14, 2026, defining the task: "An input is
complement-coded and presented on the list of input neurons. The desired
output is exactly the input expressed across the list of output neurons."
And, the same day, on the permutation: "Permuting patterns should no longer
matter. All neurons are first-class citizens of the population."
Complement coding as the only coding, Byron, September 17, 2026, 15:10 MDT
[RECORD §4.3, §8].*
*Of the engines:* the raw bits are the input zone's width less the clock
neurons of 5.3, halved; an odd remainder is refused rather than rounded.

**5.3 Clock neurons are input neurons whose bit is always 1.** A problem
says how many. They are the first that many neurons of the input zone, they
take no raw bits and the coding of 5.2 leaves them alone, and their coded
bit is 1 every epoch, so the drive fires them as it fires any bit-1 input.
They are inputs and not outputs: no read sees them.
*Byron, September 15, 2026: "Clock neurons can be created for a task as
input neurons always driven by 1" [RECORD §4.3].*
*Of the engines:* the coded pattern is the clock neurons' 1s followed by
the complement-coded bits, in that order.

**5.4 The drive is a Poisson rate drive.** An independent Poisson process
drives each input neuron across the epoch: at INPUT_RATE where its coded
bit is 1, at INPUT_RATE_OFF where it is 0. The arrivals of one neuron are
drawn $\mathrm{Exp}(\lambda)$ apart from the network's own seeded stream,
starting at the epoch's moment $t_e$, until a draw falls at or past the
horizon — that last arrival is drawn and discarded. The neurons are drawn
place by place in place order, and a place whose rate is zero draws
nothing. The whole epoch's arrivals are drawn before the schedule runs, so
they precede that epoch's firing draws in the stream.
*Byron, September 13, 2026: "I want to have an option to present the inputs
as a probabilistic firing rate rather than presenting them all at once and
seeing what happens." The default from September 14, 2026, and the drive
this specification states, Byron, September 17, 2026, 14:58 MDT: "Please
include the Poisson drive as it ran" [RECORD §4.3].*
*Of the engines:* all three take these draws from the one stream in this
order, so a seed gives every engine the same arrivals; an engine that
generates its own would agree on nothing else.

**5.5 These are arrivals, not spikes.** An arrival is a mandated spike and
not a charge: it adds nothing to the potential, and a neuron that takes one
spikes as it spikes from any other cause. An arrival landing while its
neuron is refractory is dropped, so a driven neuron fires at the first
arrival after its refractory period ends and its spike train is a renewal
process with dead time,
$$\text{ISI} = \text{REFRACTORY} + \mathrm{Exp}(\lambda).$$
Two arrivals at one moment fire one spike. A driven neuron fires in its
wave's fire phase before any neuron decides, and takes no firing decision
of its own in that wave. The refractory period that counts is the neuron's
own, so a spike the network itself caused silences the drive too; arrivals
are therefore emitted in full and never thinned in advance.
*Byron, September 14, 2026: "actual neural signalling is NOT a Poisson
process. If we want to use a Poisson process to drive the input neuron, it
should have its own, much higher rate. The effect should be to ensure the
input neuron fires at the first Poisson arrival after the refractory period
ends" [RECORD §4.3].*

**5.6 The drive is specified by its coefficient of variation.** INPUT_CV is
the constant, 0.6. The driving rate follows from it exactly, and is what
the schedule draws with:
$$\lambda = \text{INPUT\_RATE} =
\frac{1 - \text{CV}}{\text{REFRACTORY} \cdot \text{CV}} = 0.133/\text{ms},
\qquad
\bar r = \frac{1 - \text{CV}}{\text{REFRACTORY}} = 80\ \text{Hz}.$$
INPUT_RATE and INPUT_RATE_OFF are the same drive in the other coordinate,
the rates of the processes that drive a bit-1 and a bit-0 neuron;
INPUT_RATE_OFF is 0, which makes a zero bit silence rather than a low rate.
The rate and the CV are one axis and not two: the dead time of 5.5 is the
only thing making the train regular, so fixing one fixes the other.
*Byron, September 14, 2026: "the drive is specified by its CV, and the
default is CV = 0.6" [RECORD §4.3].*
*Of the engines:* $\lambda$ is derived from INPUT_CV by the formula above,
never stored as a second constant that could drift from it.

**5.7 Nothing is driven at the epoch's moment.** No neuron is made to spike
at $t_e$: a driven neuron's first arrival is $t_e + \mathrm{Exp}(\lambda)$,
and each arrival starts its own cascade on its own phase. No problem
overrides the drive.
*Byron, September 14, 2026, asking what an epoch resets: "The question I am
getting at is whether there is a unified wave front in the input process at
time zero. I don't want any such thing" [RECORD §4.3].*
*Of the engines:* the process is regenerated from $t_e$ each epoch rather
than carrying a pending arrival across the boundary; a Poisson process has
independent increments, so restarting at $t_e$ is a continuation and not a
seam.

**5.8 A driven spike is marked as driven.** A neuron the drive fired this
epoch carries a mark for the epoch, cleared at the epoch's reset. Nothing
in this section reads it: the learning rule does, and what it does with it
is open there (§8) — the update as it ran passes over every synapse whose
target was driven this epoch, and the record leaves that open in three
forms.
*The mark is the drive's and has been there since forced drive [RECORD
§4.3]; that it is stated here, with its use left open in §8, is Claude's
reading, to be corrected in a word [RECORD §6.7].*

**5.9 The inputs are drawn up front, from a stream of their own.** A run's
inputs are one raw-bit pattern per epoch, drawn before the first epoch runs
from `random.Random(f"walnutbutter inputs {seed}")` — not the network's
stream, and consumed by nothing else — each bit a fair coin flip. A
**dataset** is a stream of the same kind with labels beside it: the split
in a seeded shuffle of its own,
`random.Random(f"walnutbutter mnist {split} {seed}")`, every image once and
then round again, each with its label, which the network holds for the
epoch. A run longer than its stream cycles it. The arrival times of 5.4 are
**not** part of this stream: they depend on the epoch's length and on
$\lambda$, and stay drawn per epoch from the network's own stream.
*Byron, September 14, 2026 [RECORD §4.5]; the dataset form September 15,
2026 [RECORD §4.5, §8].*
*Of the engines:* seed $s$ means the same patterns in the same order
whatever the network is built like, which is what makes two runs comparable
epoch by epoch and not only on average.

**5.10 The read is the count read.** Each output neuron's spikes since the
epoch's reset, over the epoch's length, is its estimated rate in hertz,
$$r_j = \frac{1000\,n_j}{\text{INTERVAL}},$$
and the neuron is **on** when $r_j$ is at least TEACHER_THRESHOLD, 14.3 Hz.
It is the only read.
*Byron, September 14, 2026: "Here's how we actually score: COUNT the number
of times each neuron fired in the epoch. ESTIMATE the firing rate based on
the count. If the firing rate estimate exceeds TEACHER_THRESHOLD, the
output neuron is 1. Otherwise it is zero." The line is his the same day, to
mean one spike at a 35 ms epoch and not for the score it yields [RECORD
§4.3, §1.2].*
*Of the engines:* the comparison is $\ge$, and each engine snapshots its
spike counts at the epoch's reset so that $n_j$ is the spikes since that
snapshot. The line is quoted per second and holds at every epoch length, so
the count it demands follows from the length: at a 35 ms epoch one spike is
28.6 Hz and the line means at least one spike; at 100 ms one spike is 10 Hz
and it means at least two.

**5.11 The output zone is complement-coded.** For $C$ classes and a
population of $P$ neurons a class the zone is $2CP$ neurons: the $C$
fire-if-one populations first, class $k$ owning the $P$ neurons from $kP$,
and then, in the same order, the $C$ fire-if-zero populations — the
ordering the input zone uses for bits and their negations, so the two zones
read alike. The problem names $C$ and $P$ (§10). The label code the read is
scored against is the label's fire-if-one population on and its
fire-if-zero population off, and every other class the other way round.
*Byron, September 16, 2026, about 11:40 MDT: "In the output, I'd like to
force complement coding. How about we try ten classes x a population of six
neurons: three fire-if-one and three fire-if-zero? We will need to change
our scoring rule accordingly." Kept the same day, 16:20 MDT, **for now**:
"we'll stick with the complement coding for now. It should not hurt us"
[RECORD §8, decision 6].*

**5.12 The class evidence.** The evidence for class $k$ is its fire-if-one
count sum minus its fire-if-zero count sum,
$$n_k = n_k^{+} - n_k^{-},$$
a spike from a fire-if-zero neuron being one unit of evidence against its
class. The counts are the read's of 5.10, before any threshold. What a
critic does with the $n_k$ is §9's.
*Claude's reading of "change our scoring rule accordingly" (September 16,
2026), to be corrected in a word [RECORD §8, decision 6].*
*Of the engines:* every engine reads the zone through the one function that
computes this, so all three agree on the evidence by construction.

**5.13 Open: whether a driven input neuron also fires on its own.** A
driven input neuron is a neuron like any other, so the firing rule of §6
applies to it between arrivals as it applies everywhere. Whether it should
is not settled.
*Recorded September 16, 2026 and left to Byron: "Whether a driven input
should carry the hazard's rest at all is a rule about the neuron, and
Byron's to call" [RECORD §8]. Until he calls it, §6 applies unchanged to
the input zone.*

---

*Named here as an intention and not as a clause (Byron, September 17, 2026,
14:58 MDT — "Please include the Poisson drive as it ran"): the drive is to
become one deterministic spike every TARGET_ISI on each driven neuron,
specified but not built, and a later change to 5.4 through 5.7 rather than
a rule now in force.*

## 6. Firing

*Firing is escape noise: at every wave each neuron that is not refractory makes a stochastic decision on its own margin above the threshold. This section states that decision as the working runs made it. Byron, September 17, 2026, 14:50 MDT, choosing it: "first we must tackle neural spike escape noise as a potential mechanism… we have a learning rule derived for it right now that is working right now"; and at 14:53 MDT, on the level it is measured from: "We are going to keep the level. The system must implement what I was just using. We can make changes later to this."*

**6.1 The potential axis is quoted per unit of fan-in.** THRESHOLD and MINIMUM_POTENTIAL are quoted at THRESHOLD_FAN_IN incoming synapses, and a container that scales starts neuron $j$, of in-degree $d_j$, at

$$\theta_j = \text{THRESHOLD}\cdot\frac{d_j}{F}, \qquad
p^{\min}_j = \text{MINIMUM\_POTENTIAL}\cdot\frac{d_j}{F}, \qquad
F = \text{THRESHOLD\_FAN\_IN} = 18 .$$

The floor moves with the threshold because the two are points on one axis and it is the axis being rescaled: $p^{\min}_j/\theta_j$ is the container's ratio at every fan-in. A neuron with no incoming synapses has nothing to scale by: it keeps the container's quoted threshold and floor at scale 1, and so keeps a width (§6.4). The threshold belongs to the container; goo's are GOO_THRESHOLD $= 0.2$ and GOO_MINIMUM_POTENTIAL $= -0.8$, in the same ratio of $-4$. These are **starting** values only — where a run turns on homeostasis or un-sticking (§8, non-defaults) a threshold moves from where it starts, and a checkpoint stores the thresholds and floors a run actually reached rather than recomputing them.
*Byron, September 14, 2026, deciding the threshold and then the floor with it, and the same day that the threshold belongs to the container. The rule is linear because that is what "scales with fan-in" says without further instruction — Claude's reading, to be corrected in a word — and the scale-1 case for a neuron that hears nothing is Claude's resolution of September 16, 2026, likewise. [RECORD §5.2, §1.2]*

**6.2 The floor applies once a wave, to the wave's total.** After every signal of a wave has been delivered and before any neuron decides, each neuron touched this wave takes $p_j \leftarrow \max(p_j,\, p^{\min}_j)$, so the floor acts on the wave's summed input and the result does not depend on the order the signals arrived in. Of the engines this requires one floor per touched neuron per wave, not one per arrival.
*The floor is MINIMUM_POTENTIAL of §6.1; its place in the wave is the record's wave, and the deliver-then-settle order is what makes the result order-independent. [RECORD §4.4, §5.1]*

**6.3 A neuron at the floor has nothing of any synapse left in its potential.** When the floor bites, every synapse's open arrivals on that neuron are closed with no credit — the debit they have accrued is settled and their traces cleared — so a neuron held down by inhibition accumulates nothing on its synapses (§8, the single-spike rule; under the leak the traces are zeroed instead).
*A consequence of §6.1's floor made explicit by Byron's single-spike rule, September 17, 2026, and by the hazard eligibility of September 15, 2026: the synapse's trace is the margin's derivative, and at the floor the potential is the floor whatever the weights. [RECORD §6.7]*

**6.4 The margin, and the width of the decision.** Neuron $j$ decides at time $t$ on its margin

$$s = p_j(t) - \theta_j(t),$$

the potential after the wave's signals and after the floor, against the threshold it faces. The decision's width is

$$\Delta_j = \Delta\,\theta_j^{\text{start}}, \qquad \Delta = \text{ESCAPE\_DELTA} = 0.455,$$

quoted in units of the threshold the container gave the neuron, so §6.1's scaling applies to the width as to the rest of the axis, a neuron of any fan-in is as soft as any other, and the width stays put when homeostasis later moves $\theta_j$. A network's widths are set once the thresholds are what the container gave them — after fan-in scaling, before anything moves them.
*Byron, September 15, 2026, choosing the stochastic decision: "Let's please go with (3). It is the most like what I want to do. Make the boredom stochastic and it is Williams's unit outright." The value 0.455 is Byron's word, September 15, 2026. [RECORD §5.2, §1.2]*

**6.5 The hazard and the chance of a spike.** A neuron that is not refractory fires at a wave with probability

$$P_j(t) = 1 - e^{-m_j(t)}, \qquad
m_j(t) = \frac{\Delta t}{\text{hop}}\,\sqrt{\frac{N_0}{N}}\;e^{\,s/\Delta_j}, \qquad
N_0 = \text{ESCAPE\_REFERENCE\_COUNT} = 60,$$

where $\Delta t$ is the time elapsed since the neuron's previous decision, or since its refractory period ended if it was refractory then, and is never negative. The neuron carries a hazard of $\sqrt{N_0/N}\,e^{s/\Delta_j}$ spikes per hop: at the reference count one expected spike per hop at threshold, $e$ times more per $\Delta_j$ of margin above it and $e$ times fewer per $\Delta_j$ below. $m$ is the number of spikes expected over the interval and $P$ the chance of at least one; the hazard is charged for elapsed time rather than tossed per wave so that a neuron's rate does not depend on how busy its neighbours are. Of the engines: $m$ is capped at $10^3$, beyond which $P$ is 1 to the last bit, and $P$ is computed as $-\text{expm1}(-m)$, in that form, so the three engines agree.
*Byron, September 15, 2026, as §6.4. The cap and the expm1 form are Claude's, required for the engines to agree bit for bit; to be corrected in a word. [RECORD §5.2]*

**6.6 The hazard falls as the square root of the count.** Every hazard in a network of $N$ neurons — $N$ being every neuron the network holds — is multiplied by $\sqrt{N_0/N}$. It is the whole hazard that is scaled and not the width, so the decision keeps its sharpness, $e$ times per $\Delta_j$, and the score of §8 keeps its form; the factor is the same at every margin, and a neuron that hears nothing takes it like any other. The reference count is the unit the width is quoted in, as THRESHOLD_FAN_IN is the unit the threshold is quoted in, and is not a knob. Of the engines: the factor is recomputed from the count when a network is built or resumed rather than stored in a checkpoint, since it is a rule and not a state.
*Byron, September 16, 2026: "A design principle of this project is that it should have as few knobs as possible and by default operate in or near a stable regime. Therefore scaling the network MUST reduce the probability of escape noise at each neuron by sqrt(N). I want to build this into the rule." [RECORD §5.2]*

**6.7 The draw is one uniform per neuron per wave.** At each wave, after the floor and before anything fires, one uniform is drawn for every neuron, in neuron order, from the exploration stream (§7.3). A neuron fires iff its uniform is strictly below $P_j(t)$. Of the engines: the draws are taken at that point in the wave, in that order, and one per neuron whether or not that neuron goes on to decide, so the three engines fire the same neurons at the same waves from the same seed.
*The rule as built and held by test since September 15, 2026; the draw's position and order are Claude's reading of what reproducibility across the engines requires, to be corrected in a word. [RECORD §5.2, §6.1]*

**6.8 Every neuron decides at every wave, touched or not.** A wave examines every neuron in the network: one decision each, whether or not a signal reached it that wave, so a neuron with enough potential waiting, or one whose margin has simply carried it, fires at the first wave after it is able to and not at the next signal to arrive. A forced neuron is the exception (§6.10), and a refractory neuron makes no decision at all (§6.12). This fixes the learning rule's clock as well as the firing rule's: every decision is a charge of the single-spike rule (§8), and a neuron's expectation of itself moves per decision.
*Stated in the record with the bored neuron, September 12, 2026, and carried into the hazard on September 15, 2026; measured to be about 3,300 decisions an epoch on the mnist goo, which is what DECISION_MEMORY is quoted against (§8). [RECORD §5.2, §5.4, §1.3]*

**6.9 A neuron with no width takes the deterministic comparison.** Where the width $\Delta_j$ is not positive — ESCAPE_DELTA $=0$, or a starting threshold that is not positive, a collapsed axis carrying no width to quote — the neuron fires iff $p_j(t) \ge \theta_j(t)$ and it is not refractory. $\Delta = 0$ is the deterministic rule word for word. Of the engines: this case is taken by the comparison and not by dividing by a zero width.
*Byron, September 15, 2026, with §6.4 ("$\Delta = 0$ is the deterministic rule word for word"); the collapsed-axis case as resolved September 16, 2026, under which, given §6.1, it arises only for a threshold given as 0 outright. [RECORD §5.2]*

**6.10 A forced neuron fires by its stimulus and decides nothing.** An input neuron driven at its stimulus's time fires regardless of its potential and its threshold, the refractory period permitting, and makes no firing decision that wave — so it charges no credit and no expectation to its synapses for that spike (§8). Forced neurons fire before the wave's deciding neurons; a stimulus listed twice fires once, the first spike making the neuron refractory.
*The record's firing rule, standing: "A forced input neuron fires at its input's time regardless of $p$ and $\theta$"; and, with escape noise, September 15, 2026: "A forced neuron fires by its stimulus and makes no decision that wave." [RECORD §5.2, §4.4]*

**6.11 The spike resets the neuron and is remembered.** A neuron that fires takes

$$p \leftarrow 0, \qquad t_{\text{prev}} \leftarrow t_{\text{fired}}, \qquad t_{\text{fired}} \leftarrow t,$$

so the potential begins accumulating afresh (§5, the evidence accumulator) and the neuron keeps its last two spike times; the gap between them is what its own rules read as its interspike interval (§7.4).
*The record's firing rule, standing since the project began. [RECORD §5.2]*

**6.12 The refractory period.** A neuron that fired at $t_{\text{fired}}$ is refractory while $t < t_{\text{fired}} + \text{REFRACTORY}$, REFRACTORY $= 5$ ms. While refractory it ignores every signal, does not integrate it, cannot be forced, and makes no decision; its hazard resumes at the period's end, which is where §6.5's $\Delta t$ then runs from. A neuron may fire any number of times, the refractory period permitting: at REFRACTORY_HOPS $= 2$ — a hop of 2.5 ms — a spike sent around a two-way loop returns exactly as the period ends and may refire the neuron.
*Byron and Cedric, from the beginning: this is a feedback control mechanism and a computational feature of the system, not a limitation to be worked around. Two hops is Byron, September 17, 2026: "Please set hops=2 by default." [RECORD §0, §5.3, §4.4, §1.2]*

## 7. Exploration

**7.1 The firing decision is the exploration.** The system explores by the neuron's own stochastic decision (§6.5) and by nothing else: nothing is added to any potential and no perturbation has to be recorded, because the decision that was taken is what the learning rule scores (§8). This is Williams's Bernoulli semilinear unit with the noise in the threshold rather than on the potential — an escape-noise integrate-and-fire neuron, the spike response model's soft threshold. ESCAPE_DELTA sets how quiet silence can be: it is the one constant that says how much the system explores.
*Byron, September 15, 2026: "Make the boredom stochastic and it is Williams's unit outright." The identification is Claude's reading of Williams §2, to be corrected in a word. [RECORD §5.2, §6.7]*

**7.2 Boredom is a rate, not a deadline.** A neuron nobody talks to sits at rest, $s = -\theta_j$, and fires on its own at $e^{-1/\Delta}\sqrt{N_0/N}$ spikes per hop, from wherever its margin sits and without waiting on any clock. This is the whole of what turns silence into a spike in this specification, and it holds for a neuron with no incoming synapses as for any other (§6.1).
*Byron, September 15, 2026, deciding that the hazard is the bored neuron and that nothing else is run on top of it: "The hazard is buying us what the bored clock was supposed to buy us, and much much more cleanly." [RECORD §5.2, §5.4]*

**7.3 One stream, one order, in every engine.** The uniforms of §6.7 come from the run's exploration stream: one seeded stream, distinct from the input stream, which supplies the firing decisions and nothing else. A network whose width is positive must be run with one. Of the engines: each takes the same number of draws from that stream, in the same order, at the same point in the wave, and the Rust loop takes the Python stream's MT19937 state and hands it back, so a seed gives the same spikes whichever engine runs — the objects and Rust to the bit, the arrays exactly on the spikes and to a part in $10^9$ on continuous quantities (the platforms clause, §0).
*The stream is Byron's separation of streams, September 14, 2026 — a run's randomness comes from a stream that nothing else consumes — carried to the hazard's draws on September 15, 2026. [RECORD §5.2, §6.1, §4.5]*

**7.4 The ISI factor.** With $I = \text{TARGET\_ISI}$ and $t$ the time since the neuron's own last spike, every charge of the learning rule made at a decision is weighed by

$$f = \frac{3x - 1}{1 + x^3}, \qquad x = \frac{t}{I}.$$

It is $-1$ at $t = 0$, zero at $t = I/3$, $1$ at $t = I$ and nowhere higher, and falls as $3(I/t)^2$ after. A neuron that has never fired is at $t = \infty$, where $f = 0$: nothing is charged before its first spike. Of the engines: each computes $x$ and then $(3x-1)/(1 + x\cdot x\cdot x)$ in that order, so the objects and Rust agree to the bit and the arrays to a part in $10^9$.
*Byron, September 17, 2026, 04:31 MDT: "With absolute refractory period ABS, we would like to be able to recognize spikes coming back after multiple hops. The desired ISI is 5.1 ms (hardcode for now). The teacher's existing reinforcement is MULTIPLIED by f(t-ISI) where f(0) = 1, f(infinity)=0, f(-ISI)= -1." And the same morning: "Please make implement the ISI mechanism I suggested, despite your reservations, in authority. Only the constants are not known." The cubic rational is the form Fable derived before leaving; it is the lowest power whose tail has a finite integral. **Open:** TARGET_ISI $= 5.1$ ms is hardcoded for now and is *not known* — Byron's words — and neither is whether $t$ should be measured from the neuron's own last spike. [RECORD §0.2, §1.3]*

**7.5 The factor weighs every charge, and is on by default.** ISI_FACTOR is on. At each decision of neuron $j$ at time $t'$ both halves of the single-spike rule's charge are multiplied by the same $f$ — the decision's credit $c_j$ and its expectation $q_j$ alike, under either eligibility (§8) —

$$e_{ij} \mathrel{+}= f\big(t' - t^{\text{fired}}_j - I\big)\,\big(c_j(t') - q_j(t')\big)\,x_{ij}(t'),$$

so a spike is credited by how near its interspike interval came to $I$ and a silence is debited by the same weight at its moment. `--no-isi-factor` runs a network without it.
*Byron, September 17, 2026, choosing between on and off: "On by default." That $t$ is the time since the neuron's own last spike, and that "the teacher's existing reinforcement" means every per-decision charge of the single-spike rule, is Claude's reading, to be corrected in a word. [RECORD §0.2, §6.7]*

**7.6 A resumed network keeps the setting it was saved under.** A checkpoint records whether its network ran with the ISI factor, and a resumed run continues under the setting the checkpoint carries unless the resuming run says otherwise, so an arm continues under the rule it started with; each run's record says which it ran.
*Claude's reading, September 17, 2026, for the sweeps that were running when the factor landed; to be corrected in a word. [RECORD §0.2]*

---

*Two later changes, named here as intentions and not as clauses of this section. **Exploration at the synapse** — Byron, September 17, 2026, 06:05 MDT: "Computationally exploration noise can be generated at the synapses, and semantically this is clean: a SYNAPSE explores its own impulse response, rather than a NEURON exploring its impulse response!" — under which the exploring object and the credited object would be the same object, and the fan-in hazard would replace §6.5. Not specified and not built. And **deprecating $\theta$**, with the margin and the width then quoted from something other than the threshold. Until each is written as a clause, §6 and §7 as written are the rule in force.*

## 8. The learning rule

One rule moves a weight: REINFORCE (Williams 1992, [1] in `BIBLIOGRAPHY.md`)
with a per-decision eligibility. The clauses below say what a synapse earns,
what it is paid, and what it has to carry to do either.

**8.1 The rule.** *(The pre-alpha's, factored out behind RULE = reinforce
[record §6.7].)* The network is paid **one scalar reward** $R$ an epoch,
from the critic the run names. A running baseline $b$ gives the
**advantage** $A = R - b$. At the read, every synapse $i \to j$ that has
charged something since the last read and whose target $j$ was **not forced**
this epoch moves by

$$w_{ij} \leftarrow \mathrm{clip}\big(w_{ij} + \text{LR}\cdot A\cdot e_{ij}\big),
\qquad \text{LR} = 0.03,$$

clipped to WEIGHT_RANGE, the same $[-1, 1]$ the weights were drawn from. A
synapse into a neuron forced this epoch is passed over: its firing was not
the network's doing. Every synapse in the network gets the same $A$; what
tells them apart is $e_{ij}$ alone.

**8.2 The baseline.** *(The pre-alpha's [record §6.7, §1.3].)* $b$ is the
running average of the reward. It starts at the first epoch's reward, so the
first epoch's advantage is zero and nothing moves; after the update has used
it, $b \leftarrow b + \text{BASELINE\_RATE}\,(R - b)$, BASELINE_RATE $=
0.05$. There is one baseline for the whole network and none per neuron.

**8.3 Two eligibilities, and the eligibility follows the neuron.** Two
survive: **hazard** and **hebb**. Both are the single-spike rule of 8.4 and
differ only in what a decision's credit and expectation are (the table
there). Unless a run names one, the eligibility follows the neuron: hazard
where the firing decision is a draw (escape noise on, the firing clause
[record §5.2]), hebb where the threshold decides. The hazard refuses a
network without escape noise — a decision that was not a draw has no
probability to differentiate — and says so rather than approximating.
*Claude's reading of the default Byron set September 15, 2026, to be
corrected in a word: the record's rule was "hazard under escape noise, else
the ELIGIBILITY constant", and that constant's value, perturb, is not in this
specification.*

**8.4 The single-spike rule.** *(Byron, September 17, 2026: "Let's derive
the REINFORCE rule for the synaptic weight update for a SINGLE postsynaptic
spike at time t. We will be modifying the existing hebb and hazard
eligibilities. Hebb assumes there is no hazard and that the postsynaptic
spike was generated by presynaptic activity. Hazard assumes that there is
escape noise." Claude's derivation, on the evidence accumulator [record
§6.7].)* At **every decision** of neuron $j$, at time $t'$, with outcome
$y_j(t') = 1$ if it fired and 0 if it did not, every synapse into $j$ moves
its score by

$$e_{ij} \mathrel{+}= f\,\big(c_j(t') - q_j(t')\big)\,x_{ij}(t'),$$

$x_{ij}$ the synapse's trace (8.5), $f$ the ISI factor (8.9), $c_j$ the
decision's credit and $q_j$ its expectation:

| eligibility | fired: $c_j$, $q_j$ | silent: $c_j$, $q_j$ | the expectation |
|---|---|---|---|
| hazard | $m\,e^{-m}/(1 - e^{-m})$, 0 | 0, $m$ | $m_j(t')$, the hazard's own (8.7) |
| hebb | 1, $\hat p_j$ | 0, $\hat p_j$ | $\hat p_j$, the neuron's own estimate of its spike (8.6) |

In words: a spike credits every arrival still standing in the potential, each
equally, and every decision debits them by what was expected of the neuron at
that moment. A neuron that fires more than once in an epoch is charged for
every decision it made.

**8.5 The trace.** $x_{ij}(t')$ is what the synapse has in its target's
potential at the decision — the derivative of the margin by $w_{ij}$, not a
stand-in for it. Under the evidence accumulator it is the **count** of the
arrivals $j$ integrated since its last spike; under the leak (8.12) it is
those arrivals each decayed by TAU. A signal that arrives while $j$ is
refractory is dropped, adds nothing to the potential, and counts nothing
here. The trace is cleared whenever the potential stops being the sum of
those arrivals: at $j$'s spike, and when the floor bites, the potential being
then the floor whatever the weights. The trace belongs to the synapse, and it
is what gives the rule per-synapse resolution [record §6.7].

**8.6 The neuron's expectation of its own spike.** *(Byron, September 17,
2026: "Expectation is changed per decision in this architecture.")* $\hat
p_j$ is undefined until $j$'s first decision, which sets it to that
decision's outcome and charges nothing. After every decision, charged first,

$$\hat p_j \leftarrow \hat p_j + \max\!\big(\text{DECISION\_MEMORY},\,1/n\big)\,(y - \hat p_j),$$

$n$ the decisions $j$ has made to date, DECISION_MEMORY $= 10^{-4}$: the
plain mean of the first ten thousand decisions, an exponential average of
about the last ten thousand after. *Open: a starting value, to be swept
[record §1.3]. The warm start by $1/n$ is Claude's reading of "changed per
decision", to be corrected in a word.*

**8.7 The hazard's credit and expectation.** *(Claude's derivation from [1],
asked for and chosen by Byron, September 15, 2026: "It is the most like what
I want to do" [record §6.7].)* $m = m_j(t')$ is the spikes the escape-noise
hazard expects of $j$ over the interval this decision covers, exactly as the
firing clause computes it [record §5.2]. The credits of the table are that
interval's log-likelihood differentiated: $\partial m/\partial w_{ij} =
(m/\Delta_j)\,x_{ij}$, and the $1/\Delta_j$ — the neuron's own width — is
folded into LR, so one learning rate serves neurons of any width.

**8.8 Hebb's credit at the spike — open.** It is a full unit, where the
hazard's credit discounts a spike that was expected. Whether it should be
discounted too is **not decided**: Byron, September 17, 2026, asked which,
answered *"Defer for now."*

**8.9 The ISI factor weighs every charge.** *(Byron, September 17, 2026, the
factor and its default; the reading of what it weighs is Claude's, to be
corrected in a word [record §0.2].)* $f$ in 8.4 is the ISI factor of the
clause at the head of this file, evaluated at the time since $j$'s own last
spike, and it multiplies **both** the credit and the expectation of the
decision it is made at: hebb and hazard alike, at the decision and not at the
read. A neuron that has never fired stands at $f = 0$, so nothing is charged
on it before its first spike. It is on by default; a run may turn it off, and
a resumed network keeps the setting it was saved under (8.14). count_hebb,
wrong_hebb and perturb had no per-decision charge to weigh; they are not in
this specification.

**8.10 Arrivals stay open across reads.** *(Byron, September 17, 2026: "Let
them run! Epochs are for the convenience of teaching the network, and to some
extent an emergent property of the time constants, but they don't really
exist in nature.")* An arrival stays in the potential until $j$ spikes, so
what it owes runs across reads. Each read pays the score charged since the
last read and clears the score; the open arrivals stay open, their debit
counting from then. The epoch bounds the payment, not the accumulator.

**8.11 What the synapse owns, under the accumulator.** Each event below is
one operation on the synapse or on the neuron, and **no loop over a fan-in
runs at a decision**. Per synapse: the trace $x_{ij}$, the note $B_{ij}$ and
the score $e_{ij}$; per neuron: $E_j$, the spikes expected over its decisions
since its last spike.

- **an arrival** integrated: $x_{ij} \mathrel{+}= 1$ and $B_{ij} \mathrel{+}= E_j$;
- **a decision**: $E_j \mathrel{+}= f\,q_j$, and nothing else;
- **the spike**: every open arrival settles, $e_{ij} \mathrel{+}= f\,c_j\,x_{ij} - (x_{ij}E_j - B_{ij})$, then $x_{ij} = 0$, $B_{ij} = 0$ and $E_j$ restarts at 0;
- **the floor, a forced spike and a discharge**: settle the same way with no credit;
- **the read**: settle the debit so far into each score, $e_{ij} \mathrel{-}= x_{ij}E_j - B_{ij}$ with $B_{ij} \leftarrow x_{ij}E_j$, pay 8.1, and clear the score.

This is a regrouping of the per-decision sum of 8.4 and equals it to about a
part in $10^{15}$, which is not bit equality: the engines are held to each
other (8.15), not to the sum.

**8.12 The leak path.** The leak is a per-run option, TAU (the neuron's
clause [record §5.1, §1.2]). Under it the rule of 8.4 is charged **per
decision**: at every decision the engine walks $j$'s incoming synapses and
adds $f\,(c_j - q_j)$ times each leaked trace. The bookkeeping of 8.11 is not
available there — the regrouping needs factors of $e^{t/\text{TAU}}$ that
overflow within a second of run — and the per-decision walk is the rule for
that path.

**8.13 A late signal counts, and there is no second trace.** LATE = count: a
signal arriving after its target has already fired earns its count like any
other, the trace saying what each signal was contributing at each decision
[record §6.7]. Neither surviving eligibility takes any other value of LATE,
and neither takes a leaky eligibility trace on top of it. A run that asks for
either is refused, not approximated.

**8.14 What a checkpoint carries.** The weights; per neuron $\hat p_j$, the
decisions to date and $E_j$; per synapse the trace and the time it was
brought up to, and the note $B_{ij}$; the teacher's baseline $b$, LR, critic
and eligibility; and whether the ISI factor was on, a resumed network keeping
the setting it was saved under unless the resuming run says otherwise
[record §6.7, §0.2]. The score since the last read is not carried: a
checkpoint is written between epochs, where the score has just been paid and
is about to be cleared. *Claude's reading, to be corrected in a word.*

**8.15 What the rule requires of an engine.** Every engine charges at the
decision, in the decision's own order, and carries the state 8.11 names —
the trace, the note and the score on each synapse, $\hat p_j$, the decision
count and $E_j$ on each neuron — so that the read is one pass over the
synapses and pays what the engine already accumulated. The object engine and
the Rust loop agree to the bit; the array engine agrees within the tolerance
the invariants clause names for it, summing a wave in matrix order. An
engine that lacks the rule refuses the run and says so.

## 9. The teacher

The teacher stands outside the network. Once an epoch it reads the output
zone, scores it, pays the one rule that pays at the read, and keeps its
book. Nothing it does reaches a neuron except through that neuron's own
threshold and its synapses' weights.

### 9.1 At most one rule pays at the read

*Byron, September 13, 2026, and it is why this section and §10 are two:*
"there is not a neuron training rule. There are multiple compatible
training rules."

A **local rule** needs nothing but the neuron's own spikes and the stamps
on its own synapses. It runs inside the wave loop, under whatever else is
running, it has its own rate, and it is off until a run asks for it (§10).

A **rule that pays at the read** needs a number from outside the network.
At most one of them runs, it runs once an epoch at the read, and a run may
have none, in which case the local rules are the whole of the learning
("no teacher for now", Byron, the same day).

The specification carries one of each: the reinforce rule pays at the read,
the quash is local. They compose; neither is a setting of the other.
[RECORD §6]

### 9.2 The order at the read

*Claude's reading of the rule as built, to be corrected in a word [RECORD
§6.7, §6.15].* When an epoch's last wave has run, the teacher does these,
in this order:

1. read the output zone and compute the epoch's reward $R$ from the critic
   (§9.4–9.6);
2. pay the rule at the read with the advantage $A = R - b$ (§9.3);
3. move each neuron's firing-rate memory (§9.8);
4. move thresholds by homeostasis (§9.9), and then by un-sticking (§9.10);
5. move the baseline $b$ (§9.3).

A neuron the stimulus forced to fire this epoch is skipped by (3) and (4).

The order is part of the rule: every engine performs the same operations in
the same order, so that a run is the same run whichever engine ran it.

### 9.3 The baseline and the advantage

The reward is paid as an advantage against a running baseline. The baseline
$b$ is set to the first epoch's reward; every epoch is paid
$A = R - b$ with $b$ as it stands, and only then does $b$ move:

$$b \mathrel{+}= \text{BASELINE\_RATE}\,(R - b), \qquad
\text{BASELINE\_RATE} = 0.05.$$

The advantage is one scalar an epoch, the same number for every synapse the
paying rule touches. [RECORD §6.7, §1.3; REINFORCE, Williams 1992, [1] in
`BIBLIOGRAPHY.md`]

### 9.4 The evidence critic

*Byron, September 16, 2026, on reading the class sums as a vote share:*
"No, the spikes are EVIDENCE for now, not a proper maximum-likelihood
estimator. We will have to sweep for temperature eventually."

The read gives one number $n_k$ per class, the class's evidence this epoch
(under complement coding, its fire-if-one sum minus its fire-if-zero sum;
the read's section defines it). The evidence critic takes those as log-odds
at a temperature $T$ = TEMPERATURE: the estimate over the $C$ classes is the
Boltzmann distribution and the reward is its log score at the label $y$,

$$q_k = \frac{e^{n_k/T}}{\sum_j e^{n_j/T}}, \qquad
R = \ln q_y = \frac{n_y}{T} - \ln \sum_j e^{n_j/T}.$$

A uniform estimate scores $\ln(1/C)$, which is $-2.30$ at ten classes; a
perfect epoch scores 0; a class with no spikes at all is weak evidence and
not $-\infty$. $T$ sets what one spike of lead is worth. It is mnist's
critic. [RECORD §8, §1.3]

TEMPERATURE is 2 **FOR NOW** — *Byron, the same day: "We will have to sweep
for temperature eventually"* — set at the middle of the first sweep's
$\{1, 2, 4\}$, and what is open is the value, which a sweep is to set.

*What it requires of the engines.* Every engine pays this reward through one
function, which subtracts the largest class sum before exponentiating so
that no temperature overflows. Its inputs are sums of integer spike counts,
so the reward is the same to the bit in all three engines and the tolerance
of the one-authority rule is not needed here.

### 9.5 The row critic

The reward is the fraction of the output neurons whose fired state this
epoch matches the target pattern the problem names — for mnist, the label's
code over the output zone. One neuron right in twelve is $1/12$; a row that
matches everywhere is 1. Kept as one of the three critics (*Byron,
September 17, 2026: "evidence, row, graded"*). [RECORD §6.7]

### 9.6 The graded critic

*Byron, September 15, 2026: "a graded critic it is."* On the same class
evidence the evidence critic reads, the reward is the fraction of the other
$C - 1$ classes the label's class strictly out-spikes: 1 when it out-spikes
every one of them, 0 when it out-spikes none, and a near miss paid for what
it beat. A tie is not beaten, so a silent output zone scores 0. [RECORD §8]

### 9.7 The fraction right — a reported measure, not a critic

Every run reports the fraction of epochs in which the label's class strictly
out-spikes every other — the same class evidence, the win read as 1 or 0 —
beside the reward it is actually paid. **No run is paid on it.** It is the
number the project is read by, and keeping it a report rather than a critic
is what lets the reward be the evidence critic's log score and the reading
still be a fraction.

*Byron, September 17, 2026, 15:10 MDT, settling it as a reported measure;
Claude's proposal, accepted in "confirm and proceed", to be corrected in a
word.* [RECORD §8; `docs/rewrite-answers.md` §4]

*What it requires of the engines.* It is read through the same function as
the critic's class evidence, so it costs an epoch one comparison and can
never disagree with the critic about what the output zone said.

### 9.8 The teacher's book: each neuron's firing-rate memory

The teacher keeps, for every neuron $j$, a running estimate $r_j$ of how
often it fires. After an epoch in which $j$ was not forced,

$$r_j \mathrel{+}= \text{RATE\_MEMORY}\,\big(\mathbb{1}[j \text{ spiked this
epoch}] - r_j\big), \qquad \text{RATE\_MEMORY} = 0.01,$$

about the last hundred epochs. An epoch in which the stimulus forced $j$
says nothing about the network and moves nothing. A neuron with $r_j$ above
STUCK_ABOVE = 0.99 is **stuck on**, one with $r_j$ below STUCK_BELOW = 0.01
is **stuck off**.

The book is the teacher's and not the network's: no neuron reads it, and it
is kept whether or not §9.9 and §9.10 run, because the report counts stuck
neurons (§9.11). [RECORD §1.3, §6.7, §6.15]

*What it requires of the engines.* Every engine keeps the book the way the
object engine does — one elementwise update per neuron, once an epoch, at
the point §9.2 gives it — so no summation order enters and the three agree
to the bit. It is saved in the checkpoint with the network, so a resumed run
carries its rates and thresholds on.

### 9.9 Homeostasis — off unless a run asks

**Non-default.** Every epoch, every neuron not forced that epoch has its
threshold moved toward its target firing rate:

$$\theta_j \mathrel{+}= \text{HOMEOSTASIS}\,(r_j - \text{TARGET\_RATE}),
\qquad \text{TARGET\_RATE} = 0.5.$$

Nothing clips where this takes a threshold. HOMEOSTASIS is **0** — the rule
does not run — unless a run asks for it; the rate the record's runs used
when it was on is $10^{-6}$.

The escape noise's width $\Delta_j$ is fixed at the neuron's *starting*
threshold, so a threshold this rule moves changes the margin the neuron
fires on and not the width it is drawn at.

*Kept as a non-default, Byron, September 17, 2026: "Homeostasis and
un-sticking, as non-defaults."* mnist asks for neither this nor §9.10
(*Byron, September 15, 2026: "Please turn off for this task"*). [RECORD
§1.3, §6.7, §8]

### 9.10 Un-sticking — off unless a run asks

**Non-default.** Every epoch, every neuron not forced that epoch whose rate
memory is outside the stuck band of §9.8 — $r_j >$ STUCK_ABOVE or
$r_j <$ STUCK_BELOW — has its threshold moved toward a target rate, and only
while it is outside that band:

$$\theta_j \mathrel{+}= \text{UNSTICK}\,(r_j - \text{UNSTICK\_TARGET}),
\qquad \text{UNSTICK\_TARGET} = 0.5.$$

**Every neuron**, not the output zone alone (*Byron, September 14, 2026:
"all neurons are first-class citizens"*). Nothing clips where it takes a
threshold. UNSTICK is **0** unless a run asks; the rate the record's runs
used when it was on is $10^{-3}$. The run reports how many un-stickings have
fired (§9.11).

*Kept as a non-default, Byron, September 17, 2026, with §9.9.* [RECORD §1.3,
§6.7]

### 9.11 What is reported

[RECORD §6.14] Every run reports:

- spikes to date, and which neurons fired this epoch;
- the epoch's reward, the mean reward to date, and an exponential moving
  average over about WINDOW = 200 epochs — $\alpha = 2/(\text{WINDOW} + 1)$,
  started at the first epoch's reward;
- the fraction right beside the reward (§9.7), wherever a reward is
  reported;
- how many neurons are stuck on and stuck off (§9.8), and how many
  un-stickings have fired (§9.10);
- at the end of a run, the mean reward over its last tenth.

A per-epoch trace may be written to a file: one line an epoch of the epoch
number, the clock time in milliseconds, and the reward. A sweep's arm
records its reward trace every so many epochs and, at the end, each neuron's
rate memory and threshold.

The network runs forever (§7): these numbers are read as health, not
convergence, and drift is normal.

### 9.12 The estimator's correlation with a supervised direction

*Byron, September 16, 2026, 04:25 MDT: "Please modify the code so we can
plot the correlation of the estimator over time after a run."*

A run may be given a **supervised direction** $d$: one number per synapse
over a named subset of the synapses. For mnist that subset is the
input-to-output synapses and

$$d_{ij} = P(\text{coded input } i \text{ on} \mid \text{the class of } j) -
P(\text{coded input } i \text{ on}),$$

reversed in sign for a fire-if-zero output neuron, which should fire when
the label is not its class. At every trace point the run then records, over
that subset: the Pearson correlation of each weight's change since the run's
start with $d$, the fraction of those synapses whose change has $d$'s sign,
and the correlation of the change since the previous trace point with $d$.
The weights are the estimator integrated, so this reads the estimator under
whatever eligibility is running.

It is a measurement and it changes nothing: no rule reads $d$, and a run
given no direction records none of this. A run resumed from a checkpoint
measures the change from the weights of its **first** start, and its trace
epochs count on from where it stopped. [RECORD §8]

## 10. Local rules

A local rule needs nothing but the neuron's own spikes and the stamps on its
own synapses, runs inside the wave loop under whatever pays at the read, and
is off until a run asks for it (§9.1). The specification carries one.

### 10.1 The quash — off unless a run asks

**Non-default.** *Byron, September 13, 2026, turning the rule around:* "when
a neuron fires itself, it has found a cycle. Cycles may be BAD. The whole
point of having an absolute refractory period is to quash short cycles ...
cycles need to be quashed, so move the contributing weights in the direction
that will actually do this."

A neuron cannot tell its own returning spike from anyone else's without
tracing ancestry, which the rules forbid, so "fires itself" is read as a
**refire**, and the delay since the neuron's previous spike is the evidence
of how tight the loop was (*Byron, the same day: "the refire delay is
evidence of how tight the loop is"*). When a neuron $j$ that has spiked
before spikes again at time $t$, its previous spike at $t_{\text{prev}}$,
every active incoming synapse of $j$ whose last integrated signal landed
after $t_{\text{prev}}$ moves by

$$w_{ij} \leftarrow \mathrm{clip}\Big(w_{ij}\big(1 - \text{QUASH\_RATE}\,
e^{-\text{QUASH\_K}\,(t - t_{\text{prev}})}\big),\ \text{WEIGHT\_RANGE}\Big).$$

Being proportional to the weight, it pulls toward zero from either side;
being restricted to the synapses that carried a signal since the previous
spike, it touches only what carried the cycle. A neuron's first spike
quashes nothing, there being no interval to read.

QUASH_RATE is **0** — the rule does not run — unless a run asks; the value
the record's runs used when it was on is 0.02, and QUASH_K is 0.2 per
millisecond. Both are **open**: *Byron, the same day, "we will have to
explore this space."* mnist does not quash.

*Kept as a non-default, Byron, September 17, 2026.* [RECORD §6.11, §1.3]

*What it requires of the engines.* The quash runs at the end of the wave in
which the refire happened, over that wave's fired neurons, before the next
wave is scheduled. It reads only the neuron's previous spike time and each
incoming synapse's last-signal stamp, which every engine already keeps; no
reward, no baseline and no teacher enters it, and it runs whether or not a
rule pays at the read.

### 10.2 The quash is the only local rule carried

Of the local rules the record composes, this specification carries the quash
alone: leaky Hebb and the weight decay are dropped with the rules they were
built beside, and stay in `RECORD.md` (§6.12, §6.8) and in the code at the
tag `lab-notebook-2026-09-17`. A local rule added later states, in its own
clause, where in the wave it acts relative to the quash, and is off until a
run asks for it. *The scope, Byron, September 17, 2026.* [RECORD §6, §6.8,
§6.12]

## 10. The problem

*The specification carries one problem. A problem is what a network is asked to do and how it is watched doing it: the layout, the inputs, and whether anything outside the network trains it [record §8].*

**10.1 mnist is the only problem.** The specification states one problem, mnist, and its layout, its stream, its critic and its reported measures are stated for it here. Where mnist names a value the problem's value stands; where it names none, the constants of Appendix A do (10.14).
*Byron, September 17, 2026, 15:02 MDT, setting the rewrite's scope: every problem but mnist leaves the specification and stays in the record. The eleven other problems the record poses are not rules and are not carried [record §8; `docs/rewrite-answers.md` §4].*

**10.2 The data.** MNIST is distributed as four IDX files, gzipped — images and labels for a training split and for a test split (LeCun, Cortes and Burges; [6] in `BIBLIOGRAPHY.md`). walnutbutter holds **two of them**, the training split's `train-images-idx3-ubyte.gz` (9,912,422 bytes) and `train-labels-idx1-ubyte.gz` (28,881 bytes), in `mnist/`, which is not in git. They are fetched from the mirror TensorFlow uses, `https://storage.googleapis.com/cvdf-datasets/mnist/<file>`, and a file whose size or SHA-256 sum is not the recorded one is refused; the sums are in `mnist/README.md` beside the source and are checked on every fetch.
*Byron, September 15, 2026: "Let's set up a new task. This one will need its own data folder: mnist." [record §8]*

**10.3 The test split is not fetched.** There is no held-out set, because there is no evaluation run to hold one out for (11.12): the score is what the living network does on the stream it is given, and the loader knows one split.
*Byron, September 15, 2026: "we do not dare touch the test split. please unfetch it." [record §8]*

**10.4 From image to bits.** Each 28 × 28 image is averaged over 2 × 2 blocks to 14 × 14, and each block is on if and only if its mean intensity is at least half of full; the 196 bits are taken row-major. The ink's weight is not carried: the network is shown a binary picture.
*Byron, September 15, 2026, choosing binary bits at 14 × 14 from the choices offered [record §8, decision 1].*

**10.5 The input zone: 395 neurons.** In order: **three clock neurons**, whose coded bit is 1 every epoch, which take no raw bit and which the coding leaves alone; the **196 pixel bits**; and the **196 complements** of those bits. Bit for neuron, in that order, and nothing rearranges them. Exactly 199 of the 395 are driven every epoch, whatever the digit. Clock neurons are inputs and not outputs, so no read sees them.
*Byron, September 15, 2026, defining the zone — "three clock neurons, the 196 on-off pixels, and the 196 complement-coded pixels" — and the same day, defining the clock: "Clock neurons can be created for a task as input neurons always driven by 1." [record §4.3, §8]*

**10.6 The output zone: 60 neurons, complement-coded.** Ten classes, each with a population of three **fire-if-one** neurons and three **fire-if-zero** neurons: the ten fire-if-one populations first, in class order, then the ten fire-if-zero populations in the same order. A class's evidence is the sum over its fire-if-one population minus the sum over its fire-if-zero population, n_k = n_k⁺ − n_k⁻ — a spike from a zero neuron is a spike against.
*Byron, September 16, 2026, about 11:40 MDT: "In the output, I'd like to force complement coding. How about we try ten classes x a population of six neurons: three fire-if-one and three fire-if-zero?" Kept at 16:20 MDT: "we'll stick with the complement coding for now. It should not hurt us." [record §8]*
*Engines: every engine reads the zone through one function, so the zone's arithmetic is written once and not three times.*

**10.7 The hidden count is the problem's.** mnist is posed on goo; the goo is its input zone, its hidden neurons and its output zone, and the hidden count is **199** unless the run says otherwise — 654 neurons at the default, 455 with none. Zero hidden neurons is a network the wiring must be able to build, the outputs hearing the inputs directly.
*Byron, September 16, 2026: "I would like to specify the number of 'hidden' neurons as hidden_neurons. One thing I neglected to do is benchmark this task without any hidden neurons. How will we know if they are buying us anything if they are always part of the economy?" [record §8]*

**10.8 The read is by count.** Each output neuron's spikes since the epoch began are its count, and the zone is read by those counts: the class evidence of 10.6 is built from them directly. The count read's line in hertz (TEACHER_THRESHOLD, Appendix A) is consulted only where a read is scored as bits, which is the row critic's case and not mnist's.
*Byron, September 14, 2026, setting the read: "COUNT the number of times each neuron fired in the epoch. ESTIMATE the firing rate based on the count." [record §4.3]*
*Engines: each engine snapshots its own spike counts at the epoch's reset, and the three are compared on them.*

**10.9 The critic is evidence, at TEMPERATURE.** The class sums are read as evidence at a temperature T: the estimate over classes is the Boltzmann distribution of the sums, and the reward is its log score, with y the label —

> q_k = e^(n_k/T) / Σ_j e^(n_j/T),  r = ln q_y = n_y/T − ln Σ_j e^(n_j/T).

T = TEMPERATURE = 2. **T's value is open in Byron's own words** — "We will have to sweep for temperature eventually" — and 2 is the middle of the first sweep's {1, 2, 4}, held until the sweep sets it.
*Byron, September 16, 2026: "No, the spikes are EVIDENCE for now, not a proper maximum-likelihood estimator. We will have to sweep for temperature eventually." [record §8, decision 4]*
*Engines: one function pays every engine and the Rust driver alike.*

**10.10 The target is the label's code.** For a critic that wants a pattern rather than the sums — the row critic — mnist's target is the label: the label's fire-if-one population on and every other off, the fire-if-zero populations the other way round.
*[record §8; the code follows 10.6's layout.]*

**10.11 The learning rate is the problem's: LR 0.002.** It is taken unless the run gives a rate of its own.
*Byron, September 16, 2026, about 10:17 MDT: "default LR to 0.002 for this task." [record §8]*

**10.12 Homeostasis and un-sticking are off for this problem.** mnist sets both rates to zero, and they stay zero unless the run says otherwise. They are non-defaults in any case; this clause is the problem's own setting of them.
*Byron, September 15, 2026, the same evening he set the output populations: "Please turn off for this task." [record §8]*
*Engines: both are the Teacher's per-epoch book rather than the network's dynamics, so an engine that mirrors them mirrors zero.*

**10.13 The stream.** A run's patterns come from the train split in a seeded shuffle of its own, `random.Random(f"walnutbutter mnist {split} {seed}")`, every image once and then round again, with each image's label beside it; the network holds the label for the epoch and the critic reads it. The shuffle is not the network's stream, so seed *s* shows the same images in the same order to any network, and a run longer than the split cycles it.
*Byron, September 14, 2026, deciding the input stream, and September 15 extending it to a dataset with labels [record §4.5, §8].*

**10.14 What mnist does not fix.** The epoch's length, the container's starting threshold and floor, the wiring, the eligibility and the engine are not the problem's: they come from the constants of Appendix A or from the run that is asked for.
*Claude's reading of what the problem sets, to be corrected in a word: the record's mnist runs gave the epoch length, the threshold and the floor on the command line rather than in the problem [record §8].*

**10.15 The fraction right is reported, not paid.** Every run reports the fraction of epochs in which the label's class **strictly out-spikes every other class** on the evidence of 10.6 — a tie is not a win, and silence is not a win — beside the reward the critic pays, and as a moving average over WINDOW epochs. Nothing is paid on it: it is a reported measure and not a critic.
*Byron, September 17, 2026, 15:10 MDT, accepting the proposal in "confirm and proceed". Claude's proposal, to be corrected in a word; the class critic whose reading it is does not survive as a critic [`docs/rewrite-answers.md` §4].*

**10.16 The estimator's correlation is reported.** For a synapse from input i onto an output of class k, the supervised direction is d_ij = P(coded input i on | class k) − P(coded input i on), taken over the training split; for a fire-if-zero output the sign is reversed, since that neuron should fire when the label is not k. A run reports the correlation of each input-to-output synapse's weight change since the run's start with d. It is an instrument for reading a run and not a rule: nothing in the network consults it, and no weight moves because of it.
*The instrument is Claude's, built September 16, 2026 at Byron's question about where the learning was; kept in the specification because every mnist run reports it [record §0.1, §8].*

---

## 11. The invariants

*These hold of the system, not of any one engine. No clause of this file may be true in one engine and approximate in another.*

**11.1 One authority, and every platform built from it.** This file is the only source a platform is built from. Every clause is implemented in every engine that runs it, and is tested before it is built.
*Byron, September 17, 2026, 15:02 MDT, on keeping more than one engine: "keep, with tolerances, as a foundational rule; all platforms are built from one clear authority."*

**11.2 An engine that lacks a rule refuses.** It says so and refuses to run the configuration; it never approximates, and a gap is closed in the engine rather than run around.
*Byron, September 14, 2026: "rules in authority.md must be implemented cross-platform." [record §7]*

**11.3 The engines.** Three: the **object engine** (neurons and a queue of waves), the **array engine** (numpy vectors and a sparse matrix) and the **Rust wave loop** behind PyO3. The Rust loop is the default engine for any sweep or long run; the array engine is the second opinion and the fallback where Rust cannot run a configuration.
*Byron, September 14, 2026: "please always default to the rust engine unless it is broken." [record §6.15]*

**11.4 What agreement means, and the one tolerance.** The object engine and the Rust loop agree **to the bit**: every spike, every score and every weight, compared with `==`. The array engine agrees **exactly on spikes** and **to a part in a billion** on continuous quantities — scores and weights — because it sums a wave's arrivals in the matrix's order where the other two add them in push order, and its exponentials are numpy's rather than libm's. That is the whole of the tolerance, and it is named here rather than left in a test.
*Byron, September 17, 2026, 15:02 MDT, keeping the array engine "with tolerances"; the tolerance's two sources are the record's [record §6.15, §7].*

**11.5 The summation order is part of the rule.** Signals due at one moment are summed in **push order**, the order the topology was built in, so an engine that flattens the topology differently sums a wave differently and lands on different bits. Every engine that claims bit agreement takes the object engine's push order.
*[record §6.15. Without this clause 11.4's `==` cannot be met, and the firing decision's order-independence — signals sharing a timestamp all arrive before any neuron fires — is a separate rule about the decision, not about the bits.]*

**11.6 One stream, one order.** Every draw a run makes comes from Python's MT19937 in one order. An engine outside Python takes that generator's state, draws the same uniforms in the same order at the same named point in the wave, and hands the state back, so Python's stream carries on from where the other engine left it. The draws are equal, not approximately equal.
*Byron, September 14, 2026, refusing an engine that would have approximated them: "rules in authority.md must be implemented cross-platform." [record §6.15]*

**11.7 A seed is the whole run.** The wiring, the weights and the exploration draws all come from the seed's stream, in every engine. The inputs do not: they come from a stream of their own, seeded separately (10.13), so that a seed shows the same patterns in the same order to any network however it was built.
*Byron and Cedric, standing; the input stream separated by Byron, September 14, 2026 [record §4.5, §7].*

**11.8 Agreement is proven on the configuration, not carried over.** The engines are compared on the container, the wiring, the drive and the read a run will actually use, before that run is read as evidence about anything.
*Claude's reading of two cases the record holds — a wave stamp that only bit a neuron left untouched for a whole epoch, and a neuron with no incoming synapses that one engine silenced where the others fired it — to be corrected in a word [record §5.2, §6.15, §7].*

**11.9 Checkpoints round-trip.** A checkpoint rebuilds the network from its seed and its settings and reloads what the run reached: weights, thresholds and floors, the clock, spike times, synapse stamps, signals in flight, each neuron's per-decision expectation of its own spike and its expected spikes and decision count, each synapse's open-arrival note, and whether the ISI factor was on. It round-trips in either engine, and a network saved under a setting resumes under it unless the resuming run overrides it explicitly.
*Byron and Cedric, standing; extended September 17, 2026 with the single-spike rule's state and the ISI factor [record §7, §0.2].*

**11.10 A rule is recomputed, a state is stored.** What a checkpoint can derive from the network's settings is derived — the escape scale is recomputed from the count, since it is a rule and not a state — and what the run moved is stored as the run left it, thresholds and floors included.
*[record §5.2, §7. This is what makes 11.9's round-trip decidable when a new field appears.]*

**11.11 A resume is a continuation, not the same run.** A network resumed from a checkpoint is rebuilt under its own layout and checked neuron for neuron and synapse for synapse against a fresh build from the seed, and it is refused if they differ. It is not the saved run to the bit: the exploration stream starts afresh (the driver's seed + 1,000,000), the reward baseline starts from the first resumed epoch, and the input stream is advanced to the epoch reached rather than restored.
*Claude's reading of the record's two senses of resume, to be corrected in a word: what a resume must reproduce is not decided [record §7, §8].*

**11.12 The network keeps living.** There is no training run and no evaluation run, only one run that keeps going. No rule may assume an end, no result is read as convergence, and a run is read as health.
*Byron and Cedric, standing. Its consequence for the data is 10.3: there is no held-out set, because there is no evaluation run to hold one out for [record §7, §6.14].*

---

## Appendix A — the constant register

**A.0 One value to a name, one home for every value.** Every constant the specification fixes has exactly one name and one home in the code, every row below names the section that owns it, and a literal of the same quantity anywhere else is a bug. The register is not a clause and carries no rules: it lists what the clauses fix.
*[record §1. `tests/test_constants.py` fails if any path grows a literal of its own.]*

*Section numbers below are the rewrite's, and are to be made exact when the sections are collated; the record citation is the value's provenance.*

**The network and its wiring**

| constant | value | what it fixes | owned by |
|---|---|---|---|
| GOO_COUNT | 60 | the neurons in a goo given no count | §4 [record §1.2] |
| GOO_SCALING_FACTOR | 0.05 | the scaled wiring's knob: every neuron hears N × s synapses in expectation | §4 [record §1.2, §3.4] |
| GOO_PROJECTION | 0.2 | ff2-partial's P: the probability on each input-to-output pair; at P = 1 it is ff2 to the bit | §4 [record §1.2, §3.4] |
| WEIGHT_RANGE | [−1, 1] | the range a weight is drawn uniformly from at the build, and clipped to by learning | §2 [record §1.1] |

**The neuron, its clock and its firing**

| constant | value | what it fixes | owned by |
|---|---|---|---|
| TAU | ∞ | the potential does not leak: the evidence accumulator. 2 ms is the leak, kept as a per-run option with its analysis | §1 [record §1.2, §5.1; Byron, September 17, 2026: "neuron: NOT leaky, but please leave the leak option with its analysis"] |
| REFRACTORY | 5 ms | the absolute refractory period | §1 [record §1.2] |
| REFRACTORY_HOPS | 2 | the refractory period in hops, so a hop is 2.5 ms | §3 [record §1.2; Byron, September 17, 2026: "Please set hops=2 by default"] |
| INTERVAL | 35 ms | the epoch's length: the spacing of inputs when no time is given | §3 [record §1.2, §4.2] |
| THRESHOLD | 0.25 | the starting θ of a container that does not set its own, quoted at THRESHOLD_FAN_IN | §6 [record §1.2, §5.2] |
| GOO_THRESHOLD | 0.2 | goo's starting θ, quoted the same way | §6 [record §1.2] |
| THRESHOLD_FAN_IN | 18 | the in-degree θ and the floor are quoted at: a container that scales starts neuron j at θ · d_j / 18 | §6 [record §5.2] |
| MINIMUM_POTENTIAL | −1 | the floor on the potential, quoted at THRESHOLD_FAN_IN and rescaled with θ, holding the floor at −4 θ | §6 [record §1.2, §5.2] |
| GOO_MINIMUM_POTENTIAL | −0.8 | goo's floor, at the same ratio to goo's threshold | §6 [record §1.2] |
| ESCAPE_DELTA | 0.455 | the width of the firing decision, in units of the neuron's starting threshold | §6 [record §1.2, §5.2] |
| ESCAPE_REFERENCE_COUNT | 60 | the count the width is quoted at: every hazard runs at √(60/N) | §6 [record §1.2, §5.2] |

**The drive and the read**

| constant | value | what it fixes | owned by |
|---|---|---|---|
| INPUT_DRIVE | rate | how a bit becomes spikes: an independent Poisson process drives each input neuron across the epoch | §5 [record §1.2, §4.3] |
| INPUT_CV | 0.6 | how that drive is specified: the coefficient of variation of the train it produces, from which the rate follows | §5 [record §1.2, §4.3] |
| INPUT_RATE, INPUT_RATE_OFF | 0.133, 0 /ms | the same drive in the other coordinate: the rates of a bit-1 and a bit-0 neuron's process | §5 [record §1.2] |
| TEACHER_THRESHOLD | 14.3 Hz | the count read's line where a read is scored as bits (the row critic); mnist's critic reads the counts themselves (10.8) | §5 [record §1.2, §4.3] |
| POPULATION | 3 | the neurons per class in each half of a complement-coded output zone (10.6) | §5 [record §1.3] |

**Learning**

| constant | value | what it fixes | owned by |
|---|---|---|---|
| LR | 0.03 | the learning rate where the problem names none (mnist names 0.002, 10.11) | §8 [record §1.3] |
| ELIGIBILITY | hazard | what the reward acts on: hazard, the score of the escape decision, or hebb, the single-spike rule | §8 [record §1.3; Claude's reading of the default, to be corrected in a word] |
| CRITIC | row | how the reward is judged where the problem names none: row, graded or evidence (mnist names evidence, 10.9) | §8 [record §1.3] |
| TEMPERATURE | 2 | the evidence critic's T; open — "We will have to sweep for temperature eventually" | §10.9 [record §1.3, §8] |
| BASELINE_RATE | 0.05 | the per-epoch update of the running reward baseline the advantage is taken against | §8 [record §1.3] |
| DECISION_MEMORY | 10⁻⁴ | the per-decision update of a neuron's expectation of its own spike; a starting value, to be swept | §8 [record §1.3, §6.7] |
| TARGET_ISI | 5.1 ms | the interspike interval the ISI factor pays most for; hardcoded for now | §0.2 [record §1.3; Byron: "The desired ISI is 5.1 ms (hardcode for now)"] |
| ISI_FACTOR | on | weigh every charge of the single-spike rule by the ISI factor | §0.2 [record §1.3; Byron: "On by default"] |
| RATE_MEMORY | 0.01 | the per-epoch update of a neuron's own observed rate, which homeostasis, un-sticking and the stuck bands read | §1 [record §1.3] |
| QUASH_RATE | 0.02 | **non-default.** The fraction of its weight a refire weakens each contributing synapse by, when a run asks for the quash | §8 [record §1.3, §6.11] |
| QUASH_K | 0.2 /ms | how fast that quash falls off with the delay since the previous spike | §8 [record §1.3] |
| HOMEOSTASIS | 10⁻⁶ | **non-default.** The per-epoch rate a threshold drifts toward its target firing rate; zero for mnist (10.12) | §8 [record §1.3] |
| TARGET_RATE | 0.5 | the firing rate homeostasis aims for | §8 [record §1.3] |
| UNSTICK | 10⁻³ | **non-default.** The per-epoch rate a stuck neuron's threshold moves toward its target; zero for mnist (10.12) | §8 [record §1.3] |
| UNSTICK_TARGET | 0.5 | the firing rate the un-sticking aims for | §8 [record §1.3] |

**The problem, and what is reported**

| constant | value | what it fixes | owned by |
|---|---|---|---|
| mnist: side, block | 28, 2 | the image as distributed, and the block averaged to one bit | 10.4 |
| mnist: bits, classes | 196, 10 | the bits an image becomes, and the classes | 10.4, 10.6 |
| mnist: binarisation | 0.5 | a block is on iff its mean is at least half of full | 10.4 |
| mnist: clock | 3 | the clock neurons at the front of the input zone | 10.5 |
| mnist: inputs, outputs | 395, 60 | the two zone widths | 10.5, 10.6 |
| mnist: hidden neurons | 199 | the hidden count unless the run says otherwise | 10.7 |
| mnist: lr | 0.002 | the problem's learning rate | 10.11 |
| mnist: homeostasis, unstick | 0, 0 | both off for this problem | 10.12 |
| STUCK_BELOW, STUCK_ABOVE | 0.01, 0.99 | reporting: a neuron whose observed rate is outside the band is read as stuck | 10.15 [record §1.3] |
| WINDOW | 200 | reporting: the epochs the reported moving average of the fraction right spans | 10.15 [record §1.3] |

**Constants no clause yet claims,** named here so they are decided rather than lost: ACROSS (8 — the zone width a goo takes when no problem names one; mnist names its own); WEIGHT_EPSILON (0.001 — under `--positive-weights` the weight range becomes [ε, 1], a network with no inhibition); RULE (the record's four rules are one now that only the reinforce rule survives); TARGET (the record's default is the reversed pattern, whose problem is dropped; mnist's target is the label, 10.10); LATE (what a signal arriving after its target fired earns — it does not apply under either surviving eligibility).

