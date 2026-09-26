# The authority

This file specifies walnutbutter. The code follows it: where the two
disagree, this file is right and the code has a bug. A change to the system
is made here first, then carried into the code and its tests.

Byron and Cedric own the substance of it. Claude keeps it and the code in
step. This file is copyright (C) 2026 Byron Shock and Cedric Shock and
licensed CC BY-SA 4.0 (`LICENSE-CC-BY-SA-4.0.txt`); the code it specifies
is under the AGPL, version 3 or later (`LICENSE`).

**This file replaced the lab notebook on September 17, 2026.** What stood
here before — every measurement, every quotation, every sweep — is
`RECORD.md`, kept verbatim at the tag `lab-notebook-2026-09-17`. What belongs
here is the specification alone:

- one rule to a clause, each clause numbered, dated and attributed;
- no sweep results, no findings, no arguments — those stay in the record, and
  a record entry names the clauses it measured;
- every clause implemented in every engine that runs it, and tested before it
  is built;
- a clause that is not yet decided says so, marked **Open**, in Byron's words
  where they exist, and names what is open.

Where a clause states a reading of an instruction rather than the instruction
itself, it says whose reading it is and that it is to be corrected in a word.
The questions this file leaves for Byron and Cedric are collected in
`docs/authority-claude-notes.md`; the drafting itself is in
`docs/authority-draft.md` and `docs/authority-draft-checks.md`.

*One caution about reading this file beside the record.* The record's
measurements were not made at this file's defaults, and the gap is wider than
one constant. Every measurement in `RECORD.md` was made at TAU 2 ms and a hop
of 1.667 ms — three hops to a refractory period — where this specification
starts from the evidence accumulator and a hop of 2.55 ms (§2.2, §3.2): half
again as long, and two hops to a refractory period and a hair rather than
three inside one, so a return that refired a neuron there clears its wall
here. Every mnist measurement in it was made at a 100 ms epoch
and at a container threshold of 0.6 with the floor at −2.4, where this
specification's defaults are INTERVAL 35 ms (§3.11) and GOO_THRESHOLD 0.2
with the floor at −0.8 (§4.10). Those three are not the problem's to set
(§11.14): they come from the constants or from the run that is asked for, so
a run that names none of them is a configuration the record has not measured.
Read a number from the record only with the configuration it was taken at.
*Byron, September 17, 2026, 14:56 MDT [`docs/rewrite-answers.md` §2] and
17:14 MDT [`docs/rewrite-answers-2.md` §20].*

## Contents

- [0. Standing values — non-negotiable](#0-standing-values--non-negotiable)
- [1. The synapse](#1-the-synapse)
- [2. The neuron](#2-the-neuron)
- [3. The clock](#3-the-clock)
- [4. The container and its wiring](#4-the-container-and-its-wiring)
- [5. The drive and the read](#5-the-drive-and-the-read)
- [6. Firing](#6-firing)
- [7. Exploration](#7-exploration)
- [8. The learning rule](#8-the-learning-rule)
- [9. The teacher](#9-the-teacher)
  - [9.1 At most one rule pays at the read](#91-at-most-one-rule-pays-at-the-read)
  - [9.2 The order at the read](#92-the-order-at-the-read)
  - [9.3 The baseline and the advantage](#93-the-baseline-and-the-advantage)
  - [9.4 The evidence critic](#94-the-evidence-critic)
  - [9.5 The row critic](#95-the-row-critic)
  - [9.6 The graded critic](#96-the-graded-critic)
  - [9.7 The class critic, and the fraction right](#97-the-class-critic-and-the-fraction-right)
  - [9.8 The teacher's book: each neuron's firing-rate memory](#98-the-teachers-book-each-neurons-firing-rate-memory)
  - [9.9 Homeostasis — off unless a run asks](#99-homeostasis--off-unless-a-run-asks)
  - [9.10 Un-sticking — off unless a run asks](#910-un-sticking--off-unless-a-run-asks)
  - [9.11 What is reported](#911-what-is-reported)
  - [9.12 The estimator's correlation with a supervised direction](#912-the-estimators-correlation-with-a-supervised-direction)
  - [9.13 The rate teacher — specified, not built](#913-the-rate-teacher--specified-not-built)
- [10. Local rules](#10-local-rules)
  - [10.1 The quash — off unless a run asks](#101-the-quash--off-unless-a-run-asks)
  - [10.2 The quash is the only local rule carried](#102-the-quash-is-the-only-local-rule-carried)
- [11. The problem](#11-the-problem)
- [12. The invariants](#12-the-invariants)
- [Appendix A — the constant register](#appendix-a--the-constant-register)
- [Appendix B — building the engine](#appendix-b--building-the-engine)

## 0. Standing values — non-negotiable

*These are values, not mechanisms: no clause of this file may contradict one, and where a clause and a value disagree the value is right and the clause is to be rewritten.*

**0.1 What a value binds, and what a direction binds.** Every clause of this file states a rule the system implements today. A value marked **a direction** states where Byron and Cedric intend the system to go: it is not implemented, no clause is written from it, and it overrides no clause in force. Where the system as it runs and a direction disagree, this file states the system as it runs.
*Byron, September 17, 2026, 14:53 MDT, keeping θ rather than deprecating it in the rewrite: "We are going to keep the level. The system must implement what I was just using. We can make changes later to this." [`docs/rewrite-answers.md` §1]*

**0.2 The inputs are the outputs.** An input neuron, a hidden neuron and an output neuron are defined only with respect to where external connections are; the neurons themselves operate identically. No neuron gets a different rule of integration, of firing or of learning by its index, its zone or its label. What a zone decides is where the drive reaches, where the read counts, and what a wiring lets project onto what (§4, §5) — that is what a zone is for. *Engines:* a neuron's index selects which zone it is in and fixes the order draws are taken in, and carries nothing else.
*Byron and Cedric, from the beginning [RECORD §0]. Restated by Byron, September 14, 2026: "All neurons are first-class citizens of the population." [RECORD §4.3, §8]*

**0.3 A neuron integrates delta functions, fires when the integral reaches its threshold, and resets.** A signal is an instant: it adds its weight to the potential of the neuron it reaches and has no width and no shape. This is an integrate-and-fire neuron. The value is that the level exists and that the spike resets the potential; how soft the level is belongs to §6. The mechanism is §2.2 and §2.4.
*Byron and Cedric, from the beginning [RECORD §0].*

**0.4 A neuron that fires is then absolutely refractory, and that is a computational feature.** While refractory it ignores its inputs and does not integrate them. This is a feedback control mechanism and a computational feature of the system. The mechanism is §2.5.
*Byron and Cedric, from the beginning [RECORD §0, §5.3]. That no rule may work around it is Claude's reading of "computational feature", confirmed by Byron, September 17, 2026.*

**0.5 Structures are permissive; nothing is restricted artificially.** A neuron may sit in several input and output zones at once. Where the system refuses a configuration, the refusal is a clause of this file with a reason, never a convenience of the code. A threshold goes wherever homeostasis and un-sticking take it and nothing clips it; the range a weight is drawn from and clipped back into is a rule of its own (§1.4), not a tidying.
*Byron and Cedric [RECORD §2]. Byron, September 14, 2026, striking the threshold clamp: "Please eliminate threshold clipping. It's artificial." [RECORD §3.4; §1.3's struck THRESHOLD_RANGE row]*

**0.6 The goal is the substance, not a task.** A problem exists only as a way of watching whether the substance is alive and learning; no rule is chosen because it scores well on one.
*Byron and Cedric, standing since the project began [RECORD §2]. The record defines walnut butter as a substance spread on the plane, and the plane's containers were archived as code on September 17, 2026 (§4.1). The plane is not what the substance is: Byron, the same day, "It is not THE idea, but AN idea of how to make neural networks useful to makers" — and "I can use the containers we have. But eventually people will need a way of interacting with them. The substance will still be spread on the plane. It's much easier to work with that way once it's stable." So goo carries the substance for now, and the plane stands as one candidate way of putting it into a maker's hands; **deferred**, and no clause is written from it.*

**0.7 As few knobs as possible, and a default regime that is stable.** Where a quantity must change with the size or the shape of a network, the change is built into the rule rather than left to a sweep.
*Byron, September 16, 2026: "A design principle of this project is that it should have as few knobs as possible and by default operate in or near a stable regime. Therefore scaling the network MUST reduce the probability of escape noise at each neuron by sqrt(N). I want to build this into the rule." [RECORD §5.2] The instance he gave it for is the firing hazard's scaling by the network's count.*

**0.8 The network keeps living.** There is no training run and no evaluation run, only one run that keeps going. No rule may assume an end, and no result is read as convergence: a run is read as health, and drift is normal.
*Byron and Cedric, standing [RECORD §6.14, §7: "The network runs forever: these are read as health, not convergence, and drift is normal."]*

**0.9 One authority, every platform.** Every engine is built from this file alone. A rule stated here is implemented in every engine that runs it, and an engine that lacks a rule says so and refuses — it never approximates. Engines are compared with `==`, with one tolerance, named in one place and hidden nowhere (§12.4). A seed is the whole run, and a checkpoint round-trips: both are invariants, and both are §12's.
*Byron, September 14, 2026: "rules in authority.md must be implemented cross-platform" [RECORD §7]. Byron, September 17, 2026, 15:02 MDT, keeping the array engine: "keep, with tolerances, as a foundational rule; all platforms are built from one clear authority."*

**0.10 Learning is paid by one global scalar.** One number scores what the network did, a baseline is subtracted from it, and every synapse's update is proportional to what is left. No rule in force hands a neuron an error of its own.
*The frame is Byron and Cedric's, from the beginning [RECORD §0]. The word *reward* is kept in this file for a dopamine-mediated reward, which is the direction and not the mechanism; what a critic produces is the epoch's **reinforcement** (§8.1). **A direction:** that the scalar be produced locally at a spike and consumed globally — dopamine — leaves this file as a mechanism and stays as an intention, Cedric's to take up. Byron, September 17, 2026, 15:02 MDT: "(1) drop but keep the idea of dopamine reinforcement around. It's a very important direction for Cedric's work, but the implementation was mine and I lacked understanding." [RECORD §6.2–§6.6 is the attempt, and stays there.]*

**0.11 The synapse is the learner — specified September 24, 2026, being built.** Specified from Byron's mechanism of September 23, 2026 and his decisions of the 24th and the 25th: §6.13, §7.5–§7.9, §8.16–§8.17 and 5.4b are its clauses, and `docs/synapse-engine-plan-2026-09-23.md` is how the engines are brought to them; until they are, those clauses are specified and not built and §6.5 runs, and once they are §6.5 stays the default — a run asking for exploration at the synapse by its EXPLORATION setting (§7.1) — until all three engines agree on mnist's configuration and mnist's learning rate has been re-found (§7.6). What the mechanism bears out of this direction, and what it does not, is in those clauses: the synapse explores, one transmission at a time, on its source's potential; what its exploration credits is the synapses into its source, one hop upstream, and not itself, because $V_{\text{threshold}}$ is the neuron's (Byron, September 23, 2026); and the neuron keeps its threshold, its deterministic spike and its refractory period as properties of the channel. The sparse-matrix form stays a direction. What follows is the direction as it stood on September 17, kept as the record of where the clauses came from.

Today the neuron is the learner: one firing decision per neuron per wave, and a synapse is credited by what it happened to have in the potential when its neuron took a chance the synapse did not take. Every synapse into one neuron shares that neuron's credit and expectation and is told apart only by its own trace (§8.4), so what per-synapse resolution the rule has is carried by the trace alone. Where the project is going is the other way round: the synapse explores, the synapse is credited for what it explored, and the neuron is the medium they speak through. Nothing in this file is written from it until Byron specifies it, and specifying it is a rethink of the system rather than a change to a clause. It stands in §0 rather than among the intended changes of §0.12 because it is where the work goes next — Byron, September 17, 2026: "This is the direction we are going with our next research spurt" — and it is issue [#15](https://github.com/byronshock/walnutbutter/issues/15).

Named so that the scale of it is not lost: it would re-ask §0.10 (what one global scalar pays, when each synapse has a score of its own), the whole of §8, the learning rate (the injected noise gains a dimension per synapse rather than per neuron), and what a synapse holds, which is today four pieces and would then include its own draw and its own expectation of its own outcome. What it would *not* re-ask is what a neuron is: the threshold, the escape noise and the refractory period become properties of the channel rather than of the learner, which is what §0.4 already calls them.

The form it is meant to take is one sparse matrix of every synaptic connection, updated in parallel — which the array engine already half is (§12.3). What makes that awkward today is that the decision is the neuron's while the state is the synapse's, so a wave gathers into per-neuron potentials and then broadcasts the neuron's credit back out; under the inversion every quantity is per synapse and the update is elementwise, with no gather and no broadcast. Two things would have to be restated with it: the draw, which becomes one uniform per synapse per wave rather than one per neuron (§3.8), so the one-stream-one-order invariant of §12.6 governs far more draws and holds the engines to far more; and the clock, which is a queue of irregular times (§3.3) where a parallel form wants regular ones.
*Byron, September 17, 2026, 06:05 MDT: "Computationally exploration noise can be generated at the synapses, and semantically this is clean: a SYNAPSE explores its own impulse response, rather than a NEURON exploring its impulse response!" And the same evening, stating it whole: "imagine, if you will, a whole bunch of synapses at work trying to figure out what their impulse response is through neural goo. The synapses are the learners, and the neurons are their communication system" — a future refactor, not for now, "because it will require a whole rethink of how the system works." Its computational form is his too: "Eventually we build a huge sparse matrix with all the synaptic connections and benefit from massively parallel computations." [RECORD §0.3, "Not specified or built yet"]. The record places this at the top at Byron's instruction and expressly not among §0's rules; it stands here as a direction and not as a rule — Claude's reading, and Byron's placing of it, September 17, 2026: it stays among the values rather than among §0.12's parked intentions because it is the one being taken up.*

**0.12 The changes intended, and not made.** These are named here and nowhere else in this file; no clause is written from one, and when one is taken up it is specified whole, from §0 down.
- ~~Exploration generated at the synapse (§0.11)~~ — taken up September 24, 2026 (§7.5). What was named with it is not: θ stays, the neuron's spike being the comparison under the mechanism (§6.13), and a hazard set by the fan-out is a run option and not the rule (§7.7). *Byron, September 17, 2026: "Theta should also be deprecated" — and, the same afternoon, 14:53 MDT, that the rewrite states the system as he has been running it, θ and its fan-in scaling included. [the quoted words are at `docs/rewrite-outline.md` §6.2, attributed to Byron there; RECORD §0.3 for "θ ignored"; `docs/rewrite-answers.md` §1 for the 14:53 counter]*
- A dopamine reward produced locally at a spike (§0.10).
- A deterministic drive: one spike every REFRACTORY + LAG on each driven neuron in place of the Poisson rate drive. The interval was quoted in TARGET_ISI, which §7.4 took out with the shaping function, and is quoted in the hop since — Byron, September 18, 2026: "REFRACTORY+LAG". *Byron, September 17, 2026, 14:58 MDT: "Please include the Poisson drive as it ran." [`docs/rewrite-answers.md` §3]*
- A signal's travel time drawn rather than fixed. §3.2 gives every connection the one hop $h = (\text{REFRACTORY} + \text{LAG})/2 = 2.55$ ms, and the arithmetic that keeps a two-hop return off a refractory boundary (§2.5, §6.12) is that one number's; a drawn hop makes each of those statements a probability, takes "spikes per hop" away from the hazard as its unit (§6.5, §7.2), and adds a consumer to the streams a seed fixes (§7.3, §12.6). Which stream it draws from, and whether a draw may fall below REFRACTORY and be dropped by §8.13, are open and Byron's. LAG stays at a very small value meanwhile. A signal still arrives whole and at an instant (§0.3): what a draw spreads is the population of signals, not the signal. *Byron, September 18, 2026, retiring TIME_CONSTANT_OF_TRANSMISSION as a name: "We keep the LAG at some very small value. Arrival times may eventually be stochastic."*
- Small-world shortcuts, to return. *Byron, September 17, 2026, 15:08 MDT: "We'll bring back small world shortcuts later."*
- The plane, as one way of putting the substance into a maker's hands rather than as a container of its own (§0.6). *Byron, September 17, 2026: "It is not THE idea, but AN idea of how to make neural networks useful to makers."*

---

## 1. The synapse

*A synapse is a one-way connection between two neurons; the read synapse of §7.9 is not one, being a device of the read with no id, no weight and none of the state below. It holds a weight,
which is what it delivers, and the state the learning rule reads — four
pieces under the evidence accumulator, five under the leak (§1.5). Where a
clause here says a target "integrated" a signal, §3.5 says what that means.*

**1.1 One-way, with its own weight.** Every connection runs one way, from a
source neuron $i$ to a target neuron $j$, and carries its own weight
$w_{ij}$: the amount delivered to $j$ when a signal travels it. Between two
neurons there are two connections, $i \to j$ and $j \to i$, with independent
weights; a pair is never one object. No neuron connects to itself.
*Byron and Cedric, with the scaffolding agreed September 10, 2026 [RECORD §3]. Byron's correction of the
verb, September 14, 2026: "Instead of 'connects' I should have said
'projects'. The connections are all one-way." Restated by him in the scaled
wiring rule, September 16, 2026, 01:18 MDT: "P(i projects onto j) = 0 if i
== j".*

**1.2 Ids from 1, in the order the connections are made.** Connections take
ids from 1 in the order they are made, so one seed rebuilds the same
network, connection for connection. The order the pairs are offered in
belongs to the wiring, not to the synapse.
*Byron and Cedric, September 10, 2026 [RECORD §3, §3.4].*

**1.3 The weight at build.** Every weight is drawn independently and
uniformly from WEIGHT_RANGE = $[-1, 1]$ from the network's seeded stream, at
the moment its connection is made — so the weights are drawn in
connection-id order, interleaved with whatever draws the wiring itself makes.
A network may instead be built with one fixed weight given to every
connection, and then no weight is drawn at all, so a fixed-weight network
and a random-weight one consume the seed's stream differently (§4.8).
*Byron and Cedric, September 10, 2026 [RECORD §1.1, §3.3].*

**1.4 Learning keeps a weight inside WEIGHT_RANGE.** Every rule that moves a
weight clips the result back into the range that network's weights were
drawn from: $w_{ij} \leftarrow \operatorname{clip}(w_{ij} + \Delta, \text{WEIGHT\_RANGE})$ — $[-1, 1]$ by default, and
$[\text{WEIGHT\_EPSILON}, 1]$ under positive weights, a network with no
inhibition. The range is both the prior the weights are drawn from and the
only bound on them; a weight that reaches an end stays there until a rule
moves it back. It is the only bound in the system: no rule clips a
threshold (§0.5).
*Byron and Cedric, September 10, 2026 [RECORD §3.3, §6.7]. Engines: the clip is part of each
update, not a tidying pass over the weights afterwards.*

**1.5 What a synapse carries.** Beside its weight a synapse carries its
trace $x_{ij}$ (§1.6) and, under the leak, the moment that trace was last
brought up to date; its stamp (§1.7); its note $B_{ij}$ (§1.8), which is the
accumulator's bookkeeping and is written by nothing under the leak; and its
score $e_{ij}$ (§1.9). Four pieces under the accumulator, five under the
leak, and nothing else per synapse is a rule of this file. Exploration at the synapse (§7.5) adds nothing to the list: a synapse's decision is drawn on its source's state, and the ventured mark of §7.9 rides on the signal in flight and not on the synapse. A checkpoint
carries all of it — the stamp, the trace with its moment, the note and the
score — so a checkpoint may be written anywhere and still round-trip, and a
run saved mid-epoch can be read afterwards for what its synapses had earned.
*The four are the single-spike rule's and the schedule's (Byron, September
17, 2026) [RECORD §4.4, §6.7, §7]. That the score is carried is Byron's,
September 17, 2026: "a checkpoint should hold a synapse's score for analysis
purposes." The clause used to leave it out and require a checkpoint to be
taken at a read; that requirement was a consequence of leaving it out, not a
rule, and goes with it.*

**1.6 The trace $x_{ij}$: what the synapse has in its target's potential.**
Under the evidence accumulator the trace is the count of the arrivals the
target integrated along this synapse since the target's last spike — an
integer, raised by one at each arrival integrated. It is the derivative of
the target's potential with respect to this weight, exactly while the floor
has not bitten: $V_j = \sum_i w_{ij} x_{ij}$, so $\partial V_j / \partial w_{ij} = x_{ij}$. It is cleared when the target spikes, when the floor
bites, and when a run's epoch reset discharges the potential (§3.9) — the
three events after which what the weights delivered is no longer the
potential. Under exploration at the synapse a run may instead have the trace count the ventured arrivals only, and it is then not the derivative (§8.17). Under the leak, which stays as a per-run option (TAU finite),
the same quantity is the leaked sum $\sum_a e^{-(t - t_a)/\text{TAU}}$
brought up to date lazily, which is why the synapse also keeps the moment
its trace was last brought up to date.
*Byron, September 17, 2026, stating the accumulator: "since we are keeping
weighted synapses the potential is a weighted count of evidence, and its
derivative delta functions of spike arrivals weighted by the evidence each
carries" [RECORD §5.1, §6.7].*

**1.7 The stamp: the last signal its target integrated.** Each connection
records the clock time of the last signal its target actually integrated
along it, and nothing if it has carried none. A signal delivered while the
target is refractory is integrated by nothing and does not move the stamp,
though it is still recorded as delivered. A ventured delivery (§7.9) moves the stamp as any signal does. It is what the quash reads (§10.1, a
non-default) to find the synapses that carried a cycle.
*Byron and Cedric, September 10, 2026 [RECORD §4.4, §6.4, §6.11].*

**1.8 The note $B_{ij}$: the debit its open arrivals carry.** Under the
evidence accumulator an arrival stays open until its target spikes, so as
each arrival is integrated the synapse adds $E_j$ to its note:
$B_{ij} \mathrel{+}= E_j$. $E_j$ is not a count of expected spikes: it is
the sum of the expectations posted at the target's decisions since its last
spike. Under exploration at the synapse the sum the note takes is the target's gain $G_j$ of §8.16 in place of $E_j$ and the credit, at each arrival the trace counts — under TRACE ventured a relayed arrival notes nothing and $B_{ij}$ stands (§8.17) — and the settle is then $e_{ij} \mathrel{+}= x_{ij}\,G_j - B_{ij}$: the credit is inside $G_j$, a net credit where $E_j$ is a debit, so the bracket below changes sign and no $c_j$ stands apart. The
note is what lets the spike settle every open arrival in one operation,
$e_{ij} \mathrel{+}= c_j x_{ij} - (x_{ij} E_j - B_{ij})$ (§8.11), and it
is cleared with the trace at that spike. Under the leak there is no note:
the score is posted per decision on the leaked trace (§8.12).
*Byron, September 17, 2026, on whether a debit may run past a read: "Let
them run! Epochs are for the convenience of teaching the network, and to
some extent an emergent property of the time constants, but they don't
really exist in nature." [RECORD §6.7.] Engines: an arrival is one operation
on the synapse and a decision one operation on the neuron — no loop over a
fan-in may run at a decision.*

**1.9 The score $e_{ij}$: what the synapse has earned since the last read.**
Each synapse carries the score the learning rule has posted to it since the
last read. At the read the debit accrued so far on the open arrivals is
settled into the score, the update is paid on it, and the score is cleared —
whether or not the advantage moved a weight. The arrivals stay open, their
debit counting from that moment, which the engines do by re-basing the note
to $B_{ij} = x_{ij} E_j$ as the score is cleared. This clause fixes the
moment: the score is cleared at the read, and since nothing is posted between
the read and the next epoch's first wave, an engine that clears it at that
epoch's reset instead is conforming.
*Byron, September 17, 2026 [RECORD §6.7]. The re-basing is Claude's
statement of the record's "the open arrivals counting from now", put to Byron
on September 17, 2026 and **deferred** by him; it stands to be corrected in a
word. Where the record says a synapse is **charged**, this
file says an entry is **posted** to its score: the accounting words are the
ones meant throughout, and *charge* is kept for electrons.*

---

*An idea, named here on September 23, 2026 and taken up as a run option on the 24th (§8.17). Byron and Cedric, September 23, 2026,
in Byron's words: "IDEA (Byron and Cedric discussed): What if the synapse
discounts spikes it passed on as signal with fidelity after V_pre exceeds
threshold? In other words, when the synapse fires deterministically, it does
not try to learn at all." It became the ventured trace of §8.17 on September 24, 2026 — a run option, not the default — and §1.6 carries it; the mechanism it belongs to is §7.5's (`docs/synaptic-escape-noise-2026-09-22.md` §7).*

## 2. The neuron

**2.1 What a neuron holds.** One of each, per neuron:

- its potential $V_j$, and the clock time that potential was last brought up to date;
- its threshold $\theta_j$ — the starting value its container gave it (§4.10), and wherever homeostasis or un-sticking have since taken it when a run turns those on (§9.9, §9.10) — and its floor $V^{\min}_j$, which is the starting value its container gave it and does not move;
- the width of its firing decision $\Delta_j$, the hazard's scaling by the network's count, and the moment its hazard has run from (§6.4, §6.5, §6.6);
- the time of its last spike $t^{\text{fired}}_j$ and of the spike before it; the number of spikes it has ever fired, and the number it had fired when the epoch began, whose difference is the count read (§5.10);
- whether it has fired in the epoch so far, and whether it carries 5.8's driven mark this epoch — forced by the drive, or delivered to by the charged drive (5.4b);
- its rate memory $r_j$ (§2.6);
- its per-decision expectation $\hat p_j$, the number of decisions it has made to date, $E_j$, and the credit its last decision left for the spike to settle (§8.6, §8.11);
- under exploration at the synapse (§7.5): its gain $G_j$ in place of $E_j$ and the credit (§8.16), and, if it is an output, its read synapse's ventured transmissions this epoch (§7.9).

Its refractory state is not held: it is read from the time of its last spike (§2.5). *Engines:* every engine carries all of these per neuron, in the same neuron order. A checkpoint carries the state — the potential and its clock, the threshold and the floor as the run reached them, $\Delta_j$, the spike times and counts, $r_j$, $\hat p_j$, $n$ and $E_j$ — and recomputes what is a rule rather than a state, the hazard's scaling by the count (§12.10); the epoch-local flags are set by the epoch's reset (§3.9).
*Claude's inventory of the state the clauses of this file require; the pieces are Byron's and Cedric's where the clauses that use them are [RECORD §0 notation, §5.1, §5.2, §6.7, §7].*

**2.2 The potential is an accumulator of evidence.** The potential is the sum of the weights of the signals the neuron has integrated since its last spike, undiminished: nothing decays. With $x_{ij}$ the number of arrivals $j$ integrated along the synapse $i \to j$ since its last spike,

$$V_j = \sum_i w_{ij}\,x_{ij}, \qquad \frac{\partial V_j}{\partial w_{ij}} = x_{ij},$$

exactly while the floor has not bitten (§6.2). Once inhibition has taken the potential to $V^{\min}_j$ the potential is the floor whatever the weights; the floor clears every open arrival, so the derivative identity resumes at the next arrival and $x_{ij}$ counts from the floor rather than from the last spike (§6.3). A signal that reaches a refractory neuron is not integrated and is not counted (§2.5).
*Byron, September 17, 2026, 14:56 MDT, choosing it for the rewrite: "neuron: NOT leaky, but please leave the leak option with its analysis. 2 hops." [`docs/rewrite-answers.md` §2] And his statement of it the same morning: "since we are keeping weighted synapses the potential is a weighted count of evidence, and its derivative delta functions of spike arrivals weighted by the evidence each carries." [RECORD §5.1]*

**2.3 The leak, as a per-run option.** A run may select a leaky potential instead of the accumulator: with the leak on, a signal of weight $w$ arriving at time $t$ first decays the potential for the time since it was last brought up to date, and is then added,

$$V \leftarrow V\,e^{-(t - t_{\text{last}})/\tau}, \qquad t_{\text{last}} \leftarrow t, \qquad V \leftarrow V + w,$$

with $\tau$ = TAU = 2 ms. The accumulator of §2.2 is the default and is $\tau = \infty$; a checkpoint carries which of the two its network ran under, and a resumed network keeps it. *Engines:* under the accumulator no engine evaluates a decay anywhere — Byron, September 17, 2026: "we DO need to skip the exponential decay calculation when we select evidence-accumulator since it's just going to slow things down" — and the skip must change no result, the factors it skips being exactly 1.
*Byron, September 17, 2026, 14:56 MDT: "please leave the leak option with its analysis." The analysis stays with it in the record and no part of it is a clause [RECORD §1.2, §5.1, §8].*

**2.4 What a spike does to the potential.** A spike resets the potential to zero and is remembered, along with the spike before it:

$$V \leftarrow 0, \qquad t_{\text{prev}} \leftarrow t^{\text{fired}}, \qquad t^{\text{fired}} \leftarrow t,$$

and the neuron's spike count rises by one. Nothing any synapse delivered is still in the potential, so every incoming synapse's trace $x_{ij}$ goes to zero with it. Under the accumulator the spike is the moment the learning rule settles every open arrival; under the leak the entries were posted at each decision and the spike only clears the traces (§8.11, §8.12). The neuron is then refractory (§2.5).
*Byron and Cedric, from the beginning [RECORD §0, §5.2]. The memory of the spike before last is the record's, stated there without a date; the quash reads it (§10.1).*

**2.5 The refractory period.** A neuron that fired at $t^{\text{fired}}$ is refractory while $t < t^{\text{fired}} + \text{REFRACTORY}$, REFRACTORY = 5 ms. While refractory it ignores every signal, does not integrate it, cannot be forced by the drive, and makes no firing decision. This clause and §6.12's resumption of the hazard are the whole of what reads $t^{\text{fired}}$: since §7.4 there is no third reader under the neuron rule (under exploration at the synapse the synapses' exposure runs from the spike as well, §7.5 and §7.8, and §6.12's resumption does not arise), and the interval between a neuron's own spikes is read by the quash alone (§10.1, a non-default). What a refractory neuron does about firing is §6.12; what it does about a signal that reaches it is §8.13.

The period is shorter than two hops by LAG (§3.2), so a neuron's own spike — which reaches it again only round a cycle, and §4.4 gives it none shorter than two connections — arrives just after that neuron recovers and never as it recovers. A one-hop arrival falls inside the period and is §8.13's. No arrival is on the boundary to begin with. *Engines:* the comparison allows the clock's slack all the same (§3.4) — the test is `now + slack(now) < t_fired + REFRACTORY` — because the times being compared are each the end of a chain of clock arithmetic and neither is exact; the slack is what makes the three engines recover the same neuron at the same wave rather than a hair either side of it.
*Byron and Cedric, from the beginning [RECORD §0, §5.3]. The clause read, until September 18, 2026, that without the slack "floating-point arithmetic would decide whether a two-hop return refires the neuron" — which is exactly the case the LAG removes, and exactly what the 2.5 ms hop the code still runs does not: there two hops land on the wall and the slack decides. Byron, September 19, 2026, fixing the hop this file had stated as a whole REFRACTORY + LAG: "The current dynamics should be 2.55 ms hops, with 2 hops landing at 5.1 ms after the refractory period they are responsible for ends at 5 ms." The 5.1 ms is two hops, and §3.2 is written that way since.*

**2.6 The rate memory.** After every epoch, each neuron that carries no driven mark (5.8) for that epoch moves its rate memory toward what it did:

$$r_j \leftarrow r_j + \text{RATE\_MEMORY}\,(y - r_j), \qquad y = 1 \text{ if it fired in the epoch, } 0 \text{ if it did not},$$

RATE_MEMORY = 0.01, about the last hundred epochs; $r_j$ starts at 0.5, which is the code's start (`neuron.py`) and is not stated in the record. A neuron forced in the epoch is left alone, its firing that epoch saying nothing about the network, and so is one the charged drive delivered to (5.4b): both carry 5.8's driven mark, and the mark is what this reads. Under exploration at the synapse *fired* means a spike of the neuron's own: an output's read-synapse escapes (§7.9) count toward its read and not toward $r_j$. $r_j$ is what homeostasis and un-sticking read when a run turns them on, and what a report means by a stuck neuron (the learning and reporting clauses).
*The pre-alpha's teacher kept $r_j$ as part of its rule [RECORD §6.7]; the constant is [RECORD §1.3], which carries no date for it. The exemption of a forced neuron is the one homeostasis and un-sticking already follow. The teacher moves it once an epoch, at the point §9.2 gives it. The charged input's exemption and the spikes-only reading are Byron's, September 25, 2026 (`docs/synapse-build-map-2026-09-24.md` §5, Q9, Q7).*

## 3. The clock

*One run has one clock, and every time in this file is a time on it.*

**3.1 Time is in nominal milliseconds.** The clock runs on from the first
input and is never restarted; an epoch is a boundary on it, not a reset of
it.
*Byron and Cedric, September 10, 2026 [RECORD §4.1].*

**3.2 The hop.** A signal generated at time $t$ is delivered at
$t + h$, one hop later, and no whole number of hops ever falls at
$t + \text{REFRACTORY}$. The hop is HOP, and it is derived: wherever this file
needs the arithmetic it is written out,

$$2h = \text{REFRACTORY} + \text{LAG} = 5.1\ \text{ms},\qquad h = \text{HOP} = 2.55\ \text{ms},$$

where $h$ is this file's shorthand for HOP in arithmetic — REFRACTORY and LAG
are the two constants it is made of, and two hops are what they add to.
LAG $= 0.1$ ms, and there is no other delay: the time in signalling is
carried by the hop alone. It is a delay and not a decay — a signal arrives
whole, one hop after it was sent (§0.3).

**Two hops are longer than the refractory period, and that is the content of
the LAG.** No whole number of hops is a refractory period. A signal that
has travelled $k$ connections is delivered $k\,h$ after the spike that
generated it, and a neuron that fired in that same wave is recovered
REFRACTORY after, so the arrival falls $k\,h - \text{REFRACTORY}$ past that
neuron's wall — $-2.45$ ms at one hop, 0.1 ms at two, 2.65 ms at three, and
never zero. In general that clearance is
$(k-2)\,\text{REFRACTORY}/2 + k\,\text{LAG}/2$, which is LAG itself at
$k = 2$ and is zero for no whole $k$ at all: $k\,h = \text{REFRACTORY}$ would
need $k = 2\,\text{REFRACTORY}/(\text{REFRACTORY} + \text{LAG}) = 1.96\ldots$,
which is not one. One hop lands inside the period, and what becomes of a signal
that arrives there is §8.13's; two hops is the tight one. A neuron's own spike reaches
it again only round a cycle, and §4.4 gives
it no cycle shorter than two connections, so its own earliest return is at
$2h = 5.1$ ms, 0.1 ms past its wall — the tight case above, and the one the LAG
is for.

The two-hop return is what the LAG is for. Without it the hop would be
REFRACTORY / 2 and that return would land exactly at the end of the neuron's
own refractory period,
and the clock's slack (§3.4) — a tolerance for rounding, not a rule about
neurons — would decide whether the signal was taken or dropped. With it, no
delivery is ever settled by which side of a rounding error it fell on. What
§2.5 and §6.12 say about arrivals and refires is this arithmetic and no
other.
*The hop is Byron's, September 11, 2026. Fixed directly by Byron, September
17, 2026: "I no longer want to specify REFRACTORY_HOPS. I want to specify $h$
directly as TIME_CONSTANT_OF_TRANSMISSION." The first half of that stands: the
hop is specified directly and not as a count of refractory periods. The name
did not — Byron retired TIME_CONSTANT_OF_TRANSMISSION on September 18, 2026,
"REFRACTORY+LAG", leaving the quantity, its two constants and its 5.1 ms
untouched; only the name went — and a name came back on September 19, 2026,
when the quantity was halved to 2.55 ms and became HOP, Byron: "Let's be
consistent and call it a HOP." It was
REFRACTORY / REFRACTORY_HOPS with REFRACTORY_HOPS 2 until then, and Byron on
why that form went rather than being renamed: "refractory_hops was a nice
convenience when we were working on an integer hex grid." There a trip had a
length in cells, so counting the refractory period in hops measured how far a
spike could travel before its neuron recovered. Goo has no distance (§4.2),
every projection is one hop, and the count has nothing to count [RECORD §1.2,
§4.1]. The 5.1 ms is REFRACTORY + LAG, which is two hops and not one: a delay, and
not an interval any rule aims at. TARGET_ISI, in the form hardcoded at 5.1 ms until September 17, 2026
— not the derived HOPS × (REFRACTORY + LAG) = 10.2 ms that briefly replaced it
— was the same number, and Byron replaced it with REFRACTORY + LAG on
September 18, 2026: a name retired in favour of the arithmetic, not two
meanings merged. Nothing in force aims at this interval.

The LAG has a floor as well as a ceiling. It must stay far above the clock's
slack (§3.4) — a LAG within slack(t) puts the delivery back inside the very
tolerance it was introduced to escape — and far below REFRACTORY, so two hops
stay a refractory period and a hair, and the hazard's per-hop unit (§6.5) does
not move. At TOLERANCE $10^{-12}$, 0.1 ms clears the slack by a factor of
$10^{11}$ at a millisecond of clock time and by ten at $10^{10}$ ms, and §0.8
says the network never stops: the margin is the reason for the value and not a
coincidence of it. Byron, September 18, 2026: "We keep the LAG at some very
small value."*

**3.3 A wave is everything at one time.** The queue is a time-ordered
schedule of signals and stimuli, not a per-hop loop, and a wave is the batch
at the front of it sharing one time — the signals arriving, and an input
stamped for that moment. A cascade from one input may still be running when
the next input's signals join the schedule: the epoch is not a barrier, and
"the queue empties, then the clock advances" does not hold.
*Decided September 11, 2026 [RECORD §4.1].*

**3.4 Two moments within the clock's tolerance are one moment.** With
$\text{slack}(t) = \text{TOLERANCE} \times \max(1, |t|)$ and TOLERANCE =
$10^{-12}$, two events whose times differ by no more than the slack of the
earlier are in the same wave, and every comparison of two moments — the end
of a refractory period included — allows the same slack. Where a wave holds
an input or any other external event, that event's exact time is the wave's
time; otherwise the earliest event's time is. A hop is rarely representable
exactly, so without that anchor a chain of hops drifts off the input clock.
*[RECORD §4.1, `clock.py`.] The record states this as "times are rounded to
a nanosecond"; the rule as built is the relative slack above, and the two
are not the same quantity — the clock is in nominal milliseconds, so the
slack is $10^{-12}$ ms — a femtosecond — up to a millisecond of clock time,
and grows in proportion after. Byron, September 17, 2026, confirming the
relative rule and setting the constant at $10^{-12}$: it is some 4,500 units
in the last place of a double at every $t$, enough to absorb the rounding of
any chain of clock arithmetic, while $10^{-9}$ — 4.5 million — would have
grown to a whole hop after about a month of clock time, and §0.8 says the
network never stops. *What this requires of an engine:* the slack must exceed
the rounding accumulated along the longest chain of times it builds, and that
is to be checked when the clock is built rather than assumed. Engines: same tolerance and same anchor, or two
engines put the same two signals in different waves.*

**3.5 A wave has two phases: deliver, then fire.** First **deliver**: under
the charged drive the wave's external deliveries come first (5.4b), each
adding its charge to an input that is not refractory and touching that input
as a signal touches its target (Byron, September 25, 2026, Q10), in the order the wave holds them, as §3.6 takes its forced neurons; then every
signal of the wave delivers its weight to its target; a target that is not
refractory integrates it, which stamps the synapse (§1.7) and moves its
trace (§1.6), and a refractory target integrates nothing. Then every neuron
the wave touched settles — its floor applies to the wave's total, the charges
included, so no result depends on the order the signals arrived in. Then **fire**: every
forced neuron fires unless it is refractory, then every neuron that is not
refractory decides, whether the wave touched it or not, and each neuron that
fires schedules its outgoing signals for $t + h$ — its active outgoing
connections, a connection's active flag being carried by every engine. No
decision in a wave sees a spike of that wave, since a wave's spikes arrive
one hop later. A neuron may fire in any number of waves, the refractory
period permitting, which is what makes the count read (§5.10) and more than
one decision an epoch possible.
*Byron and Cedric [RECORD §4.4]. What a decision is belongs to the firing
clauses (the escape noise, RECORD §5.2); that every neuron is asked, and not
only the touched, is what lets a neuron nobody talks to spike.*

**3.6 The order the fire phase asks in.** The forced neurons first, in the
order the wave holds them; then the neurons the wave touched, in the order
they were touched; then every other neuron, in index order. The order is a
rule, not an implementation's choice: the order neurons fire in is the order
their signals are pushed onto the schedule, and push order is the order a
later wave sums them in (§3.7). Under exploration at the synapse the wave's
escapes (§7.5) are pushed after all of its spikes' signals, in edge order —
sources in index order, each source's outgoing synapses in the order the
topology was built (§12.5) — so a spike's signals and a ventured one due at
the same later moment are summed spikes first.
*Claude's reading of what the three engines do and of what §3.7 requires of
them. Byron left the call to Claude, September 17, 2026 — "your choice" — and
it is kept as a rule because §12.4 compares two engines with `==`, which is
undecidable unless the order neurons fire in is fixed [RECORD §4.4, §6.15].
The escapes' place is Byron's, September 25, 2026 (`docs/synapse-build-map-2026-09-24.md` §5, Q11).*

**3.7 A wave's deliveries are summed in push order.** Signals due at one
moment are summed in the order they were pushed onto the schedule, so an
engine that flattens the topology differently sums a wave differently and
lands on different bits. The object engine and the Rust loop sum in push
order; the array engine sums a wave in matrix order, and what that costs is
the tolerance of §12.4.
*Claude's reading of what the three engines do. Byron left the call to
Claude, September 17, 2026 — "your choice" — and it is kept as a rule for the
reason §3.6 gives
[RECORD §6.15].*

**3.8 The wave's draws are taken between the phases.** Where the network's
firing decision is a draw — ESCAPE_DELTA positive (§6.9) — one uniform is taken per
neuron, in neuron order, from the run's exploration stream (§7.3), after the
floor has settled and before any neuron fires. Every neuron takes a draw
whether it can use it or not, so the stream's position depends only on the
count of neurons and the count of waves. Under the deterministic rule the
width is not positive and nothing is drawn at any wave. Under exploration at the synapse (§7.5) the draws are one uniform per synapse, in edge order (§12.5) and an inactive connection's included (§3.5), then one per output neuron for its read synapse (§7.9), in output order — at the same place in the wave, and whether or not each can be used, so the stream's position again depends only on counts. What an inactive synapse does with its draw is not specified, and under exploration at the synapse a network holding an inactive connection is refused (§12.2).
*Byron decided the escape noise, September 15, 2026, and its scaling by the
count on September 16 [RECORD §5.2]. That the draw is one uniform per neuron
per wave, in neuron order, is the record's statement of what the engines do
[RECORD §5.2, §6.15]. What the draw decides is §6.7's. Engines: a draw that
is not used is still taken. Byron, September 25, 2026, that an inactive synapse keeps its draw and is refused until its part is specified (`docs/synapse-build-map-2026-09-24.md` §5, Q12).*

**3.9 An epoch.** An epoch is one input and the schedule run to the
**horizon**, $t_e + \text{INTERVAL}$. In order: (1) **reset** — every
neuron's fired-this-epoch and forced-this-epoch state is cleared; potentials,
spike times, spike counts, traces, stamps and signals in flight are all
kept, so the refractory period outlives the epoch boundary, and each open
arrival's debit counts from the read just paid (§1.9), which is what re-bases
the notes. A run may instead ask that the reset **discharge** the potentials,
zeroing them and settling what is open on every synapse with no credit
(§8.11); the discharge stays in the specification as a
non-default — Byron, September 17, 2026: "--discharge stays as a
non-default" — so §8.11's arm that settles it with no credit stands; (2) **input** — the epoch's pattern is placed on the
input neurons at the times the drive gives it; (3) **run** — the schedule
runs wave by wave to the horizon; (4) **report** — the output is read and,
under the reinforce rule, scored and paid.
*Byron and Cedric [RECORD §4.2]. The record's third step, a per-epoch
exploration nudge of every potential, goes with the perturb eligibility;
what a wave draws is §3.8.*

**3.10 Signals due at or after the horizon wait.** They stay on the schedule
and join the next epoch's waves. An input may not be given a time before the
horizon the schedule has already run to.
*Byron and Cedric, September 10, 2026 [RECORD §4.2].*

**3.11 INTERVAL.** INTERVAL is 35 ms: the epoch's length, and the spacing of
inputs when no time is given — the first input at 0 and each next one
INTERVAL after the last. No problem sets it, so the epoch's length lives in
one place; a run may set it for the whole of that run.
*Byron, September 14, 2026, deciding it on the epoch sweep [RECORD §1.2,
§4.2]; the record states the decision, it does not quote him.*

## 4. The container and its wiring

One container survives, goo, with three wirings of it; a wiring decides only what projects onto what, and the container hands each neuron the potential axis its fan-in earns.

**4.1 Goo is the only container.** The system builds one kind of network: goo. The hex grid, the hexagonal columns, the lattice and a spread leave the specification, and with them the positions they needed, the hex metric, the guaranteed neighbourhood, the small-world shortcuts and the constants OMEGA, REACH and ROWS. Nothing is deleted: they stay in `RECORD.md` §2–§3 and in the code at the tag `lab-notebook-2026-09-17`, which is what archived means here, and the rebuild does not carry them. Small-world shortcuts are named here as an intention to return, not as a clause, and no rule below may assume them. [RECORD §3.2, §3.4]
*Byron, September 17, 2026, 15:08 MDT: "archive the hex containers, lattice, spread, the whole shebang. We'll bring back small world shortcuts later." [RECORD §2, §3.4]*

**4.2 What goo is.** Goo is $N$ neurons with no positions at all. No distance between two of them is defined, so there is no neighbourhood, no near and no far, and nothing for a shortcut to get past. What connects to what is decided by a probabilistic rule over ordered pairs (4.5–4.7) and by nothing else. The count is GOO_COUNT = 60 unless a run gives another, and the zone widths (4.3) set the smallest count a goo can be built at.
*Byron, September 14, 2026, asking for a fourth container and saying what it was — fully connected goo — and, the same day, setting its count: "We will speed everything up by selecting 60 units of goo". [RECORD §2, §3.4]*

**4.3 Zones by index.** With no positions there is no bottom row to be the input, so the zones go by index: the **first** `across` neurons in index order are the input zone and the **last** `outputs` are the output zone, `outputs` being `across` unless the problem gives its own two widths. A neuron in neither zone is a hidden neuron. Both widths must be at least one, and the count must leave room for both (4.5–4.7). The zones are addressed by place: `get_neuron_at(place, 1)` is the input zone and `get_neuron_at(place, 0)` the output zone, and goo reports `rows` = 2 because it is counting those two zones and not any depth. Being in a zone says where a neuron is driven, where it is read and what may project onto it (4.5–4.7); no rule of a neuron's own integration, firing or learning branches on its index or its zone (§0.2). The zones may not overlap, whatever the wiring: the count must be at least the two widths together, and a goo that would overlap them is refused. Structures are permissive — a neuron may sit in several input and output zones at once — and every wiring here needs its zones disjoint; whether the permissive reading returns is open and Byron's. How the input zone's width is made up — the clock neurons at its front, then the coded bits — is §5.2 and §5.3's.
*Zones by index are Claude's reading of what "fully connected goo" has to mean to be buildable, September 14, 2026, written down here so that changing it is editing a working system; the two widths are from mnist, September 15, 2026. The permissive rule is the last line of the record's §2 — "A neuron may sit in several input and output zones at once; structures are permissive, never artificially restricted" — carried into this file as §0.5. [RECORD §2, §3.4]*

**4.4 A projection is an ordered pair.** Every projection is one-way, from a source to a target, with its own weight; between two neurons there may be two, $i \to j$ and $j \to i$, each with an independent weight, and no neuron projects onto itself. A wiring is a rule that gives $P(i \to j)$ for every ordered pair of neurons. The pairs are taken in $(i, j)$ order — source index, then target index — and the projections made are numbered from 1 in that order, which is the order every engine reads them in.
*The ordered-pair rule is the record's standing rule for a connection, from the scaffolding agreed September 10, 2026 [RECORD §3]. Byron corrected the verb on September 14, 2026: "Instead of 'connects' I should have said 'projects'. The connections are all one-way." The zone rule he gave that day is quoted at 4.5, where the scaled rule that replaced it lives. [RECORD §3, §3.4]*

**4.5 The scaled rule — the default wiring.** With $s$ = GOO_SCALING_FACTOR, $I$ and $O$ the zones' widths:

$$P(i \to j) = \begin{cases}
0 & i = j \\
0 & i,\ j \text{ both in the input zone} \\
0 & i \text{ in the output zone},\ j \text{ in either zone} \\
\min\!\big(1,\ Ns / A_j\big) & \text{otherwise}
\end{cases}$$

where $A_j$ is the sources $j$ may hear under the three zeros above: the $N - I - O$ hidden neurons for an input neuron, the $N - O$ inputs and hidden neurons for an output neuron, the $N - 1$ others for a hidden neuron. So every neuron with sources enough hears $Ns$ synapses in expectation and the density is the same at any count; where $Ns$ exceeds $A_j$ the probability stops at 1 and the neuron hears $A_j$ of them. An output projects onto hidden neurons alone; an input hears hidden neurons alone; with no hidden neurons the goo is inputs onto outputs and nothing else, and every input hears nothing (4.11). The zones may not overlap (4.3). GOO_SCALING_FACTOR = 0.05.
*Byron, September 16, 2026, 01:18 MDT: "P(i projects onto j) = 0 if i == j; 0 if i and j are both in the input zone; P_ij necessary to give j an average of N * scaling_factor inputs. Please default scaling_factor to 0.05." The third line is Byron the same night, 03:05 MDT: "I would like the outputs kept apart from one another again. With hidden=0 we have no cycles, eliminate interference from other output neurons, a two-layer feedforward network" — that an output also projects onto no input neuron is Claude's reading of "no cycles" and "feedforward", and is still to be corrected in a word: Byron's answer of September 17, 2026 was expressly for the feedforward wirings (4.6, 4.7) and does not reach the scaled rule, where an output does project onto hidden neurons. [RECORD §3.4]*

**4.6 ff2 — fully connected, feedforward.**

$$P(i \to j) = \begin{cases} 1 & i \text{ in the input zone},\ j \text{ in the output zone} \\ 0 & \text{otherwise} \end{cases}$$

Every input projects onto every output and nothing else projects at all. Nothing about the topology is drawn: the count and the two widths fix it. It wires two layers and no more — a goo with hidden neurons is refused under it, so the count must be exactly the two widths together. How the two layers generalise to the zones of a recurrent goo is Byron's to say and is not built.

**An output neuron projects nowhere here**, and that is the point of the wiring rather than an accident of it: it has the fan-in its inputs give it, its spikes are read (§5.10), and nothing further is done with them. Byron, September 17, 2026: "For the feedforward mnist problem, output neurons do not project. They have the required fan-in at their inputs, but we read the spikes directly and do nothing further with them. This is a desirable optimization for the feedforward problem only." *Of the engines:* an output neuron has no outgoing connections to hold and its spike schedules nothing, so an engine gives it none rather than an empty list it walks. This is particular to the two feedforward wirings and is not a rule about output neurons; under the scaled rule (4.5) an output projects onto hidden neurons.
*Byron, September 16, 2026, 16:35 MDT: "Please make a fully-connected-feedforward-2 rule: There are two 'layers', the input 'layer' and the output 'layer'. P_ij = P (i is in the input layer and j is in the output layer). These 'layers' generalize to zones in a recurrent goo, but for the feedforward problem I want everything to connect fully." [RECORD §3.4]*

**4.7 ff2-partial — the same two layers at a probability.** With $P$ = GOO_PROJECTION, which must be in $(0, 1]$:

$$P(i \to j) = \begin{cases} P & i \text{ in the input zone},\ j \text{ in the output zone} \\ 0 & \text{otherwise} \end{cases}$$

each input-to-output pair its own draw, nothing else projecting — so an output projects nowhere here either, with the optimization 4.6 names — two layers and no hidden neurons as under ff2, of which it is the general form: at $P = 1$ it is ff2 to the bit — the same projections, the same weights, the same stream — and every engine must satisfy that exactly. An output hears $IP$ inputs in expectation. GOO_PROJECTION = 0.2 — the value the constant inherited from a wiring that has left the specification, not one chosen for this wiring; the value for ff2-partial is **open** and Byron's [RECORD §1.2, §8].
*Byron, September 17, 2026, about 02:40 MDT: "I'd like to define a partly connected topology that is otherwise identical (2 layers), where P(neuron i projects onto neuron j) = P iff i in inputs, j in outputs". [RECORD §3.4]*

**4.8 The draw on the seed's stream.** Nothing is drawn where the probability is 0: no projection is made and no number is taken. Where it is 1, the projection is made without a draw. Where it is strictly between, one uniform from the network's seeded stream decides it, and the projection is made if the uniform is below the probability. A weight follows a projection that is made — one uniform on WEIGHT_RANGE, unless the run fixes every weight instead — so a pair is taken projection first and then weight, pair by pair in $(i, j)$ order (4.4), and nothing else of the topology is drawn. *What this requires of the engines:* the wiring is drawn once, in the container, and the array engine and the Rust loop take the projections and weights already made, in the container's order; no engine draws a topology of its own, so all three carry the same wiring and the same starting weights to the bit, with none of the tolerance the array engine is allowed on continuous quantities elsewhere.
*Byron, September 14, 2026, on the first probabilistic wiring; the order within a pair and the stream's ownership are Claude's reading of it, unchanged since. [RECORD §3.3, §3.4]*

**4.9 A wiring that was drawn needs its seed to be rebuilt.** Where any pair's probability was strictly between 0 and 1, the seed that drew the wiring is part of the network's identity: without it the goo cannot be rebuilt, and a checkpoint of one that has no seed is refused rather than rebuilt differently. Where no probability was strictly between 0 and 1 — ff2, or a scaled goo whose every probability is 0 or 1 — the count, the two widths and the wiring fix the topology, and it rebuilds with no seed at all. A checkpoint carries the count, the two widths, the wiring, its knob ($s$ or $P$) and the seed, and restores under the wiring it was built with, never under whatever the default has since become. *What this requires of the engines:* whether a wiring drew anything is decided from the wiring and its own knob, not from a stored probability the wiring never reads — an ff2 checkpoint needs no seed, and an engine that refuses one for want of a seed has a bug.
*Claude's reading of Byron's probabilistic wiring of September 14, 2026, written down so that changing it is editing a working system; the restore-under-its-own-wiring rule is Claude's, September 15–16, 2026, as each new wiring was added. [RECORD §3.4]*

**4.10 The potential axis scales with fan-in.** The container gives neuron $j$, of in-degree $d_j$, its starting threshold and its floor:

$$\theta_j = \text{GOO\_THRESHOLD} \cdot \frac{d_j}{F}, \qquad
V^{\min}_j = \text{GOO\_MINIMUM\_POTENTIAL} \cdot \frac{d_j}{F}, \qquad
F = \text{THRESHOLD\_FAN\_IN} = 18,$$

with GOO_THRESHOLD = 0.2 and the floor at $-4 \times$ GOO_THRESHOLD, so GOO_MINIMUM_POTENTIAL = −0.8 and follows the threshold if that constant moves: the ratio is the rule and the number is its value. Both points of the axis are rescaled by the same factor [RECORD §5.2]. The scaling is applied after the wiring, since it reads the in-degree; it is a per-run switch, on for goo unless a run turns it off, and a checkpoint restores under the setting it was saved with. These are **starting** values only: homeostasis and un-sticking (non-defaults) move a threshold from where it starts, the firing section quotes the escape width in units of the starting threshold, and a checkpoint stores the threshold and floor a run actually reached rather than recomputing them.
**The container's numbers are rough, for now.** GOO_THRESHOLD and the floor that follows it are not settled: the record's mnist runs were made at other values, for the reason the record gives, and the pair has never been swept on its own. Byron, September 17, 2026: the container "may be a little rough around the edges. For now." A run that takes these defaults is taking a starting point and not a decision. [RECORD §8]
*Byron, September 14, 2026, choosing the threshold out of the three readings the record named, and the floor with it; goo's own two numbers date from the same day. That the rule is linear is Claude's reading — that is what "scales with fan-in" says without further instruction — decided for now, and a different slope, or a threshold quoted against something other than the fan-in, is Byron's to call. $F = 18$ is the in-degree the numbers were first quoted at, an interior hex cell's two rings, and stays the unit although that container has left the specification (4.1). [RECORD §5.2]*

**4.11 A neuron that hears nothing.** A neuron with no incoming synapses is left at the container's quoted threshold and floor, scale 1 — there is nothing to scale by — so it keeps a positive threshold and the width that is quoted against it: it fires at the rest rate of the escape hazard like any other neuron, and by its drive if it is an input neuron, and never otherwise. This is a rule about the scaling alone; a threshold given as 0 outright still means the deterministic comparison (the firing section). All three engines are held to it alike.
*Claude's resolution, September 16, 2026, of the question the outputs-apart rule (4.5) opened by leaving every input neuron of a two-layer goo hearing nothing; confirmed by Byron, September 17, 2026. [RECORD §5.2]*

## 5. The drive and the read

*What reaches the network from outside, and what is read back out of it.
Everything between the two belongs to the neuron (§2), the clock (§3) and
the rule (§8).*

**5.1 The input zone is the input; the output zone is the output.** The
drive of 5.4 reaches the neurons of the input zone and no others; the read
of 5.10 counts the neurons of the output zone and no others. A zone is
where the external connections are and nothing more: per §0.1 the neurons
themselves are identical, and no rule may give a neuron a role beyond
membership of a zone. A problem names both widths (§11); a container says
which neurons a zone holds (§4).
*The Strong Statement, non-negotiable [RECORD §0], which §4.3 applies; the record neither dates it nor attributes it jointly, and neither does this clause.*

**5.2 An input is complement-coded, and coded no other way.** A pattern is
$k$ raw bits followed by their negations, $2k$ coded bits onto $2k$ input
neurons, so exactly half of the complement-coded bits are driven whatever
the raw bits are. Place $i$ of the input zone shows place $i$ of the coded
pattern of 5.3: there is no permutation. The pattern presented is the pattern the read is scored
against — nothing corrupts an input on the way in.
*Byron, September 14, 2026, defining the copy task [RECORD §8]: "An input is
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
drives each input neuron across the presentation window of 5.4a: at
INPUT_RATE where its coded bit is 1, at INPUT_RATE_OFF where it is 0. The
arrivals of one neuron are drawn $\mathrm{Exp}(\lambda)$ apart from the
network's own seeded stream, starting at the epoch's moment $t_e$, until a
draw falls at or past $t_e + \text{PRESENTATION\_TIME}$ — that last arrival is
drawn and discarded. The neurons are drawn
place by place in place order, and a place whose rate is zero draws
nothing. The whole epoch's arrivals are drawn before the schedule runs, so
the epoch's arrivals are fixed before any of its decisions, which draw from the exploration stream (§7.3).
*Byron, September 13, 2026: "I want to have an option to present the inputs
as a probabilistic firing rate rather than presenting them all at once and
seeing what happens." The default from September 14, 2026, and the drive
this specification states, Byron, September 17, 2026, 14:58 MDT: "Please
include the Poisson drive as it ran" [RECORD §4.3].*
*Of the engines:* all three take these draws from the one stream in this
order, so a seed gives every engine the same arrivals; an engine that
generates its own would agree on nothing else.

**5.4a The presentation window.** The drive runs from the epoch's moment to
$t_e + \text{PRESENTATION\_TIME}$ and not after it. PRESENTATION_TIME $=$
INTERVAL, so by default the window is the whole epoch and the drive is 5.4's
unchanged; a run that shortens it presents the input over the front of the
epoch and leaves the rest to the network. The window is the drive's alone: it
reaches the input zone, which is where the drive reaches (§5.1), and the clock
neurons of 5.3 with it, they being input neurons and no rule of this file
giving a neuron a role its zone does not (§0.1). PRESENTATION_TIME greater
than INTERVAL is **refused** and not clipped (§0.5): a window past the horizon
would schedule arrivals into an epoch that has already been read (§3.10).

*What the tail is, and what it is not.* After the window the input zone is
undriven, which is not the same as silent: every neuron decides at every wave
(§6.8) and an undriven neuron carries the hazard's rest rate (§7.2) — or, under exploration at the synapse, its synapses' whisper (§7.6) — so the
tail is the network's own doing and the drive's cascade finishing, not a gap.
What the read measures is therefore the epoch's count as it always was
(§5.10), over a whole epoch of which only the front was presented — the
denominator stays INTERVAL and does not follow the window.

*What it collides with, and is not settled.* §5.8 marks a neuron the drive
fired **this epoch**, and §8.1 skips every synapse into a neuron so marked
because "its firing was not the network's doing". That reason holds over the
window and not over the tail: a neuron driven in the front of the epoch stays
marked through a tail in which its firing *is* the network's doing, so a short
window makes the rule refuse to learn from the very part of the epoch the
window creates. Either the mark narrows to the window and §8.1 skips only what
the drive caused, or §8.1's skip stays the epoch's and the tail is unlearnable
for every driven neuron. This is Byron's, and it blocks a short window rather
than the clause.

*Byron, September 18, 2026: "Please add a new parameter, PRESENTATION_TIME.
From the beginning of the epoch to PRESENTATION_TIME, input zone neurons are
driven; from PRESENTATION_TIME to the end of the epoch they are not." The
default is INTERVAL because that is the only value under which nothing already
measured moves; what a shorter window is for, and what it should be, are open
and Byron's. **Not yet built:** the code drives the whole epoch, and this
clause is one the engines do not meet (`docs/conformance-checks.md`).*

**5.4b The charged drive — a run option, specified September 24, 2026.** A run may name the drive **charged**. The arrivals are drawn as 5.4 draws them, from the network's stream and in the same order, at DRIVE_STEPS times the rate, and each delivers $\theta_i/\text{DRIVE\_STEPS}$ to the input neuron's potential in place of forcing a spike — the threshold the input holds when the delivery lands, divided by DRIVE_STEPS in that form and at that moment. DRIVE_STEPS is 3 unless the run names another: it is a run option, a count of deliveries and so a whole number of at least 1 — a run that names any other value is refused (§12.2) — recorded in the run's record and carried by its checkpoints (§12.9). The input reaches its threshold about as often as it was fired before — from rest after DRIVE_STEPS deliveries, or after one more where their sum falls a rounding short of the threshold, which is accepted as it falls — and spikes by the comparison (§6.13), its potential readable by its synapses between deliveries (§7.5). A wave takes its external deliveries first, then its signals, so a charge due at the moment of a signal is integrated before it (§3.5). Nothing is driven at the epoch's moment (5.7), a delivery to a refractory neuron is dropped (§2.5), and the deliveries are external — they come along no synapse and count on no trace. A charged input carries the driven mark of 5.8 as a forced one does — set at its first delivery of the epoch, a delivery dropped at a refractory input included (it is still delivered, §1.7), and read as forced wherever the mark is read: 8.1 skips its incoming synapses at the read, and the rate memory (§2.6) and §9.2's steps (3) and (4) pass over it. The charged drive runs under exploration at the synapse only: under the neuron rule of §6.5 a charged input would fire by its own hazard on the potential the charges build (§6.8, §5.13), not by the comparison this clause's arithmetic is written on, and a run that names both is refused (§12.2). The Poisson forced drive of 5.4 stays the default.
*The second toy's "potential" drive (`docs/synaptic-noise-toy2.py`; `docs/synaptic-escape-noise-2026-09-22.md` §7.11), under which an input's synapses whisper the pattern and a ventured trace has something to learn from, where a forced input sits at zero between its spikes and its synapses whisper as an off-input's do. Byron, September 24, 2026, carrying it as a run option and keeping the mark. §0.12's deterministic drive is a separate intention and stands. Byron, September 25, 2026 (`docs/synapse-build-map-2026-09-24.md` §5): the amount at delivery and in that form, a further delivery accepted, the charge before the wave's signals (Q10); DRIVE_STEPS a run option (Q16); the mark set by any delivery (Q8) and read as forced by §2.6 and §9.2 as by 8.1 (Q9); the charged drive refused under the neuron rule (Q18). Specified September 24, 2026; the engines are brought to it under `docs/synapse-engine-plan-2026-09-23.md`, and until one carries it a run that asks for it is refused (§12.2).*

**5.5 These are arrivals, not spikes.** An arrival is a mandated spike and
not a signal: it adds nothing to the potential, and a neuron that takes one
spikes as it spikes from any other cause. An arrival landing while its
neuron is refractory is dropped, so a driven neuron fires at the first
arrival after its refractory period ends and its spike train is, within the
presentation window of 5.4a, a renewal process with dead time,
$$\text{ISI} = \text{REFRACTORY} + \mathrm{Exp}(\lambda).$$
Past the window there are no arrivals and the train is the neuron's own
(§7.2); at the default PRESENTATION_TIME the window is the epoch and there is
no past.
Two arrivals at one moment fire one spike. An arrival that fires the neuron
resets its potential as any spike does (§2.4), so under the accumulator the
evidence gathered since that neuron's last spike is discharged with it. A driven neuron fires in its
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
The rate and the CV are one axis and not two, so fixing one fixes the other
[RECORD §4.3].
*Decided by Byron, September 14, 2026 [RECORD §4.3]; the record states the
decision, and has no quoted words of his for it.*
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
than carrying a pending arrival across the boundary [RECORD §4.3].

**5.8 A driven spike is marked as driven.** A neuron the drive fired this
epoch carries a mark for the epoch, cleared at the epoch's reset, and so does
an input the charged drive delivered to this epoch, from its first delivery, one
dropped at a refractory input included (5.4b). Nothing in this section reads
it: the learning rule does, and what it does is §8.1's — the update **skips**
every synapse whose target was driven this epoch, its firing not having been
the network's doing — and so do the rate memory (§2.6) and §9.2's steps (3)
and (4), which pass over a marked neuron (Byron, September 25, 2026;
`docs/synapse-build-map-2026-09-24.md` §5, Q8, Q9). An input neuron the drive
neither fired nor delivered to this epoch is an ordinary neuron and is paid
like one.
*The mark is the drive's and has been there since forced drive [RECORD
§4.3]; that it is stated here, with its use in §8.1, is Claude's reading, put to
Byron on September 17, 2026 and **deferred** by him; it stands to be corrected
in a word [RECORD §6.7].*

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
whatever the network is built like [RECORD §4.5]. The shuffle's seed is the
run's `--input-seed` where one is given and the run's own seed otherwise, and
it indexes the shuffle alone, never the network's build. What turns a
dataset's images into raw bits is the problem's (§11.4).

**5.10 The read is the count read.** The read of an output neuron is
$n_j$, the spikes it fired since the epoch's reset — a count, and nothing
else. **It is the default read**, and the one every rule of this file is
written on: where a clause says what is read, it means this. A rate follows
from it and the epoch's length where a rule wants one,
$1000\,n_j/\text{INTERVAL}$ in hertz, and what counts as a neuron being **on**
belongs to the critic that asks (§9.5). Under exploration at the synapse the count is the output's spikes plus its read synapse's ventured transmissions this epoch (§7.9), and every clause that reads a count reads that sum. It is the one place the escapes are summed in: what reads whether a neuron fired — the rate memory of §2.6, and through it the stuck counts (§9.8, §9.11) — reads the neuron's own spikes, and the non-default reads below are refused under exploration at the synapse until a clause says what each reads (§12.2). *Byron, September 25, 2026 (`docs/synapse-build-map-2026-09-24.md` §5, Q7).*

*The other reads are non-defaults, and a run asks for each by name* (§9.1's
rule for a rule that does not run unless asked, applied to a read). They are
**fired**, on if the neuron spiked at all this epoch; **again**, on if it
spiked after the epoch's input moment; **window**, on if it spiked within
READ_WINDOW of the read; and **rate**, the neuron's own exponential firing-rate
estimate over RATE_TAU, scored against RATE_ON for a 1 and RATE_OFF for a 0
and clipped to $[0, 1]$ — the only read that is graded rather than a count.
None of them is quoted by any other clause, no result of this project rests on
one, and a run that names one is outside what the rest of this file describes.
*Byron, September 19, 2026, keeping them: "5.10 should say count is the
default read. Keep the rest."*
*Byron, September 14, 2026: "Here's how we actually score: COUNT the number
of times each neuron fired in the epoch. ESTIMATE the firing rate based on
the count. If the firing rate estimate exceeds TEACHER_THRESHOLD, the
output neuron is 1. Otherwise it is zero." The line is his the same day, to
mean one spike at a 35 ms epoch and not for the score it yields [RECORD
§4.3, §1.2].*
*Of the engines:* each snapshots its spike counts at the epoch's reset, so
that $n_j$ is the spikes since that snapshot and the read is an integer in
every engine — no tolerance applies to it, and the three agree on it exactly.
The rate line this clause used to carry became the row critic's pickiness
(§9.5) on September 17, 2026, which is a count and not a rate, so the read no
longer changes meaning with the epoch's length [RECORD §1.2, §4.3, §8].
This clause read "It is the only read" until September 19, 2026, when Byron
kept the other four rather than strike them; what changed is the clause and
not the code, which had offered five all along and defaulted to **fired** —
that default is now the count's, which is the part of the disagreement that
was the code's [`docs/conformance-checks.md` C1].

**5.11 The output zone is complement-coded.** For $C$ classes and a
population of $P$ neurons in each half — $P$ fire-if-one and $P$
fire-if-zero a class — the zone is $2CP$ neurons: the $C$
fire-if-one populations first, class $k$ owning the $P$ neurons from $kP$,
and then, in the same order, the $C$ fire-if-zero populations — the
ordering the input zone uses for bits and their negations, so the two zones
read alike. The problem names $C$ and $P$ (§11). The label code the **row critic**
scores against (§9.5) is the label's fire-if-one population on and its
fire-if-zero population off, and every other class the other way round;
mnist's own critic reads the class evidence of 5.12 and no code.
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
2026), confirmed by Byron, September 17, 2026 [RECORD §8, decision 6].*
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
become one deterministic spike every REFRACTORY + LAG on each driven neuron,
specified but not built, and a later change to 5.4 through 5.7 rather than
a rule now in force. The interval was quoted in TARGET_ISI, which §7.4 took
out, and is quoted in the hop since (§0.12, §3.2). Note what 5.7 costs it:
no neuron may be driven at the epoch's moment, so a deterministic drive gives
each driven neuron its own phase and not one lattice for the network.*

## 6. Firing

*Firing is escape noise: at every wave each neuron that is not refractory makes a stochastic decision on its own margin above the threshold — under the neuron rule. Under exploration at the synapse (§6.13, §7.5) the escape noise is the synapses' and the neuron's decision is the comparison: 6.4 to 6.8 then describe the neuron rule, which is what runs under EXPLORATION neuron — the default until §7.6's conditions are met, and a run's to name after them (§7.1, §7.6) — and where one of them says "the neuron's decision", "one uniform per neuron" or "every decision posts an entry", the synapse clauses govern instead (§3.8, §7.5, §8.16). This section states that decision as the working runs made it. Byron, September 17, 2026, 14:50 MDT, choosing it: "first we must tackle neural spike escape noise as a potential mechanism… we have a learning rule derived for it right now that is working right now"; and at 14:53 MDT, on the level it is measured from: "We are going to keep the level. The system must implement what I was just using. We can make changes later to this."*

**6.1 The axis a neuron decides on is the one its container gave it.** The
threshold $\theta_j$ and the floor $V^{\min}_j$ a neuron faces are the
container's, scaled by its fan-in (§4.10); THRESHOLD_FAN_IN $= 18$ is the
in-degree they are quoted at, THRESHOLD and MINIMUM_POTENTIAL are that
quotation for a container that sets no pair of its own, and goo's are
GOO_THRESHOLD $= 0.2$ and GOO_MINIMUM_POTENTIAL $= -0.8$, at the ratio of
$-4$. Goo scales, so under this specification every neuron's axis is scaled
by $d_j/F$ unless a run turns the scaling off; a neuron that hears nothing
keeps the quoted pair at scale 1 and so keeps a width (§4.11, §6.4). These
are **starting** values only: where a run turns on homeostasis or un-sticking
(§9.9, §9.10, non-defaults) a threshold moves from where it starts, and a
checkpoint stores the thresholds and floors a run actually reached rather
than recomputing them.
*The axis and its scaling are §4.10's, and the provenance is there. [RECORD
§5.2, §1.2]*

**6.2 The floor applies once a wave, to the wave's total.** After every signal of a wave has been delivered and before any neuron decides, each neuron touched this wave takes $V_j \leftarrow \max(V_j,\, V^{\min}_j)$, so the floor acts on the wave's summed input and the result does not depend on the order the signals arrived in. Of the engines this requires one floor per touched neuron per wave, not one per arrival.
*The floor is the container's of §4.10. Byron and Cedric, from the beginning, for the floor itself [RECORD §5.1]; that it applies once a wave, on the wave's summed input, is the record's wave of September 10, 2026 — Claude's reading of it, confirmed by Byron, September 17, 2026. [RECORD §4.4, §5.1]*

**6.3 A neuron at the floor has nothing of any synapse left in its potential.** When the floor bites, every synapse's open arrivals on that neuron are closed with no credit — the debit they have accrued is settled and their traces cleared — so a neuron held down by inhibition accumulates nothing on its synapses (§8.11; under the leak the traces are zeroed instead). A forced spike (§6.10) and a discharge (§3.9) settle the same way.
*Claude's derivation of September 15 and 17, 2026, on Byron's decision to derive the rule; confirmed by Byron, September 17, 2026. [RECORD §6.7]*

**6.4 The margin, and the width of the decision.** Neuron $j$ decides at time $t$ on its margin

$$s = V_j(t) - \theta_j(t),$$

the potential after the wave's signals and after the floor, against the threshold it faces. The decision's width is

$$\Delta_j = \Delta\,\theta_j^{\text{start}}, \qquad \Delta = \text{ESCAPE\_DELTA} = 0.455,$$

quoted in units of the threshold the container gave the neuron, so §6.1's scaling applies to the width as to the rest of the axis, a neuron of any fan-in is as soft as any other, and the width stays put when homeostasis later moves $\theta_j$. A network's widths are set once the thresholds are what the container gave them — after fan-in scaling, before anything moves them.
$\theta_j$ is constant through a run except as homeostasis and un-sticking move it between epochs (§9.9, §9.10, non-defaults); no rule of this specification lowers a threshold within a run. Under the leak $V_j(t)$ in the margin is the potential decayed to the decision's moment (§2.3). *Of the engines:* each neuron's width is stored in a checkpoint and restored, never recomputed from the threshold the run has since reached — unlike §6.6's count factor.
*Byron, September 15, 2026, choosing the stochastic decision: "Let's please go with (3). It is the most like what I want to do. Make the boredom stochastic and it is Williams's unit outright." The value 0.455 is Byron's word, September 15, 2026. [RECORD §5.2, §1.2]*

**6.5 The hazard and the chance of a spike.** A neuron that is not refractory fires at a wave with probability

$$P_j(t) = 1 - e^{-m_j(t)}, \qquad
m_j(t) = \frac{\Delta t}{\text{hop}}\,\kappa(N)\;e^{\,s/\Delta_j},$$

where hop is the signal's travel time of §3.2, $\kappa(N)$ the count's factor of §6.6, and $\Delta t$ the time elapsed since the neuron's previous decision, or since its refractory period ended if it was refractory then. The neuron carries a hazard of $\kappa(N)\,e^{s/\Delta_j}$ spikes per hop: at the reference count one expected spike per hop at threshold, $e$ times more per $\Delta_j$ of margin above it and $e$ times fewer per $\Delta_j$ below. $m$ is the number of spikes expected over the interval and $P$ the chance of at least one; the hazard runs on elapsed time rather than being tossed per wave [RECORD §5.2].

**$\Delta t$ is never negative.** The refractory test of §2.5 allows the clock's slack, so a neuron counts as recovered up to that slack before its exposure formally begins; asked to decide in that sliver it has been exposed to nothing, and $\Delta t$ is zero. This is a rule of the decision and not an engine's convenience: $m$ is also the expectation posted to every incoming synapse at a silent decision (§8.4), so a negative $m$ would put a wrong-signed entry across the whole fan-in, once per spike and always the same way.

Of the engines: $m$ is capped at $10^3$ — beyond which $P$ is 1 to the last bit — and $P$ is computed as $-\text{expm1}(-m)$, in that form, so the three engines agree and so that a small $m$ keeps its precision.
*Byron, September 15, 2026, as §6.4, and September 17, 2026 confirming the rule and promoting the clamp on $\Delta t$ out of the engine note. The cap and the expm1 form are Claude's, required for the engines to agree bit for bit; confirmed by Byron, September 17, 2026. [RECORD §5.2]*

**6.6 The hazard falls as the square root of the count.** Every hazard in a network of $N$ neurons — $N$ being every neuron the network holds — is multiplied by $\kappa(N) = \sqrt{N_0/N}$, with $N_0 = \text{ESCAPE\_REFERENCE\_COUNT} = 60$. It is the whole hazard that is scaled and not the width, so the decision keeps its sharpness, $e$ times per $\Delta_j$, and the score of §8 keeps its form; the factor is the same at every margin, and a neuron that hears nothing takes it like any other. The reference count is the unit the width is quoted in, as THRESHOLD_FAN_IN is the unit the threshold is quoted in, and is not a knob. Of the engines: the factor is recomputed from the count when a network is built or resumed rather than stored in a checkpoint, since it is a rule and not a state.
*Byron, September 16, 2026: "A design principle of this project is that it should have as few knobs as possible and by default operate in or near a stable regime. Therefore scaling the network MUST reduce the probability of escape noise at each neuron by sqrt(N). I want to build this into the rule." [RECORD §5.2]*

**6.7 A neuron fires iff its draw falls below its chance.** The wave's draws are taken where §3.8 says, one uniform per neuron in neuron order, and neuron $j$ fires iff its uniform is **strictly below** $P_j(t)$. Of the engines: one draw per neuron whether or not that neuron goes on to decide, so the three engines fire the same neurons at the same waves from the same seed.
*The rule as built and held by test since September 15, 2026; the draw's position and order are Claude's reading of what reproducibility across the engines requires, confirmed by Byron, September 17, 2026. [RECORD §5.2, §6.1]*

**6.8 Every neuron decides at every wave, touched or not.** A wave examines every neuron in the network: one decision each, whether or not a signal reached it that wave, so a neuron with enough potential waiting, or one whose margin has simply carried it, fires at the first wave after it is able to and not at the next signal to arrive. A forced neuron is the exception (§6.10), and a refractory neuron makes no decision at all (§6.12). This fixes the learning rule's clock as well as the firing rule's: every decision posts an entry under the single-spike rule (§8.4), and a neuron's expectation of itself moves per decision and not per epoch (§8.6).
*Claude's reading, built September 12, 2026 under Byron's boredom decision, and carried into the hazard on September 15, 2026; confirmed by Byron, September 17, 2026. [RECORD §5.2, §5.4, §1.3]*

**6.9 A neuron with no width takes the deterministic comparison.** Where the width $\Delta_j$ is not positive — ESCAPE_DELTA $=0$, or a starting threshold that is not positive, a collapsed axis carrying no width to quote — the neuron fires iff $V_j(t) \ge \theta_j(t)$ and it is not refractory. $\Delta = 0$ is the deterministic rule word for word. Of the engines: this case is taken by the comparison and not by dividing by a zero width.
*Byron, September 15, 2026, with §6.4 ("$\Delta = 0$ is the deterministic rule word for word"); the collapsed-axis case as resolved September 16, 2026, under which, given §6.1, it arises only for a threshold given as 0 outright. [RECORD §5.2]*

**6.10 A forced neuron fires by its stimulus and decides nothing.** An input neuron driven at its stimulus's time fires regardless of its potential and its threshold, the refractory period permitting, and makes no firing decision that wave, so no credit is posted for that spike; the arrivals it closes settle their debit as at the floor (§6.3). Forced neurons fire before the wave's deciding neurons; a stimulus listed twice fires once, the first spike making the neuron refractory.
*The record's firing rule, standing: "A forced input neuron fires at its input's time regardless of $p$ and $\theta$"; and, with escape noise, September 15, 2026: "A forced neuron fires by its stimulus and makes no decision that wave." [RECORD §5.2, §4.4]*

**6.11 The spike resets the neuron and is remembered.** A neuron that fires takes

$$V \leftarrow 0, \qquad t_{\text{prev}} \leftarrow t_{\text{fired}}, \qquad t_{\text{fired}} \leftarrow t,$$

so the potential begins accumulating afresh (§2.2, §2.4) and the neuron keeps its last two spike times, its outgoing signals being scheduled one hop later (§3.5). $t_{\text{fired}}$ is what §2.5 reads the refractory period from and, under the neuron rule, nothing else reads (under exploration at the synapse the synapses' exposure runs from it too, §7.5); the gap back to $t_{\text{prev}}$ is read by the quash alone (§10.1, a non-default). No rule in force reads the interval since a neuron's last spike at a decision (§7.4).
*The record's firing rule, standing since the project began. [RECORD §5.2]*

**6.12 A refractory neuron makes no decision.** The refractory period and what it suspends are §2.5's. What this section adds is where the hazard picks up: it resumes at the period's end, which is where §6.5's $\Delta t$ then runs from, so the suspension costs the neuron no accumulated exposure and buys it none.

A neuron may fire any number of times, the period permitting, and the hop is what permits it. Its own spike comes back to it only round a cycle, and §4.4 gives it none shorter than a reciprocal pair, so the earliest return is two hops — $2h = 10.2$ ms, a clear 5.2 ms past the wall (§3.2). Another neuron's spike, sent in the same wave, arrives LAG past the wall, which is the tightest arrival there is. Either may refire the neuron, and neither is the clock's accident: under this hop no arrival ever lands on a refractory boundary at all. Nothing scores a refire as such: since §7.4 no rule in force reads how long a neuron waited between its own spikes, and a refire is paid exactly as any other spike is, by what §8.4 posts at the decisions that led to it.
*Byron and Cedric, from the beginning: this is a feedback control mechanism and a computational feature of the system. [RECORD §0, §5.3, §4.4]. The clause said, until September 18, 2026, that the two-hop return was "the return the shaping function pays most for"; with §7.4 out, no return is paid more than another.*

**6.13 Under exploration at the synapse, the neuron's decision is the comparison.** When a run explores at the synapse — its EXPLORATION setting is synapse (§7.1, §7.5) — every neuron fires iff $V_j(t) \ge \theta_j(t)$ and it is not refractory — §6.9, the deterministic rule word for word — at every wave as §6.8 says, and its own escape noise is off: every width $\Delta_j$ is 0 and ESCAPE_DELTA is not consulted. A run that asks for a positive neuron width and exploration at the synapse together is **refused**, with this clause as the reason: the system explores by one thing (§7.1). The neuron keeps its threshold, its deterministic spike and reset (§6.11) and its refractory period (§2.5); what changes is who takes the chance. A forced neuron is §6.10's, unchanged, and a neuron with no width posts nothing at its decision — the decisions the rule scores are its synapses' (§8.16).
*Byron, September 23, 2026, answering the fourth question of `docs/synaptic-escape-noise-2026-09-22.md` §7.9: "The neuron keeps its threshold and its deterministic spike." September 24, 2026, on both explorations together: refuse now, revisit if a measurement wants it. Specified September 24, 2026; the engines are brought to it under `docs/synapse-engine-plan-2026-09-23.md`, and until one carries it a run that asks for it is refused (§12.2).*

## 7. Exploration

**7.1 The firing decision is the exploration.** The system explores by the neuron's own stochastic decision (§6.5) — or, under exploration at the synapse, by its synapses' decisions (§7.5) — and by nothing else: nothing is added to any potential and no perturbation has to be recorded, because the decision that was taken is what the learning rule scores (§8). Under the neuron rule ESCAPE_DELTA sets how quiet silence can be and is the one constant that says how much the system explores; under exploration at the synapse that constant is SYNAPSE_HAZARD_REST (§7.6), and ESCAPE_DELTA is not consulted (§6.13). Which of the two a run uses is its **EXPLORATION** setting, neuron or synapse. No value of the rest hazard turns exploration at the synapse off — a rest hazard of 0 does not stop it under the linear family, whose hazard is then the potential's fraction of the threshold itself, and under the loglinear the synapses still take their draws (§7.6); under the neuron rule a width of 0 is §6.9's deterministic network, which draws nothing (§3.8). EXPLORATION is neuron unless a run says synapse until §7.6's default moves, and synapse after it unless a run says neuron — which a run also says by giving a positive neuron width (§7.6).
*Byron, September 15, 2026: "Make the boredom stochastic and it is Williams's unit outright." What the decision is — Williams's Bernoulli semilinear unit with the noise in the threshold rather than on the potential, an escape-noise integrate-and-fire neuron — is Claude's reading of Williams §2, and is the record's [RECORD §5.2, §6.7]. The EXPLORATION setting is Byron's, September 25, 2026 (`docs/synapse-build-map-2026-09-24.md` §5, Q1).*

**7.2 Boredom is a rate, not a deadline.** A neuron nobody talks to sits at rest, $s = -\theta_j$, and fires on its own at $e^{-1/\Delta}\sqrt{N_0/N}$ spikes per hop, from wherever its margin sits and without waiting on any clock. This is the whole of what turns silence into a spike in this specification, and it holds for a neuron with no incoming synapses as for any other (§6.1). Under exploration at the synapse a silent neuron does not fire on its own: its synapses whisper at $h_0\,\kappa_i$ spikes per hop (§7.6, §7.7), and a spontaneous spike arises only downstream, where whispers from many sources accumulate in a neuron with no leak until its threshold is crossed.
*Byron, September 15, 2026, deciding that the hazard is the bored neuron and that nothing else is run on top of it: "The hazard is buying us what the bored clock was supposed to buy us, and much much more cleanly." [RECORD §5.2, §5.4]*

**7.3 One stream, one order, in every engine.** The uniforms of §3.8 come from the run's exploration stream: one seeded stream of its own, which supplies the firing decisions — or, under exploration at the synapse, the synapses' and the read synapses' decisions (§3.8) — and nothing else. A run has three, and no two share draws: this one; the network's own stream, which draws the wiring, the weights and the drive's arrival times (§4.8, §5.4); and the input stream, which draws the epoch's patterns (§5.9). A network whose width is positive, or whose EXPLORATION is synapse (§7.1) — its synapses taking their draws whatever the rest hazard, $h_0 = 0$ included (§7.6) — must be run with one. Of the engines: each takes the same number of draws from that stream, in the same order, at the same point in the wave, and the Rust loop takes the Python stream's MT19937 state and hands it back, so a seed gives the same spikes whichever engine runs — the objects and Rust to the bit, the arrays exactly on the spikes and to a part in $10^9$ on continuous quantities (the platforms clause, §0).
*The stream is Byron's separation of streams, September 14, 2026 — a run's randomness comes from a stream that nothing else consumes — carried to the hazard's draws on September 15, 2026. [RECORD §5.2, §6.1, §4.5]*

**7.4 No shaping function.** What a decision posts is what §8.4 gives it and
nothing else. No factor weighs an entry by the time since the neuron's own
last spike, and no rule in force reads that interval at all.

What the system still has of a neuron's own timing is the refractory period
and the hop, and neither is a learning signal: §2.5 suspends the neuron for
REFRACTORY and reads $t^{\text{fired}}$ for nothing else, §3.2 delivers every
signal one hop later, and §6.12 lets a spike come back and refire the neuron
without that refire being worth more or less than any other spike. A neuron's
interspike interval is an outcome of those three and of the weights; it is not
a quantity the rule aims at, and the one clause that reads it — the quash,
§10.1 — is a non-default and reads it to punish a tight loop, not to set a
rate.

*Byron, September 18, 2026, taking it out: "I want to factor out shaping. I
don't understand it. A fundamental principle of this project is that we ONLY
INCLUDE MECHANICS WE UNDERSTAND. We don't invent something that's not there,
although we are abstracting the neuron as much as possible while keeping what
it does."*

*What went, and what was measured before it did.* The shaping function was a
factor $f$ on every entry, anchored at $-1$ at the refractory wall and $+1$ at
an interval TARGET_ISI, meant as a set point against saturation: a neuron
firing as fast as its refractory period allows would have its own evidence
turned against it. Two findings of September 18, 2026 are why it is not
carried. It is not a set point — $f$ enters the rule only as a multiplier, no
term anywhere depends on $t$ alone, so where the reward gradient is flat it
multiplies zero and exerts no pressure at all, which is the saturating neuron
it was for. And the advantage is a reward minus a running baseline (§9.3) and
is negative about as often as positive, so on a negative-advantage epoch a
wall-hugging spike takes $f < 0$ times $A < 0$ and is **reinforced**. The
mechanism was 'unlearn from the early decisions', not 'aim the interval', and
the file said the second. With it go TARGET_ISI, HOPS,
EARLY_ARRIVAL_PUNISHMENT_FACTOR and ISI_FACTOR, and with it went the clause then numbered §7.6, which
said a resumed network keeps the setting it was saved under (the number has belonged to the hazard family since September 24, 2026). LAG stays: it is
§3.2's and belongs to the hop. What §0.12's deterministic drive and §9.13's
rate teacher were to be quoted in, having quoted TARGET_ISI, was answered
twice over on September 18, 2026. The drive's interval is REFRACTORY + LAG —
"REFRACTORY+LAG" — which is 2 HOP, written as the sum because that is how
Byron gave it (§3.2); that is not TARGET_ISI restored under another name, since being
the unit the drive quotes makes the hop no more a set point than it was, no
rule in force reading a neuron's own interval, which is this clause. The rate
teacher's 1-target is not quoted in the hop at all: it is ONE_TARGET, a
constant of its own (§9.13). Neither clause is in force. [RECORD §0.2, §6.7]*

---

**7.5 Exploration at the synapse — specified September 24, 2026, being built.** Below threshold, every synapse of a neuron takes its own chance. At every wave, after the floor has settled and after the wave's spikes are decided (§6.13), every synapse $i \to j$ whose source $i$ did not spike this wave decides — its source refractory or not (7.8). The spikes the synapse's hazard expects of it over the interval since $i$'s synapses last decided, or since $i$'s spike where that came later, are

$$m_i(t) = \frac{\Delta t}{\text{hop}}\;\kappa_i\;h(u_i), \qquad u_i = \frac{\operatorname{clip}\big(V_i(t),\,0,\,\theta_i\big)}{\theta_i},$$

the same $m$ for every synapse of $i$: $h$ is the hazard family of 7.6, $\kappa_i$ the scaling of 7.7, and $\Delta t$ never negative (§6.5). The synapse **escapes** iff its uniform (§3.8) is strictly below $P = 1 - e^{-m_i}$, computed as $-\operatorname{expm1}(-m_i)$ with $m_i$ capped at $10^3$, as §6.5 computes the neuron's. An escape delivers the synapse's weight to its target one hop later, as a spike's signal would (7.9), and leaves $V_i$ where it was. At threshold the source spikes and every synapse it has transmits (§6.13), so $u_i = 1$ never reaches a synapse decision. One exposure clock per source serves all its synapses, since they decide together on one $u_i$; it is the clock §2.1 already holds, whose neuron-side use §6.13 ends. A threshold at or below zero leaves $u_i$ undefined, and under exploration at the synapse a network holding one is refused (§12.2) — at the build, and at the moment homeostasis or un-sticking (§9.9, §9.10) takes a threshold there. *Of the engines:* $u_i$ is computed as $\min(\max(V_i, 0), \theta_i)/\theta_i$; the loglinear hazard as `h0 ** (1 - u)` and the linear as `h0 + (1 - h0) * u`; $m_i$ as $\Delta t/\text{hop}$, times $\kappa_i$, times $h$, left to right as §6.5's is, then capped; and $\kappa_i$ once, where the network is built or resumed, handed to every engine — Claude's forms, as §6.5's are, required for the object engine and the Rust loop to agree bit for bit, and confirmed by Byron, September 25, 2026 (`docs/synapse-build-map-2026-09-24.md` §5, Q13, Q15).
*Byron, September 23, 2026, 03:31 MDT, stating the mechanism: "the synapse ESCAPES with probability density monotonically increasing as (min(max(V_pre,0),V_threshold)/V_threshold. At V_threshold and greater this is equal to 1, so the synapse 'fires' deterministically"; his answers of that morning, that $V_{\text{threshold}}$ is the neuron's and that an escape leaves $V_{\text{pre}}$ alone (`docs/synaptic-escape-noise-2026-09-22.md` §7.10); and September 24, 2026, choosing that a synapse decides at every wave, as neurons do today, over deciding only at the waves that reach its source. The continuous form — a rate on elapsed exposure rather than a toss per hop — is §6.5's carried over: the second toy decided once a hop, and Byron asked that time stay continuous. Specified September 24, 2026; the engines are brought to it under `docs/synapse-engine-plan-2026-09-23.md`, and until one carries it a run that asks for it is refused (§12.2).*

**7.6 The hazard family and the rest hazard.** Two families, each $h_0$ at rest and one spike per hop at threshold, quoted per hop before the scaling of 7.7:

- **loglinear**, $h(u) = h_0^{\,1-u}$ — the family today's neuron hazard belongs to, §6.5 being this with $h_0 = e^{-1/\Delta}$, clipped at zero potential and cut off at threshold;
- **linear**, $h(u) = h_0 + (1 - h_0)\,u$ — Byron's statement read literally, rising in proportion to the potential.

SYNAPSE_HAZARD_FAMILY names the family a run uses, loglinear unless the run says linear, and every engine carries both. SYNAPSE_HAZARD_REST $= h_0 = 0.01$ is the rest hazard: what a synapse of a source at zero potential speculates at, and so what a silent neuron's synapses whisper at (§7.2). It lies in $[0, 1)$: at 1 the hazard is flat and scores nothing, above 1 the family turns over, below 0 it is undefined, and a run that asks for any of those is refused (§12.2). With $h_0 = 0$ under the loglinear family no synapse speculates below threshold and the network spikes as §6.9's deterministic one does, word for word, though its synapses still take their draws (§3.8): the run explores at the synapse by its setting (§7.1), not by $h_0$. This section becomes the exploration in force by default — EXPLORATION synapse, ESCAPE_DELTA's neuron width 0 — once all three engines carry it and agree on mnist's configuration (§12.8), and mnist's learning rate has been re-found under §8.16 (11.11); until both, §6.5 runs by default and a run asks for this section by name. Once the default has moved, a run that gives a positive neuron width asks for the neuron rule of §6.5 by that alone — a width of 0 given explicitly is the width §6.13 gives every neuron and asks for nothing — and one that asks for exploration at the synapse and a positive width together is refused (§6.13).
*Byron, September 23, 2026, on the rest hazard: "To be determined through rigorous investigation." September 24, 2026, choosing both families as a run option with loglinear the default, and 0.01 — the one value inside every regime's plateau in the second toy's sweeps (`docs/synaptic-escape-noise-2026-09-22.md` §7.13; the four sweeps of September 23 tabled in `docs/synapse-engine-plan-2026-09-23.md` §6) — as its default, to be re-measured on mnist. Byron, September 25, 2026: the default moves only when three engines agree and the rate has been re-found, a width given explicitly then asks for the neuron rule, and $h_0$ lies in $[0, 1)$ (`docs/synapse-build-map-2026-09-24.md` §5, Q2, Q14).*

**7.7 The synapse hazard is scaled by the count, or by the fan-out.** Every hazard is scaled by $\kappa(N) = \sqrt{N_0/N}$ (§6.6), the synapse's included: $\kappa_i = \kappa(N)$ is the default. A run may instead scale by the source's fan-out, $\kappa_i = \kappa(N)/F_i$, $F_i$ the number of synapses $i$ has, so that a neuron's speculation per hop, summed over its synapses, does not grow with how many it has. SYNAPSE_HAZARD_SCALING names which, count unless the run says fan-out. The factor is a rule and not a state: recomputed at build and at resume, stored in no checkpoint (§12.10).
*Byron, September 16, 2026, for the count (§6.6); September 24, 2026, choosing both as a run option with the count the default. The fan-out form is the record's $1/d$ per hop (RECORD §0.3), untested, and the two are one constant apart on any one goo.*

**7.8 A refractory source's synapses whisper at rest.** A spike resets the source to zero and makes it refractory (§2.4, §2.5), and its synapses go on deciding at every wave of the period at $u_i = 0$, the rest hazard $h_0$: the synapse follows its source's potential and not its silence. The exposure a decision covers runs from the previous decision, or from the spike where that came later, and nothing suspends it. A spike is therefore followed by whispers at rest, not by quiet.
*Byron, September 24, 2026, choosing this over the second toy's rule, under which a refractory source's synapses were suspended and resumed with the period's end (`docs/synaptic-escape-noise-2026-09-22.md` §7.11). The toy's numbers are therefore not this clause's, and the plan's §5 re-measures.*

**7.9 A ventured signal, and the read synapse.** An escape schedules a signal one hop later (§3.5) carrying a **ventured** mark, and it is a signal in every other way: delivered whole at an instant (§0.3), dropped if its target is refractory (§8.13), floored with the wave's total (§6.2), integrated into the potential and the trace (§1.6, §8.17), and recorded on the stamp (§1.7). The mark rides on the signal in flight and on nothing else, and a checkpoint carries it with the signal (§12.9).

Under exploration at the synapse every output neuron has one **read synapse**: an outgoing synapse to the read and to no neuron, which decides at every wave as any synapse does, on the output's own $u$, its draw taken after the network's synapses' (§3.8). Its ventured transmissions this epoch are added to the output's spikes in the count read (§5.10), each counted at the wave of its decision — it delivers nothing, so nothing of it is in flight, and an escape in an epoch's last hop counts in that epoch — so an output's speculation reaches the reward and the critic scores the sum. It is not a connection of §1: it takes no id (§1.2) and no weight (§1.3), carries none of §1.5's state, has no place in edge order (§3.8) and none in the topology a seed builds (§4.8); it is one of the output's $F_i$ synapses for 7.7's scaling and for §8.16's decisions, and its only state is the per-epoch count §2.1 gives the output. The read synapse delivers nothing, learns nothing and is credited nothing; it exists so that the output's incoming synapses can be — an output projects nowhere else on a goo with no hidden neurons, and §8.16 credits a synapse only through the decisions of its target's synapses.
*The device is the second toy's (`docs/synaptic-escape-noise-2026-09-22.md` §7.11, "the read on transmissions"), without which the exact rule posts nothing to an output's fan-in. Byron, September 24, 2026, making it part of the mechanism rather than an option, and choosing that a ventured delivery moves the stamp as any signal does. Byron, September 25, 2026, that a read escape counts at its decision (`docs/synapse-build-map-2026-09-24.md` §5, Q6).*

*Deprecating $\theta$, named here on September 17, 2026 as a later change, is not taken up: the neuron keeps its threshold (§6.13). The margin and the width of §6.4 stay quoted from it, and no clause is written from the deprecation.*

## 8. The learning rule

One rule pays at the read: REINFORCE (Williams 1992, [1] in
`BIBLIOGRAPHY.md`) with a per-decision eligibility. The clauses below say
what a synapse earns, what it is paid, and what it has to carry to do
either. It is not the only rule that moves a weight: the quash moves one
too, locally and at a refire, and composes with this rule rather than
replacing it (§10.1).

*Under exploration at the synapse (§7.5) the decisions the rule scores are the synapses', and 8.16 says how 8.4's row and 8.11's bookkeeping are posted for them; 8.4, 8.6 to 8.8 and 8.11 to 8.12 as written describe the neuron's decisions, the rule that runs when a run's EXPLORATION is neuron (§7.1), as every run's is by default until §7.6's default moves.*

**8.1 The rule.** *(Williams's rule [1], carried from the pre-alpha and
asked for on this substance by Byron, September 13, 2026: "I want to try the
same thing with the REINFORCE algorithm" [record §6.7]; LR and BASELINE_RATE
are the standing constants of [record §1.3].)* The network is paid **one scalar** an epoch, its **reinforcement** $R$,
from the critic the run names. A running baseline $b$ gives the
**advantage** $A = R - b$. At the read, every synapse $i \to j$ that has
posted something since the last read and whose target $j$ carries no **driven mark** (5.8)
this epoch moves by

$$w_{ij} \leftarrow \mathrm{clip}\big(w_{ij} + \text{LR}\cdot A\cdot e_{ij}\big),
\qquad \text{LR} = 0.03,$$

clipped to the range that network's weights were drawn from (§1.4). A
synapse into a neuron forced this epoch is **skipped**: its firing was not
the network's doing. A synapse into an input the charged drive delivered to this epoch is skipped the same way (5.4b): what the skip reads is 5.8's driven mark, which a forced neuron and a charged one both carry, a delivery dropped at a refractory input included (Byron, September 24, 2026, keeping the mark, and September 25, 2026, that any delivery sets it). An input neuron the drive neither fired nor delivered to this epoch is an
ordinary neuron, and its incoming synapses are paid like any other's. Every
synapse in the network gets the same $A$; what tells them apart is $e_{ij}$
alone.

**8.2 The baseline.** *(The pre-alpha's, carried; the constant is [record
§1.3]. This clause owns the baseline; §9.3 cites it.)* $b$ is the
running average of the reinforcement. It starts at the first epoch's reinforcement, so the
first epoch's advantage is zero and nothing moves; after the update has used
it, $b \leftarrow b + \text{BASELINE\_RATE}\,(R - b)$, BASELINE_RATE $= 0.05$. There is one baseline for the whole network and none per neuron.

**8.3 Two eligibilities, and which one a run gets.** Two
survive: **hazard** and **hebb**. Both are the single-spike rule of 8.4 and
differ in what a decision's credit and expectation are (the table
there) — and in one thing the table does not show. The hazard's
$(c_j - q_j)$ is exactly zero-mean at every decision, both branches having
mean $m e^{-m}$, so nothing systematic is posted by a neuron whose behaviour
carries no information. Hebb's is $y - \hat p_j$, whose mean is the lag of the
neuron's own estimate, $P_j - \hat p_j$; it is zero only where that estimate
has caught up, and DECISION_MEMORY sets how long that takes (§8.6). A run
under hebb is therefore posting a systematic component wherever a neuron's
rate is moving. A run may name one. Where it names none and the firing
decision is a draw, the eligibility is hazard. Where the threshold decides and nothing else draws, **no eligibility runs and
the rule refuses to learn.** Byron, September 17, 2026: "Refuse to learn."
REINFORCE estimates a gradient from the randomness of the decision; with no
width there is no randomness and nothing to estimate, so a run that asks for
the reinforce rule on a network without escape noise is refused rather than
given a rule with no derivation behind it (§0.9). The record's fallback was
the ELIGIBILITY constant, whose value was perturb, and perturb leaves this
specification. The hazard
refuses a network without escape noise and says so rather than approximating:
a decision that was not a draw has no probability to differentiate. Which
eligibility runs is independent of what makes the firing decision — a run may
name hebb on a network with escape noise, and then the decision is still a
draw but is posted against the neuron's own expectation. Under exploration at the synapse the decisions are the synapses', the eligibility is the hazard's and is posted as 8.16 says, and hebb, having no neuron decision to centre, is refused.
*The hazard default is Claude's reading of the default Byron set September
15, 2026, to be corrected in a word; the record's rule was "hazard under
escape noise, else the ELIGIBILITY constant" [record §1.3, §6.7].*

**8.4 The single-spike rule.** *(Byron, September 17, 2026: "Let's derive
the REINFORCE rule for the synaptic weight update for a SINGLE postsynaptic
spike at time t. We will be modifying the existing hebb and hazard
eligibilities. Hebb assumes there is no hazard and that the postsynaptic
spike was generated by presynaptic activity. Hazard assumes that there is
escape noise." Claude's derivation, on the evidence accumulator [record
§6.7].)* At **every decision** of neuron $j$, at time $t'$, with outcome
$y_j(t') = 1$ if it fired and 0 if it did not, every synapse into $j$ moves
its score by

$$e_{ij} \mathrel{+}= \big(c_j(t') - q_j(t')\big)\,x_{ij}(t'),$$

$x_{ij}$ the synapse's trace (8.5), $c_j$ the
decision's credit and $q_j$ its expectation:

| eligibility | fired: $c_j$, $q_j$ | silent: $c_j$, $q_j$ | the expectation |
|---|---|---|---|
| hazard | $m\,e^{-m}/(1 - e^{-m})$, 0 | 0, $m$ | $m_j(t')$, the hazard's own (8.7) |
| hebb | 1, $\hat p_j$ | 0, $\hat p_j$ | $\hat p_j$, the neuron's own estimate of its spike (8.6) |

In words: a spike credits every arrival still standing in the potential, each
equally, and every decision debits them by what was expected of the neuron at
that moment. A neuron that fires more than once in an epoch has an entry
posted for every decision it made.
*Claude's derivation on Byron's decision of September 17, 2026, and his three decisions on what it left open; confirmed by Byron, September 17, 2026 — he asked for this one to be read twice before confirming it [record §6.7].*

**8.5 The trace.** $x_{ij}(t')$ is what the synapse has in its target's
potential at the decision — the derivative of the margin by $w_{ij}$, not a
stand-in for it. Under TRACE $=$ ventured it counts the ventured arrivals only and is then not the derivative (8.17). Under the evidence accumulator it is the **count** of the
arrivals $j$ integrated since its last spike; under the leak (8.12) it is
those arrivals each decayed by TAU. A signal that arrives while $j$ is
refractory is dropped, adds nothing to the potential, and counts nothing
here. The trace is cleared whenever the potential stops being the sum of
those arrivals: at $j$'s spike, when the floor bites, and at a discharge
(§1.6) — the potential being then the floor, or zero, whatever the weights.
The trace belongs to the synapse, and it is what gives the rule per-synapse
resolution.
*Claude's derivation, September 17, 2026, on Byron's decision to derive the rule; confirmed by Byron, September 17, 2026 [record §6.7].*

**8.6 The neuron's expectation of its own spike.** *(Byron, September 17,
2026: "Expectation is changed per decision in this architecture.")* $\hat p_j$ is undefined until $j$'s first decision, which sets it to that
decision's outcome and posts nothing. After every decision, the entry posted
first,

$$\hat p_j \leftarrow \hat p_j + \max\!\big(\text{DECISION\_MEMORY},\,1/n\big)\,(y - \hat p_j),$$

$n$ the decisions $j$ has made to date, DECISION_MEMORY $= 10^{-4}$: the
plain mean of the first ten thousand decisions, an exponential average of
about the last ten thousand after. The estimate's own move takes the outcome
unweighed — what the learning rule weighs is what it posts, not what the
neuron remembers — and a neuron whose decisions the rule in force does not
post for keeps $\hat p_j$ and $n$ untouched. *The warm start by $1/n$ is Claude's reading of
"changed per decision", kept by Byron, September 17, 2026: "We don't worry
about what happens in the neuron's first ten thousand decisions as it's just
waking up and hasn't had its coffee yet ... I'm more interested in what the
neuron is doing after an hour or two." The decisions it covers are a few
epochs against the hundred thousand an hour of clock holds, and the record
carries the count. Open: DECISION_MEMORY is a starting value, to be swept
[record §1.3].*

**8.7 The hazard's credit and expectation.** *(Claude's derivation from [1],
asked for and chosen by Byron, September 15, 2026: "It is the most like what
I want to do" [record §6.7].)* $m = m_j(t')$ is the spikes the escape-noise
hazard expects of $j$ over the interval this decision covers, exactly as the
firing clause computes it [record §5.2]. The credits of the table are that
interval's log-likelihood differentiated: $\partial m/\partial w_{ij} = (m/\Delta_j)\,x_{ij}$, and the $1/\Delta_j$ — the neuron's own width — is
folded into LR, so one learning rate serves neurons of any width. Under exploration at the synapse what is folded is $1/\theta_i$ and, under the loglinear family, the family's constant log-derivative $-\ln h_0$; under the linear family the log-derivative, $(1 - h_0)/h(u_i)$, varies with the source's potential and is carried in the gain, not folded (8.16).

**8.8 Hebb's credit at the spike — open, and a full unit until it is
settled.** It is 1, where the hazard's credit discounts a spike that was
expected. Whether it should be discounted too is **open**: Byron, September
17, 2026, *"Defer for now"*, and later the same day, settling the default
without settling the question, *"Defaults to 1? 1 is the best number to
default to. 0 is the second-best."*

A full unit is also what keeps the rule as near zero-mean as hebb comes: with
it $E[c_j] = P_j$ against $E[q_j] = \hat p_j$, so the only gap is the
estimate's lag (§8.3), where any discount $d < 1$ gives $E[c_j] = d\,P_j$ and
widens it, unless $d$ happened to equal $\hat p_j / P_j$.


**8.10 Arrivals stay open across reads.** *(Byron, September 17, 2026: "Let
them run! Epochs are for the convenience of teaching the network, and to some
extent an emergent property of the time constants, but they don't really
exist in nature.")* An arrival stays in the potential until $j$ spikes, so
what it owes runs across reads. Each read pays the score posted since the
last read and clears it, whether or not the advantage moved a weight (§1.9);
the open arrivals stay open, their debit counting from then. The epoch bounds
the payment, not the accumulator.

**8.11 What the synapse owns, under the accumulator.** Each event below is
one operation on the synapse or on the neuron, and **no loop over a fan-in
runs at a decision**. Per synapse: the trace $x_{ij}$, the note $B_{ij}$ and
the score $e_{ij}$; per neuron: $E_j$, the spikes expected over its decisions
since its last spike.

- **an arrival** integrated: $x_{ij} \mathrel{+}= 1$ and $B_{ij} \mathrel{+}= E_j$;
- **a decision**: $E_j \mathrel{+}= q_j$, and nothing else;
- **the spike**: every open arrival settles, $e_{ij} \mathrel{+}= c_j\,x_{ij} - (x_{ij}E_j - B_{ij})$, then $x_{ij} = 0$, $B_{ij} = 0$ and $E_j$ restarts at 0;
- **the floor, a forced spike and a discharge**: settle the same way with no credit;
- **the read**: settle the debit so far into each score, $e_{ij} \mathrel{-}= x_{ij}E_j - B_{ij}$ with $B_{ij} \leftarrow x_{ij}E_j$, pay 8.1, and clear the score.

This bookkeeping is the rule: it is a regrouping of the per-decision sum of
8.4 and is not required to reproduce that sum bit for bit. The engines are
held to each other (8.15) [record §6.7].

**8.12 The leak path.** The leak is a per-run option, TAU (the neuron's
clause [record §5.1, §1.2]). Under it the rule of 8.4 is posted **per
decision**: at every decision the engine walks $j$'s incoming synapses and
adds $(c_j - q_j)$ times each leaked trace. The bookkeeping of 8.11 is not used on this
path, and there is no note (§1.8): the per-decision walk is the rule for it
[record §6.7].

**8.13 The late-signal question does not arise.** A signal that arrives while
$j$ is refractory is dropped and counts nothing (§2.5); one that arrives
after $j$'s spike opens the next interval and is credited there, the
potential having been reset at that spike. The trace says what each signal
was contributing at each decision, so neither surviving eligibility takes a
late-signal rule or an eligibility trace of its own.
*Claude's reading of the record's "LATE does not apply", confirmed by Byron,
September 17, 2026 [record §6.7].*

**8.14 What a checkpoint carries of this rule.** The round-trip itself is
§12.9's. Of this rule: the weights; per neuron $\hat p_j$, the decisions to
date and $E_j$; per synapse the trace and the time it was brought up to, and
the note $B_{ij}$; the teacher's baseline $b$, LR, critic and eligibility; under exploration at the synapse, each neuron's gain $G_j$, each output's read count this epoch, the ventured mark on every signal in flight, and the settings of §7.1, §7.6, §7.7, 8.17 and 5.4b — a checkpoint that carries no §7.1 setting was saved under the neuron rule, and §12.9's refusal of a resume under the other exploration reads it so;
[record §6.7, §0.2]. The score since the last read is carried too
(§1.5), so a checkpoint is not confined to a read. *Byron, September 17,
2026: "a checkpoint should hold a synapse's score for analysis purposes."*

**8.15 What the rule requires of an engine.** *(Byron, September 14, 2026:
"rules in authority.md must be implemented cross-platform"; and September 17,
2026: "keep, with tolerances, as a foundational rule; all platforms are built
from one clear authority" [record §7].)* Every engine posts at the
decision, in the decision's own order, and carries the state 8.11 names —
the trace, the note and the score on each synapse, $\hat p_j$, the decision
count and $E_j$ — or, under exploration at the synapse, the gain $G_j$ (8.16) — on each neuron — so that the read is one pass over the
synapses and pays what the engine already accumulated. The object engine and
the Rust loop agree to the bit; the array engine agrees within the tolerance
the invariants clause names for it, summing a wave in matrix order. An
engine that lacks the rule refuses the run and says so.

**8.16 The single-spike rule at the synapses' decisions — specified September 24, 2026, being built.** Under exploration at the synapse (§7.5) the decisions are the synapses', and the rule of 8.4 is posted for them. At every wave, for every source $i$ whose $F_i$ synapses decided (an output's read synapse among them, §7.9), $a_i$ of them escaping: each escape's credit is the hazard's, $c = m\,e^{-m}/(1 - e^{-m})$, each silent decision's expectation is $q = m$, with $m = m_i(t)$ of §7.5 — 8.4's hazard row, the decision being the synapse's rather than the neuron's and $F_i$ of them a wave instead of one — and the wave posts to every synapse $k \to i$ **into the source**

$$e_{ki} \mathrel{+}= \big(a_i\,c - (F_i - a_i)\,m\big)\,x_{ki},$$

$x_{ki}$ the trace of $k \to i$ (8.5, 8.17). The wave posts this only while the source's potential is above zero, $V_i(t) > 0$: at or below zero §7.5's clip holds the hazard at $h_0$ whatever the synapses into $i$ delivered, the derivative below is zero, and the wave posts nothing for $i$ — as the second toy posted, whose numbers set §7.6's default. The odds of an escape of $i \to j$ are set by $V_i$, which the synapses into $i$ built through their traces, so they are what its outcome credits; the escaping synapse's own weight $w_{ij}$ shapes no odds of its own and is credited, as today, through the decisions of $j$'s synapses and its trace $x_{ij}$. The entry's mean at every wave is zero, both branches having mean $m\,e^{-m}$ (8.3). The deterministic spike of §6.13 posts nothing: it is not a draw. The row's credits are the log-likelihood of the wave's decisions differentiated by $w_{ki}$, as 8.7 differentiates the neuron's: $\partial m_i/\partial w_{ki} = m_i\,\rho(u_i)\,x_{ki}/\theta_i$, with $\rho(u) = h'(u)/h(u)$ the family's log-derivative. Under the loglinear family $\rho = -\ln h_0$, a constant, and it and $1/\theta_i$ are folded into LR as 8.7 folds $1/\Delta_j$, so the entry stands as written; under the linear family $\rho(u_i) = (1 - h_0)/h(u_i)$ varies with the source's potential and multiplies the wave's entry, $1/\theta_i$ alone being folded. The learning rate is quoted plain and its value re-found under this rule (11.11 for mnist). The second toy applied neither factor under the linear family, so §7.13's linear numbers are of a rule one factor short of this one.

The bookkeeping of 8.11 carries over with one substitution. Per neuron a running **gain** $G_j$ replaces $E_j$ and the credit: at each wave that posts for $j$ — its synapses decided and $V_j > 0$ — $G_j \mathrel{+}= \tilde\rho_j\,\big(a_j\,c - (F_j - a_j)\,m\big)$, $\tilde\rho_j$ being what of $\rho$ the rule above leaves unfolded: 1 under the loglinear family, $(1 - h_0)/h(u_j)$ under the linear; an arrival integrated on $k \to j$ notes $B_{kj} \mathrel{+}= G_j$; a settle — $j$'s spike, the floor, a forced spike and a discharge — posts $e_{kj} \mathrel{+}= x_{kj}\,G_j - B_{kj}$, clears $x$ and $B$, and restarts $G_j$ at 0; the read posts the same and re-bases $B_{kj} \leftarrow x_{kj}\,G_j$, leaving $x$ and $G_j$ as they are (§1.9). No loop over a fan-in runs at a decision (§1.8). Under the leak (8.12) the walk over the fan-in runs once a wave per source the wave posts for. Nothing new is carried per synapse (§1.5). *Of the engines:* $c$ is computed as `m * exp(-m) / -expm1(-m)`, the form the neuron's credit takes today, and only where $m > 0$ — at $m = 0$ nothing can escape and the entry is 0; the entry as `a * c - (F - a) * m`, $F - a$ an integer. Under the linear family, where $m$ is below its cap, $m\,\tilde\rho$ is exactly $(\Delta t/\text{hop})\,\kappa_i\,(1 - h_0)$, and the entry is written so that no silent decision divides by $h$: `a * ((1 - h0) / h * c) - (F - a) * (dt / hop * kappa * (1 - h0))`, the escapes' part taken only where $a > 0$, so a potential decayed toward zero under the leak, whose hazard underflows, posts a finite entry; at the cap, $\tilde\rho$ multiplies the finished entry, `(1 - h0) / h * (a * c - (F - a) * m)`. An entry that is still not finite — an escape at a hazard that underflowed, a chance of some $10^{-308}$ — is refused (§12.2). Claude's forms, confirmed by Byron, September 25, 2026 (Q15), the linear family's silent part rewritten on his word the same day.
*Claude's derivation of September 23, 2026 (`docs/synaptic-escape-noise-2026-09-22.md` §7.9: Williams's rule credits whatever parameter shaped a draw's odds), settled by Byron's first answer of that morning — $V_{\text{threshold}}$ is the neuron's, so the local case does not arise — and the bookkeeping as proposed in `docs/synapse-engine-plan-2026-09-23.md` §1.8, taken up by Byron on September 24, 2026. The engines are held to each other on it (8.15). Byron, September 25, 2026: the entry posted only while $V_i > 0$, as the toy posted, and the unfolded factor carried in the gain (`docs/synapse-build-map-2026-09-24.md` §5, Q4, Q3). Specified September 24, 2026; the engines are brought to it under `docs/synapse-engine-plan-2026-09-23.md`, and until one carries it a run that asks for it is refused (§12.2).*

**8.17 What the trace counts is a run option: every delivery, or the ventured ones.** TRACE $=$ all, the default: $x_{ij}$ counts every arrival $j$ integrated along $i \to j$, relayed with $i$'s spike or ventured by the synapse, and stays the derivative of the potential by the weight (§1.6, 8.5), so the rule is Williams's estimator. TRACE $=$ ventured: $x_{ij}$ counts the ventured arrivals only — a synapse that transmitted with its source's spike does not try to learn from it — while the potential takes every delivery. The trace is then not the derivative and the estimator is biased toward what was ventured; this clause says so, and a run that names it says so in its record. Under it a relayed arrival is, for the synapse's learning, not there: it notes nothing ($B_{ij}$ stands), it leaves $x_{ij}$ as it is, and under the leak it neither decays the trace nor moves the moment the trace was last brought up to date (§1.6) — so the settle's $x\,G - B$ covers exactly the arrivals the trace counted. Nothing else of 8.4–8.16 changes.
*Byron and Cedric's idea of September 23, 2026 (§1's closing note: "when the synapse fires deterministically, it does not try to learn at all"), measured that day in the second toy (`docs/synaptic-escape-noise-2026-09-22.md` §7.12) and in the four sweeps of September 23: under the file's forced drive it learns nothing, and under a charged input (5.4b) it costs about 0.03 and one weak seed in ten at its plateau. Byron, September 24, 2026, choosing every delivery as the default with ventured a named option. Byron, September 25, 2026, that a relayed arrival touches none of the synapse's bookkeeping under it (`docs/synapse-build-map-2026-09-24.md` §5, Q5).*

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

*Claude's reading of the rule as built, confirmed by Byron, September 17, 2026 [RECORD
§6.7, §6.15].* When an epoch's last wave has run, the teacher does these,
in this order:

1. read the output zone and compute the epoch's reinforcement $R$ from the critic
   (§9.4–§9.7);
2. pay the rule at the read with the advantage $A = R - b$ (§9.3);
3. move each neuron's firing-rate memory (§9.8);
4. move thresholds by homeostasis (§9.9), and then by un-sticking (§9.10);
5. move the baseline $b$ (§9.3).

A neuron the stimulus forced to fire this epoch is skipped by (3) and (4), and so is one the charged drive delivered to (5.4b; Byron, September 25, 2026): each carries 5.8's driven mark, and the mark is what the skip reads.
Step (2) passes over it as well, by the paying rule's own clause (§8.1); this
enumeration is not the whole of what a forced neuron is exempt from.

The order is part of the rule: every engine performs the same operations in
the same order, so that a run is the same run whichever engine ran it.

### 9.3 The baseline and the advantage

The reinforcement reaches a synapse as an advantage against a running baseline, which is the
paying rule's own and is stated once, at §8.2: $A = R - b$ with $b$ as it
stands, and $b$ moves only after the update has used it. The advantage is one
scalar an epoch, the same number for every synapse the paying rule touches.

The baseline is part of what a checkpoint carries and a resume restores
(§12.9, §12.11): a resumed run is the same run continued, so it is paid
against the baseline the run had reached and not against a fresh one. [RECORD
§6.7, §1.3, §8; REINFORCE, Williams 1992, [1] in `BIBLIOGRAPHY.md`]

### 9.4 The evidence critic

*Byron, September 16, 2026, on reading the class sums as a vote share:*
"No, the spikes are EVIDENCE for now, not a proper maximum-likelihood
estimator. We will have to sweep for temperature eventually."

The read gives one number $n_k$ per class, the class's evidence this epoch
(under complement coding, its fire-if-one sum minus its fire-if-zero sum;
the read's section defines it). The evidence critic takes those as log-odds
at a temperature $T$ = TEMPERATURE: the estimate over the $C$ classes is the
Boltzmann distribution and the reinforcement is its log score at the label $y$,

$$q_k = \frac{e^{n_k/T}}{\sum_j e^{n_j/T}}, \qquad
R = \ln q_y = \frac{n_y}{T} - \ln \sum_j e^{n_j/T}.$$

A uniform estimate scores $\ln(1/C)$, which is $-2.30$ at ten classes; a
perfect epoch scores 0; a class with no spikes at all is weak evidence and
not $-\infty$. $T$ sets what one spike of lead is worth. It is mnist's
critic. [RECORD §8, §1.3]

TEMPERATURE is 2 **FOR NOW** — *Byron, the same day: "We will have to sweep
for temperature eventually"*. What is open is the value, which a sweep is to
set. [RECORD §1.3, §8]

*What it requires of the engines.* Every engine computes this reinforcement through one
function, which subtracts the largest class sum before exponentiating so
that no temperature overflows. Its inputs are sums of integer spike counts,
so the reinforcement is the same to the bit in all three engines and the tolerance
of the one-authority rule is not needed here.

### 9.5 The row critic

An output neuron is **on** for this critic when its count for the epoch is at
least ROW_CRITIC_PICKINESS_IN_SPIKES $= 2$, an integer; no other critic reads
the zone as bits. The reinforcement is the fraction of the output neurons whose
state this epoch matches the target pattern the problem names; a row that
matches everywhere is 1.

Two is chosen against the rest rate of the escape hazard (§7.2): a neuron that
hears nothing still fires at that rate under the neuron rule (under exploration at the synapse it does not, §7.2), so a pickiness too low reads
background as signal. How low is too low depends on the epoch's length and —
through $\kappa(N)$ — on the count of neurons, so this pair is chosen
together and neither travels alone: **at INTERVAL 35 ms and a pickiness of
2**, background costs the least of the pairs the record measured, and a
longer epoch at the same pickiness lets more of it through.
*Byron, September 17, 2026: "The row critic needs a parameter,
ROW_CRITIC_PICKINESS_IN_SPIKES. It should be an integer, probably 1 or 2," and
the same afternoon: "An epoch is going to be 35 ms at pickiness 2."* Under complement coding that pattern is the label's
fire-if-one population on and its fire-if-zero population off, every other
class the other way round (§5.11). Kept as one of the four critics (*Byron,
September 17, 2026: "evidence, row, graded"*, and at 15:12 MDT "keep the
class critic"). [RECORD §6.7, §8]

### 9.6 The graded critic

*Byron, September 15, 2026: "a graded critic it is."* On the same class
evidence the evidence critic reads, the reinforcement is the fraction of the other
$C - 1$ classes the label's class strictly out-spikes: 1 when it out-spikes
every one of them, 0 when it out-spikes none, and a near miss paid for what
it beat. A tie is not beaten, so a silent output zone scores 0. [RECORD §8]

### 9.7 The class critic, and the fraction right

On the same class evidence the evidence critic reads, the reinforcement is **1**
when the label's class strictly out-spikes every other class and **0**
otherwise. A tie is not a win, and a silent output zone is not a win.

The fraction of epochs a run wins on that rule is the **fraction right**, and
it is reported beside whatever reinforcement the run is paid, whether or not this
critic is the one paying. The measure and the critic are one rule: the number
reported is this critic's own.

*Byron, September 17, 2026, 15:12 MDT — "keep the class critic" — reversing
the three-critic list he gave at 15:09 MDT.* [RECORD §8, §1.3;
`docs/rewrite-answers.md` §4]

*What it requires of the engines.* It is read through the same function as
the critic's class evidence, so it costs an epoch one comparison and can
never disagree with the critic about what the output zone said. Every run reports it beside the
reinforcement it is paid, whichever critic pays: the class critic's number is
what the project is read by, so a run that does not report it cannot be read.
The lab notebook's Rust driver accumulated it over a run's last tenth and
only under the evidence critic; that is the notebook's, not this rule's.

### 9.8 The teacher's book: each neuron's firing-rate memory

The teacher keeps, for every neuron $j$, a running estimate $r_j$ of how
often it fires. After an epoch in which $j$ carried no driven mark (5.8), neither forced by the stimulus nor delivered to by the charged drive (5.4b; Byron, September 25, 2026),

$$r_j \mathrel{+}= \text{RATE\_MEMORY}\,\big(\mathbf{1}[j \text{ spiked this
epoch}] - r_j\big), \qquad \text{RATE\_MEMORY} = 0.01,$$

about the last hundred epochs. An epoch in which the stimulus forced $j$
says nothing about the network and moves nothing. A neuron with $r_j$ above
STUCK_ABOVE = 0.99 is **stuck on**, one with $r_j$ below STUCK_BELOW = 0.01
is **stuck off**.

$r_j$ is the neuron's own state (§2.1, §2.6) and the teacher is what moves
it, once an epoch, at the point §9.2 gives it; no rule of the neuron reads
it. The band is what gates un-sticking (§9.10), and it is kept whether or not
§9.9 and §9.10 run, because the report counts stuck neurons (§9.11). [RECORD §1.3, §6.7, §6.15]

*What it requires of the engines.* Every engine keeps the book the way the
object engine does — one elementwise update per neuron, once an epoch, at
the point §9.2 gives it — so no summation order enters and the three agree
to the bit. It is saved in the checkpoint with the network, so a resumed run
carries its rates and thresholds on.

### 9.9 Homeostasis — off unless a run asks

**Non-default.** Every epoch, every neuron that carried no driven mark that epoch (5.8: neither forced nor delivered to by the charged drive, 5.4b; Byron, September 25, 2026) has its
threshold moved toward its target firing rate:

$$\theta_j \mathrel{+}= \text{HOMEOSTASIS}\,(r_j - \text{TARGET\_RATE}),
\qquad \text{TARGET\_RATE} = 0.5.$$

Nothing clips where this takes a threshold (§0.5). HOMEOSTASIS is
$10^{-6}$, and the rule does not run unless a run asks for it: zero switches
it off, and that is where every run this specification states leaves it.
The constant keeps its value and the run switches it off, rather than the
constant being zero and a run supplying the value. Byron, September 17, 2026:
these are "latent knobs" — a rule that is off carries the value it would run
at, so turning it on is asking for it and not inventing it. The same holds of
un-sticking (§9.10) and the quash (§10.1).

The escape noise's width $\Delta_j$ is fixed at the neuron's *starting*
threshold, so a threshold this rule moves changes the margin the neuron
fires on and not the width it is drawn at.

*Kept as a non-default, Byron, September 17, 2026: "Homeostasis and
un-sticking, as non-defaults."* mnist asks for neither this nor §9.10
(*Byron, September 15, 2026: "Please turn off for this task"*). [RECORD
§1.3, §6.7, §8]

### 9.10 Un-sticking — off unless a run asks

**Non-default.** Every epoch, every neuron that carried no driven mark that epoch (5.8: neither forced nor delivered to by the charged drive, 5.4b; Byron, September 25, 2026) whose rate
memory is outside the stuck band of §9.8 — $r_j >$ STUCK_ABOVE or
$r_j <$ STUCK_BELOW — has its threshold moved toward a target rate, and only
while it is outside that band:

$$\theta_j \mathrel{+}= \text{UNSTICK}\,(r_j - \text{UNSTICK\_TARGET}),
\qquad \text{UNSTICK\_TARGET} = 0.5.$$

**Every neuron**, not the output zone alone (*Byron, September 14, 2026:
"all neurons are first-class citizens"*). Nothing clips where it takes a
threshold (§0.5). UNSTICK is $10^{-3}$, and the rule does not run unless a
run asks for it: zero switches it off, and the constant keeps its value as a latent knob
(§9.9). The run reports how many un-stickings have
fired (§9.11).

*Kept as a non-default, Byron, September 17, 2026, with §9.9.* [RECORD §1.3,
§6.7]

### 9.11 What is reported

[RECORD §6.14] Every run reports:

- spikes to date, and which neurons fired this epoch;
- the epoch's reinforcement, the mean reinforcement to date, and an exponential moving
  average over about WINDOW = 200 epochs — $\alpha = 2/(\text{WINDOW} + 1)$,
  started at the first epoch's reinforcement;
- the fraction right beside the reinforcement (§9.7) — as built, over a run's last
  tenth and only under the evidence critic; that it is reported at every
  reinforcement is Claude's reading (§9.7);
- how many neurons are stuck on and stuck off (§9.8), and how many
  un-stickings have fired (§9.10);
- at the end of a run, the mean reinforcement over its last tenth.

A per-epoch trace may be written to a file: one line an epoch of the epoch
number, the clock time in milliseconds, and the reinforcement. A sweep's arm
records its reinforcement trace every so many epochs and, at the end, each neuron's
rate memory, threshold, per-decision expectation $\hat p_j$ and decisions to
date, so that a continuation posts from where it was.

The network runs forever (§0.8, §12.12): these numbers are read as health,
not convergence, and drift is normal.

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

*What it requires of the engines.* Every engine that runs a problem with a
supervised direction measures this, as §12.1 requires of every clause. The
lab notebook's Rust path was the only one that did; it is not a precedent,
and no exception to §12.1 is claimed here.

It is a measurement and it changes nothing: no rule reads $d$, and a run
given no direction records none of this. A run resumed from a checkpoint
measures the change from the weights of its **first** start, and its trace
epochs count on from where it stopped. [RECORD §8]

### 9.13 The rate teacher — specified, not built

**Not in force.** This clause states what has been specified of the teacher
meant to replace the external teacher the specification dropped, and marks
what is not decided. Until the open parts below are settled, §9.1–§9.12 are
the rule and nothing is built from this.

The teacher's reinforcement is proportional to the difference between a
neuron's observed firing rate and its target rate:

$$\text{reinforcement} \;\propto\; r^{\text{obs}}_j - r^{\text{target}}_j.$$

As specified on September 17, 2026 this difference was soft-gated by the ISI
factor. The gate went with §7.4 on September 18 and is not carried; what
stands is the bare difference. The target for a coded bit of 1 is
**ONE_TARGET**, a constant of its own in §7.2's unit — spikes per hop —
ONE_TARGET $= 1$; and the target for a 0 is the **exploration rate**: the
hazard's rest rate (§7.2), which is new, is not zero, and is a set point
rather than an extreme. In that one unit the two are 1.000 and, at $N = 60$,
0.111 spikes per hop, against the 1.020 spikes per hop the refractory period
allows. (Under exploration at the synapse a neuron at rest fires at no rate of its own, §7.2, and the corresponding quantity is its synapses' whisper rate $h_0\,\kappa_i$; which the 0-target then takes is open.) Being a rate difference it needs no bit, so the row critic's pickiness
(§9.5) has no job under it.

**ONE_TARGET is quoted off nothing.** It is not the drive's rate, not an
interval, and not derived from the clock: it is a number of spikes per hop
that a run sets, and the default is the value the clause has always run at.
That is what makes where it sits a choice. At ONE_TARGET $= 1$ the target is
98% of the 1.020 the refractory period allows, which is where this clause
stood when the interval carried it, and the teacher it replaces is criticised
in this file for targets "at the two extremes of what a neuron can do"
[RECORD §6.9] — so 1 is a starting value, to be swept, and the range it is
swept over runs from the 0-target up to 1.020 and no further. Two points on
that range are already fixed by clauses in force and are where a sweep would
start: the drive of §5.6 gives a bit-1 neuron $\bar r = 80$ Hz, which is 0.408
spikes per hop, and §7.2's rest at $N = 60$ gives 0.111. The 1 the default
carries is the deterministic drive's rate, and that drive is not in force
(§0.12).

**Open. Each part blocks a build:**

- **Its name.**
- **Its observation window**, the window the observed rate is taken over.
  Three candidates stand in the constants: one interspike interval, a rate
  read off a single spike; the rate memory's hundred epochs (§2.6); or the
  epoch itself. The neuron keeps a rate memory over a window of its own
  (§2.6), and whether the teacher's window is that one or another is the
  same question asked twice.
- **Its critic**, or whether a rate difference implies one at all.
- **Whether it pays once at the horizon or continuously.** A teacher that
  posts per decision has no single reading instant, and if it posts
  continuously the epoch stops being a unit of learning (§3.9, §3.11) and
  the constants quoted per epoch — BASELINE_RATE, RATE_MEMORY — lose their
  unit.

**What it collides with, and what must be settled with it:**

- **§0.10.** That value says learning is paid by one global scalar and that no
  rule in force hands a neuron an error of its own. A rate difference is a
  per-neuron error. Either the differences are reduced to one scalar before
  anything is paid, or §0.10 is rewritten with this clause.
- **The 0-target is the firing clause's.** Setting it at the exploration rate
  makes the teacher depend on ESCAPE_DELTA (§6.4), a link §6 and §9 do not
  have today — and the rest rate carries the count's factor $\kappa(N)$
  (§6.6), so the target for a 0 moves when the network is resized. Whether a
  target that follows the network's size is wanted is open.
One collision the specification carried on September 17, 2026 is gone with
§7.4, and is named so that taking the gate up again re-asks it: that a neuron
which has never fired stands at $f = 0$, so a gated teacher posts nothing to
it and a network that starts silent cannot be taught out of silence. The other
is not gone but reduced. TARGET_ISI held three jobs, and that was the reason
the collision was recorded; REFRACTORY + LAG holds two — §3.2's hop and
§0.12's drive interval — and this clause's 1-target, which was briefly the
third, is ONE_TARGET and is quoted off nothing.

*Byron, September 17, 2026: the reinforcement rule and its soft gate are his,
specified and not built [RECORD §0.2]; the gate went out with §7.4 on
September 18 and the rule stands without it. The targets are settled in
direction and not in value. The old teacher this replaces, with its targets at the two
extremes of what a neuron can do and the risk Byron asked to be recorded on
September 14, 2026, is [RECORD §6.9] and is not carried. The four open parts
are decision 7 of `docs/rewrite-decisions.md`, unanswered; the unit of the
1-target, which that decision also carried, is answered above — Byron,
September 18, 2026: "REFRACTORY+LAG". The 1-target was quoted there for the
rest of that day and then separated, Byron the same evening: "Please separate
the 1-target from REFRACTORY+LAG ... ONE_TARGET = 1 (by default)." What the
separation buys is that the 98% is now a property of the default and not of
the specification: the constant can be swept without moving the clock.*


## 10. Local rules

A local rule needs nothing but the neuron's own spikes and the stamps on its
own synapses, runs inside the wave loop under whatever pays at the read, and
is off until a run asks for it (§9.1). The specification carries one.

### 10.1 The quash — off unless a run asks

**Non-default.** *Byron, September 13, 2026, turning the rule around:* when
a neuron fires itself, it has found a cycle. Cycles may be BAD. The whole
point of having an absolute refractory period is to quash short cycles.
Sustain may be exactly what we don't want a neural network doing. We may have
had the learning rule backward: cycles need to be quashed, so move the
contributing weights in the direction that will actually do this.

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

QUASH_RATE is 0.02 and QUASH_K is 0.2 per millisecond, and the rule does not
run unless a run asks for it: zero switches it off, and mnist leaves it off
(11.12). Both values are **open**: *Byron, the same day, "we will have to
explore this space."*

*Kept as a non-default, Byron, September 17, 2026.* [RECORD §6.11, §1.3]

*What it requires of the engines.* The quash runs at the end of the wave in
which the refire happened, over that wave's fired neurons, before the next
wave is scheduled. It reads only the neuron's previous spike time and each
incoming synapse's last-signal stamp, which every engine already keeps; no
reinforcement, no baseline and no teacher enters it, and it runs whether or not a
rule pays at the read.

### 10.2 The quash is the only local rule carried

This specification carries the quash alone. A local rule added later states,
in its own clause, where in the wave it acts relative to the quash, and is
off until a run asks for it. *The scope, Byron, September 17, 2026.* [RECORD
§6, the fixed within-wave order]

## 11. The problem

*The specification carries one problem. A problem is what a network is asked to do and how it is watched doing it: the layout, the inputs, and whether anything outside the network trains it [record §8].*

**11.1 mnist is the only problem.** The specification states one problem, mnist, and its layout, its stream, its critic and its reported measures are stated for it here. The run's own value stands first, then the problem's, then the constants of Appendix A (11.14).
*Byron, September 17, 2026, 15:02 MDT, setting the rewrite's scope: every problem but mnist leaves the specification and stays in the record. The eleven other problems the record poses are not rules and are not carried [record §8; `docs/rewrite-answers.md` §4].*

**11.2 The data.** MNIST is distributed as four IDX files, gzipped — images and labels for a training split and for a test split (LeCun, Cortes and Burges; [6] in `BIBLIOGRAPHY.md`). walnutbutter holds **two of them**, the training split's `train-images-idx3-ubyte.gz` (9,912,422 bytes) and `train-labels-idx1-ubyte.gz` (28,881 bytes), in `mnist/`; the data files are not in git (`.gitignore`), while `mnist/README.md`, which records the source, the sizes and the sums, is. They are fetched from the mirror TensorFlow uses, `https://storage.googleapis.com/cvdf-datasets/mnist/<file>`, and a file whose size or SHA-256 sum is not the recorded one is refused; the sums are in `mnist/README.md` beside the source and are checked on every fetch.
*Byron, September 15, 2026: "Let's set up a new task. This one will need its own data folder: mnist." [record §8]*

**11.3 The test split is not fetched.** There is no held-out set, because there is no evaluation run to hold one out for (12.12): the score is what the living network does on the stream it is given, and the loader knows one split.
*Byron, September 15, 2026: "we do not dare touch the test split. please unfetch it." [record §8]*

**11.4 From image to bits.** Each 28 × 28 image is averaged over 2 × 2 blocks to 14 × 14, and each block is on if and only if its mean intensity is at least half of full; the 196 bits are taken row-major. The ink's weight is not carried: the network is shown a binary picture.
*Byron, September 15, 2026, choosing binary bits at 14 × 14 from the choices offered [record §8, decision 1].*

**11.5 The input zone: 395 neurons.** In order: **three clock neurons**, whose coded bit is 1 every epoch, which take no raw bit and which the coding leaves alone; the **196 pixel bits**; and the **196 complements** of those bits. Bit for neuron, in that order, and nothing rearranges them. Exactly 199 of the 395 are driven every epoch, whatever the digit. Clock neurons are inputs and not outputs, so no read sees them.
*Byron, September 15, 2026, defining the zone — "three clock neurons, the 196 on-off pixels, and the 196 complement-coded pixels" — and the same day, defining the clock: "Clock neurons can be created for a task as input neurons always driven by 1." [record §4.3, §8]*

**11.6 The output zone: 60 neurons, complement-coded.** Ten classes, each with a population of three **fire-if-one** neurons and three **fire-if-zero** neurons: the ten fire-if-one populations first, in class order, then the ten fire-if-zero populations in the same order. A class's evidence is the sum over its fire-if-one population minus the sum over its fire-if-zero population, n_k = n_k⁺ − n_k⁻ — a spike from a zero neuron is a spike against.
*Byron, September 16, 2026, about 11:40 MDT: "In the output, I'd like to force complement coding. How about we try ten classes x a population of six neurons: three fire-if-one and three fire-if-zero?" Kept at 16:20 MDT: "we'll stick with the complement coding for now. It should not hurt us." The evidence rule $n_k = n_k^+ - n_k^-$ is Claude's reading of "accordingly", confirmed by Byron, September 17, 2026. [record §8, decision 6]*
*Engines: every engine reads the zone through one function, so the zone's arithmetic is written once and not three times.*

**11.7 The hidden count is the problem's.** mnist is posed on goo; the goo is its input zone, its hidden neurons and its output zone, and the hidden count is **199** unless the run says otherwise — 654 neurons at the default, 455 with none. Zero hidden neurons is a network the wiring must be able to build, the outputs hearing the inputs directly.
*Byron, September 16, 2026: "I would like to specify the number of 'hidden' neurons as hidden_neurons. One thing I neglected to do is benchmark this task without any hidden neurons. How will we know if they are buying us anything if they are always part of the economy?" [record §8]*

**11.8 The read is by count.** Each output neuron's spikes since the epoch began are its count — plus, under exploration at the synapse, its read synapse's ventured transmissions (§5.10, §7.9) — and the zone is read by those counts: the class evidence of 11.6 is built from them directly. Turning a count into a bit is the row critic's business alone (§9.5, its pickiness) and mnist does not ask it: its critic reads the counts themselves.
*Byron, September 14, 2026, setting the read: "COUNT the number of times each neuron fired in the epoch. ESTIMATE the firing rate based on the count." [record §4.3]*
*Engines: each engine snapshots its own spike counts at the epoch's reset, and the three are compared on them.*

**11.9 The critic is evidence, at TEMPERATURE.** The class sums are read as evidence at a temperature T: the estimate over classes is the Boltzmann distribution of the sums, and the reinforcement is its log score, with y the label —

> q_k = e^(n_k/T) / Σ_j e^(n_j/T),  r = ln q_y = n_y/T − ln Σ_j e^(n_j/T).

T = TEMPERATURE = 2. **T's value is open in Byron's own words** — "We will have to sweep for temperature eventually" — and 2 is the middle of the first sweep's {1, 2, 4}, held until the sweep sets it.
*Byron, September 16, 2026: "No, the spikes are EVIDENCE for now, not a proper maximum-likelihood estimator. We will have to sweep for temperature eventually." [record §8, decision 4]*
*Engines: one function pays every engine and the Rust driver alike.*

**11.10 The target is the label's code.** For a critic that wants a pattern rather than the sums — the row critic — mnist's target is the label: the label's fire-if-one population on and every other off, the fire-if-zero populations the other way round.
*[record §8; the code follows 11.6's layout.]*

**11.11 The learning rate is the problem's: LR 0.002.** It is taken unless the run gives a rate of its own.
*Byron, September 16, 2026, about 10:17 MDT: "default LR to 0.002 for this task." [record §8]*

**11.12 Homeostasis, un-sticking and the quash are off for this problem.** mnist sets all three rates to zero, and they stay zero unless the run says otherwise. They are non-defaults in any case; this clause is the problem's own setting of them. That mnist sets the quash off is how the problem is built rather than anything Byron said in words, and is to be confirmed in a word.
*Byron, September 15, 2026, the same evening he set the output populations: "Please turn off for this task." [record §8]*
*Engines: both are the Teacher's per-epoch book rather than the network's dynamics, so an engine that mirrors them mirrors zero.*

**11.13 The stream.** A run's patterns come from the train split in a seeded shuffle of its own, `random.Random(f"walnutbutter mnist {split} {seed}")`, every image once and then round again, with each image's label beside it; the network holds the label for the epoch and the critic reads it. The shuffle's seed is the run's `--input-seed` where one is given and the run's own seed otherwise, and it indexes the shuffle alone. The shuffle is not the network's stream, so seed *s* shows the same images in the same order to any network, and a run longer than the split cycles it.
*Byron, September 14, 2026, deciding the input stream, and September 15 extending it to a dataset with labels [record §4.5, §8].*

**11.14 What mnist does not fix.** The epoch's length, the container's starting threshold and floor, the wiring, the eligibility and the engine are not the problem's: they come from the constants of Appendix A or from the run that is asked for.
*Claude's reading of what the problem sets, confirmed by Byron, September 17, 2026: the record's mnist runs gave the epoch length, the threshold and the floor on the command line rather than in the problem [record §8].*

**11.15 The fraction right is reported beside the reinforcement.** mnist is paid on the evidence critic (11.9) and reports beside it the class critic's own number (§9.7): the fraction of epochs in which the label's class **strictly out-spikes every other class** on the evidence of 11.6 — a tie is not a win, and silence is not a win. As built that fraction is accumulated over a run's last tenth and only under the evidence critic; the WINDOW moving average a run reports is of the **reinforcement**, not of the fraction (§9.11).
*Byron, September 17, 2026, 15:12 MDT: "keep the class critic", under which the measure and the critic are one rule and not two [`docs/rewrite-answers.md` §4].*

**11.16 The estimator's correlation is reported.** For a synapse from input i onto an output of class k, the supervised direction is d_ij = P(coded input i on | class k) − P(coded input i on), taken over the training split; for a fire-if-zero output the sign is reversed, since that neuron should fire when the label is not k. A run reports the correlation of each input-to-output synapse's weight change since the run's start with d. It is an instrument for reading a run and not a rule: nothing in the network consults it, and no weight moves because of it.
*The instrument is Byron's instruction, September 16, 2026, 04:25 MDT: "Please modify the code so we can plot the correlation of the estimator over time after a run." The supervised direction it measures against is Claude's, from the earlier gradient check [record §0.1, §8].*

---

## 12. The invariants

*These hold of the system, not of any one engine. No clause of this file may be true in one engine and approximate in another.*

**12.1 One authority, and every platform built from it.** This file is the only source a platform is built from. Every clause is implemented in every engine that runs it, and is tested before it is built.
*Byron, September 17, 2026, 15:02 MDT, on keeping more than one engine: "keep, with tolerances, as a foundational rule; all platforms are built from one clear authority."*

**12.2 An engine that lacks a rule refuses.** It says so and refuses to run the configuration; it never approximates, and a gap is closed in the engine rather than run around.
*Byron, September 14, 2026: "rules in authority.md must be implemented cross-platform." [record §7]*

**12.3 The engines.** Three: the **object engine** (neurons and a queue of waves), the **array engine** (numpy vectors and a sparse matrix) and the **Rust wave loop** behind PyO3. The Rust loop is the default engine for any sweep or long run; the array engine is the second opinion and the fallback where Rust cannot run a configuration.
*Byron, September 14, 2026: "please always default to the rust engine unless it is broken." [record §6.15]*

**12.4 What agreement means, and the one tolerance.** The object engine and the Rust loop agree **to the bit**: every spike, every score and every weight, compared with `==`. The array engine agrees **to a part in a billion** on continuous quantities — scores and weights — because it sums a wave's arrivals in the matrix's order where the other two add them in push order (12.5), and its exponentials are numpy's rather than libm's; the tests hold weights, thresholds and rates to $10^{-12}$ absolute. A part in a billion is of what a value is made from: a score a settle makes as the difference of two terms — $x\,G - B$ (§8.16), or 8.11's $x\,E - B$ — agrees to a part in a billion of the larger term, since two equal terms cancel to rounding the engines round differently, 0 in one and a few units in the last place in the other. The array engine's agreement is held over a short run, the tests' twenty epochs: with learning on, a weight's last-bit difference feeds back into the potentials and grows, and over hundreds of epochs the arrays part from the objects beyond this tolerance and at last on a spike. Even inside twenty epochs about one learning run in a hundred parts — twelve of 1,280 measured over forty seeds — on a weight or a potential near zero, where a part in a billion of itself is finer than the rounding it has carried; none has parted inside twenty epochs on a spike, a read count, a mark, the queue or the stream, and the tests name the runs that part rather than choose seeds that do not. Sameness over a run of any length is the object engine's and the Rust loop's, to the bit; the arrays keep numpy's exponentials, libm's halving their partings at two to three times their cost. Spikes have agreed on every configuration compared, and are not guaranteed to: a potential within an ulp of the decision can part two engines, which is why 12.8 requires the comparison on the configuration. That is the whole of the tolerance, and it is named here rather than left in a test.
*Byron, September 17, 2026, 15:02 MDT, keeping the array engine "with tolerances"; the tolerance's two sources are the record's [record §6.15, §7]. Byron, September 25, 2026, when the array engine took up exploration at the synapse: a cancelled score measured against its terms, the agreement held over a stated horizon, numpy's exponentials kept; and the same day, once the horizon was measured, that the file says the one run in a hundred.*

**12.5 The summation order is part of the rule.** Signals due at one moment are summed in **push order**, the order the topology was built in, so an engine that flattens the topology differently sums a wave differently and lands on different bits. Every engine that claims bit agreement takes the object engine's push order.
*[record §6.15.]*

**12.6 One stream, one order.** Every draw a run makes comes from Python's MT19937 in one order. An engine outside Python takes that generator's state, draws the same uniforms in the same order at the same named point in the wave, and hands the state back, so Python's stream carries on from where the other engine left it. The draws are equal, not approximately equal.
*Byron, September 14, 2026, refusing an engine that would have approximated them: "rules in authority.md must be implemented cross-platform." [record §6.15]*

**12.7 A seed is the whole run.** One seed number fixes three streams, and there are no others: the network's own stream, which draws the wiring, the weights and the drive's Poisson arrival times; the exploration stream, seeded from the same number, which draws one uniform per neuron per wave — or, under exploration at the synapse, one per synapse and one per output (§3.8) — and is the stream an engine outside Python takes over and hands back (12.6); and the input stream, separate, which draws the epoch's patterns (11.13) so that a seed shows the same patterns in the same order to any network however it was built.
*Byron and Cedric, standing; the input stream separated by Byron, September 14, 2026 [record §4.5, §7].*

**12.8 Agreement is proven on the configuration, not carried over.** The engines are compared on the container, the wiring, the drive and the read a run will actually use, before that run is read as evidence about anything.
*Claude's reading of two cases the record holds, confirmed by Byron, September 17, 2026 [record §5.2, §6.15, §7].*

**12.9 Checkpoints round-trip.** A checkpoint rebuilds the network from its seed and its settings and reloads what the run reached: weights, thresholds and floors, potentials, the clock, spike times, each neuron's firing-rate memory and its decision width, synapse stamps, signals in flight, each neuron's per-decision expectation of its own spike and its expected spikes and decision count, and each synapse's open-arrival note; under exploration at the synapse, each neuron's gain, each output's read count this epoch, the ventured marks on the signals in flight, and the settings of §7.1, §7.6, §7.7, §8.17 and 5.4b — the exploration, the family and rest hazard, the scaling, the trace, the drive and DRIVE_STEPS (§8.14). It also carries the state of all three of the run's streams and the reinforcement baseline, which is what §12.11's exactness rests on. It round-trips in every engine, and a network saved under a setting resumes under it unless the resuming run overrides it explicitly. The exploration is the exception: a checkpoint saved under one resumes under that one only, and a run that would resume it under the other is refused (§12.2), nothing mapping the neuron rule's per-decision bookkeeping onto the gain or back (Byron, September 25, 2026; `docs/synapse-build-map-2026-09-24.md` §5, Q17). *Of the engines:* an engine outside Python takes the exploration stream's state and hands it back (§12.6), so the state a checkpoint stores is the one the next draw comes from whichever engine ran the epoch before it.
*Byron and Cedric, standing; extended September 17, 2026 with the single-spike rule's state [record §7, §0.2].*

**12.10 A rule is recomputed, a state is stored.** What a checkpoint can derive from the network's settings is derived — the escape scale is recomputed from the count, since it is a rule and not a state — and what the run moved is stored as the run left it, thresholds and floors included.
*[record §5.2, §7.]*

**12.11 A resume is exact.** A run resumed from a checkpoint produces, bit for bit, what the uninterrupted run would have produced: the same spikes at the same waves, the same weights, the same reinforcement, epoch after epoch, until the two are stopped. A resume is not a new run that starts where an old one left off; it is the same run, continued. The network is rebuilt under its own layout and checked neuron for neuron and synapse for synapse against a fresh build from the seed, and refused if they differ.

What that requires is §12.9's list, and the part of it a resume alone needs is the state of every stream the run draws from: the exploration stream that supplies the firing decisions (or the synapses', §7.5), the network's own stream that supplied the wiring and the weights and goes on supplying the drive's arrival times, and the input stream that draws the epoch's patterns. A checkpoint carries each generator's state, not its seed and a count of draws, so the next number is the number the uninterrupted run would have taken. The reinforcement baseline is restored with them (§9.3).
*Byron, September 17, 2026: "I would like resumes to be exact ... the resumed run reproduces, bit for bit, what the uninterrupted run would have done ... It's worth having." This replaces the lab notebook's resume, which started the exploration stream afresh at the driver's seed + 1,000,000, started the baseline from the first resumed epoch, and advanced the input stream rather than restoring it — so a resumed arm there was not comparable to an uninterrupted one [record §7, §8].*

**12.12 The network keeps living.** There is no training run and no evaluation run, only one run that keeps going. No rule may assume an end, no result is read as convergence, and a run is read as health.
*Byron and Cedric, standing. Its consequence for the data is 11.3: there is no held-out set, because there is no evaluation run to hold one out for [record §7, §6.14].*

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
| WEIGHT_RANGE | [−1, 1] | the range a weight is drawn uniformly from at the build, and clipped to by learning | §1 [record §1.1] |

**The neuron, its clock and its firing**

| constant | value | what it fixes | owned by |
|---|---|---|---|
| TAU | ∞ | the potential does not leak: the evidence accumulator. 2 ms is the leak, kept as a per-run option with its analysis | §2 [record §1.2, §5.1; Byron, September 17, 2026: "neuron: NOT leaky, but please leave the leak option with its analysis"] |
| REFRACTORY | 5 ms | the absolute refractory period | §2.5 [record §1.2] |
| LAG | 0.1 ms | how long after a target could take it a signal actually arrives, so no delivery lands on the end of a refractory period | §3.2 [Byron, September 17, 2026] |
| TOLERANCE | 10⁻¹² | the clock's slack, relative: two moments within slack(t) = TOLERANCE × max(1, \|t\|) are one moment | §3.4 [Byron, September 17, 2026] |
| HOP | 2.55 ms | **derived:** (REFRACTORY + LAG) / 2. How long a signal takes to travel any connection, and the only delay in signalling. A delay, not a decay. Two hops are REFRACTORY + LAG = 5.1 ms, which is what clears a neuron's own return of its wall | §3.2 [Byron, September 17, 2026; TIME_CONSTANT_OF_TRANSMISSION retired September 18, 2026; halved and named HOP September 19, 2026] |
| INTERVAL | 35 ms | the epoch's length: the spacing of inputs when no time is given | §3 [record §1.2, §4.2] |
| THRESHOLD | 0.25 | the starting θ of a container that does not set its own, quoted at THRESHOLD_FAN_IN | §6 [record §1.2, §5.2] |
| GOO_THRESHOLD | 0.2 | goo's starting θ, quoted the same way | §6 [record §1.2] |
| THRESHOLD_FAN_IN | 18 | the in-degree θ and the floor are quoted at: a container that scales starts neuron j at θ · d_j / 18 | §6 [record §5.2] |
| MINIMUM_POTENTIAL | −1 | the floor on the potential, quoted at THRESHOLD_FAN_IN and rescaled with θ, holding the floor at −4 θ | §6 [record §1.2, §5.2] |
| GOO_MINIMUM_POTENTIAL | −0.8 | goo's floor, at the same ratio to goo's threshold | §6 [record §1.2] |
| ESCAPE_DELTA | 0.455 | the width of the firing decision, in units of the neuron's starting threshold; 0 whenever EXPLORATION is synapse (§6.13) | §6 [record §1.2, §5.2] |
| ESCAPE_REFERENCE_COUNT | 60 | the count the width is quoted at: every hazard runs at √(60/N) | §6 [record §1.2, §5.2] |

**The drive and the read**

| constant | value | what it fixes | owned by |
|---|---|---|---|
| INPUT_DRIVE | rate | how a bit becomes spikes: an independent Poisson process drives each input neuron across the presentation window; charged, a run option under exploration at the synapse only, delivers θ/DRIVE_STEPS per arrival instead, at DRIVE_STEPS times the rate (5.4b) | §5 [record §1.2, §4.3; 5.4b, Byron, September 24, 2026; DRIVE_STEPS a run option and the charged drive refused under the neuron rule, September 25, 2026] |
| PRESENTATION_TIME | INTERVAL | how far into the epoch the drive runs; the default is the whole of it, and more than INTERVAL is refused | §5.4a [Byron, September 18, 2026] |
| INPUT_CV | 0.6 | how that drive is specified: the coefficient of variation of the train it produces, from which the rate follows | §5 [record §1.2, §4.3] |
| INPUT_RATE, INPUT_RATE_OFF | 0.133, 0 /ms | the same drive in the other coordinate: the rates of a bit-1 and a bit-0 neuron's process | §5 [record §1.2] |
| ROW_CRITIC_PICKINESS_IN_SPIKES | 2 | the spikes an output neuron must fire in an epoch to count as on for the row critic, and for nothing else; an integer. mnist's critic reads the counts themselves (11.8) | §9.5 [Byron, September 17, 2026] |
| POPULATION | 3 | the neurons per class in each half of a complement-coded output zone (11.6) | §5.11 [record §1.3] |
| READ_WINDOW | 5 ms | **non-default read.** How recently before the read a spike must have fallen to count, under the *window* read of §5.10 | §5.10 [record §1.3] |
| RATE_TAU | 5 ms | **non-default read.** The exponential window a neuron's own firing-rate estimate is taken over, under the *rate* read of §5.10 | §5.10 [record §1.3] |
| RATE_ON, RATE_OFF | 200, 0 Hz | **non-default read.** What the *rate* read of §5.10 scores a 1 and a 0 against: saturation, which is 1/REFRACTORY, and silence. These are the two extremes §9.13 was written to leave behind, and they stay because the read that uses them stays | §5.10 [record §1.3; Byron, September 19, 2026] |

**Learning**

| constant | value | what it fixes | owned by |
|---|---|---|---|
| LR | 0.03 | the learning rate where the problem names none (mnist names 0.002, 11.11) | §8 [record §1.3] |
| ELIGIBILITY | hazard | what the reinforcement acts on: hazard, the score of the escape decision, or hebb, the neuron's own expectation. The record's stored value was perturb, which leaves the specification, so the constant becomes hazard — what the neuron already chose under escape noise; where the threshold decides and nothing else draws, no eligibility runs and the rule refuses (§8.3); under exploration at the synapse the hazard is posted at the synapses' decisions (§8.16) | §8.3 [record §1.3; the hazard default is Claude's reading, to be corrected in a word] |
| CRITIC | row | how the reinforcement is judged where the problem names none: row, class, graded or evidence (mnist names evidence, 11.9) | §9.4–§9.7 [record §1.3] |
| TEMPERATURE | 2 | the evidence critic's T; open — "We will have to sweep for temperature eventually" | §9.4 [record §1.3, §8] |
| BASELINE_RATE | 0.05 | the per-epoch update of the running reinforcement baseline the advantage is taken against | §8 [record §1.3] |
| DECISION_MEMORY | 10⁻⁴ | the per-decision update of a neuron's expectation of its own spike; a starting value, to be swept | §8.6 [record §1.3, §6.7] |
| RATE_MEMORY | 0.01 | the per-epoch update of a neuron's own observed rate, which homeostasis, un-sticking and the stuck bands read | §2.6 [record §1.3] |
| QUASH_RATE | 0.02 | **non-default.** The fraction of its weight a refire weakens each contributing synapse by, when a run asks for the quash | §10.1 [record §1.3, §6.11] |
| QUASH_K | 0.2 /ms | how fast that quash falls off with the delay since the previous spike | §10.1 [record §1.3] |
| HOMEOSTASIS | 10⁻⁶ | **non-default.** The per-epoch rate a threshold drifts toward its target firing rate; zero for mnist (11.12) | §9.9 [record §1.3] |
| TARGET_RATE | 0.5 | the firing rate homeostasis aims for | §9.9 [record §1.3] |
| UNSTICK | 10⁻³ | **non-default.** The per-epoch rate a stuck neuron's threshold moves toward its target; zero for mnist (11.12) | §9.10 [record §1.3] |
| UNSTICK_TARGET | 0.5 | the firing rate the un-sticking aims for | §9.10 [record §1.3] |
| ONE_TARGET | 1 spike per hop | **not in force.** The rate teacher's target for a coded bit of 1, in §7.2's unit and quoted off nothing else; a starting value, to be swept, with 1.020 the most the refractory period allows | §9.13 [Byron, September 18, 2026] |

**The problem, and what is reported**

| constant | value | what it fixes | owned by |
|---|---|---|---|
| mnist: side, block | 28, 2 | the image as distributed, and the block averaged to one bit | 11.4 |
| mnist: bits, classes | 196, 10 | the bits an image becomes, and the classes | 11.4, 11.6 |
| mnist: binarisation | 0.5 | a block is on iff its mean is at least half of full | 11.4 |
| mnist: clock | 3 | the clock neurons at the front of the input zone | 11.5 |
| mnist: inputs, outputs | 395, 60 | the two zone widths | 11.5, 11.6 |
| mnist: hidden neurons | 199 | the hidden count unless the run says otherwise | 11.7 |
| mnist: lr | 0.002 | the problem's learning rate | 11.11 |
| mnist: homeostasis, unstick | 0, 0 | both off for this problem | 11.12 |
| STUCK_BELOW, STUCK_ABOVE | 0.01, 0.99 | the band outside which a neuron is stuck: what gates un-sticking, and what the reports count | §9.10 [record §1.3] |
| WINDOW | 200 | reporting: the epochs the reported moving average of the **reinforcement** spans | §9.11 [record §1.3, §6.14] |

**Constants no clause yet claims,** named here so they are decided rather than lost: PROBLEM (the default problem, still "reversal" in the code, a problem the specification does not carry — it becomes mnist or it goes); ACROSS (8 — the zone width a goo takes when no problem names one; mnist names its own); WEIGHT_EPSILON (0.001 — under `--positive-weights` the weight range becomes [ε, 1], a network with no inhibition); RULE (the record's four rules are one now that only the reinforce rule survives); TARGET (the record's default is the reversed pattern, whose problem is dropped; mnist's target is the label, 11.10); LATE (what a signal arriving after its target fired earns — it does not apply under either surviving eligibility).

---

**Exploration at the synapse (§7.1, §7.5–§7.9, §8.16–§8.17, 5.4b) — specified September 24 and 25, 2026, being built**

| constant | value | what it fixes | owned by |
|---|---|---|---|
| EXPLORATION | neuron | what explores: the neuron's escape noise (§6.5) or its synapses' (§7.5); synapse becomes the default once §7.6's conditions are met, and a positive neuron width given explicitly then asks for neuron | §7.1 [Byron, September 25, 2026] |
| SYNAPSE_HAZARD_REST | 0.01 | the rest hazard $h_0$, per hop before the scaling: what a synapse of a source at zero potential speculates at; in $[0, 1)$, and 0 under the loglinear family spikes as the deterministic network does | §7.6 [Byron, September 24, 2026; the range, and 0 no switch of the exploration, September 25, 2026] |
| SYNAPSE_HAZARD_FAMILY | loglinear | $h(u) = h_0^{1-u}$; a run may say linear, $h_0 + (1 - h_0)\,u$ | §7.6 [Byron, September 24, 2026] |
| SYNAPSE_HAZARD_SCALING | count | $\kappa(N)$ on every synapse hazard; a run may say fan-out, $\kappa(N)/F_i$ | §7.7 [Byron, September 24, 2026] |
| TRACE | all | what a trace counts: every delivery, or the ventured ones alone | §8.17 [Byron, September 24, 2026] |
| DRIVE_STEPS | 3 | under the charged drive, the deliveries that take an input from rest to threshold, or one more where their sum falls a rounding short of it; each is θ/DRIVE_STEPS on the threshold the input holds when the delivery lands, at DRIVE_STEPS times the rate; a run option | 5.4b [Byron, September 24, 2026; a run option, and the amount at delivery, September 25, 2026] |

## Appendix B — building the engine

**The engine is built for the machine that will run it.** `rust/.cargo/config.toml` carries `-C target-cpu=native`, so a build uses everything the chip has rather than a baseline processor nobody here owns. Both of us build on our own machines, which is what makes this the right default rather than a preference; it would have to be revisited only if the engine were ever built somewhere neither of us controls.

*Byron, September 19, 2026: "We both want native CPU optimizations when we build things. That is the right way to do things: Build them for the target system."*

**Native does not cost the bit agreement of 12.4.** Rust keeps IEEE semantics operation by operation: it never fuses a multiply-add and never reorders a float reduction, so wider registers change which instructions do the arithmetic, not the arithmetic itself. Measured September 19, 2026 on a 7950X3D: the built engine carries AVX-512 and AVX2 instructions and no fused multiply-add at all, the object engine and the Rust loop agreed to the bit over 300 epochs, and the wave loop ran about 7% faster than the same source built for the baseline. A change of build flags is checked with that comparison (12.8), not assumed to be safe.
