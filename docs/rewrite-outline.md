# The rewrite: the outline, what it drops, and a draft of the first section (September 17, 2026)

*Working notes for the rewrite of `AUTHORITY.md`, produced by reading `RECORD.md` section by section (ten agents: eight readers, a drafter and a critic). Not the specification: the specification is `AUTHORITY.md`, and nothing here is decided until Byron says so. The decisions this raises are in `rewrite-decisions.md`.*

## What the readers found

| part of the record | clauses | kept | superseded | open | record-only |
|---|---|---|---|---|---|
| zero | 28 | 11 | 6 | 6 | 5 |
| constants | 69 | 12 | 18 | 30 | 9 |
| substance | 38 | 16 | 9 | 11 | 2 |
| signalling | 52 | 27 | 10 | 10 | 5 |
| activation | 40 | 15 | 12 | 5 | 8 |
| learning1 | 56 | 20 | 15 | 19 | 2 |
| learning2 | 51 | 21 | 10 | 12 | 8 |
| problems | 50 | 19 | 5 | 14 | 12 |
| **total** | **384** | **141** | **85** | **107** | **51** |

## The proposed outline

# AUTHORITY.md — proposed table of contents

*Preamble (already written, 27 lines): the file wins over the code; Byron and Cedric own the substance; how a clause is written. Unnumbered.*

## §0 Standing values — non-negotiable
Values, not mechanisms. No clause below may contradict one.

- 0.1 The inputs are the outputs: input, hidden and output are defined only by where external connections are, and the neurons operate identically — **kept** [§0 bullet 1]
- 0.2 A neuron integrates delta functions to determine its potential — **kept** [§0 bullet 2]
- 0.3 The potential does not leak; the impulse response is infinite — **kept** [§0 bullet 2 as first written; §5.1]
- 0.4 A neuron that fires is absolutely refractory, ignoring its inputs and not integrating them; this is a computational feature — **kept** [§0 bullet 3]
- 0.5 Exploration belongs to the synapse: a synapse explores its own impulse response — **kept** [§0.3]
- 0.6 Learning is paid by one global scalar, produced locally and consumed globally — **open** [§0 bullet 4]
- 0.7 Structures are permissive; nothing is restricted artificially — **kept** [§2; §3.4]
- 0.8 As few knobs as possible, a stable default regime, and scale built into a rule rather than swept — **kept** [§1.2; §5.2]
- 0.9 The goal is the substance, not a task; a task only watches whether the substance is alive — **kept** [§2]
- 0.10 The network keeps living: no rule may assume an end — **kept** [§7]
- 0.11 Struck from §0, with the words that struck them: the threshold sentence and the dopamine timing law — **kept** (as a deprecation record) [§0 bullets 2 and 4]

## §1 The neuron
- 1.1 A neuron has a potential p_j, a last-spike time, a previous-spike time, a rate estimate r_j and a per-decision expectation of its own spike p̂_j — **kept** [§0 notation; §6.7]
- 1.2 p_j is the sum of the weights of the arrivals it integrated since its last spike, undiminished — **kept** [§5.1]
- 1.3 No decay term is evaluated anywhere in any engine; TAU is deprecated — **kept** [§5.1; §1.2]
- 1.4 What a spike does to the potential (reset to zero, subtract a level, or nothing) — **open** [§0 bullet 2; §5.2]
- 1.5 Whether the potential is bounded below, and in what unit — **open** [§1.2 MINIMUM_POTENTIAL; §5.2]
- 1.6 A neuron that fired at t is refractory while t < t_fired + REFRACTORY; it ignores every signal and does not integrate it — **kept** [§5.3]
- 1.7 A refractory neuron cannot be forced — **kept** [§5.3]
- 1.8 r_j, the neuron's own observed rate, and the window it is estimated over — **open** [§1.2 RATE_TAU; §1.3 RATE_MEMORY]

## §2 The synapse
- 2.1 Every connection is one-way from source to target and carries its own weight; a pair of neurons has two, with independent weights — **kept** [§3]
- 2.2 Connections take ids from 1 in the order they are made — **kept** [§3]
- 2.3 Whether a neuron may connect to itself — **open** [§3]
- 2.4 A weight is drawn uniformly from WEIGHT_RANGE, in connection-id order, from the seed's stream — **kept** [§3.3]
- 2.5 What bounds a weight once learning moves it — **open** [§3.3; §1.1]
- 2.6 A synapse carries its own trace x_ij: the count of arrivals its target integrated since its target's last spike — **kept** [§5.1; §6.7]
- 2.7 A synapse carries its open-arrival note B_ij and the score e_ij charged since the last read — **kept** [§6.7]
- 2.8 Each connection stamps the time of the last signal its target actually integrated — **kept** [§4.4]

## §3 The clock
- 3.1 Time is in nominal milliseconds — **kept** [§4.1]
- 3.2 A signal takes one hop, h = REFRACTORY / REFRACTORY_HOPS; at REFRACTORY_HOPS = 2 a hop is 2.5 ms and the refractory period is two hops — **kept** [§4.1]
- 3.3 There is no delay other than the hop — **kept** [§4.1]
- 3.4 A wave is everything scheduled for one time t; the queue is a time-ordered schedule, not a per-hop loop — **kept** [§4.1; §4.4]
- 3.5 Signals sharing a timestamp sum before any neuron fires, so no decision depends on arrival order — **kept** [§4.1; §4.4]
- 3.6 Times are rounded to a nanosecond — **kept** [§4.1]
- 3.7 A wave has two phases, deliver then fire; a firing neuron schedules its outgoing signals at t + h — **kept** [§4.4]
- 3.8 A cascade may still be running when the next input lands; the epoch is not a barrier — **kept** [§4.1]
- 3.9 An epoch is one input and the schedule run to the horizon t_e + INTERVAL; signals due at or after it wait — **kept** [§4.2]
- 3.10 At reset the fired-this-epoch state clears; potentials, spike times and signals in flight are kept — **kept** [§4.2]
- 3.11 INTERVAL's value, and whether a problem may set it — **open** [§1.2; §4.2]
- 3.12 Whether the epoch is still the learning cadence or only a reporting boundary — **open** [§4.2; §6.7]

## §4 The container and its wiring
- 4.1 A container decides only where neurons sit and what connects to what; every container shares the same neurons, synapses, clock and learning — **kept** [§2]
- 4.2 Which containers exist — **open** [§2; §3.1; §3.2]
- 4.3 Goo is N neurons with no positions, wired by a probabilistic rule over ordered pairs — **kept** [§2; §3.4]
- 4.4 Goo's zones are named by index, an input width and an output width, computed from three integers — **kept** [§3.4]
- 4.5 Whether a zone must be a prefix and a suffix, or may be any set of neurons — **open** [§3.4]
- 4.6 Zones may overlap; only a count below the input width is refused, and each wiring states its own refusal — **kept** [§2; §3.4]
- 4.7 The scaled rule: P(i→j) = 0 if i = j, 0 if both are in the input zone, 0 if i is in the output zone, else min(1, N·s / A_j) — **kept** [§3.4]
- 4.8 The scaling factor is the wiring's knob, 0.05 — **kept** [§1.2; §3.4]
- 4.9 ff2: every input projects onto every output and nothing else; a goo with hidden neurons is refused — **kept** [§3.4]
- 4.10 ff2-partial: P on each input→output pair, and at P = 1 it is ff2 to the bit — **kept** [§3.4]
- 4.11 Which wiring is the default, and whether equal fan-in is still required — **open** [§3.4; §0.4]
- 4.12 Where a probability is 0 or 1 nothing is drawn; otherwise pair by pair in (i, j) order, the projection draw then the weight — **kept** [§3.4]
- 4.13 A checkpoint restores a goo under the wiring it was built with — **kept** [§3.4]
- 4.14 Whether there is a default network count, or the count comes from the problem's zones — **open** [§1.2 GOO_COUNT]

## §5 The drive and the read
- 5.1 The network's input is its input zone and its output its output zone — **kept** [§4.3]
- 5.2 A problem names its own zone widths — **kept** [§1.1; §3.4]
- 5.3 Inputs are drawn up front from a stream of their own, never the network's; a dataset is such a stream with labels — **kept** [§4.5]
- 5.4 A pattern is complement-coded, so exactly half the zone is driven whatever the bits — **kept** [§4.3]
- 5.5 A problem may repeat each bit over POPULATION neurons — **kept** [§4.3]
- 5.6 Clock neurons: input neurons whose bit is 1 every epoch — **kept** [§4.3]
- 5.7 The drive is one deterministic spike every TARGET_ISI on each driven neuron, presented all at once — **open** (specified September 17, not built) [replaces §1.2 INPUT_DRIVE]
- 5.8 What an undriven (bit-0) input neuron does — **open** [§1.2 INPUT_RATE_OFF]
- 5.9 What "on" means at the read, if a bit read survives at all — **open** [§4.3]

## §6 Firing
- 6.1 What fires a neuron — **open** [§0 bullet 2; §5.2]
- 6.2 θ is deprecated, and with it the quoting of any constant in threshold units and the rescaling of the potential axis by fan-in — **kept** (a deprecation clause) [§5.2; §1.2]
- 6.3 A spike is remembered, along with the spike before it — **kept** [§5.2]
- 6.4 A neuron may fire any number of times, the refractory period permitting — **kept** [§4.4]
- 6.5 Whether anything may still force a spike, now that the drive is deterministic — **open** [§5.2; §4.3]

## §7 Exploration
- 7.1 Exploration is generated at the synapse: the exploring object and the credited object are the same object — **kept** [§0.3]
- 7.2 The hazard: per synapse or per neuron, per hop or per refractory period, and at what rate — **open** [§0.3; §0.4]
- 7.3 What becomes of an exploration event whose target is refractory — **open** [§0 bullet 3; §0.4]
- 7.4 Whether a network's exploration still falls as the square root of its size — **open** [§5.2 ESCAPE_REFERENCE_COUNT]
- 7.5 A neuron nobody talks to must still be able to spike — **kept** (the need; the mechanism is 7.2) [§5.4; §1.3 UNSTICK]
- 7.6 Every exploration draw is taken from one stream, in one order, at one named point in the wave, in every engine — **kept** [§5.2; §6.15]

## §8 The learning rule
- 8.1 Which eligibility the system runs — **open** [§1.3 ELIGIBILITY; §6.7]
- 8.2 The single-spike rule: at every decision of neuron j, each incoming synapse takes e_ij += (c_j − q_j)·x_ij — **kept** [§6.7]
- 8.3 Under hebb, c_j is 1 at a spike and 0 at a silence, and q_j is p̂_j at every decision — **kept** [§6.7]
- 8.4 p̂_j is set by the first decision and charges nothing; after every decision, charged first, it moves by max(DECISION_MEMORY, 1/n)(y − p̂_j) — **kept** [§6.7; §1.3]
- 8.5 DECISION_MEMORY's window — **open** [§1.3]
- 8.6 An arrival stays open until its target spikes, so a debit runs across reads — Byron: "Let them run!" — **kept** [§6.7]
- 8.7 What each event costs: an arrival counts one and notes E_j; a decision adds q_j to E_j; the spike settles every open arrival and clears; the read pays and clears the score — **kept** [§6.7]
- 8.8 Whether hebb's credit at the spike is a full unit or discounted by the expectation — Byron: "Defer for now" — **open** [§6.7]
- 8.9 The ISI factor f(t − I) = (3x − 1)/(1 + x³), x = t/I: −1 at t = 0, 1 at t = I and nowhere higher — **kept** [§0.2]
- 8.10 Every engine computes x = (t′ − t_fired_j)/I, then (3x − 1)/(1 + x·x·x), in that order — **kept** [§0.2]
- 8.11 The factor is on by default, in the library constructor as well as the command line — **kept** [§0.2]
- 8.12 What the factor weighs (the teacher alone, or every charge), and what t is measured from — **open** [§0.2]
- 8.13 What reaches a neuron that has never fired, where f = 0 — **open** [§0.2]
- 8.14 TARGET_ISI = 5.1 ms, hardcoded for now — **open** [§0.2; §1.3]
- 8.15 The update: w_ij moves by LR · A · e_ij at the read, bounded per 2.5 — **kept** [§6.7; §6.9]
- 8.16 LR's default, now that the injected noise has as many dimensions as there are synapses — **open** [§1.3; §0.3]
- 8.17 A running baseline is subtracted to give the advantage A — **kept** [§1.3 BASELINE_RATE]
- 8.18 Whether weights forget on their own — **open** [§1.3 WEIGHT_DECAY; §6.8]
- 8.19 Whether a refire weakens the synapses that caused it — **open** [§1.3 QUASH_RATE; §6.11]

## §9 The teacher
- 9.1 The teacher's name — **open** [Byron, September 17, 2026]
- 9.2 The teacher's reinforcement is proportional to (observed rate − target rate), soft-gated by the ISI factor — **open** (specified, not built) [§0.2; replaces §6.9]
- 9.3 The target is the drive rate for a 1 and the exploration rate for a 0 — **open** [replaces §1.2 RATE_ON / RATE_OFF]
- 9.4 The observation window the observed rate is taken over — **open** [§1.2 RATE_TAU; §1.3 RATE_MEMORY]
- 9.5 The critic — **open** [§1.3 CRITIC]
- 9.6 Whether the teacher pays once at the horizon or continuously — **open** [§4.2; §6.9]
- 9.7 A network is reported and read as health, not as convergence; drift is normal — **kept** [§6.14; §7]

## §10 The problems
- 10.1 A problem is what a network is asked to do and how it is watched: the layout, the inputs, and whether anything outside trains it — **kept** [§8]
- 10.2 Which problems exist — **open** [§8; §1.2 PROBLEM]
- 10.3 copy: four raw bits, complement-coded onto the input zone, unpermuted; output place i shows coded bit i — **open** [§8]
- 10.4 mnist: 14×14 binary pixels; 3 clock + 196 pixels + 196 complements in, five neurons a class out — **kept** [§8]
- 10.5 mnist's hidden count is the problem's, and zero hidden is a network the wiring must be able to build — **kept** [§8]
- 10.6 The test split is not fetched: there is no evaluation run and so no held-out set — **kept** [§8; §7]
- 10.7 Whether a problem may switch off a clause of this file — **open** [§8]

## §11 The invariants
- 11.1 Which engines are built — **open** [§7; §6.15]
- 11.2 Every clause is implemented in every engine that runs it, and tested before it is built — **kept** [preamble; §7]
- 11.3 An engine that lacks a rule says so and refuses; it never approximates — **kept** [§7]
- 11.4 The engines are compared with ==; any tolerance is a named exception naming its clause — **open** [§7; §0.2]
- 11.5 A seed is the whole run: wiring, weights, inputs and exploration all come from it — **kept** [§7]
- 11.6 A non-Python engine takes Python's MT19937 state, draws in the same order, and hands it back — **kept** [§6.15]
- 11.7 A checkpoint round-trips, and a resumed network keeps the settings it was saved under unless the resuming run overrides them explicitly — **kept** [§7; §0.2]
- 11.8 Where the constants live when more than one engine can build a network — **open** [§1]

## Appendix A — the constant register
- A.0 One value to a name, one home for every value, and every row names the clause that owns it; a literal anywhere else is a bug — **kept** [§1] *(the register itself is not a clause and carries no rules)*

## What the specification would deliberately drop

## What the new specification deliberately does not carry

Everything below stays in `RECORD.md`, which holds the quotations and the measurements. Nothing here is lost; it is just no longer a rule.

### 1. The threshold, and everything quoted against it
θ_j, THRESHOLD (0.25), GOO_THRESHOLD (0.2), THRESHOLD_FAN_IN (18), the rescaling of a neuron's whole potential axis by d_j/18, the comparison p ≥ θ, threshold homeostasis (HOMEOSTASIS, TARGET_RATE), un-sticking (UNSTICK, UNSTICK_TARGET, STUCK_BELOW/ABOVE as a trigger), the bored clock (BORED_AFTER) and `threshold_at`, the struck THRESHOLD_RANGE row.

*Why:* Byron, September 17, 2026 — "Theta should also be deprecated." Eight §1 constants exist only to set a threshold or to quote something against it, and 18 is a hex-lattice number that means nothing to goo. Two needs they served are carried forward as clauses without their mechanisms: a neuron nobody talks to must still be able to spike (§7.5), and a definition of "stuck" for reading a run (reporting, not a rule). What is *not* carried is the habit of quoting a constant in threshold units — the floor, the hazard's width and the escape scale were all quoted that way, and the unit is gone.

### 2. The leak
TAU (2 ms), the lazy decay on arrival, the half-life arithmetic, the per-run choice between the leaky neuron and the accumulator, `--tau inf`, the checkpoint field that records it, and §6.1's note that the accumulated perturbation scale depends on TAU.

*Why:* Byron, September 17, 2026 — "please deprecate and let me know if there are uses of this still." The accumulator is the neuron, not an option, so the rule is stated positively ("the potential does not leak") rather than as TAU = ∞, which keeps a deprecated constant alive inside the words of the rule. SYNAPSE_TAU is a different constant, deliberately decoupled on September 13; it survives or dies with the eligibility decision, not with this one.

### 3. The neuron's exploration
ESCAPE_DELTA (0.455) and the margin s = p − θ_j, the hazard 1 − e^(−m) with m = (Δt/hop)·√(N₀/N)·e^(s/Δ_j), ESCAPE_REFERENCE_COUNT (60) as a constant, SIGMA and the additive Gaussian nudge, EXPLORE (wave/epoch) and the Box-Muller stream consumer, §4.2's per-epoch explore step.

*Why:* Byron, September 17, 2026 (§0.3) — a synapse explores its own impulse response, not a neuron. Doubly gone: the margin it is computed from and the unit its width is quoted in both die with θ. The *principle* attached to ESCAPE_REFERENCE_COUNT survives as §0.8 (few knobs, scale built into the rule); whether the √N law is discharged by a fan-in hazard is a decision, not an inheritance. Additive SIGMA is dropped even as a switch: the record measures three things tried against it and none moved it off chance.

### 4. The dopamine machinery
§6.2–§6.6 whole: the gamma release density, DOPAMINE_RELEASE_ALPHA/THETA, DOPAMINE_TAU, DOPAMINE_EXPECTATION_TAU/START, DOPAMINE_ORDER, DOPAMINE_PUNISH, DOPAMINE_PUNISH_GAIN, and §0's fourth bullet's timing law (maximum immediately after the refractory period, decaying exponentially from 1 unit).

*Why:* Byron, September 17, 2026 — "its leak bullet and dopamine bullet no longer describe the system." The pool has decided nothing since the external teacher landed on September 12; the Rust engine never implemented the rule at all, so it cannot satisfy "every clause in every engine" as it stands; and the timing law is the opposite shape to what is built, since f is negative at t = 0 and peaks at TARGET_ISI. What is carried is the *frame* and not the law, as an open clause (§0.6): one global scalar, produced locally at a spike, consumed globally. If the student-as-teacher is taken up again it is respecified whole, from §0 down.

### 5. The Poisson drive and its coordinates
INPUT_DRIVE as a menu (rate / forced), INPUT_CV (0.6), INPUT_RATE (0.133/ms), INPUT_RATE_OFF as an obvious zero.

*Why:* replaced by one deterministic spike every TARGET_ISI, presented all at once (Byron, September 17, 2026). The new file writes one drive and not a choice of three. One thing must not be lost with it, and is carried into the drive's clause as a constraint rather than a mode: Byron's September 14 reason for dropping `forced` — a single synchronised wave at t_e locks every spike in the network onto one hop lattice.

### 6. The retired reads and critics
The `window` read (READ_WINDOW), the rate read's level ℓ_j = clip(r_j/RATE_ON, 0, 1), the kinder/population critic and TEACHER_CREDIT's per-neuron path, `decoded`.

*Why:* the window read was abandoned twice for the same reason and measured worse at seven widths; the level exists to convert a rate into the old teacher's bit, and the new teacher is a rate difference and needs no bit; the kinder critic bought nothing by its own measure and cost under the stricter one. `fired`, `count`, `class` and `graded` are not dropped — they are candidates inside the open critic clause (§9.5), which the teacher decision settles.

### 7. The retired wirings
All-pairs goo, `--wiring zones`, `zones-equal`, `uniform`, `scaled-open`, GOO_PROJECTION (0.2), and the `direct = False` cut.

*Why:* each was superseded by the next within days, and the zone rule's own text says it contains the direct cut. `scaled-open` never had a checkpoint saved under it and exists only so two temperature sweeps can be read — that is exactly what the record is for. Four surviving wirings (scaled, ff2, ff2-partial, and whatever the container decision leaves) are enough; seven in three engines is not.

### 8. Four of the five eligibilities
perturb, wrong_hebb, count_hebb, hazard — subject to the eligibility decision, which may keep one of them instead of hebb.

*Why:* this is the single largest driver of the rebuild's size — each eligibility is a separate implementation in every engine, held to the bit. On the record's own evidence perturb never moved off chance, wrong_hebb carries no digit, hazard is defined on a margin θ's deprecation removes, and count_hebb is the epoch form the single-spike rule replaced on September 17. The new file names the one rule it keeps and says what weighs it; it does not carry a menu that exists because each item was tried.

### 9. Grid furniture — conditional on the container decision
ROWS, OMEGA, REACH, the hex-step metric and the guaranteed neighbourhood, the small-world shortcut draw with its rejection loop, the columns' horizontal-distance ε, the lattice/spread, the input permutation, TARGET = reversed, PROBLEM = reversal, ACROSS as "cells across" and the k = ACROSS/2 sizing convention, and goo's grid-style `get_neuron_at(place, row)` addressing with `rows = 2`.

*Why:* goo is the plane taken away and every September result is on goo; the permutation and the reversed target are relabellings of interchangeable neurons by Byron's own September 14 words ("Permuting patterns should no longer matter. All neurons are first-class citizens"). The row addressing is scaffolding that makes goo impersonate a grid so grid-era code keeps working — the rewrite exists to shed exactly that. If the grid is rebuilt it maps onto zones, not the other way round.

### 10. Compatibility and reporting scaffolding
`--no-isi-factor` as a bit-compatibility flag, the "a checkpoint saved before the factor resumes with it off" migration, the exemption list (count_hebb, wrong_hebb and perturb run unweighted), WINDOW (marked "reporting only" in the record's own table), the section-wide **kept**/**open** banners, and the preamble sentence "an open section currently describes what the code does today" with its companion "where it conflicts with §0, §0 wins and the code is behind."

*Why:* the first three keep faith with runs that are finished and recorded; a rebuilt system has no pre-factor rule to reproduce. WINDOW changes nothing about what the system does. The banners have nothing to mark once the clause is the unit — and that preamble sentence is precisely the licence that let a specification become a lab notebook, and the mechanism by which a rule could be non-negotiable and unimplemented at the same time. If an ablation switch on the ISI factor is wanted, it gets its own clause saying what it is for.

### 11. Findings, arguments and derivations
§0.1 and its five consequences, the Werfel/Seung note, the n = 3 derivation of the cubic rational, the escape-scale arithmetic, every sweep table, every "why" and "what it is" paragraph, and every "measured" block.

*Why:* the file's own rule — no results, no findings, no arguments; a record entry names the clauses it measured. Two things move rather than die: the d_ij instrument of §0.1 (the correlation of a weight change with P(pixel i on | class of j) − P(pixel i on)) is a measuring device and belongs with the sweep tooling that is carried over deliberately; and Seung (2003) — the actual precedent for a synapse exploring by its own stochastic events — should be added to BIBLIOGRAPHY.md with a copy fetched into `references/` under the usual provenance rule.

## A draft of the first section, for Byron to react to

# AUTHORITY.md — §0, drafted for Byron to react to

*(The preamble already in the file stands unchanged above this. What follows replaces "0. The Strong Statement".)*

---

## 0. Standing values — non-negotiable

These are values, not mechanisms. No clause below §0 may contradict one; where a clause and a value disagree, the value is right and the clause is to be rewritten. Each value is dated and attributed like every other clause, and §0 is revised the same way any clause is: by writing the value that replaces it, with the words that decided it.

**0.1 The inputs are the outputs.** An input neuron, a hidden neuron and an output neuron are defined only with respect to where external connections exist. The neurons themselves operate identically. No rule may branch on a neuron's index, its zone or its label.
*Byron and Cedric, from the beginning. Restated by Byron, September 14, 2026: "All neurons are first-class citizens of the population."*

**0.2 A neuron integrates delta functions to determine its potential.**
*Byron and Cedric, from the beginning. The mechanism is §1.2.*

**0.3 The potential does not leak, not even lazily: the impulse response is infinite.**
*Byron and Cedric, as §0 was first written; reversed September 12, 2026 and restored by Byron, September 17, 2026, specifying the evidence accumulator ("I don't know that there are any changes other than Tau=infinity") and then deprecating the leak outright ("please deprecate and let me know if there are uses of this still"). The mechanism is §1.2; TAU is deprecated by §1.3.*

**0.4 A neuron that fires is then absolutely refractory: it ignores its inputs and does not integrate them.** This is a feedback control mechanism and a computational feature of the system, not a limitation to be worked around.
*Byron and Cedric, from the beginning. The mechanism is §1.6.*

**0.5 Exploration belongs to the synapse.** A synapse explores its own impulse response; the object that explores and the object that is credited for what it explored are the same object.
*Byron, September 17, 2026, 06:05 MDT: "Computationally exploration noise can be generated at the synapses, and semantically this is clean: a SYNAPSE explores its own impulse response, rather than a NEURON exploring its impulse response!" The mechanism is §7 — open.*

**0.6 Learning is paid by one global scalar, produced locally and consumed globally, and a neuron becomes eligible when it fires.** — **open.**
*The frame is Byron and Cedric's, from the beginning. Its timing law is struck (§0.11), and what replaces it waits on the teacher (§9) and on which eligibility survives (§8.1). Until Byron settles it, this clause states the frame and nothing about when or how much.*

**0.7 Structures are permissive; nothing is restricted artificially.** A neuron may sit in several input and output zones at once. Where the system refuses a configuration, the refusal is a clause of this file with a reason, never a convenience of the code.
*Byron, September 14, 2026, striking the threshold clamp: "Please eliminate threshold clipping. It's artificial." And the same day, extending un-sticking from the output row to every neuron.*

**0.8 As few knobs as possible, and a default regime that is stable.** Where a quantity must change with the size or the shape of a network, the change is built into the rule rather than left to a sweep.
*Byron, September 16, 2026: "A design principle of this project is that it should have as few knobs as possible and by default operate in or near a stable regime. Therefore scaling the network MUST reduce the probability of escape noise at each neuron by sqrt(N). I want to build this into the rule."*

**0.9 The goal is the substance, not a task.** A problem exists only as a way of watching whether the substance is alive and learning; no clause may be chosen because it scores well on one.
*Byron and Cedric, standing since the project began.*

**0.10 The network keeps living.** There is no training run and no evaluation run, only one run that keeps going. No rule may assume an end, and no result is read as convergence.
*Byron and Cedric, standing. Its consequence for data is §10.6: there is no held-out set, because there is no evaluation run to hold one out for.*

**0.11 Struck from §0, and by whom.** Two of the four statements this section carried until September 17, 2026 no longer describe the system, and are recorded here as struck rather than quietly dropped:

- *"…and fires when the integral exceeds a threshold, at which point it resets."* Struck by Byron, September 17, 2026: **"Theta should also be deprecated."** What fires a neuron is now §6.1 and is **open**; what a spike does to the potential is §1.4 and is **open**. The two questions are separated because they were never separate rules, only one sentence.
- *"A neuron becomes eligible for dopamine release when it fires… Maximum dopamine is released when the neuron fires immediately after the refractory period. The amount of dopamine released exponentially decays with time…"* Struck by Byron, September 17, 2026: **"its leak bullet and dopamine bullet no longer describe the system."** The timing law is gone; the frame survives as §0.6, open. The machinery that implemented it (the gamma release, the pool, the expectation trace and the punishment switch) is in `RECORD.md` and is not carried into this file; if it is taken up again it is respecified whole, beginning here.

---

*Three notes on what this draft assumes, for Byron to accept or reject:*

1. *§0 is now values only. Every mechanism that used to live in it — the threshold, the reset, the leak's time constant, the dopamine timing — has moved out into a numbered clause that can be dated, tested and revised without touching a non-negotiable section. That is the answer this draft gives to the question of whether §0 stays non-negotiable: it stays, and it stops containing mechanisms, because a non-negotiable section that is revised twice a week teaches the opposite of what the marking intends.*
2. *§0.11 exists so that a reader of this file alone can see what was removed and who removed it. If you would rather §0 carried no history at all, it moves to the head of `RECORD.md` and §0 loses four lines.*
3. *§0.6 is the only open value. Everything else here is settled in your or Cedric's words. If the answer to the dopamine question is "no global scalar at all — the teacher's rate error is the whole signal", then §0.6 is struck too and §0 has nine values, not ten.*

## The critic's gaps

- **must-have** — §6 head (Byron, September 13, 2026): The composition principle is nowhere in the outline or the dropped list: "What I realised today is that there is not a neuron training rule. There are multiple compatible training rules." With it go three rules of structure — local rules (needing only the neuron's own spikes and its synapses' stamps) run under every configuration and compose; at most one rule pays at the read, named by RULE, and RULE = local means none does ("no teacher for now"); and within a wave the local rules run in a fixed order (quash, then leaky Hebb) so a potentiation is not discounted the moment it is made. The outline's §8 and §9 silently assume exactly one eligibility plus one teacher, which is a narrowing of Byron's own words and needs to be his decision, not an omission.
- **must-have** — §6.12 (leaky Hebb) and §1.3 (SYNAPSE_TAU, HEBB_RATE, LEAKY_ELIGIBILITY): An entire built rule — Byron's own, September 13, 2026, specified in his words ("we are going to append f(t - t_fired_i) to the chain of multiplications. Call this leaky_hebb") — appears neither as a clause nor in the eleven things the specification deliberately drops. Its constant SYNAPSE_TAU is mentioned once, in the leak's drop note, as surviving or dying "with the eligibility decision", which is not the same decision: leaky_hebb is a local rule with its own rate, not a value of ELIGIBILITY, and LEAKY_ELIGIBILITY (appending the same trace to the reinforce rule) is a third thing again. Keep it, drop it, or fold it into the eligibility decision — but name it.
- **must-have** — §6.10 (ADALINE) and §1.3 (RULE): ADALINE with an eligibility trace is decided, built, measured, and is one of the four values of RULE; it appears nowhere in the outline and nowhere in the dropped list. It also carries a §0 conflict the record flags explicitly and that no clause now carries: it assigns each output its own error, so "the signal is no longer global" — a departure from §0's one global scalar (outline 0.6). Either it is dropped with the dopamine machinery, or §0.6 has to say that a per-neuron error is permitted.
- **must-have** — §0.3 and §0.4: The two statements of the new exploration rate disagree and clause 7.2 carries neither. §0.3: "a neuron of fan-in d taking a hazard spike within a hop with probability exactly 1/d". §0.4: "a probability of (2d - 1)/(2d^2) of an exploration spike per refractory period ... (FOR NOW)", from which the whole 1.98 s / perceptual-frame arithmetic is computed. At REFRACTORY_HOPS = 2 those are about a factor of two apart in rate. An open clause is required to say so in Byron's words and name the unit (per hop or per refractory period), rather than leave 7.2 with no content at all.
- **must-have** — §0.3 (with §5.2 and §6.7): What an exploration event at a synapse actually *is* is unspecified, and two incompatible readings sit side by side in the record. Byron's mechanism is a fan-in hazard that produces a spike of the *neuron* ("a neuron of fan-in d taking a hazard spike"), while the precedent he cites, Seung (2003), is stochastic *transmission*: the synapse releases or fails when a real presynaptic spike arrives. These differ in what the event delivers (a spike of the target, or w_ij into the target's potential), in whether it requires a presynaptic spike at all, and in what it does to the trace x_ij that 8.2 credits. Since 7.1 insists the exploring object and the credited object are the same object, this is the decision §7 and §8 both rest on.
- **must-have** — §5.2 ("A neuron that hears nothing") with §5.4, §1.3 UNSTICK: A fan-in hazard is undefined at d = 0, and clause 7.5 ("a neuron nobody talks to must still be able to spike") is precisely the case it cannot cover: (2d - 1)/(2d^2) is undefined at zero fan-in and 1/d diverges. The record settled this once for the neuron hazard, on September 16, at some length — a neuron with no incoming synapses takes the container's quoted width and "fires at the hazard's rest like any neuron" — after the array engine had silently silenced 395 mnist inputs by dividing by a zero width. The synapse hazard needs the same clause written before the code, or 7.5 is unimplementable.
- **must-have** — §4.3 ("rate is the default", Byron, September 14, 2026): Clause 5.7's new drive — one deterministic spike every TARGET_ISI, "presented all at once" — reinstates exactly what Byron struck: "The question I am getting at is whether there is a unified wave front in the input process at time zero. I don't want any such thing." The measurement beside it is that under forced drive 100% of every spike in the network sat on the lattice t_e + k*hop against 0.0% under rate drive. The outline's own drop list says that reason is "carried into the drive's clause as a constraint rather than a mode", and then 5.7 contradicts it. The conflict is Byron's to resolve (5.1 ms against a 2.5 ms hop is not a multiple, which may be the answer), but the clause must state it.
- **must-have** — §6.7 ("Three forms follow, and which one is meant is open"): The forced skip is an open learning decision the outline does not carry. The update passes over every synapse whose target was forced this epoch, and the record leaves three forms open: (a) as written, (b) lift the skip and take the eligibility from the refire, (c) REINFORCE on the teacher's trace. Clause 6.5 asks only whether forcing survives; it does not ask whether a driven neuron's incoming synapses learn. With the drive now deterministic and every driven neuron spiking every 5.1 ms, this decides whether the input zone's synapses learn at all.
- **must-have** — §8 (the input code the outputs receive, September 16, 2026): "Whether a driven input should carry the hazard's rest at all is a rule about the neuron, and Byron's to call." Measured: an on-pixel input fires 3.5 spikes an epoch and an off-pixel input 1.75, so "a synapse from an off pixel earns eligibility as if its pixel were half on", and quieting the escape was worth +0.093 of estimator correlation on ten seeds of ten. Clause 5.8 asks what an undriven input neuron does under the *drive*; it does not ask whether an input neuron explores. Separate question, explicitly handed to Byron, missing.
- **should-have** — §2 ("The substance"): The definition of walnut butter itself — a substance spread on the plane, its density the neuron density, one neuron per unit cell at unit density, butter near butter connects — is neither a clause nor in the dropped list, while goo is "the plane taken away" and every September result is on goo. Clause 0.9 keeps "the goal is the substance, not a task" without the specification ever saying what the substance now is. Either §0 says what butter means with no plane, or the drop list says the plane goes with the grid furniture.
- **should-have** — §0.4 (Byron, September 17, 2026): The goal — "I am okay waiting 2 seconds for the network to learn each novel thing. Ideally I'd like this to be on the order of the perceptual frame" — appears only as a bracketed citation on 7.2, 7.3 and 4.11, never as a clause. It is the only stated criterion by which the open hazard rate can be decided (the record's arithmetic: about 5 ms / T_f per refractory period, 0.05 at a 100 ms frame, a rate set by the frame and not by the fan-in), so it belongs in §0 or at the head of §7 as a standing target, in Byron's words.
- **should-have** — §4.3 (a noisy input) and §1.3 FLIP: A problem may corrupt what it presents: each coded bit flipped independently with probability FLIP from the network's seeded stream, which creates two patterns where there was one — the *presented* pattern that drives the neurons and the *target* pattern the read is scored against. The network is then asked to repair its input. Neither the mechanism nor the two-pattern distinction is in the outline or the dropped list, and population_denoise is built on it.
- **should-have** — §1.1 WEIGHT_EPSILON: `--positive-weights` narrows WEIGHT_RANGE to [epsilon, 1]: a network with no inhibition at all. Clause 2.5 asks what bounds a weight once learning moves it, which is a different question; a switch that removes inhibition from the substance should be a clause of its own or an explicit drop, particularly now that the floor (1.5) is open and inhibition is the only thing the floor ever capped.
- **should-have** — §1.2 ESCAPE_DELTA ("a network built in the library is deterministic until set_delta"): Where the exploration rate is set is a rule and is missing: a network built in the library has no exploration until a Teacher or the command line gives it one, while the command line and the sweep driver apply it. Under the old neuron hazard this meant a library network was deterministic; under a synapse hazard the same question decides whether a bare network explores at all, and the answer has to be the same in three engines. (The outline's 8.11 gestures at a library/command-line distinction for the ISI factor, where the record makes none.)
- **should-have** — §5.2 (escape scale) and §5.2 (starting values) with §7: The checkpoint rule that separates a rule from a state is missing: the escape scale "a checkpoint recomputes rather than storing, since it is a rule and not a state", while thresholds and floors are stored as the run actually reached them rather than recomputed. That distinction is what makes 11.7's round-trip meaningful and 4.13's "restores under the wiring it was built with" decidable; without it, every future field is an argument.
- **should-have** — §6.9 ("A problem names the rule it is posed for") and §8 (mnist, Problem.lr): What a problem may set, beyond its zones, is scattered and partly missing: a problem names the rule it is posed for; mnist sets its own learning rate (Byron, September 16: "default LR to 0.002 for this task", taken unless --lr is given); mnist switched homeostasis and un-sticking off ("Please turn off for this task"). Clause 10.7 asks only whether a problem may switch off a clause of the file. Whether a problem may set a constant — and which — is the general form of that question and of 3.11, 4.14 and 8.16.
- **nice-to-have** — §4.3 (coding) and §4.2: Three smaller drive-and-epoch rules are in neither list: `--ecc`, which encodes 4 data bits to 7 (Hamming) or 6 (parity) before complement coding; `coding = raw` and `coding = population-complement` (5.5 carries only population); and `--discharge`, under which the epoch's reset zeroes potentials instead of keeping them (3.10 states only the keeping). Also from §4.2: "An input may not be given a time before the horizon the schedule has already run to", which is the one constraint that makes 3.9's horizon well defined.
- **nice-to-have** — §1.3 LATE and §6.7: LATE — what a signal arriving after its target fired earns: count, ignore, or depress — is neither kept nor dropped. It "does not apply" under both surviving eligibilities, so it should go in the dropped list beside the four eligibilities with that reason stated, rather than be left for a reader to discover in the constant register.
- **nice-to-have** — §3.3 and §4.5: Two small build-and-stream rules are unstated: a network may be built with a fixed weight given to all connections in place of the uniform draw (2.4 states only the draw), and a run longer than its input stream cycles it (which is what makes mnist's "100,000 epochs is one and two thirds of a pass" true, and what a million-epoch run depends on).

### Clauses the critic says are misstated

- **4.7 — "The scaled rule: P(i→j) = 0 if i = j, 0 if both are in the input zone, 0 if i is in the output zone, else min(1, N·s / A_j)"**
  - The third case is misstated and forbids projections the rule permits. The record's amended rule (§3.4, Byron, September 16, 03:05 MDT) is "0 if i in the output zone, **and j in either zone**" — an output projects onto the hidden neurons, and only onto them; the outline's wording makes an output project onto nothing, which would make the mnist goo of 644 a different network from every one measured. The clause is also incomplete without A_j, which is part of the rule and not an abbreviation: the N − I − O hidden neurons for an input, the N − O inputs and hidden for an output, the N − 1 others for a hidden neuron.
- **10.4 — "mnist: 14×14 binary pixels; 3 clock + 196 pixels + 196 complements in, five neurons a class out" — kept**
  - The output layout is the superseded one. Decision 6 (Byron, September 16, about 11:40 MDT: "In the output, I'd like to force complement coding. How about we try ten classes x a population of six neurons: three fire-if-one and three fire-if-zero?") replaced it with sixty outputs — the ten fire-if-one populations of three first, then the ten fire-if-zero in the same order — population 3, class evidence n_k = n_k⁺ − n_k⁻, and it was kept on the measurement ("You missed that there is less variance in the complement-coded results, so we'll stick with the complement coding for now"). Five a class / fifty outputs is the September 15 layout, and the last three sweeps of the project all ran the complement zone.
- **4.6 — "Zones may overlap; only a count below the input width is refused" — kept**
  - That is the all-pairs goo's rule of September 14 and it was superseded twice over. Under the two-width zones "count must exceed the two widths together"; under the scaled rule the requirement is "only that the zones not overlap"; ff2 refuses a goo with hidden neurons. As written, 4.6 contradicts 4.7 and 4.9. What survives is the second half — each wiring states its own refusal — and 4.5 (prefix/suffix or any set) is where the permissiveness question actually lives.
- **2.3 — "Whether a neuron may connect to itself — open"**
  - Not open. §3 decided it on September 10 ("No neuron connects to itself"), it has held through every wiring, and Byron restated it in his own words on September 16 when he gave the scaled rule: "P(i projects onto j) = 0 if i == j". If §0.7's no-artificial-restrictions is thought to reopen it, that is a new question of Byron's, and the clause should say so rather than present a decided rule as undecided.
- **3.5 — "Signals sharing a timestamp sum before any neuron fires, so no decision depends on arrival order" — kept**
  - True of the firing decision, false of the bits, and the difference is load-bearing for §11. §6.15: "signals due at one moment are summed in push order, so an engine that flattens the topology differently sums a wave differently and lands on different bits" — which is why the array engine agrees on every spike but only to a part in 10⁹ on weights, and why the Rust loop had to take the object engine's push order. Either 3.5 names the summation order as normative, or 11.4's "compared with ==" cannot be met.
- **8.7 — "an arrival counts one and notes E_j; a decision adds q_j to E_j; the spike settles every open arrival and clears; the read pays and clears the score"**
  - Three settling events are missing from the list. §6.7: "the **floor**, a **forced spike** and a **discharge** settle with no credit". The floor's settle is not bookkeeping — it is the rule that stops an inhibited neuron accumulating eligibility it did not earn — and it is the one event that disappears if 1.5 decides there is no floor, so the clause has to name it while the floor is open.
- **2.6 — "A synapse carries its own trace x_ij: the count of arrivals its target integrated since its target's last spike" (and 1.2, "p_j is the sum of the weights of the arrivals it integrated since its last spike, undiminished")**
  - Both need the floor's qualification. §6.7: the trace "is zeroed when j spikes (the potential resets) **and when the floor bites** (the potential is then the floor whatever the weights)". §5.1 states the identity p_j = Σ w_ij x_ij and ∂p_j/∂w_ij = x_ij "exactly **while the floor has not bitten**". As written, 1.2 and 2.6 are true only of a neuron that has never been floored, and it is that identity that makes x_ij the derivative rather than a stand-in.
- **4.10 — "ff2-partial: P on each input→output pair" — kept, against the dropped list's "GOO_PROJECTION (0.2)"**
  - The drop list removes ff2-partial's only knob. GOO_PROJECTION (`--projection`) *is* the P of ff2-partial (Byron, September 17, about 02:40 MDT), and the sweep that produced the best read of any mnist run — 0.217 and 0.212 right at P 0.25 and 0.5 — is a sweep of it. Drop it as the retired wirings' knob if you like, but 4.10 then needs a projection constant of its own in the register, or the two entries cancel.
- **11.7 — "A checkpoint round-trips, and a resumed network keeps the settings it was saved under unless the resuming run overrides them explicitly" — kept**
  - The record has two different things under one name and the clause promises the stronger one. §7's checkpoint round-trips (weights, thresholds, clock, spike times, synapse stamps, signals in flight, p̂_j, expected count, the synapse's note, the ISI setting). The driver's `--resume-from` is explicitly *not* to the bit: "the exploration stream starts afresh (seed + 1,000,000) and the reward baseline from the first resumed epoch", and the input stream's position is not in a checkpoint at all — the watch's own checkpoints "restore those weights but not the stream's position, the exploration stream's state or the baseline, so a run from it is a continuation, not the same run". What a resume must reproduce is an open decision, not a kept invariant.
- **1.1 — "A neuron has a potential p_j, a last-spike time, a previous-spike time, a rate estimate r_j and a per-decision expectation of its own spike p̂_j"**
  - Two more pieces of per-neuron state carry the single-spike rule and are checkpointed: E_j, the neuron's expected spikes since its last spike ("a decision adds its q_j to E_j"), and the count of decisions to date, n, which 8.4's max(DECISION_MEMORY, 1/n) reads. §7's checkpoint bullet lists both. Without them 8.4 and 8.7 cannot be implemented from the clause.
- **0.5 — "Exploration belongs to the synapse: a synapse explores its own impulse response — kept [§0.3]"**
  - The citation does not support a kept §0 value. §0.3 is headed "an observation", is placed "at the top at Byron's instruction, beside §0.1 and §0.2 and **not among §0's rules**", and ends "Not specified or built yet." Byron's words are strong and he may well want this in §0 — but promoting an observation to a non-negotiable standing value is a decision of his to make, and the clause should present it as one rather than as inherited.
- **8.11 — "The factor is on by default, in the library constructor as well as the command line"**
  - The record says only "On by default" (Byron, choosing between on and off) with `--no-isi-factor`, and ISI_FACTOR = on in §1.3. The library-versus-command-line distinction it echoes is ESCAPE_DELTA's — "a network built in the library is deterministic until set_delta, as it has no σ until a Teacher gives it one" — and that rule is the one the outline is missing. Attributing it to the ISI factor states something the record does not.

### On the ordering

Yes, in five places.

(1) The ISI factor is buried, against Byron's explicit instruction. §0.2 exists where it does because he said so — "Please make the factor authority near the very top" — and the outline puts it at 8.9 to 8.14, three sections down, while 9.2 (the new teacher, "soft-gated by the ISI factor") uses it before §8 defines it. Either honour the instruction and put the factor in §0 beside the values, or have §9 cite §8 explicitly; as it stands the file's first mention of f is a use, not a definition.

(2) Exploration comes after firing, and on the record's own measurements that is backwards. Clause 6.1, "what fires a neuron", is open and cannot be read at all without 7.2; every mnist sweep of the last two days reports outputs firing at the exploration rest rather than from their inputs ("they fire by the hazard and not by their inputs, as the arithmetic said"), and the fully connected goo built its whole estimator alignment "entirely from rest-firing decisions". Exploration is now a firing mechanism, not an addition to one. Put §7 before or inside §6 — and before §8, since 7.1 says the exploring object and the credited object are the same object, which is the premise 8.2's x_ij depends on.

(3) §2 (the synapse) is stated before §3 (the clock) defines the terms it uses. 2.6's "arrivals its target integrated" and 2.8's "the last signal its target actually integrated" are only defined once 3.7's deliver-then-fire phases and 1.6's refractory rule are in hand (a signal arriving at a refractory neuron is dropped and integrated by nothing). And 2.6's trace depends on two clauses that are still open above it — 1.4 (what a spike does to the potential) and 1.5 (whether there is a floor) — so a reader meets the trace before either thing that resets it.

(4) The same open question is asked twice in two sections without saying they are the same: 1.8 (the window r_j is estimated over) and 9.4 (the window the teacher's observed rate is taken over), with §1.2 RATE_TAU and §1.3 RATE_MEMORY cited under both. Settle them as one clause or say in 9.4 how the teacher's window differs from the neuron's — otherwise the rewrite can implement two windows by accident, which is exactly how RATE_TAU and DECISION_MEMORY came to be quoted in units nobody had measured.

(5) A reader needs 10.1 — what a problem is and what it may set — long before §10. Whether a problem may set INTERVAL (3.11), the network count (4.14), its zone widths (5.2, kept), the learning rate (§8's missing clause; mnist sets LR 0.002), and whether it may switch off a clause (10.7) are the same question, met five times across seven sections and answered at the end. Move the definition of a problem to the head of §5, where the drive and the read first depend on it, and leave §10 for the problems themselves.

One smaller structural note: 0.11 is a deprecation record, not a standing value, and sits oddly among ten things that are. Consider an appendix for deprecations beside Appendix A's constant register — it keeps §0 to values only, and it gives the θ clause (6.2) and the leak clause (1.3) somewhere to point that is not §0.
