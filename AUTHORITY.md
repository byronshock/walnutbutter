# The authority

This file specifies walnutbutter. The code follows it: where the two
disagree, this file is right and the code has a bug. A change to the system
is made here first, then carried into the code and its tests.

Byron and Cedric own the substance of it. Claude keeps it and the code in
step. §0 is **non-negotiable**. Every other section says whether it is
**kept** (the connectivity and signalling scaffolding, agreed on
September 10, 2026, refactored lightly as needed) or **open** (to be
respecified: the constants, the activation rule, the learning rule). An
open section currently describes what the code does today, so that editing
it is editing a working system; where it conflicts with §0, §0 wins and the
code is behind.

Notation: a neuron $j$ has potential $p_j$, threshold $\theta_j$, and firing
rate estimate $r_j$. A connection $i \to j$ has weight $w_{ij}$. Time $t$ is
in nominal milliseconds. Every symbol in small capitals is a constant from
§1 and from `src/walnutbutter/constants.py`.

## 0. The Strong Statement — non-negotiable

The following are non-negotiable in this system. Everything else is tunable
parameters with sweeps.

* The inputs are the outputs. An "input" neuron, a "hidden" neuron, and an
  "output" neuron are defined only with respect to where external
  connections exist. The neurons themselves operate identically.
* A neuron integrates delta functions to determine its potential and fires
  when the integral exceeds a threshold, at which point it resets. This is
  an integrate-and-fire neuron. Note that there is no leak, not even a lazy
  leak. *Revised (Byron, September 12, 2026):* bring back the leaky
  integrate-and-fire neuron; the infinite impulse response without the
  leak is not wanted. The leak is lazy, with time constant TAU (§5.1).
* A neuron firing is followed by an absolute refractory period during which
  the neuron ignores its inputs and does not integrate them. This is a
  feedback control mechanism and computational feature of the system.
* A neuron becomes eligible for dopamine release when it fires. Dopamine is
  the global reward used for reinforcement learning. It is produced locally
  and consumed globally. Local production initiates when the neuron returns
  online from its refractory period and fires again. Maximum dopamine is
  released when the neuron fires immediately after the refractory period.
  The amount of dopamine released exponentially decays with time from
  1 unit at time t_first_fired + refractory_period + epsilon.

## 1. Global constants — open

One value each, for the whole network. `constants.py` is their only home in
the code; the constructors, the clock, the Teacher and the command line all
read from it, and `tests/test_constants.py` fails if any path grows a
literal of its own.

### 1.1 The default network

| constant | value | meaning |
|---|---|---|
| ACROSS | 8 | cells across; the input row has one neuron per coded bit |
| ROWS | 10 | rows of cells; input at the bottom, output at the top |
| OMEGA | 0.2 | proportion of all connections that are small-world shortcuts, $0 \le \omega < 1$ |
| REACH | 2 | lattice wiring: every pair within this many unit distances connects |
| WEIGHT_RANGE | [−1, 1] | random weights are drawn uniformly from this range, and learning clips to it |
| WEIGHT_EPSILON | 0.001 | under `--positive-weights` the range becomes [ε, 1]: no inhibition |

Goo (§3.4) has no cells to count, so it reads these two differently: ACROSS
is the width of its input and output zones, and ACROSS × ROWS is how many
neurons `--goo` makes when it is given no number — the default network's
eighty, so that a goo run and a grid run compare at equal size. OMEGA and
REACH do not reach it at all.

### 1.2 The neuron and its clock

| constant | value | meaning |
|---|---|---|
| THRESHOLD | 0.25 | $\theta$ every neuron starts with, quoted at THRESHOLD_FAN_IN incoming synapses (§5.2) |
| THRESHOLD_FAN_IN | 18 | the in-degree THRESHOLD and MINIMUM_POTENTIAL are quoted at: an interior hex cell's two rings at REACH 2. A container that scales rescales neuron $j$'s whole potential axis by $d_j / \text{THRESHOLD\_FAN\_IN}$ (§5.2); goo does, nothing else does yet |
| TAU | 2 ms | leak time constant of the potential, computed lazily on arrival, and of the eligibility trace on a synapse (§6.12), which is taken to be the same constant; $\infty$ switches it off (§5.1) |
| MINIMUM_POTENTIAL | −1 | floor on $p$: inhibition and carried-over charge go no lower. Quoted at THRESHOLD_FAN_IN like $\theta$, and rescaled with it (§5.2), so $p^{\min}/\theta$ stays −4 |
| REFRACTORY | 5 ms | absolute refractory period |
| REFRACTORY_HOPS | 3 | the refractory period divided by the time a signal takes to travel one hop; not an integer, started at 3 |
| INTERVAL | 35 ms | the epoch's length: the spacing of inputs when no time is given. Swept September 14, 2026 (§4.2); no problem overrides it |
| INPUT_DRIVE | rate | how a bit becomes spikes (§4.3): a Poisson process drives each input neuron across the epoch. `forced`, one mandated spike at $t_e$, locks every spike onto a hop lattice and is retired as the default |
| INPUT_CV | 0.6 | how the drive is specified (§4.3): the coefficient of variation of the spike train it produces. The rate follows exactly, $(1-\text{CV})/\text{REFRACTORY}$, so 80 Hz and 2.8 spikes an epoch |
| INPUT_RATE, INPUT_RATE_OFF | 0.133, 0 /ms | the same drive in the other coordinate: the rates of the Poisson processes that *drive* a bit-1 and a bit-0 neuron. INPUT_RATE is derived from INPUT_CV; the neuron fires at the first arrival after its refractory period (§4.3) |
| RATE_TAU | 5 ms | the exponential window the rate read estimates over (§4.3); equal to REFRACTORY, so one spike reads exactly RATE_ON |
| READ_WINDOW | 5 ms | the window of the `window` read: a bit, counting only spikes this recently before the epoch's end (§4.3) |
| RATE_ON, RATE_OFF | 200, 0 Hz | the rates a target-on and a target-off output are driven to: saturation ($1/$REFRACTORY) and silence (§6.9) |
| BORED_AFTER | 0 (off) | when positive, silence after which a neuron's threshold has fallen to zero and it fires on its own (§5.4). Off since September 14, 2026: every value below the epoch floods |

### 1.3 Learning

| constant | value | meaning |
|---|---|---|
| RULE | teacher | which learning rule runs: teacher (§6.9), adaline (§6.10), dopamine (§6.2-6.6) or reinforce (§6.7) |
| TEACHER_CREDIT | none | credit per neuron in the teacher's score; unset normalises it to span [−1, 1] for a zone of any size (0.25 at four neurons, 1/12 at twelve) |
| POPULATION | 3 | neurons per raw bit under population coding (§4.3) |
| FLIP | 1/12 | the probability a problem that corrupts its input flips each coded bit with (§4.3, Byron, September 13, 2026); a network in the library does not flip until it is asked to |
| QUASH_RATE | 0.02 | a refire weakens each contributing synapse by this fraction of its weight (§6.11); 0 = off |
| QUASH_K | 0.2 /ms | how fast that quash falls off with the delay since the previous spike |
| LEAKY_ELIGIBILITY | off | append §6.12's trace to the reinforce rule's chain, so the global reward reaches each synapse by what it still had in its target (§6.7) |
| SYNAPSE_TAU | 10 ms | the leak of the eligibility trace on a synapse (§6.12), the synapse's own and no longer the neuron's; it governs leaky Hebb and the reinforce rule's leaky eligibility alike |
| HEBB_RATE | 0.01 | leaky Hebb (§6.12): a firing neuron potentiates each gated synapse by this much times its leaky trace; a starting value, to be swept; 0 = off |
| LR | 0.03 | learning rate |
| SIGMA | 0.1 | exploration noise: standard deviation added to each neuron's potential |
| EXPLORE | wave | when that draw is taken (§6.1): afresh before every firing decision, or `epoch`, once at the input's moment (the pre-alpha's) |
| DOPAMINE_RELEASE_ALPHA, DOPAMINE_RELEASE_THETA | 2, 1 ms | shape and scale of the gamma density that gives the amount released against the refire delay past the refractory period |
| DOPAMINE_TAU | 20 ms | decay of the global dopamine value |
| DOPAMINE_EXPECTATION_TAU | 10 min | exponential window of the expected dopamine trace |
| DOPAMINE_EXPECTATION_START | 0 | where that trace starts; a high start holds early learning back (Byron, September 12, 2026) |
| DOPAMINE_ORDER | release-first | at a refire, release before the weight update (or update-first) |
| DOPAMINE_PUNISH | on | an input neuron whose bit is 0 has its update reversed when it refires |
| DOPAMINE_PUNISH_GAIN | 2 | and that reversed update is this many times a reward (§6.8) |
| WEIGHT_DECAY | 10⁻⁴ | every weight moves toward 0 by this fraction each epoch: synapses that forget (§6.8) |

The reinforce rule, factored out behind RULE = reinforce, keeps its own:

| constant | value | meaning |
|---|---|---|
| TARGET | reversed | what the output should show, derived from the input (§4.3) |
| CRITIC | row | how the reward is judged (§6.7) |
| ELIGIBILITY | perturb | what the reward acts on: the exploration noise, or a Hebbian ±1 |
| LATE | count | what a signal arriving after its target fired earns (§6.7) |
| BASELINE_RATE | 0.05 | per-epoch update of the running reward baseline |
| HOMEOSTASIS | 10⁻⁶ | per-epoch rate a threshold drifts toward its target firing rate; 0 = off |
| TARGET_RATE | 0.5 | firing rate homeostasis aims for |
| UNSTICK | 10⁻³ | per-epoch rate a stuck output's threshold moves toward UNSTICK_TARGET; 0 = off |
| UNSTICK_TARGET | 0.5 | firing rate the un-sticking aims for |
| THRESHOLD_RANGE | [−5, 5] | limits on where homeostasis and un-sticking may move a threshold |
| RATE_MEMORY | 0.01 | per-epoch update of $r_j$: about the last 100 epochs |
| STUCK_BELOW, STUCK_ABOVE | 0.01, 0.99 | a neuron with $r_j$ outside this band is stuck |
| WINDOW | 200 | epochs the reported moving-average accuracy spans (reporting only) |

## 2. The substance — kept

Walnut butter is a substance spread on the plane. Its density is neuron
density: butter spread at unit density puts one neuron per unit cell of a
hexagonal lattice. Butter spread near other butter connects (§3). The goal
is to create walnut butter, not to solve a particular task: a task is a way
of watching whether the substance is alive and learning.

Four containers build a network from it, all sharing the same neurons,
connections, signalling and learning:

- **The hex grid.** A rectangle of ACROSS × ROWS hexagonal cells, one neuron
  each, pointy-top, odd rows shifted half a cell. Row 0 is the top.
- **Hexagonal columns** (`--layers N`). The same field of cells extruded
  into columns of $N$ neurons in $\mathbb{R}^3$: a cell is one unit across,
  neighbouring cells are half a unit apart in the plane, the layers of a
  column an eighth of a unit apart. With one layer the stack *is* the hex
  grid, bit for bit.
- **The lattice / a spread** (`--nodes`). Neurons at $(x, y)$ positions in
  unit distances: a hexagonal lattice, a random scatter, or the positions a
  butter recipe describes.
- **Goo** (`--goo N`). The plane taken away: $N$ neurons with no positions
  at all, every ordered pair connected, no neighbourhood and no shortcuts
  (§3.4). Its input and output zones are picked by index, because there are
  no places to pick them by. It is the control the other three are measured
  against — whatever the geometry is worth is what goo is missing.

A neuron may sit in several input and output zones at once; structures are
permissive, never artificially restricted.

## 3. Connectivity — kept

Every connection is one-way, from a source to a target, with its own weight.
Between two neurons $i$ and $j$ there are two connections, $i \to j$ and
$j \to i$, each with an independent weight. No neuron connects to itself.
Connections have ids from 1 in the order they are made, so a seed rebuilds
the same network.

### 3.1 The guaranteed neighbourhood

- **Hex grid.** Every neuron connects to every cell within `reach` hex
  steps: with the default reach 2, its six neighbours (the first ring) and
  the twelve neighbours of those neighbours (the second ring), eighteen
  local targets for an interior cell; reach 3 adds the eighteen cells of
  the third ring, thirty-six in all.
- **Columns.** Same position, never; horizontal distance $\le 1 + \varepsilon$
  (measured in the plane, ignoring height), always. The radius of one unit
  takes in a neuron's own column and the eighteen columns around it, in
  every layer.
- **Lattice / spread.** Every ordered pair within REACH units connects. At
  unit density with REACH = 2 that is the same eighteen neighbours.

### 3.2 Shortcuts

Everything further away is reached only by small-world shortcuts. With $L$
guaranteed connections, $S = \mathrm{round}\big(\omega L / (1 - \omega)\big)$
shortcuts are added so that they are the fraction OMEGA of all connections.
Each runs one way from a random neuron to a random neuron that is not
itself, not a guaranteed partner, and not already a target of the source.
The lattice and a spread have no shortcuts.

### 3.3 Weights

Every weight is drawn independently and uniformly from WEIGHT_RANGE, in
connection-id order from the seeded stream, unless a fixed weight is given
to all. Learning (§6) keeps every weight inside WEIGHT_RANGE.

### 3.4 Goo — the container decided (Byron, September 14, 2026), the rest written for revising

*Byron asked for a fourth container and said what it was: fully connected
goo. Everything below it — the zones, the id order, the refusals — is
Claude's reading of what that has to mean to be buildable, written down here
so that changing it is editing a working system. §2's ownership stands: the
substance is Byron and Cedric's, and any line of this is theirs to overturn.*

Goo is butter with the plane taken away. Its neurons have no positions, so
no distance between two of them is defined, and neither §3.1's guaranteed
neighbourhood nor §3.2's shortcuts has anything to measure. What is left is
the only wiring that needs no ruler: **every ordered pair connects**. $N$
neurons give $N(N-1)$ one-way connections, each of kind `goo`, and at the
default $N = \text{ACROSS} \times \text{ROWS} = 80$ that is 6,320 against
the 8 × 10 hex grid's 1,395 — the same eighty neurons, four and a half
times the wiring.

Everything else in §3 holds unchanged: one way, an independent weight each
direction, no self-connection, ids from 1. The id order is source index
then target index, so $N$ alone fixes the topology — nothing is drawn to
decide it, and the seed's stream is spent only on the weights (§3.3) and
the permutation. A checkpoint therefore rebuilds goo from its count, with
no shortcut list to verify.

**Zones by index.** With no rows there is no bottom row to be the input.
The **first** ACROSS neurons in index order are the input zone and the
**last** ACROSS are the output zone. Both are addressed the way every other
container's rows are, so nothing above §3 has to know: `get_neuron_at(place,
1)` is the input zone and `get_neuron_at(place, 0)` the output, and `rows`
is 2 because it is counting those two zones and not any depth. The coded
bits are permuted across the input zone exactly as they are across a row
(§4.3). Where $N < 2\,\text{ACROSS}$ the zones overlap, and at
$N = \text{ACROSS}$ they are the same neurons reading and being read:
permitted per §2, not prevented. $N < \text{ACROSS}$ is the one refusal —
there would be nowhere to put a bit.

**What it is for.** Every other container spends its structure on putting
the output away from the input: on the 8 × 10 grid the bottom row is nine
hex steps from the top, five hops of the guaranteed neighbourhood, and the
OMEGA shortcuts are the only thing that shortens the trip. Goo has no far.
The output zone is one hop from the input zone and from everything else,
and every neuron sees every other. So goo and the grid at the same neuron
count differ in exactly two things — locality and depth — and whatever the
grid scores above goo is what those two are worth. If it scores nothing
above goo, the lattice has been decoration.

**Goo saturated at the grid's constants, and the fix was the whole potential
axis** *(September 14, 2026; 120 epochs, 5 seeds, no learning — a check on
the container, not a result).* Each goo neuron has 79 incoming synapses drawn
uniformly from [−1, 1] against a THRESHOLD of 0.25, where the grid gives it
about 18. "Distinct words" counts how many different patterns the output zone
produced across a run, per seed: 1 means the output never changed at all,
whatever the input.

| | fires per epoch | output zone on | distinct words |
|---|---|---|---|
| hex grid, 8 × 10 | 87.0% | 76.4% | 2.8 |
| goo, no scaling | 98.0% | **100%** | **1.0** |
| goo, threshold scaled only | 98.9% | 99.1% | 2.8 |
| goo, threshold and floor (§5.2) | 90.4% | 91.3% | **8.2** |

Unscaled, goo's output zone is on every epoch whatever the input, and the
read carries nothing whatsoever. **Scaling the threshold alone barely helps**
— still 99% on — because what sustains the activity is the recurrence rather
than the size of any single arrival, and raising $\theta$ against an unmoved
floor caps inhibition while excitation keeps piling up. Moving both is what
lands it: the output zone varies, and at 8.2 distinct words goo produces more
variety in its read than the grid does in its own.

Byron chose the threshold out of the three alternatives, then the floor with
it once the measurement showed the threshold alone was not the lever. **FOR
NOW**, in his words: §5.2 records the rule, and the exponent is linear
because that is the plain reading of "scales with fan-in", not because it was
fitted. Sweeping the floor's exponent found 1 to be where the gain arrives
and 1.5 and 2 to add nothing (8.2 words against 8.4), which is what holding
$p^{\min}/\theta$ fixed predicts: once inhibition is no longer capped, a
deeper floor is never reached.

**Swept (Byron, September 14, 2026: "sweep goo with threshold and floor
with 10 seeds at 250000 epochs").** Three arms, ten seeds each, 250,000
epochs, the reversal problem at every default — the reinforce rule with the
perturb eligibility, $\sigma$ 0.1 — on the array engine, every arm at seed
$s$ given the same input stream (§4.5) so the arms pair epoch by epoch. Two
arms were added to the one asked for, because a control's number means
nothing without the thing it controls for: the same goo run flat, and the
8 × 10 grid. `docs/goo-250k.md`, `goo-250k-score.png`, the checkpoints under
`runs/goo-250k/`.

| arm | last 25,000 epochs | over seeds | stuck on | stuck off |
|---|---|---|---|---|
| hex grid, 8 × 10 | **0.520** | 0.499–0.550 | 47.5 | 12.6 |
| goo, flat | 0.511 | 0.497–0.573 | 62.2 | 6.3 |
| goo, threshold and floor scaled | 0.507 | 0.500–0.528 | 41.4 | 27.0 |

**Everything is at chance, the grid included.** Paired on the seed, the grid
beats scaled goo by 0.013 ($t = 2.2$) and flat goo by 0.009 ($t = 1.5$), and
the two goos differ by 0.004 ($t = 0.5$) — a borderline edge for the
baseline, measured against a baseline that itself learned nothing in a
quarter of a million epochs. That is not a surprise this file did not already
hold: §6.1 and §6.7 record the perturb rule at chance on the grid under both
drives (0.543 under rate drive; 56.8% at five rows), while the hebb
eligibility of the same rule reaches 0.80 and 0.978. The sweep ran the
container's defaults, and the container's defaults are the rule that does not
learn.

So **this says nothing about locality or depth.** A control can only measure
what the baseline achieves, and here the baseline achieved nothing, so the
gap between them is a gap between two chance results. What it does say is
narrower and still worth having: (i) the rescaling of §5.2 did what the
activity table above said it would and no more — flat goo ends with 62 of 80
neurons stuck on, the saturation of the unscaled axis, scaled goo with 41 on
and 27 off — it fixed the read carrying nothing, not the rule learning
nothing; and (ii) 250,000 epochs of the perturb rule on this problem is not a
learning run for any of the three, which the grid's own figure, 0.520, states
on its own.

**Swept again with the hebb eligibility (Byron, September 14, 2026: "the
exact same sweep with eligibility=hebb, since that's the rule where we are
seeing learning emerge on the hex grid").** The same three arms, ten seeds,
250,000 epochs, reversal, every arm at seed $s$ on the same input stream —
on the Rust wave loop this time (§6.15), the Teacher's homeostasis and
un-sticking mirrored at the command line's constants so it is the same
experiment. `docs/goo-250k-hebb.md`, `goo-250k-hebb-score.png`.

| arm | last 25,000 epochs | over seeds | stuck on | stuck off |
|---|---|---|---|---|
| goo, threshold and floor scaled | **0.513** | 0.500–0.553 | 51.3 | 8.7 |
| goo, flat | 0.506 | 0.500–0.524 | 75.7 | 0.3 |
| hex grid, 8 × 10 | 0.502 | 0.500–0.514 | 61.5 | 7.1 |

**Still chance, and this time the grid is the worst of the three.** Paired
on the seed, scaled goo beats the grid by 0.011 ($t = 1.9$) and flat goo by
0.007 ($t = 1.1$); flat goo beats the grid by 0.004 ($t = 1.2$). None of it
survives, and the grid at 0.502 is the number to read: **on reversal, at ten
rows, the hex grid does not learn under the hebb eligibility either.** The
0.978 and 0.80 that §6.7 records for that eligibility are on shallow_copy —
twelve across, *two* rows, one hop — and the 0.64 of §4.3's CV sweep is on
reaching_copy, five rows wired to REACH 5 so that the bottom row synapses
straight onto the top: one hop again. The rule that learns learns where the
task is one hop wide, and reversal at ten rows, nine hex steps and five hops,
is not that task. Both sweeps of this section measured goo against a grid on
a problem the grid cannot do.

That points the comparison somewhere more useful than a working baseline.
reaching_copy is the grid with its **depth** taken away and its locality
kept; goo is the grid with both taken away, so a comparison at equal depth
would leave **locality alone** as what separates them — the cleaner half of
the question this container was built to ask.

**The task, defined (Byron, September 14, 2026).** Neither reversal nor
reaching_copy is the task Byron then set out, and §8 records it in his words
as **copy**: an input complement-coded onto eight input neurons, and the
desired output exactly that input across eight output neurons, place for
place, unpermuted. Every sweep of this section so far ran on reversal, the
input *reversed* on the outputs, with the input permuted. **For goo that
makes no difference** (§4.3): its zones are interchangeable neurons, so
reversed, permuted and copy are one task by relabelling, and the goo arms
of both sweeps stand as measurements of copy — at chance, 250,000 epochs,
under either eligibility. For the grid it does make a difference, so the
grid arms do not transfer and the baseline on copy is unmeasured. From here
the comparisons, and the count sweep below, are posed on copy
(`docs/rust-sweep.py --problem copy [--goo N]`, the Rust loop). Not run.

*Byron's reading, after the two sweeps (September 14, 2026): "I think we may
actually have too many neurons. FOR NOW."* Counted as synapses per raw bit of
task rather than as neurons, the day's evidence lines up behind it:

| network | neurons | connections | synapses per raw bit | hebb, best seen |
|---|---|---|---|---|
| shallow_copy | 24 | 215 | ~54 | **0.978** (§6.7) |
| reaching_copy | 80 | 3,122 | ~780 | 0.64 (§4.3) |
| hex grid 8 × 10, reversal | 80 | 1,395 | ~349 | 0.502 |
| goo 80, reversal | 80 | 6,320 | ~1,580 | 0.513 |

One scalar reward spread over more synapses moves each of them less, and the
one network that learns outright is the one with an order of magnitude fewer
to move; reaching_copy, one hop but 780 a bit, gets partway. That is a
"too many" story, and a depth story only secondarily. Recorded as a
hypothesis and not as a change to §1.1: the default network stays at eighty
neurons until this is measured. The test is a count sweep on hebbian goo on copy (§8) —
`--goo N` for $N$ in {8, 12, 16, 24, 32, 48, 64, 80}, ten seeds — because
$N$ is goo's only knob, so nothing has to be re-tiled to vary it, and goo 16
at 240 connections is shallow_copy's regime with no geometry at all.

**Swept (Byron, September 14, 2026: "sweep goo in {24, 32, 40, 48, 64, 80} x
threshold (and resulting floor) in {0.15 0.2 0.25 0.3 0.5} for 100000 epochs
across 10 seeds").** Goo's count against THRESHOLD, the floor following the
threshold at the grid's ratio of −4 so that every cell moves the whole axis
and not the ratio, and goo then scaling both by $(N - 1)/18$; copy (§8), the
hebb eligibility, the Rust loop, 300 arms in 742 seconds.
`docs/goo-count-threshold.md`, `goo-count-threshold-score.png`. Accuracy over
the last tenth, mean of ten seeds; stuck-on counts beneath:

| goo \ THRESHOLD | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| 24 | 0.579 | 0.560 | **0.609** | 0.595 | 0.584 |
| 32 | 0.558 | 0.530 | 0.537 | 0.574 | 0.586 |
| 40 | 0.538 | 0.530 | 0.555 | 0.573 | 0.550 |
| 48 | 0.527 | 0.512 | 0.532 | 0.531 | 0.526 |
| 64 | 0.505 | 0.506 | 0.539 | 0.516 | 0.572 |
| 80 | 0.531 | 0.518 | 0.524 | 0.516 | **0.634** |
| *stuck on, of 24* | 7 | 9 | 3 | 6 | 4 |
| *stuck on, of 80* | 67 | 67 | 61 | 61 | **0** |

**Two results, and they pull in opposite directions.**

*At the shipped threshold, fewer neurons is much better.* At THRESHOLD 0.25
goo 24 scores 0.609 against goo 80's 0.524 — +0.085 paired on the seed,
$t = 6.5$ — and the column falls monotonically from 24 to 64. The stuck-on
counts say why: goo 24 ends with 3 of its 24 neurons stuck on, goo 80 with 61
of 80. Byron's reading holds exactly where it was made.

*But the best cell of the sweep is the biggest goo at the highest
threshold.* Goo 80 at THRESHOLD 0.5 — $\theta$ 2.19, floor −8.78 — scores
**0.634**: nine seeds of ten above 0.55 and the tenth at chance, a whole-run
mean of 0.637 against 0.516 at the default (it learns from the start, not at
the end), and its stuck-on count falls from 67 to **0**, the un-stick nudges
from 549,000 to 28,000, because the outputs stop needing to be dragged off
the rails. Against its own default it is +0.110, $t = 4.6$; against goo 24
at the same threshold +0.050, $t = 5.1$ — the count effect *reverses sign*.
And against goo 24 at its own best it is +0.024, $t = 1.3$: not separable.

So the count was never the problem — **the threshold for the fan-in was.**
Eighty neurons learn as well as twenty-four once their threshold keeps them
from saturating, and §5.2's linear scaling does not get them there: at
$N = 80$ it starts $\theta$ at 1.10, and the saturation only clears at about
2.2, twice that. The stuck-on rows trace the same curve at every count — 64
needs 0.5 too, 40 clears at 0.3, 24 at anything — so the threshold that stops
a goo saturating grows *faster* than its fan-in, which is what the activity
scan above already hinted (exponent about 1.3 against the grid) and what the
sum of $d$ independent weights predicts if it is the variance and not the
mean that matters. That is a measurement against §5.2's "linear because that
is the plain reading", and it is Byron's to act on: FOR NOW was the word.

*Two caveats.* The axis is truncated where it matters — for $N \ge 64$ the
best cell is the last one, so the optimum is at 0.5 or past it, and no
exponent can be read off five points that stop there. And the seed spread is
wide everywhere, 0.50 to 0.70 in most cells: a cell's mean is ten seeds of
which some catch and some do not, and goo 80 at 0.5 is the one cell where
nearly all of them catch. Drawn over the run, every seed
(`goo-count-threshold-threshold0.5-goo80-trace.png`, Byron asking to see it):
the mean is past 0.55 by epoch 6,000 and peaks near 0.66 at 13,000, then
drifts down and wanders between 0.61 and 0.65 for the remaining 85,000 — a
network that keeps living rather than one that converges (§7). And the one
seed that ends at chance is not one that never caught: it learned with the
others and held for 70,000 epochs, then lost it in the last 15,000. What
"caught" means here is a state the network can also leave.

*The next sweep, if it is wanted:* THRESHOLD in {0.5, 0.75, 1.0, 1.5} for
$N$ in {48, 64, 80}, the floor following — 120 arms, about five minutes on
the Rust loop. It would locate the optimum for the big goos and put a number
on the exponent. Not run.

The two rejected alternatives are still on the table if that sweep finds the
rescaling wanting: a weight range scaling as $1/\sqrt{N}$, and the reading
that THRESHOLD was never a constant of the substance but a constant of the
grid.

*Also not decided here:* whether the pairs should connect with a probability
less than 1, which would make "fully connected" one end of a density axis
rather than the whole of goo, and whether the two zones should default to
disjoint at all. Both are sweeps, and neither has been run.

## 4. Signalling — kept, on a schedule

### 4.1 The clock

Time is in nominal milliseconds. A signal takes one **hop** to travel a
connection, $h = \text{REFRACTORY} / \text{REFRACTORY\_HOPS}$ (Byron,
September 11, 2026; not an integer, started at 3). Each input has a time
$t_e$; the first is at 0 and by default each is INTERVAL after the last.
There is no other delay: the time component of signalling is carried by
the hop and the refractory period.

*Decided (September 11, 2026):* a wave is "everything that happens at time
$t$", not "everything one hop after the last wave". With a non-integer
ratio, refractory recovery and hop arrivals fall at different times, and a
second input can land mid-cascade. The queue is therefore a time-ordered
**schedule** of signals, and a wave is the batch at the front with the same
timestamp, including any external signals stamped for that moment.
Same-time signals sum within a wave before anyone fires, which keeps the
floor and the firing decision order-independent; signals one hop apart are
separate waves. Epochs stop being the unit: a cascade from one input can
still be running when the next input's signals join the schedule, so "the
queue empties, then the clock advances" no longer holds. Learning has its
own cadence (§6). Times are rounded to a nanosecond so that two signals
computed to arrive at the same moment are the same moment in both engines.

### 4.2 An epoch

An epoch is one input and the schedule run up to the next input's time, the
**horizon**, $t_e + \text{INTERVAL}$ by default. Signals due at or after the
horizon wait for the next epoch. An input may not be given a time before
the horizon the schedule has already run to. In order:

1. **Reset.** Every neuron's fired-this-epoch state is cleared. Potentials
   are kept (`--discharge` zeroes them instead). Spike times are kept: the
   refractory period outlives the epoch. Signals in flight stay scheduled.
2. **Input.** The pattern is placed on the input neurons (§4.3), stamped
   $t_e$.
3. **Explore.** Each potential is nudged by the exploration noise (§6.1),
   floored at MINIMUM_POTENTIAL.
4. **Run.** The schedule runs wave by wave (§4.4) until the horizon; a
   neuron that refires learns as it fires (§6).
5. **Report.** The output is read (and, under the reinforce rule, scored
   and reinforced, §6.7).

*Swept, September 14, 2026: how long should an epoch be?* INTERVAL was never
chosen, only inherited. On shallow_copy at five rows under rate drive with
reward-modulated Hebb, six seeds and 100,000 epochs, against both critics:

| epoch | 5 | 19 | 20 | 21 | 25 | 26 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|
| row critic | 0.529 | 0.781 | 0.794 | 0.799 | | 0.830 | | | | |
| population critic | 0.535 | 0.800 | 0.814 | 0.818 | 0.847 | 0.849 | 0.858 | **0.869** | **0.869** | 0.865 |

Three findings. **Nothing distinguishes 20 ms**: 19, 20 and 21 lie on a smooth
monotonic line with no step at the boundary, which is what should be expected
once stated — nothing in the model knows what the interval is supposed to be,
so the epoch edge has no status of its own. **The optimum is a plateau at
35–40 ms**, about twice the inherited value: every step up to 35 wins in six
seeds of six, 40 over 35 wins in three of six and gains 0.0007, and 45 loses
in four of six. Seed-to-seed spread also triples past 30 ms (sd 0.003 below,
0.012 above), so the far end is both no better and less reliable. **And the
critic does not interact with the epoch at all**: the population critic sits a
near-constant +0.019 above the row critic at every interval but the collapsed
one. Same shape, shifted — the signature of a kinder yardstick rather than a
different behaviour, agreeing with §6.13's cross-scoring.

The 5 ms arm collapses to chance for four reasons at once: the epoch equals
REFRACTORY so no neuron can fire twice, three waves barely span the two hops
the task needs, and at 0.1/ms half of all epochs present no input at all.

*Not yet separated:* at a fixed input rate a longer epoch buys both more
settling time and more input spikes (2.0 at 20 ms, 3.5 at 35). Scaling the
rate as 1/INTERVAL would hold the spike count constant and say which of the
two the plateau is about.

*Decided (Byron, September 14, 2026), on the strength of the above:* INTERVAL
is **35 ms**, and no problem overrides it. The five that pinned 20 ms — a
value inherited from the first sustain task and never chosen — no longer set
an interval at all, so the epoch's length now lives in exactly one place.
Anything measured before this ran at 20 ms (or at 10 ms for reversal) and is
labelled as such where it matters.

### 4.3 Input and output

The network's input is its bottom row (the bottom layer, for columns); its
output is its top row (top layer). Per §0 the neurons are the same
everywhere; only the external connections differ. An input is $k$ =
ACROSS / 2 raw bits, drawn as fair coin flips from the seeded stream (or
given). They are **complement-coded**, the bits followed by their
negations, so $2k$ bits reach the row and exactly half of it fires whatever
the raw bits. The coded bits are then **permuted** by a random permutation
drawn once per network and fixed for its life: place $i$ along the row
shows coded bit $\pi(i)$. With an error-correcting code (`--ecc`), 4 data
bits are first encoded to 7 (Hamming) or 6 (parity) before complement
coding.

*Byron, September 14, 2026: "Permuting patterns should no longer matter.
All neurons are first-class citizens of the population." And: "I don't
think asking for the target reversed should matter either."* Both are
statements about geometry, and both are exactly right where there is none.
The permutation was there for the grid, where a place is a position and a
bit's neighbours are the bits beside it; a reversed target puts output
place $7 - i$ half a row away from input place $i$. On goo (§3.4) every
neuron of a zone is wired identically to every other, so a permutation of
the input zone and a reversal of the output zone are each a relabelling of
interchangeable neurons: they change which weight draw sits where and
nothing else, and copy, reversal and any permutation of either are one task.
The copy problem (§8) runs unpermuted. The permutation rule above stays as
written for the problems measured with it, and on the grid both still bite.

(A problem may instead lay the raw bits down as they are, `coding = raw`,
which sustain_inputs does, or repeat each bit over POPULATION neurons,
`coding = population`, which population_copy does: 1001 becomes
111000000111 on a twelve-wide row (Byron, September 13, 2026). Or both in
order, `coding = population-complement` (Byron, September 14, 2026): repeat
each bit, then complement-code the whole run, so with POPULATION 2 the four
bits 1001 become 11000011 and then 1100001100111100 over sixteen neurons,
which doubled_copy does. §8. A problem also says what "on" means at the read:
fired this epoch, spiked again after the input's moment, or fired within a
window before the horizon.) The neurons whose bit is 1 are **forced** to fire
at $t_e$, refractory period permitting. A neuron forced this epoch is marked as such, which only
the reinforce rule (§6.7) consults.

**How a bit becomes spikes (Byron, September 13, 2026): "I want to have an
option to present the inputs as a probabilistic firing rate rather than
presenting them all at once and seeing what happens."** INPUT_DRIVE names it.

- `forced`, and everything above this section: every neuron whose bit is 1 is
  made to spike once, at $t_e$, all of them in one wave.
- `rate`: an independent **Poisson process drives** each input neuron across
  the epoch, at INPUT_RATE where its bit is 1 and INPUT_RATE_OFF where it is
  0. Arrivals are drawn $\mathrm{Exp}(\lambda)$ apart from the network's own
  seeded stream, in place order, so a seed reproduces them and both engines
  draw the same ones.

**These are arrivals, not spikes** *(Byron, September 14, 2026: "actual neural
signalling is NOT a Poisson process. If we want to use a Poisson process to
drive the input neuron, it should have its own, much higher rate. The effect
should be to ensure the input neuron fires at the first Poisson arrival after
the refractory period ends.")* An arrival landing while the neuron is
refractory is dropped, exactly as a forced stimulus is, so the neuron fires at
the first arrival after its refractory period ends and its spike train is a
**renewal process with dead time**:

$$\text{ISI} = \text{REFRACTORY} + \mathrm{Exp}(\lambda), \qquad
\bar r = \frac{1}{\text{REFRACTORY} + 1/\lambda}, \qquad
\text{CV} = \frac{1/\lambda}{\text{REFRACTORY} + 1/\lambda}.$$

**The rate and the CV are one axis, not two** *(September 14, 2026, working
out what a CV sweep would be sweeping)*. Eliminating $\lambda$ between those
two expressions gives

$$\bar r = \frac{1 - \text{CV}}{\text{REFRACTORY}}, \qquad
\lambda = \frac{1 - \text{CV}}{\text{REFRACTORY} \cdot \text{CV}},$$

so at REFRACTORY = 5 ms the train runs at exactly $200(1 - \text{CV})$ Hz. The
irregularity and the rate are in exact bijection: there is no setting of
$\lambda$ that makes the train fast and irregular, or slow and clocklike,
because the dead time is the only thing making it regular and it is a fixed 5
ms. To vary one while holding the other needs a second knob — a refractory
period that moves with it, or a drive that is not a Poisson process. Sweeping
CV and sweeping $\lambda$ are therefore the same experiment in two coordinate
systems, and a result along that axis cannot say which of the two the network
is responding to.

At the default CV of **0.6** that is $\lambda = 0.133$/ms, a mean interval of
12.5 ms, 80 Hz — 40% of saturation, 2.8 spikes across a 35 ms epoch — against a
Poisson train's CV of exactly 1 and a visual cortical train's 0.5–1.0 [1]. Two
arrivals in five are discarded, which is the point: the rate sets how soon after
the dead time the neuron goes, and the refractory period sets everything else. A
low $\lambda$ leaves the refractory period rarely binding and the train nearly
Poisson (CV 0.67 at 0.1/ms); a high one drives it toward clockwork (CV 0.05 at
3.8/ms).

*Where the 0.5–1.0 comes from, and what it does not cover (references added at
Byron's request, September 14, 2026).*

1. **Softky & Koch (1993)**, *J. Neurosci.* **13**(1):334–350, "The highly
   irregular firing of cortical cells is inconsistent with temporal integration
   of random EPSPs". V1 and MT of awake behaving macaque; CV of the ISI
   distribution generally **0.5 to 1.0**, near-Poisson, for non-bursting cells
   at sustained rates up to 300 Hz. This is the source of the figure quoted
   above. Their argument is the one this model sits inside: a neuron
   integrating many random EPSPs *should* fire regularly, and cortex does not.
2. **Shadlen & Newsome (1998)**, *J. Neurosci.* **18**(10):3870–3896, "The
   variable discharge of cortical neurons". The standard answer to (1): an
   integrate-and-fire unit in a high-input regime, with excitation balanced
   against inhibition, produces CV near 1 on its own. The irregularity is a
   property of the input, not of the spike generator.
3. **Holt, Softky, Koch & Douglas (1996)**, *J. Neurophysiol.*
   **75**(5):1806–1814. The same cells fire **regularly in slice** under
   constant current and **irregularly in vivo**, which settles (2)
   experimentally and introduces CV2, the adjacent-interval measure that is
   insensitive to a drifting rate.
4. **Maimon & Assad (2009)**, *Neuron* **62**(3):426–440, "Beyond Poisson:
   increased spike-time regularity across primate parietal cortex". The
   counterexample: visual areas fire irregularly, but association and
   motor-like parietal areas are **markedly more regular**. Poisson-like
   firing is not a universal property of neocortex, so "0.5–1.0" is a claim
   about visual cortex and not about cortex.
5. **Berry & Meister (1998)**, *J. Neurosci.* **18**(6):2200–2211,
   "Refractoriness and neural precision". Retinal ganglion cells modelled as
   probabilistic firing gated by a recovery function — **this model's
   dead-time renewal process, with a soft recovery instead of a hard one** —
   and longer refractoriness makes the response *more* reproducible. The
   mechanism by which REFRACTORY pulls this system's CV down is the one they
   describe.

*What the references do not license.* Our CV is produced entirely by the dead
time, and is therefore locked to the rate by the identity above. Cortex gets
its irregularity from (2) and (3), the structure of the input, at whatever rate
it happens to be firing. So matching cortex on CV here is matching one number,
not the mechanism that produces it, and a network tuned to CV 0.5–1.0 by
lowering $\lambda$ is a network that has been made **quiet**, not one that has
been made cortical.

*Swept (Byron, September 14, 2026): CV in steps of 0.05 from 0.05 to 0.95 on
reaching_copy, six seeds, 100,000 epochs, one shared input stream per seed
(§4.5), the Rust loop.* 114 arms. The answer is **nothing across the whole
usable range, then a collapse when the input runs out**:

![CV against score](cv-reaching-100k.png)

| band | rate | spikes/epoch | score |
|---|---|---|---|
| CV 0.05–0.30, near-clockwork | 190–140 Hz | 6.7–4.9 | 0.6364 |
| CV 0.35–0.65 | 130–70 Hz | 4.6–2.5 | 0.6500 |
| CV 0.70–0.95, near-Poisson | 60–10 Hz | 2.1–0.35 | 0.6372 |

Paired on the seed, the middle band beats the fast one by 0.0136 ($t = 2.1$)
and the slow one by 0.0128 ($t = 1.1$) — neither survives contact with the
noise. The spread across the whole CV axis is 0.068, the spread across *seeds*
is 0.035, and the seed-by-CV residual has a standard deviation of 0.030. **The
noise in one cell is the size of the entire effect.**

The one real result is at the far end: everything else beats CV 0.95 by
**+0.0435 ($t = 3.65$), 6 seeds of 6**. That is 10 Hz, 0.35 spikes in a 35 ms
epoch, so most epochs present *no input at all*. It is not a fact about
regularity; it is the drive being switched off.

So over a **nineteen-fold range of input rate** — 190 Hz clockwork down to
10 Hz near-Poisson — this network scores the same. What it reads is which
neurons fired at all within the epoch (`read = "fired"`), and one spike is
enough for that, so neither the rate above one spike per epoch nor the
regularity of the train reaches the readout. The result is a property of the
read, and the way to make the input's timing matter is to change what the
teacher looks at, not how the input is driven.

*This also undercuts an earlier finding.* The 108-arm doubled_copy sweep put
$\lambda = 1$ above $\lambda = 0.1$ by 0.0172 in 6 seeds of 6 — the same axis,
in the CV coordinate 0.17 against 0.67, where this sweep finds 0.638 against
0.644, i.e. nothing. That sweep predates §4.5, so its arms did not share an
input stream; a 0.017 difference is inside the 0.030 residual measured here.
Treat it as unreplicated.

**Decided (Byron, September 14, 2026): the drive is specified by its CV, and
the default is CV = 0.6.** INPUT_CV is the constant; $\lambda$ =
INPUT_RATE follows from it as $(1-\text{CV})/(\text{REFRACTORY}\cdot\text{CV})$
and is what the schedule actually draws with. `--cv` is the flag; `--input-rate`
still takes the other coordinate and reports the CV it implies. CV is the end of
the axis worth naming because it is the quantity the literature reports, and
because it is bounded: 0 and 1 are the two physical limits, where a rate has no
natural scale.

CV 0.6 is $\lambda = 0.133$/ms, a 12.5 ms mean interval, **80 Hz, 2.8 spikes
across a 35 ms epoch**. It replaces $\lambda = 0.5$/ms (CV 0.29, 143 Hz).

*What this choice does and does not rest on.* Not on the sweep: the whole middle
of the CV axis measured flat, and CV 0.6's first place there is inside the
noise. It rests on the two things the sweep did establish and on the biology —
CV 0.6 is inside the 0.5–1.0 that visual cortex shows (Softky & Koch 1993,
[1] above), and it is far from the only region the sweep found to matter, the
starvation at CV 0.95 where an epoch holds a third of a spike. Between a
defensible prior and a flat measurement, the prior decides.

**`rate` is the default from September 14, 2026, and `forced` is what it
replaces.** *Byron, asking what the epoch resets and what it does not: "The
question I am getting at is whether there is a unified wave front in the input
process at time zero. I don't want any such thing."* There was one, and it was
total. Under forced drive every bit-1 neuron spikes at exactly $t_e$, and
because the hop is a fixed 1.667 ms the whole network is then locked to the
lattice $t_e + k\,\text{hop}$: **100% of every spike in the network**, in every
row, for the whole epoch, measured. It is not a wave front so much as a
metronome that $t_e$ starts. Under rate drive nothing fires at $t_e$ — the
first arrival is $t_e + \mathrm{Exp}(\lambda)$ — each arrival starts its own
cascade off its own phase, and **0.0%** of spikes share any lattice. No
problem overrides the drive, so this is the default throughout.

*Everything measured before this date under forced drive was measured on a
synchronous lattice*, the 0.978 two-hop solve of §6.7 included, and that
result may be about the lattice rather than about the task.

*And there is no seam at the epoch boundary, contrary to what Claude claimed
when first raising this.* `input_schedule` regenerates the process from $t_e$
each epoch rather than carrying a pending arrival across, but a Poisson
process has **independent increments**, so arrivals in $[T, T+I)$ do not
depend on anything before $T$ and restarting at $T$ *is* a continuation.
Simulated against a single unbroken process over 40,000 epochs the two agree
on everything: rate 1.0000 against 1.0004, mean gap within an epoch 0.9704
against 0.9701, mean gap straddling a boundary 2.0072 against 2.0009, and
arrival density in the first millisecond of an epoch 0.0286 against a
mid-epoch 0.0286. (The straddling gap being twice the mean is the inspection
paradox, not a seam: the interval containing any fixed instant is
length-biased. Measuring it and reading it as evidence of a seam was the
second error in the same place.)

The refractory period that matters is the **neuron's own**, so a spike the
mesh drove silences the drive too; this is why the arrivals are emitted in
full rather than thinned in advance, and why the process cannot be
precomputed. `Network.input_events` holds the (place, time) **arrivals** an
epoch used — not its spikes — because `input_schedule()` draws afresh each
time it is called. An off-rate above zero makes a zero bit a *low* rate rather
than silence; at zero it means silence, as forced drive does.

This is also the only change that lets §0's "the inputs are the outputs"
be tested rather than worked around. Under forced drive a stimulus overwrites
the neuron's state, so a read of the input zone asks what that state was
after destroying it, which is why §8's input-zone problems have stayed at
chance under every rule. Under rate drive the input arrives as spikes the
neuron's own dynamics must accommodate, and there is something left to read.
That is a prediction, not a result: no problem uses rate drive yet.

**What the teacher reads (Byron, September 14, 2026): a rate, not a bit.**
"Let's make the teacher read spike counts instead of a bit. It should
estimate the rate using an exponential decay window with time constant 5 ms:
if there are no spikes, the rate is zero." A fourth `read` joins the three
above:

$$r_j(t) = \frac{1}{\text{RATE\_TAU}}\sum_{k}
e^{-(t - t^{(k)}_j)/\text{RATE\_TAU}},$$

the sum over $j$'s spikes, read at the horizon and expressed in Hz. Each
spike puts one spike's worth on the trace and it decays with the same
constant, so a neuron that has never fired reads zero, and so does one whose
spikes are long past. The trace is a property of the neuron and is **not**
reset between epochs; at a 35 ms epoch a spike from the epoch before
contributes 0.09% of one.

Two coincidences are not coincidences. RATE_TAU = 5 ms is REFRACTORY, so a
**single spike read at its own moment reads exactly 200 Hz**; and 200 Hz is
$1/\text{REFRACTORY}$, the fastest the absolute refractory period allows.
RATE_ON is therefore saturation in the strict sense — no train can average
more.

*Claude's reading of what this costs, worth knowing before it is swept.* The
estimator is unbiased in time-average: a train at exactly the refractory
period averages 200.5 Hz over one interval. Its **instantaneous** value is
not, and swings from 117 Hz to 316 Hz across that interval depending on where
in it the read happens to fall. So a perfectly saturated neuron read at the
horizon scores a level anywhere from 0.58 to 1.00, and a perfect answer will
often not read as one. That is a property of reading an exponential window at
a single instant, not of the network.

> **⚠ The window strategy may be wrong, recorded at Byron's request
> (September 14, 2026).** *Byron:* "The biggest problem is still with the
> teacher. It is watching a 20 ms window and asking how many times the neuron
> fired. I don't want to change the window strategy — which may be WRONG
> because it does not follow the neural dynamics — for now."
>
> *Claude's reading of the objection.* Everything about this read is imposed
> from outside the network. RATE_TAU is a constant the experimenter picks,
> not anything the neuron has; the read happens at the horizon, an instant
> the experimenter picks; and none of the network's own timescales — the 2 ms
> leak, the 5 ms refractory period, the 1.67 ms hop — enters the estimate
> except by the coincidence that RATE_TAU was set equal to one of them. A
> read that followed the dynamics would ask the neuron something about its
> own state rather than counting its spikes against a wall clock. Kept for
> now, and swept rather than assumed: §8's read-window sweep varies RATE_TAU
> from 2 ms to the full 20 ms epoch.
>
> What that sweep also varies, which is worth knowing when reading it: one
> spike read at its own moment reads $1000/\text{RATE\_TAU}$ Hz, so the number
> of spikes needed to read as fully on is $\text{RATE\_TAU}/5$. At 5 ms it is
> exactly one spike, at 2 ms it is 0.4 of one — a single spike over-saturates
> and the read is a bit in disguise — and at 20 ms it is four. So the sweep
> runs from a binary read to a genuinely graded one, and the window's width
> and the read's resolution move together rather than independently.

> *Swept, September 14, 2026: eight windows from 2 ms to the full 20 ms
> epoch, six seeds, 100,000 epochs, on shallow_copy at five rows under rate
> drive with reward-modulated Hebb.* The optimum is **RATE_TAU = 5 ms**,
> which beats both ends in six seeds of six — and 5 ms is REFRACTORY, the one
> width at which a single spike reads exactly RATE_ON. So the best window is
> the most bit-like one that is not degenerate, and the read gets worse the
> more genuinely graded it becomes. The whole sweep spans 0.517 to 0.526,
> against a floor of 0.5.
>
> Scoring the trained networks under both reads settles what that means. A
> network **taught on the bit** scores 0.568 under the rate read and 0.741
> under the bit read; the best network taught on the rate scores 0.520 and
> 0.609. The rate teacher produces a worse network **by the rate's own
> measure**, so this is not a harsher yardstick applied to the same
> behaviour.
>
> And the risk §6.9 records is realised, not hypothetical: the rate-taught
> networks carry mean $|w|$ of 0.54 with 39 to 80 weights at the rails,
> against 0.34 and 9 for the bit-taught one, while firing **fewer** spikes an
> epoch (22.7 against 30.5). The teacher asks for more rate, the refractory
> period caps what it can deliver, and the weights absorb the difference.
> This is the evidence for Byron's own reservation above rather than against
> it.

> *Decided (Byron, September 14, 2026):* "We are going to go forward with bit
> reading and a five-millisecond window to get the job done." So the read is
> `window` at READ_WINDOW = 5 ms — a **bit**, but counting only spikes within
> the last 5 ms before the epoch's end — and the rate read stays built,
> swept and available rather than in use. That window is the one §8 recorded
> abandoning on September 12, for reading one phase in three of a sustaining
> neuron as off; it returns because the alternative measured worse, not
> because that objection was answered.

> *Validated at Byron's request, September 14, 2026, and the validation
> refuses half of the decision.* Eight input interspike intervals from 2 ms
> to 20 ms against that fixed read, six seeds, 100,000 epochs. The input's
> timescale barely matters: the whole sweep spans 0.523 to 0.528 and the
> per-seed winners are scattered across six of the eight intervals, so no ISI
> is distinguishable from its neighbours. (Byron predicted 12.5 ms; it came
> sixth of eight, losing to 10 ms in five seeds of six by 0.004.)
>
> What the sweep does show is that the **window** is the wrong half of the
> decision. Every arm sits near 0.525 where the same configuration read as
> `fired` scored 0.80, and cross-scoring settles that this is the teacher and
> not the yardstick: a network taught on `fired` scores **0.571** under the
> 5 ms window and 0.755 under `fired`, while one taught on the 5 ms window
> scores 0.533 and 0.623. The window-taught network is worse **under the
> window's own read**. Its weights show the §6.9 signature again — mean $|w|$
> 0.551 with 25 at the rails against 0.363 and 18, firing fewer spikes an
> epoch (20.7 against 28.3).
>
> So the bit was right and the narrowing was not. Both alternatives to
> `fired` tried so far — the rate read of §4.3 and this 5 ms window — make a
> worse network by their own measures, and both inflate the weights while
> reducing the firing. READ_WINDOW's width has not itself been swept, and
> `window` at 20 ms is `fired`, so that is one axis rather than a dichotomy.

> *Swept, September 14, 2026: READ_WINDOW over 5, 5.5, 7.5, 10, 12.5, 15 and
> 20 ms, six seeds, 100,000 epochs, input ISI 10 ms.* **Wider is better,
> monotonically, all the way to the epoch**, and every step wins in six seeds
> of six bar the last:
>
> | window | 5 | 5.5 | 7.5 | 10 | 12.5 | 15 | 20 |
> |---|---|---|---|---|---|---|---|
> | score | 0.528 | 0.532 | 0.547 | 0.576 | 0.638 | 0.783 | **0.794** |
>
> The axis is calibrated at both ends: the 20 ms arm *is* `fired`, and it
> scores 0.7935 — the rate-drive sweep's figure for `fired` at this input
> rate, to four decimals. So there is no useful width between the two, and
> narrowing the window only ever costs: 0.27 of score from 20 ms down to 5.
>
> The shape is not a learning curve but a ceiling. Every width is flat from
> its first ten thousand epochs onward — 12.5 ms reaches 0.639 by epoch
> 20,000 and is still at 0.638 at 100,000 — so the narrow windows are not
> learning slowly, they have converged to less. And what sets the ceiling is
> plain: the output row's spikes are spread nearly uniformly across the
> epoch (10–17% in each 2.5 ms slice), so a window of width $W$ can see only
> $W/20$ of them. A 5 ms window catches 20% of the spikes the network
> produces and scores 0.53; a 12.5 ms window catches 57% and scores 0.64.
> The teacher is not measuring the network, it is measuring a sample of it,
> and the sample size is the score.
>
> *So the September 12 objection was right and its remedy was too.* The
> window read was abandoned then for reading a sustaining neuron as off; it
> is abandoned again now, for the same reason measured at seven widths.
> **The read stays `fired`.**

**A noisy input (Byron, September 13, 2026).** A problem may corrupt what it
presents: each bit of the coded, permuted pattern is **flipped independently
with probability FLIP** before the row is forced. The flips come from the
network's own seeded stream, so a seed reproduces them, and nothing is drawn
when FLIP is 0, which leaves every run made before this bit-for-bit as it was.
Two patterns then exist where one did before: the **presented** pattern, which
says which neurons are forced, and the **target** pattern, the clean one, which
is what the read is scored against and what `should_fire` records. The network
is therefore asked to *repair* its input rather than merely carry it. With
population coding the redundancy is three neurons a bit, so a single flip inside
a patch is outvoted by its two neighbours, which is the whole reason the
corruption is correctable at all. FLIP = 0 collapses the two patterns into one
and this section reads as it did before.

### 4.4 Waves

Firing is queued, never recursive. A wave is every event scheduled for one
time $t$: the signals arriving, and the stimulus if an input lands then. It
has two phases:

1. **Deliver.** Every arriving signal $i \to j$ delivers $w_{ij}$ to $j$
   (§5.1). Then every neuron touched this wave settles: its potential is
   floored at MINIMUM_POTENTIAL, so the floor acts on the wave's summed input
   and the result does not depend on the order the signals arrived in.
2. **Fire.** Every forced neuron fires unless refractory; then every touched
   neuron whose potential now meets its threshold, and is not refractory,
   fires (§5.2). Each firing neuron's active outgoing connections are
   scheduled for $t + h$. The refires in the wave then learn (§6).

A neuron may fire any number of times, the refractory period permitting; a
tight loop of about REFRACTORY_HOPS hops can bring a neuron's own spike
back to refire it. A refractory neuron ignores every signal, forced
stimulus included; the delivery is still recorded as such. Each connection
stamps the time of the last signal its target actually integrated: that is
the only trace of activity the target has (§6.4).

### 4.5 The input stream — decided (Byron, September 14, 2026)

A run's inputs are **drawn up front**, one raw-bit pattern per epoch, from a
stream of their own: `random.Random(f"walnutbutter inputs {seed}")`, which is
not the network's stream and is consumed by nothing else. Ask for a million
epochs and a million patterns are generated before the first one runs.

*Why it is not the same as drawing them as you go.* Every run before this date
drew its input bits from the network's own `_rng` — the same stream that placed
the small-world shortcuts, drew the random weights and shuffled the
permutation. That stream is therefore consumed differently by every network
that is built differently. Two arms of a sweep that differ in reach, in rows,
in omega, or in anything that changes how much randomness the build eats,
**saw different input sequences at the same seed**. They were comparable on
average, over enough epochs, and not comparable epoch by epoch; and a paired
seed comparison between two such arms was pairing the seed, not the inputs.

With an input stream, seed $s$ means the same million patterns in the same
order whatever the network is, so two arms differ only in what is under test.
The Poisson arrival times of §4.3 are *not* part of the stream: they depend on
INTERVAL and on $\lambda$, which are themselves things a sweep varies, so they
stay drawn per epoch from the network's stream.

`network.input_stream(count, raw_bits, seed)` makes one and
`Network.use_input_stream(patterns)` attaches it; `--input-seed` does both from
the command line, and the drivers pass a stream to every arm. A run longer than
its stream cycles it. Without a stream a network draws as it always did, so
nothing that does not ask for one changes.

## 5. Activation rule — open

Per §0 as revised, the neuron is leaky integrate-and-fire with an absolute
refractory period.

### 5.1 Integration, with a lazy leak

Nothing happens to a quiet neuron. When a signal of weight $w$ arrives at
time $t$, the potential is first decayed for the time since it was last
brought up to date, then the signal is added:

$$p \leftarrow p \, e^{-(t - t_{\text{last}}) / \tau}, \qquad t_{\text{last}} \leftarrow t, \qquad p \leftarrow p + w,$$

with $\tau$ = TAU (Byron, September 12, 2026, bringing the leak back:
"I don't like the infinite impulse response without the leak"; the default
is the 2 ms his earlier sweep chose, and a hop is now 1.7 ms, so the leak
acts within a cascade as well as between inputs). The exploration noise
(§6.1) is added on top of the leaked potential at the input's moment, and
a bored neuron's check (§5.4) compares the leaked potential against the
falling threshold. Inhibition ($w < 0$) pushes the potential down, and
once per wave the floor applies: $p \leftarrow \max(p, \text{MINIMUM\_POTENTIAL})$.

### 5.2 Firing

A neuron fires in a wave iff $p \ge \theta$ and it is not refractory. The
spike resets it and is remembered, along with the spike before it:

$$p \leftarrow 0, \qquad t_{\text{prev}} \leftarrow t_{\text{fired}}, \qquad t_{\text{fired}} \leftarrow t .$$

A forced input neuron fires at its input's time regardless of $p$ and
$\theta$, refractory period permitting.

**The potential axis scales with fan-in — decided for now (Byron,
September 14, 2026: the threshold out of §3.4's three, then the floor with
it).** THRESHOLD is one number, but a neuron with 79 incoming synapses is
not being asked the same question as one with 18: the $\theta$ that makes an
interior grid cell selective makes a goo neuron (§3.4) fire on nearly
anything. So the potential axis is quoted *per unit of fan-in*, and both of
the points on it move together. A container that scales starts neuron $j$,
of in-degree $d_j$, at

$$\theta_j = \text{THRESHOLD} \cdot \frac{d_j}{F}, \qquad
p^{\min}_j = \text{MINIMUM\_POTENTIAL} \cdot \frac{d_j}{F}, \qquad
F = \text{THRESHOLD\_FAN\_IN} = 18,$$

the in-degree the 0.25 and the −1 were chosen against: an interior cell's
two hex rings at REACH 2. A neuron wired like that cell keeps both exactly.
The rule is linear because that is what "scales with fan-in" says without
further instruction.

*Measured, September 14, 2026 (§3.4, the count × threshold sweep):* linear
under-corrects. On copy under hebb, goo 80 saturates at every THRESHOLD up
to 0.3 — $\theta$ up to 1.32 — and comes alive at 0.5, $\theta$ 2.19,
twice what this rule gives it, scoring 0.634 with nothing stuck; at every
count measured the threshold that stops a goo saturating grows faster than
its fan-in. The rule stands as written until Byron moves it.

**The floor moves because it is not a second decision.** $\theta$ and
$p^{\min}$ are two points on one axis, and it is the axis being rescaled.
Scaling $\theta$ alone silently squeezes the usable negative range from four
times the threshold to 0.91 times it, so inhibition hits its cap while
excitation keeps piling up — and §3.4 measures that asymmetry to be most of
what saturates goo. Holding $p^{\min}/\theta$ at the grid's −4 keeps the
shape and changes only the units.

These are **starting** values only. Homeostasis and un-sticking (§1.3) move a
threshold from wherever it starts, §5.4's boredom clock reads it as before,
and a checkpoint stores the thresholds and floors a run actually reached
rather than recomputing them.

**Goo scales; nothing else does yet.** Turning it on for the grid would move
every threshold every result to date was measured at, and that is its own
decision — `--scale-with-fan-in` makes the rule available to any container
so the question can be asked without a code change. On the grid it is not a
no-op even in principle: in-degree runs from 7 at a corner to 24, so
seventeen distinct thresholds replace the one.

### 5.3 Refractory period

A neuron that fired at $t_{\text{fired}}$ is refractory while
$t < t_{\text{fired}} + \text{REFRACTORY}$. While refractory it ignores every
signal, does not integrate it, and cannot be forced. This is a feedback
control mechanism and computational feature of the system (§0).

### 5.4 Threshold homeostasis: bored neurons — decided for now

*Decided (Byron, September 12, 2026):* we need a form of threshold
homeostasis so bored neurons tend to fire after about 200 ms all on their
own. *Claude's reading, built:* the threshold a neuron faces falls
linearly with the silence since its last spike,

$$\theta_j(t) = \theta_j \Big(1 - \frac{t - t_{\text{fired}}}{\text{BORED\_AFTER}}\Big),$$

reaching zero at BORED_AFTER and continuing below it, so a neuron whose
potential sits under zero fires later still; a spike resets it to
$\theta_j$. A neuron that has never fired counts as silent since the
clock started. Every wave now checks every neuron, touched or not, so a
bored neuron fires at the first wave after its threshold has fallen to
its potential: in a live mesh within a hop, in a dead one at the next
input. (Before this, a neuron was only checked when a signal reached it,
which also left a neuron recovered from its refractory period with enough
potential waiting to be touched; both engines now fire it at once.) The
spike it fires on its own is a spike like any other: it releases by its
delay, which at 200 ms is nothing, and it resets the clock on its
boredom.

## 6. Learning rules — open

*Byron, September 13, 2026, and it reorganises this section:* "What I
realised today is that there is not a neuron training rule. There are
multiple compatible training rules."

So §6 is not a menu of alternatives with one switch. It holds two kinds of
rule, and they compose:

- **Local rules**, which need nothing but the neuron's own spikes and the
  stamps on its synapses, and which run under every configuration: the
  quash (§6.11), leaky Hebb (§6.12), and the weight decay (§6.8). Each has
  its own rate and each is off until asked for.
- **A rule that pays at the read**, of which at most one runs: the external
  teacher (§6.9), ADALINE (§6.10), dopamine (§6.2–6.6) or REINFORCE (§6.7).
  RULE names it, and `RULE = local` means none of them does and the local
  rules are the whole of the learning ("no teacher for now", Byron, the
  same day).

Within a wave the local rules run in a fixed order — quash, then leaky Hebb
— so a potentiation is not discounted the moment it is made. Only the quash
depends on the weight it acts on, so no other order matters.

Per §0: a neuron becomes eligible for dopamine release when it fires.
Dopamine is the global reward used for reinforcement learning. It is
produced locally and consumed globally.

*Decided (Byron, September 11, 2026):* Learning happens when a neuron that
previously fired fires again and is proportional to the global dopamine
value. We can take care of this when the neuron fires. The strategy of
REINFORCE remains the same as much as we can keep it with the global
dopamine reinforcer.

*Decided (Byron, September 12, 2026):* the dopamine eligibility is
calculated both locally and lazily. It is a property of a neuron. Signals
are NOT traced back to their ancestors. The only trace available to the
neuron is the activity across its synapses.

*Decided FOR NOW (Byron, September 12, 2026):* the eligibility is the same
for all of a neuron's weights. For now, dopamine is released and then all
weights move together. *Open question:* whether the dopamine release or the
weight update should happen first.

### 6.1 Exploration — revised, per wave

*Revised by Byron, September 13, 2026, on the diagnosis below.* Every neuron
$j$ (inputs included) draws $\xi_j \sim \mathcal{N}(0, \sigma^2)$ with
$\sigma$ = SIGMA and adds it to its potential, floored. The draw is taken
**afresh before every wave's firing decision**, after that wave's signals
have been integrated and settled and before any neuron fires, so a neuron's
$\xi_j$ is the perturbation it actually decided under. EXPLORE = epoch
restores the pre-alpha's single draw at the input's moment. Both engines draw
from the same Box-Muller stream, so a seed gives the same noise whichever
engine runs; 0 switches it off.

A neuron that has already fired this epoch **keeps** the $\xi_j$ it decided
under rather than taking the new draw, because §6.7 credits one $e_j$ per
epoch and it must refer to the perturbation that produced the spike, not to a
later one that explains nothing. Its potential still takes the new draw: the
hold is on the record, not on the dynamics.

*Why (Claude, September 13, 2026, and it is a diagnosis rather than a
result).* §6.7's score function $e_j = \xi_j/\sigma$ is what makes that rule
a gradient estimator, and under a single draw at $t_e$ it was formally
correct about the wrong event: a neuron firing 12 ms later has spiked and
reset in between, so its $\xi_j$ carries no information about the decision
being credited. That rule stayed at chance under both input drives — 0.568
forced, 0.543 rate — while the deterministic reward-modulated Hebb of the
same section reached 0.978 and 0.80. Drawing per wave is the change that
follows: it does not touch §6.7, only what $\xi_j$ refers to.

Note what the leak does to the scale. A draw lands every hop and decays with
TAU, so the accumulated perturbation reaches a standard deviation of
$\sigma/\sqrt{1 - e^{-2\,\text{hop}/\text{TAU}}} = 1.11\,\sigma$ at the
default constants — the noise is refreshed at each decision without
accumulating, and SIGMA keeps very nearly the meaning it had.

*Measured, September 13, 2026, and the diagnosis did not survive it.* The
same thirty arms — perturb at $\sigma$ 0.1 under rate drive on shallow_copy
at five rows, five input rates by six seeds, 100,000 epochs — score **0.533
± 0.023** drawing per wave against **0.543 ± 0.017** drawing per epoch, and
beat the per-epoch arms in 11 of 30 paired runs. The change is a wash, and if
anything slightly worse. Reward-modulated Hebb on the identical arms scores
0.799 and beats wave exploration in 30 of 30.

So the gradient form's difficulty is not *when* $\xi_j$ is drawn. Three
things have now been tried against it — the timing here, the missing
presynaptic factor (§6.12's trace, which also made it worse), and the
baseline (§6.7, whose variance reduction measures at nothing on this problem)
— and none moves it off chance, while the rule that abandons the gradient
entirely reaches 0.80 under rate drive and 0.978 under forced. That is
evidence about the estimator's variance rather than about a missing factor in
it, and it is the reason §6.1 keeps EXPLORE as a switch rather than a
decision.

One thing the per-wave draw does show: it is the only arm of the three that
**moves**. Its mean rises 0.514 → 0.530 over the hundred thousand epochs
while the per-epoch arms sit flat at 0.53–0.54 throughout. Slow, small, and
not yet worth anything.

Nothing measured before this revision is affected by it. The hook is off
whenever $\sigma$ is 0, which is what the hebb eligibility runs at, so every
result in §6.7 and §8 obtained with that eligibility stands unchanged.

### 6.2 Release

A neuron whose previous spike was at $t_{\text{prev}}$ and which fires
again at $t$ (necessarily $t \ge t_{\text{prev}} + \text{REFRACTORY}$), a
delay $\delta = t - t_{\text{prev}} - \text{REFRACTORY}$ past the end of its
refractory period, releases the gamma density of that delay (Byron,
September 12, 2026, replacing the exponential
$e^{-\delta / \text{DOPAMINE\_RELEASE\_TAU}}$):

$$d_j = \frac{\delta^{\alpha - 1} e^{-\delta / \theta}}{\Gamma(\alpha)\,\theta^{\alpha}},
\qquad \alpha = \text{DOPAMINE\_RELEASE\_ALPHA} = 2,\ \theta = \text{DOPAMINE\_RELEASE\_THETA} = 1\text{ ms}.$$

*Claude's implementation note:* the density is averaged over the hop that
follows the delay, $d_j = \big[F(\delta + h) - F(\delta)\big] / h$ with $F$
the gamma distribution function and $h$ the hop. That is the density as the
hop shrinks, and it is finite for every $\alpha$: for $\alpha < 1$ the
density itself is infinite at zero delay, and an instant refire is the
common case here (the first sweep arm at $\alpha = 0.5$ read "dopamine
inf" within a minute). With $\alpha = 2$, $\theta = 1$ ms an instant refire
releases about 0.3, the release peaks a little later, and it decays from
there. *Tension with §0*, which says maximum dopamine is released when the
neuron fires immediately after the refractory period: for $\alpha > 1$ the
maximum comes later, and $\alpha \le 1$ recovers §0's shape. For Byron and
Cedric to settle in §0. A neuron's first ever spike releases nothing.
Forced neurons are neurons (§0): a forced refire releases and learns like
any other.

### 6.3 The global value and the expectation

The global dopamine value $D(t)$ is a pool every release adds to, decaying
lazily with DOPAMINE_TAU: $D(t) = D(t_0)\, e^{-(t - t_0)/\tau_D}$ between
events. Reading it does not deplete it.

*Decided (Byron, September 12, 2026):* instead of a baseline, actual
reinforcement learning with an internal expectation, `dopamine_expected`,
calculated globally under the assumption that dopamine is global. For now,
the actual counts (not biologically realistic) of dopamine to date divided
by the current time. The reinforcement is proportional to
(dopamine − dopamine_expected), not to dopamine.

*Decided (Byron, September 12, 2026), replacing the counts-over-time
estimator, which could only creep:* start with dopamine_expected = 0 and use
an exponential decay window of 10 minutes for the expected dopamine trace.
At every wave, after the wave's releases and updates, the expectation moves
toward the value by the fraction of the window that has elapsed since the
last wave:

$$E \leftarrow E + \big(1 - e^{-\Delta t / \tau_E}\big)\,(D(t) - E),
\qquad A(t) = D(t) - E ,$$

with $\tau_E$ = DOPAMINE_EXPECTATION_TAU and $A$ read before the move. This
may start off slow but may eventually get us where we are trying to get.

### 6.4 Eligibility — for now

One value per neuron, computed when it refires, from nothing but its own
spikes and the stamps on its synapses (§4.4): $e_j = d_j$, its own release
amount. A tight refire is both the biggest release and the biggest
eligibility.

### 6.5 The update — gated, for now

When neuron $j$ refires at $t$, every incoming connection $i \to j$ that
carried a signal $j$ integrated since its previous spike (the gate,
$x_i = 1$; otherwise $x_i = 0$) moves together:

$$w_{ij} \leftarrow \mathrm{clip}\big(w_{ij} + \text{LR} \cdot A(t) \cdot e_j \cdot x_i,\ \text{WEIGHT\_RANGE}\big).$$

This keeps REINFORCE's shape, presynaptic activity × postsynaptic
eligibility × global signal, with the global signal now $A(t)$.

*Decided (Byron, September 12, 2026):* the system has no knowledge of how
it is being scored: a neuron that should fire is forced, checked, and
rewarded or anti-rewarded by the rule, but a neuron that SHOULD NOT FIRE
(input bit 0) is free to be recruited into sustaining the others. For
neurons that should not fire, the learning rule is applied but the reward
is reversed upon firing: if the dopamine is greater than expected, the
neuron is punished for firing instead of rewarded. In the update above,
$\text{LR}$ takes a minus sign for a refiring input neuron whose bit this
epoch is 0. Its release, and hidden and forced neurons, are unchanged.
DOPAMINE_PUNISH switches it (`--no-punish`).

### 6.6 Order within a wave

All the refires in one wave are handled together: their releases are
computed, then (DOPAMINE_ORDER = release-first) added to the pool before
$A(t)$ is read and the updates applied, or (update-first) after. Release
first lets a neuron's own release count toward what it consumes; update
first keeps the signal strictly what others released. A switch, to be
swept.

### 6.7 The reinforce rule — factored out

RULE = reinforce is the rule of the pre-alpha, kept for comparison and run
by the Teacher on a trained problem (§8). One scalar reward per epoch: the
**row** critic scores the fraction of output neurons whose fired state this
epoch matches the target pattern (decoded critics read the row as a code
word first). A running baseline $b$ (BASELINE_RATE) gives the advantage
$A = R - b$. For every connection $i \to j$ that carried a signal this
epoch into a neuron $j$ not forced this epoch,
$w_{ij} \leftarrow \mathrm{clip}(w_{ij} + \text{LR} \cdot A \cdot e_j)$ with
$e_j = \xi_j/\sigma$ (ELIGIBILITY = perturb) or $\pm 1$ by whether $j$ fired
(hebb). LATE says what a signal arriving after its target fired earns:
count, ignore, or depress. The Teacher also keeps each unforced neuron's
firing rate ($r_j$, RATE_MEMORY) and drifts thresholds toward TARGET_RATE
(HOMEOSTASIS), nudging stuck outputs faster (UNSTICK), within
THRESHOLD_RANGE. Under RULE = dopamine the Teacher only scores and reports;
the network learns by §6.2–6.6.

*Byron, September 13, 2026: "I want to try the same thing with the REINFORCE
algorithm."* So §6.12's trace is appended to this rule's chain of
multiplications too, behind LEAKY_ELIGIBILITY (`--leaky`, off by default so
the pre-alpha's rule is unchanged):

$$w_{ij} \leftarrow \mathrm{clip}\Big(w_{ij} + \text{LR}\cdot A\cdot e_j\cdot
e^{-(t^{\text{read}}_j - t^{\text{fired}}_i)/\text{TAU}}\Big),$$

with $t^{\text{read}}_j$ the moment $j$'s answer was fixed: its spike if it
fired, and the arrival of the signal itself if it did not. It multiplies the
existing eligibility rather than replacing it, so $e_j$ keeps the per-neuron
sign the rule already has and the trace adds per-synapse resolution the rule
never had: without it every synapse that delivered into $j$ receives the same
update whatever it delivered.

*Claude's reading of the non-firing case, and why it is not the horizon.* A
neuron that never fired offers no moment at which its synapses can be told
apart, so all of them take $e^{-\text{hop}/\text{TAU}} = 0.435$ and only the
scale changes. Evaluating the trace at the horizon instead — the first
reading, built and measured — annihilates that half of the signal entirely:
at TAU 2 ms over a 20 ms epoch the trace there is $5\times10^{-5}$, and
since $e_j = -1$ for a non-firing neuron under the hebb eligibility, it is
exactly the **depressive** half that vanishes. Measured on shallow_copy at
two rows it took the rule from 0.948 to 0.626 and the activity from 12.8 to
14.1 spikes an epoch, and at five rows from 26 to 129.

*Measured, September 13, 2026, and it is the largest result on the branch so
far.* The trace hurts this rule wherever it is tried — but trying it turned
up what does work. With ELIGIBILITY = hebb the Teacher sets $\sigma$ to zero,
so there is no exploration noise at all and $e_j = \pm 1$ by whether $j$
fired: the rule is then **reward-modulated Hebb**, a Hebbian sign times a
global advantage $A = R - b$, with no per-neuron target and no perturbation
to correlate against. On shallow_copy, reading each network at the $\sigma$ it
was trained with:

| rule | 2 rows | 5 rows, $\omega$ 0 (two hops) |
|---|---|---|
| REINFORCE, hebb | 90.5%, 14/16 responses | **97.5%, 16/16, all twelve right in 88.2% of epochs** |
| REINFORCE, hebb + leaky trace | — | 54.0%, 4/16 |
| REINFORCE, perturb | 75.4%, 16/16 | 56.8% |
| ADALINE | 97.1%, 16/16 | 61.3%, 5/16 |

So the two-hop problem is solved, and solved by the rule that gets **one
global number** and no target at all — while ADALINE, which is handed a
per-neuron error, cannot do it, because §6.10 moves only the readout's
incoming weights. What reaches the interior is not a richer signal but a
rule that is allowed to touch it. This retires the reading of §6.10 that
credit assignment to the interior was the thing not yet built: it was built,
in §6.7, and had only ever been run with the perturb eligibility.

It also qualifies §6.12. leaky_hebb run without a teacher collapses the
network to a single response; the same Hebbian sign with a global factor
multiplying it solves two hops. The missing ingredient there was the third
factor, not competition.

*Asked by Byron, September 13, 2026: can we use REINFORCE instead of
ADALINE?* The rule may now be asked for on any problem, not only a trained
one; the gate that refused it was plumbing, not a decision, and nothing
outside the network moves a threshold either way. But on a problem that
reads the **input zone** the rule as written above is blind on half of what
it scores, for two independent reasons, and both are §6 decisions to make
rather than bugs to fix:

1. **The forced skip.** The update passes over every connection whose
   target was forced this epoch, because "its firing was not the network's
   doing". That is exactly right when the read is *fired this epoch* and
   exactly wrong when the read is *spiked again* (§4.3): under that read the
   forced spike is not what is scored, the refire is, and the refire is the
   network's doing. Measured on population_denoise, 74 of the 166 synapses
   entering the scored zone are skipped.
2. **The eligibility says nothing about a refire.** $e_j = \xi_j/\sigma$ is
   the exploration noise added to $j$'s potential at $t_e$; a forced neuron
   spikes regardless of it and resets, so the draw cannot explain whether it
   refired 6 ms later. $e_j = \pm 1$ by `has_fired` (hebb) is $+1$ for every
   forced neuron by construction. In one measured epoch the five forced
   neurons read True, True, True, True, True under hebb while the read they
   are scored on was False, True, True, True, False.

Three forms follow, and which one is meant is open. (a) The rule exactly as
written, which now runs. (b) The read as the eligibility: lift the skip
under `read = "again"` and take $e_j = \pm 1$ by whether $j$ refired, which
is the score function of a Bernoulli "did I refire" policy. (c) REINFORCE on
§6.9's trace: keep the teacher's eligibility, the release on gated synapses,
and pay it with the advantage $A = R - b$ in place of the raw score $S$ —
the same machinery as the current rule with one substitution. Under (a) and
(b) there is no weight decay (§6.8), because the decay is carried by the
dopamine pool and the reinforce rule runs without one.

### 6.8 Earned activity — decided for now

*Proposed by Claude, September 12, 2026, after the punishment rule (§6.5)
moved the weights into the input row but not the score; decided by Byron
the same day: "I do want synapses that forget on their own. For now."* The
diagnosis:
under these constants the mesh reverberates on its own, so every input
neuron is driven every 6 ms whatever its bit, and the same synapses carry
that drive in an epoch where it should fire and in one where it should
not. Reward on the one and punishment on the other land on the same
weights and cancel. Nothing a neuron can see distinguishes the two
epochs except its own forced spike, and the rule gives that spike no role
because the rest of the mesh fires the same either way. Two decisions
together would give it one; either alone does not.

1. **Punishment outweighs reward.** A refire that should not have
   happened is punished DOPAMINE_PUNISH_GAIN times as hard as a refire that
   should have is rewarded (proposed 2). For an input neuron driven by an
   autonomous loop, the net over its bit-1 and bit-0 epochs is then
   negative, and its incoming weights fall until the loop no longer fires
   it unforced. For a loop that only runs when the neuron's own forced
   spike starts it, there are no bit-0 refires to punish, the net is
   positive, and it grows. The fixed point of the rule is the boundary the
   task asks for: on when forced, off when not. Alone this fails, because
   in a fully reverberating mesh the drive does not depend on the forced
   spike, so the neuron falls silent in both cases and the score stays at
   chance.

2. **Weights decay toward zero.** Every weight moves toward 0 by a fraction
   WEIGHT_DECAY per epoch (proposed $10^{-4}$: about two hundred epochs of
   decay per learning step at LR 0.02). Activity that is not rewarded on
   net dissolves, the intrinsic reverberation included, so sustained
   activity has to be earned, and the only structures that earn it on net
   are loops that depend on a forced spike. Alone this fails too: a quiet
   mesh gives nothing to reward until exploration noise finds a refire, and
   without (1) an autonomous loop, once found, is net neutral and survives.

Together: decay empties the mesh of unearned reverberation, exploration
seeds refires, reward grows the loops that a forced spike starts, and the
asymmetric punishment dissolves any loop that learns to run without one.
Both are single constants and stay local: DOPAMINE_PUNISH_GAIN
(`--punish-gain`) and WEIGHT_DECAY (`--weight-decay`), the decay applied
to every weight once per epoch, after the epoch's waves, in both engines.
The alternative not taken: a global inhibition proportional to the mesh's
activity, which also makes reverberation something to be earned but
couples every neuron to the whole.

### 6.9 The external teacher — decided, the current rule

*Decided (Byron, September 12, 2026):* the dopamine student-as-teacher is
very hard to figure out for now, so revert to an external teacher. This
teacher assigns reinforcement globally according to a score:

- $+0.25$ for a forced-input neuron that sustains,
- $+0.25$ for an input neuron whose input is zero and does not fire,
- $-0.25$ for a forced-input neuron that fails to sustain,
- $-0.25$ for an input neuron whose input is zero and does fire.

Thus the possible scores for any input on reading its sustain behaviour are
$-1$, $-0.5$, $0$, $0.5$ and $1$ (four input neurons at TEACHER_CREDIT
each; a wider zone widens the range in the same steps).

**Scoring a rate (Byron, September 14, 2026): "the target rates will be 200
Hz on and 0 Hz off. FOR NOW we drive the neurons toward saturation if they
are firing correctly and silence if they are not."** The read of §4.3 becomes
a **level** $\ell_j = \mathrm{clip}(r_j/\text{RATE\_ON}, 0, 1)$, with 0
silence and 1 saturation, and every score in this section is computed from
$\ell_j$ in place of the bit:

$$S = \frac{1}{n}\sum_j \big(1 - 2\,|\ell_j - d_j|\big), \qquad
R = \frac{1}{n}\sum_j \big(1 - |\ell_j - d_j|\big),$$

with $d_j \in \{0, 1\}$ the target bit. When $\ell_j$ is 0 or 1 these are
exactly the old $\pm$credit score and the old row critic, so every rule
scored under a boolean read is scored as it always was, and $S = 2R - 1$
still holds. ADALINE's per-neuron error becomes $d_j - \ell_j$, now
continuous on $[-1, 1]$ rather than $\{-1, 0, +1\}$.

> **⚠ A decision with a known risk, recorded at Byron's request (September
> 14, 2026).** The target rates are the two **extremes** of what a neuron can
> do: saturation for an output that should be on, silence for one that should
> be off. There is no intermediate set point anywhere in this rule. A neuron
> that is firing correctly is pushed to fire *faster* no matter how fast it
> is already going, and the only thing that stops it is the absolute
> refractory period — a cap on the spikes, not on the weights driving them,
> which go on growing until they hit WEIGHT_RANGE. This could have drastic
> consequences if neurons are driven into unstable regions, and it is adopted
> FOR NOW with that understood rather than because the extremes are believed
> to be the right targets. What to watch: weights at the rails, spikes per
> epoch climbing without the score climbing, and the §5.4 bored-neuron
> threshold interacting with a population that never falls silent. RATE_ON is
> tunable (`--rate-on`), so a middling target is one flag away when it is
> wanted.

*Claude's reading, built:* the score is known only at the read, so the
epoch's credit is carried by an **eligibility trace**. Everything in
§6.2–6.5 stands except who supplies the global signal and when the weights
move. During the epoch a refiring neuron's release (§6.2) is added to the
`eligibility` of each of its gated incoming synapses (the gate of §6.5: a
synapse that carried a signal the neuron integrated since its previous
spike) instead of moving that synapse. At the read the teacher computes the
score $S$ and every synapse moves by

$$w_{ij} \leftarrow \mathrm{clip}\big(w_{ij} + \text{LR} \cdot S \cdot e_{ij},\ \text{WEIGHT\_RANGE}\big),$$

with $e_{ij}$ the trace, which is then cleared: an epoch's credit never
carries into the next. The dopamine pool still runs and is still reported,
but it no longer decides anything. The §6.5 punishment of a bit-0 input
neuron is switched **off** under this rule, because the teacher's score
already knows which neurons should not have fired and reversing the sign
locally would double-count it; `--punish` turns it back on. Weight decay
(§6.8) and the bored neurons (§5.4) are unchanged. A problem names the rule
it is posed for: reversal the reinforce rule, sustain_inputs and
improved_sustain the teacher.

### 6.10 ADALINE with an eligibility trace — decided, built

*Decided (Byron, September 13, 2026):* implement the ADALINE learning rule
with an eligibility trace.

*Claude's reading, built (`--rule adaline`).* Widrow-Hoff assigns each
output its own error rather than broadcasting one scalar, so this is a
**departure from §0's global reinforcement**: the signal is no longer
global, and only the scored neurons' incoming weights learn. Noted for
Byron and Cedric, not smuggled.

The error at output neuron $j$ is desired minus actual,

$$\epsilon_j = d_j - y_j \in \{-1, 0, +1\},$$

with $d_j$ the neuron's bit this epoch (1 it should sustain, 0 it should
not fire) and $y_j$ whether it was read as on (§8). A neuron read correctly
has $\epsilon_j = 0$ and moves nothing: ADALINE corrects mistakes only,
where the teacher of §6.9 paid every synapse on every epoch.

The eligibility trace is the presynaptic activity, accumulated through the
epoch because the input to a spiking neuron is spread over it: each synapse
counts the signals its target actually integrated along it (a signal
dropped into a refractory neuron does not count),

$$x_{ij} = \#\{\text{signals } i \to j \text{ integrated this epoch}\}.$$

At the read every synapse moves by the rule's own form, and the trace is
cleared so an epoch's activity never counts twice:

$$w_{ij} \leftarrow \mathrm{clip}\big(w_{ij} + \text{LR} \cdot \epsilon_j \cdot x_{ij},\ \text{WEIGHT\_RANGE}\big).$$

A synapse whose target is not scored has $\epsilon_j = 0$, so the mesh
behind the scored neurons is an untrained reservoir and only the readout
learns, which is what a single-layer rule means here. True ADALINE uses the
analog net input rather than the binary output; a spiking neuron that
resets has no such quantity to hand, so the binary form is used and the
difference is noted. The dopamine pool still runs and is still reported but
decides nothing, the bit-0 punishment is off (the error already knows), and
the exploration noise, the bored neurons and the weight decay are
unchanged.

### 6.11 Quashing cycles — decided, built

*Byron, September 13, 2026, turning the rule around:* when a neuron fires
itself, it has found a cycle. Cycles may be BAD. The whole point of having
an absolute refractory period is to quash short cycles. Sustain may be
exactly what we don't want a neural network doing. We may have had the
learning rule backward: cycles need to be quashed, so move the contributing
weights in the direction that will actually do this. The quash is
proportional to the synaptic gating, the weight, and an exponential decay
$e^{-k(t - t_{\text{fired}})}$.

*Built.* A neuron cannot tell its own returning spike from anyone else's
without tracing ancestry, which §6.4 rules out, so "fires itself" is read
as a refire, and the delay since the previous spike is the evidence of how
tight the loop was (Byron, September 13, 2026: "the refire delay is
evidence of how tight the loop is"). When a neuron refires at $t$ with its
previous spike at $t_{\text{prev}}$, every incoming synapse that carried a
signal it integrated since that spike, the gate of §6.5, moves by

$$w_{ij} \leftarrow \mathrm{clip}\Big(w_{ij}\big(1 - \text{QUASH\_RATE}\,
e^{-\text{QUASH\_K}\,(t - t_{\text{prev}})}\big),\ \text{WEIGHT\_RANGE}\Big).$$

Being proportional to the weight it pulls toward zero from either side, and
being proportional to the gate it touches only the synapses that carried
the cycle. It is local, lazy, needs no external signal, and runs under
every rule, so it composes with the teacher rather than replacing it;
*Measured, September 13, 2026: the quash is not optional.* On shallow_copy at
five rows with no shortcuts — a genuine two-hop copy — the rule of §6.7 with
the hebb eligibility reaches 0.978 with QUASH_RATE 0.02 and 0.965 with 0.1,
and **0.502 with the quash off**, where the mesh reverberates at 220 spikes an
epoch against 28 and learns nothing at all. Byron's reading of the refractory
period was right: without something that closes short loops the network has
no quiet in which a reward can mean anything. The earlier reading, that the
quash turns destructive with depth, came from running it against §6.10, which
trains only the readout and so cannot compensate for what the quash removes;
it does not generalise to a rule that reaches the interior.

QUASH_RATE 0 switches it off, and a problem says whether it quashes. The
constants are to be explored (Byron: "we will have to explore this space"),
and the driver sweeps `quash` and `quash_k`.

*What it answers.* Every sweep of September 12 and 13 ended with the mesh
either reverberating or dead (`docs/findings-2026-09-13.md`), and the rules
that paid for refires drove it to whichever of those its constants chose.
Quashing inverts the sign on exactly that quantity, so activity that loops
gets weaker and activity that crosses the mesh once does not: the dynamics
are self-limiting for the first time. §6.10's note, that a refire inside
the epoch that forced a neuron is the one pattern-dependent event a neuron
can observe locally, still holds; the change is that such a cycle is now
treated as something to remove rather than to reward.

### 6.12 Leaky Hebb — decided, built

*Byron, September 13, 2026.* The system is primarily defined through two
time constants: the leak rate parameter, and the absolute refractory window
implicitly defined through refractory_hops. It is also defined by the
**synapse's** leak rate parameter, which we take for computational
simplicity to be the same as the postsynaptic neuron's leak rate parameter.
The leak rate fully specifies an exponential-decay window

$$f\big(t - t^{\text{fired}}_i\big) = e^{-k\,(t - t^{\text{fired}}_i)},$$

and we define the **leaky eligibility trace on the synapse** as that $f$.
For Hebbian learning the weight update rule would normally be $w_{ij}
\leftarrow w_{ij} \times \text{pre\_gate} \times \text{post\_gate} \times
\text{LR}$, or somesuch; we are going to append $f(t - t^{\text{fired}}_i)$
to the chain of multiplications. Call this **leaky_hebb**.

*Claude's reading.* With $i$ presynaptic and $j$ postsynaptic as everywhere
in §6, and $k = 1/\text{TAU}$: at the moment $j$ fires, at $t =
t^{\text{fired}}_j$, every incoming synapse moves by

$$w_{ij} \leftarrow \mathrm{clip}\Big(w_{ij} + \text{LR}\; g_{ij}\;
e^{-(t^{\text{fired}}_j - t^{\text{fired}}_i)/\text{TAU}},\
\text{WEIGHT\_RANGE}\Big),$$

where $g_{ij}$ is the pre-gate of §6.5 — the synapse carried a signal $j$
integrated since $j$'s previous spike — and the post-gate is the occasion
itself: nothing happens on a synapse whose target did not fire. The chain
adds to $w_{ij}$ rather than multiplying it, since gates of 0 and 1
multiplying a weight would zero it rather than leave it alone.

**Equal leaks bought an identity — and broke the rule.** *Revised by Byron,
September 13, 2026: the synapse's leak is its own constant, SYNAPSE_TAU,
and is no longer taken equal to the neuron's.* The identity below is real,
and it is what the equality buys; the measurement that follows is why it was
not worth buying. A signal of size $w_{ij}$ that arrived at $t_a$ is still
contributing $w_{ij}e^{-(t - t_a)/\text{TAU}}$ to $j$'s potential at $t$
(§5.1), and travel takes exactly one hop, so $t_a = t^{\text{fired}}_i +
\text{hop}$ and

$$e^{-(t^{\text{fired}}_j - t^{\text{fired}}_i)/\text{TAU}} =
e^{-\text{hop}/\text{TAU}} \cdot \frac{\text{what that synapse still
contributed to } p_j \text{ when } j \text{ fired}}{w_{ij}}.$$

The trace is exactly proportional to the residual charge the synapse still
had in the neuron at the moment of the spike it helped cause — absent the
floor (MINIMUM_POTENTIAL), and outside the reset, which the gate already
excludes. Setting the synapse's leak to the neuron's is what buys that
identity: with two different constants the rule credits a synapse by
something other than its contribution.

*And that is the trade, measured.* The trace is a gradient — it equals
$\partial p_j/\partial w_{ij}$ at the spike, to six figures, times the
constant $e^{\text{hop}/\text{TAU}}$ — so appending it makes a rule **more**
gradient-like with respect to the potential, not less. What it also does, at
the neuron's own leak, is attenuate by 148 times across the 10 ms a network
computes over, which lands essentially all the credit on the last synapse to
arrive before each spike and starves every earlier link in a chain. Sweeping
the trace's leak on shallow_copy at five rows with no shortcuts, the neuron's
leak held at 2 ms throughout and only the synapse's moving:

| synapse leak | trace's range over 10 ms | score |
|---|---|---|
| 2 ms (= TAU) | 148× | 0.500 |
| 5 ms | 7.4× | 0.983 |
| 10 ms | 2.7× | 0.981 |
| 20 ms | 1.6× | 0.973 |
| 50 ms | 1.2× | 0.984 |
| 200 ms | 1.1× | 0.970 |
| no trace at all | 1.0× | 0.978 |

Early learning slows monotonically as the trace sharpens (0.654 down to
0.506 over the first thousand epochs), which is the dose-response of a
dynamic-range effect rather than of a wrong gradient. SYNAPSE_TAU is
therefore 10 ms: clear of the cliff, and every value from 5 ms up is the
same within one seed's noise. Note what the sweep does **not** show — above
5 ms the trace neither helps nor hurts. It has stopped costing anything; it
has not yet been shown to buy anything.

*And the same sweep run on leaky_hebb itself changes nothing*, which is the
control that says what the sweep above was about. Unsupervised on
shallow_copy, with SYNAPSE_TAU at 2, 10 and 50 ms:

| | distinct responses | consistency | row fires |
|---|---|---|---|
| 2 rows, untrained | 7/16 | 36.6% | 73.9% |
| 2 rows, leaky_hebb, any leak | 1–2/16 | 85–89% | 97–98% |
| 5 rows, untrained | 5/16 | 76.6% | 84.3% |
| 5 rows, leaky_hebb, any leak | 8–12/16 | **15–17%** | 42–45% |
| 5 rows, REINFORCE hebb | 16/16 | 58.0% | 57.7% |

At two rows it collapses to one all-on answer whatever the leak. At five it
does the opposite and answers almost at random — the count of distinct
responses rises, but the modal answer holds only one epoch in six, against
77% untrained, so that is chaos rather than selectivity. Flattening the trace
moves neither outcome. The dynamic range was decisive for the trace **inside
a rule with a global factor** and is irrelevant to the trace on its own,
which is the same conclusion §6.12 reached by the other road: what leaky_hebb
lacks is the third factor, not a better-shaped eligibility.

**What the two time constants do to it.** Write $\rho = \text{REFRACTORY} /
\text{TAU} = 2.5$ and $h = \text{REFRACTORY\_HOPS} = 3$. The soonest $j$ can
fire after $i$ did is one hop, so the trace never exceeds $e^{-\rho/h} =
0.435$; one further refractory window on it is down to $e^{-\rho(1 + 1/h)} =
0.036$. The leak sets the window's shape and the refractory period its
width, and the ratio $\rho$ alone sets the dynamic range across it, here a
factor of $e^{\rho} = 12.2$.

**It needs no new state.** A connection already stamps `last_signal`, the
arrival of the last signal its target integrated, and travel is exactly one
hop, so $t^{\text{fired}}_i = \texttt{last\_signal} - \text{hop}$ is to
hand. That also settles which presynaptic spike is meant when $i$ has fired
more than once: the one whose signal $j$ actually integrated, rather than
$i$'s most recent — the local reading, and the one that makes the identity
above exact.

**Settled (Byron, September 13, 2026).**

1. *Sign.* leaky_hebb potentiates and never depresses, and it gets no
   anti-Hebbian arm of its own, because the depression is another rule's
   work — see the head of §6. In practice the quash **is** that arm, and it
   tracks leaky Hebb closely without being asked to: both are driven by the
   same §6.5 gate and the same exponential, Hebb on every spike and the
   quash on every refire, and in a busy mesh most spikes are refires.
   Measured on shallow_copy at two rows, HEBB_RATE 0.01 adds 1.251 of
   weight per epoch while the quash takes 1.227 and the decay 0.012, a net
   of +0.013 on a turnover a hundred times its size. At 0.001 the three come
   to −0.0008. The pair is self-balancing; what the rate chooses is the
   turnover it balances at, and at 0.01 that equilibrium sits at the rails.
2. *No teacher, for now.* The rule is unsupervised and is meant to be:
   RULE = local. It is the first rule here that touches the **interior** of
   the network at all — §6.10 moves only the readout's incoming weights, and
   five rows showed that is the binding constraint.

*Measured, September 13, 2026, and it qualifies the above.* Balancing the
weight budget is not the same as preserving selectivity. Run unsupervised on
shallow_copy, leaky Hebb with the quash and the decay drives the network to a
single fixed response: of the sixteen possible inputs the top row answers all
sixteen identically, every neuron on, every epoch, at HEBB_RATE 0.01 and
0.001 alike. The same network before any learning answers 6 of the 16
distinguishably, so the composition made it strictly **less** informative
than its initialisation; ADALINE on the same task answers all 16 apart. The
row critic reads that collapse as 0.505, because an all-on row scores exactly
one half against a code that is half on — the trap §8 already records for
sustain_inputs, in a new place.

The gap is that no rule in the composition removes weight from a synapse
*for being uninformative*. Hebb adds on coincidence, the quash removes on
recurrence, the decay removes uniformly; none of them is selective against a
synapse that fires with everything. What such compositions conventionally
need is competition — a neuron's incoming weights normalised against each
other, or depression on post-without-pre, or a threshold that rises with a
neuron's own activity. The last of those is half-built already: §5.4 lowers
a bored neuron's threshold when it is silent and nothing raises it when it
is overactive. Making that symmetric would be the smallest addition
consistent with the head of §6, and is not yet decided.
3. *Name.* leaky_hebb, and it is a local rule with its own rate
   (HEBB_RATE), not a value of RULE. ELIGIBILITY = hebb inside the reinforce
   rule (§6.7) is something else, $\pm 1$ by whether $j$ fired.

### 6.13 The kinder teacher — decided, built

*Byron, September 14, 2026.* "Let's make a new teacher for this task. This
teacher is kinder: if two of the three neurons fired, that is a 1. If one of
the three fired, that is a 0. Three is a 1, and none is a 0. Output patterns
will be scored against the desired input as follows: for each bit, award 0.25
points if the output bit matches the desired input."

So the output row is read as the population code it is (§4.3), not as twelve
independent neurons. Each group of POPULATION neurons votes, the majority
carries the raw bit, and the score is the fraction of raw bits the vote gets
right — a quarter each, because there are four of them:

$$b_g = \Big[\textstyle\sum_{j \in g} \text{fired}_j > \tfrac{\text{POPULATION}}{2}\Big],
\qquad R = \frac{1}{k}\sum_{g} \big[\,b_g = d_g\,\big].$$

CRITIC = population selects it. **Where the kindness is:** under the row
critic every one of the twelve neurons must be right, so one wrong neuron in
a group of three costs a twelfth of the score; under this one the majority
still carries the bit and it costs nothing. A group can be wrong by a third
and still be read correctly. What it cannot do is disagree two to one.

*Claude's reading, built.* The read and the target are decoded by the same
vote, so a corrupted input (§4.3) is scored against the clean code and any
target arrangement works. The trivial floor is unchanged at 0.5: a silent
row votes every bit 0 and a saturated row votes every bit 1, and half the
raw bits are 1 on average, so both score two quarters — the same floor every
other critic has, which keeps the numbers comparable.

**And this fixes a wart rather than adding one.** The teacher's signal is now
$S = 2R - 1$ through whichever critic is running, where before it was
computed neuron by neuron regardless of the critic, so the rule of §6.9 could
pay a signal the critic did not agree with. For the row critic the two are
identically equal — $\frac{1}{n}\sum(1 - 2|\ell_j - d_j|)$ *is*
$2 \cdot \frac{1}{n}\sum(1 - |\ell_j - d_j|) - 1$ — so nothing measured
before changes, and TEACHER_CREDIT set by hand still takes the old per-neuron
path, which is the only place it means anything.

*Measured, September 14, 2026: six seeds, 100,000 epochs, shallow_copy at
five rows under rate drive with reward-modulated Hebb.* On its own trace the
kinder teacher reads **0.814** against the row critic's 0.794 — but a kinder
yardstick reads higher on the same behaviour, so that number settles nothing.
Freezing both networks and scoring each under both critics does:

| network | population critic | row critic |
|---|---|---|
| taught by the kinder | 0.757 | 0.713 |
| taught by the row | **0.762** | **0.755** |

Under the **population critic the two are a tie** — 0.005 apart, the
row-taught network ahead in three seeds of six. Under the row critic the
row-taught network wins in five of six. So the kindness buys nothing by its
own measure and costs under the stricter one.

Where it goes is visible. Counting the output groups that disagree with
themselves — one or two of three fired, rather than none or all — the
kind-taught networks split **30.3%** of their groups against the row-taught
networks' **12.0%**. Forgiving a wrong neuron in a group produces wrong
neurons in groups: the network learns exactly what it is graded on, and the
margin the tolerance creates is spent on sloppiness rather than banked as
robustness.

One thing it does buy, and it is the §6.9 signature running the right way for
once: the kind-taught networks carry **no weights at the rails** against 18,
at a lower mean $|w|$ (0.320 against 0.363). A teacher that stops asking once
the majority is safe stops pushing the weights, which is exactly what the
extremes of §6.9 do not do. CRITIC stays `row` for every problem; the kinder
teacher is available with `--critic population`.

### 6.14 What is reported

Spikes to date, the neurons fired this epoch, the dopamine value and its
expectation, releases and updates to date; the score of every epoch, its
mean to date, and an exponential moving average over about WINDOW epochs;
and a per-epoch trace file of epoch, time, dopamine, expected and score.
The window colours every neuron by the time of its last spike, hot (red)
at the end of the epoch cooling to blue over an epoch's length, and Space
pauses the free run at the end of an epoch to show its trace, a raster of
every spike. The network runs forever: these are read as health, not
convergence, and drift is normal.

### 6.15 A third engine — built, and every rule in it

*Byron, September 14, 2026, asking whether another language would make the
scheduling significantly faster, and then: "please build and test the rust
scheduler module."*

Profiling the array engine settled the first question. The arithmetic is
**1.3% of the runtime**: the sparse matrix-vector product that performs every
signal delivery costs 42 ms out of 3,271, and the rest is Python interpreter
overhead. The network is small — 60 neurons, 766 connections — so a numpy call
touches sixty doubles, about 50 ns of work behind 1–5 µs of call overhead.
Vectorising cannot pay at this scale; it was worth doing for the second
opinion it gave, not for the speed. A compiled loop that owns the state across
a run avoids both the interpreter and the per-call overhead, and 30–100× is
the expected order.

`rust/` holds that loop: the wave batching of §4.4, the neuron dynamics of §5,
the quash of §6.11 and leaky Hebb of §6.12, behind PyO3, with `fast.py` to
build it from a grid and `compare()` to check it lands on the same bits as the
object engine.

*As first written it refused exploration noise rather than approximating it,
because the draws would have to come from Python's stream in the same order
for the engines to agree — and the perturb eligibility of §6.7 with it. Byron,
September 14, 2026: "rules in authority.md must be implemented
cross-platform."* So it takes the stream rather than refusing it: the engine
runs Python's own MT19937, seeded by handing over `rng.getstate()`, draws the
same uniforms in the same order as the other two engines, pairs them by the
same Box-Muller (§6.1), and hands the state back so Python's stream carries on
from where Rust left it. The draws are equal, not approximately equal;
`tests/test_fast.py` compares them with `==`, and compares the three engines
wave by wave and weight by weight under both eligibilities of §6.7. The
per-wave draw lands in propagation.py's position exactly: after the floor,
before anything fires, and a neuron that has already fired keeps the draw it
decided under.

**It is built and run.** The toolchain arrived after this section was first
written; `fast.available()` is true, `compare()` returns nothing on every
configuration it is asked about, and the engine ran the CV sweep of §4.3.
Two tests need no extension — that the module refuses clearly when absent,
and that the edge order matches the object engine's push order — and that
second one is not cosmetic: signals due at one moment are summed in push
order, so an engine that flattens the topology differently sums a wave
differently and lands on different bits.

*The Teacher's threshold moves (Byron, September 14, 2026: "please always
default to the rust engine unless it is broken").* The firing-rate memory,
homeostasis and un-sticking of §1.3 are the Teacher's, not the network's,
and `fast.train` mirrors them once an epoch in Python — the same operations
in the same order as the object engine, the thresholds pushed to the loop —
at the constants the command line runs them at, so a Rust run is the run
`walnutbutter --seeds` would do. Every rust-sweep before that date ran with
them off, the CV sweep of §4.3 included. `docs/rust-sweep.py` builds goo
(§3.4) as well as the grid, and the Rust loop is now the default engine for
any sweep: the array engine is the fallback when Rust cannot run a
configuration, and a gap is closed in Rust rather than run around.

*What it still lacks:* the dopamine rule's in-loop weight updates
(§6.2–6.6), the teacher (§6.9) and ADALINE (§6.10) as updates rather than
as the eligibility they earn, LATE other than count, and the leaky trace on
the reinforce rule. Those pay at the read, in Python, and a run that needs
them takes the array engine — and says so.

The profile also named an algorithmic win that needs no new language: **the hop
delay is constant**, so every signal from a wave arrives at exactly
$t + \text{hop}$ and a ring buffer indexed by hop count would replace the
general priority queue, O(1) where the heap is O(log n). Only the Poisson
stimuli land at arbitrary times. That is available in Python today and may be
worth more than the rewrite.

## 7. Invariants the scaffolding guarantees — kept

- **Three engines, one network.** The object engine (neurons and a queue of
  waves), the array engine (numpy vectors, a scipy sparse matrix) and the
  Rust wave loop (§6.15) run the same network: same ids, same firing wave by
  wave, same exploration draws, same weights to $10^{-12}$, and
  `tests/test_arrays.py` and `tests/test_fast.py` run them side by side. A
  rule is implemented in every engine that runs it and must pass the same
  tests; an engine that lacks a rule says so and refuses, never
  approximates (Byron, September 14, 2026: "rules in authority.md must be
  implemented cross-platform").
- **A seed is the whole run.** Shortcuts, weights, permutation, inputs and
  exploration noise all come from the seed's stream, in both engines.
- **Checkpoints round-trip.** A checkpoint rebuilds the network from its
  seed and settings and reloads its weights, thresholds, clock, spike
  times, synapse stamps, signals in flight and dopamine, in either engine.
- **The network keeps living.** There is no training run and no evaluation
  run, only one run that keeps going; a rule may not assume an end.

## 8. Problems — open

A problem is what a network is asked to do and how it is watched doing it:
the layout, the inputs, and whether anything outside the network trains it.
`--problem` picks one; PROBLEM (§1) is the default.

- **reversal** (default today). The 8 × 10 hex grid; 4 random bits,
  complement-coded and permuted onto the bottom row; the top row is taught
  to show the bottom row reversed by a Teacher with a target and a critic
  (§6.2). The task of the pre-alpha, kept as the baseline.
- **copy** *(Byron, September 14, 2026: "Let's define this task very
  clearly: An input is complement-coded and presented on the list of input
  neurons. The desired output is exactly the input expressed across the list
  of output neurons. As things are currently set up, there should be eight
  input neurons and eight output neurons.")* Four raw bits, complement-coded
  to eight onto the input neurons, **unpermuted**; output place $i$ is
  taught to show coded bit $i$ — exactly the input, place for place — by the
  reinforce rule with the row critic. *Byron, the same day, on why there is
  no permutation: "Permuting patterns should no longer matter. All neurons
  are first-class citizens of the population." And on the target: "I don't
  think asking for the target reversed should matter either, frankly."* On
  goo neither does, by symmetry (§4.3); on the grid both do. On the
  8 × 10 grid the inputs are the bottom row and the outputs the top; on goo
  (§3.4) the first eight neurons and the last eight. It is reversal with the
  target set to copy and the permutation off, and it is the task the goo
  comparisons of §3.4 are posed on from here. Every sweep of that section
  before this definition ran on reversal, whose target is the input
  reversed; none of them measured this task.
- **sustain_inputs** (Byron, September 12, 2026). The same 16 inputs will
  be used across 8 neurons. However, this network is not trained
  externally: the neurons will utilize the new eligibility rule (§6).

  *The training epoch (Byron, September 12, 2026):* the sustain_inputs task
  receives an input on the same forced neurons, reverberates for 20
  milliseconds, and the same inputs (the inputs are the outputs) are read
  so we have a trace of the "score"; the previous Teacher metric computes
  the score even though it is not used for reinforcement. Time or epoch
  number, expected dopamine value, and score are recorded for each
  training epoch.

  *Claude's reading of the read:* the outputs are the input row, the target
  is the input pattern itself (copy), and a neuron is read as on if it
  fired within the last refractory period before the read at the end of
  the 20 ms, so the forced spike itself does not count and only a
  sustained neuron scores. Nothing outside the network moves a threshold
  (homeostasis and un-sticking off). The record goes to a CSV next to the
  checkpoint: epoch, time, dopamine, expected, score.

  *Decided (Byron, September 12, 2026), replacing the row critic:* what we
  are interested in is: did the neurons that were forced to fire sustain?
  Only the four neurons that were forced to fire are scored. The score is
  the **sustained** critic: of the input neurons the pattern forced, the
  fraction read as on at the end of the epoch. The unforced four are not
  scored at all. (Under the row critic a uniform row, all on or all off,
  scored 0.5 whatever was forced, which is why every sweep to this point
  read 0.500.)

  *Decided (Byron, September 12, 2026), simplifying the task:* rather than
  complement-coding the inputs, the sixteen inputs are presented as 4 bits
  each, laid down as they are on a 4-across, 10-row grid (no complement
  coding: 0000 forces nothing, 1111 forces all four). The score is the
  original row critic over all four input neurons, forced or not: the
  forced ones should be on at the read and the others off. The `sustained`
  critic stays available (`--critic sustained`).

  *Decided (Byron, September 12, 2026), replacing the read window:* the
  read asks whether the neuron fired again at all, rather than whether it
  fired in the last 5 ms. A neuron is on at the read if it spiked at any
  moment strictly after the input's moment $t_e$: for a forced neuron, a
  refire; for an unforced one, any spike in the epoch. (The 5 ms window read
  one phase in three of a fully sustaining neuron as off, on a 2 ms hop
  grid with a 6 ms cadence: the 2/3 ceiling of the low grid.) The window
  read stays available to a problem as `read = "window"`.

- **improved_sustain** (Byron, September 12, 2026). A different topology
  for this problem: the hex grid consists of 7 rows by 10 across. The input
  is presented in the MIDDLE, row 4, places 4 through 7 counted from 1
  (row 3, places 3 to 6 counted from 0: the middle four of the middle row).
  The reach is 3: every neuron is wired to every cell within three hex
  steps (§3.1). No permutation, the bits land where they are. Everything
  else as sustain_inputs: raw coding, the same four neurons
  read back as spiked again, the row critic, learning by dopamine, nothing
  outside the network training it. A problem may name any input zone, a
  list of (place, row) cells (`Network.set_input_cells`), in place of the
  bottom row; a neuron may sit in several zones at once.

- **population_copy** (Byron, September 13, 2026), the problem the quashing
  rule (§6.11) is tested on. The four raw bits are population-coded, three
  neurons a bit, onto a twelve-wide bottom row of a ten-row grid: 1001
  becomes 111000000111, 0001 becomes 000000000111. The outputs are read
  across the top, and the target is the copied population code. The teacher
  scores the top row in $[-1, 1]$, six of twelve right being zero, which is
  §6.9's score with the credit normalised to the zone. No permutation, a
  20 ms epoch, reach 2, and each output read as fired this epoch. Cycles
  are quashed; the teacher also pays, and `--lr 0` isolates the quash.

- **doubled_copy** (Byron, September 14, 2026). The four raw bits are
  **doubled** (POPULATION 2, so 1001 becomes 11000011), then
  **complement-coded** (11000011 becomes 1100001100111100), then spread over a
  sixteen-wide input row by the **consistent random permutation** of §4.3. The
  top row of an eight-row grid should show the same code: the target is copy,
  the row critic scores it, and cycles are quashed as elsewhere.

  *What the coding takes away.* Complement coding means **exactly half the row
  fires whatever the bits** — verified over all sixteen inputs — so total
  activity is constant and carries nothing. The permutation means **adjacency
  carries nothing** either: a bit and its double, and a bit and its negation,
  land wherever the permutation put them rather than side by side. So neither
  of the two shortcuts the earlier population problems left open is available.
  In population_copy a bit's three neurons were contiguous and a row's total
  firing tracked the number of ones; here both are gone by construction, and
  what is left to learn is the mapping itself.

  A first run over 3,000 epochs on 128 neurons reaches 0.518.

- **reaching_copy** (Byron, September 14, 2026). doubled_copy's inputs
  exactly — the four raw bits doubled, complement-coded, and spread by the
  consistent random permutation over a sixteen-wide input row — copied to the
  top of a **five-row** grid wired to **reach 5**. Target copy, row critic,
  quash on, everything else as doubled_copy.

  *What the two changes do.* They pull in opposite directions on the same
  quantity, the number of hops between a bit and its answer. Five rows instead
  of eight shortens the path; reach 5 on a five-row grid abolishes it. The
  bottom row and the top row are four hex steps apart, inside the reach, so
  **every input neuron synapses directly onto the output row**: 100 of the 256
  input-output pairs are one hop, every horizontal offset from -3 to +3. The
  mesh is no longer a depth through which a signal must be relayed; it is a
  near-complete bipartite map with a body of interneurons attached.

  The cost is density. 80 neurons carry 3,122 local connections, a mean
  out-degree of 39 — half the network — against 13.1 at reach 2 on the same
  grid and doubled_copy's 1,856 on 128 neurons. At the default OMEGA of 0.2 the
  small-world shortcuts bring the built network to 3,903 connections against
  doubled_copy's 2,320. So this is half again as many synapses on two thirds as
  many neurons, and the quash of §6.11 has correspondingly more cycles to find.

  *What the permutation does and does not do (Byron, September 14, 2026,
  correcting this entry).* **The task is a copy, not an unscrambling.** What is
  permuted is the input: the coded bits are spread over the input row by $\pi$,
  and the pattern that lands there is the pattern the top row must show. Output
  place $i$ is scored against input place $i$. The permutation destroys the
  adjacency the coding would otherwise leave — a bit and its double, a bit and
  its negation, no longer neighbours — and that is all it is for; the network is
  never asked to invert it.

  So **every one of the sixteen routes is one hop**: place $i$ at the bottom is
  four hex steps from place $i$ at the top, inside the reach, directly wired.
  Which is the whole point of reach 5, and what separates this problem from
  doubled_copy, where the same copy has eight rows to cross and no direct
  synapse anywhere.

  (The first draft of this entry said the network had to undo the permutation
  and counted 6.25 of the sixteen routes as one hop. That was wrong about the
  specification, not about the code: `set_input` has always stored the permuted
  pattern as the target, so the code was already doing the copy. The arithmetic
  about $\pi^{-1}$ described a problem nobody had posed.)

  *A first run, September 14, 2026: 20,000 epochs, six seeds, the Rust loop, one
  shared input stream per seed (§4.5), everything else at the defaults.* A
  reach-2 arm at the same five rows is run alongside, so that depth and reach
  come apart:

  | arm | score | sd |
  |---|---|---|
  | shallow_copy (12 wide, 2 rows, no complement, no permutation) | 0.880 | 0.038 |
  | **reaching_copy** (5 rows, reach 5) | **0.667** | 0.036 |
  | the same at reach 2 | 0.618 | 0.017 |
  | doubled_copy (8 rows, reach 2) | 0.596 | 0.034 |

  Paired on the seed, reaching_copy beats doubled_copy by **+0.070 in 6 of 6**
  and its own reach-2 control by **+0.049 in 5 of 6**. So the gain splits about
  one part depth to two parts reach: going from eight rows to five is worth
  +0.022, and wiring those five rows through is worth twice as much again. The
  direct synapse is doing the work, which is what the problem was posed to ask.

  It still leaves 0.21 to shallow_copy, so **the coding costs more than the
  topology can win back**. Nothing here is converged at 20,000 epochs;
  doubled_copy's own million-epoch value at these settings is 0.599, which this
  run reaches at 20,000 with the shared stream.

  (The first version of this run, before the stream, put reaching_copy at 0.633
  and doubled_copy at 0.575 — the same ordering, the same conclusion, every
  level shifted by about the size of one seed's spread. That is the measure of
  what the input lottery was worth: about as much as the effects being
  measured.)

- **shallow_not** (Byron, September 14, 2026): "It's the same as the old
  problem, only it's not." shallow_copy in every respect — the same 12-wide
  population-coded input on the same grid, read the same way, scored the same
  way — except that the target is the **complement** of the code.

  *Why it is not a small variation.* Copy can be had by excitation alone:
  something fires, something downstream fires. The complement asks an output
  neuron to fire **because nothing told it to**. Silence propagates no signal,
  sends nothing along any synapse, and arrives nowhere, so no weight on any
  incoming connection can drive an output whose whole input group is quiet —
  there is no signal there to weight. The half of the task that says "be off
  where the input was on" is easy; the half that says "be on where the input
  was silent" has no causal path to it.

  The only mechanism in the system that turns silence into a spike is the
  bored-neuron threshold of §5.4, and it is calibrated at BORED_AFTER = 200 ms
  against a 35 ms epoch: six epochs of silence before a neuron goes on its
  own, by which time the input has changed six times. So the expectation,
  stated in advance, is a score near the 0.5 floor for a structural reason
  rather than a tuning one. A first run at five rows reaches 0.537 over 3,000
  epochs.

  What would change it is not a learning rate. It is a standing background of
  activity that inhibition can sculpt — a tonic drive, or a boredom clock near
  the epoch's own length — which is how cortex computes absence, and which no
  problem here has ever asked for.

  *Swept across depth, September 14, 2026: rows 2 to 8, six seeds, 20,000
  epochs, with shallow_copy run identically as the control.* The complement
  never leaves the floor at any depth, and depth makes it worse:

  | rows | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|
  | shallow_not | 0.559 | 0.566 | 0.537 | 0.537 | 0.508 | 0.512 | 0.506 |
  | shallow_copy | 0.896 | **0.904** | 0.850 | 0.838 | 0.787 | 0.782 | 0.707 |

  Splitting the complement's errors shows the predicted mechanism exactly. On
  the half with no causal path — fire where the input group was silent —
  accuracy falls from 48.9% at two rows to 31.3% at eight, while the other
  half rises to 68%, and the output row's firing falls from 44.6% to 31.5%
  against a target wanting 49% on. **The network is not failing to learn the
  complement; it is learning to shut up**, which is the closest thing to an
  answer that excitation can express, and deeper meshes are quieter.

  *And the control earned its keep twice over.* Copy also declines with depth
  under these defaults, 0.904 at three rows to 0.707 at eight, which reverses
  the September 13 finding that five rows beat two. Depth used to help and now
  hurts; what changed is forced drive giving way to rate (§4.3). So the
  narrowing gap between the two rows of the table is copy falling, not the
  complement rising — and the two-hop solve of §6.7 looks increasingly like a
  property of the synchronous lattice rather than of the task.

  *The boredom clock is not the lever (Byron, September 14, 2026, asking for
  it against copy first, which was the right order).* BORED_AFTER swept at 10,
  15, 20, 25 and 30 ms — every value below the 35 ms epoch, so a silent neuron
  goes on its own within one — **destroys copy at every value**, scoring 0.542
  to 0.557 against 0.838 at the default 200 ms and losing in 6 seeds of 6
  throughout. The reason is that it floods:

  | BORED_AFTER | spikes/epoch | row fires | on target-1 | on target-0 | mean $\|w\|$ |
  |---|---|---|---|---|---|
  | 10 ms | 131 | 96.5% | 100.0% | 7.3% | 0.72 |
  | 20 ms | 91 | 94.6% | 100.0% | 11.2% | 0.98 |
  | 30 ms | 75 | 88.3% | 92.8% | 16.5% | 0.95 |
  | 200 ms | 55 | 40.0% | 71.9% | 93.8% | 0.37 |

  Which is the complement's failure mode in a mirror. shallow_not fails by
  going silent and gets the target-0 half; a fast boredom clock fails by never
  going silent and gets the target-1 half. Neither carries information,
  because the clock is an **untargeted** background: it fires every neuron
  alike, and nothing about it depends on the input. A background that
  inhibition could sculpt has to be one the active input groups can suppress.
  Whether that is reachable is untested — the same sweep against shallow_not
  would say, and is the obvious next measurement rather than another rule.

- **population_denoise** (Byron, September 13, 2026), the same network
  inspected somewhere else. *Byron:* "I want to know whether this is doing
  something useful, and it's not at the output zone that I want to look. I
  guess this is a new subtask with the same teacher that looks at the output
  zone. I want to use that teacher but inspect the INPUT zone under the
  condition that the input signal has flipped bits with probability 1/12."

  Everything is population_copy's — the twelve-wide bottom row of a ten-row
  grid, population coding, no permutation, 20 ms epochs, reach 2, the §6.9
  teacher with its credit normalised to 1/12, cycles quashed — except where the
  read is taken and what reaches the row. Each of the twelve coded bits is
  flipped with probability FLIP = 1/12 (§4.3), so about one neuron an epoch is
  wrong; the **input zone** is read back, a neuron on if it spiked again after
  the input's moment; and the score is the row critic against the **clean**
  code, not the corrupted one.

  *Claude's reading, built:* the target is the clean pattern, because that is
  the only reading under which the corruption carries information — with the
  corrupted pattern as the target this is sustain_inputs with extra variety.
  Three baselines say what the number means. A silent network reads all-off and
  scores 0 on average, the clean code carrying six ones and six zeros. A network
  that perfectly sustains whatever it was forced with scores
  $(11 - 1)/12 = 0.833$, carrying the flip through faithfully. Only a network
  that repairs the flipped patch reaches 1. So the band above 0.833 is
  correction, and the band below it is the cost of the quash, which works
  against sustaining by construction; `--quash 0` separates the two.

- **shallow_copy** (Byron, September 13, 2026): "we are going to shrink the
  network to see if it can learn at all with a new rule. The same, with
  output at the top and input at the bottom, but we'll have just two rows."
  So: population_copy's task on the smallest network that still has an input
  row and an output row. Twelve across, **two rows**, the four raw bits
  population-coded onto the bottom, the top read as fired this epoch, the
  same teacher, the same row critic, no permutation, a 20 ms epoch, reach 2.
  Twenty-four neurons and 215 connections, against 120 and 2,195.

  The point is that the task is now one hop wide: every input neuron is
  directly wired to the output neurons above it, so a rule that can learn
  anything should learn this, and a rule that cannot learn this cannot be
  rescued by depth. It is the floor the rules are measured against, not a
  problem worth solving for its own sake.

  *Claude's reading, built:* reach 2 on two rows wires the rows to each other
  **and to themselves**, and 56 of the 215 connections run top to bottom, so
  the output row feeds back into the input row. That feedback is the only
  cycle generator in the network, which makes this also the cleanest place to
  watch the quash (§6.11) work. Cycles are quashed as in population_copy, and
  `--quash 0` switches it off. The rule is the problem's default teacher;
  `--rule` picks any of the four.
