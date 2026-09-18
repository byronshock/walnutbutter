# walnutbutter

A living network of simple neurons.

The system is specified in `AUTHORITY.md`; the code follows it. Each
neuron is an integrate-and-fire unit with no leak: weighted input
accumulates until it crosses a threshold, the neuron fires and resets, and
for an absolute refractory period it ignores everything. Its signal travels
one-way along weighted connections, one hop per connection, on a clock in
nominal milliseconds; a wave is everything that happens at one moment. A
neuron may fire as often as its refractory period allows, so a tight loop
can carry a neuron's own spike back to refire it, and activity sustains
itself. The network's input is a complement-coded, permuted bit pattern
forced onto its input zone; its output is the output zone. Learning is
**dopamine**: a neuron that fires again after its refractory period
releases dopamine, most when it refires the instant it may, into one
global pool; at that same refire its incoming synapses that carried a
signal since its previous spike move together in proportion to the pool
minus the network's expectation of it. There is no training run and no
evaluation run, only one run that keeps going: a free-running system that
learns, checkpoints itself, and reports as it goes.

**Goo** is the only container (AUTHORITY.md §4.1). It has no positions at
all, so there is no distance to measure and nothing to be near; what decides
the architecture is the scaled wiring rule alone. Sixty neurons at its own
threshold, 0.2, wired by a rule about zones, every neuron un-sticking itself
(§4.4, §4.11). The hex grid, the hexagonal columns, the lattice and the
spread left the specification on September 17, 2026; nothing is deleted --
they stay in `RECORD.md` §2-§3 and in the code at the tag
`lab-notebook-2026-09-17`, which is what archived means here.

## Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra includes pytest, pygame, numpy and scipy. To install only
what the visualizer needs, use `pip install -e ".[viz]"`; only what the
array engine needs, `pip install -e ".[arrays]"`. The object engine has no
dependencies at all.

## Usage

```bash
walnutbutter                 # open the window, free-run, learn, report accuracy; close it to stop
walnutbutter --headless --epochs 20000 -q   # the same without a window, for a fixed number of epochs
walnutbutter --no-learn      # just watch the untrained network
walnutbutter --problem copy             # 4 bits complement-coded onto 8 input neurons, and the 8 outputs taught to show exactly that, place for place, unpermuted; an output is on if its spike count this epoch, as a rate, exceeds --teacher-threshold (AUTHORITY.md §8, §4.3; the goo task)
walnutbutter --problem sustain_inputs   # the 16 four-bit inputs as they are on 4 neurons; score = which input neurons spiked again, against the pattern; learns by dopamine (AUTHORITY.md §8)
walnutbutter --problem population_copy  # 4 bits population-coded over 12 neurons (1001 -> 111000000111), copied to the output zone, cycles quashed
walnutbutter --problem shallow_copy  # the same task on 24 neurons, one hop from input to output, the floor every rule is measured against
walnutbutter --problem population_denoise  # the same network, but one bit in twelve flips on the way in and the INPUT zone is read back against the clean code
walnutbutter --trace runs/sustain.csv   # one line per epoch: epoch, time, dopamine, expected, score (default: next to the checkpoint)
walnutbutter --rule reinforce           # the pre-alpha's global-reward rule instead, run by the Teacher
walnutbutter --refractory-hops 3.7 --release-theta 2 --order update-first   # the clock and dopamine knobs (see --help)
walnutbutter --goo 200 --across 24       # a bigger goo than the default sixty: 12 input bits, coded to 24
walnutbutter --ecc                        # 4 data bits -> Hamming (7, 4) -> 14 across
walnutbutter --ecc parity64               # the (6, 4) detect-only code on 12 across instead
walnutbutter --input 1011                 # choose the 4 input bits
walnutbutter --no-permute                 # coded bits in order on the input zone
walnutbutter --seed 42       # repeat a particular random wiring
walnutbutter --weight 1      # a fixed weight on every connection instead
walnutbutter --positive-weights --threshold 2   # no inhibition: weights kept in [epsilon, 1]
python -m walnutbutter       # same thing without the installed command
```

**Input.** The network's input is its input zone. Each epoch draws 4 random
bits (half the count across), complement-codes them by appending their negations,
and scrambles the 8 coded bits with a random permutation of the places in the zone that
is drawn once per run and never changes. The input-zone neurons whose bit is
1 are forced to fire at the input's time (refractory period permitting), so
up to half of the zone fires every time.
The raw bits, the coded bits and the permuted row are printed, and the
permutation is printed once at the start. `--input 1011` supplies specific
bits for the first epoch, `--no-permute` lays the coded bits down in order,
and `--seed` reproduces the permutation and the whole sequence of random
inputs. Columns must be even. From Python, `run_epoch(grid)` resets the
mesh and presents the next input.

## Comparing seeds

Outcomes vary a lot between seeds: with identical settings, some seeds reach
the high 80s within a million epochs while others sit in the low 60s. So
the best use of a many-core machine is to run several seeds at once and
keep the best:

```bash
walnutbutter --seeds 15 --epochs 1000000 --seed 1
```

runs seeds 1 to 15 in parallel, headless, one process per core, prints a table
sorted best first (accuracy over each run's last tenth, and to date),
and checkpoints every run to `runs/` so the winner can be loaded with
`--load-weights`. Without `--seed` the base seed is random and printed.

## From Python

```python
from walnutbutter import main
from walnutbutter.monitor import run_epoch

net = main(count=60, across=8)  # builds the goo and runs the first epoch on its input zone
run_epoch(net)                  # reset every neuron and present a new random input
print(len(net.fired_neurons()))
net.reset()                     # allow every neuron to fire again
```

## Goo

The plane taken away (AUTHORITY.md §4.1). A container with positions has to
say what "near" means before it can say what connects; goo has no positions, so
there is no distance to measure and nothing to be near, and what is left is
the scaled rule (AUTHORITY.md §3.4, Byron, September 16, 2026): no neuron
projects onto itself, no input neuron onto another, and every other
ordered pair projects, one way, at the probability that gives its target
N times `--scaling-factor` synapses in expectation over the sources it may
hear (GOO_SCALING_FACTOR, 0.05: 32 synapses a neuron on the mnist goo of
644, 3 on goo 60). The input zone is kept from talking to itself and the
outputs are kept apart (Byron, the same night: "With hidden=0 we have no
cycles ... a two-layer feedforward network"): an input hears the hidden
neurons, an output hears the inputs directly and the hidden neurons and
projects onto the hidden alone, so no interior is needed, only that the
zones not overlap, and with none the goo is inputs -> outputs and nothing
else. A neuron that hears nothing is left at the container's threshold
(AUTHORITY.md §5.2). `--goo` makes 60 neurons
(GOO_COUNT; it was the grid's 80 while the two were compared): about 180
projections at 0.05, the seed's choice of them. `--wiring` reaches the three
earlier rules at `--projection`: `zones-equal`, the zone rule of September
14 with the equal fan-in of the 15th (the zones never project onto each
other and an interior-to-zone projection is scaled up so every neuron hears
the same number; the rule until the 16th, and what every result below ran
under; at 0.2 about 650 projections, at 1 the fully connected goo of 3,300),
`zones` (without the equal fan-in) and `uniform` (one probability over every
ordered pair); `scaled-open` is the night's first scaled rule, the outputs
open to every zone.

With no positions there is nowhere to put the input but the index, so the
zones go by index (AUTHORITY.md §4.3): the first `--across` neurons are the
input zone and the last `--across` the output. They are addressed by place
(`get_neuron_at(place, 1)` in, `get_neuron_at(place, 0)` out), so learning,
the teacher, the checkpoints and the array engine need to know nothing about
it. The zones may not overlap: a goo that would overlap them is refused.

```bash
walnutbutter --epochs 1000                  # the default goo, sixty neurons
walnutbutter --goo 200 --engine arrays      # a bigger goo, on the array engine
walnutbutter --goo 200 --seeds 8 --epochs 50000   # a batch
```

The seed decides the topology, pair by pair in (i, j) order, the projection
draw and then the weight, so a goo needs its seed to be rebuilt, and a
checkpoint records the wiring it was built under (`wiring`, `scaling_factor`,
`projection`) and restores under it. It has no geometry, so every run is
headless.

**Its potential axis scales with fan-in**, and nothing else's does
(AUTHORITY.md §5.2), and since September 14, 2026 it scales **its own
constants**: GOO_THRESHOLD 0.2 and GOO_MINIMUM_POTENTIAL -0.8, the grid's
ratio kept. Quoted at an interior hex cell's 18 incoming synapses; a goo
neuron of 60 has 59, so it starts at a threshold of 0.2 x 59/18 = 0.66 and a
floor of -2.62 -- set from a sweep at 0.01 steps with every neuron
un-sticking (below), where 0.15-0.40 is a plateau and 0.20 the level where
every seed learned. (It was 1, theta 3.28, chosen to sit past the
saturation edge; once the direct projection was cut that was a dead
interior. At the grid's 0.25 and 80 neurons it started at 1.097 and -4.389,
where the sweeps began.)

Both move, because they are two points on one axis and it is the axis being
rescaled. Unscaled, goo's output zone is on 100% of the time whatever the
input and the read carries nothing at all. Scaling the threshold alone
barely helps -- still 99% on -- because the recurrence sustains the activity,
and raising the threshold against an unmoved floor caps inhibition while
excitation piles up. Moving both lands it: the output zone varies, and goo
then produces more distinct output words than the grid does.

That is activity, not learning. Swept twice at 250,000 epochs and ten seeds
against the same goo run flat and against the grid (AUTHORITY.md §3.4,
`docs/goo-250k.md`, `docs/goo-250k-hebb.md`): **all three at chance both
times** -- under the perturb eligibility grid 0.520, flat goo 0.511, scaled
goo 0.507; under wrong_hebb scaled goo 0.513, flat goo 0.506, grid 0.502. The grid
does not learn reversal at ten rows under either rule; where wrong_hebb learns
(§6.7's 0.98, §4.3's 0.64) the task is one hop wide. The rescaling fixed the
read carrying nothing, not the rule learning nothing, and the comparison goo
was built for wants a one-hop task the grid can do: reaching_copy, which is
the grid with its depth taken away, against goo, which is the grid with its
depth and its locality taken away. `--no-scale-with-fan-in` runs goo flat,
and `--scale-with-fan-in` offers the rule to any other container.

Then, on copy under wrong_hebb, goo's count against THRESHOLD with the floor
following at the grid's ratio (`docs/goo-count-threshold.md`, 300 arms):
**at the shipped threshold fewer neurons is much better** -- goo 24 scores
0.609, goo 80 0.524, t 6.5 -- **but the best cell of all is goo 80 at
THRESHOLD 0.5**: 0.634, nine seeds of ten learning, not one neuron stuck on.
The count was never the problem; the threshold for the fan-in was, and the
linear scaling of §5.2 under-corrects by about 2x at eighty neurons.

Past that edge (`docs/goo-count-threshold-high.md`, goo 64 to 120 against
THRESHOLD 0.5 to 2) there is **no optimum at all**: twenty cells at 0.56 to
0.64 whose spread is the seed noise, over a sevenfold range of theta and a
twofold range of count. The threshold's only job is to get a goo out of
saturation. At goo 120 most of the goo can be silent -- 55 to 77 of 120
neurons stuck off -- and the copy survives on the eight inputs projecting
straight onto the eight outputs. Everything that escapes saturation lands at
0.60-0.64, where the wrong_hebb eligibility also lands on reaching_copy: the
ceiling is the rule's, not the network's.

Under the count read (§4.3) the working network -- goo 60 at its own
constants, copy -- scores **0.556** with the wrong_hebb eligibility and 0.514
(chance) with perturb over ten seeds and 100,000 epochs
(`docs/goo60-count.md`, `docs/goo60-count-perturb.md`), at 7,000 epochs a
second: the read that stops counting a stray spike as a one moves the
ceiling down, not up.

Then the bug: a fully connected goo wires input i straight onto output i,
and that one synapse was what the rule had learned. Cut the input-to-output
projection and every seed sits at 0.500 -- a dead interior at theta 3.28,
since nothing walked a silent interior neuron's threshold down. So **un-sticking now
reaches every neuron**, not the output row only, and with it the interior
lives at every threshold from 0.10 to 0.50 (`docs/goo60-nodirect-threshold.md`,
410 arms): a plateau at ~0.56 from 0.15 to 0.40, 0.20 the level where every
seed learned a two-hop copy, and the old default of 1 off the plateau at
0.502. GOO_THRESHOLD is 0.2 since.

## Learning

**Dopamine** (AUTHORITY.md §6, the default, `--rule dopamine`). A neuron
whose previous spike was at `t_prev` and which fires again at `t`, a delay
`d = t - t_prev - refractory` past the end of its refractory period,
releases the gamma density of that delay, `d^(alpha-1) exp(-d/theta) /
(Gamma(alpha) theta^alpha)`, averaged over the hop that follows it so that
it is finite for every alpha (`--release-alpha` 2, `--release-theta` 1 ms
by default: about 0.3 for an instant refire, a peak a little later, then a
decay). Releases pool into one
global value that decays with `--dopamine-tau` (default 20 ms) and is read
without being depleted. The network's expectation of it is an exponential
moving average of the value over `--expectation-tau` (default 10 minutes),
starting at 0. At the refire,
every incoming synapse that carried a signal the neuron integrated since
its previous spike moves together by `lr * (dopamine - expected) *
release`, clipped to the weight range: REINFORCE's shape, presynaptic
activity x postsynaptic eligibility x global signal, with the eligibility
the neuron's own release and nothing traced back through the network.
`--order` says whether a wave's releases join the pool before its updates
read it (`release-first`, default) or after. An input neuron whose bit is
0 this epoch, one that should not fire, has the sign of its update
reversed when it refires, so above-expected dopamine punishes it, and
`--punish-gain` (default 2) times as hard as a reward (`--no-punish`
switches that off). Every weight also moves toward zero by
`--weight-decay` per epoch (default 1e-4): synapses that forget on their
own, so that sustained activity has to be earned (AUTHORITY.md §6.8). Every neuron learns this way,
forced inputs included: input, hidden and output neurons differ only in
where external connections land. At every input each neuron's potential is
nudged by exploration noise (`--sigma`, default 0.1; 0 switches it off).
Both engines do the same arithmetic and agree to the last bit.

**The external teacher** (`--rule teacher`, the default for the sustain
problems, AUTHORITY.md §6.10). The teacher scores the read: +0.25 for each
input neuron that is on when its bit is 1 or off when its bit is 0, −0.25
for each one that is not, so four inputs give a score of −1, −0.5, 0, 0.5
or 1. During the epoch a refire adds its release to the `eligibility` of
each gated incoming synapse instead of moving it; at the read every synapse
moves by `lr * score * eligibility` and the trace is cleared. The dopamine
pool still runs and is reported but decides nothing, and the bit-0
punishment is off, since the score already accounts for those neurons.

**Quashing cycles** (`--quash RATE`, AUTHORITY.md §6.11). A refire is a
cycle, and a cycle is treated as something to remove: every incoming
synapse that carried a signal the neuron integrated since its previous
spike is multiplied by `1 - quash * exp(-quash_k * delay)`, where the delay
is the time since that spike. It pulls a weight toward zero from either
side, touches only the synapses that carried the cycle, and needs no
external signal, so it runs alongside whichever rule is chosen. A problem
says whether it quashes; `--quash 0` switches it off.

**ADALINE** (`--rule adaline`, AUTHORITY.md §6.10). Widrow-Hoff with an
eligibility trace: each scored neuron gets its own error, +1 if it should
have been on and was not, −1 if it was on and should not have been, 0 if it
was read correctly, and every synapse into it moves by `lr * error * what
that synapse delivered this epoch`. A correct epoch moves nothing, and only
the scored neurons' incoming weights learn, so the mesh behind them is an
untrained reservoir. Unlike the other rules the signal is not global.

**The reinforce rule** (`--rule reinforce`) is the pre-alpha's global
reinforcement, factored out and kept for comparison. On a trained problem
(`--problem reversal`) the `Teacher` tells the network each epoch what the
output zone should have shown, by default a **reversed** copy of the input-zone
input (`--target reversed`; `copy`, `all-off` and `all-on` also exist), and
scores the fraction of output neurons that match; under the dopamine rule
it only scores and reports. Under the reinforce rule a single scalar
reward, the epoch's accuracy, is compared with a running average to give an
*advantage*, and every connection that carried a signal into a neuron that
was not a forced input moves by `lr * advantage * eligibility`, where the
eligibility is the target neuron's exploration noise (node-perturbation
REINFORCE); with `--eligibility hebb`, what the synapse delivered times the
target's spike count minus the target's own running expectation of it (the
centred Hebbian rule, AUTHORITY.md §6.7); with `--eligibility wrong_hebb`, +1
if the target fired and -1 if not (the uncentred rule hebb replaced on
September 16, 2026); or with `--eligibility hazard` the score of the
escape-noise decision on each synapse's trace. Forced inputs are never
adjusted and weights stay within [-1, 1].

The **mnist** problem (AUTHORITY.md §8) reads the handwritten digits from
`mnist/` (the training split's two IDX files, not in git; `mnist/README.md`
says where from and `walnutbutter.mnist.fetch()` gets them; the test split is
not fetched, by decision), each image averaged to 14 × 14 and
thresholded at half; the input zone is three clock neurons always driven,
the 196 on-off pixels and their 196 complements; ten classes on 30 outputs
read by count, and the class critic: the label's three outputs out-spike
every other class's three, or nothing. `walnutbutter --problem mnist` builds
a goo of 644 for it, 199 hidden neurons between the zones; `--hidden-neurons H` sets the hidden count (0 is allowed under the scaled rule: the outputs hear the inputs directly) and `--goo N` the total.

With `--delta D` (ESCAPE_DELTA, AUTHORITY.md §5.2) the firing decision
itself is the draw: a neuron that is not refractory fires at a wave with
probability `1 - exp(-m)`, `m = (dt / hop) * exp(s / delta_j)`, where `s`
is its margin against the threshold it faces and `delta_j` is D times its
starting threshold -- one expected spike per hop at threshold, e times more
per `delta_j` above it, so a neuron nobody talks to fires on its own at a
rate its margin sets. That is Williams's Bernoulli unit with the noise in
the threshold, and `--eligibility hazard` pays it with Williams's own
eligibility: the score of each decision, summed over the epoch on each
synapse's trace of what it still had in the potential (§6.7). It needs a
positive D and runs in all three engines. Since September 15, 2026 the
command line's default is D = 0.455 (ESCAPE_DELTA, Byron's word from the
Δ × LR grid of AUTHORITY.md §3.4) with the hazard eligibility; `--delta 0`
is the deterministic neuron of every earlier result, and a network built in
the library stays deterministic until `network.set_delta(D)`.
Under the schedule the mesh reverberates on its own, so no performance is
claimed for this rule any more.

A signal that arrives after its target has already fired is dropped on
delivery and changes nothing in the epoch, yet by default its connection is
still reinforced: pre fired, post fired, and the global reward says whether
the coincidence was good. That is a local Hebbian term riding on the
perturbation estimator, strictly a bias with respect to the reward
gradient, but it is the biological shape of the rule (local eligibility,
global signal) and it learns faster: on the 8x10 reversed task at 100k
epochs, every seed tried did better with it (last tenth 0.76-0.91 against
0.61-0.75). `--late` chooses what a late signal earns: `count` (the
default), `ignore` (nothing: only the signals that landed, the
node-perturbation estimator proper) or `depress` (the opposite update, the
shape of spike-timing-dependent plasticity, where a presynaptic spike after
the postsynaptic one weakens the synapse). `Teacher` wraps all this; use
`teacher.epoch()` instead of `run_epoch(grid)` so the exploration noise is
injected.

**The trace.** Every scored run appends one line per epoch to a CSV next
to its checkpoint (`--trace FILE` to choose it, `--no-trace` to skip it):
epoch, time in ms, the dopamine value, the expectation, and the score.
In the window every neuron is coloured by the time of its last spike, red
at the end of the epoch cooling to blue over one epoch's length, and Space
pauses the free run at the end of an epoch to show its trace, a raster of
every spike, time across and neuron down; Space resumes.

**Output.** Runs are silent apart from the progress reports and the final
summary: printing is far slower than learning, and a headless run of millions
of epochs would otherwise spend its time writing to the terminal. `-v` /
`--verbose` prints a line for every epoch's input and every neuron that
fires, for short inspections. From Python the same switch is
`Neuron.verbose`, off by default; `run_epoch(grid)` prints its one input
line unless told `verbose=False`.

**Saving what it learned.** Every run writes a JSON checkpoint at every
progress report and on exit, by default to `runs/<date>-<time>-seed<seed>.json`
(the path is printed at the start; `runs/` is ignored by git). `--save-weights FILE`
chooses the file, `--no-save` skips it. A checkpoint holds: every weight by connection ID, plus
the count, the wiring rule, threshold, seed, input permutation, epoch count and
the learning statistics. `--load-weights FILE` rebuilds that exact network from
the stored seed and settings, restores the weights, and carries on counting
epochs and accuracy to date from where the file left off. The sequence of
random inputs starts afresh, so a resumed run is not epoch-for-epoch
identical to an uninterrupted one, but the learned weights are the same.
From Python: `persistence.checkpoint(grid, path, teacher)` and
`grid, data = persistence.restore(path)`.

**Stuck neurons and homeostasis.** A neuron whose input sits far from its
threshold is never flipped by the exploration noise, so it gets no learning
signal and stays "stuck" always on or always off; on a long run most hidden
neurons end up that way. To counter it, every
neuron tracks its own firing rate and slowly moves its threshold toward a
target rate (`--homeostasis`, default 1e-6 per epoch,
`--target-rate`, default 0.5); firing too often raises the threshold, too
rarely lowers it. Neurons forced this epoch are left out for that epoch, as
reinforcement leaves them out; an unforced input neuron is treated like any
other. `--homeostasis 0` switches it off. Per-neuron thresholds are saved in
checkpoints.

**Un-sticking.** A saturated neuron, one that fires on every input or on
none, gets no learning signal at all, because nothing changes what it does.
`--unstick RATE` (default 0.001) moves the threshold of any neuron that is
stuck, firing more than 99% or less than 1% of the time, toward
`--unstick-target` (default 0.5), and stops the moment it is no longer
stuck; a neuron forced this epoch is left alone. It was the output row only
until September 14, 2026, when the interior of a goo with no direct
projection turned out to be dead for want of exactly this (AUTHORITY.md
§6.7): all neurons are first-class citizens, and with every neuron
un-sticking the starting threshold stops mattering much -- the network
finds its own operating point.

Accuracy **to date** is the mean over every epoch since the start; the
**recent** figure is an exponential average over roughly the last 200.

**Where this stands.** Input-independent targets are learned: `all-off`
passes 90% within a couple of thousand epochs on an 8x4 mesh. For
input-dependent targets learning is real but slow: on an 8x4 mesh with both
rings of neighbours, `reversed` climbs from 50% to the low 60s within ten
thousand epochs. On a 24x20 mesh nothing measurable happened within fifteen
thousand epochs. Reinforcement learning of this kind pays for its generality
with variance, and the variance grows with the number of neurons being
perturbed.

```python
from walnutbutter.learning import Teacher
grid = main(across=24, rows=20, seed=1)
teacher = Teacher(grid, target="reversed", lr=0.03, sigma=0.1, seed=1)
for _ in range(10000):
    teacher.epoch(verbose=False)   # exploration noise, new input, propagate, reinforce
print(teacher.status())
```

## Tests

```bash
pytest
```

## Layout

```
AUTHORITY.md   the specification: constants, substance, connectivity, signalling, activation, learning; the code follows it
src/walnutbutter/
  connection.py Connection: ID, source and target neurons, weight, is_active, kind
  neuron.py    Neuron: threshold, potential, refractory period, receive(), fire(), reset(); no leak
  clock.py     the clock's tolerance: when two moments are the same moment
  propagation.py Schedule: the time-ordered queue of signals in flight, run a wave at a time; propagate()
  dopamine.py  Dopamine: the global pool, its expectation, and the learning at a refire
  arrays.py    ArrayNetwork: the same network as numpy vectors and a scipy sparse matrix (--engine arrays)
  fast.py      the Rust wave loop (rust/), built from a network; train(), compare() against the object engine
  exploration.py the Box-Muller noise draws both engines share
  constants.py every global constant: the default network, the neuron's clock, the learning rule's knobs
  goo.py       Goo: no positions, wired by the scaled rule, zones by index; the only container
  inputs.py    random bits, complement coding, parsing and formatting
  monitor.py   main(): build a network and run its first epoch; run_epoch(): reset and present a new input
  learning.py  output targets, reward, the global-reinforcement rule, and Teacher
  problems.py  the problems (--problem): layout, inputs, and whether a Teacher trains the network
  persistence.py checkpoint() and restore() for learned weights
  cli.py       argument parsing and the `walnutbutter` command
  __main__.py  lets you run `python -m walnutbutter`
tests/         pytest tests for the neuron, the network, and the command line
pyproject.toml project metadata, dependencies, and the command definition
```

## What we mean it for

`VALUES.md` is a statement of values from Byron and Cedric Shock: run it on
non-fission renewable energy, and if it ends up in a robot, that robot
follows the three laws of robotics. It is not a licence term and restricts
nothing the licence grants.

## Licence

Copyright (C) 2026 Byron Shock and Cedric Shock.

The code is free software under the **GNU Affero General Public License,
version 3 or (at your option) any later version** -- `LICENSE`. It comes with
no warranty. The Affero clause is the point: anyone who lets others use a
modified walnutbutter over a network must offer them its source.

Four additional terms, permitted by section 7 of that licence, are appended
to `LICENSE`: preserve the attributions and notices; mark a modified version
as changed; do not use our names to promote your version; and indemnify us
for warranties or support you promise. They add no restriction beyond what
section 7 allows and take nothing away from what the licence grants.

`AUTHORITY.md`, the specification, is licensed **Creative Commons
Attribution-ShareAlike 4.0 International** (CC BY-SA 4.0) --
`LICENSE-CC-BY-SA-4.0.txt`. Share it, change it, build on it; keep the
attribution and pass it on under the same terms.
