# Synapse-level exploration and learning in the engines: the plan

September 23, 2026, 15:30 MDT. Byron's decision, this afternoon: "I want to modify
the existing engine to incorporate synapse-level exploration and learning." This is
the plan for doing it in all three engines, from the clauses down, as §0.1 and §12.1
require. It is written from Byron's mechanism as he stated it in §7 of
`synaptic-escape-noise-2026-09-22.md` and answered it in §7.10, and from what the
second toy measured (§7.11–§7.13 and the four sweeps of this afternoon, tabled at the
end). It proposes an answer to every open question so that the engineering can be
scoped, and marks each one **his**: nothing here is a clause until he writes it.

## 0. What is being built, in one paragraph

Below threshold, every synapse of a neuron makes its own escape decision (lazily with each incoming wave) at a hazard that rises with its source's potential as a fraction of the
neuron's threshold and is certain at threshold, where the neuron spikes
deterministically and every synapse it has transmits. An escape delivers the
synapse's weight to its target *one hop later* and leaves the source's potential
alone. The presynaptic neuron keeps its threshold, its deterministic spike and its refractory
period; its own escape noise (§6.5) is turned off when the synapse's is on. Learning
stays REINFORCE paid by one scalar at the read (§8.1): the score of each synapse's
decision is posted to the synapses **into its source**, through their traces, exactly
as §8.4 posts a neuron's decision to its fan-in today — the exploring object and the
credited object are neighbours one hop apart (§7.9, closed by Byron's first answer).
A synapse's own weight is credited, as today, by what it had standing in its target
when the target's synapses took their chances. Two options ride on it, both measured
in the toy: the trace may count every delivery or only the ventured ones (§7.12), and
an input may be charged by the drive instead of forced (§7.11).

**Time stays continuous.** The toy decided once per hop; the engines run a clock of
irregular moments (§3.3) and the hazard is a rate that runs on elapsed time
(§6.5: $m = (\Delta t / \text{hop})\,\kappa(N)\,h$, exposure since the previous decision or the
refractory period's end). The synapse's hazard takes exactly that form, with $h$ a
function of the source's potential instead of the neuron's margin. Nothing about the
wave structure changes; what changes is who decides in phase two of a wave.



## 1. The mechanism as clauses — proposed text for Byron to accept, amend or strike

Each item names the clause it changes, proposes the rule, and says what the toy
measured where it bears. Items marked **his** are decisions only he can make.

**1.1 §6 Firing: the neuron.** A neuron fires iff $V_j(t) \ge \theta_j(t)$ (please replace p to mean potential with V to mean voltage throughout, so we don't get potentials mixed up with probabilities) and it is not refractory (§6.9, the deterministic rule word for word) whenever the synapse
hazard is on. ESCAPE_DELTA is then 0 for every neuron, and a run that asks for both
a neuron width and a synapse hazard is **refused** with a reason (§0.5): the system
explores by one thing (§7.1). *His:* whether both may ever run together as an
experiment. Refuse now, revisit if a measurement wants it.

**1.2 §7 Exploration: the synapse's decision.** At every wave, after the floor and
after the wave's spikes are decided, every synapse $i \to j$ whose source $i$ is not
refractory and did not spike this wave decides. Its expected escapes over the
interval since $i$'s synapses last decided (or since $i$'s refractory period ended)
are

$$m_{i}(t) = \frac{\Delta t}{\text{hop}}\;\kappa(N)\;h(u_i), \qquad u_i = \frac{\operatorname{clip}(V_i(t), 0, \theta_i)}{\theta_i},$$

the same $m$ for every synapse of $i$, and the synapse escapes iff its uniform is
strictly below $P = 1 - e^{-m}$, computed as $-\operatorname{expm1}(-m)$ with $m$ capped at $10^3$ (§6.5's
engine rules, unchanged). $\Delta t$ is never negative (§6.5). One exposure clock per
source neuron serves all its synapses, since they decide together on the same $u$;
it is today's `exposed_since`, whose neuron-side use ends with 1.1.
*Byron's second answer: an escape does not move $V_i$. His fourth: the neuron keeps
its deterministic spike, so $u = 1$ never reaches a synapse decision.*

**1.3 §7 The hazard family and the rest hazard.** Two families, both $h_0$ at rest
and 1 spike per hop at threshold, quoted before $\kappa(N)$:
loglinear $h(u) = h_0^{\,1-u}$ (today's exponential hazard is this family with
$h_0 = e^{-1/\Delta}$, §7.11) and linear $h(u) = h_0 + (1-h_0)\,u$ (Byron's words read
literally, §7.13). Two constants in Appendix A: SYNAPSE_HAZARD_REST ($h_0$; 0 turns
the mechanism off and leaves §6 as it stands) and SYNAPSE_HAZARD_FAMILY.
*His:* the family and the default $h_0$. Proposed: loglinear, $h_0 = 0.01$ — the one
value that sat on the plateau in every toy regime (forced/full 0.986–0.991 with
sd ≤ 0.009; charged/full 0.974–0.986; charged/ventured 0.924–0.940), where 0.003
cost seeds under the charged drive and 0.03 cost seeds under the forced one at long
epochs. The sweep that sets it on mnist is §5 below.

**1.4 §6.6 Scaling.** $\kappa(N) = \sqrt{N_0/N}$ multiplies every hazard, the synapse's
included: the clause already says "every hazard", and the toy ran it so. *His:*
whether a neuron of fan-out $F$, which now speculates at $F$ times the per-synapse
hazard, should instead be scaled by its fan-out (the record's $1/d$ per hop, §0.3).
Proposed: $\kappa(N)$ now, fan-out scaling as a later measurement, since §0.7 wants
the scaling built into the rule and the two are one constant apart on a fixed goo.

**1.5 §2.5 / §6.12 A refractory source.** Its synapses make no decision and accrue no
exposure; they resume at the period's end. The toy did this (`act_in`), and it is
what §0.4's "ignores everything" says. *His:* the alternative is that a refractory
source's synapses whisper at $h_0$, its potential being 0. Proposed: suspended.

**1.6 §3.5 / §3.8 The draws and the signal.** One uniform per synapse per wave, in
edge order (the object engine's push order, §12.5), from the exploration stream,
taken after the floor and before anything fires (§3.8's place), whether or not the
synapse can use it, so the stream's position depends only on the count of synapses
and the count of waves. An escape schedules a signal one hop later like any spike's
signal (§3.5), carrying a **ventured** mark; delivery, the refractory drop (§8.13),
the floor and the stamp treat it as any signal. *His:* whether a ventured delivery
moves the stamp the quash reads (§1.7, §10.1, a non-default). Proposed: yes, it is a
signal.

**1.7 §8.4 The rule.** At every wave, for every source $i$ whose synapses decided,
with $n$ of its $F_i$ deciding synapses escaping: the credit of each escape is
$c = m\,e^{-m}/(1 - e^{-m})$ and the expectation of each silent decision is $q = m$
(the hazard row of §8.4, unchanged), and the wave posts to every synapse $k \to i$

$$e_{ki} \mathrel{+}= \big(n\,c - (F_i - n)\,m\big)\,x_{ki},$$

$x_{ki}$ the trace of $k \to i$ (§8.5). This is §8.4 with the decision being the
synapse's rather than the neuron's and $F_i$ decisions a wave instead of one. Its
mean at every wave is zero, as today's is (§8.3): both branches have mean $m e^{-m}$.

**1.8 §8.11 The bookkeeping.** No loop over a fan-in runs at a decision (§1.8). Per
neuron, a running **gain** $G_i$ replaces today's pair $E_j$ and credit: at each wave
$G_i \mathrel{+}= n\,c - (F_i - n)\,m$. Per synapse the trace, the note and the score
stay what §1.5 names — the mechanism adds **no per-synapse state**. An arrival
integrated on $k \to i$: $x_{ki} \mathrel{+}= 1$, $B_{ki} \mathrel{+}= G_i$. A settle —
$i$'s spike, the floor, a forced spike, a discharge, and the read — posts
$e_{ki} \mathrel{+}= x_{ki} G_i - B_{ki}$, then clears $x$ and $B$ and restarts $G_i$
at 0 (the read re-bases $B_{ki} \leftarrow x_{ki} G_i$ instead, §1.9). Today's rule
is the case $G = -E$ plus one credit at the spike; the identity is the same one, and
the tests hold it against the per-wave sum on small networks. Under the leak (§8.12)
the walk over the fan-in runs once a wave per source that decided, as it runs once a
decision today. The deterministic spike itself posts nothing: it is not a draw.

**1.9 §1.6 / §8.5 The trace's content — a run option.** TRACE $=$ all (default):
$x_{ki}$ counts every arrival $i$ integrated along $k \to i$, ventured or relayed, and
stays the derivative of the potential by the weight. TRACE $=$ ventured: it counts
the ventured ones only — Byron and Cedric's idea of §7.12 — and is then not the
derivative but a biased estimator, which the clause says in words. The potential
takes every delivery under both. *His:* the default. Proposed: all, the gradient,
with ventured a named option; the toy's numbers are in the table at the end.

**1.10 §5.4 The drive — a second kind, charged.** The Poisson rate drive stays the
default as it ran (§5.4, Byron's word of September 17). A run may name the drive
**charged**: each arrival the Poisson process would have fired the neuron with
instead delivers $\theta_i / \text{DRIVE\_STEPS}$ to its potential, DRIVE_STEPS $= 3$,
at DRIVE_STEPS times the rate, so the input reaches threshold about as often as it
was fired before and spikes by the deterministic rule of 1.1, its potential readable
by its synapses between deliveries. The arrivals are drawn from the network's stream
in the order §5.4 fixes; nothing is driven at the epoch's moment (§5.7). A charged
input keeps the driven mark of §5.8, so §8.1 still skips its incoming synapses.
*His:* whether to carry the charged drive at all, given §0.12's deterministic drive
intention; and the mark. Proposed: carry it as a run option, since the toy says the
ventured trace has nothing to learn from under a forced input and the file's own
drive is forced.

**1.11 §5.10 / §9.4 The read — the read synapse.** With the synapse hazard on, each
output neuron has one outgoing **read synapse** to the read: it decides like any
synapse, on the output's potential, and the count read counts its ventured
transmissions beside the output's spikes; the evidence critic scores the sum. This
is the toy's device (§7.11), and it is what lets the goo with no hidden neurons
learn: an output projects nowhere else, so without it the exact rule posts nothing
to an output's fan-in. It is one decision per output per wave, its draws taken after
the network's synapses' in output order, and it carries a per-epoch count and nothing
else. *His:* whether the read synapse is part of the mechanism or an option, and
whether it should exist at all when hidden neurons give the outputs projections.
Proposed: part of the mechanism — the read is where the reward is, and the outputs'
speculation should reach it whatever the wiring.

**1.12 §0.10, §0.11, §7's closing note, §0.12.** §0.11 stops being a direction and
becomes the rule; §0.12's first line is taken up; the closing note of §7 goes. §0.10
is unchanged in substance: one scalar still pays every synapse, and what tells them
apart is the score. §8.7's remark that $1/\Delta_j$ is folded into LR falls away
with the width: the learning rate is quoted plain, and its value is re-found (§5).

**1.13 What a checkpoint carries (§8.14, §12.9).** Per neuron $G_i$ in place of $E_j$
and the credit; the per-epoch read-synapse counts; the ventured mark on every signal
in flight (`engine_pending`); the settings — $h_0$, the family, the trace, the drive
and DRIVE_STEPS — so a network saved under them resumes under them (§12.9). The
escape scale is recomputed from the count (§12.10). A resume stays exact (§12.11):
the exploration stream's state is carried as today; the draw count per wave is the
edge count, so its position is a function of the record alone.

## 2. The engines

Every rule above is implemented in all three engines and compared (§12.4): objects
and Rust to the bit, arrays to a part in a billion. The order below is the order the
comparison harness needs them in.

**2.1 The object engine** — `neuron.py`, `connection.py`, `propagation.py`,
`network.py`, `learning.py`.
- `Network.set_synapse_hazard(h0, family)` beside `set_delta`: sets every neuron's
  $h_0$ and family, the escape scale, forces every width to 0 and refuses a positive
  `escape_delta` (1.1). `Network.centre` stays for hebb; hebb under the synapse
  mechanism is out of scope (there is no neuron decision for it to centre) and is
  refused.
- `Neuron.decide` returns the deterministic comparison when the synapse hazard is on
  and posts nothing. New `Neuron.speculate(now, draws)`: $u_i$, $\Delta t$, $m_i$,
  one Bernoulli per outgoing active synapse in edge order, returning the escaped
  connections for the schedule and posting the wave's entry to $G_i$ (or, under the
  leak, walking the fan-in per §8.12). `exposed_since` is the source clock.
- `propagation.Schedule.run`: phase two becomes — forced neurons fire; every neuron
  decides deterministically (touched first, then everyone, as today); then every
  neuron speculates, in neuron order, with the wave's per-edge draws; escapes are
  scheduled at `time + hop` as `Signal`s carrying `ventured=True`. Delivery treats a
  ventured signal as any signal except that under TRACE = ventured a relayed arrival
  does not raise the trace.
- `Connection`: no new field. `Signal` and the schedule's event tuple gain the
  ventured bit.
- The read synapse: an `Output` bookkeeping list on the network — per output a
  per-epoch ventured count, decided in `speculate`'s pass after the network's
  synapses; `learning.class_sums` and `evidence_score` read spikes plus that count.
- The charged drive: `inputs.py` / `network.py` where the Poisson arrivals become
  events — arrivals of kind EXTERNAL with amount $\theta_i / \text{DRIVE\_STEPS}$ in
  place of STIMULUS, the driven mark set on the neuron at its first arrival.
- `learning.reinforce`: `settle_scores` posts $x G - B$ and re-bases; the pay is
  §8.1 unchanged.

**2.2 The Rust wave loop** — `rust/src/lib.rs`, mirroring 2.1 line for line.
- Per-neuron arrays: `h0`, `gain` (replacing `expected`/`credit` when the mechanism
  is on), `family`; per-edge `draw` sized to the edge count; `synapse_draws()` takes
  the uniforms from Python's MT19937 in edge order (the existing `hazard_draws`
  pattern); read-synapse draws follow in output order.
- Events: the heap payload gains a ventured bit; `pending_events()` and
  `push_events` carry it (§12.11).
- The loop: after the floor and the fires, `speculate(i, time)` for every neuron in
  order; escapes push `(time + hop, SIGNAL, edge, ventured)`.
- Setters and getters for the gain, the read counts and the settings, so
  `fast.build`, `_save_network` and `resume_grid` round-trip everything (1.13).
- Cost: draws per wave equal the edge count — 1,396 on goo 455, ~13,000 with 60
  hidden — against the neuron count today. At 39 waves an epoch that is 54,000 and
  510,000 uniforms an epoch; MT19937 in Rust is near 10 ns a draw, so 0.5 ms and
  5 ms an epoch against the 35 ms an epoch the accumulator runs now: under 15 per
  cent on the largest current goo. Measured after the build, not assumed.

**2.3 The array engine** — `arrays.py`: the per-edge decision vectorised over the
edge arrays ($u$ gathered by source index, $m$ per edge, draws per edge); ventured
arrivals are per-edge deliveries rather than per-source rows of the matrix, so the
wave's delivery gains a gather of escaped edges before the matrix product; the
gain's entries scatter-added per source. Tolerance as §12.4.

**2.4 The Python side** — `fast.py` (`build`, `train`, `compare` mirror the new
setters and the read; `compare` refuses what it cannot mirror, as today),
`persistence.py` (1.13), `cli.py` and `constants.py` (the flags and the register:
`--synapse-hazard H0`, `--hazard-family`, `--trace`, `--drive charged`,
`--drive-steps`; SYNAPSE_HAZARD_REST = 0 so that nothing changes until a run or a
clause turns it on), `docs/rust-sweep.py` (KNOBS gains `synapse_hazard` so $h_0$ can
be swept; family, trace and drive fixed per sweep and recorded in every arm's json).

**2.5 Refusals (§12.2).** A neuron width with a synapse hazard; hebb with a synapse
hazard; the reinforce rule with neither width nor synapse hazard (already refused);
an engine asked for a family it lacks; TRACE = ventured on the arrays engine until it
is built there. Each with a reason.

## 3. Tests — written before the engines, held by all three

1. `test_synapse_hazard.py`: $m$ and $P$ per clause at $u \in \{0, \tfrac12, 1^-\}$ for
   both families; $h_0 = 0$ under loglinear is the deterministic network word for
   word (no draw escapes); the wave's entry has mean zero over draws; the
   bookkeeping identity $\sum (x G - B)$ equals the per-wave sum on a three-neuron
   chain, accumulator and leak; TRACE = ventured leaves a relayed arrival out of the
   trace and in the potential; a refractory source's synapses accrue no exposure;
   the read synapse's count reaches the evidence critic.
2. Agreement (§12.8): objects against Rust to the bit and arrays to $10^{-9}$ on goo
   60 copy and on goo 455 mnist, for {forced, charged} × {all, ventured}, 40 epochs
   each, before any sweep — `fast.compare` extended, in `test_hazard.py`'s pattern.
3. The stream: the draws taken per wave equal the edge count plus the output count
   whatever fired, so a seed's spikes do not depend on the engine (§3.8, §12.6).
4. Checkpoints and resume: `test_resume_exact_rust.py`'s five cases with the
   mechanism on, plus one where ventured signals are in flight at the cut.
5. Refusals of 2.5, each by message.
6. The driver: an arm's json names $h_0$, the family, the trace and the drive.

## 4. Sequence, and what it costs

| phase | what                                                                                    | who                   | time        |
| ----- | --------------------------------------------------------------------------------------- | --------------------- | ----------- |
| 0     | the clauses of §1, decided and written into AUTHORITY.md; the constants named           | Byron (Cedric on 1.9) | one sitting |
| 1     | tests of §3.1 and §3.5; the object engine (2.1) green on them                           | Claude                | a day       |
| 2     | the Rust loop (2.2); objects and Rust to the bit on both configurations                 | Claude                | a day       |
| 3     | the array engine (2.3) to tolerance; checkpoints, resume, CLI, driver (2.4, §3.4, §3.6) | Claude                | a day       |
| 4     | §5's measurements                                                                       | the machine           | see §5      |

The engine build must not touch the venv's `.so` while the three mnist sweeps run
(about twelve hours more): the work goes in a worktree with the isolated wheel and
PYTHONPATH recipe already used on September 21, and lands in the venv when the
sweeps are done. A branch `synapse/exploration` off `code_follows_the_file`, one PR
per phase or one for all three, with the suite green locally with the engine built
(CI does not build it).

## 5. Measurement, once the engines agree

1. **Compare on the configuration** (§12.8): goo 455, interval 100, the mnist read,
   each drive and trace — done in §3.2, repeated on the built wheel.
2. **Smoke and resume rehearsal**: 3,000 epochs, 3 seeds, cut and continued.
3. **The learning rate, re-found**: 25,000 epochs, 3 seeds, LR in
   {0.0005, 0.001, 0.002, 0.005, 0.01} at $h_0 = 0.01$, charged drive, full trace —
   the toy's 0.01 is not the engine's 0.01 (the engine's best rate today is 0.002 and
   every step up cost it accuracy).
4. **The four corners at 100k, 3 seeds**: {forced, charged} × {all, ventured} at the
   rate found, $h_0 \in \{0.003, 0.01, 0.03\}$ — 36 arms, about a day at 30 workers.
   Ranked on accuracy with `stuck_off` beside it, never on score.
5. **The best corner at 1M, ten seeds**, against the accumulator's 0.2577 and the
   leak's 0.2875. Then hidden neurons, and the fan-out scaling of 1.4.

## 6. What the toy says the numbers will look like — the afternoon's four sweeps

Accuracy at 25,000 epochs on the toy, ten seeds, LR tied to the epoch at 0.35, all at
the tie's centre LR 0.01 / 35 waves unless noted; chance 0.15.

| drive, trace | $h_0$ 0.001 | 0.003 | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.07 |
|---|---|---|---|---|---|---|---|---|
| forced, all | 0.972 ± 0.056 | **0.992 ± 0.004** | 0.991 ± 0.004 | — | 0.985 ± 0.005 | — | — | — |
| forced, ventured | 0.116 | 0.108 | 0.114 | — | 0.110 | — | — | — |
| charged, all | 0.940 ± 0.112 | 0.974 ± 0.049 | 0.974 ± 0.054 | — | **0.988 ± 0.007** | — | — | — |
| charged, ventured | 0.854 ± 0.142 | 0.892 ± 0.124 | 0.938 ± 0.081 | 0.961 ± 0.053 | 0.962 ± 0.054 | 0.959 ± 0.054 | 0.883 ± 0.205 | 0.843 ± 0.191 |

Three things to carry into §1's defaults: the ventured trace needs a charged input
and then costs about 0.03 and one weak seed in ten at its plateau of 0.02 to 0.04;
the full trace on the charged drive is as good as on the forced one at $h_0$ 0.03
and slower to 0.8; and $h_0 = 0.01$ is the one value inside every regime's plateau.
Every figure is a toy's — 64 fully connected inputs on five outputs, fan-in 64,
scaling 0.93 against mnist's 0.36 — and §5 is what replaces them.
