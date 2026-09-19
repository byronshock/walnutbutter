# The code's checks against the rewritten AUTHORITY.md (September 17, 2026)

*The companion to `authority-draft-checks.md`. That one read the draft against
the record and the code, and its findings were fixes to the draft. This one
reads the **code** against the file the draft became, and its findings are
changes to the code — the file wins.*

*Produced by seventy-three agents: one auditor per landed conformance commit,
a second reader told to refute every finding it raised (sixty-six raised, six
survived), two site maps, and a completeness pass that diffed Appendix A row
by row against `constants.py`. Every claim below was then checked by hand
against the clause and the line. Nothing here is a matter of taste: each item
is a sentence in the file that the code does not say.*

---

## One thing that blocks the rest

**B1. ~~The Rust extension in the venv is not the Rust source.~~ Withdrawn —
this one was wrong.** The audit compared the installed `.so`'s mtime (05:32)
against `rust/src/lib.rs`'s (12:35) and read seven hours of staleness into the
gap. A git checkout rewrites a working file's mtime without touching its
content, and there had been two that day — the merge of PR 16 and the move of
three commits off `main` — so the comparison measured the checkouts, not the
build.

The binary was current. `fast.build` calls `engine.set_isi_factor(...)`
unconditionally (`fast.py:91`) and every Rust test passes through it, so a
binary predating that method would have failed all of them; they all passed.
Rebuilding settled it: cargo produced a `.so` of identical size, and the suite
gave identical results either side of it.

*Recorded rather than deleted, because mtime is a tempting and useless way to
ask this question and the next reader will be tempted the same way. The way to
ask it is to rebuild and compare, which takes three seconds.*

With the rebuild done, `fast.compare` agrees on the configurations the read
change touched — `copy` and `mnist`, each under the hazard and the hebb
eligibility, at `read=count` and pickiness 2 — so §12.4's bit agreement holds
across the change that landed this afternoon.

**B2. ~~`improved_sustain` cannot be posed.~~ Done — Byron removed it,
September 17, 2026.** Its `input_cells` were hex-grid coordinates and goo has
two rows, so `--problem improved_sustain` exited 2 before the first epoch. It
was also the only user of `input_cells` at all, so the whole (place, row)
input-zone mechanism went with it — the thing §4.3 replaced with "the first
`across` neurons are the input zone". The ten remaining problems all run.

*Whether the rest of the problems survive is still Byron's: Appendix A's
unclaimed list leaves PROBLEM open.*

---

## A. The clock and the neuron's default

*These move every number the project has measured, though by 2% and not by
half — see A1, corrected September 19, 2026. They belong together in one
re-timing, followed by a re-measurement of mnist — not one at a time.*

**A1. ~~§3.2 — the hop is short by the LAG.~~ Done, September 19, 2026.** The file fixes
$2h = \text{REFRACTORY} + \text{LAG} = 5.1$ ms, so $h = 2.55$ ms, written out
and with no name of its own since September 18, 2026, and retires the old form
in Byron's words: *"I no longer want to specify REFRACTORY_HOPS. I want to
specify $h$ directly."* `neuron.py:35` still computes
`REFRACTORY / REFRACTORY_HOPS` = 2.5 ms. The gap is the LAG: every delay in the
system is 2% short of the specified one, and the LAG's whole purpose fails with
it, since two hops then land at exactly 5.0 ms — **on** the refractory wall,
where §3.2 puts them 0.1 ms past it and the clock's slack (§3.4) decides
nothing. The tight case is the neuron's own earliest return, two connections
by §4.4, and not a one-hop arrival: one hop lands inside the period either way
and is §8.13's.

*This entry read, until September 19, 2026, that the hop was half what the file
said and that every delay was half the specified one. That followed §3.2 as it
was then written, which made a whole REFRACTORY + LAG one hop rather than two.
Byron, September 19, 2026: "The current dynamics should be 2.55 ms hops, with 2
hops landing at 5.1 ms after the refractory period they are responsible for
ends at 5 ms." §3.2 and §2.5 were corrected to say so, and the code's error is
2% rather than 50%.*

**A2. ~~Appendix A — LAG has no home, and the code has no hop to add it to.~~ Done with A1.**
`rg '\bLAG\b'` over `src/` and `rust/` returns nothing, so the register's one
remaining unhoused clock constant is LAG. The hop is now written out as
REFRACTORY + LAG and has no name to look for: what the code must carry is the
sum, and it carries `REFRACTORY / REFRACTORY_HOPS` instead (A1). HOPS and
TIME_CONSTANT_OF_TRANSMISSION are no longer register names at all — HOPS went
with the shaping function on September 18, 2026, and the name
TIME_CONSTANT_OF_TRANSMISSION was retired the same day, leaving the quantity.
`REFRACTORY_HOPS` survives in the code as the divisor of A1 and is a different
quantity under a similar name: the refractory period divided by the hop, not
the connections a spike travels. It cannot carry the specified hop even in
principle — the period over 2.55 ms is 1.9607843137…, not a number to write in
a register — which is the arithmetic behind Byron's wanting $h$ direct. A.0 wants one home for every value in the
register.

*The fix, as Byron settled it September 19, 2026.* The code carries **HOP**,
specified directly, and REFRACTORY_HOPS is retired — it cannot hold the
specified hop in any case, REFRACTORY / 2.55 being 1.9607843137… LAG joins the
register beside it, and HOP is what the two make: (REFRACTORY + LAG) / 2. A
checkpoint written with `refractory_hops` **converts on load**, hop =
refractory / refractory_hops, preserving the timing the run actually used,
rather than being refused under §12.2: the field records what a run ran, and
refusing would orphan every checkpoint already on disk. Byron closed the naming
question the same day — *"You just called it a HOP. Let's be consistent and
call it a HOP"* — so §3.2 no longer says the hop has no name of its own, the
register carries HOP at 2.55 ms as (REFRACTORY + LAG) / 2 in place of the
derived REFRACTORY + LAG row, and §0.12's drive quotes its interval as 2 HOP.

*Applied September 19, 2026, and one thing it changed that was not foreseen.*
`constants.py` carries `HOP = (REFRACTORY + LAG) / 2` and `LAG`, the class
attribute is `Neuron.hop` rather than a `hop()` over a ratio, `--hop` replaces
`--refractory-hops`, checkpoints write `hop`, and `persistence.hop_of`
converts an old file by `refractory / refractory_hops`, with a test for both
directions. What was not foreseen: **INTERVAL is no longer a whole number of
hops.** It was exactly 14 at 2.5 ms and is 13.7255 at 2.55, so under `forced`
drive each epoch's driven spikes now start a lattice of their own instead of
continuing the run's — per-epoch alignment falls from 100% to 11%, while the
run-wide lattice holds at 91%, the missing 9% being the driven spikes
themselves at multiples of INTERVAL. `forced` is not the default and §4.3
prefers the rate drive, so nothing in force depends on it; the effect weakens
the very synchrony Byron struck `forced` for on September 14, 2026. It is
recorded here because a future INTERVAL chosen as a whole number of hops would
bring it back without anyone intending it.

**A3. ~~§7.4 — TARGET_ISI is derived, and the code's is the superseded one.~~
Dissolved — Byron took the shaping function out, September 18, 2026.** The
register's derived TARGET_ISI = 10.2 ms and the code's free 5.1 disagreed; the
constant is gone from both.

**A4. ~~§7.4 — the shaping function's negative region never runs.~~
Dissolved with it.** The file anchored $f$ at $-1$ at the refractory wall and
all three engines computed the superseded $u = t/I$, so the negative region no
decision ever reached was still the one that ran. The discrepancy was real and
is what sent the question back to Byron; what came back was that the mechanism
goes. *Byron, September 18, 2026: "I want to factor out shaping. I don't
understand it. A fundamental principle of this project is that we ONLY INCLUDE
MECHANICS WE UNDERSTAND."* §7.4 now says there is no shaping function and no
rule reads the interval since a neuron's own last spike; the code says the
same, in all three engines.

**A5. ~~Appendix A — TAU is ∞, and the code leaks by default.~~ Done,
September 19, 2026.** The register: *"TAU | ∞ | the potential does not leak:
the evidence accumulator. 2 ms is the leak, kept as a per-run option with its
analysis"*, quoting Byron: *"neuron: NOT leaky, but please leave the leak
option with its analysis."* `constants.py` had `TAU = 2.0` and `--tau`
defaulted to it, so every run leaked and the accumulator — the thing §1.2 and
§8 are written on — was the branch a run had to ask for. `TAU` is now
`math.inf`.

*Where the 2 went.* It is `LEAK_TAU`, named rather than deleted, because the
register keeps the leak "as a per-run option with its analysis" and a bare 2.0
in a help string is not that. It is not a second name for TAU: it is the value
the option was swept to, and `--tau` is how a run asks for it. **Every number
measured before this date was measured under the leak**, which is the whole
reason section A goes together and is followed by a re-measurement.

**A6. ~~§3.4 — TOLERANCE is a thousand times looser than specified.~~ Done,
September 19, 2026.** The register fixes $10^{-12}$; `clock.py` had `1e-9` and
`rust/src/lib.rs` a **second literal** of the same quantity at the same wrong
value, which A.0 forbids on its own. Both now read $10^{-12}$.

*What became of the second literal.* A Rust compile-time constant cannot be
taken from Python, so it stays — and a test holds it equal to `clock.TOLERANCE`
and skips when the loop is not built, which is what makes it a mirror rather
than an independent value. That is A.0's guarantee for this quantity: if the
two ever part, every comparison of two moments parts with them and the engines
stop agreeing on which events share a wave.

**A7. ~~§5.4a — the presentation window is not built.~~ Done, September 19,
2026.** The clause is new, of
September 18, 2026: the drive runs to $t_e + \text{PRESENTATION\_TIME}$ and not
past it, defaulting to INTERVAL. No engine carries the constant, and all three
draw arrivals to $t_e + \text{INTERVAL}$. At the default the behaviour is
identical, so nothing measured moves until a run shortens the window — which
is why this sits with the re-timing rather than ahead of it.

*Two things to get right when it is built.* The arrivals are drawn from the
network's own stream, so a shorter window takes **fewer draws** and shifts
every later draw in that stream; the three engines must shorten identically or
§12.6 fails. And `fast.compare` cannot catch a wrong window on its own: at the
default there is nothing to see, and away from the default every engine that
shares the same drawing code would be wrong together. The test has to be
against the clause — that no arrival lands past the window — and not against
another engine. §12.11's exact resume needs the constant in the checkpoint.

*How it was built, and why the traps did not bite.* One function draws the
arrivals — `Network.input_schedule` — and all three engines take its list, the
Rust loop included, so the three shorten identically by construction and there
is no second drawing site to keep in step. `PRESENTATION_TIME` is a sentinel
`None` meaning the whole epoch rather than the number 35, because INTERVAL is a
per-run and per-problem value and the default window must follow it. The
refusal of §0.5 is raised when the window is set, not at the first draw, so a
run that asks for a window past the horizon fails before it starts. The test is
against the clause — no arrival past the window, fewer draws than the whole
epoch, and the default drawing the whole of it — and the engines were compared
at windows of 35, 10 and 2.55 ms, all three agreeing to the bit.

---

## B. Rules the specification does not carry — **done, September 17, 2026**

**B3. ~~§9.1 — one rule pays at the read, and it is not the one that runs.~~ Done.** The
clause: *"At most one of them runs... The specification carries one of each:
the reinforce rule pays at the read, the quash is local."* The code carries
five rules, and `RULE` defaults to `"teacher"`. Live and outside the file:

- **the dopamine rule** — `dopamine.py` (372 lines), eight constants, its CLI
  flags, its checkpoint record, and its construction inside `Teacher`. None of
  the eight names appears in Appendix A, in the table or in the unclaimed list.
- **the external teacher** (`teacher_score`, TEACHER_CREDIT) and **adaline**.
- **leaky Hebb** (HEBB_RATE, `--hebb`, and `leaky_hebb` in all three engines) —
  a second local rule where §10.2 says the quash is the whole of them.
- **the per-epoch weight decay** (WEIGHT_DECAY, `network.py`, `arrays.py`,
  `lib.rs::forget`) — a third.

**B4. ~~§8.3 — three of five eligibilities leave.~~ Done.** `hazard` and `hebb` survive;
`perturb`, `wrong_hebb` and `count_hebb` do not. The map found **117 sites**.
The clause settles which `hebb` survives: §8.4's table gives credit 1 and
expectation $\hat p_j$ **per decision**, which is the code's `hebb`
(`network.centre`, DECISION_MEMORY). The code's `count_hebb` is a per-epoch
rule with no $c_j/q_j$ pair, and goes.

*Its RNG risk is the reverse of what one expects.* Removing `perturb` removes
no draw from a taught run — the Teacher already zeroes sigma for anything but
perturb, and the additive block is gated on `sigma > 0` in all three engines.
The hazard is at the untrained path: `cli.py` runs `run_epoch(..., noise=args.sigma)`
with no Teacher to zero it and `--sigma` defaults to 0.1, and the Box-Muller
block draws **before** the hazard uniforms from the same stream. Deleting SIGMA
deletes $n + (n \bmod 2)$ uniforms per wave, so every `--no-learn` run with a
width lands on different spikes.

**B5. ~~§5.2 — complement coding only, *and there is no permutation*.~~ Done.** The
clause is three removals, not one: the dropped codings (raw, population,
population-complement), the error-correcting codes and the input flips, and
**the permutation itself** — *"Place $i$ of the input zone shows place $i$ of
the coded pattern of 5.3: there is no permutation. The pattern presented is the
pattern the read is scored against — nothing corrupts an input on the way in."*
The map found **124 sites**.

*What this costs.* The shuffle draws from the network's own seeded stream, by a
variable number of words per draw, so no "burn $k$ draws" shim reproduces the
old stream. But **mnist is untouched**: it is `permute=False` and `flip=0`, so
no mnist arm ever took either draw and every mnist checkpoint and every mnist
number survives. What breaks is the permuted set — `reversal` (the default
problem), `doubled_copy`, `reaching_copy`, and
`checkpoints/reversal-79pct.json`, which stores `[7,4,5,0,2,3,6,1]` — and
`population_denoise`, whose flip draws per epoch and so cannot be reproduced at
all afterwards.

*One silent failure to avoid.* `persistence.py` reads `data["permutation"]`
hard. If the writer goes and the reader stays, old checkpoints still load; if
the reader goes, a run scored against a scrambled zone resumes against an
unscrambled one with no error at all. Bump FORMAT and refuse a file carrying a
non-identity permutation rather than defaulting it away.

---

## C. The reads, and the two non-defaults that are on

**C1. §5.10 — "It is the only read", and it is one of five.** The network's
default read is `"fired"` and the command line offers five. Only mnist sets
`read="count"`, so every other run reads the zone a way no clause carries — and
under `"fired"` the row critic's pickiness of 2, which landed this afternoon,
is never applied at all.

**C2. ~~§9.9 and §9.10 — marked Non-default, and defaulted on.~~ Done —
D5 answered by the file, September 18, 2026.** The Teacher and the command line
both defaulted to HOMEOSTASIS and UNSTICK, so `reversal` and `copy` drifted
thresholds by a rule the file calls non-default; only mnist asked for them off.

> **D5 is closed, and §9.9 answered it.** Byron had answered "the constants
> become 0"; the clause, quoting him the day before, says the opposite — these
> are "latent knobs", so the constant keeps the value it would run at and the
> run switches it off. Asked which stood, he chose the file's reading. The
> constants are untouched; off is now the default everywhere, and a run asks:
> bare `--homeostasis` runs it at 1e-6, `--homeostasis 5e-6` at that, absent is
> off. The quash was already right and was the model.

**C3. §9.13, §7.5 — the old rate teacher's two extremes are still live.**
RATE_ON = 200 Hz and RATE_OFF = 0 are exactly the saturation and silence the
new set point replaces. They drive the `"rate"` read through `output_levels`
and have CLI flags. None of RATE_ON, RATE_OFF or RATE_TAU is in Appendix A.

---

## D. Resume and checkpoints

**D1. §12.11 — a resume is exact, and nothing carries a generator's state.**
*"A run resumed from a checkpoint produces, bit for bit, what the uninterrupted
run would have produced... A checkpoint carries each generator's state, not its
seed and a count of draws."* `getstate` appears nowhere in `persistence.py`;
the sweep driver still restarts the exploration stream at seed + 1,000,000 and
advances the input stream rather than restoring it — which is, word for word,
what the clause's own note says it replaced.

> **`rewrite-decisions-2.md` lists this as W5, "not decided". That is stale.**
> §12.11 settles it and quotes Byron settling it: *"I would like resumes to be
> exact ... the resumed run reproduces, bit for bit, what the uninterrupted run
> would have done ... It's worth having."* One fewer open question than the
> decisions document says.

**D2. §12.9 — the round trip is not whole in every engine.** `fast.build` hands
Rust the topology, weights, thresholds, floors, deltas and expectations and no
clock state — no potentials, spike times, synapse stamps, traces, open-arrival
notes or signals in flight — and zeroes $E_j$. A Rust arm's checkpoint is then
written from the object network the engine was built from, which never ran, so
the fields §12.9 requires are recorded as fresh-build values.

**D3. §12.11 — the rebuild is not checked.** The clause wants a network
"checked neuron for neuron and synapse for synapse against a fresh build from
the seed, and refused if they differ". `load_weights` compares `across`, `rows`,
`seed` and the connection **count**, so a checkpoint whose wiring differs edge
for edge loads silently as long as the totals match.

---

## E. The register, and A.0

The Appendix A diff was run row by row. **Everything not listed below matches**
— sixty-odd rows verified equal, the mnist rows included.

| row | register | code |
|---|---|---|
| TAU | ∞ | 2.0 |
| TOLERANCE | 10⁻¹² | 1e-9 |
| LAG | 0.1 ms | absent |
| REFRACTORY + LAG | 5.1 ms | 2.5 ms, as `REFRACTORY / REFRACTORY_HOPS` |
| ELIGIBILITY | hazard | "perturb" |

*TARGET_ISI, HOPS and EARLY_ARRIVAL_PUNISHMENT_FACTOR left this table with the
shaping function on September 18, 2026 (A3, A4): the register no longer carries
them and neither does the code. TIME_CONSTANT_OF_TRANSMISSION left it the same
day, as a name and not as a quantity — Byron retired the name, so the row is
keyed on the arithmetic and the gap it records is A1's.*

*In the code and in no row of the register, on this same subject:*
`REFRACTORY_HOPS` *— the divisor A1 is about. It goes when the hop does.*

In the code and in no row of the register: COUNT_MEMORY, SIGMA, EXPLORE,
LEAKY_ELIGIBILITY, FLIP, READ_WINDOW, BORED_AFTER, TEACHER_CREDIT, RATE_ON,
RATE_OFF, RATE_TAU, WEIGHT_DECAY, HEBB_RATE, SYNAPSE_TAU, and the eight
dopamine constants.

**A.0, one value to a name:** `mnist.THRESHOLD = 0.5` (the binarisation)
collides with `constants.THRESHOLD = 0.25` (the firing threshold) — two
quantities of Appendix A sharing one name.

---

## F. Smaller, and one latent bug

**F1. §12.3 — the default engine cannot be selected.** The clause makes the
Rust loop "the default engine for any sweep or long run"; `--engine` offers
`objects` and `arrays` only, and defaults to `objects`. Every long run started
through `walnutbutter` rather than `docs/rust-sweep.py` runs on the engine the
clause puts third.

**F2. §9.12 — the estimator's correlation is Rust-only.** The clause names this
itself as the one place §12.1 is not met and says it is not a precedent. It is
K2 in `rewrite-decisions-2.md`, and still open.

**F3. §5.3 — the permutation currently scrambles the clock neurons.**
`network.py` prepends the clock's 1s and *then* permutes, so with
`permute=True` and `clock > 0` the always-on bits land at arbitrary places,
against §5.3's "They are the first that many neurons of the input zone". No
problem hits it today — mnist is the only one with a clock and it is
unpermuted — and it disappears when the permutation goes under §5.2. Recorded
so it is not reintroduced.

---

## What is already done

- **§4.1** — the plane and its containers are out of the tree; goo is the only
  container.
- **§5.10, §9.5** — the read is the count, and the line is
  ROW_CRITIC_PICKINESS_IN_SPIKES = 2 in all three engines. (C1 above is the
  part still outstanding: the count read is not yet the *only* read.)
- **§9.3** — the reinforcement baseline survives a resume, in both the sweep
  driver and the watch. (D1 is the part still outstanding: exactness needs the
  generators' state.)
- **§8.3** — the reinforce rule refuses to learn where the threshold decides,
  in all three engines.
- **§12.4** — the Rust extension is rebuilt, and `fast.compare` agrees with the
  object engine on `copy` and `mnist`, each under hazard and hebb, at the new
  count read and pickiness 2.
- **§9.1, §8.3, §5.2 — the three deletions.** One rule pays at the read and one
  local rule; two eligibilities; complement coding and no permutation. About
  4,000 lines out across three commits, the Rust engine rebuilt at each, the
  suite green and the engines compared at each. Seven of the ten problems went
  with them — each was posed on the external teacher *and* on a dropped coding —
  leaving `reversal`, `copy` and `mnist`. `improved_sustain` had gone already.

What the tree still owes is Group A (the clock, the shaping function and TAU),
Group C (the reads and the two non-defaults), Group D (resume and checkpoints),
Group E (the register) and Group F.
