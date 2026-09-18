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

**B2. `improved_sustain` cannot be posed.** Its `input_cells` are hex-grid
coordinates (`problems.py:96`, row 3), and goo has two rows. `--problem
improved_sustain` exits 2 before the first epoch. Every other problem runs.
Whether the problems survive at all is Byron's: Appendix A's unclaimed list
leaves PROBLEM open.

---

## A. The clock, the shaping function and the neuron's default

*These four move every number the project has measured. They belong together
in one re-timing, followed by a re-measurement of mnist — not one at a time.*

**A1. §3.2 — the hop is half what the file says.** The file fixes
$h = \text{REFRACTORY} + \text{LAG} = 5.1$ ms and retires the old form in
Byron's words: *"I no longer want to specify REFRACTORY_HOPS. I want to specify
$h$ directly as TIME_CONSTANT_OF_TRANSMISSION."* `neuron.py:51` still computes
`REFRACTORY / REFRACTORY_HOPS` = 2.5 ms. Every delay in the system is half the
specified one, and the LAG's whole purpose fails with it: a signal sent to a
neuron that fired in the same wave now lands at $t + 2.5$ ms, **inside** that
neuron's refractory period, where §3.2 requires it to land at $t + 5.1$ ms,
just after recovery.

**A2. Appendix A — LAG, HOPS and TIME_CONSTANT_OF_TRANSMISSION have no home.**
`rg '\bLAG\b'` and `rg TIME_CONSTANT_OF_TRANSMISSION` over `src/` and `rust/`
return nothing. HOPS exists only as `REFRACTORY_HOPS`, which is a different
quantity under a similar name — the refractory period divided by the hop, not
the connections a spike travels. The value coincides at 2; the meaning does
not. A.0 wants one home for every value in the register.

**A3. §7.4 — TARGET_ISI is derived, and the code's is the superseded one.**
The register fixes TARGET_ISI = 10.2 ms, **derived** as HOPS × (REFRACTORY +
LAG), and the clause says plainly that it "is not a knob of its own" and that
"what a sweep varies is therefore one of those three and not the target
itself". `constants.py` holds 5.1 as a free constant, and `rust/src/lib.rs:300`
carries the same stale default.

**A4. §7.4 — the shaping function's negative region never runs.** The file
gives

$$f = \frac{3u - 1}{1 + u^3}, \qquad u = \frac{t - R}{I - R},$$

anchored at $f(R) = -1$ at the refractory wall. All three engines compute
$u = t/I$ instead (`neuron.py:20`, `arrays.py:391`, `rust/src/lib.rs:103`) —
the first anchoring, which the file's own provenance note says put the $-1$ and
the zero crossing at $t = 0$ and $t = 1.70$ ms, "both inside the refractory
period, so no decision ever reached them and the factor was a positive discount
only. The wall anchor is what makes the negative region live." The three
engines agree with each other and all three disagree with the file. This is
precisely what the commit *"The shaping function re-anchored at the refractory
wall"* changed in the file and not in the code.

**A5. Appendix A — TAU is ∞, and the code leaks by default.** The register:
*"TAU | ∞ | the potential does not leak: the evidence accumulator. 2 ms is the
leak, kept as a per-run option with its analysis"*, quoting Byron: *"neuron:
NOT leaky, but please leave the leak option with its analysis."* `constants.py`
has `TAU = 2.0` and `--tau` defaults to it, so every run leaks and the
accumulator — the thing §1.2 and §8 are written on — is the branch a run has to
ask for.

**A6. §3.4 — TOLERANCE is a thousand times looser than specified.** The
register fixes $10^{-12}$; `clock.py:12` has `1e-9`, and
`rust/src/lib.rs:21` carries a **second literal** of the same quantity at the
same wrong value, which A.0 forbids on its own.

---

## B. Rules the specification does not carry

**B3. §9.1 — one rule pays at the read, and it is not the one that runs.** The
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

**B4. §8.3 — three of five eligibilities leave.** `hazard` and `hebb` survive;
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

**B5. §5.2 — complement coding only, *and there is no permutation*.** The
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

**C2. §9.9 and §9.10 — marked Non-default, and defaulted on.** *"the rule does
not run unless a run asks for it: zero switches it off, and that is where every
run this specification states leaves it."* The `Teacher` defaults
`homeostasis=HOMEOSTASIS` and `unstick=UNSTICK`, and so do `--homeostasis` and
`--unstick`, so every run that does not explicitly pass 0 drifts thresholds by
a rule the file says is off. mnist passes 0; nothing else does.

> **This is D5, and the code is wrong under either reading of it.** Byron
> answered *"The constants become 0"*. The clauses as written say the opposite
> in as many words — *"The constant keeps its value and the run switches it
> off, rather than the constant being zero and a run supplying the value."*
> The two readings differ about where the default lives; they agree that the
> effective default is **off**, which is not what the code does. §9.9, §9.10
> and §10.1 need Byron's edit before the code can follow either way.

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
| TARGET_ISI | 10.2 ms, derived | 5.1, free |
| LAG | 0.1 ms | absent |
| HOPS | 2 | absent (REFRACTORY_HOPS is another quantity) |
| TIME_CONSTANT_OF_TRANSMISSION | 5.1 ms, derived | absent |
| EARLY_ARRIVAL_PUNISHMENT_FACTOR | −1 | a bare literal in three engines |
| ELIGIBILITY | hazard | "perturb" |

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
