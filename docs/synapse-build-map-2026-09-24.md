# Synapse build map, September 24, 2026

Untracked working notes, not the specification. A read-only survey (nine readers and a critic) of how the three engines, the checkpoints, the CLI and the driver meet the synapse clauses of AUTHORITY.md at 1e3460f on spec/synapse-exploration. Section 5 lists the questions for Byron; nothing here is a decision until it is clause text. His answers of September 25, 2026 are at the end, and are written into AUTHORITY.md.


Scope and state, checked at 1e3460f on `spec/synapse-exploration`:
- The three synapse commits change only `AUTHORITY.md` and `docs/synapse-engine-plan-2026-09-23.md`. `src/`, `rust/`, `tests/` and `docs/rust-sweep.py` are the same as on the trunk (ce9cce3).
- None of the mechanism exists in code. There are no constants, flags, setters, fields or refusals for it.
- `sweeps/checkpoint-every` (#21) is not in HEAD. It changes three files this build also changes: `docs/rust-sweep.py` (+96/−28), `src/walnutbutter/fast.py` (+23) and `tests/test_resume_exact_rust.py` (+168). It also carries `BIBLIOGRAPHY.md`, `references/README.md` and `docs/mnist-rebaseline-2026-09-21.md`, which this build does not touch.

Where the readings disagreed, I re-opened the files:
- **Charged-drive arithmetic, by IEEE check.** With `θ = 0.2*(d/18.0)` (network.py:203-204), three sequential additions of `θ/3.0` fall short of θ for 23 of d = 1..399 (35, 70, 81, 85, 140, …) and overshoot for 31. With the multiply form `θ*(1/3.0)`, 127 fall short, including d = 35 and 37. Over 1..2000 the division form falls short for 94. θ = 0.2 is exact.
- `tests/test_resume_exact_rust.py:13-18` has four cases, not five.
- Rust line numbers: `external` is lib.rs:506, `hazard_draws` is lib.rs:846-858, and the stream check in `set_deltas` is lib.rs:380-384.
- The mnist goo is 654 neurons (395 + 199 + 60, problems.py:89). "644" is stale in three places:
  - goo.py:28, the docstring;
  - problems.py:85, mnist's description;
  - docs/mnist-evidence-diag.py:23, which passes `goo 644.0` through `grid_of`. `grid_of` never runs cli.py:710's size check, so that driver builds a 644-neuron goo.
- Inactive edges are counted differently: `fast.flatten` includes them (fast.py:55-61), while the array engine's `_out_degree` counts only active ones (arrays.py:174).

---

## 1. What the file now requires

Each clause below lists its touchpoints per engine: **O** is objects, **R** is Rust plus fast.py, **A** is arrays, **P** is persistence, CLI and driver.

### 1.1 Selecting the mechanism, and its settings (§6.13, §7.1, §7.6, §7.7, §8.17, 5.4b, App. A, A.0)

The file requires:
- Five register values: SYNAPSE_HAZARD_REST 0.01, SYNAPSE_HAZARD_FAMILY loglinear, SYNAPSE_HAZARD_SCALING count, TRACE all, DRIVE_STEPS 3.
- A run may explore at the synapse and may name linear, fan-out, ventured or charged.
- A neuron width and a synapse hazard asked for together are refused, citing §6.13.
- Unknown names are refused (§12.2).
- The file names no on/off switch (Q1).

Touchpoints:
- **constants.py.** Add the five rows beside ESCAPE_DELTA (constants.py:93-100) and INPUT_DRIVE (:103). The ESCAPE_DELTA comment at :98-100 becomes wrong at the flip.
- **O.**
  - Add mode and setting attributes beside network.py:90-92.
  - Add a setter beside `set_delta` (network.py:156-175).
    - It refuses when the network already has a width (`escape_delta > 0`), citing §6.13. In the library a width is always an explicit request, because only `set_delta` sets one (constants.py:99-100: "a network built in the library is deterministic until Network.set_delta"). So this is exactly §6.13's "asked for together", as long as the CLI and the drivers never apply the default width under the mechanism (P below, step 5).
    - Otherwise it sets every width to 0, sets κ_i and sets `traced = True` on every neuron.
    - It refuses unless the Rust edge layout is the identity (1.4 R) when a Rust engine is built from it.
  - `set_delta(>0)` refuses while the mechanism is on.
  - `hazard` (network.py:124-126) keeps its meaning, "a neuron width is positive".
  - Add an "explores" property (width > 0 or mechanism on). These all key on it: explorer (network.py:523-527), the `traced` property (network.py:128-131), learning.py:457, :526, fast.py:383 and arrays.py:534.
- **R.**
  - New fields beside lib.rs:173-179: mode flag, h0, family, trace mode, κ_i, output list, gain, read counts, and preallocated E and O draw buffers.
  - A setter after `set_deltas` (lib.rs:375-389). It refuses a positive delta alongside the mechanism, requires the stream, and refuses unless `out_edges` is the identity (1.4 R).
  - A `traced()` getter, for the test in §4 (Rust has none today; `traced` is only the field at lib.rs:179).
- **A.** Copy the settings in `__init__` at arrays.py:105-117. Settings are copied once (arrays.py:83, :105) and remain ordinary writable attributes, so every refusal is checked again at `fire_input`/`_run` (1.10).
- **P.**
  - New CLI flags. Do not spell TRACE `--trace`: that name is taken by the CSV path at cli.py:514-519, so `--trace ventured` today silently writes a file called `ventured`.
  - Add `charged` to `--drive` (cli.py:298-305).
  - `--delta` becomes a `None` sentinel in the same step as the new flags (step 5), not at the flip:
    - Record whether it was given, then resolve None to ESCAPE_DELTA under the neuron rule and to no width under the mechanism.
    - Do this at the three sites that apply a width today: cli.py:757-758, the `--seeds` worker (cli.py:939, fed from the job dict at :1035), and `grid_of` (docs/rust-sweep.py:166).
    - An explicit `--delta > 0` together with the mechanism is §6.13's refusal.
    - Three readers of `args.delta` would break on None and change in the same step: cli.py:998-999 (`args.delta > 0`), and the banners at cli.py:808-809 and :1054 (`{args.delta:g}`).
  - rust-sweep KNOBS (docs/rust-sweep.py:58-78) gains the h0 knob. Family, scaling, trace and drive become string knobs, handled the way eligibility is (:97-103).

### 1.2 The neuron: the comparison, no width, posts nothing (§6.13, §6.9, §6.8, §8.6)

Every neuron fires iff V ≥ θ and it is not refractory, at every wave, in §3.6's order. Its decision posts nothing, and p̂_j and n stay untouched.

- **O.** Already true once `delta == 0` and `centred` is False: neuron.py:248-250 takes `can_fire` with m = 0, and :269-270 returns without posting.
- **R.** Same, at lib.rs:870-872 and the `m > 0.0` guard at :896 (the early return at :904-906).
- **A.** The deterministic branch at arrays.py:297-298. `self.hazard` is False, so no neuron draws are taken (arrays.py:271-272).
- No new decision code. The guarantee comes from the setter (1.1).

### 1.3 The draws (§3.8, §7.3, §7.9, §12.6, §12.7)

Each wave:
- takes E uniforms in edge order (flatten order, and so push order), then O uniforms in output order;
- takes them after the floor and before any firing, all of them, whatever fires;
- takes them only from the exploration stream. Rust takes the MT19937 state and hands it back.

Touchpoints:
- **O.**
  - `_explore` (network.py:517-521) takes E + O draws into a per-wave buffer. The buffer is not on `Connection`, because §1.5 adds no per-synapse state.
  - `explorer` (network.py:523-527) returns that hook under the mechanism. Its call sites stay: propagation.py:188-189 inside `Schedule.run`, which `fire_input` (network.py:458-459) and `Network.propagate` (network.py:498-499) both reach.
  - `hazard_draws(rng, count)` (exploration.py:36-38) is reused.
  - `run_epoch` falls back to the global `random` (monitor.py:44). Under the mechanism it must refuse instead (§7.3, §12.7).
- **R.**
  - Gate lib.rs:596-598 on the mode.
  - `synapse_draws` follows the pattern of lib.rs:846-858, filling a preallocated buffer indexed by edge id (1.4 R). lib.rs:848 today allocates a new Vec every wave. lib.rs:855 returns silently with no stream, so the setter's stream check is the only guard.
  - Rust needs the output order through a new setter. Today only fast.py knows `out` (fast.py:224, :402).
  - Stream handover: extend the condition at fast.py:81-84 and Rust's check at lib.rs:380-384 to cover the mechanism.
- **A.**
  - arrays.py:271-272 takes E + O draws.
    - It draws from `self.explore_rng` unchecked. That value is copied from `mesh.explore_rng` at arrays.py:84, and it is None on any network that `run_epoch` has not touched (network.py:90; `run_epoch` sets it at monitor.py:44).
    - A direct `ArrayNetwork.fire_input()` then fails with an AttributeError inside `hazard_draws`. Refuse a missing stream at the draw site, in `_run`.
  - The edge vectors are in connection-id order (arrays.py:130-135). Assert that this equals flatten order (it holds because goo wires source-major, goo.py:291-304), or permute.
  - Output order is `output_index` (arrays.py:147).

### 1.4 The synapse decision (§7.5, §7.6, §7.7, §7.8, §6.5)

After the fire phase, every source i that did not spike this wave decides, refractory or not:
- u_i = clip(V_i(t), 0, θ_i)/θ_i, with V decayed to t under the leak.
- Δt ≥ 0.
- m_i = (Δt/hop)·κ_i·h(u_i), capped at 1e3.
- P = −expm1(−m_i).
- Each synapse, and an output's read synapse, escapes iff its uniform is strictly below P.
- An escape leaves V_i, t_fired, the spike count, the fired flag and the rate trace unchanged.

Touchpoints:
- **O.**
  - Add a post-fire phase between propagation.py:200 and :201. `Schedule.run` has only a pre-fire hook today.
  - Pass it at both call sites of `Schedule.run` that have a network: `fire_input` (network.py:458-459) and `Network.propagate` (network.py:498-499). A hook passed only from `fire_input` would let a cascade through `Network.propagate` take the E + O draws and make no synapse decision at all.
  - Arithmetic templates: neuron.py:226-230, :253 and :266.
  - "Spiked this wave" is `fired_in_wave == wave.number` (set in `fire`, neuron.py:283-290).
  - The module-level `propagate()` (propagation.py:213-235) has no network, so it must refuse the mechanism.
- **R.**
  - New pass after lib.rs:616-620, before the quash at :622-624.
  - Skip sources with `fired_wave[i] == wave_no` (lib.rs:992).
  - A source's synapses are `out_edges[k]` for k in `out_start[i]..out_start[i+1]` (lib.rs:147). Those are positions in `out_edges`, and the entries are edge ids, used as `self.out_edges[k]` in `fire` (lib.rs:1002-1004). Index the E draws by the edge id `out_edges[k]`, not by k.
    - Positions equal edge ids only because `fast.flatten` is source-major (fast.py:56-61) and `bucket` is stable (lib.rs:1049-1058). `Engine::new` (lib.rs:237-240) never checks this.
    - §3.8's draw layout and Q11's push order both assume the identity, so the mechanism's setter refuses unless `out_edges` is the identity.
  - V comes from `potential_at` (lib.rs:797-804).
- **A.**
  - New block after `fired` is formed (arrays.py:324). Its mask is `~fired`, not `deciding` (arrays.py:284).
  - Per edge: `U[:E] < P[source]`. Then a_i = bincount of the escaped sources, plus the read escapes at `output_index`.

### 1.5 One exposure clock per source (§7.5, §7.8, §2.5, §6.11)

- After each wave's synapse pass, `exposed_since_i = t` for every source that did not spike.
- At a spike, `exposed_since_i = t` rather than t + REFRACTORY.
- **Change, all three engines:** neuron.py:301, lib.rs:980 and arrays.py:337 each gain a mode branch. The neuron-rule updates (neuron.py:254, lib.rs:880, arrays.py:296) stay.
- **Pin across engines:** the clock is updated for sources with F_i = 0 too. Nothing reads it there, but checkpoints carry it. It is compared across engines (1.13, step 4 gate), since nothing compares it today: fast.compare's checks at fast.py:255-283 omit it, and the only test that reads it is the objects round trip at test_hazard.py:208.
- **P.** No new key: persistence.py:91 and :333-334, fast.py:105, rust-sweep.py:203-204. What the key means now depends on the mode.

### 1.6 The ventured signal (§7.9, §1.7, §8.13, §3.5, §3.7, §3.10)

An escape at t schedules a signal for t + HOP that carries a ventured mark. The signal:
- delivers w as read at delivery;
- is dropped at a refractory target but still recorded as delivered;
- is floored with the wave's total;
- moves the stamp and, per TRACE, the trace;
- at or after the horizon, waits and keeps its mark.

Touchpoints:
- **O.**
  - Push through `Schedule.signal` (propagation.py:104-106) with the bit inside the payload. The kind must stay SIGNAL (propagation.py:45): the heap orders by (time, kind, seq), so a new kind would sum ventured signals after relayed ones.
  - `Wave.delivered` stays a list of Connections, because learning.py:437-439 and `Wave.__eq__` (:84-88) read it. Add a parallel list of bits.
  - `Signal` gains a `ventured` slot (:55).
  - `pending()` (:119-121) returns the bit.
  - Delivery code is at :163-178, with the weight read at :167 and the stamp at :168.
- **R.**
  - `Event` (lib.rs:110-114) gains `ventured: bool`, kept out of `Ord` (lib.rs:120-128).
  - Delivery is at lib.rs:552-578.
- **A.** Add a per-edge arrival schedule beside the per-source rows (arrays.py:184-200). Delivery gathers the ventured edges into `incoming`/`touched`, `last_signal` and `delivered_wave` (arrays.py:250-260).

### 1.7 The read synapse (§7.9, §5.10, §11.8, §2.1, §3.9)

- Every output-zone neuron has one read synapse. It is not a Connection: it has no id, and it stays out of `network.connections`, `flatten`, the edge arrays and `wiring_digest` (persistence.py:147-163).
- It decides on the output's own u, m and clock, with the uniform at index E + k.
- An escape adds 1 to the output's read count for the epoch. Nothing is scheduled.
- It counts in F_i and a_i.
- The count read is spikes plus the read count, everywhere a count is read. The read count is zeroed at the epoch's reset.

Touchpoints:
- **O.**
  - A per-output counter beside `spikes_at_reset` (neuron.py:68), zeroed in `Neuron.reset` (:314-330).
  - Add it in `output_counts` (network.py:228-230). That also reaches the count-read pickiness line (:242-243) and `class_sums`/`evidence_score` (learning.py:171-173, :220).
  - Add it in `output_counts_hz` (network.py:246-249), which reads `epoch_spikes` directly.
- **R.**
  - `read_count: Vec<u64>`, zeroed in `reset` (lib.rs:330-353), with a getter and a setter.
  - Add it wherever counts are read: fast.py:122 (`_outputs_on`), fast.py:134 (`_reward`), docs/rust-sweep.py:275 (`_output_counts`), docs/mnist-watch.py:111, docs/mnist-evidence-diag.py:30 and docs/mnist-evidence-gradient.py:41.
- **A.** An O-vector, used in `output_counts` (arrays.py:478-480), `output_fired`'s count branch (:501-502) and `output_counts_hz` (:505-507), and zeroed in `reset` (:418-431).
- **Unchanged pending Q7:** the rate memory and stuck counts read spikes only (learning.py:340-351, fast.py:169, arrays.py:546-549).

### 1.8 The rule at the synapses' decisions, and the gain (§8.16, §8.11, §1.8, §1.9, §8.12, §8.1, §8.2)

The rule:
- G_j replaces E_j and the credit.
- At each wave, for each deciding source with m > 0:
  - c = m e^−m/(1−e^−m);
  - entry = a·c − (F−a)·m, times ρ(u) under linear (Q3, Q4).
- Under the accumulator, G += entry after this wave's arrivals have noted.
- An arrival does x += 1 (per TRACE) and B += G, where G is the gain before this wave's entries.
- The spike, the floor, a forced spike and a discharge each do e += xG − B, clear x and B, and set G = 0.
- The read does e += xG − B, then B = xG, leaving x and G as they are. The epoch reset sets B = xG.
- Under the leak there is no G: a fan-in walk runs once a wave per deciding source.
- The pay (§8.1, §8.2) is unchanged.

Touchpoints:
- **O.**
  - Add G to `Neuron` (beside neuron.py:53-55). The posting in `decide` (:264-280) is not used under the mechanism.
  - `settle_arrivals` (:162-176) takes the G form and restarts G.
  - `clear_arrivals` (:178-188) is called from the floor (:150-160) and from a discharge (:323-326), and now restarts G, per neuron and unconditionally. E is not restarted there today.
  - `fire` settles and restarts G (:302-309).
  - The delivery note at propagation.py:172.
  - `Network.settle_scores` (network.py:140-154): the sign flips and B = xG.
  - `Network.reset` re-base (:509-514).
  - The `traced` guards at network.py:148 and :509 must be true under the mechanism, through the property at :128-131.
- **R.**
  - `gain` replaces the use of `expected`/`credit` (lib.rs:183-185).
  - The note at lib.rs:566; `settle_arrivals` (lib.rs:820-835); `clear_arrivals` (lib.rs:837-843, per neuron); `fire` (lib.rs:981-990).
  - `settle_scores` (lib.rs:441-453): opposite sign.
  - `reset` re-base (lib.rs:343-349).
  - The leak walk's template is at lib.rs:913-922.
  - `reinforce_scores` (lib.rs:468-493) is unchanged.
- **A.**
  - `expected` (arrays.py:116) becomes G.
  - The note at arrays.py:264.
  - Restart G on neuron masks, not through `_close_arrivals`.
    - `_close_arrivals` (:352-364) works on an edge mask and touches only edges with an open arrival (:356). A floored or discharged neuron with no open arrival (arrays.py:268-270, :420-422) would keep its old G, while objects and Rust restart G per neuron.
    - The offset cancels in xG − B in exact arithmetic but not in floating point.
    - Restart G on `clipped` at the floor, on `idx` at a spike (in place of :340's zeroing of E), and on every neuron at a discharge.
  - `settle_scores` (:366-374) and the reset re-base (:425-426).
  - The `credit_v`/q block (:299-323) is replaced. The leak walk's template is :320-323.
- **Pay.** learning.py:442-475, lib.rs:468-493 and arrays.py:532-544 change only the refusal key (1.10).

### 1.9 TRACE (§8.17, §1.6, §8.5)

- **all:** every integrated arrival raises x.
- **ventured:** only ventured arrivals raise x. V and the stamp take every delivery. B moves only where x moves (Q5(i)).
  - Under TAU finite, the file does not say whether a relayed arrival decays x_ij to now and moves `trace_at` (adding 0), or leaves both alone (Q5(ii)). Today every integrated arrival decays x and moves `trace_at`: propagation.py:174-175, lib.rs:568-571 and arrays.py:266-267.
  - The two readings are equal in exact arithmetic but not in floating point. The leak walk also multiplies x·exp(−(t − trace_at)/τ) (lib.rs:919, arrays.py:323).
  - Under the accumulator `trace_at` is moved too (propagation.py:175, lib.rs:571) and checkpointed ("traces"), though no arithmetic reads it. Pin it in both TAU.
- A run under ventured says in its record that the estimator is biased.
- **Touchpoints:** O propagation.py:169-175; R lib.rs:559-572; A arrays.py:261-267. The record: the checkpoint, the arm json (docs/rust-sweep.py:347-370), the banner (cli.py:807-813) and the seeds header (cli.py:1054).

### 1.10 Eligibility and refusals (§8.3, §6.13, §7.3, §12.2)

- **Reinforce is allowed under the mechanism.**
  - Re-key learning.py:457-458 and :526-527, fast.py:383-386 and arrays.py:534-535. Keep the prefix "refuses to learn", which test_hazard.py:122-133 matches.
  - Re-key or delete Rust's `reinforce_hazard` (lib.rs:456-463), which refuses unless `self.hazard`. It has no caller in src/, docs/ or tests/, so it is stale rather than live, but it would refuse a mechanism run.
- **Hebb is refused under the mechanism** in:
  - the Teacher (learning.py:545 calls `centre`);
  - fast.train (fast.py:381-382, :395);
  - fast.train's keyword default `eligibility="hebb"` (fast.py:321);
  - rust-sweep's default (docs/rust-sweep.py:98) and `grid_of`'s own keyword default `"hebb"` (docs/rust-sweep.py:119).
- **`traced` must survive every recompute:**
  - the `Network.traced` property (network.py:128-131), which the array engine inherits (arrays.py:113);
  - `Network.centre` (network.py:133-138) and `set_delta` (:175);
  - restore (persistence.py:328);
  - `arrays.centre` (arrays.py:376-381);
  - Rust `set_deltas`/`set_centred` (lib.rs:387, :405).

  Otherwise constructing a Teacher silently turns every trace off. The test in §4 checks all of them, through the new Rust `traced()` getter.
- **No stream is refused** at network.py:525-526, monitor.py:44, fast.py:81-83, lib.rs:380-384, and in the array engine at the draw site in `_run` (arrays.py:271-272; 1.3 A).
- **The array engine's refusals** (the mechanism, charged, linear/fan-out/ventured, a missing stream) are checked in `fire_input`/`_run` as well as in `__init__`. Its settings are copied once (arrays.py:83, :105) and stay writable, and `run_epoch` itself assigns `explore_rng` after wrapping (monitor.py:44).

### 1.11 The charged drive (5.4b, §5.4, §5.7, §5.8, §3.4, §2.5)

- **Draw.**
  - `input_schedule` (network.py:397-440) draws at rate × DRIVE_STEPS from `network._rng`, in place order: 0.4/ms for bit 1, nothing for bit 0.
  - Refuse unknown drive names. Today any drive other than `rate` silently becomes one spike at t_e (network.py:425-426), which §5.7 forbids.
- **Deliver.** Each arrival is an EXTERNAL event of θ_i/DRIVE_STEPS (Q10). It:
  - is dropped while the input is refractory;
  - touches the input;
  - anchors the wave (§3.4);
  - moves no trace, note, stamp or score;
  - sets the driven mark at the first delivery of the epoch (Q8, Q9);
  - forces no spike.
- **O.**
  - `Schedule.external` and its delivery (propagation.py:112-114, :181-185; `receive` at neuron.py:134-148) already do most of this.
  - The mark: today only a fired STIMULUS sets `forced` (propagation.py:190-193).
  - `fire_input` (network.py:455-456) schedules charges, not stimuli.
- **R.**
  - Add `charge_many` beside `stimulate_many` (lib.rs:361-369).
  - Handle EXTERNAL at lib.rs:580, which today silently ignores it.
  - Anchoring already exists (lib.rs:532-534).
  - fast.train (fast.py:445) and compare (fast.py:237-238) push charges.
- **A.**
  - Keep per-index charge lists on the pattern of `_stimulate_at` (arrays.py:206-219), added one delivery at a time before the matrix sum.
  - Anchor the wave on the external's time. Today arrays.py:241-246 takes the last merged stimulus key, while the objects take the earliest non-SIGNAL event (propagation.py:151-158).
  - `ArrayNetwork.fire_input` inherits `input_schedule` and hands every event to `_stimulate_at` (arrays.py:458-460). Until the charge path exists, `fire_input` refuses `drive == "charged"` itself, or step 2's `input_schedule` would reach arrays as forced spikes.
- **P.** `--drive charged` (cli.py:298-305). The code's `forced` (constants.py:103-106) is not the file's "forced drive"; the file's forced drive is the code's `rate`.

### 1.12 Checkpoints and resume (§8.14, §12.9, §12.10, §12.11)

A checkpoint must carry:
- G per neuron;
- the read count per output;
- the ventured mark on each pending signal;
- the mode, h0, family, scaling and TRACE (and DRIVE_STEPS, Q16).

κ_i is recomputed at build and at resume (fan-out included) and never stored. A resume is exact, including signals in flight carrying a ventured mark. A network saved under a setting resumes under it unless the resuming run overrides it explicitly (§12.9).

- **O.**
  - Add keys in `checkpoint` (persistence.py:31-102).
  - `_restore_clock` (persistence.py:299-360):
    - `traced` from the mode (:328);
    - κ (:329-332);
    - 3-element pending (:345-347).
  - A missing key reads as the neuron rule, never as SYNAPSE_HAZARD_REST.
  - **Format.**
    - `read_checkpoint` refuses anything whose format is not FORMAT (persistence.py:183-189), so a bare bump to 3 would refuse every format-2 file on disk.
    - That is 796 files: 755 sweep network checkpoints under runs/<sweep>/ (24 sweeps, among them the 37 carrying `engine_pending`), 39 under runs/archive/, and 2 at the top of runs/.
    - Make `read_checkpoint` accept {2, 3}, and write format 3 only when the mechanism is on. Neuron-rule files stay format 2, so trunk code (FORMAT = 2) still reads them and refuses a synapse file.
  - Refuse a switch of mode at resume (Q17).
  - **Events waiting past the horizon.** An object checkpoint loses stimuli and charges that wait past the horizon:
    - `Schedule.pending()` keeps only SIGNAL events (propagation.py:119-121), and restore pushes signals only (persistence.py:345-347).
    - Rust's `pending_events` keeps every kind (lib.rs:724-728), which `_save_network` stores as `engine_pending` (rust-sweep.py:218).
    - `input_schedule` keeps draws with `when < end` (network.py:436; `end` is the horizon when the presentation window is the whole epoch, network.py:388-389, the default). `Schedule.run` runs only events `before(t, until)`, that is t < until − slack (clock.py:19-21).
    - So a stimulus or charge drawn in [until − slack, until) waits in the heap and is lost at an object checkpoint.
    - This is pre-existing for the rate drive's stimuli; charged adds EXTERNAL.
    - Carry STIMULUS and EXTERNAL events (with kind, and the charge's amount rule per Q10) in the objects' pending list and restore them, or refuse an object checkpoint while one is pending.
  - **§8.14's score.** §8.14 says "the score since the last read is carried too … so a checkpoint is not confined to a read". `checkpoint` (persistence.py:31-102) carries no score, and no `spikes_at_reset`, `has_fired`/`fired_in_wave` or `forced` either. This is a pre-existing gap, and no entry in docs/conformance-checks.md cites §8.14.
    - Add the score per §8.14.
    - For an exact mid-epoch object checkpoint, also add `spikes_at_reset`, `has_fired`/`fired_in_wave` and `forced` beside the read count.
    - Until then, checkpoints are taken at epoch boundaries only, and the tests say so.
- **R.**
  - Add `set_gains`/`gains` and `set_read_counts`/`read_counts`.
  - `pending_events` returns 4-tuples (lib.rs:724-728; note its `unwrap` on NaN at :726).
  - `push_events` takes the marks and refuses unknown kinds (lib.rs:729-737).
  - fast.build accepts 3- and 4-tuples (fast.py:107-108).
- **Driver.**
  - `_save_network` (docs/rust-sweep.py:170-220) adds gains, read counts, marks and settings.
  - The synapse settings must not join `RESUMED_SETTINGS` (:223-225), which writes the fresh arm's values over the checkpoint's (:250-251).
  - `drive` is already in `RESUMED_SETTINGS` (:223), so a charged checkpoint resumed by an arm that names no drive comes back on the rate drive without a word. Drop `drive` from `RESUMED_SETTINGS`, and apply it only when the arm (as a knob) or the sweep's fixed words name it. Existing arms all resume on rate either way, so no existing bit changes.
- **A.**
  - `__init__` sends ventured pending signals to the per-edge schedule (arrays.py:150-154).
  - `sync_to_mesh` pushes a single ventured edge. Today :602-606 expands every pending source into all of its active edges.
  - Sync G and the read counts.
- **CLI.**
  - Do not extend the overwrite-on-load pattern (cli.py:754-773) to the synapse settings. Rewrite cli.py:757-758 in step 5 (1.1 P).
  - The drive is already overwritten there. cli.py:768 sets `grid.drive = args.drive` on every run, loaded ones included, and `args.drive` has been resolved to the problem's or INPUT_DRIVE by then (cli.py:610-611). Record whether `--drive` was given before that resolution, and set `grid.drive` on a loaded network only when it was.

### 1.13 Agreement (§8.15, §12.4, §12.8)

- **fast.compare (fast.py:194-289).**
  - Compare G in place of `expected` (:268).
  - Compare the read counts.
  - Compare `exposed_since` (`==`, via `engine.exposed_since()`). Today :255-283 compare spikes, rewards, rates, thresholds, scores, notes, expected counts, expectations, decisions and weights, but not `exposed_since`, potentials or `trace_at`. Ventured signals and charges move the last two, and the getters exist (`engine.potentials()`, `engine.trace_ats()`, used at rust-sweep.py:191, :209), so compare them too.
  - Check the stream's end state under the mechanism; today it is checked only `if network.hazard` (:286).
  - Replay charges.
  - Refuse discharge, which is never mirrored today.
- **The matrix:** {rate, charged} × {all, ventured} × {loglinear, linear} × {count, fan-out} × TAU {inf, 2}, on goo 60 copy and goo 455 with the mnist read. The TAU 2 × ventured cells carry Q5(ii).

---

## 2. Where the September 23 plan is superseded

| Plan says | The file says | The build does |
|---|---|---|
| §0: synapses decide "lazily with each incoming wave" | §7.5: at every wave, every non-spiking source | Walk every neuron after the fire phase, touched or not |
| §0, §1.2, §2.1: exposure runs from the end of the refractory period | §7.5, §7.8, §2.5, §6.11: from the spike, never suspended | `exposed_since = t` at the spike (neuron.py:301, lib.rs:980, arrays.py:337) |
| §1.2, §1.5: a refractory source's synapses are suspended; §3.1 tests "accrue no exposure" | §7.8: they whisper at u = 0 | Mask is "did not spike", not `~refractory` (arrays.py:284). Invert the test. The plan's §6 toy numbers are not predictions |
| §1.4: κ(N) now, fan-out later (§5.5) | §7.7: both, a run option; F_i includes the read synapse (§7.9) | A κ_i vector in every engine, recomputed at build and resume; a flag; persisted; fan-out arms in the agreement tests |
| §1.3, §2.4: SYNAPSE_HAZARD_REST = 0, and "0 turns it off" | App. A: 0.01; §7.6: the default once built; h0 = 0 under loglinear is deterministic, and linear h0 = 0 still explores | Constant 0.01; a separate mode selector (Q1); the flip is the last step (Q2) |
| §1.3: the family is open; §2.5: "lacks a family" is a lasting refusal | §7.6: every engine carries both | Linear is built everywhere; the refusal is only transitional (§12.2) |
| §1.7, §1.12: the entry is the same for any family; the 1/Δ fold "falls away" | §8.16, §8.7: loglinear folds −ln h0 and 1/θ; linear multiplies by ρ(u) | ρ in the linear entry and in G (Q3) |
| §1.11: the read synapse's place in the rule is unsaid | §8.16, §7.9: in F_i and a_i | For an output, F = out-degree + 1; one G addition per wave |
| §1.8: the read "clears x and B and restarts G" (with a parenthetical) | §8.16, §1.9: the read posts xG − B and re-bases B = xG; x and G stay | A separate read path, not the settle path |
| §2.1: "one Bernoulli per outgoing active synapse" | §3.8: one uniform per synapse, whether usable or not | Draw for every edge in flatten order, indexed by edge id (Q12 covers inactive edges) |
| §1.13, §2.2: draws per wave = the edge count | §3.8, §12.7: E + O | Buffers sized E + O |
| §1.13: settings are h0, family, trace, drive, DRIVE_STEPS | §8.14, §12.9: the settings of §7.6, §7.7, §8.17, so the scaling too | Persist the scaling. DRIVE_STEPS is a register constant, not a named run option: no `--drive-steps` (Q16) |
| §2.1: class_sums and evidence_score read spikes + count | §5.10, §11.8: every count reader | All the readers listed in 1.7 |
| §2.1: mark "at its first arrival"; §2.2 and §2.3 omit charged | 5.4b: "first delivery of the epoch"; §12.2 | Rust and arrays carry charged, or refuse it until they do |
| §2.1: `set_synapse_hazard` refuses any positive `escape_delta` | §6.13 refuses only an explicit request for both | In the library a width is only ever explicit, so the setter refuses `escape_delta > 0`. The CLI and drivers stop applying the default width under the mechanism (step 5). Key `traced`, the draw hook and the reinforce gate on the mode |
| §2.4: `--trace` for TRACE | — (the name collides with cli.py:514-519) | A distinct flag name |
| §3.2: {forced, charged} × {all, ventured} | §7.6, §7.7, §12.8 | Add {loglinear, linear} × {count, fan-out} and TAU |
| §3.4: "five cases" | tests/test_resume_exact_rust.py:13-18 has four | Four × the mechanism, plus a ventured-in-flight case |
| §4: phase 0 is Byron writing the clauses; branch off code_follows_the_file; hold off while the mnist sweeps run | Clauses done (859c357, 1e3460f); the trunk lacks them (14 commits behind) | Branch off `spec/synapse-exploration` in a worktree (protects the Obsidian checkout); Byron's answers land as clause text before step 1; add a final phase for the flip and the marker edits (Byron's) |
| §2.2 cost: 39 waves/epoch, 54k uniforms, "under 15 per cent" | docs/waves-per-hop-10k.md:44-48: about 1,500 waves/epoch on goo 455 | The plan is about 40× low; see §6 |
| §5.3: LR search on the charged drive | 5.4b: rate (Poisson forced) stays the default; §8.16 re-finds LR under the default | Search on the rate drive, charged as an arm; grid may need rates below 0.0005 |
| §5.5 baselines 0.2577 / 0.2875 | (recon, runs/) leak ten-seed 0.2704; accumulator 0.3143 at lr 0.0005 | Use the current baselines |
| §1.12: §0.11 "stops being a direction"; §7's note goes | §0.11 is "specified, being built" and keeps its text; the sparse matrix stays a direction; §7's note says deprecating θ is not taken up | The marker edits are Byron's, in the last phase |

---

## 3. Build order

### Step 0: branch, baseline, and the answers
- Make a worktree branch off `spec/synapse-exploration` (proposed name `synapse/engines`). Use the isolated wheel and PYTHONPATH recipe so the main checkout's venv and `.so` stay untouched.
- Decide on #21 first: merge it into the trunk and rebase, or build on `sweeps/checkpoint-every`. It changes rust-sweep.py, fast.py and test_resume_exact_rust.py.
- **Byron answers before step 1, not step 2.** Q1 decides the API step 1 builds (the mode setter, the "explores" property, the refusals keyed on the mode), and step 1 writes every §4 test, which encodes the answers. Each answer lands as clause text in AUTHORITY.md, Byron's edit, before the code that depends on it.
  - **Answers that change the bits:** Q1, Q3, Q4, Q5 (both parts), Q6, Q8, Q9, Q10, Q11 and Q12, with Q15's forms accepted.
  - **Q4(b) needs a clause edit.** It contradicts §8.16 as written: "At every wave, for every source i whose F_i synapses decided … the wave posts to every synapse k→i", and the bookkeeping's "at each wave G_j += …". Both would gain the condition 0 < V_i < θ_i, and "the entry's mean at every wave is zero" would read as of the waves it posts at.
  - **Refusals only:** Q7, Q13, Q14, Q17 and Q18 are built as interim refusals in step 1 whatever the answer, so they do not block.
  - **Before step 5:** Q2 (what an explicit `--delta` means, which the step-5 sentinel implements) and Q16 (whether DRIVE_STEPS gets a flag).
- **Gate:** the existing suite is green locally with the wheel built.

### Step 1: tests, settings and refusals (no mechanism yet)
- Add the constants, the settings attributes, the mode setter, the "explores" property, and the refusals of the table below.
- Write every test of §4. Mark the engine-specific ones `xfail(strict=True)` so they flip to passing as each engine lands.
- Close the pre-existing silent paths:
  - an unknown drive name (network.py:425-426);
  - EXTERNAL through `push_events` (lib.rs:580);
  - `run_epoch` without an rng under the mechanism (monitor.py:44);
  - the array engine drawing from a None stream (arrays.py:271-272, an AttributeError today).
- **Gate:** the refusal tests pass; every existing test is unchanged and green (no default moves).

### Step 2: the object engine
1. The E + O draw hook (1.3).
2. Neuron changes: G, the settle forms, the exposure at the spike, the read settle and the reset re-base (1.5, 1.8).
3. The post-fire synapse pass (1.4, 1.6, 1.7), passed at both `Schedule.run` call sites (network.py:458-459 and :498-499). It covers escapes pushed as SIGNAL with the bit, read synapses, a_i, G or the leak walk, and the clock.
4. TRACE at delivery, with Q5's `trace_at` rule (1.9).
5. Read counts in every object reader (1.7).
6. The charged drive (1.11).
7. Linear and fan-out.
- **Gate:** `tests/test_synapse_hazard.py` (objects part), the refusal tests, and the full existing suite.

### Step 3: Rust and fast.py, to the bit
1. Fields and setters (1.1, 1.3): the out_edges identity check, the output order, and the `traced()` getter. Re-key or delete `reinforce_hazard` (lib.rs:456-463).
2. `run()`: charge delivery; synapse draws indexed by edge id; the synapse pass after lib.rs:616-620; the ventured push; the Event bit.
3. `fire`, `clear_arrivals`, `settle_scores`, `reset` (1.5, 1.8).
4. `pending_events`/`push_events` with marks (1.12).
5. fast.py:
   - `build`: stream condition, setters, 3- and 4-tuples.
   - `_outputs_on`, `_reward`: read counts.
   - `train`: refusal key, eligibility default, `charge_many`.
   - `compare`: G, counts, `exposed_since`, potentials, `trace_at`, stream, charges, refusals.
- **Gate (locally, since CI does not build Rust):**
  - `fast.compare(...) == []` over the 1.13 matrix on goo 24/60 copy and goo 455 mnist;
  - the handed-back stream equals Python's;
  - an engine-level heap round trip: `pending_events()` pushed into a fresh engine with `push_events` keeps delivery order and marks.
  - The exact resume through `test_resume_exact_rust` is **not** in this gate. It goes through `_save_network` → `persistence.checkpoint` (rust-sweep.py:217) and `resume_grid` → `restore` (:228-254), which refuse a synapse network until step 5. It is in step 5's gate.

### Step 4: the array engine, to tolerance
- Build:
  - settings;
  - E + O draws, with the no-stream refusal in `_run` (fast option: numpy MT19937 loaded from Python's state and handed back);
  - the per-edge escape pass;
  - the per-edge ventured schedule;
  - G, restarted on neuron masks (1.8 A);
  - read counts;
  - charges added one at a time;
  - the anchor fix;
  - the edge-order assertion;
  - the refusals repeated in `fire_input`/`_run` (1.10).
- **Gate:** against objects:
  - spikes and read counts `==`;
  - scores, weights and G `np.allclose(rtol=1e-9, atol=0)`;
  - thresholds, rates and `exposed_since` to 1e-12 absolute (§12.4);
  - the round trip through `__init__`/`sync_to_mesh` with ventured signals pending.

### Step 5: checkpoints, resume, CLI and driver
- **persistence:**
  - `read_checkpoint` accepts {2, 3}; format 3 is written only under the mechanism;
  - keys and fallbacks;
  - §8.14's score, plus the mid-epoch fields or a boundary-only rule (1.12);
  - pending STIMULUS/EXTERNAL events carried or refused;
  - the mode-switch refusal;
  - `traced` and κ_i on restore.
- **Driver:** `_save_network`, `resume_grid`, KNOBS, the arm json and `grid_of` (resolving the `--delta` sentinel at :166). Drop `drive` from `RESUMED_SETTINGS` and apply it only when the arm names it.
- The four drivers that read counts.
- **CLI:**
  - flags with `None` sentinels, `--delta` among them (moved here from step 7);
  - rewrite cli.py:757-758;
  - apply the drive on load only when `--drive` was given (cli.py:768);
  - rewrite cli.py:998-999 and the banners (:808-809, :1054);
  - the `--seeds` job dict and worker (cli.py:1023-1038, :939);
  - test_constants.py:30 changes here, since the parser's `--delta` default becomes None.
- **Gate:**
  - test_persistence: synapse round trip; format-2 fallback; format 3 written only under the mechanism; the drive kept on load; a charge pending at until − slack/2; exact resume.
  - test_resume_exact_rust, moved here from step 3: the four cases × the mechanism; ventured signals in flight at the cut; G open across the epoch boundary with the finished epoch's read count; 3-tuple back-compatibility.
  - test_fast: arm json and refusals.
  - CLI refusal tests.

### Step 6: compare on the configuration and measure (§12.8)
- Run the 1.13 matrix on the built wheel at the mnist operating point.
- Time an epoch before and after (neuron rule vs mechanism, goo 455 and goo 654).
- Smoke run: 3,000 epochs, 3 seeds, cut and continued.
- **Gate:** agreement holds, and the costs are recorded.

### Step 7: the default flip and the marker edits (Byron approves)
- The default mode becomes synapse. The `--delta` sentinel already exists from step 5. Update the tests listed in §5.
- Add a new section to docs/conformance-checks.md.
- Byron strikes the "being built" markers: §0.11, 5.4b, §6.13, §7.5, §8.16, the provenance lines of §7.6 and §7.9, and the Appendix A heading.

### Refusals each engine carries until it is built (§12.2)

Each message names the clause.

| Asked for | objects | Rust (fast.build/train/compare; lib.rs setters, `push_events`) | arrays (`ArrayNetwork.__init__`, and again in `fire_input`/`_run`) | persistence / CLI / driver |
|---|---|---|---|---|
| mechanism on (§7.5) | until step 2 | until step 3 | until step 4 | `checkpoint()` refuses a synapse network until step 5; the flags don't exist until step 5 |
| linear (§7.6), fan-out (§7.7), TRACE ventured (§8.17) | until step 2 | until step 3 | until step 4 | until step 5 |
| charged (5.4b) | until step 2 | until step 3 (and EXTERNAL in `push_events` from step 1) | until step 4 | until step 5 |
| ventured pending, G or read counts in a restored network (§12.9) | until step 2 | until step 3 | until step 4 | until step 5 |

Permanent refusals:
- a width and the mechanism asked for together (§6.13);
- hebb under the mechanism (§8.3);
- the mechanism with no exploration stream, in every engine: objects (including the `run_epoch` fallback), Rust, and arrays at the draw site (§7.3);
- an unknown family, scaling, trace or drive (§12.2).

Recommended interim refusals, pending Byron:
- θ_i ≤ 0 (Q13);
- an inactive connection (Q12);
- h0 outside [0, 1) (Q14);
- the fired, again, window and rate reads under the mechanism (Q7);
- `readout = "input"` under the mechanism (no clause; §5.1, §7.9 "every output neuron");
- `bored_after > 0` under the mechanism (no clause; u reads θ_i);
- charged under the neuron rule (Q18);
- a resume across modes (Q17);
- discharge on the fast path (pre-existing).

---

## 4. The tests to write

**tests/test_synapse_hazard.py (new; objects first, then parametrised over engines)**
- m and P at u ∈ {0, ½, 1−ε} for loglinear and linear, in the pinned arithmetic (Q15).
- Loglinear h0 = 0: spikes equal the Δ = 0 network's on the same seed and drive; nothing is posted.
- The stream advances by exactly (E+O)·waves `random()` calls whatever fires: its state equals a fresh `Random(seed)` stepped that many times.
- Draw layout: a stubbed stream of known values produces escapes in edge order, then output order.
- Under the mechanism every width is 0; after a run, every neuron's expectation is `None` and its decisions 0 (§8.6).
- The first decision after a spike at t_s covers Δt = t − t_s (§7.8).
- An F_i = 0 source's `exposed_since` moves to t at every wave.
- A refractory source's synapses escape at h0·κ_i per hop during the period (statistical).
- A neuron with V = 0 and no input never spikes; its synapses escape at h0·κ_i per hop (§7.2).
- An escape leaves V_i, t_fired, the spike count, `has_fired` and the rate trace unchanged.
- A ventured delivery adds w as read at delivery, moves the stamp, and raises x under TRACE all.
- A ventured signal to a refractory target is dropped, recorded as delivered, and leaves x and the stamp alone.
- Escapes are pushed after the wave's spike signals, in edge order (the next wave's `delivered` order).
- A ventured signal at the horizon waits, keeps its mark and is delivered next epoch.
- The entry has mean zero at fixed m over many draws, both families.
- Summed scores equal the per-wave sum of entry·x (tolerance), under the accumulator. The run goes through a mid-interval read, more arrivals, then the floor, a forced spike, a discharge and a spike.
- Under the leak, the once-a-wave walk equals the direct per-wave sum.
- Under linear, ρ(u) multiplies the entry, and the identity holds with ρ inside G.
- G restarts at the spike, the floor, a forced spike and a discharge, including for a neuron with no open arrival. The read leaves x and G and re-bases B = xG.
- A deterministic spike posts nothing; the escape of i→j posts nothing to i→j.
- The read synapse's count is zeroed at reset, reaches `output_counts`, `output_counts_hz`, `class_sums` and `evidence_score`, and is in F_i and a_i.
- The read synapse is absent from `network.connections`, flatten and `wiring_digest`.
- Fan-out: κ_i = κ(N)/F_i, with F_i counting the read synapse; a source with F_i = 0 makes no decision.
- TRACE ventured:
  - a relayed arrival raises V and the stamp but not x or B; a ventured one raises all;
  - under TAU 2, after a relayed then a ventured arrival, x_ij and `trace_at` are exactly what Q5(ii) pins.
- `traced` stays True under the mechanism after each of: constructing `Teacher(eligibility="hazard")`, fast.train's `centre` call (fast.py:395), `set_delta(0)`, restore, and wrapping in `ArrayNetwork`. The check covers every `neuron.traced`, `network.traced`, the array engine's `traced` and the Rust engine's `traced()`.
- `Network.propagate` under the mechanism makes synapse decisions (stubbed stream); the module-level `propagate()` refuses.
- Charged arrival times equal the rate drive's draws at 3× the rate from the network stream; nothing at t_e.
- A charged delivery adds θ/3, is dropped while refractory, touches, anchors the wave, and moves no trace or stamp.
- A charged input spikes only by the comparison; its mark is set at the first delivery; §8.1 skips its fan-in.
- Refusals, each matched by message:
  - width + mechanism;
  - hebb + mechanism;
  - no stream, including `run_epoch` without an rng and `ArrayNetwork.fire_input` with `explore_rng` None;
  - unknown names;
  - each interim refusal from §3.

**tests/test_hazard.py (extend)**
- Three-engine agreement under the mechanism in the pattern of :136-169 and :393-458, over the 1.13 matrix × TAU {inf, 2}. It compares read counts (`==`), G (`==` Rust, rtol 1e-9 arrays) and `exposed_since` (`==` Rust, 1e-12 absolute arrays) as well as spikes, scores and weights.
- The existing neuron-rule tests stay pinned to an explicit `set_delta`.

**tests/test_propagation.py**
- A ventured signal keeps kind SIGNAL and sums in push order with relayed signals.
- An EXTERNAL charge sorts first in its wave, touches, and anchors the wave's time.

**tests/test_fast.py**
- compare refuses the mechanism until step 3, then is inverted. Under the mechanism it reports a planted `exposed_since` difference.
- `build` accepts 3- and 4-tuple pending events.
- `pending_events()` pushed into a fresh engine keeps order and ventured marks.
- `train` refuses hebb and accepts reinforce under the mechanism, and its default eligibility is not hebb.
- The mechanism's setter refuses a non-identity `out_edges`.
- The arm json names the mode, h0, family, scaling, trace and drive.
- A ventured-trace arm's json says the estimator is biased.
- A resumed arm that names no drive keeps the checkpoint's drive.

**tests/test_resume_exact_rust.py**
- The four cases of :13-18 with the mechanism on.
- Ventured signals in flight at the cut.
- Through #21's callback, which runs after an epoch's update and before the next reset: G open across the epoch boundary, and the finished epoch's read count at the cut, zeroed by the next reset after resume. This is not a mid-epoch case.
- A 3-tuple `engine_pending` loads as relayed.

**tests/test_persistence.py**
- A synapse round trip carries G, read counts, marks and settings.
- A format-2 file with no synapse keys and 2-element pending restores as the neuron rule, with its signals relayed.
- Formats:
  - a synapse checkpoint is written as format 3 and a neuron-rule checkpoint as format 2, so trunk code's format-2 reader (persistence.py:183) refuses the first and reads the second;
  - `read_checkpoint` accepts 2 and 3 and refuses any other format.
- κ_i is recomputed, not stored (fan-out included).
- An exact object resume under the mechanism (pattern of :269-316).
- A charge or stimulus placed at until − slack/2 survives an object checkpoint (or the checkpoint refuses).
- The score is carried (§8.14).
- A network saved under charged and loaded with no `--drive` stays charged.
- A resume under the other mode is refused.

**tests/test_arrays.py**
- The assertion that edge order equals connection-id order.
- Ventured per-edge arrivals survive `__init__` from a mesh and `sync_to_mesh`.
- The refusal until step 4, including a mode or drive set after wrapping, caught by `fire_input`.
- A floored or discharged neuron with no open arrival restarts G.

**tests/test_constants.py**
- The five new register values.
- The parser defaults equal them, with the mode sentinel.
- :30 changes in step 5, when `--delta`'s parser default becomes None; the `ESCAPE_DELTA == 0.455` assertion stays.

**tests/test_mnist.py**
- compare on goo 455 with the mnist read under the mechanism (Rust-only; skipped on CI).

**At the flip: tests/conftest.py, tests/test_goo.py, tests/test_monitor.py**
- A fixture that pins the neuron rule, on the pattern of `forced_input` (conftest.py:10-22).
- Rewrite test_goo.py:447-448 (the banner), :449-452 (hebb with no `--delta`) and test_monitor.py:132-138 (the fired-line count).

---

## 5. Questions for Byron before code

**Q1. What switches the mechanism on, and is h0 = 0 "off"?** (§6.13, §7.3, §7.6, §8.3)
- Reading (a): the mechanism is on iff h0 > 0. Then h0 = 0 takes no draws and reinforce is refused.
- Reading (b): a separate mode. h0 = 0 still takes E + O draws a wave with P = 0; reinforce is allowed and posts nothing.
- Bits: whether the stream advances, and whether the Teacher refuses.
- **Recommend (b), with a named selector** (proposed EXPLORATION = neuron | synapse, with an Appendix A row under A.0). Under linear, h0 = 0 gives h = u and still explores, so h0 cannot be the switch.

**Q2. When does the default flip, and how does a run ask for the neuron rule?** (§7.6, App. A ESCAPE_DELTA row)
- The flip happens either when all three engines carry the section, or when objects and Rust do.
- `--delta` defaults to 0.455 (cli.py:372-380), so after the flip a run with no flags "asks for both".
- **Recommend:**
  - flip only after all three engines carry it and compare on the configuration;
  - an explicitly given `--delta > 0` (or selector = neuron) asks for the neuron rule and turns the synapse default off;
  - only an explicit request for both is refused.
- The sentinel that implements "explicitly given" lands in step 5, with the flags.
- Byron's call: whether the flip waits for §8.16's LR to be re-found. 11.11's 0.002 was found under the neuron rule, and the −ln h0 fold ties LR to h0. **Recommend waiting.**

**Q3. Under linear, does G carry ρ(u)?** (§8.16: the rule paragraph against the bookkeeping paragraph)
- (a) G += ρ(u)·(a c − (F−a) m).
- (b) The bookkeeping formula as written, without ρ.
- Bits: every score into a linear-family source.
- **Recommend (a).** The settle only regroups the per-wave sum if G carries ρ.

**Q4. Is an entry posted where V_i ≤ 0, where the clip makes the true derivative 0?** (§8.16, §7.5)
- (a) Post at every wave, as displayed.
- (b) Post only where 0 < V_i < θ_i, as the toy did (docs/synaptic-noise-toy2.py:190-195).
- Learning:
  - Under loglinear, (a) adds zero-mean noise, not bias.
  - Under linear, (a) weights rest-state entries by ρ(0) = (1−h0)/h0 = 99 at h0 = 0.01. Inhibited outputs are common on goo 455, so this matters there.
- **Recommend (b), with nothing posted at V = 0 exactly.** It is the log-likelihood derivative §8.16 says the entry is, and it is the rule that produced §7.13's numbers and the choice of h0 = 0.01.
- (b) needs §8.16's text changed first (step 0).

**Q5. Under TRACE ventured, what does a relayed arrival do to the trace's bookkeeping?** (§8.17, §8.16, §8.12)
- **(i) B.**
  - (a) Only arrivals that raise x note B.
  - (b) Every integrated arrival notes B.
  - **Recommend (a).** Under (b), xG − B subtracts G for arrivals that x never counted.
- **(ii) x and `trace_at`, both TAU.**
  - (a) A relayed arrival leaves x_ij and `trace_at` untouched.
  - (b) It decays x_ij to now and moves `trace_at`, adding 0.
  - They are equal in exact arithmetic but not in floating point (x·e^−a/τ·e^−b/τ ≠ x·e^−(a+b)/τ). The leak walk reads `trace_at` (lib.rs:919, arrays.py:323), and checkpoints carry it under either TAU.
  - **Recommend (a).**
- A sentence in §8.17 settles both.

**Q6. When does a read-synapse escape count: in the epoch of its decision, or one hop later?** (§7.9, §5.10, §3.10)
- (a) At decision time t; no event.
- (b) At t + HOP, so an escape in the last hop counts next epoch and needs a checkpointed event.
- **Recommend (a).** "The read synapse delivers nothing", and it matches the toy (toy2.py:194).

**Q7. Does the read count make an output "fired"?** (§2.6, §5.10, §9.11)
- It affects the rate memory, the stuck counts, and the fired/again/window/rate reads.
- **Recommend:**
  - §2.6 and the stuck counts use spikes only (literal: "fired");
  - the non-count reads are refused under the mechanism until specified.
- Note: the reversal problem, the CLI default (constants.py:155), reads `fired` through the `Problem` default `read = "fired"` (problems.py:34). The sweep worker also falls back to `fired` (cli.py:941).

**Q8. Does a charged delivery that is dropped at a refractory input set the driven mark?** (5.4b, §1.7)
- **Recommend yes, on any delivery.** §1.7 calls a dropped signal "delivered". This is rare on mnist (about 14 arrivals per bit-1 input an epoch).

**Q9. Does the charged mark also exempt the input from §2.6 and from §9.2 steps (3) and (4)?**
- §9.2 skips only "the stimulus forced to fire". The code has one `forced` flag (learning.py:340-351, fast.py:169-185, arrays.py:546-564).
- **Recommend yes, with one flag.** §2.6's own reason covers a charged input: "its firing that epoch saying nothing about the network". Confirm with a word in §9.2.
- Homeostasis and un-sticking are off for mnist (11.12), so only r_j and the stuck report move.

**Q10. What is the exact arithmetic of a charged delivery, and where does it sit in a wave?** (5.4b, App. A DRIVE_STEPS, §3.6, §3.7)
- The choices: the form θ_i/3.0 or θ_i·(1/3), and the current θ or the starting θ.
- The division form leaves 23 of in-degrees 1..399 one ulp short after three deliveries from rest (d = 35 among them), so those inputs need a fourth. The multiply form leaves 127 short.
- **Recommend:**
  - `θ_i / DRIVE_STEPS`, a division, on the current θ, computed at delivery. This avoids a stale amount for an arrival within slack of the horizon that waits past a threshold move (and such an arrival must survive a checkpoint, 1.12).
  - Accept the fourth delivery ("about as often").
  - The charge touches the input, and it sorts before signals of the same moment (EXTERNAL = 0, as both heaps do today).

**Q11. In what order are one wave's escapes pushed?** (§3.6, §3.7, §3.8, §7.5)
- **Recommend:** after all of the wave's spike pushes, in edge order (sources in index order, each source's outgoing list in order). Write it into §3.6 or §7.9, since §12.4 compares with `==`.

**Q12. Do inactive connections draw, escape, or count in F_i?** (§3.5, §3.8, §7.7)
- **Recommend:** every synapse draws, inactive ones included (as flatten counts them). Refuse any inactive connection under the mechanism until Byron rules. Nothing builds one today; only tests set it (test_arrays.py:95-110).

**Q13. What is u_i when θ_i ≤ 0?** (§7.5, §6.9, §0.5)
- u is 0/0, or the clip bounds are reversed. One form gives u = 1 at rest.
- The case is reachable: test_hazard.py:108-118 builds a θ = 0 network, and homeostasis and un-sticking are unclipped.
- **Recommend refusing** at build, and if a threshold moves to ≤ 0.

**Q14. What range of h0 is allowed?** (§7.6, §0.5)
- **Recommend 0 ≤ h0 < 1.**
  - At h0 = 1, h is flat: the loglinear fold posts noise where the true gradient is 0.
  - h0 > 1 inverts the family.
  - h0 < 0 makes `pow` return NaN.
  - Guard m > 0 before computing ρ: under linear with h0 = 0, ρ(0) is infinite.

**Q15. The floating-point forms** (an engine note like §6.5's; Claude's to propose, Byron's to accept)
- u = min(max(V, 0), θ)/θ.
- Loglinear h = h0 ** (1.0 − u). CPython's float `**` and Rust's `powf` reach the same glibc `pow` in the same process.
- Linear h = h0 + (1.0 − h0)·u.
- κ_i is computed once in Python and handed over.
- m = min(Δt / hop * κ_i * h, 1e3), left to right as neuron.py:230.
- c = m*exp(−m)/−expm1(−m), as neuron.py:266, only when m > 0. In Python 0.0/0.0 raises, while Rust gives NaN.
- entry = a*c − (F − a)*m, with F − a an integer; ρ multiplies the finished entry.

**Q16. Is DRIVE_STEPS a run option?** (App. A, §12.9)
- **Recommend:** a register constant only, with no `--drive-steps`. Carry it in checkpoints anyway.

**Q17. May a checkpoint resume under the other exploration mode?** (§12.9)
- Nothing maps E and the credit to G. Under the neuron rule, `exposed_since` holds t_fired + REFRACTORY.
- **Recommend refusing.**

**Q18. Is the charged drive allowed under the neuron rule?** (5.4b: "spikes by the comparison (§6.13)")
- **Recommend refusing.**

### Consequences of §7.6's default-on (from the plumbing reading)

**CLI and driver**
- `--delta`'s default (cli.py:372-380) must become a `None` sentinel, or every run with no flags asks for both explorations and §6.13 refuses it. The sentinel lands in step 5, with the flags that select the mechanism. Until then the default width is applied unconditionally at cli.py:757-758, cli.py:939 and docs/rust-sweep.py:166, and every step-5 or step-6 run that selects the mechanism without `--delta 0` would ask for both.
- cli.py:757-758 compares against the constant. If the default moves without rewriting it, every `--load-weights` of a width checkpoint calls `set_delta(0)`.
- cli.py:768 and `RESUMED_SETTINGS` (docs/rust-sweep.py:223) already write the fresh run's drive over a checkpoint's (1.12).
- Also touched:
  - the eligibility default (cli.py:759-760, :998-999);
  - the banners (:807-813, :1054);
  - the `--seeds` job and worker (:1023-1038, :939);
  - `grid_of` (docs/rust-sweep.py:166) and the eligibility defaults (:98, and `grid_of`'s own `"hebb"` at :119);
  - fast.train's `eligibility="hebb"` (fast.py:321).
- Four drivers built on `grid_of` would switch rule silently: docs/mnist-watch.py:74, docs/mnist-evidence-diag.py:23, docs/mnist-evidence-gradient.py:30, and docs/waves-per-hop.py:88 (untracked).

**Tests**
- Assert the width default: test_constants.py:30 (changes in step 5) and test_goo.py:447-448.
- Hebb run with no `--delta`, which would be refused: test_goo.py:449-452.
- test_monitor.py:132-138's count of 3 "fired" lines holds only if the synapse hazard is off too.
- The array-engine CLI runs (test_arrays.py:145-165) would be refused until step 4.
- CLI tests that would silently change mechanism:
  - test_goo.py:107, 185-188, 382-452, 546, 585;
  - test_problems.py:114, 424;
  - test_time.py:215-229;
  - test_monitor.py:86-175;
  - test_critic.py:52.
- A flip in the Goo constructor rather than the CLI would also break test_hazard.py:33-40, 122-133 and 173-193, and test_fast.py:55-99. Keep the flip where ESCAPE_DELTA is applied today.

**Checkpoints**
- 202 files sit in runs/*.json; 196 are format 1 and already refused.
- Format 2 is on 796 files: 755 sweep network checkpoints under runs/<sweep>/ (24 sweeps), 39 under runs/archive/, and 2 at the top of runs/.
- 37 of the sweep checkpoints carry `engine_pending`: mnist-1m-lowlr 12, mnist-1m-hidden 15, mnist-1m-leaky-10seed 10.
- They keep resuming bit for bit only if all of these hold:
  - `read_checkpoint` still accepts format 2 (a bare FORMAT bump refuses them all);
  - missing synapse keys read as the neuron rule;
  - the resume paths stop writing defaults over them (cli.py:754-773, docs/rust-sweep.py:223-225);
  - 2- and 3-element pending entries load as relayed;
  - `traced` and κ are restored from the mode;
  - the rule for the exposure clock at a spike comes from the checkpoint's mode.

**Past sweeps**
- Arms that took 0.455 by default (goo455-ff-lr-100k, goo60-hazard-lr, mnist-ff-lr100k) would silently rerun under the synapse rule.
- Arms that named `--delta` need "an explicit delta asks for the neuron rule".
- `run_arm` skips an arm whose CSV exists (docs/rust-sweep.py:291), and #21 resumes arms by itself, so relaunching under the same `--name` would mix both regimes.

**Operating points**
- mnist's LR 0.002 (problems.py:90, 11.11) and pickiness 2 were set under the neuron rule's rest rate (§9.5). Under the mechanism a silent neuron has no rest rate of its own (§7.2).

**Cost**
- Draws per wave go from N to E + O: on goo 455, 455 → about 1,451 (3.2×); on goo 654, 654 → about 21,446 (about 33×). Details in §6.

**Credit flow on goo 455**
- Only the 60 read synapses' decisions post, crediting the input→output synapses.
- Under the default rate drive, inputs sit at V = 0. Their 1,391 synapses whisper at h0·κ = 0.0036 per hop, about 69 ventured arrivals an epoch, carrying no information about the pattern.
- Only the charged drive gives the input synapses' escapes the pattern.

---

## 6. Risks to bit agreement and to cost

### Objects vs Rust (`==`)
- **Arithmetic form.** h, u, m, κ_i, ρ and the entry are new arithmetic. Any difference in association (for example a·(c+m) − F·m), or using exp(log) for pow, parts the engines (Q15). Rust emits no FMA (App. B).
- **Push order** of escapes (Q11).
- **Draw layout.**
  - E draws in flatten order, including inactive edges, then O in output order, all taken.
  - Rust must index the E draws by edge id (`out_edges[k]`, lib.rs:1002-1004), not by position in `out_start[i]..out_start[i+1]`. The two coincide only because flatten is source-major and `bucket` is stable (lib.rs:1049-1058), and `Engine::new` never checks it.
  - Rust must be given the output order, not assume "the last O". rust-sweep's `_output_counts` makes that assumption today (docs/rust-sweep.py:267-276).
- **The ventured bit** must ride outside the heap key: `Event` Ord (lib.rs:120-128), and the payload of the Python heap tuple (propagation.py:98). It must not be a new kind.
- **TRACE ventured's `trace_at`** under the leak: one engine decaying x at a relayed arrival while the other skips it parts the scores (Q5(ii)).
- **State changes one engine could miss:**
  - `exposed_since = t` at a spike, and for every non-spiking source each wave (compared in fast.compare from step 3);
  - G restarting at the floor and at a discharge, where E never restarted (neuron.py:157-160, 323-326; lib.rs:820-843);
  - B += G at delivery;
  - the sign of the read settle (lib.rs:441-453 and network.py:140-154 subtract today);
  - an output's read escape inside a_i as a single G addition.
- **u** must read `potential_at` under the leak and θ_i rather than `threshold_at`. Refuse `bored_after > 0`.
- **Guards:** m > 0 before c and ρ; skip F_i = 0.
- **Stream plumbing:**
  - handover gated on width only (fast.py:81, lib.rs:380-384);
  - compare's end-state check gated on width (fast.py:286);
  - `pending_events` unwraps on NaN (lib.rs:726);
  - `push_events` accepts EXTERNAL and `run` drops it silently (lib.rs:580);
  - `hazard_draws` returns silently with no stream (lib.rs:855).
- **The charged amount** computed at scheduling in one engine and at delivery in another differs for an arrival within slack of the horizon after a threshold move (Q10).
- **Checkpoints disagree on waiting events.** A stimulus or charge in [until − slack, until) survives a Rust-path checkpoint (`engine_pending`, lib.rs:724-728) but not an object one (propagation.py:119-121, persistence.py:345-347). This is pre-existing for rate-drive stimuli (1.12).
- **`traced` switched off** by `centre`, `set_delta`, restore or the Teacher (network.py:128-138, :175; persistence.py:328; learning.py:545; fast.py:395): a run silently posts nothing. The Rust side cannot be checked without the new `traced()` getter.
- **`reinforce_hazard`** (lib.rs:456-463) keyed on the width would refuse a mechanism run if anything called it.

### Arrays (spikes `==`, continuous values to 1e-9)
- Escapes are not guaranteed exact. P comes from V summed in matrix order and from numpy's exp/pow/expm1, not libm. An escape within an ulp of P can part the engines, as a spike can today (§12.4).
- Charged sums must be added one delivery at a time, in the objects' order, or the 23 in-degrees on the threshold boundary part the spikes. Mixed with signals (mnist inputs hear hidden neurons), only the tolerance holds.
- The per-source pending round trip turns one ventured edge into a full relay (arrays.py:150-154, :602-606) until the per-edge schedule exists.
- The edge-order assumption (arrays.py:130-135).
- The wave anchor already differs from the objects (arrays.py:241-246 vs propagation.py:151-158).
- `_out_degree` counts active edges only (arrays.py:174), while flatten counts all (fast.py:55-61).
- G restarted through `_close_arrivals`' edge mask would miss neurons with no open arrival (arrays.py:352-364, 268-270, 420-422); restart it on neuron masks.
- A missing stream crashes rather than refuses (arrays.py:84, :271-272). Settings set after wrapping bypass `__init__`'s refusals.

### Cost (estimates from the readings' counts; not measured)
- **goo 455, rate drive.**
  - About 1,494-1,534 waves/epoch were measured under the neuron rule (docs/waves-per-hop-10k.md:44-48).
  - The mechanism adds about 70 whisper waves, for about 2.25M uniforms/epoch against about 0.69M today.
  - At 5-10 ns a uniform, that adds roughly 11-23 ms to a 19 ms epoch (52 epochs/s per arm, :65).
  - Loglinear adds one `pow` per non-spiking source per wave.
- **goo 455, charged drive.** Arrivals triple, and input u sits at ⅓ or ⅔ (h = 0.046, 0.215). That is about 3,500-3,700 waves and about 5M uniforms per epoch.
- **goo 654 (mnist default, 199 hidden).** About 21,446 draws per wave: at least 33M uniforms/epoch, or 0.17-0.33 s/epoch in draws alone. A 1M-epoch arm becomes days per core. Measure in step 6 before planning sweeps.
- **The plan's figure** (docs/synapse-engine-plan-2026-09-23.md:218-222: 39 waves, 54k uniforms, "under 15 per cent") is about 40× low.
- **Objects:** a Python loop over E per wave. Tests must use goo 24/60 and few epochs.
- **Arrays:** `hazard_draws` is a Python list. Use numpy MT19937 loaded with Python's state and hand the state back exactly (checked equal over 100,003 draws in the arrays reading).
- **Leak:** the fan-in walk is O(E) per wave with an `exp` per nonzero trace, the same order as the neuron rule's walk today. Q4(b) shrinks it.
- **Allocation:** preallocate the per-edge draw buffer; lib.rs:848 allocates per wave today.
- **Parallel scheduler:** the MT19937 stream grows 3-33× per wave. Jump-ahead stays possible because consumption does not depend on outcomes (§3.8).
- **Verification:** CI does not build Rust (.github/workflows/tests.yml). Rust agreement and exact resume are shown locally only, and each PR must say so.

---

## Revisions

1. **Step 3's gate no longer includes the exact resume.** `test_resume_exact_rust` goes through `_save_network` → `persistence.checkpoint` (rust-sweep.py:217) and `resume_grid` → `restore` (:243), which are step-5 work. Step 3's gate is now compare + stream + an engine-level `pending_events`/`push_events` round trip; the resume case is in step 5's gate.
2. **FORMAT.** `read_checkpoint` refuses any format but FORMAT (persistence.py:183-189), so a bare bump would refuse every format-2 file. There are 796 on disk (counted), including all 37 `engine_pending` checkpoints. It now accepts {2, 3}, writes 3 only under the mechanism, and gains matching tests.
3. **Questions gate.**
   - Moved from before step 2 to before step 1.
   - The list is corrected to Q1, Q3-Q6, Q8-Q12 plus Q15. Q7, Q13, Q14, Q17 and Q18 are marked as interim refusals, and Q2 and Q16 as due before step 5.
   - Each answer must land as clause text first.
   - Q4(b) is noted as requiring an edit to §8.16's "at every wave" posting sentences (verified in AUTHORITY.md:1203-1208).
4. **Q cross-references renumbered against §5:** 1.12 (Q16, Q17), the §2 table (Q12, Q16), the interim refusals (Q13, Q12, Q14, Q17, Q18) and §4 (Q15).
5. **TRACE ventured and `trace_at`.** Added as Q5(ii), in 1.9, §4 and §6. Both engines decay and move `trace_at` at every integrated arrival (propagation.py:174-175, lib.rs:568-571), and the leak walk reads it (lib.rs:919).
6. **The drive is overwritten on load and resume** (cli.py:768; `drive` in RESUMED_SETTINGS, rust-sweep.py:223, applied at :250-251), against §12.9's "resumes under it unless the resuming run overrides it explicitly". Added to 1.12, step 5 and the tests.
7. **The `--delta` sentinel moved from step 7 to step 5**, with the three width sites (cli.py:757-758, :939, rust-sweep.py:166) and the None-breaking readers (cli.py:998-999, :808-809, :1054). test_constants.py:30 moves with it. The 1.1 setter wording and the §2 table row are reconciled: a library width is always explicit (constants.py:99-100).
8. **`Network.propagate`** (network.py:498-499) added as a second call site for the post-fire hook, with a test.
9. **Array engine.**
   - A missing stream is refused at the draw site (arrays.py:84, :271-272; `explore_rng` defaults to None at network.py:90). Arrays are added to the permanent no-stream list.
   - Refusals are repeated in `fire_input`/`_run`.
   - The charged path is refused there too, because `fire_input` hands every `input_schedule` event to `_stimulate_at` (arrays.py:458-460).
10. **Rust edge indexing.** The E draws are indexed by the edge id `out_edges[k]` (lib.rs:147, 1002-1004), and the setter checks that `out_edges` is the identity, since `Engine::new` (lib.rs:237-240) does not.
11. **#21's callback runs after an epoch's update, before the next reset**, not mid-epoch (verified in the sweeps/checkpoint-every diff of fast.py, and fast.py:437-453). The test is restated. §8.14's carried score, absent from `checkpoint`, is added as a pre-existing gap, with the mid-epoch fields or a boundary-only rule.
12. **Citations fixed:** #21 rust-sweep.py +96/−28; lib.rs:380-384 (four places); problems.py:34 for `read = "fired"`; the stale 644 also at problems.py:85 and docs/mnist-evidence-diag.py:23.
13. **`exposed_since`** compared in fast.compare (absent from fast.py:255-283) and in the three-engine and array gates. Potentials and `trace_at` were added to compare, since their getters exist. An F_i = 0 clock test was added.
14. **Array G restart moved to neuron masks.** `_close_arrivals` only touches open edges (arrays.py:356). Read counts (`==`) and G (rtol 1e-9) were added to the array gate and to test_hazard.
15. **`traced` test** across Teacher, fast.train, `set_delta(0)`, restore and ArrayNetwork, plus a Rust `traced()` getter (none exists). The `Network.traced` property (network.py:128-131) was added to the recompute list.
16. **`reinforce_hazard`** (lib.rs:456-463, keyed on `self.hazard`, no caller in src/docs/tests) added to the re-key list for step 3.
17. **Events waiting past the horizon.** Stimuli and charges drawn in [until − slack, until) are dropped by the objects' `pending()` (propagation.py:121) and restore (persistence.py:345-347) but kept by Rust's `pending_events` (lib.rs:724-728). Now carried or refused, with a test. Noted as pre-existing for rate-drive stimuli.
18. **Grid_of's eligibility default.** Its own keyword default `"hebb"` (docs/rust-sweep.py:119) was added beside :98.

Note: gap 9's supporting evidence is partly wrong. test_problems.py:223 reads `net.drive` rather than assigning it after wrapping; that test sets the drive before wrapping, at :218. The fix still applies, because the settings stay writable attributes and `run_epoch` assigns `explore_rng` after wrapping (monitor.py:44).

---

## Byron's answers, September 25, 2026

Put to him in five rounds of multiple choice, about 02:00 MDT. Each is written into AUTHORITY.md, at the clause named.

| Q | answer | clause |
|---|---|---|
| Q1 | a named setting, EXPLORATION = neuron or synapse; a rest hazard of 0 does not switch it | §7.1, §6.13, §7.6, App. A |
| Q2 | the default moves once all three engines agree on mnist's configuration and the LR has been re-found; an explicit width then asks for the neuron rule; both together refused | §7.6, App. A |
| Q3 | the gain carries the unfolded factor: 1 under loglinear, (1 − h0)/h(u) under linear | §8.16 |
| Q4 | the entry is posted only while the source's V > 0, as the toy posted | §8.16 |
| Q5 | under TRACE ventured, a relayed arrival notes nothing and leaves x and its decay clock as they are | §8.17 |
| Q6 | a read escape counts at the wave of its decision | §7.9 |
| Q7 | the rate memory and stuck counts read spikes only; the non-count reads are refused under the mechanism | §5.10, §2.6 |
| Q8 | any charged delivery sets the driven mark, one dropped at a refractory input included | 5.4b |
| Q9 | a charged input is exempt from the rate memory and §9.2's steps 3 and 4, by the one driven mark | 5.4b, §2.6, §9.2 |
| Q10 | θ/DRIVE_STEPS, a division, on the threshold at delivery; a further delivery accepted; charges before signals in a wave | 5.4b |
| Q11 | escapes pushed after all the wave's spike signals, in edge order | §3.6 |
| Q12 | an inactive synapse keeps its draw; a network holding one is refused until specified | §3.8 |
| Q13 | a threshold at or below zero is refused, at build and when one moves there | §7.5 |
| Q14 | 0 ≤ h0 < 1 | §7.6, App. A |
| Q15 | the floating-point forms accepted as an engine note | §7.5, §8.16 |
| Q16 | **not as recommended:** DRIVE_STEPS is a run option (`--drive-steps`), recorded per run and carried by checkpoints | 5.4b, §12.9, §8.14, App. A |
| Q17 | a resume across explorations is refused | §12.9 |
| Q18 | the charged drive under the neuron rule is refused | 5.4b |

