# The second round: what AUTHORITY.md still asks of Byron

*September 17, 2026, after the rewrite landed on `rewrite`. The file has 138
clauses. Six carry **Open**, two say **not decided**, and twenty-eight say
"to be corrected in a word" — the rule is stated, but on Claude's reading of
an instruction rather than on Byron's decision. This is all of them.*

*Answers go to `docs/rewrite-answers.md`, in Byron's words, as they are given.*

---

## Part 1 — the decisions (8)

**D1. §0.6 — what is the substance, now the plane is archived?** The record
defines walnut butter as a substance spread on the plane, one neuron per unit
cell. The plane and its containers left the specification at 15:08 MDT.
§0.6 still says the goal is the substance and not a task.

**D2. §5.10 — is TEACHER_THRESHOLD a rate or a line?** 14.3 Hz was set to mean
one spike at a 35 ms epoch. Read as a rate that holds at any epoch length, the
count it demands changes with the length — one spike at 35 ms, two at 100 ms.
Read as a line, it means one spike whatever the epoch. The clause states the
first. The mnist runs used both epoch lengths.

**D3. §7.4 — TARGET_ISI's value, and is it one constant or three?** Now 10 ms
and open. It is the shaping function's peak, the deterministic drive's period
(§0.12) and the rate teacher's target rate for a 1 (§9.13). One constant means
a sweep on any one job moves the other two; three sharing a default costs
nothing now and stops that.

**D4. §9.3 — what does the reinforcement baseline do on a resume?** A
checkpoint saves and restores it; a sweep's `--resume-from` starts it from the
first resumed epoch. The code disagrees with itself.

**D5. §9.9, §9.10, §10.1 — do the three non-default constants become 0?**
HOMEOSTASIS 10⁻⁶, UNSTICK 10⁻³, QUASH_RATE 0.02 with every run switching them
off, which is what the clauses say — or the constant itself becomes 0 and the
record's values are what a run asks for. mnist is unaffected either way.

**D6. §3.9 — does `--discharge` stay?** Zeroing the potentials at an epoch's
reset is in neither the kept list nor the dropped one, and §8.11 names a
discharge as an event that settles with no credit. Keep it as a non-default,
or drop it and delete that arm of the settling rule.

**D7. §8.3 — which eligibility runs when the threshold decides?** The record's
rule was "hazard under escape noise, else the ELIGIBILITY constant", and that
constant was perturb, which the scope drops. Under escape noise it is hazard;
with the threshold deciding, nothing stands there.

**D8. §8.8 — is hebb's credit at the spike a full unit or discounted?** The
hazard's credit discounts a spike that was expected; hebb's does not. Byron,
September 17: *"Defer for now."* Still deferred.

---

## Part 2 — the twenty-eight words

### 2a. Confirm or correct (21)

Each states a rule Claude read off an instruction or off what the three
engines do. None is believed to be contentious; all are load-bearing enough
that a rebuild would carry them.

| | clause | the reading |
|---|---|---|
| 1 | §0.4 | that the refractory period being a "computational feature" means **no rule may work around it** |
| 2 | §1.5 | what a checkpoint holds of a synapse: the stamp, the trace with its moment, the note — not the score |
| 3 | §1.9 | the read re-bases the note to $B_{ij} = x_{ij}E_j$, which is the record's "the open arrivals counting from now" |
| 4 | §3.4 | the clock's slack is relative — $10^{-9}\max(1,\lvert t\rvert)$ ms, a picosecond under a millisecond — not the record's "rounded to a nanosecond" |
| 5 | §3.6 | the fire phase asks: forced neurons first, then the touched in the order touched, then every other in index order |
| 6 | §3.7 | a wave's deliveries are summed in push order, and that is a rule rather than an implementation's choice |
| 7 | §4.11 | a neuron with no incoming synapses keeps the container's quoted threshold and floor at scale 1, so it keeps a width |
| 8 | §5.8 | the driven mark belongs to the drive and is stated there, its use being §8.1's forced skip |
| 9 | §6.2 | the floor applies once a wave, to the wave's summed input, before anyone fires |
| 10 | §6.3 | the floor, a forced spike and a discharge each settle every open arrival with no credit |
| 11 | §6.5 | $\Delta t$ clamped at zero, $m$ capped at $10^3$, $P$ computed as $-\mathrm{expm1}(-m)$ — engine requirements for bit agreement |
| 12 | §6.7 | one uniform per neuron per wave, in neuron order, after the floor and before anything fires |
| 13 | §6.8 | every neuron decides at every wave, touched or not |
| 14 | §7.6 | a resumed network keeps the shaping-function setting it was saved under unless the resuming run overrides it |
| 15 | §8.4 | the single-spike rule as derived: $e_{ij} \mathrel{+}= f(c_j - q_j)x_{ij}$ at every decision |
| 16 | §8.5 | the trace is the derivative of the margin, cleared at the spike, at the floor and at a discharge |
| 17 | §8.13 | the late-signal question does not arise under either surviving eligibility |
| 18 | §8.14 | the score is not checkpointed, because a checkpoint is written between epochs where it has just been paid |
| 19 | §9.2 | the order at the read: score, pay, rate memories, homeostasis, un-sticking, baseline |
| 20 | §11.14 | what mnist does not fix — the epoch's length, the threshold and floor, the wiring, the eligibility, the engine |
| 21 | §12.8 | agreement between engines is proven on the configuration a run will use, not carried over from another |

### 2b. Real choices (6)

**W1. §4.5 — may an output neuron project onto the input zone?** The scaled
rule's third line forbids it. Byron's words were "no cycles" and "a two-layer
feedforward network", said of a goo with no hidden neurons; that they also
forbid output→input is Claude's reading.

**W2. §5.12 and §11.6 — is the class evidence $n_k = n_k^+ - n_k^-$?** Byron
settled the layout — three fire-if-one and three fire-if-zero a class — and
said "We will need to change our scoring rule accordingly". The signed sum is
Claude's reading of *accordingly*.

**W3. §7.5 and §8.9 — does the shaping function weigh every entry, or the
read's reinforcement alone?** Byron's sentence was "the teacher's existing
reinforcement is MULTIPLIED by f(t-ISI)", which names the payment. What is
built weighs every per-decision entry. Since the wall anchor made the negative
region reachable, the difference is now visible in a run rather than academic.

**W4. §8.6 — is the warm start $\max(\text{DECISION\_MEMORY}, 1/n)$ right?**
$\hat p_j$ is the plain mean of a neuron's first $1/\text{DECISION\_MEMORY}$
decisions and an exponential average after. Byron said only "Expectation is
changed per decision in this architecture".

**W5. §12.11 — what must a resume reproduce?** The clause states the
continuation as the code has it: the exploration stream starts afresh at seed
+ 1,000,000, the baseline from the first resumed epoch, the input stream
advanced rather than restored. Whether a resume should instead be the saved
run to the bit is not decided.

**W6. §0.11 — does the synapse-as-learner direction belong in §0 at all?** The
record places it at the top at Byron's instruction and expressly not among
§0's rules. It stands in the values section as a direction that binds nothing.

### 2c. Answers that are work, not wording (2)

**K1. §9.7 — should every run report the fraction right beside every
reinforcement?** As built, only the Rust driver computes it, over a run's last
tenth and only under the evidence critic. The clause asks for more than any
engine does; saying yes is work in the object and array engines.

**K2. §9.12 — must every engine measure the estimator's correlation?** It is
the Rust path's alone today. This is the one place in the file where §12.1 —
every clause in every engine — is not met. Either the other two grow it, or
the clause names the exception.

---

*Twenty-one confirmations, fourteen decisions. When they are answered the file
stops being Claude's reading of Byron's system and becomes Byron's
specification of it.*
