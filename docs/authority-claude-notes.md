# What I changed, and what is still yours

*Claude's pass over `docs/authority-draft.md`, September 17, 2026. The
merged result — Byron's ordering, Claude's corrections — is
`docs/authority-merged.md`, and the clause numbers below are its. The two
passes and the base are all kept: `authority-byron.md`, `authority-claude.md`,
`authority-draft.md`.*

Byron's pass moved the synapse ahead of the neuron (§1 and §2) and added the
contents; he changed no rule. Everything below is the content pass.

## 1. Things no one checker could see

Eight drafters each wrote a whole section, so the file stated some rules
twice, in different words, and in three places the two statements disagreed.
One clause now owns each rule and the others cite it.

| the rule | owned by | what the other clause became |
|---|---|---|
| the hop | §3.2 | §2.6 deleted |
| the per-decision expectation | §8.6 | §2.8 deleted; the two rules §8 lacked moved into §8.6 |
| the potential axis and its fan-in scaling | §4.10 | §6.1 is a pointer |
| the refractory period | §2.5 | §6.12 keeps only "makes no decision" |
| the spike's reset | §2.4 | §6.11 points |
| the wave's draw | §3.8 | §6.7 keeps only "fires iff below $P_j$" |
| the baseline | §8.2 | §9.3 points, and names the resume question |
| the weight clip | §1.4 | §8.1 points |
| the count's $\sqrt{N_0/N}$ | §6.6 | §6.5 calls it $\kappa(N)$ |

Three of those were contradictions, not repetitions:

- **The rate memory.** §2.1 held $r_j$ as the neuron's state; §9.8 said "the
  book is the teacher's and not the network's". It is the neuron's state, and
  the teacher is what moves it, once an epoch.
- **The score's clearing.** §1.9 cleared it at the read, §3.9 at the next
  epoch's reset. The read is the rule; nothing charges between the two, so an
  engine may do it at the reset.
- **The ISI factor.** §7.4 defines $f$ on $x = t/I$, with $f(0) = -1$ and
  $f(I) = 1$. §7.5 then called it as $f(t' - t^{\text{fired}} - I)$, which is
  Byron's parameterisation, where $f(0) = 1$. Same curve, two origins — a
  rebuilder reading both would have shifted the factor by one ISI. §7.5 now
  calls §7.4's $f$, with a line saying the two are one function.

Two sections were both numbered 10. Local rules is §10, the problem §11, the
invariants §12. Every `§N.M` in the file now resolves to a clause that
exists; I checked it mechanically, not by eye.

## 2. Rules the draft had wrong

- **The class critic is alive.** Your 15:12 answer — "keep the class critic" —
  came two minutes after the three-critic list the drafters worked from, so
  the draft wrote it out of existence and put "**No run is paid on it**" in
  bold. §9.7 is now the class critic *and* the fraction right as one rule,
  which is what your answer says they are. §9.5, §11.15 and the register
  follow.
- §0.2 forbade any rule branching on a neuron's index — which goo's zones do
  at every wiring case. Restricted to a neuron's own integration, firing and
  learning.
- §0.5 said no quantity is clipped. Learning clips every weight. What you
  struck was the threshold clamp.
- §8's preamble said one rule moves a weight. The quash moves weights too,
  and you kept it.
- §8.3 let **hebb** in as the default eligibility when the threshold decides.
  Nobody set that: the record's default named ELIGIBILITY, whose value was
  perturb, and perturb is dropped. It now says the question is open.
- §5.8 said the update "passes over" every synapse whose target was driven —
  which reads both as *skips* and as *goes over each of*. It skips.
- §5.11 defined $P$ as the population per class in one sentence and per half
  in the formula: 120 output neurons or 60, depending which you read.
- §6.10 had a forced spike charging neither credit nor expectation. It
  withholds the credit; the debit settles as at the floor.
- §1.5 said "exactly four pieces of state" and §1.6 then named a fifth.
- §4.10 gave the floor as a free constant, so a change to GOO_THRESHOLD would
  have left it behind. It is $-4 \times$ the threshold.
- §3.4 called the clock's slack a nanosecond. The clock is in nominal
  milliseconds and TOLERANCE is relative, so it is a picosecond.
- §5.4 drew arrivals "until the horizon"; the bound is $t_e + $ INTERVAL,
  which need not be the same instant.
- §5.2 said place $i$ shows coded bit $i$ — false wherever there are clock
  neurons, which is mnist.
- §4.9 said an ff2 goo rebuilds with no seed. The code refuses it. I stated
  the rule and named the code's bug, the file winning.

## 3. Taken out because the file's own rule excludes it

The 3,300 decisions an epoch; the part in $10^{15}$; the 1/12 output zone;
"why a rate and not a coin per wave"; the argument for small-world shortcuts;
the independent-increments rebuttal; the two engine-bug stories in §12.8; the
sweep's $\{1, 2, 4\}$ behind TEMPERATURE. All of it is in `RECORD.md`.

I also deleted the **Open** note on INTERVAL: the record decides 35 ms and no
problem overrides it, and the openness was the drafter's invention from what
the mnist runs happened to use. If 35 against 100 is a real question it is
one to put to you, not to write into the specification.

## 4. Open, and yours

Grouped so you can answer in batches.

**One word each**
1. The three non-defaults: does the constant become 0 (HOMEOSTASIS, UNSTICK,
   QUASH_RATE), or keep its value with every run switching it off? I wrote
   the second. mnist is unaffected either way.
2. `--discharge` — zeroing potentials at the epoch's reset — is in neither
   your kept list nor your dropped one, and §8.11 names a discharge as an
   event that settles with no credit. Keep as a non-default, or drop it and
   delete that arm.
3. Which eligibility runs when the threshold decides, now perturb is gone.
4. Whether an output neuron may project onto the input zone (§4.5's third
   line is my reading of "no cycles").
5. Whether mnist's quash-off is really yours, or just how the problem was
   built.

**Values a sweep is to set** — TEMPERATURE 2, TARGET_ISI 5.1 ms, GOO_PROJECTION
0.2 (inherited from a wiring that has left the specification), DECISION_MEMORY
$10^{-4}$, QUASH_RATE and QUASH_K.

**Two the code disagrees with itself about**
6. The baseline on a resume: a checkpoint restores it, a sweep's resume
   starts it from the first resumed epoch.
7. TEACHER_THRESHOLD: a rate that holds at every epoch length, so the count
   it demands changes with the length, or a line set to mean one spike at
   35 ms.

**Two that are work, not wording**
8. §9.7 as I wrote it has every run report the fraction right beside every
   reward. As built, only the Rust driver does, over a run's last tenth and
   only under the evidence critic. The clause says so, but it is asking for
   work in two engines.
9. §9.12's estimator correlation is Rust-only, which is the one place in the
   file where §12.1 — every clause in every engine — is not met. Either the
   others grow it or the clause names the exception.

**Placement**
10. You asked that the ISI factor's authority stand near the very top. It is
    at §7.4, in Exploration, because that is where the firing decision it
    weighs is stated. Say the word and it moves into §0.
11. What the substance is, now the plane and its containers are archived.
    §0.6 still says the goal is the substance.

**Constants no clause claims** — PROBLEM (still "reversal" in the code, a
problem this file does not carry), ACROSS, WEIGHT_EPSILON, RULE, TARGET, LATE.
