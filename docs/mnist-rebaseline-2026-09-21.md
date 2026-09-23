# mnist re-baselined under the September 19 constants

September 20–22, 2026. One run — the ten-seed leaky comparator, §6 — was
lost to a reboot and awaits a relaunch; everything else is final.

`docs/conformance-checks.md` A1 and A2 asked for one re-timing of the hop
followed by one re-measurement of mnist. Between 16:31 and 17:35 on September
19 four things landed on `code_follows_the_file` that touch a mnist run:
the hop 2.50 → 2.55 ms (`29adb7a`, A1/A2), TAU 2.0 → infinite (`e249db3`, A5),
the clock's tolerance 1e-9 → 1e-12 (`215baab`, A6), and the count read made the
default (`8a4d6c0`, C1 — a no-op for mnist, which has set `read="count"` since
September 16). This is the re-measurement, and it separates the four.

## The operating point

Every number below is at goo 455 (395 in, 0 hidden, 60 out), threshold 0.6,
interval 100 ms, delta 0.3125, hazard eligibility, count read at pickiness 2,
evidence critic, hop 2.55, homeostasis and un-sticking off — the September 19
delta sweep's configuration — unless a column says otherwise. Two early probes
were run without pinning `--lr` and `--interval` and silently took the problem's
defaults (0.002 and 35 ms); they are not tabled here and every later run names
its full operating point. Accuracy is the fraction right over the last tenth of
the epochs an arm ran. `runs/<name>/<arm>.json` holds each arm.

## 1. The hop is a non-event

The leaky configuration, at its old floor and learning rate, under the new hop:

| | hop 2.50 | hop 2.55 |
|---|---|---|
| leaky (TAU 2.0), 100k, 3 seeds | 0.177 (RECORD, September 16) | **0.1825 ± 0.0084** (0.1738, 0.1831, 0.1906) |

A 2×2 at 5,000 epochs (`ab.jsonl`, 3 seeds each) says the same from the other
side — the hop column barely moves anything, the TAU row moves everything:

| TAU | hop | output spikes/epoch | outputs silent | stuck off | accuracy |
|---|---|---|---|---|---|
| 2.0 | 2.50 | 1.17 | 30/60 | 0 | 0.075 |
| 2.0 | 2.55 | 0.88 | 32/60 | 0 | 0.083 |
| inf | 2.50 | 0.00 | 60/60 | 57 | 0.004 |
| inf | 2.55 | 0.05 | 59/60 | 56 | 0.007 |

A1 was a correctness fix — the two-hop return now lands LAG clear of the
refractory wall instead of on it, where §3.4's slack decided — with no
measurable cost. That closes A1/A2.

## 2. The accumulator dies at the inherited floor

With no leak the potential integrates every arrival since the last spike. Net
inhibition walks it down to MINIMUM_POTENTIAL and pins it there; nothing
restores it, because the leak was doing that silently. At V = −2.4, θ = 0.6,
Δ = 0.3125 and escape scale √(60/455) = 0.363, the hazard is
0.363·e^(−9.6) ≈ 2.5×10⁻⁵ a hop; over ~39 hops an epoch and 60 outputs that
predicts 0.06 output spikes an epoch, and 0.00–0.13 was measured.

It is a learning-rate runaway, and the floor rescues it (10,000 epochs, 3 seeds):

| floor | lr | spikes/epoch | silent | stuck off | accuracy |
|---|---|---|---|---|---|
| −2.4 | 0.002 | 0.91 | 92% | 50.7 | 0.056 |
| −2.4 | 0.0075 | 0.00 | 100% | 58.0 | 0.003 |
| −0.1 | 0.002 | 4.16 | 21% | 0.0 | 0.100 |
| −0.1 | 0.0075 | 0.76 | 50% | 0.0 | 0.100 |

The full 6 floors × 3 learning rates probe (`runs/mnist-accumulator-floor-lr`,
54 arms) puts the alive frontier at floor ≤ −0.4 for lr 0.002, ≤ −0.2 for
0.004, and −0.1 only for 0.0075. `stuck_off` is zero for every floor above
−0.8. Two lessons from that probe: a silent network earns the evidence critic's
uniform score log(1/10) = −2.3026 and *outranks* every live network that
guesses wrong, so a sweep ranked on score selects the dead; and output spikes
per epoch do not measure learning — trained networks here fire 0.3–1.3 an
epoch whether or not they learn. `stuck_off` was the one column that held.

## 3. Re-tuned, the accumulator learns

At 100,000 epochs, 9 cells × 3 seeds (`runs/archive/two-stage-2026-09-20/mnist-accumulator-100k`):

| lr | floor | accuracy | sd | spikes/epoch |
|---|---|---|---|---|
| 0.002 | −0.2 | 0.1658 | 0.0126 | 0.89 |
| 0.003 | −0.4 | 0.1657 | 0.0273 | 0.41 |
| 0.002 | −0.1 | 0.1626 | 0.0180 | 1.26 |
| 0.003 | −0.2 | 0.1597 | 0.0116 | 0.69 |
| 0.002 | −0.4 | 0.1584 | 0.0247 | 0.71 |
| 0.003 | −0.1 | 0.1558 | 0.0312 | 0.92 |
| 0.004 | −0.1 | 0.1485 | 0.0335 | 0.84 |
| 0.004 | −0.2 | 0.1419 | 0.0129 | 0.66 |
| 0.004 | −0.4 | 0.1401 | 0.0266 | 0.34 |

lr dominates and the floor barely matters once it is off −2.4. At three seeds
the top four are inside one seed-sd of each other.

At 1,000,000 epochs straight through, 10 seeds each:

| lr | floor | accuracy | sd | seeds, sorted |
|---|---|---|---|---|
| 0.002 | −0.2 | **0.2577** | 0.0139 | 0.2414 0.2473 0.2482 0.2504 0.2526 0.2545 0.2576 0.2617 0.2783 0.2849 |
| 0.003 | −0.4 | 0.2335 | 0.0176 | 0.2046 0.2196 0.2235 0.2240 0.2305 0.2310 0.2416 0.2427 0.2504 0.2671 |

(`runs/mnist-1m-lr002-floor02`, `runs/mnist-1m-lr003-floor04`.) Both cells
read higher on their first three seeds than on ten. Every arm was still
improving when it stopped — `mean` below `last_tenth` throughout — and
`stuck_off` is zero in all twenty.

## 4. The leak is ahead

| neuron | floor | lr | epochs | seeds | accuracy | sd | score |
|---|---|---|---|---|---|---|---|
| leaky, TAU 2.0 | −2.4 | 0.0075 | 100k | 3 | 0.1825 | 0.0084 | −2.1779 |
| leaky, TAU 2.0 | −2.4 | 0.0075 | 1M † | 3 | **0.2875** | 0.0043 | −1.9350 |
| accumulator | −0.2 | 0.002 | 100k | 3 | 0.1658 | 0.0126 | −2.3864 |
| accumulator | −0.2 | 0.002 | 1M | 10 | 0.2577 | 0.0139 | −2.0577 |

The gap widened with training: 0.017 at 100k, 0.030 at 1M. The accumulator's
best of ten seeds (0.2849) sits two ten-thousandths under the leaky worst of
three (0.2851).

Two caveats. The comparison is ten seeds against three, and both accumulator
cells showed sd 0.014–0.018 at ten where they had shown 0.002–0.023 at three;
with a conservative leaky sd of 0.015 the difference is about t = 3 — the
ordering holds, it is not overwhelming. And lr 0.0075, the leaky winner's rate,
was never run past 10,000 epochs on the accumulator; if the deficit is a
learning-rate mismatch rather than the leak, that is where it shows.
The second is closed in §4.1; the first waits on §6. † The leaky 1M is a
resumed run — see §5.

This is filed as [#17](https://github.com/byronshock/walnutbutter/issues/17):
§2.2 makes the accumulator the default and §2.3 fixes TAU infinite on Byron's
words; the measurement says the leak is worth three points on this task.

### 4.1 The leak's learning rate is not what the accumulator was missing

`mnist-1m-lr0075-floor01`, landed September 21 at 23:32: the accumulator at
the leak's lr 0.0075, floor −0.1 (the only floor alive at that rate in §2's
probe), ten seeds, 1M straight through.

| configuration | n | accuracy | sd | last_tenth | spk/ep | silent | stuck_off |
|---|---|---|---|---|---|---|---|
| accumulator, lr 0.0075, floor −0.1 | 10 | **0.1446** | 0.0171 | −2.3691 | 0.54 | 65% | 0 |

Per seed 0.1078 to 0.1677. Against 0.2577 at lr 0.002 that is 0.11 lower —
below where the lr 0.002 cell stood at 100,000 epochs — and on the log score it
sits below the uniform line (−2.37 against −2.30), which a silent network
beats. It is alive: `stuck_off` is zero in all ten, and `mean` is below
`last_tenth` in nine of ten, so it is still climbing, from a low base.

Every step up in lr has cost the accumulator accuracy at 1M, with the floor
moved each time to keep it alive: 0.2577 at 0.002 (floor −0.2), 0.2335 at
0.003 (floor −0.4), 0.1446 at 0.0075 (floor −0.1). The deficit in §4 is not a
learning-rate mismatch in the direction the leak would suggest. The grid never
looked *below* 0.002, mnist's default; that is the one lr corner still open.

### 4.2 Sixty hidden neurons buy nothing measurable

`mnist-1m-lr003-floor04-hidden60`, landed September 22 at 02:33: the lr 0.003
/ floor −0.4 cell with 60 hidden goo neurons, ten seeds, 1M straight through —
Goo(515 neurons; 395 in, 60 hidden, 60 out; 13,128–13,432 projections at
scaling factor 0.05, the wiring varying by seed), escape scale √(60/515) =
0.341.

| configuration | n | accuracy | sd | last_tenth | out spk/ep | silent | stuck_off | stuck_on |
|---|---|---|---|---|---|---|---|---|
| 60 hidden, lr 0.003, floor −0.4 | 10 | **0.2436** | 0.0136 | −2.1083 | 0.43 | 80% | 0 | 0–14 |
| 0 hidden, the same cell (§4) | 10 | 0.2335 | 0.0176 | −2.0945 | 0.41 | 79% | 0 | 0–2 |

+0.010 over the same cell without a hidden layer — inside one sd of either,
not separable at ten seeds. It does not reach the best 0-hidden cell (0.2577
at lr 0.002, floor −0.2), let alone the leak (0.2875). The hidden zone is
quiet: rate memory 0.08–0.11 against 0.18–0.24 at the outputs and 0.30–0.39
at the inputs. `stuck_on` runs 0–14 here against 0–3 across the 0-hidden 1M runs. The floor and lr tuned at 0 hidden held —
`stuck_off` zero in all ten. 15.2 epochs/s a thread; the launch probe's 11
was early-training activity, as §6 allowed.

Byron's question of September 16 — "How will we know if they are buying us
anything if they are always part of the economy?" — has its first answer
under the accumulator: on this cell, not measurably. A hidden layer tuned on
its own terms (count, floor, lr) is a different question from one dropped
into a cell tuned without it, and that question is open.

### 4.3 What the learning rate and the floor are bracketed to

Three accumulator cells have now run the full million, and each moved the
floor along with the learning rate to keep the network alive. That is the
first thing to say about them: **there is no pair at 1M that differs in the
floor alone**, so the two axes are confounded and what follows brackets them
together. Accuracy and the score are each arm's own record; the score is the
engine's last tenth.

| lr | floor | n | accuracy | sd | score | out spk/ep | silent | stuck_off |
|---|---|---|---|---|---|---|---|---|
| 0.002 | −0.2 | 10 | **0.2577** | 0.0139 | −2.0671 | 0.83 | 61% | 0 |
| 0.003 | −0.4 | 10 | 0.2335 | 0.0176 | −2.1045 | 0.48 | 77% | 0 |
| 0.0075 | −0.1 | 10 | 0.1446 | 0.0171 | −2.3691 | 0.54 | 65% | 0 |

Every arm at a given seed sees the same images in the same order (§4.5), so
the cells compare seed by seed: lr 0.002 over lr 0.003 is +0.0242 ± 0.0081,
winning 9 of 10; over lr 0.0075 it is +0.1130 ± 0.0046, winning 10 of 10.

**The learning rate is bracketed above and open below.** Accuracy over each
tenth of the run:

```
lr 0.002    0.130  0.183  0.212  0.221  0.230  0.240  0.248  0.247  0.253  0.258
lr 0.003    0.128  0.182  0.202  0.218  0.227  0.226  0.231  0.233  0.230  0.233
lr 0.0075   0.112  0.129  0.137  0.140  0.139  0.141  0.144  0.147  0.146  0.145
```

The higher rates are not faster early — all three are level in the first
tenth — they stop sooner. lr 0.0075 is flat from the fourth tenth, lr 0.003
from the sixth, and lr 0.002 is still climbing at a million. What a larger
rate costs here is a ceiling, and it does not buy the thing a larger rate is
usually paid for. Below 0.002 nothing has been run: it is mnist's default and
the lowest rate ever given to a long run, so the low end of the bracket is
the edge of the grid and not a measurement. Since the best cell had not
finished climbing, a lower rate is the live hypothesis rather than a
formality.

**The floor is bracketed only as fatal against alive.** At −2.4, the grid's
inherited floor, the accumulator pins and produces nothing (§2). Between −0.1
and −0.4 every one of the thirty 1M arms ended with `stuck_off` zero. Within
that band nothing ranks the floors: the one clean floor axis,
`mnist-accumulator-floor` at a fixed lr 0.002 over nine seeds, ran 10,000
epochs and landed between 0.067 and 0.083 — at or below the 0.1 chance line,
which separates dead from alive and nothing finer: a measurement taken where
nothing has learned yet ranks noise. The busiest cell is also the
best, which fits a shallow floor keeping more neurons in play, but with the
rate moving too that is a reading and not a result.

Two probes close the two corners, and neither is expensive beside a 1M sweep:
rates below 0.002 at a held floor of −0.2, and a floor axis at a held lr
0.002 run long enough to be above chance.

## 5. A resumed run is not the same run continued

The four 1M cells were first run as 100k then `--resume-from` for 900k, and
then, for cleanliness of the measuring window, as 1M straight through. The
same seed on the two paths should be the same run (§12.11). Row by row:
one-shot against the 100k run — identical, all 100 sampled rows, all six arms;
one-shot against the resumed run — identical for the 100 carried rows, then
divergent at 101,000, the first sample after the checkpoint, in all six.

| lr | floor | resumed (100k + 900k), seeds 7 8 9 | one-shot, seeds 7 8 9 |
|---|---|---|---|
| 0.002 | −0.2 | 0.2590 0.2566 0.2543 | 0.2526 0.2783 0.2576 |
| 0.003 | −0.4 | 0.2206 0.2571 0.2626 | 0.2046 0.2196 0.2504 |

Five kinds of state the engine keeps across epochs — `train` calls
`engine.reset(False)` each epoch — that `fast.build()` never loaded into a
resumed engine and `_save_network` never read back: the exploration stream
(saved null, `fast.sync_explore()` having no call site, so `fast.train:374`
reseeded); potentials, the per-synapse traces and the §6.7 `expected` debits;
the absolute times `fired_at`, `previous_fired_at` and `exposed_since` that
the refractory test and the hazard's elapsed run from; and the event heap —
never cleared by `reset`, so a spike in an epoch's last hop delivers in the
next — with per-edge `last_signal`; and under the leak (§2.3), `last_update`
and `trace_at`, the clocks its lazy decays run from, without which the traces
agreed and the weights still ended 5e-6 apart. The drive stream and input
position were fine. The Rust-path test asserted bookkeeping and resumed at
`seed + 1_000_000`, the convention §12.11 replaced. Every rust-sweep
checkpoint since D1 is affected, and `mnist-watch.py --resume` with it. The
resumed runs above are legitimate runs of their configuration, not the runs
their seeds name; the archive keeps them at
`runs/archive/two-stage-2026-09-20/`. Fixed in `6d2d422` on branch
`fix/resume-explore-state` — PR #19 into `code_follows_the_file`, closing
issue #18 — setters and a heap dump on the engine, the load
in `build()`, the write-back in `_save_network`, `pending_events` through
`train` — with a test that compares a resumed trace against the uninterrupted
one, under the accumulator and the leak with homeostasis off and on, and
passes; the suite passes (258, 1 skipped) and the engines still agree to the
bit on the mnist configuration. Landing needs a venv rebuild (`maturin develop --release`); nothing holds the
`.so` now.

## 6. Pending

- `mnist-1m-leaky-10seed`: the leaky configuration, ten seeds, 1M straight
  through — replaces the resumed control and makes §4 ten against ten.

Launched September 21 at 06:57 beside the lr 0.0075 run (06:58, landed 23:32
at 17.1 epochs/s a thread — §4.1) and the hidden-60 run (07:31, landed 02:33
on the 22nd at 15.2 — §4.2): thirty arms on sixteen cores, each holding one of
the 32 SMT threads rather than a core, so the whole-core rates (accumulator
~29 epochs/s, leaky ~14) ran at roughly 0.6. Alone on the machine from 02:33, it
was still running at 05:58 with 82,715 CPU-seconds an arm — an average under
12.09 epochs/s, so 976k–993k of a million done — and the machine was shut
down at 06:21 on the 22nd to install a graphics card, which was dead on
arrival. Arms write only on completion: all ten were lost within minutes of
finishing. The ten-against-ten of §4 waits on a relaunch of the same command,
~23 hours at the leaky neuron's 12.3 epochs/s, which no load lightens.

Relaunched September 22 at 17:58, the same command on ten workers: ten arms on
sixteen cores hold a core each rather than an SMT thread, so the arms run at the
leak's own rate instead of the 12.09 average the thirty-arm launch managed. The
engines were compared on this cell first, on the binary rebuilt that afternoon
after the resume fix landed — forty epochs, no disagreement. It is the last
sweep that will be able to lose a day this way: see §7.

## 7. A sweep cut short costs the epochs since its last checkpoint

The loss above was not that the machine went down; it was that an arm wrote
nothing until it was finished, so 98 per cent of a run was worth the same as
none of it. `--checkpoint-every N` (25,000 epochs by default) writes each arm's
checkpoint as the run goes, with the arm's record so far inside it, and an arm
that has a checkpoint and no `.csv` was cut short — so running the sweep's own
command again picks every such arm up where it stopped and finishes the epochs
asked for. Recovery is the same line again.

What makes this safe is §12.11. A checkpoint is a read of the engine, never a
touch, so a run that writes them is the run that does not: the tests hold a
checkpointed run against an unchecked one, score for score and weight for
weight, and hold an arm killed just after a checkpoint against the
uninterrupted arm, whose record it must reproduce row for row. The file is
moved into place once written, so a crash during a write leaves the checkpoint
before it standing rather than a half-written one.

One number is not recoverable that way. `fast.train` measures the last tenth of
the run it was asked for, so a leg that finishes an interrupted arm would
measure a tenth of the leg — a tenth of 25,000 epochs where its siblings report
a tenth of a million, which read beside them is simply a wrong number. A
continued arm's summary is therefore taken from its record over the last tenth
of the whole run, and says so in `last_tenth_over`. The fraction right comes
back exactly (each traced point is the mean over its own interval); the log
score is those points' mean, sampled every `--trace-every` epochs rather than
counted over all of them.

## Provenance

- The delta sweep this work waited on (`goo455-ff-delta-fine-100k`) completed
  waves 1–4 — 120 arms, seeds 1–12, all at hop 2.50 and TAU 2.0 — before the
  code landed mid-wave-5; wave 5 was killed with nothing on disk. Its `.md`
  shows only seeds 10–12 because the driver rewrote it each wave; the arms are
  all in `runs/`. It is a record of a configuration nothing runs any more.
- Throughput on the 7950X3D: accumulator arms 18 epochs/s at 27 arms, 29 at
  12; leaky arms 12–13. The sparse regime is cheaper to simulate.
- Engines compared on this configuration before any sweep (§12.8): object and
  Rust agree to the bit over 300 epochs at hop 2.55, read count, goo 455.
  The same at goo 515 (60 hidden) over 40 epochs, before the hidden-60 sweep.
- The test suite at `2be9814`: 254 passed, 1 skipped.
