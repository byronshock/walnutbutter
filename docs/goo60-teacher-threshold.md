# Sweep goo60-teacher-threshold (September 14, 2026)

TEACHER_THRESHOLD against 10 seeds of goo (AUTHORITY.md §3.4), the copy problem (§8), the reinforce rule with the wrong_hebb eligibility, the Rust wave loop (§6.15), homeostasis and un-sticking at the command line's constants, everything else at the defaults in `constants.py`. Goo(60 neurons fully connected, 3540 connections; 8 in, 8 out). Every level at seed *s* is given the same input stream (§4.5), so levels pair epoch by epoch. About 6,239 epochs a second an arm.

Driver: `docs/rust-sweep.py --name goo60-teacher-threshold --problem reversal --goo --eligibility wrong_hebb --teacher_threshold ... --seed 1 ... 10 --epochs N`; report: `docs/goo-threshold-report.py --name goo60-teacher-threshold --knob teacher_threshold`; figure: `goo60-teacher-threshold-score.png`. Every arm's trace and summary is under `runs/goo60-teacher-threshold/` (not in git).

| teacher_threshold | accuracy, last tenth | over seeds | stuck on | stuck off | vs default |
|---|---|---|---|---|---|
| 25 | **0.6297** | 0.504 to 0.667 | 0.0 | 5.8 | +0.0733 (t +3.0) |
| 30 | 0.5564 | 0.511 to 0.630 | 4.5 | 16.2 | +0.0000 (t +inf) |
| 35 | 0.5564 | 0.511 to 0.630 | 4.5 | 16.2 | +0.0000 (t +inf) |
| 40 | 0.5564 | 0.511 to 0.630 | 4.5 | 16.2 | the default |
| 45 | 0.5564 | 0.511 to 0.630 | 4.5 | 16.2 | +0.0000 (t +inf) |
| 50 | 0.5564 | 0.511 to 0.630 | 4.5 | 16.2 | +0.0000 (t +inf) |
| 60 | 0.5467 | 0.527 to 0.588 | 0.0 | 6.9 | -0.0097 (t -0.7) |
| 80 | 0.5467 | 0.527 to 0.588 | 0.0 | 6.9 | -0.0097 (t -0.7) |
| 100 | 0.5132 | 0.500 to 0.564 | 0.0 | 40.6 | -0.0432 (t -3.2) |

Best level: teacher_threshold 25, 0.6297 over the last tenth, averaged over 10 seeds.

