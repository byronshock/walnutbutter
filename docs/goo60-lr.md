# Sweep goo60-lr (September 14, 2026)

LR against 10 seeds of goo (AUTHORITY.md §3.4), the copy problem (§8), the reinforce rule with the hebb eligibility, the Rust wave loop (§6.15), homeostasis and un-sticking at the command line's constants, everything else at the defaults in `constants.py`. Goo(60 neurons fully connected, 3476 connections; 8 in, 8 out, no direct projection). Every level at seed *s* is given the same input stream (§4.5), so levels pair epoch by epoch. About 7,082 epochs a second an arm.

Driver: `docs/rust-sweep.py --name goo60-lr --problem reversal --goo --eligibility hebb --lr ... --seed 1 ... 10 --epochs N`; report: `docs/goo-threshold-report.py --name goo60-lr --knob lr`; figure: `goo60-lr-score.png`. Every arm's trace and summary is under `runs/goo60-lr/` (not in git).

| lr | accuracy, last tenth | over seeds | stuck on | stuck off | vs default |
|---|---|---|---|---|---|
| 0.003 | 0.5003 | 0.499 to 0.503 | 0.0 | 48.9 | -0.0014 (t -1.0) |
| 0.005 | 0.5010 | 0.500 to 0.508 | 0.0 | 47.1 | -0.0007 (t -0.3) |
| 0.0075 | 0.4999 | 0.499 to 0.500 | 0.0 | 47.9 | -0.0018 (t -1.0) |
| 0.015 | 0.5003 | 0.500 to 0.502 | 0.8 | 42.8 | -0.0013 (t -0.8) |
| 0.0225 | 0.5004 | 0.500 to 0.503 | 0.8 | 48.0 | -0.0012 (t -0.7) |
| 0.03 | **0.5017** | 0.500 to 0.517 | 0.0 | 47.6 | the default |
| 0.225 | 0.5000 | 0.500 to 0.500 | 7.2 | 37.4 | -0.0016 (t -1.0) |

Best level: lr 0.03, 0.5017 over the last tenth, averaged over 10 seeds.

