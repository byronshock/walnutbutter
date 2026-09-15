# Sweep goo-250k-hebb (September 14, 2026)

Does goo learn once its potential axis is scaled with fan-in (AUTHORITY.md §5.2)? Three arms, 10 seeds each, 250,000 epochs per run, the reversal problem, the reinforce rule with the **hebb** eligibility (σ 0, as the Teacher sets it), the Rust wave loop (§6.15), everything else at the defaults in `constants.py`. Every arm at seed *s* is given the same input stream (§4.5), so the arms are paired epoch by epoch and not merely on average.

Driver: `docs/rust-sweep.py --name goo-250k-hebb-<arm> --problem reversal [--goo [--no-scale-with-fan-in]] --eligibility hebb --seed 1 ... 10 --epochs 250000`, once per arm; report: `docs/goo-250k-report.py --name goo-250k-hebb --eligibility hebb --source rust`; figure: `goo-250k-hebb-score.png`. Every run's trace and summary is under `runs/goo-250k-hebb-<arm>/` (not in git).

| arm | accuracy, last 25,000 epochs | range over seeds | accuracy, to date | stuck on | stuck off |
|---|---|---|---|---|---|
| goo, threshold and floor scaled | **0.5131** | 0.500 to 0.553 | 0.5142 | 51.3 | 8.7 |
| goo, flat threshold and floor | **0.5062** | 0.500 to 0.524 | 0.5040 | 75.7 | 0.3 |
| hex grid 8 x 10 (baseline) | **0.5022** | 0.500 to 0.514 | 0.5039 | 61.5 | 7.1 |

Paired on the seed, over the last 25,000 epochs of each run:

| comparison | mean difference | t |
|---|---|---|
| goo, threshold and floor scaled − goo, flat threshold and floor | +0.0069 | +1.11 |
| goo, threshold and floor scaled − hex grid 8 x 10 (baseline) | +0.0109 | +1.93 |
| goo, flat threshold and floor − hex grid 8 x 10 (baseline) | +0.0040 | +1.21 |

Per seed, accuracy over the last 25,000 epochs:

| seed | goo, threshold and floor scaled | goo, flat threshold and floor | hex grid 8 x 10 (baseline) |
|---|---|---|---|
| 1 | 0.507 | 0.500 | 0.501 |
| 2 | 0.505 | 0.500 | 0.500 |
| 3 | 0.503 | 0.522 | 0.507 |
| 4 | 0.509 | 0.524 | 0.500 |
| 5 | 0.505 | 0.500 | 0.500 |
| 6 | 0.500 | 0.500 | 0.514 |
| 7 | 0.511 | 0.506 | 0.501 |
| 8 | 0.553 | 0.500 | 0.500 |
| 9 | 0.522 | 0.510 | 0.500 |
| 10 | 0.516 | 0.500 | 0.500 |

