# Sweep goo-250k (September 14, 2026)

Does goo learn once its potential axis is scaled with fan-in (AUTHORITY.md §5.2)? Three arms, 10 seeds each, 250,000 epochs per run, the reversal problem, the reinforce rule with the **perturb** eligibility (σ 0.1), the array engine, everything else at the defaults in `constants.py`. Every arm at seed *s* is given the same input stream (§4.5), so the arms are paired epoch by epoch and not merely on average.

Driver: `walnutbutter --goo --eligibility perturb --seeds 10 --seed 1 --input-seed 1 --epochs 250000 --engine arrays`, once per arm; report: `docs/goo-250k-report.py --name goo-250k --eligibility perturb`; figure: `goo-250k-score.png`. Every run's checkpoint and log is under `runs/goo-250k/` (not in git).

| arm | accuracy, last 25,000 epochs | range over seeds | accuracy, to date | stuck on | stuck off |
|---|---|---|---|---|---|
| goo, threshold and floor scaled | **0.5072** | 0.500 to 0.528 | 0.5072 | 41.4 | 27.0 |
| goo, flat threshold and floor | **0.5108** | 0.497 to 0.573 | 0.5155 | 62.2 | 6.3 |
| hex grid 8 x 10 (baseline) | **0.5201** | 0.499 to 0.550 | 0.5169 | 47.5 | 12.6 |

Paired on the seed, over the last 25,000 epochs of each run:

| comparison | mean difference | t |
|---|---|---|
| goo, threshold and floor scaled − goo, flat threshold and floor | -0.0036 | -0.46 |
| goo, threshold and floor scaled − hex grid 8 x 10 (baseline) | -0.0129 | -2.19 |
| goo, flat threshold and floor − hex grid 8 x 10 (baseline) | -0.0093 | -1.49 |

Per seed, accuracy over the last 25,000 epochs:

| seed | goo, threshold and floor scaled | goo, flat threshold and floor | hex grid 8 x 10 (baseline) |
|---|---|---|---|
| 1 | 0.501 | 0.500 | 0.511 |
| 2 | 0.500 | 0.573 | 0.546 |
| 3 | 0.509 | 0.515 | 0.515 |
| 4 | 0.528 | 0.521 | 0.548 |
| 5 | 0.507 | 0.500 | 0.506 |
| 6 | 0.506 | 0.500 | 0.550 |
| 7 | 0.509 | 0.500 | 0.499 |
| 8 | 0.508 | 0.501 | 0.508 |
| 9 | 0.502 | 0.497 | 0.509 |
| 10 | 0.502 | 0.501 | 0.509 |

