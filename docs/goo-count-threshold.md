# Sweep goo-count-threshold (September 14, 2026)

goo neurons × THRESHOLD, 10 seeds, the copy problem (AUTHORITY.md §8), the reinforce rule with the wrong_hebb eligibility, the Rust wave loop (§6.15), homeostasis and un-sticking at the command line's constants. Every arm at seed *s* is given the same input stream (§4.5). The floor follows the threshold at -4 × THRESHOLD, so floor / theta is -4 in every cell; goo then scales both by its fan-in over 18 (§5.2).

Driver: `docs/rust-sweep.py --name goo-count-threshold --problem copy --goo ... --threshold ... --floor-ratio -4.0 --eligibility wrong_hebb --seed 1 ... 10 --epochs N`; report: `docs/goo-grid-report.py --name goo-count-threshold`; figure: `goo-count-threshold-score.png`. Every arm's trace and summary is under `runs/goo-count-threshold/` (not in git).

**Accuracy over the last tenth, mean over seeds**

| goo neurons \ THRESHOLD | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| 24 | 0.579 | 0.560 | 0.609 | 0.595 | 0.584 |
| 32 | 0.558 | 0.530 | 0.537 | 0.574 | 0.586 |
| 40 | 0.538 | 0.530 | 0.555 | 0.573 | 0.550 |
| 48 | 0.527 | 0.512 | 0.532 | 0.531 | 0.526 |
| 64 | 0.505 | 0.506 | 0.539 | 0.516 | 0.572 |
| 80 | 0.531 | 0.518 | 0.524 | 0.516 | 0.634 |

**Range over seeds**

| goo neurons \ THRESHOLD | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| 24 | 0.514–0.620 | 0.516–0.631 | 0.540–0.695 | 0.504–0.700 | 0.503–0.622 |
| 32 | 0.514–0.675 | 0.501–0.629 | 0.500–0.607 | 0.500–0.637 | 0.508–0.695 |
| 40 | 0.500–0.627 | 0.509–0.642 | 0.500–0.637 | 0.499–0.634 | 0.502–0.662 |
| 48 | 0.500–0.696 | 0.500–0.540 | 0.499–0.612 | 0.500–0.630 | 0.500–0.619 |
| 64 | 0.500–0.520 | 0.500–0.517 | 0.500–0.660 | 0.500–0.571 | 0.501–0.648 |
| 80 | 0.500–0.665 | 0.499–0.642 | 0.500–0.588 | 0.500–0.577 | 0.504–0.673 |

**Neurons stuck on / off at the end, mean over seeds**

| goo neurons \ THRESHOLD | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| 24 | 7 | 9 | 3 | 6 | 4 |
| 32 | 12 | 15 | 17 | 6 | 7 |
| 40 | 25 | 26 | 16 | 14 | 6 |
| 48 | 33 | 30 | 26 | 29 | 22 |
| 64 | 60 | 54 | 30 | 44 | 14 |
| 80 | 67 | 67 | 61 | 61 | 0 |

**Neurons stuck off at the end, mean over seeds**

| goo neurons \ THRESHOLD | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| 24 | 0 | 0 | 0 | 0 | 1 |
| 32 | 0 | 1 | 2 | 1 | 2 |
| 40 | 2 | 1 | 2 | 1 | 6 |
| 48 | 1 | 1 | 1 | 2 | 7 |
| 64 | 1 | 3 | 4 | 10 | 4 |
| 80 | 1 | 3 | 7 | 8 | 8 |

**What a neuron starts at: theta (floor is the ratio times this)**

| goo neurons \ THRESHOLD | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| 24 | 0.19 | 0.26 | 0.32 | 0.38 | 0.64 |
| 32 | 0.26 | 0.34 | 0.43 | 0.52 | 0.86 |
| 40 | 0.32 | 0.43 | 0.54 | 0.65 | 1.08 |
| 48 | 0.39 | 0.52 | 0.65 | 0.78 | 1.31 |
| 64 | 0.53 | 0.70 | 0.88 | 1.05 | 1.75 |
| 80 | 0.66 | 0.88 | 1.10 | 1.32 | 2.19 |

Best cell: goo neurons 80, THRESHOLD 0.5: 0.6335 over 10 seeds.

