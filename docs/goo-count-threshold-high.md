# Sweep goo-count-threshold-high (September 14, 2026)

goo neurons × THRESHOLD, 10 seeds, the copy problem (AUTHORITY.md §8), the reinforce rule with the hebb eligibility, the Rust wave loop (§6.15), homeostasis and un-sticking at the command line's constants. Every arm at seed *s* is given the same input stream (§4.5). The floor follows the threshold at -4 × THRESHOLD, so floor / theta is -4 in every cell; goo then scales both by its fan-in over 18 (§5.2).

Driver: `docs/rust-sweep.py --name goo-count-threshold-high --problem copy --goo ... --threshold ... --floor-ratio -4.0 --eligibility hebb --seed 1 ... 10 --epochs N`; report: `docs/goo-grid-report.py --name goo-count-threshold-high`; figure: `goo-count-threshold-high-score.png`. Every arm's trace and summary is under `runs/goo-count-threshold-high/` (not in git).

**Accuracy over the last tenth, mean over seeds**

| goo neurons \ THRESHOLD | 0.5 | 0.75 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|
| 64 | 0.572 | 0.632 | 0.643 | 0.605 | 0.612 |
| 80 | 0.634 | 0.612 | 0.597 | 0.638 | 0.580 |
| 100 | 0.632 | 0.563 | 0.603 | 0.615 | 0.587 |
| 120 | 0.620 | 0.618 | 0.617 | 0.623 | 0.637 |

**Range over seeds**

| goo neurons \ THRESHOLD | 0.5 | 0.75 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|
| 64 | 0.501–0.648 | 0.548–0.700 | 0.506–0.749 | 0.501–0.661 | 0.500–0.673 |
| 80 | 0.504–0.673 | 0.504–0.734 | 0.498–0.672 | 0.554–0.684 | 0.497–0.679 |
| 100 | 0.570–0.663 | 0.498–0.666 | 0.500–0.668 | 0.500–0.669 | 0.501–0.667 |
| 120 | 0.502–0.675 | 0.497–0.664 | 0.514–0.687 | 0.499–0.687 | 0.500–0.687 |

**Neurons stuck on / off at the end, mean over seeds**

| goo neurons \ THRESHOLD | 0.5 | 0.75 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|
| 64 | 14 | 0 | 4 | 15 | 11 |
| 80 | 0 | 17 | 14 | 0 | 32 |
| 100 | 0 | 37 | 18 | 10 | 34 |
| 120 | 11 | 11 | 0 | 11 | 11 |

**Neurons stuck off at the end, mean over seeds**

| goo neurons \ THRESHOLD | 0.5 | 0.75 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|
| 64 | 4 | 0 | 13 | 29 | 33 |
| 80 | 8 | 5 | 7 | 30 | 21 |
| 100 | 0 | 33 | 38 | 31 | 35 |
| 120 | 2 | 77 | 66 | 66 | 55 |

**What a neuron starts at: theta (floor is the ratio times this)**

| goo neurons \ THRESHOLD | 0.5 | 0.75 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|
| 64 | 1.75 | 2.62 | 3.50 | 5.25 | 7.00 |
| 80 | 2.19 | 3.29 | 4.39 | 6.58 | 8.78 |
| 100 | 2.75 | 4.12 | 5.50 | 8.25 | 11.00 |
| 120 | 3.31 | 4.96 | 6.61 | 9.92 | 13.22 |

Best cell: goo neurons 64, THRESHOLD 1: 0.6433 over 10 seeds.

