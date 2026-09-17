# Sweep goo60-zones-sparse (September 14, 2026)

THRESHOLD × projection P, 10 seeds, the copy problem (AUTHORITY.md §8), the reinforce rule with the wrong_hebb eligibility, the Rust wave loop (§6.15), homeostasis and un-sticking at the command line's constants. Every arm at seed *s* is given the same input stream (§4.5). The floor follows the threshold at -4 × THRESHOLD, so floor / theta is -4 in every cell; goo then scales both by its fan-in over 18 (§5.2).

Driver: `docs/rust-sweep.py --name goo60-zones-sparse --problem copy --goo ... --threshold ... --floor-ratio -4.0 --eligibility wrong_hebb --seed 1 ... 10 --epochs N`; report: `docs/goo-grid-report.py --name goo60-zones-sparse`; figure: `goo60-zones-sparse-score.png`. Every arm's trace and summary is under `runs/goo60-zones-sparse/` (not in git).

**Accuracy over the last tenth, mean over seeds**

| THRESHOLD \ projection P | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.45 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.585 | 0.633 | 0.626 | 0.605 | 0.600 | 0.612 | 0.603 | 0.607 | 0.594 |
| 0.26 | 0.587 | 0.608 | 0.604 | 0.612 | 0.600 | 0.597 | 0.609 | 0.586 | 0.624 |
| 0.35 | 0.579 | 0.604 | 0.600 | 0.590 | 0.608 | 0.590 | 0.589 | 0.615 | 0.605 |
| 0.44 | 0.581 | 0.595 | 0.591 | 0.595 | 0.592 | 0.613 | 0.606 | 0.605 | 0.597 |

**Range over seeds**

| THRESHOLD \ projection P | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.45 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.543–0.669 | 0.586–0.696 | 0.600–0.688 | 0.565–0.671 | 0.531–0.643 | 0.565–0.655 | 0.534–0.644 | 0.574–0.692 | 0.533–0.649 |
| 0.26 | 0.550–0.638 | 0.546–0.666 | 0.573–0.667 | 0.534–0.693 | 0.546–0.654 | 0.517–0.635 | 0.541–0.675 | 0.508–0.625 | 0.557–0.678 |
| 0.35 | 0.542–0.661 | 0.532–0.668 | 0.538–0.652 | 0.543–0.628 | 0.509–0.654 | 0.514–0.660 | 0.515–0.668 | 0.578–0.668 | 0.556–0.665 |
| 0.44 | 0.546–0.617 | 0.547–0.647 | 0.543–0.662 | 0.568–0.631 | 0.538–0.669 | 0.554–0.654 | 0.549–0.654 | 0.559–0.641 | 0.500–0.631 |

**Neurons stuck on / off at the end, mean over seeds**

| THRESHOLD \ projection P | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.45 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.35 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| 0.44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Neurons stuck off at the end, mean over seeds**

| THRESHOLD \ projection P | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.45 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.35 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| 0.44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**What a neuron starts at: theta (floor is the ratio times this)**

| THRESHOLD \ projection P | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.45 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.11 | 0.03 | 0.07 | 0.10 | 0.18 | 0.16 | 0.24 | 0.23 | 0.23 |
| 0.26 | 0.14 | 0.04 | 0.09 | 0.13 | 0.23 | 0.20 | 0.32 | 0.30 | 0.30 |
| 0.35 | 0.19 | 0.06 | 0.12 | 0.17 | 0.31 | 0.27 | 0.43 | 0.41 | 0.41 |
| 0.44 | 0.24 | 0.07 | 0.15 | 0.22 | 0.39 | 0.34 | 0.54 | 0.51 | 0.51 |

Best cell: THRESHOLD 0.2, projection P 0.15: 0.6326 over 10 seeds.

**Stability by THRESHOLD** (full cells only): the mean over every projection P, the five-level band with the highest rolling mean, how many cells had every seed above 0.55, and the cell with the highest worst seed

| THRESHOLD | mean | best band | every seed > 0.55 | worst-seed leader |
|---|---|---|---|---|
| 0.2 | 0.6070 | 0.15–0.35: 0.6149 | 5 of 9 | projection P 0.2: 0.6255, worst 0.600 |
| 0.26 | 0.6029 | 0.2–0.4: 0.6042 | 2 of 9 | projection P 0.2: 0.6042, worst 0.573 |
| 0.35 | 0.5977 | 0.3–0.5: 0.6014 | 2 of 9 | projection P 0.45: 0.6153, worst 0.578 |
| 0.44 | 0.5973 | 0.3–0.5: 0.6028 | 3 of 9 | projection P 0.25: 0.5946, worst 0.568 |

Cells where every seed learned (above 0.55): THRESHOLD 0.2 / projection P 0.15, THRESHOLD 0.2 / projection P 0.2, THRESHOLD 0.2 / projection P 0.25, THRESHOLD 0.2 / projection P 0.35, THRESHOLD 0.2 / projection P 0.45, THRESHOLD 0.26 / projection P 0.2, THRESHOLD 0.26 / projection P 0.5, THRESHOLD 0.35 / projection P 0.45, THRESHOLD 0.35 / projection P 0.5, THRESHOLD 0.44 / projection P 0.25, THRESHOLD 0.44 / projection P 0.35, THRESHOLD 0.44 / projection P 0.45

