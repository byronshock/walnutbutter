# Sweep epoch-pop (September 12, 2026)

shallow_copy, 12x5 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 100,000 epochs per arm, 20 ms epochs. Fixed: input rate (/ms) 0.1. Swept: interval (ms), seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `epoch-pop-expected.png`, `epoch-pop-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/epoch-pop/` (not in git).

| interval (ms) | seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 0.532 | 0.536 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,490,668 | 60 | 0 | 0 |
| 5 | 2 | 0.532 | 0.529 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,511,602 | 60 | 0 | 0 |
| 5 | 3 | 0.535 | 0.536 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,476,814 | 60 | 0 | 0 |
| 5 | 4 | 0.533 | 0.530 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,484,637 | 60 | 0 | 0 |
| 5 | 5 | 0.535 | 0.537 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,510,720 | 60 | 0 | 0 |
| 5 | 6 | 0.534 | 0.539 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,408,695 | 60 | 0 | 0 |
| 19 | 1 | 0.795 | 0.798 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,871,030 | 60 | 0 | 4 |
| 19 | 2 | 0.800 | 0.800 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,977,768 | 60 | 0 | 2 |
| 19 | 3 | 0.792 | 0.795 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,659,953 | 60 | 0 | 8 |
| 19 | 4 | 0.797 | 0.801 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,752,688 | 60 | 0 | 20 |
| 19 | 5 | 0.802 | 0.802 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,763,974 | 60 | 0 | 2 |
| 19 | 6 | 0.791 | 0.800 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,875,360 | 60 | 0 | 11 |
| 20 | 1 | 0.805 | 0.811 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,924,563 | 60 | 0 | 0 |
| 20 | 2 | 0.810 | 0.820 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,025,418 | 60 | 0 | 12 |
| 20 | 3 | 0.810 | 0.818 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,887,587 | 60 | 0 | 4 |
| 20 | 4 | 0.805 | 0.810 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,835,297 | 60 | 0 | 10 |
| 20 | 5 | 0.807 | 0.813 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,934,471 | 60 | 0 | 6 |
| 20 | 6 | 0.805 | 0.812 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,882,391 | 60 | 0 | 3 |
| 21 | 1 | 0.816 | 0.825 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,933,248 | 60 | 0 | 16 |
| 21 | 2 | 0.820 | 0.823 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,944,973 | 60 | 0 | 14 |
| 21 | 3 | 0.797 | 0.800 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,046,273 | 60 | 0 | 7 |
| 21 | 4 | 0.802 | 0.818 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,834,180 | 60 | 0 | 10 |
| 21 | 5 | 0.816 | 0.820 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,911,284 | 60 | 0 | 26 |
| 21 | 6 | 0.813 | 0.822 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,049,485 | 60 | 0 | 0 |
| 26 | 1 | 0.846 | 0.848 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,243,528 | 60 | 0 | 7 |
| 26 | 2 | 0.842 | 0.853 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,279,252 | 60 | 0 | 39 |
| 26 | 3 | 0.839 | 0.841 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,265,269 | 60 | 0 | 11 |
| 26 | 4 | 0.837 | 0.853 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,278,133 | 60 | 0 | 2 |
| 26 | 5 | 0.846 | 0.851 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,299,380 | 60 | 0 | 6 |
| 26 | 6 | 0.841 | 0.846 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,176,860 | 60 | 0 | 19 |

Averaged over the 6 seeds (mean, then the range across seeds):

| interval (ms) | seeds | score last tenth | range | score to date | range | spikes | pinned at +1 |
|---|---|---|---|---|---|---|---|
| 5 | 6 | 0.535 | 0.529 to 0.539 | 0.533 | 0.532 to 0.535 | 1,480,523 | 0.0 |
| 19 | 6 | 0.800 | 0.795 to 0.802 | 0.796 | 0.791 to 0.802 | 3,816,796 | 0.0 |
| 20 | 6 | 0.814 | 0.810 to 0.820 | 0.807 | 0.805 to 0.810 | 3,914,954 | 0.0 |
| 21 | 6 | 0.818 | 0.800 to 0.825 | 0.811 | 0.797 to 0.820 | 3,953,240 | 0.0 |
| 26 | 6 | 0.849 | 0.841 to 0.853 | 0.842 | 0.837 to 0.846 | 4,257,070 | 0.0 |
