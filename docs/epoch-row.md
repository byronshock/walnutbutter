# Sweep epoch-row (September 12, 2026)

shallow_copy, 12x5 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 100,000 epochs per arm, 20 ms epochs. Fixed: input rate (/ms) 0.1. Swept: interval (ms), seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `epoch-row-expected.png`, `epoch-row-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/epoch-row/` (not in git).

| interval (ms) | seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 0.530 | 0.530 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,410,505 | 60 | 0 | 0 |
| 5 | 2 | 0.527 | 0.529 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,393,194 | 60 | 0 | 0 |
| 5 | 3 | 0.528 | 0.525 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,422,940 | 60 | 0 | 0 |
| 5 | 4 | 0.529 | 0.527 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,402,914 | 60 | 0 | 0 |
| 5 | 5 | 0.530 | 0.529 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,410,599 | 60 | 0 | 0 |
| 5 | 6 | 0.526 | 0.534 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,376,339 | 60 | 0 | 0 |
| 19 | 1 | 0.776 | 0.779 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,826,006 | 60 | 0 | 9 |
| 19 | 2 | 0.777 | 0.785 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,697,928 | 60 | 0 | 8 |
| 19 | 3 | 0.771 | 0.779 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,902,633 | 60 | 0 | 22 |
| 19 | 4 | 0.775 | 0.786 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,625,867 | 60 | 0 | 14 |
| 19 | 5 | 0.771 | 0.777 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,757,870 | 60 | 0 | 4 |
| 19 | 6 | 0.771 | 0.778 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,889,187 | 60 | 0 | 8 |
| 20 | 1 | 0.786 | 0.791 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,832,055 | 60 | 0 | 18 |
| 20 | 2 | 0.788 | 0.792 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,888,162 | 60 | 0 | 6 |
| 20 | 3 | 0.782 | 0.787 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,954,516 | 60 | 0 | 4 |
| 20 | 4 | 0.779 | 0.804 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,673,358 | 60 | 0 | 12 |
| 20 | 5 | 0.785 | 0.789 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,985,079 | 60 | 0 | 7 |
| 20 | 6 | 0.780 | 0.800 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,709,961 | 60 | 0 | 16 |
| 21 | 1 | 0.788 | 0.795 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,963,728 | 60 | 0 | 5 |
| 21 | 2 | 0.792 | 0.795 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,918,474 | 60 | 0 | 22 |
| 21 | 3 | 0.791 | 0.792 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,988,003 | 60 | 0 | 5 |
| 21 | 4 | 0.794 | 0.803 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,910,432 | 60 | 0 | 6 |
| 21 | 5 | 0.798 | 0.804 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,875,031 | 60 | 0 | 14 |
| 21 | 6 | 0.791 | 0.803 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,943,304 | 60 | 0 | 49 |
| 26 | 1 | 0.830 | 0.832 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,320,112 | 60 | 0 | 2 |
| 26 | 2 | 0.823 | 0.837 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,367,088 | 60 | 0 | 6 |
| 26 | 3 | 0.814 | 0.816 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,160,251 | 60 | 0 | 19 |
| 26 | 4 | 0.826 | 0.834 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,166,184 | 60 | 0 | 10 |
| 26 | 5 | 0.829 | 0.835 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,381,988 | 60 | 0 | 10 |
| 26 | 6 | 0.818 | 0.829 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,102,648 | 60 | 0 | 35 |

Averaged over the 6 seeds (mean, then the range across seeds):

| interval (ms) | seeds | score last tenth | range | score to date | range | spikes | pinned at +1 |
|---|---|---|---|---|---|---|---|
| 5 | 6 | 0.529 | 0.525 to 0.534 | 0.528 | 0.526 to 0.530 | 1,402,748 | 0.0 |
| 19 | 6 | 0.780 | 0.777 to 0.786 | 0.774 | 0.771 to 0.777 | 3,783,248 | 0.0 |
| 20 | 6 | 0.794 | 0.787 to 0.804 | 0.783 | 0.779 to 0.788 | 3,840,522 | 0.0 |
| 21 | 6 | 0.799 | 0.792 to 0.804 | 0.792 | 0.788 to 0.798 | 3,933,162 | 0.0 |
| 26 | 6 | 0.830 | 0.816 to 0.837 | 0.823 | 0.814 to 0.830 | 4,249,712 | 0.0 |
