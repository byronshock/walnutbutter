# Sweep rate-drive (September 12, 2026)

shallow_copy, 12x5 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 100,000 epochs per arm, 20 ms epochs. Swept: input rate (/ms), seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `rate-drive-expected.png`, `rate-drive-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/rate-drive/` (not in git).

| input rate (/ms) | seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 1 | 0.786 | 0.791 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,832,055 | 60 | 0 | 18 |
| 0.1 | 2 | 0.788 | 0.792 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,888,162 | 60 | 0 | 6 |
| 0.1 | 3 | 0.782 | 0.787 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,954,516 | 60 | 0 | 4 |
| 0.1 | 4 | 0.779 | 0.804 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,673,358 | 60 | 0 | 12 |
| 0.1 | 5 | 0.785 | 0.789 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,985,079 | 60 | 0 | 7 |
| 0.1 | 6 | 0.780 | 0.800 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,709,961 | 60 | 0 | 16 |
| 0.125 | 1 | 0.787 | 0.784 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,045,344 | 60 | 0 | 7 |
| 0.125 | 2 | 0.788 | 0.797 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,128,412 | 60 | 0 | 9 |
| 0.125 | 3 | 0.787 | 0.798 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,015,787 | 60 | 0 | 5 |
| 0.125 | 4 | 0.786 | 0.786 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,071,711 | 60 | 0 | 32 |
| 0.125 | 5 | 0.797 | 0.805 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,968,430 | 60 | 0 | 26 |
| 0.125 | 6 | 0.795 | 0.801 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,889,573 | 60 | 0 | 9 |
| 0.15 | 1 | 0.799 | 0.801 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,027,375 | 60 | 0 | 9 |
| 0.15 | 2 | 0.793 | 0.792 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,156,250 | 60 | 0 | 2 |
| 0.15 | 3 | 0.800 | 0.807 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,988,127 | 60 | 0 | 7 |
| 0.15 | 4 | 0.796 | 0.800 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,070,756 | 60 | 0 | 0 |
| 0.15 | 5 | 0.797 | 0.805 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,059,191 | 60 | 0 | 5 |
| 0.15 | 6 | 0.791 | 0.805 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,023,833 | 60 | 0 | 1 |
| 0.175 | 1 | 0.801 | 0.807 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,185,346 | 60 | 0 | 6 |
| 0.175 | 2 | 0.800 | 0.807 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,263,887 | 60 | 0 | 44 |
| 0.175 | 3 | 0.794 | 0.792 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,137,058 | 60 | 0 | 1 |
| 0.175 | 4 | 0.798 | 0.806 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,239,096 | 60 | 0 | 1 |
| 0.175 | 5 | 0.802 | 0.806 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,259,972 | 60 | 0 | 6 |
| 0.175 | 6 | 0.786 | 0.789 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,103,904 | 60 | 0 | 14 |
| 0.2 | 1 | 0.795 | 0.803 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,282,358 | 60 | 0 | 37 |
| 0.2 | 2 | 0.796 | 0.801 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,364,954 | 60 | 0 | 4 |
| 0.2 | 3 | 0.790 | 0.798 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,190,828 | 60 | 0 | 13 |
| 0.2 | 4 | 0.800 | 0.810 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,335,196 | 60 | 0 | 2 |
| 0.2 | 5 | 0.799 | 0.804 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,354,159 | 60 | 0 | 20 |
| 0.2 | 6 | 0.783 | 0.791 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,315,210 | 60 | 0 | 20 |

Averaged over the 6 seeds (mean, then the range across seeds):

| input rate (/ms) | seeds | score last tenth | range | score to date | range | spikes | pinned at +1 |
|---|---|---|---|---|---|---|---|
| 0.1 | 6 | 0.794 | 0.787 to 0.804 | 0.783 | 0.779 to 0.788 | 3,840,522 | 0.0 |
| 0.125 | 6 | 0.795 | 0.784 to 0.805 | 0.790 | 0.786 to 0.797 | 4,019,876 | 0.0 |
| 0.15 | 6 | 0.802 | 0.792 to 0.807 | 0.796 | 0.791 to 0.800 | 4,054,255 | 0.0 |
| 0.175 | 6 | 0.801 | 0.789 to 0.807 | 0.797 | 0.786 to 0.802 | 4,198,210 | 0.0 |
| 0.2 | 6 | 0.801 | 0.791 to 0.810 | 0.794 | 0.783 to 0.800 | 4,307,118 | 0.0 |
