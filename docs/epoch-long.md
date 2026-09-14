# Sweep epoch-long (September 12, 2026)

shallow_copy, 12x5 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 100,000 epochs per arm, 20 ms epochs. Fixed: input rate (/ms) 0.1. Swept: interval (ms), seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `epoch-long-expected.png`, `epoch-long-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/epoch-long/` (not in git).

| interval (ms) | seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 25 | 1 | 0.845 | 0.852 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,205,974 | 60 | 0 | 27 |
| 25 | 2 | 0.842 | 0.846 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,337,548 | 60 | 0 | 3 |
| 25 | 3 | 0.842 | 0.845 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,178,930 | 60 | 0 | 1 |
| 25 | 4 | 0.841 | 0.846 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,156,899 | 60 | 0 | 3 |
| 25 | 5 | 0.843 | 0.848 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,186,196 | 60 | 0 | 8 |
| 25 | 6 | 0.835 | 0.842 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,086,634 | 60 | 0 | 33 |
| 30 | 1 | 0.858 | 0.857 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,427,856 | 60 | 0 | 39 |
| 30 | 2 | 0.857 | 0.860 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,518,204 | 60 | 0 | 24 |
| 30 | 3 | 0.854 | 0.854 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,369,260 | 60 | 0 | 47 |
| 30 | 4 | 0.850 | 0.859 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,391,078 | 60 | 0 | 1 |
| 30 | 5 | 0.862 | 0.863 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,564,233 | 60 | 0 | 3 |
| 30 | 6 | 0.849 | 0.857 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,473,878 | 60 | 0 | 23 |
| 35 | 1 | 0.854 | 0.857 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,766,852 | 60 | 0 | 15 |
| 35 | 2 | 0.867 | 0.865 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,717,131 | 60 | 0 | 20 |
| 35 | 3 | 0.858 | 0.875 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,741,413 | 60 | 0 | 2 |
| 35 | 4 | 0.865 | 0.868 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,763,194 | 60 | 0 | 8 |
| 35 | 5 | 0.873 | 0.878 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,781,888 | 60 | 0 | 32 |
| 35 | 6 | 0.862 | 0.869 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,689,724 | 60 | 0 | 19 |
| 40 | 1 | 0.872 | 0.874 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,942,273 | 60 | 0 | 9 |
| 40 | 2 | 0.873 | 0.876 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,715,559 | 60 | 0 | 5 |
| 40 | 3 | 0.864 | 0.866 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,613,030 | 60 | 0 | 3 |
| 40 | 4 | 0.842 | 0.844 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,323,672 | 60 | 0 | 30 |
| 40 | 5 | 0.873 | 0.876 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,859,098 | 60 | 0 | 10 |
| 40 | 6 | 0.867 | 0.880 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,730,379 | 60 | 0 | 7 |
| 45 | 1 | 0.868 | 0.875 | 1.000 | nan | nan | nan | nan | 0 | 0 | 5,007,954 | 60 | 0 | 31 |
| 45 | 2 | 0.870 | 0.867 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,930,316 | 60 | 0 | 9 |
| 45 | 3 | 0.858 | 0.857 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,959,514 | 60 | 0 | 3 |
| 45 | 4 | 0.835 | 0.844 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,564,670 | 60 | 1 | 63 |
| 45 | 5 | 0.871 | 0.876 | 1.000 | nan | nan | nan | nan | 0 | 0 | 5,027,034 | 60 | 0 | 31 |
| 45 | 6 | 0.862 | 0.867 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,980,645 | 60 | 0 | 14 |

Averaged over the 6 seeds (mean, then the range across seeds):

| interval (ms) | seeds | score last tenth | range | score to date | range | spikes | pinned at +1 |
|---|---|---|---|---|---|---|---|
| 25 | 6 | 0.847 | 0.842 to 0.852 | 0.841 | 0.835 to 0.845 | 4,192,030 | 0.0 |
| 30 | 6 | 0.858 | 0.854 to 0.863 | 0.855 | 0.849 to 0.862 | 4,457,418 | 0.0 |
| 35 | 6 | 0.869 | 0.857 to 0.878 | 0.863 | 0.854 to 0.873 | 4,743,367 | 0.0 |
| 40 | 6 | 0.869 | 0.844 to 0.880 | 0.865 | 0.842 to 0.873 | 4,697,335 | 0.0 |
| 45 | 6 | 0.864 | 0.844 to 0.876 | 0.861 | 0.835 to 0.871 | 4,911,689 | 0.2 |
