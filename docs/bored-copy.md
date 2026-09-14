# Sweep bored-copy (September 12, 2026)

shallow_copy, 12x5 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 20,000 epochs per arm, 20 ms epochs. Fixed: rows 5. Swept: bored after (ms), seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `bored-copy-expected.png`, `bored-copy-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/bored-copy/` (not in git).

| bored after (ms) | seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1 | 0.537 | 0.532 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,766,948 | 60 | 0 | 3 |
| 10 | 2 | 0.531 | 0.540 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,822,479 | 60 | 0 | 8 |
| 10 | 3 | 0.537 | 0.538 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,698,491 | 60 | 0 | 13 |
| 10 | 4 | 0.541 | 0.545 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,675,619 | 60 | 0 | 39 |
| 10 | 5 | 0.535 | 0.542 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,856,469 | 60 | 0 | 8 |
| 10 | 6 | 0.535 | 0.542 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,665,208 | 60 | 0 | 0 |
| 15 | 1 | 0.551 | 0.548 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,156,220 | 60 | 0 | 22 |
| 15 | 2 | 0.549 | 0.554 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,157,477 | 60 | 0 | 118 |
| 15 | 3 | 0.548 | 0.551 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,154,454 | 60 | 0 | 38 |
| 15 | 4 | 0.550 | 0.552 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,152,350 | 60 | 0 | 88 |
| 15 | 5 | 0.549 | 0.552 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,197,060 | 60 | 0 | 49 |
| 15 | 6 | 0.548 | 0.555 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,124,791 | 60 | 0 | 0 |
| 20 | 1 | 0.554 | 0.553 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,882,921 | 60 | 0 | 69 |
| 20 | 2 | 0.552 | 0.555 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,862,249 | 60 | 0 | 336 |
| 20 | 3 | 0.552 | 0.551 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,857,308 | 60 | 0 | 56 |
| 20 | 4 | 0.555 | 0.557 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,853,280 | 60 | 0 | 340 |
| 20 | 5 | 0.555 | 0.558 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,864,570 | 60 | 0 | 108 |
| 20 | 6 | 0.551 | 0.558 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,829,421 | 60 | 0 | 21 |
| 25 | 1 | 0.556 | 0.556 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,682,934 | 60 | 0 | 154 |
| 25 | 2 | 0.554 | 0.557 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,675,514 | 60 | 0 | 481 |
| 25 | 3 | 0.553 | 0.552 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,671,610 | 60 | 0 | 56 |
| 25 | 4 | 0.555 | 0.557 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,673,468 | 60 | 0 | 411 |
| 25 | 5 | 0.556 | 0.560 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,683,038 | 60 | 0 | 196 |
| 25 | 6 | 0.552 | 0.559 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,647,269 | 60 | 0 | 31 |
| 30 | 1 | 0.552 | 0.548 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,553,849 | 60 | 0 | 102 |
| 30 | 2 | 0.551 | 0.555 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,539,226 | 60 | 0 | 326 |
| 30 | 3 | 0.550 | 0.548 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,534,748 | 60 | 0 | 87 |
| 30 | 4 | 0.553 | 0.555 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,534,049 | 60 | 0 | 196 |
| 30 | 5 | 0.552 | 0.555 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,547,546 | 60 | 0 | 225 |
| 30 | 6 | 0.550 | 0.555 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,508,469 | 60 | 0 | 41 |

Averaged over the 6 seeds (mean, then the range across seeds):

| bored after (ms) | seeds | score last tenth | range | score to date | range | spikes | pinned at +1 |
|---|---|---|---|---|---|---|---|
| 10 | 6 | 0.540 | 0.532 to 0.545 | 0.536 | 0.531 to 0.541 | 2,747,536 | 0.0 |
| 15 | 6 | 0.552 | 0.548 to 0.555 | 0.549 | 0.548 to 0.551 | 2,157,059 | 0.0 |
| 20 | 6 | 0.556 | 0.551 to 0.558 | 0.553 | 0.551 to 0.555 | 1,858,292 | 0.0 |
| 25 | 6 | 0.557 | 0.552 to 0.560 | 0.554 | 0.552 to 0.556 | 1,672,306 | 0.0 |
| 30 | 6 | 0.552 | 0.548 to 0.555 | 0.551 | 0.550 to 0.553 | 1,536,314 | 0.0 |
