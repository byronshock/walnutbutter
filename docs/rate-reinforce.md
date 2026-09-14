# Sweep rate-reinforce (September 12, 2026)

shallow_copy, 12x5 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 100,000 epochs per arm, 20 ms epochs. Fixed: sigma 0.1. Swept: input rate (/ms), seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `rate-reinforce-expected.png`, `rate-reinforce-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/rate-reinforce/` (not in git).

| input rate (/ms) | seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 1 | 0.544 | 0.544 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,404,968 | 60 | 0 | 6 |
| 0.1 | 2 | 0.543 | 0.524 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,349,001 | 60 | 0 | 1 |
| 0.1 | 3 | 0.530 | 0.521 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,257,181 | 60 | 0 | 0 |
| 0.1 | 4 | 0.561 | 0.595 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,397,040 | 60 | 0 | 7 |
| 0.1 | 5 | 0.538 | 0.541 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,388,744 | 60 | 0 | 2 |
| 0.1 | 6 | 0.530 | 0.527 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,245,745 | 60 | 0 | 1 |
| 0.125 | 1 | 0.536 | 0.538 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,514,083 | 60 | 0 | 4 |
| 0.125 | 2 | 0.545 | 0.552 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,484,463 | 60 | 0 | 2 |
| 0.125 | 3 | 0.542 | 0.560 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,554,532 | 60 | 0 | 2 |
| 0.125 | 4 | 0.537 | 0.531 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,367,537 | 60 | 0 | 1 |
| 0.125 | 5 | 0.546 | 0.548 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,540,257 | 60 | 0 | 5 |
| 0.125 | 6 | 0.551 | 0.564 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,640,664 | 60 | 0 | 0 |
| 0.15 | 1 | 0.538 | 0.546 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,631,437 | 60 | 0 | 4 |
| 0.15 | 2 | 0.546 | 0.541 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,626,642 | 60 | 0 | 3 |
| 0.15 | 3 | 0.522 | 0.525 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,535,388 | 60 | 0 | 5 |
| 0.15 | 4 | 0.539 | 0.553 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,676,884 | 60 | 0 | 1 |
| 0.15 | 5 | 0.537 | 0.560 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,535,168 | 60 | 0 | 2 |
| 0.15 | 6 | 0.535 | 0.542 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,547,182 | 60 | 0 | 3 |
| 0.175 | 1 | 0.549 | 0.536 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,821,966 | 60 | 0 | 7 |
| 0.175 | 2 | 0.536 | 0.524 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,772,952 | 60 | 0 | 9 |
| 0.175 | 3 | 0.534 | 0.520 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,596,518 | 60 | 0 | 4 |
| 0.175 | 4 | 0.533 | 0.532 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,707,564 | 60 | 0 | 1 |
| 0.175 | 5 | 0.552 | 0.557 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,657,280 | 60 | 0 | 7 |
| 0.175 | 6 | 0.543 | 0.541 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,707,113 | 60 | 0 | 1 |
| 0.2 | 1 | 0.553 | 0.569 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,836,920 | 60 | 0 | 5 |
| 0.2 | 2 | 0.525 | 0.528 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,783,963 | 60 | 0 | 1 |
| 0.2 | 3 | 0.532 | 0.529 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,697,331 | 60 | 0 | 3 |
| 0.2 | 4 | 0.545 | 0.557 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,797,197 | 60 | 0 | 1 |
| 0.2 | 5 | 0.542 | 0.561 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,700,755 | 60 | 0 | 4 |
| 0.2 | 6 | 0.537 | 0.527 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,778,453 | 60 | 0 | 1 |

Averaged over the 6 seeds (mean, then the range across seeds):

| input rate (/ms) | seeds | score last tenth | range | score to date | range | spikes | pinned at +1 |
|---|---|---|---|---|---|---|---|
| 0.1 | 6 | 0.542 | 0.521 to 0.595 | 0.541 | 0.530 to 0.561 | 2,340,446 | 0.0 |
| 0.125 | 6 | 0.549 | 0.531 to 0.564 | 0.543 | 0.536 to 0.551 | 2,516,923 | 0.0 |
| 0.15 | 6 | 0.545 | 0.525 to 0.560 | 0.536 | 0.522 to 0.546 | 2,592,117 | 0.0 |
| 0.175 | 6 | 0.535 | 0.520 to 0.557 | 0.541 | 0.533 to 0.552 | 2,710,566 | 0.0 |
| 0.2 | 6 | 0.545 | 0.527 to 0.569 | 0.539 | 0.525 to 0.553 | 2,765,770 | 0.0 |
