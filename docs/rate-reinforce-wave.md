# Sweep rate-reinforce-wave (September 12, 2026)

shallow_copy, 12x5 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 100,000 epochs per arm, 20 ms epochs. Fixed: sigma 0.1. Swept: input rate (/ms), seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `rate-reinforce-wave-expected.png`, `rate-reinforce-wave-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/rate-reinforce-wave/` (not in git).

| input rate (/ms) | seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 1 | 0.558 | 0.557 | 1.000 | nan | nan | nan | nan | 0 | 0 | 2,976,901 | 60 | 0 | 95 |
| 0.1 | 2 | 0.551 | 0.558 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,729,464 | 60 | 0 | 106 |
| 0.1 | 3 | 0.518 | 0.523 | 1.000 | nan | nan | nan | nan | 0 | 0 | 13,233,288 | 60 | 0 | 0 |
| 0.1 | 4 | 0.544 | 0.559 | 1.000 | nan | nan | nan | nan | 0 | 0 | 6,153,612 | 60 | 0 | 69 |
| 0.1 | 5 | 0.519 | 0.521 | 1.000 | nan | nan | nan | nan | 0 | 0 | 13,288,255 | 60 | 0 | 0 |
| 0.1 | 6 | 0.555 | 0.560 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,817,112 | 60 | 0 | 118 |
| 0.125 | 1 | 0.514 | 0.518 | 1.000 | nan | nan | nan | nan | 0 | 0 | 13,943,202 | 60 | 0 | 0 |
| 0.125 | 2 | 0.514 | 0.516 | 1.000 | nan | nan | nan | nan | 0 | 0 | 13,880,649 | 60 | 0 | 0 |
| 0.125 | 3 | 0.560 | 0.566 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,667,220 | 60 | 0 | 74 |
| 0.125 | 4 | 0.530 | 0.562 | 1.000 | nan | nan | nan | nan | 0 | 0 | 10,476,147 | 60 | 0 | 126 |
| 0.125 | 5 | 0.514 | 0.518 | 1.000 | nan | nan | nan | nan | 0 | 0 | 13,976,942 | 60 | 0 | 0 |
| 0.125 | 6 | 0.515 | 0.513 | 1.000 | nan | nan | nan | nan | 0 | 0 | 13,899,332 | 60 | 0 | 0 |
| 0.15 | 1 | 0.513 | 0.517 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,350,786 | 60 | 0 | 0 |
| 0.15 | 2 | 0.539 | 0.562 | 1.000 | nan | nan | nan | nan | 0 | 0 | 9,058,992 | 60 | 0 | 105 |
| 0.15 | 3 | 0.513 | 0.515 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,165,577 | 60 | 0 | 0 |
| 0.15 | 4 | 0.515 | 0.516 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,323,818 | 60 | 0 | 0 |
| 0.15 | 5 | 0.515 | 0.512 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,143,087 | 60 | 0 | 0 |
| 0.15 | 6 | 0.534 | 0.561 | 1.000 | nan | nan | nan | nan | 0 | 0 | 10,201,221 | 60 | 0 | 106 |
| 0.175 | 1 | 0.515 | 0.513 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,475,147 | 60 | 0 | 0 |
| 0.175 | 2 | 0.515 | 0.507 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,437,968 | 60 | 0 | 0 |
| 0.175 | 3 | 0.521 | 0.570 | 1.000 | nan | nan | nan | nan | 0 | 0 | 12,983,163 | 60 | 0 | 160 |
| 0.175 | 4 | 0.516 | 0.521 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,461,584 | 60 | 0 | 0 |
| 0.175 | 5 | 0.530 | 0.564 | 1.000 | nan | nan | nan | nan | 0 | 0 | 11,351,348 | 60 | 0 | 151 |
| 0.175 | 6 | 0.513 | 0.514 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,502,584 | 60 | 0 | 0 |
| 0.2 | 1 | 0.513 | 0.518 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,664,089 | 60 | 0 | 0 |
| 0.2 | 2 | 0.515 | 0.516 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,700,592 | 60 | 0 | 0 |
| 0.2 | 3 | 0.511 | 0.511 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,844,348 | 60 | 0 | 0 |
| 0.2 | 4 | 0.566 | 0.569 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,232,203 | 60 | 0 | 59 |
| 0.2 | 5 | 0.513 | 0.512 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,741,894 | 60 | 0 | 0 |
| 0.2 | 6 | 0.513 | 0.517 | 1.000 | nan | nan | nan | nan | 0 | 0 | 14,642,652 | 60 | 0 | 0 |

Averaged over the 6 seeds (mean, then the range across seeds):

| input rate (/ms) | seeds | score last tenth | range | score to date | range | spikes | pinned at +1 |
|---|---|---|---|---|---|---|---|
| 0.1 | 6 | 0.547 | 0.521 to 0.560 | 0.541 | 0.518 to 0.558 | 7,366,439 | 0.0 |
| 0.125 | 6 | 0.532 | 0.513 to 0.566 | 0.525 | 0.514 to 0.560 | 11,640,582 | 0.0 |
| 0.15 | 6 | 0.530 | 0.512 to 0.562 | 0.521 | 0.513 to 0.539 | 12,707,247 | 0.0 |
| 0.175 | 6 | 0.532 | 0.507 to 0.570 | 0.518 | 0.513 to 0.530 | 13,701,966 | 0.0 |
| 0.2 | 6 | 0.524 | 0.511 to 0.569 | 0.522 | 0.511 to 0.566 | 12,970,963 | 0.0 |
