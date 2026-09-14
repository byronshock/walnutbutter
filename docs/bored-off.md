# Sweep bored-off (September 12, 2026)

shallow_copy, 12x5 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 20,000 epochs per arm, 20 ms epochs. Fixed: bored after (ms) 0, rows 5. Swept: seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `bored-off-expected.png`, `bored-off-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/bored-off/` (not in git).

| seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.758 | 0.770 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,614,150 | 60 | 0 | 15 |
| 2 | 0.760 | 0.762 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,525,286 | 60 | 0 | 21 |
| 3 | 0.751 | 0.779 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,564,932 | 60 | 0 | 9 |
| 4 | 0.754 | 0.758 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,585,192 | 60 | 0 | 20 |
| 5 | 0.761 | 0.777 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,571,241 | 60 | 0 | 5 |
| 6 | 0.762 | 0.786 | 1.000 | nan | nan | nan | nan | 0 | 0 | 1,509,058 | 60 | 0 | 3 |
