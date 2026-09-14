# Sweep kind-teacher (September 12, 2026)

shallow_copy, 12x5 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 100,000 epochs per arm, 20 ms epochs. Fixed: input rate (/ms) 0.1. Swept: seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `kind-teacher-expected.png`, `kind-teacher-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/kind-teacher/` (not in git).

| seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.805 | 0.811 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,924,563 | 60 | 0 | 0 |
| 2 | 0.810 | 0.820 | 1.000 | nan | nan | nan | nan | 0 | 0 | 4,025,418 | 60 | 0 | 12 |
| 3 | 0.810 | 0.818 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,887,587 | 60 | 0 | 4 |
| 4 | 0.805 | 0.810 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,835,297 | 60 | 0 | 10 |
| 5 | 0.807 | 0.813 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,934,471 | 60 | 0 | 6 |
| 6 | 0.805 | 0.812 | 1.000 | nan | nan | nan | nan | 0 | 0 | 3,882,391 | 60 | 0 | 3 |
