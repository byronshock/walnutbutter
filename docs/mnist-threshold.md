# Sweep mnist-threshold (September 15, 2026)

THRESHOLD against 5 seeds of goo (AUTHORITY.md §3.4), the mnist problem (§8), the reinforce rule with the hazard eligibility, the Rust wave loop (§6.15), homeostasis and un-sticking at the command line's constants, everything else at the defaults in `constants.py`. Goo(644 neurons, 20654 projections at P 0.05; 395 in, 50 out). Every level at seed *s* is given the same input stream (§4.5), so levels pair epoch by epoch. About 5 epochs a second an arm.

Driver: `docs/rust-sweep.py --name mnist-threshold --problem mnist --goo ... --eligibility hazard --threshold ... --seed 1 ... 5 --epochs N`; report: `docs/goo-threshold-report.py --name mnist-threshold --knob threshold`; figure: `mnist-threshold-score.png`. Every arm's trace and summary is under `runs/mnist-threshold/` (not in git).

Goo scales its potential axis by fan-in (§5.2), so the theta a goo neuron actually starts at is the level times 3.28, and the floor is MINIMUM_POTENTIAL × 3.28 = -3.28 throughout: sweeping THRESHOLD alone sweeps the floor-to-threshold ratio with it.

| THRESHOLD | theta on goo | floor / theta | accuracy, last tenth | over seeds | stuck on | stuck off | vs default |
|---|---|---|---|---|---|---|---|
| 0.45 | 1.475 | -2.22 | 0.0998 | 0.090 to 0.112 | 173.0 | 0.0 | — |
| 0.55 | 1.803 | -1.82 | 0.0891 | 0.078 to 0.095 | 181.4 | 0.0 | — |
| 0.6 | 1.967 | -1.67 | 0.0903 | 0.081 to 0.096 | 175.0 | 0.0 | — |
| 0.65 | 2.131 | -1.54 | **0.1000** | 0.092 to 0.111 | 165.0 | 0.0 | — |
| 0.7 | 2.294 | -1.43 | 0.0967 | 0.090 to 0.108 | 169.8 | 0.0 | — |
| 0.8 | 2.622 | -1.25 | 0.0974 | 0.088 to 0.108 | 164.0 | 0.0 | — |

Best level: threshold 0.65, 0.1000 over the last tenth, averaged over 5 seeds.

