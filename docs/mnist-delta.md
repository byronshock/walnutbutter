# Sweep mnist-delta (September 15, 2026)

DELTA against 10 seeds of goo (AUTHORITY.md §3.4), the mnist problem (§8), the reinforce rule with the hazard eligibility, the Rust wave loop (§6.15), homeostasis and un-sticking at the command line's constants, everything else at the defaults in `constants.py`. Goo(644 neurons, 20884 projections at P 0.05; 395 in, 50 out, zones apart). Every level at seed *s* is given the same input stream (§4.5), so levels pair epoch by epoch. About 5 epochs a second an arm.

Driver: `docs/rust-sweep.py --name mnist-delta --problem mnist --goo ... --eligibility hazard --delta ... --seed 1 ... 10 --epochs N`; report: `docs/goo-threshold-report.py --name mnist-delta --knob delta`; figure: `mnist-delta-score.png`. Every arm's trace and summary is under `runs/mnist-delta/` (not in git).

| delta | accuracy, last tenth | over seeds | stuck on | stuck off | vs default |
|---|---|---|---|---|---|
| 0.1 | 0.3732 | 0.329 to 0.418 | 4.3 | 146.2 | — |
| 0.15 | 0.3943 | 0.358 to 0.432 | 5.0 | 11.6 | — |
| 0.2 | 0.4380 | 0.423 to 0.450 | 10.1 | 0.8 | — |
| 0.225 | 0.4387 | 0.424 to 0.455 | 14.2 | 0.0 | — |
| 0.25 | 0.4451 | 0.429 to 0.456 | 22.1 | 0.0 | — |
| 0.275 | **0.4549** | 0.438 to 0.467 | 30.7 | 0.0 | — |

Best level: delta 0.275, 0.4549 over the last tenth, averaged over 10 seeds.

