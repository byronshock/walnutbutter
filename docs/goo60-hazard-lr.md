# Sweep goo60-hazard-lr (September 15, 2026)

LR against 10 seeds of goo (AUTHORITY.md §3.4), the copy problem (§8), the reinforce rule with the hazard eligibility, the Rust wave loop (§6.15), homeostasis and un-sticking at the command line's constants, everything else at the defaults in `constants.py`. Goo(60 neurons, 651 projections at P 0.2; 8 in, 8 out, zones apart). Every level at seed *s* is given the same input stream (§4.5), so levels pair epoch by epoch. About 3,379 epochs a second an arm.

Driver: `docs/rust-sweep.py --name goo60-hazard-lr --problem copy --goo 60 --eligibility hazard --lr ... --seed 1 ... 10 --epochs N`; report: `docs/goo-threshold-report.py --name goo60-hazard-lr --knob lr`; figure: `goo60-hazard-lr-score.png`. Every arm's trace and summary is under `runs/goo60-hazard-lr/` (not in git).

| lr | accuracy, last tenth | over seeds | stuck on | stuck off | vs default |
|---|---|---|---|---|---|
| 0.0275 | 0.6564 | 0.594 to 0.767 | 0.0 | 0.0 | — |
| 0.04 | 0.6674 | 0.618 to 0.762 | 0.0 | 0.0 | — |
| 0.05 | 0.6711 | 0.611 to 0.717 | 0.0 | 0.0 | — |
| 0.0525 | **0.6760** | 0.627 to 0.776 | 0.1 | 0.0 | — |

Best level: lr 0.0525, 0.6760 over the last tenth, averaged over 10 seeds.

