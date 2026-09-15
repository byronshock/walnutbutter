# Sweep goo60-hazard-lr-fine (September 15, 2026)

LR against 10 seeds of goo (AUTHORITY.md §3.4), the copy problem (§8), the reinforce rule with the hazard eligibility, the Rust wave loop (§6.15), homeostasis and un-sticking at the command line's constants, everything else at the defaults in `constants.py`. Goo(60 neurons, 651 projections at P 0.2; 8 in, 8 out, zones apart). Every level at seed *s* is given the same input stream (§4.5), so levels pair epoch by epoch. About 3,011 epochs a second an arm.

Driver: `docs/rust-sweep.py --name goo60-hazard-lr-fine --problem copy --goo 60 --eligibility hazard --lr ... --seed 1 ... 10 --epochs N`; report: `docs/goo-threshold-report.py --name goo60-hazard-lr-fine --knob lr`; figure: `goo60-hazard-lr-fine-score.png`. Every arm's trace and summary is under `runs/goo60-hazard-lr-fine/` (not in git).

| lr | accuracy, last tenth | over seeds | stuck on | stuck off | vs default |
|---|---|---|---|---|---|
| 0.02625 | 0.6694 | 0.614 to 0.718 | 0.0 | 0.1 | — |
| 0.0275 | 0.6564 | 0.594 to 0.767 | 0.0 | 0.0 | — |
| 0.02875 | 0.6681 | 0.620 to 0.729 | 0.0 | 0.0 | — |
| 0.03 | 0.6636 | 0.602 to 0.749 | 0.0 | 0.0 | — |
| 0.03125 | 0.6627 | 0.596 to 0.726 | 0.0 | 0.1 | — |
| 0.0325 | 0.6738 | 0.617 to 0.723 | 0.0 | 0.0 | — |
| 0.03375 | **0.6756** | 0.582 to 0.755 | 0.0 | 0.0 | — |
| 0.035 | 0.6676 | 0.606 to 0.725 | 0.0 | 0.1 | — |
| 0.03625 | 0.6617 | 0.579 to 0.723 | 0.0 | 0.3 | — |
| 0.0375 | 0.6742 | 0.629 to 0.747 | 0.0 | 0.0 | — |
| 0.03875 | 0.6609 | 0.607 to 0.726 | 0.0 | 0.0 | — |
| 0.04 | 0.6674 | 0.618 to 0.762 | 0.0 | 0.0 | — |

Best level: lr 0.03375, 0.6756 over the last tenth, averaged over 10 seeds.

