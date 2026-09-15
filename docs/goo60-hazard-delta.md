# Sweep goo60-hazard-delta (September 15, 2026)

DELTA against 10 seeds of goo (AUTHORITY.md §3.4), the copy problem (§8), the reinforce rule with the hazard eligibility, the Rust wave loop (§6.15), homeostasis and un-sticking at the command line's constants, everything else at the defaults in `constants.py`. Goo(60 neurons, 651 projections at P 0.2; 8 in, 8 out, zones apart). Every level at seed *s* is given the same input stream (§4.5), so levels pair epoch by epoch. About 3,063 epochs a second an arm.

Driver: `docs/rust-sweep.py --name goo60-hazard-delta --problem copy --goo 60 --eligibility hazard --delta ... --seed 1 ... 10 --epochs N`; report: `docs/goo-threshold-report.py --name goo60-hazard-delta --knob delta`; figure: `goo60-hazard-delta-score.png`. Every arm's trace and summary is under `runs/goo60-hazard-delta/` (not in git).

| delta | accuracy, last tenth | over seeds | stuck on | stuck off | vs default |
|---|---|---|---|---|---|
| 0.7 | **0.6578** | 0.601 to 0.705 | 0.0 | 0.0 | — |
| 1.05 | 0.6196 | 0.562 to 0.687 | 0.0 | 0.0 | — |
| 1.4 | 0.5673 | 0.523 to 0.614 | 0.0 | 0.0 | — |

Best level: delta 0.7, 0.6578 over the last tenth, averaged over 10 seeds.

