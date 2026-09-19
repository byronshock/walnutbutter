# goo455-ff-delta-100k: the estimator's correlation over time (September 19, 2026)

Over the input-to-output synapses, the correlation of the weight change with the supervised
direction d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is
`goo455-ff-delta-100k-delta.png`.

Every arm below shares one configuration and differs only in delta: lr 0.0075, interval 100,
threshold 0.6, floor -2.4, hazard eligibility, the evidence critic, the count read, scaled wiring,
tau 2.0, 60 outputs, no hidden neurons, 100,000 epochs. 90 arms in all. Deltas 0.455 and above
carry 10 seeds; 0.1 through 0.4, run September 19, carry 5.

| delta | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |
|---|---|---|---|---|---|---|
| 0.1 | 5 | +0.114 ± 0.033 | 0.439 | +0.012 | -2.24 | 0.131 |
| 0.2 | 5 | +0.247 ± 0.029 | 0.467 | +0.015 | -2.12 | 0.196 |
| 0.25 | 5 | +0.311 ± 0.033 | 0.474 | +0.019 | -2.10 | 0.198 |
| 0.3 | 5 | +0.333 ± 0.026 | 0.480 | +0.022 | -2.13 | 0.200 |
| 0.35 | 5 | +0.360 ± 0.023 | 0.487 | +0.017 | -2.16 | 0.197 |
| 0.4 | 5 | +0.370 ± 0.019 | 0.493 | +0.020 | -2.19 | 0.179 |
| 0.455 | 10 | +0.363 ± 0.033 | 0.487 | +0.015 | -2.24 | 0.182 |
| 0.6 | 10 | +0.340 ± 0.016 | 0.475 | +0.018 | -2.39 | 0.138 |
| 0.8 | 10 | +0.285 ± 0.026 | 0.468 | +0.012 | -2.56 | 0.109 |
| 1 | 10 | +0.219 ± 0.023 | 0.456 | +0.004 | -2.82 | 0.100 |
| 1.1 | 10 | +0.203 ± 0.021 | 0.452 | +0.003 | -2.90 | 0.094 |
| 1.2 | 10 | +0.186 ± 0.030 | 0.445 | +0.001 | -3.01 | 0.097 |

**Two plateaus that overlap between 0.3 and 0.35.** The correlation is flat from 0.35 to 0.455
(+0.360, +0.370, +0.363, inside its own ±0.02 to ±0.03), and the fraction right is flat from 0.2
to 0.35 (0.196 to 0.200). Only 0.3 and 0.35 sit on both. Below 0.2 and above 0.455 every column
falls together.

**The two metrics part above 0.35,** and the parting is the finding: the correlation keeps rising
to its peak at 0.4 while the fraction right has already dropped from 0.200 to 0.179. The estimator
measures which way the weights move, not whether the read decodes what they carry, so taken alone
it would choose a delta that is measurably worse at the task. Where the two disagree, §9.12 is the
diagnostic and the fraction right is the result.

**The windowed correlation is near zero at every delta** -- +0.001 to +0.022, against cumulative
values reaching +0.370. By the last quarter of 100,000 epochs no delta is still moving the weights
along the supervised direction; the cumulative figure is earned early and then held. That is a
statement about where this knob stops working, not about the network's health.

**The best-measured region is the least-well-sampled.** The peak lies entirely among the 5-seed
arms, and the differences that would place it are about 0.02 -- the size of the spread itself.
