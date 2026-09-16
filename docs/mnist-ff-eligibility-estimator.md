# mnist-ff-eligibility: the estimator's correlation over time (September 16, 2026)

Over the input-to-output synapses, the correlation of the weight change with the supervised direction d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is `mnist-ff-eligibility-estimator.png`.

| arm | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |
|---|---|---|---|---|---|---|
| lr0.001-threshold0.6-hidden_neurons0-temperature2-eligibilityhazard | 10 | +0.109 ± 0.045 | 0.433 | +0.014 | -2.82 | 0.074 |
| lr0.001-threshold0.6-hidden_neurons0-temperature2-eligibilityhebb | 10 | +0.012 ± 0.026 | 0.424 | +0.001 | -2.88 | 0.074 |

## The reading

Ten seeds each at LR 0.001, T = 2, on the feedforward goo (AUTHORITY.md §8,
the sweeps of 04:25 MDT). Under the hazard eligibility the cumulative
correlation rises from +0.007 at 1,000 epochs through +0.042 at 5,000 and
+0.065 at 10,000 to +0.109 at 25,000, about as the square root of the
epochs; under hebb it ends at +0.012 with a window correlation of zero. So
the hazard's estimator is aligned with the supervised direction and
accumulates, and hebb's is not aligned at all. Neither moved the read in
25,000 epochs: rewards -2.82 and -2.88 against a chance of -3.40, the
fraction right 0.074 for both against 0.06, the output zone at its rest
throughout. The sign agreement of 0.43 for both, under a half, is the
label-blind per-output push deciding most signs. At this signal-to-noise a
correlation of a half would take about twenty times the epochs; the
learning rate scales signal and noise alike, so it is not the lever for
their ratio.
