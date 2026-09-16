# mnist-ff-interval: the estimator's correlation over time (September 16, 2026)

Over the input-to-output synapses, the correlation of the weight change with the supervised direction d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is `mnist-ff-interval-estimator.png`.

| arm | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |
|---|---|---|---|---|---|---|
| interval100-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.207 ± 0.042 | 0.446 | +0.018 | -3.09 | 0.124 |
| interval20-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.201 ± 0.042 | 0.453 | +0.024 | -2.56 | 0.071 |
| interval25-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.199 ± 0.030 | 0.449 | +0.022 | -2.62 | 0.077 |
| interval30-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.216 ± 0.032 | 0.457 | +0.023 | -2.66 | 0.081 |
| interval35-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.219 ± 0.033 | 0.458 | +0.023 | -2.70 | 0.084 |
| interval50-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.217 ± 0.034 | 0.448 | +0.022 | -2.84 | 0.093 |
