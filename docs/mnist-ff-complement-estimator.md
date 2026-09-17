# mnist-ff-complement: the estimator's correlation over time (September 16, 2026)

Over the input-to-output synapses, the correlation of the weight change with the supervised direction d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is `mnist-ff-complement-estimator.png`.

| arm | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |
|---|---|---|---|---|---|---|
| interval100-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.213 ± 0.023 | 0.455 | +0.017 | -3.21 | 0.116 |
| interval35-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.216 ± 0.024 | 0.459 | +0.020 | -2.79 | 0.081 |
