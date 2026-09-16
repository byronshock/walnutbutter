# mnist-ff-lr100k: the estimator's correlation over time (September 16, 2026)

Over the input-to-output synapses, the correlation of the weight change with the supervised direction d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is `mnist-ff-lr100k-estimator.png`.

| arm | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |
|---|---|---|---|---|---|---|
| lr0.001-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.226 ± 0.035 | 0.461 | +0.010 | -2.72 | 0.082 |
| lr0.002-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.238 ± 0.024 | 0.462 | +0.011 | -2.66 | 0.086 |
| lr0.003-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.219 ± 0.036 | 0.451 | +0.009 | -2.60 | 0.087 |
| lr0.005-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.220 ± 0.023 | 0.448 | +0.007 | -2.50 | 0.092 |
| lr0.01-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.191 ± 0.027 | 0.431 | +0.005 | -2.43 | 0.081 |
| lr0.02-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.162 ± 0.028 | 0.425 | +0.008 | -2.34 | 0.063 |
| lr0.03-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.158 ± 0.025 | 0.422 | +0.006 | -2.33 | 0.063 |
