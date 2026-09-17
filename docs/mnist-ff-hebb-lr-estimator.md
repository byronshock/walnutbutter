# mnist-ff-hebb-lr: the estimator's correlation over time (September 16, 2026)

Over the input-to-output synapses, the correlation of the weight change with the supervised direction d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is `mnist-ff-hebb-lr-estimator.png`.

| arm | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |
|---|---|---|---|---|---|---|
| lr0.003-threshold0.6-hidden_neurons0-temperature2 | 3 | +0.206 ± 0.023 | 0.462 | -0.006 | -2.72 | 0.072 |
| lr0.005-threshold0.6-hidden_neurons0-temperature2 | 3 | +0.179 ± 0.034 | 0.455 | +0.003 | -2.58 | 0.063 |
| lr0.01-threshold0.6-hidden_neurons0-temperature2 | 3 | +0.160 ± 0.006 | 0.455 | -0.001 | -2.44 | 0.063 |
| lr0.02-threshold0.6-hidden_neurons0-temperature2 | 3 | +0.141 ± 0.026 | 0.449 | +0.001 | -2.47 | 0.067 |
| lr0.03-threshold0.6-hidden_neurons0-temperature2 | 3 | +0.097 ± 0.016 | 0.433 | +0.001 | -2.50 | 0.059 |
| lr0.05-threshold0.6-hidden_neurons0-temperature2 | 3 | +0.059 ± 0.015 | 0.424 | +0.000 | -2.51 | 0.058 |
