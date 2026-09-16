# mnist-ff-tau: the estimator's correlation over time (September 16, 2026)

Over the input-to-output synapses, the correlation of the weight change with the supervised direction d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is `mnist-ff-tau-estimator.png`.

| arm | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |
|---|---|---|---|---|---|---|
| interval100-tau10-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.096 ± 0.034 | 0.421 | +0.010 | -2.61 | 0.094 |
| interval100-tau20-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.044 ± 0.022 | 0.416 | +0.009 | -2.62 | 0.074 |
| interval100-tau5-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.150 ± 0.021 | 0.442 | +0.010 | -2.88 | 0.122 |
| interval100-tau50-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.036 ± 0.039 | 0.415 | +0.011 | -2.49 | 0.050 |
