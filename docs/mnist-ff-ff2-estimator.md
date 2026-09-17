# mnist-ff-ff2: the estimator's correlation over time (September 16, 2026)

Over the input-to-output synapses, the correlation of the weight change with the supervised direction d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is `mnist-ff-ff2-estimator.png`.

| arm | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |
|---|---|---|---|---|---|---|
| interval100-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.336 ± 0.010 | 0.483 | +0.032 | -2.92 | 0.112 |
