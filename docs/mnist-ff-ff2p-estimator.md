# mnist-ff-ff2p: the estimator's correlation over time (September 17, 2026)

Over the input-to-output synapses, the correlation of the weight change with the supervised direction d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is `mnist-ff-ff2p-estimator.png`.

| arm | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |
|---|---|---|---|---|---|---|
| interval100-threshold0.6-hidden_neurons0-temperature2-projection0.25 | 10 | +0.462 ± 0.013 | 0.507 | +0.017 | -2.32 | 0.217 |
| interval100-threshold0.6-hidden_neurons0-temperature2-projection0.5 | 10 | +0.472 ± 0.022 | 0.512 | +0.017 | -2.32 | 0.212 |
