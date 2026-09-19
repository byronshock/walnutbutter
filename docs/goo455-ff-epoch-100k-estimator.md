# goo455-ff-epoch-100k: the estimator's correlation over time (September 18, 2026)

Over the input-to-output synapses, the correlation of the weight change with the supervised direction d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is `goo455-ff-epoch-100k-estimator.png`.

| arm | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |
|---|---|---|---|---|---|---|
| interval100-lr0.0075-threshold0.6-minimum_potential-2.4-hidden_neurons0 | 10 | +0.363 ± 0.031 | 0.487 | +0.015 | -2.24 | 0.182 |
| interval50-lr0.0075-threshold0.6-minimum_potential-2.4-hidden_neurons0 | 10 | +0.396 ± 0.027 | 0.492 | +0.026 | -2.27 | 0.148 |
| interval75-lr0.0075-threshold0.6-minimum_potential-2.4-hidden_neurons0 | 10 | +0.370 ± 0.025 | 0.482 | +0.021 | -2.25 | 0.163 |
