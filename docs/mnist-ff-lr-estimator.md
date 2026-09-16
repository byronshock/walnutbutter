# mnist-ff-lr: the estimator's correlation over time (September 16, 2026)

Over the input-to-output synapses, the correlation of the weight change with the supervised direction d = P(pixel on | class) - P(pixel on) (AUTHORITY.md §8), mean over seeds; the figure is `mnist-ff-lr-estimator.png`.

| arm | seeds | corr, cumulative, final | sign agreement, final | corr, window, last quarter | reward, last tenth | right, last tenth |
|---|---|---|---|---|---|---|
| lr0.0005-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.087 ± 0.041 | 0.427 | +0.013 | -2.87 | 0.072 |
| lr0.001-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.109 ± 0.045 | 0.433 | +0.014 | -2.82 | 0.074 |
| lr0.002-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.126 ± 0.036 | 0.433 | +0.015 | -2.78 | 0.075 |
| lr0.005-threshold0.6-hidden_neurons0-temperature2 | 10 | +0.124 ± 0.031 | 0.431 | +0.011 | -2.73 | 0.077 |

## The reading

Ten seeds a learning rate under the hazard eligibility, T = 2, on the
feedforward goo (AUTHORITY.md §8). The cumulative correlation at 25,000
epochs is +0.087, +0.109, +0.126 and +0.124 at LR 0.0005, 0.001, 0.002 and
0.005: within a few hundredths across a tenfold range, rising a little with
the rate as the weights move further, with the window correlation over the
last quarter at +0.011 to +0.015 everywhere. The learning rate scales the
estimator's signal and noise alike, so it does not move their ratio; what it
moves is the drift -- at 0.005 the output zone has begun to quiet (rate
memory 0.75, two spikes a neuron against three at rest) and the reward is a
tenth higher for it (-2.73 against -2.87 at 0.0005; chance -3.40), while the
fraction right stays at 0.072 to 0.077 against 0.06. The arm at 0.001 equals
sweep 1's hazard arm to the bit on all ten seeds, the replicate the fixed
rate of sweep 1 was chosen for.
