# Sweep mnist-evidence-h0 (September 16, 2026): the task without hidden neurons

Byron: "One thing I neglected to do is benchmark this task without any
hidden neurons. How will we know if they are buying us anything if they are
always part of the economy?" So the sweep of `mnist-evidence.md` again with
`--hidden-neurons 0`: a goo of 445, the 395 inputs and the 50 outputs and
nothing between them, the outputs hearing the inputs directly under the
scaled rule (§3.4) at 22 synapses a neuron in expectation; T in {1, 2, 4} x
LR in {0.001, 0.003, 0.006, 0.01, 0.03}, one seed, 10,000 epochs, threshold
0.6 with the floor at -2.4, Delta 0.455, fifteen arms on fifteen workers,
landed 02:52 MDT. Each cell gives this benchmark first and the 199-hidden
sweep beside it in parentheses; chance is each network's learning-off
reward at that temperature (seed 1, 500 epochs; `runs/mnist-evidence-h0-chance.json`),
which is lower here because a smaller goo is noisier at rest. The figure
(`mnist-evidence-h0-score.png`) draws the traces against it.

**Reward over the last tenth, no hidden (199 hidden)**

| T \ LR | 0.001 | 0.003 | 0.006 | 0.01 | 0.03 | learning off |
|---|---|---|---|---|---|---|
| 1 | -4.26 (-3.94) | -4.01 (-4.06) | -3.62 (-3.65) | -3.34 (-3.38) | -2.62 (-2.98) | -6.61 (-5.60) |
| 2 | -2.89 (-2.76) | -2.88 (-2.83) | -2.80 (-2.83) | -2.72 (-2.75) | -2.67 (-2.62) | -3.84 (-3.39) |
| 4 | -2.58 (-2.47) | -2.50 (-2.43) | -2.46 (-2.46) | -2.41 (-2.41) | -2.44 (-2.42) | -2.76 (-2.60) |

**Fraction right over the last tenth, the class critic, no hidden (199 hidden)**

| T \ LR | 0.001 | 0.003 | 0.006 | 0.01 | 0.03 | learning off |
|---|---|---|---|---|---|---|
| 1 | 0.07 (0.09) | 0.07 (0.09) | 0.07 (0.07) | 0.06 (0.08) | 0.05 (0.06) | 0.10 (0.06) |
| 2 | 0.07 (0.07) | 0.07 (0.07) | 0.07 (0.08) | 0.08 (0.09) | 0.05 (0.08) | 0.10 (0.06) |
| 4 | 0.07 (0.07) | 0.08 (0.09) | 0.08 (0.08) | 0.08 (0.08) | 0.06 (0.07) | 0.10 (0.06) |

**The output zone's final rate memory, no hidden: the fraction of epochs an output spiked in**

| T \ LR | 0.001 | 0.003 | 0.006 | 0.01 | 0.03 | learning off |
|---|---|---|---|---|---|---|
| 1 | 0.85 | 0.68 | 0.55 | 0.31 | 0.09 | about 0.9 |
| 2 | 0.89 | 0.80 | 0.74 | 0.60 | 0.34 | about 0.9 |
| 4 | 0.91 | 0.87 | 0.80 | 0.73 | 0.43 | about 0.9 |

## What it says

**The same picture with or without the hidden neurons.** Every arm ends
above its chance, most of all at T = 1 where chance is lowest; the fraction
right is 0.05 to 0.08 in every cell of both sweeps, against a learning-off
0.06 to 0.10; and the reward's rise is again the class sums evening out and
quieting, here visible in the output zone's rate memory falling with the
learning rate, from 0.85 to 0.09 at T = 1 -- at LR 0.03 the outputs spike in
one epoch in eleven, and a silent output zone is the uniform estimate. The
199 hidden neurons buy nothing measurable yet, but not because they are
idle: nothing is learned in either economy, so there is nothing for them to
add to. The benchmark stands to be rerun the day something learns; the
knob is in place for it.

Two small things the run found in the driver, fixed after it: a goo sized
by `--hidden-neurons` was called "hex grid" in the log's summary header,
and the record's last-epoch output counts were read from the mesh, which
the Rust engine does not write back, so they are zeros here.
