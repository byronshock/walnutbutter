# goo455-ff-delta-100k: mnist on the goo, 100,000 epochs an arm, the Rust wave loop (§6.15), hazard eligibility

| arm | first tenth | last tenth | best |
|---|---|---|---|
| interval100-lr0.0075-delta0.2-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed3 | -2.5621 | -1.6477 | -0.0613 |
| interval100-lr0.0075-delta0.4-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed3 | -3.3706 | -1.8956 | -0.1636 |
| interval100-lr0.0075-delta0.25-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed3 | -2.1414 | -1.9010 | -0.3428 |
| interval100-lr0.0075-delta0.35-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed3 | -3.3500 | -1.9342 | -0.3099 |
| interval100-lr0.0075-delta0.25-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed5 | -3.5989 | -2.0109 | -0.1040 |
| interval100-lr0.0075-delta0.35-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed1 | -3.7535 | -2.0178 | -0.0255 |
| interval100-lr0.0075-delta0.1-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed2 | -3.0970 | -2.0363 | -0.4725 |
| interval100-lr0.0075-delta0.2-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed2 | -2.6301 | -2.0732 | -0.1667 |
| interval100-lr0.0075-delta0.1-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed4 | -3.1394 | -2.1010 | -0.6065 |
| interval100-lr0.0075-delta0.1-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed5 | -3.5404 | -2.1386 | -0.5025 |
| interval100-lr0.0075-delta0.3-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed4 | -3.7046 | -2.1402 | -0.3895 |
| interval100-lr0.0075-delta0.2-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed5 | -3.1351 | -2.2042 | -0.4014 |
| interval100-lr0.0075-delta0.25-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed4 | -3.3374 | -2.2058 | -0.7936 |
| interval100-lr0.0075-delta0.2-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed1 | -2.6950 | -2.2236 | -0.1047 |
| interval100-lr0.0075-delta0.3-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed2 | -3.7607 | -2.2286 | -0.1950 |
| interval100-lr0.0075-delta0.4-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed2 | -2.6296 | -2.2473 | -0.5352 |
| interval100-lr0.0075-delta0.1-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed1 | -2.8523 | -2.2534 | -0.6148 |
| interval100-lr0.0075-delta0.1-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed3 | -3.2956 | -2.2576 | -0.5229 |
| interval100-lr0.0075-delta0.25-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed2 | -2.2597 | -2.2656 | -0.1830 |
| interval100-lr0.0075-delta0.4-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed5 | -2.7622 | -2.2656 | -0.4190 |
| interval100-lr0.0075-delta0.25-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed1 | -2.4390 | -2.2966 | -0.0402 |
| interval100-lr0.0075-delta0.3-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed3 | -2.6972 | -2.2999 | -0.1163 |
| interval100-lr0.0075-delta0.35-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed2 | -2.6902 | -2.3037 | -0.3347 |
| interval100-lr0.0075-delta0.2-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed4 | -3.8451 | -2.3139 | -0.6294 |
| interval100-lr0.0075-delta0.4-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed1 | -3.7852 | -2.3923 | -0.1306 |
| interval100-lr0.0075-delta0.35-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed5 | -3.7815 | -2.3939 | -0.4587 |
| interval100-lr0.0075-delta0.3-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed1 | -2.7723 | -2.4508 | -0.1551 |
| interval100-lr0.0075-delta0.3-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed5 | -2.6424 | -2.5382 | -0.2776 |
| interval100-lr0.0075-delta0.35-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed4 | -2.2673 | -2.7077 | -0.7034 |
| interval100-lr0.0075-delta0.4-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed4 | -3.1763 | -2.8014 | -0.6812 |
