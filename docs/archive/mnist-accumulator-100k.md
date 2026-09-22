# mnist-accumulator-100k: mnist on the goo, 100,000 epochs an arm, the Rust wave loop (§6.15), hazard eligibility

Mean score over the last tenth of each run, averaged over seeds.

| minimum_potential \ lr | 0.002 | 0.003 | 0.004 |
|---|---|---|---|
| -0.4 | -2.1729 | -2.2469 | -2.3583 |
| -0.2 | -2.4477 | -2.6188 | -2.3985 |
| -0.1 | -2.8099 | -2.4056 | -2.2805 |

| arm | first tenth | last tenth | best |
|---|---|---|---|
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed9 | -4.8457 | -2.0400 | -0.6724 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed8 | -4.6362 | -2.0541 | -0.2694 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed7 | -5.2513 | -2.0572 | -0.2290 |
| interval100-lr0.003-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed9 | -5.2721 | -2.0627 | -0.5854 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed9 | -4.9319 | -2.0856 | -0.0361 |
| interval100-lr0.003-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed9 | -3.9030 | -2.0966 | -0.2204 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed9 | -2.7339 | -2.1798 | -0.1506 |
| interval100-lr0.003-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed9 | -4.1956 | -2.2214 | -0.3234 |
| interval100-lr0.003-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed7 | -4.7249 | -2.2980 | -0.1825 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed9 | -3.7687 | -2.3253 | -0.2030 |
| interval100-lr0.003-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed7 | -4.6922 | -2.3309 | -0.0918 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed8 | -4.1776 | -2.3535 | -0.0605 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed7 | -4.2372 | -2.3789 | -0.4141 |
| interval100-lr0.003-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed8 | -5.4632 | -2.3800 | -0.4611 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed9 | -4.7542 | -2.3999 | -0.1019 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed7 | -4.2592 | -2.4078 | -0.0885 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed9 | -3.6069 | -2.4251 | -0.1388 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed8 | -3.6601 | -2.4319 | -0.1810 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed7 | -5.0082 | -2.4481 | -0.6076 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed7 | -3.1633 | -2.4632 | -0.4516 |
| interval100-lr0.003-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed8 | -2.6126 | -2.4747 | -0.1259 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed8 | -7.6255 | -2.5356 | -0.2238 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed8 | -4.5463 | -2.6918 | -0.1117 |
| interval100-lr0.003-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed8 | -5.8223 | -2.7894 | -0.0446 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed8 | -5.5243 | -2.8130 | -0.0408 |
| interval100-lr0.003-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed7 | -5.3697 | -3.1603 | -0.3478 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed7 | -3.7900 | -3.3127 | -0.1303 |
