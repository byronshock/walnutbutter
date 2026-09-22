# mnist-accumulator-floor-lr: mnist on the goo, 10,000 epochs an arm, the Rust wave loop (§6.15), hazard eligibility

Mean score over the last tenth of each run, averaged over seeds.

| minimum_potential \ lr | 0.002 | 0.004 | 0.0075 |
|---|---|---|---|
| -2.4 | -4.3454 | -1.5704 | -2.3026 |
| -1.2 | -1.7485 | -1.9548 | -1.9155 |
| -0.8 | -6.2172 | -2.3734 | -2.3014 |
| -0.4 | -6.1333 | -2.2961 | -2.3665 |
| -0.2 | -3.0965 | -2.9339 | -2.1873 |
| -0.1 | -3.3155 | -2.8766 | -3.9894 |

| arm | first tenth | last tenth | best |
|---|---|---|---|
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed8 | -6.8415 | -0.0525 | -0.0525 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-1.2-hidden_neurons0-seed8 | -2.3371 | -0.5652 | -0.5652 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-1.2-hidden_neurons0-seed8 | -11.6682 | -0.8647 | -0.1107 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-1.2-hidden_neurons0-seed9 | -12.0102 | -1.0533 | -1.0533 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed9 | -5.1636 | -1.0716 | -0.1506 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed8 | -13.9797 | -1.3767 | -1.3767 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed9 | -3.8066 | -1.7128 | -0.9040 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed9 | -5.5866 | -1.7948 | -1.6260 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed9 | -3.0613 | -1.9166 | -1.4288 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed8 | -2.0674 | -1.9584 | -1.9584 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed8 | -0.0408 | -2.0477 | -0.0408 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed8 | -8.0895 | -2.0708 | -1.4055 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-1.2-hidden_neurons0-seed7 | -0.1035 | -2.0860 | -0.1035 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed8 | -13.6264 | -2.1357 | -1.3647 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.8-hidden_neurons0-seed8 | -2.9431 | -2.1495 | -2.0244 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-1.2-hidden_neurons0-seed7 | -3.5125 | -2.1587 | -1.8628 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed7 | -5.8222 | -2.1975 | -1.8549 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.8-hidden_neurons0-seed8 | -10.9111 | -2.2195 | -1.6883 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-1.2-hidden_neurons0-seed9 | -0.6850 | -2.2223 | -0.6591 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.8-hidden_neurons0-seed7 | -1.1141 | -2.2479 | -1.1141 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed7 | -2.1249 | -2.3026 | -2.1249 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed8 | -0.0042 | -2.3026 | -0.0042 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed9 | -4.5449 | -2.3026 | -2.3026 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-1.2-hidden_neurons0-seed7 | -5.5322 | -2.3278 | -2.2624 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.8-hidden_neurons0-seed7 | -4.6353 | -2.3278 | -1.8450 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed7 | -4.3767 | -2.3427 | -0.3605 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-1.2-hidden_neurons0-seed8 | -7.1825 | -2.3654 | -2.2373 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed9 | -2.5820 | -2.3845 | -2.0140 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.8-hidden_neurons0-seed9 | -3.3378 | -2.4270 | -2.1962 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed9 | -0.9782 | -2.4611 | -0.9782 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.8-hidden_neurons0-seed9 | -0.4879 | -2.6526 | -0.4879 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed7 | -3.6400 | -2.8089 | -1.8326 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.8-hidden_neurons0-seed8 | -8.0595 | -2.8475 | -0.1308 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed7 | -0.2428 | -3.1704 | -0.2428 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed9 | -4.0433 | -3.2004 | -2.4371 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-1.2-hidden_neurons0-seed9 | -4.5399 | -3.2132 | -1.3971 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed7 | -3.4484 | -3.2389 | -1.9199 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed7 | -7.1967 | -3.2618 | -1.6888 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed7 | -7.0320 | -3.3159 | -1.8294 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed8 | -10.2330 | -3.3212 | -1.5617 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed9 | -0.1388 | -3.3815 | -0.1388 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed7 | -7.3350 | -3.3918 | -0.8988 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed7 | -7.1576 | -3.5535 | -3.1551 |
| interval100-lr0.004-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed7 | -5.9533 | -3.6810 | -0.9078 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed9 | -7.7679 | -4.1084 | -0.2913 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed8 | -11.5255 | -4.2225 | -1.0225 |
| interval100-lr0.0075-delta0.3125-threshold0.6-minimum_potential-0.1-hidden_neurons0-seed8 | -8.1238 | -4.6210 | -1.8884 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.2-hidden_neurons0-seed9 | -0.5008 | -4.7426 | -0.5008 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed8 | -8.9398 | -5.2265 | -0.9483 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed9 | -1.0082 | -5.6358 | -0.2910 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.8-hidden_neurons0-seed7 | -3.4497 | -7.1333 | -2.1268 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-2.4-hidden_neurons0-seed8 | -19.7053 | -7.3898 | -1.9504 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.4-hidden_neurons0-seed7 | -4.5969 | -7.5377 | -0.6280 |
| interval100-lr0.002-delta0.3125-threshold0.6-minimum_potential-0.8-hidden_neurons0-seed9 | -6.6420 | -8.6709 | -1.1301 |
