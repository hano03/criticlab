
### HalfCheetah-v5 (final probe at or before 300,000 steps)

| Group | n | Probe step(s) | Return | Norm. bias (median) | Norm. bias (mean) | Spearman(Q, G) | Explained var. | Episode length |
|---|---|---|---|---|---|---|---|---|
| DDPG | 5 | 300,000 | 7580 ± 829 (med 7371) | +0.025 ± 0.018 (med +0.031) | +0.205 ± 0.391 (med +0.038) | 0.53 ± 0.08 (med 0.50) | – | 1000 ± 0 (med 1000) |
| TD3 w/o clipped double-Q | 5 | 300,000 | 7635 ± 688 (med 7805) | -0.131 ± 0.015 (med -0.129) | -0.123 ± 0.009 (med -0.118) | 0.59 ± 0.03 (med 0.58) | – | 1000 ± 0 (med 1000) |
| TD3 | 5 | 300,000 | 7510 ± 418 (med 7690) | -0.258 ± 0.013 (med -0.255) | -0.257 ± 0.011 (med -0.255) | 0.63 ± 0.04 (med 0.64) | – | 1000 ± 0 (med 1000) |
| SAC | 5 | 300,000 | 9044 ± 817 (med 8617) | -0.098 ± 0.023 (med -0.095) | -0.092 ± 0.024 (med -0.078) | 0.72 ± 0.04 (med 0.72) | – | 1000 ± 0 (med 1000) |
| PPO, λ = 0.95 | 5 | 300,000 | 1912 ± 815 (med 1542) | -0.003 ± 0.032 (med +0.001) | +0.040 ± 0.088 (med +0.013) | 0.45 ± 0.06 (med 0.46) | +0.77 ± 0.20 (med +0.64) | 1000 ± 0 (med 1000) |

Welch's t-test against PPO, λ = 0.95 (per-seed values; p, two-sided):

| Group | Return | Norm. bias (median) | Norm. bias (mean) | Spearman(Q, G) | Explained var. | Episode length |
|---|---|---|---|---|---|---|
| DDPG | t = +10.90, p < 0.001 | t = +1.71, p = 0.136 | t = +0.92, p = 0.404 | t = +1.84, p = 0.105 | – | t = +nan, p = nan |
| TD3 w/o clipped double-Q | t = +12.00, p < 0.001 | t = -8.09, p < 0.001 | t = -4.11, p = 0.014 | t = +4.86, p = 0.002 | – | t = +nan, p = nan |
| TD3 | t = +13.67, p < 0.001 | t = -16.48, p < 0.001 | t = -7.51, p = 0.001 | t = +5.82, p < 0.001 | – | t = +nan, p = nan |
| SAC | t = +13.81, p < 0.001 | t = -5.39, p < 0.001 | t = -3.24, p = 0.026 | t = +8.61, p < 0.001 | – | t = +nan, p = nan |

### Hopper-v5 (final probe at or before 300,000 steps)

| Group | n | Probe step(s) | Return | Norm. bias (median) | Norm. bias (mean) | Spearman(Q, G) | Explained var. | Episode length |
|---|---|---|---|---|---|---|---|---|
| DDPG | 5 | 300,000 | 1124 ± 730 (med 926) | +0.946 ± 0.419 (med +0.822) | +1.453 ± 0.911 (med +1.132) | 0.54 ± 0.24 (med 0.63) | – | 410 ± 246 (med 341) |
| TD3 w/o clipped double-Q | 5 | 300,000 | 1597 ± 934 (med 1745) | +0.823 ± 0.632 (med +0.592) | +1.032 ± 0.727 (med +0.699) | 0.49 ± 0.29 (med 0.56) | – | 609 ± 337 (med 553) |
| TD3 | 5 | 300,000 | 1475 ± 1179 (med 890) | -0.176 ± 0.111 (med -0.216) | -0.119 ± 0.137 (med -0.163) | 0.73 ± 0.24 (med 0.85) | – | 482 ± 384 (med 278) |
| SAC | 5 | 300,000 | 2781 ± 990 (med 3265) | -0.061 ± 0.020 (med -0.056) | -0.058 ± 0.018 (med -0.054) | 0.86 ± 0.08 (med 0.85) | – | 755 ± 320 (med 946) |
| PPO, λ = 0.95 | 5 | 300,000 | 2727 ± 1036 (med 3259) | -0.002 ± 0.010 (med -0.001) | +0.001 ± 0.018 (med +0.001) | 0.89 ± 0.08 (med 0.90) | +0.99 ± 0.00 (med +0.99) | 563 ± 222 (med 616) |

Welch's t-test against PPO, λ = 0.95 (per-seed values; p, two-sided):

| Group | Return | Norm. bias (median) | Norm. bias (mean) | Spearman(Q, G) | Explained var. | Episode length |
|---|---|---|---|---|---|---|
| DDPG | t = -2.83, p = 0.025 | t = +5.06, p = 0.007 | t = +3.56, p = 0.023 | t = -3.08, p = 0.029 | – | t = -1.03, p = 0.332 |
| TD3 w/o clipped double-Q | t = -1.81, p = 0.108 | t = +2.92, p = 0.043 | t = +3.17, p = 0.034 | t = -2.91, p = 0.038 | – | t = +0.25, p = 0.808 |
| TD3 | t = -1.78, p = 0.113 | t = -3.49, p = 0.025 | t = -1.96, p = 0.120 | t = -1.35, p = 0.237 | – | t = -0.41, p = 0.695 |
| SAC | t = +0.08, p = 0.935 | t = -5.91, p = 0.001 | t = -5.22, p < 0.001 | t = -0.54, p = 0.605 | – | t = +1.10, p = 0.306 |
