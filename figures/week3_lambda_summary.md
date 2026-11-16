
### Hopper-v5

| Group | n | Probe step(s) | Return | Norm. bias (median) | Norm. bias (mean) | Spearman(Q, G) | Explained var. |
|---|---|---|---|---|---|---|---|
| PPO, λ = 0.95 | 5 | 999,424 | 3278 ± 66 (med 3305) | +0.002 ± 0.001 (med +0.002) | +0.053 ± 0.098 (med +0.012) | 0.89 ± 0.05 (med 0.90) | +1.00 ± 0.00 (med +1.00) |
| PPO, λ = 0.8 | 5 | 999,424 | 2105 ± 1162 (med 2506) | -0.005 ± 0.024 (med -0.011) | +0.046 ± 0.074 (med +0.010) | 0.84 ± 0.08 (med 0.82) | +0.99 ± 0.01 (med +0.99) |
| PPO, λ = 1.0 | 5 | 999,424 | 1977 ± 1009 (med 1428) | -0.002 ± 0.011 (med -0.002) | +0.040 ± 0.052 (med +0.016) | 0.87 ± 0.09 (med 0.89) | +0.85 ± 0.29 (med +0.98) |

Welch's t-test against PPO, λ = 0.95 (per-seed values; p, two-sided):

| Group | Return | Norm. bias (median) | Norm. bias (mean) | Spearman(Q, G) | Explained var. |
|---|---|---|---|---|---|
| PPO, λ = 0.8 | t = -2.26, p = 0.087 | t = -0.64, p = 0.559 | t = -0.14, p = 0.895 | t = -1.12, p = 0.303 | t = -0.98, p = 0.370 |
| PPO, λ = 1.0 | t = -2.88, p = 0.045 | t = -0.69, p = 0.525 | t = -0.25, p = 0.807 | t = -0.47, p = 0.654 | t = -1.08, p = 0.343 |
