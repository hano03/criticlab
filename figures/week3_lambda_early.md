
### Hopper-v5 (per-seed mean over probes from 20,000 to 250,000 steps)

| Group | n | Probes per seed | Return | Norm. bias (median) | Norm. bias (mean) | Spearman(Q, G) | Explained var. | Episode length |
|---|---|---|---|---|---|---|---|---|
| PPO, λ = 0.95 | 5 | 24 | 1219 ± 482 (med 961) | +0.009 ± 0.003 (med +0.010) | +0.045 ± 0.017 (med +0.048) | 0.88 ± 0.04 (med 0.89) | +0.96 ± 0.02 (med +0.97) | 303 ± 66 (med 270) |
| PPO, λ = 0.8 | 5 | 24 | 751 ± 625 (med 391) | +0.036 ± 0.029 (med +0.033) | +0.063 ± 0.037 (med +0.053) | 0.85 ± 0.05 (med 0.86) | +0.98 ± 0.00 (med +0.98) | 224 ± 135 (med 141) |
| PPO, λ = 1.0 | 5 | 24 | 788 ± 171 (med 684) | -0.015 ± 0.009 (med -0.018) | +0.059 ± 0.027 (med +0.068) | 0.85 ± 0.02 (med 0.84) | +0.84 ± 0.04 (med +0.84) | 212 ± 16 (med 207) |

Welch's t-test against PPO, λ = 0.95 (per-seed values; p, two-sided):

| Group | Return | Norm. bias (median) | Norm. bias (mean) | Spearman(Q, G) | Explained var. | Episode length |
|---|---|---|---|---|---|---|
| PPO, λ = 0.8 | t = -1.33, p = 0.224 | t = +2.10, p = 0.102 | t = +1.01, p = 0.355 | t = -1.12, p = 0.301 | t = +2.33, p = 0.075 | t = -1.18, p = 0.285 |
| PPO, λ = 1.0 | t = -1.88, p = 0.119 | t = -5.74, p = 0.003 | t = +1.01, p = 0.348 | t = -1.86, p = 0.110 | t = -7.00, p < 0.001 | t = -2.99, p = 0.035 |
