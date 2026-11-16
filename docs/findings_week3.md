# Week 3: PPO's state-value critic

*criticlab · 2026-11-16 · final for week 3 · reproduce with the commands in §7*

**Data.** `scripts/run_week3.py`, 20 runs, 1M steps, 10 probe episodes per probe, every 10k steps:
PPO with GAE λ = 0.95 (CleanRL's default) on Hopper-v5 and HalfCheetah-v5, plus λ = 0.8 and λ = 1.0
on Hopper, 5 seeds each. The cross-algorithm comparison uses the week-2 runs and the PPO runs cut at
300k steps, so step counts match. Tables come from `scripts/summarize_sweep.py`: each seed's value at
its final probe (PPO ends at step 999,424, the last whole 2,048-step rollout), then mean ± sample std
across seeds, with Welch's t-test on the per-seed values.

**What is probed.** PPO's critic is V(s) for its stochastic policy, trained on rewards divided by a
running scale σ. The probe compares σ·V(s) with Monte Carlo returns of the same stochastic policy.
The same (Q − G) / max(|G|, 10) normalisation as week 2 applies.

## 1. Is the implementation right?

| Environment | criticlab PPO, 1M steps | CleanRL `ppo_continuous_action.py`, 1M steps (v2 envs) |
|---|---|---|
| Hopper-v5 | 3,278 ± 66 | 2,231 ± 657 |
| HalfCheetah-v5 | 2,793 ± 1,260 (median 1,914) | 1,823 ± 928 |

Both are in or above CleanRL's published range. The comparison is approximate: CleanRL's figures are
for the v2 environments, and `ppo.py` bootstraps through time limits where CleanRL does not. The
InvertedPendulum check (Day 15) had already confirmed the σ·V unit conversion on a task with a known
value.

## 2. Is PPO's critic biased?

No, to within a fraction of a percent.

| | Hopper-v5 | HalfCheetah-v5 |
|---|---|---|
| Median normalised bias at 1M | +0.002 ± 0.001 | −0.003 ± 0.007 |
| Median normalised bias at 300k | −0.002 ± 0.010 | −0.003 ± 0.032 |

At 300k neither differs from zero (one-sample t-test on the 5 per-seed values: p = 0.68 and 0.84).
At 1M, Hopper's +0.002 is consistently above zero (p = 0.01), but it is 0.2% of the return. The *mean*
over states is larger on Hopper (+0.053 ± 0.098 at 1M, median across seeds +0.012): as in week 2, a
minority of states with small |G| dominates the mean, which is why the headline uses the median.

**Why this is expected.** The overestimation of DDPG and TD3 comes from an actor maximising a noisy
learned Q, then bootstrapping from that maximum (Fujimoto et al., 2018). PPO's critic has no
maximisation in its target: it regresses V(s) onto GAE λ-returns of the current policy, which are
mostly real rewards, from fresh rollouts of the states it is later probed on. So this result is a
second validation of the probe, after week 2's DDPG → TD3: it measures zero where theory predicts zero.

## 3. PPO beside the Q-critics (300k steps)

| Environment | Algorithm | Return | Median bias | Bias ≠ 0? (p) | Spearman(Q, G) |
|---|---|---|---|---|---|
| Hopper-v5 | DDPG | 1,124 ± 730 | +0.946 | 0.007 | 0.54 |
| Hopper-v5 | TD3 w/o clipped double-Q | 1,597 ± 934 | +0.823 | 0.044 | 0.49 |
| Hopper-v5 | TD3 | 1,475 ± 1,179 | −0.176 | 0.024 | 0.73 |
| Hopper-v5 | SAC | 2,781 ± 990 | −0.061 | 0.002 | 0.86 |
| Hopper-v5 | **PPO** | 2,727 ± 1,036 | **−0.002** | **0.68** | **0.89** |
| HalfCheetah-v5 | DDPG | 7,580 ± 829 | +0.025 | 0.036 | 0.53 |
| HalfCheetah-v5 | TD3 w/o clipped double-Q | 7,635 ± 688 | −0.131 | < 0.001 | 0.59 |
| HalfCheetah-v5 | TD3 | 7,510 ± 418 | −0.258 | < 0.001 | 0.63 |
| HalfCheetah-v5 | SAC | 9,044 ± 817 | −0.098 | < 0.001 | 0.72 |
| HalfCheetah-v5 | **PPO** | 1,912 ± 815 | **−0.003** | **0.84** | **0.45** |

*"Bias ≠ 0?": one-sample t-test of the 5 per-seed medians against 0, computed from the table's
rounded means and stds. All comparisons between algorithms below are exact Welch tests on the
per-seed values, from `summarize_sweep.py --ref ppo` (§7).*

- **PPO is the only algorithm whose bias is indistinguishable from zero in both environments.**
  On Hopper its bias differs from every other algorithm's: SAC (p = 0.001), DDPG (p = 0.007),
  TD3 (p = 0.025), TD3 without clipped double-Q (p = 0.043). On HalfCheetah it differs from SAC, TD3
  and TD3 without clipped double-Q (all p < 0.001) but not from DDPG (+0.025, p = 0.136), which is also
  close to zero there.
- **Unbiased is not the same as informative.** On HalfCheetah, PPO's critic has the lowest rank
  calibration of all five (Spearman 0.45): below SAC (0.72, p < 0.001), TD3 (0.63, p < 0.001) and TD3
  without clipped double-Q (0.59, p = 0.002); not distinguishable from DDPG (0.53, p = 0.105). It is
  right on average but poor at telling better states from worse. On Hopper it ranks as well as SAC
  (0.89 against 0.86, p = 0.605) and better than DDPG (0.54, p = 0.029) and TD3 without clipped
  double-Q (0.49, p = 0.038).
- **Returns.** On Hopper, PPO matches SAC at 300k (p = 0.935) and beats DDPG (p = 0.025). On
  HalfCheetah it is far behind every off-policy method (1,912 against 7,510–9,044, all p < 0.001), the
  usual sample-efficiency gap of on-policy learning.
- **Reading panel 3 with care.** PPO's HalfCheetah calibration line sits on the diagonal even though
  its Spearman is 0.45. Panel 3 pools all seeds' states, so the large differences in return *between*
  seeds spread the bins along the diagonal. Within one run, ranking is weaker than the panel suggests.
  This applies to the week-2 panels too.

## 4. GAE λ (Hopper, 1M steps)

| λ | Return | Median bias | Spearman | Explained variance |
|---|---|---|---|---|
| 0.8 | 2,105 ± 1,162 | −0.005 ± 0.024 | 0.84 ± 0.08 | +0.99 ± 0.01 |
| 0.95 | 3,278 ± 66 | +0.002 ± 0.001 | 0.89 ± 0.05 | +1.00 ± 0.00 |
| 1.0 | 1,977 ± 1,009 | −0.002 ± 0.011 | 0.87 ± 0.09 | +0.85 ± 0.29 |

- **Return: λ = 0.95 was best and by far the most consistent across seeds,** but the evidence is
  modest. Welch p = 0.087 against λ = 0.8 and 0.045 against λ = 1.0. With two comparisons, neither
  survives a Bonferroni correction (0.17 and 0.09). The direction agrees with the GAE paper, where
  intermediate λ worked best (Schulman et al., 2016).
- **Final bias does not depend on λ** (all p > 0.5). That fits theory: for the current policy,
  tabular TD(λ) has the same fixed point, V^π, for every λ. With function approximation the fixed
  point can depend on λ, but a flexible network trained to convergence on each rollout should leave
  little of that difference.

**Early training (window test).** Two patterns in the figures were tested on one number per seed:
the mean of each seed's probes from 20k to 250k steps.

| λ | Return | Median bias | Spearman | Explained variance | Episode length |
|---|---|---|---|---|---|
| 0.8 | 751 ± 625 | +0.036 ± 0.029 | 0.85 ± 0.05 | +0.98 ± 0.00 | 224 ± 135 |
| 0.95 | 1,219 ± 482 | +0.009 ± 0.003 | 0.88 ± 0.04 | +0.96 ± 0.02 | 303 ± 66 |
| 1.0 | 788 ± 171 | −0.015 ± 0.009 | 0.85 ± 0.02 | +0.84 ± 0.04 | 212 ± 16 |

*Welch against λ = 0.95: median bias p = 0.102 (λ = 0.8) and 0.003 (λ = 1.0); explained variance
p = 0.075 and < 0.001; episode length p = 0.285 and 0.035.*

1. **Early bias is ordered by λ, but the λ = 0.8 effect is not established.** The figures suggested
   that λ = 0.8 overestimates early. Its early median bias is indeed the highest (+0.036 against
   +0.009), but with p = 0.102 and a large spread across seeds that is a direction, not a result.
   λ = 1.0, unexpectedly, sits slightly *below* zero (−0.015, p = 0.003 against λ = 0.95). So the three
   means fall in order of λ: smaller λ, more positive early bias.
   - *Possible explanation (post hoc, untested):* two effects pull in opposite directions while the
     policy improves. Critic lag pushes V below the current policy's returns for every λ. With smaller λ,
     targets lean more on V itself, so the fact that an early Hopper episode ends in a fall reaches
     earlier states only slowly, which pushes V up. λ = 1.0 would show the lag alone, λ = 0.8 mostly the
     optimism, λ = 0.95 a near balance.
   - *A confound that does not explain it:* shorter episodes mean smaller |G|, which inflates normalised
     bias for the same raw error. But λ = 1.0 has episodes as short as λ = 0.8's (212 against 224 steps)
     and the opposite sign, so episode length alone cannot be the cause.
   - *What would test it:* the link to falls needs each probed state's distance to the end of its
     episode, which the reports do not record yet, plus more λ values and seeds (see §7).
2. **Explained variance follows the targets' noise, not the critic's accuracy.** Early in training it
   orders the runs 0.8 > 0.95 > 1.0 (0.98, 0.96, 0.84). λ = 1.0's explained variance is far below
   λ = 0.95's (p < 0.001), yet the two critics rank states about equally well (Spearman 0.85 against
   0.88, p = 0.110) and both stay within 0.02 of unbiased (−0.015 and +0.009). λ = 0.8 scores highest
   while having the largest mean bias, though that difference is not significant. The explanation:
   explained variance is computed against each run's own λ-return targets. With λ = 1 those are full
   Monte Carlo returns, the noisiest targets; with small λ they are built largely from V itself, so V
   "explains" them partly by construction. **Explained variance is not comparable across λ.**

## 5. Explained variance against the probe

| Run | Explained variance | Median bias | Spearman |
|---|---|---|---|
| InvertedPendulum-v5, solved (Day 15, 6 seeds) | −3.3 to +0.21 | −0.021 to −0.013 | – |
| HalfCheetah-v5, 1M | +0.68 ± 0.08 | −0.003 ± 0.007 | 0.52 ± 0.08 |
| Hopper-v5, 1M | +1.00 ± 0.00 | +0.002 ± 0.001 | 0.89 ± 0.05 |

Explained variance, the critic metric every PPO implementation logs, measures how well V fits its
own rollout targets in scaled units. It says nothing about the sign or size of the bias: critics
within 2% of the truth score anywhere from −3.3 to 1.0, and §4 shows that across λ it mostly reflects
how noisy the training targets are. On HalfCheetah its moderate value (0.68) agrees with the
probe's Spearman (0.52) that ranking is weak, but only the probe adds that there is no systematic
bias.

## 6. Limitations

- **Different quantities.** PPO is probed as V of a stochastic policy, TD3 and DDPG as Q of a
  deterministic one, SAC as a soft Q. All answer "how much return does the critic expect from states
  its policy visits?", but PPO's is the easiest version: on-policy data, no maximisation.
- **One probe per seed at 300k and 1M,** with 10 probe and 5 evaluation episodes; 5 seeds.
- **Post hoc.** The window and its 20k–250k range were chosen after seeing the figures, and the
  explanation in §4 was built after seeing the result. Treat §4.1 as a hypothesis for a new
  experiment, not a finding.
- **Early HalfCheetah transient.** PPO's median bias dips to about −0.75 around 60k, while its returns
  are near zero. There max(|G|, 10) magnifies small raw errors, and the policy is changing fast. Not
  analysed further.
- **Environment versions.** The benchmark comparison is against CleanRL's v2 results.

## 7. Reproduce, and what comes next

```bash
python scripts/run_week3.py                                                    # the 20 runs
python scripts/summarize_sweep.py runs/week3 --only ppo                        # §1, §2
python scripts/summarize_sweep.py runs/week2 runs/week3 --at-step 300000 --only ddpg td3 td3_no_cdq sac ppo --ref ppo   # §3
python scripts/summarize_sweep.py "runs/week3/Hopper-v5__*" --ref ppo          # §4, end of training
python scripts/summarize_sweep.py "runs/week3/Hopper-v5__*" --window 20000 250000 --ref ppo   # §4, early training
python scripts/plot_report.py runs/week2 runs/week3 --max-step 300000 --only ddpg td3 td3_no_cdq sac ppo --out figures/all_300k.png
```

**Week-5 candidate (4): does small λ make early value estimates optimistic, and is it the falls?**
A dedicated, cheap experiment. Record each probed state's distance to the end of its episode and
whether that end was a fall or the time limit. Run Hopper with λ ∈ {0.5, 0.8, 0.9, 0.95, 0.99, 1.0},
10 seeds, 250k steps, about 15M steps in total, so roughly 6 CPU-hours by the week-3 speeds. The
hypothesis predicts that the early positive bias grows as λ falls and is concentrated in states many
steps before a fall.

## References

- Fujimoto, van Hoof, Meger (2018). *Addressing Function Approximation Error in Actor-Critic Methods.*
- Schulman, Moritz, Levine, Jordan, Abbeel (2016). *High-Dimensional Continuous Control Using
  Generalized Advantage Estimation.* ICLR.
- Huang et al. (2022). *CleanRL.* JMLR 23(274). PPO benchmark table in the CleanRL documentation.
