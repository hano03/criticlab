# Week 2 findings: DDPG, TD3 and SAC critics on Hopper and HalfCheetah

*criticlab · 2026-11-12 (statistics corrected the same day) · 4 configurations × 2 environments × 5 seeds, 300k steps, 10 probe episodes per probe*

**Headline: the critics differ far more than the returns do.** Clipped double-Q removes DDPG-style overestimation where it exists (Hopper), but turns into consistent underestimation where it doesn't (HalfCheetah). In neither environment did those large differences in critic bias change the return reached by 300k steps.

## Setup

- **Algorithms:** DDPG (TD3's settings minus its three fixes), TD3, TD3 without clipped double-Q (target smoothing and delayed updates kept), SAC (automatic α). All are single-file PyTorch implementations with CleanRL's hyperparameters.
- **Environments:** Hopper-v5 (terminates when the hopper falls) and HalfCheetah-v5 (never terminates; 1,000-step time limit).
- **Measurement:** every 10k steps, 10 probe episodes compare the critic's Q(s, a) with the discounted Monte Carlo return G (soft returns for SAC). States within 500 steps of a time-limit cutoff are excluded. Normalised bias per state is (Q − G) / max(|G|, 10), as in REDQ. Each probe reports the mean and the median over its states.
- **Statistics:** values are mean ± sample standard deviation (ddof = 1) across 5 seeds at the final probe (300k). Comparisons use Welch's t-test on the 5 per-seed values. With n = 5, treat p-values as rough guides.
- **Correction:** the first version of this note took population standard deviations as sample ones, which overstated every t by about 12%. All statistics below are recomputed from the raw per-seed values. No conclusion changed; the SAC Hopper return comparison moved from p ≈ 0.07 to p ≈ 0.10.

## Results at 300k steps

| Environment | Algorithm | Return | Normalised bias, mean | Normalised bias, median | Spearman(Q, G) |
|---|---|---|---|---|---|
| Hopper-v5 | DDPG | 1,124 ± 730 | +1.453 ± 0.911 | +0.946 ± 0.419 | 0.54 |
| Hopper-v5 | TD3 w/o clipped double-Q | 1,597 ± 934 | +1.032 ± 0.727 | +0.823 ± 0.632 | 0.49 |
| Hopper-v5 | TD3 | 1,475 ± 1,179 | −0.119 ± 0.137 | −0.176 ± 0.111 | 0.73 |
| Hopper-v5 | SAC | 2,781 ± 990 | −0.058 ± 0.018 | −0.061 ± 0.020 | 0.86 |
| HalfCheetah-v5 | DDPG | 7,580 ± 829 | +0.205 ± 0.391 | +0.025 ± 0.018 | 0.53 |
| HalfCheetah-v5 | TD3 w/o clipped double-Q | 7,635 ± 688 | −0.123 ± 0.009 | −0.131 ± 0.015 | 0.59 |
| HalfCheetah-v5 | TD3 | 7,510 ± 418 | −0.257 ± 0.011 | −0.258 ± 0.013 | 0.63 |
| HalfCheetah-v5 | SAC | 9,044 ± 817 | −0.092 ± 0.024 | −0.098 ± 0.023 | 0.72 |

Returns are from the deterministic policy (for SAC, the deterministic evaluation). Spearman is the mean over seeds at the final probe.

## The four week-2 questions

### 1. Does DDPG overestimate?

**On Hopper, clearly; on HalfCheetah, barely.**

- **Hopper:** the median bias is +0.946. Both the mean and the median were positive in all five seeds (means +0.61 to +2.74). DDPG's critic also had the highest TD error on Hopper: the across-seed mean peaked at 6.9, against 2.5 for TD3 and 2.6 for SAC.
- **HalfCheetah:** the median is +0.025 ± 0.018, close to unbiased. The mean (+0.205 ± 0.391) is pulled up by one seed (seed 2: mean +0.904, median +0.040), where a tail of badly overestimated states dominated the average.

So overestimation is environment-dependent. It appeared in the environment where episodes terminate and policies are unstable; with only two environments, that link is a hypothesis, not an explanation.

### 2. Does TD3 fix it?

**On Hopper, yes. On HalfCheetah it overcorrects.**

- **Hopper:** TD3's median bias is −0.176, against DDPG's +0.946 (t = 5.8, p ≈ 0.003).
- **HalfCheetah:** TD3 underestimates by about a quarter (median −0.258), with almost no spread across seeds (± 0.013). That is far below DDPG's near-zero median (t = 28.4, p < 0.001). The consistency points to a systematic effect of TD3's corrections, not noise: the pessimism TD3 deliberately accepts in exchange for removing overestimation. Question 3 shows the clipped minimum causes only about half of it.

### 3. Which TD3 component matters most?

**Clipped double-Q does almost all of the bias work on Hopper.** Removing it leaves the bias at DDPG's level:

| Hopper, median bias | Value | Share of the DDPG → TD3 gap |
|---|---|---|
| DDPG | +0.946 | |
| + target smoothing and delayed updates (TD3 w/o clipped double-Q) | +0.823 | 11% (not distinguishable from DDPG: t = 0.4, p ≈ 0.73) |
| + clipped double-Q (TD3) | −0.176 | 89% (t = 3.5, p ≈ 0.023) |

Using the mean instead of the median gives the same picture: 27% and 73%.

**On HalfCheetah, both parts push the critic down.** Smoothing and delay move the median from +0.025 to −0.131, and clipped double-Q moves it further to −0.258. Both steps are clear (t ≈ 15 each).

*Limit:* this ablation removes only clipped double-Q, so it cannot separate target smoothing from delayed updates.

### 4. Where does SAC sit, and is its critic better calibrated?

**SAC had the best returns and the best-ranked critic, with mild underestimation.**

- **Returns:** SAC is clearly ahead on HalfCheetah (9,044 against TD3's 7,510; t = 3.7, p ≈ 0.010). On Hopper it is ahead but not conclusively with 5 seeds (2,781 against 1,475; t = 1.9, p ≈ 0.10).
- **Bias:** SAC's critic sits slightly below the soft returns in both environments (median −0.061 on Hopper, −0.098 on HalfCheetah), with the smallest spread across seeds.
- **Ranking:** SAC's final rank calibration was the highest in both environments (Spearman 0.86 on Hopper and 0.72 on HalfCheetah, against 0.73 and 0.63 for TD3). The two overestimating critics ranked states worst on Hopper (0.54 and 0.49).

## The finding behind the headline: bias does not predict return here

| Return at 300k | Comparison | Difference | Welch t, p |
|---|---|---|---|
| Hopper | TD3 − DDPG | +352 | 0.57, 0.59 |
| Hopper | TD3 − TD3 w/o clipped double-Q | −121 | −0.18, 0.86 |
| HalfCheetah | TD3 − DDPG | −70 | −0.17, 0.87 |
| HalfCheetah | TD3 − TD3 w/o clipped double-Q | −125 | −0.35, 0.74 |

On Hopper, the critics of DDPG and TD3 differ by more than one normalised unit, yet their returns are indistinguishable. On HalfCheetah, three critics with clearly different biases (+0.03, −0.13 and −0.26 median) reach the same return of about 7,500. **Within 300k steps, a learning curve alone would not reveal any of these critic differences.** That is the case for measuring the critic directly, which is what this repo does.

This does not contradict the TD3 paper, which reports TD3's return advantage over 1M steps. Our runs stop at 300k.

## Other observations

- **When a policy collapses, the critic does not notice.** TD3 without clipped double-Q, Hopper, seed 4: at 150k steps the policy fell almost at once (probe return 6.6; mean real return per state 0.7), yet its critic still predicted about 303. The raw overestimation (+303) was similar to the other four seeds at that probe (+151 to +270). Dividing by max(|G|, 10) turned it into a normalised median of +32.3, against +1.0 to +2.1 for the other seeds. The policy recovered by 160k. Collapses like this are why the README figure aggregates robustly twice: the median over states within each probe, then the median and interquartile range across seeds.
- **Mean versus median within a probe.** HalfCheetah DDPG seed 2 (mean +0.904, median +0.040) shows the mean can be driven by a tail of states. The full figure's panel 2 keeps REDQ's mean; the README figure uses the median.
- **Twin disagreement tracks instability.** On Hopper, averaged over 100k–300k steps, TD3 without clipped double-Q had about 3 times the twin disagreement of TD3 and SAC (5.2 against 1.7 and 1.6). Its two critics are trained on the same target, so the disagreement reflects unstable values rather than a design difference. On HalfCheetah all three twin-critic methods sat near 2.

## Benchmark check

- **TD3:** our implementation matched CleanRL's own TD3 on the same machine (Hopper-v5, seeds 1–3, 100k steps: 318.9 against 298.1 mean training return). Every algorithmic component matches CleanRL's code.
- **Published anchors** (1M steps, Hopper-v4 and HalfCheetah-v4, CleanRL): TD3 about 3,134 on Hopper and 9,583 on HalfCheetah; SAC about 2,310 on Hopper and 9,634 on HalfCheetah ([source](https://www.codesota.com/benchmark/mujoco)). Our 300k results sit below or near these 1M figures, as expected at 30% of the training. Our SAC already exceeds CleanRL's 1M figure on Hopper (2,781) and approaches it on HalfCheetah (9,044), which argues against a bug: bugs usually make results worse, not better.
- **Conclusion:** no sign of an implementation problem. A direct 300k comparison would need CleanRL runs at 300k; not needed unless a later result depends on exact parity.

## Limits

- 300k steps, not 1M; 5 seeds; 2 environments.
- The ablation removes only clipped double-Q.
- Probes measure the critic on states the *current* policy visits, not on the replay buffer.
- Normalisation by max(|G|, 10) and the 500-step truncation horizon are design choices (see `docs/diagnostics.md`). The seed-4 case shows the normalisation can magnify a fixed raw error tenfold or more when returns collapse.

## Candidate questions for week 5 (a new finding rather than a reproduction)

1. **Can cheap online signals predict the measured bias?** Twin disagreement and TD error need no extra rollouts. If either tracks the probe's bias across runs and probes, practitioners get a free warning light. All 40 report files already contain the data.
2. **Is clipped double-Q's underestimation on HalfCheetah costing anything?** Its returns match DDPG's at 300k. Longer runs, or an adjustable pessimism level, would show whether it matters later.
3. **Critic lag around collapses:** the seed-4 case shows Q staying put while returns fall to almost zero. Do twin disagreement or TD error move *before* such a collapse, giving a usable warning?
