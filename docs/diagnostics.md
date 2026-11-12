# What criticlab measures, and how to read it

Every `probe_every` steps (default 10,000), each algorithm hands `criticlab/diagnostics.py` two
small functions: its policy (`act_fn`) and its critic (`q_fn`). The probe plays a few episodes on a
separate copy of the environment, writes down what actually happened, and asks the critic what it
*expected* to happen. Everything below compares those two.

## 1. The ground truth: Monte Carlo returns

For every state–action pair (s_t, a_t) visited in a probe episode, the return actually received
from that point on is

    G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …        (γ = 0.99)

A perfect critic would predict Q(s_t, a_t) = E[G_t]. One episode gives one noisy sample of that
expectation, which is why metrics are averaged over many states and several episodes.

**How each episode ends matters.**

| Episode ended by | Meaning | Which states are used |
|---|---|---|
| Termination (the hopper fell) | The return really stops | All of them |
| Time-limit truncation (1,000 steps) | Rewards after the cut are missing, so late G_t are wrong | Only states with at least H = 500 rewards observed after them (γ^500 ≈ 0.007) |

`probe/usable_states` reports how many states survived. On Pendulum (200-step episodes) the
horizon must be lowered to 50; because Pendulum rewards are never positive, the missing tail can
only make G too high there, so the measured bias is a lower bound on overestimation.

**SAC is different.** SAC's critic predicts a *soft* value: reward plus an entropy bonus
−α log π for every future action. Comparing it with plain returns would make SAC look
overestimated when it is not. So for SAC the probe runs the stochastic policy, records each
action's log-probability, and uses

    G_t^soft = r_t + Σ_{k≥1} γ^k (r_{t+k} − α log π(a_{t+k} | s_{t+k}))

The entropy term starts at t+1 because Q(s_t, a_t) already conditions on a_t. This matches the
recursion in REDQ's evaluator (`redq/utils/bias_utils.py`).

## 2. The metrics

All live in `runs/<run>/critic_report.json`, one entry per probe.

| Metric | Definition | How to read it |
|---|---|---|
| `bias/bias_mean` | mean of Q − G over usable states, in reward units | > 0 overestimation, < 0 underestimation. Not comparable across environments |
| `bias/norm_bias_mean` | mean of (Q − G) / max(\|G\|, 10), per state (REDQ's definition) | Comparable across environments. The headline number |
| `bias/norm_bias_median` | median of the same per-state values | Robust companion to the mean. If mean ≫ median, a minority of states drives the bias (see §3) |
| `bias/pearson` | linear correlation of Q and G | Can stay near 1 just because a few states have very different values |
| `bias/spearman` | rank correlation of Q and G | Does a higher Q really mean a better outcome, state by state? |
| `bias_min/*` | the same metrics using min(Q1, Q2) | Twin critics only: the value TD3 and SAC actually bootstrap from. `bias/*` uses their average |
| `twin/abs_diff_mean` | mean \|Q1 − Q2\| | Twin critics only. A cheap uncertainty signal |
| `td/abs_mean`, `td/target_q_mean`, `td/target_q_abs_max` | TD error and target size on the latest replay batch | A steadily rising TD error or target is an early warning of divergence |
| `probe/return_mean` | undiscounted return of the probe episodes | For SAC this is the *stochastic* policy |
| `eval/deterministic_return_mean` | SAC only: return of tanh(mean action) | The return to compare with TD3 and DDPG |
| `probe/alpha` | SAC only: α used for the soft returns | |
| `perf/sps_window` | environment steps per second since the previous probe (probe time included) | Use this, not the running average, for compute budgets |

## 3. Known effects to keep in mind

- **Critic lag after a policy collapse.** When a policy temporarily breaks, episodes get short,
  G drops, and the critic still predicts the values it learned for the better policy. The bias is
  real (the critic is wrong about the current policy), but dividing by max(|G|, 10) magnifies it,
  because many states then have small |G|. On the Day 6 Hopper rehearsal, DDPG's probe at 80k
  steps had return 114.6, only 306 usable states and normalised bias +9.4. Check the median and
  the return panel before reading a single spike as a trend.
- **Probes see on-policy states only.** The critic is trained on the replay buffer, which holds
  older policies' states. A critic can be accurate on probe states and wrong elsewhere, or the
  reverse.
- **Noise.** Each probe uses a handful of episodes (5 by default, 10 in the week-2 sweep). A
  single probe from a single seed is an observation, not a result; claims need several seeds.

## 4. Reading the standard figure

`scripts/plot_report.py` draws one line per algorithm, mean ± std across seeds:

1. **Return.** Is the agent learning? (SAC: deterministic evaluation.)
2. **Normalised Q bias.** The headline: above the zero line means overestimation.
3. **Calibration at the final probe.** Mean G in 10 equal-count bins of predicted Q. On the dotted
   line, the critic is calibrated; below it, the critic promises more than it delivers.
4. **Rank calibration.** Spearman(Q, G) over time.
5. **Twin-critic disagreement.**
6. **TD error on replay batches.**

`<name>_hero.png` repeats panel 2 alone; it is the README figure.
