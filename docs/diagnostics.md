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

**PPO is different again.** Its critic is a state-value function V(s) for the *stochastic*
policy, so the probe runs that policy and ignores the action. Two details matter:

- PPO learns V on rewards divided by a running scale σ (the std of the discounted return), so
  the probe compares σ·V(s) with G. Without this, V is in the wrong units and the bias is
  meaningless. Observations are normalised with the training statistics, frozen during the probe.
- `ppo.py` bootstraps through time limits (γ·V(final state) is added to the last reward), so V
  estimates an infinite-horizon value, like the off-policy critics.

Check: on InvertedPendulum-v5, a solved policy earns 1 per step for 1,000 steps, so G ≈ 99.3–100
for every usable state. After 150k steps, three seeds gave σ·V ≈ 97.8 and normalised bias −0.021,
with almost no spread. Repeated on a second machine: σ·V 97.8–98.4, bias −0.021 to −0.013. Across
those six seeds PPO's own explained variance ranged from −3.3 to +0.21: when every return is nearly
the same, explained variance divides by a variance close to zero and says nothing about the
critic. The probe still does.

**DQN and Double DQN.** The critic is Q(s, a) of the greedy policy (the policy the max in the
target evaluates), so the probe runs with ε = 0. Two details:

- Probe episodes may run past the training time limit (`--probe-max-episode-steps`). The critic
  bootstraps through time limits, so it estimates the infinite-horizon value; longer probe episodes
  just give more ground truth. CartPole-v1 stops at 500 steps, so with H = 500 a solved policy would
  leave one usable state per episode; with probe episodes of 1,000 steps it leaves 501.
- Bounded rewards bound Q. CartPole pays +1 per step, so no policy is worth more than
  1 / (1 − γ) = 100; Acrobot pays −1 per step, so Q ≤ 0. `bias/q_max`, the largest prediction on
  probe states, shows a critic going past that ceiling ("soft divergence", van Hasselt et al., 2018)
  without needing any Monte Carlo return.
- With hard target copies, honest values can grow by at most one reward step per copy, so after n
  copies they are capped by the *copy envelope* r_max (1 − γ^(n+1)) / (1 − γ). When comparing target
  periods, compare max Q with this envelope at a matched number of copies; otherwise a slow target
  looks safe merely because its values have not had time to grow (see `docs/findings_week4.md`).

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
| `probe/return_mean` | undiscounted return of the probe episodes | For SAC and PPO this is the *stochastic* policy |
| `eval/deterministic_return_mean` | SAC and PPO: return of the mean action (tanh for SAC) | The return to compare with TD3 and DDPG |
| `probe/alpha` | SAC only: α used for the soft returns | |
| `bias/q_max` | largest critic prediction on probe states | Above r_max / (1 − γ), no policy can earn it: soft divergence |
| `dqn/epsilon` | DQN only: exploration rate at the probe | |
| `ppo/reward_scale` | PPO only: σ, the running std of the discounted return | V is multiplied by it before comparison with G |
| `ppo/explained_variance` | PPO only: 1 − Var(R − V)/Var(R) on the last rollout, in scaled units | PPO's usual critic metric. Uninformative when returns barely vary (see §1) |
| `ppo/value_loss`, `ppo/approx_kl`, `ppo/clipfrac`, `ppo/entropy` | PPO only: last update's statistics | Standard PPO health checks |
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
