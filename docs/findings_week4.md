# Week 4: DQN, Double DQN and the target network

*criticlab · 2026-11-17 · pre-registered test (spec v3.2, written before the sweep) · reproduce with §7*

**Question.** On CartPole, DQN's largest prediction climbs to about 250 although no policy can earn
more than 100, then falls back. Double DQN does not change it. Is the cause the target network,
which CleanRL copies every 500 environment steps (every 50 gradient updates)?

**Answer.** On CartPole, yes: with the target copied every 5,000 steps the overshoot disappears, for
DQN and Double DQN alike, and Double DQN makes no measurable difference at either copy rate. The
price is slower learning. On Acrobot the predictions failed: its occasional values above the ceiling
are a different effect, untouched by the copy rate.

## 1. Setup

- **Algorithms.** `dqn.py` with CleanRL's `dqn.py` hyperparameters (500k steps, 10k replay buffer,
  one update every 10 steps, ε 1 → 0.05 by 250k), and `--double` for Double DQN.
- **Sweep.** `scripts/run_week4.py`: CartPole-v1 and Acrobot-v1 × {DQN, Double DQN} × target copied
  every {500, 5,000} steps × 5 seeds, 10 probe episodes every 10k steps. The 5,000-step runs last
  1.5M steps, with ε still reaching 0.05 at 250k (why: §2).
- **Implementation check.** Greedy return averaged over probes 400k–500k with CleanRL's settings:
  CartPole 476.6 ± 26.5 (DQN) and 492.0 ± 10.2 (Double DQN); Acrobot −84.9 ± 1.3 and −83.6 ± 3.9.
  CleanRL reports CartPole-v1 488.69 ± 16.11 and Acrobot-v1 −91.54 ± 7.20. In range.

## 2. Measuring an overshoot fairly: the ceiling and the copy envelope

**The ceiling.** Rewards are bounded, so values are too. CartPole pays +1 per step, so no state is
worth more than 1 / (1 − γ) = 100; Acrobot pays −1 per step, so no state is worth more than 0. A
prediction above the ceiling is wrong for *every* policy, so `bias/q_max` (the largest prediction on
probe states) shows the problem without any Monte Carlo return. van Hasselt et al. (2018) call
this *soft divergence*.

**The copy envelope.** With hard target copies, the critic regresses onto r + γ·max Q_target, and
Q_target changes only at a copy. So each copy lets honest values grow by at most one more step of
reward: after n copies, predictions supported by the targets are capped at

    E(n) = r_max (1 − γ^(n+1)) / (1 − γ)          (CartPole: 100 (1 − 0.99^(n+1)))

With a copy every 5,000 steps, 500k steps give only 98 copies, so E = 63. A low peak there would
just mean the values had not had time to grow, and would have confirmed the hypothesis trivially. A
one-seed pilot showed exactly this (max-Q 67 against an envelope of 63 at 500k), so before the sweep
the design was changed: the primary measure became max-Q's **excess over the envelope within the
first 290 copies** (about where the default peaks), and the 5,000-step runs were lengthened to 1.5M
steps (298 copies). Acrobot's rewards are never positive, so its envelope is simply its ceiling, 0.

## 3. Predictions and results

Written before the sweep: **P1** within the first 290 copies, the excess over the envelope is much
smaller with 5,000-step copies than with 500, for DQN and Double DQN. **P1b** over the whole 1.5M
steps, the 5,000-step runs stay within ceiling + 10. **P2** Double DQN stays close to DQN at both
copy rates. The predictions did not name an environment, so they are judged on both.

### CartPole (ceiling 100)

| Group | Excess over envelope, first 290 copies | Peak max-Q, whole run | Probes > 110 | Return, 400k–500k | Median bias, 400k–500k |
|---|---|---|---|---|---|
| DQN, copy every 500 | +154.9 ± 5.1 | 247.5 ± 4.4 | 56% | 476.6 ± 26.5 | +0.097 ± 0.127 |
| Double DQN, copy every 500 | +157.6 ± 12.0 | 250.3 ± 11.3 | 56% | 492.0 ± 10.2 | +0.029 ± 0.006 |
| DQN, copy every 5,000 | +7.1 ± 2.8 | 98.8 ± 2.0 | 0% | 236.5 ± 128.7 | −0.070 ± 0.145 |
| Double DQN, copy every 5,000 | +5.3 ± 1.6 | 97.0 ± 0.5 | 0% | 311.7 ± 140.0 | −0.201 ± 0.165 |

*Mean ± sample std over 5 seeds; differences below are a − b with 95% Welch confidence intervals.*

| Comparison (a − b) | Excess, first 290 copies | Return, 400k–500k |
|---|---|---|
| DQN: copy 5,000 vs 500 | −147.8 [−154.1, −141.5], p < 0.001 | −240.1 [−398.3, −81.9], p = 0.013 |
| Double DQN: copy 5,000 vs 500 | −152.3 [−167.1, −137.4], p < 0.001 | −180.3 [−353.9, −6.7], p = 0.045 |
| Copy 500: Double vs DQN | +2.7 [−12.1, +17.4], p = 0.668 | +15.4 [−17.0, +47.7], p = 0.279 |
| Copy 5,000: Double vs DQN | −1.8 [−5.3, +1.7], p = 0.253 | +75.2 [−121.2, +271.6], p = 0.403 |

- **P1 holds.** A slower target removes almost all of the overshoot: from about +155 above the
  envelope to about +6. The steps-matched measure gives the same answer (first 500k steps: +4.6 and
  +4.0 against +154.9 and +157.6), so the result does not depend on how the runs are aligned.
- **P1b holds.** No slow-target probe went above 110 in 1.5M steps; the largest prediction was 101.5.
- **P2 holds.** At the default copy rate, Double DQN's effect on the overshoot is between −12 and
  +17, against an overshoot of about 155. With two actions, maximisation bias is not what drives it.
- **The cost.** The slow target cut the return over 400k–500k by 240 and 180. Its critic is
  pessimistic for most of training (values can only rise one reward step per copy) and its returns
  reach 450–500 only around 1.2–1.5M steps. Meanwhile the default runs reached 500 while predicting
  values 2.5 times higher than possible: as in week 2, a badly wrong critic did not stop the policy.

### Acrobot (ceiling 0)

| Group | Peak max-Q, first 290 copies | Probes > 10 | Return, 400k–500k |
|---|---|---|---|
| DQN, copy every 500 | +32.6 ± 12.7 | 17% | −84.9 ± 1.3 |
| Double DQN, copy every 500 | +31.7 ± 16.8 | 16% | −83.6 ± 3.9 |
| DQN, copy every 5,000 | +34.1 ± 14.0 | 6% | −146.9 ± 36.9 |
| Double DQN, copy every 5,000 | +31.0 ± 21.2 | 4% | −152.5 ± 56.7 |

- **P1 fails.** The copy rate does not change the peak: +1.4 [−18.1, +21.0] (DQN) and −0.8
  [−28.9, +27.4] (Double DQN).
- **P1b fails.** Slow-target runs still went more than 10 above the ceiling in 6% and 4% of probes.
- **P2 holds** (all Double-vs-DQN intervals span zero), but the intervals are wide.
- **Why the difference.** CartPole's overshoot lifts most of the critic: in the Day 22 check, 90% of
  probe states were above the ceiling around the peak. Acrobot's crossings are isolated states, 1–6% of
  probe states, and they come and go between probes. They look like local fitting error at a few
  rarely visited states, not runaway bootstrapping, and the copy rate does not reach them.

## 4. The same limit, mirrored (observation)

With slow copies, CartPole's critic underestimates for most of training (median normalised bias
about −0.9 early, near 0 only after about 1M steps), while Acrobot's overestimates for its first ~400k steps
(about +0.8 falling to 0). Both follow from the copy limit: values start near 0 and each copy can
move them by at most one reward step. With +1 rewards they cannot rise fast enough, so the critic is
pessimistic; with −1 rewards they cannot fall fast enough, so it is optimistic. This was not
predicted in advance and is reported as an observation.

## 5. What this means

- **For DQN users.** On CartPole with CleanRL's defaults, values more than double past what any
  policy can earn. The cause is the target network, refreshed every 50 updates, not maximisation;
  Double DQN does not help. Copying every 5,000 steps removes it but slows learning several-fold
  (returns near 500 by about 300k steps with the default, only around 1.2M with the slow target).
  A soft (Polyak) target update or an intermediate period may give a better trade-off; untested here.
- **For criticlab.** Two diagnostics earned their place: `bias/q_max` against the reward ceiling,
  which flags soft divergence with no Monte Carlo return, and the copy envelope, which stops a slower
  target from looking good merely because its values have not had time to grow.
- **Relation to prior work.** van Hasselt et al. (2018) already found that target networks reduce
  soft divergence. This is a clean, pre-registered reproduction of that effect on a small task, with
  a measurement that controls for value propagation. It is not a new phenomenon.

## 6. Limitations

- **Matching copies does not match training stage.** 290 copies come at about 155k steps (ε ≈ 0.45)
  with 500-step copies and at 1.46M (ε = 0.05) with 5,000. The steps-matched measure agrees, but the
  clean test would vary gradient updates per copy at a fixed copy rate.
- **The design was changed after a pilot.** The envelope measure and the 1.5M-step runs were
  introduced after one pilot run and before the sweep; the pilot is in the progress log.
- **Two small environments, two actions and three actions.** With more actions, maximisation bias
  could matter more and Double DQN might help; this sweep cannot say.
- **5 seeds.** Enough for the CartPole effect (non-overlapping intervals), not for the return
  differences between Double DQN and DQN.
- **Seeds are not reproducible across machines** (Windows vs Linux floating point); only aggregates are.

## 7. Reproduce

```bash
python scripts/dqn_check.py --max-parallel 12            # Day 22 check, 3 seeds, CleanRL settings
python scripts/run_week4.py --max-parallel 10            # the 40 runs, figures, figures/week4_summary.md
python scripts/run_week4.py --summary-only               # tables only, from existing runs
```

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning* (DQN).
- van Hasselt, Guez, Silver (2016). *Deep Reinforcement Learning with Double Q-learning.*
- van Hasselt et al. (2018). *Deep Reinforcement Learning and the Deadly Triad* (soft divergence).
- Huang et al. (2022). *CleanRL.* JMLR 23(274). DQN classic-control results in the CleanRL documentation.
