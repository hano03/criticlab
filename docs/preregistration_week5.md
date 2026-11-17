# Week 5 pre-registration: gradient updates per target copy

*criticlab · written 2026-11-17, before the sweep · committed before any week-5 run is launched, so
the commit date is the registration date · analysis: `scripts/run_week5.py`*

## 1. Question

In week 4, DQN on CartPole predicted values up to about 250, although no policy is worth more than
100. Copying the target network every 5,000 env steps instead of every 500 removed this overshoot.
Measured as the peak excess of max-Q over the copy envelope, it fell from +155 to +7
(`docs/findings_week4.md`).

That change did two things at once:

1. **Updates per copy.** It gave the online network 500 gradient updates against each frozen
   target instead of 50.
2. **Copy interval in env steps.** It spread the copies ten times further apart in env steps.
   Matched numbers of copies therefore came at very different stages of training: 290 copies
   arrive at about 155k steps with ε ≈ 0.45 for the default, but at 1.46M steps with ε = 0.05 for
   the slow target.

Week 4 cannot tell these two apart.

**This test fixes the copy interval at 500 env steps and varies only the number of gradient updates
per copy.** Every run makes its copies at the same steps, so the copy count, the envelope and ε are
identical at every probe.

## 2. Hypothesis and alternatives

- **H.** The overshoot is controlled by how many gradient updates the online network takes against
  each frozen target. With 500 updates per copy it disappears, whatever the copy interval.
- **A1, copy interval.** What matters is the copy interval in env steps, or the training stage at
  which copies happen. Under A1, more updates per copy at a 500-step interval do not remove the
  overshoot.
- **A2, replay ratio (a confound of this design).** With the copy interval fixed, more updates per
  copy also means more updates per env step:
  - 0.1 updates per env step at 50 updates per copy;
  - 0.3 at 150;
  - 1.0 at 500.
  
  On a 10k-transition buffer, more fitting per new transition plausibly increases overestimation,
  if it changes it at all. This works against H; it has not been tested here.

## 3. Design

| Configuration | `--gradient-steps` | Updates per target copy | Updates per env step | Gradient updates in total |
|---|---|---|---|---|
| `dqn` (CleanRL default) | 1 | 50 | 0.1 | 49,000 |
| `dqn_g3` | 3 | 150 | 0.3 | 147,000 |
| `dqn_g10` | 10 | 500 | 1.0 | 490,000 |

- **Fixed in every run.** CartPole-v1, plain DQN (no `--double`), 500k env steps, target copied
  every 500 env steps, one training call every 10 env steps, learning from 10k steps. Every other
  hyperparameter is CleanRL's default.
- **Seeds.** 0–9: 30 runs.
- **Probes.** Every 10k steps, 10 probe episodes each, of up to 1,000 steps.
- **What `--gradient-steps k` does.** It runs k updates per training call, each on a fresh batch
  from the same buffer, against the same frozen target network. With k = 1 a run is bit-identical to
  one made before the flag existed, and a test checks this.
- **Compute.** About 9.5 CPU-hours, about 1 h wall time at 10 runs at once:
  `python scripts/run_week5.py --dry-run --max-parallel 10`.

## 4. Primary measure

For each seed, the **peak excess of max-Q over the copy envelope** is computed over all probes
after learning starts, up to 500k steps:

  max over probes of [ `bias/q_max` − E(n) ],  where E(n) = 100 (1 − 0.99^(n+1)) and n = step // 500 − 10,000 // 500.

- `bias/q_max` is the largest prediction Q(s, a) over all probe states, under the greedy policy.
- E(n) is the largest value that n hard copies can honestly support (week 4, §3).
- This is week 4's measure with one difference: in week 4 it was restricted to the first 290
  copies, because copies had to be matched across intervals. Here copies already match at every
  probe, so the whole run is used.

## 5. Predictions and the outcome rule

- **P1 (primary).** Compare 500 against 50 updates per copy. The mean peak excess falls by at least
  75%, and the 95% Welch confidence interval for the difference in means (500 − 50) lies entirely
  below 0.
- **P2 (dose-response).** The group means are ordered 50 > 150 > 500. Across all 30 runs,
  Spearman's ρ between log(updates per copy) and the per-seed peak excess is negative, with
  p < 0.05.
- **P3.** With 500 updates per copy, no probe of any seed has max-Q above ceiling + 10 = 110.

**The outcome is decided by P1 alone.** Here the reduction is 1 − mean(500) / mean(50).

| Outcome | Rule |
|---|---|
| **SUPPORTS H** | Reduction ≥ 75% and the interval lies below 0 |
| **AGAINST H** | Reduction ≤ 25%, including any increase |
| **PARTIAL** | Anything in between, including a reduction ≥ 75% whose interval reaches 0 |
| **NO VERDICT** | The default runs do not reproduce the overshoot: mean peak excess < +50. Week 4 measured +155 |

P2 and P3 are reported as holding or not, and do not change the outcome.

`run_week5.summary()` applies the outcome rule, and `tests/test_run_week5.py` checks it against
these rules. Any analysis added after the sweep will be labelled post hoc.

## 6. Reported, not predicted

For each configuration, the summary also reports:

- the excess within the first 290 copies (week 4's primary measure);
- peak max-Q and the step where it occurs;
- the share of probes above 110;
- the greedy return and the median normalised bias, averaged over probes 400k–500k.

It also reports Welch intervals for 150 vs 50, 500 vs 150 and 500 vs 50, on both peak excess and
late return.

## 7. What each outcome would mean

- **SUPPORTS.** The number of updates against each frozen target is what removed the overshoot in
  week 4. Spacing copies further apart in env steps is not needed.
  - This holds despite the replay-ratio confound, which plausibly pushes the other way.
  - Practical reading: in CleanRL's DQN, 50 updates per copy leave the target too close to the
    online network.
- **AGAINST.** At a fixed copy interval, updates per copy do not matter. This cannot separate A1
  from A2.
  - Next test: copies every 5,000 env steps with 50 updates per copy (`--train-frequency 100`).
    That holds updates per copy at the default and lowers the replay ratio, so it separates the
    factors from the other side.
- **PARTIAL.** Both factors contribute. The effect size is reported, and the next test is the same
  as for AGAINST.
- **NO VERDICT.** Check the default runs against week 4 before interpreting anything.

## 8. Known limitations

- **Replay ratio is confounded with updates per copy** (§2, A2). A separate test would be needed to
  hold both fixed while changing the copy interval.
- **Peak measures are maxima over probes, so they are noisy.** Week 4's between-seed SD for the
  default was 5–12 against an effect of about 150, so 10 seeds per group give a wide margin for P1.
  P2 has 30 runs.
- **Seeds are not reproducible across machines** (Day 22). Only aggregates are compared.
- **The labels assume `train_frequency` = 10**, so they show "150 updates per copy". The analysis
  computes updates per copy from each run's metadata.

## 9. Exploratory, not pre-registered

`dqn_g10` and week 4's `dqn_tgt5000` both make 500 updates per copy. They differ in the copy
interval (500 vs 5,000 env steps) and the replay ratio (1.0 vs 0.1). They come from different
sweeps, so any comparison between them is exploratory.

## 10. Pilot (disclosed)

Three runs were made on Claude's machine before this note was written, only to measure speed:

- seed 99, 30k steps, with 1, 3 and 10 gradient steps;
- ε reached 0.05 at 15k steps because `--exploration-fraction` scales with the run length.

Speeds were 2,800, 1,300–2,000 and 600–700 steps/s, 3 runs at once on 2 cores; the dry-run
estimate scales these ratios to week 4's speed.

At 30k steps, only 40 copies had been made (E = 34). The mean Q on probe states was 26–32 for all
three runs, below the envelope, and max-Q was not inspected. The overshoot in week 4 peaked at about
140k steps, so the pilot carries no information about P1–P3.
