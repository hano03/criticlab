# Can cheap signals replace the probe? (first look)

**Question.** TD error and twin-critic disagreement cost nothing to log. If either one tracked
the bias the probe measures, you could skip the probe rollouts. Do they?

**Short answer: no, not on this data.** Both signals separate *algorithms* from one another.
Within a run, once the shared training trend and the size of Q are accounted for, neither
tracks the measured bias, and neither warns of a collapse more often than chance.

**Data.** The 40 week-2 runs: 2 environments × 4 algorithm variants × 5 seeds, 300k steps,
probed every 10k steps. Probes before 50k steps are dropped, which leaves 26 per run. The bias is
the per-probe median normalised bias. TD error is available for all 40 runs; twin disagreement
only for the 30 runs with twin critics (not DDPG).

Reproduce with:

```bash
python scripts/analyze_signals.py runs/week2
python scripts/analyze_signals.py runs/week2 --algo td3_no_cdq   # one algorithm only
```

## Why a raw correlation misleads

Two things inflate any correlation between these signals and the bias:

1. **Shared trends.** Q, TD error, twin disagreement and bias all drift over training, so two
   series that are unrelated probe-to-probe still correlate over a whole run.
2. **Coupling through the size of Q.** When Q grows, |Q1 − Q2| and TD error grow with it. So
   does (Q − G) when Q runs ahead of G. A raw correlation can therefore be just "Q got bigger".

The script reports five views, from naive to controlled.

## Results

| View | What it asks | HalfCheetah: TD / twin | Hopper: TD / twin |
|---|---|---|---|
| 1. Across runs (late-training means) | Do runs with more signal have more bias? | ρ −0.20 / −0.37 | **ρ +0.71 / +0.61** |
| 2. Within runs, probe-to-probe changes | When the signal moves, does bias move? | mean ρ +0.16 / +0.10 | +0.03 / −0.07 |
| 2b. Signal change → *next* bias change | Does the signal lead? | +0.16 / +0.15 | +0.01 / +0.02 |
| 3. Partial r, controlling log\|Q\|, step, run | Is there anything beyond scale and time? | r +0.01 / −0.09 | +0.00 / +0.18 |
| 5. Detrended levels (rolling median, 5 probes) | Do the deviations from trend match? | median ρ +0.05 / −0.03 | +0.11 / −0.02 |
| 5b. Detrended signal → next bias change, controlling ΔQ | Does it lead once Q-scale is removed? | −0.02 / −0.04 | +0.04 / +0.07 |

How to read it:

- **View 1 is the only strong result, and it is about algorithm identity.** On Hopper, DDPG and
  TD3 without clipped double-Q have both the largest bias and the largest signals. With 5 seeds
  per algorithm, the across-run correlation mostly says "these are different algorithms", which
  the probe already tells you more precisely.
- **Within runs there is little.** HalfCheetah shows weak co-movement of changes (view 2: mean
  ρ +0.16 for TD, positive in 16 of 20 runs, p ≈ 0.006). That is consistent across runs, but
  small: ρ ≈ 0.15 explains about 2% of the variance. It disappears after detrending (view 5).
  Hopper shows nothing in views 2 and 5.
- **The one partial correlation that clears p < 0.001** (Hopper twin, r = +0.18, view 3) pools 390
  autocorrelated probes, so its p-value is far too optimistic. It does not survive detrending
  (view 5: −0.02).
- **No view gives a correlation that would let you read bias off a signal.** The largest
  controlled value is |0.18|, which explains about 3% of the variance.

## Collapses (view 4)

There were 23 return collapses: return fell by more than 60% from a level above 300 between two
probes, all on Hopper. TD error had risen in the probe before the collapse in 12 of 23 (about
half: chance). Twin disagreement had risen in 3 of the 12 twin-critic collapses. Neither is an
early warning at a 10k-step probe interval.

## Limitations

- **TD error is noisy here.** It comes from one replay batch at the last update before each probe,
  not an average. A smoothed TD error might do better; this data cannot say.
- **Probe resolution.** 10k steps between probes may be too coarse to see a signal lead a
  collapse by a few thousand steps.
- **Five seeds per cell.** Per-algorithm results have wide ranges. The one hint in the controlled
  data is TD3 without clipped double-Q on Hopper (`--algo td3_no_cdq`, view 5b): both signals
  lead the next bias change in 5 of 5 seeds (median ρ +0.25 for twin disagreement, +0.12 for TD
  error; p = 0.06, the smallest a two-sided Wilcoxon test can give with 5 runs). That is the
  family with the largest collapses. It is a lead, not a finding.
- **Two environments, 300k steps.**

## What this means for criticlab

The negative result supports the repo's premise: the cheap signals that are already logged
are not a substitute for measuring the critic against what actually happens. It is a useful
paragraph for the README. It is too thin to be the week-5 "new finding" on its own.
