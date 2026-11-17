# Week 5c pre-registration: replay buffer size

*criticlab · written 2026-11-17, after week 5 and before any new run · committed before launch, so
the commit date is the registration date · analysis: `scripts/run_week5c.py`*

## 1. Why this test

Week 5 (`docs/findings_week5.md`) found two things about DQN's overshoot on CartPole:

- the copy interval in env steps controls it;
- the number of gradient updates per copy does not.

Copies every 500 env steps on a 10k buffer mean 20 target copies per turnover of the buffer; copies
every 5,000 mean 2. Two explanations fit, and they predict **opposite** effects of buffer size:

- **Turnover.** Each new target is fitted largely on the same transitions as the last, so an error
  written into one target is bootstrapped again and again from the same data. More copies per
  turnover, more overshoot. A **larger buffer** at a fixed copy interval means more copies per
  turnover, so **more** overshoot.
- **Overfitting.** The network fits a small, narrow dataset and extrapolates badly outside it. A
  **larger**, more varied buffer means **less** overshoot.

**What buffer size leaves fixed.** Buffer size does not change how often each transition is replayed:
0.1 updates per env step × batch 128 = 12.8 replays over a transition's lifetime, whatever the buffer
size. So the test moves the copies per turnover and the breadth of the data, not the reuse of each
transition.

## 2. Design

All runs: CartPole-v1, plain DQN, target copied every 500 env steps, one update every 10 env steps,
500k steps, 10 seeds per cell, and 10 probe episodes of up to 1,000 steps. All other settings are
CleanRL's defaults.

| Buffer | Copies per turnover | Configuration | Source |
|---|---|---|---|
| 2,500 | 5 | `dqn_buf2500` | **new**, `runs/week5c`, seeds 0–9 |
| 10,000 | 20 | `dqn` | week 5's default runs, `runs/week5`, seeds 0–9 (seen) |
| 40,000 | 80 | `dqn_buf40000` | **new**, `runs/week5c`, seeds 0–9 |

- **Sources.** Each cell is read only from its listed folder and seeds. `runs/week5` also holds
  `dqn_g3` and `dqn_g10`; those are other configurations and are never read.
- **Duplicates.** The same (cell, seed) twice within the designated sources is an error.
- **Compute.** 20 new runs, about 3 CPU-hours, about 20–30 minutes of wall time at 10 runs at once.

**What is already known.** The 10,000 cell's results have been seen: +156.9 ± 6.6. The two new cells
have not been run.

## 3. Primary measure

For each seed: the **peak excess of max-Q over the copy envelope** E(n) = 100 (1 − 0.99^(n+1)),
over all probes up to 500k steps. This is test 1's measure. The copy schedule is the same in every
run, so copies, envelope and ε match at every probe.

## 4. The outcome rule

No direction is predicted: each explanation predicts one. The rules are checked in this order:

| Outcome | Rule |
|---|---|
| **NO VERDICT** | The 10,000 cell's mean is below +50 |
| **TURNOVER** | All of: the 95% Welch interval for 40,000 − 2,500 lies above 0; the means rise, 2,500 < 10,000 < 40,000; Spearman ρ(log buffer, per-seed excess) over the 30 runs is > 0 with p < 0.05 |
| **OVERFITTING** | The mirror image: the interval lies below 0, the means fall in order, and ρ < 0 with p < 0.05 |
| **NO EFFECT** | The interval for 40,000 − 2,500 lies within ±25% of the 10,000 cell's mean (about ±39) |
| **INCONCLUSIVE** | Anything else: a mixed or non-monotone pattern, or an interval too wide to say |

`run_week5c.summary()` applies the rule, and `tests/test_run_week5c.py` checks it.

## 5. Reported, not predicted

- The excess within the first 290 copies.
- Peak max-Q and the step where it occurs.
- The share of probes above 110.
- The greedy return and the median normalised bias over probes 400k–500k.
- The contrasts 2,500 vs 10,000 and 40,000 vs 10,000, on the primary measure and on return.

## 6. What each outcome would mean

- **TURNOVER.** The copy interval's effect runs through how often targets are rebuilt on the same
  data. The week-5 note would then state a tested mechanism. The practical rule would be to set the
  copy interval relative to the buffer size, not in absolute env steps.
- **OVERFITTING.** Narrow data, not copies per turnover, drives the overshoot. The turnover reading
  of week 5 is ruled out. Why the copy interval matters would stay open: one candidate is that a
  slow target limits how often the overfitted errors are copied.
- **NO EFFECT.** Neither explanation, as stated, accounts for the copy interval's effect, and the
  note would say so.
- **INCONCLUSIVE.** The cell means are reported and no mechanism is claimed.

Whatever the outcome, the result goes into `docs/findings_week5.md` as an addendum, and week 6 (the
launch) follows.

## 7. Known limitations

- **Buffer size also changes how old the data are.** A 40k buffer at 500k steps holds transitions
  from about 40k steps of policies, a 2.5k buffer only the latest. Older data are more off-policy,
  and early on they come from more exploratory policies (ε falls to 0.05 by 250k). This is a third
  possible mechanism, which the test cannot separate from the other two.
- **A 2,500-transition buffer may destabilise learning** (forgetting). Returns are reported for
  every cell.
- **One environment, one copy interval, one replay ratio.**
- **The 10,000 cell comes from an earlier sweep.** It was run on the same machine, and the code path
  for a 10,000-transition buffer has not changed since.
