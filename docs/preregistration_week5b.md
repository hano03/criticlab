# Week 5b pre-registration: copy interval × replay ratio

*criticlab · written 2026-11-17, after the week-5 result and before any new run · committed before
launch, so the commit date is the registration date · analysis: `scripts/run_week5b.py`*

## 1. Why this follow-up

The week-5 test (`docs/preregistration_week5.md`) asked whether the number of gradient updates per
target copy controls DQN's overshoot on CartPole. The pre-registered outcome was **AGAINST**.
At a fixed copy interval of 500 env steps, raising the updates per copy from 50 to 500 more than
doubled the peak excess over the copy envelope: +156.9 ± 6.6 → +388.6 ± 138.8, difference +231.8
[+132.4, +331.1].

Three quantities are tied together:

  updates per copy = copy interval (env steps) × updates per env step (the replay ratio)

So any one experiment can only move two of them:

- **Week 4** raised the copy interval at a fixed replay ratio, and the overshoot vanished.
- **Week 5** raised the replay ratio at a fixed copy interval, and the overshoot grew.

The week-5 pre-registration named, as its next test, copies every 5,000 steps with 50 updates per
copy. That test also lowers the replay ratio, to 0.01, so it would not have separated anything. It
is withdrawn; the correction is recorded in the progress log (Day 30).

What can be learned is how the two free quantities act together. This sweep fills the missing
corner of a 2×2: **a slow target at the high replay ratio.**

## 2. Design

All runs use CartPole-v1, plain DQN and 10 seeds per cell. Probes run 10 episodes of up to 1,000
steps. All other settings are CleanRL's defaults.

| Cell | Target copied every | Updates per env step | Updates per copy | Steps | Configuration | Source |
|---|---|---|---|---|---|---|
| A | 500 | 0.1 | 50 | 500k | `dqn` | week 5, seeds 0–9 (seen) |
| B | 500 | 1.0 | 500 | 500k | `dqn_g10` | week 5, seeds 0–9 (seen) |
| C | 5,000 | 0.1 | 500 | 1.5M | `dqn_tgt5000` | week 4, seeds 0–4 (seen) + **new seeds 5–9** |
| D | 5,000 | 1.0 | 5,000 | 1.5M | `dqn_tgt5000_g10` | **new, seeds 0–9** |

- **Run length and exploration.** The 1.5M-step runs use `--exploration-fraction 1/6`, so ε still
  reaches 0.05 at 250k, as in week 4. They make 298 copies; the default overshoot peaks at about
  290.
- **Pooling cell C.** Cell C combines two sweeps. The code path is the same: with gradient steps = 1
  a run is bit-identical to one made before the flag existed, and a test checks this. The machine
  and the flags are the same too.
- **Designated sources only.** `run_week5b.py` reads each cell only from the folder and seeds in the
  table. `runs/week4` also holds plain DQN with copies every 500 steps (cell A's configuration);
  those runs are not read.
- **No seed counted twice.** Within the designated sources, `run_week5b.py` refuses to count any
  (cell, seed) twice.
- **New compute.** 15 runs, about 19 CPU-hours, about 2 h of wall time at 10 runs at once.

**What is already known.** Cells A and B and five seeds of C have been seen. The question was chosen
after seeing them. What is fixed here, before any new run, is the measure and the rule that turns D
(and C's new seeds) into an outcome.

## 3. Primary measure

For each seed: the **peak excess of max-Q over the copy envelope within the first 290 target
copies.**

- The envelope after n copies is E(n) = 100 (1 − 0.99^(n+1)).
- This is week 4's primary measure. The copy count has to be matched, because the copy interval
  differs between cells.

## 4. Hypotheses and the outcome rule

No direction is predicted; both answers are plausible:

- **H-interval.** The copy interval in env steps decides the outcome. Between copies, half of the
  10k buffer is replaced, so errors fitted into one target cannot feed back on the same data copy
  after copy. Under this hypothesis D ≈ C.
- **H-ratio.** The replay ratio decides the outcome. Replaying each transition about 128 times
  (1.0 update per step × batch 128) overfits the small buffer whatever the target does. Under this
  hypothesis D ≈ B, or worse.

The **protection** P measures how much of the high-ratio overshoot the slow target removes,
computed from the cell means:

  P = (B − D) / (B − C)      (D = C gives P = 1; D = B gives P = 0)

| Outcome | Rule |
|---|---|
| **PROTECTS** | P ≥ 0.75, and the 95% Welch interval for D − B lies below 0 |
| **DOES NOT** | P ≤ 0.25, so D is close to B or worse |
| **PARTLY** | Anything in between, including P ≥ 0.75 with an interval that reaches 0 |
| **NO VERDICT** | C's mean exceeds +30 (week 4 measured +7.1; the result did not hold up on new seeds), or B − C < 100 (no high-ratio overshoot to protect against) |

As a reference only, not a prediction: if the two effects added on this scale, D would be about
B + C − A ≈ 218, giving P ≈ 0.42 (PARTLY). Excess cannot go far below 0, so additivity is doubtful
near the floor.

`run_week5b.summary()` applies the rule, and `tests/test_run_week5b.py` checks it.

## 5. Reported, not predicted

- **The interaction** (D − C) − (B − A), with a Welch–Satterthwaite interval.
- **Other excess measures:** peak excess over the whole run and within the first 500k steps.
- **Overshoot frequency:** the share of probes above 110.
- **Late behaviour:** the greedy return and the median normalised bias over each run's last 100k
  steps.
- **The pairwise contrasts** C − A, B − A and D − C, on the primary measure and on return.

## 6. What each outcome would mean for the week-5 note

- **PROTECTS.** A slow target is enough, even when each transition is replayed heavily. What decides
  whether DQN's values run away on CartPole is how much new data arrives between copies, not how
  hard the network is trained on it. The week-5 result then shows that the protection comes from
  spacing the copies out, not from training more against each target.
- **DOES NOT.** The replay ratio dominates. A slow target protects only when each transition is
  replayed rarely. This is a practical warning for sample-efficiency methods that raise the replay
  ratio.
- **PARTLY.** Both matter. The note reports the size of each effect and the interaction.

## 7. Known limitations

- **Copies-matched is not stage-matched.** The 290th copy comes at about 155k steps in A and B, and
  at about 1.46M in C and D (week 4's limitation, unchanged). The excess within the first 500k steps
  is reported for the other side.
- **Cell D is extreme.** It takes 5,000 updates against each frozen target and about 1.47M updates
  in total on a 10k buffer. It may fail to learn CartPole, and its returns are reported either way.
- **Seeds differ across machines.** They are not reproducible from one machine to another, so only
  aggregates are compared.
- **Small differences between sweeps are possible.** Cell C pools seeds run weeks apart, on the same
  machine and with the same code.
