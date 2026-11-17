"""The week-5 sweep: at a fixed target-copy interval, do more gradient updates per copy stop DQN's
soft divergence on CartPole?

    python scripts/run_week5.py --dry-run           # list the 30 jobs and the time estimate
    python scripts/run_week5.py --max-parallel 10   # run them, then draw figures and the summary
    python scripts/run_week5.py --summary-only      # summary of existing runs only

The 30 jobs: CartPole-v1 x DQN x {1, 3, 10} gradient updates every 10 env steps x seeds 0-9,
500k steps, 10 probe episodes (up to 1,000 steps each), target copied every 500 env steps (CleanRL's
default) in every run. Updates against each frozen target: 50 (CleanRL's default), 150, 500.

Why. In week 4, copying the target every 5,000 env steps instead of 500 removed the overshoot
(excess of max-Q over the copy envelope +155 -> +7). That change did two things at once: it gave the
online network 500 instead of 50 gradient updates against each frozen target, and it spread the
copies 10x further apart in env steps, so matched copies came at different stages of training.
Here the copy schedule is identical in every run (same copies, same envelope, same epsilon at every
copy), and only the updates per copy change. The price: updates per env step change with them
(0.1, 0.3, 1.0), so the replay ratio is confounded with updates per copy. A higher replay ratio is
usually thought to make overestimation worse, not better, so it works against the hypothesis.

Pre-registration (Day 29, before the sweep; also docs/preregistration_week5.md and spec v3.4).
H: the overshoot is controlled by the number of gradient updates against each frozen target.
Primary measure, per seed: peak excess of max-Q over the copy envelope E(n) = 100 (1 - 0.99^(n+1))
over all probes up to 500k steps.
  P1  Primary. With 500 updates per copy, the mean peak excess falls by at least 75% relative to 50
      updates per copy, and the 95% Welch interval for the difference lies below 0.
  P2  Dose-response. Group means fall in order 50 > 150 > 500, and Spearman's rho between
      log(updates per copy) and per-seed peak excess (30 runs) is negative with p < 0.05.
  P3  With 500 updates per copy, no probe of any seed goes above ceiling + 10 = 110.
Outcome of the test, decided by P1 alone:
  SUPPORTS    reduction >= 75% and the interval below 0
  AGAINST     reduction <= 25% (including an increase)
  PARTIAL     anything in between
  NO VERDICT  if the default runs do not reproduce the overshoot (mean peak excess < +50)
Not predicted, reported: excess within the first 290 copies (week 4's measure), peak step, share of
probes above 110, greedy return and median normalised bias over probes 400k-500k.

Resumable like run_week2.py. Figures: figures/week5.png (+ _hero), figures/week5_dose.png;
tables and the outcome: figures/week5_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_week2 as sweep  # noqa: E402
import run_week4 as week4  # noqa: E402
from launch import run_all  # noqa: E402

ENV = "CartPole-v1"
CEILING, R_MAX = 100.0, 1.0
KEYS = ["dqn", "dqn_g3", "dqn_g10"]          # 1, 3, 10 gradient updates per training call
DEFAULT, SLOW = "dqn", "dqn_g10"
SEEDS = list(range(10))
TOTAL = 500_000
MARGIN = week4.MARGIN                          # 10
MATCHED_COPIES = week4.MATCHED_COPIES          # 290, week 4's primary window (reported, not tested)
LATE = week4.LATE                              # probes 400k-500k
SUPPORT_REDUCTION, AGAINST_REDUCTION = 0.75, 0.25
MIN_DEFAULT_EXCESS = 50.0
OUTCOME_TEXT = {
    "SUPPORTS": "More updates against each frozen target remove the overshoot at a fixed copy interval. "
                "Week 4's slow target worked through its 500 updates per copy; spacing the copies "
                "further apart in env steps is not needed.",
    "AGAINST": "At a fixed copy interval, more updates per copy do not remove the overshoot. Week 4's "
               "effect then comes from the copy interval in env steps, unless the higher replay ratio "
               "here cancels it. Next test: copies every 5,000 env steps with 50 updates per copy "
               "(--train-frequency 100), the same separation from the other side.",
    "PARTIAL": "Updates per copy explain part of the overshoot, not all of it. Report the size; the "
               "next test is the same as for AGAINST.",
    "NO VERDICT": "The default runs did not reproduce the week-4 overshoot, so the comparison cannot "
                  "test H. Check the runs before reading anything else.",
}


def updates_per_copy(meta: dict) -> int:
    period = int(meta.get("target_network_frequency", sweep.DEFAULT_TARGET_PERIOD))
    return period // int(meta.get("train_frequency", 10)) * int(meta.get("gradient_steps", 1))


def week5_jobs(seeds, total, probe_episodes, runs_dir) -> list[dict]:
    jobs = sweep.build_jobs(KEYS, [ENV], seeds, total, probe_episodes, runs_dir)  # slowest first
    for job in jobs:
        job["argv"] = list(job["argv"]) + week4.CARTPOLE_FLAGS
    return jobs


def load(runs_dir: Path) -> dict:
    """key -> list of per-seed measures. Unfinished runs (_incomplete/) are skipped."""
    out = defaultdict(list)
    for path in sorted(runs_dir.glob("*/critic_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        meta = report["meta"]
        key = sweep.algo_key(meta)
        if meta.get("env_id") != ENV or key not in KEYS or not report.get("probes"):
            continue
        row = week4.per_seed(report, CEILING, R_MAX, late=LATE, matched_steps=TOTAL)
        row["primary"] = row["excess_steps"]  # peak excess over the envelope, all probes <= 500k
        row["upc"] = updates_per_copy(meta)
        row["updates"] = (int(meta["total_timesteps"]) - int(meta.get("learning_starts", 0))) \
            // int(meta.get("train_frequency", 10)) * int(meta.get("gradient_steps", 1))
        out[key].append(row)
    return out


def verdict(default: list[float], slow: list[float]) -> dict:
    """P1 and the outcome of the test, by the pre-registered rule."""
    default, slow = np.asarray(default, float), np.asarray(slow, float)
    d, lo, hi, p = week4.welch_ci(slow, default)
    base = float(default.mean())
    reduction = 1.0 - float(slow.mean()) / base if base else float("nan")
    if base < MIN_DEFAULT_EXCESS:
        outcome = "NO VERDICT"
    elif reduction >= SUPPORT_REDUCTION and hi < 0:
        outcome = "SUPPORTS"
    elif reduction <= AGAINST_REDUCTION:
        outcome = "AGAINST"
    else:
        outcome = "PARTIAL"
    return {"diff": d, "lo": lo, "hi": hi, "p": p, "reduction": reduction, "outcome": outcome}


def dose(data: dict) -> dict:
    """P2: ordered means and Spearman's rho between log(updates per copy) and per-seed excess."""
    means = [float(np.mean([r["primary"] for r in data[k]])) for k in KEYS]
    x = [np.log(r["upc"]) for k in KEYS for r in data[k]]
    y = [r["primary"] for k in KEYS for r in data[k]]
    rho, p = stats.spearmanr(x, y)
    ordered = all(a > b for a, b in zip(means, means[1:]))
    return {"means": means, "rho": float(rho), "p": float(p), "ordered": ordered,
            "holds": bool(ordered and rho < 0 and p < 0.05)}


def _p(p: float) -> str:
    return "–" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def summary(runs_dir: Path) -> str:
    data = load(runs_dir)
    keys = [k for k in KEYS if data.get(k)]
    if not keys:
        return f"No finished {ENV} runs of {KEYS} under {runs_dir}."
    sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else 0.0  # noqa: E731
    fmt = lambda a, f: f"{np.mean(a):{f}} ± {sd(a):{f.lstrip('+')}}"  # noqa: E731
    col = lambda k, m: [r[m] for r in data[k]]  # noqa: E731
    late = f"{LATE[0] // 1000}k–{LATE[1] // 1000}k"
    lines = [f"### {ENV}: gradient updates per target copy (copy every 500 env steps)", "",
             "| Updates per copy | Seeds | Gradient updates in total | **Peak excess over envelope, ≤ 500k (primary)** "
             f"| Excess, first {MATCHED_COPIES} copies | Peak max-Q (step) | Probes > {CEILING + MARGIN:g} "
             f"| Return, {late} | Median bias, {late} |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        lines.append(
            f"| {data[k][0]['upc']} | {len(data[k])} | {data[k][0]['updates']:,} | **{fmt(col(k, 'primary'), '+.1f')}** "
            f"| {fmt(col(k, 'excess_copies'), '+.1f')} | {fmt(col(k, 'peak'), '.1f')} "
            f"({np.mean(col(k, 'peak_step')) / 1000:.0f}k) | {100 * np.mean(col(k, 'over')):.0f}% "
            f"| {fmt(col(k, 'late_return'), '.1f')} | {fmt(col(k, 'late_bias'), '+.3f')} |")

    complete = all(len(data.get(k, [])) >= 2 for k in KEYS)
    lines += ["", "### Pre-registered tests", ""]
    if not complete:
        lines.append("Not every group has at least 2 finished seeds yet: no tests.")
        return "\n".join(lines)
    v = verdict(col(DEFAULT, "primary"), col(SLOW, "primary"))
    dz = dose(data)
    worst = max(col(SLOW, "peak_whole"))
    p3 = worst <= CEILING + MARGIN
    lines += ["| Prediction | Result | Holds? |", "|---|---|---|",
              f"| P1: 500 vs 50 updates per copy, peak excess | {v['diff']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}], "
              f"p {_p(v['p'])}; reduction {100 * v['reduction']:.0f}% (needs ≥ {100 * SUPPORT_REDUCTION:.0f}% "
              f"and interval < 0) | {'yes' if v['outcome'] == 'SUPPORTS' else 'no'} |",
              "| P2: means fall with updates per copy; Spearman ρ < 0, p < 0.05 | means "
              + " > ".join(f"{m:+.1f}" for m in dz["means"]) + f" ({'in order' if dz['ordered'] else 'not in order'}); "
              f"ρ = {dz['rho']:+.2f}, p {_p(dz['p'])} | {'yes' if dz['holds'] else 'no'} |",
              f"| P3: 500 updates per copy never above {CEILING + MARGIN:g} | largest max-Q of any seed {worst:.1f} "
              f"| {'yes' if p3 else 'no'} |",
              "", f"**Outcome (rule on P1): {v['outcome']}.** {OUTCOME_TEXT[v['outcome']]}",
              "", "| Other comparisons (a − b, not predicted) | Peak excess, ≤ 500k | Return, " + late + " |",
              "|---|---|---|"]
    for name, a, b in [("150 vs 50 updates per copy", "dqn_g3", DEFAULT), ("500 vs 150 updates per copy", SLOW, "dqn_g3"),
                       ("500 vs 50 updates per copy", SLOW, DEFAULT)]:
        cells = []
        for m in ("primary", "late_return"):
            d, lo, hi, p = week4.welch_ci(col(a, m), col(b, m))
            cells.append(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}], p {_p(p)}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "*Envelope after n target copies: 100 (1 − 0.99^(n+1)); every run has the same copy schedule, "
              "so copies, envelope and ε match at every probe. The replay ratio (updates per env step) rises "
              "with updates per copy: 0.1, 0.3, 1.0.*"]
    return "\n".join(lines)


def dose_figure(runs_dir: Path, out: Path) -> Path:
    """(a) max-Q over training, median and IQR across seeds, against the envelope and the ceiling;
    (b) each seed's primary measure against updates per copy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    import plot_report as pr

    groups = pr.load_groups([str(runs_dir)]).get(ENV, {})
    data = load(runs_dir)
    keys = [k for k in KEYS if groups.get(k) and data.get(k)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.patch.set_facecolor(pr.SURFACE)
    pr.style_axis(ax1, "a  Largest prediction on probe states", "max Q")
    pr.style_axis(ax2, "b  Peak excess over the envelope, per seed", "max Q − envelope (≤ 500k)")
    ls = 10_000
    steps = np.arange(ls, TOTAL + 1, 1000)
    env_line = [week4.envelope(week4.copies(s, sweep.DEFAULT_TARGET_PERIOD, ls), CEILING, R_MAX) for s in steps]
    (env_h,) = ax1.plot(steps, env_line, color=pr.INK_2, linewidth=1.2, linestyle="--", label="copy envelope")
    ceil_h = ax1.axhline(CEILING + MARGIN, color=pr.INK_2, linewidth=1.0, linestyle=":", label=f"ceiling + {MARGIN:g}")
    handles = []
    for k in keys:
        s, med, q25, q75 = pr.stack_quantiles(groups[k], "bias/q_max")
        line = pr.line_with_band(ax1, s, med, None, k, q25, q75)
        if line is not None:
            line.set_label(f"{data[k][0]['upc']} updates per copy" + (" (default)" if k == DEFAULT else ""))
            handles.append(line)
    ax1.set_xlabel("environment steps (median and IQR across seeds)", color=pr.INK_2, fontsize=9)
    ax1.set_xlim(0, TOTAL)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    # the three lines converge near 100 late in training, so identity goes in a legend, not end labels
    ax1.legend(handles=handles + [env_h, ceil_h], frameon=False, fontsize=8, loc="upper right")
    rng = np.random.default_rng(0)
    for k in keys:
        upc = data[k][0]["upc"]
        y = np.array([r["primary"] for r in data[k]])
        x = upc * np.exp(rng.uniform(-0.08, 0.08, len(y)))
        ax2.plot([upc / 1.3, upc * 1.3], [y.mean()] * 2, color=pr.INK_2, linewidth=1.5, zorder=2)
        ax2.scatter(x, y, s=64, color=pr.COLORS[k], edgecolor=pr.SURFACE, linewidth=2, zorder=3)
    ax2.axhline(0, color=pr.INK_2, linewidth=1.0)
    ax2.set_xscale("log")
    ax2.set_xticks([data[k][0]["upc"] for k in keys])
    ax2.set_xticklabels([str(data[k][0]["upc"]) for k in keys])
    ax2.minorticks_off()
    ax2.set_xlabel("gradient updates per target copy (bar = mean)", color=pr.INK_2, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=pr.SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--total-timesteps", type=int, default=TOTAL)
    p.add_argument("--probe-episodes", type=int, default=10)
    p.add_argument("--max-parallel", type=int, default=6)
    p.add_argument("--runs-dir", default=str(Path("runs") / "week5"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    runs_dir = Path(args.runs_dir)

    if not args.summary_only:
        jobs = week5_jobs(args.seeds, args.total_timesteps, args.probe_episodes, runs_dir)
        to_run, skipped, moved = sweep.prepare(jobs, runs_dir, args.total_timesteps, args.dry_run)
        cpu_h, wall_h = sweep.estimate_hours(to_run, args.total_timesteps, args.max_parallel)
        print(f"{len(jobs)} jobs: {len(skipped)} already finished, {len(to_run)} to run, "
              f"{len(moved)} unfinished run folder(s) {'would be' if args.dry_run else ''} moved to _incomplete/")
        print(f"estimate ({sweep.DQN_SPS} / {sweep.DQN_G3_SPS} / {sweep.DQN_G10_SPS} SPS per run for 50 / 150 / 500 "
              f"updates per copy): {cpu_h:.1f} CPU-hours, about {wall_h:.1f} h wall at {args.max_parallel} at once")
        if args.dry_run:
            for job in to_run:
                print("  python -m " + " ".join(job["argv"]))
            return
        results = run_all([(j["name"], j["argv"]) for j in to_run], log_dir=runs_dir / "logs",
                          max_parallel=args.max_parallel)
        failed = [name for name, code in results if code != 0]
        if failed:
            raise SystemExit(f"FAILED ({len(failed)}): {failed}  -> rerun the same command to retry only these")

    if not args.no_plot:
        import plot_report
        for path in plot_report.plot([str(runs_dir)], str(Path("figures") / "week5.png")):
            print(f"wrote {path}")
        print(f"wrote {dose_figure(runs_dir, Path('figures') / 'week5_dose.png')}")
    text = summary(runs_dir)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    out = Path("figures") / "week5_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
