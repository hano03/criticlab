"""The week-4 sweep: does a slower target network stop DQN's soft divergence on CartPole?

    python scripts/run_week4.py --dry-run           # list the 40 jobs and the time estimate
    python scripts/run_week4.py --max-parallel 10   # run them, then draw figures and the summary
    python scripts/run_week4.py --summary-only      # summary of existing runs only

The 40 jobs: CartPole-v1, Acrobot-v1 x {DQN, Double DQN} x target network copied every
{500 (CleanRL's default), 5,000} environment steps x seeds 0-4, 10 probe episodes, CleanRL's
other hyperparameters. CartPole probe episodes may run to 1,000 steps.

Why the 5,000-step runs are 1.5M steps long. With hard copies, each copy lets values grow by at
most one more step of reward: after n copies, honest predictions are capped by the copy envelope
E(n) = r_max (1 - gamma^(n+1)) / (1 - gamma) (CartPole: 100 (1 - 0.99^(n+1))). With 5,000 steps
between copies, 500k steps give only 98 copies, so E = 63: a low peak there would just mean the
values had not had time to grow. The default reaches about 290 copies at its peak (~150k steps);
1.5M steps give the 5,000-step runs 298 copies. Their exploration schedule is rescaled
(--exploration-fraction 1/6) so epsilon still reaches 0.05 at 250k steps, as in the 500k runs.

Predictions, written down before the sweep (spec v3.2, Day 23; revised after a one-run pilot, see
the progress log):
  P1  Primary. Within the first 290 target copies, each seed's peak excess of max-Q over the copy
      envelope is much smaller with copies every 5,000 steps than every 500, for DQN and for
      Double DQN alike.
  P1b Over the whole 1.5M-step run, the 5,000-step runs stay within ceiling + 10.
  P2  Double DQN stays close to DQN at both target periods (with 2 actions, the max adds little).
  If the 5,000-step runs overshoot their envelope as much as the 500-step runs, the target period
  is not the cause; the next suspects are the 10k replay buffer and the learning rate.

Measures, per seed (Acrobot's rewards are never positive, so its envelope is simply its ceiling 0):
  peak max-Q and its excess over the envelope, first 500k steps (matched steps) and first 290
  copies (matched copies); share of probes more than 10 above the ceiling and peak max-Q over the
  whole run; mean greedy return and median normalised bias over probes 400k-500k.
Comparisons: difference in means with a 95% Welch confidence interval and p-value.

Resumable like run_week2.py. Figures: figures/week4__<env>.png (+ _hero), all steps;
figures/week4_500k__<env>.png, cut at 500k; tables: figures/week4_summary.md.
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
from launch import run_all  # noqa: E402

GAMMA = 0.99
ENVS = {"CartPole-v1": (100.0, 1.0), "Acrobot-v1": (0.0, None)}  # env -> (Q ceiling, r_max if > 0)
CARTPOLE_FLAGS = ["--probe-max-episode-steps", "1000"]
KEYS = ["dqn", "double_dqn", "dqn_tgt5000", "double_dqn_tgt5000"]
TOTAL, LONG_TOTAL = 500_000, 1_500_000
LONG_FLAGS = ["--exploration-fraction", str(250_000 / LONG_TOTAL)]  # epsilon reaches 0.05 at 250k
MARGIN = 10.0
MATCHED_COPIES = 290
LATE = (400_000, 500_000)
# (label, group a, group b): each row reports mean(a) - mean(b)
COMPARISONS = [
    ("P1 DQN: target 5,000 vs 500", "dqn_tgt5000", "dqn"),
    ("P1 Double DQN: target 5,000 vs 500", "double_dqn_tgt5000", "double_dqn"),
    ("P2 target 500: Double vs DQN", "double_dqn", "dqn"),
    ("P2 target 5,000: Double vs DQN", "double_dqn_tgt5000", "dqn_tgt5000"),
]


def week4_jobs(seeds, total, probe_episodes, runs_dir, long_total: int = LONG_TOTAL) -> list[dict]:
    jobs = sweep.build_jobs(KEYS, list(ENVS), seeds, total, probe_episodes, runs_dir)
    for job in jobs:
        argv = list(job["argv"])
        if job["key"][1].endswith("_tgt5000"):
            argv[argv.index("--total-timesteps") + 1] = str(long_total)
            argv += LONG_FLAGS
            job["total"] = long_total
        if job["key"][0] == "CartPole-v1":
            argv += CARTPOLE_FLAGS
        job["argv"] = argv
    jobs.sort(key=lambda j: -j.get("total", total))  # long runs first: shorter total wall time
    return jobs


def copies(step: int, period: int, learning_starts: int) -> int:
    """Target copies made by `step`: one every `period` steps once learning has started."""
    return max(0, step // period - learning_starts // period)


def envelope(n: int, ceiling: float, r_max) -> float:
    """Largest value n hard target copies can honestly support (the ceiling if r_max is None)."""
    if r_max is None:
        return ceiling
    return min(ceiling, r_max * (1 - GAMMA ** (n + 1)) / (1 - GAMMA))


def per_seed(report: dict, ceiling: float, r_max=None, late=LATE, matched_steps: int = TOTAL,
             matched_copies: int = MATCHED_COPIES) -> dict:
    """The pre-registered measures for one run."""
    meta = report["meta"]
    period = int(meta.get("target_network_frequency", sweep.DEFAULT_TARGET_PERIOD))
    ls = int(meta.get("learning_starts", 0))
    pts = [(p["step"], copies(p["step"], period, ls), p["metrics"]["bias/q_max"])
           for p in report["probes"] if p["step"] > ls and p["metrics"].get("bias/q_max") is not None]
    nan = float("nan")

    def peak(rows):
        return max(rows, key=lambda r: r[2]) if rows else (nan, nan, nan)

    def excess(rows):
        return max((q - envelope(n, ceiling, r_max) for _, n, q in rows), default=nan)

    early = [r for r in pts if r[0] <= matched_steps]
    by_copies = [r for r in pts if r[1] <= matched_copies]
    late_p = [p["metrics"] for p in report["probes"] if late[0] <= p["step"] <= late[1]]

    def late_mean(key):
        vals = [m[key] for m in late_p if m.get(key) is not None]
        return float(np.mean(vals)) if vals else nan

    step, _, q = peak(early)
    return {
        "peak": float(q), "peak_step": float(step),
        "excess_steps": float(excess(early)),
        "excess_copies": float(excess(by_copies)),
        "copies_reached": int(max((r[1] for r in pts), default=0)),
        "peak_whole": float(peak(pts)[2]),
        "over": float(np.mean([r[2] > ceiling + MARGIN for r in pts])) if pts else nan,
        "late_return": late_mean("eval/deterministic_return_mean"),
        "late_bias": late_mean("bias/norm_bias_median"),
    }


def welch_ci(a, b, level: float = 0.95):
    """mean(a) - mean(b), its Welch confidence interval, and the two-sided p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return diff, diff, diff, float("nan")
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = stats.t.ppf(0.5 + level / 2, df) * se
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    return diff, diff - half, diff + half, float(p)


def load(runs_dir: Path) -> dict:
    """(env, key) -> list of per-seed measure dicts. Unfinished runs (_incomplete/) are skipped."""
    out = defaultdict(list)
    for path in sorted(runs_dir.glob("*/critic_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        env_id = report["meta"]["env_id"]
        if env_id in ENVS and report.get("probes"):
            ceiling, r_max = ENVS[env_id]
            out[(env_id, sweep.algo_key(report["meta"]))].append(per_seed(report, ceiling, r_max))
    return out


def summary(runs_dir: Path) -> str:
    data = load(runs_dir)
    sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else 0.0  # noqa: E731
    fmt = lambda a, f: f"{np.mean(a):{f}} ± {sd(a):{f.lstrip('+')}}"  # noqa: E731
    col = lambda rows, k: [r[k] for r in rows]  # noqa: E731
    order = ["dqn", "double_dqn", "dqn_tgt5000", "double_dqn_tgt5000"]
    late = f"{LATE[0] // 1000}k–{LATE[1] // 1000}k"
    lines = []
    for env_id, (ceiling, _) in ENVS.items():
        keys = [k for k in order if (env_id, k) in data]
        if not keys:
            continue
        lines += [f"\n### {env_id} (Q ceiling {ceiling:g})", "",
                  "| Group | Seeds | Copies reached | Peak max-Q, first 500k (step) | Excess over envelope, first 500k "
                  f"| **Excess over envelope, first {MATCHED_COPIES} copies** | Peak max-Q, whole run "
                  f"| Probes > ceiling + {MARGIN:g}, whole run | Return, {late} | Median bias, {late} |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for k in keys:
            rows = data[(env_id, k)]
            lines.append(
                f"| {k} | {len(rows)} | {min(col(rows, 'copies_reached'))} | {fmt(col(rows, 'peak'), '.1f')} "
                f"({np.mean(col(rows, 'peak_step')) / 1000:.0f}k) | {fmt(col(rows, 'excess_steps'), '+.1f')} | "
                f"**{fmt(col(rows, 'excess_copies'), '+.1f')}** | {fmt(col(rows, 'peak_whole'), '.1f')} | "
                f"{100 * np.mean(col(rows, 'over')):.0f}% | {fmt(col(rows, 'late_return'), '.1f')} | "
                f"{fmt(col(rows, 'late_bias'), '+.3f')} |")
        lines += ["", f"| Comparison (a − b) | **Excess, first {MATCHED_COPIES} copies** | Peak max-Q, first 500k "
                  f"| Return, {late} |", "|---|---|---|---|"]
        for name, a, b in COMPARISONS:
            if (env_id, a) not in data or (env_id, b) not in data:
                continue
            cells = []
            for measure in ("excess_copies", "peak", "late_return"):
                xa, xb = col(data[(env_id, a)], measure), col(data[(env_id, b)], measure)
                if len(xa) < 2 or len(xb) < 2:
                    cells.append("–")
                    continue
                d, lo, hi, p = welch_ci(xa, xb)
                p_txt = "–" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")
                cells.append(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}], p {p_txt}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", f"*Envelope after n target copies: r_max (1 − γ^(n+1)) / (1 − γ), capped at the ceiling "
              f"(Acrobot: the ceiling, 0). Copies-matched excess uses probes with at most {MATCHED_COPIES} copies; "
              f"runs that never reach that many copies are compared over the copies they have (see 'Copies reached').*"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--total-timesteps", type=int, default=TOTAL, help="target period 500 runs")
    p.add_argument("--long-total-timesteps", type=int, default=LONG_TOTAL, help="target period 5,000 runs")
    p.add_argument("--probe-episodes", type=int, default=10)
    p.add_argument("--max-parallel", type=int, default=6)
    p.add_argument("--runs-dir", default=str(Path("runs") / "week4"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    runs_dir = Path(args.runs_dir)

    if not args.summary_only:
        jobs = week4_jobs(args.seeds, args.total_timesteps, args.probe_episodes, runs_dir,
                          long_total=args.long_total_timesteps)
        to_run, skipped, moved = sweep.prepare(jobs, runs_dir, args.total_timesteps, args.dry_run)
        cpu_h, wall_h = sweep.estimate_hours(to_run, args.total_timesteps, args.max_parallel)
        print(f"{len(jobs)} jobs: {len(skipped)} already finished, {len(to_run)} to run, "
              f"{len(moved)} unfinished run folder(s) {'would be' if args.dry_run else ''} moved to _incomplete/")
        print(f"estimate ({sweep.DQN_SPS} SPS per run): {cpu_h:.1f} CPU-hours, about {wall_h:.1f} h wall "
              f"at {args.max_parallel} at once (speed per run drops as more run at once)")
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
        for path in (plot_report.plot([str(runs_dir)], str(Path("figures") / "week4.png"))
                     + plot_report.plot([str(runs_dir)], str(Path("figures") / "week4_500k.png"),
                                        max_step=TOTAL)):
            print(f"wrote {path}")
    text = summary(runs_dir)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    out = Path("figures") / "week4_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
