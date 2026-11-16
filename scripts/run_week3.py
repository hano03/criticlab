"""The week-3 PPO sweep: the main grid and the GAE lambda study, run in parallel.

    python scripts/run_week3.py --dry-run          # list the 20 jobs and the time estimate, start nothing
    python scripts/run_week3.py                    # run them, 6 at a time, then draw the figures
    python scripts/run_week3.py --max-parallel 8

The 20 jobs:
    main grid      PPO (GAE lambda 0.95)   x Hopper-v5, HalfCheetah-v5 x seeds 0-4
    lambda study   PPO, lambda 0.8 and 1.0  x Hopper-v5                 x seeds 0-4
                   (lambda 0.95 on Hopper comes from the main grid)

1M steps (CleanRL's PPO default, for the benchmark check), 10 probe episodes per probe. PPO runs
whole 2,048-step rollouts, so a 1M-step run ends, and is counted as finished, at step 999,424.

Same resume rules as run_week2.py: Ctrl+C is safe; rerunning skips finished runs and moves
unfinished ones to <runs-dir>/_incomplete/. Figures written at the end:
    figures/week3__<env>.png (+ _hero)   PPO lambda 0.95, Hopper and HalfCheetah
    figures/week3_lambda.png (+ _hero)   Hopper, lambda 0.8 / 0.95 / 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_week2 as sweep  # noqa: E402
from launch import run_all  # noqa: E402

MAIN_ENVS = ["Hopper-v5", "HalfCheetah-v5"]
LAMBDA_ENV = "Hopper-v5"
LAMBDA_KEYS = ["ppo_lam0.8", "ppo_lam1.0"]


def week3_jobs(seeds, total, probe_episodes, runs_dir, with_lambda: bool = True) -> list[dict]:
    jobs = sweep.build_jobs(["ppo"], MAIN_ENVS, seeds, total, probe_episodes, runs_dir)
    if with_lambda:
        jobs += sweep.build_jobs(LAMBDA_KEYS, [LAMBDA_ENV], seeds, total, probe_episodes, runs_dir)
    return jobs


def draw_figures(runs_dir: Path) -> list[Path]:
    import plot_report
    written = plot_report.plot([str(runs_dir)], str(Path("figures") / "week3.png"), only=["ppo"])
    lam_runs = str(runs_dir / f"{LAMBDA_ENV}__*")
    if any(plot_report.group_key(r["meta"]) in LAMBDA_KEYS
           for g in plot_report.load_groups([lam_runs]).values() for rs in g.values() for r in rs):
        written += plot_report.plot([lam_runs], str(Path("figures") / "week3_lambda.png"),
                                    only=["ppo", *LAMBDA_KEYS])
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--probe-episodes", type=int, default=10)
    p.add_argument("--max-parallel", type=int, default=6)
    p.add_argument("--runs-dir", default=str(Path("runs") / "week3"))
    p.add_argument("--no-lambda", action="store_true", help="main grid only")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    jobs = week3_jobs(args.seeds, args.total_timesteps, args.probe_episodes, runs_dir,
                      with_lambda=not args.no_lambda)
    to_run, skipped, moved = sweep.prepare(jobs, runs_dir, args.total_timesteps, args.dry_run)
    cpu_h, wall_h = sweep.estimate_hours(to_run, args.total_timesteps, args.max_parallel)

    print(f"{len(jobs)} jobs: {len(skipped)} already finished, {len(to_run)} to run, "
          f"{len(moved)} unfinished run folder(s) {'would be' if args.dry_run else ''} moved to _incomplete/")
    print(f"estimate ({sweep.PPO_SPS} SPS per PPO run): {cpu_h:.1f} CPU-hours, about {wall_h:.1f} h wall "
          f"at {args.max_parallel} at once (more if per-process speed drops at higher parallelism)")
    if args.dry_run:
        for job in to_run:
            print("  python -m " + " ".join(job["argv"]))
        return

    results = run_all([(j["name"], j["argv"]) for j in to_run], log_dir=runs_dir / "logs",
                      max_parallel=args.max_parallel)
    failed = [name for name, code in results if code != 0]
    if failed:
        print(f"FAILED ({len(failed)}): {failed}  -> rerun the same command to retry only these")
    if not args.no_plot and not failed:
        for path in draw_figures(runs_dir):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
