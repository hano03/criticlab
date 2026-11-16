"""The week-2 sweep, or any smaller one: algorithms x environments x seeds, run in parallel.

    python scripts/run_week2.py --dry-run            # list the 40 jobs and the time estimate, start nothing
    python scripts/run_week2.py                      # run them, 6 at a time, then draw the figures
    python scripts/run_week2.py --max-parallel 8

A smaller sweep with the same tool (the Day 7 TD3 check):
    python scripts/run_week2.py --algos td3 --envs Hopper-v5 --seeds 1 2 3 --total-timesteps 100000 --runs-dir runs/td3_check

Safe to stop (Ctrl+C) and restart: finished runs are skipped, and an unfinished run of the same
configuration is moved to <runs-dir>/_incomplete/ before it starts again, so figures never mix in
partial runs. Slow jobs (SAC) start first, which shortens the total wall time.

PPO configurations (ppo, ppo_lam0.8, ppo_lam1.0) are available too; the week-3 sweep is
scripts/run_week3.py, which uses the functions here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import run_all  # noqa: E402

# key -> (module, extra flags). The key is also the run's --exp-name.
ALGOS = {
    "sac": ("criticlab.sac", []),
    "td3": ("criticlab.td3", []),
    "td3_no_cdq": ("criticlab.td3", ["--no-clipped-double-q"]),
    "ddpg": ("criticlab.ddpg", []),
    "ppo": ("criticlab.ppo", []),                       # GAE lambda 0.95, CleanRL's default
    "ppo_lam0.8": ("criticlab.ppo", ["--gae-lambda", "0.8"]),
    "ppo_lam1.0": ("criticlab.ppo", ["--gae-lambda", "1.0"]),
}
SLOW_FIRST = ["sac", "td3", "td3_no_cdq", "ddpg"]           # the week-2 sweep, slowest first
JOB_ORDER = SLOW_FIRST + ["ppo", "ppo_lam0.8", "ppo_lam1.0"]  # PPO is the fastest per step
# Day 6 dress rehearsal, Hopper-v5, one thread per process, 3 running at once. PPO: estimated on
# Day 16 from Hopper/HalfCheetah runs with 10 probe episodes, scaled to the user's machine by the
# Day 15 InvertedPendulum speed ratio (977 vs 870 SPS).
PPO_SPS = 700
MEASURED_SPS = {"sac": 75, "td3": 127, "td3_no_cdq": 127, "ddpg": 127,
                "ppo": PPO_SPS, "ppo_lam0.8": PPO_SPS, "ppo_lam1.0": PPO_SPS}
DEFAULT_GAE_LAMBDA = 0.95


def algo_key(meta: dict) -> str:
    """The configuration name of a run, from its metadata (same rule as plot_report.group_key)."""
    algo = meta.get("algo")
    if algo == "td3" and meta.get("clipped_double_q") is False:
        return "td3_no_cdq"
    if algo == "ppo":
        lam = float(meta.get("gae_lambda", DEFAULT_GAE_LAMBDA))
        return "ppo" if abs(lam - DEFAULT_GAE_LAMBDA) < 1e-9 else f"ppo_lam{lam}"
    return algo


def config_key(meta: dict) -> tuple:
    return (meta.get("env_id"), algo_key(meta), meta.get("seed"))


def expected_last_step(meta: dict) -> int | None:
    """Step of a finished run's final probe. PPO stops after whole rollouts: 1M steps with
    2048-step rollouts ends at 488 x 2048 = 999,424."""
    total = meta.get("total_timesteps")
    if total is None:
        return None
    if meta.get("algo") == "ppo" and meta.get("num_steps"):
        return (total // meta["num_steps"]) * meta["num_steps"]
    return total


def build_jobs(algos, envs, seeds, total, probe_episodes, runs_dir) -> list[dict]:
    jobs = []
    for key in [k for k in JOB_ORDER if k in algos]:
        module, extra = ALGOS[key]
        for env_id in envs:
            for seed in seeds:
                jobs.append({
                    "name": f"{env_id}__{key}__s{seed}",
                    "key": (env_id, key, seed),
                    "argv": [module, "--env-id", env_id, "--seed", str(seed),
                             "--total-timesteps", str(total), "--probe-episodes", str(probe_episodes),
                             "--runs-dir", str(runs_dir), "--exp-name", key, *extra],
                })
    return jobs


def scan(runs_dir: Path, total: int):
    """Finished configuration keys, and the run folders of unfinished ones."""
    finished, unfinished = set(), defaultdict(list)
    for report in runs_dir.glob("*/critic_report.json"):  # top level only: skips _incomplete/
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        steps = [p["step"] for p in data.get("probes", [])]
        key = config_key(data["meta"])
        meta = data["meta"]
        if meta.get("total_timesteps") == total and steps and steps[-1] == expected_last_step(meta):
            finished.add(key)
        else:
            unfinished[key].append(report.parent)
    return finished, unfinished


def prepare(jobs, runs_dir: Path, total: int, dry_run: bool):
    """Drop finished jobs; move unfinished runs of the remaining ones aside (unless dry run)."""
    finished, unfinished = scan(runs_dir, total)
    to_run, skipped, moved = [], [], []
    for job in jobs:
        if job["key"] in finished:
            skipped.append(job)
            continue
        for folder in unfinished.get(job["key"], []):
            moved.append(folder)
            if not dry_run:
                target = runs_dir / "_incomplete" / folder.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(folder), str(target))
        to_run.append(job)
    return to_run, skipped, moved


def estimate_hours(jobs, total, max_parallel) -> tuple[float, float]:
    """CPU-hours, and wall-hours from simulating the launcher: jobs start in list order, each on
    the first free slot. (20 equal jobs at 6 at once take 4 rounds, not 20/6 = 3.3.)"""
    hours = [total / MEASURED_SPS[j["key"][1]] / 3600 for j in jobs]
    slots = [0.0] * max(1, min(max_parallel, len(jobs)))
    for h in hours:
        i = slots.index(min(slots))
        slots[i] += h
    return sum(hours), max(slots) if hours else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--algos", nargs="+", default=SLOW_FIRST, choices=list(ALGOS),
                   help="default: the four week-2 configurations")
    p.add_argument("--envs", nargs="+", default=["Hopper-v5", "HalfCheetah-v5"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--probe-episodes", type=int, default=10)
    p.add_argument("--max-parallel", type=int, default=6)
    p.add_argument("--runs-dir", default=str(Path("runs") / "week2"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    jobs = build_jobs(args.algos, args.envs, args.seeds, args.total_timesteps,
                      args.probe_episodes, runs_dir)
    to_run, skipped, moved = prepare(jobs, runs_dir, args.total_timesteps, args.dry_run)
    cpu_h, wall_h = estimate_hours(to_run, args.total_timesteps, args.max_parallel)

    print(f"{len(jobs)} jobs: {len(skipped)} already finished, {len(to_run)} to run, "
          f"{len(moved)} unfinished run folder(s) {'would be' if args.dry_run else ''} moved to _incomplete/")
    print(f"estimate (measured speeds): {cpu_h:.1f} CPU-hours, about {wall_h:.1f} h wall at "
          f"{args.max_parallel} at once (more if per-process speed drops at higher parallelism)")
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
        import plot_report
        out = Path("figures") / f"{runs_dir.name}.png"
        for path in plot_report.plot([str(runs_dir)], str(out)):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
