"""Day 6 dress rehearsal: TD3, DDPG and SAC on Hopper-v5, 100k steps, seed 0, all three at once.

    python scripts/dress_rehearsal.py

Logs go to runs/dress_rehearsal/logs/<algo>.log; the figure to figures/dress_rehearsal.png
(and _hero.png). Each algorithm keeps its own default learning_starts (TD3 and DDPG 25k, SAC 5k).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import run_all  # noqa: E402
import plot_report  # noqa: E402

RUNS_DIR = Path("runs") / "dress_rehearsal"
COMMON = ["--env-id", "Hopper-v5", "--seed", "0", "--total-timesteps", "100000",
          "--runs-dir", str(RUNS_DIR)]
JOBS = [(algo, [f"criticlab.{algo}", *COMMON]) for algo in ("td3", "ddpg", "sac")]


def main() -> None:
    if RUNS_DIR.exists() and any(RUNS_DIR.rglob("critic_report.json")):
        raise SystemExit(f"{RUNS_DIR} already holds runs; rename or delete it first, "
                         "so old and new runs are not mixed in one figure")
    results = run_all(JOBS, log_dir=RUNS_DIR / "logs", max_parallel=len(JOBS))
    failed = [name for name, code in results if code != 0]
    if failed:
        raise SystemExit(f"failed: {failed}; see {RUNS_DIR / 'logs'}")
    for path in plot_report.plot([str(RUNS_DIR)], str(Path("figures") / "dress_rehearsal.png")):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
