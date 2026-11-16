"""Day 16 PPO check: InvertedPendulum-v5, 150k steps, seeds 0-2, all three at once.

    python scripts/ppo_check.py

Why this task: a solved policy earns reward 1 for each of 1,000 steps, so the true discounted
value of every probed state is known in advance, about 99.3-100 with gamma = 0.99. If PPO's
critic is converted back to reward units correctly (sigma * V), the probe must report a value
near that and a normalised bias near 0. If the conversion were missing, V would be ~30x too small.

Logs go to runs/ppo_check/logs/; the table at the end is what to report back.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import run_all  # noqa: E402

RUNS_DIR = Path("runs") / "ppo_check"
SEEDS = (0, 1, 2)
JOBS = [(f"ppo_s{s}", ["criticlab.ppo", "--env-id", "InvertedPendulum-v5", "--seed", str(s),
                       "--total-timesteps", "150000", "--runs-dir", str(RUNS_DIR)]) for s in SEEDS]


def summary_row(report: Path) -> str:
    data = json.loads(report.read_text(encoding="utf-8"))
    last = data["probes"][-1]
    m = last["metrics"]
    sps = [p["metrics"].get("perf/sps_window") for p in data["probes"][1:]]
    sps = [x for x in sps if x is not None]
    f = lambda k, fmt: format(m[k], fmt) if m.get(k) is not None else "n/a"  # noqa: E731
    return (f"seed {data['meta']['seed']}  step {last['step']:>6}  "
            f"eval {f('eval/deterministic_return_mean', '7.1f')}  "
            f"sigma*V {f('bias/q_mean', '6.2f')}  "
            f"norm_bias median {f('bias/norm_bias_median', '+.3f')}  "
            f"EV {f('ppo/explained_variance', '+.2f')}  "
            f"sigma {f('ppo/reward_scale', '5.2f')}  "
            f"SPS {sum(sps) / max(len(sps), 1):5.0f}")


def main() -> None:
    if RUNS_DIR.exists() and any(RUNS_DIR.rglob("critic_report.json")):
        raise SystemExit(f"{RUNS_DIR} already holds runs; rename or delete it first")
    results = run_all(JOBS, log_dir=RUNS_DIR / "logs", max_parallel=len(JOBS))
    failed = [name for name, code in results if code != 0]
    if failed:
        raise SystemExit(f"failed: {failed}; see {RUNS_DIR / 'logs'}")
    print("\nFinal probe of each seed (expect eval 1000, sigma*V ~ 97-100, norm_bias within +-0.05):")
    for report in sorted(RUNS_DIR.rglob("critic_report.json")):
        print("  " + summary_row(report))


if __name__ == "__main__":
    main()
