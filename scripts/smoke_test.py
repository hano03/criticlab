"""Smoke test: each algorithm (TD3, DDPG, SAC, PPO) runs 5k steps on Pendulum-v1 and writes a valid critic report.

Usage (from the repo root):
    python scripts/smoke_test.py          # every algorithm in ALGOS
    python scripts/smoke_test.py td3      # just one
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ALGOS = ["td3", "ddpg", "sac", "ppo"]
# Off-policy algorithms start learning after 1k random steps. PPO has no warm-up; instead its
# rollout length must divide 5000 so the run ends exactly at the second probe (10 updates x 500).
ALGO_ARGS = {"ppo": ["--num-steps", "500", "--num-minibatches", "5", "--update-epochs", "2"]}
DEFAULT_ARGS = ["--learning-starts", "1000"]


def smoke(algo: str) -> None:
    with tempfile.TemporaryDirectory() as runs_dir:
        cmd = [sys.executable, "-m", f"criticlab.{algo}",
               "--env-id", "Pendulum-v1",
               "--total-timesteps", "5000",
               *ALGO_ARGS.get(algo, DEFAULT_ARGS),
               "--probe-every", "2500",
               "--probe-episodes", "2",
               "--probe-horizon", "50",  # Pendulum episodes are 200 steps; 500 would leave no states
               "--runs-dir", runs_dir]
        subprocess.run(cmd, check=True, env={**os.environ, "OMP_NUM_THREADS": "1"})

        reports = list(Path(runs_dir).glob("*/critic_report.json"))
        assert len(reports) == 1, f"{algo}: expected one critic_report.json, found {len(reports)}"
        data = json.loads(reports[0].read_text(encoding="utf-8"))
        steps = [p["step"] for p in data["probes"]]
        assert steps == [2500, 5000], f"{algo}: probe steps {steps}"
        last = data["probes"][-1]["metrics"]
        assert last["probe/usable_states"] > 0, f"{algo}: no usable probe states"
        assert last["bias/norm_bias_mean"] is not None, f"{algo}: bias metric missing"
        print(f"[smoke] {algo}: OK  (final norm_bias={last['bias/norm_bias_mean']:.3f})")


def main() -> None:
    for algo in sys.argv[1:] or ALGOS:
        smoke(algo)
    print("[smoke] all passed")


if __name__ == "__main__":
    main()
