"""Day 22 DQN check: DQN and Double DQN on CartPole-v1 and Acrobot-v1, seeds 0-2, 500k steps.

    python scripts/dqn_check.py                    # 12 runs, 6 at a time (about 10 minutes)
    python scripts/dqn_check.py --max-parallel 12

Two questions:
  1. Is the implementation right? Final greedy returns should be in the range of CleanRL's dqn.py
     (CartPole-v1 488.69 +- 16.11, Acrobot-v1 -91.54 +- 7.20).
  2. Does the critic go past what any policy can earn? Rewards are bounded, so Q is too:
     CartPole pays +1 per step, so Q <= 1 / (1 - 0.99) = 100; Acrobot pays -1 per step (0 at the
     goal), so Q <= 0. A prediction outside that range is "soft divergence" (van Hasselt et al.,
     2018), visible without any Monte Carlo return. The table counts probes whose largest
     prediction is more than MARGIN = 10 (a tenth of the 0-100 value range) above the ceiling, so
     a near-goal Acrobot state predicted at +0.5 is not counted as divergence.

CartPole probe episodes may run to 1,000 steps (--probe-max-episode-steps): the training limit of
500 would leave a solved policy almost no states with 500 observed rewards.
Logs: runs/dqn_check/logs/. Figures: figures/dqn_check__<env>.png (+ _hero).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import run_all  # noqa: E402

RUNS_DIR = Path("runs") / "dqn_check"
ENVS = {  # env -> (extra flags, upper bound on Q)
    "CartPole-v1": (["--probe-max-episode-steps", "1000"], 100.0),
    "Acrobot-v1": ([], 0.0),
}
VARIANTS = {"dqn": [], "double_dqn": ["--double"]}
MARGIN = 10.0


def jobs(seeds) -> list[tuple[str, list[str]]]:
    out = []
    for env_id, (extra, _) in ENVS.items():
        for key, flags in VARIANTS.items():
            for s in seeds:
                out.append((f"{env_id}__{key}__s{s}",
                            ["criticlab.dqn", "--env-id", env_id, "--seed", str(s), "--exp-name", key,
                             "--runs-dir", str(RUNS_DIR), *extra, *flags]))
    return out


def summarize(runs_dir: Path) -> str:
    rows = defaultdict(list)
    for report in sorted(runs_dir.glob("*/critic_report.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        meta = data["meta"]
        bound = ENVS[meta["env_id"]][1]
        learned = [p["metrics"] for p in data["probes"] if p["step"] > meta["learning_starts"]]
        steps = [p["step"] for p in data["probes"] if p["step"] > meta["learning_starts"]]
        q_max = [m.get("bias/q_max") for m in learned]
        pairs = [(s, q) for s, q in zip(steps, q_max) if q is not None]
        last = data["probes"][-1]["metrics"]
        peak_step, peak = max(pairs, key=lambda x: x[1]) if pairs else (0, float("nan"))
        rows[(meta["env_id"], "double_dqn" if meta["double"] else "dqn")].append({
            "ret": last["eval/deterministic_return_mean"],
            "bias": last.get("bias/norm_bias_median"),
            "peak": peak, "peak_step": peak_step,
            "final_qmax": last.get("bias/q_max"),
            "over": float(np.mean([q > bound + MARGIN for _, q in pairs])) if pairs else float("nan"),
        })
    lines = ["| Env (ceiling) | Variant | Seeds | Final return | Final median bias | Peak max-Q (step) "
             f"| Final max-Q | Probes > ceiling + {MARGIN:g} |",
             "|---|---|---|---|---|---|---|---|"]
    for (env_id, key), rs in sorted(rows.items()):
        f = lambda k: np.array([r[k] for r in rs if r[k] is not None], float)  # noqa: E731
        sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else 0.0  # noqa: E731
        lines.append(f"| {env_id} ({ENVS[env_id][1]:g}) | {key} | {len(rs)} | "
                     f"{f('ret').mean():.1f} ± {sd(f('ret')):.1f} | {f('bias').mean():+.3f} ± {sd(f('bias')):.3f} | "
                     f"{f('peak').mean():.1f} ± {sd(f('peak')):.1f} ({f('peak_step').mean() / 1000:.0f}k) | "
                     f"{f('final_qmax').mean():.1f} | {100 * f('over').mean():.0f}% |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max-parallel", type=int, default=6)
    p.add_argument("--summary-only", action="store_true", help="skip training, summarise existing runs")
    args = p.parse_args()
    if not args.summary_only:
        if RUNS_DIR.exists() and any(RUNS_DIR.rglob("critic_report.json")):
            raise SystemExit(f"{RUNS_DIR} already holds runs; rename or delete it, or use --summary-only")
        results = run_all(jobs(args.seeds), log_dir=RUNS_DIR / "logs", max_parallel=args.max_parallel)
        failed = [name for name, code in results if code != 0]
        if failed:
            raise SystemExit(f"failed: {failed}; see {RUNS_DIR / 'logs'}")
    import plot_report
    for path in plot_report.plot([str(RUNS_DIR)], str(Path("figures") / "dqn_check.png")):
        print(f"wrote {path}")
    text = summarize(RUNS_DIR)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("\n" + text)
    out = Path("figures") / "dqn_check_summary.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
