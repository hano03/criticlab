"""Compare training-episode returns near the end of runs, on equal terms across implementations.

Probe returns (deterministic policy) and CleanRL's printed returns (training episodes, with
exploration noise) are not the same quantity. This script reads the same quantity from both:
the returns of training episodes that ended in the last `--window` steps.

    python scripts/last_returns.py runs/td3_check                  # our runs (TensorBoard files)
    python scripts/last_returns.py ref_td3_s1.log ref_td3_s2.log   # CleanRL console logs
    python scripts/last_returns.py runs/td3_check ref_td3_s*.log   # both (wildcards expanded here)
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

# CleanRL prints e.g. "global_step=98765, episodic_return=[312.4]" or "...episodic_return=312.4"
LOG_LINE = re.compile(r"global_step=(\d+),\s*episodic_return=\[?\s*(-?[\d.eE+]+)")


def from_log(path: Path) -> list[tuple[int, float]]:
    pairs = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOG_LINE.search(line)
        if m:
            pairs.append((int(m.group(1)), float(m.group(2))))
    return pairs


def from_tensorboard(run_dir: Path) -> list[tuple[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})  # 0 = keep every point
    acc.Reload()
    if "charts/episodic_return" not in acc.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in acc.Scalars("charts/episodic_return")]


def sources(inputs: list[str]) -> list[tuple[str, list[tuple[int, float]]]]:
    out = []
    for item in inputs:
        for match in sorted(glob.glob(item)) or [item]:
            p = Path(match)
            if p.is_file():
                out.append((p.name, from_log(p)))
            elif p.is_dir():
                runs = sorted({f.parent for f in p.rglob("events.out.tfevents.*")
                               if "_incomplete" not in f.parts})
                out += [(r.name, from_tensorboard(r)) for r in runs]
    return out


def summarise(pairs: list[tuple[int, float]], window: int):
    if not pairs:
        return None
    last_step = max(s for s, _ in pairs)
    tail = [r for s, r in pairs if s > last_step - window]
    return last_step, len(tail), float(np.mean(tail)), float(np.max(tail))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("inputs", nargs="+", help="run folders and/or CleanRL log files")
    p.add_argument("--window", type=int, default=10_000, help="steps before the end to average over")
    args = p.parse_args()
    print(f"{'run':<55} {'last step':>9} {'episodes':>8} {'mean':>8} {'max':>8}")
    for name, pairs in sources(args.inputs):
        s = summarise(pairs, args.window)
        if s is None:
            print(f"{name:<55}  no episodic returns found")
        else:
            print(f"{name:<55} {s[0]:>9} {s[1]:>8} {s[2]:>8.1f} {s[3]:>8.1f}")


if __name__ == "__main__":
    main()
