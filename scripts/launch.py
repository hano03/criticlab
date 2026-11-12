"""Run training jobs in parallel: one CPU thread each, one log file per job. Windows-safe.

Used by dress_rehearsal.py (Day 6) and run_week2.py (Day 8). Each job is (name, args), and runs
as `python -u -m <args...>` with OMP_NUM_THREADS=1 so parallel jobs do not fight over cores.
Ctrl+C stops every running job.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def run_all(jobs: list[tuple[str, list[str]]], log_dir, max_parallel: int = 6,
            poll_seconds: float = 2.0) -> list[tuple[str, int]]:
    """Run jobs at most `max_parallel` at a time. Returns (name, exit code) in finishing order."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    pending = list(jobs)
    running: list[tuple[str, subprocess.Popen, object]] = []
    results: list[tuple[str, int]] = []
    try:
        while pending or running:
            while pending and len(running) < max_parallel:
                name, args = pending.pop(0)
                log = open(log_dir / f"{name}.log", "w", encoding="utf-8")
                proc = subprocess.Popen([sys.executable, "-u", "-m", *args],
                                        stdout=log, stderr=subprocess.STDOUT, env=env)
                running.append((name, proc, log))
                print(f"[launch] started  {name}", flush=True)
            for item in list(running):
                name, proc, log = item
                code = proc.poll()
                if code is not None:
                    log.close()
                    running.remove(item)
                    results.append((name, code))
                    status = "finished" if code == 0 else f"FAILED (exit {code}), see {log_dir / (name + '.log')}"
                    print(f"[launch] {status}  {name}  ({len(results)}/{len(jobs)})", flush=True)
            if running:
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        for _, proc, log in running:
            proc.terminate()
            log.close()
        print("[launch] interrupted: stopped all running jobs", flush=True)
        raise
    return results
