"""launch.run_all: parallel jobs, per-job logs, exit codes reported (no torch needed)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "launch", Path(__file__).resolve().parents[1] / "scripts" / "launch.py")
launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)


def test_runs_jobs_and_reports_exit_codes(tmp_path):
    jobs = [("ok_a", ["json.tool", "--help"]),
            ("ok_b", ["json.tool", "--help"]),
            ("broken", ["criticlab_module_that_does_not_exist"])]
    results = dict(launch.run_all(jobs, log_dir=tmp_path / "logs", max_parallel=2, poll_seconds=0.1))
    assert results["ok_a"] == 0 and results["ok_b"] == 0
    assert results["broken"] != 0
    assert "No module named" in (tmp_path / "logs" / "broken.log").read_text(encoding="utf-8")
    assert (tmp_path / "logs" / "ok_a.log").stat().st_size > 0
