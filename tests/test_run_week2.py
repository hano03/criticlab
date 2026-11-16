"""run_week2.py: job list, slow-first order, ablation flags, resume logic (no torch needed)."""

import importlib.util
import json
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))
_spec = importlib.util.spec_from_file_location("run_week2", _scripts / "run_week2.py")
run_week2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_week2)

ENVS = ["Hopper-v5", "HalfCheetah-v5"]


def _jobs(runs_dir, algos=run_week2.SLOW_FIRST, seeds=(0, 1, 2, 3, 4)):
    return run_week2.build_jobs(algos, ENVS, list(seeds), 300_000, 10, runs_dir)


def _fake_run(runs_dir: Path, name: str, algo: str, env_id: str, seed: int, last_step: int,
              total: int = 300_000, **meta) -> Path:
    folder = runs_dir / name
    folder.mkdir(parents=True)
    report = {"meta": {"algo": algo, "env_id": env_id, "seed": seed, "total_timesteps": total, **meta},
              "probes": [{"step": s, "metrics": {}, "raw": {"q": [], "g": []}}
                         for s in range(10_000, last_step + 1, 10_000)]}
    (folder / "critic_report.json").write_text(json.dumps(report), encoding="utf-8")
    return folder


def test_full_sweep_is_40_jobs_slowest_first(tmp_path):
    jobs = _jobs(tmp_path)
    assert len(jobs) == 40
    assert [j["key"][1] for j in jobs[:10]] == ["sac"] * 10
    assert len({j["name"] for j in jobs}) == 40


def test_ablation_job_flags():
    jobs = [j for j in _jobs(Path("runs")) if j["key"][1] == "td3_no_cdq"]
    assert len(jobs) == 10
    argv = jobs[0]["argv"]
    assert argv[0] == "criticlab.td3" and "--no-clipped-double-q" in argv
    assert argv[argv.index("--exp-name") + 1] == "td3_no_cdq"
    assert argv[argv.index("--probe-episodes") + 1] == "10"


def test_resume_skips_finished_and_moves_unfinished(tmp_path):
    _fake_run(tmp_path, "done", "td3", "Hopper-v5", 0, 300_000)
    half = _fake_run(tmp_path, "half", "td3", "Hopper-v5", 1, 120_000)
    _fake_run(tmp_path, "done_ablation", "td3", "Hopper-v5", 0, 300_000, clipped_double_q=False)
    jobs = _jobs(tmp_path, algos=["td3", "td3_no_cdq"], seeds=(0, 1))
    to_run, skipped, moved = run_week2.prepare(jobs, tmp_path, 300_000, dry_run=False)
    assert {j["key"] for j in skipped} == {("Hopper-v5", "td3", 0), ("Hopper-v5", "td3_no_cdq", 0)}
    assert len(to_run) == len(jobs) - 2
    assert moved == [half]
    assert not half.exists() and (tmp_path / "_incomplete" / "half" / "critic_report.json").exists()


def test_dry_run_moves_nothing(tmp_path):
    half = _fake_run(tmp_path, "half", "sac", "HalfCheetah-v5", 2, 50_000)
    _, _, moved = run_week2.prepare(_jobs(tmp_path), tmp_path, 300_000, dry_run=True)
    assert moved == [half] and half.exists()


def test_a_shorter_run_does_not_count_as_finished(tmp_path):
    _fake_run(tmp_path, "rehearsal", "ddpg", "Hopper-v5", 0, 100_000, total=100_000)
    _, skipped, moved = run_week2.prepare(_jobs(tmp_path, algos=["ddpg"]), tmp_path, 300_000, dry_run=True)
    assert skipped == [] and len(moved) == 1


def test_estimate_uses_measured_speeds(tmp_path):
    cpu, wall = run_week2.estimate_hours(_jobs(tmp_path), 300_000, 6)
    expected = (10 * 300_000 / 75 + 30 * 300_000 / 127) / 3600
    assert abs(cpu - expected) < 1e-9
    assert expected / 6 <= wall <= expected / 6 + 300_000 / 75 / 3600  # rounds, at most one job extra


def test_wall_estimate_counts_rounds(tmp_path):
    jobs = run_week2.build_jobs(["ddpg"], ENVS, [0, 1, 2], 300_000, 10, tmp_path)  # 6 equal jobs
    one = 300_000 / 127 / 3600
    assert abs(run_week2.estimate_hours(jobs, 300_000, 6)[1] - one) < 1e-9
    assert abs(run_week2.estimate_hours(jobs, 300_000, 4)[1] - 2 * one) < 1e-9   # 4 + 2: two rounds


def test_week2_default_is_unchanged_by_ppo():
    assert run_week2.SLOW_FIRST == ["sac", "td3", "td3_no_cdq", "ddpg"]
