"""run_week3.py: the 20 PPO jobs, lambda flags, PPO's finish rule, and one naming rule everywhere."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _scripts / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_week2 = _load("run_week2")
run_week3 = _load("run_week3")
TOTAL = 1_000_000
LAST = 999_424  # 488 whole 2,048-step rollouts


def _jobs(runs_dir, **kw):
    return run_week3.week3_jobs([0, 1, 2, 3, 4], TOTAL, 10, runs_dir, **kw)


def _fake_ppo(runs_dir: Path, name: str, env_id: str, seed: int, last_step: int, lam: float = 0.95):
    folder = runs_dir / name
    folder.mkdir(parents=True)
    steps = list(range(10_000, last_step + 1, 10_000))
    if last_step % 10_000:
        steps.append(last_step)
    meta = {"algo": "ppo", "env_id": env_id, "seed": seed, "total_timesteps": TOTAL,
            "num_steps": 2048, "gae_lambda": lam}
    report = {"meta": meta, "probes": [{"step": s, "metrics": {}, "raw": {"q": [], "g": []}} for s in steps]}
    (folder / "critic_report.json").write_text(json.dumps(report), encoding="utf-8")
    return folder


def test_twenty_jobs_main_grid_then_lambda_study(tmp_path):
    jobs = _jobs(tmp_path)
    assert len(jobs) == 20 and len({j["name"] for j in jobs}) == 20
    main = [j for j in jobs if j["key"][1] == "ppo"]
    assert sorted({j["key"][0] for j in main}) == ["HalfCheetah-v5", "Hopper-v5"] and len(main) == 10
    lam = [j for j in jobs if j["key"][1] != "ppo"]
    assert {j["key"][0] for j in lam} == {"Hopper-v5"} and len(lam) == 10
    assert jobs.index(main[-1]) < jobs.index(lam[0])
    assert len(_jobs(tmp_path, with_lambda=False)) == 10


@pytest.mark.parametrize("key, flag", [("ppo", None), ("ppo_lam0.8", "0.8"), ("ppo_lam1.0", "1.0")])
def test_job_flags(tmp_path, key, flag):
    argv = next(j["argv"] for j in _jobs(tmp_path) if j["key"][1] == key)
    assert argv[0] == "criticlab.ppo"
    assert argv[argv.index("--exp-name") + 1] == key
    assert argv[argv.index("--total-timesteps") + 1] == str(TOTAL)
    assert argv[argv.index("--probe-episodes") + 1] == "10"
    if flag is None:
        assert "--gae-lambda" not in argv
    else:
        assert argv[argv.index("--gae-lambda") + 1] == flag


def test_key_from_metadata_matches_job_key_and_plot_groups():
    plot_report = _load("plot_report")
    cases = {0.95: "ppo", 0.8: "ppo_lam0.8", 1.0: "ppo_lam1.0"}
    for lam, key in cases.items():
        meta = {"algo": "ppo", "gae_lambda": lam}
        assert run_week2.algo_key(meta) == key == plot_report.group_key(meta)
    for meta in ({"algo": "td3", "clipped_double_q": False}, {"algo": "td3", "clipped_double_q": True},
                 {"algo": "sac"}, {"algo": "ddpg"}, {"algo": "ppo"}):
        assert run_week2.algo_key(meta) == plot_report.group_key(meta)


def test_ppo_run_ending_at_999424_counts_as_finished(tmp_path):
    _fake_ppo(tmp_path, "done", "Hopper-v5", 0, LAST)
    half = _fake_ppo(tmp_path, "half", "Hopper-v5", 1, 500_000)
    _fake_ppo(tmp_path, "done_lam", "Hopper-v5", 2, LAST, lam=0.8)
    to_run, skipped, moved = run_week2.prepare(_jobs(tmp_path), tmp_path, TOTAL, dry_run=True)
    assert {j["key"] for j in skipped} == {("Hopper-v5", "ppo", 0), ("Hopper-v5", "ppo_lam0.8", 2)}
    assert len(to_run) == 18 and moved == [half]


def test_expected_last_step():
    assert run_week2.expected_last_step({"algo": "ppo", "total_timesteps": TOTAL, "num_steps": 2048}) == LAST
    assert run_week2.expected_last_step({"algo": "td3", "total_timesteps": 300_000}) == 300_000


def test_estimate_four_rounds_at_six(tmp_path):
    cpu, wall = run_week2.estimate_hours(_jobs(tmp_path), TOTAL, 6)
    one = TOTAL / run_week2.PPO_SPS / 3600
    assert abs(cpu - 20 * one) < 1e-9 and abs(wall - 4 * one) < 1e-9
