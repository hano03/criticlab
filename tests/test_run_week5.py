"""run_week5.py: the 30 jobs, updates-per-copy keys, and the pre-registered outcome rule."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _scripts / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_week2 = _load("run_week2")
run_week5 = _load("run_week5")
plot_report = _load("plot_report")


def test_thirty_jobs_slowest_first_fixed_copy_interval(tmp_path):
    jobs = run_week5.week5_jobs(list(range(10)), 500_000, 10, tmp_path)
    assert len(jobs) == 30 and len({j["name"] for j in jobs}) == 30
    assert [j["key"][1] for j in jobs[::10]] == ["dqn_g10", "dqn_g3", "dqn"]
    for job in jobs:
        argv = job["argv"]
        steps = {"dqn": None, "dqn_g3": "3", "dqn_g10": "10"}[job["key"][1]]
        assert (argv[argv.index("--gradient-steps") + 1] if "--gradient-steps" in argv else None) == steps
        assert "--target-network-frequency" not in argv and "--double" not in argv
        assert argv[argv.index("--probe-max-episode-steps") + 1] == "1000"
        assert job["key"][0] == "CartPole-v1" and "total" not in job


def test_keys_labels_and_updates_per_copy():
    assert run_week2.algo_key({"algo": "dqn"}) == "dqn"                       # week-4 runs unchanged
    assert run_week2.algo_key({"algo": "dqn", "gradient_steps": 1}) == "dqn"
    assert run_week2.algo_key({"algo": "dqn", "gradient_steps": 10}) == "dqn_g10"
    assert run_week2.algo_key({"algo": "dqn", "double": True, "target_network_frequency": 5000,
                               "gradient_steps": 3}) == "double_dqn_tgt5000_g3"
    meta = {"algo": "dqn", "gradient_steps": 3}
    assert plot_report.group_key(meta) == run_week2.algo_key(meta)
    assert plot_report.label("dqn_g3") == "DQN, 150 updates per copy"
    assert plot_report.label("dqn_g10") == "DQN, 500 updates per copy"
    assert run_week5.updates_per_copy({"target_network_frequency": 500, "train_frequency": 10}) == 50
    assert run_week5.updates_per_copy({"target_network_frequency": 500, "train_frequency": 10,
                                       "gradient_steps": 10}) == 500
    assert run_week5.updates_per_copy({"target_network_frequency": 5000, "train_frequency": 10}) == 500
    assert plot_report.COLORS["dqn_g10"] == plot_report.COLORS["dqn_tgt5000"]  # same 500 per copy


def test_outcome_rule_follows_the_preregistration():
    default = [150, 160, 155, 148, 152]
    assert run_week5.verdict(default, [5, 8, 6, 4, 7])["outcome"] == "SUPPORTS"
    assert run_week5.verdict(default, [140, 150, 145, 160, 155])["outcome"] == "AGAINST"
    assert run_week5.verdict(default, [170, 180, 175, 190, 185])["outcome"] == "AGAINST"   # an increase
    assert run_week5.verdict(default, [70, 80, 75, 72, 78])["outcome"] == "PARTIAL"
    assert run_week5.verdict([10, 20, 15, 12, 18], [1, 2, 1, 2, 1])["outcome"] == "NO VERDICT"
    # a 90% mean reduction whose interval still reaches 0 is not support
    v = run_week5.verdict(default, [-200, 230, 5, -150, 190])
    assert v["reduction"] >= 0.75 and v["hi"] >= 0 and v["outcome"] == "PARTIAL"


def _report(folder: Path, seed: int, gradient_steps: int, q_max: list[float]):
    folder.mkdir(parents=True)
    meta = {"algo": "dqn", "env_id": "CartPole-v1", "seed": seed, "double": False, "total_timesteps": 500_000,
            "target_network_frequency": 500, "train_frequency": 10, "gradient_steps": gradient_steps,
            "learning_starts": 10_000}
    steps = np.linspace(50_000, 500_000, len(q_max)).astype(int)
    probes = [{"step": int(s), "metrics": {"bias/q_max": q, "eval/deterministic_return_mean": 500.0 - seed - 0.1 * q,
                                           "bias/norm_bias_median": 0.0}} for s, q in zip(steps, q_max)]
    (folder / "critic_report.json").write_text(json.dumps({"meta": meta, "probes": probes}))


def test_summary_on_synthetic_runs_reports_dose_response(tmp_path):
    for seed in range(3):
        _report(tmp_path / f"a{seed}", seed, 1, [80, 250 + seed, 120, 101])
        _report(tmp_path / f"b{seed}", seed, 3, [70, 160 + seed, 105, 100])
        _report(tmp_path / f"c{seed}", seed, 10, [50, 98 + seed, 99, 99])
    data = run_week5.load(tmp_path)
    assert {k: len(v) for k, v in data.items()} == {"dqn": 3, "dqn_g3": 3, "dqn_g10": 3}
    assert [data[k][0]["upc"] for k in run_week5.KEYS] == [50, 150, 500]
    assert data["dqn"][0]["updates"] == 49_000 and data["dqn_g10"][0]["updates"] == 490_000
    env = run_week5.week4.envelope(run_week5.week4.copies(200_000, 500, 10_000), 100.0, 1.0)
    assert abs(data["dqn"][0]["primary"] - (250 - env)) < 1e-9     # the peak at 200k sets the excess
    d = run_week5.dose(data)
    assert d["ordered"] and d["rho"] < 0 and d["holds"]
    text = run_week5.summary(tmp_path)
    assert "**Outcome (rule on P1): SUPPORTS.**" in text and "| P3:" in text
    out = run_week5.dose_figure(tmp_path, tmp_path / "dose.png")
    assert out.is_file() and out.stat().st_size > 10_000
