"""run_week4.py: the 40 jobs, target-period keys, the pre-registered measures and Welch intervals."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _scripts / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_week2 = _load("run_week2")
run_week4 = _load("run_week4")
plot_report = _load("plot_report")


def test_forty_jobs_long_runs_first_with_matching_flags(tmp_path):
    jobs = run_week4.week4_jobs([0, 1, 2, 3, 4], 500_000, 10, tmp_path)
    assert len(jobs) == 40 and len({j["name"] for j in jobs}) == 40
    assert all(j.get("total") == 1_500_000 for j in jobs[:20])      # long runs start first
    for job in jobs:
        env_id, key, _ = job["key"]
        argv = job["argv"]
        total = argv[argv.index("--total-timesteps") + 1]
        assert ("--double" in argv) == key.startswith("double")
        if key.endswith("tgt5000"):
            assert argv[argv.index("--target-network-frequency") + 1] == "5000"
            assert total == "1500000" and job["total"] == 1_500_000
            frac = float(argv[argv.index("--exploration-fraction") + 1])
            assert abs(frac * 1_500_000 - 250_000) < 1e-6           # epsilon reaches 0.05 at 250k
        else:
            assert "--target-network-frequency" not in argv and "--exploration-fraction" not in argv
            assert total == "500000" and "total" not in job
        assert ("--probe-max-episode-steps" in argv) == (env_id == "CartPole-v1")


def test_resume_uses_each_jobs_own_length(tmp_path):
    import json
    jobs = run_week4.week4_jobs([0], 500_000, 10, tmp_path)
    def fake(name, period, total):
        folder = tmp_path / name
        folder.mkdir()
        meta = {"algo": "dqn", "env_id": "CartPole-v1", "seed": 0, "double": False,
                "target_network_frequency": period, "total_timesteps": total}
        (folder / "critic_report.json").write_text(json.dumps(
            {"meta": meta, "probes": [{"step": total, "metrics": {}}]}))
    fake("short_done", 500, 500_000)            # finished 500k run with the default period
    fake("long_as_short", 5000, 500_000)        # a 5,000-period run of the wrong length
    to_run, skipped, moved = run_week2.prepare(jobs, tmp_path, 500_000, dry_run=True)
    assert [j["key"][1] for j in skipped] == ["dqn"]
    assert [m.name for m in moved] == ["long_as_short"]


def test_copies_and_envelope():
    assert run_week4.copies(10_000, 500, 10_000) == 0
    assert run_week4.copies(155_000, 500, 10_000) == 290
    assert run_week4.copies(1_460_000, 5000, 10_000) == 290
    assert abs(run_week4.envelope(0, 100.0, 1.0) - 1.0) < 1e-12
    assert abs(run_week4.envelope(98, 100.0, 1.0) - 100 * (1 - 0.99 ** 99)) < 1e-9
    assert abs(run_week4.envelope(10_000, 100.0, 1.0) - 100.0) < 1e-9   # approaches the ceiling
    assert run_week4.envelope(5, 0.0, None) == 0.0                  # Acrobot: the ceiling itself


def test_target_period_names_agree_between_sweep_and_plots():
    for double in (False, True):
        for period, suffix in ((500, ""), (5000, "_tgt5000"), (2000, "_tgt2000")):
            meta = {"algo": "dqn", "double": double, "target_network_frequency": period}
            key = ("double_dqn" if double else "dqn") + suffix
            assert run_week2.algo_key(meta) == key == plot_report.group_key(meta)
    assert plot_report.label("double_dqn_tgt5000") == "Double DQN, target every 5,000"
    assert plot_report.label("dqn_tgt2000") == "DQN, target every 2,000"


def _report(q_max_by_step, returns_by_step, period=500, learning_starts=10_000):
    probes = [{"step": st, "metrics": {"bias/q_max": q, "eval/deterministic_return_mean": returns_by_step[st],
                                       "bias/norm_bias_median": 0.1}}
              for st, q in q_max_by_step.items()]
    return {"meta": {"learning_starts": learning_starts, "target_network_frequency": period}, "probes": probes}


def test_per_seed_measures():
    q = {10_000: 500.0, 100_000: 250.0, 200_000: 105.0, 400_000: 115.0, 500_000: 101.0, 600_000: 300.0}
    r = {10_000: 9.0, 100_000: 200.0, 200_000: 300.0, 400_000: 480.0, 500_000: 500.0, 600_000: 500.0}
    m = run_week4.per_seed(_report(q, r), ceiling=100.0, r_max=1.0)
    assert m["peak"] == 250.0 and m["peak_step"] == 100_000   # 10k is before learning; 600k is past 500k
    assert m["peak_whole"] == 300.0
    env_100k = run_week4.envelope(run_week4.copies(100_000, 500, 10_000), 100.0, 1.0)
    assert np.isclose(m["excess_steps"], 250.0 - env_100k)
    assert np.isclose(m["excess_copies"], 250.0 - env_100k)  # 290 copies come at 155k: only 100k counts
    assert m["over"] == 0.6                                    # 250, 115, 300 of 5 probes exceed 110
    assert m["late_return"] == 490.0 and np.isclose(m["late_bias"], 0.1)


def test_welch_interval_matches_scipy():
    a, b = [10.0, 12.0, 11.0, 13.0, 9.0], [20.0, 25.0, 22.0, 30.0, 18.0]
    d, lo, hi, p = run_week4.welch_ci(a, b)
    assert np.isclose(d, -12.0)
    assert np.isclose(p, stats.ttest_ind(a, b, equal_var=False).pvalue)
    ci = stats.ttest_ind(a, b, equal_var=False).confidence_interval(0.95)
    assert np.isclose(lo, ci.low) and np.isclose(hi, ci.high)
