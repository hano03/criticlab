"""run_week5b.py: the 15 new jobs, pooling cells from three sweeps, and the pre-registered rule."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _scripts / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_week5b = _load("run_week5b")


def test_fifteen_new_jobs_long_with_rescaled_exploration(tmp_path):
    jobs = run_week5b.week5b_jobs(tmp_path)
    assert [(j["key"][1], j["key"][2]) for j in jobs] == (
        [("dqn_tgt5000_g10", s) for s in range(10)] + [("dqn_tgt5000", s) for s in range(5, 10)])
    for job in jobs:
        argv = job["argv"]
        assert job["total"] == 1_500_000 and argv[argv.index("--total-timesteps") + 1] == "1500000"
        assert argv[argv.index("--target-network-frequency") + 1] == "5000"
        assert abs(float(argv[argv.index("--exploration-fraction") + 1]) * 1_500_000 - 250_000) < 1e-6
        assert argv[argv.index("--probe-max-episode-steps") + 1] == "1000"
        assert ("--gradient-steps" in argv) == (job["key"][1] == "dqn_tgt5000_g10")


def test_outcome_rule_follows_the_preregistration():
    a, b, c = [155, 160, 150, 158], [370, 300, 420, 380], [7, 5, 9, 6]
    assert run_week5b.verdict(a, b, c, [10, 8, 12, 9])["outcome"] == "PROTECTS"
    assert run_week5b.verdict(a, b, c, [360, 390, 340, 400])["outcome"] == "DOES NOT"
    assert run_week5b.verdict(a, b, c, [500, 600, 550, 520])["outcome"] == "DOES NOT"      # worse than B
    assert run_week5b.verdict(a, b, c, [200, 180, 220, 190])["outcome"] == "PARTLY"
    assert run_week5b.verdict(a, b, [40, 45, 50, 35], [10, 8, 12, 9])["outcome"] == "NO VERDICT"
    assert run_week5b.verdict(a, [90, 100, 95, 105], c, [10, 8, 12, 9])["outcome"] == "NO VERDICT"
    v = run_week5b.verdict(a, b, c, [10, 8, 12, 9])
    assert abs(v["protection"] - (np.mean(b) - 9.75) / (np.mean(b) - 6.75)) < 1e-12
    est, lo, hi = v["interaction"]
    assert abs(est - ((9.75 - 6.75) - (np.mean(b) - np.mean(a)))) < 1e-9 and lo < est < hi
    # a large mean protection whose interval for D - B still reaches 0 is not PROTECTS
    v = run_week5b.verdict(a, b, c, [-900, 20, 30, 900])
    assert v["protection"] >= 0.75 and v["hi"] >= 0 and v["outcome"] == "PARTLY"


def _report(folder: Path, key: str, seed: int, bump: float, finished: bool = True, total=None):
    """50 probes that track the copy envelope (1 below it), except probe 10, which exceeds it by `bump`."""
    _, period, ratio = next(v for v in run_week5b.CELLS.values() if v[0] == key)
    total = total or (500_000 if period == 500 else 1_500_000)
    folder.mkdir(parents=True)
    meta = {"algo": "dqn", "env_id": "CartPole-v1", "seed": seed, "double": False, "total_timesteps": total,
            "target_network_frequency": period, "train_frequency": 10, "gradient_steps": int(ratio * 10),
            "learning_starts": 10_000}
    if ratio == 0.1:
        del meta["gradient_steps"]                     # week-4 runs predate the field
    w4 = run_week5b.week4
    steps = [total * (i + 1) // 50 - (0 if finished else 1) for i in range(50)]
    probes = []
    for i, st in enumerate(steps):
        q = w4.envelope(w4.copies(st, period, 10_000), 100.0, 1.0) + (bump if i == 10 else -1.0)
        probes.append({"step": st, "metrics": {"bias/q_max": q, "eval/deterministic_return_mean": 480.0 - seed - 0.1 * q,
                                               "bias/norm_bias_median": 0.001 * q * (seed + 1)}})
    (folder / "critic_report.json").write_text(json.dumps({"meta": meta, "probes": probes}))


def _sweeps(tmp_path):
    d4, d5, d5b = tmp_path / "w4", tmp_path / "w5", tmp_path / "w5b"
    for s in range(5):
        _report(d4 / f"c{s}", "dqn_tgt5000", s, 6 + s)
    for s in range(10):
        _report(d5 / f"a{s}", "dqn", s, 150 + s)
        _report(d5 / f"b{s}", "dqn_g10", s, 350 + 7 * s)
        _report(d5b / f"d{s}", "dqn_tgt5000_g10", s, 9 + s)
    for s in range(5, 10):
        _report(d5b / f"c{s}", "dqn_tgt5000", s, 6 + s)
    _report(d5b / "unfinished", "dqn_tgt5000_g10", 11, 9, finished=False)
    _report(d5 / "wrong_length", "dqn", 12, 150, total=300_000)
    _report(d4 / "b_wrong_length", "dqn_g10", 13, 350, total=1_500_000)
    return run_week5b.sources(d5, d4, d5b)


def test_cells_pool_three_sweeps_and_skip_unfinished_runs(tmp_path):
    src = _sweeps(tmp_path)
    data = run_week5b.load(src)
    assert {c: sorted(r["seed"] for r in rows) for c, rows in data.items()} == {
        "A": list(range(10)), "B": list(range(10)), "C": list(range(10)), "D": list(range(10))}
    assert abs(np.mean([r["primary"] for r in data["B"]]) - (350 + 7 * 4.5)) < 1e-9   # the bump, exactly
    text = run_week5b.summary(src)
    assert "**Outcome: PROTECTS.**" in text and "Interaction" in text
    out = run_week5b.figure(src, tmp_path / "f.png")
    assert out.is_file() and out.stat().st_size > 10_000


def test_week4_cell_a_runs_are_not_read(tmp_path):
    """Regression (Day 31 dry run): week 4 also ran plain DQN with copies every 500 steps, cell A's
    configuration. Cell A comes from week 5 only, so those runs neither count nor trip the duplicate check."""
    src = _sweeps(tmp_path)
    d4 = src["C"][0][0]
    for s in range(5):
        _report(d4 / f"week4_dqn_{s}", "dqn", s, 900 + s)            # week 4's plain DQN, seeds 0-4
    _report(d4 / "week4_double", "dqn", 0, 900)                      # and its Double DQN, same key family
    meta_path = d4 / "week4_double" / "critic_report.json"
    report = json.loads(meta_path.read_text())
    report["meta"]["double"] = True
    meta_path.write_text(json.dumps(report))
    data = run_week5b.load(src)
    assert sorted(r["seed"] for r in data["A"]) == list(range(10))
    assert max(r["primary"] for r in data["A"]) < 200                # none of week 4's +900 bumps


def test_seeds_outside_a_cells_designated_range_are_ignored(tmp_path):
    src = _sweeps(tmp_path)
    d5b = src["D"][0][0]
    _report(d5b / "c_seed0_in_week5b", "dqn_tgt5000", 0, 500)       # C seed 0 belongs to week 4 only
    _report(d5b / "d_seed10", "dqn_tgt5000_g10", 10, 500)            # D has seeds 0-9 only
    data = run_week5b.load(src)
    assert sorted(r["seed"] for r in data["C"]) == list(range(10)) and max(r["primary"] for r in data["C"]) < 100
    assert sorted(r["seed"] for r in data["D"]) == list(range(10))


def test_the_same_seed_twice_in_a_designated_source_is_an_error(tmp_path):
    src = _sweeps(tmp_path)
    _report(src["A"][0][0] / "a3_again", "dqn", 3, 150)              # a second week-5 cell-A run of seed 3
    with pytest.raises(SystemExit, match="seed counted twice: cell A, seed 3"):
        run_week5b.load(src)
    src = _sweeps(tmp_path / "second")
    _report(src["C"][0][0] / "c2_again", "dqn_tgt5000", 2, 6)         # a second week-4 cell-C run of seed 2
    with pytest.raises(SystemExit, match="seed counted twice: cell C, seed 2"):
        run_week5b.load(src)
