"""run_week5c.py: the 20 new jobs, buffer-size keys, designated sources and the pre-registered rule."""

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


run_week2 = _load("run_week2")
run_week5c = _load("run_week5c")
plot_report = _load("plot_report")


def test_twenty_new_jobs_vary_only_the_buffer(tmp_path):
    jobs = run_week5c.week5c_jobs(tmp_path)
    assert [(j["key"][1], j["key"][2]) for j in jobs] == (
        [("dqn_buf2500", s) for s in range(10)] + [("dqn_buf40000", s) for s in range(10)])
    for job in jobs:
        argv = job["argv"]
        size = {"dqn_buf2500": "2500", "dqn_buf40000": "40000"}[job["key"][1]]
        assert argv[argv.index("--buffer-size") + 1] == size
        assert argv[argv.index("--total-timesteps") + 1] == "500000" and "total" not in job
        assert argv[argv.index("--probe-max-episode-steps") + 1] == "1000"
        for flag in ("--gradient-steps", "--target-network-frequency", "--double", "--exploration-fraction"):
            assert flag not in argv


def test_buffer_keys_and_labels():
    assert run_week2.algo_key({"algo": "dqn", "buffer_size": 10_000}) == "dqn"           # earlier runs unchanged
    assert run_week2.algo_key({"algo": "dqn", "buffer_size": 2_500}) == "dqn_buf2500"
    assert run_week2.algo_key({"algo": "dqn", "gradient_steps": 10, "buffer_size": 40_000}) == "dqn_g10_buf40000"
    meta = {"algo": "dqn", "buffer_size": 40_000}
    assert plot_report.group_key(meta) == run_week2.algo_key(meta)
    assert plot_report.label("dqn_buf2500") == "DQN, buffer 2,500"


def _rule(small, default, large):
    return run_week5c.verdict({2_500: small, 10_000: default, 40_000: large})["outcome"]


def test_outcome_rule_follows_the_preregistration():
    default = [150, 160, 155, 158, 152]
    assert _rule([60, 70, 65, 62, 68], default, [300, 320, 310, 290, 305]) == "TURNOVER"
    assert _rule([300, 320, 310, 290, 305], default, [60, 70, 65, 62, 68]) == "OVERFITTING"
    assert _rule([150, 158, 152, 160, 155], default, [156, 150, 159, 153, 157]) == "NO EFFECT"
    # extremes clearly differ but the middle is out of order: not a monotone pattern
    assert _rule([60, 70, 65, 62, 68], default, [140, 145, 138, 142, 146]) == "INCONCLUSIVE"
    assert _rule([60, 70, 65, 62, 68], [400, 410, 405, 395, 402], [300, 320, 310, 290, 305]) == "INCONCLUSIVE"
    # an interval too wide to call either a direction or no effect
    assert _rule([10, 300, 50, 250, 100], default, [20, 350, 60, 280, 90]) == "INCONCLUSIVE"
    assert _rule([10, 12, 11, 9, 10], [20, 30, 25, 22, 28], [40, 45, 41, 43, 42]) == "NO VERDICT"


def _report(folder: Path, key: str, seed: int, bump: float, buffer: int, gradient_steps: int = 1):
    """50 probes that track the copy envelope (1 below it), except probe 10, which exceeds it by `bump`."""
    folder.mkdir(parents=True)
    meta = {"algo": "dqn", "env_id": "CartPole-v1", "seed": seed, "double": False, "total_timesteps": 500_000,
            "target_network_frequency": 500, "train_frequency": 10, "gradient_steps": gradient_steps,
            "buffer_size": buffer, "learning_starts": 10_000}
    w4 = run_week5c.week4
    probes = []
    for i in range(50):
        st = 10_000 * (i + 1)
        q = w4.envelope(w4.copies(st, 500, 10_000), 100.0, 1.0) + (bump if i == 10 else -1.0)
        probes.append({"step": st, "metrics": {"bias/q_max": q, "eval/deterministic_return_mean": 480.0 - seed - 0.1 * q,
                                               "bias/norm_bias_median": 0.001 * q * (seed + 1)}})
    (folder / "critic_report.json").write_text(json.dumps({"meta": meta, "probes": probes}))


def _sweeps(tmp_path):
    w5, w5c = tmp_path / "w5", tmp_path / "w5c"
    for s in range(10):
        _report(w5 / f"dqn{s}", "dqn", s, 150 + s, 10_000)
        _report(w5 / f"g3_{s}", "dqn_g3", s, 140, 10_000, gradient_steps=3)     # week 5's other arms
        _report(w5 / f"g10_{s}", "dqn_g10", s, 380, 10_000, gradient_steps=10)
        _report(w5c / f"small{s}", "dqn_buf2500", s, 60 + s, 2_500)
        _report(w5c / f"large{s}", "dqn_buf40000", s, 300 + 2 * s, 40_000)
    _report(w5 / "stray_small", "dqn_buf2500", 0, 999, 2_500)                  # not a designated source
    return run_week5c.sources(w5, w5c)


def test_cells_come_only_from_their_designated_sources(tmp_path):
    src = _sweeps(tmp_path)
    data = run_week5c.load(src)
    assert {b: sorted(r["seed"] for r in rows) for b, rows in data.items()} == {b: list(range(10)) for b in (2_500, 10_000, 40_000)}
    assert abs(np.mean([r["primary"] for r in data[10_000]]) - 154.5) < 1e-9   # week 5's dqn only, not g3 or g10
    assert max(r["primary"] for r in data[2_500]) < 100                        # the stray week-5 run is not read
    text = run_week5c.summary(src)
    assert "**Outcome: TURNOVER.**" in text and "Spearman" in text
    out = run_week5c.figure(src, tmp_path / "f.png")
    assert out.is_file() and out.stat().st_size > 10_000


def test_the_same_seed_twice_in_a_designated_source_is_an_error(tmp_path):
    src = _sweeps(tmp_path)
    _report(src[40_000][0] / "large3_again", "dqn_buf40000", 3, 300, 40_000)
    with pytest.raises(SystemExit, match="seed counted twice: buffer 40000, seed 3"):
        run_week5c.load(src)
