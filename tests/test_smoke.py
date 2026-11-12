"""Tiny in-process runs of each algorithm on Pendulum-v1 (seconds, CPU, no MuJoCo)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from criticlab import ddpg, sac, td3  # noqa: E402


def _tiny(module, tmp_path, **overrides):
    # defaults first, then overrides: a test may override any field, including probe_horizon
    kwargs = dict(env_id="Pendulum-v1", total_timesteps=600, learning_starts=200,
                  probe_every=300, probe_episodes=1, probe_horizon=50, log_every=100,
                  runs_dir=str(tmp_path))
    kwargs.update(overrides)
    run_dir = module.main(module.Args(**kwargs))
    return json.loads((Path(run_dir) / "critic_report.json").read_text(encoding="utf-8"))


def test_td3_writes_report_with_bias_and_twin_metrics(tmp_path):
    data = _tiny(td3, tmp_path)
    assert data["meta"]["algo"] == "td3"
    assert [p["step"] for p in data["probes"]] == [300, 600]
    m = data["probes"][-1]["metrics"]
    assert m["probe/usable_states"] == 151          # 200-step episode, horizon 50: t <= 150
    assert m["bias/norm_bias_mean"] is not None
    assert m["twin/abs_diff_mean"] is not None
    assert m["td/abs_mean"] is not None            # latest TD stats travel with each probe
    assert m["perf/sps_window"] > 0


def test_td3_ablation_runs(tmp_path):
    data = _tiny(td3, tmp_path, clipped_double_q=False)
    assert data["meta"]["clipped_double_q"] is False


def test_ddpg_writes_report_with_single_critic_metrics(tmp_path):
    data = _tiny(ddpg, tmp_path)
    assert data["meta"]["algo"] == "ddpg"
    assert [p["step"] for p in data["probes"]] == [300, 600]
    m = data["probes"][-1]["metrics"]
    assert m["probe/usable_states"] == 151
    assert m["bias/norm_bias_mean"] is not None
    # one critic: no twin disagreement and no min-of-twins bias
    assert "twin/abs_diff_mean" not in m
    assert not any(k.startswith("bias_min/") for k in m)


def test_ddpg_has_none_of_td3s_fixes():
    fields = set(ddpg.Args.__dataclass_fields__)
    assert not fields & {"policy_frequency", "policy_noise", "noise_clip", "clipped_double_q"}


def test_sac_probes_soft_values_and_evaluates_deterministically(tmp_path):
    data = _tiny(sac, tmp_path)
    assert data["meta"]["algo"] == "sac"
    assert [p["step"] for p in data["probes"]] == [300, 600]
    m = data["probes"][-1]["metrics"]
    assert m["probe/usable_states"] == 151
    assert m["probe/alpha"] > 0                      # soft returns used the current alpha
    assert m["bias/norm_bias_mean"] is not None
    assert m["twin/abs_diff_mean"] is not None
    assert m["eval/deterministic_return_mean"] is not None


def test_sac_fixed_alpha_is_passed_to_the_probe(tmp_path):
    data = _tiny(sac, tmp_path, autotune=False, alpha=0.1)
    assert all(p["metrics"]["probe/alpha"] == 0.1 for p in data["probes"])


def test_default_horizon_gives_no_usable_states_on_pendulum(tmp_path):
    # 200-step episodes are always truncated before 500 rewards: bias is not computed, by design
    data = _tiny(td3, tmp_path, probe_horizon=500)
    m = data["probes"][-1]["metrics"]
    assert m["probe/usable_states"] == 0
    assert "bias/norm_bias_mean" not in m
