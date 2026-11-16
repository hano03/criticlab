"""PPO's pieces that decide the critic's targets: GAE, the running normaliser, and a tiny run."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from criticlab import ppo  # noqa: E402


def naive_gae(rewards, values, dones, next_value, gamma, lam):
    """Definition: A_t = sum_l (gamma*lam)^l * delta_{t+l}, stopping at the end of the episode."""
    n, out = len(rewards), []
    vals = list(values) + [next_value]
    for t in range(n):
        total, coef = 0.0, 1.0
        for k in range(t, n):
            delta = rewards[k] + gamma * vals[k + 1] * (1 - dones[k]) - vals[k]
            total += coef * delta
            if dones[k]:
                break
            coef *= gamma * lam
        out.append(total)
    return np.array(out)


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_gae_matches_its_definition(lam):
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=12), rng.normal(size=12)
    d = np.zeros(12)
    d[[3, 8]] = 1.0  # two episode ends inside the rollout
    adv, ret = ppo.compute_gae(r, v, d, next_value=0.7, gamma=0.9, lam=lam)
    np.testing.assert_allclose(adv, naive_gae(r, v, d, 0.7, 0.9, lam), atol=1e-12)
    np.testing.assert_allclose(ret, adv + v)


def test_gae_lambda_one_gives_monte_carlo_returns():
    # lambda = 1, gamma = 1: the critic's target is the plain sum of the remaining rewards
    r = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([0.0, 1.0, 0.0, 0.0])          # first episode ends at t = 1
    _, ret = ppo.compute_gae(r, np.zeros(4), d, next_value=10.0, gamma=1.0, lam=1.0)
    np.testing.assert_allclose(ret, [3.0, 2.0, 3.0 + 4.0 + 10.0, 4.0 + 10.0])


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(1)
    data = rng.normal(3.0, 2.0, size=(1000, 4))
    rms = ppo.RunningMeanStd(shape=(4,))
    for chunk in np.array_split(data, 37):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, data.mean(axis=0), atol=1e-3)
    np.testing.assert_allclose(rms.var, data.var(axis=0), rtol=1e-3)


def test_tiny_ppo_run_writes_report_in_real_reward_units(tmp_path):
    args = ppo.Args(env_id="Pendulum-v1", total_timesteps=600, num_steps=100, num_minibatches=4,
                    update_epochs=2, probe_every=300, probe_episodes=1, probe_horizon=50,
                    eval_episodes=1, runs_dir=str(tmp_path))
    data = json.loads((Path(ppo.main(args)) / "critic_report.json").read_text(encoding="utf-8"))
    assert data["meta"]["algo"] == "ppo"
    assert [p["step"] for p in data["probes"]] == [300, 600]
    m = data["probes"][-1]["metrics"]
    assert m["probe/usable_states"] == 151
    assert m["bias/norm_bias_median"] is not None
    assert m["ppo/reward_scale"] > 0
    assert "ppo/explained_variance" in m
    assert "twin/abs_diff_mean" not in m          # one critic
