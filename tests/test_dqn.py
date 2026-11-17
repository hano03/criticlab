"""dqn.py: the DQN and Double DQN targets, the epsilon schedule, and a tiny CartPole run."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from criticlab import dqn


class FixedQ(nn.Module):
    """Returns a fixed table of Q-values for every row of the batch (ignores the input)."""

    def __init__(self, table):
        super().__init__()
        self.table = torch.tensor(table, dtype=torch.float32)

    def forward(self, x):
        return self.table.expand(x.shape[0], -1)


def _batch(term: float):
    return {"obs": torch.zeros(1, 4), "acts": torch.zeros(1, dtype=torch.int64),
            "rews": torch.tensor([1.0]), "next_obs": torch.zeros(1, 4), "terms": torch.tensor([term])}


def test_dqn_target_takes_the_target_networks_max():
    online, target = FixedQ([[0.0, 5.0, 1.0]]), FixedQ([[2.0, 3.0, 10.0]])
    y = dqn.td_target(online, target, _batch(0.0), gamma=0.9, double=False)
    assert torch.allclose(y, torch.tensor([1.0 + 0.9 * 10.0]))


def test_double_dqn_target_lets_online_choose_and_target_value():
    online, target = FixedQ([[0.0, 5.0, 1.0]]), FixedQ([[2.0, 3.0, 10.0]])
    y = dqn.td_target(online, target, _batch(0.0), gamma=0.9, double=True)
    assert torch.allclose(y, torch.tensor([1.0 + 0.9 * 3.0]))  # online picks action 1, target says 3


@pytest.mark.parametrize("double", [False, True])
def test_termination_stops_the_bootstrap(double):
    online, target = FixedQ([[0.0, 5.0]]), FixedQ([[7.0, 9.0]])
    assert torch.allclose(dqn.td_target(online, target, _batch(1.0), 0.99, double), torch.tensor([1.0]))


def test_double_target_never_exceeds_the_dqn_target():
    # the max over the target network bounds any single entry of it
    rng = np.random.default_rng(0)
    for _ in range(20):
        online, target = FixedQ(rng.normal(size=(1, 4))), FixedQ(rng.normal(size=(1, 4)))
        assert dqn.td_target(online, target, _batch(0.0), 0.99, True) <= dqn.td_target(
            online, target, _batch(0.0), 0.99, False) + 1e-6


def test_linear_schedule_matches_cleanrl():
    assert dqn.linear_schedule(1.0, 0.05, 100, 0) == 1.0
    assert abs(dqn.linear_schedule(1.0, 0.05, 100, 50) - 0.525) < 1e-12
    assert abs(dqn.linear_schedule(1.0, 0.05, 100, 100) - 0.05) < 1e-12
    assert dqn.linear_schedule(1.0, 0.05, 100, 10_000) == 0.05


def test_replay_buffer_keeps_integer_actions_and_terminated_only():
    rb = dqn.ReplayBuffer(4, obs_dim=2)
    rb.add(np.ones(2), 1, 1.0, np.zeros(2), terminated=False)
    rb.add(np.ones(2), 0, 1.0, np.zeros(2), terminated=True)
    batch = rb.sample(64, np.random.default_rng(0))
    assert batch["acts"].dtype == torch.int64
    assert set(batch["terms"].tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("double", [False, True])
def test_tiny_cartpole_run_writes_a_report(tmp_path, double):
    args = dqn.Args(env_id="CartPole-v1", total_timesteps=600, learning_starts=100, train_frequency=1,
                    target_network_frequency=50, probe_every=300, probe_episodes=2, probe_horizon=10,
                    probe_max_episode_steps=1000, eval_episodes=2, log_every=100, runs_dir=str(tmp_path), double=double,
                    exp_name="double_dqn" if double else "dqn")
    run_dir = dqn.main(args)
    data = json.loads((Path(run_dir) / "critic_report.json").read_text(encoding="utf-8"))
    assert data["meta"]["algo"] == "dqn" and data["meta"]["double"] is double
    assert [p["step"] for p in data["probes"]] == [300, 600]
    m = data["probes"][-1]["metrics"]
    assert m["probe/usable_states"] > 0 and m["bias/norm_bias_median"] is not None
    assert m["eval/deterministic_return_mean"] > 0 and m["bias/q_max"] >= m["bias/q_mean"]
    assert "td/abs_mean" in m and "twin/abs_diff_mean" not in m
    assert m["dqn/epsilon"] < 1.0
