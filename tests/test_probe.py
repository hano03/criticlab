"""End-to-end tests of probe() with a tiny fake environment (no MuJoCo needed).

FakeEnv: observation = number of steps taken so far, reward 1 every step.
It terminates after `term_at` steps, or truncates at `limit` steps.
"""

import json
import math

import numpy as np
import pytest

from criticlab.diagnostics import CriticReport, probe


class FakeEnv:
    def __init__(self, term_at=None, limit=10):
        self.term_at, self.limit, self.t = term_at, limit, 0

    def reset(self, seed=None):
        self.t = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        terminated = self.term_at is not None and self.t >= self.term_at
        truncated = (not terminated) and self.t >= self.limit
        return np.array([float(self.t)], dtype=np.float32), 1.0, terminated, truncated, {}


def deterministic(obs):
    return np.zeros(1, dtype=np.float32), None


def true_g(obs, term_at, gamma):
    """Exact return from state t when the episode terminates after term_at steps."""
    t = obs[:, 0].astype(int)
    return np.array([sum(gamma**k for k in range(term_at - ti)) for ti in t])


def test_terminated_episodes_constant_critic():
    env = FakeEnv(term_at=5, limit=100)
    m, raw = probe(env, deterministic, lambda o, a: np.full(len(o), 2.0),
                   n_episodes=2, gamma=0.5, horizon=500)
    g = np.array([1.9375, 1.875, 1.75, 1.5, 1.0])
    assert m["probe/usable_states"] == 10            # termination: every state usable
    assert math.isclose(m["bias/bias_mean"], 2.0 - g.mean())
    assert m["probe/return_mean"] == 5.0
    assert len(raw["q"]) == len(raw["g"]) == 10


def test_truncation_drops_tail_states():
    env = FakeEnv(term_at=None, limit=10)
    m, _ = probe(env, deterministic, lambda o, a: np.zeros(len(o)),
                 n_episodes=3, gamma=0.9, horizon=4)
    assert m["probe/usable_states"] == 3 * 7          # t <= 10 - 4 -> 7 states per episode


def test_all_states_truncated_away():
    env = FakeEnv(term_at=None, limit=3)
    m, raw = probe(env, deterministic, lambda o, a: np.zeros(len(o)),
                   n_episodes=2, gamma=0.9, horizon=500)
    assert m["probe/usable_states"] == 0 and raw == {"q": [], "g": []}


def test_twin_critics():
    gamma, term_at = 0.5, 5
    env = FakeEnv(term_at=term_at, limit=100)
    q_fn = lambda o, a: (true_g(o, term_at, gamma) + 1.0, true_g(o, term_at, gamma) + 3.0)
    m, _ = probe(env, deterministic, q_fn, n_episodes=2, gamma=gamma)
    assert math.isclose(m["bias/bias_mean"], 2.0)      # average of the twins
    assert math.isclose(m["bias_min/bias_mean"], 1.0)  # minimum of the twins
    assert math.isclose(m["twin/abs_diff_mean"], 2.0)
    assert math.isclose(m["bias/pearson"], 1.0)


def test_soft_returns_used_when_alpha_given():
    # logp = -1 every step, alpha = 0.5, gamma = 0.5, 3 steps:
    # H_2 = 1, H_1 = 1 + 0.5*(1 + 0.5) = 1.75, H_0 = 1 + 0.5*(1.75 + 0.5) = 2.125
    env = FakeEnv(term_at=3, limit=100)
    stochastic = lambda obs: (np.zeros(1, dtype=np.float32), -1.0)
    exact = {0: 2.125, 1: 1.75, 2: 1.0}
    q_fn = lambda o, a: np.array([exact[int(x)] for x in o[:, 0]])
    m, _ = probe(env, stochastic, q_fn, n_episodes=1, gamma=0.5, alpha=0.5)
    assert abs(m["bias/bias_mean"]) < 1e-12


def test_alpha_without_logps_raises():
    env = FakeEnv(term_at=3, limit=100)
    with pytest.raises(ValueError):
        probe(env, deterministic, lambda o, a: np.zeros(len(o)), n_episodes=1, alpha=0.2)


def test_report_round_trip(tmp_path):
    env = FakeEnv(term_at=4, limit=100)
    m, raw = probe(env, deterministic, lambda o, a: np.zeros(len(o)), n_episodes=1)
    path = tmp_path / "run" / "critic_report.json"
    report = CriticReport(path, meta={"algo": "td3", "seed": 0})
    report.add(10_000, {**m, "bias/pearson": float("nan")}, raw)
    data = json.loads(path.read_text(encoding="utf-8"))  # strict JSON: NaN must become null
    assert data["meta"]["algo"] == "td3"
    assert data["probes"][0]["step"] == 10_000
    assert data["probes"][0]["metrics"]["bias/pearson"] is None
