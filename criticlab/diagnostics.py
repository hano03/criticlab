"""Critic diagnostics: does a critic's Q(s, a) match what actually happens?

This is the only module shared by the algorithm files. It is framework-agnostic:
everything here works on NumPy arrays, and algorithms hand in two small callables.

    act_fn(obs) -> (action, logp)
        One observation in, one action out. `logp` is the log-probability of that
        action under the policy (SAC), or None for deterministic policies (TD3, DDPG).

    q_fn(obs_batch, act_batch) -> q
        Batched critic. Returns a (N,) array, or a tuple of (N,) arrays for twin critics.
        The algorithm file wraps its PyTorch network, e.g.
            q_fn = lambda o, a: (qf1(T(o), T(a)).numpy().ravel(), qf2(T(o), T(a)).numpy().ravel())

Definitions follow REDQ (Chen et al., 2021, redq/utils/bias_utils.py) where they overlap:
    - soft Monte Carlo return:  H_t = r_t + gamma * (H_{t+1} - alpha * logp_{t+1}),  H_{T-1} = r_{T-1}
    - normalised bias per state: (Q - G) / max(|G|, 10)

Where this module differs from REDQ:
    - Truncation (time limit) and termination are treated differently. After a termination the
      return really ends, so every state is usable. After a time-limit truncation the tail is
      missing, so only states with at least `horizon` observed rewards are used.
    - Calibration (Pearson / Spearman between Q and G) and twin-critic disagreement are reported.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

NORM_FLOOR = 10.0  # REDQ's floor for the normalisation denominator


# ---------------------------------------------------------------------------
# Monte Carlo ground truth
# ---------------------------------------------------------------------------

def discounted_returns(rewards, gamma: float) -> np.ndarray:
    """G_t = sum_{k>=0} gamma^k r_{t+k}, computed backwards. No bootstrapping."""
    r = np.asarray(rewards, dtype=np.float64)
    out = np.zeros_like(r)
    running = 0.0
    for t in range(len(r) - 1, -1, -1):
        running = r[t] + gamma * running
        out[t] = running
    return out


def soft_returns(rewards, logps, alpha: float, gamma: float) -> np.ndarray:
    """Entropy-augmented return that a soft critic (SAC) should predict.

    H_t = r_t + gamma * (H_{t+1} - alpha * logp_{t+1}),  H_{T-1} = r_{T-1}.
    The entropy bonus starts at t+1 because Q(s_t, a_t) is conditioned on a_t.
    """
    r = np.asarray(rewards, dtype=np.float64)
    lp = np.asarray(logps, dtype=np.float64)
    if r.shape != lp.shape:
        raise ValueError(f"rewards {r.shape} and logps {lp.shape} must have the same shape")
    out = np.zeros_like(r)
    if r.size == 0:
        return out
    out[-1] = r[-1]
    for t in range(len(r) - 2, -1, -1):
        out[t] = r[t] + gamma * (out[t + 1] - alpha * lp[t + 1])
    return out


def usable_mask(length: int, truncated: bool, horizon: int) -> np.ndarray:
    """Which states have a trustworthy Monte Carlo return.

    terminated episode -> every state (the return really ends there).
    truncated episode  -> only states with at least `horizon` rewards observed from t on,
                          i.e. t <= length - horizon. With gamma = 0.99 and horizon = 500
                          the missing tail is weighted by at most 0.99**500 ~= 0.007.
    """
    mask = np.ones(length, dtype=bool)
    if truncated:
        mask[max(length - horizon + 1, 0):] = False
    return mask


# ---------------------------------------------------------------------------
# Probe rollouts
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    obs: np.ndarray               # (T, obs_dim)  state *before* each action
    acts: np.ndarray              # (T, act_dim)
    rewards: np.ndarray           # (T,)
    logps: Optional[np.ndarray]   # (T,) or None for deterministic policies
    truncated: bool               # True: cut by a time limit; False: terminated


def rollout(env, act_fn: Callable, seed: Optional[int] = None, max_steps: int = 100_000) -> Episode:
    """Run one episode with a Gymnasium-style env (reset -> (obs, info); step -> 5-tuple).

    Use a separate env instance from the training env, so probes never disturb training.
    If `terminated` and `truncated` are both True on the same step, the episode counts as
    terminated (Gymnasium's convention: the return really ended).
    """
    obs, _ = env.reset(seed=seed)
    obs_l, act_l, rew_l, logp_l = [], [], [], []
    truncated = True  # also covers hitting the max_steps safety cap
    for _ in range(max_steps):
        action, logp = act_fn(obs)
        next_obs, reward, terminated, trunc, _ = env.step(action)
        obs_l.append(np.asarray(obs, dtype=np.float32))
        act_l.append(np.asarray(action, dtype=np.float32))
        rew_l.append(float(reward))
        logp_l.append(logp)
        obs = next_obs
        if terminated:
            truncated = False
            break
        if trunc:
            break
    logps = None if any(lp is None for lp in logp_l) else np.asarray(logp_l, dtype=np.float64)
    return Episode(
        obs=np.stack(obs_l),
        acts=np.stack(act_l),
        rewards=np.asarray(rew_l, dtype=np.float64),
        logps=logps,
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), 0-based."""
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    avg = ends - (counts - 1) / 2.0 - 1.0
    return avg[inv.ravel()]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bias_metrics(q, g, floor: float = NORM_FLOOR) -> dict:
    """Compare critic predictions q with Monte Carlo returns g (same shape)."""
    q = np.asarray(q, dtype=np.float64).ravel()
    g = np.asarray(g, dtype=np.float64).ravel()
    if q.shape != g.shape:
        raise ValueError(f"q {q.shape} and g {g.shape} must have the same shape")
    nan = float("nan")
    if q.size == 0:
        keys = ["bias_mean", "bias_std", "bias_abs_mean", "norm_bias_mean", "norm_bias_median",
                "norm_bias_std", "pearson", "spearman", "q_mean", "q_max", "g_mean"]
        return {"n": 0, **{k: nan for k in keys}}
    bias = q - g
    norm = bias / np.maximum(np.abs(g), floor)
    return {
        "n": int(q.size),
        "bias_mean": float(bias.mean()),           # > 0 means overestimation
        "bias_std": float(bias.std()),
        "bias_abs_mean": float(np.abs(bias).mean()),
        "norm_bias_mean": float(norm.mean()),
        # robust companion: a few near-terminal states (small |G|, floor 10) cannot dominate it
        "norm_bias_median": float(np.median(norm)),
        "norm_bias_std": float(norm.std()),
        "pearson": _corr(q, g),
        "spearman": _corr(_rank(q), _rank(g)),
        "q_mean": float(q.mean()),
        # largest prediction: above r_max / (1 - gamma) no policy can earn it ("soft divergence",
        # van Hasselt et al., 2018), whatever the Monte Carlo returns say
        "q_max": float(q.max()),
        "g_mean": float(g.mean()),
    }


def td_error_stats(q_pred, target) -> dict:
    """For algorithm files to log on each critic update: TD error size and target magnitude.

    A steadily growing target_q_abs_max is an early warning of divergence.
    """
    q_pred = np.asarray(q_pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    return {
        "td/abs_mean": float(np.abs(q_pred - target).mean()),
        "td/target_q_mean": float(target.mean()),
        "td/target_q_abs_max": float(np.abs(target).max()),
    }


def probe(
    env,
    act_fn: Callable,
    q_fn: Callable,
    n_episodes: int = 5,
    gamma: float = 0.99,
    horizon: int = 500,
    alpha: Optional[float] = None,
    seed: int = 0,
    max_points: int = 2000,
) -> tuple[dict, dict]:
    """Roll out the current policy and compare the critic with Monte Carlo returns.

    alpha=None -> plain returns (TD3, DDPG). alpha=float -> soft returns (SAC); act_fn must
    then return log-probs, and should sample from the stochastic policy the critic evaluates.

    For twin critics (q_fn returns a tuple), "bias/*" uses the average of the critics
    (REDQ's convention) and "bias_min/*" uses their minimum (what TD3/SAC bootstrap on).

    Returns (metrics, raw). `raw` holds up to `max_points` (q, g) pairs for calibration plots.
    Note: probe/return_mean is the return of the policy used here, which for SAC is the
    stochastic policy, not the deterministic evaluation policy.
    """
    episodes = [rollout(env, act_fn, seed=seed + i) for i in range(n_episodes)]

    obs_parts, act_parts, g_parts = [], [], []
    for ep in episodes:
        if alpha is None:
            g = discounted_returns(ep.rewards, gamma)
        else:
            if ep.logps is None:
                raise ValueError("alpha was given (soft critic) but act_fn returned no log-probs")
            g = soft_returns(ep.rewards, ep.logps, alpha, gamma)
        m = usable_mask(len(ep.rewards), ep.truncated, horizon)
        obs_parts.append(ep.obs[m])
        act_parts.append(ep.acts[m])
        g_parts.append(g[m])

    obs = np.concatenate(obs_parts)
    acts = np.concatenate(act_parts)
    g = np.concatenate(g_parts)
    n = int(g.size)

    metrics: dict[str, Any] = {
        "probe/return_mean": float(np.mean([ep.rewards.sum() for ep in episodes])),
        "probe/length_mean": float(np.mean([len(ep.rewards) for ep in episodes])),
        "probe/usable_states": n,
    }
    if n == 0:  # e.g. every episode truncated before `horizon` steps
        return metrics, {"q": [], "g": []}

    out = q_fn(obs, acts)
    if isinstance(out, (tuple, list)):
        qs = np.stack([np.asarray(o, dtype=np.float64).ravel() for o in out])
        q = qs.mean(axis=0)
        metrics["twin/abs_diff_mean"] = float((qs.max(axis=0) - qs.min(axis=0)).mean())
        for k, v in bias_metrics(qs.min(axis=0), g).items():
            metrics[f"bias_min/{k}"] = v
    else:
        q = np.asarray(out, dtype=np.float64).ravel()
    for k, v in bias_metrics(q, g).items():
        metrics[f"bias/{k}"] = v

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n, max_points), replace=False)
    raw = {"q": q[idx].tolist(), "g": g[idx].tolist()}
    return metrics, raw


# ---------------------------------------------------------------------------
# Report file
# ---------------------------------------------------------------------------

def _json_safe(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None  # strict JSON has no NaN / Infinity
    if isinstance(v, (np.floating, np.integer)):
        return _json_safe(v.item())
    return v


class CriticReport:
    """Appends every probe to runs/<run>/critic_report.json, rewritten atomically each time,
    so a crashed run still leaves a valid file and plots never need a rerun."""

    def __init__(self, path, meta: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {"meta": {k: _json_safe(v) for k, v in meta.items()}, "probes": []}

    def add(self, step: int, metrics: dict, raw: dict) -> None:
        self.data["probes"].append({
            "step": int(step),
            "metrics": {k: _json_safe(v) for k, v in metrics.items()},
            "raw": raw,
        })
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic on Windows and Linux
