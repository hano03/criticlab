"""DDPG (Lillicrap et al., 2016) in one file, with critic diagnostics.

Written as td3.py minus TD3's three fixes, so `diff td3.py ddpg.py` shows exactly what TD3 adds:
  - no target policy smoothing  (the target action is target_actor(s') with no noise)
  - no clipped double-Q         (one critic; the target bootstraps from Q'(s', a'))
  - no delayed policy updates   (actor and targets update after every critic update)
Everything else is identical to td3.py: networks, learning rate, batch size, tau, exploration
noise, replay buffer, terminated-vs-truncated handling and the diagnostics probe.

This is the "DDPG with TD3's architecture and hyperparameters" baseline that Fujimoto et al.
(2018) compare against (OurDDPG in their code). It differs from CleanRL's ddpg_continuous_action.py,
which keeps a delayed policy update (policy_frequency=2).

Run:
    python -m criticlab.ddpg --env-id Hopper-v5 --seed 0
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from criticlab.diagnostics import CriticReport, probe, td_error_stats


@dataclass
class Args:
    env_id: str = "Hopper-v5"
    seed: int = 0
    total_timesteps: int = 300_000
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    """target network update rate"""
    batch_size: int = 256
    exploration_noise: float = 0.1
    """std of Gaussian exploration noise, as a fraction of the action range"""
    learning_starts: int = 25_000
    """random actions and no updates before this many steps"""
    probe_every: int = 10_000
    probe_episodes: int = 5
    probe_horizon: int = 500
    """after a time-limit truncation, only states with this many observed rewards are probed"""
    log_every: int = 1_000
    torch_threads: int = 1
    runs_dir: str = "runs"
    exp_name: str = "ddpg"
    save_model: bool = False


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, low: np.ndarray, high: np.ndarray):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)
        # buffers, not parameters: moved with the model, never trained
        self.register_buffer("action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x)) * self.action_scale + self.action_bias


class ReplayBuffer:
    """Plain ring buffer. Stores `terminated` (true endings) only, never `truncated`."""

    def __init__(self, size: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros(size, dtype=np.float32)
        self.terms = np.zeros(size, dtype=np.float32)
        self.size, self.ptr, self.full = size, 0, False

    def add(self, obs, act, rew, next_obs, terminated):
        i = self.ptr
        self.obs[i], self.acts[i], self.rews[i] = obs, act, rew
        self.next_obs[i], self.terms[i] = next_obs, float(terminated)
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, n: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self.size if self.full else self.ptr, size=n)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "acts": torch.as_tensor(self.acts[idx]),
            "rews": torch.as_tensor(self.rews[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "terms": torch.as_tensor(self.terms[idx]),
        }


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


def main(args: Args) -> str:
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = os.path.join(args.runs_dir, run_name)
    writer = SummaryWriter(run_dir)
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                    "\n".join(f"|{k}|{v}|" for k, v in asdict(args).items()))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    rng = np.random.default_rng(args.seed)

    env = gym.make(args.env_id)
    probe_env = gym.make(args.env_id)  # separate instance: probes never disturb training
    assert isinstance(env.action_space, gym.spaces.Box), "DDPG needs a continuous action space"
    env.action_space.seed(args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    low, high = env.action_space.low, env.action_space.high

    actor = Actor(obs_dim, act_dim, low, high)
    qf1 = QNetwork(obs_dim, act_dim)
    target_actor = Actor(obs_dim, act_dim, low, high)
    qf1_target = QNetwork(obs_dim, act_dim)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = torch.optim.Adam(qf1.parameters(), lr=args.learning_rate)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(min(args.buffer_size, args.total_timesteps), obs_dim, act_dim)

    # --- diagnostics hooks: NumPy in, NumPy out --------------------------------------
    def act_fn(o):
        """Deterministic policy, no exploration noise: the policy the critic should evaluate."""
        with torch.no_grad():
            a = actor(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        return a.numpy(), None

    def q_fn(o, a, chunk: int = 4096):
        """One critic: returns a single array, so no twin/bias_min metrics are logged."""
        q1s = []
        with torch.no_grad():
            for i in range(0, len(o), chunk):
                ot = torch.as_tensor(o[i:i + chunk], dtype=torch.float32)
                at = torch.as_tensor(a[i:i + chunk], dtype=torch.float32)
                q1s.append(qf1(ot, at).squeeze(-1).numpy())
        return np.concatenate(q1s)

    report = CriticReport(os.path.join(run_dir, "critic_report.json"),
                          meta={"algo": "ddpg", "run_name": run_name, **asdict(args)})
    # ------------------------------------------------------------------------------------

    obs, _ = env.reset(seed=args.seed)
    ep_return, ep_len = 0.0, 0
    actor_loss = None
    start = time.time()
    last_td: dict = {}
    window_start = (0, start)  # (step, time) at the previous probe, for windowed SPS

    for step in range(1, args.total_timesteps + 1):  # step = env steps taken so far
        # 1. act
        if step <= args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                a += torch.normal(0.0, actor.action_scale * args.exploration_noise)
            action = a.numpy().clip(low, high)

        # 2. step the env and store the transition (real final obs, terminated only)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rb.add(obs, action, reward, next_obs, terminated)
        ep_return += float(reward)
        ep_len += 1
        if terminated or truncated:
            writer.add_scalar("charts/episodic_return", ep_return, step)
            writer.add_scalar("charts/episodic_length", ep_len, step)
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0
        else:
            obs = next_obs

        # 3. learn
        if step > args.learning_starts:
            data = rb.sample(args.batch_size, rng)
            with torch.no_grad():
                # no smoothing noise, one critic: the target trusts Q'(s', a') as it is
                next_actions = target_actor(data["next_obs"])
                q_next = qf1_target(data["next_obs"], next_actions)
                target_q = (data["rews"].unsqueeze(-1)
                            + (1.0 - data["terms"].unsqueeze(-1)) * args.gamma * q_next)

            q1 = qf1(data["obs"], data["acts"])
            q_loss = F.mse_loss(q1, target_q)
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            # no delay: actor and targets update after every critic update
            actor_loss = -qf1(data["obs"], actor(data["obs"])).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            soft_update(target_actor, actor, args.tau)
            soft_update(qf1_target, qf1, args.tau)

            if step % args.log_every == 0:
                writer.add_scalar("losses/q_loss", q_loss.item(), step)
                writer.add_scalar("losses/q1_values", q1.mean().item(), step)
                if actor_loss is not None:
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), step)
                last_td = td_error_stats(q1.detach().numpy(), target_q.numpy())
                for k, v in last_td.items():
                    writer.add_scalar(k, v, step)
                writer.add_scalar("charts/SPS", int(step / (time.time() - start)), step)

        # 4. probe the critics
        if step % args.probe_every == 0 or step == args.total_timesteps:
            metrics, raw = probe(probe_env, act_fn, q_fn, n_episodes=args.probe_episodes,
                                 gamma=args.gamma, horizon=args.probe_horizon,
                                 seed=args.seed * 1_000_000 + step)
            now = time.time()
            metrics["perf/sps_window"] = (step - window_start[0]) / max(now - window_start[1], 1e-9)
            window_start = (step, now)  # window = training steps since the last probe, probe time included
            metrics.update(last_td)
            report.add(step, metrics, raw)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    writer.add_scalar(k, v, step)
            print(f"step={step:>8}  return={metrics['probe/return_mean']:9.1f}  "
                  f"norm_bias={metrics.get('bias/norm_bias_mean', float('nan')):7.3f}  "
                  f"usable={metrics['probe/usable_states']:>5}  "
                  f"SPS={int(metrics['perf/sps_window'])} (avg {int(step / (now - start))})", flush=True)

    if args.save_model:
        torch.save({"actor": actor.state_dict(), "qf1": qf1.state_dict()},
                   os.path.join(run_dir, "model.pt"))
    env.close()
    probe_env.close()
    writer.close()
    return run_dir


if __name__ == "__main__":
    main(tyro.cli(Args))
