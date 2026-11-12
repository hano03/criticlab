"""SAC (Haarnoja et al., 2018) in one file, with critic diagnostics.

Starts from CleanRL's sac_continuous_action.py: same defaults, same networks (tanh-squashed
Gaussian actor, twin soft Q-functions), automatic entropy tuning, same update schedule.
The same three deliberate changes as td3.py:
  1. One environment, reset by hand (no vector env, no auto-reset).
  2. The replay buffer stores `terminated` only; time-limit truncation still bootstraps.
  3. Every `probe_every` steps, criticlab.diagnostics.probe compares the critics with Monte Carlo
     returns. SAC's critics predict *soft* values (reward plus entropy bonus), so the probe uses
     the stochastic policy, its log-probs and the current alpha to compute soft returns.
     Comparing them with plain returns would make SAC look overestimated when it is not.

Because the probe runs the stochastic policy, its return (probe/return_mean) is the stochastic
policy's return. The usual evaluation return of the deterministic policy (the tanh of the mean)
is logged separately as eval/deterministic_return_mean.

Run:
    python -m criticlab.sac --env-id Hopper-v5 --seed 0
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
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    """target network update rate"""
    batch_size: int = 256
    learning_starts: int = 5_000
    """random actions and no updates before this many steps"""
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    """learning rate of the critics and of alpha"""
    policy_frequency: int = 2
    """actor updates happen every this many steps, policy_frequency times each (same total count)"""
    target_network_frequency: int = 1
    alpha: float = 0.2
    """entropy coefficient; only the starting point when autotune is on"""
    autotune: bool = True
    """learn alpha so the policy's entropy tracks -action_dim; --no-autotune keeps alpha fixed"""
    probe_every: int = 10_000
    probe_episodes: int = 5
    probe_horizon: int = 500
    """after a time-limit truncation, only states with this many observed rewards are probed"""
    eval_episodes: int = 5
    """deterministic-policy evaluation episodes at every probe"""
    log_every: int = 1_000
    torch_threads: int = 1
    runs_dir: str = "runs"
    exp_name: str = "sac"
    save_model: bool = False


LOG_STD_MAX = 2
LOG_STD_MIN = -5


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
    """Gaussian policy squashed by tanh into the action range."""

    def __init__(self, obs_dim: int, act_dim: int, low: np.ndarray, high: np.ndarray):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        self.register_buffer("action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))  # squash, then rescale into [LOG_STD_MIN, LOG_STD_MAX]
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        """Returns (sampled action, its log-prob, deterministic action)."""
        mean, log_std = self(x)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x_t = normal.rsample()  # reparameterised: gradients flow through the sample
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # change of variables for tanh and the action scaling
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


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
    assert isinstance(env.action_space, gym.spaces.Box), "SAC needs a continuous action space"
    env.action_space.seed(args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    low, high = env.action_space.low, env.action_space.high

    actor = Actor(obs_dim, act_dim, low, high)
    qf1, qf2 = QNetwork(obs_dim, act_dim), QNetwork(obs_dim, act_dim)
    qf1_target, qf2_target = QNetwork(obs_dim, act_dim), QNetwork(obs_dim, act_dim)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)

    # automatic entropy tuning: learn log(alpha) so that entropy tracks -action_dim
    if args.autotune:
        target_entropy = -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True)
        alpha = log_alpha.exp().item()
        a_optimizer = torch.optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(min(args.buffer_size, args.total_timesteps), obs_dim, act_dim)

    # --- diagnostics hooks: NumPy in, NumPy out --------------------------------------
    def act_fn(o):
        """Stochastic policy with its log-prob: soft Q-values are defined for this policy."""
        with torch.no_grad():
            a, logp, _ = actor.get_action(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
        return a.squeeze(0).numpy(), float(logp.item())

    def q_fn(o, a, chunk: int = 4096):
        q1s, q2s = [], []
        with torch.no_grad():
            for i in range(0, len(o), chunk):
                ot = torch.as_tensor(o[i:i + chunk], dtype=torch.float32)
                at = torch.as_tensor(a[i:i + chunk], dtype=torch.float32)
                q1s.append(qf1(ot, at).squeeze(-1).numpy())
                q2s.append(qf2(ot, at).squeeze(-1).numpy())
        return np.concatenate(q1s), np.concatenate(q2s)

    def evaluate_deterministic(n: int, seed: int) -> float:
        """The usual evaluation return: act with tanh(mean), no sampling."""
        returns = []
        for i in range(n):
            o, _ = probe_env.reset(seed=seed + i)
            total = 0.0
            while True:
                with torch.no_grad():
                    _, _, a = actor.get_action(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
                o, r, terminated, truncated, _ = probe_env.step(a.squeeze(0).numpy())
                total += float(r)
                if terminated or truncated:
                    break
            returns.append(total)
        return float(np.mean(returns))

    report = CriticReport(os.path.join(run_dir, "critic_report.json"),
                          meta={"algo": "sac", "run_name": run_name, **asdict(args)})
    # ------------------------------------------------------------------------------------

    obs, _ = env.reset(seed=args.seed)
    ep_return, ep_len = 0.0, 0
    actor_loss = alpha_loss = None
    start = time.time()
    last_td: dict = {}
    window_start = (0, start)  # (step, time) at the previous probe, for windowed SPS

    for step in range(1, args.total_timesteps + 1):  # step = env steps taken so far
        # 1. act
        if step <= args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _, _ = actor.get_action(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = a.squeeze(0).numpy()

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
                # soft target: min of the twin targets, minus the entropy term of the next action
                next_actions, next_logp, _ = actor.get_action(data["next_obs"])
                q_next = torch.min(qf1_target(data["next_obs"], next_actions),
                                   qf2_target(data["next_obs"], next_actions)) - alpha * next_logp
                target_q = (data["rews"].unsqueeze(-1)
                            + (1.0 - data["terms"].unsqueeze(-1)) * args.gamma * q_next)

            q1 = qf1(data["obs"], data["acts"])
            q2 = qf2(data["obs"], data["acts"])
            q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            # delayed actor updates, repeated so the total number of actor updates is unchanged
            if step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, logp, _ = actor.get_action(data["obs"])
                    min_q_pi = torch.min(qf1(data["obs"], pi), qf2(data["obs"], pi))
                    actor_loss = (alpha * logp - min_q_pi).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, logp, _ = actor.get_action(data["obs"])
                        alpha_loss = (-log_alpha.exp() * (logp + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if step % args.target_network_frequency == 0:
                soft_update(qf1_target, qf1, args.tau)
                soft_update(qf2_target, qf2, args.tau)

            if step % args.log_every == 0:
                writer.add_scalar("losses/q_loss", q_loss.item() / 2.0, step)
                writer.add_scalar("losses/q1_values", q1.mean().item(), step)
                writer.add_scalar("losses/alpha", alpha, step)
                if actor_loss is not None:
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), step)
                if alpha_loss is not None:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), step)
                last_td = td_error_stats(q1.detach().numpy(), target_q.numpy())
                for k, v in last_td.items():
                    writer.add_scalar(k, v, step)
                writer.add_scalar("charts/SPS", int(step / (time.time() - start)), step)

        # 4. probe the critics (soft returns) and evaluate the deterministic policy
        if step % args.probe_every == 0 or step == args.total_timesteps:
            probe_seed = args.seed * 1_000_000 + step
            metrics, raw = probe(probe_env, act_fn, q_fn, n_episodes=args.probe_episodes,
                                 gamma=args.gamma, horizon=args.probe_horizon,
                                 alpha=alpha, seed=probe_seed)
            metrics["probe/alpha"] = alpha
            metrics["eval/deterministic_return_mean"] = evaluate_deterministic(
                args.eval_episodes, probe_seed + 500_000)
            now = time.time()
            metrics["perf/sps_window"] = (step - window_start[0]) / max(now - window_start[1], 1e-9)
            window_start = (step, now)  # window = training steps since the last probe, probe time included
            metrics.update(last_td)
            report.add(step, metrics, raw)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    writer.add_scalar(k, v, step)
            print(f"step={step:>8}  eval={metrics['eval/deterministic_return_mean']:9.1f}  "
                  f"probe={metrics['probe/return_mean']:9.1f}  "
                  f"norm_bias={metrics.get('bias/norm_bias_mean', float('nan')):7.3f}  "
                  f"alpha={alpha:.4f}  usable={metrics['probe/usable_states']:>5}  "
                  f"SPS={int(metrics['perf/sps_window'])} (avg {int(step / (now - start))})", flush=True)

    if args.save_model:
        torch.save({"actor": actor.state_dict(), "qf1": qf1.state_dict(), "qf2": qf2.state_dict()},
                   os.path.join(run_dir, "model.pt"))
    env.close()
    probe_env.close()
    writer.close()
    return run_dir


if __name__ == "__main__":
    main(tyro.cli(Args))
