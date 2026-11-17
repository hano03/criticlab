"""DQN (Mnih et al., 2015) and Double DQN (van Hasselt, Guez & Silver, 2016) in one file,
with critic diagnostics.

Starts from CleanRL's dqn.py: same defaults (CartPole-v1, 500k steps, 120-84 MLP, Adam 2.5e-4,
10k replay buffer, batch 128, one update every 10 steps, hard target copy every 500 steps,
epsilon 1 -> 0.05 over the first half of training). Deliberate changes:

  1. One environment, reset by hand, like td3.py: the transition that ends an episode stores the
     real final observation, and the buffer stores `terminated` only, so a time limit is not
     treated as the end of the world. (CleanRL's DQN does the same through
     handle_timeout_termination=False.)
  2. --double: Double DQN. The online network picks the next action, the target network values it:
         DQN:         y = r + gamma * max_a' Q_target(s', a')
         Double DQN:  y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
     This is the discrete analogue of TD3's clipped double-Q: both stop one noisy estimate from
     both choosing and valuing the maximising action.
  3. Every `probe_every` steps the probe compares Q(s, a) with Monte Carlo returns of the greedy
     policy (epsilon = 0), the policy the max in the target evaluates. A separate greedy
     evaluation on the standard environment gives `eval/deterministic_return_mean`, the return to
     compare with CleanRL's benchmark.
  4. --probe-max-episode-steps lets probe episodes run past the training time limit. CartPole-v1
     stops at 500 steps, so with probe_horizon = 500 a solved policy would leave almost no usable
     states. The critic bootstraps through the time limit, so it estimates the infinite-horizon
     value, and longer probe episodes simply give more ground truth for it.

Run:
    python -m criticlab.dqn --env-id CartPole-v1 --seed 0 --probe-max-episode-steps 1000
    python -m criticlab.dqn --env-id CartPole-v1 --seed 0 --probe-max-episode-steps 1000 --double
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

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
    env_id: str = "CartPole-v1"
    seed: int = 0
    total_timesteps: int = 500_000
    learning_rate: float = 2.5e-4
    buffer_size: int = 10_000
    gamma: float = 0.99
    tau: float = 1.0
    """target update rate; 1.0 = hard copy"""
    target_network_frequency: int = 500
    """steps between target network updates"""
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    """fraction of total_timesteps over which epsilon decays from start_e to end_e"""
    learning_starts: int = 10_000
    train_frequency: int = 10
    """one gradient step per this many environment steps"""
    double: bool = False
    """Double DQN target: online network picks the action, target network values it"""
    probe_every: int = 10_000
    probe_episodes: int = 5
    probe_horizon: int = 500
    """after a time-limit truncation, only states with this many observed rewards are probed"""
    probe_max_episode_steps: Optional[int] = None
    """time limit for probe episodes (default: the environment's own)"""
    eval_episodes: int = 5
    """greedy episodes on the standard environment, for the benchmark return"""
    log_every: int = 1_000
    torch_threads: int = 1
    runs_dir: str = "runs"
    exp_name: str = "dqn"
    save_model: bool = False


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, n_actions),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Plain ring buffer. Stores `terminated` (true endings) only, never `truncated`."""

    def __init__(self, size: int, obs_dim: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros(size, dtype=np.int64)
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


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def td_target(q_network: nn.Module, target_network: nn.Module, data: dict, gamma: float,
              double: bool) -> torch.Tensor:
    """The bootstrapped target, (B,). Only true terminations stop the bootstrap."""
    with torch.no_grad():
        q_next_target = target_network(data["next_obs"])
        if double:
            next_actions = q_network(data["next_obs"]).argmax(dim=1, keepdim=True)
            q_next = q_next_target.gather(1, next_actions).squeeze(1)
        else:
            q_next = q_next_target.max(dim=1).values
        return data["rews"] + gamma * q_next * (1.0 - data["terms"])


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
    probe_env = (gym.make(args.env_id, max_episode_steps=args.probe_max_episode_steps)
                 if args.probe_max_episode_steps else gym.make(args.env_id))
    eval_env = gym.make(args.env_id)  # standard time limit: the benchmark return
    assert isinstance(env.action_space, gym.spaces.Discrete), "DQN needs a discrete action space"
    env.action_space.seed(args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    q_network = QNetwork(obs_dim, n_actions)
    target_network = QNetwork(obs_dim, n_actions)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(min(args.buffer_size, args.total_timesteps), obs_dim)

    # --- diagnostics hooks: NumPy in, NumPy out --------------------------------------
    def greedy(o) -> int:
        with torch.no_grad():
            return int(q_network(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).argmax(dim=1).item())

    def act_fn(o):
        """Greedy policy, epsilon = 0: the policy the max in the target evaluates."""
        return greedy(o), None

    def q_fn(o, a, chunk: int = 4096):
        out = []
        with torch.no_grad():
            for i in range(0, len(o), chunk):
                ot = torch.as_tensor(o[i:i + chunk], dtype=torch.float32)
                at = torch.as_tensor(np.asarray(a[i:i + chunk]).reshape(-1), dtype=torch.int64)
                out.append(q_network(ot).gather(1, at.unsqueeze(1)).squeeze(1).numpy())
        return np.concatenate(out)

    def evaluate_greedy(n: int, seed: int) -> float:
        returns = []
        for i in range(n):
            o, _ = eval_env.reset(seed=seed + i)
            done, total = False, 0.0
            while not done:
                o, r, term, trunc, _ = eval_env.step(greedy(o))
                total += float(r)
                done = term or trunc
            returns.append(total)
        return float(np.mean(returns))

    report = CriticReport(os.path.join(run_dir, "critic_report.json"),
                          meta={"algo": "dqn", "run_name": run_name, **asdict(args)})
    # ------------------------------------------------------------------------------------

    obs, _ = env.reset(seed=args.seed)
    ep_return, ep_len = 0.0, 0
    start = time.time()
    last_td: dict = {}
    window_start = (0, start)
    epsilon = args.start_e

    for step in range(1, args.total_timesteps + 1):  # step = env steps taken so far
        # 1. act, epsilon-greedy
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  step - 1)
        action = env.action_space.sample() if random.random() < epsilon else greedy(obs)

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
            if step % args.train_frequency == 0:
                data = rb.sample(args.batch_size, rng)
                target = td_target(q_network, target_network, data, args.gamma, args.double)
                q_pred = q_network(data["obs"]).gather(1, data["acts"].unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q_pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % args.log_every == 0:
                    writer.add_scalar("losses/td_loss", loss.item(), step)
                    writer.add_scalar("losses/q_values", q_pred.mean().item(), step)
                    last_td = td_error_stats(q_pred.detach().numpy(), target.numpy())
                    for k, v in last_td.items():
                        writer.add_scalar(k, v, step)
                    writer.add_scalar("charts/epsilon", epsilon, step)
                    writer.add_scalar("charts/SPS", int(step / (time.time() - start)), step)
            if step % args.target_network_frequency == 0:
                for tp, p in zip(target_network.parameters(), q_network.parameters()):
                    tp.data.copy_(args.tau * p.data + (1.0 - args.tau) * tp.data)

        # 4. probe the critic
        if step % args.probe_every == 0 or step == args.total_timesteps:
            metrics, raw = probe(probe_env, act_fn, q_fn, n_episodes=args.probe_episodes,
                                 gamma=args.gamma, horizon=args.probe_horizon,
                                 seed=args.seed * 1_000_000 + step)
            metrics["eval/deterministic_return_mean"] = evaluate_greedy(
                args.eval_episodes, seed=args.seed * 1_000_000 + step + 500_000)
            now = time.time()
            metrics["perf/sps_window"] = (step - window_start[0]) / max(now - window_start[1], 1e-9)
            window_start = (step, now)
            metrics["dqn/epsilon"] = epsilon
            metrics.update(last_td)
            report.add(step, metrics, raw)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    writer.add_scalar(k, v, step)
            print(f"step={step:>8}  eval={metrics['eval/deterministic_return_mean']:8.1f}  "
                  f"norm_bias median={metrics.get('bias/norm_bias_median', float('nan')):7.3f}  "
                  f"Q={metrics.get('bias/q_mean', float('nan')):7.1f}  G={metrics.get('bias/g_mean', float('nan')):7.1f}  "
                  f"usable={metrics['probe/usable_states']:>5}  eps={epsilon:.2f}  "
                  f"SPS={int(metrics['perf/sps_window'])} (avg {int(step / (now - start))})", flush=True)

    if args.save_model:
        torch.save({"q_network": q_network.state_dict()}, os.path.join(run_dir, "model.pt"))
    for e in (env, probe_env, eval_env):
        e.close()
    writer.close()
    return run_dir


if __name__ == "__main__":
    main(tyro.cli(Args))
