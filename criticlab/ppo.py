"""PPO (Schulman et al., 2017) for continuous actions in one file, with critic diagnostics.

Starts from CleanRL's ppo_continuous_action.py: same defaults (2048-step rollouts, 10 epochs,
32 minibatches, GAE lambda 0.95, clipped value loss, observation and reward normalisation,
orthogonal init, linear learning-rate decay). Deliberate changes:

  1. One environment, reset by hand, with the normalisers written out here instead of Gymnasium
     wrappers. The probe can then use the *same* statistics, frozen, so the critic sees probe
     states exactly as it sees training states.
  2. Time-limit truncation bootstraps: when an episode is cut by the time limit, gamma * V(final
     state) is added to the last reward before the episode is closed. (CleanRL treats truncation
     as termination, which teaches the critic that the world ends at the time limit.)
  3. Every `probe_every` steps, the probe compares V(s) with Monte Carlo returns of the *stochastic*
     policy, which is the policy V evaluates. V is learned on rewards divided by the running scale
     sigma, so the probe converts it back to real reward units: V_raw(s) = sigma * V(s). Bias is
     then comparable with DDPG, TD3 and SAC. (The +-10 reward clip is ignored in this conversion;
     it rarely binds.)

PPO's critic is a state-value function V(s), not Q(s, a): the probe ignores the action.

Run:
    python -m criticlab.ppo --env-id Hopper-v5 --seed 0
    python -m criticlab.ppo --env-id Hopper-v5 --seed 0 --gae-lambda 1.0   # week-3 lambda study
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
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from criticlab.diagnostics import CriticReport, probe


@dataclass
class Args:
    env_id: str = "Hopper-v5"
    seed: int = 0
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    """decay the learning rate linearly to 0 over training"""
    num_steps: int = 2048
    """environment steps per rollout (one policy update per rollout)"""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_obs: bool = True
    norm_reward: bool = True
    probe_every: int = 10_000
    probe_episodes: int = 5
    probe_horizon: int = 500
    """after a time-limit truncation, only states with this many observed rewards are probed"""
    eval_episodes: int = 5
    """deterministic-policy evaluation episodes at every probe"""
    torch_threads: int = 1
    runs_dir: str = "runs"
    exp_name: str = "ppo"
    save_model: bool = False


class RunningMeanStd:
    """Running mean and variance (parallel algorithm), as in Gymnasium's normalisation wrappers."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        b_mean, b_var, b_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta, total = b_mean - self.mean, self.count + b_count
        self.mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total
        self.var, self.count = m2 / total, total


def compute_gae(rewards, values, dones, next_value, gamma: float, lam: float):
    """Generalised advantage estimation over one rollout.

    dones[t] = 1 if the episode ended at step t (the transition from s_t was its last), so the
    value of s_{t+1} is masked. Truncated episodes were already bootstrapped in the reward.
    next_value = V of the observation after the rollout's last step.
    Returns (advantages, returns), with returns = advantages + values (the critic's targets).
    """
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in reversed(range(n)):
        v_next = next_value if t == n - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * v_next * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + np.asarray(values, dtype=np.float64)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x)


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

    env = gym.make(args.env_id)
    probe_env = gym.make(args.env_id)  # separate instance: probes never disturb training
    assert isinstance(env.action_space, gym.spaces.Box), "this PPO needs a continuous action space"
    env.action_space.seed(args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    low, high = env.action_space.low, env.action_space.high

    agent = Agent(obs_dim, act_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- normalisers, shared with the probe ------------------------------------------------
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    ret_rms = RunningMeanStd()
    disc_return = 0.0  # running discounted return used to estimate the reward scale

    def norm_obs(o):
        o = np.asarray(o, dtype=np.float64)
        if not args.norm_obs:
            return o.astype(np.float32)
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10).astype(np.float32)

    def reward_scale() -> float:
        return float(np.sqrt(ret_rms.var + 1e-8)) if args.norm_reward else 1.0

    # --- diagnostics hooks: NumPy in, NumPy out -------------------------------------------
    def act_fn(o):
        """Stochastic policy (the one V evaluates), normalised with frozen statistics."""
        with torch.no_grad():
            a, _, _, _ = agent.get_action_and_value(torch.as_tensor(norm_obs(o)).unsqueeze(0))
        return np.clip(a.squeeze(0).numpy(), low, high), None

    def q_fn(o, a, chunk: int = 4096):
        """V(s) converted back to real reward units; the action is ignored."""
        vs = []
        with torch.no_grad():
            for i in range(0, len(o), chunk):
                x = torch.as_tensor(norm_obs(o[i:i + chunk]))  # normalisation broadcasts over the batch
                vs.append(agent.get_value(x).squeeze(-1).numpy())
        return np.concatenate(vs) * reward_scale()

    def evaluate_deterministic(n: int, seed: int) -> float:
        returns = []
        for i in range(n):
            o, _ = probe_env.reset(seed=seed + i)
            total = 0.0
            while True:
                with torch.no_grad():
                    mean = agent.actor_mean(torch.as_tensor(norm_obs(o)).unsqueeze(0))
                o, r, terminated, truncated, _ = probe_env.step(np.clip(mean.squeeze(0).numpy(), low, high))
                total += float(r)
                if terminated or truncated:
                    break
            returns.append(total)
        return float(np.mean(returns))

    report = CriticReport(os.path.join(run_dir, "critic_report.json"),
                          meta={"algo": "ppo", "run_name": run_name, **asdict(args)})
    # ----------------------------------------------------------------------------------------

    obs_buf = np.zeros((args.num_steps, obs_dim), dtype=np.float32)
    act_buf = np.zeros((args.num_steps, act_dim), dtype=np.float32)
    logp_buf = np.zeros(args.num_steps, dtype=np.float32)
    rew_buf = np.zeros(args.num_steps, dtype=np.float64)
    done_buf = np.zeros(args.num_steps, dtype=np.float64)
    val_buf = np.zeros(args.num_steps, dtype=np.float64)

    raw_obs, _ = env.reset(seed=args.seed)
    obs_rms.update(raw_obs)
    obs = norm_obs(raw_obs)
    ep_return, ep_len = 0.0, 0
    num_updates = args.total_timesteps // args.num_steps
    last_steps = num_updates * args.num_steps
    last_update: dict = {}
    global_step = 0
    start = time.time()
    window_start = (0, start)

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1.0) / num_updates) * args.learning_rate

        # 1. collect one rollout
        for t in range(args.num_steps):
            global_step += 1
            obs_buf[t] = obs
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(torch.as_tensor(obs).unsqueeze(0))
            action = action.squeeze(0).numpy()
            act_buf[t], logp_buf[t], val_buf[t] = action, float(logp), float(value)

            raw_next, reward, terminated, truncated, _ = env.step(np.clip(action, low, high))
            ep_return += float(reward)
            ep_len += 1

            # reward scaling, as Gymnasium's NormalizeReward: divide by the std of the discounted return
            disc_return = disc_return * args.gamma + float(reward)
            ret_rms.update(np.array([disc_return]))
            r = float(np.clip(reward / reward_scale(), -10, 10)) if args.norm_reward else float(reward)

            if truncated and not terminated:
                # bootstrap through the time limit: the episode is cut, not over
                with torch.no_grad():
                    r += args.gamma * float(agent.get_value(torch.as_tensor(norm_obs(raw_next)).unsqueeze(0)))
            rew_buf[t] = r
            done_buf[t] = float(terminated or truncated)

            if terminated or truncated:
                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/episodic_length", ep_len, global_step)
                raw_next, _ = env.reset()
                ep_return, ep_len, disc_return = 0.0, 0, 0.0
            obs_rms.update(raw_next)
            obs = norm_obs(raw_next)

            # probe between environment steps; parameters only change at the update below
            if global_step % args.probe_every == 0 or global_step == last_steps:
                probe_seed = args.seed * 1_000_000 + global_step
                metrics, raw = probe(probe_env, act_fn, q_fn, n_episodes=args.probe_episodes,
                                     gamma=args.gamma, horizon=args.probe_horizon, seed=probe_seed)
                metrics["eval/deterministic_return_mean"] = evaluate_deterministic(
                    args.eval_episodes, probe_seed + 500_000)
                metrics["ppo/reward_scale"] = reward_scale()
                now = time.time()
                metrics["perf/sps_window"] = (global_step - window_start[0]) / max(now - window_start[1], 1e-9)
                window_start = (global_step, now)
                metrics.update(last_update)
                report.add(global_step, metrics, raw)
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        writer.add_scalar(k, v, global_step)
                print(f"step={global_step:>8}  eval={metrics['eval/deterministic_return_mean']:9.1f}  "
                      f"probe={metrics['probe/return_mean']:9.1f}  "
                      f"norm_bias={metrics.get('bias/norm_bias_mean', float('nan')):7.3f}  "
                      f"EV={last_update.get('ppo/explained_variance', float('nan')):5.2f}  "
                      f"usable={metrics['probe/usable_states']:>5}  "
                      f"SPS={int(metrics['perf/sps_window'])} (avg {int(global_step / (now - start))})", flush=True)

        # 2. advantages and value targets (GAE)
        with torch.no_grad():
            next_value = float(agent.get_value(torch.as_tensor(obs).unsqueeze(0)))
        adv, ret = compute_gae(rew_buf, val_buf, done_buf, next_value, args.gamma, args.gae_lambda)

        # 3. PPO update
        b_obs, b_act = torch.as_tensor(obs_buf), torch.as_tensor(act_buf)
        b_logp, b_adv = torch.as_tensor(logp_buf), torch.as_tensor(adv, dtype=torch.float32)
        b_ret, b_val = torch.as_tensor(ret, dtype=torch.float32), torch.as_tensor(val_buf, dtype=torch.float32)
        mb_size = args.num_steps // args.num_minibatches
        clipfracs = []
        for _ in range(args.update_epochs):
            perm = np.random.permutation(args.num_steps)
            for s in range(0, args.num_steps, mb_size):
                mb = perm[s:s + mb_size]
                _, new_logp, entropy, new_val = agent.get_action_and_value(b_obs[mb], b_act[mb])
                log_ratio = new_logp - b_logp[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_adv[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()

                new_val = new_val.view(-1)
                if args.clip_vloss:
                    v_clipped = b_val[mb] + torch.clamp(new_val - b_val[mb], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max((new_val - b_ret[mb]) ** 2, (v_clipped - b_ret[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_val - b_ret[mb]) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # 4. how well did the critic fit this rollout's targets? (before the update's values)
        var_ret = np.var(ret)
        explained_var = float("nan") if var_ret == 0 else float(1 - np.var(ret - val_buf) / var_ret)
        last_update = {
            "ppo/explained_variance": explained_var,
            "ppo/value_loss": float(v_loss.item()),
            "ppo/policy_loss": float(pg_loss.item()),
            "ppo/approx_kl": float(approx_kl.item()),
            "ppo/clipfrac": float(np.mean(clipfracs)),
            "ppo/entropy": float(entropy.mean().item()),
        }
        for k, v in last_update.items():
            if np.isfinite(v):
                writer.add_scalar(k, v, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)

    if args.save_model:
        torch.save({"agent": agent.state_dict(), "obs_mean": obs_rms.mean, "obs_var": obs_rms.var,
                    "ret_var": ret_rms.var}, os.path.join(run_dir, "model.pt"))
    env.close()
    probe_env.close()
    writer.close()
    return run_dir


if __name__ == "__main__":
    main(tyro.cli(Args))
