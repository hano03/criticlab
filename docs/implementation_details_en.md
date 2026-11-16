# CriticLab implementation details

## Shared by TD3, DDPG and SAC

### 1. Why reset a single environment by hand?

With same-step autoreset, `step()` can return the new episode's initial observation in place of the final observation. Storing it as `next_obs` links two unrelated episodes and can corrupt a time-limit bootstrap target. CriticLab stores the transition from `step()` before calling `reset()`. [`td3.py`](../criticlab/td3.py#L204)

### 2. Why store `terminated` in the replay buffer?

A terminal fall ends the task, so the target is just the last reward. A time limit cuts off data collection while the underlying task can continue, so the target still includes $\gamma Q_{\text{target}}(s',a')$ using the actual final observation. The mask `1 - terminated` preserves that term for time-limit truncations. [`td3.py`](../criticlab/td3.py#L204)

### 3. Why start with random actions?

Random actions fill replay with a broader set of transitions before an untrained actor can steer the data toward a narrow set of behaviors. TD3 and DDPG use 25,000 warm-up steps and SAC uses 5,000. SAC keeps sampling from a stochastic policy and encourages entropy after warm-up, which helps it continue exploring. [`td3.py`](../criticlab/td3.py#L48), [`ddpg.py`](../criticlab/ddpg.py#L49), [`sac.py`](../criticlab/sac.py#L49)

### 4. Why use a separate probe environment?

A probe resets the environment and runs full episodes. On the training instance, that would discard the live simulator state while the loop still holds its previous observation, breaking the next replay transition and the episode statistics. A separate instance keeps the training trajectory intact, and `seed * 1_000_000 + step` assigns probe reset seeds by run and checkpoint. [`td3.py`](../criticlab/td3.py#L146), [`td3.py`](../criticlab/td3.py#L260)

## TD3

### 5. Why smooth the target action?

Clipped Gaussian noise makes the backup use nearby target actions, so a narrow, spuriously high Q peak has less influence. `action_scale` converts noise from normalized coordinates to each dimension's action units using half its range. After adding noise, the code clamps each action component to its own lower and upper bounds. [`td3.py`](../criticlab/td3.py#L221)

### 6. Why use the smaller of two target Q values?

Both target critics score the same next action. Their minimum makes an optimistic error from just one critic less likely to enter the backup. The trade-offs are possible underestimation and the compute cost of a second critic. [`td3.py`](../criticlab/td3.py#L226)

### 7. Why delay the actor and target updates?

By default the critic gets two updates per actor update, giving it more time to fit the value estimates before the actor follows their gradients. The target actor and both target critics are soft-updated in the same delayed block. They stay fixed between those updates, so the critics fit against a slower-moving target. [`td3.py`](../criticlab/td3.py#L233)

### 8. Why does the actor use `qf1` alone?

TD3 updates its actor with the gradient of $Q_1(s,\pi(s))$ and uses the minimum in the critic backup to limit propagated overestimation. Using the minimum in the actor objective would route each sample's gradient through whichever critic is lower, which can switch as the action changes and steer the actor toward a more pessimistic objective. [`td3.py`](../criticlab/td3.py#L228), [`td3.py`](../criticlab/td3.py#L240)

## DDPG

### 9. What makes this DDPG a baseline for TD3?

It uses TD3's architecture and shared hyperparameters with one critic, an unsmoothed target action, and actor and target updates after every critic update. The CleanRL DDPG reference used here keeps the actor delay at two steps. This baseline removes all three TD3 changes together, while the `--no-clipped-double-q` TD3 run isolates the effect of the minimum in the backup. [`ddpg.py`](../criticlab/ddpg.py#L1), [`td3.py`](../criticlab/td3.py#L226)

## SAC

### 10. Why bound log standard deviation with `tanh`?

The actor maps its raw output smoothly into $[-5,2]$ before exponentiating it, keeping the Gaussian scale away from near-zero or excessively large values. A hard clamp has zero gradient outside the interval, while `tanh` tapers the gradient as it approaches the bounds. [`sac.py`](../criticlab/sac.py#L104)

### 11. What do `rsample` and the log-probability correction do?

`rsample()` writes the Gaussian draw as $x=\mu+\sigma\epsilon$, so gradients can flow through the sampled action to the actor parameters. For $a=\text{scale}\cdot\tanh(x)+\text{bias}$, change of variables subtracts the log absolute Jacobian from the Gaussian log density. The code subtracts $\log(\text{scale}\cdot(1-\tanh^2(x))+10^{-6})$ per dimension and sums the corrected log densities. [`sac.py`](../criticlab/sac.py#L112)

### 12. How does SAC compensate for delayed actor updates?

SAC performs two actor optimizer steps every two environment steps, using the same replay batch and drawing fresh policy actions for each update. This keeps the average at about one actor update per critic update, while TD3 performs just one actor update per two critic updates. SAC's target critics follow a separate schedule and soft-update every step by default once learning starts. [`sac.py`](../criticlab/sac.py#L284), [`td3.py`](../criticlab/td3.py#L240)

### 13. How does automatic $\alpha$ tuning work?

Optimizing unconstrained `log_alpha` gives a positive temperature via $\alpha=\exp(\text{log\_alpha})$. The target entropy $-\dim(A)$ is a heuristic that scales linearly with the number of action dimensions, and $\alpha$ rises when measured entropy is below that target. The log-probability is computed under `no_grad()` so the temperature loss updates only `log_alpha`. [`sac.py`](../criticlab/sac.py#L189), [`sac.py`](../criticlab/sac.py#L294)

### 14. Why use stochastic soft-return probes and deterministic evaluation?

The soft critic predicts returns under the stochastic policy, so the probe samples from that policy and uses its log-probabilities and the current $\alpha$ to compute soft Monte Carlo returns. Entropy starts at $t+1$ because $Q(s_t,a_t)$ already conditions on $a_t$. The separate `eval` score uses the squashed, rescaled Gaussian mean and raw episode rewards so task performance can be compared with TD3 and DDPG. [`sac.py`](../criticlab/sac.py#L200), [`sac.py`](../criticlab/sac.py#L321), [`diagnostics.py`](../criticlab/diagnostics.py#L55)

## Engineering

### 15. Why use one CPU thread per process?

With several runs in parallel, a large thread pool per run can oversubscribe the CPU, adding scheduling and synchronization overhead to small network updates. The launcher sets OpenMP and MKL to one thread per worker, and the algorithms default to one PyTorch intra-op thread. This can improve total throughput across runs even when a single job could benefit from more threads. [`launch.py`](../scripts/launch.py#L18), [`td3.py`](../criticlab/td3.py#L140)

### 16. Why write the report to a temporary file and replace it?

The writer finishes the JSON in a sibling `.json.tmp` file, then publishes it with `os.replace()`. Readers see a complete old or new report, and an interruption before replacement leaves the previous report intact. Undefined scalar metrics such as `NaN` become `null` so strict JSON readers can parse them as missing values. [`diagnostics.py`](../criticlab/diagnostics.py#L268)
