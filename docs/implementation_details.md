# Implementation details

> **For you to write.** Each entry names a choice, where it lives, and a question. Answer each in
> 2–4 sentences of your own, without looking at the code first, then check against the code.
> Delete these quote blocks when done. This file is your interview cheat sheet: if you can't
> answer an entry, that's the part of the code to re-read.

## Shared by TD3, DDPG and SAC

### 1. One environment, reset by hand
*Where:* `main()`, step 2 of the loop, in each algorithm file.
*Question:* What goes wrong with the replay buffer when a vector env auto-resets, and how does a
single hand-reset env avoid it?

### 2. The buffer stores `terminated`, never `truncated`
*Where:* `ReplayBuffer.add`, and the `(1 - terms)` factor in the target.
*Question:* Why must a time-limit cutoff still bootstrap from Q(s', a'), while a fall must not?

### 3. Random actions before `learning_starts`
*Where:* step 1 of the loop. TD3 and DDPG use 25k, SAC 5k (CleanRL defaults).
*Question:* What does a purely random start buy, and why might SAC need less of it?

### 4. A separate environment for probes
*Where:* `probe_env`, and the probe seeds `seed * 1_000_000 + step`.
*Question:* What would break if probes used the training environment?

## TD3

### 5. Target policy smoothing
*Where:* the `noise = ...` lines in the learn block.
*Question:* Why add clipped noise to the *target* action, why multiply it by `action_scale`, and
why clamp per dimension?

### 6. Clipped double-Q
*Where:* `torch.min(q1_next, q2_next)`; switched off by `--no-clipped-double-q`.
*Question:* Why does taking the minimum of two critics reduce overestimation, and what does it
cost?

### 7. Delayed policy and target updates
*Where:* `if step % args.policy_frequency == 0`.
*Question:* Why update the actor less often than the critics, and why do the targets move only
inside this block?

### 8. The actor uses `qf1` only
*Where:* `actor_loss = -qf1(...)`.
*Question:* Why not min(Q1, Q2) for the actor too?

## DDPG

### 9. "TD3 minus its three fixes"
*Where:* `git diff --no-index criticlab/td3.py criticlab/ddpg.py`.
*Question:* How does this DDPG differ from CleanRL's, and why does that matter for the ablation
story?

## SAC

### 10. Squashed log-std
*Where:* `Actor.forward`, `LOG_STD_MIN`, `LOG_STD_MAX`.
*Question:* Why bound log σ to [−5, 2] with a tanh instead of clipping?

### 11. Reparameterisation and the tanh correction
*Where:* `Actor.get_action`: `rsample`, and `log(action_scale * (1 - y²) + 1e-6)`.
*Question:* What does `rsample` give that `sample` doesn't, and where does the correction term
come from?

### 12. Compensated delayed actor updates
*Where:* `for _ in range(args.policy_frequency)`.
*Question:* The actor updates every 2 steps but twice each time. Why is that not the same as
TD3's delay?

### 13. Automatic α
*Where:* `log_alpha`, `target_entropy = -act_dim`, `alpha_loss`.
*Question:* Why learn log α rather than α, why −dim(A) as the target, and why is the log-prob
computed under `no_grad` here?

### 14. Probing SAC
*Where:* `act_fn` (stochastic, returns log-prob), `probe(..., alpha=alpha)`,
`evaluate_deterministic`.
*Question:* Why must the probe use the stochastic policy and soft returns, while the reported
return uses the deterministic policy?

## Engineering

### 15. One thread per process
*Where:* `torch.set_num_threads(args.torch_threads)`, `OMP_NUM_THREADS=1` in `scripts/launch.py`.
*Question:* Why do parallel runs get faster overall with one thread each?

### 16. Report files
*Where:* `CriticReport.save` in `diagnostics.py`.
*Question:* Why write to a temporary file and `os.replace` it, and why turn NaN into `null`?
