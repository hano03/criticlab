# CriticLab 实现细节

## TD3、DDPG 和 SAC 共用的设计

### 1. 为什么用单环境并手动重置？

向量环境如果在同一步自动重置，`step()` 可能返回新回合的初始观测。若把它存成 `next_obs`，就会把两个回合接在一起，导致时间截断时用错状态来估计后续价值。CriticLab 先保存 `step()` 返回的完整转移，再调用 `reset()`。 [`td3.py`](../criticlab/td3.py#L204)

### 2. 为什么回放缓冲区只存 `terminated`？

摔倒导致任务真正终止时，训练目标只有当步奖励。时间上限只中断采样，任务本身仍可继续，所以要用截断时的真实下一状态加上 $\gamma Q_{\text{target}}(s',a')$。用 `1 - terminated` 作掩码就能保留这项后续价值。 [`td3.py`](../criticlab/td3.py#L204)

### 3. 为什么先用随机动作预热？

随机动作先给回放缓冲区积累较多样的转移，避免未训练的 actor 一开始就把数据集中到少数行为上。TD3 和 DDPG 默认预热 25,000 步，SAC 默认 5,000 步。SAC 在预热后仍从随机策略采样，并在优化目标里鼓励熵，因此可以继续探索。 [`td3.py`](../criticlab/td3.py#L48)、[`ddpg.py`](../criticlab/ddpg.py#L49)、[`sac.py`](../criticlab/sac.py#L49)

### 4. 为什么探针要用独立环境？

探针会重置环境并跑完整回合。如果共用训练环境，模拟器状态会被改写，而训练循环还拿着旧观测，导致下一条回放转移和回合统计都出错。独立环境保留了训练轨迹，`seed * 1_000_000 + step` 则按训练种子和步数为探针分配重置种子。 [`td3.py`](../criticlab/td3.py#L146)、[`td3.py`](../criticlab/td3.py#L260)

## TD3

### 5. 为什么要平滑目标动作？

给目标动作加限幅高斯噪声，让训练目标利用动作邻域的估值，减少 critic 上狭窄的虚高尖峰对目标的影响。`action_scale` 是各动作维度范围的一半，用它把归一化噪声转换成该维度的动作单位。加噪后再按各维度自己的上下界裁剪，保证动作有效。 [`td3.py`](../criticlab/td3.py#L221)

### 6. 为什么两个目标 Q 要取较小值？

两个目标 critic 给同一个下一步动作估值，取较小值能降低单个 critic 的高估被传入训练目标的概率。代价是可能引入低估，同时需要额外训练和计算第二个 critic。 [`td3.py`](../criticlab/td3.py#L226)

### 7. 为什么延迟更新 actor 和目标网络？

默认先让 critic 更新两次，再让 actor 沿 critic 的梯度更新一次，给 critic 更多时间拟合价值。目标 actor 和两个目标 critic 在同一个分支里软更新。两次软更新之间目标网络固定，critic 拟合的目标变化更慢。 [`td3.py`](../criticlab/td3.py#L233)

### 8. 为什么 actor 只用 `qf1`？

TD3 用 $Q_1(s,\pi(s))$ 的梯度更新 actor，并在 critic 的训练目标里取最小值来抑制高估传播。如果 actor 也取最小值，每个样本的梯度会来自估值较低的 critic，动作变化时还可能在两个 critic 之间切换，使策略优化更偏保守。 [`td3.py`](../criticlab/td3.py#L228)、[`td3.py`](../criticlab/td3.py#L240)

## DDPG

### 9. 这里的 DDPG 为什么适合作为 TD3 的基线？

它沿用 TD3 的网络结构和共有超参数，只用一个 critic，目标动作不加平滑噪声，并在每次 critic 更新后更新 actor 和目标网络。项目参照的 CleanRL DDPG 仍保留每两步一次的 actor 更新。这里的 DDPG 一次移除 TD3 的三项改动，而 TD3 的 `--no-clipped-double-q` 单独检验训练目标取最小值的作用。 [`ddpg.py`](../criticlab/ddpg.py#L1)、[`td3.py`](../criticlab/td3.py#L226)

## SAC

### 10. 为什么用 `tanh` 限制对数标准差？

网络先把原始输出平滑映射到 $[-5,2]$，再取指数，让高斯分布的标准差保持在合适范围内。硬裁剪在越界区间的梯度为零，`tanh` 的梯度在接近边界时逐渐减小。 [`sac.py`](../criticlab/sac.py#L104)

### 11. `rsample` 和对数概率修正分别做了什么？

`rsample()` 把高斯样本写成 $x=\mu+\sigma\epsilon$，让梯度能通过采样动作传回 actor。动作经过 $a=\text{scale}\cdot\tanh(x)+\text{bias}$ 后，变量变换要求从原高斯对数密度里减去雅可比行列式绝对值的对数。代码逐维减去 $\log(\text{scale}\cdot(1-\tanh^2(x))+10^{-6})$，再把各维结果相加。 [`sac.py`](../criticlab/sac.py#L112)

### 12. SAC 怎样补偿延迟的 actor 更新？

SAC 每两步集中做两次 actor 更新，两次用同一批回放数据，但各自重新采样策略动作。这样平均每次 critic 更新对应约一次 actor 更新，TD3 则是每两次 critic 更新对应一次。SAC 的目标 critic 单独调度，开始学习后默认每步软更新。 [`sac.py`](../criticlab/sac.py#L284)、[`td3.py`](../criticlab/td3.py#L240)

### 13. 自动调节 $\alpha$ 的原理是什么？

直接优化无约束的 `log_alpha`，再取指数得到正的温度系数 $\alpha$。目标熵 $-\dim(A)$ 是随动作维数线性缩放的经验设定，策略的估计熵低于目标时，$\alpha$ 会增大。`no_grad()` 让这次更新只调整 `log_alpha`。 [`sac.py`](../criticlab/sac.py#L189)、[`sac.py`](../criticlab/sac.py#L294)

### 14. 为什么探针用随机策略和软回报，评估用确定性策略？

软 critic 估计的是随机策略下的回报，所以探针从该策略采样，用动作的对数概率和当前 $\alpha$ 计算软 Monte Carlo 回报。熵项从 $t+1$ 开始，因为 $Q(s_t,a_t)$ 已经以当前动作 $a_t$ 为条件。单独报告的 `eval` 用经过 `tanh` 和缩放的高斯均值作为动作，只累加环境原始奖励，便于和 TD3、DDPG 比较任务表现。 [`sac.py`](../criticlab/sac.py#L200)、[`sac.py`](../criticlab/sac.py#L321)、[`diagnostics.py`](../criticlab/diagnostics.py#L55)

## 工程实现

### 15. 为什么每个进程用一个 CPU 线程？

多组实验并行时，各进程如果都开一组线程，容易争用 CPU，小网络更新还要承担额外的调度和同步开销。启动器将每个子进程的 OpenMP 和 MKL 设为单线程，算法默认也将 PyTorch 算子内部的并行线程数设为 1。这样通常能提高并行实验的总吞吐量。 [`launch.py`](../scripts/launch.py#L18)、[`td3.py`](../criticlab/td3.py#L140)

### 16. 为什么先写临时文件，再替换报告？

先把完整 JSON 写到同目录的 `.json.tmp`，再用 `os.replace()` 原子替换。读取方只会看到完整的旧版或新版报告，写入在替换前中断时旧报告仍可用。`NaN` 等非有限的标量指标转成 `null`，让严格的 JSON 解析器能把它们读作缺失值。 [`diagnostics.py`](../criticlab/diagnostics.py#L268)
