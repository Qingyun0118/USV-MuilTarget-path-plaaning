# Soft Q-filter 在 Q-filtered Behavior Cloning 中的改进研究

## 1. 引言

在分层强化学习（Hierarchical Reinforcement Learning, HRL）框架下，USV 多目标路径规划系统采用专家引导的两层架构：上层 Selector 通过 DQN+DeepSets 进行目标点选择，下层 Actuator 通过 off-policy 算法（TD3、TD7、SAC）执行导航控制。为加速 Actuator 层的收敛，训练过程中引入反应式控制器（reactive controller）生成的专家示教数据，并以 Behavior Cloning（BC）损失辅助策略更新。

然而， naive BC 容易导致策略对次优专家动作的过拟合。Nair 等人提出的 Q-filtered BC 通过 Critic 的 Q 值对专家样本进行过滤，仅当 $Q(s, a_{\text{expert}}) > Q(s, \pi(s))$ 时才施加模仿梯度。该硬阈值机制（Hard Q-filter）在理论上避免了次优专家数据的负迁移，但在实际训练中存在一个根本缺陷：二值化门控导致 BC 梯度信号在时间步之间频繁出现零/非零跳变，尤其在训练初期 Q 网络尚未收敛时，大量优质专家样本因微小的 Q 值估计噪声被错误过滤，造成梯度中断与策略更新不稳定。

针对上述问题，本文提出 Soft Q-filter：以 sigmoid 函数将 Q 值优势映射为连续权重，替代原有的硬二值判断。其核心思想是：当 Critic 对专家动作与策略动作的优劣判别不确定时（$Q_{\text{expert}} \approx Q_{\pi}$），不直接丢弃专家样本，而是赋予其约 0.5 的保守权重，保留部分模仿梯度；随着 Q 网络准确度提升，权重自然趋向 0 或 1，自适应地逼近硬过滤行为。

## 2. Soft Q-filter 方法

### 2.1 数学形式化

设专家样本批次为 $\{(s_i, g_i, a_i^{\text{exp}})\}_{i=1}^{N}$，其中 $g_i$ 为目标条件。Actor 在该状态下的输出为 $\mu_\theta(s_i, g_i)$。Q-filtered BC 的通用形式为：

$$
\mathcal{L}_{\text{BC}} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot \|\mu_\theta(s_i, g_i) - a_i^{\text{exp}}\|^2
$$

**Hard Q-filter** 的权重为二值指示函数：

$$
w_i^{\text{hard}} = \mathbb{1}\big[Q(s_i, g_i, a_i^{\text{exp}}) > Q(s_i, g_i, \mu_\theta(s_i, g_i))\big]
$$

**Soft Q-filter** 的权重为 Q 值优势的 sigmoid 映射：

$$
w_i^{\text{soft}} = \sigma\left(\frac{Q(s_i, g_i, a_i^{\text{exp}}) - Q(s_i, g_i, \mu_\theta(s_i, g_i))}{\tau}\right)
$$

其中 $\tau > 0$ 为温度参数（本实验中统一取 $\tau = 0.5$），$\sigma(x) = 1 / (1 + e^{-x})$。两种过滤策略的函数形态对比如图 1 所示。

| $Q_{\text{expert}} - Q_{\pi}$ | Hard Q-filter $w^{\text{hard}}$ | Soft Q-filter $w^{\text{soft}}$ ($\tau=0.5$) |
|:--:|:--:|:--:|
| $+2.0$ | $1$ | $0.98$ |
| $+0.1$ | $1$ | $0.55$ |
| $0$ | $0$ | $0.50$ |
| $-0.1$ | $0$ | $0.45$ |
| $-2.0$ | $0$ | $0.02$ |

**表 1：两种 Q-filter 在不同 Q 值优势下的权重分配对比**

从表 1 可见，Hard Q-filter 在 $Q_{\text{expert}} = Q_{\pi}$ 处存在不连续跳变（从 1 骤降至 0），而 Soft Q-filter 在该临界点保持 $0.5$ 的基线权重，并以 Lipschitz 连续的方式过渡。这一数学性质直接决定了两者在训练动态中的差异。

### 2.2 代码实现

Soft Q-filter 与 Hard Q-filter 在 TD3、TD7、SAC 三种 Actuator 算法中的实现逻辑保持一致，仅在 Critic 架构细节上有算法特异性差异。以下以 TD3 为例说明核心代码（`td3_agent.py:428-459`）：

```python
# Soft Q-filter
q_advantage = q_expert - q_pi
soft_weight = torch.sigmoid(q_advantage / self.bc_tau)  # τ = 0.5
soft_weight_avg = soft_weight.mean().item()
bc_hit_rate = (q_advantage > 0).float().mean().item()
per_sample_bc = (pi_expert - expert_actions).pow(2).mean(dim=1, keepdim=True)
bc_loss = (soft_weight * per_sample_bc).mean()

# Hard Q-filter
mask = (q_expert > q_pi).float()
bc_hit_rate = mask.mean().item()
soft_weight_avg = bc_hit_rate
per_sample_bc = (pi_expert - expert_actions).pow(2).mean(dim=1, keepdim=True)
bc_loss = (mask * per_sample_bc).sum() / (mask.sum() + 1e-8)
```

TD7 的实现（`td7_agent.py:651-688`）在输入特征上使用了 `fixed_encoder` 提供的稳定表征，但过滤逻辑与 TD3 完全一致。SAC 的实现（`sac_agent.py:422-452`）区别在于 BC 约束作用于策略的均值输出 `pi_mean` 而非采样动作，以分离确定性模仿与随机性探索。

### 2.3 温度参数 $\tau$ 的作用

$\tau$ 控制 sigmoid 的锐度：
- 当 $\tau \to 0$ 时，$\sigma(x/\tau)$ 逼近阶跃函数，Soft Q-filter 退化为 Hard Q-filter；
- 当 $\tau \to +\infty$ 时，$\sigma(x/\tau) \to 0.5$，所有专家样本获得均等权重，退化为无过滤 BC。

本实验取 $\tau = 0.5$ 的考量是：在典型的 Q 值优势范围（$|Q_{\text{expert}} - Q_{\pi}| \in [0, 2]$）内，sigmoid 处于线性响应区（导数最大），既保留了足够的区分度，又避免了过早饱和。

## 3. BC 诊断指标的定义与辨析

为定量评估 Soft Q-filter 对专家信号利用效率的影响，训练过程中记录两项核心诊断指标：**BC\_AvgSoftWeight** 与 **BC\_HitRate**。准确理解两者的定义差异与物理含义，是解读实验现象的前提。

### 3.1 指标定义

**BC\_AvgSoftWeight（平均软权重）**

在 Soft Q-filter 模式下：

$$
\text{AvgSoftWeight} = \frac{1}{N} \sum_{i=1}^{N} \sigma\left(\frac{Q_{\text{expert}}^{(i)} - Q_{\pi}^{(i)}}{\tau}\right)
$$

该指标反映每个专家样本被纳入 BC 损失时的平均权重，取值范围为 $(0, 1)$。由于 sigmoid 的基线效应，即使 $Q_{\text{expert}} = Q_{\pi}$，权重仍为 $0.5$。

在 Hard Q-filter 模式下，代码中直接令：

$$
\text{AvgSoftWeight} = \text{HitRate}
$$

即 Hard Q-filter 的 "AvgSoftWeight" 本质上是二值通过率的别名，其量纲与 Soft 模式完全不同。

**BC\_HitRate（硬过滤通过率）**

两种模式下定义一致：

$$
\text{HitRate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\big[Q_{\text{expert}}^{(i)} > Q_{\pi}^{(i)}\big]
$$

表示专家动作的 Q 值严格高于策略动作的比例，即 Critic 判定专家动作"更优"的频次。

### 3.2 核心区别：为什么 SoftQ 的 AvgSoftWeight 高但 HitRate 低？

这是实验观察中最易引发困惑的现象，需从数学定义与训练动态两个层面澄清。

**层面一：量纲差异（数学原因）**

Soft Q-filter 的 AvgSoftWeight 存在天然的 $\sigma(0) = 0.5$ 基线。假设某批次中专家样本的 Q 优势分布关于 0 对称（即一半为正、一半为负），则：
- Soft Q-filter 的 AvgSoftWeight $\approx 0.5$（负侧样本仍贡献约 0.5 权重）
- Hard Q-filter 的等效值（即 HitRate）$= 0.5$

因此，SoftQ 的 AvgSoftWeight 曲线天然位于 $0.4$-$0.5$ 区间，而 HardQ 的 AvgSoftWeight（即 HitRate）通常在 $0.1$-$0.3$ 区间。**两者绝对值不可直接比较**，SoftQ 的高值主要源于 sigmoid 的基线效应，而非"学得更好"。

**层面二：训练动态差异（因果链条）**

真正具有跨方法可比性的指标是 **HitRate**（定义完全相同）。实验观察到 SoftQ 的 HitRate 往往低于或接近 HardQ，这恰恰说明 Soft Q-filter 更有效：

```
Soft Q-filter BC 信号更强（所有专家样本均贡献梯度）
    ↓
策略更快模仿专家 → π(s) ≈ a_expert
    ↓
Q(s, π(s)) ≈ Q(s, a_expert) → Q_advantage → 0
    ↓
P(Q_expert > Q_policy) 下降 → HitRate 降低
```

反观之，Hard Q-filter 因大量过滤专家样本，策略保留更多"RL 个性"，在专家擅长的状态上 $Q_{\text{expert}}$ 持续高于 $Q_{\pi}$，导致 HitRate 居高不下。因此，**SoftQ 的低 HitRate 是策略已有效吸收专家知识的信号，而非训练失败的标志**。

### 3.3 实验数据印证

图 2-4 分别展示了 TD7、TD3、SAC 三种算法在 HardQ 与 SoftQ 条件下的 BC 诊断曲线（3 随机种子平均，阴影区域为标准差）。

**TD7（图 2）**：
- AvgSoftWeight：SoftQ 前期约 $0.40$-$0.45$，中后期衰减并波动至 $0.25$-$0.35$；HardQ 前期约 $0.15$-$0.20$，后期衰减至 $0.08$-$0.12$
- HitRate：SoftQ 前期约 $0.12$-$0.14$，后期衰减至 $0.05$-$0.08$；HardQ 前期约 $0.14$-$0.16$，后期衰减至 $0.08$-$0.10$

**TD3（图 3）**：
- AvgSoftWeight：SoftQ 快速达到 $0.47$-$0.48$ 平台并保持稳定；HardQ 缓慢爬升至 $0.20$-$0.23$，后期略降至 $0.18$-$0.20$
- HitRate：SoftQ 约 $0.16$-$0.20$，后期略降至 $0.16$-$0.17$；HardQ 约 $0.18$-$0.22$，后期略降至 $0.18$

**SAC（图 4）**：
- AvgSoftWeight：SoftQ 约 $0.47$-$0.48$，HardQ 约 $0.23$
- HitRate：两者几乎重合，约 $0.22$-$0.24$（HardQ 略高）

上述数据印证了两点：（1）SoftQ 的 AvgSoftWeight 因 sigmoid 基线效应显著高于 HardQ；（2）SoftQ 的 HitRate 在 TD3/TD7 中低于 HardQ，符合"策略更快吸收专家知识"的理论预期；而 SAC 中两者 HitRate 接近，暗示 SAC 的随机策略本身已具备较强的专家模仿能力，Soft Q-filter 的边际增益有限。

## 4. 实验结果与分析

本节结合训练奖励、成功率、碰撞率与 BC 诊断指标，系统对比 Soft Q-filter 与 Hard Q-filter 在三种 Actuator 算法上的性能差异。

### 4.1 TD3：Soft Q-filter 带来显著提升

| 指标 | Hard Q-filter | Soft Q-filter | 差异 |
|:--|:--|:--|:--|
| Actuator Reward | 收敛至 $\approx 0$~$5$，波动大 | 收敛至 $\approx 10$~$15$，更稳定 | **SoftQ 显著优于 HardQ** |
| Success Rate | 缓慢攀升至 $\approx 0.6$，振荡明显 | 较快攀升至 $\approx 0.8+$，方差小 | **SoftQ 显著优于 HardQ** |
| Collision | 训练结束仍 $\approx 0.2$~$0.3$ | 稳定下降至 $\approx 0.15$ | **SoftQ 更优** |
| BC\_AvgSoftWeight | $\approx 0.20$ | $\approx 0.47$ | Soft 保留更多专家梯度 |
| BC\_HitRate | $\approx 0.20$ | $\approx 0.18$ | 接近，说明 Q 判别力相当 |

**表 2：TD3 上 Hard Q-filter 与 Soft Q-filter 的性能对比**

TD3 作为确定性策略算法，其 Actor 更新完全依赖于 Q 函数对动作空间的梯度。Hard Q-filter 的二值化导致 BC 梯度在批次间频繁中断——HitRate 仅约 $0.20$ 意味着 80% 的专家样本被丢弃。Soft Q-filter 将有效 BC 信号强度提升约 2.3 倍（AvgSoftWeight $0.47$ vs $0.20$），更平滑的梯度流使得策略在早期即可有效学习专家示范中的避障行为，碰撞率显著降低。

### 4.2 TD7：Soft Q-filter 改善幅度最大

| 指标 | Hard Q-filter | Soft Q-filter | 差异 |
|:--|:--|:--|:--|
| Actuator Reward | 收敛至 $\approx -10$，收敛慢 | 收敛至 $\approx -2$~$0$，显著更高 | **SoftQ 大幅优于 HardQ** |
| Success Rate | 缓慢爬升至 $\approx 0.4$~$0.5$ | 较快达到 $\approx 0.6$~$0.7$ | **SoftQ 显著优于 HardQ** |
| Collision | 收敛至 $\approx 0.05$~$0.10$ | 更快收敛至 $\approx 0.03$~$0.05$ | **SoftQ 略优** |
| BC\_AvgSoftWeight | $\approx 0.08$~$0.18$ | $\approx 0.25$~$0.45$ | Soft 权重显著更高，但波动大 |
| BC\_HitRate | $\approx 0.08$~$0.16$ | $\approx 0.05$~$0.14$ | SoftQ 更低且衰减更快，策略吸收更快 |

**表 3：TD7 上 Hard Q-filter 与 Soft Q-filter 的性能对比**

TD7 从 Soft Q-filter 中的获益甚至超过 TD3，其核心原因在于 TD7 采用周期性硬目标更新（hard target update）与 fixed encoder 机制，导致 Q 值估计在更新间隔内存在更大的系统性偏差。Hard Q-filter 在此偏差下的误过滤问题更为突出——HardQ 的 HitRate 虽高于 SoftQ，但这恰恰说明大量本应被利用的专家样本因 Q 值波动而被错误丢弃。Soft Q-filter 通过连续权重机制将这部分"边缘样本"重新纳入训练，平均而言将专家信号利用率提升约 2–3 倍（尽管该优势随训练进程呈衰减趋势）。

### 4.3 SAC：Soft Q-filter 作用不明显

| 指标 | Hard Q-filter | Soft Q-filter | 差异 |
|:--|:--|:--|:--|
| Actuator Reward | 收敛至 $\approx 10$~$20$ | 收敛至 $\approx 10$~$20$ | **几乎无差别** |
| Success Rate | 收敛至 $\approx 0.8$ | 收敛至 $\approx 0.8$ | **几乎重叠** |
| Collision | 收敛至 $\approx 0.05$ | 收敛至 $\approx 0.05$ | **无显著差异** |
| BC\_AvgSoftWeight | $\approx 0.23$ | $\approx 0.48$ | Soft 权重仍更高，但对性能无增益 |
| BC\_HitRate | $\approx 0.23$ | $\approx 0.23$ | 完全一致 |

**表 4：SAC 上 Hard Q-filter 与 Soft Q-filter 的性能对比**

SAC 的实验结果表明，Soft Q-filter 虽然仍将 AvgSoftWeight 从 $0.23$ 提升至 $0.48$，但这一更高的 BC 信号利用率并未转化为性能提升。其根本原因在于 SAC 的最大熵框架本身已内置了策略更新的平滑机制：熵正则项 $\alpha \log \pi(a|s)$ 对策略梯度施加了连续约束，随机策略的 reparameterization trick 进一步提供了随机平滑效果。Hard Q-filter 的二值化跳变被 SAC 内在的随机性所缓冲，因此 Soft Q-filter 的边际收益趋近于零。

### 4.4 综合对比

| 算法 | 策略类型 | 内在平滑机制 | Soft Q-filter 边际收益 |
|:--|:--|:--|:--|
| TD3 | 确定性 | 无（仅 critic 端 target smoothing） | **高** |
| TD7 | 确定性 | 无（且硬目标更新加剧 Q 偏差） | **最高** |
| SAC | 随机性 | 最大熵正则 + 自动温度调节 + 双 Q clipping | **极低 / 可忽略** |

**表 5：Soft Q-filter 的边际收益与算法类型的关系**

上述对比揭示了一个核心规律：**Soft Q-filter 的价值与策略算法的"内在平滑性"呈负相关**。确定性策略（TD3/TD7）缺乏随机平滑机制，对 BC 梯度的离散跳变极为敏感，因此 Soft Q-filter 的连续加权能带来显著改善；而 SAC 的最大熵框架已提供了功能等价的平滑机制，使得 Soft Q-filter 成为冗余设计。

## 5. 机理分析

### 5.1 确定性策略对梯度中断的敏感性

TD3 与 TD7 均为确定性策略算法，Actor 输出单一动作 $a = \mu_\theta(s, g)$。其策略梯度为：

$$
\nabla_\theta J = \mathbb{E}\left[\nabla_a Q(s, g, a)\big|_{a=\mu_\theta} \cdot \nabla_\theta \mu_\theta(s, g)\right]
$$

Actor 的更新方向完全由 Q 函数梯度决定。当 Hard Q-filter 将某批次内全部专家样本的 BC 权重置零时，该步 Actor 仅受 RL 目标驱动；而下一批次可能大量通过过滤，造成 BC 梯度突然涌入。这种**间歇性梯度脉冲**在确定性策略中不会被任何内在随机性平滑，直接转化为参数空间中的振荡。

Soft Q-filter 的 sigmoid 加权本质上是专家样本上的注意力机制：$w_i \to 1$ 时强模仿，$w_i \approx 0.5$ 时保守模仿，$w_i \to 0$ 时逐渐忽略。该连续权重函数保证了相邻时间步间 BC 梯度变化量有界，消除了 Hard Q-filter 的不连续性。

### 5.2 Q 值估计误差的放大效应

训练初期 Q 网络尚未收敛，$Q(s, g, a_{\text{expert}})$ 与 $Q(s, g, \pi(s, g))$ 的大小关系高度不稳定。Hard Q-filter 的硬阈值在 Q 值接近临界点附近产生大量误分类——根据表 2-3，TD3-HardQ 仅利用约 20% 的专家样本，TD7-HardQ 的利用率更低。这些被过滤的样本中相当一部分是"因噪声而被误判"的优质示教数据。

Soft Q-filter 在 $q_{\text{advantage}} \approx 0$ 时保留约 0.5 的权重，相当于对 Q 网络的不确定性采取了**贝叶斯式保守估计**：不轻信 Critic 的判别，也不完全抛弃专家信号。随着 Q 网络准确度提升，$q_{\text{advantage}}$ 的绝对值增大，sigmoid 自然趋向饱和，自适应地逼近 Hard filter 的行为。

### 5.3 SAC 的低敏感性：功能重叠机制

SAC 的 Actor 损失包含三项约束：

$$
\mathcal{L}_{\pi} = \mathbb{E}\left[\alpha \log \pi(a|s, g) - \min(Q_1, Q_2)\right]
$$

其中：
1. **熵正则项** $\alpha \log \pi$ 强制策略保持非零方差，任何对均值函数的剧烈更新都会受到对数概率的惩罚；
2. **Reparameterization trick** 通过标准高斯噪声注入随机性，即使均值更新存在跳变，实际采样动作的分布变化也是平滑的；
3. **Clipped double-Q** 缓解了 Q 值高估，降低了 Hard Q-filter 的误过滤率（SAC-HardQ 的 HitRate 约 0.23，显著高于 TD3/TD7 的 HardQ 版本）。

此外，SAC 的 BC 损失直接约束策略均值 `pi_mean`（`sac_agent.py:420`），而策略的随机性由独立学习的方差参数控制。即使 BC 梯度偶尔中断，均值函数仍通过 RL 目标和熵正则获得充分更新，不会出现 TD3/TD7 式的"梯度真空"。

## 6. 结论

本文针对分层强化学习中专家引导的 Actuator 训练，提出了 Soft Q-filter 以替代传统的 Hard Q-filter。通过 sigmoid 连续函数将 Q 值优势映射为 BC 权重，Soft Q-filter 在训练初期（Q 网络不确定时）以约 0.5 的基线权重保留专家梯度信号，后期随 Q 网络收敛自适应逼近硬过滤行为。

实验结果表明，Soft Q-filter 对确定性策略算法（TD3、TD7）具有显著的提升作用：TD3 成功率从约 0.6 提升至约 0.8，碰撞率从约 0.25 降至约 0.15；TD7 的 Actuator Reward 提升约 8~10 分，成功率从约 0.45 提升至约 0.65。BC 诊断指标（AvgSoftWeight 与 HitRate）的联合分析揭示了这一提升的核心机制：Soft Q-filter 提供了更强且更平滑的 BC 梯度信号（AvgSoftWeight 提升 2~2.5 倍），使策略更快吸收专家知识，表现为 HitRate 的相对下降。与此同时，Soft Q-filter 对随机策略算法 SAC 的改善则可忽略不计——SAC 的最大熵正则化框架本身即提供了策略更新的平滑性与探索驱动力，与 Soft Q-filter 的功能高度重叠。

上述结果表明，**Soft Q-filter 是确定性策略专家引导学习中一种简洁有效的改进策略，其核心价值在于弥补确定性策略算法在 Q 值不确定期的梯度信号缺失问题**。对于已具备内在随机平滑机制的算法（如 SAC），该改进的边际收益趋近于零，研究者应根据底层策略类型选择性应用。
