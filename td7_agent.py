"""
TD7 (TD3 + SALE + LAP + Checkpoint Policy) Actuator Agent — Goal-Conditioned

严格对齐 Fujimoto et al., "TD7: Revisiting Fundamentals of Off-Policy RL",
ICML 2023 的开源实现 (github.com/sfujim/TD7)。

TD3 基础上增加三项改进:

1. SALE (State-Action Learned Embeddings):
   - 独立 Encoder 模块, 自监督训练: 预测下一状态嵌入 MSE(g(f(s,g),a), f(s',g))
   - Encoder 输出 zs (状态嵌入, AvgL1Norm) 和 zsa (状态-动作嵌入)
   - fixed_encoder: 冻结快照, 供 Actor/Critic 使用 (稳定特征)
   - 周期性硬重置: fixed_encoder ← encoder (每 target_update_rate 步)

2. LAP (Loss-Adjusted Priorities):
   - LAP Huber 损失函数 (小 TD error 用 MSE, 大 TD error 用线性)
   - 优先级 = max(td_loss, min_priority)^alpha
   - 不使用 IS 权重修正

3. Checkpoint Policy:
   - 数据收集时 50/50 使用当前/历史最优策略
   - checkpoint 同时保存 actor + encoder
   - 防止策略坍塌, 保证 buffer 中高质量数据

与开源 TD7 的区别 (领域适配):
- SensorCNN: Conv1d 处理障碍物传感器数据 (保留空间分辨率)
- Goal-Conditioned: Encoder/Actor/Critic 均接受相对目标位置
- BC Loss: 支持 Expert 行为克隆 (Q-filtered)

与原 TD7 完全一致的核心机制:
- AvgL1Norm 归一化 (非 LayerNorm)
- 周期性硬目标更新 (非 EMA 软更新)
- LAP Huber 损失 (非 MSE)
- 值裁剪 (target Q clipping)
- Encoder → fixed_encoder → fixed_encoder_target 三级冻结
- Critic 使用 [zsa, zs] 拼接嵌入 (非内积 Q 分解)
- ELU 激活 (Encoder/Critic), ReLU 激活 (Actor)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hrl_config import device

# 状态维度布局: [o_usv(7) | o_detector(15) | o_current(4)] = 26
SENSOR_START = 7   # o_detector 在 state 中的起始索引
SENSOR_END = 22    # o_detector 在 state 中的结束索引 (不含)
SENSOR_DIM = SENSOR_END - SENSOR_START  # 15


class SensorCNN(nn.Module):
    """
    传感器 CNN: 保留空间分辨率, 提取障碍物空间特征

    使用 Flatten + FC, 保留左/右方向的空间信息,
    使 Agent 能区分障碍物方位。

    Input:  (batch, sensor_dim)  e.g. (batch, 15)
    Output: (batch, output_dim)  e.g. (batch, 64)
    """

    def __init__(self, sensor_dim=SENSOR_DIM, output_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * sensor_dim, output_dim)

    def forward(self, sensor):
        x = sensor.unsqueeze(1)              # (batch, 1, sensor_dim)
        x = F.relu(self.conv1(x))            # (batch, 16, sensor_dim)
        x = F.relu(self.conv2(x))            # (batch, 32, sensor_dim)
        x = x.view(x.size(0), -1)           # (batch, 32 * sensor_dim)
        return F.relu(self.fc(x))            # (batch, output_dim)


def _split_state(state):
    """将 state 拆分为 non_sensor 和 sensor 两部分"""
    non_sensor = torch.cat([state[:, :SENSOR_START],
                            state[:, SENSOR_END:]], dim=1)
    sensor = state[:, SENSOR_START:SENSOR_END]
    return non_sensor, sensor


# ============================================================================
# ========================= Utility Functions ================================
# ============================================================================

def AvgL1Norm(x, eps=1e-8):
    """
    Average L1 Normalization (TD7 论文核心归一化方式)

    x / mean(|x|): 保持向量方向, 归一化到平均绝对值为 1 的尺度。
    比 LayerNorm 更简洁, 不依赖可学习参数。
    """
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def LAP_huber(x, min_priority=1):
    """
    LAP Huber Loss (TD7 论文 Critic 损失函数)

    Huber-like loss, 以 min_priority 为阈值:
    - |x| < min_priority: 0.5 * x^2 (MSE-like, 小误差时平滑)
    - |x| >= min_priority: min_priority * |x| (线性, 大误差时鲁棒)

    先 sum 过 Q1/Q2 维度 (dim=1), 再 mean 过 batch。
    """
    return torch.where(x < min_priority, 0.5 * x.pow(2),
                       min_priority * x).sum(1).mean()


# ============================================================================
# ========================= Neural Networks ==================================
# ============================================================================

class Encoder(nn.Module):
    """
    SALE Encoder (Goal-Conditioned)

    自监督学习模块: 预测下一状态嵌入。
    - zs(state, goal) → 状态嵌入 z_s ∈ R^{zs_dim}, AvgL1Norm 归一化
    - zsa(z_s, action) → 状态-动作嵌入 z_{sa} ∈ R^{zs_dim}

    训练目标: MSE(zsa(zs_t, a_t), zs_{t+1})
    迫使 encoder 学习能预测未来状态的有意义表征。

    使用 SensorCNN 处理传感器数据 (state encoder 部分)。
    使用 ELU 激活函数 (与参考实现一致)。
    """

    def __init__(self, state_dim, goal_dim, action_dim, zs_dim=256,
                 hdim=256, cnn_output_dim=64):
        super().__init__()
        non_sensor_dim = state_dim - SENSOR_DIM

        # State encoder (with SensorCNN)
        self.sensor_cnn = SensorCNN(SENSOR_DIM, cnn_output_dim)
        fused_dim = non_sensor_dim + cnn_output_dim + goal_dim

        self.zs1 = nn.Linear(fused_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        # State-action encoder (takes zs + action, no SensorCNN)
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state, goal):
        """状态嵌入: f(s, g) → z_s, AvgL1Norm 归一化"""
        non_sensor, sensor = _split_state(state)
        cnn_out = self.sensor_cnn(sensor)
        x = torch.cat([non_sensor, cnn_out, goal], dim=1)
        x = F.elu(self.zs1(x))
        x = F.elu(self.zs2(x))
        x = AvgL1Norm(self.zs3(x))
        return x

    def zsa(self, zs, action):
        """状态-动作嵌入: g(z_s, a) → z_{sa}"""
        x = torch.cat([zs, action], dim=1)
        x = F.elu(self.zsa1(x))
        x = F.elu(self.zsa2(x))
        x = self.zsa3(x)
        return x


class Actor(nn.Module):
    """
    TD7 Actor (Goal-Conditioned)

    forward(state, goal, zs) → action ∈ [-1, 1]

    架构 (对齐参考实现):
    1. SensorCNN 处理传感器 → 拼接 non_sensor + goal → l0 → AvgL1Norm
    2. 与 Encoder 状态嵌入 zs 拼接
    3. 2 层隐藏层 (ReLU) → tanh 输出

    AvgL1Norm 替代 LayerNorm, ReLU 激活 (与参考 Actor 一致)。
    """

    def __init__(self, state_dim, goal_dim, action_dim, zs_dim=256,
                 hdim=256, cnn_output_dim=64):
        super().__init__()
        non_sensor_dim = state_dim - SENSOR_DIM
        self.sensor_cnn = SensorCNN(SENSOR_DIM, cnn_output_dim)
        fused_dim = non_sensor_dim + cnn_output_dim + goal_dim

        self.l0 = nn.Linear(fused_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, goal, zs):
        non_sensor, sensor = _split_state(state)
        cnn_out = self.sensor_cnn(sensor)
        fused = torch.cat([non_sensor, cnn_out, goal], dim=1)

        a = AvgL1Norm(self.l0(fused))
        a = torch.cat([a, zs], dim=1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))


class Critic(nn.Module):
    """
    TD7 Twin Critic (Goal-Conditioned)

    forward(state, goal, action, zsa, zs) → [Q1, Q2] (batch, 2)

    架构 (对齐参考实现):
    1. SensorCNN 处理传感器 → 拼接 non_sensor + goal + action → q0 → AvgL1Norm
    2. 与 Encoder 嵌入 [zsa, zs] 拼接 (2 * zs_dim)
    3. 2 层隐藏层 (ELU) → Q 值输出

    Q1 和 Q2 各有独立的 SensorCNN 和全连接层。
    使用 ELU 激活 (与参考 Critic 一致)。
    """

    def __init__(self, state_dim, goal_dim, action_dim, zs_dim=256,
                 hdim=256, cnn_output_dim=64):
        super().__init__()
        non_sensor_dim = state_dim - SENSOR_DIM

        # Q1 和 Q2 各有独立 SensorCNN
        self.sensor_cnn1 = SensorCNN(SENSOR_DIM, cnn_output_dim)
        self.sensor_cnn2 = SensorCNN(SENSOR_DIM, cnn_output_dim)

        sa_fused_dim = non_sensor_dim + cnn_output_dim + goal_dim + action_dim

        # Q1
        self.q01 = nn.Linear(sa_fused_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        # Q2
        self.q02 = nn.Linear(sa_fused_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, goal, action, zsa, zs):
        """返回 [Q1, Q2], shape (batch, 2)"""
        non_sensor, sensor = _split_state(state)
        embeddings = torch.cat([zsa, zs], dim=1)  # (batch, 2 * zs_dim)

        # Q1
        cnn_out1 = self.sensor_cnn1(sensor)
        sa1 = torch.cat([non_sensor, cnn_out1, goal, action], dim=1)
        q1 = AvgL1Norm(self.q01(sa1))
        q1 = torch.cat([q1, embeddings], dim=1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)

        # Q2
        cnn_out2 = self.sensor_cnn2(sensor)
        sa2 = torch.cat([non_sensor, cnn_out2, goal, action], dim=1)
        q2 = AvgL1Norm(self.q02(sa2))
        q2 = torch.cat([q2, embeddings], dim=1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)

        return torch.cat([q1, q2], dim=1)  # (batch, 2)


# ============================================================================
# ========================= TD7 Actuator Agent ===============================
# ============================================================================

class TD7ActuatorAgent:
    """
    TD7 Actuator Agent — Goal-Conditioned

    严格对齐 Fujimoto et al. (2023) 开源实现:

    组件:
    - Encoder: 自监督训练, 预测下一状态嵌入
      · encoder (不断训练)
      · fixed_encoder (冻结快照, Actor/Critic 使用, 周期性同步)
      · fixed_encoder_target (冻结快照, target Q 使用, 周期性同步)
      · checkpoint_encoder (最优策略快照)
    - Actor: 确定性策略, 使用 Encoder 嵌入
      · actor (当前策略)
      · actor_target (目标策略, 周期性硬拷贝)
      · actor_checkpoint (最优策略快照)
    - Critic: Twin Q, 使用 Encoder 嵌入, LAP Huber 损失
      · critic (当前)
      · critic_target (目标, 周期性硬拷贝)

    训练流程:
    1. Encoder 自监督更新: MSE(g(f(s,g),a), f(s',g))
    2. Critic 更新: LAP Huber 损失 + fixed_encoder 嵌入
    3. LAP 优先级更新
    4. Actor 延迟更新: -Q.mean() + BC loss
    5. 周期性硬目标更新 (所有 target 网络 + fixed_encoder 同步)
    """

    def __init__(self, state_dim, goal_dim, action_dim, max_action=1.0,
                 lr_actor=3e-4, lr_critic=3e-4, lr_encoder=3e-4,
                 gamma=0.99, target_update_rate=750,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 zs_dim=256, lap_alpha=0.4, lap_min_priority=1.0):

        self.max_action = max_action
        self.gamma = gamma
        self.target_update_rate = target_update_rate
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.zs_dim = zs_dim
        self.lap_alpha = lap_alpha
        self.lap_min_priority = lap_min_priority

        # ---- Encoder: current + fixed + fixed_target + checkpoint ----
        self.encoder = Encoder(
            state_dim, goal_dim, action_dim, zs_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=lr_encoder)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        # ---- Actor: current + target + checkpoint ----
        self.actor = Actor(
            state_dim, goal_dim, action_dim, zs_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr_actor)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_checkpoint = copy.deepcopy(self.actor)

        # ---- Critic: current + target ----
        self.critic = Critic(
            state_dim, goal_dim, action_dim, zs_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic)
        self.critic_target = copy.deepcopy(self.critic)

        # ---- Checkpoint state ----
        self.checkpoint_ready = False
        self.total_it = 0

        # ---- Value clipping (跟踪 Q 值范围, 防止 target Q 爆炸) ----
        self.max_Q = -1e8
        self.min_Q = 1e8
        self.max_Q_target = 0
        self.min_Q_target = 0

    def select_action(self, state, goal, noise=0.0, use_checkpoint=False):
        """
        选择动作, 支持 Checkpoint Policy

        - 默认: 使用 fixed_encoder + actor (当前策略)
        - checkpoint: 使用 checkpoint_encoder + actor_checkpoint (最优策略)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device)
        with torch.no_grad():
            if use_checkpoint and self.checkpoint_ready:
                zs = self.checkpoint_encoder.zs(state_t, goal_t)
                action = self.actor_checkpoint(state_t, goal_t, zs)
            else:
                zs = self.fixed_encoder.zs(state_t, goal_t)
                action = self.actor(state_t, goal_t, zs)

            if noise > 0:
                action = action + torch.randn_like(action) * noise

        return action.clamp(-1, 1).cpu().numpy().flatten() * self.max_action

    def update_checkpoint(self):
        """将当前 actor + fixed_encoder 复制到 checkpoint"""
        self.actor_checkpoint.load_state_dict(self.actor.state_dict())
        self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())
        self.checkpoint_ready = True

    def train(self, replay_buffer, batch_size=256, bc_weight=0.0):
        """
        TD7 训练步 (对齐参考实现)

        训练流程:
        1. Encoder 自监督更新: MSE(encoder.zsa(zs, a), encoder.zs(s', g))
        2. Critic 更新: LAP Huber 损失, 使用 fixed_encoder 嵌入
        3. LAP 优先级更新: td_loss.max(Q1,Q2).clamp(min).pow(alpha)
        4. Actor 延迟更新: -Q.mean() + BC loss (Q-filtered)
        5. 周期性硬目标更新 (每 target_update_rate 步)
        """
        if len(replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0

        sample_result = replay_buffer.sample(batch_size)

        # 兼容 PER buffer (9 元素) 和普通 buffer (7 元素)
        if len(sample_result) == 9:
            state, action, reward, next_state, done, goal, is_expert, \
                indices, _is_weights = sample_result
            # LAP: 不使用 IS 权重, _is_weights 被丢弃
        else:
            state, action, reward, next_state, done, goal, is_expert = sample_result
            indices = None

        self.total_it += 1
        not_done = 1.0 - done

        # ==================================================================
        # 1. Encoder 自监督更新
        # ==================================================================
        # 目标: encoder 学习预测下一状态嵌入 (自监督信号)
        # next_zs (stop gradient) 作为回归目标
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state, goal)

        zs = self.encoder.zs(state, goal)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # ==================================================================
        # 2. Critic 更新 (LAP Huber + fixed_encoder 嵌入)
        # ==================================================================
        with torch.no_grad():
            # 使用 fixed_encoder_target 计算 target embeddings
            fixed_target_zs = self.fixed_encoder_target.zs(next_state, goal)

            # Target policy smoothing (TD3)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(
                next_state, goal, fixed_target_zs) + noise).clamp(-1, 1)

            fixed_target_zsa = self.fixed_encoder_target.zsa(
                fixed_target_zs, next_action)

            # Target Q (min of Q1, Q2)
            Q_target = self.critic_target(
                next_state, goal, next_action,
                fixed_target_zsa, fixed_target_zs
            ).min(1, keepdim=True)[0]

            # Value clipping: 限制 target Q 范围, 防止爆炸
            Q_target = reward + not_done * self.gamma * Q_target.clamp(
                self.min_Q_target, self.max_Q_target)

            # 跟踪 Q 值范围
            self.max_Q = max(self.max_Q, float(Q_target.max()))
            self.min_Q = min(self.min_Q, float(Q_target.min()))

            # 使用 fixed_encoder 计算当前 batch 的嵌入
            fixed_zs = self.fixed_encoder.zs(state, goal)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        # Critic forward: Q(s, g, a, zsa, zs) → [Q1, Q2]
        Q = self.critic(state, goal, action, fixed_zsa, fixed_zs)
        td_loss = (Q - Q_target).abs()  # (batch, 2)

        # LAP Huber loss
        critic_loss = LAP_huber(td_loss, self.lap_min_priority)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ==================================================================
        # 3. LAP 优先级更新 (对齐参考实现, 跳过 buffer 内部 α 变换)
        # ==================================================================
        if indices is not None:
            # max over Q1/Q2, clamp to min_priority, 再做 pow(alpha)
            # 直接传入最终优先级, 使用 update_priorities_raw 跳过二次变换
            priority = td_loss.max(1)[0].clamp(
                min=self.lap_min_priority).pow(self.lap_alpha).detach()
            replay_buffer.update_priorities_raw(indices, priority)

        # ==================================================================
        # 4. Actor 延迟更新 (with BC loss)
        # ==================================================================
        actor_loss_val = 0.0
        bc_loss_val = 0.0
        if self.total_it % self.policy_freq == 0:
            # fixed_zs: 从 no_grad 块获取的 detached tensor (稳定特征)
            pi_action = self.actor(state, goal, fixed_zs)
            fixed_zsa_pi = self.fixed_encoder.zsa(fixed_zs, pi_action)
            Q_pi = self.critic(state, goal, pi_action,
                               fixed_zsa_pi, fixed_zs)

            # RL 目标: 最大化平均 Q 值 (Q1 + Q2 都参与, 与参考一致)
            rl_loss = -Q_pi.mean()

            # 行为克隆损失: 仅在 Expert transition 上, Q-filtered
            bc_loss = torch.tensor(0.0, device=device)
            if bc_weight > 0.0 and is_expert.sum() > 0:
                expert_mask = (is_expert > 0.5).squeeze()
                if expert_mask.any():
                    with torch.no_grad():
                        # Q of expert's original actions
                        Q_expert = self.critic(
                            state[expert_mask], goal[expert_mask],
                            action[expert_mask],
                            fixed_zsa[expert_mask],
                            fixed_zs[expert_mask]
                        ).mean(dim=1)
                        # Q of our policy actions on expert states
                        Q_policy = Q_pi[expert_mask].mean(dim=1).detach()
                        bc_filter = (Q_expert > Q_policy)

                    if bc_filter.any():
                        bc_loss = F.mse_loss(
                            pi_action[expert_mask][bc_filter],
                            action[expert_mask][bc_filter])

            total_actor_loss = rl_loss + bc_weight * bc_loss

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            actor_loss_val = rl_loss.item()
            bc_loss_val = bc_loss.item()

        # ==================================================================
        # 5. 周期性硬目标更新 (非 EMA, 与参考完全一致)
        # ==================================================================
        if self.total_it % self.target_update_rate == 0:
            # 硬拷贝所有 target 网络
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(
                self.fixed_encoder.state_dict())

            # 核心: 将训练中的 encoder 同步到 fixed_encoder
            # fixed_encoder 作为稳定特征提供者, 只在这里更新
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            # 更新值裁剪范围
            self.max_Q_target = self.max_Q
            self.min_Q_target = self.min_Q

        return critic_loss.item(), actor_loss_val, bc_loss_val
