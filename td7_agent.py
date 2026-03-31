"""
TD7 (TD3 + SALE + LAP + Checkpoint Policy) Actuator Agent — Goal-Conditioned

TD3 基础上增加三项改进 (Fujimoto et al., 2023):
1. SALE (State-Action Learned Embeddings): Q(s,a) = f(s)^T · g(s,a) / sqrt(d)
   - 共享状态编码器 f (在 Q1/Q2 之间共享)
   - 独立状态-动作编码器 g1/g2
   - 内积 Q 分解限制 Q 值过估计 (高维空间中两个独立的嵌入向量难以同时过高)
   - EMA 目标编码器 + 周期性硬重置

2. LAP (Loss-Adjusted Priorities): 基于 critic loss 的优先级采样
   - 优先级 = max(per_sample_critic_loss, 1.0)
   - 不使用 IS 权重修正 (优先级已学习化)

3. Checkpoint Policy: 数据收集时 50/50 使用当前/历史最优策略
   - 防止策略坍塌 (TD3 已知问题: SR 峰值后崩溃到 0)
   - 保证 buffer 中始终有来自最优策略的高质量数据

网络结构与 SAC/TD3 版对齐:
- SensorCNN: Flatten+FC 保留空间分辨率
- 3 层隐藏层 + LayerNorm + 输入归一化 (Actor)
- 2 层隐藏层 + LayerNorm (SALE Encoder)
- 早期目标融合
"""

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
# ========================= Neural Networks ==================================
# ============================================================================

class Actor(nn.Module):
    """
    TD7 Actor (Goal-Conditioned) — 与 TD3 版完全相同

    确定性策略: (State, Goal) → Action
    SensorCNN + 3 层隐藏层 + LayerNorm + 输入归一化
    """

    def __init__(self, state_dim, goal_dim, action_dim, max_action=1.0,
                 cnn_output_dim=64, hidden_dim=256):
        super().__init__()
        self.max_action = max_action
        non_sensor_dim = state_dim - SENSOR_DIM

        # 传感器 CNN
        self.sensor_cnn = SensorCNN(SENSOR_DIM, cnn_output_dim)

        # 输入维度: non_sensor(11) + cnn_out(64) + goal(2) = 77
        fused_dim = non_sensor_dim + cnn_output_dim + goal_dim

        # 输入归一化
        self.input_ln = nn.LayerNorm(fused_dim)

        # 3 层隐藏层 + LayerNorm
        self.fc1 = nn.Linear(fused_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # 输出头
        self.fc_out = nn.Linear(hidden_dim, action_dim)

        # 权重初始化
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, goal):
        non_sensor, sensor = _split_state(state)
        cnn_out = self.sensor_cnn(sensor)

        # 早期目标融合: goal 从第一层就参与
        x = torch.cat([non_sensor, cnn_out, goal], dim=1)
        x = self.input_ln(x)

        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))

        return self.max_action * torch.tanh(self.fc_out(x))


# ---- SALE Components (State-Action Learned Embeddings) ----

class StateEncoder(nn.Module):
    """
    SALE 状态编码器 f(s, g) → z_s ∈ R^{embed_dim}

    在 Q1 和 Q2 之间共享, 提供统一的状态表征。
    使用 SensorCNN 处理传感器数据, 2 层隐藏层 + LayerNorm。
    输出为线性嵌入 (无激活函数), 用于内积 Q 计算。
    """

    def __init__(self, state_dim, goal_dim, embed_dim=256,
                 cnn_output_dim=64, hidden_dim=256):
        super().__init__()
        non_sensor_dim = state_dim - SENSOR_DIM
        self.sensor_cnn = SensorCNN(SENSOR_DIM, cnn_output_dim)
        fused_dim = non_sensor_dim + cnn_output_dim + goal_dim

        self.input_ln = nn.LayerNorm(fused_dim)
        self.fc1 = nn.Linear(fused_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, embed_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, goal):
        non_sensor, sensor = _split_state(state)
        cnn_out = self.sensor_cnn(sensor)
        x = torch.cat([non_sensor, cnn_out, goal], dim=1)
        x = self.input_ln(x)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc_out(x)  # 线性输出, 不加激活


class StateActionEncoder(nn.Module):
    """
    SALE 状态-动作编码器 g(s, g, a) → z_{sa} ∈ R^{embed_dim}

    Q1 和 Q2 各自拥有独立的 StateActionEncoder (含独立 SensorCNN),
    确保 Twin Q 真正独立。
    输出为线性嵌入, 用于内积 Q 计算。
    """

    def __init__(self, state_dim, goal_dim, action_dim, embed_dim=256,
                 cnn_output_dim=64, hidden_dim=256):
        super().__init__()
        non_sensor_dim = state_dim - SENSOR_DIM
        self.sensor_cnn = SensorCNN(SENSOR_DIM, cnn_output_dim)
        fused_dim = non_sensor_dim + cnn_output_dim + goal_dim + action_dim

        self.input_ln = nn.LayerNorm(fused_dim)
        self.fc1 = nn.Linear(fused_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, embed_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, goal, action):
        non_sensor, sensor = _split_state(state)
        cnn_out = self.sensor_cnn(sensor)
        x = torch.cat([non_sensor, cnn_out, goal, action], dim=1)
        x = self.input_ln(x)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc_out(x)  # 线性输出, 不加激活


class SALECritic(nn.Module):
    """
    SALE Twin Critic (Goal-Conditioned)

    Q_i(s, g, a) = f(s, g)^T · g_i(s, g, a) / sqrt(embed_dim)

    组件:
    - f (StateEncoder): 共享状态编码器 (一个 SensorCNN, 在 Q1/Q2 之间共享)
    - g1, g2 (StateActionEncoder): 独立状态-动作编码器 (各自拥有独立 SensorCNN)

    内积结构的优势:
    - Q 值 = 两个嵌入向量的内积 / sqrt(d), 天然限制过大
    - 高维空间中两个独立学习的嵌入向量的内积增长受限 (类似 Transformer 注意力缩放)
    - 共享状态编码器 f 提供一致的状态表征, 减少 Q1/Q2 之间的碎片化

    (State, Goal, Action) → Q1, Q2
    """

    def __init__(self, state_dim, goal_dim, action_dim,
                 embed_dim=256, cnn_output_dim=64, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** 0.5

        # 共享状态编码器 f (一个 SensorCNN)
        self.state_encoder = StateEncoder(
            state_dim, goal_dim, embed_dim, cnn_output_dim, hidden_dim)

        # 独立状态-动作编码器 g1, g2 (各自拥有独立 SensorCNN)
        self.sa_encoder1 = StateActionEncoder(
            state_dim, goal_dim, action_dim, embed_dim, cnn_output_dim, hidden_dim)
        self.sa_encoder2 = StateActionEncoder(
            state_dim, goal_dim, action_dim, embed_dim, cnn_output_dim, hidden_dim)

    def forward(self, state, goal, action):
        """返回 Q1, Q2 (内积 Q 值)"""
        zs = self.state_encoder(state, goal)           # (batch, embed_dim)
        zsa1 = self.sa_encoder1(state, goal, action)   # (batch, embed_dim)
        zsa2 = self.sa_encoder2(state, goal, action)   # (batch, embed_dim)

        q1 = (zs * zsa1).sum(dim=1, keepdim=True) / self.scale
        q2 = (zs * zsa2).sum(dim=1, keepdim=True) / self.scale
        return q1, q2

    def Q1(self, state, goal, action):
        """仅计算 Q1 (用于 Actor 策略梯度优化)"""
        zs = self.state_encoder(state, goal)
        zsa1 = self.sa_encoder1(state, goal, action)
        return (zs * zsa1).sum(dim=1, keepdim=True) / self.scale


# ============================================================================
# ========================= TD7 Actuator Agent ===============================
# ============================================================================

class TD7ActuatorAgent:
    """
    TD7 Actuator Agent — Goal-Conditioned

    TD3 基础上增加:
    1. SALE Critic (内积 Q 分解, 抑制 Q 值过估计)
    2. LAP (loss-based 优先级采样, 不使用 IS 权重修正)
    3. Checkpoint Policy (50/50 actor 混合, 防止策略坍塌)

    直接控制 USV 的油门和舵角, 目标为到达 Selector 指定的目标点。
    (State, RelativeGoal) → (Throttle, Rudder)
    """

    def __init__(self, state_dim, goal_dim, action_dim, max_action=1.0,
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 embed_dim=256, encoder_reset_interval=100000,
                 lap_min_priority=1.0):
        # ---- Actor: current + target + checkpoint ----
        self.actor = Actor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_checkpoint = Actor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_checkpoint.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # ---- SALE Critic: current + target ----
        self.critic = SALECritic(
            state_dim, goal_dim, action_dim, embed_dim).to(device)
        self.critic_target = SALECritic(
            state_dim, goal_dim, action_dim, embed_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic)

        # ---- Hyperparameters ----
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # SALE
        self.embed_dim = embed_dim
        self.encoder_reset_interval = encoder_reset_interval

        # LAP
        self.lap_min_priority = lap_min_priority

        # Checkpoint readiness: 初始 checkpoint 与 actor 相同 (随机权重),
        # 在被有意义地更新前不应使用
        self.checkpoint_ready = False

        self.total_it = 0

    def select_action(self, state, goal, noise=0.0, use_checkpoint=False):
        """
        选择动作, 支持 Checkpoint Policy

        Args:
            state: 环境状态 (numpy array)
            goal: 相对目标位置 (numpy array)
            noise: 探索噪声标准差
            use_checkpoint: 是否使用 checkpoint actor (50/50 策略)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device)
        with torch.no_grad():
            if use_checkpoint:
                action = self.actor_checkpoint(state_t, goal_t).cpu().numpy().flatten()
            else:
                action = self.actor(state_t, goal_t).cpu().numpy().flatten()
        if noise > 0:
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(
                -self.max_action, self.max_action)
        return action

    def update_checkpoint(self):
        """将当前 actor 复制到 checkpoint (当发现更好的策略时调用)"""
        self.actor_checkpoint.load_state_dict(self.actor.state_dict())
        self.checkpoint_ready = True

    def train(self, replay_buffer, batch_size=256, bc_weight=0.0):
        """
        TD7 训练步 (SALE + LAP + BC)

        与 TD3 的关键区别:
        1. Critic 使用 SALE 内积结构: Q = f(s,g)^T · g(s,g,a) / sqrt(d)
        2. 优先级基于 per-sample critic loss (LAP), 不使用 IS 权重修正
        3. Critic target 使用 EMA 更新 + 周期性硬重置
        4. Actor target 仍使用 TD3 的 delayed soft update

        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            bc_weight: 行为克隆损失权重
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

        # ---- Critic Update (SALE) ----
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, goal) + noise).clamp(
                -self.max_action, self.max_action)
            # 使用 EMA target critic 计算 target Q
            target_Q1, target_Q2 = self.critic_target(
                next_state, goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, goal, action)

        # Per-sample critic loss (用于 LAP 优先级更新)
        per_sample_loss = (current_Q1 - target_Q).pow(2) + \
                          (current_Q2 - target_Q).pow(2)

        # LAP: 不使用 IS 权重, 直接求均值
        critic_loss = per_sample_loss.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---- LAP Priority Update ----
        # 优先级 = max(per_sample_loss, min_priority)
        # 传给 buffer.update_priorities, buffer 内部会应用 (|x|+ε)^α 变换
        if indices is not None:
            lap_priorities = torch.clamp(
                per_sample_loss.detach().squeeze(),
                min=self.lap_min_priority)
            replay_buffer.update_priorities(indices, lap_priorities)

        # ---- EMA Update: Critic Target (every step) ----
        for p, tp in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # ---- Periodic Encoder Reset (SALE 核心机制) ----
        # 周期性将 target critic 硬重置为当前 critic,
        # 防止 EMA target 过于陈旧导致 Q 估计偏差累积
        if self.total_it % self.encoder_reset_interval == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())

        # ---- Delayed Actor Update (with BC Loss) ----
        actor_loss_val = 0.0
        bc_loss_val = 0.0
        if self.total_it % self.policy_freq == 0:
            pi_action = self.actor(state, goal)

            # 标准 RL 目标: 最大化 Q 值
            rl_loss = -self.critic.Q1(state, goal, pi_action).mean()

            # 行为克隆损失: 仅在 Expert transition 上, Q-filtered
            bc_loss = torch.tensor(0.0, device=device)
            if bc_weight > 0.0 and is_expert.sum() > 0:
                expert_mask = (is_expert > 0.5).squeeze()  # (batch,)
                if expert_mask.any():
                    expert_states = state[expert_mask]
                    expert_goals = goal[expert_mask]
                    expert_actions = action[expert_mask]
                    pi_expert = pi_action[expert_mask]

                    # Q-filter: 只在 Q(s,g,a_expert) > Q(s,g,π(s,g)) 时模仿
                    with torch.no_grad():
                        q_expert = self.critic.Q1(
                            expert_states, expert_goals, expert_actions)
                        q_pi = self.critic.Q1(
                            expert_states, expert_goals, pi_expert)
                        bc_filter = (q_expert > q_pi).squeeze()

                    if bc_filter.any():
                        bc_loss = F.mse_loss(
                            pi_expert[bc_filter], expert_actions[bc_filter])

            total_actor_loss = rl_loss + bc_weight * bc_loss

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft update actor target (同 TD3, 仅在 policy_freq 时更新)
            for p, tp in zip(self.actor.parameters(),
                             self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            actor_loss_val = rl_loss.item()
            bc_loss_val = bc_loss.item()

        return critic_loss.item(), actor_loss_val, bc_loss_val
