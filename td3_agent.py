"""
TD3 (Twin Delayed DDPG) Actuator Agent — Goal-Conditioned

直接控制 USV 的油门和舵角, 目标为到达 Selector 指定的目标点。
(State, RelativeGoal) → (Throttle, Rudder)

对齐 Fujimoto et al. (2018) 原始 TD3 实现:
1. SensorCNN: Flatten+FC 保留空间分辨率 (领域定制)
2. 2 层隐藏层 (256), ReLU 激活, 无 LayerNorm
3. Critic Q1/Q2 拥有独立 SensorCNN
4. 早期目标融合 (Goal-Conditioned)
5. 无梯度裁剪, 使用 copy.deepcopy 初始化 target 网络
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
# ========================= Neural Networks ==================================
# ============================================================================

class Actor(nn.Module):
    """
    TD3 Actor (Goal-Conditioned) — 对齐原始 TD3

    结构: SensorCNN + 2 层隐藏层 (256), 无 LayerNorm
    对齐 Fujimoto et al. (2018): 简洁的 2 层 MLP + ReLU

    (State, Goal) → Action
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

        # 2 层隐藏层 (对齐原始 TD3, 无 LayerNorm)
        self.fc1 = nn.Linear(fused_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, goal):
        non_sensor, sensor = _split_state(state)
        cnn_out = self.sensor_cnn(sensor)

        x = torch.cat([non_sensor, cnn_out, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.max_action * torch.tanh(self.fc_out(x))


class Critic(nn.Module):
    """
    TD3 Twin Critic (Goal-Conditioned) — 对齐原始 TD3

    结构: 独立 SensorCNN + 2 层隐藏层 (256), 无 LayerNorm
    关键: Critic 不使用 LayerNorm, 保留 Q 值的原始尺度信息

    (State, Goal, Action) → Q1, Q2
    """

    def __init__(self, state_dim, goal_dim, action_dim,
                 cnn_output_dim=64, hidden_dim=256):
        super().__init__()
        non_sensor_dim = state_dim - SENSOR_DIM

        # 独立 SensorCNN
        self.sensor_cnn1 = SensorCNN(SENSOR_DIM, cnn_output_dim)
        self.sensor_cnn2 = SensorCNN(SENSOR_DIM, cnn_output_dim)

        # 输入维度: non_sensor(11) + cnn_out(64) + goal(2) + action(2) = 79
        fused_dim = non_sensor_dim + cnn_output_dim + goal_dim + action_dim

        # Q1 网络 (2 层, 无 LayerNorm)
        self.l1 = nn.Linear(fused_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3_out = nn.Linear(hidden_dim, 1)

        # Q2 网络 (2 层, 无 LayerNorm)
        self.l4 = nn.Linear(fused_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6_out = nn.Linear(hidden_dim, 1)

    def _q1_forward(self, non_sensor, sensor, goal, action):
        cnn_out = self.sensor_cnn1(sensor)
        x = torch.cat([non_sensor, cnn_out, goal, action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3_out(x)

    def _q2_forward(self, non_sensor, sensor, goal, action):
        cnn_out = self.sensor_cnn2(sensor)
        x = torch.cat([non_sensor, cnn_out, goal, action], dim=1)
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return self.l6_out(x)

    def forward(self, state, goal, action):
        non_sensor, sensor = _split_state(state)
        q1 = self._q1_forward(non_sensor, sensor, goal, action)
        q2 = self._q2_forward(non_sensor, sensor, goal, action)
        return q1, q2

    def Q1(self, state, goal, action):
        non_sensor, sensor = _split_state(state)
        return self._q1_forward(non_sensor, sensor, goal, action)


# ============================================================================
# ========================= Expert Demo Buffer ===============================
# ============================================================================

class ExpertDemoBuffer:
    """
    Expert 演示经验永久缓冲区 (DDPGfD-style)

    固定容量的环形缓冲区, 仅存储 Expert 控制器生成的 transition,
    永远不会被 RL 自身产生的数据覆盖。

    解决问题: 主 Replay Buffer 容量有限, 训练后期 Expert 数据被新 RL 数据
    (大量碰撞样本) 覆盖, 导致 Critic 对避障动作的 Q 估计崩溃 (外推误差),
    Actor 丧失避障能力, 形成灾难性遗忘的死亡螺旋。

    用法:
      1. 训练时, 每当 is_expert=True, 同时 push 到主 Buffer 和 DemoBuffer
      2. train() 中从 DemoBuffer 采样 demo_batch_size 条, 与主 Buffer 采样合并
      3. DemoBuffer 的数据不参与 PER 优先级, 均匀采样即可

    存储 (state, action, reward, next_state, done, goal) 六元组
    """

    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, goal):
        data = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
            np.array(goal, dtype=np.float32),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """
        均匀采样

        Returns:
            states, actions, rewards, next_states, dones, goals, is_experts
            (is_experts 全为 1.0, 因为 DemoBuffer 只存放 Expert 数据)
        """
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states, dones, goals = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).reshape(-1, 1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).reshape(-1, 1).to(device),
            torch.FloatTensor(np.array(goals)).to(device),
            torch.ones(batch_size, 1, device=device),  # is_expert = 1.0
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# ========================= TD3 Actuator Agent ===============================
# ============================================================================

class TD3ActuatorAgent:
    """
    TD3 Actuator Agent — Goal-Conditioned

    直接控制 USV 的油门和舵角, 目标为到达 Selector 指定的目标点。
    (State, RelativeGoal) → (Throttle, Rudder)
    """

    def __init__(self, state_dim, goal_dim, action_dim, max_action=1.0,
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 bc_tau=0.5, use_soft_q_filter=True):
        self.actor = Actor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.bc_tau = bc_tau  # Soft Q-filter 温度参数
        self.use_soft_q_filter = use_soft_q_filter
        self.total_it = 0

        # 训练期 BC 统计（按 actor 更新累计，供主循环按 episode 消费）
        self._bc_metric_sum_soft_weight = 0.0
        self._bc_metric_sum_hit_rate = 0.0
        self._bc_metric_sum_expert_ratio = 0.0
        self._bc_metric_count = 0

    def select_action(self, state, goal, noise=0.0):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state_t, goal_t).cpu().numpy().flatten()
        if noise > 0:
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(
                -self.max_action, self.max_action)
        return action

    def _record_bc_metrics(self, soft_weight_avg, bc_hit_rate, expert_sample_ratio):
        self._bc_metric_sum_soft_weight += float(soft_weight_avg)
        self._bc_metric_sum_hit_rate += float(bc_hit_rate)
        self._bc_metric_sum_expert_ratio += float(expert_sample_ratio)
        self._bc_metric_count += 1

    def consume_bc_metrics(self):
        if self._bc_metric_count == 0:
            return {
                'avg_soft_weight': 0.0,
                'bc_hit_rate': 0.0,
                'expert_sample_ratio': 0.0,
                'num_updates': 0,
            }

        inv_n = 1.0 / float(self._bc_metric_count)
        stats = {
            'avg_soft_weight': self._bc_metric_sum_soft_weight * inv_n,
            'bc_hit_rate': self._bc_metric_sum_hit_rate * inv_n,
            'expert_sample_ratio': self._bc_metric_sum_expert_ratio * inv_n,
            'num_updates': self._bc_metric_count,
        }
        self._bc_metric_sum_soft_weight = 0.0
        self._bc_metric_sum_hit_rate = 0.0
        self._bc_metric_sum_expert_ratio = 0.0
        self._bc_metric_count = 0
        return stats

    def train(self, replay_buffer, batch_size=256, bc_weight=0.0,
              demo_buffer=None, demo_ratio=0.25):
        """
        TD3 训练 (Goal-Conditioned) + 行为克隆损失 (BC Loss) + Expert Demo Buffer

        参考 DDPGfD / TD3fD 的做法:
        - 对 Expert 专家生成的 transition, 除了标准 Q-maximization 外,
          额外添加 BC loss 让 actor 直接模仿 Expert 动作
        - BC loss 权重随训练推进衰减 (与 Expert 概率同步)
        - 若提供 demo_buffer (ExpertDemoBuffer), 每个 batch 中混入永久 Expert 数据,
          防止主 Buffer 覆盖 Expert 后导致灾难性遗忘

        联合 Actor 损失:
          L_actor = -E[Q(s,g,π(s,g))] + bc_weight * E_{expert}[||π(s,g) - a_expert||²]

        其中第二项仅在 Expert transition 上计算, 仅当 Q(s,g,a_expert) > Q(s,g,π(s,g))
        时施加 (即只在 Expert 动作确实比当前策略好时才模仿, 避免模仿劣质 Expert 动作)

        Args:
            replay_buffer: 经验回放缓冲区 (主 Buffer)
            batch_size: 总批次大小 (含 demo 部分)
            bc_weight: 行为克隆损失权重 (通常随训练衰减)
            demo_buffer: ExpertDemoBuffer 实例, 永久保存 Expert 数据 (可选)
            demo_ratio: demo_buffer 在每个 batch 中的占比, 默认 0.25
        """
        if len(replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0

        # ---- 混合采样: 主 Buffer + Demo Buffer ----
        use_demo = (demo_buffer is not None and len(demo_buffer) > 0)
        if use_demo:
            demo_batch_size = max(1, int(batch_size * demo_ratio))
            main_batch_size = batch_size - demo_batch_size
        else:
            main_batch_size = batch_size
            demo_batch_size = 0

        sample_result = replay_buffer.sample(main_batch_size)

        # 兼容 PER buffer (返回 9 元素) 和普通 buffer (返回 7 元素)
        if len(sample_result) == 9:
            state, action, reward, next_state, done, goal, is_expert, \
                indices, is_weights = sample_result
        else:
            state, action, reward, next_state, done, goal, is_expert = sample_result
            indices = None
            is_weights = torch.ones(main_batch_size, 1, device=state.device)

        # 合并 Demo Buffer 采样
        if use_demo:
            demo_sample = demo_buffer.sample(demo_batch_size)
            d_state, d_action, d_reward, d_next_state, d_done, d_goal, d_is_expert = demo_sample
            state = torch.cat([state, d_state], dim=0)
            action = torch.cat([action, d_action], dim=0)
            reward = torch.cat([reward, d_reward], dim=0)
            next_state = torch.cat([next_state, d_next_state], dim=0)
            done = torch.cat([done, d_done], dim=0)
            goal = torch.cat([goal, d_goal], dim=0)
            is_expert = torch.cat([is_expert, d_is_expert], dim=0)
            # Demo 部分 IS 权重 = 1.0 (均匀采样, 无需修正)
            is_weights = torch.cat([
                is_weights,
                torch.ones(demo_batch_size, 1, device=device)
            ], dim=0)

        self.total_it += 1

        # ---- Critic Update ----
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, goal) + noise).clamp(
                -self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, goal, action)

        # 逐样本 TD error (用于 PER 优先级更新)
        td_error1 = (current_Q1 - target_Q).abs()
        td_error2 = (current_Q2 - target_Q).abs()
        td_errors = torch.max(td_error1, td_error2).detach()

        # IS 权重加权的 critic loss
        critic_loss = (is_weights * (current_Q1 - target_Q).pow(2)).mean() + \
                      (is_weights * (current_Q2 - target_Q).pow(2)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Delayed Actor Update (with BC Loss) ----
        actor_loss_val = 0.0
        bc_loss_val = 0.0
        if self.total_it % self.policy_freq == 0:
            pi_action = self.actor(state, goal)

            expert_sample_ratio = (is_expert > 0.5).float().mean().item()
            soft_weight_avg = 0.0
            bc_hit_rate = 0.0

            # 标准 RL 目标: 最大化 Q 值
            rl_loss = -self.critic.Q1(state, goal, pi_action).mean()

            # 行为克隆损失: 仅在 Expert transition 上施加
            # 使用 Soft Q-filter: 用 sigmoid 软权重替代硬二值过滤
            #   w_i = σ((Q(s,g,a_expert) - Q(s,g,π(s,g))) / τ)
            # 训练初期 Q 不准 → q_adv ≈ 0 → w ≈ 0.5, 保留一半梯度
            # 训练后期 Q 准确 → 优质动作 w→1, 劣质动作 w→0, 自然过滤
            bc_loss = torch.tensor(0.0, device=device)
            if bc_weight > 0.0 and is_expert.sum() > 0:
                expert_mask = (is_expert > 0.5).squeeze()  # (batch,)
                if expert_mask.any():
                    expert_states = state[expert_mask]
                    expert_goals = goal[expert_mask]
                    expert_actions = action[expert_mask]  # Expert 专家动作
                    pi_expert = pi_action[expert_mask]    # actor 当前输出

                    if self.use_soft_q_filter:
                        # Soft Q-filter: sigmoid 连续权重
                        with torch.no_grad():
                            q_expert = self.critic.Q1(expert_states, expert_goals, expert_actions)
                            q_pi = self.critic.Q1(expert_states, expert_goals, pi_expert)
                            q_advantage = q_expert - q_pi  # (n_expert, 1)
                            soft_weight = torch.sigmoid(q_advantage / self.bc_tau)  # (n_expert, 1)
                            soft_weight_avg = soft_weight.mean().item()
                            bc_hit_rate = (q_advantage > 0).float().mean().item()

                        per_sample_bc = (pi_expert - expert_actions).pow(2).mean(dim=1, keepdim=True)
                        bc_loss = (soft_weight * per_sample_bc).mean()
                    else:
                        # 硬 Q-filter
                        with torch.no_grad():
                            q_expert = self.critic.Q1(expert_states, expert_goals, expert_actions)
                            q_pi = self.critic.Q1(expert_states, expert_goals, pi_expert)
                            # 仅保留 Q(expert) > Q(pi) 的样本
                            mask = (q_expert > q_pi).float()
                            bc_hit_rate = mask.mean().item()
                            soft_weight_avg = bc_hit_rate
                        
                        per_sample_bc = (pi_expert - expert_actions).pow(2).mean(dim=1, keepdim=True)
                        bc_loss = (mask * per_sample_bc).sum() / (mask.sum() + 1e-8)

            total_actor_loss = rl_loss + bc_weight * bc_loss

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            actor_loss_val = rl_loss.item()
            bc_loss_val = bc_loss.item()
            self._record_bc_metrics(soft_weight_avg, bc_hit_rate, expert_sample_ratio)

        # 更新 PER 优先级 (仅主 Buffer 部分, Demo Buffer 无 PER)
        if indices is not None:
            replay_buffer.update_priorities(indices, td_errors[:main_batch_size])

        return critic_loss.item(), actor_loss_val, bc_loss_val
