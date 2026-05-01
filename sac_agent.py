"""
SAC (Soft Actor-Critic) Actuator Agent — Goal-Conditioned

使用 Soft Actor-Critic 算法控制 USV 的油门和舵角。
(State, RelativeGoal) → (Throttle, Rudder)

与 TD3 的主要区别:
1. Actor: 确定性策略 → 高斯随机策略 (reparameterization trick)
2. 无 target actor 网络, 仅保留 target critic
3. 熵正则化: 自动调节温度参数 alpha, 平衡探索与利用
4. 无需 TD3 的 delayed policy update 和 target policy smoothing noise
5. 动作采样自带随机性, 无需显式添加探索噪声
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from hrl_config import device

# SAC 常量
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
SAC_EPSILON = 1e-6


# ============================================================================
# ========================= Neural Networks ==================================
# ============================================================================

# 状态维度布局: [o_usv(7) | o_detector(15) | o_current(4)] = 26
SENSOR_START = 7   # o_detector 在 state 中的起始索引
SENSOR_END = 22    # o_detector 在 state 中的结束索引 (不含)
SENSOR_DIM = SENSOR_END - SENSOR_START  # 15


class SensorCNN(nn.Module):
    """
    传感器 CNN: 保留空间分辨率, 提取障碍物空间特征

    与之前的 AdaptiveAvgPool 版本不同, 这里使用 Flatten + FC,
    保留左/右方向的空间信息, 使 Agent 能区分障碍物方位。

    Input:  (batch, sensor_dim)  e.g. (batch, 15)
    Output: (batch, output_dim)  e.g. (batch, 64)
    """

    def __init__(self, sensor_dim=SENSOR_DIM, output_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * sensor_dim, output_dim)

    def forward(self, sensor):
        # sensor: (batch, sensor_dim)
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


class GaussianActor(nn.Module):
    """
    SAC Gaussian Actor (Goal-Conditioned) — 改进版

    改进:
    1. SensorCNN 处理传感器数据, 保留空间分辨率
    2. 早期目标融合: goal 在第一层就参与计算
    3. 3 层隐藏层 + LayerNorm, 增强表达能力和训练稳定性
    4. 输入特征 LayerNorm 归一化

    (State, Goal) → mean, log_std → action via reparameterization trick
    使用 tanh 压缩输出到 [-max_action, max_action]
    """

    def __init__(self, state_dim, goal_dim, action_dim, max_action=1.0,
                 cnn_output_dim=64, hidden_dim=256):
        super().__init__()
        self.max_action = max_action
        non_sensor_dim = state_dim - SENSOR_DIM  # 26 - 15 = 11

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
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

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

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, goal):
        """
        采样动作 (reparameterization trick + tanh squashing)

        Returns:
            action: squashed action ∈ [-max_action, max_action]
            log_prob: log probability (已包含 tanh 修正)
            mean_action: deterministic action (tanh(mean))
        """
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action

        # log_prob 修正: 减去 tanh 的 Jacobian 对数行列式
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + SAC_EPSILON)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean_action = torch.tanh(mean) * self.max_action
        return action, log_prob, mean_action

    def get_deterministic_action(self, state, goal):
        """获取确定性动作 (评估时使用)"""
        mean, _ = self.forward(state, goal)
        return torch.tanh(mean) * self.max_action


class SACCritic(nn.Module):
    """
    SAC Twin Critic (Goal-Conditioned) — 改进版

    改进:
    1. Q1/Q2 各自拥有独立的 SensorCNN, 确保 Twin Q 真正獨立
    2. 3 层隐藏层 + LayerNorm
    3. 输入归一化

    (State, Goal, Action) → Q1, Q2
    """

    def __init__(self, state_dim, goal_dim, action_dim,
                 cnn_output_dim=64, hidden_dim=256):
        super().__init__()
        non_sensor_dim = state_dim - SENSOR_DIM  # 11

        # 独立 SensorCNN
        self.sensor_cnn1 = SensorCNN(SENSOR_DIM, cnn_output_dim)
        self.sensor_cnn2 = SensorCNN(SENSOR_DIM, cnn_output_dim)

        # 输入维度: non_sensor(11) + cnn_out(64) + goal(2) + action(2) = 79
        fused_dim = non_sensor_dim + cnn_output_dim + goal_dim + action_dim

        # Q1 网络
        self.ln1_input = nn.LayerNorm(fused_dim)
        self.l1 = nn.Linear(fused_dim, hidden_dim)
        self.ln1_1 = nn.LayerNorm(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1_2 = nn.LayerNorm(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1_3 = nn.LayerNorm(hidden_dim)
        self.l3_out = nn.Linear(hidden_dim, 1)

        # Q2 网络
        self.ln2_input = nn.LayerNorm(fused_dim)
        self.l4 = nn.Linear(fused_dim, hidden_dim)
        self.ln2_1 = nn.LayerNorm(hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2_2 = nn.LayerNorm(hidden_dim)
        self.l6 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2_3 = nn.LayerNorm(hidden_dim)
        self.l6_out = nn.Linear(hidden_dim, 1)

    def _q1_forward(self, non_sensor, sensor, goal, action):
        cnn_out = self.sensor_cnn1(sensor)
        x = torch.cat([non_sensor, cnn_out, goal, action], dim=1)
        x = self.ln1_input(x)
        x = F.relu(self.ln1_1(self.l1(x)))
        x = F.relu(self.ln1_2(self.l2(x)))
        x = F.relu(self.ln1_3(self.l3(x)))
        return self.l3_out(x)

    def _q2_forward(self, non_sensor, sensor, goal, action):
        cnn_out = self.sensor_cnn2(sensor)
        x = torch.cat([non_sensor, cnn_out, goal, action], dim=1)
        x = self.ln2_input(x)
        x = F.relu(self.ln2_1(self.l4(x)))
        x = F.relu(self.ln2_2(self.l5(x)))
        x = F.relu(self.ln2_3(self.l6(x)))
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
# ========================= SAC Actuator Agent ===============================
# ============================================================================

class SACActuatorAgent:
    """
    SAC Actuator Agent — Goal-Conditioned

    使用 Soft Actor-Critic 算法控制 USV 的油门和舵角。
    (State, RelativeGoal) → (Throttle, Rudder)

    与 TD3 的主要区别:
    1. 随机策略 (Gaussian) + reparameterization trick
    2. 熵正则化 + 自动温度调节
    3. 无 target actor, 无 delayed policy update
    4. 每步都同时更新 critic 和 actor
    """

    def __init__(self, state_dim, goal_dim, action_dim, max_action=1.0,
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005,
                 alpha=0.2, automatic_entropy_tuning=True,
                 target_update_interval=1, lr_alpha=None,
                 bc_tau=0.5, use_soft_q_filter=True):
        # Actor (Gaussian Policy)
        self.actor = GaussianActor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic (Twin Q-Networks)
        self.critic = SACCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = SACCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.target_update_interval = target_update_interval
        self.bc_tau = bc_tau
        self.use_soft_q_filter = use_soft_q_filter
        self.total_it = 0

        # 训练期 BC 统计（按 train 调用累计，供主循环按 episode 消费）
        self._bc_metric_sum_soft_weight = 0.0
        self._bc_metric_sum_hit_rate = 0.0
        self._bc_metric_sum_expert_ratio = 0.0
        self._bc_metric_count = 0

        # 熵正则化
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if lr_alpha is None:
            lr_alpha = lr_actor
        if automatic_entropy_tuning:
            # 目标熵 = -0.5 * dim(action)
            self.target_entropy = -0.5 * float(action_dim)
            self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
            self.log_alpha_min = np.log(0.01)  # alpha 下限 clamp
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

    def select_action(self, state, goal, evaluate=False):
        """
        选取动作

        Args:
            state: actuator 状态
            goal: 相对目标位置
            evaluate: 是否为评估模式 (True 时使用确定性动作)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device)
        with torch.no_grad():
            if evaluate:
                action = self.actor.get_deterministic_action(state_t, goal_t)
            else:
                action, _, _ = self.actor.sample(state_t, goal_t)
        return action.cpu().numpy().flatten()

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

    def train(self, replay_buffer, batch_size=256, bc_weight=0.0):
        """
        SAC 训练 (Goal-Conditioned) + 行为克隆损失 (BC Loss) + PER

        支持优先经验回放: 采样权重修正 critic loss, 返回 TD error 用于更新优先级。

        Returns:
            critic_loss, actor_loss (RL部分), bc_loss, td_errors (tensor), indices
        """
        if len(replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0, None, None

        sample_result = replay_buffer.sample(batch_size)

        # 兼容 PER buffer (返回 9 元素) 和普通 buffer (返回 7 元素)
        if len(sample_result) == 9:
            state, action, reward, next_state, done, goal, is_expert, \
                indices, is_weights = sample_result
        else:
            state, action, reward, next_state, done, goal, is_expert = sample_result
            indices = None
            is_weights = torch.ones(batch_size, 1, device=state.device)

        self.total_it += 1

        # ---- Critic Update ----
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state, goal)
            target_Q1, target_Q2 = self.critic_target(next_state, goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pi
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
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---- Actor Update ----
        pi_action, log_pi, pi_mean = self.actor.sample(state, goal)
        q1_pi, q2_pi = self.critic(state, goal, pi_action)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # SAC actor loss: 最小化 α*log_π - Q
        rl_loss = (self.alpha * log_pi - min_q_pi).mean()

        expert_sample_ratio = (is_expert > 0.5).float().mean().item()
        soft_weight_avg = 0.0
        bc_hit_rate = 0.0

        # 行为克隆损失 (Q-filtered BC)
        bc_loss = torch.tensor(0.0, device=device)
        if bc_weight > 0.0 and is_expert.sum() > 0:
            expert_mask = (is_expert > 0.5).squeeze()
            if expert_mask.any():
                expert_states = state[expert_mask]
                expert_goals = goal[expert_mask]
                expert_actions = action[expert_mask]
                # 使用 actor 的 mean 输出进行 BC (确定性部分)
                expert_pi_mean = pi_mean[expert_mask]

                if self.use_soft_q_filter:
                    with torch.no_grad():
                        q1_expert, q2_expert = self.critic(
                            expert_states, expert_goals, expert_actions)
                        q1_pi, q2_pi = self.critic(
                            expert_states, expert_goals, expert_pi_mean)
                        q_expert = 0.5 * (q1_expert + q2_expert)
                        q_pi = 0.5 * (q1_pi + q2_pi)
                        q_advantage = q_expert - q_pi
                        soft_weight = torch.sigmoid(q_advantage / self.bc_tau)
                        soft_weight_avg = soft_weight.mean().item()
                        bc_hit_rate = (q_advantage > 0).float().mean().item()

                    per_sample_bc = (expert_pi_mean - expert_actions).pow(2).mean(dim=1, keepdim=True)
                    bc_loss = (soft_weight * per_sample_bc).mean()
                else:
                    # Q-filter: 只在 Q(s,g,a_expert) > Q(s,g,π(s,g)) 时模仿
                    with torch.no_grad():
                        q1_expert, q2_expert = self.critic(
                            expert_states, expert_goals, expert_actions)
                        q1_pi, q2_pi = self.critic(
                            expert_states, expert_goals, expert_pi_mean)
                        q_expert = 0.5 * (q1_expert + q2_expert)
                        q_pi = 0.5 * (q1_pi + q2_pi)
                        bc_filter = (q_expert > q_pi).squeeze()
                        bc_hit_rate = bc_filter.float().mean().item()
                        soft_weight_avg = bc_hit_rate

                    if bc_filter.any():
                        bc_loss = F.mse_loss(
                            expert_pi_mean[bc_filter], expert_actions[bc_filter])

        total_actor_loss = rl_loss + bc_weight * bc_loss

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---- 自动熵温度调节 ----
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # clamp log_alpha 防止 alpha 过小
            with torch.no_grad():
                self.log_alpha.clamp_(min=self.log_alpha_min)
            self.alpha = self.log_alpha.exp().item()

        # ---- Soft update target critic ----
        if self.total_it % self.target_update_interval == 0:
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # 更新 PER 优先级
        if indices is not None:
            replay_buffer.update_priorities(indices, td_errors)

        self._record_bc_metrics(soft_weight_avg, bc_hit_rate, expert_sample_ratio)

        return critic_loss.item(), rl_loss.item(), bc_loss.item(), td_errors, indices
