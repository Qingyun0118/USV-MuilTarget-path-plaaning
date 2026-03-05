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

class GaussianActor(nn.Module):
    """
    SAC Gaussian Actor (Goal-Conditioned):
    (State, Goal) → mean, log_std → action via reparameterization trick

    使用 tanh 压缩输出到 [-max_action, max_action]
    """

    def __init__(self, state_dim, goal_dim, action_dim, max_action=1.0):
        super().__init__()
        input_dim = state_dim + goal_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)
        self.max_action = max_action

        # 权重初始化
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
    SAC Twin Critic (Goal-Conditioned): (State, Goal, Action) → Q1, Q2
    """

    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        total_dim = state_dim + goal_dim + action_dim
        # Q1
        self.l1 = nn.Linear(total_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(total_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)


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
                 target_update_interval=1):
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
        self.target_update_interval = target_update_interval
        self.total_it = 0

        # 熵正则化
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            # 目标熵 = -dim(action)
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_actor)
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

    def train(self, replay_buffer, batch_size=256, bc_weight=0.0):
        """
        SAC 训练 (Goal-Conditioned) + 行为克隆损失 (BC Loss)

        联合 Actor 损失:
          L_actor = E[α * log π(a|s,g) - Q(s,g,a)] + bc_weight * E_{mpc}[||μ(s,g) - a_mpc||²]

        BC loss 使用 actor 的 mean 输出 (确定性部分) 与 MPC 动作的 MSE,
        仅在 Q(s,g,a_mpc) > Q(s,g,π(s,g)) 时施加 (Q-filtered BC)。

        Returns:
            critic_loss, actor_loss (RL部分), bc_loss
        """
        if len(replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0

        state, action, reward, next_state, done, goal, is_mpc = \
            replay_buffer.sample(batch_size)
        self.total_it += 1

        # ---- Critic Update ----
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state, goal)
            target_Q1, target_Q2 = self.critic_target(next_state, goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pi
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, goal, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

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

        # 行为克隆损失 (Q-filtered BC)
        bc_loss = torch.tensor(0.0, device=device)
        if bc_weight > 0.0 and is_mpc.sum() > 0:
            mpc_mask = (is_mpc > 0.5).squeeze()
            if mpc_mask.any():
                mpc_states = state[mpc_mask]
                mpc_goals = goal[mpc_mask]
                mpc_actions = action[mpc_mask]
                # 使用 actor 的 mean 输出进行 BC (确定性部分)
                mpc_pi_mean = pi_mean[mpc_mask]

                # Q-filter: 只在 Q(s,g,a_mpc) > Q(s,g,π(s,g)) 时模仿
                with torch.no_grad():
                    q_mpc = self.critic.Q1(mpc_states, mpc_goals, mpc_actions)
                    q_pi = self.critic.Q1(mpc_states, mpc_goals, mpc_pi_mean)
                    bc_filter = (q_mpc > q_pi).squeeze()

                if bc_filter.any():
                    bc_loss = F.mse_loss(
                        mpc_pi_mean[bc_filter], mpc_actions[bc_filter])

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
            self.alpha = self.log_alpha.exp().item()

        # ---- Soft update target critic ----
        if self.total_it % self.target_update_interval == 0:
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), rl_loss.item(), bc_loss.item()
