"""
TD3 (Twin Delayed DDPG) Actuator Agent — Goal-Conditioned

直接控制 USV 的油门和舵角, 目标为到达 Selector 指定的目标点。
(State, RelativeGoal) → (Throttle, Rudder)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hrl_config import device


# ============================================================================
# ========================= Neural Networks ==================================
# ============================================================================

class Actor(nn.Module):
    """TD3 Actor: (State, Goal) → Action"""

    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """TD3 Twin Critic: (State, Goal, Action) → Q1, Q2"""

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
# ========================= TD3 Actuator Agent ===============================
# ============================================================================

class TD3ActuatorAgent:
    """
    TD3 Actuator Agent — Goal-Conditioned

    直接控制 USV 的油门和舵角, 目标为到达 Selector 指定的目标点。
    (State, RelativeGoal) → (Throttle, Rudder)
    """

    def __init__(self, state_dim, goal_dim, action_dim, max_action=1.0,
                 lr_actor=1e-4, lr_critic=3e-4, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = Actor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state, goal, noise=0.0):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state_t, goal_t).cpu().numpy().flatten()
        if noise > 0:
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(
                -self.max_action, self.max_action)
        return action

    def train(self, replay_buffer, batch_size=256, bc_weight=0.0):
        """
        TD3 训练 (Goal-Conditioned) + 行为克隆损失 (BC Loss)

        参考 DDPGfD / TD3fD 的做法:
        - 对 Expert 专家生成的 transition, 除了标准 Q-maximization 外,
          额外添加 BC loss 让 actor 直接模仿 Expert 动作
        - BC loss 权重随训练推进衰减 (与 Expert 概率同步)

        联合 Actor 损失:
          L_actor = -E[Q(s,g,π(s,g))] + bc_weight * E_{expert}[||π(s,g) - a_expert||²]

        其中第二项仅在 Expert transition 上计算, 仅当 Q(s,g,a_expert) > Q(s,g,π(s,g))
        时施加 (即只在 Expert 动作确实比当前策略好时才模仿, 避免模仿劣质 Expert 动作)

        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            bc_weight: 行为克隆损失权重 (通常随训练衰减)
        """
        if len(replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0

        state, action, reward, next_state, done, goal, is_expert = \
            replay_buffer.sample(batch_size)
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
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---- Delayed Actor Update (with BC Loss) ----
        actor_loss_val = 0.0
        bc_loss_val = 0.0
        if self.total_it % self.policy_freq == 0:
            pi_action = self.actor(state, goal)

            # 标准 RL 目标: 最大化 Q 值
            rl_loss = -self.critic.Q1(state, goal, pi_action).mean()

            # 行为克隆损失: 仅在 Expert transition 上, 且仅当 Expert 动作比
            # 当前策略动作的 Q 值更高时才施加 (filtered BC)
            bc_loss = torch.tensor(0.0, device=device)
            if bc_weight > 0.0 and is_expert.sum() > 0:
                expert_mask = (is_expert > 0.5).squeeze()  # (batch,)
                if expert_mask.any():
                    expert_states = state[expert_mask]
                    expert_goals = goal[expert_mask]
                    expert_actions = action[expert_mask]  # Expert 专家动作
                    pi_expert = pi_action[expert_mask]    # actor 当前输出

                    # Q-filter: 只在 Q(s,g,a_expert) > Q(s,g,π(s,g)) 时模仿
                    with torch.no_grad():
                        q_expert = self.critic.Q1(expert_states, expert_goals, expert_actions)
                        q_pi = self.critic.Q1(expert_states, expert_goals, pi_expert)
                        bc_filter = (q_expert > q_pi).squeeze()  # (n_expert,)

                    if bc_filter.any():
                        bc_loss = F.mse_loss(
                            pi_expert[bc_filter], expert_actions[bc_filter])

            total_actor_loss = rl_loss + bc_weight * bc_loss

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft update target networks
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            actor_loss_val = rl_loss.item()
            bc_loss_val = bc_loss.item()

        return critic_loss.item(), actor_loss_val, bc_loss_val
