"""
MAPPO (Multi-Agent PPO) — CTDE 多 USV 协同导航

组件:
  - CentralizedCritic:   V(s_global) 集中式价值函数
  - MAPPORolloutBuffer:  On-policy 轨迹存储 + 逐智能体 GAE
  - compute_log_prob:    处理 tanh 压缩动作的 log π 重新计算
  - MAPPOTrainer:        PPO 训练逻辑 (共享 Actor + 集中式 Critic)

设计原则:
  - Parameter Sharing: 所有智能体共享 Actor 和 Critic 参数
  - Actor 使用预训练 SAC 的 GaussianActor (局部观测 26D + 相对目标 2D)
  - Critic 使用全局状态 (7N + 3T + 4 + N(N-1)) 进行价值估计
  - PPO clip 目标函数, 保证策略更新稳定
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from hrl_config import device
from sac_agent import SAC_EPSILON


# ============================================================================
# ========================= Centralized Critic ===============================
# ============================================================================

class CentralizedCritic(nn.Module):
    """
    集中式价值函数 V(s_global).

    输入全局状态向量, 输出标量价值估计.
    使用 LayerNorm + Orthogonal Init (PPO 标准做法).
    """

    def __init__(self, global_state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(global_state_dim),
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, global_state):
        """
        Args:
            global_state: (batch, global_state_dim)
        Returns:
            value: (batch, 1)
        """
        return self.net(global_state)


# ============================================================================
# ========================= Rollout Buffer ===================================
# ============================================================================

class MAPPORolloutBuffer:
    """
    On-policy 轨迹缓冲区, 支持逐智能体 GAE 计算.

    存储布局: (T, N, dim)
      T = 时间步数 (可变)
      N = 智能体数量

    使用流程:
      1. rollout 阶段: 逐步 add()
      2. rollout 结束: compute_gae(last_values)
      3. PPO 更新:    get_batches(batch_size) 产生 mini-batch
      4. 更新完毕:    clear()
    """

    def __init__(self, max_steps, num_agents, obs_dim, action_dim,
                 global_state_dim, goal_dim):
        self.max_steps = max_steps
        self.num_agents = num_agents

        # (T, N, dim)
        self.obs = np.zeros((max_steps, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_steps, num_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.rewards = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.terminals = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.goals = np.zeros((max_steps, num_agents, goal_dim), dtype=np.float32)
        self.values = np.zeros((max_steps, num_agents), dtype=np.float32)

        # (T, global_state_dim) — 同一时间步所有智能体共享
        self.global_states = np.zeros((max_steps, global_state_dim), dtype=np.float32)

        # GAE 计算结果
        self.advantages = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.returns = np.zeros((max_steps, num_agents), dtype=np.float32)

        self.ptr = 0

    def add(self, obs_arr, actions_arr, log_probs_arr, rewards_arr,
            terminals_arr, goals_arr, global_state, values_arr):
        """
        添加一个时间步的全部智能体数据.

        Args:
            obs_arr:        (N, obs_dim)
            actions_arr:    (N, action_dim)
            log_probs_arr:  (N,)
            rewards_arr:    (N,)
            terminals_arr:  (N,)  — 1.0 表示真正终止 (非 truncation)
            goals_arr:      (N, goal_dim)
            global_state:   (global_state_dim,)
            values_arr:     (N,)
        """
        t = self.ptr
        self.obs[t] = obs_arr
        self.actions[t] = actions_arr
        self.log_probs[t] = log_probs_arr
        self.rewards[t] = rewards_arr
        self.terminals[t] = terminals_arr
        self.goals[t] = goals_arr
        self.global_states[t] = global_state
        self.values[t] = values_arr
        self.ptr += 1

    def compute_gae(self, last_values, gamma=0.99, gae_lambda=0.95):
        """
        逐智能体计算 GAE 优势和 returns.

        Args:
            last_values: list/array of length N, 最后一步的 bootstrap 价值.
                         真正终止时为 0, truncation 时为 V(s_final).
        """
        n = self.ptr
        for i in range(self.num_agents):
            gae = 0.0
            for t in reversed(range(n)):
                next_val = last_values[i] if t == n - 1 else self.values[t + 1, i]
                non_terminal = 1.0 - self.terminals[t, i]
                delta = (self.rewards[t, i]
                         + gamma * next_val * non_terminal
                         - self.values[t, i])
                gae = delta + gamma * gae_lambda * non_terminal * gae
                self.advantages[t, i] = gae
            self.returns[:n, i] = self.advantages[:n, i] + self.values[:n, i]

    def get_batches(self, batch_size):
        """
        将 (T, N, dim) 展平为 (T*N, dim), 随机打乱后产生 mini-batch.

        Yields:
            dict with keys: obs, actions, old_log_probs, advantages,
                            returns, goals, global_states
        """
        n = self.ptr
        total = n * self.num_agents

        obs_flat = self.obs[:n].reshape(total, -1)
        actions_flat = self.actions[:n].reshape(total, -1)
        log_probs_flat = self.log_probs[:n].reshape(total)
        advantages_flat = self.advantages[:n].reshape(total)
        returns_flat = self.returns[:n].reshape(total)
        goals_flat = self.goals[:n].reshape(total, -1)

        # 全局状态: (T, gs_dim) → 复制 N 次 → (T*N, gs_dim)
        gs_expanded = np.repeat(self.global_states[:n], self.num_agents, axis=0)

        indices = np.random.permutation(total)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]
            yield {
                'obs': torch.FloatTensor(obs_flat[idx]).to(device),
                'actions': torch.FloatTensor(actions_flat[idx]).to(device),
                'old_log_probs': torch.FloatTensor(log_probs_flat[idx]).to(device),
                'advantages': torch.FloatTensor(advantages_flat[idx]).to(device),
                'returns': torch.FloatTensor(returns_flat[idx]).to(device),
                'goals': torch.FloatTensor(goals_flat[idx]).to(device),
                'global_states': torch.FloatTensor(gs_expanded[idx]).to(device),
            }

    def clear(self):
        self.ptr = 0


# ============================================================================
# ========================= Log-Prob Utility =================================
# ============================================================================

def compute_log_prob(actor, state_t, goal_t, action_t):
    """
    重新计算 log π(action | state, goal), 处理 tanh 压缩.

    用于 PPO 比率计算: ratio = exp(new_log_prob - old_log_prob).

    Args:
        actor:    GaussianActor 实例
        state_t:  (batch, state_dim)
        goal_t:   (batch, goal_dim)
        action_t: (batch, action_dim) — 已经过 tanh 压缩的动作

    Returns:
        log_prob: (batch,)
        entropy:  (batch,)
    """
    mean, log_std = actor(state_t, goal_t)
    std = log_std.exp()

    # 反 tanh: action = tanh(x) * max_action → x = atanh(action / max_action)
    max_action = actor.max_action
    a_norm = (action_t / max_action).clamp(-0.999, 0.999)
    x_t = torch.atanh(a_norm)

    dist = Normal(mean, std)
    log_prob = dist.log_prob(x_t)
    # Jacobian 修正: log |d(tanh)/dx| = log(1 - tanh²(x))
    log_prob -= torch.log(max_action * (1 - a_norm.pow(2)) + SAC_EPSILON)
    log_prob = log_prob.sum(dim=-1)

    entropy = dist.entropy().sum(dim=-1)

    return log_prob, entropy


# ============================================================================
# ========================= MAPPO Trainer ====================================
# ============================================================================

class MAPPOTrainer:
    """
    MAPPO 训练器: 共享 Actor + 集中式 Critic.

    Actor:  GaussianActor (从预训练 SAC 加载, 使用较小学习率)
    Critic: CentralizedCritic (全新初始化, 使用较大学习率)

    PPO 更新要点:
      - Clip 目标函数 (防止策略大幅偏移)
      - GAE 优势标准化
      - 梯度裁剪
      - 熵正则化 (鼓励探索)
    """

    def __init__(self, actor, critic,
                 lr_actor=3e-5, lr_critic=3e-4,
                 clip_eps=0.2, entropy_coef=0.01,
                 max_grad_norm=0.5, ppo_epochs=10,
                 mini_batch_size=256, gamma=0.99, gae_lambda=0.95):
        self.actor = actor
        self.critic = critic

        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=lr_critic, eps=1e-5)

        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    # ---- 推理辅助 ----

    @torch.no_grad()
    def get_action_and_value(self, state, goal, global_state):
        """
        采样动作并估计价值 (rollout 阶段使用).

        Args:
            state:        np.array (obs_dim,)
            goal:         np.array (goal_dim,)
            global_state: np.array (global_state_dim,)

        Returns:
            action:   np.array (action_dim,)
            log_prob: float
            value:    float
        """
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        g = torch.FloatTensor(goal).unsqueeze(0).to(device)
        gs = torch.FloatTensor(global_state).unsqueeze(0).to(device)

        action, log_prob, _ = self.actor.sample(s, g)
        value = self.critic(gs)

        return (action.cpu().numpy().flatten(),
                log_prob.cpu().item(),
                value.cpu().item())

    @torch.no_grad()
    def get_value(self, global_state):
        """获取价值估计 (用于 GAE bootstrap)."""
        gs = torch.FloatTensor(global_state).unsqueeze(0).to(device)
        return self.critic(gs).item()

    @torch.no_grad()
    def get_deterministic_action(self, state, goal):
        """获取确定性动作 (评估模式)."""
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        g = torch.FloatTensor(goal).unsqueeze(0).to(device)
        action = self.actor.get_deterministic_action(s, g)
        return action.cpu().numpy().flatten()

    # ---- PPO 更新 ----

    def update(self, buffer):
        """
        PPO 多 epoch mini-batch 更新.

        Args:
            buffer: MAPPORolloutBuffer (已调用 compute_gae)

        Returns:
            dict: {'actor_loss', 'critic_loss', 'entropy', 'approx_kl'}
        """
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _epoch in range(self.ppo_epochs):
            for batch in buffer.get_batches(self.mini_batch_size):
                obs = batch['obs']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                goals = batch['goals']
                global_states = batch['global_states']

                # 标准化优势
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ---- Actor 更新 ----
                new_log_probs, entropy = compute_log_prob(
                    self.actor, obs, goals, actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_mean = entropy.mean()

                actor_total = actor_loss - self.entropy_coef * entropy_mean

                self.actor_optimizer.zero_grad()
                actor_total.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # ---- Critic 更新 ----
                values = self.critic(global_states).squeeze(-1)
                critic_loss = F.mse_loss(values, returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # 统计
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean().item()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_mean.item()
                total_approx_kl += approx_kl
                n_updates += 1

        n = max(1, n_updates)
        return {
            'actor_loss': total_actor_loss / n,
            'critic_loss': total_critic_loss / n,
            'entropy': total_entropy / n,
            'approx_kl': total_approx_kl / n,
        }
