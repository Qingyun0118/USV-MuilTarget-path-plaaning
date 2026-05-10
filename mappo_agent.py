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

    def __init__(self, global_state_dim, hidden_dim=512):
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
        self.active_masks = np.zeros((max_steps, num_agents), dtype=np.float32)

        # (T, global_state_dim) — 同一时间步所有智能体共享
        self.global_states = np.zeros((max_steps, global_state_dim), dtype=np.float32)

        # GAE 计算结果
        self.advantages = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.returns = np.zeros((max_steps, num_agents), dtype=np.float32)

        self.ptr = 0

    def add(self, obs_arr, actions_arr, log_probs_arr, rewards_arr,
            terminals_arr, goals_arr, global_state, values_arr,
            active_masks_arr=None):
        """
        添加一个时间步的全部智能体数据.

        Args:
            obs_arr:          (N, obs_dim)
            actions_arr:      (N, action_dim)
            log_probs_arr:    (N,)
            rewards_arr:      (N,)
            terminals_arr:    (N,)  — 1.0 表示真正终止 (非 truncation)
            goals_arr:        (N, goal_dim)
            global_state:     (global_state_dim,)
            values_arr:       (N,)
            active_masks_arr: (N,)  — 1.0 活跃, 0.0 空闲 (可选, 默认全活跃)
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
        if active_masks_arr is not None:
            self.active_masks[t] = active_masks_arr
        else:
            self.active_masks[t] = 1.0
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
        values_flat = self.values[:n].reshape(total)
        active_masks_flat = self.active_masks[:n].reshape(total)

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
                'old_values': torch.FloatTensor(values_flat[idx]).to(device),
                'active_masks': torch.FloatTensor(active_masks_flat[idx]).to(device),
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

    # NaN 防护: 如果 Actor 权重爆炸导致输出 NaN, 用零均值 + 大方差兜底
    if torch.isnan(mean).any() or torch.isnan(std).any():
        mean = torch.zeros_like(mean)
        std = torch.ones_like(std)

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
                 mini_batch_size=256, gamma=0.99, gae_lambda=0.95,
                 target_kl=0.02, huber_delta=10.0):
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
        self.target_kl = target_kl
        self.huber_delta = huber_delta

        # Return 归一化: 追踪 returns 的运行均值/标准差, 用于缩放 Critic Loss
        self._ret_rms_mean = 0.0
        self._ret_rms_var = 1.0
        self._ret_rms_count = 1e-4  # 避免除零

    def _update_ret_rms(self, returns_np):
        """用 Welford 算法更新 return 运行统计量."""
        batch_mean = returns_np.mean()
        batch_var = returns_np.var()
        batch_count = len(returns_np)

        delta = batch_mean - self._ret_rms_mean
        total = self._ret_rms_count + batch_count
        self._ret_rms_mean += delta * batch_count / total
        m_a = self._ret_rms_var * self._ret_rms_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self._ret_rms_count * batch_count / total
        self._ret_rms_var = m2 / total
        self._ret_rms_count = total

    @property
    def _ret_std(self):
        return max(np.sqrt(self._ret_rms_var), 1e-6)

    def _cal_value_loss(self, values, old_values_norm, returns_norm,
                        active_masks=None):
        """双重裁剪 Value Loss + Huber (对齐参考 MAPPO)."""
        value_pred_clipped = old_values_norm + (
            values - old_values_norm
        ).clamp(-self.clip_eps, self.clip_eps)

        loss_original = F.huber_loss(
            values, returns_norm, reduction='none', delta=self.huber_delta)
        loss_clipped = F.huber_loss(
            value_pred_clipped, returns_norm,
            reduction='none', delta=self.huber_delta)
        value_loss = torch.max(loss_original, loss_clipped)

        if active_masks is not None and active_masks.sum() > 0:
            return (value_loss * active_masks).sum() / active_masks.sum()
        return value_loss.mean()

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

        # 反归一化: Critic 输出归一化 V(s), 还原到原始 reward 尺度用于 GAE
        denorm_value = value.cpu().item() * self._ret_std + self._ret_rms_mean

        return (action.cpu().numpy().flatten(),
                log_prob.cpu().item(),
                denorm_value)

    @torch.no_grad()
    def get_value(self, global_state):
        """获取价值估计 (用于 GAE bootstrap), 反归一化到原始尺度."""
        gs = torch.FloatTensor(global_state).unsqueeze(0).to(device)
        v = self.critic(gs).item()
        return v * self._ret_std + self._ret_rms_mean

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

        改进 (对齐参考 MAPPO):
          - 双重裁剪 Value Loss + Huber
          - Active Masks 排除空闲智能体
          - Masked Advantage 归一化

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
        kl_early_stopped = False

        import copy
        actor_snapshot = copy.deepcopy(self.actor.state_dict())

        n = buffer.ptr
        all_returns = buffer.returns[:n].reshape(-1)
        self._update_ret_rms(all_returns)

        for _epoch in range(self.ppo_epochs):
            if kl_early_stopped:
                break
            for batch in buffer.get_batches(self.mini_batch_size):
                obs = batch['obs']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                goals = batch['goals']
                global_states = batch['global_states']
                old_values = batch['old_values']
                active_masks = batch['active_masks']

                returns_norm = (returns - self._ret_rms_mean) / self._ret_std
                old_values_norm = (old_values - self._ret_rms_mean) / self._ret_std

                # Masked Advantage 归一化 (排除空闲智能体)
                if active_masks.sum() > 1:
                    active_adv = advantages[active_masks == 1]
                    adv_mean = active_adv.mean()
                    adv_std = active_adv.std()
                    adv = ((advantages - adv_mean) / (adv_std + 1e-5)
                           ).clamp(-10.0, 10.0)
                elif advantages.numel() > 1:
                    adv = ((advantages - advantages.mean())
                           / (advantages.std() + 1e-5)).clamp(-10.0, 10.0)
                else:
                    adv = (advantages - advantages.mean()).clamp(-10.0, 10.0)

                # ---- Actor 更新 ----
                new_log_probs, entropy = compute_log_prob(
                    self.actor, obs, goals, actions)

                if torch.isnan(new_log_probs).any():
                    print("[WARNING] NaN in Actor output, skipping batch")
                    values = self.critic(global_states).squeeze(-1)
                    critic_loss = self._cal_value_loss(
                        values, old_values_norm, returns_norm, active_masks)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    total_critic_loss += critic_loss.item()
                    n_updates += 1
                    continue

                ratio = torch.exp(new_log_probs - old_log_probs)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                if self.target_kl is not None and approx_kl > self.target_kl:
                    kl_early_stopped = True
                    values = self.critic(global_states).squeeze(-1)
                    critic_loss = self._cal_value_loss(
                        values, old_values_norm, returns_norm, active_masks)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    total_critic_loss += critic_loss.item()
                    total_approx_kl += approx_kl
                    n_updates += 1
                    break

                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2)
                entropy_mean = entropy.mean()

                # Masked Actor Loss
                if active_masks.sum() > 0:
                    actor_loss = ((policy_loss * active_masks).sum()
                                  / active_masks.sum())
                else:
                    actor_loss = policy_loss.mean()

                actor_total = actor_loss - self.entropy_coef * entropy_mean

                self.actor_optimizer.zero_grad()
                actor_total.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                if any(torch.isnan(p).any() for p in self.actor.parameters()):
                    print("[WARNING] Actor weights NaN, rolling back")
                    self.actor.load_state_dict(actor_snapshot)
                    kl_early_stopped = True
                    break

                # ---- Critic 更新 (双重裁剪 + Huber) ----
                values = self.critic(global_states).squeeze(-1)
                critic_loss = self._cal_value_loss(
                    values, old_values_norm, returns_norm, active_masks)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

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
            'kl_early_stopped': kl_early_stopped,
        }

    def update_critic_only(self, buffer):
        """
        仅更新 Critic, 冻结 Actor (Critic warmup 阶段使用).
        """
        total_critic_loss = 0.0
        n_updates = 0

        n = buffer.ptr
        all_returns = buffer.returns[:n].reshape(-1)
        self._update_ret_rms(all_returns)

        for _epoch in range(self.ppo_epochs):
            for batch in buffer.get_batches(self.mini_batch_size):
                global_states = batch['global_states']
                returns = batch['returns']
                old_values = batch['old_values']
                active_masks = batch['active_masks']

                returns_norm = (returns - self._ret_rms_mean) / self._ret_std
                old_values_norm = (old_values - self._ret_rms_mean) / self._ret_std

                values = self.critic(global_states).squeeze(-1)
                critic_loss = self._cal_value_loss(
                    values, old_values_norm, returns_norm, active_masks)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_critic_loss += critic_loss.item()
                n_updates += 1

        n = max(1, n_updates)
        return {
            'actor_loss': 0.0,
            'critic_loss': total_critic_loss / n,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'kl_early_stopped': False,
        }
