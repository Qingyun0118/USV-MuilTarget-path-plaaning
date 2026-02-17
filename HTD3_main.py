"""
HRL-MG-TD3 + HGG 实现。
将 Yan 等人的分层强化学习架构 (HRL-MG) 应用于 USV 多目标导航任务，
并使用 HGG (Hindsight Goal Generation) 替代原始 HER 进行探索。

参考文献:
- Yan et al., "Hierarchical RL for handling sparse rewards in multi-goal navigation", AIR 2024
- Ren et al., "Exploration via Hindsight Goal Generation", NeurIPS 2019

架构:
  Selector (DQN) → Upper Actuator (TD3 + HGG) → Lower Actuator (TD3 + HGG) → USV Environment

HGG 核心思想:
  在每次迭代开始时，通过 Wasserstein Barycenter 二部图匹配，
  从历史轨迹中选出既容易短期达成、又能引导智能体长期接近真实目标的 hindsight 目标，
  用于替代原始目标进行探索。策略训练仍使用 HER 从 replay buffer 采样。
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime
from collections import deque
from scipy.optimize import linear_sum_assignment  # HGG 二部图匹配 (匈牙利算法)
from my_USV_Env import MyUsvEnv, X_MAX, Y_MAX

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# ========================= Replay Buffers ===================================
# ============================================================================

class ReplayBuffer:
    """Standard Replay Buffer for Selector (DQN) — 与原始实现相同，无需修改"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.LongTensor(np.array(action)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)


class TrajectoryBuffer:
    """
    轨迹级别缓冲区 — 替代原始 HER_ReplayBuffer
    
    同时支持:
    1. HGG 的轨迹级别访问 (用于二部图匹配)
    2. HER 风格的 transition 采样 (用于策略训练)
    
    与原始 HER_ReplayBuffer 的区别:
    - 存储结构更清晰: 每条轨迹包含 (初始位置, 目标, transitions)
    - 提供 get_all_trajectories() 供 HGG 匹配使用
    - HER 采样逻辑内置于 sample_with_her() 中
    
    每条 trajectory 的格式:
    {
        'init_pos': np.ndarray (2,),       # 轨迹起始时的归一化 USV 位置 (φ(s₀))
        'desired_goal': np.ndarray (2,),   # 该轨迹的期望目标
        'transitions': [                    # 时间步列表
            {
                'state': np.ndarray,        # 完整状态向量
                'action': np.ndarray,       # 动作
                'next_state': np.ndarray,   # 下一状态
                'achieved_pos': np.ndarray (2,),  # 该步达到的归一化位置 φ(s_t)
                'desired_goal': np.ndarray (2,),  # 期望目标 (相对或绝对)
            },
            ...
        ]
    }
    """
    def __init__(self, capacity):
        """
        Args:
            capacity: 最大存储轨迹数量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store_trajectory(self, trajectory):
        """
        存储一条完整轨迹
        
        Args:
            trajectory: dict, 包含 'init_pos', 'desired_goal', 'transitions'
        """
        self.buffer.append(trajectory)

    def get_all_trajectories(self):
        """返回所有轨迹供 HGG 匹配使用"""
        return list(self.buffer)

    def sample_with_her(self, batch_size, compute_reward_fn, k_future=4):
        """
        HER (future strategy) 风格采样 — 用于策略训练
        
        HGG 论文 Algorithm 1 Step 16: "Sample a minibatch b from replay buffer using HER"
        即 HGG 用于探索，HER 仍然用于训练时的经验回放。
        
        Args:
            batch_size: 批量大小
            compute_reward_fn: 奖励计算函数 f(achieved, desired) -> reward
            k_future: HER future strategy 的采样比例
        
        Returns:
            (state, action, reward, next_state, done, desired_goal) 的 Tensor 元组
        """
        state_batch, action_batch, reward_batch = [], [], []
        next_state_batch, done_batch, goal_batch = [], [], []

        num_trajectories = len(self.buffer)
        if num_trajectories == 0:
            return None

        # 采样 batch_size 个 transition
        traj_indices = np.random.randint(0, num_trajectories, batch_size)

        for traj_idx in traj_indices:
            traj = self.buffer[traj_idx]
            transitions = traj['transitions']
            ep_len = len(transitions)
            if ep_len == 0:
                continue

            # 随机选择一个时间步 t
            t = np.random.randint(0, ep_len)
            trans = transitions[t]

            # HER: 以概率 k/(k+1) 使用 future strategy 重标记目标
            use_her = (np.random.random() < k_future / (k_future + 1.0))

            if use_her and t < ep_len - 1:
                # Future strategy: 从 [t+1, ep_len) 中随机选一个未来时刻
                t_future = np.random.randint(t + 1, ep_len)
                # 使用未来时刻的 achieved_pos 作为新目标
                new_goal = transitions[t_future]['achieved_pos'].copy()
                # 重新计算奖励: 当前 achieved 与新目标的距离
                achieved = trans['achieved_pos']
                reward = compute_reward_fn(achieved, new_goal)
            else:
                # 使用原始目标
                new_goal = trans['desired_goal'].copy()
                achieved = trans['achieved_pos']
                reward = compute_reward_fn(achieved, new_goal)

            state_batch.append(trans['state'])
            action_batch.append(trans['action'])
            # [Fix] 碰撞惩罚加入HER训练奖励 (物理事件，与目标选择无关)
            if trans.get('crush', False):
                reward -= 10.0
            reward_batch.append(reward)
            next_state_batch.append(trans['next_state'])
            # [Fix] 碰撞或到达目标都是终止状态，必须设 done=1 防止错误 bootstrap
            is_terminal = trans.get('crush', False) or trans.get('done', False)
            done_batch.append(1.0 if is_terminal else 0.0)
            goal_batch.append(new_goal)

        if len(state_batch) == 0:
            return None

        return (
            torch.FloatTensor(np.array(state_batch)).to(device),
            torch.FloatTensor(np.array(action_batch)).to(device),
            torch.FloatTensor(np.array(reward_batch)).reshape(-1, 1).to(device),
            torch.FloatTensor(np.array(next_state_batch)).to(device),
            torch.FloatTensor(np.array(done_batch)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(goal_batch)).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# ========================= HGG Goal Generator ==============================
# ============================================================================

class HGG_GoalGenerator:
    """
    HGG (Hindsight Goal Generation) 目标生成器
    
    实现 Ren et al. NeurIPS 2019 Algorithm 1 的核心逻辑:
    1. 从目标分布 T* 采样 K 个任务 {(ŝ₀ⁱ, ĝⁱ)}
    2. 通过加权二部图匹配，从轨迹缓冲区 B 中找到 K 条不同轨迹
    3. 从匹配的轨迹中提取最优 hindsight 目标
    
    代价函数 (Eq. 9):
        w((ŝ₀, ĝ), τ) = c·‖φ(ŝ₀) - φ(s₀)‖₂ + min_t(‖ĝ - φ(sₜ)‖₂ - (1/L)·V^π(s₀, φ(sₜ)))
    
    其中:
    - c: 初始状态距离权重 (控制初始状态相似性的重要程度)
    - L: Lipschitz 常数 (控制值函数项的权重，L越大值函数影响越小)
    - V^π(s₀, φ(sₜ)): 策略 π 在初始状态 s₀ 下到达目标 φ(sₜ) 的值函数估计
    """
    def __init__(self, K=50, L=5.0, c=3.0, max_traj_for_matching=200,
                 max_trans_per_traj=10):
        """
        Args:
            K: 每次迭代生成的 hindsight 目标数量
            L: Lipschitz 常数 (论文建议 L=5.0)
            c: 初始状态距离权重 (论文建议 c=3.0)
            max_traj_for_matching: 二部图匹配时最多使用的轨迹数量 (控制计算量)
            max_trans_per_traj: 每条轨迹最多采样的 transition 数 (下采样, 控制计算量)
        """
        self.K = K
        self.L = L
        self.c = c
        self.max_traj_for_matching = max_traj_for_matching
        self.max_trans_per_traj = max_trans_per_traj

    def _subsample_transitions(self, transitions):
        """对轨迹的 transitions 进行均匀下采样，保留首尾"""
        T = len(transitions)
        if T <= self.max_trans_per_traj:
            return transitions
        # 均匀采样索引，始终包含首尾
        indices = np.linspace(0, T - 1, self.max_trans_per_traj, dtype=int)
        return [transitions[i] for i in indices]

    def generate_goals(self, trajectory_buffer, s0_batch, g_batch, value_fn):
        """
        HGG 核心: 生成 hindsight 目标 (批量化优化版本)
        
        对应 Algorithm 1 的 Step 5-6:
        - Step 5: 二部图匹配找到最优轨迹
        - Step 6: 从匹配轨迹中提取 hindsight 目标
        
        性能优化:
        - 将所有 value_fn 调用收集后一次性批量推理
        - 对每条轨迹的 transitions 进行下采样
        - 使用向量化 numpy 操作替代 Python 循环
        
        Args:
            trajectory_buffer: TrajectoryBuffer 实例 (轨迹缓冲区 B)
            s0_batch: np.ndarray (K, 2), 从 T* 采样的初始位置
            g_batch:  np.ndarray (K, 2), 从 T* 采样的目标位置
            value_fn: callable, 支持两种调用方式:
                      - value_fn(state, goal) -> float (单个)
                      - value_fn.estimate_value_batch(states, goals) -> np.ndarray (批量)
                      传入的是 TD3Agent.estimate_value，通过 agent 对象获取批量方法
        
        Returns:
            generated_goals: np.ndarray (K, 2), 生成的 hindsight 目标
            如果缓冲区为空或轨迹不足，返回原始目标 g_batch
        """
        trajectories = trajectory_buffer.get_all_trajectories()
        num_traj = len(trajectories)

        # 预热阶段: 轨迹不足时直接返回原始目标
        if num_traj < 2:
            return g_batch.copy()

        # 限制匹配的轨迹数量以控制计算量
        if num_traj > self.max_traj_for_matching:
            traj_indices = np.random.choice(num_traj, self.max_traj_for_matching, replace=False)
            trajectories_subset = [trajectories[i] for i in traj_indices]
        else:
            trajectories_subset = trajectories

        K = min(self.K, len(s0_batch))
        M = len(trajectories_subset)

        # ---- 预处理: 提取并下采样所有轨迹的 achieved_pos ----
        # 构建扁平化的 (state, goal) 对列表，用于批量 value_fn 调用
        traj_init_positions = []       # 每条轨迹的 init_pos
        traj_achieved_list = []        # 每条轨迹下采样后的 achieved_pos 列表
        all_vfn_states = []            # 扁平化: 所有需要估值的 state
        all_vfn_goals = []             # 扁平化: 所有需要估值的 goal
        traj_offsets = []              # 每条轨迹在扁平数组中的起始偏移
        traj_lengths = []              # 每条轨迹下采样后的 transition 数

        offset = 0
        for j, traj in enumerate(trajectories_subset):
            traj_init_pos = traj['init_pos']
            transitions = traj['transitions']
            if len(transitions) == 0:
                traj_init_positions.append(traj_init_pos)
                traj_achieved_list.append(np.empty((0, 2)))
                traj_offsets.append(offset)
                traj_lengths.append(0)
                continue

            # 下采样 transitions
            sampled_trans = self._subsample_transitions(transitions)
            achieved_arr = np.array([t['achieved_pos'] for t in sampled_trans])

            traj_init_positions.append(traj_init_pos)
            traj_achieved_list.append(achieved_arr)
            traj_offsets.append(offset)
            traj_lengths.append(len(sampled_trans))

            # 收集 value_fn 输入: (init_pos, achieved_pos) 对
            for achieved in achieved_arr:
                all_vfn_states.append(traj_init_pos)
                all_vfn_goals.append(achieved)

            offset += len(sampled_trans)

        # ---- 批量计算所有 value_fn ----
        total_pairs = len(all_vfn_states)
        if total_pairs > 0:
            all_vfn_states = np.array(all_vfn_states)
            all_vfn_goals = np.array(all_vfn_goals)

            # 尝试使用批量方法 (通过 value_fn 所属对象的 estimate_value_batch)
            try:
                agent = value_fn.__self__  # 获取绑定方法的对象 (TD3Agent)
                all_values = agent.estimate_value_batch(all_vfn_states, all_vfn_goals)
            except (AttributeError, Exception):
                # 回退到逐个调用 (不应该发生，但作为安全保障)
                all_values = np.zeros(total_pairs)
                for idx in range(total_pairs):
                    try:
                        all_values[idx] = value_fn(all_vfn_states[idx], all_vfn_goals[idx])
                    except Exception:
                        all_values[idx] = 0.0
        else:
            all_values = np.array([])

        # ---- Step 5: 构建代价矩阵 (向量化) ----
        cost_matrix = np.full((K, M), fill_value=1e10)

        # 预计算初始状态距离矩阵: init_dists[i, j] = c * ||s0_batch[i] - traj_init[j]||
        traj_init_arr = np.array(traj_init_positions)  # (M, 2)
        # 广播计算: (K, 1, 2) - (1, M, 2) -> (K, M, 2) -> norm -> (K, M)
        init_dists = self.c * np.linalg.norm(
            s0_batch[:K, np.newaxis, :] - traj_init_arr[np.newaxis, :, :], axis=2)

        for j in range(M):
            T_j = traj_lengths[j]
            if T_j == 0:
                continue

            achieved_j = traj_achieved_list[j]  # (T_j, 2)
            v_offset = traj_offsets[j]
            values_j = all_values[v_offset:v_offset + T_j]  # (T_j,)

            # 向量化计算: goal_dists[i, t] = ||g_batch[i] - achieved_j[t]||
            # g_batch[:K] shape: (K, 2), achieved_j shape: (T_j, 2)
            goal_dists = np.linalg.norm(
                g_batch[:K, np.newaxis, :] - achieved_j[np.newaxis, :, :], axis=2)  # (K, T_j)

            # cost[i, t] = goal_dist[i, t] - (1/L) * values_j[t]
            costs_per_t = goal_dists - (1.0 / self.L) * values_j[np.newaxis, :]  # (K, T_j)

            # min over t
            best_cost_per_task = np.min(costs_per_t, axis=1)  # (K,)

            cost_matrix[:, j] = init_dists[:, j] + best_cost_per_task

        # 使用匈牙利算法求解最小代价二部图匹配
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except Exception:
            return g_batch.copy()

        # ---- Step 6: 从匹配的轨迹中提取 hindsight 目标 (向量化) ----
        generated_goals = g_batch.copy()

        for idx in range(len(row_ind)):
            i = row_ind[idx]
            j = col_ind[idx]

            T_j = traj_lengths[j]
            if T_j == 0:
                continue

            achieved_j = traj_achieved_list[j]  # (T_j, 2)
            v_offset = traj_offsets[j]
            values_j = all_values[v_offset:v_offset + T_j]

            g_hat = g_batch[i]

            # 向量化: 找到最优 t*
            goal_dists = np.linalg.norm(achieved_j - g_hat, axis=1)  # (T_j,)
            costs = goal_dists - (1.0 / self.L) * values_j
            best_t = np.argmin(costs)

            generated_goals[i] = achieved_j[best_t].copy()

        return generated_goals

    def filter_unsafe_goals(self, generated_goals, original_goals, obstacles, scale):
        """
        [Fix] 安全过滤: 将落在障碍物安全区域内的 HGG 目标替换为原始目标
        
        HGG 生成的目标来自历史轨迹的 achieved_pos，可能落在障碍物附近或内部。
        如果直接使用这些目标，Lower agent 将试图导航到不可达位置并碰撞。
        
        Args:
            generated_goals: np.ndarray (K, 2), HGG 生成的归一化目标
            original_goals: np.ndarray (K, 2), 原始目标 (回退方案)
            obstacles: list, 环境障碍物列表
            scale: np.ndarray (2,), 归一化→绝对坐标的缩放因子
        
        Returns:
            filtered_goals: np.ndarray (K, 2), 过滤后的安全目标
        """
        filtered = generated_goals.copy()
        for i in range(len(filtered)):
            goal_abs = filtered[i] * scale
            # 边界检查
            if (goal_abs[0] < 2.0 or goal_abs[0] > scale[0] - 2.0 or
                goal_abs[1] < 2.0 or goal_abs[1] > scale[1] - 2.0):
                filtered[i] = original_goals[i]
                continue
            # 障碍物碰撞检查
            for obs in obstacles:
                if obs.is_point_in_safety_zone(goal_abs):
                    filtered[i] = original_goals[i]
                    break
        return filtered


# ============================================================================
# ========================= Neural Networks ==================================
# ============================================================================

class SelectorQNet(nn.Module):
    """DQN for Selector: State -> Q-values for each target — 与原始实现相同"""
    def __init__(self, state_dim, num_actions):
        super(SelectorQNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    """TD3 Actor: (State + Goal) -> Action — 与原始实现相同"""
    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """TD3 Critic: (State + Goal + Action) -> Q-value — 与原始实现相同"""
    def __init__(self, state_dim, goal_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 128)
        self.l6 = nn.Linear(128, 1)

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
        q1 = self.l3(q1)
        return q1


# ============================================================================
# ========================= Agents ===========================================
# ============================================================================

class SelectorAgent:
    """Selector Agent (DQN) — 与原始实现相同，不使用 HGG"""
    def __init__(self, state_dim, num_targets, lr=1e-4, gamma=0.99, epsilon=0.3):
        self.q_net = SelectorQNet(state_dim, num_targets).to(device)
        self.target_q_net = SelectorQNet(state_dim, num_targets).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_targets = num_targets

    def select_action(self, state, valid_mask=None, eval=False):
        """
        状态：numpy 数组
        有效掩码：布尔数组（如果目标可用/未访问则为 True）
        """
        if not eval and random.random() < self.epsilon:
            if valid_mask is not None:
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    return np.random.choice(valid_indices)
            return np.random.randint(0, self.num_targets)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state_t).cpu().numpy().flatten()

        if valid_mask is not None:
            q_values[~valid_mask.astype(bool)] = -1e9

        return np.argmax(q_values)

    def train(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_q = self.target_q_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (1 - done) * self.gamma * next_q

        current_q = self.q_net(state).gather(1, action)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self, tau=0.005):
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class TD3Agent:
    """
    TD3 Agent — 在原始实现基础上新增 estimate_value() 方法供 HGG 使用
    
    新增方法:
    - estimate_value(state, goal): 估计 V^π(s, g) ≈ Q(s, g, π(s, g))
    - train_with_her_buffer(): 从 TrajectoryBuffer 的 HER 采样中训练
    """
    def __init__(self, state_dim, goal_dim, action_dim, max_action,
                 lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, tau=0.05,
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

        # 保存维度信息供 estimate_value 使用
        self.state_dim = state_dim
        self.goal_dim = goal_dim

    def select_action(self, state, goal, noise=0.0):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device)

        with torch.no_grad():
            action = self.actor(state_t, goal_t).cpu().numpy().flatten()

        if noise > 0:
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(
                -self.max_action, self.max_action)

        return action

    def estimate_value(self, state, goal):
        """
        [HGG 新增] 估计值函数 V^π(s, g)
        
        用于 HGG 代价函数 (Eq. 9) 中的 V^π(s₀, φ(sₜ)) 项。
        在 TD3 框架中: V^π(s, g) ≈ Q(s, g, π(s, g))
        
        Args:
            state: np.ndarray, 状态向量 (可以是位置向量，会被零填充到 state_dim)
            goal:  np.ndarray, 目标向量 (可以是位置向量，会被零填充到 goal_dim)
        
        Returns:
            float: 值函数估计值
        """
        # 将输入填充到正确的维度
        state_padded = np.zeros(self.state_dim, dtype=np.float32)
        state_padded[:min(len(state), self.state_dim)] = state[:self.state_dim]

        goal_padded = np.zeros(self.goal_dim, dtype=np.float32)
        goal_padded[:min(len(goal), self.goal_dim)] = goal[:self.goal_dim]

        state_t = torch.FloatTensor(state_padded).unsqueeze(0).to(device)
        goal_t = torch.FloatTensor(goal_padded).unsqueeze(0).to(device)

        with torch.no_grad():
            action = self.actor(state_t, goal_t)
            q1, _ = self.critic(state_t, goal_t, action)
            return q1.item()

    def estimate_value_batch(self, states, goals):
        """
        [HGG 性能优化] 批量估计值函数 V^π(s, g)
        
        将所有 (state, goal) 对一次性送入网络推理，避免逐个调用的巨大开销。
        单次调用 estimate_value 需要 1 次 GPU 推理，
        而 estimate_value_batch(N) 也只需要 1 次 GPU 推理（或少量分批）。
        
        Args:
            states: np.ndarray (N, state_dim 或更小), 状态向量批量
            goals:  np.ndarray (N, goal_dim 或更小), 目标向量批量
        
        Returns:
            np.ndarray (N,): 值函数估计值数组
        """
        N = len(states)
        if N == 0:
            return np.array([])

        # 批量填充到正确维度
        states_padded = np.zeros((N, self.state_dim), dtype=np.float32)
        goals_padded = np.zeros((N, self.goal_dim), dtype=np.float32)

        s_copy_dim = min(states.shape[1], self.state_dim) if states.ndim > 1 else min(len(states), self.state_dim)
        g_copy_dim = min(goals.shape[1], self.goal_dim) if goals.ndim > 1 else min(len(goals), self.goal_dim)

        if states.ndim == 1:
            # 单个 state 重复 N 次的情况
            states_padded[:, :s_copy_dim] = states[:s_copy_dim]
        else:
            states_padded[:, :s_copy_dim] = states[:, :s_copy_dim]

        if goals.ndim == 1:
            goals_padded[:, :g_copy_dim] = goals[:g_copy_dim]
        else:
            goals_padded[:, :g_copy_dim] = goals[:, :g_copy_dim]

        # 分批推理以避免 GPU 内存溢出
        BATCH_LIMIT = 4096
        all_values = []

        with torch.no_grad():
            for start in range(0, N, BATCH_LIMIT):
                end = min(start + BATCH_LIMIT, N)
                state_t = torch.FloatTensor(states_padded[start:end]).to(device)
                goal_t = torch.FloatTensor(goals_padded[start:end]).to(device)
                action = self.actor(state_t, goal_t)
                q1, _ = self.critic(state_t, goal_t, action)
                all_values.append(q1.squeeze(-1).cpu().numpy())

        return np.concatenate(all_values)

    def train_from_buffer(self, trajectory_buffer, batch_size=100):
        """
        [HGG 修改] 从 TrajectoryBuffer 使用 HER 采样进行训练
        
        对应 HGG Algorithm 1 Step 16-17:
        "Sample a minibatch b from replay buffer using HER"
        "Perform one step on value and policy update"
        
        Args:
            trajectory_buffer: TrajectoryBuffer 实例
            batch_size: 批量大小
        
        Returns:
            (critic_loss, actor_loss) 元组
        """
        if len(trajectory_buffer) < 2:
            return 0, 0

        # 使用 HER 采样
        sample = trajectory_buffer.sample_with_her(
            batch_size, compute_reward, k_future=4
        )
        if sample is None:
            return 0, 0

        state, action, reward, next_state, done, desired_goal = sample

        self.total_it += 1

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state, desired_goal) + noise).clamp(
                -self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, desired_goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, desired_goal, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss = 0
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(
                state, desired_goal, self.actor(state, desired_goal)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss if isinstance(actor_loss, int) else actor_loss.item()


# ============================================================================
# ========================= Helper Functions =================================
# ============================================================================

def get_selector_state(obs_dict):
    """Extract state for Selector: USV pos info + Targets info"""
    return np.concatenate([obs_dict['o_usv'][0:2], obs_dict['targets']])


def get_upper_state(obs_dict):
    """Extract state for Upper Actuator: USV pos info + Lidar"""
    return np.concatenate([obs_dict['o_usv'][0:2], obs_dict['o_detector']])


def get_lower_state(obs_dict):
    """Extract state for Lower Actuator: USV full state + Lidar + Current"""
    return np.concatenate([obs_dict['o_usv'], obs_dict['o_detector'], obs_dict['o_current']])


def get_target_pos(obs_dict, target_idx, num_targets):
    """Get normalized position of the selected target"""
    base_idx = target_idx * 3
    tx = obs_dict['targets'][base_idx]
    ty = obs_dict['targets'][base_idx + 1]
    return np.array([tx, ty])


def get_usv_pos(obs_dict):
    """Get normalized USV position"""
    return obs_dict['o_usv'][0:2]


def compute_reward(achieved_goal, desired_goal, info=None):
    """
    计算奖励函数 (支持 numpy 和 torch)
    
    用于:
    1. HER 采样时重新计算奖励
    2. 训练循环中的奖励计算
    
    Args:
        achieved_goal: 实际达到的位置
        desired_goal: 期望的目标位置
    
    Returns:
        奖励值 (距离越近奖励越大，最大为0)
    """
    if isinstance(achieved_goal, torch.Tensor):
        dist = torch.norm(achieved_goal - desired_goal, dim=-1, keepdim=True)
        return -1.0 * dist
    else:
        achieved = np.array(achieved_goal)
        desired = np.array(desired_goal)
        dist = np.linalg.norm(achieved - desired)
        return -1.0 * dist


def check_subgoal_collision(usv_pos, sub_goal_rel, obstacles, margin=2.0):
    """检查生成的子目标是否落在障碍物膨胀区域内"""
    sub_goal_global = usv_pos + sub_goal_rel
    for obs in obstacles:
        if obs.is_point_in_safety_zone(sub_goal_global):
            return True
    return False


# ============================================================================
# ========================= Main HRL + HGG Training Loop ====================
# ============================================================================

def train_hrl(args):
    """
    HRL-MG + HGG 训练主循环
    
    与原始 HER 版本的核心区别:
    1. 使用 TrajectoryBuffer 替代 HER_ReplayBuffer
    2. 在每次 Selector 选择目标后，Upper 层使用 HGG 生成 hindsight 全局目标进行探索
    3. 在每次 Upper 生成子目标后，Lower 层使用 HGG 生成 hindsight 子目标进行探索
    4. 策略训练仍使用 HER 从 TrajectoryBuffer 采样 (HGG Algorithm 1 Step 16)
    """
    # 1. 环境初始化
    env = MyUsvEnv(
        num_targets=args.num_targets,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        record_flag=False,
        usv_length=args.usv_length,
        usv_width=args.usv_width,
        target_radius=args.target_radius,
        obstacle_radius_range=(args.obstacle_radius_min, args.obstacle_radius_max),
        use_random_shapes=args.use_random_shapes
    )

    # 获取维度
    obs = env.reset()
    selector_state_dim = get_selector_state(obs).shape[0]
    upper_state_dim = get_upper_state(obs).shape[0]
    lower_state_dim = get_lower_state(obs).shape[0]

    # 动作与目标维度
    selector_action_dim = args.num_targets
    upper_action_dim = 2   # (dx, dy) 子目标相对偏移
    lower_action_dim = 2   # (throttle, rudder)
    upper_goal_dim = 2     # 相对目标 (到全局目标的向量)
    lower_goal_dim = 2     # 相对子目标 (到子目标的向量)

    # 2. 智能体初始化
    selector = SelectorAgent(selector_state_dim, selector_action_dim,
                             epsilon=args.epsilon_start)
    upper_agent = TD3Agent(upper_state_dim, upper_goal_dim, upper_action_dim,
                           max_action=args.subgoal_range,
                           lr_actor=1e-4, lr_critic=1e-3)
    lower_agent = TD3Agent(lower_state_dim, lower_goal_dim, lower_action_dim,
                           max_action=1.0,
                           lr_actor=1e-4, lr_critic=1e-3)

    # 3. 缓冲区设置 — 使用 TrajectoryBuffer 替代 HER_ReplayBuffer
    selector_buffer = ReplayBuffer(args.selector_buffer_size)
    upper_buffer = TrajectoryBuffer(capacity=5000)   # Upper 层轨迹缓冲区
    lower_buffer = TrajectoryBuffer(capacity=10000)  # Lower 层轨迹缓冲区

    # 4. HGG 目标生成器 — 双层 HGG (批量化优化版本)
    hgg_upper = HGG_GoalGenerator(
        K=args.hgg_K, L=args.hgg_L, c=args.hgg_c,
        max_traj_for_matching=args.hgg_max_traj,
        max_trans_per_traj=args.hgg_max_trans
    )
    hgg_lower = HGG_GoalGenerator(
        K=args.hgg_K, L=args.hgg_L, c=args.hgg_c,
        max_traj_for_matching=args.hgg_max_traj,
        max_trans_per_traj=args.hgg_max_trans
    )

    writer = SummaryWriter(
        f'htd3_output/runs/HRL_HGG_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')

    global_step = 0
    SCALE = np.array([X_MAX, Y_MAX])  # 坐标反归一化因子 [100, 100]

    best_reward = -float('inf')

    for episode in range(args.max_episodes):
        # 课程学习: 逐步增加环境多样性
        if episode < args.curriculum_episodes:
            # Phase 1: 固定环境，专注学习基本导航
            obs_dict = env.reset(seed=args.seed)
        elif episode < args.curriculum_episodes * 2:
            # Phase 2: 5个环境轮换，学习泛化
            obs_dict = env.reset(seed=args.seed + (episode % 5))
        else:
            # Phase 3: 每个episode新环境，完全泛化
            obs_dict = env.reset(seed=args.seed + episode)
        usv_pos = get_usv_pos(obs_dict)  # 归一化坐标 [0, 1]

        # [Fix] 自适应探索噪声: 预训练阶段更大噪声加速 Lower 学习
        if episode < args.lower_pretrain_episodes:
            current_noise = max(args.exploration_noise, 0.4)
        elif episode < args.hgg_warmup:
            current_noise = args.exploration_noise
        else:
            # HGG 阶段适当降低噪声，让 HGG 引导探索方向
            current_noise = args.exploration_noise * 0.8

        target_status = obs_dict['targets'][2::3]  # 0: unvisited
        valid_mask = (target_status == 0)

        # 记录累积奖励
        episode_lower_reward = 0
        episode_upper_reward = 0
        episode_selector_reward = 0

        done = False
        attempts = 0
        crush = False

        # ==================== Selector Loop ====================
        while not (done or attempts >= args.num_targets or crush):
            attempts += 1
            selector_state = get_selector_state(obs_dict)
            selector_start_pos = usv_pos
            selector_start_state = selector_state

            # Selector 选择一个全局目标
            target_idx = selector.select_action(selector_state, valid_mask)
            g_des_global_norm = get_target_pos(obs_dict, target_idx, args.num_targets)

            print(f'EP:{episode} | Selector Target: {target_idx} | '
                  f'Pos: {g_des_global_norm * 100}')

            selector_step_reward = 0
            selector_done = False

            # ---- HGG Upper: 生成 hindsight 全局目标 ----
            # HGG Algorithm 1 Step 4-6
            # [Fix] 增加 buffer 最小数量要求，确保有足够轨迹进行有意义的匹配
            hgg_upper_active = (episode >= args.hgg_warmup and
                                len(upper_buffer) >= args.min_buffer_for_hgg)
            if hgg_upper_active:
                # Step 4: 从 T*_upper 采样 K 个任务
                s0_upper_batch, g_upper_batch = env.sample_tasks_batch(
                    args.hgg_K, target_idx=target_idx)

                # Step 5-6: 二部图匹配 + 提取 hindsight 目标
                hgg_upper_goals = hgg_upper.generate_goals(
                    upper_buffer, s0_upper_batch, g_upper_batch,
                    value_fn=upper_agent.estimate_value
                )
                # [Fix] 安全过滤: 排除落在障碍物内的目标
                hgg_upper_goals = hgg_upper.filter_unsafe_goals(
                    hgg_upper_goals, g_upper_batch, env.obstacles, SCALE)
                # 使用第一个生成的 HGG 目标替代原始全局目标
                g_des_global_norm_hgg = hgg_upper_goals[0]
            else:
                # 预热阶段: 使用原始目标
                g_des_global_norm_hgg = g_des_global_norm.copy()

            # 临时存储 Upper 轨迹
            upper_trajectory_transitions = []

            # ==================== Upper Agent Loop ====================
            u_step = 0
            while not (done or selector_done or crush or u_step >= args.upper_steps):
                u_step += 1
                upper_state = get_upper_state(obs_dict)
                usv_pos_start_upper = get_usv_pos(obs_dict)

                # Upper Goal: HGG 目标 - 当前位置 (相对向量)
                # [HGG 修改] 使用 HGG 生成的目标替代原始目标
                g_des_rel = g_des_global_norm_hgg - usv_pos_start_upper

                # Upper Action: 生成子目标 (相对偏移)
                g_sub_rel = upper_agent.select_action(
                    upper_state, g_des_rel, noise=current_noise)

                # 计算子目标绝对位置 (归一化)
                g_sub_global_norm = usv_pos_start_upper + g_sub_rel
                g_sub_global_norm = np.clip(g_sub_global_norm, 0, 1)

                # ---- HGG Lower: 生成 hindsight 子目标 ----
                # [Fix] 增加 buffer 最小数量要求
                hgg_lower_active = (episode >= args.hgg_warmup and
                                    len(lower_buffer) >= args.min_buffer_for_hgg)
                if hgg_lower_active:
                    # 为 Lower 层生成 HGG 目标
                    # T*_lower: (当前USV位置, Upper给出的子目标)
                    s0_lower_batch = np.tile(usv_pos_start_upper, (args.hgg_K, 1))
                    g_lower_batch = np.tile(g_sub_global_norm, (args.hgg_K, 1))

                    hgg_lower_goals = hgg_lower.generate_goals(
                        lower_buffer, s0_lower_batch, g_lower_batch,
                        value_fn=lower_agent.estimate_value
                    )
                    # [Fix] 安全过滤
                    hgg_lower_goals = hgg_lower.filter_unsafe_goals(
                        hgg_lower_goals, g_lower_batch, env.obstacles, SCALE)
                    g_sub_global_norm_hgg = hgg_lower_goals[0]
                else:
                    g_sub_global_norm_hgg = g_sub_global_norm.copy()

                # 临时存储 Lower 轨迹
                lower_trajectory_transitions = []
                sub_done = False

                # ==================== Lower Agent Loop ====================
                l_step = 0
                while not (done or selector_done or sub_done or crush
                           or l_step >= args.lower_steps):
                    l_step += 1
                    lower_state = get_lower_state(obs_dict)
                    curr_usv_pos = get_usv_pos(obs_dict)

                    # Lower Goal: HGG 子目标 - 当前位置
                    # [HGG 修改] 使用 HGG 生成的子目标
                    curr_g_sub_rel = g_sub_global_norm_hgg - curr_usv_pos

                    # Lower Action: [Throttle, Rudder]
                    lower_action = lower_agent.select_action(
                        lower_state, curr_g_sub_rel, noise=current_noise)

                    # 将坐标转为绝对坐标传给环境
                    g_sub_global_abs = g_sub_global_norm_hgg * SCALE
                    next_obs_dict, sub_done, crush, truncated, info = env.step(
                        lower_action, g_sub_global_abs)

                    next_usv_pos = get_usv_pos(next_obs_dict)

                    # 计算实际位移 (Achieved Goal)
                    actual_move = next_usv_pos - curr_usv_pos

                    # ---- Lower Reward (用于日志监控，训练奖励由HER采样决定) ----
                    # 1. 接近奖励: 靠近子目标则为正，远离则为负
                    prev_dist = np.linalg.norm(curr_g_sub_rel)
                    next_dist = np.linalg.norm(g_sub_global_norm_hgg - next_usv_pos)
                    approach_reward = (prev_dist - next_dist) * 10.0

                    # 2. 障碍物接近惩罚 (传感器最小读数 < 0.3)
                    min_sensor = float(min(next_obs_dict['o_detector']))
                    proximity_penalty = -max(0.0, 0.3 - min_sensor) * 5.0

                    # 3. 终止事件
                    if crush:
                        r_low_dense = -10.0
                    elif sub_done:
                        r_low_dense = approach_reward + 5.0
                    else:
                        r_low_dense = approach_reward + proximity_penalty - 0.01

                    episode_lower_reward += r_low_dense

                    # 存储到 Lower 轨迹 (用于 TrajectoryBuffer)
                    lower_trajectory_transitions.append({
                        'state': lower_state,
                        'action': lower_action,
                        'next_state': get_lower_state(next_obs_dict),
                        'achieved_pos': next_usv_pos.copy(),
                        'desired_goal': curr_g_sub_rel.copy(),
                        'crush': crush,  # [Fix] 存储碰撞状态
                        'done': sub_done
                    })

                    obs_dict = next_obs_dict
                    usv_pos = next_usv_pos
                    global_step += 1

                    # >>>>>> 动态目标检测 (DGT) — 与原始实现相同 <<<<<<
                    reached_indices = info.get('reached_targets', [])
                    if len(reached_indices) > 0:
                        next_selector_state = get_selector_state(next_obs_dict)
                        current_valid_mask = (next_obs_dict['targets'][2::3] == 0)
                        is_all_done = not np.any(current_valid_mask)

                        for r_idx in reached_indices:
                            if r_idx == target_idx:
                                print(f"  >>> SUCCESS: Reached Intended Target {target_idx}")
                                selector_step_reward = 1.0
                                selector_done = True
                            else:
                                print(f"  >>> DGT HIT: Reached Target {r_idx} "
                                      f"(aiming for {target_idx})")
                                selector_buffer.push(
                                    selector_start_state,
                                    r_idx,   # Fake Action
                                    1.0,     # Fake Reward
                                    next_selector_state,
                                    is_all_done
                                )
                                selector_done = True

                # --- End of Lower Loop ---

                # 1. 存储 Lower 轨迹到 TrajectoryBuffer
                if len(lower_trajectory_transitions) > 0:
                    lower_buffer.store_trajectory({
                        'init_pos': usv_pos_start_upper.copy(),
                        'desired_goal': g_sub_global_norm_hgg.copy(),
                        'transitions': lower_trajectory_transitions
                    })

                # 2. 训练 Lower Agent (HGG Algorithm 1 Step 16-17)
                # [Fix] 预训练阶段增加训练迭代次数，加速 Lower 学习基本避障
                if len(lower_buffer) > args.actuator_batch_size // 10:
                    n_lower_iters = (args.lower_train_iters
                                     if episode < args.lower_pretrain_episodes
                                     else min(l_step, 10))
                    for _ in range(n_lower_iters):
                        lower_agent.train_from_buffer(
                            lower_buffer, args.actuator_batch_size)

                # 3. 记录 Upper Transition
                next_upper_state = get_upper_state(obs_dict)
                usv_pos_end_upper = get_usv_pos(obs_dict)

                # Upper Achieved Goal: 这一阶段的总位移
                actual_upper_displacement = usv_pos_end_upper - usv_pos_start_upper

                # ---- Upper Reward (用于日志监控) ----
                prev_dist_upper = np.linalg.norm(g_des_rel)
                next_dist_upper = np.linalg.norm(g_des_global_norm_hgg - usv_pos_end_upper)
                approach_upper = (prev_dist_upper - next_dist_upper) * 10.0

                if crush:
                    r_up_dense = -10.0
                elif done or selector_done:
                    r_up_dense = approach_upper + 10.0
                else:
                    r_up_dense = approach_upper - 0.01

                episode_upper_reward += r_up_dense

                upper_trajectory_transitions.append({
                    'state': upper_state,
                    'action': g_sub_rel,
                    'next_state': next_upper_state,
                    'achieved_pos': usv_pos_end_upper.copy(),
                    'desired_goal': g_des_rel.copy(),
                    'crush': crush,
                    'done': done or selector_done
                })

            # --- End of Upper Loop ---

            # 1. 存储 Upper 轨迹到 TrajectoryBuffer
            if len(upper_trajectory_transitions) > 0:
                upper_buffer.store_trajectory({
                    'init_pos': selector_start_pos.copy(),
                    'desired_goal': g_des_global_norm_hgg.copy(),
                    'transitions': upper_trajectory_transitions
                })

            # 2. 训练 Upper Agent (HGG Algorithm 1 Step 16-17)
            # [Fix] 预训练阶段适度增加 Upper 训练强度
            if len(upper_buffer) > args.actuator_batch_size // 10:
                n_upper_iters = (min(u_step, 15)
                                 if episode < args.lower_pretrain_episodes
                                 else min(u_step, 10))
                for _ in range(n_upper_iters):
                    upper_agent.train_from_buffer(
                        upper_buffer, args.actuator_batch_size)

            # 3. Selector Update
            target_status = obs_dict['targets'][2::3]
            valid_mask = (target_status == 0)
            done = not np.any(valid_mask)

            if done:
                selector_step_reward = 2.0  # 完成所有任务的大奖励

            final_selector_reward = selector_step_reward if selector_done else -0.1

            selector_buffer.push(
                selector_start_state, target_idx, final_selector_reward,
                get_selector_state(obs_dict), done
            )

            if len(selector_buffer) > args.selector_batch_size:
                selector.train(selector_buffer, args.selector_batch_size)

            episode_selector_reward += final_selector_reward

        # --- End of Selector Loop ---

        # 衰减 Epsilon
        selector.epsilon = max(args.epsilon_end,
                               selector.epsilon * args.epsilon_decay)

        # [Fix] Lower 预训练阶段结束时保存检查点
        if episode == args.lower_pretrain_episodes - 1:
            ckpt_dir = os.path.join(args.output_dir, "checkpoints", "lower_pretrain")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(lower_agent.actor.state_dict(), os.path.join(ckpt_dir, "lower_actor.pth"))
            torch.save(lower_agent.critic.state_dict(), os.path.join(ckpt_dir, "lower_critic.pth"))
            torch.save(upper_agent.actor.state_dict(), os.path.join(ckpt_dir, "upper_actor.pth"))
            torch.save(upper_agent.critic.state_dict(), os.path.join(ckpt_dir, "upper_critic.pth"))
            print(f"  [Checkpoint] Lower pretrain complete at episode {episode+1}")

        targets_reached = int(np.sum(env.target_status))
        # [Fix] 显示训练阶段和 buffer 大小
        phase = ("PreTrain" if episode < args.lower_pretrain_episodes
                 else ("Warmup" if episode < args.hgg_warmup else "HGG"))
        print(f"EP {episode} [{phase}] | "
              f"R_low: {episode_lower_reward:.1f} | "
              f"R_up: {episode_upper_reward:.1f} | "
              f"R_sel: {episode_selector_reward:.1f} | "
              f"Targets: {targets_reached}/{args.num_targets} | "
              f"Collision: {crush} | "
              f"Steps: {global_step} | "
              f"Eps: {selector.epsilon:.3f} | "
              f"Buf: U={len(upper_buffer)},L={len(lower_buffer)}")

        writer.add_scalar('Reward/Lower', episode_lower_reward, episode)
        writer.add_scalar('Reward/Upper', episode_upper_reward, episode)
        writer.add_scalar('Reward/Selector', episode_selector_reward, episode)
        writer.add_scalar('Metrics/Collision', float(crush), episode)
        writer.add_scalar('Metrics/TargetsReached', targets_reached, episode)
        writer.add_scalar('Metrics/Epsilon', selector.epsilon, episode)
        writer.add_scalar('Buffer/Upper', len(upper_buffer), episode)
        writer.add_scalar('Buffer/Lower', len(lower_buffer), episode)
        writer.add_scalar('Training/Noise', current_noise, episode)
        writer.add_scalar('Training/Phase',
                          0 if episode < args.lower_pretrain_episodes
                          else (1 if episode < args.hgg_warmup else 2), episode)

        # 定期保存检查点
        if (episode + 1) % 200 == 0:
            ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"ep_{episode+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(selector.q_net.state_dict(), os.path.join(ckpt_dir, "selector.pth"))
            torch.save(upper_agent.actor.state_dict(), os.path.join(ckpt_dir, "upper_actor.pth"))
            torch.save(lower_agent.actor.state_dict(), os.path.join(ckpt_dir, "lower_actor.pth"))
            # [Fix] 保存 critic 状态 — HGG 依赖值函数
            torch.save(upper_agent.critic.state_dict(), os.path.join(ckpt_dir, "upper_critic.pth"))
            torch.save(lower_agent.critic.state_dict(), os.path.join(ckpt_dir, "lower_critic.pth"))
            print(f"  [Checkpoint] Saved at episode {episode+1}")

        # 保存最佳模型
        if episode_lower_reward > best_reward and not crush:
            best_reward = episode_lower_reward
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(selector.q_net.state_dict(), os.path.join(args.output_dir, "best_selector.pth"))
            torch.save(upper_agent.actor.state_dict(), os.path.join(args.output_dir, "best_upper_actor.pth"))
            torch.save(lower_agent.actor.state_dict(), os.path.join(args.output_dir, "best_lower_actor.pth"))
            # [Fix] 保存 critic 以保留值函数状态 (HGG 依赖值函数)
            torch.save(upper_agent.critic.state_dict(), os.path.join(args.output_dir, "best_upper_critic.pth"))
            torch.save(lower_agent.critic.state_dict(), os.path.join(args.output_dir, "best_lower_critic.pth"))

    # Save Models
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(selector.q_net.state_dict(),
               os.path.join(args.output_dir, "selector.pth"))
    torch.save(upper_agent.actor.state_dict(),
               os.path.join(args.output_dir, "upper_actor.pth"))
    torch.save(lower_agent.actor.state_dict(),
               os.path.join(args.output_dir, "lower_actor.pth"))
    # [Fix] 保存 critic 状态
    torch.save(upper_agent.critic.state_dict(),
               os.path.join(args.output_dir, "upper_critic.pth"))
    torch.save(lower_agent.critic.state_dict(),
               os.path.join(args.output_dir, "lower_critic.pth"))
    print("Training Complete.")


# ============================================================================
# ========================= Evaluation =======================================
# ============================================================================

def evaluate_hrl(args):
    """
    评估函数 — 评估时不使用 HGG（直接使用训练好的策略）
    与原始实现逻辑基本相同
    """
    print("Starting Evaluation...")
    # 1. Environment Setup
    env = MyUsvEnv(
        num_targets=args.num_targets,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        record_flag=False,
        usv_length=args.usv_length,
        usv_width=args.usv_width,
        target_radius=args.target_radius,
        obstacle_radius_range=(args.obstacle_radius_min, args.obstacle_radius_max),
        use_random_shapes=args.use_random_shapes
    )

    # Dimensions
    obs = env.reset()
    selector_state_dim = get_selector_state(obs).shape[0]
    upper_state_dim = get_upper_state(obs).shape[0]
    lower_state_dim = get_lower_state(obs).shape[0]

    # Action Dims
    selector_action_dim = args.num_targets
    upper_action_dim = 2
    lower_action_dim = 2

    # Goal Dims
    upper_goal_dim = 2
    lower_goal_dim = 2

    # 2. Agents Setup & Load Models
    selector = SelectorAgent(selector_state_dim, selector_action_dim)
    upper_agent = TD3Agent(upper_state_dim, upper_goal_dim, upper_action_dim,
                           max_action=args.subgoal_range)
    lower_agent = TD3Agent(lower_state_dim, lower_goal_dim, lower_action_dim,
                           max_action=1.0)

    model_dir = args.output_dir
    if os.path.exists(os.path.join(model_dir, "selector.pth")):
        selector.q_net.load_state_dict(
            torch.load(os.path.join(model_dir, "selector.pth"), map_location=device))
        upper_agent.actor.load_state_dict(
            torch.load(os.path.join(model_dir, "upper_actor.pth"), map_location=device))
        lower_agent.actor.load_state_dict(
            torch.load(os.path.join(model_dir, "lower_actor.pth"), map_location=device))
        print("Models loaded successfully.")
    else:
        print("No models found, running with random weights.")

    eval_dir = os.path.join(args.output_dir, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)

    SCALE = np.array([X_MAX, Y_MAX])

    for episode in range(args.eval_episodes):
        obs_dict = env.reset()
        done = False
        truncated = False
        crush = False
        episode_lower_reward = 0
        attempts = 0
        csv_file = os.path.join(eval_dir, f"hrl_path_episode_{episode + 1}.csv")
        if os.path.exists(csv_file):
            os.remove(csv_file)

        while not (done or attempts >= args.num_targets or crush):
            attempts += 1
            selector_state = get_selector_state(obs_dict)
            target_status = obs_dict['targets'][2::3]
            valid_mask = (target_status == 0)

            target_idx = selector.select_action(selector_state, valid_mask, eval=True)
            g_des_global = get_target_pos(obs_dict, target_idx, args.num_targets)

            print(f"Episode {episode}: Selector chose Target {target_idx}")

            step_count = 0
            total_env_steps = 0
            max_eval_steps = args.upper_steps * args.lower_steps * args.num_targets * 2  # 评估最大步数安全限制
            while not (done or crush or truncated or total_env_steps >= max_eval_steps):
                upper_state = get_upper_state(obs_dict)
                usv_pos = get_usv_pos(obs_dict)
                g_des_rel = g_des_global - usv_pos

                # Upper Action
                g_sub_rel = upper_agent.select_action(upper_state, g_des_rel, noise=0.0)

                # Lower Loop
                g_sub_global = usv_pos + g_sub_rel
                sub_done = False
                l_step = 0
                while not (done or sub_done or crush or truncated
                           or l_step >= args.lower_steps
                           or total_env_steps >= max_eval_steps):
                    l_step += 1
                    total_env_steps += 1
                    lower_state = get_lower_state(obs_dict)
                    curr_usv_pos = get_usv_pos(obs_dict)
                    curr_g_sub_rel = g_sub_global - curr_usv_pos

                    lower_action = lower_agent.select_action(
                        lower_state, curr_g_sub_rel, noise=0.0)

                    g_sub_global_abs = g_sub_global * SCALE
                    next_obs_dict, sub_done, crush, truncated, info = env.step(
                        lower_action, g_sub_global_abs)

                    # Record Path
                    env.record_points(np.array([env.usv.x, env.usv.y]), csv_file)

                    obs_dict = next_obs_dict
                    episode_lower_reward += compute_reward(
                        get_usv_pos(obs_dict), g_sub_global)

                    # Check if target reached
                    is_reached = (obs_dict['targets'][target_idx * 3 + 2] == 1.0)

                    if is_reached:
                        print(f"  Target {target_idx} reached!")
                        target_status = obs_dict['targets'][2::3]
                        valid_mask = (target_status == 0)
                        if np.any(valid_mask):
                            selector_state = get_selector_state(obs_dict)
                            target_idx = selector.select_action(
                                selector_state, valid_mask, eval=True)
                            g_des_global = get_target_pos(
                                obs_dict, target_idx, args.num_targets)
                            print(f"  Selector chose new Target {target_idx}")
                            break
                        else:
                            done = True
                            break

                step_count += 1

            if total_env_steps >= max_eval_steps:
                print(f"  Episode {episode}: Reached max eval steps ({max_eval_steps}), truncating.")

        print(f"Episode {episode}: lower Reward {episode_lower_reward:.2f} "
              f"Steps {step_count}, Collisions {info.get('collision', False)}")
        env.close_writer()

        # Draw path
        env.draw_path(csv_file, ifshow=False, ifsave=True, savePath=eval_dir)


# ============================================================================
# ========================= Main Entry Point =================================
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HRL-MG + HGG for USV Multi-Target Navigation")

    # --- 环境参数 ---
    parser.add_argument("--num_targets", default=2, type=int)
    parser.add_argument("--num_obstacles", default=5, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--usv_length", default=3.0, type=float)
    parser.add_argument("--usv_width", default=1.5, type=float)
    parser.add_argument("--target_radius", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_min", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_max", default=10.0, type=float)
    parser.add_argument("--use_random_shapes", action="store_true", default=True)

    # --- 训练参数 ---
    parser.add_argument("--max_episodes", default=3000, type=int)
    parser.add_argument("--upper_steps", default=20, type=int,
                        help="Upper steps per Selector step")
    parser.add_argument("--lower_steps", default=100, type=int,
                        help="Lower steps per Upper step")
    parser.add_argument("--selector_batch_size", default=64, type=int)
    parser.add_argument("--actuator_batch_size", default=256, type=int)
    parser.add_argument("--selector_buffer_size", default=50000, type=int)
    parser.add_argument("--warmup_steps", default=10000, type=int)
    parser.add_argument("--curriculum_episodes", default=200, type=int,
                        help="课程学习: 前 N 个 episode 使用固定环境")

    # --- 探索参数 ---
    parser.add_argument("--epsilon_start", default=1.0, type=float)
    parser.add_argument("--epsilon_end", default=0.05, type=float)
    parser.add_argument("--epsilon_decay", default=0.998, type=float)
    parser.add_argument("--exploration_noise", default=0.2, type=float)
    parser.add_argument("--subgoal_range", default=0.3, type=float,
                        help="Range of sub-goal (normalized)")

    # --- HGG 参数 (新增) ---
    parser.add_argument("--hgg_K", default=50, type=int,
                        help="HGG: 每次迭代生成的 hindsight 目标数量")
    parser.add_argument("--hgg_L", default=5.0, type=float,
                        help="HGG: Lipschitz 常数 (控制值函数项权重)")
    parser.add_argument("--hgg_c", default=3.0, type=float,
                        help="HGG: 初始状态距离权重")
    parser.add_argument("--hgg_warmup", default=300, type=int,
                        help="HGG: 启动前的预热 episode 数 (确保值函数先收敛)")
    parser.add_argument("--hgg_max_traj", default=100, type=int,
                        help="HGG: 二部图匹配时最多使用的轨迹数量")
    parser.add_argument("--hgg_max_trans", default=10, type=int,
                        help="HGG: 每条轨迹最多采样的 transition 数 (下采样)")
    parser.add_argument("--min_buffer_for_hgg", default=30, type=int,
                        help="HGG: 启用所需最小轨迹数量 (buffer 轨迹数 >= 此值才启用)")
    parser.add_argument("--lower_pretrain_episodes", default=100, type=int,
                        help="Lower 层预训练 episode 数 (更大噪声 + 更密集训练, 先学避障)")
    parser.add_argument("--lower_train_iters", default=20, type=int,
                        help="Lower 层预训练阶段每步训练迭代次数 (默认阶段为 min(l_step,10))")

    # --- 输出参数 ---
    parser.add_argument("--output_dir", default="hrl_output", type=str)

    # --- 模式 ---
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_episodes", default=5, type=int)

    args = parser.parse_args()

    if args.do_train:
        train_hrl(args)

    if args.do_eval:
        evaluate_hrl(args)
