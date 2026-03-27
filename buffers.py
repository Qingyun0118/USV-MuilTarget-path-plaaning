"""
经验回放缓冲区 (Replay Buffers)

包含:
- ReplayBuffer: 标准回放缓冲区, 用于 Selector (DQN)
- GoalConditionedReplayBuffer: 目标条件回放缓冲区, 用于 Actuator (TD3/SAC)
  支持优先经验回放 (Prioritized Experience Replay, PER)
"""

import random
import numpy as np
import torch

from hrl_config import device


class SumTree:
    """
    Sum Tree 数据结构, 用于 O(log N) 的优先级采样。

    叶节点存储 transition 的优先级 (priority),
    内部节点存储子树优先级之和。
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_pos = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        tree_idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(tree_idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        """根据累积优先级 s 查找对应的 (tree_idx, priority, data)"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def get_data_by_index(self, data_idx):
        return self.data[data_idx]

    def set_data_by_index(self, data_idx, data):
        self.data[data_idx] = data

    @property
    def last_data_idx(self):
        """最近一次写入的 data 索引"""
        return (self.write_pos - 1) % self.capacity


class ReplayBuffer:
    """Standard Replay Buffer for Selector (DQN)"""

    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        data = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

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


class GoalConditionedReplayBuffer:
    """
    Goal-Conditioned Prioritized Experience Replay Buffer for Actuator (TD3/SAC)

    基于 SumTree 实现优先经验回放 (PER):
    - 新体验以最大优先级插入, 确保至少被采样一次
    - 采样概率 P(i) = p_i^α / Σ p_j^α
    - 重要性采样权重 w_i = (N * P(i))^(-β), β 从 beta0 线性退火到 1.0
    - Expert 体验的优先级额外加成 expert_bonus, 延缓被覆盖后价值流失

    存储 (state, action, reward, next_state, done, goal, is_expert) 七元组
    """

    def __init__(self, capacity, alpha=0.6, beta0=0.4, beta_frames=100000,
                 expert_bonus=0.5):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha          # 优先级指数: 0=均匀, 1=完全优先
        self.beta0 = beta0          # IS 权重初始 β
        self.beta_frames = beta_frames
        self.expert_bonus = expert_bonus
        self.max_priority = 1.0     # 跟踪当前最大优先级
        self.frame = 0              # 用于 β 退火

    def push(self, state, action, reward, next_state, done, goal, is_expert=False):
        data = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
            np.array(goal, dtype=np.float32),
            float(is_expert)
        )
        # 新样本用当前最大优先级, Expert 额外加成
        priority = self.max_priority
        if is_expert:
            priority += self.expert_bonus
        self.tree.add(priority ** self.alpha, data)

    def sample(self, batch_size):
        """
        按优先级采样

        Returns:
            states, actions, rewards, next_states, dones, goals, is_experts,
            indices (tree indices), weights (IS weights)
        """
        self.frame += 1
        beta = min(1.0, self.beta0 + (1.0 - self.beta0) * self.frame / self.beta_frames)

        indices = []
        priorities = []
        batch = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            tree_idx, priority, data = self.tree.get(s)
            # 防止 priority=0 导致的问题
            priority = max(priority, 1e-8)
            indices.append(tree_idx)
            priorities.append(priority)
            batch.append(data)

        # 重要性采样权重
        priorities = np.array(priorities, dtype=np.float64)
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.size * sampling_probs) ** (-beta)
        weights /= weights.max()  # 归一化, 最大权重=1

        states, actions, rewards, next_states, dones, goals, is_experts = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).reshape(-1, 1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).reshape(-1, 1).to(device),
            torch.FloatTensor(np.array(goals)).to(device),
            torch.FloatTensor(np.array(is_experts)).reshape(-1, 1).to(device),
            indices,
            torch.FloatTensor(weights).reshape(-1, 1).to(device),
        )

    def update_priorities(self, indices, td_errors):
        """
        用 TD error 更新优先级

        Args:
            indices: tree indices (从 sample 返回)
            td_errors: numpy array or tensor, shape (batch,) or (batch,1)
        """
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        td_errors = np.squeeze(td_errors)

        for idx, td in zip(indices, td_errors):
            # 加小常数避免零优先级
            priority = (abs(td) + 1e-6) ** self.alpha
            # Expert 数据加成
            data_idx = idx - self.tree.capacity + 1
            data = self.tree.get_data_by_index(data_idx)
            if data is not None and data[6] > 0.5:  # is_expert
                priority += self.expert_bonus ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def update_last_n_rewards(self, n, reward_modifier_fn):
        """
        修改最近 n 条 transition 的 reward (用于碰撞回溯惩罚)

        Args:
            n: 回溯步数
            reward_modifier_fn: callable(k, old_reward) -> new_reward
                k: 距最新 transition 的步数 (2, 3, ..., n)
        """
        for k in range(2, n + 1):
            data_idx = (self.tree.write_pos - k) % self.capacity
            if data_idx < 0 or data_idx >= self.tree.size:
                break
            old = self.tree.get_data_by_index(data_idx)
            if old is None:
                break
            new_reward = reward_modifier_fn(k, old[2])
            new_data = (old[0], old[1], new_reward, old[3], old[4], old[5], old[6])
            self.tree.set_data_by_index(data_idx, new_data)
            # 提高被修改样本的优先级 (惩罚加大了, 更值得学习)
            tree_idx = data_idx + self.capacity - 1
            self.tree.update(tree_idx, self.max_priority ** self.alpha)

    def __len__(self):
        return self.tree.size
