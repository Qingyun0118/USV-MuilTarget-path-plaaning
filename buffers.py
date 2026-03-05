"""
经验回放缓冲区 (Replay Buffers)

包含:
- ReplayBuffer: 标准回放缓冲区, 用于 Selector (DQN)
- GoalConditionedReplayBuffer: 目标条件回放缓冲区, 用于 Actuator (TD3/SAC)
"""

import random
import numpy as np
import torch
from collections import deque

from hrl_config import device


class ReplayBuffer:
    """Standard Replay Buffer for Selector (DQN)"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        ))

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
    Goal-Conditioned Replay Buffer for Actuator (TD3/SAC)
    存储 (state, action, reward, next_state, done, goal, is_mpc) 七元组

    is_mpc 标志用于区分 MPC 专家生成的 transition 和 RL 策略生成的 transition,
    以便在训练时对 MPC transition 施加行为克隆损失 (Behavioral Cloning Loss)。
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, goal, is_mpc=False):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
            np.array(goal, dtype=np.float32),
            float(is_mpc)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, goals, is_mpcs = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).reshape(-1, 1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).reshape(-1, 1).to(device),
            torch.FloatTensor(np.array(goals)).to(device),
            torch.FloatTensor(np.array(is_mpcs)).reshape(-1, 1).to(device),
        )

    def __len__(self):
        return len(self.buffer)
