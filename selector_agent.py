"""
Selector Agent (DQN) — 目标选择器

选择下一个要访问的目标点, 使用 ε-greedy 策略的 DQN 网络。

包含:
- SelectorQNet: DQN 网络结构
- SelectorAgent: 完整的 DQN 智能体 (选择、训练、更新)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hrl_config import device


class SelectorQNet(nn.Module):
    """DQN for Selector: State → Q-values (每个目标一个)"""

    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SelectorAgent:
    """
    Selector Agent (DQN)
    选择下一个要访问的目标点, ε-greedy 策略
    """

    def __init__(self, state_dim, num_targets, lr=1e-4, gamma=0.99, epsilon=1.0):
        self.q_net = SelectorQNet(state_dim, num_targets).to(device)
        self.target_q_net = SelectorQNet(state_dim, num_targets).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_targets = num_targets

    def select_action(self, state, valid_mask=None, eval_mode=False):
        """选择目标, ε-greedy, 仅从 valid (未访问) 目标中选择"""
        if not eval_mode and random.random() < self.epsilon:
            if valid_mask is not None:
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    return int(np.random.choice(valid_indices))
            return np.random.randint(0, self.num_targets)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state_t).cpu().numpy().flatten()

        if valid_mask is not None:
            q_values[~valid_mask.astype(bool)] = -1e9

        return int(np.argmax(q_values))

    def train(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return 0.0
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_q = self.target_q_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (1 - done) * self.gamma * next_q

        current_q = self.q_net(state).gather(1, action)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self, tau=0.005):
        for p, tp in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
