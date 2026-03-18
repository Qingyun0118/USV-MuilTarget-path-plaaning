"""
Selector Agent (DQN) — 目标选择器

选择下一个要访问的目标点, 使用 ε-greedy 策略的 DQN 网络。

使用 DeepSets 风格的 Set Encoder 架构, 使网络与目标数量解耦:
- 每个目标独立编码 (φ), 聚合后生成全局上下文
- 结合全局上下文与逐目标嵌入, 为每个目标输出 Q 值
- 训练时用 N 个目标, 评估时可用任意数量目标

包含:
- SelectorQNet: DeepSets-style DQN 网络
- SelectorAgent: 完整的 DQN 智能体 (选择、训练、更新)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hrl_config import device

# Flat state 中 USV 信息和每个目标信息的维度
_USV_DIM = 2       # [x_norm, y_norm]
_TARGET_DIM = 3    # [tx_norm, ty_norm, status]


class SelectorQNet(nn.Module):
    """
    DeepSets-style DQN — 可处理任意数量目标

    Flat state 格式: [usv_x, usv_y, t1_x, t1_y, t1_status, t2_x, ...]
    内部自动解析为:
      - usv_state : [B, 2]
      - targets   : [B, N, 3]    (N = 目标数, 可变)

    输出: Q-values [B, N] — 每个目标一个 Q 值
    """

    def __init__(self, embed_dim=64):
        super().__init__()
        encoder_input_dim = _USV_DIM + _TARGET_DIM  # 5

        # φ: 逐目标编码器 (共享权重), 输入 = [usv_state, target_i]
        self.target_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU()
        )

        # ρ: Q 值头, 输入 = [usv_state, global_embed, target_i_embed]
        self.q_head = nn.Sequential(
            nn.Linear(_USV_DIM + embed_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, flat_state):
        """
        Args:
            flat_state: [B, 2 + 3*N]  flat state vector
        Returns:
            q_values:   [B, N]
        """
        # ---- 解析 flat state ----
        usv_state = flat_state[:, :_USV_DIM]                          # [B, 2]
        targets_flat = flat_state[:, _USV_DIM:]                       # [B, 3*N]
        batch_size = flat_state.shape[0]
        num_targets = targets_flat.shape[1] // _TARGET_DIM
        targets = targets_flat.reshape(batch_size, num_targets, _TARGET_DIM)  # [B, N, 3]

        # ---- 逐目标编码 ----
        usv_exp = usv_state.unsqueeze(1).expand(-1, num_targets, -1)  # [B, N, 2]
        encoder_in = torch.cat([usv_exp, targets], dim=-1)            # [B, N, 5]
        target_embeds = self.target_encoder(encoder_in)               # [B, N, E]

        # ---- 全局聚合 (mean pooling) ----
        global_embed = target_embeds.mean(dim=1, keepdim=True)        # [B, 1, E]
        global_embed = global_embed.expand(-1, num_targets, -1)       # [B, N, E]

        # ---- 逐目标 Q 值 ----
        q_in = torch.cat([usv_exp, global_embed, target_embeds], dim=-1)  # [B, N, 2+E+E]
        q_values = self.q_head(q_in).squeeze(-1)                     # [B, N]

        return q_values


class SelectorAgent:
    """
    Selector Agent (DQN)
    选择下一个要访问的目标点, ε-greedy 策略
    使用 Set Encoder 架构, 支持可变目标数
    """

    def __init__(self, embed_dim=64, lr=1e-4, gamma=0.99, epsilon=1.0):
        self.q_net = SelectorQNet(embed_dim).to(device)
        self.target_q_net = SelectorQNet(embed_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state, valid_mask=None, eval_mode=False):
        """选择目标, ε-greedy, 仅从 valid (未访问) 目标中选择"""
        num_targets = (len(state) - _USV_DIM) // _TARGET_DIM

        if not eval_mode and random.random() < self.epsilon:
            if valid_mask is not None:
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    return int(np.random.choice(valid_indices))
            return np.random.randint(0, num_targets)

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
