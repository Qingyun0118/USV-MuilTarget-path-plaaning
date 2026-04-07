"""
Coordinated DQN Selector — 协调 DQN 目标选择器

在原始 DeepSets DQN 基础上加入多智能体协调信息,
保持 off-policy 训练方式 (Double DQN + Target Network + Experience Replay).

包含:
  - CoordSelectorQNet:     协调 DQN 网络
  - SelectorDQNReplayBuffer: 稀疏决策点经验回放
  - CoordSelectorDQNAgent:  完整的协调 DQN 智能体
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hrl_config import device

# Flat state 维度
_USV_DIM = 2       # [x_norm, y_norm]
_TARGET_DIM = 3    # [tx_norm, ty_norm, status]


# ============================================================================
# ============================ Q Network =====================================
# ============================================================================

class CoordSelectorQNet(nn.Module):
    """
    协调 DQN — 在原始 DeepSets DQN 基础上加入多智能体协调信息.

    原 SelectorQNet:
      target_encoder(usv + target_i) → embeds [B, N, E]
      mean pool → global [B, 1, E]
      q_head(usv, global, embed_i) → Q(s, a_i)

    协调扩展:
      coord_encoder(其他USV位置+已分配目标) → coord_embed [B, E]
      q_head(usv, global, embed_i, coord_embed) → Q(s, a_i)
    """

    def __init__(self, embed_dim=64, max_usvs=4):
        super().__init__()
        self.embed_dim = embed_dim
        encoder_input_dim = _USV_DIM + _TARGET_DIM  # 5

        # φ: 逐目标编码器 (与 SelectorQNet 相同)
        self.target_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU()
        )

        # 协调编码器: 其他 USV 的 (x, y, tgt_x, tgt_y)
        coord_input_dim = 4 * (max_usvs - 1)
        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU()
        )

        # Q 值头 (输入: usv + global + target_embed + coord_embed)
        self.q_head = nn.Sequential(
            nn.Linear(_USV_DIM + embed_dim + embed_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, flat_state, coord_info=None):
        """
        Args:
            flat_state: [B, 2 + 3*N]
            coord_info: [B, 4*(max_usvs-1)] 或 None (用零向量)
        Returns:
            q_values: [B, N]
        """
        usv_state = flat_state[:, :_USV_DIM]
        targets_flat = flat_state[:, _USV_DIM:]
        batch_size = flat_state.shape[0]
        num_targets = targets_flat.shape[1] // _TARGET_DIM
        targets = targets_flat.reshape(batch_size, num_targets, _TARGET_DIM)

        # 逐目标编码
        usv_exp = usv_state.unsqueeze(1).expand(-1, num_targets, -1)
        encoder_in = torch.cat([usv_exp, targets], dim=-1)
        target_embeds = self.target_encoder(encoder_in)       # [B, N, E]

        # 全局聚合
        global_embed = target_embeds.mean(dim=1, keepdim=True).expand(
            -1, num_targets, -1)                              # [B, N, E]

        # 协调编码
        if coord_info is not None:
            coord_embed = self.coord_encoder(coord_info)      # [B, E]
        else:
            coord_embed = torch.zeros(
                batch_size, self.embed_dim, device=flat_state.device)
        coord_embed = coord_embed.unsqueeze(1).expand(
            -1, num_targets, -1)                              # [B, N, E]

        # Q 值
        q_in = torch.cat(
            [usv_exp, global_embed, target_embeds, coord_embed], dim=-1)
        q_values = self.q_head(q_in).squeeze(-1)              # [B, N]
        return q_values


# ============================================================================
# ========================= Replay Buffer ====================================
# ============================================================================

class SelectorDQNReplayBuffer:
    """Selector DQN 的经验回放缓冲区 (稀疏决策点)."""

    def __init__(self, max_size=10000):
        from collections import deque as _deque
        self.buffer = _deque(maxlen=max_size)

    def add(self, state, coord, valid_mask, action, reward,
            next_state, next_coord, next_valid_mask, done):
        self.buffer.append((
            state.copy(), coord.copy(), valid_mask.copy(),
            int(action), float(reward),
            next_state.copy(), next_coord.copy(),
            next_valid_mask.copy(), float(done),
        ))

    def sample(self, batch_size):
        import random as _random
        batch = _random.sample(
            list(self.buffer), min(batch_size, len(self.buffer)))
        states, coords, valids, actions, rewards, \
            next_states, next_coords, next_valids, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(coords)).to(device),
            torch.BoolTensor(np.array(valids)).to(device),
            torch.LongTensor(actions).unsqueeze(1).to(device),
            torch.FloatTensor(rewards).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(next_coords)).to(device),
            torch.BoolTensor(np.array(next_valids)).to(device),
            torch.FloatTensor(dones).unsqueeze(1).to(device),
        )

    @property
    def size(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# ============================ DQN Agent =====================================
# ============================================================================

class CoordSelectorDQNAgent:
    """
    协调 DQN Selector — 保持 DQN 结构, 加入多智能体协调信息微调.

    与冻结 DQN 的区别:
      - 输入增加其他 USV 的位置 + 已分配目标 (coordination info)
      - 参数可训练, 使用 ε-greedy 探索 + 衰减
      - 使用累积奖励 (分配→下次分配期间) 作为 DQN reward
      - Double DQN + Target network + 经验回放, off-policy 训练

    从预训练 DQN 热启动:
      target_encoder 完全迁移, q_head 部分迁移 (第一层拓宽).
    """

    def __init__(self, embed_dim=64, lr=1e-4, gamma=0.99,
                 epsilon_start=0.3, epsilon_end=0.01, epsilon_decay=0.995,
                 max_usvs=4, buffer_size=10000, batch_size=64,
                 target_update_tau=0.005):
        self.q_net = CoordSelectorQNet(embed_dim, max_usvs).to(device)
        self.target_q_net = CoordSelectorQNet(embed_dim, max_usvs).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_tau = target_update_tau

        self.buffer = SelectorDQNReplayBuffer(buffer_size)
        self._pending_decisions = {}

    def load_from_dqn(self, dqn_path):
        """从预训练 SelectorQNet 热启动."""
        from selector_agent import SelectorQNet
        dqn = SelectorQNet()
        dqn.load_state_dict(
            torch.load(dqn_path, map_location=device, weights_only=True))

        # target_encoder: 完全迁移
        self.q_net.target_encoder.load_state_dict(
            dqn.target_encoder.state_dict())
        self.target_q_net.target_encoder.load_state_dict(
            dqn.target_encoder.state_dict())

        # q_head: 第一层加宽 (130 → 130+E), 部分迁移
        with torch.no_grad():
            old_w = dqn.q_head[0].weight.data   # [128, 130]
            old_b = dqn.q_head[0].bias.data      # [128]
            self.q_net.q_head[0].weight.data[:, :old_w.shape[1]].copy_(old_w)
            self.q_net.q_head[0].bias.data.copy_(old_b)
            self.target_q_net.q_head[0].weight.data[
                :, :old_w.shape[1]].copy_(old_w)
            self.target_q_net.q_head[0].bias.data.copy_(old_b)

        # 后续层完全迁移 (Linear(128,64), Linear(64,1))
        for i in [2, 4]:
            self.q_net.q_head[i].weight.data.copy_(
                dqn.q_head[i].weight.data)
            self.q_net.q_head[i].bias.data.copy_(
                dqn.q_head[i].bias.data)
            self.target_q_net.q_head[i].weight.data.copy_(
                dqn.q_head[i].weight.data)
            self.target_q_net.q_head[i].bias.data.copy_(
                dqn.q_head[i].bias.data)

        print(f"[CoordDQN] 从预训练 DQN 热启动完成: {dqn_path}")

    @torch.no_grad()
    def select_action(self, sel_state, valid_mask, coord_info=None,
                      eval_mode=False):
        """ε-greedy 选择目标."""
        import random as _random
        num_targets = (len(sel_state) - _USV_DIM) // _TARGET_DIM

        if not eval_mode and _random.random() < self.epsilon:
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                return int(np.random.choice(valid_indices))
            return np.random.randint(0, num_targets)

        s = torch.FloatTensor(sel_state).unsqueeze(0).to(device)
        ci = (torch.FloatTensor(coord_info).unsqueeze(0).to(device)
              if coord_info is not None else None)
        q_values = self.q_net(s, ci).cpu().numpy().flatten()

        if valid_mask is not None:
            q_values[~valid_mask.astype(bool)] = -1e9

        return int(np.argmax(q_values))

    def train_step(self):
        """Double DQN 更新一步."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, coords, valids, actions, rewards, \
            next_states, next_coords, next_valids, dones = \
            self.buffer.sample(self.batch_size)

        with torch.no_grad():
            # Double DQN: online net 选动作, target net 评估
            next_q_online = self.q_net(next_states, next_coords)
            next_q_online[~next_valids] = -1e9
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            next_q_target = self.target_q_net(next_states, next_coords)
            next_q_value = next_q_target.gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_value

        current_q = self.q_net(states, coords).gather(1, actions)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Soft update target network."""
        tau = self.target_update_tau
        for p, tp in zip(self.q_net.parameters(),
                         self.target_q_net.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def decay_epsilon(self):
        """衰减探索率."""
        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_q_net.load_state_dict(ckpt['target_q_net'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'epsilon' in ckpt:
            self.epsilon = ckpt['epsilon']
