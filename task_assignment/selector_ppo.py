"""
Selector PPO Agent — 目标选择器 (PPO 版, 用于 CTDE 微调)

将原 DQN Selector 改为 on-policy PPO, 使其可与 MAPPO Actuator 联合训练.

架构:
  - SelectorPolicyNet: DeepSets-style 策略网络, 输出 Categorical 分布
  - SelectorValueNet:  DeepSets-style 集中式价值网络, V(global_state)
  - SelectorPPORolloutBuffer: 稀疏决策点的 on-policy 缓冲区
  - SelectorPPOAgent:  PPO 训练逻辑 + 推理接口

与 Actuator MAPPO 的协同:
  - Selector 在 episode 开始和目标到达时做决策 (稀疏)
  - 累积 Actuator 执行期间的奖励作为 Selector 的 reward
  - 使用集中式 global_state 作为 Critic 输入
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from hrl_config import device

# Flat state 维度 (与 selector_agent.py 一致)
_USV_DIM = 2       # [x_norm, y_norm]
_TARGET_DIM = 3    # [tx_norm, ty_norm, status]


# ============================================================================
# ============================ Policy Network ================================
# ============================================================================

class SelectorPolicyNet(nn.Module):
    """
    DeepSets-style 策略网络 — 输出 Categorical 分布的 logits.

    结构与 SelectorQNet 一致, 但输出语义为 logits(非 Q 值),
    配合 Categorical 采样和 PPO 更新.

    Flat state: [usv_x, usv_y, t1_x, t1_y, t1_status, t2_x, ...]

    额外输入: other_assignments — 其他 USV 的已分配目标 (协调信息)
    """

    def __init__(self, embed_dim=64, use_coordination=True, max_usvs=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_coordination = use_coordination

        encoder_input_dim = _USV_DIM + _TARGET_DIM  # 5

        # φ: 逐目标编码器
        self.target_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU()
        )

        # 协调编码器: 编码其他 USV 的分配信息
        if use_coordination:
            # 输入: 其他 USV 的 (x, y, assigned_target_x, assigned_target_y)
            coord_input_dim = 4 * (max_usvs - 1)
            self.coord_encoder = nn.Sequential(
                nn.Linear(coord_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, embed_dim),
                nn.ReLU()
            )
            head_input_dim = _USV_DIM + embed_dim + embed_dim + embed_dim
        else:
            self.coord_encoder = None
            head_input_dim = _USV_DIM + embed_dim + embed_dim

        # logit 头
        self.logit_head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, flat_state, valid_mask=None, coord_info=None):
        """
        Args:
            flat_state: [B, 2 + 3*N]
            valid_mask: [B, N] bool, True = 可选目标
            coord_info: [B, 4*(max_usvs-1)] 其他 USV 协调信息 (可选)

        Returns:
            logits: [B, N] (已 mask 无效目标)
        """
        usv_state = flat_state[:, :_USV_DIM]                           # [B, 2]
        targets_flat = flat_state[:, _USV_DIM:]                        # [B, 3*N]
        batch_size = flat_state.shape[0]
        num_targets = targets_flat.shape[1] // _TARGET_DIM
        targets = targets_flat.reshape(batch_size, num_targets, _TARGET_DIM)

        # 逐目标编码
        usv_exp = usv_state.unsqueeze(1).expand(-1, num_targets, -1)   # [B, N, 2]
        encoder_in = torch.cat([usv_exp, targets], dim=-1)             # [B, N, 5]
        target_embeds = self.target_encoder(encoder_in)                # [B, N, E]

        # 全局聚合
        global_embed = target_embeds.mean(dim=1, keepdim=True)         # [B, 1, E]
        global_embed = global_embed.expand(-1, num_targets, -1)        # [B, N, E]

        # 构建 logit 输入
        if self.use_coordination and coord_info is not None:
            coord_embed = self.coord_encoder(coord_info)               # [B, E]
            coord_embed = coord_embed.unsqueeze(1).expand(
                -1, num_targets, -1)                                   # [B, N, E]
            logit_in = torch.cat(
                [usv_exp, global_embed, target_embeds, coord_embed],
                dim=-1)
        else:
            logit_in = torch.cat(
                [usv_exp, global_embed, target_embeds], dim=-1)

        logits = self.logit_head(logit_in).squeeze(-1)                 # [B, N]

        # mask 无效目标
        if valid_mask is not None:
            logits[~valid_mask] = -1e9

        return logits

    def get_dist(self, flat_state, valid_mask=None, coord_info=None):
        """返回 Categorical 分布."""
        logits = self.forward(flat_state, valid_mask, coord_info)
        return Categorical(logits=logits)


# ============================================================================
# ============================ Value Network =================================
# ============================================================================

class SelectorValueNet(nn.Module):
    """
    Selector 的集中式价值网络.

    与 Actuator 的 CentralizedCritic 共享全局状态输入,
    但独立参数 (Selector 的 reward 定义不同于 Actuator).
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
            nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, global_state):
        return self.net(global_state)


# ============================================================================
# ========================= Rollout Buffer ===================================
# ============================================================================

class SelectorPPORolloutBuffer:
    """
    Selector 决策点的 on-policy 缓冲区.

    与 Actuator 的 MAPPORolloutBuffer 不同:
      - 数据稀疏: 仅在 episode 开始 / 目标到达时记录
      - 每条记录包含: selector_state, action(目标idx), log_prob, value,
        累积奖励(分配→下次分配期间的折扣和), global_state
      - 可跨多 episode 累积 (因为数据太稀疏)
    """

    def __init__(self, max_size=2048):
        self.max_size = max_size

        self.sel_states = []       # selector flat state
        self.actions = []          # int, chosen target_idx
        self.log_probs = []        # log π(action)
        self.values = []           # V(global_state)
        self.rewards = []          # 累积折扣奖励
        self.dones = []            # episode 终止标志
        self.valid_masks = []      # [N] bool
        self.coord_infos = []      # 协调信息 (可选)
        self.global_states = []    # 全局状态

    @property
    def size(self):
        return len(self.sel_states)

    def add(self, sel_state, action, log_prob, value,
            reward, done, valid_mask, global_state, coord_info=None):
        """添加一条 Selector 决策记录."""
        self.sel_states.append(sel_state.copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.valid_masks.append(valid_mask.copy())
        self.global_states.append(global_state.copy())
        self.coord_infos.append(
            coord_info.copy() if coord_info is not None else None)

        # 容量保护
        if len(self.sel_states) > self.max_size:
            n_remove = len(self.sel_states) - self.max_size
            for attr in ['sel_states', 'actions', 'log_probs', 'values',
                         'rewards', 'dones', 'valid_masks', 'global_states',
                         'coord_infos']:
                setattr(self, attr, getattr(self, attr)[n_remove:])

    def compute_gae(self, last_value=0.0, gamma=0.99, gae_lambda=0.95):
        """计算 GAE 优势和 returns."""
        n = self.size
        if n == 0:
            return

        values = np.array(self.values, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * non_terminal - values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + values

    def get_batches(self, batch_size):
        """产生 mini-batch. 注意: valid_mask 维度可能不同(不同目标数),
        这里假定同一训练周期内目标数相同."""
        n = self.size
        if n == 0:
            return

        sel_states = np.array(self.sel_states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int64)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)
        advantages = self.advantages.copy()
        returns = self.returns.copy()
        valid_masks = np.array(self.valid_masks, dtype=bool)
        global_states = np.array(self.global_states, dtype=np.float32)

        has_coord = self.coord_infos[0] is not None
        if has_coord:
            coord_infos = np.array(
                [c if c is not None else np.zeros(1)
                 for c in self.coord_infos], dtype=np.float32)

        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            batch = {
                'sel_states': torch.FloatTensor(sel_states[idx]).to(device),
                'actions': torch.LongTensor(actions[idx]).to(device),
                'old_log_probs': torch.FloatTensor(
                    old_log_probs[idx]).to(device),
                'advantages': torch.FloatTensor(advantages[idx]).to(device),
                'returns': torch.FloatTensor(returns[idx]).to(device),
                'valid_masks': torch.BoolTensor(valid_masks[idx]).to(device),
                'global_states': torch.FloatTensor(
                    global_states[idx]).to(device),
            }
            if has_coord:
                batch['coord_infos'] = torch.FloatTensor(
                    coord_infos[idx]).to(device)
            yield batch

    def clear(self):
        for attr in ['sel_states', 'actions', 'log_probs', 'values',
                     'rewards', 'dones', 'valid_masks', 'global_states',
                     'coord_infos']:
            setattr(self, attr, [])
        self.advantages = None
        self.returns = None


# ============================================================================
# ============================ PPO Agent =====================================
# ============================================================================

class SelectorPPOAgent:
    """
    Selector PPO Agent — 目标选择策略的 PPO 训练.

    训练特点:
      - 决策稀疏: 每 episode 仅 1-3 次决策, 需跨 episode 累积
      - reward 来源: 分配→下次分配区间内 Actuator 的累积折扣奖励
      - 可从预训练 DQN 的权重热启动 logit_head

    用法:
      agent = SelectorPPOAgent(global_state_dim)
      agent.load_from_dqn(pretrained_selector_path)  # 可选

      # rollout 阶段
      action, log_prob = agent.select_action(sel_state, valid_mask, coord_info)
      value = agent.get_value(global_state)

      # 累积奖励后
      agent.buffer.add(sel_state, action, log_prob, value, cum_reward, done,
                        valid_mask, global_state, coord_info)

      # 更新 (达到最小决策数后)
      if agent.buffer.size >= min_decisions:
          losses = agent.update()
    """

    def __init__(self, global_state_dim, embed_dim=64,
                 lr_policy=1e-4, lr_value=3e-4,
                 clip_eps=0.2, entropy_coef=0.02,
                 max_grad_norm=0.5, ppo_epochs=5,
                 mini_batch_size=64, gamma=0.99, gae_lambda=0.95,
                 use_coordination=True, max_usvs=4,
                 buffer_size=2048):

        self.policy = SelectorPolicyNet(
            embed_dim, use_coordination, max_usvs).to(device)
        self.value_net = SelectorValueNet(
            global_state_dim).to(device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr_policy, eps=1e-5)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=lr_value, eps=1e-5)

        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.buffer = SelectorPPORolloutBuffer(buffer_size)

    def load_from_dqn(self, dqn_path):
        """
        从预训练 DQN Selector (SelectorQNet) 的权重热启动.

        SelectorQNet.target_encoder → SelectorPolicyNet.target_encoder
        SelectorQNet.q_head → SelectorPolicyNet.logit_head
        """
        from selector_agent import SelectorQNet
        dqn = SelectorQNet()
        dqn.load_state_dict(
            torch.load(dqn_path, map_location=device, weights_only=True))

        # 迁移 target_encoder
        self.policy.target_encoder.load_state_dict(
            dqn.target_encoder.state_dict())
        print(f"[SelectorPPO] target_encoder ← DQN")

        # 迁移 q_head → logit_head (结构完全相同: Linear→ReLU→Linear→ReLU→Linear)
        if not self.policy.use_coordination:
            # 无协调时 head 输入维度与 DQN 相同, 可直接迁移
            self.policy.logit_head.load_state_dict(
                dqn.q_head.state_dict())
            print(f"[SelectorPPO] logit_head ← DQN q_head (direct)")
        else:
            # 有协调时 head 输入维度更大, 仅迁移后面的层
            # DQN q_head: Linear(130, 128), ReLU, Linear(128, 64), ReLU, Linear(64, 1)
            # PPO logit_head: Linear(130+E, 128), ReLU, ...  — 第一层不同
            # 迁移第 2,3,4 层 (index 2,3,4)
            for i in [2, 4]:
                self.policy.logit_head[i].weight.data.copy_(
                    dqn.q_head[i].weight.data)
                self.policy.logit_head[i].bias.data.copy_(
                    dqn.q_head[i].bias.data)
            print(f"[SelectorPPO] logit_head ← DQN q_head (partial, "
                  f"coordination mode)")

        print(f"[SelectorPPO] 从 DQN 热启动完成: {dqn_path}")

    # ---- 推理接口 ----

    @torch.no_grad()
    def select_action(self, sel_state, valid_mask, coord_info=None,
                      eval_mode=False):
        """
        选择目标.

        Args:
            sel_state:  np.array [2 + 3*N]
            valid_mask: np.array [N] bool
            coord_info: np.array [4*(max_usvs-1)] 或 None
            eval_mode:  True 时取 argmax, False 时采样

        Returns:
            action:   int (目标索引)
            log_prob: float
        """
        s = torch.FloatTensor(sel_state).unsqueeze(0).to(device)
        vm = torch.BoolTensor(valid_mask).unsqueeze(0).to(device)
        ci = (torch.FloatTensor(coord_info).unsqueeze(0).to(device)
              if coord_info is not None else None)

        dist = self.policy.get_dist(s, vm, ci)

        if eval_mode:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    @torch.no_grad()
    def get_value(self, global_state):
        """获取价值估计."""
        gs = torch.FloatTensor(global_state).unsqueeze(0).to(device)
        return self.value_net(gs).item()

    # ---- PPO 更新 ----

    def update(self, last_value=0.0):
        """
        PPO 更新 (跨 episode 累积后调用).

        Returns:
            dict: {'sel_policy_loss', 'sel_value_loss', 'sel_entropy'}
        """
        if self.buffer.size < 2:
            return {'sel_policy_loss': 0, 'sel_value_loss': 0,
                    'sel_entropy': 0}

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _epoch in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size):
                sel_states = batch['sel_states']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                valid_masks = batch['valid_masks']
                global_states = batch['global_states']
                coord_infos = batch.get('coord_infos', None)

                # 标准化优势
                adv = ((advantages - advantages.mean())
                       / (advantages.std() + 1e-8))

                # ---- 策略更新 ----
                dist = self.policy.get_dist(
                    sel_states, valid_masks, coord_infos)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_mean = entropy.mean()

                policy_total = policy_loss - self.entropy_coef * entropy_mean

                self.policy_optimizer.zero_grad()
                policy_total.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                # ---- 价值更新 ----
                values = self.value_net(global_states).squeeze(-1)
                value_loss = F.mse_loss(values, returns)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                n_updates += 1

        self.buffer.clear()

        n = max(1, n_updates)
        return {
            'sel_policy_loss': total_policy_loss / n,
            'sel_value_loss': total_value_loss / n,
            'sel_entropy': total_entropy / n,
        }

    # ---- 保存/加载 ----

    def save(self, path):
        """保存 Selector PPO 模型."""
        torch.save({
            'policy': self.policy.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, path)

    def load(self, path):
        """加载 Selector PPO 模型."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.policy.load_state_dict(ckpt['policy'])
        self.value_net.load_state_dict(ckpt['value_net'])
        if 'policy_optimizer' in ckpt:
            self.policy_optimizer.load_state_dict(ckpt['policy_optimizer'])
        if 'value_optimizer' in ckpt:
            self.value_optimizer.load_state_dict(ckpt['value_optimizer'])
