"""
MakespanSelectorPPOAssigner — 基于 PPO 的学习型 Makespan 分配器 (Proposed 2)

从 selector_ppo.py 迁移核心网络 (SelectorPolicyNet, SelectorValueNet),
封装为 BaseAssigner 接口, 使用 makespan 改造后的奖励:

  r_sel = alpha * reach_bonus - beta * est_remaining_makespan + gamma * cum_actuator_reward
"""

from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from hrl_config import device
from task_assignment.base import BaseAssigner
from hrl_helpers import get_selector_state, get_usv_pos, get_target_pos

# Flat state 维度 (与 selector_agent.py / selector_ppo.py 一致)
_USV_DIM = 2
_TARGET_DIM = 3


# ============================================================================
# Policy / Value Networks (从 selector_ppo.py 迁移)
# ============================================================================

class SelectorPolicyNet(nn.Module):
    """DeepSets-style 策略网络 — 输出 Categorical 分布的 logits."""

    def __init__(self, embed_dim=64, use_coordination=True, max_usvs=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_coordination = use_coordination

        encoder_input_dim = _USV_DIM + _TARGET_DIM

        self.target_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU()
        )

        if use_coordination:
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
        usv_state = flat_state[:, :_USV_DIM]
        targets_flat = flat_state[:, _USV_DIM:]
        batch_size = flat_state.shape[0]
        num_targets = targets_flat.shape[1] // _TARGET_DIM
        targets = targets_flat.reshape(batch_size, num_targets, _TARGET_DIM)

        usv_exp = usv_state.unsqueeze(1).expand(-1, num_targets, -1)
        encoder_in = torch.cat([usv_exp, targets], dim=-1)
        target_embeds = self.target_encoder(encoder_in)

        global_embed = target_embeds.mean(dim=1, keepdim=True)
        global_embed = global_embed.expand(-1, num_targets, -1)

        if self.use_coordination and coord_info is not None:
            coord_embed = self.coord_encoder(coord_info)
            coord_embed = coord_embed.unsqueeze(1).expand(-1, num_targets, -1)
            logit_in = torch.cat(
                [usv_exp, global_embed, target_embeds, coord_embed], dim=-1)
        else:
            logit_in = torch.cat(
                [usv_exp, global_embed, target_embeds], dim=-1)

        logits = self.logit_head(logit_in).squeeze(-1)

        if valid_mask is not None:
            logits[~valid_mask] = -1e9

        return logits

    def get_dist(self, flat_state, valid_mask=None, coord_info=None):
        logits = self.forward(flat_state, valid_mask, coord_info)
        return Categorical(logits=logits)


class SelectorValueNet(nn.Module):
    """Selector 的集中式价值网络."""

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
# Rollout Buffer
# ============================================================================

class SelectorPPORolloutBuffer:
    """Selector 决策点的 on-policy 缓冲区, 可跨多 episode 累积."""

    def __init__(self, max_size=2048):
        self.max_size = max_size
        self.sel_states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.valid_masks = []
        self.coord_infos = []
        self.global_states = []

    @property
    def size(self):
        return len(self.sel_states)

    def add(self, sel_state, action, log_prob, value,
            reward, done, valid_mask, global_state, coord_info=None):
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

        if len(self.sel_states) > self.max_size:
            n_remove = len(self.sel_states) - self.max_size
            for attr in ['sel_states', 'actions', 'log_probs', 'values',
                         'rewards', 'dones', 'valid_masks', 'global_states',
                         'coord_infos']:
                setattr(self, attr, getattr(self, attr)[n_remove:])

    def compute_gae(self, last_value=0.0, gamma=0.99, gae_lambda=0.95):
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
                'old_log_probs': torch.FloatTensor(old_log_probs[idx]).to(device),
                'advantages': torch.FloatTensor(advantages[idx]).to(device),
                'returns': torch.FloatTensor(returns[idx]).to(device),
                'valid_masks': torch.BoolTensor(valid_masks[idx]).to(device),
                'global_states': torch.FloatTensor(global_states[idx]).to(device),
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
# 协调信息构建 (从 coord_utils.py 内联)
# ============================================================================

def _build_coord_info(obs_dict, num_usvs, usv_idx,
                      current_target_idx, target_pos_norm, max_usvs=4):
    """构建 USV i 的协调信息向量."""
    info = np.zeros(4 * (max_usvs - 1), dtype=np.float32)
    slot = 0
    for j in range(num_usvs):
        if j == usv_idx:
            continue
        if slot >= max_usvs - 1:
            break
        pos = get_usv_pos(obs_dict[f"usv_{j}"])
        tgt = target_pos_norm[j] if j < len(target_pos_norm) else np.zeros(2)
        base = slot * 4
        info[base:base + 2] = pos
        info[base + 2:base + 4] = tgt
        slot += 1
    return info


# ============================================================================
# Main Assigner
# ============================================================================

class MakespanSelectorPPOAssigner(BaseAssigner):
    """
    基于 PPO 的学习型 Makespan 目标分配器.

    与原始 selector_ppo 的关键区别 — Makespan 改造奖励:
      r_sel = alpha * reach_bonus - beta * est_remaining_makespan
              + gamma * cum_actuator_reward

    pending decision 机制:
      - assign_initial / reassign: 为每个USV创建 pending 记录
      - on_step: 累积 actuator 奖励到 pending
      - on_episode_end: 结算所有 pending, 写入 PPO buffer
      - update: 调用 PPO 更新
    """

    name = "mappo_makespan"
    is_learnable = True

    def __init__(self, global_state_dim, embed_dim=64,
                 lr_policy=1e-4, lr_value=3e-4,
                 clip_eps=0.2, entropy_coef=0.02,
                 max_grad_norm=0.5, ppo_epochs=5,
                 mini_batch_size=64, gamma=0.99, gae_lambda=0.95,
                 use_coordination=True, max_usvs=4,
                 buffer_size=2048,
                 alpha=1.0, beta=0.1, gamma_weight=0.5,
                 usv_speed_estimate=2.0):
        self.policy = SelectorPolicyNet(
            embed_dim, use_coordination, max_usvs).to(device)
        self.value_net = SelectorValueNet(global_state_dim).to(device)

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
        self._pending_decisions: Dict[int, dict] = {}

        # Makespan 奖励权重
        self.alpha = alpha
        self.beta = beta
        self.gamma_weight = gamma_weight
        self.usv_speed = usv_speed_estimate

        self._use_coordination = use_coordination
        self._max_usvs = max_usvs

    # ==========================================================================
    # 初始分配
    # ==========================================================================

    def assign_initial(
        self,
        obs_dict: dict,
        num_usvs: int,
        valid_mask: np.ndarray,
        scale: float,
        global_state: Optional[np.ndarray] = None,
        eval_mode: bool = False,
    ):
        target_idx = []
        target_norm = []
        target_abs = []
        sel_log_probs = []

        for i in range(num_usvs):
            sel_state = get_selector_state(obs_dict[f"usv_{i}"])
            coord = _build_coord_info(
                obs_dict, num_usvs, i, target_idx,
                target_norm if target_norm else [np.zeros(2)] * num_usvs,
                max_usvs=self._max_usvs)

            agent_valid = valid_mask.copy()
            for prev_idx in target_idx:
                if prev_idx < len(agent_valid):
                    agent_valid[prev_idx] = False
            if not np.any(agent_valid):
                agent_valid = valid_mask.copy()

            t_idx, log_prob = self._select_action(
                sel_state, agent_valid, coord, eval_mode=eval_mode)
            t_norm = get_target_pos(obs_dict[f"usv_{i}"], t_idx)
            target_idx.append(t_idx)
            target_norm.append(t_norm)
            target_abs.append(t_norm * scale)
            sel_log_probs.append(log_prob)

        # 创建 pending decisions
        if not eval_mode and global_state is not None:
            for i in range(num_usvs):
                self._pending_decisions[i] = {
                    'sel_state': get_selector_state(obs_dict[f"usv_{i}"]),
                    'action': target_idx[i],
                    'log_prob': sel_log_probs[i],
                    'value': self.get_value(global_state),
                    'valid_mask': valid_mask.copy(),
                    'global_state': global_state.copy(),
                    'coord_info': _build_coord_info(
                        obs_dict, num_usvs, i,
                        target_idx, target_norm, max_usvs=self._max_usvs),
                    'cum_reward': 0.0,
                    'reached': False,
                }

        return target_idx, target_norm, target_abs, (
            sel_log_probs if eval_mode else None)

    # ==========================================================================
    # 选择动作 (推理)
    # ==========================================================================

    @torch.no_grad()
    def _select_action(self, sel_state, valid_mask, coord_info=None,
                       eval_mode=False):
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
        gs = torch.FloatTensor(global_state).unsqueeze(0).to(device)
        return self.value_net(gs).item()

    # ==========================================================================
    # 重分配
    # ==========================================================================

    def reassign(
        self,
        obs_dict: dict,
        usv_indices: List[int],
        valid_mask: np.ndarray,
        current_target_idx: List[int],
        target_pos_norm: List[np.ndarray],
        target_pos_abs: List[np.ndarray],
        scale: float,
        global_state: Optional[np.ndarray] = None,
        eval_mode: bool = False,
    ):
        num_usvs = len(current_target_idx)
        re_log_probs = {}

        for i in usv_indices:
            # 结算旧的 pending decision
            if not eval_mode and i in self._pending_decisions:
                dec = self._pending_decisions.pop(i)
                dec['cum_reward'] += 5.0  # 到达奖励
                dec['reached'] = True
                self.buffer.add(
                    dec['sel_state'], dec['action'], dec['log_prob'],
                    dec['value'], dec['cum_reward'],
                    False,  # 非终止
                    dec['valid_mask'], dec['global_state'],
                    dec['coord_info'])

            # 选择新目标
            sel_state = get_selector_state(obs_dict[f"usv_{i}"])
            coord = _build_coord_info(
                obs_dict, num_usvs, i,
                current_target_idx, target_pos_norm,
                max_usvs=self._max_usvs)

            t_idx, log_prob = self._select_action(
                sel_state, valid_mask, coord, eval_mode=eval_mode)
            current_target_idx[i] = t_idx
            target_pos_norm[i] = get_target_pos(obs_dict[f"usv_{i}"], t_idx)
            target_pos_abs[i] = target_pos_norm[i] * scale
            re_log_probs[i] = log_prob

            # 创建新的 pending decision
            if not eval_mode and global_state is not None:
                self._pending_decisions[i] = {
                    'sel_state': sel_state,
                    'action': t_idx,
                    'log_prob': log_prob,
                    'value': self.get_value(global_state),
                    'valid_mask': valid_mask.copy(),
                    'global_state': global_state.copy(),
                    'coord_info': coord,
                    'cum_reward': 0.0,
                    'reached': False,
                }

        return re_log_probs if not eval_mode else None

    # ==========================================================================
    # 每步回调: 累积 rewards + makespan 惩罚
    # ==========================================================================

    def on_step(self, rewards_arr: np.ndarray, info: dict):
        for i, dec in self._pending_decisions.items():
            if i < len(rewards_arr):
                dec['cum_reward'] += self.gamma_weight * rewards_arr[i]

    # ==========================================================================
    # Episode 结束回调
    # ==========================================================================

    def on_episode_end(self, obs_dict: dict, info: dict):
        """结算所有 pending decisions, 包括 makespan 惩罚."""
        for i, dec in self._pending_decisions.items():
            # 如果 USV 到达了, 加到达奖励
            if dec.get('reached', False):
                dec['cum_reward'] += self.alpha * 5.0

            # Makespan 惩罚 (基于剩余未分配目标)
            # 估计: 剩余目标数 × 平均距离 / 速度
            remaining, avg_dist = self._estimate_remaining(info, obs_dict, i)
            dec['cum_reward'] -= self.beta * remaining * avg_dist / self.usv_speed

            self.buffer.add(
                dec['sel_state'], dec['action'], dec['log_prob'],
                dec['value'], dec['cum_reward'],
                True,  # episode 终止
                dec['valid_mask'], dec['global_state'],
                dec['coord_info'])

        self._pending_decisions = {}

    def _estimate_remaining(self, info, obs_dict, usv_idx):
        """估计剩余 makespan: 剩余目标数 和 平均距离."""
        target_status = info.get('target_status', None)
        if target_status is not None:
            remaining = int(np.sum(target_status == 0))
        else:
            remaining = 1
        # 粗略平均距离估计
        avg_dist = 100.0
        return remaining, avg_dist

    # ==========================================================================
    # PPO 更新
    # ==========================================================================

    def update(self) -> dict:
        if self.buffer.size < 2:
            return {'sel_policy_loss': 0, 'sel_value_loss': 0,
                    'sel_entropy': 0}

        self.buffer.compute_gae(0.0, self.gamma, self.gae_lambda)

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

                adv = ((advantages - advantages.mean())
                       / (advantages.std() + 1e-8))

                # 策略更新
                dist = self.policy.get_dist(sel_states, valid_masks, coord_infos)
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

                # 价值更新
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

    # ==========================================================================
    # 从预训练 DQN 热启动
    # ==========================================================================

    def load_from_dqn(self, dqn_path):
        """从预训练 DQN Selector (SelectorQNet) 的权重热启动."""
        from selector_agent import SelectorQNet
        dqn = SelectorQNet()
        dqn.load_state_dict(
            torch.load(dqn_path, map_location=device, weights_only=True))

        self.policy.target_encoder.load_state_dict(
            dqn.target_encoder.state_dict())

        if not self._use_coordination:
            self.policy.logit_head.load_state_dict(dqn.q_head.state_dict())
        else:
            for i in [2, 4]:
                self.policy.logit_head[i].weight.data.copy_(
                    dqn.q_head[i].weight.data)
                self.policy.logit_head[i].bias.data.copy_(
                    dqn.q_head[i].bias.data)
        print(f"[MakespanSelectorPPO] DQN 热启动完成: {dqn_path}")

    # ==========================================================================
    # 保存/加载
    # ==========================================================================

    def save(self, path: str):
        torch.save({
            'policy': self.policy.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.policy.load_state_dict(ckpt['policy'])
        self.value_net.load_state_dict(ckpt['value_net'])
        if 'policy_optimizer' in ckpt:
            self.policy_optimizer.load_state_dict(ckpt['policy_optimizer'])
        if 'value_optimizer' in ckpt:
            self.value_optimizer.load_state_dict(ckpt['value_optimizer'])

    def state_dict_for_checkpoint(self) -> dict:
        return {
            'policy': self.policy.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }

    def load_from_checkpoint(self, ckpt: dict):
        sp = ckpt.get('assigner') or ckpt.get('sel_ppo')
        if sp is None:
            return
        self.policy.load_state_dict(sp['policy'])
        self.value_net.load_state_dict(sp['value_net'])
        if 'policy_optimizer' in sp:
            self.policy_optimizer.load_state_dict(sp['policy_optimizer'])
        if 'value_optimizer' in sp:
            self.value_optimizer.load_state_dict(sp['value_optimizer'])
