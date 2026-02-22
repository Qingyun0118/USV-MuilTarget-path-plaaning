"""
MPC-Guided Two-Layer HRL for USV Multi-Target Navigation

参考文献:
- Dawood et al., "Handling Sparse Rewards in RL Using MPC", Applied Intelligence, 2023
- Yan et al., "Hierarchical RL for Multi-Goal Navigation", AIR 2024

架构 (简化为两层):
  Selector (DQN) → Actuator (TD3, Goal-Conditioned) → USV Environment
  + MPC 教师在训练初期提供专家示范

核心改进:
1. 去掉 Upper Actuator 层, 减少分层复杂度
2. 训练初期通过 MPC (比例导航 + 反应式避障) 控制 USV, 生成成功经验
3. MPC 概率按 Dawood et al. 指数衰减方式随训练推进下降, RL 逐渐接管控制
4. 行为克隆损失 (BC Loss): 在 MPC transition 上显式教 actor 模仿专家动作
5. Q-filtered BC: 仅在 MPC 动作比当前策略更好时才模仿, 避免模仿劣质 MPC
6. Selector 增加总距离惩罚, 鼓励最短路径目标序列
7. 直接在 3目标 6障碍物 环境中训练 (无课程学习)
8. 移除 HER/HGG/DGT, 简化训练流程
"""

import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 评估指标与雷达图模块
from evaluation_metrics import (
    evaluate_single_episode, evaluate_all_episodes,
    compute_success_rate, print_evaluation_report,
    get_normalized_metrics_for_radar
)
from radar_chart import plot_single_radar, plot_multi_radar, plot_metrics_bar_chart
from torch.utils.tensorboard import SummaryWriter
import datetime
from collections import deque
from my_USV_Env import MyUsvEnv, X_MAX, Y_MAX

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 坐标反归一化因子
SCALE = np.array([X_MAX, Y_MAX])


# ============================================================================
# ========================= Replay Buffers ===================================
# ============================================================================

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
    Goal-Conditioned Replay Buffer for Actuator (TD3)
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


# ============================================================================
# ========================= MPC Controller ===================================
# ============================================================================

class MPCController:
    """
    MPC 控制器: 比例导航 + 反应式避障

    参考 Dawood et al. 2023 的思路, 在训练初期提供 "专家" 示范:
    1. 比例导航: 计算期望航向, 按比例施加油门和舵角
    2. 反应式避障: 利用传感器读数检测障碍物, 进行转向规避
    3. 将 MPC 动作存入 Replay Buffer, 作为执行器 TD3 的学习依据

    Note: MPC 不需要完美, 只需要偶尔成功到达目标即可提供正向经验。
    """

    def __init__(self, kp_heading=2.0, kp_throttle=0.0006,
                 obs_avoid_threshold=0.8, obs_critical_threshold=0.45,
                 min_throttle_for_turn=0.03, max_throttle=0.04):
        """
        MPC 控制器参数

        注意: USV 动力学中阻力极低 (a12=-0.00125), 因此:
        - max_throttle 必须很小 (~0.04) 以限制巡航速度 ~5-6 m/s
        - 检测到障碍物时必须使用负推力主动制动, 不能仅降低油门
        - 制动响应: 从 6m/s 全力制动到停止约需 6 步 (~0.8m)

        Args:
            kp_heading: 航向比例增益
            kp_throttle: 油门比例增益 (极小, 因为 USV 阻力低)
            obs_avoid_threshold: 避障触发阈值 (~12m)
            obs_critical_threshold: 紧急避障阈值 (~6.75m)
            min_throttle_for_turn: 避障转向时最低正油门 (维持舵效)
            max_throttle: 巡航最大油门 (限制 USV 最高速度)
        """
        self.kp_heading = kp_heading
        self.kp_throttle = kp_throttle
        self.obs_avoid_threshold = obs_avoid_threshold
        self.obs_critical_threshold = obs_critical_threshold
        self.min_throttle_for_turn = min_throttle_for_turn
        self.max_throttle = max_throttle

    def compute_action(self, usv_x, usv_y, usv_heading,
                       goal_x, goal_y, sensor_readings=None, usv_u=0.0):
        """
        计算导航动作 (比例导航 + 反应式避障)

        关键设计:
        - USV 动力学: 转向力矩 ∝ T_thrust * sin(δ), 推力为 0 则无法转向
        - 因此即使在避障时也保持最低油门以维持舵效
        - 仅在极度危险(即将碰撞)时才使用倒车

        传感器布局: N 根探测线均匀分布在 150° 扇形 FOV (前向),
        索引 0 = 右侧边缘, 索引 N-1 = 左侧边缘, 中间 = 正前方。

        Returns: np.array([throttle, rudder]) ∈ [-1, 1]²

        *** 关键物理约束 ***
        USV 偏航力矩 = b31 * T_thrust * sin(δ)
        当 T_thrust < 0 (倒车) 时, 舵角方向效果反转!
        → 必须在负油门时反转 rudder 符号以保持预期转向方向。
        """
        # ---- 1. 比例导航 ----
        dx = goal_x - usv_x
        dy = goal_y - usv_y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        desired_heading = np.arctan2(dy, dx)

        heading_error = desired_heading - usv_heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # 基础油门 (限制最大值以控制惯性, USV 阻力系数低易超速)
        throttle = np.clip(self.kp_throttle * dist, 0.005, self.max_throttle)
        if dist < 5.0:
            throttle *= max(0.2, dist / 5.0)

        # 舵角
        rudder = np.clip(self.kp_heading * heading_error / np.pi, -1.0, 1.0)

        # ---- 2. 反应式避障 ----
        if sensor_readings is not None and len(sensor_readings) > 0:
            readings = np.array(sensor_readings)
            min_reading = float(np.min(readings))
            n = len(readings)

            if min_reading < self.obs_avoid_threshold:
                # 5 分区
                n5 = max(n // 5, 1)
                zones = [
                    readings[:n5],              # far_right
                    readings[n5:2 * n5],        # right
                    readings[2 * n5:3 * n5],    # center
                    readings[3 * n5:4 * n5],    # left
                    readings[4 * n5:],           # far_left
                ]
                zone_mins = [float(np.min(z)) for z in zones]
                fr_min, r_min, c_min, l_min, fl_min = zone_mins

                left_openness = float(np.mean(np.concatenate([zones[3], zones[4]])))
                right_openness = float(np.mean(np.concatenate([zones[0], zones[1]])))
                avoid_left = (left_openness >= right_openness)

                # ---- Level 0: 极度危险 (<1.8m) — 全力制动 + 转向 ----
                if min_reading < 0.12:
                    throttle = -1.0
                    rudder = 1.0 if avoid_left else -1.0

                # ---- Level 1: 非常近 (<3.75m) — 强力制动 + 急转 ----
                elif min_reading < 0.25:
                    throttle = -0.7
                    rudder = 0.95 if avoid_left else -0.95

                # ---- Level 2: 接近 (<6.75m) — 中等制动 + 大幅转向 ----
                elif min_reading < self.obs_critical_threshold:
                    if c_min < self.obs_critical_threshold:
                        throttle = -0.35
                        rudder = 0.9 if avoid_left else -0.9
                    elif min(r_min, fr_min) < self.obs_critical_threshold:
                        throttle = -0.2
                        rudder = 0.85
                    elif min(l_min, fl_min) < self.obs_critical_threshold:
                        throttle = -0.2
                        rudder = -0.85
                    else:
                        throttle = -0.15
                        rudder = 0.75 if avoid_left else -0.75

                # ---- Level 3: 警戒 (<12m) — 正油门转向 (保留舵效) ----
                else:
                    # 渐进减速但保持正油门 (正油门下舵角方向正常)
                    brake_ratio = 1.0 - (min_reading - self.obs_critical_threshold) / (
                            self.obs_avoid_threshold - self.obs_critical_threshold)
                    throttle = self.max_throttle * max(0.15, 1.0 - 0.7 * brake_ratio)

                    # 方向偏转
                    if c_min < 0.6:
                        rudder = 0.7 if avoid_left else -0.7
                    elif min(r_min, fr_min) < 0.6:
                        rudder = max(rudder, 0.5)
                    elif min(l_min, fl_min) < 0.6:
                        rudder = min(rudder, -0.5)
                    else:
                        intensity = 0.3 * brake_ratio
                        if avoid_left:
                            rudder = np.clip(rudder + intensity, -1.0, 1.0)
                        else:
                            rudder = np.clip(rudder - intensity, -1.0, 1.0)

        # ============================================================
        # 修正1: 防止倒车 (传感器只朝前, 倒车时完全盲区)
        # 当 u < 0: 覆盖为正油门, 恢复前进
        # 当 u 很小且油门为负: 改为滑行, 防止速度穿越零点
        # ============================================================
        if usv_u < -0.3:
            # 正在倒退, 传感器盲区 — 全力向前刹停倒车
            throttle = self.max_throttle
            rudder = np.clip(self.kp_heading * heading_error / np.pi, -1.0, 1.0)
        elif usv_u < 1.5 and throttle < -0.01:
            # 速度很低时不再主动制动, 用小正油门维持舵效
            throttle = self.min_throttle_for_turn

        # ============================================================
        # 修正2: USV 偏航耦合补偿
        # 偏航力矩 = b31 * T_thrust * sin(δ)
        # 当 T_thrust < 0, sin(δ) 贡献反号
        # → 反转 rudder 以维持预期转向方向
        # ============================================================
        if throttle < -0.01:
            rudder = -rudder

        return np.array([
            np.clip(throttle, -1.0, 1.0),
            np.clip(rudder, -1.0, 1.0)
        ], dtype=np.float32)


# ============================================================================
# ================ MPC Probability Scheduler (Dawood et al.) =================
# ============================================================================

class DawoodMPCScheduler:
    """
    Dawood et al. (2023) 指数衰减 MPC 使用概率调度器

    参考: Handling Sparse Rewards in RL Using Model Predictive Control,
          Applied Intelligence, 2023.

    核心思想:
    - 训练初期以高概率使用 MPC 专家策略, 为 RL 提供成功经验
    - 随训练推进, MPC 概率按指数衰减, RL 逐渐接管控制
    - 指数衰减公式:
        p_MPC(e) = max(p_min, p_0 * exp(-λ * e / E_total))
      其中 λ = ln(p_0 / p_min), 确保在 E_total 集结束时 p_MPC ≈ p_min

    与线性/分阶段衰减相比, 指数衰减的优势:
    1. 训练初期 MPC 概率下降缓慢, 保证充足的专家经验
    2. 中后期衰减加速, 让 RL 更快获得独立探索机会
    3. 平滑过渡, 避免阶段跳变带来的训练不稳定
    """

    def __init__(self, p0=0.9, p_min=0.01, total_episodes=5000,
                 success_window=100):
        """
        Args:
            p0: 初始 MPC 使用概率
            p_min: 最低 MPC 使用概率 (保留少量 MPC 防止遗忘)
            total_episodes: 总训练 episode 数 (用于计算衰减速率)
            success_window: 成功率统计滑动窗口大小
        """
        self.p0 = p0
        self.p_min = p_min
        self.total_episodes = total_episodes
        # λ = ln(p0 / p_min), 确保在 total_episodes 时衰减到 p_min
        self.decay_lambda = np.log(p0 / max(p_min, 1e-8))
        self.success_history = deque(maxlen=success_window)

    def get_mpc_probability(self, episode: int) -> float:
        """
        计算当前 episode 的 MPC 使用概率 (Dawood 指数衰减)

        p_MPC(e) = max(p_min, p_0 * exp(-λ * e / E_total))
        """
        p = self.p0 * np.exp(-self.decay_lambda * episode / self.total_episodes)
        return max(self.p_min, p)

    def record_episode(self, success: bool):
        """记录一个 episode 的成功/失败"""
        self.success_history.append(1.0 if success else 0.0)

    def get_success_rate(self) -> float:
        if len(self.success_history) == 0:
            return 0.0
        return float(np.mean(self.success_history))


# ============================================================================
# ========================= Neural Networks ==================================
# ============================================================================

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


class Actor(nn.Module):
    """TD3 Actor: (State, Goal) → Action"""

    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """TD3 Twin Critic: (State, Goal, Action) → Q1, Q2"""

    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        total_dim = state_dim + goal_dim + action_dim
        # Q1
        self.l1 = nn.Linear(total_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(total_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

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
        return self.l3(q1)


# ============================================================================
# ========================= Agents ===========================================
# ============================================================================

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


class TD3ActuatorAgent:
    """
    TD3 Actuator Agent — Goal-Conditioned

    直接控制 USV 的油门和舵角, 目标为到达 Selector 指定的目标点。
    (State, RelativeGoal) → (Throttle, Rudder)
    """

    def __init__(self, state_dim, goal_dim, action_dim, max_action=1.0,
                 lr_actor=1e-4, lr_critic=3e-4, gamma=0.99, tau=0.005,
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

    def select_action(self, state, goal, noise=0.0):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state_t, goal_t).cpu().numpy().flatten()
        if noise > 0:
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(
                -self.max_action, self.max_action)
        return action

    def train(self, replay_buffer, batch_size=256, bc_weight=0.0):
        """
        TD3 训练 (Goal-Conditioned) + 行为克隆损失 (BC Loss)

        参考 DDPGfD / TD3fD 的做法:
        - 对 MPC 专家生成的 transition, 除了标准 Q-maximization 外,
          额外添加 BC loss 让 actor 直接模仿 MPC 动作
        - BC loss 权重随训练推进衰减 (与 MPC 概率同步)

        联合 Actor 损失:
          L_actor = -E[Q(s,g,π(s,g))] + bc_weight * E_{mpc}[||π(s,g) - a_mpc||²]

        其中第二项仅在 MPC transition 上计算, 仅当 Q(s,g,a_mpc) > Q(s,g,π(s,g))
        时施加 (即只在 MPC 动作确实比当前策略好时才模仿, 避免模仿劣质 MPC 动作)

        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            bc_weight: 行为克隆损失权重 (通常随训练衰减)
        """
        if len(replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0

        state, action, reward, next_state, done, goal, is_mpc = \
            replay_buffer.sample(batch_size)
        self.total_it += 1

        # ---- Critic Update ----
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, goal) + noise).clamp(
                -self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, goal, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---- Delayed Actor Update (with BC Loss) ----
        actor_loss_val = 0.0
        bc_loss_val = 0.0
        if self.total_it % self.policy_freq == 0:
            pi_action = self.actor(state, goal)

            # 标准 RL 目标: 最大化 Q 值
            rl_loss = -self.critic.Q1(state, goal, pi_action).mean()

            # 行为克隆损失: 仅在 MPC transition 上, 且仅当 MPC 动作比
            # 当前策略动作的 Q 值更高时才施加 (filtered BC)
            bc_loss = torch.tensor(0.0, device=device)
            if bc_weight > 0.0 and is_mpc.sum() > 0:
                mpc_mask = (is_mpc > 0.5).squeeze()  # (batch,)
                if mpc_mask.any():
                    mpc_states = state[mpc_mask]
                    mpc_goals = goal[mpc_mask]
                    mpc_actions = action[mpc_mask]  # MPC 专家动作
                    pi_mpc = pi_action[mpc_mask]    # actor 当前输出

                    # Q-filter: 只在 Q(s,g,a_mpc) > Q(s,g,π(s,g)) 时模仿
                    with torch.no_grad():
                        q_mpc = self.critic.Q1(mpc_states, mpc_goals, mpc_actions)
                        q_pi = self.critic.Q1(mpc_states, mpc_goals, pi_mpc)
                        bc_filter = (q_mpc > q_pi).squeeze()  # (n_mpc,)

                    if bc_filter.any():
                        bc_loss = F.mse_loss(
                            pi_mpc[bc_filter], mpc_actions[bc_filter])

            total_actor_loss = rl_loss + bc_weight * bc_loss

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft update target networks
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            actor_loss_val = rl_loss.item()
            bc_loss_val = bc_loss.item()

        return critic_loss.item(), actor_loss_val, bc_loss_val


# ============================================================================
# ========================= Helper Functions =================================
# ============================================================================

def get_selector_state(obs_dict):
    """Selector 状态: USV 位置 (2) + 所有目标信息 (3 * num_targets)"""
    return np.concatenate([obs_dict['o_usv'][0:2], obs_dict['targets']])


def get_actuator_state(obs_dict):
    """Actuator 状态: USV 完整状态 (7) + 传感器 (15) + 海流 (4) = 26"""
    return np.concatenate([obs_dict['o_usv'], obs_dict['o_detector'], obs_dict['o_current']])


def get_usv_pos(obs_dict):
    """获取归一化 USV 位置"""
    return obs_dict['o_usv'][0:2].copy()


def get_target_pos(obs_dict, target_idx):
    """获取归一化目标位置"""
    base = target_idx * 3
    return np.array([obs_dict['targets'][base], obs_dict['targets'][base + 1]])


def compute_actuator_reward(prev_dist, next_dist, min_sensor, crush, sub_done):
    """
    计算执行器奖励

    Args:
        prev_dist: 上一步到目标的归一化距离
        next_dist: 当前步到目标的归一化距离
        min_sensor: 传感器最小读数 (归一化, 0=碰撞)
        crush: 是否碰撞
        sub_done: 是否到达目标区域
    """
    # 接近奖励
    approach = (prev_dist - next_dist) * 10.0
    # 障碍物接近惩罚
    proximity_penalty = -max(0.0, 0.3 - min_sensor) * 5.0

    if crush:
        return -10.0
    elif sub_done:
        return approach + 5.0
    else:
        return approach + proximity_penalty - 0.01


# ============================================================================
# ========================= Training Loop ====================================
# ============================================================================

def set_global_seed(seed: int):
    """
    固定所有随机源, 确保训练可复现

    覆盖:
    - Python random
    - NumPy
    - PyTorch CPU / CUDA
    - cuDNN 确定性模式
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  #会降低GPU性能, 但保证完全可复现
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_mpc_hrl(args):
    """
    MPC-Guided Two-Layer HRL 训练主循环

    流程:
    1. 固定  目标  障碍物环境
    2. Selector 选择目标
    3. Actuator 导航到目标 (MPC 或 RL 决策, MPC 概率按 Dawood 指数衰减)
    4. 经验存入 Replay Buffer
    5. 训练 Actuator (TD3) 和 Selector (DQN)
    """
    # ---- 0. 全局种子固定 (可复现性) ----
    set_global_seed(args.seed)
    print(f"Global seed set to {args.seed} (deterministic mode)")

    # ---- 1. 环境初始化 ----
    NUM_TARGETS = args.train_targets
    NUM_OBSTACLES = args.train_obstacles
    env = MyUsvEnv(
        num_targets=NUM_TARGETS,
        num_obstacles=NUM_OBSTACLES,
        seed=args.seed,
        record_flag=False,
        usv_length=args.usv_length,
        usv_width=args.usv_width,
        target_radius=args.target_radius,
        obstacle_radius_range=(args.obstacle_radius_min, args.obstacle_radius_max),
        use_random_shapes=args.use_random_shapes,
        current_type="generated"
    )

    # 获取维度
    obs = env.reset()
    selector_state_dim = get_selector_state(obs).shape[0]  # 2 + 3*3 = 11
    actuator_state_dim = get_actuator_state(obs).shape[0]  # 7 + 15 + 4 = 26
    actuator_goal_dim = 2  # 相对目标位置
    actuator_action_dim = 2  # (throttle, rudder)

    print(f"State Dims — Selector: {selector_state_dim}, Actuator: {actuator_state_dim}")
    print(f"Goal Dim: {actuator_goal_dim}, Action Dim: {actuator_action_dim}")
    print(f"Environment: {NUM_TARGETS} targets, {NUM_OBSTACLES} obstacles")
    print(f"Device: {device}")

    # ---- 2. 智能体初始化 ----
    selector = SelectorAgent(
        selector_state_dim, NUM_TARGETS,
        lr=args.lr_selector, epsilon=args.epsilon_start
    )
    actuator = TD3ActuatorAgent(
        actuator_state_dim, actuator_goal_dim, actuator_action_dim,
        max_action=1.0,
        lr_actor=args.lr_actor, lr_critic=args.lr_critic,
        tau=args.tau, policy_noise=args.policy_noise,
        noise_clip=args.noise_clip, policy_freq=args.policy_freq
    )

    # ---- 3. Replay Buffers ----
    selector_buffer = ReplayBuffer(args.selector_buffer_size)
    actuator_buffer = GoalConditionedReplayBuffer(args.actuator_buffer_size)

    # ---- 4. MPC Controller ----
    mpc = MPCController()

    # ---- 5. Dawood MPC Probability Scheduler ----
    mpc_scheduler = DawoodMPCScheduler(
        p0=args.mpc_p0,
        p_min=args.mpc_p_min,
        # total_episodes=args.max_episodes,
        total_episodes=1000,
        success_window=args.success_window
    )

    # ---- 6. Logging ----
    run_name = f'MPC_HRL_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(os.path.join(args.output_dir, 'runs', run_name))

    global_step = 0
    best_reward = -float('inf')
    start_episode = 0

    # 训练历史记录 (用于样本效率评估)
    training_history = {
        'episode_steps': [],
        'episode_successes': [],
    }

    # ---- 7. 从检查点恢复训练 (可选) ----
    if getattr(args, 'resume', None):
        resume_path = args.resume
        # 支持直接传 episode 数字, 自动推断路径
        if resume_path.isdigit():
            resume_path = os.path.join(
                args.output_dir, "checkpoints", f"ep_{resume_path}",
                "checkpoint.pth")
        elif os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, "checkpoint.pth")

        if os.path.exists(resume_path):
            print(f"\n[Resume] 正在从检查点恢复: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)

            # 恢复网络权重
            selector.q_net.load_state_dict(ckpt['selector_q_net'])
            selector.target_q_net.load_state_dict(ckpt['selector_target_q_net'])
            actuator.actor.load_state_dict(ckpt['actuator_actor'])
            actuator.actor_target.load_state_dict(ckpt['actuator_actor_target'])
            actuator.critic.load_state_dict(ckpt['actuator_critic'])
            actuator.critic_target.load_state_dict(ckpt['actuator_critic_target'])

            # 恢复优化器状态
            selector.optimizer.load_state_dict(ckpt['selector_optimizer'])
            actuator.actor_optimizer.load_state_dict(ckpt['actuator_actor_optimizer'])
            actuator.critic_optimizer.load_state_dict(ckpt['actuator_critic_optimizer'])

            # 恢复训练状态
            start_episode = ckpt['episode'] + 1
            global_step = ckpt['global_step']
            best_reward = ckpt['best_reward']
            selector.epsilon = ckpt['epsilon']
            actuator.total_it = ckpt['actuator_total_it']

            # 恢复训练历史
            training_history = ckpt.get('training_history', training_history)

            # 恢复 MPC Scheduler 成功历史
            saved_success_history = ckpt.get('mpc_success_history', [])
            mpc_scheduler.success_history = deque(
                saved_success_history, maxlen=args.success_window)

            print(f"[Resume] 从 Episode {start_episode} 继续训练")
            print(f"[Resume] global_step={global_step}, "
                  f"best_reward={best_reward:.2f}, "
                  f"epsilon={selector.epsilon:.4f}, "
                  f"success_rate={mpc_scheduler.get_success_rate():.2f}")
        else:
            print(f"\n[Resume] 检查点不存在: {resume_path}, 将从头开始训练")

    print(f"\n{'=' * 60}")
    print(f"  Training Start — {NUM_TARGETS}T/{NUM_OBSTACLES}O")
    print(f"  MPC Decay: Dawood Exponential (p0={args.mpc_p0}, p_min={args.mpc_p_min})")
    print(f"  Max Episodes: {args.max_episodes}")
    if start_episode > 0:
        print(f"  Resuming from Episode: {start_episode}")
    print(f"{'=' * 60}\n")

    # ======================== Episode Loop ========================
    for episode in range(start_episode, args.max_episodes):
        # ---- 环境重置 (固定 3 目标 6 障碍物) ----
        active_targets = NUM_TARGETS

        # 前 N 个 episode 使用固定种子 (稳定学习), 之后使用变化种子 (泛化)
        if episode < args.fixed_env_episodes:
            obs_dict = env.reset(seed=args.seed)
        else:
            obs_dict = env.reset(seed=args.seed + episode)

        # ---- MPC 概率 (Dawood 指数衰减) & 探索噪声 ----
        mpc_prob = mpc_scheduler.get_mpc_probability(episode)
        exploration_noise = args.exploration_noise * max(
            0.3, 1.0 - episode / args.noise_decay_episodes)

        # ---- Episode 级跟踪变量 ----
        usv_pos = get_usv_pos(obs_dict)
        prev_pos_abs = usv_pos * SCALE
        episode_total_distance = 0.0
        episode_actuator_reward = 0.0
        episode_selector_reward = 0.0
        targets_reached_count = 0
        mpc_steps = 0
        rl_steps = 0

        ep_done = False  # 所有活跃目标完成
        crush = False  # 碰撞
        total_steps = 0
        max_episode_steps = args.max_steps_per_target * active_targets

        target_status = obs_dict['targets'][2::3]
        valid_mask = (target_status == 0)

        # ==================== Selector Loop ====================
        selector_attempts = 0
        while not (ep_done or crush or total_steps >= max_episode_steps):
            selector_attempts += 1
            if selector_attempts > active_targets + 2:
                break  # 安全退出

            # Selector 状态
            selector_state = get_selector_state(obs_dict)
            selector_start_state = selector_state.copy()

            # Selector 选择目标
            target_idx = selector.select_action(selector_state, valid_mask)
            target_pos_norm = get_target_pos(obs_dict, target_idx)
            target_pos_abs = target_pos_norm * SCALE

            if episode % args.log_interval == 0:
                print(f'  EP:{episode} | '
                      f'Attempt:{selector_attempts} | Target:{target_idx} | '
                      f'Pos:({target_pos_abs[0]:.1f},{target_pos_abs[1]:.1f})')

            selector_step_reward = 0.0
            target_reached = False

            # ==================== Actuator Navigation Loop ====================
            nav_steps = 0
            while not (ep_done or target_reached or crush
                       or nav_steps >= args.max_steps_per_target
                       or total_steps >= max_episode_steps):
                nav_steps += 1
                total_steps += 1
                global_step += 1

                # 当前状态
                actuator_state = get_actuator_state(obs_dict)
                curr_usv_pos = get_usv_pos(obs_dict)
                relative_goal = target_pos_norm - curr_usv_pos
                prev_dist = np.linalg.norm(relative_goal)

                # ---- 动作选择: MPC 或 RL ----
                use_mpc = (random.random() < mpc_prob)

                if use_mpc:
                    action = mpc.compute_action(
                        env.usv.x, env.usv.y, env.usv.heading,
                        target_pos_abs[0], target_pos_abs[1],
                        obs_dict['o_detector'], usv_u=env.usv.u
                    )
                    # 添加小噪声增加多样性
                    action = action + np.random.normal(0, 0.03, size=action.shape)
                    action = np.clip(action, -1.0, 1.0)
                    mpc_steps += 1
                else:
                    action = actuator.select_action(
                        actuator_state, relative_goal, noise=exploration_noise)
                    rl_steps += 1

                # ---- 环境步进 ----
                next_obs_dict, sub_done, crush, truncated, info = env.step(
                    action, target_pos_abs)

                next_usv_pos = get_usv_pos(next_obs_dict)
                next_actuator_state = get_actuator_state(next_obs_dict)
                next_dist = np.linalg.norm(target_pos_norm - next_usv_pos)

                # 累计行走距离
                curr_pos_abs = next_usv_pos * SCALE
                step_distance = np.linalg.norm(curr_pos_abs - prev_pos_abs)
                episode_total_distance += step_distance
                prev_pos_abs = curr_pos_abs.copy()

                # ---- 执行器奖励 ----
                min_sensor = float(np.min(next_obs_dict['o_detector']))
                r_actuator = compute_actuator_reward(
                    prev_dist, next_dist, min_sensor, crush, sub_done)
                episode_actuator_reward += r_actuator

                # ---- 存入 Replay Buffer (带 MPC 标志) ----
                is_terminal = crush or sub_done
                actuator_buffer.push(
                    actuator_state, action, r_actuator,
                    next_actuator_state, float(is_terminal), relative_goal,
                    is_mpc=use_mpc
                )

                # ---- 训练 Actuator (带行为克隆损失) ----
                # BC 权重与 MPC 概率成正比: 前期强模仿, 后期弱模仿
                bc_w = args.bc_weight * mpc_prob
                if len(actuator_buffer) >= args.warmup_steps:
                    for _ in range(args.actuator_train_iters):
                        actuator.train(actuator_buffer, args.actuator_batch_size,
                                       bc_weight=bc_w)

                # ---- 检查目标到达 ----
                reached_indices = info.get('reached_targets', [])
                if len(reached_indices) > 0:
                    for r_idx in reached_indices:
                        targets_reached_count += 1

                    if target_idx in reached_indices:
                        # 选中的目标被到达
                        target_reached = True
                        selector_step_reward = 1.0
                    else:
                        # 非选中目标被意外到达, 给予部分奖励并中断重选
                        target_reached = True
                        selector_step_reward = 0.5

                # 更新状态
                obs_dict = next_obs_dict
                usv_pos = next_usv_pos

                # 检查是否所有活跃目标完成
                target_status = obs_dict['targets'][2::3]
                valid_mask = (target_status == 0)
                ep_done = not np.any(valid_mask)

            # ---- End of Actuator Navigation Loop ----

            # ---- Selector 奖励计算 ----
            if ep_done:
                # 所有目标完成 — 基础大奖励 + 总距离惩罚
                max_possible_dist = active_targets * np.sqrt(X_MAX ** 2 + Y_MAX ** 2)
                distance_ratio = episode_total_distance / max_possible_dist
                distance_penalty = -args.distance_penalty_coeff * distance_ratio
                selector_step_reward += 2.0 + distance_penalty

            if not target_reached and not ep_done:
                selector_step_reward = -0.1  # 未到达目标

            # 存入 Selector Buffer
            next_selector_state = get_selector_state(obs_dict)
            selector_buffer.push(
                selector_start_state, target_idx, selector_step_reward,
                next_selector_state, float(ep_done)
            )

            # 训练 Selector
            if len(selector_buffer) > args.selector_batch_size:
                selector.train(selector_buffer, args.selector_batch_size)

            episode_selector_reward += selector_step_reward

        # ==================== End of Selector Loop ====================

        # ---- Epsilon 衰减 & Target 网络更新 ----
        selector.epsilon = max(args.epsilon_end,
                               selector.epsilon * args.epsilon_decay)
        selector.update_target()

        # ---- 记录成功率 ----
        all_active_reached = (targets_reached_count >= active_targets)
        ep_success = all_active_reached and not crush
        mpc_scheduler.record_episode(ep_success)

        # 记录训练历史 (样本效率)
        training_history['episode_steps'].append(total_steps)
        training_history['episode_successes'].append(bool(ep_success))

        # ---- Logging ----
        success_rate = mpc_scheduler.get_success_rate()

        if episode % args.log_interval == 0 or episode < 10:
            print(f"EP {episode} [{NUM_TARGETS}T/{NUM_OBSTACLES}O] | "
                  f"R_act:{episode_actuator_reward:.1f} R_sel:{episode_selector_reward:.1f} | "
                  f"Tgt:{targets_reached_count}/{active_targets} | "
                  f"Dist:{episode_total_distance:.0f} | "
                  f"Crush:{crush} | "
                  f"MPC/RL:{mpc_steps}/{rl_steps} p={mpc_prob:.3f} | "
                  f"Eps:{selector.epsilon:.3f} | "
                  f"SR:{success_rate:.2f} | "
                  f"Steps:{total_steps}")

        writer.add_scalar('Reward/Actuator', episode_actuator_reward, episode)
        writer.add_scalar('Reward/Selector', episode_selector_reward, episode)
        writer.add_scalar('Metrics/TargetsReached', targets_reached_count, episode)
        writer.add_scalar('Metrics/TotalDistance', episode_total_distance, episode)
        writer.add_scalar('Metrics/Collision', float(crush), episode)
        writer.add_scalar('Metrics/SuccessRate', success_rate, episode)
        writer.add_scalar('Metrics/AllReached', float(all_active_reached), episode)
        writer.add_scalar('Training/MPC_Probability', mpc_prob, episode)
        writer.add_scalar('Training/MPC_Steps', mpc_steps, episode)
        writer.add_scalar('Training/RL_Steps', rl_steps, episode)
        writer.add_scalar('Training/Epsilon', selector.epsilon, episode)
        writer.add_scalar('Training/ExplorationNoise', exploration_noise, episode)
        writer.add_scalar('Training/BC_Weight', args.bc_weight * mpc_prob, episode)
        writer.add_scalar('Buffer/Actuator', len(actuator_buffer), episode)
        writer.add_scalar('Buffer/Selector', len(selector_buffer), episode)

        # ---- 定期保存检查点 (完整训练快照, 支持断点续训) ----
        if (episode + 1) % args.checkpoint_interval == 0:
            ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"ep_{episode + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            checkpoint = {
                # 网络权重 (含 target 网络)
                'selector_q_net': selector.q_net.state_dict(),
                'selector_target_q_net': selector.target_q_net.state_dict(),
                'actuator_actor': actuator.actor.state_dict(),
                'actuator_actor_target': actuator.actor_target.state_dict(),
                'actuator_critic': actuator.critic.state_dict(),
                'actuator_critic_target': actuator.critic_target.state_dict(),
                # 优化器状态
                'selector_optimizer': selector.optimizer.state_dict(),
                'actuator_actor_optimizer': actuator.actor_optimizer.state_dict(),
                'actuator_critic_optimizer': actuator.critic_optimizer.state_dict(),
                # 训练状态
                'episode': episode,
                'global_step': global_step,
                'best_reward': best_reward,
                'epsilon': selector.epsilon,
                'actuator_total_it': actuator.total_it,
                # 训练历史 (样本效率)
                'training_history': training_history,
                # MPC 调度器成功历史
                'mpc_success_history': list(mpc_scheduler.success_history),
            }
            torch.save(checkpoint, os.path.join(ckpt_dir, "checkpoint.pth"))
            # 同时保存独立权重文件 (兼容评估加载)
            torch.save(selector.q_net.state_dict(),
                       os.path.join(ckpt_dir, "selector.pth"))
            torch.save(actuator.actor.state_dict(),
                       os.path.join(ckpt_dir, "actuator_actor.pth"))
            torch.save(actuator.critic.state_dict(),
                       os.path.join(ckpt_dir, "actuator_critic.pth"))
            print(f"  [Checkpoint] Saved at episode {episode + 1} (完整快照)")

        # ---- 保存最佳模型 ----
        total_reward = episode_actuator_reward + episode_selector_reward
        if total_reward > best_reward and not crush and all_active_reached:
            best_reward = total_reward
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(selector.q_net.state_dict(),
                       os.path.join(args.output_dir, "best_selector.pth"))
            torch.save(actuator.actor.state_dict(),
                       os.path.join(args.output_dir, "best_actuator_actor.pth"))
            torch.save(actuator.critic.state_dict(),
                       os.path.join(args.output_dir, "best_actuator_critic.pth"))



    # ======================== Training Complete ========================
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(selector.q_net.state_dict(),
               os.path.join(args.output_dir, "selector.pth"))
    torch.save(actuator.actor.state_dict(),
               os.path.join(args.output_dir, "actuator_actor.pth"))
    torch.save(actuator.critic.state_dict(),
               os.path.join(args.output_dir, "actuator_critic.pth"))

    # 保存训练历史 (样本效率评估需要)
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f)
    print(f"训练历史已保存至: {history_path}")

    writer.close()
    print(f"\nTraining Complete. Episodes: {args.max_episodes}")
    print(f"Best Reward: {best_reward:.2f}")


# ============================================================================
# ========================= Evaluation =======================================
# ============================================================================

def evaluate_mpc_hrl(args, eval_mode='rl'):
    """
    评估函数

    Args:
        args: 命令行参数
        eval_mode: 评估模式
            - 'rl':  执行器使用纯 RL 策略 (无 MPC)
            - 'mpc': 执行器使用纯 MPC 控制, 决策器仍使用 RL
    """
    mode_desc = {'rl': '纯 RL Actuator', 'mpc': 'MPC Actuator + RL Selector'}
    print("=" * 60)
    print(f"  Evaluation — {mode_desc.get(eval_mode, eval_mode)}")
    print("=" * 60)

    num_targets = args.eval_targets

    # 1. 环境初始化
    env = MyUsvEnv(
        num_targets=num_targets,
        num_obstacles=args.eval_obstacles,
        seed=args.eval_seed,
        record_flag=False,
        usv_length=args.usv_length,
        usv_width=args.usv_width,
        target_radius=args.target_radius,
        obstacle_radius_range=(args.obstacle_radius_min, args.obstacle_radius_max),
        use_random_shapes=args.use_random_shapes,
        current_type="generated"
    )

    obs = env.reset()
    selector_state_dim = get_selector_state(obs).shape[0]
    actuator_state_dim = get_actuator_state(obs).shape[0]

    # 2. 创建 Agent 并加载模型
    selector = SelectorAgent(selector_state_dim, num_targets)
    actuator = TD3ActuatorAgent(actuator_state_dim, 2, 2, max_action=1.0)
    mpc = MPCController()  # MPC 控制器 (mpc 模式下使用)

    model_dir = args.output_dir
    model_prefix = "best_" if os.path.exists(
        os.path.join(model_dir, "best_selector.pth")) else ""

    try:
        selector.q_net.load_state_dict(torch.load(
            os.path.join(model_dir, f"{model_prefix}selector.pth"),
            map_location=device, weights_only=False))
        if eval_mode == 'rl':
            actuator.actor.load_state_dict(torch.load(
                os.path.join(model_dir, f"{model_prefix}actuator_actor.pth"),
                map_location=device, weights_only=False))
            print("Models loaded successfully (Selector + Actuator).")
        else:
            print("Selector model loaded. Actuator using MPC control.")
    except FileNotFoundError:
        print("WARNING: No models found, running with random weights.")

    eval_dir = os.path.join(args.output_dir, "evaluation_results",
                             f"eval_{eval_mode}")
    os.makedirs(eval_dir, exist_ok=True)

    save_csv = getattr(args, 'eval_save_csv', False)
    tb_motion = getattr(args, 'eval_tb_motion', False)
    eval_results = []
    total_eval_start = time.time()

    # TensorBoard: 记录每步 USV 运动参数 (可选)
    eval_writer = None
    eval_tb_dir = None
    if tb_motion:
        eval_tb_dir = os.path.join(eval_dir, 'runs',
                                   f'eval_{eval_mode}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
        eval_writer = SummaryWriter(eval_tb_dir)
        print(f"  TensorBoard eval log: {eval_tb_dir}")

    for ep in range(args.eval_episodes):
        obs_dict = env.reset(seed=args.eval_seed + ep)
        csv_file = os.path.join(eval_dir, f"mpc_hrl_path_{eval_mode}_ep_{ep + 1}.csv")
        if save_csv and os.path.exists(csv_file):
            os.remove(csv_file)

        ep_done = False
        crush = False
        targets_reached = 0
        total_distance = 0.0
        prev_pos_abs = get_usv_pos(obs_dict) * SCALE
        total_steps = 0
        max_steps = args.max_steps_per_target * num_targets * 2  # 评估时允许更多步数
        target_order = []
        ep_start_time = time.time()

        print(f"\n--- Episode {ep + 1} ---")

        while not (ep_done or crush or total_steps >= max_steps):
            target_status = obs_dict['targets'][2::3]
            valid_mask = (target_status == 0)
            if not np.any(valid_mask):
                ep_done = True
                break

            # Selector 选目标 (评估模式, 无随机)
            selector_state = get_selector_state(obs_dict)
            target_idx = selector.select_action(selector_state, valid_mask, eval_mode=True)
            target_pos_norm = get_target_pos(obs_dict, target_idx)
            target_pos_abs = target_pos_norm * SCALE

            print(f"  Target {target_idx}: ({target_pos_abs[0]:.1f}, {target_pos_abs[1]:.1f})")

            # 导航到目标
            target_reached = False
            nav_steps = 0
            while not (ep_done or target_reached or crush
                       or nav_steps >= args.max_steps_per_target
                       or total_steps >= max_steps):
                nav_steps += 1
                total_steps += 1

                actuator_state = get_actuator_state(obs_dict)
                curr_usv_pos = get_usv_pos(obs_dict)
                relative_goal = target_pos_norm - curr_usv_pos

                if eval_mode == 'mpc':
                    # MPC 控制执行器
                    action = mpc.compute_action(
                        env.usv.x, env.usv.y, env.usv.heading,
                        target_pos_abs[0], target_pos_abs[1],
                        obs_dict['o_detector'], usv_u=env.usv.u
                    )
                else:
                    # RL 控制执行器
                    action = actuator.select_action(
                        actuator_state, relative_goal, noise=0.0)

                next_obs_dict, sub_done, crush, truncated, info = env.step(
                    action, target_pos_abs)

                # ---- TensorBoard: 记录 USV 运动参数 (每步, 可选) ----
                if tb_motion:
                    _tb_tag = f'ep{ep+1}'  # 按 episode 分组
                    _tb_step = total_steps
                    # 动作
                    eval_writer.add_scalar(f'{_tb_tag}/Action/throttle', float(action[0]), _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Action/rudder', float(action[1]), _tb_step)
                    # 位置
                    eval_writer.add_scalar(f'{_tb_tag}/Position/x', env.usv.x, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Position/y', env.usv.y, _tb_step)
                    # 航向角
                    eval_writer.add_scalar(f'{_tb_tag}/Heading/rad', env.usv.heading, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Heading/deg', np.degrees(env.usv.heading), _tb_step)
                    # 速度 (体坐标系)
                    eval_writer.add_scalar(f'{_tb_tag}/Velocity/surge_u', env.usv.u, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Velocity/sway_v', env.usv.v, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Velocity/yaw_rate_r', env.usv.r, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Velocity/speed',
                                           np.sqrt(env.usv.u**2 + env.usv.v**2), _tb_step)
                    # 传感器 (障碍物距离)
                    _sensor = np.array(env.usv.sensor_readings)
                    eval_writer.add_scalar(f'{_tb_tag}/Sensor/min_distance', _sensor.min(), _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Sensor/mean_distance', _sensor.mean(), _tb_step)
                    for _si, _sd in enumerate(_sensor):
                        eval_writer.add_scalar(f'{_tb_tag}/Sensor/ray_{_si}', _sd, _tb_step)
                    # 海流
                    _flow_u, _flow_v = env.get_flow_velocity(env.usv.x, env.usv.y, env.current_time)
                    eval_writer.add_scalar(f'{_tb_tag}/Flow/u', _flow_u, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Flow/v', _flow_v, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Flow/speed',
                                           np.sqrt(_flow_u**2 + _flow_v**2), _tb_step)
                    # 到当前目标的距离
                    _dist_to_target = np.linalg.norm(
                        np.array([env.usv.x, env.usv.y]) - target_pos_abs)
                    eval_writer.add_scalar(f'{_tb_tag}/Navigation/dist_to_target', _dist_to_target, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Navigation/current_target_idx', target_idx, _tb_step)

                # 记录路径
                if save_csv:
                    env.record_points(np.array([env.usv.x, env.usv.y]),
                                      csv_file, heading=env.usv.heading)

                # 距离跟踪
                curr_pos_abs = get_usv_pos(next_obs_dict) * SCALE
                total_distance += np.linalg.norm(curr_pos_abs - prev_pos_abs)
                prev_pos_abs = curr_pos_abs.copy()

                obs_dict = next_obs_dict

                # 检查目标
                reached_indices = info.get('reached_targets', [])
                if len(reached_indices) > 0:
                    for r_idx in reached_indices:
                        targets_reached += 1
                        target_order.append(r_idx)
                        print(f"    Reached target {r_idx}!")
                    if target_idx in reached_indices:
                        target_reached = True

                # 检查全部完成
                target_status = obs_dict['targets'][2::3]
                valid_mask = (target_status == 0)
                ep_done = not np.any(valid_mask)

        env.close_writer()

        # TensorBoard: 记录 episode 级汇总 (可选)
        if tb_motion:
            eval_writer.add_scalar('Summary/targets_reached', targets_reached, ep + 1)
            eval_writer.add_scalar('Summary/total_distance', total_distance, ep + 1)
            eval_writer.add_scalar('Summary/collision', float(crush), ep + 1)
            eval_writer.add_scalar('Summary/total_steps', total_steps, ep + 1)
            eval_writer.add_scalar('Summary/success',
                                   float(targets_reached >= num_targets and not crush), ep + 1)

        ep_elapsed = time.time() - ep_start_time

        result = {
            'episode': ep + 1,
            'targets_reached': targets_reached,
            'total_distance': total_distance,
            'collision': crush,
            'total_steps': total_steps,
            'target_order': target_order,
            'success': (targets_reached >= num_targets and not crush),
            'time': ep_elapsed
        }
        eval_results.append(result)

        print(f"  Result: {targets_reached}/{num_targets} targets | "
              f"Distance: {total_distance:.1f} | "
              f"Collision: {crush} | "
              f"Steps: {total_steps} | "
              f"Time: {ep_elapsed:.2f}s | "
              f"Order: {target_order}")

        #绘制路径 (仅在未指定 --eval_save_csv 时绘制, 该参数表示只保存 CSV)
        if save_csv and os.path.exists(csv_file):
            try:
                env.draw_path(csv_file, ifshow=False, ifsave=True, savePath=eval_dir)
            except Exception as e:
                print(f"  Warning: Draw path failed: {e}")

    # 关闭评估 TensorBoard writer (可选)
    if tb_motion and eval_writer is not None:
        eval_writer.close()
        print(f"  TensorBoard eval log saved to: {eval_tb_dir}")

    # 汇总
    total_eval_time = time.time() - total_eval_start
    print(f"\n{'=' * 60}")
    print(f"  Evaluation Summary [{eval_mode.upper()} mode]")
    print(f"{'=' * 60}")
    print(f"  Environment:  {num_targets} targets, {args.eval_obstacles} obstacles")
    print(f"  Episodes:     {len(eval_results)}")
    success_count = sum(1 for r in eval_results if r['success'])
    collision_count = sum(1 for r in eval_results if r['collision'])
    success_results = [r for r in eval_results if r['success']]
    print(f"  Success Rate: {success_count}/{len(eval_results)} "
          f"({100 * success_count / max(len(eval_results), 1):.0f}%)")
    if success_results:
        avg_distance = np.mean([r['total_distance'] for r in success_results])
        avg_time = np.mean([r['time'] for r in success_results])
        print(f"  Avg Distance: {avg_distance:.1f}  (success only)")
        print(f"  Avg Time:     {avg_time:.2f}s per episode  (success only)")
    else:
        print(f"  Avg Distance: N/A  (no successful episode)")
        print(f"  Avg Time:     N/A  (no successful episode)")
    print(f"  Total Time:   {total_eval_time:.2f}s")
    print(f"  Collisions:   {collision_count}")

    # ====================================================================
    # 综合评估: 6 项指标计算 + 雷达图
    # ====================================================================
    if save_eval_report := getattr(args, 'save_eval_report', True):
        print(f"\n--- 运行综合指标评估 (6 项指标) ---")

        # 海流速度获取函数
        flow_fn = lambda x, y: env.get_flow_velocity(x, y)

        # 加载训练历史 (用于样本效率评估, 仅训练后可用)
        history_path = os.path.join(args.output_dir, "training_history.json")
        episode_steps_train = None
        episode_successes_train = None
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            episode_steps_train = history.get('episode_steps')
            episode_successes_train = history.get('episode_successes')
            print(f"已加载训练历史 ({len(episode_steps_train)} episodes)")
        else:
            print(f"未找到训练历史 ({history_path}), 样本效率指标将跳过")

        # 运行综合评估
        report = evaluate_all_episodes(
            eval_results=eval_results,
            csv_dir=eval_dir,
            env=env,
            eval_mode=eval_mode,
            flow_fn=flow_fn,
            dt=env.dt,
            episode_steps_train=episode_steps_train,
            episode_successes_train=episode_successes_train,
        )

        # 打印评估报告
        algo_name = 'MPC-HRL (RL)' if eval_mode == 'rl' else 'MPC-HRL (MPC)'
        print_evaluation_report(report, algorithm_name=algo_name)

        # 获取雷达图数据并绘制
        radar_metrics = get_normalized_metrics_for_radar(report)
        radar_save = os.path.join(eval_dir, f"radar_{eval_mode}.png")
        plot_single_radar(
            radar_metrics,
            algorithm_name=algo_name,
            save_path=radar_save,
            show=False
        )

        # 保存评估结果到文本 (追加模式, 带时间戳和参数)
        result_file = os.path.join(eval_dir, f"evaluation_report_{eval_mode}.txt")
        with open(result_file, 'a', encoding='utf-8') as f:
            import io, contextlib
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n{'=' * 70}\n")
            f.write(f"  评估时间: {timestamp}\n")
            f.write(f"{'=' * 70}\n")
            # 环境参数
            f.write(f"\n--- 环境参数 ---\n")
            f.write(f"  评估模式:       {eval_mode}\n")
            f.write(f"  目标数量:       {args.eval_targets}\n")
            f.write(f"  障碍物数量:     {args.eval_obstacles}\n")
            f.write(f"  评估 episodes:  {args.eval_episodes}\n")
            f.write(f"  评估种子:       {args.eval_seed}\n")
            f.write(f"  USV 尺寸:       {args.usv_length} x {args.usv_width} m\n")
            f.write(f"  目标半径:       {args.target_radius} m\n")
            f.write(f"  障碍物半径:     {args.obstacle_radius_min}~{args.obstacle_radius_max} m\n")
            f.write(f"  随机形状:       {args.use_random_shapes}\n")
            f.write(f"  模型目录:       {args.output_dir}\n")
            # 训练参数 (重要)
            f.write(f"\n--- 训练参数 ---\n")
            f.write(f"  训练 episodes:  {args.max_episodes}\n")
            f.write(f"  每目标最大步数: {args.max_steps_per_target}\n")
            f.write(f"  Actor LR:       {args.lr_actor}\n")
            f.write(f"  Critic LR:      {args.lr_critic}\n")
            f.write(f"  Selector LR:    {args.lr_selector}\n")
            f.write(f"  TD3 tau:        {args.tau}\n")
            f.write(f"  Policy noise:   {args.policy_noise}\n")
            f.write(f"  MPC p0:         {args.mpc_p0}\n")
            f.write(f"  MPC p_min:      {args.mpc_p_min}\n")
            f.write(f"  BC weight:      {args.bc_weight}\n")
            f.write(f"  Seed:           {args.seed}\n")
            # 汇总
            f.write(f"\n--- 评估结果汇总 ---\n")
            f.write(f"  Success Rate: {success_count}/{len(eval_results)} "
                    f"({100 * success_count / max(len(eval_results), 1):.0f}%)\n")
            if success_results:
                avg_distance = np.mean([r['total_distance'] for r in success_results])
                avg_time = np.mean([r['time'] for r in success_results])
                f.write(f"  Avg Distance: {avg_distance:.1f}  (success only)\n")
                f.write(f"  Avg Time:     {avg_time:.2f}s per episode  (success only)\n")
            else:
                f.write(f"  Avg Distance: N/A  (no successful episode)\n")
                f.write(f"  Avg Time:     N/A  (no successful episode)\n")
            f.write(f"  Total Time:   {total_eval_time:.2f}s\n")
            f.write(f"  Collisions:   {collision_count}\n")
            # 评估报告
            f.write(f"\n--- 综合评估报告 ---\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_evaluation_report(report, algorithm_name=algo_name)
            f.write(buf.getvalue())
            f.write(f"\n{'─' * 70}\n\n")
        print(f"评估报告已追加至: {result_file}")


# ============================================================================
# ========================= Main Entry Point =================================
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MPC-Guided Two-Layer HRL for USV Multi-Target Navigation")

    # --- 环境参数 ---
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--usv_length", default=3.0, type=float)
    parser.add_argument("--usv_width", default=1.5, type=float)
    parser.add_argument("--target_radius", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_min", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_max", default=10.0, type=float)
    parser.add_argument("--use_random_shapes", action="store_true", default=True)
    parser.add_argument("--train_targets", default=3, type=int,
                        help="训练时目标数量")
    parser.add_argument("--train_obstacles", default=6, type=int,
                        help="训练时障碍物数量")

    # --- 训练参数 ---
    parser.add_argument("--max_episodes", default=1500, type=int,
                        help="总训练 episode 数")
    parser.add_argument("--max_steps_per_target", default=1000, type=int,
                        help="导航到单个目标的最大步数")
    parser.add_argument("--warmup_steps", default=1000, type=int,
                        help="Actuator 开始训练前的最小 buffer 数据量")
    parser.add_argument("--fixed_env_episodes", default=0, type=int,
                        help="前 N 个 episode 使用固定种子环境")

    # --- 学习率 ---
    parser.add_argument("--lr_actor", default=1e-4, type=float)
    parser.add_argument("--lr_critic", default=3e-4, type=float)
    parser.add_argument("--lr_selector", default=1e-4, type=float)

    # --- TD3 参数 ---
    parser.add_argument("--tau", default=0.005, type=float,
                        help="Target network soft update 系数")
    parser.add_argument("--policy_noise", default=0.2, type=float,
                        help="TD3 target policy smoothing noise")
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int,
                        help="TD3 延迟策略更新频率")
    parser.add_argument("--actuator_batch_size", default=256, type=int)
    parser.add_argument("--actuator_buffer_size", default=1000000, type=int)
    parser.add_argument("--actuator_train_iters", default=2, type=int,
                        help="每步 Actuator 训练迭代次数")

    # --- Selector 参数 ---
    parser.add_argument("--selector_batch_size", default=64, type=int)
    parser.add_argument("--selector_buffer_size", default=50000, type=int)
    parser.add_argument("--epsilon_start", default=1.0, type=float)
    parser.add_argument("--epsilon_end", default=0.05, type=float)
    parser.add_argument("--epsilon_decay", default=0.999, type=float)
    parser.add_argument("--distance_penalty_coeff", default=1.0, type=float,
                        help="Selector 总距离惩罚系数 (越大越鼓励短路径)")

    # --- 探索参数 ---
    parser.add_argument("--exploration_noise", default=0.3, type=float,
                        help="RL 策略探索噪声标准差")
    parser.add_argument("--noise_decay_episodes", default=3000, type=int,
                        help="探索噪声衰减到最低值的 episode 数")

    # --- MPC 概率衰减参数 (Dawood et al. 指数衰减) ---
    parser.add_argument("--mpc_p0", default=1.0, type=float,
                        help="MPC 初始使用概率")
    parser.add_argument("--mpc_p_min", default=0.5, type=float,
                        help="MPC 最低使用概率")
    parser.add_argument("--success_window", default=100, type=int,
                        help="成功率统计滑动窗口大小")

    # --- 行为克隆参数 (BC Loss) ---
    parser.add_argument("--bc_weight", default=3.0, type=float,
                        help="行为克隆损失基础权重 (实际权重 = bc_weight * mpc_prob)")

    # --- 输出参数 ---
    parser.add_argument("--output_dir", default="mpc_hrl_output", type=str)
    parser.add_argument("--checkpoint_interval", default=200, type=int)
    parser.add_argument("--log_interval", default=50, type=int,
                        help="每 N 个 episode 打印日志")

    # --- 断点续训 ---
    parser.add_argument("--resume", default=None, type=str,
                        help="从检查点恢复训练。可传入: "
                             "(1) episode 编号, 如 '200'; "
                             "(2) checkpoint 目录, 如 'mpc_hrl_output/checkpoints/ep_200'; "
                             "(3) checkpoint.pth 文件完整路径")

    # --- 模式 ---
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true",
                        help="评估纯 RL 控制效果")
    parser.add_argument("--do_eval_mpc", action="store_true",
                        help="评估纯 MPC 控制效果 (决策器仍用 RL)")
    parser.add_argument("--eval_episodes", default=5, type=int)
    parser.add_argument("--eval_targets", default=3, type=int,
                        help="评估时目标数量")
    parser.add_argument("--eval_obstacles", default=6, type=int,
                        help="评估时障碍物数量")
    parser.add_argument("--eval_seed", default=42, type=int)
    parser.add_argument("--eval_save_csv", action="store_true",
                        help="评估时保存路径 CSV 文件")
    parser.add_argument("--eval_tb_motion", action="store_true",
                        help="评估时用 TensorBoard 记录 USV 运动参数 (油门/航向/传感器等)")
    parser.add_argument("--save_eval_report", action="store_true",
                        help="评估后保存综合评估报告文本")

    args = parser.parse_args()

    if args.do_train:
        train_mpc_hrl(args)

    if args.do_eval:
        evaluate_mpc_hrl(args, eval_mode='rl')

    if args.do_eval_mpc:
        evaluate_mpc_hrl(args, eval_mode='mpc')

    if not args.do_train and not args.do_eval and not args.do_eval_mpc:
        print("请指定 --do_train / --do_eval / --do_eval_mpc")
        print("示例: python MPC_HRL_main.py --do_train")
        print("      python MPC_HRL_main.py --do_eval        # 评估 RL")
        print("      python MPC_HRL_main.py --do_eval_mpc    # 评估 MPC")
