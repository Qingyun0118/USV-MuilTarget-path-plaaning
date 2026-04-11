"""
Expert-Guided Two-Layer HRL for USV Multi-Target Navigation — TD7 版

参考文献:
- Fujimoto et al., "TD7: Revisiting Fundamentals of Off-Policy RL", ICML 2023
- Dawood et al., "Handling Sparse Rewards in RL Using Expert", Applied Intelligence, 2023
- Yan et al., "Hierarchical RL for Multi-Goal Navigation", AIR 2024

架构:
  Selector (DQN) → Actuator (TD7, Goal-Conditioned) → USV Environment
  + Expert 教师在训练初期提供专家示范

TD7 = TD3 + 三项改进 (严格对齐 Fujimoto et al. 2023 开源实现):
1. SALE (State-Action Learned Embeddings):
   - 独立 Encoder 自监督训练, 预测下一状态嵌入
   - fixed_encoder 冻结快照, 供 Actor/Critic 使用
   - 周期性硬重置 (非 EMA): fixed_encoder ← encoder
2. LAP (Loss-Adjusted Priorities): LAP Huber 损失 + td_loss 优先级 (不使用 IS 权重)
3. Checkpoint Policy: 数据收集时 50/50 使用当前/历史最优策略, 防止策略坍塌

与 TD3 版的关键区别:
- Actuator 使用 TD7ActuatorAgent (独立 Encoder + SALE Critic + LAP + Checkpoint)
- Encoder 自监督训练: MSE(encoder.zsa(zs, a), encoder.zs(s', g))
- 周期性硬目标更新 (非 EMA 软更新), 含 encoder → fixed_encoder 同步
- 数据收集: RL 动作时以 50% 概率使用 checkpoint actor + checkpoint encoder
- 保存/恢复: 额外保存 encoder + fixed_encoder + checkpoint 权重
"""

import argparse
import json
import os
import random
import time
import numpy as np
import torch

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

# --- 公共模块导入 ---
from hrl_config import device, SCALE, set_global_seed
from buffers import ReplayBuffer
from reactive_controller import ReactiveController
from expert_scheduler import ExpertScheduler
from selector_agent import SelectorAgent
from hrl_helpers import (
    get_selector_state, get_actuator_state,
    get_usv_pos, get_target_pos, compute_actuator_reward
)
from td7_agent import TD7ActuatorAgent, GoalConditionedLAPBuffer



# ============================================================================
# ========================= Training Loop ====================================
# ============================================================================

def train_expert_hrl(args):
    """
    Expert-Guided Two-Layer HRL 训练主循环 (TD7 版)

    流程:
    1. 固定 目标 障碍物环境
    2. Selector 选择目标
    3. Actuator 导航到目标 (Expert 或 RL 决策, Expert 概率按 Dawood 指数衰减)
       - RL 动作时: 50% 使用当前策略, 50% 使用 checkpoint 策略 (TD7 Checkpoint Policy)
    4. 经验存入 Replay Buffer
    5. 训练 Actuator (TD7: SALE + LAP) 和 Selector (DQN)
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
    actuator_state_dim = get_actuator_state(obs).shape[0]  # 7 + 15 + 4 = 26
    actuator_goal_dim = 2  # 相对目标位置
    actuator_action_dim = 2  # (throttle, rudder)

    print(f"State Dims — Actuator: {actuator_state_dim}")
    print(f"Goal Dim: {actuator_goal_dim}, Action Dim: {actuator_action_dim}")
    print(f"Environment: {NUM_TARGETS} targets, {NUM_OBSTACLES} obstacles")
    print(f"Device: {device}")

    # ---- 2. 智能体初始化 ----
    selector = SelectorAgent(
        lr=args.lr_selector, epsilon=args.epsilon_start
    )
    actuator = TD7ActuatorAgent(
        actuator_state_dim, actuator_goal_dim, actuator_action_dim,
        max_action=1.0,
        lr_actor=args.lr_actor, lr_critic=args.lr_critic,
        lr_encoder=args.lr_encoder,
        gamma=args.gamma, target_update_rate=args.target_update_rate,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip, policy_freq=args.policy_freq,
        zs_dim=args.zs_dim,
        lap_alpha=args.lap_alpha,
        lap_min_priority=args.lap_min_priority
    )

    # ---- 3. Replay Buffers ----
    selector_buffer = ReplayBuffer(args.selector_buffer_size)
    actuator_buffer = GoalConditionedLAPBuffer(
        actuator_state_dim, actuator_goal_dim, actuator_action_dim,
        max_size=args.actuator_buffer_size,
        batch_size=args.actuator_batch_size
    )

    # ---- 4. Expert Controller ----
    expert = ReactiveController()

    # ---- 5. Dawood Expert Probability Scheduler ----
    expert_scheduler = ExpertScheduler(
        p0=args.expert_p0,
        p_min=args.expert_p_min,
        total_episodes=args.max_episodes,
        success_window=args.success_window
    )

    # ---- 6. Logging ----
    run_name = f'Expert_HRL_TD7_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(os.path.join(args.output_dir, 'runs', run_name))

    global_step = 0
    best_reward = -float('inf')
    best_reward_episode = -1
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

            # TD7: 恢复 Encoder 状态 (SALE 核心)
            if 'actuator_encoder' in ckpt:
                actuator.encoder.load_state_dict(ckpt['actuator_encoder'])
                actuator.fixed_encoder.load_state_dict(ckpt['actuator_fixed_encoder'])
                actuator.fixed_encoder_target.load_state_dict(
                    ckpt['actuator_fixed_encoder_target'])
                actuator.checkpoint_encoder.load_state_dict(
                    ckpt['actuator_checkpoint_encoder'])

            # TD7: 恢复 checkpoint actor
            if 'actuator_actor_checkpoint' in ckpt:
                actuator.actor_checkpoint.load_state_dict(
                    ckpt['actuator_actor_checkpoint'])
                actuator.checkpoint_ready = ckpt.get(
                    'checkpoint_ready', True)

            # TD7: 恢复值裁剪范围
            actuator.max_Q = ckpt.get('max_Q', -1e8)
            actuator.min_Q = ckpt.get('min_Q', 1e8)
            actuator.max_Q_target = ckpt.get('max_Q_target', 0)
            actuator.min_Q_target = ckpt.get('min_Q_target', 0)

            # TD7: 恢复 batch training checkpoint 追踪
            actuator.eps_since_update = ckpt.get('eps_since_update', 0)
            actuator.timesteps_since_update = ckpt.get('timesteps_since_update', 0)
            actuator.max_eps_before_update = ckpt.get('max_eps_before_update', 1)
            actuator.mean_return = ckpt.get('mean_return', 0.0)
            actuator.best_mean_return = ckpt.get('best_mean_return', -1e8)

            # 恢复优化器状态
            selector.optimizer.load_state_dict(ckpt['selector_optimizer'])
            actuator.actor_optimizer.load_state_dict(ckpt['actuator_actor_optimizer'])
            actuator.critic_optimizer.load_state_dict(ckpt['actuator_critic_optimizer'])
            if 'actuator_encoder_optimizer' in ckpt:
                actuator.encoder_optimizer.load_state_dict(
                    ckpt['actuator_encoder_optimizer'])

            # 恢复训练状态
            start_episode = ckpt['episode'] + 1
            global_step = ckpt['global_step']
            best_reward = ckpt['best_reward']
            selector.epsilon = ckpt['epsilon']
            actuator.total_it = ckpt['actuator_total_it']

            # 恢复训练历史
            training_history = ckpt.get('training_history', training_history)

            # 恢复 Expert Scheduler 成功历史
            saved_success_history = ckpt.get('expert_success_history', [])
            expert_scheduler.success_history = deque(
                saved_success_history, maxlen=args.success_window)

            print(f"[Resume] 从 Episode {start_episode} 继续训练")
            print(f"[Resume] global_step={global_step}, "
                  f"best_reward={best_reward:.2f}, "
                  f"epsilon={selector.epsilon:.4f}, "
                  f"success_rate={expert_scheduler.get_success_rate():.2f}")
        else:
            print(f"\n[Resume] 检查点不存在: {resume_path}, 将从头开始训练")

    print(f"\n{'=' * 60}")
    print(f"  Training Start — TD7 — {NUM_TARGETS}T/{NUM_OBSTACLES}O")
    print(f"  SALE zs_dim={args.zs_dim}, "
          f"target_update_rate={args.target_update_rate}")
    print(f"  LAP alpha={args.lap_alpha}, min_priority={args.lap_min_priority}")
    print(f"  Checkpoint Policy: batch training + adaptive checkpoint "
          f"(对齐参考实现 maybe_train_and_checkpoint)")
    print(f"  Expert Decay: Dawood Exponential "
          f"(p0={args.expert_p0}, p_min={args.expert_p_min})")
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

        # ---- Expert 概率 (Dawood 指数衰减) & 探索噪声 ----
        effective_ep = max(0, episode - args.fixed_env_episodes)
        expert_prob = expert_scheduler.get_expert_probability(effective_ep)
        # TD7 参考实现使用恒定探索噪声 (确定性策略需要持续探索)
        exploration_noise = args.exploration_noise

        # ---- Episode 级跟踪变量 ----
        usv_pos = get_usv_pos(obs_dict)
        prev_pos_abs = usv_pos * SCALE
        episode_total_distance = 0.0
        episode_actuator_reward = 0.0
        episode_selector_reward = 0.0
        targets_reached_count = 0
        expert_steps = 0
        rl_steps = 0
        checkpoint_steps = 0  # TD7: 跟踪 checkpoint actor 使用次数

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

                # ---- 动作选择: Expert 或 RL (TD7 Checkpoint Policy) ----
                use_expert = (random.random() < expert_prob)

                if use_expert:
                    action = expert.compute_action(
                        env.usv.x, env.usv.y, env.usv.heading,
                        target_pos_abs[0], target_pos_abs[1],
                        obs_dict['o_detector'], usv_u=env.usv.u
                    )
                    # 添加小噪声增加多样性
                    action = action + np.random.normal(0, 0.03, size=action.shape)
                    action = np.clip(action, -1.0, 1.0)
                    expert_steps += 1
                else:
                    # TD7 Checkpoint Policy: 50% 使用当前策略, 50% 使用 checkpoint
                    # 关键: checkpoint 未被有意义地更新前不使用 (随机权重会导致碰撞)
                    use_ckpt = (actuator.checkpoint_ready
                                and random.random() < 0.5)
                    action = actuator.select_action(
                        actuator_state, relative_goal,
                        noise=exploration_noise,
                        use_checkpoint=use_ckpt)
                    rl_steps += 1
                    if use_ckpt:
                        checkpoint_steps += 1

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
                    prev_dist, next_dist, min_sensor, crush, sub_done,
                    usv_pos_norm=info.get('usv_pos_raw_norm', next_usv_pos))
                episode_actuator_reward += r_actuator

                # ---- 存入 Replay Buffer (带 Expert 标志) ----
                is_terminal = crush or sub_done
                actuator_buffer.add(
                    actuator_state, action, r_actuator,
                    next_actuator_state, float(is_terminal), relative_goal,
                    is_expert=use_expert
                )

                # ---- 碰撞回溯惩罚 ----
                if crush:
                    backtrack_steps = min(nav_steps, 10)
                    actuator_buffer.update_last_n_rewards(
                        backtrack_steps + 1,
                        lambda k, old_r: old_r + (-5.0 * (1.0 - (k - 1) / backtrack_steps))
                    )

                # (TD7: 数据收集阶段不训练, episode 结束后批量训练)

                # ---- 检查目标到达 ----
                reached_indices = info.get('reached_targets', [])
                if len(reached_indices) > 0:
                    for r_idx in reached_indices:
                        targets_reached_count += 1

                    if target_idx in reached_indices:
                        target_reached = True
                        selector_step_reward = 1.0
                    else:
                        target_reached = True
                        selector_step_reward = 0.1

                # 更新状态
                obs_dict = next_obs_dict
                usv_pos = next_usv_pos

                # 检查是否所有活跃目标完成
                target_status = obs_dict['targets'][2::3]
                valid_mask = (target_status == 0)
                ep_done = not np.any(valid_mask)

            # ---- End of Actuator Navigation Loop ----

            # ---- Selector 奖励计算 ----
            if crush:
                selector_step_reward = -2.0

            if ep_done:
                max_possible_dist = active_targets * np.sqrt(X_MAX ** 2 + Y_MAX ** 2)
                distance_ratio = episode_total_distance / max_possible_dist
                distance_penalty = -args.distance_penalty_coeff * distance_ratio
                selector_step_reward += 2.0 + distance_penalty

            if not target_reached and not ep_done and not crush:
                selector_step_reward = -0.3

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
        expert_scheduler.record_episode(ep_success)

        # ---- TD7 批量训练 + Checkpoint (对齐参考实现) ----
        # 数据收集阶段不训练, episode 结束后批量训练所有累积 timesteps
        bc_w = args.bc_weight * expert_prob
        if len(actuator_buffer) >= args.warmup_steps:
            actuator.maybe_train_and_checkpoint(
                total_steps, episode_actuator_reward,
                actuator_buffer, args.actuator_batch_size,
                bc_weight=bc_w
            )

        # 记录训练历史 (样本效率)
        training_history['episode_steps'].append(total_steps)
        training_history['episode_successes'].append(bool(ep_success))

        # ---- Logging ----
        success_rate = expert_scheduler.get_success_rate()

        if episode % args.log_interval == 0 or episode < 10:
            print(f"EP {episode} [TD7 {NUM_TARGETS}T/{NUM_OBSTACLES}O] | "
                  f"R_act:{episode_actuator_reward:.1f} "
                  f"R_sel:{episode_selector_reward:.1f} | "
                  f"Tgt:{targets_reached_count}/{active_targets} | "
                  f"Dist:{episode_total_distance:.0f} | "
                  f"Crush:{crush} | "
                  f"Expert/RL/Ckpt:{expert_steps}/{rl_steps}/{checkpoint_steps} "
                  f"p={expert_prob:.3f} | "
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
        writer.add_scalar('Training/Expert_Probability', expert_prob, episode)
        writer.add_scalar('Training/Expert_Steps', expert_steps, episode)
        writer.add_scalar('Training/RL_Steps', rl_steps, episode)
        writer.add_scalar('Training/Epsilon', selector.epsilon, episode)
        writer.add_scalar('Training/ExplorationNoise', exploration_noise, episode)
        writer.add_scalar('Training/BC_Weight', args.bc_weight * expert_prob, episode)
        writer.add_scalar('Buffer/Actuator', len(actuator_buffer), episode)
        writer.add_scalar('Buffer/Selector', len(selector_buffer), episode)
        # TD7 特有指标
        writer.add_scalar('TD7/Checkpoint_Steps', checkpoint_steps, episode)
        writer.add_scalar('TD7/Checkpoint_Ratio',
                          checkpoint_steps / max(rl_steps, 1), episode)
        writer.add_scalar('TD7/Target_Update_Count',
                          actuator.total_it // args.target_update_rate, episode)
        writer.add_scalar('TD7/Max_Q', actuator.max_Q, episode)
        writer.add_scalar('TD7/Min_Q', actuator.min_Q, episode)

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
                'actuator_actor_checkpoint': actuator.actor_checkpoint.state_dict(),
                'actuator_critic': actuator.critic.state_dict(),
                'actuator_critic_target': actuator.critic_target.state_dict(),
                # TD7 Encoder 状态 (SALE)
                'actuator_encoder': actuator.encoder.state_dict(),
                'actuator_fixed_encoder': actuator.fixed_encoder.state_dict(),
                'actuator_fixed_encoder_target': actuator.fixed_encoder_target.state_dict(),
                'actuator_checkpoint_encoder': actuator.checkpoint_encoder.state_dict(),
                # 优化器状态
                'selector_optimizer': selector.optimizer.state_dict(),
                'actuator_actor_optimizer': actuator.actor_optimizer.state_dict(),
                'actuator_critic_optimizer': actuator.critic_optimizer.state_dict(),
                'actuator_encoder_optimizer': actuator.encoder_optimizer.state_dict(),
                # 训练状态
                'episode': episode,
                'global_step': global_step,
                'best_reward': best_reward,
                'epsilon': selector.epsilon,
                'actuator_total_it': actuator.total_it,
                # 训练历史 (样本效率)
                'training_history': training_history,
                # Expert 调度器成功历史
                'expert_success_history': list(expert_scheduler.success_history),
                # TD7 checkpoint 状态
                'checkpoint_ready': actuator.checkpoint_ready,
                # TD7 batch training checkpoint 追踪
                'eps_since_update': actuator.eps_since_update,
                'timesteps_since_update': actuator.timesteps_since_update,
                'max_eps_before_update': actuator.max_eps_before_update,
                'mean_return': actuator.mean_return,
                'best_mean_return': actuator.best_mean_return,
                # TD7 值裁剪范围
                'max_Q': actuator.max_Q,
                'min_Q': actuator.min_Q,
                'max_Q_target': actuator.max_Q_target,
                'min_Q_target': actuator.min_Q_target,
            }
            torch.save(checkpoint, os.path.join(ckpt_dir, "checkpoint.pth"))
            # 同时保存独立权重文件 (兼容评估加载)
            torch.save(selector.q_net.state_dict(),
                       os.path.join(ckpt_dir, "selector.pth"))
            torch.save(actuator.actor.state_dict(),
                       os.path.join(ckpt_dir, "actuator_actor.pth"))
            torch.save(actuator.critic.state_dict(),
                       os.path.join(ckpt_dir, "actuator_critic.pth"))
            torch.save(actuator.actor_checkpoint.state_dict(),
                       os.path.join(ckpt_dir, "actuator_actor_checkpoint.pth"))
            torch.save(actuator.fixed_encoder.state_dict(),
                       os.path.join(ckpt_dir, "actuator_encoder.pth"))
            print(f"  [Checkpoint] Saved at episode {episode + 1} (完整快照)")

        # (TD7 checkpoint 更新已由 maybe_train_and_checkpoint 自动管理)

        # ---- 保存最佳模型 ----
        total_reward = episode_actuator_reward + episode_selector_reward
        if total_reward > best_reward and not crush and all_active_reached:
            best_reward = total_reward
            best_reward_episode = episode
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(selector.q_net.state_dict(),
                       os.path.join(args.output_dir, "best_selector.pth"))
            torch.save(actuator.actor.state_dict(),
                       os.path.join(args.output_dir, "best_actuator_actor.pth"))
            torch.save(actuator.critic.state_dict(),
                       os.path.join(args.output_dir, "best_actuator_critic.pth"))
            torch.save(actuator.fixed_encoder.state_dict(),
                       os.path.join(args.output_dir, "best_actuator_encoder.pth"))
            # TD7: 发现更好策略时更新 checkpoint
            actuator.update_checkpoint()


    # ======================== Training Complete ========================
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(selector.q_net.state_dict(),
               os.path.join(args.output_dir, "selector.pth"))
    torch.save(actuator.actor.state_dict(),
               os.path.join(args.output_dir, "actuator_actor.pth"))
    torch.save(actuator.critic.state_dict(),
               os.path.join(args.output_dir, "actuator_critic.pth"))
    torch.save(actuator.fixed_encoder.state_dict(),
               os.path.join(args.output_dir, "actuator_encoder.pth"))
    torch.save(actuator.actor_checkpoint.state_dict(),
               os.path.join(args.output_dir, "actuator_actor_checkpoint.pth"))

    # 保存训练历史 (样本效率评估需要)
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f)
    print(f"训练历史已保存至: {history_path}")

    writer.close()
    print(f"\nTraining Complete. Episodes: {args.max_episodes}")
    print(f"Best Reward: {best_reward:.2f} at Episode {best_reward_episode}")


# ============================================================================
# ========================= Evaluation =======================================
# ============================================================================

def evaluate_expert_hrl(args, eval_mode='rl'):
    """
    评估函数

    Args:
        args: 命令行参数
        eval_mode: 评估模式
            - 'rl':  执行器使用纯 RL 策略 (无 Expert)
            - 'expert': 执行器使用纯 Expert 控制, 决策器仍使用 RL
    """
    mode_desc = {'rl': '纯 RL Actuator (TD7)', 'expert': 'Expert Actuator + RL Selector'}
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
    actuator_state_dim = get_actuator_state(obs).shape[0]

    # 2. 创建 Agent 并加载模型
    selector = SelectorAgent()
    actuator = TD7ActuatorAgent(
        actuator_state_dim, 2, 2, max_action=1.0,
        zs_dim=args.zs_dim)
    expert = ReactiveController()  # Expert 控制器 (expert 模式下使用)

    default_selector_model = os.path.join(args.output_dir, "best_selector.pth")
    default_actuator_model = os.path.join(args.output_dir, "best_actuator_actor.pth")
    selector_model_path = args.eval_selector_model_path or default_selector_model
    actuator_model_path = args.eval_actuator_model_path or default_actuator_model

    # TD7: encoder 默认使用 output_dir 下 best_encoder;
    # 若传入 actuator actor 路径且同目录存在配套 encoder 文件, 则优先使用该配套 encoder。
    default_encoder_model = os.path.join(args.output_dir, "best_actuator_encoder.pth")
    encoder_model_path = default_encoder_model
    if args.eval_actuator_model_path:
        actor_name = os.path.basename(actuator_model_path)
        inferred_encoder_name = actor_name.replace("actuator_actor", "actuator_encoder")
        inferred_encoder_path = os.path.join(
            os.path.dirname(actuator_model_path), inferred_encoder_name)
        if inferred_encoder_name != actor_name and os.path.exists(inferred_encoder_path):
            encoder_model_path = inferred_encoder_path

    selector_loaded = False
    try:
        selector.q_net.load_state_dict(torch.load(
            selector_model_path, map_location=device, weights_only=False))
        selector_loaded = True
        print(f"Selector model loaded: {selector_model_path}")
    except FileNotFoundError:
        print(f"WARNING: Selector model not found: {selector_model_path}")
    except Exception as e:
        print(f"WARNING: Failed to load selector model ({selector_model_path}): {e}")

    if eval_mode == 'rl':
        actor_loaded = False
        try:
            actuator.actor.load_state_dict(torch.load(
                actuator_model_path, map_location=device, weights_only=False))
            actor_loaded = True
            print(f"TD7 actuator actor loaded: {actuator_model_path}")
        except FileNotFoundError:
            print(f"WARNING: TD7 actuator actor not found: {actuator_model_path}")
        except Exception as e:
            print(f"WARNING: Failed to load TD7 actuator actor ({actuator_model_path}): {e}")

        encoder_loaded = False
        if os.path.exists(encoder_model_path):
            try:
                actuator.fixed_encoder.load_state_dict(torch.load(
                    encoder_model_path, map_location=device, weights_only=False))
                encoder_loaded = True
                print(f"TD7 actuator encoder loaded: {encoder_model_path}")
            except Exception as e:
                print(f"WARNING: Failed to load TD7 actuator encoder ({encoder_model_path}): {e}")
        else:
            print(f"WARNING: TD7 actuator encoder not found: {encoder_model_path}")

        if selector_loaded and actor_loaded and encoder_loaded:
            print("Models loaded successfully (Selector + TD7 Actuator + Encoder).")
        else:
            print("WARNING: Some TD7 evaluation models are missing, corresponding modules use random weights.")
    else:
        if selector_loaded:
            print("Selector model loaded. Actuator using Expert control.")
        else:
            print("WARNING: Selector model unavailable, selector uses random weights. Actuator using Expert control.")

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
        csv_file = os.path.join(eval_dir, f"expert_hrl_path_{eval_mode}_ep_{ep + 1}.csv")
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

                if eval_mode == 'expert':
                    action = expert.compute_action(
                        env.usv.x, env.usv.y, env.usv.heading,
                        target_pos_abs[0], target_pos_abs[1],
                        obs_dict['o_detector'], usv_u=env.usv.u
                    )
                else:
                    # 评估时使用当前 actor (不使用 checkpoint, 无探索噪声)
                    action = actuator.select_action(
                        actuator_state, relative_goal, noise=0.0)

                next_obs_dict, sub_done, crush, truncated, info = env.step(
                    action, target_pos_abs)

                # ---- TensorBoard: 记录 USV 运动参数 (每步, 可选) ----
                if tb_motion:
                    _tb_tag = f'ep{ep+1}'
                    _tb_step = total_steps
                    eval_writer.add_scalar(f'{_tb_tag}/Action/throttle', float(action[0]), _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Action/rudder', float(action[1]), _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Position/x', env.usv.x, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Position/y', env.usv.y, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Heading/rad', env.usv.heading, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Heading/deg', np.degrees(env.usv.heading), _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Velocity/surge_u', env.usv.u, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Velocity/sway_v', env.usv.v, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Velocity/yaw_rate_r', env.usv.r, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Velocity/speed',
                                           np.sqrt(env.usv.u**2 + env.usv.v**2), _tb_step)
                    _sensor = np.array(env.usv.sensor_readings)
                    eval_writer.add_scalar(f'{_tb_tag}/Sensor/min_distance', _sensor.min(), _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Sensor/mean_distance', _sensor.mean(), _tb_step)
                    for _si, _sd in enumerate(_sensor):
                        eval_writer.add_scalar(f'{_tb_tag}/Sensor/ray_{_si}', _sd, _tb_step)
                    _flow_u, _flow_v = env.get_flow_velocity(env.usv.x, env.usv.y, env.current_time)
                    eval_writer.add_scalar(f'{_tb_tag}/Flow/u', _flow_u, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Flow/v', _flow_v, _tb_step)
                    eval_writer.add_scalar(f'{_tb_tag}/Flow/speed',
                                           np.sqrt(_flow_u**2 + _flow_v**2), _tb_step)
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
    print(f"  Evaluation Summary [TD7 {eval_mode.upper()} mode]")
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
    # 综合评估: 评估报告保存
    # ====================================================================
    if save_eval_report := getattr(args, 'save_eval_report', True):
        result_file = os.path.join(eval_dir, f"evaluation_report_{eval_mode}.txt")
        with open(result_file, 'a', encoding='utf-8') as f:
            import io, contextlib
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n{'=' * 70}\n")
            f.write(f"  评估时间: {timestamp}\n")
            f.write(f"  算法: TD7 (SALE + LAP + Checkpoint Policy)\n")
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
            f.write(f"  Selector模型:   {selector_model_path}\n")
            if eval_mode == 'rl':
                f.write(f"  Actuator模型:   {actuator_model_path}\n")
                f.write(f"  Encoder模型:    {encoder_model_path}\n")
            else:
                f.write(f"  Actuator模型:   Expert Controller (不加载RL模型)\n")
            # 训练参数
            f.write(f"\n--- 训练参数 ---\n")
            f.write(f"  训练 episodes:  {args.max_episodes}\n")
            f.write(f"  每目标最大步数: {args.max_steps_per_target}\n")
            f.write(f"  Actor LR:       {args.lr_actor}\n")
            f.write(f"  Critic LR:      {args.lr_critic}\n")
            f.write(f"  Selector LR:    {args.lr_selector}\n")
            f.write(f"  TD7 tau:        N/A (周期性硬更新, rate={args.target_update_rate})\n")
            f.write(f"  Policy noise:   {args.policy_noise}\n")
            f.write(f"  Expert p0:      {args.expert_p0}\n")
            f.write(f"  Expert p_min:   {args.expert_p_min}\n")
            f.write(f"  BC weight:      {args.bc_weight}\n")
            f.write(f"  Seed:           {args.seed}\n")
            # TD7 特有参数
            f.write(f"\n--- TD7 参数 (对齐参考实现) ---\n")
            f.write(f"  SALE zs_dim:                {args.zs_dim}\n")
            f.write(f"  Target update rate:         {args.target_update_rate}\n")
            f.write(f"  Encoder LR:                 {args.lr_encoder}\n")
            f.write(f"  LAP alpha:                  {args.lap_alpha}\n")
            f.write(f"  LAP min_priority:           {args.lap_min_priority}\n")
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
            f.write(f"\n{'─' * 70}\n\n")
        print(f"评估报告已追加至: {result_file}")


# ============================================================================
# ========================= Main Entry Point =================================
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expert-Guided Two-Layer HRL for USV — TD7 (SALE+LAP+Checkpoint)")

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
    parser.add_argument("--max_episodes", default=1000, type=int,
                        help="总训练 episode 数")
    parser.add_argument("--max_steps_per_target", default=1000, type=int,
                        help="导航到单个目标的最大步数")
    parser.add_argument("--warmup_steps", default=5000, type=int,
                        help="Actuator 开始训练前的最小 buffer 数据量 (参考实现: 25000)")
    parser.add_argument("--fixed_env_episodes", default=0, type=int,
                        help="前 N 个 episode 使用固定种子环境")

    # --- 学习率 ---
    parser.add_argument("--lr_actor", default=3e-4, type=float)
    parser.add_argument("--lr_critic", default=3e-4, type=float)
    parser.add_argument("--lr_selector", default=1e-4, type=float)

    # --- TD3 基础参数 ---
    parser.add_argument("--gamma", default=0.99, type=float,
                        help="折扣因子")
    parser.add_argument("--policy_noise", default=0.2, type=float,
                        help="Target policy smoothing noise")
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int,
                        help="延迟策略更新频率")
    parser.add_argument("--actuator_batch_size", default=256, type=int)
    parser.add_argument("--actuator_buffer_size", default=1000000, type=int)

    # --- TD7 特有参数 (对齐 Fujimoto et al. 2023 开源实现) ---
    parser.add_argument("--zs_dim", default=256, type=int,
                        help="SALE Encoder 嵌入维度 (zs, zsa 均为此维度)")
    parser.add_argument("--target_update_rate", default=250, type=int,
                        help="周期性硬目标更新间隔 (training steps, 非 EMA)")
    parser.add_argument("--lr_encoder", default=3e-4, type=float,
                        help="SALE Encoder 学习率")
    parser.add_argument("--lap_alpha", default=0.4, type=float,
                        help="LAP 优先级指数 (priority = td.pow(alpha))")
    parser.add_argument("--lap_min_priority", default=1.0, type=float,
                        help="LAP 最小优先级 (priority = max(td, min_priority))")
    parser.add_argument("--checkpoint_eval_interval", default=50, type=int,
                        help="Checkpoint 策略周期性评估/更新间隔 (episodes)")

    # --- Selector 参数 ---
    parser.add_argument("--selector_batch_size", default=64, type=int)
    parser.add_argument("--selector_buffer_size", default=50000, type=int)
    parser.add_argument("--epsilon_start", default=1.0, type=float)
    parser.add_argument("--epsilon_end", default=0.05, type=float)
    parser.add_argument("--epsilon_decay", default=0.999, type=float)
    parser.add_argument("--distance_penalty_coeff", default=2.0, type=float,
                        help="Selector 总距离惩罚系数")

    # --- 探索参数 (对齐参考实现: 0.1) ---
    parser.add_argument("--exploration_noise", default=0.1, type=float,
                        help="RL 策略探索噪声标准差 (参考实现: 0.1)")
    parser.add_argument("--noise_decay_episodes", default=800, type=int,
                        help="探索噪声衰减到最低值的 episode 数")
    parser.add_argument("--noise_min_ratio", default=0.3, type=float,
                        help="探索噪声最低比例")

    # --- Expert 概率衰减参数 ---
    parser.add_argument("--expert_p0", default=0.5, type=float,
                        help="Expert 初始使用概率 (TD7 建议略高于 TD3, 因 checkpoint 会分流 RL 步数)")
    parser.add_argument("--expert_p_min", default=0.05, type=float,
                        help="Expert 最低使用概率")
    parser.add_argument("--success_window", default=100, type=int,
                        help="成功率统计滑动窗口大小")

    # --- 行为克隆参数 ---
    parser.add_argument("--bc_weight", default=1.0, type=float,
                        help="行为克隆损失基础权重 (实际权重 = bc_weight * expert_prob)")

    # --- 输出参数 ---
    parser.add_argument("--output_dir", default="expert_hrl_td7_output", type=str)
    parser.add_argument("--checkpoint_interval", default=200, type=int)
    parser.add_argument("--log_interval", default=50, type=int,
                        help="每 N 个 episode 打印日志")

    # --- 断点续训 ---
    parser.add_argument("--resume", default=None, type=str,
                        help="从检查点恢复训练。可传入: "
                             "(1) episode 编号, 如 '200'; "
                             "(2) checkpoint 目录; "
                             "(3) checkpoint.pth 文件完整路径")

    # --- 模式 ---
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true",
                        help="评估纯 RL 控制效果")
    parser.add_argument("--do_eval_expert", action="store_true",
                        help="评估纯 Expert 控制效果 (决策器仍用 RL)")
    parser.add_argument("--eval_episodes", default=5, type=int)
    parser.add_argument("--eval_targets", default=3, type=int,
                        help="评估时目标数量")
    parser.add_argument("--eval_obstacles", default=6, type=int,
                        help="评估时障碍物数量")
    parser.add_argument("--eval_seed", default=42, type=int)
    parser.add_argument("--eval_selector_model_path", default=None, type=str,
                        help="评估时 Selector 模型路径 (默认: output_dir/best_selector.pth)")
    parser.add_argument("--eval_actuator_model_path", default=None, type=str,
                        help="评估时 Actuator Actor 模型路径 (默认: output_dir/best_actuator_actor.pth)")
    parser.add_argument("--eval_save_csv", action="store_true",
                        help="评估时保存路径 CSV 文件")
    parser.add_argument("--eval_tb_motion", action="store_true",
                        help="评估时用 TensorBoard 记录 USV 运动参数")
    parser.add_argument("--save_eval_report", action="store_true",
                        help="评估后保存综合评估报告文本")

    args = parser.parse_args()

    if args.do_train:
        train_expert_hrl(args)

    if args.do_eval:
        evaluate_expert_hrl(args, eval_mode='rl')

    if args.do_eval_expert:
        evaluate_expert_hrl(args, eval_mode='expert')

    if not args.do_train and not args.do_eval and not args.do_eval_expert:
        print("请指定 --do_train / --do_eval / --do_eval_expert")
        print("示例: python Expert_HRL_TD7_main.py --do_train")
        print("      python Expert_HRL_TD7_main.py --do_eval        # 评估 RL")
        print("      python Expert_HRL_TD7_main.py --do_eval_expert # 评估 Expert")
