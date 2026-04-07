"""
CTDE MAPPO 训练脚本 — 多 USV 多目标协同导航 (含 COLREGs)

使用预训练单 USV SAC Actuator + DQN Selector 热启动,
通过 MAPPO (Multi-Agent PPO) 进行集中训练分布执行 (CTDE) 微调.

架构:
  ┌──────────────────────────────────────────────────────────────┐
  │  Decentralized Actor (共享参数, 预训练初始化)                  │
  │    Input:  actuator_state(26) + relative_goal(2)             │
  │    Output: Gaussian(throttle, rudder)                        │
  ├──────────────────────────────────────────────────────────────┤
  │  Centralized Critic (随机初始化)                              │
  │    Input:  global_state (7N + 3T + 4 + N(N-1))              │
  │    Output: V(s)                                              │
  ├──────────────────────────────────────────────────────────────┤
  │  Frozen Selector (预训练 DQN, 不参与梯度更新)                  │
  │    为每个 USV 独立选择下一个目标                                │
  └──────────────────────────────────────────────────────────────┘

训练流程:
  1. 每 episode: 环境 reset, Selector 为各 USV 分配初始目标
  2. 导航循环: 所有 USV 并行执行 Actor, step_marl 获取 per-agent reward
  3. 目标到达时 Selector 重新分配, 记录 rollout 数据
  4. Episode 结束后: 计算 GAE, PPO 多 epoch 更新

用法:
  python CTDE_MAPPO_train.py --do_train --pretrained_dir expert_hrl_sac_output_329
  python CTDE_MAPPO_train.py --do_eval  --pretrained_dir ctde_mappo_output
"""

import argparse
import json
import os
import datetime
import time
import numpy as np
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from my_USV_Env import MultiUsvEnv, X_MAX, Y_MAX
from hrl_config import device, SCALE, set_global_seed
from sac_agent import GaussianActor
from selector_agent import SelectorAgent
from hrl_helpers import (
    get_selector_state, get_actuator_state,
    get_usv_pos, get_target_pos
)
from mappo_agent import (
    CentralizedCritic, MAPPORolloutBuffer, MAPPOTrainer
)


# ============================================================================
# ========================= Training Loop ====================================
# ============================================================================

def train_ctde_mappo(args):
    """
    CTDE MAPPO 训练主循环.

    1. 加载预训练 Actor (SAC GaussianActor) 和 Selector (DQN)
    2. 初始化集中式 Critic
    3. 每 episode 收集 rollout → GAE → PPO 更新
    """
    # ---- 0. 全局种子 ----
    set_global_seed(args.seed)
    print(f"[CTDE-MAPPO] Global seed = {args.seed}, device = {device}")

    # ---- 1. 环境 ----
    env = MultiUsvEnv(
        num_usvs=args.num_usvs,
        per_usv_max_steps=args.max_steps_per_target * args.num_targets,
        colregs_k_C=args.colregs_k_C,
        starboard_threshold=args.starboard_threshold,
        # MyUsvEnv kwargs
        num_targets=args.num_targets,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        record_flag=False,
        usv_length=args.usv_length,
        usv_width=args.usv_width,
        target_radius=args.target_radius,
        obstacle_radius_range=(args.obstacle_radius_min, args.obstacle_radius_max),
        use_random_shapes=True,
        current_type="generated",
    )

    obs_dict = env.reset()
    actuator_state_dim = get_actuator_state(obs_dict["usv_0"]).shape[0]  # 26
    goal_dim = 2
    action_dim = 2
    global_state_dim = env.get_global_state_dim()

    print(f"[Env] {args.num_usvs} USVs, {args.num_targets} targets, "
          f"{args.num_obstacles} obstacles")
    print(f"[Dims] actuator_state={actuator_state_dim}, goal={goal_dim}, "
          f"action={action_dim}, global_state={global_state_dim}")

    # ---- 2. 模型初始化 ----
    # Actor: 预训练 GaussianActor
    actor = GaussianActor(
        actuator_state_dim, goal_dim, action_dim, max_action=1.0
    ).to(device)

    pretrained_actor_path = os.path.join(
        args.pretrained_dir, "best_actuator_actor.pth")
    if os.path.exists(pretrained_actor_path):
        actor.load_state_dict(
            torch.load(pretrained_actor_path, map_location=device,
                       weights_only=True))
        print(f"[Load] Actor ← {pretrained_actor_path}")
    else:
        print(f"[Warning] 预训练 Actor 不存在: {pretrained_actor_path}, 随机初始化")

    # Critic: 集中式 V-function (全新)
    critic = CentralizedCritic(global_state_dim).to(device)
    print(f"[Init] CentralizedCritic (global_state_dim={global_state_dim})")

    # Selector: 预训练 DQN (冻结)
    selector = SelectorAgent()
    pretrained_selector_path = os.path.join(
        args.pretrained_dir, "best_selector.pth")
    if os.path.exists(pretrained_selector_path):
        selector.q_net.load_state_dict(
            torch.load(pretrained_selector_path, map_location=device,
                       weights_only=True))
        selector.target_q_net.load_state_dict(selector.q_net.state_dict())
        print(f"[Load] Selector ← {pretrained_selector_path}")
    else:
        print(f"[Warning] 预训练 Selector 不存在: {pretrained_selector_path}")
    selector.epsilon = 0.0  # 评估模式, 无随机探索

    # ---- 3. MAPPO Trainer ----
    trainer = MAPPOTrainer(
        actor, critic,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

    # ---- 4. Rollout Buffer ----
    max_ep_steps = args.max_steps_per_target * args.num_targets
    buffer = MAPPORolloutBuffer(
        max_ep_steps, args.num_usvs,
        actuator_state_dim, action_dim,
        global_state_dim, goal_dim,
    )

    # ---- 5. Logging ----
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = (f'CTDE_MAPPO_{args.num_usvs}USV_'
                f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    writer = SummaryWriter(os.path.join(args.output_dir, 'runs', run_name))

    best_reward = -float('inf')
    best_reward_episode = 0
    success_history = deque(maxlen=100)
    start_episode = 0

    training_history = {
        'episode_rewards': [],
        'episode_steps': [],
        'episode_successes': [],
        'episode_colregs_violations': [],
    }

    # ---- 6. 恢复训练 (可选) ----
    if args.resume:
        resume_path = args.resume
        if resume_path.isdigit():
            resume_path = os.path.join(
                args.output_dir, "checkpoints",
                f"ep_{resume_path}", "checkpoint.pth")
        elif os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, "checkpoint.pth")

        if os.path.exists(resume_path):
            print(f"\n[Resume] ← {resume_path}")
            ckpt = torch.load(resume_path, map_location=device,
                              weights_only=False)
            actor.load_state_dict(ckpt['actor'])
            critic.load_state_dict(ckpt['critic'])
            trainer.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
            trainer.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
            start_episode = ckpt['episode'] + 1
            best_reward = ckpt['best_reward']
            training_history = ckpt.get('training_history', training_history)
            print(f"[Resume] 从 Episode {start_episode} 继续, "
                  f"best_reward={best_reward:.2f}")
        else:
            print(f"[Resume] 检查点不存在: {resume_path}")

    # ---- 打印配置 ----
    print(f"\n{'=' * 60}")
    print(f"  CTDE-MAPPO Training — {args.num_usvs} USVs / "
          f"{args.num_targets}T / {args.num_obstacles}O")
    print(f"  lr_actor={args.lr_actor}, lr_critic={args.lr_critic}")
    print(f"  PPO: clip={args.clip_eps}, epochs={args.ppo_epochs}, "
          f"entropy={args.entropy_coef}")
    print(f"  COLREGs: k_C={args.colregs_k_C}, "
          f"starboard_th={args.starboard_threshold}")
    print(f"  Max Episodes: {args.max_episodes}")
    if start_episode > 0:
        print(f"  Resuming from Episode: {start_episode}")
    print(f"{'=' * 60}\n")

    # ================================================================
    # ======================== Episode Loop ==========================
    # ================================================================
    for episode in range(start_episode, args.max_episodes):
        ep_start_time = time.time()
        buffer.clear()

        # ---- 环境 Reset ----
        if episode < args.fixed_env_episodes:
            obs_dict = env.reset(seed=args.seed)
        else:
            obs_dict = env.reset(seed=args.seed + episode)

        # ---- 初始目标分配 (Selector, 简单协调避免重叠) ----
        target_status = obs_dict["usv_0"]["targets"][2::3]
        valid_mask = (target_status == 0).astype(bool)

        current_target_idx = []
        target_pos_norm = []
        target_pos_abs = []
        assigned_targets = set()

        for i in range(args.num_usvs):
            sel_state = get_selector_state(obs_dict[f"usv_{i}"])

            # 避免多 USV 选择相同目标
            agent_valid = valid_mask.copy()
            for idx in assigned_targets:
                if idx < len(agent_valid):
                    agent_valid[idx] = False
            if not np.any(agent_valid):
                agent_valid = valid_mask.copy()  # 回退: 允许重叠

            t_idx = selector.select_action(sel_state, agent_valid, eval_mode=True)
            assigned_targets.add(t_idx)

            t_norm = get_target_pos(obs_dict[f"usv_{i}"], t_idx)
            t_abs = t_norm * SCALE

            current_target_idx.append(t_idx)
            target_pos_norm.append(t_norm)
            target_pos_abs.append(t_abs)

        # ---- Episode 统计 ----
        episode_rewards = np.zeros(args.num_usvs)
        episode_steps = 0
        targets_reached_count = 0
        colregs_violations = 0
        crush = False
        episode_done = False

        # ---- 导航循环 ----
        while not episode_done and episode_steps < max_ep_steps:
            global_state = env.get_global_state()

            # 收集所有智能体的动作
            obs_arr = np.zeros((args.num_usvs, actuator_state_dim),
                               dtype=np.float32)
            actions_arr = np.zeros((args.num_usvs, action_dim),
                                   dtype=np.float32)
            log_probs_arr = np.zeros(args.num_usvs, dtype=np.float32)
            values_arr = np.zeros(args.num_usvs, dtype=np.float32)
            goals_arr = np.zeros((args.num_usvs, goal_dim), dtype=np.float32)

            for i in range(args.num_usvs):
                act_state = get_actuator_state(obs_dict[f"usv_{i}"])
                usv_pos = get_usv_pos(obs_dict[f"usv_{i}"])
                rel_goal = target_pos_norm[i] - usv_pos

                action, log_prob, value = trainer.get_action_and_value(
                    act_state, rel_goal, global_state)

                obs_arr[i] = act_state
                actions_arr[i] = action
                log_probs_arr[i] = log_prob
                values_arr[i] = value
                goals_arr[i] = rel_goal

            # ---- 环境步进 ----
            next_obs, rewards_dict, dones_dict, truncated_dict, info = \
                env.step_marl(actions_arr, target_pos_abs)

            episode_steps += 1

            # ---- 奖励和终止标志 ----
            rewards_arr = np.array(
                [rewards_dict[f"usv_{i}"] for i in range(args.num_usvs)],
                dtype=np.float32)

            # terminal = 真正终止 (碰撞/全部到达), 不含 truncation
            terminal_flags = np.array([
                float(dones_dict[f"usv_{i}"]
                      and not truncated_dict[f"usv_{i}"])
                for i in range(args.num_usvs)
            ], dtype=np.float32)

            # 存入 buffer
            buffer.add(obs_arr, actions_arr, log_probs_arr, rewards_arr,
                       terminal_flags, goals_arr, global_state, values_arr)

            episode_rewards += rewards_arr

            # COLREGs 违规统计
            if 'reward_details' in info:
                for i in range(args.num_usvs):
                    pen = info['reward_details'].get(i, {}).get('colregs', 0)
                    if pen < -0.01:
                        colregs_violations += 1

            # ---- 检查终止 ----
            if dones_dict["__all__"]:
                episode_done = True
                crush = (any(info['obstacle_collisions'])
                         or len(info['usv_collision_pairs']) > 0)
                break

            # 全部目标到达
            new_target_status = info['target_status']
            if np.all(new_target_status == 1.0):
                episode_done = True
                break

            # ---- 目标重新分配 ----
            new_valid = (new_target_status == 0).astype(bool)
            for i in range(args.num_usvs):
                if info['reached_goals'][i]:
                    targets_reached_count += 1
                    if np.any(new_valid):
                        sel_st = get_selector_state(next_obs[f"usv_{i}"])
                        new_idx = selector.select_action(
                            sel_st, new_valid, eval_mode=True)
                        current_target_idx[i] = new_idx
                        target_pos_norm[i] = get_target_pos(
                            next_obs[f"usv_{i}"], new_idx)
                        target_pos_abs[i] = target_pos_norm[i] * SCALE

            obs_dict = next_obs

        # ==================== End of Episode ====================

        # 最终目标到达数 (从 target_status 统计)
        final_targets_reached = int(np.sum(info['target_status']))

        # ---- 计算 GAE ----
        if episode_done and not truncated_dict.get("__all__", False):
            # 真正终止: bootstrap = 0
            last_values = [0.0] * args.num_usvs
        else:
            # Truncation / 超步: bootstrap with V(s)
            gs = env.get_global_state()
            last_values = [trainer.get_value(gs)] * args.num_usvs

        buffer.compute_gae(last_values, args.gamma, args.gae_lambda)

        # ---- PPO 更新 ----
        if buffer.ptr >= args.min_update_steps:
            losses = trainer.update(buffer)
        else:
            losses = {'actor_loss': 0, 'critic_loss': 0,
                      'entropy': 0, 'approx_kl': 0}

        # ---- 统计 ----
        all_reached = (final_targets_reached >= args.num_targets)
        success = all_reached and not crush
        success_history.append(float(success))
        success_rate = np.mean(success_history)
        avg_reward = float(episode_rewards.mean())

        training_history['episode_rewards'].append(avg_reward)
        training_history['episode_steps'].append(episode_steps)
        training_history['episode_successes'].append(bool(success))
        training_history['episode_colregs_violations'].append(colregs_violations)

        ep_time = time.time() - ep_start_time

        # ---- 日志打印 ----
        if episode % args.log_interval == 0 or episode < 10:
            print(
                f"EP {episode} [{args.num_usvs}USV] | "
                f"R:{avg_reward:.1f} | "
                f"Tgt:{final_targets_reached}/{args.num_targets} | "
                f"Crush:{crush} | "
                f"CR_viol:{colregs_violations} | "
                f"SR:{success_rate:.2f} | "
                f"Steps:{episode_steps} | "
                f"AL:{losses['actor_loss']:.4f} "
                f"CL:{losses['critic_loss']:.4f} "
                f"Ent:{losses['entropy']:.3f} | "
                f"{ep_time:.1f}s"
            )

        # ---- TensorBoard ----
        writer.add_scalar('Reward/Average', avg_reward, episode)
        for i in range(args.num_usvs):
            writer.add_scalar(f'Reward/USV_{i}',
                              episode_rewards[i], episode)
        writer.add_scalar('Metrics/TargetsReached',
                          final_targets_reached, episode)
        writer.add_scalar('Metrics/Collision', float(crush), episode)
        writer.add_scalar('Metrics/SuccessRate', success_rate, episode)
        writer.add_scalar('Metrics/COLREGs_Violations',
                          colregs_violations, episode)
        writer.add_scalar('Metrics/EpisodeSteps', episode_steps, episode)
        writer.add_scalar('Training/ActorLoss',
                          losses['actor_loss'], episode)
        writer.add_scalar('Training/CriticLoss',
                          losses['critic_loss'], episode)
        writer.add_scalar('Training/Entropy', losses['entropy'], episode)
        writer.add_scalar('Training/ApproxKL',
                          losses['approx_kl'], episode)

        # ---- 定期保存检查点 ----
        if (episode + 1) % args.checkpoint_interval == 0:
            ckpt_dir = os.path.join(
                args.output_dir, "checkpoints", f"ep_{episode + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = {
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'actor_optimizer': trainer.actor_optimizer.state_dict(),
                'critic_optimizer': trainer.critic_optimizer.state_dict(),
                'episode': episode,
                'best_reward': best_reward,
                'training_history': training_history,
            }
            torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint.pth"))
            torch.save(actor.state_dict(),
                       os.path.join(ckpt_dir, "actor.pth"))
            torch.save(critic.state_dict(),
                       os.path.join(ckpt_dir, "critic.pth"))
            print(f"  [Checkpoint] Saved at ep {episode + 1}")

        # ---- 保存最佳模型 ----
        if avg_reward > best_reward and success:
            best_reward = avg_reward
            best_reward_episode = episode
            torch.save(actor.state_dict(),
                       os.path.join(args.output_dir, "best_actor.pth"))
            torch.save(critic.state_dict(),
                       os.path.join(args.output_dir, "best_critic.pth"))

    # ======================== Training Complete ========================
    torch.save(actor.state_dict(),
               os.path.join(args.output_dir, "actor.pth"))
    torch.save(critic.state_dict(),
               os.path.join(args.output_dir, "critic.pth"))

    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f)
    print(f"训练历史已保存至: {history_path}")

    writer.close()
    print(f"\n{'=' * 60}")
    print(f"Training Complete. Episodes: {args.max_episodes}")
    print(f"Best Avg Reward: {best_reward:.2f} at Episode {best_reward_episode}")
    print(f"{'=' * 60}")


# ============================================================================
# ========================= Evaluation =======================================
# ============================================================================

def evaluate_ctde_mappo(args):
    """
    CTDE MAPPO 评估.

    使用确定性策略运行多个 episode, 统计:
      - 成功率, 碰撞率
      - 目标到达数, 总步数
      - COLREGs 违规次数
    """
    set_global_seed(args.eval_seed)

    # ---- 环境 ----
    env = MultiUsvEnv(
        num_usvs=args.num_usvs,
        per_usv_max_steps=args.max_steps_per_target * args.eval_targets,
        colregs_k_C=args.colregs_k_C,
        starboard_threshold=args.starboard_threshold,
        num_targets=args.eval_targets,
        num_obstacles=args.eval_obstacles,
        seed=args.eval_seed,
        record_flag=False,
        usv_length=args.usv_length,
        usv_width=args.usv_width,
        target_radius=args.target_radius,
        obstacle_radius_range=(args.obstacle_radius_min, args.obstacle_radius_max),
        use_random_shapes=True,
        current_type="generated",
    )

    obs_dict = env.reset()
    act_dim = get_actuator_state(obs_dict["usv_0"]).shape[0]
    gs_dim = env.get_global_state_dim()

    # ---- 加载模型 ----
    actor_path = os.path.join(args.output_dir, "best_actor.pth")
    if not os.path.exists(actor_path):
        actor_path = os.path.join(args.output_dir, "actor.pth")

    actor = GaussianActor(act_dim, 2, 2, max_action=1.0).to(device)
    actor.load_state_dict(
        torch.load(actor_path, map_location=device, weights_only=True))
    actor.eval()
    print(f"[Eval] Actor ← {actor_path}")

    critic = CentralizedCritic(gs_dim).to(device)
    critic_path = os.path.join(args.output_dir, "best_critic.pth")
    if not os.path.exists(critic_path):
        critic_path = os.path.join(args.output_dir, "critic.pth")
    if os.path.exists(critic_path):
        critic.load_state_dict(
            torch.load(critic_path, map_location=device, weights_only=True))
    critic.eval()

    trainer = MAPPOTrainer(actor, critic)

    selector = SelectorAgent()
    sel_path = os.path.join(args.pretrained_dir, "best_selector.pth")
    if os.path.exists(sel_path):
        selector.q_net.load_state_dict(
            torch.load(sel_path, map_location=device, weights_only=True))
    selector.epsilon = 0.0

    # ---- 轨迹记录目录 ----
    eval_dir = os.path.join(args.output_dir, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)
    record_path = getattr(args, 'eval_record_path', True)

    # ---- 评估循环 ----
    results = []
    max_ep_steps = args.max_steps_per_target * args.eval_targets

    print(f"\n{'=' * 60}")
    print(f"  Evaluation: {args.eval_episodes} episodes, "
          f"{args.num_usvs} USVs, {args.eval_targets}T/{args.eval_obstacles}O")
    if record_path:
        print(f"  Trajectory recording → {eval_dir}")
    print(f"{'=' * 60}\n")

    for ep in range(args.eval_episodes):
        obs_dict = env.reset(seed=args.eval_seed + ep)

        # 目标分配
        t_status = obs_dict["usv_0"]["targets"][2::3]
        valid = (t_status == 0).astype(bool)

        tgt_idx = []
        tgt_norm = []
        tgt_abs = []
        assigned = set()

        for i in range(args.num_usvs):
            ss = get_selector_state(obs_dict[f"usv_{i}"])
            av = valid.copy()
            for a in assigned:
                if a < len(av):
                    av[a] = False
            if not np.any(av):
                av = valid.copy()
            ti = selector.select_action(ss, av, eval_mode=True)
            assigned.add(ti)
            tn = get_target_pos(obs_dict[f"usv_{i}"], ti)
            tgt_idx.append(ti)
            tgt_norm.append(tn)
            tgt_abs.append(tn * SCALE)

        ep_steps = 0
        ep_rewards = np.zeros(args.num_usvs)
        cr_viols = 0
        ep_done = False
        crush = False

        # 轨迹记录: ep 前缀
        ep_csv_prefix = f"ctde_ep_{ep}"

        while not ep_done and ep_steps < max_ep_steps:
            actions = np.zeros((args.num_usvs, 2), dtype=np.float32)
            for i in range(args.num_usvs):
                ast = get_actuator_state(obs_dict[f"usv_{i}"])
                upos = get_usv_pos(obs_dict[f"usv_{i}"])
                rg = tgt_norm[i] - upos
                actions[i] = trainer.get_deterministic_action(ast, rg)

            obs_dict, rwd, dn, trunc, info = env.step_marl(
                actions, tgt_abs)
            ep_steps += 1
            ep_rewards += np.array(
                [rwd[f"usv_{i}"] for i in range(args.num_usvs)])

            # 记录轨迹
            if record_path:
                env.record_points(csv_prefix=ep_csv_prefix,
                                  output_dir=eval_dir)

            if 'reward_details' in info:
                for i in range(args.num_usvs):
                    if info['reward_details'].get(i, {}).get('colregs', 0) < -0.01:
                        cr_viols += 1

            if dn["__all__"]:
                ep_done = True
                crush = (any(info['obstacle_collisions'])
                         or len(info['usv_collision_pairs']) > 0)
                break

            if np.all(info['target_status'] == 1.0):
                ep_done = True
                break

            nv = (info['target_status'] == 0).astype(bool)
            for i in range(args.num_usvs):
                if info['reached_goals'][i] and np.any(nv):
                    ss = get_selector_state(obs_dict[f"usv_{i}"])
                    ni = selector.select_action(ss, nv, eval_mode=True)
                    tgt_idx[i] = ni
                    tgt_norm[i] = get_target_pos(obs_dict[f"usv_{i}"], ni)
                    tgt_abs[i] = tgt_norm[i] * SCALE

        final_reached = int(np.sum(info['target_status']))
        success = (final_reached >= args.eval_targets) and not crush

        # 关闭本 episode 的 CSV 写入器 & 绘制轨迹
        if record_path:
            env.close_writers()
            env.draw_path(
                csv_prefix=ep_csv_prefix,
                output_dir=eval_dir,
                ifshow=False,
                ifsave=True,
                savePath=eval_dir,
                show_usv_shape=False,
                usv_interval=max(1, ep_steps // 20),
            )

        result = {
            'episode': ep,
            'success': success,
            'crush': crush,
            'targets_reached': final_reached,
            'steps': ep_steps,
            'avg_reward': float(ep_rewards.mean()),
            'colregs_violations': cr_viols,
        }
        results.append(result)

        status = "✓" if success else ("✗ CRASH" if crush else "✗ FAIL")
        print(f"  EP {ep}: {status} | Tgt:{final_reached}/{args.eval_targets} | "
              f"R:{ep_rewards.mean():.1f} | Steps:{ep_steps} | "
              f"CR_viol:{cr_viols}")

    # ---- 汇总 ----
    successes = sum(r['success'] for r in results)
    crashes = sum(r['crush'] for r in results)
    avg_reached = np.mean([r['targets_reached'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['avg_reward'] for r in results])
    avg_cr_viols = np.mean([r['colregs_violations'] for r in results])

    print(f"\n{'=' * 60}")
    print(f"  Evaluation Summary ({args.eval_episodes} episodes)")
    print(f"  Success Rate:      {successes}/{args.eval_episodes} "
          f"({100 * successes / args.eval_episodes:.1f}%)")
    print(f"  Collision Rate:    {crashes}/{args.eval_episodes}")
    print(f"  Avg Targets:       {avg_reached:.1f}/{args.eval_targets}")
    print(f"  Avg Steps:         {avg_steps:.0f}")
    print(f"  Avg Reward:        {avg_reward:.2f}")
    print(f"  Avg COLREGs Viol:  {avg_cr_viols:.1f}")
    print(f"{'=' * 60}")

    # 保存评估结果
    eval_path = os.path.join(eval_dir, "ctde_eval_results.json")
    with open(eval_path, 'w') as f:
        json.dump({
            'summary': {
                'success_rate': successes / args.eval_episodes,
                'collision_rate': crashes / args.eval_episodes,
                'avg_targets_reached': avg_reached,
                'avg_steps': avg_steps,
                'avg_reward': avg_reward,
                'avg_colregs_violations': avg_cr_viols,
            },
            'episodes': results,
        }, f, indent=2)
    print(f"评估结果已保存至: {eval_path}")

    return results


# ============================================================================
# ========================= CLI Entry Point ==================================
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CTDE MAPPO for Multi-USV Navigation with COLREGs")

    # ---- 环境 ----
    parser.add_argument("--num_usvs", default=2, type=int,
                        help="USV 数量")
    parser.add_argument("--num_targets", default=2, type=int,
                        help="目标点数量")
    parser.add_argument("--num_obstacles", default=6, type=int,
                        help="障碍物数量")
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--usv_length", default=3.0, type=float)
    parser.add_argument("--usv_width", default=1.5, type=float)
    parser.add_argument("--target_radius", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_min", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_max", default=10.0, type=float)
    parser.add_argument("--max_steps_per_target", default=1000, type=int)

    # ---- COLREGs ----
    parser.add_argument("--colregs_k_C", default=5.0, type=float,
                        help="COLREGs 惩罚系数 k_C")
    parser.add_argument("--starboard_threshold", default=-0.05, type=float,
                        help="右转判定阈值 (rudder < threshold 为右转)")

    # ---- 预训练 ----
    parser.add_argument("--pretrained_dir", default="expert_hrl_sac_output_329",
                        type=str, help="预训练模型目录")

    # ---- MAPPO ----
    parser.add_argument("--lr_actor", default=3e-5, type=float,
                        help="Actor 学习率 (较小, 保留预训练知识)")
    parser.add_argument("--lr_critic", default=3e-4, type=float,
                        help="Critic 学习率")
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--gae_lambda", default=0.95, type=float,
                        help="GAE λ 参数")
    parser.add_argument("--clip_eps", default=0.2, type=float,
                        help="PPO clip ε")
    parser.add_argument("--entropy_coef", default=0.01, type=float,
                        help="熵正则化系数")
    parser.add_argument("--max_grad_norm", default=0.5, type=float)
    parser.add_argument("--ppo_epochs", default=10, type=int,
                        help="每轮 rollout 的 PPO 更新 epoch 数")
    parser.add_argument("--mini_batch_size", default=256, type=int)

    # ---- 训练 ----
    parser.add_argument("--max_episodes", default=3000, type=int)
    parser.add_argument("--fixed_env_episodes", default=200, type=int,
                        help="前 N 个 episode 使用固定环境 (同一 seed)")
    parser.add_argument("--min_update_steps", default=64, type=int,
                        help="PPO 更新所需的最少 rollout 步数")
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--checkpoint_interval", default=200, type=int)
    parser.add_argument("--output_dir", default="ctde_mappo_output", type=str)

    # ---- 恢复 ----
    parser.add_argument("--resume", default=None, type=str,
                        help="检查点路径或 episode 编号")

    # ---- 评估 ----
    parser.add_argument("--eval_episodes", default=5, type=int)
    parser.add_argument("--eval_targets", default=2, type=int)
    parser.add_argument("--eval_obstacles", default=6, type=int)
    parser.add_argument("--eval_seed", default=42, type=int)

    # ---- 轨迹记录 ----
    parser.add_argument("--eval_record_path", action="store_true",
                        default=True,
                        help="评估时记录并绘制各 USV 轨迹")

    # ---- 模式 ----
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    args = parser.parse_args()

    if args.do_train:
        train_ctde_mappo(args)

    if args.do_eval:
        evaluate_ctde_mappo(args)

    if not args.do_train and not args.do_eval:
        print("请指定 --do_train 和/或 --do_eval")
        parser.print_help()


if __name__ == "__main__":
    main()
