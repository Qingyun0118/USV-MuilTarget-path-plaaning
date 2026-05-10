"""
CTDE MAPPO 训练循环 — 基于 BaseAssigner 统一接口

消除所有分配方法分支逻辑, 训练流程:
  1. 创建环境、Actor、Critic、Assigner
  2. 每 episode: reset → assign_initial → 导航循环 → PPO 更新
  3. assigner.on_step / assigner.on_episode_end / assigner.update 回调
"""

import json
import os
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from my_USV_Env import MultiUsvEnv
from hrl_config import device, SCALE, set_global_seed
from sac_agent import GaussianActor, SAC_EPSILON
from hrl_helpers import (
    get_actuator_state, get_usv_pos, get_target_pos,
)
from mappo_agent import (
    CentralizedCritic, MAPPORolloutBuffer, MAPPOTrainer
)
from task_assignment import create_assigner


# ============================================================================
# ==================== Actor 工厂 ============================================
# ============================================================================

def detect_algo_from_dir(pretrained_dir):
    """从预训练目录名检测算法类型 (sac/td3/td7)."""
    name = os.path.basename(pretrained_dir).lower()
    if 'td7' in name:
        return 'td7'
    elif 'td3' in name:
        return 'td3'
    return 'sac'


class StochasticWrapper(nn.Module):
    """为确定性 Actor (TD3/TD7) 添加随机策略接口, 适配 MAPPO PPO 训练."""

    def __init__(self, deterministic_actor, action_dim, max_action=1.0,
                 log_std_init=-0.5, is_td7=False, encoder=None):
        super().__init__()
        self.actor = deterministic_actor
        self.max_action = max_action
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        self.is_td7 = is_td7
        self.encoder = encoder  # TD7 Encoder (nn.Module), 自动纳入 state_dict

    def _get_mean(self, state, goal):
        if self.is_td7:
            if self.encoder is not None:
                zs = self.encoder.zs(state, goal)
            else:
                zs_dim = self.actor.l1.in_features - self.actor.l0.out_features
                zs = torch.zeros(state.size(0), zs_dim, device=state.device)
            return self.actor(state, goal, zs)
        return self.actor(state, goal)

    def forward(self, state, goal):
        mean = self._get_mean(state, goal)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def sample(self, state, goal):
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + SAC_EPSILON)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean_action = torch.tanh(mean) * self.max_action
        return action, log_prob, mean_action

    def get_deterministic_action(self, state, goal):
        mean = self._get_mean(state, goal)
        return torch.tanh(mean) * self.max_action


def create_actor_for_algo(algo, state_dim, goal_dim, action_dim):
    """根据算法类型创建 Actor (不加载权重)."""
    if algo == 'sac':
        return GaussianActor(state_dim, goal_dim, action_dim, max_action=1.0).to(device)
    elif algo == 'td3':
        from td3_agent import Actor as TD3Actor
        det_actor = TD3Actor(state_dim, goal_dim, action_dim, max_action=1.0).to(device)
        return StochasticWrapper(det_actor, action_dim, max_action=1.0, is_td7=False).to(device)
    elif algo == 'td7':
        from td7_agent import Actor as TD7Actor, Encoder
        det_actor = TD7Actor(state_dim, goal_dim, action_dim).to(device)
        encoder = Encoder(state_dim, goal_dim, action_dim).to(device)
        return StochasticWrapper(det_actor, action_dim, max_action=1.0,
                                 is_td7=True, encoder=encoder).to(device)
    else:
        raise ValueError(f"Unsupported algo: {algo}, expected sac/td3/td7")


def create_pretrained_actor(algo, state_dim, goal_dim, action_dim, pretrained_path):
    """根据算法类型创建并加载预训练 Actor (含 TD7 Encoder)."""
    actor = create_actor_for_algo(algo, state_dim, goal_dim, action_dim)

    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        if algo == 'sac':
            actor.load_state_dict(state_dict)
        else:
            actor.actor.load_state_dict(state_dict)
        print(f"[Load] Actor ({algo.upper()}) ← {pretrained_path}")

        # TD7: 同时加载 Encoder
        if algo == 'td7' and actor.encoder is not None:
            pretrained_dir = os.path.dirname(pretrained_path)
            encoder_loaded = False
            for enc_name in ["actuator_encoder.pth", "best_actuator_encoder.pth"]:
                enc_path = os.path.join(pretrained_dir, enc_name)
                if os.path.exists(enc_path):
                    actor.encoder.load_state_dict(
                        torch.load(enc_path, map_location=device, weights_only=True))
                    encoder_loaded = True
                    print(f"[Load] Encoder (TD7) ← {enc_path}")
                    break
            if not encoder_loaded:
                print("[Warning] TD7 Encoder 未找到, zs 将使用零向量")
    elif pretrained_path:
        print(f"[Warning] 预训练 Actor 不存在: {pretrained_path}, 随机初始化")

    return actor


def train_ctde_mappo(args):
    """CTDE MAPPO 训练主循环 — 统一 BaseAssigner 接口, 无分支."""

    # ---- 0. 全局种子 ----
    set_global_seed(args.seed)
    print(f"[CTDE-MAPPO] Global seed = {args.seed}, device = {device}")

    # ---- 1. 环境 ----
    env = MultiUsvEnv(
        num_usvs=args.num_usvs,
        per_usv_max_steps=args.max_steps_per_target * args.num_targets,
        colregs_k_C=args.colregs_k_C,
        colregs_k_R=args.colregs_k_R,
        starboard_threshold=args.starboard_threshold,
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
    actuator_state_dim = get_actuator_state(obs_dict["usv_0"]).shape[0]
    goal_dim = 2
    action_dim = 2
    global_state_dim = env.get_global_state_dim()

    print(f"[Env] {args.num_usvs} USVs, {args.num_targets} targets, "
          f"{args.num_obstacles} obstacles")
    print(f"[Dims] actuator_state={actuator_state_dim}, goal={goal_dim}, "
          f"action={action_dim}, global_state={global_state_dim}")

    # ---- 2. 模型初始化 ----
    algo = getattr(args, 'algo', None)
    if not algo:
        algo = detect_algo_from_dir(args.pretrained_dir)
    print(f"[Algo] 检测到预训练算法: {algo.upper()}")

    # 优先加载最后模型, 其次 best 模型
    pretrained_actor_path = None
    for fname in ["actuator_actor.pth", "best_actuator_actor.pth"]:
        candidate = os.path.join(args.pretrained_dir, fname)
        if os.path.exists(candidate):
            pretrained_actor_path = candidate
            break
    if pretrained_actor_path is None:
        pretrained_actor_path = os.path.join(
            args.pretrained_dir, "actuator_actor.pth")  # 触发 warning

    actor = create_pretrained_actor(
        algo, actuator_state_dim, goal_dim, action_dim,
        pretrained_actor_path)

    critic = CentralizedCritic(global_state_dim).to(device)
    print(f"[Init] CentralizedCritic (global_state_dim={global_state_dim})")

    # ---- 3. 分配器 (统一工厂) ----
    assigner = create_assigner(args, global_state_dim, args.pretrained_dir)
    print(f"[Init] Assigner: {assigner.name} "
          f"(learnable={assigner.is_learnable})")

    # ---- 4. MAPPO Trainer ----
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
        target_kl=args.target_kl,
    )

    # ---- 5. Rollout Buffer ----
    max_ep_steps = args.max_steps_per_target * args.num_targets
    buffer = MAPPORolloutBuffer(
        max_ep_steps, args.num_usvs,
        actuator_state_dim, action_dim,
        global_state_dim, goal_dim,
    )

    # ---- 6. Logging ----
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

    # ---- 7. 恢复训练 ----
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
            if assigner.is_learnable and 'assigner' in ckpt:
                assigner.load_from_checkpoint(ckpt)
                print("[Resume] Assigner 状态已恢复")
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
    print(f"  Assignment: {assigner.name}")
    print(f"  lr_actor={args.lr_actor}, lr_critic={args.lr_critic}")
    print(f"  PPO: clip={args.clip_eps}, epochs={args.ppo_epochs}, "
          f"entropy={args.entropy_coef}")
    if assigner.is_learnable:
        print(f"  Selector PPO: lr_policy={args.sel_lr_policy}, "
              f"lr_value={args.sel_lr_value}, "
              f"entropy={args.sel_entropy_coef}")
    print(f"  COLREGs: k_C={args.colregs_k_C}, k_R={args.colregs_k_R}, "
          f"anneal_eps={args.colregs_anneal_episodes}")
    print(f"  Critic warmup: {args.freeze_actor_episodes} episodes")
    print(f"  KL early stopping: target_kl={args.target_kl}")
    print(f"  Max Episodes: {args.max_episodes}")
    if start_episode > 0:
        print(f"  Resuming from Episode: {start_episode}")
    print(f"{'=' * 60}\n")

    # ====================================================================
    # ======================== Episode Loop ==============================
    # ====================================================================
    for episode in range(start_episode, args.max_episodes):
        ep_start_time = time.time()
        buffer.clear()

        # ---- 线性学习率衰减 ----
        lr_frac = 1.0 - episode / args.max_episodes
        for pg in trainer.actor_optimizer.param_groups:
            pg['lr'] = args.lr_actor * lr_frac
        for pg in trainer.critic_optimizer.param_groups:
            pg['lr'] = args.lr_critic * lr_frac

        # ---- COLREGs 退火 ----
        if episode < args.colregs_anneal_episodes:
            progress = episode / max(1, args.colregs_anneal_episodes)
            current_k_C = (args.colregs_k_C_start
                           + (args.colregs_k_C - args.colregs_k_C_start) * progress)
            current_k_R = args.colregs_k_R * progress
        else:
            current_k_C = args.colregs_k_C
            current_k_R = args.colregs_k_R
        env.colregs_k_C = current_k_C
        env.colregs_k_R = current_k_R

        # ---- 环境 Reset ----
        if episode < args.fixed_env_episodes:
            obs_dict = env.reset(seed=args.seed)
        else:
            obs_dict = env.reset(seed=args.seed + episode)

        # ---- 初始目标分配 ----
        target_status = obs_dict["usv_0"]["targets"][2::3]
        valid_mask = (target_status == 0).astype(bool)
        global_state_init = env.get_global_state()

        current_target_idx, target_pos_norm, target_pos_abs, _ = \
            assigner.assign_initial(
                obs_dict, args.num_usvs, valid_mask, SCALE,
                global_state=global_state_init)

        # ---- Episode 统计 ----
        episode_rewards = np.zeros(args.num_usvs)
        episode_steps = 0
        targets_reached_count = 0
        colregs_violations = 0
        crush = False
        obs_crush = False
        usv_crush = False
        episode_done = False

        # per-USV 完成时间 (步数), -1 表示未完成全部序列
        usv_finish_steps = np.full(args.num_usvs, -1, dtype=np.int32)
        # 跟踪每艘USV已到达的目标数
        usv_targets_reached = np.zeros(args.num_usvs, dtype=np.int32)
        # 跟踪空闲USV (已完成序列且无剩余目标可分配)
        usv_idle = np.zeros(args.num_usvs, dtype=bool)

        # ---- 导航循环 ----
        while not episode_done and episode_steps < max_ep_steps:
            global_state = env.get_global_state()

            obs_arr = np.zeros((args.num_usvs, actuator_state_dim),
                               dtype=np.float32)
            actions_arr = np.zeros((args.num_usvs, action_dim),
                                   dtype=np.float32)
            log_probs_arr = np.zeros(args.num_usvs, dtype=np.float32)
            values_arr = np.zeros(args.num_usvs, dtype=np.float32)
            goals_arr = np.zeros((args.num_usvs, goal_dim), dtype=np.float32)
            active_masks_arr = np.ones(args.num_usvs, dtype=np.float32)

            for i in range(args.num_usvs):
                act_state = get_actuator_state(obs_dict[f"usv_{i}"])
                usv_pos = get_usv_pos(obs_dict[f"usv_{i}"])

                if usv_idle[i]:
                    rel_goal = np.zeros(goal_dim, dtype=np.float32)
                    action = np.zeros(action_dim, dtype=np.float32)
                    log_prob = 0.0
                    value = trainer.get_value(global_state)
                    active_masks_arr[i] = 0.0
                else:
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

            rewards_arr = np.array(
                [rewards_dict[f"usv_{i}"] for i in range(args.num_usvs)],
                dtype=np.float32)

            terminal_flags = np.array([
                float(dones_dict[f"usv_{i}"]
                      and not truncated_dict[f"usv_{i}"])
                for i in range(args.num_usvs)
            ], dtype=np.float32)

            buffer.add(obs_arr, actions_arr, log_probs_arr, rewards_arr,
                       terminal_flags, goals_arr, global_state, values_arr,
                       active_masks_arr)

            episode_rewards += rewards_arr

            # ---- Assigner 每步回调 ----
            assigner.on_step(rewards_arr, info)

            # ---- COLREGs 违规统计 ----
            if 'reward_details' in info:
                for i in range(args.num_usvs):
                    pen = info['reward_details'].get(i, {}).get('colregs', 0)
                    if pen < -0.01:
                        colregs_violations += 1

            # ---- 检查终止 ----
            if dones_dict["__all__"]:
                episode_done = True
                obs_crush = any(info['obstacle_collisions'])
                usv_crush = len(info['usv_collision_pairs']) > 0
                crush = obs_crush or usv_crush
                break

            new_target_status = info['target_status']
            if np.all(new_target_status == 1.0):
                episode_done = True
                # 记录最后完成者的时间
                for i in range(args.num_usvs):
                    if usv_finish_steps[i] < 0:
                        usv_finish_steps[i] = episode_steps
                break

            # ---- 目标重新分配 ----
            new_valid = (new_target_status == 0).astype(bool)
            reassign_usv_indices = []

            # 情况A: USV本步到达子目标
            for i in range(args.num_usvs):
                if info['reached_goals'][i]:
                    targets_reached_count += 1
                    usv_targets_reached[i] += 1
                    reassign_usv_indices.append(i)

            # 情况B: 空闲USV仍有剩余目标可接管
            if np.any(new_valid):
                for i in range(args.num_usvs):
                    if usv_idle[i] and i not in reassign_usv_indices:
                        reassign_usv_indices.append(i)
                        usv_idle[i] = False

            if reassign_usv_indices and np.any(new_valid):
                new_gs = env.get_global_state()
                assigner.reassign(
                    next_obs, reassign_usv_indices, new_valid,
                    current_target_idx, target_pos_norm, target_pos_abs,
                    SCALE, global_state=new_gs)

                # 检查 reassign 后是否有USV未获得有效目标 → 标记空闲
                for i in reassign_usv_indices:
                    t_idx = current_target_idx[i]
                    if t_idx < 0 or (t_idx < len(new_valid) and not new_valid[t_idx]):
                        usv_idle[i] = True
                        if usv_finish_steps[i] < 0:
                            usv_finish_steps[i] = episode_steps
            elif reassign_usv_indices and not np.any(new_valid):
                # 无剩余目标, 所有刚到达的USV变为空闲
                for i in reassign_usv_indices:
                    usv_idle[i] = True
                    if usv_finish_steps[i] < 0:
                        usv_finish_steps[i] = episode_steps

            obs_dict = next_obs

        # ==================== End of Episode ====================

        # ---- Assigner episode 结束回调 ----
        assigner.on_episode_end(obs_dict, info)

        final_targets_reached = int(np.sum(info['target_status']))

        # ---- GAE (Actuator) ----
        if episode_done and not truncated_dict.get("__all__", False):
            last_values = [0.0] * args.num_usvs
        else:
            gs = env.get_global_state()
            last_values = [trainer.get_value(gs)] * args.num_usvs

        buffer.compute_gae(last_values, args.gamma, args.gae_lambda)

        # ---- Actuator PPO 更新 ----
        if buffer.ptr >= args.min_update_steps:
            if episode < args.freeze_actor_episodes:
                losses = trainer.update_critic_only(buffer)
            else:
                losses = trainer.update(buffer)
        else:
            losses = {'actor_loss': 0, 'critic_loss': 0,
                      'entropy': 0, 'approx_kl': 0}

        # ---- Assigner 更新 ----
        sel_losses = {'sel_policy_loss': 0, 'sel_value_loss': 0,
                      'sel_entropy': 0}
        if assigner.is_learnable:
            buf_size = assigner.buffer.size
            if buf_size >= args.sel_min_decisions:
                sel_losses = assigner.update()
            # 传递 buffer size 用于日志
            sel_losses['_buf_size'] = buf_size

        # ---- 统计 ----
        all_reached = (final_targets_reached >= args.num_targets)
        success = all_reached and not crush
        success_history.append(float(success))
        success_rate = np.mean(success_history)
        avg_reward = float(episode_rewards.mean())

        # Makespan 指标: 未完成的USV用 episode_steps 填充
        for i in range(args.num_usvs):
            if usv_finish_steps[i] < 0:
                usv_finish_steps[i] = episode_steps
        makespan = int(np.max(usv_finish_steps))
        load_balance_std = float(np.std(usv_finish_steps))
        per_usv_finish = usv_finish_steps.tolist()

        training_history['episode_rewards'].append(avg_reward)
        training_history['episode_steps'].append(episode_steps)
        training_history['episode_successes'].append(bool(success))
        training_history['episode_colregs_violations'].append(colregs_violations)
        training_history.setdefault('episode_makespan', []).append(makespan)
        training_history.setdefault('episode_load_balance_std', []).append(
            load_balance_std)
        training_history.setdefault('episode_per_usv_finish', []).append(
            per_usv_finish)

        ep_time = time.time() - ep_start_time

        # ---- 日志打印 ----
        if episode % args.log_interval == 0 or episode < 10:
            sel_info = ""
            if assigner.is_learnable:
                sel_info = (f" SL:{sel_losses['sel_policy_loss']:.4f}"
                            f" SE:{sel_losses['sel_entropy']:.3f}"
                            f" SB:{sel_losses.get('_buf_size', 0)}")
            warmup_tag = "[WARMUP] " if episode < args.freeze_actor_episodes else ""
            finish_str = "/".join(str(s) for s in per_usv_finish)
            print(
                f"EP {episode} {warmup_tag}[{args.num_usvs}USV|{assigner.name}] | "
                f"R:{avg_reward:.1f} | "
                f"Tgt:{final_targets_reached}/{args.num_targets} | "
                f"MS:{makespan}({finish_str}) LB:{load_balance_std:.1f} | "
                f"Crush:{crush}(obs:{obs_crush},usv:{usv_crush}) | "
                f"CR_viol:{colregs_violations} | "
                f"SR:{success_rate:.2f} | "
                f"Steps:{episode_steps} | "
                f"AL:{losses['actor_loss']:.4f} "
                f"CL:{losses['critic_loss']:.4f} "
                f"Ent:{losses['entropy']:.3f} "
                f"kC:{current_k_C:.2f}"
                f"{sel_info} | "
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
        writer.add_scalar('Metrics/Collision_Obstacle', float(obs_crush), episode)
        writer.add_scalar('Metrics/Collision_USV', float(usv_crush), episode)
        writer.add_scalar('Metrics/SuccessRate', success_rate, episode)
        writer.add_scalar('Metrics/COLREGs_Violations',
                          colregs_violations, episode)
        writer.add_scalar('Metrics/EpisodeSteps', episode_steps, episode)
        writer.add_scalar('Metrics/Makespan', makespan, episode)
        writer.add_scalar('Metrics/LoadBalanceStd', load_balance_std, episode)
        for i in range(args.num_usvs):
            writer.add_scalar(f'Metrics/USV_{i}_FinishSteps',
                              per_usv_finish[i], episode)
        writer.add_scalar('Training/ActorLoss', losses['actor_loss'], episode)
        writer.add_scalar('Training/CriticLoss', losses['critic_loss'], episode)
        writer.add_scalar('Training/Entropy', losses['entropy'], episode)
        writer.add_scalar('Training/ApproxKL', losses['approx_kl'], episode)
        writer.add_scalar('Training/COLREGs_kC', current_k_C, episode)
        writer.add_scalar('Training/CriticWarmup',
                          float(episode < args.freeze_actor_episodes), episode)
        writer.add_scalar('Training/LR_Actor',
                          trainer.actor_optimizer.param_groups[0]['lr'], episode)
        writer.add_scalar('Training/LR_Critic',
                          trainer.critic_optimizer.param_groups[0]['lr'], episode)
        if assigner.is_learnable:
            writer.add_scalar('Selector/PolicyLoss',
                              sel_losses['sel_policy_loss'], episode)
            writer.add_scalar('Selector/ValueLoss',
                              sel_losses['sel_value_loss'], episode)
            writer.add_scalar('Selector/Entropy',
                              sel_losses['sel_entropy'], episode)
            writer.add_scalar('Selector/BufferSize',
                              sel_losses.get('_buf_size', 0), episode)

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
                'assign_method': args.assign_method,
                'algo': algo,
            }
            if assigner.is_learnable:
                ckpt['assigner'] = assigner.state_dict_for_checkpoint()
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
            if assigner.is_learnable:
                assigner.save(os.path.join(
                    args.output_dir, "best_selector_ppo.pth"))

    # ======================== Training Complete ========================
    torch.save(actor.state_dict(),
               os.path.join(args.output_dir, "actor.pth"))
    torch.save(critic.state_dict(),
               os.path.join(args.output_dir, "critic.pth"))
    if assigner.is_learnable:
        assigner.save(os.path.join(args.output_dir, "selector_ppo.pth"))

    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f)
    print(f"训练历史已保存至: {history_path}")

    writer.close()
    print(f"\n{'=' * 60}")
    print(f"Training Complete. Episodes: {args.max_episodes}")
    print(f"Best Avg Reward: {best_reward:.2f} at Episode {best_reward_episode}")
    print(f"{'=' * 60}")
