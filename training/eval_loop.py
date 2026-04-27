"""
CTDE MAPPO 评估循环 — 基于 BaseAssigner 统一接口

使用确定性策略运行多个 episode, 统计成功率、碰撞率、COLREGs 违规等.
"""

import json
import os
import numpy as np
import torch

from my_USV_Env import MultiUsvEnv
from hrl_config import device, SCALE, set_global_seed
from hrl_helpers import get_actuator_state, get_usv_pos
from mappo_agent import CentralizedCritic, MAPPOTrainer
from task_assignment import create_assigner
from training.train_loop import (
    detect_algo_from_dir, create_actor_for_algo,
)


def evaluate_ctde_mappo(args):
    """CTDE MAPPO 评估 — 统一 BaseAssigner 接口, 无分支."""

    set_global_seed(args.eval_seed)

    # ---- 环境 ----
    env = MultiUsvEnv(
        num_usvs=args.num_usvs,
        per_usv_max_steps=args.max_steps_per_target * args.eval_targets,
        colregs_k_C=args.colregs_k_C,
        colregs_k_R=args.colregs_k_R,
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

    # ---- 加载 Actor ----
    actor_path = os.path.join(args.output_dir, "best_actor.pth")
    if not os.path.exists(actor_path):
        actor_path = os.path.join(args.output_dir, "actor.pth")

    state_dict = torch.load(actor_path, map_location=device, weights_only=True)

    # 从 state_dict 检测算法类型
    if 'log_std_linear.weight' in state_dict:
        algo = 'sac'
    elif 'actor.l0.weight' in state_dict:
        algo = 'td7'
    elif 'actor.fc1.weight' in state_dict:
        algo = 'td3'
    else:
        algo = detect_algo_from_dir(args.output_dir)

    actor = create_actor_for_algo(algo, act_dim, 2, 2)
    actor.load_state_dict(state_dict)
    actor.eval()
    print(f"[Eval] Actor ({algo.upper()}) ← {actor_path}")

    # ---- 加载 Critic ----
    critic = CentralizedCritic(gs_dim).to(device)
    critic_path = os.path.join(args.output_dir, "best_critic.pth")
    if not os.path.exists(critic_path):
        critic_path = os.path.join(args.output_dir, "critic.pth")
    if os.path.exists(critic_path):
        critic.load_state_dict(
            torch.load(critic_path, map_location=device, weights_only=True))
    critic.eval()

    trainer = MAPPOTrainer(actor, critic)

    # ---- 分配器 (统一工厂, eval 模式) ----
    assigner = create_assigner(args, gs_dim)
    if assigner.is_learnable:
        ppo_path = os.path.join(args.output_dir, "best_selector_ppo.pth")
        if not os.path.exists(ppo_path):
            ppo_path = os.path.join(args.output_dir, "selector_ppo.pth")
        if os.path.exists(ppo_path):
            assigner.load(ppo_path)
            print(f"[Eval] Assigner ← {ppo_path}")
        else:
            print("[Eval] WARNING: No assigner weights found, using random!")
    print(f"[Eval] Assigner: {assigner.name}")

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

        t_status = obs_dict["usv_0"]["targets"][2::3]
        valid = (t_status == 0).astype(bool)
        global_state = env.get_global_state()

        tgt_idx, tgt_norm, tgt_abs, _ = assigner.assign_initial(
            obs_dict, args.num_usvs, valid, SCALE,
            global_state=global_state, eval_mode=True)

        ep_steps = 0
        ep_rewards = np.zeros(args.num_usvs)
        cr_viols = 0
        ep_done = False
        crush = False
        obs_crush = False
        usv_crush = False

        ep_csv_prefix = f"ctde_ep_{ep}"

        while not ep_done and ep_steps < max_ep_steps:
            actions = np.zeros((args.num_usvs, 2), dtype=np.float32)
            for i in range(args.num_usvs):
                ast = get_actuator_state(obs_dict[f"usv_{i}"])
                upos = get_usv_pos(obs_dict[f"usv_{i}"])
                rg = tgt_norm[i] - upos
                actions[i] = trainer.get_deterministic_action(ast, rg)

            obs_dict, rwd, dn, trunc, info = env.step_marl(actions, tgt_abs)
            ep_steps += 1
            ep_rewards += np.array(
                [rwd[f"usv_{i}"] for i in range(args.num_usvs)])

            if record_path:
                env.record_points(csv_prefix=ep_csv_prefix, output_dir=eval_dir)

            if 'reward_details' in info:
                for i in range(args.num_usvs):
                    if info['reward_details'].get(i, {}).get('colregs', 0) < -0.01:
                        cr_viols += 1

            if dn["__all__"]:
                ep_done = True
                obs_crush = any(info['obstacle_collisions'])
                usv_crush = len(info['usv_collision_pairs']) > 0
                crush = obs_crush or usv_crush
                break

            if np.all(info['target_status'] == 1.0):
                ep_done = True
                break

            nv = (info['target_status'] == 0).astype(bool)
            reached_usvs = [i for i in range(args.num_usvs)
                            if info['reached_goals'][i] and np.any(nv)]
            if reached_usvs:
                global_state = env.get_global_state()
                assigner.reassign(
                    obs_dict, reached_usvs, nv, tgt_idx, tgt_norm, tgt_abs,
                    SCALE, global_state=global_state, eval_mode=True)

        final_reached = int(np.sum(info['target_status']))
        success = (final_reached >= args.eval_targets) and not crush

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
            'obs_crush': obs_crush,
            'usv_crush': usv_crush,
            'targets_reached': final_reached,
            'steps': ep_steps,
            'avg_reward': float(ep_rewards.mean()),
            'colregs_violations': cr_viols,
        }
        results.append(result)

        if crush:
            crash_type = "OBS" if obs_crush else ""
            if usv_crush:
                crash_type += ("+" if crash_type else "") + "USV"
            status = f"✗ CRASH({crash_type})"
        else:
            status = "✓" if success else "✗ FAIL"
        print(f"  EP {ep}: {status} | Tgt:{final_reached}/{args.eval_targets} | "
              f"R:{ep_rewards.mean():.1f} | Steps:{ep_steps} | "
              f"CR_viol:{cr_viols}")

    # ---- 汇总 ----
    successes = sum(r['success'] for r in results)
    crashes = sum(r['crush'] for r in results)
    obs_crashes = sum(r['obs_crush'] for r in results)
    usv_crashes = sum(r['usv_crush'] for r in results)
    avg_reached = np.mean([r['targets_reached'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['avg_reward'] for r in results])
    avg_cr_viols = np.mean([r['colregs_violations'] for r in results])

    print(f"\n{'=' * 60}")
    print(f"  Evaluation Summary ({args.eval_episodes} episodes)")
    print(f"  Success Rate:      {successes}/{args.eval_episodes} "
          f"({100 * successes / args.eval_episodes:.1f}%)")
    print(f"  Collision Rate:    {crashes}/{args.eval_episodes} "
          f"(obs:{obs_crashes}, usv:{usv_crashes})")
    print(f"  Avg Targets:       {avg_reached:.1f}/{args.eval_targets}")
    print(f"  Avg Steps:         {avg_steps:.0f}")
    print(f"  Avg Reward:        {avg_reward:.2f}")
    print(f"  Avg COLREGs Viol:  {avg_cr_viols:.1f}")
    print(f"{'=' * 60}")

    eval_path = os.path.join(eval_dir, "ctde_eval_results.json")
    with open(eval_path, 'w') as f:
        json.dump({
            'summary': {
                'success_rate': successes / args.eval_episodes,
                'collision_rate': crashes / args.eval_episodes,
                'obs_collision_rate': obs_crashes / args.eval_episodes,
                'usv_collision_rate': usv_crashes / args.eval_episodes,
                'avg_targets_reached': avg_reached,
                'avg_steps': avg_steps,
                'avg_reward': avg_reward,
                'avg_colregs_violations': avg_cr_viols,
            },
            'episodes': results,
        }, f, indent=2)
    print(f"评估结果已保存至: {eval_path}")

    return results
