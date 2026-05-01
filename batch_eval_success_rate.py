"""
批量评估任务成功率脚本（支持多训练种子目录一键评估）

对多个训练种子的模型分别评估，每个模型在相同的 eval 种子集合下各跑 M 个 episode，
计算每个训练种子的成功率，最后汇总输出 method-level 的均值和样本均方差。

用法（单目录，兼容旧版）:
    python batch_eval_success_rate.py --model_dir expert_hrl_td7_seed2022_softQ --eval_episodes 100

用法（多训练种子目录，推荐）:
    python batch_eval_success_rate.py \
        --model_dirs \
            expert_hrl_td7_seed2021_softQ \
            expert_hrl_td7_seed2022_softQ \
            expert_hrl_td7_seed2023_softQ \
        --eval_episodes 100
"""

import argparse
import math
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from my_USV_Env import MyUsvEnv
from hrl_config import device, SCALE, set_global_seed
from hrl_helpers import (
    get_selector_state, get_actuator_state,
    get_usv_pos, get_target_pos,
)
from selector_agent import SelectorAgent

# ============================================================================
# 默认评估随机种子列表（覆盖不同场景布局）
# ============================================================================
DEFAULT_EVAL_SEEDS = [42, 123, 456, 789, 1024]


# ============================================================================
# 障碍物半径自动调整
# ============================================================================

def auto_obstacle_radius_max(num_obstacles, map_size=100.0, rou_c=1.5, rou_s=1.5,
                             base_n=6, base_rmax=10.0, r_min=3.0):
    """
    根据障碍物数量自动调整最大半径，保持空间密度大致恒定。
    """
    if num_obstacles <= base_n:
        return base_rmax

    base_margin = base_rmax + rou_c + rou_s + 5
    base_avail = (map_size - 2 * base_margin) ** 2
    base_avg_r = (r_min + base_rmax) / 2
    base_safe_r = base_avg_r + rou_c + rou_s
    base_total_area = base_n * math.pi * base_safe_r ** 2
    target_ratio = base_avail / base_total_area

    margin_const = rou_c + rou_s + 5
    A = map_size - 2 * margin_const
    c = r_min / 2 + rou_c + rou_s
    B = target_ratio * num_obstacles * math.pi

    sqrt_B = math.sqrt(B)
    numerator = A - sqrt_B * c
    denominator = 2 + sqrt_B / 2

    if denominator <= 0 or numerator <= 0:
        return r_min

    rmax = numerator / denominator
    return max(r_min, min(base_rmax, rmax))


# ============================================================================
# 模型加载工具
# ============================================================================

def detect_algo(model_dir):
    """从文件夹名检测算法类型 (td7 / td3 / sac)。"""
    name = os.path.basename(os.path.normpath(model_dir)).lower()
    if 'td7' in name:
        return 'td7'
    elif 'td3' in name:
        return 'td3'
    elif 'sac' in name:
        return 'sac'
    raise SystemExit(
        f"无法从文件夹名检测算法类型 (td7/td3/sac): {name}\n"
        f"请确保文件夹名包含 'td7'、'td3' 或 'sac'。")


def pick_model(model_dir, candidates):
    """按顺序尝试候选文件名，返回第一个存在的文件路径，都不存在返回 None。"""
    for name in candidates:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    return None


def safe_load(path):
    """跨版本兼容的 torch.load。"""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except (TypeError, ValueError):
        return torch.load(path, map_location=device, weights_only=False)


def load_selector(model_dir):
    """加载 Selector 模型。"""
    selector = SelectorAgent()
    model_path = pick_model(model_dir, ["selector.pth", "best_selector.pth"])
    if model_path is None:
        print(f"  WARNING: 未找到 Selector 模型 (selector.pth / best_selector.pth)")
        return None
    try:
        selector.q_net.load_state_dict(safe_load(model_path))
        print(f"  Selector 已加载: {os.path.basename(model_path)}")
        return selector
    except Exception as e:
        print(f"  WARNING: Selector 加载失败 ({model_path}): {e}")
        return None


def load_actuator(algo, model_dir, state_dim):
    """根据算法类型加载 Actuator 模型。"""
    actor_path = pick_model(model_dir, ["actuator_actor.pth", "best_actuator_actor.pth"])
    if actor_path is None:
        print(f"  WARNING: 未找到 Actuator 模型")
        return None

    if algo == 'td7':
        from td7_agent import TD7ActuatorAgent
        actuator = TD7ActuatorAgent(state_dim, 2, 2, max_action=1.0)
        actuator.actor.load_state_dict(safe_load(actor_path))
        print(f"  TD7 Actor 已加载:   {os.path.basename(actor_path)}")

        encoder_path = pick_model(model_dir, ["actuator_encoder.pth", "best_actuator_encoder.pth"])
        if encoder_path:
            actuator.fixed_encoder.load_state_dict(safe_load(encoder_path))
            print(f"  TD7 Encoder 已加载: {os.path.basename(encoder_path)}")
        else:
            print(f"  WARNING: 未找到 TD7 Encoder，使用随机权重")

    elif algo == 'td3':
        from td3_agent import TD3ActuatorAgent
        actuator = TD3ActuatorAgent(state_dim, 2, 2, max_action=1.0)
        actuator.actor.load_state_dict(safe_load(actor_path))
        print(f"  TD3 Actor 已加载:   {os.path.basename(actor_path)}")

    elif algo == 'sac':
        from sac_agent import SACActuatorAgent
        actuator = SACActuatorAgent(state_dim, 2, 2, max_action=1.0)
        actuator.actor.load_state_dict(safe_load(actor_path))
        print(f"  SAC Actor 已加载:   {os.path.basename(actor_path)}")

    return actuator


# ============================================================================
# 单 episode 评估
# ============================================================================

@torch.no_grad()
def run_eval_episode(env, selector, actuator, algo, max_steps_per_target, episode_seed):
    """
    运行单个评估 episode。

    Returns:
        (success, targets_reached, total_steps, collision)
    """
    obs_dict = env.reset(seed=episode_seed)
    num_targets = env.num_targets

    ep_done = False
    crush = False
    targets_reached = 0
    total_steps = 0
    max_steps = max_steps_per_target * num_targets * 2

    while not (ep_done or crush or total_steps >= max_steps):
        target_status = obs_dict['targets'][2::3]
        valid_mask = (target_status == 0)
        if not np.any(valid_mask):
            ep_done = True
            break

        selector_state = get_selector_state(obs_dict)
        target_idx = selector.select_action(selector_state, valid_mask, eval_mode=True)
        target_pos_norm = get_target_pos(obs_dict, target_idx)
        target_pos_abs = target_pos_norm * SCALE

        target_reached = False
        nav_steps = 0
        while not (ep_done or target_reached or crush
                   or nav_steps >= max_steps_per_target
                   or total_steps >= max_steps):
            nav_steps += 1
            total_steps += 1

            actuator_state = get_actuator_state(obs_dict)
            curr_usv_pos = get_usv_pos(obs_dict)
            relative_goal = target_pos_norm - curr_usv_pos

            if algo == 'sac':
                action = actuator.select_action(actuator_state, relative_goal, evaluate=True)
            else:
                action = actuator.select_action(actuator_state, relative_goal, noise=0.0)

            next_obs_dict, sub_done, crush, truncated, info = env.step(action, target_pos_abs)
            obs_dict = next_obs_dict

            reached_indices = info.get('reached_targets', [])
            if len(reached_indices) > 0:
                for r_idx in reached_indices:
                    targets_reached += 1
                if target_idx in reached_indices:
                    target_reached = True

            target_status = obs_dict['targets'][2::3]
            valid_mask = (target_status == 0)
            ep_done = not np.any(valid_mask)

    success = (targets_reached >= num_targets and not crush)
    return success, targets_reached, total_steps, crush


# ============================================================================
# 单模型（单训练种子）评估
# ============================================================================

def eval_single_model(model_dir, args, state_dim, env_kwargs):
    """
    评估单个训练种子的模型。

    Returns:
        dict with keys: model_dir, algo, seed_results, mean_rate, std_rate, total_episodes, elapsed
    """
    if not os.path.isdir(model_dir):
        print(f"[SKIP] 目录不存在: {model_dir}")
        return None

    algo = detect_algo(model_dir)
    eval_seeds = args.seeds if args.seeds is not None else DEFAULT_EVAL_SEEDS
    M = args.eval_episodes
    N_eval = len(eval_seeds)

    print(f"\n{'=' * 60}")
    print(f"  评估模型: {os.path.basename(os.path.normpath(model_dir))}")
    print(f"  算法: {algo.upper()} | eval_seeds={N_eval} | episodes/seed={M}")
    print(f"{'=' * 60}")

    selector = load_selector(model_dir)
    actuator = load_actuator(algo, model_dir, state_dim)
    if selector is None or actuator is None:
        print(f"  [ERROR] 模型加载失败，跳过: {model_dir}")
        return None

    selector.q_net.eval()
    actuator.actor.eval()
    if algo == 'td7' and hasattr(actuator, 'fixed_encoder'):
        actuator.fixed_encoder.eval()

    per_seed_rates = []
    all_results = []
    model_start = time.time()

    for seed_idx, seed in enumerate(eval_seeds):
        env = MyUsvEnv(**env_kwargs, seed=seed)

        successes = 0
        collisions = 0
        total_reached = 0
        total_steps_sum = 0
        seed_start = time.time()

        for ep in range(M):
            success, reached, steps, crush = run_eval_episode(
                env, selector, actuator, algo,
                args.max_steps_per_target,
                episode_seed=seed + ep,
            )
            successes += int(success)
            collisions += int(crush)
            total_reached += reached
            total_steps_sum += steps

            if (ep + 1) % 50 == 0:
                print(f"  [EvalSeed {seed}] Ep {ep + 1}/{M}: running SR={successes / (ep + 1):.3f}")

        seed_elapsed = time.time() - seed_start
        rate = successes / M
        per_seed_rates.append(rate)

        print(f"  [EvalSeed {seed}] 完成: SR={rate:.4f} ({successes}/{M}), "
              f"碰撞={collisions}, 平均目标={total_reached/M:.1f}, "
              f"耗时={seed_elapsed:.1f}s")

        all_results.append({
            'eval_seed': seed,
            'success_rate': rate,
            'successes': successes,
            'episodes': M,
            'collisions': collisions,
            'avg_targets': total_reached / M,
            'avg_steps': total_steps_sum / M,
        })

        if hasattr(env, 'close'):
            env.close()

    model_elapsed = time.time() - model_start
    rates = np.array(per_seed_rates)
    mean_rate = float(np.mean(rates))
    std_rate = float(np.std(rates, ddof=1))

    # 保存单模型结果（追加到该目录下的文件）
    result_path = os.path.join(model_dir, "batch_eval_results.txt")
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"批量评估结果 — {os.path.basename(os.path.normpath(model_dir))}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"算法:     {algo.upper()}\n")
        f.write(f"模型目录: {model_dir}\n")
        f.write(f"评估种子数: {N_eval}\n")
        f.write(f"每种子评估回合数: {M}\n")
        f.write(f"总评估回合数:     {N_eval * M}\n")
        f.write(f"环境:     {args.eval_targets}T / {args.eval_obstacles}O\n")
        f.write(f"总耗时:   {model_elapsed:.1f}s\n")
        f.write(f"{'─' * 60}\n")
        f.write(f"评估种子列表: {eval_seeds}\n")
        f.write(f"各评估种子成功率:\n")
        for r in all_results:
            f.write(f"  EvalSeed {r['eval_seed']:5d}: {r['success_rate']:.4f} "
                    f"({r['successes']}/{r['episodes']})  "
                    f"碰撞={r['collisions']}  "
                    f"平均目标={r['avg_targets']:.1f}  "
                    f"平均步数={r['avg_steps']:.1f}\n")
        f.write(f"{'─' * 60}\n")
        f.write(f"成功率均值 (Mean):  {mean_rate:.4f}\n")
        f.write(f"成功率均方差 (Std): {std_rate:.4f}\n")
        f.write(f"Mean ± Std:         {mean_rate:.4f} ± {std_rate:.4f}\n")

    print(f"  单模型结果已保存至: {result_path}")

    return {
        'model_dir': model_dir,
        'algo': algo,
        'eval_seed_results': all_results,
        'mean_rate': mean_rate,
        'std_rate': std_rate,
        'total_episodes': N_eval * M,
        'elapsed': model_elapsed,
    }


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="批量评估任务成功率 — 多训练种子 × 多评估场景 × 多 episode")

    # 模型目录参数（二选一）
    parser.add_argument("--model_dir", type=str, default=None,
                        help="(兼容旧版) 单个模型文件夹路径")
    parser.add_argument("--model_dirs", type=str, nargs="+", default=None,
                        help="多训练种子的模型目录列表（推荐，用于同方法多种子评估）")

    # 评估参数
    parser.add_argument("--eval_episodes", "-m", type=int, default=100,
                        help="每个评估种子的 episode 数 (默认: 100)")
    parser.add_argument("--eval_targets", type=int, default=3,
                        help="目标数量 (默认: 3)")
    parser.add_argument("--eval_obstacles", type=int, default=6,
                        help="障碍物数量 (默认: 6)")
    parser.add_argument("--max_steps_per_target", type=int, default=1000,
                        help="每个目标的最大导航步数 (默认: 1000)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="自定义评估随机种子列表，覆盖默认")

    # 环境参数
    parser.add_argument("--usv_length", type=float, default=3.0)
    parser.add_argument("--usv_width", type=float, default=1.5)
    parser.add_argument("--target_radius", type=float, default=5.0)
    parser.add_argument("--obstacle_radius_min", type=float, default=3.0)
    parser.add_argument("--obstacle_radius_max", type=float, default=10.0)
    parser.add_argument("--use_random_shapes", action="store_true", default=True)

    args = parser.parse_args()

    # --- 确定模型目录列表 ---
    if args.model_dirs is not None:
        model_dirs = args.model_dirs
    elif args.model_dir is not None:
        model_dirs = [args.model_dir]
    else:
        parser.error("必须指定 --model_dir 或 --model_dirs 之一")

    # --- 自动调整障碍物最大半径 ---
    original_rmax = args.obstacle_radius_max
    adjusted_rmax = auto_obstacle_radius_max(args.eval_obstacles)
    if adjusted_rmax != original_rmax:
        args.obstacle_radius_max = adjusted_rmax
        print(f"[AUTO] 障碍物数={args.eval_obstacles}，自动调整 obstacle_radius_max: "
              f"{original_rmax:.1f} -> {adjusted_rmax:.2f}")

    # --- 统一的环境参数（所有模型共用） ---
    env_kwargs = {
        'num_targets': args.eval_targets,
        'num_obstacles': args.eval_obstacles,
        'usv_length': args.usv_length,
        'usv_width': args.usv_width,
        'target_radius': args.target_radius,
        'obstacle_radius_range': (args.obstacle_radius_min, args.obstacle_radius_max),
        'use_random_shapes': args.use_random_shapes,
        'current_type': "generated",
    }

    # --- 创建临时环境获取 state_dim ---
    temp_env = MyUsvEnv(**env_kwargs, seed=0)
    obs = temp_env.reset()
    state_dim = get_actuator_state(obs).shape[0]
    if hasattr(temp_env, 'close'):
        temp_env.close()

    # --- 逐个模型评估 ---
    print("\n" + "=" * 70)
    print("  多模型批量评估")
    print("=" * 70)
    print(f"  模型数量:     {len(model_dirs)}")
    print(f"  评估种子数:   {len(args.seeds) if args.seeds else len(DEFAULT_EVAL_SEEDS)}")
    print(f"  每种子回合数: {args.eval_episodes}")
    print(f"  环境:         {args.eval_targets}T / {args.eval_obstacles}O")
    print(f"  State dim:    {state_dim}")
    print("=" * 70)

    all_model_results = []
    total_start = time.time()

    for idx, model_dir in enumerate(model_dirs):
        print(f"\n[{idx + 1}/{len(model_dirs)}] 开始评估: {model_dir}")
        result = eval_single_model(model_dir, args, state_dim, env_kwargs)
        if result is not None:
            all_model_results.append(result)

    total_elapsed = time.time() - total_start

    if len(all_model_results) == 0:
        print("\n[ERROR] 没有成功评估任何模型。")
        sys.exit(1)

    # --- 汇总输出 ---
    print("\n" + "=" * 70)
    print("  汇总结果")
    print("=" * 70)

    rates = np.array([r['mean_rate'] for r in all_model_results])
    method_mean = float(np.mean(rates))
    method_std = float(np.std(rates, ddof=1))
    method_sem = method_std / math.sqrt(len(rates))

    print(f"\n  评估模型数:         {len(all_model_results)}")
    print(f"  总评估回合数:       {sum(r['total_episodes'] for r in all_model_results)}")
    print(f"  总耗时:             {total_elapsed:.1f}s")
    print(f"\n  {'模型目录':<40} {'Mean':>8} {'Std':>8} {'Algo':>6}")
    print(f"  {'─' * 66}")
    for r in all_model_results:
        short_name = os.path.basename(os.path.normpath(r['model_dir']))
        print(f"  {short_name:<40} {r['mean_rate']:>8.4f} {r['std_rate']:>8.4f} {r['algo']:>6}")

    print(f"\n  ── Method-level 统计 ──")
    print(f"  成功率均值 (Mean):      {method_mean:.4f}")
    print(f"  成功率均方差 (Std):     {method_std:.4f}")
    print(f"  标准误 (SEM):           {method_sem:.4f}")
    print(f"  95% 置信区间:           [{method_mean - 1.96*method_sem:.4f}, {method_mean + 1.96*method_sem:.4f}]")
    print(f"  Mean ± Std:             {method_mean:.4f} ± {method_std:.4f}")
    print("=" * 70)

    # --- 保存汇总结果 ---
    summary_dir = os.path.dirname(os.path.abspath(model_dirs[0]))
    summary_path = os.path.join(summary_dir, "batch_eval_summary.txt")

    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"多模型批量评估汇总 — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"环境参数: {args.eval_targets}T / {args.eval_obstacles}O, "
                f"obs_rmax={args.obstacle_radius_max:.2f}\n")
        f.write(f"评估种子列表: {args.seeds if args.seeds else DEFAULT_EVAL_SEEDS}\n")
        f.write(f"每种子评估回合: {args.eval_episodes}\n")
        f.write(f"总耗时: {total_elapsed:.1f}s\n")
        f.write(f"\n{'模型目录':<40} {'Mean':>8} {'Std':>8} {'Algo':>6}\n")
        f.write(f"{'─' * 66}\n")
        for r in all_model_results:
            short_name = os.path.basename(os.path.normpath(r['model_dir']))
            f.write(f"{short_name:<40} {r['mean_rate']:>8.4f} {r['std_rate']:>8.4f} {r['algo']:>6}\n")
        f.write(f"\nMethod-level 统计:\n")
        f.write(f"  成功率均值 (Mean):    {method_mean:.4f}\n")
        f.write(f"  成功率均方差 (Std):   {method_std:.4f}\n")
        f.write(f"  标准误 (SEM):         {method_sem:.4f}\n")
        f.write(f"  95% CI:               [{method_mean - 1.96*method_sem:.4f}, {method_mean + 1.96*method_sem:.4f}]\n")
        f.write(f"  Mean ± Std:           {method_mean:.4f} ± {method_std:.4f}\n")
        f.write(f"{'=' * 70}\n")

    print(f"\n汇总结果已保存至: {summary_path}")


if __name__ == "__main__":
    main()
