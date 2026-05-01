"""
批量评估任务成功率脚本

对多个随机种子分别评估 M 个 episode，计算每个种子的成功率，
最后输出 N 个成功率的均值和样本均方差。

用法:
    python batch_eval_success_rate.py --model_dir expert_hrl_td7_0426_seed2022_softQ --eval_episodes 100
    python batch_eval_success_rate.py --model_dir expert_hrl_sac_output --eval_episodes 100 --seeds 42 123 456
"""

import argparse
import math
import os
import sys
import time
import numpy as np
import torch

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from my_USV_Env import MyUsvEnv
from hrl_config import device, SCALE, set_global_seed
from hrl_helpers import (
    get_selector_state, get_actuator_state,
    get_usv_pos, get_target_pos,
)
from selector_agent import SelectorAgent

# ============================================================================
# 硬编码的评估随机种子列表
# ============================================================================
EVAL_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]


# ============================================================================
# 障碍物半径自动调整
# ============================================================================

def auto_obstacle_radius_max(num_obstacles, map_size=100.0, rou_c=1.5, rou_s=1.5,
                             base_n=6, base_rmax=10.0, r_min=3.0):
    """
    根据障碍物数量自动调整最大半径，保持空间密度大致恒定。

    推导思路：
    - 可用区域边长 = map_size - 2*(r_max + rou_c + rou_s + 5)
    - 单个障碍物平均投影面积 ≈ π * ((r_min+r_max)/2 + rou_c + rou_s)^2
    - 令"可用面积 / 总投影面积"与 base case (N=6, rmax=10) 保持一致

    Returns:
        调整后的 obstacle_radius_max
    """
    if num_obstacles <= base_n:
        return base_rmax

    # base case 的空间冗余比例（约 2.4，已包含随机放置的 inefficiency）
    base_margin = base_rmax + rou_c + rou_s + 5
    base_avail = (map_size - 2 * base_margin) ** 2
    base_avg_r = (r_min + base_rmax) / 2
    base_safe_r = base_avg_r + rou_c + rou_s
    base_total_area = base_n * math.pi * base_safe_r ** 2
    target_ratio = base_avail / base_total_area

    # 求解方程：(A - 2*rmax)^2 = B * (rmax/2 + c)^2
    # 其中 A = map_size - 2*(rou_c+rou_s+5), c = r_min/2 + rou_c + rou_s
    margin_const = rou_c + rou_s + 5          # 默认 8.0
    A = map_size - 2 * margin_const           # 默认 84.0
    c = r_min / 2 + rou_c + rou_s             # 默认 4.5
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
    """
    加载 Selector 模型。
    优先加载 selector.pth（最后模型），fallback 到 best_selector.pth。
    """
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
    """
    根据算法类型加载 Actuator 模型。
    优先加载 actuator_actor.pth（最后模型），fallback 到 best_actuator_actor.pth。
    TD7 额外加载 encoder。
    """
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
        # 检查活跃目标
        target_status = obs_dict['targets'][2::3]
        valid_mask = (target_status == 0)
        if not np.any(valid_mask):
            ep_done = True
            break

        # Selector 选择目标 (评估模式: argmax)
        selector_state = get_selector_state(obs_dict)
        target_idx = selector.select_action(selector_state, valid_mask, eval_mode=True)
        target_pos_norm = get_target_pos(obs_dict, target_idx)
        target_pos_abs = target_pos_norm * SCALE

        # 导航到目标
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

            # 动作选择 (确定性，无探索)
            if algo == 'sac':
                action = actuator.select_action(actuator_state, relative_goal, evaluate=True)
            else:
                action = actuator.select_action(actuator_state, relative_goal, noise=0.0)

            next_obs_dict, sub_done, crush, truncated, info = env.step(action, target_pos_abs)
            obs_dict = next_obs_dict

            # 检查到达目标
            reached_indices = info.get('reached_targets', [])
            if len(reached_indices) > 0:
                for r_idx in reached_indices:
                    targets_reached += 1
                if target_idx in reached_indices:
                    target_reached = True

            # 检查是否所有目标完成
            target_status = obs_dict['targets'][2::3]
            valid_mask = (target_status == 0)
            ep_done = not np.any(valid_mask)

    success = (targets_reached >= num_targets and not crush)
    return success, targets_reached, total_steps, crush


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="批量评估任务成功率 — 多随机种子 × 多 episode")

    # 必需参数
    parser.add_argument("--model_dir", type=str, required=True,
                        help="模型文件夹路径 (如 expert_hrl_td7_0426_seed2022_softQ)")

    # 评估参数
    parser.add_argument("--eval_episodes", "-m", type=int, default=100,
                        help="每个种子的评估 episode 数 (默认: 100)")
    parser.add_argument("--eval_targets", type=int, default=3,
                        help="目标数量 (默认: 3)")
    parser.add_argument("--eval_obstacles", type=int, default=6,
                        help="障碍物数量 (默认: 6)")
    parser.add_argument("--max_steps_per_target", type=int, default=1000,
                        help="每个目标的最大导航步数 (默认: 1000)")

    # 可选：覆盖硬编码种子
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="自定义随机种子列表，覆盖默认硬编码种子")

    # 环境参数
    parser.add_argument("--usv_length", type=float, default=3.0)
    parser.add_argument("--usv_width", type=float, default=1.5)
    parser.add_argument("--target_radius", type=float, default=5.0)
    parser.add_argument("--obstacle_radius_min", type=float, default=3.0)
    parser.add_argument("--obstacle_radius_max", type=float, default=10.0)
    parser.add_argument("--use_random_shapes", action="store_true", default=True)

    args = parser.parse_args()

    # --- 校验 ---
    if not os.path.isdir(args.model_dir):
        print(f"ERROR: 模型目录不存在: {args.model_dir}")
        sys.exit(1)

    # --- 检测算法 ---
    algo = detect_algo(args.model_dir)
    seeds = args.seeds if args.seeds is not None else EVAL_SEEDS
    M = args.eval_episodes
    N = len(seeds)

    # --- 自动调整障碍物最大半径（高密度场景） ---
    original_rmax = args.obstacle_radius_max
    adjusted_rmax = auto_obstacle_radius_max(args.eval_obstacles)
    if adjusted_rmax != original_rmax:
        args.obstacle_radius_max = adjusted_rmax
        print(f"[AUTO] 障碍物数={args.eval_obstacles}，自动调整 obstacle_radius_max: "
              f"{original_rmax:.1f} → {adjusted_rmax:.2f}")

    print("=" * 60)
    print(f"  批量评估 — {algo.upper()}")
    print("=" * 60)
    print(f"  模型目录:     {args.model_dir}")
    print(f"  随机种子数:   {N}")
    print(f"  每种子评估数: {M}")
    print(f"  总评估回合:   {N * M}")
    print(f"  环境:         {args.eval_targets} 目标 / {args.eval_obstacles} 障碍物")
    print(f"  障碍半径范围: {args.obstacle_radius_min:.1f} ~ {args.obstacle_radius_max:.2f}")
    print(f"  种子列表:     {seeds}")

    # --- 加载模型 (一次性) ---
    print(f"\n--- 加载模型 ---")

    # 临时环境获取 state_dim
    temp_env = MyUsvEnv(
        num_targets=args.eval_targets,
        num_obstacles=args.eval_obstacles,
        seed=0,
        usv_length=args.usv_length,
        usv_width=args.usv_width,
        target_radius=args.target_radius,
        obstacle_radius_range=(args.obstacle_radius_min, args.obstacle_radius_max),
        use_random_shapes=args.use_random_shapes,
        current_type="generated",
    )
    obs = temp_env.reset()
    state_dim = get_actuator_state(obs).shape[0]
    temp_env.close() if hasattr(temp_env, 'close') else None

    selector = load_selector(args.model_dir)
    actuator = load_actuator(algo, args.model_dir, state_dim)

    if selector is None or actuator is None:
        print("\nERROR: 模型加载失败，无法继续评估。")
        sys.exit(1)

    # 设为评估模式
    selector.q_net.eval()
    actuator.actor.eval()
    if algo == 'td7' and hasattr(actuator, 'fixed_encoder'):
        actuator.fixed_encoder.eval()

    # --- 逐种子评估 ---
    print(f"\n{'=' * 60}")
    print(f"  开始批量评估...")
    print(f"{'=' * 60}")

    per_seed_rates = []
    all_results = []
    total_start = time.time()

    for seed_idx, seed in enumerate(seeds):
        env = MyUsvEnv(
            num_targets=args.eval_targets,
            num_obstacles=args.eval_obstacles,
            seed=seed,
            usv_length=args.usv_length,
            usv_width=args.usv_width,
            target_radius=args.target_radius,
            obstacle_radius_range=(args.obstacle_radius_min, args.obstacle_radius_max),
            use_random_shapes=args.use_random_shapes,
            current_type="generated",
        )

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

            # 每 50 回合打印进度
            if (ep + 1) % 50 == 0:
                print(f"  [Seed {seed}] Ep {ep + 1}/{M}: "
                      f"running SR={successes / (ep + 1):.3f}")

        seed_elapsed = time.time() - seed_start
        rate = successes / M
        per_seed_rates.append(rate)

        print(f"  [Seed {seed}] 完成: SR={rate:.4f} ({successes}/{M}), "
              f"碰撞={collisions}, "
              f"平均目标数/ep={total_reached / M:.1f}, "
              f"平均步数/ep={total_steps_sum / M:.1f}, "
              f"耗时={seed_elapsed:.1f}s")

        all_results.append({
            'seed': seed,
            'success_rate': rate,
            'successes': successes,
            'episodes': M,
            'collisions': collisions,
            'avg_targets': total_reached / M,
            'avg_steps': total_steps_sum / M,
        })

    total_elapsed = time.time() - total_start

    # --- 汇总 ---
    rates = np.array(per_seed_rates)
    mean_rate = float(np.mean(rates))
    std_rate = float(np.std(rates, ddof=1))  # 样本均方差 (ddof=1)

    print(f"\n{'=' * 60}")
    print(f"  批量评估汇总")
    print(f"{'=' * 60}")
    print(f"  算法:           {algo.upper()}")
    print(f"  模型目录:       {args.model_dir}")
    print(f"  种子数 × 回合:  {N} × {M} = {N * M}")
    print(f"  环境:           {args.eval_targets}T / {args.eval_obstacles}O")
    print(f"  总耗时:         {total_elapsed:.1f}s")
    print(f"{'─' * 60}")
    print(f"  各种子成功率:")
    for r in all_results:
        print(f"    Seed {r['seed']:5d}  {r['success_rate']:.4f}  "
              f"({r['successes']}/{r['episodes']})")
    print(f"{'─' * 60}")
    print(f"  成功率均值 (Mean):    {mean_rate:.4f}")
    print(f"  成功率均方差 (Std):   {std_rate:.4f}")
    print(f"  Mean ± Std:           {mean_rate:.4f} ± {std_rate:.4f}")
    print(f"{'=' * 60}")

    # --- 保存结果文件 ---
    result_path = os.path.join(args.model_dir, "batch_eval_results.txt")
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write(f"批量评估结果\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"算法:     {algo.upper()}\n")
        f.write(f"模型目录: {args.model_dir}\n")
        f.write(f"种子数:   {N}\n")
        f.write(f"每种子评估回合数: {M}\n")
        f.write(f"总评估回合数:     {N * M}\n")
        f.write(f"环境:     {args.eval_targets}T / {args.eval_obstacles}O\n")
        f.write(f"总耗时:   {total_elapsed:.1f}s\n")
        f.write(f"{'─' * 60}\n")
        f.write(f"种子列表: {seeds}\n")
        f.write(f"各种子成功率:\n")
        for r in all_results:
            f.write(f"  Seed {r['seed']:5d}: {r['success_rate']:.4f} "
                    f"({r['successes']}/{r['episodes']})  "
                    f"碰撞={r['collisions']}  "
                    f"平均目标={r['avg_targets']:.1f}  "
                    f"平均步数={r['avg_steps']:.1f}\n")
        f.write(f"{'─' * 60}\n")
        f.write(f"成功率均值 (Mean):  {mean_rate:.4f}\n")
        f.write(f"成功率均方差 (Std): {std_rate:.4f}\n")
        f.write(f"Mean ± Std:         {mean_rate:.4f} ± {std_rate:.4f}\n")

    print(f"\n结果已保存至: {result_path}")


if __name__ == "__main__":
    main()
