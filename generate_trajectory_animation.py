"""
生成 COLREGs 测试场景（E1-E6）双 USV 轨迹动画。

默认使用 CTDE MAPPO（hungarian 目录）控制策略，
也支持 Expert-HRL-SAC 或随机动作。
"""

import argparse
import os
import random
import numpy as np
import torch

from hrl_config import set_global_seed
from my_USV_Env import X_MAX, Y_MAX, create_colregs_test_env, list_colregs_test_cases
from colregs_batch_policy_test import CTDEHungarianPolicy, ExpertHRLSACPolicy


def _prepare_goals(env):
    goals_norm = []
    for i in range(env.num_usvs):
        t_idx = min(i, len(env.targets) - 1)
        tx = env.targets[t_idx].x
        ty = env.targets[t_idx].y
        goals_norm.append(np.array([tx / X_MAX, ty / Y_MAX], dtype=np.float32))
    return goals_norm


def _build_policy(policy_name, model_dir):
    name = policy_name.lower().strip()
    if name == "ctde":
        return CTDEHungarianPolicy(model_dir), "CTDE_MAPPO_HUNGARIAN"
    if name == "expert":
        return ExpertHRLSACPolicy(model_dir), "EXPERT_HRL_SAC_329"
    if name == "random":
        return None, "RANDOM"
    raise ValueError("policy must be one of: ctde, expert, random")


def main():
    available_cases = sorted(list_colregs_test_cases().keys())
    parser = argparse.ArgumentParser(description="生成 COLREGs 场景双USV轨迹动画")
    parser.add_argument("--case", type=str, default="E3",
                        help=f"场景编号: {' / '.join(available_cases)}")
    parser.add_argument("--policy", type=str, default="ctde",
                        help="策略类型: ctde / expert / random")
    parser.add_argument("--model_dir", type=str, default="",
                        help="模型目录；ctde 默认 ctde_mappo_output_hungarian，expert 默认 expert_hrl_sac_output_329")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=180)
    parser.add_argument("--interval", type=int, default=120,
                        help="动画帧间隔(ms)")
    parser.add_argument("--speed", type=int, default=3,
                        help="动画每帧前进点数")
    parser.add_argument("--output_dir", type=str, default="colregs_animation")
    args = parser.parse_args()

    case_name = args.case.upper().strip()
    if case_name not in available_cases:
        raise ValueError(
            f"Unknown case '{args.case}', valid cases: {', '.join(available_cases)}")

    set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.model_dir:
        if args.policy.lower() == "ctde":
            model_dir = "ctde_mappo_output_hungarian"
        elif args.policy.lower() == "expert":
            model_dir = "expert_hrl_sac_output_329"
        else:
            model_dir = ""
    else:
        model_dir = args.model_dir

    policy, policy_tag = _build_policy(args.policy, model_dir)

    env = create_colregs_test_env(case_name=case_name, seed=args.seed, rou_c=1.5, rou_s=1.5)
    obs = env.reset(seed=args.seed)
    goals_norm = _prepare_goals(env)

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"{case_name.lower()}_{policy_tag.lower()}_seed{args.seed}"

    step_count = 0
    done_reason = "max_steps"

    for _ in range(args.max_steps):
        actions = np.zeros((env.num_usvs, 2), dtype=np.float32)

        if policy is None:
            actions = env.action_space.sample().astype(np.float32)
        else:
            for i in range(env.num_usvs):
                actions[i] = policy.act(obs[f"usv_{i}"], goals_norm[i])

        goal_abs = [g * np.array([X_MAX, Y_MAX], dtype=np.float32) for g in goals_norm]
        obs, _, dn, trunc, info = env.step_marl(actions, goal_abs)
        env.record_points(csv_prefix=prefix, output_dir=args.output_dir)

        step_count += 1
        if dn["__all__"]:
            done_reason = info.get("termination_reason", "done")
            break
        if trunc["__all__"]:
            done_reason = "truncated"
            break

    env.close_writers()

    gif_path = os.path.join(args.output_dir, f"{prefix}.gif")
    env.animate(
        csv_prefix=prefix,
        output_dir=args.output_dir,
        interval=args.interval,
        speed=max(1, args.speed),
        trail_length=0,
        save_path=gif_path,
    )

    print(f"Policy: {policy_tag}")
    print(f"Case: {case_name}")
    print(f"Steps: {step_count}")
    print(f"End reason: {done_reason}")
    print(f"Animation: {gif_path}")


if __name__ == "__main__":
    main()
