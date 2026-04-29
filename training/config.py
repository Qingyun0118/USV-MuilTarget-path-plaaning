"""
训练配置 — argparse 参数解析

精简为 3 种分配方法的参数:
  - hungarian:           匈牙利算法 (Baseline 2)
  - hungarian_makespan:  最小化 Makespan 匈牙利 (Proposed 1)
  - mappo_makespan:      PPO 学习型 Makespan (Proposed 2)
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTDE MAPPO for Multi-USV Navigation with COLREGs")

    # ---- 环境 ----
    parser.add_argument("--num_usvs", default=3, type=int)
    parser.add_argument("--num_targets", default=6, type=int)
    parser.add_argument("--num_obstacles", default=6, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--usv_length", default=3.0, type=float)
    parser.add_argument("--usv_width", default=1.5, type=float)
    parser.add_argument("--target_radius", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_min", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_max", default=10.0, type=float)
    parser.add_argument("--max_steps_per_target", default=1000, type=int)

    # ---- COLREGs ----
    parser.add_argument("--colregs_k_C", default=10.0, type=float)
    parser.add_argument("--colregs_k_R", default=1.0, type=float)
    parser.add_argument("--starboard_threshold", default=-0.05, type=float)

    # ---- 预训练 ----
    parser.add_argument("--pretrained_dir", default="expert_hrl_td7_0426_seed2022_softQ",
                        type=str)
    parser.add_argument("--algo", default=None, type=str,
                        choices=["sac", "td3", "td7"],
                        help="预训练 Actor 算法类型 (默认从 pretrained_dir 自动检测)")

    # ---- MAPPO ----
    parser.add_argument("--lr_actor", default=3e-4, type=float)
    parser.add_argument("--lr_critic", default=3e-4, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--gae_lambda", default=0.95, type=float)
    parser.add_argument("--clip_eps", default=0.2, type=float)
    parser.add_argument("--entropy_coef", default=0.01, type=float)
    parser.add_argument("--max_grad_norm", default=10.0, type=float)
    parser.add_argument("--ppo_epochs", default=10, type=int)
    parser.add_argument("--mini_batch_size", default=256, type=int)
    parser.add_argument("--target_kl", default=0.15, type=float)

    # ---- 训练 ----
    parser.add_argument("--max_episodes", default=1000, type=int)
    parser.add_argument("--fixed_env_episodes", default=0, type=int)
    parser.add_argument("--min_update_steps", default=32, type=int)

    # ---- Critic warmup & COLREGs 退火 ----
    parser.add_argument("--freeze_actor_episodes", default=30, type=int)
    parser.add_argument("--colregs_k_C_start", default=1.0, type=float)
    parser.add_argument("--colregs_anneal_episodes", default=800, type=int)
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--checkpoint_interval", default=200, type=int)
    parser.add_argument("--output_dir", default=None, type=str)

    # ---- 目标分配方法 (3 种) ----
    parser.add_argument("--assign_method", default="hungarian", type=str,
                        choices=["hungarian", "hungarian_makespan",
                                 "mappo_makespan"])

    # ---- Makespan 参数 ----
    parser.add_argument("--makespan_speed_estimate", default=2.0, type=float)

    # ---- Selector PPO 参数 (仅 mappo_makespan 使用) ----
    parser.add_argument("--sel_embed_dim", default=64, type=int)
    parser.add_argument("--sel_lr_policy", default=1e-4, type=float)
    parser.add_argument("--sel_lr_value", default=3e-4, type=float)
    parser.add_argument("--sel_entropy_coef", default=0.02, type=float)
    parser.add_argument("--sel_ppo_epochs", default=5, type=int)
    parser.add_argument("--sel_mini_batch_size", default=64, type=int)
    parser.add_argument("--sel_min_decisions", default=32, type=int)
    parser.add_argument("--sel_buffer_size", default=2048, type=int)
    parser.add_argument("--sel_alpha_reach", default=1.0, type=float)
    parser.add_argument("--sel_beta_makespan", default=0.1, type=float)
    parser.add_argument("--sel_gamma_actuator", default=0.5, type=float)

    # ---- 恢复 ----
    parser.add_argument("--resume", default=None, type=str)

    # ---- 评估 ----
    parser.add_argument("--eval_episodes", default=5, type=int)
    parser.add_argument("--eval_targets", default=6, type=int)
    parser.add_argument("--eval_obstacles", default=6, type=int)
    parser.add_argument("--eval_seed", default=42, type=int)
    parser.add_argument("--eval_record_path", action="store_true", default=True)

    # ---- 模式 ----
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    args = parser.parse_args()

    # 自动设置输出目录
    if args.output_dir is None:
        args.output_dir = f"ctde_mappo_output_{args.assign_method}"

    return args
