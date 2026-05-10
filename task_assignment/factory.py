"""
create_assigner() — 任务分配器工厂函数

根据 --assign_method 参数创建对应的 BaseAssigner 实例。
"""

import os
from task_assignment.base import BaseAssigner
from task_assignment.hungarian import HungarianAssigner
from task_assignment.hungarian_makespan import MakespanHungarianAssigner
from task_assignment.selector_ppo_makespan import MakespanSelectorPPOAssigner


def create_assigner(args, global_state_dim=None, pretrained_dir=None) -> BaseAssigner:
    """
    根据 args.assign_method 创建分配器。

    Args:
        args:              argparse.Namespace
        global_state_dim:  全局状态维度 (学习型方法需要)
        pretrained_dir:    预训练模型目录 (用于热启动)

    Returns:
        BaseAssigner 实例
    """
    method = args.assign_method

    if method == 'hungarian':
        return HungarianAssigner()

    elif method == 'hungarian_makespan':
        return MakespanHungarianAssigner(
            usv_speed_estimate=getattr(args, 'makespan_speed_estimate', 2.0))

    elif method == 'mappo_makespan':
        if global_state_dim is None:
            raise ValueError(
                "mappo_makespan 需要 global_state_dim, 但收到 None")
        assigner = MakespanSelectorPPOAssigner(
            global_state_dim=global_state_dim,
            embed_dim=getattr(args, 'sel_embed_dim', 64),
            lr_policy=args.sel_lr_policy,
            lr_value=args.sel_lr_value,
            clip_eps=args.clip_eps,
            entropy_coef=args.sel_entropy_coef,
            ppo_epochs=args.sel_ppo_epochs,
            mini_batch_size=args.sel_mini_batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            use_coordination=True,
            max_usvs=max(4, args.num_usvs),
            buffer_size=getattr(args, 'sel_buffer_size', 2048),
            # Makespan 奖励权重
            alpha=getattr(args, 'sel_alpha_reach', 1.0),
            beta=getattr(args, 'sel_beta_makespan', 0.1),
            gamma_weight=getattr(args, 'sel_gamma_actuator', 0.5),
        )
        # 从预训练 DQN 热启动
        if pretrained_dir:
            dqn_path = os.path.join(pretrained_dir, "best_selector.pth")
            if os.path.exists(dqn_path):
                assigner.load_from_dqn(dqn_path)
        return assigner

    else:
        raise ValueError(f"未知的分配方法: {method}, "
                         f"可选: hungarian, hungarian_makespan, mappo_makespan")
