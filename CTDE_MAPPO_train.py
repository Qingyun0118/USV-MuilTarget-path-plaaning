"""
CTDE MAPPO 训练脚本 — 多 USV 多目标协同导航 (含 COLREGs)

目标分配方法 (--assign_method):
  - hungarian:          匈牙利算法全局最优二部图匹配 (Baseline 1)
  - hungarian_makespan: 最小化 Makespan 的匈牙利分配器 (Proposed 1)
  - mappo_makespan:     PPO 学习型 Makespan 分配器 (Proposed 2)

用法:
  # Baseline 1: 匈牙利
  python CTDE_MAPPO_train.py --do_train --assign_method hungarian --max_episodes 10

  # Proposed 1: Makespan 匈牙利
  python CTDE_MAPPO_train.py --do_train --assign_method hungarian_makespan --max_episodes 10

  # Proposed 2: MAPPO 学习型
  python CTDE_MAPPO_train.py --do_train --assign_method mappo_makespan --max_episodes 10

  # 评估
  python CTDE_MAPPO_train.py --do_eval --assign_method hungarian --eval_episodes 5
  python CTDE_MAPPO_train.py --do_eval --assign_method hungarian_makespan --eval_episodes 5
  python CTDE_MAPPO_train.py --do_eval --assign_method mappo_makespan --eval_episodes 5
"""

from training.config import parse_args
from training.train_loop import train_ctde_mappo
from training.eval_loop import evaluate_ctde_mappo


def main():
    args = parse_args()

    if args.do_train:
        train_ctde_mappo(args)

    if args.do_eval:
        evaluate_ctde_mappo(args)

    if not args.do_train and not args.do_eval:
        print("请指定 --do_train 和/或 --do_eval")


if __name__ == "__main__":
    main()
