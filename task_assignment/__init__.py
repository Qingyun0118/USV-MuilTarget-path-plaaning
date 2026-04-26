"""
task_assignment — 任务分配算法包

三种分配方法:
  - hungarian:          匈牙利算法全局最优二部图匹配 (Baseline 2)
  - hungarian_makespan: 最小化 Makespan 的匈牙利分配器 (Proposed 1)
  - mappo_makespan:     PPO 学习型 Makespan 分配器 (Proposed 2)
"""

from task_assignment.base import BaseAssigner
from task_assignment.hungarian import HungarianAssigner, hungarian_assign
from task_assignment.hungarian_makespan import MakespanHungarianAssigner
from task_assignment.selector_ppo_makespan import MakespanSelectorPPOAssigner
from task_assignment.factory import create_assigner

__all__ = [
    'BaseAssigner',
    'HungarianAssigner', 'hungarian_assign',
    'MakespanHungarianAssigner',
    'MakespanSelectorPPOAssigner',
    'create_assigner',
]
