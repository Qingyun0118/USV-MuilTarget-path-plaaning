"""
task_assignment — 任务分配算法包

将不同的目标分配策略封装为独立模块:
  - hungarian:    匈牙利算法 (全局最优分配)
  - selector_ppo: PPO 目标选择器 (on-policy, 可学习)
  - coord_dqn:    协调 DQN 目标选择器 (off-policy, 可学习)
  - coord_utils:  多智能体协调信息构建工具

用法:
  from task_assignment import hungarian_assign, build_coord_info
  from task_assignment import SelectorPPOAgent
  from task_assignment import CoordSelectorDQNAgent
"""

from task_assignment.hungarian import hungarian_assign
from task_assignment.coord_utils import build_coord_info
from task_assignment.selector_ppo import (
    SelectorPolicyNet, SelectorValueNet,
    SelectorPPORolloutBuffer, SelectorPPOAgent,
)
from task_assignment.coord_dqn import (
    CoordSelectorQNet, SelectorDQNReplayBuffer,
    CoordSelectorDQNAgent,
)

__all__ = [
    # 匈牙利算法
    'hungarian_assign',
    # 协调工具
    'build_coord_info',
    # PPO Selector
    'SelectorPolicyNet', 'SelectorValueNet',
    'SelectorPPORolloutBuffer', 'SelectorPPOAgent',
    # 协调 DQN Selector
    'CoordSelectorQNet', 'SelectorDQNReplayBuffer',
    'CoordSelectorDQNAgent',
]
