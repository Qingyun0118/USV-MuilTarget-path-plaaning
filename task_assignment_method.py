"""
task_assignment_method — 向后兼容入口

所有任务分配算法已拆分至 task_assignment 包:
  - task_assignment.hungarian:    匈牙利算法
  - task_assignment.selector_ppo: PPO 目标选择器
  - task_assignment.coord_dqn:    协调 DQN 目标选择器
  - task_assignment.coord_utils:  多智能体协调工具

本文件保留所有原始导出名, 确保外部 import 不受影响.
"""

# 向后兼容: 从 task_assignment 包重新导出所有公开 API
from task_assignment.selector_ppo import (       # noqa: F401
    SelectorPolicyNet,
    SelectorValueNet,
    SelectorPPORolloutBuffer,
    SelectorPPOAgent,
)
from task_assignment.coord_dqn import (          # noqa: F401
    CoordSelectorQNet,
    SelectorDQNReplayBuffer,
    CoordSelectorDQNAgent,
)
from task_assignment.hungarian import (          # noqa: F401
    hungarian_assign,
)
from task_assignment.coord_utils import (        # noqa: F401
    build_coord_info,
)
