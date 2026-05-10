"""
BaseAssigner — 任务分配器抽象基类

所有目标分配方法 (匈牙利、Makespan匈牙利、MAPPO学习型) 统一实现此接口。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict

import numpy as np


class BaseAssigner(ABC):
    """任务分配器抽象基类, 定义统一接口。"""

    name: str = "base"
    is_learnable: bool = False

    # ==========================================================================
    # 核心接口
    # ==========================================================================

    @abstractmethod
    def assign_initial(
        self,
        obs_dict: dict,
        num_usvs: int,
        valid_mask: np.ndarray,
        scale: float,
        global_state: Optional[np.ndarray] = None,
        eval_mode: bool = False,
    ):
        """
        初始目标分配 — 为每个 USV 分配第一个目标。

        Returns:
            target_idx:   List[int], 每个 USV 分配的目标索引
            target_norm:  List[np.ndarray], 归一化目标位置
            target_abs:   List[np.ndarray], 绝对坐标目标位置
            extra:        dict | None, 额外信息 (如 log_probs)
        """
        ...

    def reassign(
        self,
        obs_dict: dict,
        usv_indices: List[int],
        valid_mask: np.ndarray,
        current_target_idx: List[int],
        target_pos_norm: List[np.ndarray],
        target_pos_abs: List[np.ndarray],
        scale: float,
        global_state: Optional[np.ndarray] = None,
        eval_mode: bool = False,
    ):
        """
        目标重新分配 — 当 USV 到达当前目标后分配新目标。

        直接 in-place 修改 current_target_idx / target_pos_norm / target_pos_abs。

        Returns:
            extra: dict | None
        """
        ...

    # ==========================================================================
    # 学习型方法回调 (默认空操作)
    # ==========================================================================

    def on_step(self, rewards_arr: np.ndarray, info: dict):
        """每步回调, 用于累积奖励到 pending decisions。"""

    def on_episode_end(self, obs_dict: dict, info: dict):
        """Episode 结束回调, 用于结算 pending decisions。"""

    def update(self) -> dict:
        """训练更新, 返回 loss 字典。"""
        return {}

    def save(self, path: str):
        """保存模型参数。"""

    def load(self, path: str):
        """加载模型参数。"""

    def state_dict_for_checkpoint(self) -> Optional[dict]:
        """返回用于检查点保存的状态字典。"""
        return None

    def load_from_checkpoint(self, ckpt: dict):
        """从检查点恢复状态。"""
