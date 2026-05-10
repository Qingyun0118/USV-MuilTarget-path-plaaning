"""
匈牙利算法全局最优目标分配模块 (Baseline 2).

基于 scipy.optimize.linear_sum_assignment, 最小化 USV→Target 总欧氏距离.
"""

from typing import List, Optional

import numpy as np

from task_assignment.base import BaseAssigner
from hrl_helpers import get_usv_pos, get_target_pos


def hungarian_assign(obs_dict, num_usvs, valid_mask, scale):
    """
    匈牙利算法全局最优目标分配 — 最小化 USV→Target 总欧氏距离.

    Args:
        obs_dict:   环境观测字典 {"usv_0": {...}, ...}
        num_usvs:   USV 数量
        valid_mask:  np.array [N] bool, True = 可选目标
        scale:      坐标缩放因子 (归一化→绝对坐标)

    Returns:
        assignments: list of int, assignments[i] = USV i 分配的目标索引
                     如果 USV 多于有效目标, 多余 USV 分配 -1
    """
    from scipy.optimize import linear_sum_assignment

    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    if n_valid == 0:
        return [-1] * num_usvs

    # 构建代价矩阵, 补齐为方阵
    n_cols = max(num_usvs, n_valid)
    cost = np.full((num_usvs, n_cols), 1e6, dtype=np.float64)

    for i in range(num_usvs):
        usv_pos = get_usv_pos(obs_dict[f"usv_{i}"]) * scale
        for j, t_idx in enumerate(valid_indices):
            tgt_pos = get_target_pos(obs_dict[f"usv_{i}"], t_idx) * scale
            cost[i, j] = np.linalg.norm(usv_pos - tgt_pos)

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments = [-1] * num_usvs
    for r, c in zip(row_ind, col_ind):
        if c < n_valid:
            assignments[r] = int(valid_indices[c])

    return assignments


class HungarianAssigner(BaseAssigner):
    """
    匈牙利算法分配器 — 每次分配运行全局最优二部图匹配.

    初始分配和重分配均调用 hungarian_assign, 最小化总欧氏距离.
    """

    name = "hungarian"
    is_learnable = False

    def assign_initial(
        self,
        obs_dict: dict,
        num_usvs: int,
        valid_mask: np.ndarray,
        scale: float,
        global_state: Optional[np.ndarray] = None,
        eval_mode: bool = False,
    ):
        assignments = hungarian_assign(obs_dict, num_usvs, valid_mask, scale)

        target_idx: List[int] = []
        target_norm: List[np.ndarray] = []
        target_abs: List[np.ndarray] = []

        for i in range(num_usvs):
            t_idx = assignments[i]
            if t_idx < 0:
                usv_pos = get_usv_pos(obs_dict[f"usv_{i}"]) * scale
                n_t = len(obs_dict[f"usv_{i}"]["targets"]) // 3
                dists = [np.linalg.norm(
                    usv_pos - get_target_pos(obs_dict[f"usv_{i}"], j) * scale)
                    for j in range(n_t)]
                t_idx = int(np.argmin(dists))
            t_norm = get_target_pos(obs_dict[f"usv_{i}"], t_idx)
            target_idx.append(t_idx)
            target_norm.append(t_norm)
            target_abs.append(t_norm * scale)

        return target_idx, target_norm, target_abs, None

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
        assignments = hungarian_assign(
            obs_dict, len(current_target_idx), valid_mask, scale)
        for i in range(len(current_target_idx)):
            if assignments[i] >= 0:
                current_target_idx[i] = assignments[i]
                target_pos_norm[i] = get_target_pos(
                    obs_dict[f"usv_{i}"], assignments[i])
                target_pos_abs[i] = target_pos_norm[i] * scale
        return None
