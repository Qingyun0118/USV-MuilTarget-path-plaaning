"""
匈牙利算法全局最优目标分配模块.

基于 scipy.optimize.linear_sum_assignment, 最小化 USV→Target 总欧氏距离.
"""

import numpy as np


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
    from hrl_helpers import get_usv_pos, get_target_pos

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
