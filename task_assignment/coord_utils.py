"""
多智能体协调信息构建工具.

为 Selector 提供其他 USV 的位置和已分配目标信息,
用于 SelectorPPOAgent 和 CoordSelectorDQNAgent 的协调输入.
"""

import numpy as np


def build_coord_info(obs_dict, num_usvs, usv_idx,
                     current_target_idx, target_pos_norm, max_usvs=4):
    """
    构建 USV i 的协调信息向量 (其他 USV 的位置 + 已分配目标位置).

    Returns:
        np.array, shape = (4 * (max_usvs - 1),)
    """
    from hrl_helpers import get_usv_pos

    info = np.zeros(4 * (max_usvs - 1), dtype=np.float32)
    slot = 0
    for j in range(num_usvs):
        if j == usv_idx:
            continue
        if slot >= max_usvs - 1:
            break
        pos = get_usv_pos(obs_dict[f"usv_{j}"])
        tgt = target_pos_norm[j] if j < len(target_pos_norm) else np.zeros(2)
        base = slot * 4
        info[base:base + 2] = pos
        info[base + 2:base + 4] = tgt
        slot += 1
    return info
