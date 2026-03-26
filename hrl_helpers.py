"""
HRL 辅助函数

包含 Selector/Actuator 的状态提取函数和执行器奖励计算函数。
"""

import numpy as np


def get_selector_state(obs_dict):
    """Selector 状态: USV 位置 (2) + 所有目标信息 (3 * num_targets)"""
    return np.concatenate([obs_dict['o_usv'][0:2], obs_dict['targets']])


def get_actuator_state(obs_dict):
    """Actuator 状态: USV 完整状态 (7) + 传感器 (15) + 海流 (4) = 26"""
    return np.concatenate([obs_dict['o_usv'], obs_dict['o_detector'], obs_dict['o_current']])


def get_usv_pos(obs_dict):
    """获取归一化 USV 位置"""
    return obs_dict['o_usv'][0:2].copy()


def get_target_pos(obs_dict, target_idx):
    """获取归一化目标位置"""
    base = target_idx * 3
    return np.array([obs_dict['targets'][base], obs_dict['targets'][base + 1]])


def compute_actuator_reward(prev_dist, next_dist, min_sensor, crush, sub_done,
                            usv_pos_norm=None):
    """
    计算执行器奖励

    Args:
        prev_dist: 上一步到目标的归一化距离
        next_dist: 当前步到目标的归一化距离
        min_sensor: 传感器最小读数 (归一化, 0=碰撞)
        crush: 是否碰撞
        sub_done: 是否到达目标区域
        usv_pos_norm: USV 归一化位置 [x, y], 范围 [0, 1], 用于边界惩罚
    """
    # 接近奖励 (缩小系数, 避免远距离高速冲刺获得过高奖励)
    approach = (prev_dist - next_dist) * 5.0

    # 障碍物接近惩罚 (指数形式, 近距离时急剧增长)
    # min_sensor ∈ [0, 1], 阈值 0.4 ≈ 6m; 越近惩罚越大
    obs_threshold = 0.4
    if min_sensor < obs_threshold:
        proximity_penalty = -2.0 * np.exp(3.0 * (obs_threshold - min_sensor)) + 2.0
    else:
        proximity_penalty = 0.0

    # 安全奖励: 远离障碍物时给予正向鼓励, 使存活本身有价值
    if min_sensor > 0.6:
        safety_bonus = 0.02
    elif min_sensor > obs_threshold:
        safety_bonus = 0.01
    else:
        safety_bonus = 0.0

    # 边界惩罚: 距边界 5% 内开始渐进惩罚, 超出边界时更强
    boundary_penalty = 0.0
    boundary_margin = 0.05
    if usv_pos_norm is not None:
        for coord in usv_pos_norm:
            if coord < 0.0:
                boundary_penalty -= (abs(coord) + boundary_margin) * 20.0
            elif coord < boundary_margin:
                boundary_penalty -= (boundary_margin - coord) * 10.0
            elif coord > 1.0:
                boundary_penalty -= (coord - 1.0 + boundary_margin) * 20.0
            elif coord > 1.0 - boundary_margin:
                boundary_penalty -= (coord - (1.0 - boundary_margin)) * 10.0

    if crush:
        return -100.0
    elif sub_done:
        return approach + 10.0 + boundary_penalty
    else:
        return approach + proximity_penalty + safety_bonus + boundary_penalty - 0.02
