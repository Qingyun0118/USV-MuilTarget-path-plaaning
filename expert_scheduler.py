"""
Dawood et al. (2023) 指数衰减 Expert 使用概率调度器

参考: Handling Sparse Rewards in RL Using Model Predictive Control,
      Applied Intelligence, 2023.

核心思想:
- 训练初期以高概率使用 Expert 专家策略, 为 RL 提供成功经验
- 随训练推进, Expert 概率按指数衰减, RL 逐渐接管控制
- 指数衰减公式:
    p_Expert(e) = max(p_min, p_0 * exp(-λ * e / E_total))
  其中 λ = ln(p_0 / p_min), 确保在 E_total 集结束时 p_Expert ≈ p_min

与线性/分阶段衰减相比, 指数衰减的优势:
1. 训练初期 Expert 概率下降缓慢, 保证充足的专家经验
2. 中后期衰减加速, 让 RL 更快获得独立探索机会
3. 平滑过渡, 避免阶段跳变带来的训练不稳定
"""

import numpy as np
from collections import deque


class ExpertScheduler:
    """
    Dawood et al. (2023) 指数衰减 Expert 使用概率调度器
    """

    def __init__(self, p0=0.9, p_min=0.01, total_episodes=5000,
                 success_window=100):
        """
        Args:
            p0: 初始 Expert 使用概率
            p_min: 最低 Expert 使用概率 (保留少量 Expert 防止遗忘)
            total_episodes: 总训练 episode 数 (用于计算衰减速率)
            success_window: 成功率统计滑动窗口大小
        """
        self.p0 = p0
        self.p_min = p_min
        self.total_episodes = total_episodes
        # λ = ln(p0 / p_min), 确保在 total_episodes 时衰减到 p_min
        # p0=0 时无需衰减, 直接返回 0
        if p0 > 0 and p_min > 0:
            self.decay_lambda = np.log(p0 / p_min)
        else:
            self.decay_lambda = 0.0
        self.success_history = deque(maxlen=success_window)

    def get_expert_probability(self, episode: int) -> float:
        """
        计算当前 episode 的 Expert 使用概率 (Dawood 指数衰减)

        p_Expert(e) = max(p_min, p_0 * exp(-λ * e / E_total))
        """
        if self.p0 <= 0:
            return 0.0
        p = self.p0 * np.exp(-self.decay_lambda * episode / self.total_episodes)
        return max(self.p_min, p)

    def record_episode(self, success: bool):
        """记录一个 episode 的成功/失败"""
        self.success_history.append(1.0 if success else 0.0)

    def get_success_rate(self) -> float:
        if len(self.success_history) == 0:
            return 0.0
        return float(np.mean(self.success_history))
