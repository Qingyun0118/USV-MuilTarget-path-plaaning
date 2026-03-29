"""
SOM-based Global Target Scheduler for USV Clusters

参考: Zhao et al. (2024) — A bi-layer task decision-making approach
      for USV clusters with multi-constrained task points.

核心思想:
  上层使用自组织映射 (SOM) 将 M 个任务目标分配给 N 个 USV,
  使每个 USV 负责空间相近的目标子集, 并通过最近邻启发式确定访问顺序。

算法流程:
  1. 以 USV 起始位置初始化 SOM 神经元权重
  2. 迭代训练: 将每个目标作为输入, 找到最佳匹配单元 (BMU),
     更新 BMU 权重向目标靠拢 (学习率和邻域半径按指数衰减)
  3. 训练结束后, 将每个目标分配给最近的 SOM 神经元 (即 USV)
  4. 对每个 USV 的目标子集用最近邻启发式排序, 生成访问序列
"""

import numpy as np
from typing import List, Tuple, Optional


class SOMScheduler:
    """
    SOM 全局目标调度器

    将 M 个目标点分配给 N 个 USV, 并为每个 USV 生成有序目标访问序列。
    """

    def __init__(self,
                 num_iterations: int = 200,
                 initial_lr: float = 0.5,
                 final_lr: float = 0.01,
                 initial_sigma: float = 30.0,
                 final_sigma: float = 0.5,
                 seed: Optional[int] = None):
        """
        Args:
            num_iterations: SOM 训练迭代次数
            initial_lr: 初始学习率
            final_lr: 最终学习率
            initial_sigma: 初始邻域半径
            final_sigma: 最终邻域半径
            seed: 随机种子
        """
        self.num_iterations = num_iterations
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.rng = np.random.RandomState(seed)

    def schedule(self,
                 usv_positions: np.ndarray,
                 target_positions: np.ndarray
                 ) -> List[List[int]]:
        """
        执行 SOM 调度, 为每个 USV 分配有序目标序列。

        Args:
            usv_positions: shape=(N, 2), 各 USV 起始位置
            target_positions: shape=(M, 2), 各目标位置

        Returns:
            sequences: 长度为 N 的列表, 每个元素是该 USV 的目标索引序列
        """
        num_usvs = len(usv_positions)
        num_targets = len(target_positions)

        if num_targets == 0:
            return [[] for _ in range(num_usvs)]

        # --- 1. SOM 训练: 神经元权重初始化为 USV 位置 ---
        weights = usv_positions.astype(np.float64).copy()  # (N, 2)

        for t in range(self.num_iterations):
            # 指数衰减学习率和邻域半径
            progress = t / max(self.num_iterations - 1, 1)
            lr = self.initial_lr * (self.final_lr / self.initial_lr) ** progress
            sigma = self.initial_sigma * (self.final_sigma / self.initial_sigma) ** progress

            # 随机选取一个目标作为输入
            idx = self.rng.randint(num_targets)
            target = target_positions[idx]

            # 找到 BMU (Best Matching Unit)
            dists = np.linalg.norm(weights - target, axis=1)
            bmu = int(np.argmin(dists))

            # 更新权重 (高斯邻域函数, 基于权重空间距离)
            for i in range(num_usvs):
                spatial_dist = np.linalg.norm(weights[i] - weights[bmu])
                h = np.exp(-(spatial_dist ** 2) / (2 * sigma ** 2))
                weights[i] += lr * h * (target - weights[i])

        # --- 2. 分配: 每个目标归属最近的神经元 ---
        assignments = [[] for _ in range(num_usvs)]
        for j in range(num_targets):
            dists = np.linalg.norm(weights - target_positions[j], axis=1)
            best_usv = int(np.argmin(dists))
            assignments[best_usv].append(j)

        # --- 3. 排序: 最近邻启发式 (从 USV 位置出发) ---
        sequences = []
        for i in range(num_usvs):
            if len(assignments[i]) == 0:
                sequences.append([])
                continue
            ordered = self._nearest_neighbor_sort(
                usv_positions[i], target_positions, assignments[i]
            )
            sequences.append(ordered)

        return sequences

    @staticmethod
    def _nearest_neighbor_sort(start_pos: np.ndarray,
                               target_positions: np.ndarray,
                               target_indices: List[int]) -> List[int]:
        """
        最近邻启发式排序: 从 start_pos 出发, 每次选最近的未访问目标。
        """
        remaining = list(target_indices)
        ordered = []
        current = start_pos.copy()

        while remaining:
            dists = [np.linalg.norm(current - target_positions[j]) for j in remaining]
            nearest_idx = int(np.argmin(dists))
            chosen = remaining.pop(nearest_idx)
            ordered.append(chosen)
            current = target_positions[chosen].copy()

        return ordered

    def schedule_with_balance(self,
                              usv_positions: np.ndarray,
                              target_positions: np.ndarray,
                              max_imbalance: int = 2
                              ) -> List[List[int]]:
        """
        带负载均衡的 SOM 调度。

        在基础 SOM 分配后, 如果某个 USV 的目标数量超过平均值 + max_imbalance,
        则将多余目标转移给最空闲且最近的 USV。

        Args:
            usv_positions: shape=(N, 2)
            target_positions: shape=(M, 2)
            max_imbalance: 允许的最大偏差

        Returns:
            sequences: 均衡后的目标序列
        """
        sequences = self.schedule(usv_positions, target_positions)
        num_usvs = len(usv_positions)
        num_targets = len(target_positions)

        if num_targets == 0 or num_usvs == 0:
            return sequences

        avg_load = num_targets / num_usvs
        upper_bound = int(np.ceil(avg_load)) + max_imbalance

        # 迭代再平衡
        changed = True
        while changed:
            changed = False
            for i in range(num_usvs):
                while len(sequences[i]) > upper_bound:
                    # 找距离该 USV 最远的目标
                    farthest_idx = -1
                    farthest_dist = -1.0
                    for seq_pos, t_idx in enumerate(sequences[i]):
                        d = np.linalg.norm(usv_positions[i] - target_positions[t_idx])
                        if d > farthest_dist:
                            farthest_dist = d
                            farthest_idx = seq_pos

                    t_to_move = sequences[i].pop(farthest_idx)

                    # 找最空闲且最近的其他 USV
                    best_j = -1
                    best_score = float('inf')
                    for j in range(num_usvs):
                        if j == i:
                            continue
                        if len(sequences[j]) >= upper_bound:
                            continue
                        d = np.linalg.norm(usv_positions[j] - target_positions[t_to_move])
                        score = len(sequences[j]) * 100.0 + d
                        if score < best_score:
                            best_score = score
                            best_j = j

                    if best_j >= 0:
                        sequences[best_j].append(t_to_move)
                    else:
                        # 无法转移, 放回
                        sequences[i].insert(farthest_idx, t_to_move)
                        break

                    changed = True

        # 对转移后的序列重新排序
        for i in range(num_usvs):
            if len(sequences[i]) > 0:
                sequences[i] = self._nearest_neighbor_sort(
                    usv_positions[i], target_positions, sequences[i]
                )

        return sequences
