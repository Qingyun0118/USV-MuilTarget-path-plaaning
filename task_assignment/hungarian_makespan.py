"""
MakespanHungarianAssigner — 最小化 Makespan 的匈牙利分配器 (Proposed 1)

初始分配使用贪心 makespan 分区 + 最近邻排序,
重分配使用 makespan 代价矩阵 + linear_sum_assignment。
"""

from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from task_assignment.base import BaseAssigner
from hrl_helpers import get_usv_pos, get_target_pos


class MakespanHungarianAssigner(BaseAssigner):
    """
    最小化 Makespan 的目标分配器。

    初始分配:
      1. 计算 USV→目标距离矩阵
      2. 贪心 makespan 分区: 逐个分配目标, 选择使全局 makespan 增量最小的 USV
      3. 对每个 USV 的队列做最近邻排序优化内部访问顺序
      4. 返回 queue[i][0] 作为各 USV 的初始目标

    重分配:
      - USV 到达目标 → 推进队列指针
      - USV 完成队列 → 从负载最重的 USV 偷取目标
      - 用 makespan 代价矩阵 + linear_sum_assignment 重新分配
    """

    name = "hungarian_makespan"
    is_learnable = False

    def __init__(self, usv_speed_estimate: float = 2.0):
        self.usv_speed = usv_speed_estimate
        self.queues: List[List[int]] = []   # per-USV 完整目标序列
        self.queue_ptr: List[int] = []       # per-USV 当前队列位置

    # ==========================================================================
    # 初始分配
    # ==========================================================================

    def assign_initial(
        self,
        obs_dict: dict,
        num_usvs: int,
        valid_mask: np.ndarray,
        scale: float,
        global_state: Optional[np.ndarray] = None,
        eval_mode: bool = False,
    ):
        valid_indices = list(np.where(valid_mask)[0])
        n_targets = len(valid_indices)

        if n_targets == 0:
            self.queues = [[] for _ in range(num_usvs)]
            self.queue_ptr = [0] * num_usvs
            return self._fallback(obs_dict, num_usvs, valid_mask, scale)

        # 计算距离矩阵
        dist = np.zeros((num_usvs, n_targets), dtype=np.float64)
        for i in range(num_usvs):
            up = get_usv_pos(obs_dict[f"usv_{i}"]) * scale
            for j, t_idx in enumerate(valid_indices):
                tp = get_target_pos(obs_dict[f"usv_{i}"], t_idx) * scale
                dist[i, j] = np.linalg.norm(up - tp)

        # 贪心 makespan 分区
        self.queues = self._greedy_makespan_partition(dist, valid_indices)
        self.queue_ptr = [0] * num_usvs

        return self._build_result(obs_dict, num_usvs, scale)

    def _greedy_makespan_partition(self, dist, valid_indices):
        """
        贪心 makespan 分区:
        - 将目标按"到最近USV距离"升序排序
        - 逐个分配: 选择使全局 makespan 增量最小的 USV
        - 对每个USV的队列做最近邻排序
        """
        num_usvs, n_targets = dist.shape
        queues = [[] for _ in range(num_usvs)]
        queue_times = [0.0] * num_usvs  # 每个USV队列的累计时间

        # 目标按"到最近USV的最小距离"排序
        min_dists = dist.min(axis=0)
        target_order = np.argsort(min_dists)

        for j in range(n_targets):
            t_idx_order = target_order[j]

            # 计算分配给每个USV后的 makespan
            best_usv = 0
            best_makespan = float('inf')
            for i in range(num_usvs):
                travel_time = dist[i, t_idx_order] / self.usv_speed
                candidate = max(queue_times[i] + travel_time,
                                max(queue_times) if queue_times else 0)
                if candidate < best_makespan:
                    best_makespan = candidate
                    best_usv = i

            queues[best_usv].append(valid_indices[t_idx_order])
            queue_times[best_usv] += dist[best_usv, t_idx_order] / self.usv_speed

        # 对每个USV的队列做最近邻排序
        for i in range(num_usvs):
            if len(queues[i]) > 1:
                queues[i] = self._nearest_neighbor_sort(
                    dist[i], queues[i], valid_indices)

        return queues

    def _nearest_neighbor_sort(self, dist_row, queue, valid_indices):
        """对队列做最近邻排序, 优化内部访问顺序。"""
        if len(queue) <= 1:
            return queue
        idx_to_col = {v: i for i, v in enumerate(valid_indices)}
        remaining = set(queue)
        ordered = [queue[0]]
        remaining.discard(queue[0])
        while remaining:
            last = ordered[-1]
            nearest = min(remaining,
                         key=lambda t: dist_row[idx_to_col[t]])
            ordered.append(nearest)
            remaining.discard(nearest)
        return ordered

    # ==========================================================================
    # 重分配
    # ==========================================================================

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
        num_usvs = len(current_target_idx)

        for i in usv_indices:
            if self.queue_ptr[i] < len(self.queues[i]):
                self.queue_ptr[i] += 1

        # 检查是否有USV完成队列, 需要从负载最重的偷取
        for i in usv_indices:
            if self.queue_ptr[i] >= len(self.queues[i]):
                self._steal_target(i)

        # 用 makespan 代价矩阵 + 匈牙利重新分配下一个目标
        self._hungarian_makespan_reassign(
            obs_dict, num_usvs, valid_mask, scale,
            current_target_idx, target_pos_norm, target_pos_abs)

    def _steal_target(self, usv_idx):
        """从估计剩余时间最长的USV队列末尾偷取目标。"""
        best_donor = -1
        best_time = -1
        for i, q in enumerate(self.queues):
            if i == usv_idx:
                continue
            remaining = len(q) - self.queue_ptr[i]
            if remaining > 1:
                # 估算剩余时间
                est_time = remaining * 20.0  # 粗略估计
                if est_time > best_time:
                    best_time = est_time
                    best_donor = i
        if best_donor >= 0:
            stolen = self.queues[best_donor].pop()
            self.queues[usv_idx].append(stolen)
            self.queue_ptr[usv_idx] = len(self.queues[usv_idx]) - 1

    def _hungarian_makespan_reassign(
        self, obs_dict, num_usvs, valid_mask, scale,
        current_target_idx, target_pos_norm, target_pos_abs
    ):
        """使用 makespan 代价矩阵 + 匈牙利算法重新分配。"""
        valid_indices = list(np.where(valid_mask)[0])
        n_valid = len(valid_indices)

        if n_valid == 0:
            return

        # 计算每个USV队列剩余时间
        remaining_times = np.zeros(num_usvs, dtype=np.float64)
        for i in range(num_usvs):
            ptr = self.queue_ptr[i]
            if ptr < len(self.queues[i]):
                # 估计: 当前目标剩余距离 + 队列中后续距离
                remaining = 0.0
                up = get_usv_pos(obs_dict[f"usv_{i}"]) * scale
                tp = get_target_pos(obs_dict[f"usv_{i}"],
                                    current_target_idx[i]) * scale
                remaining += np.linalg.norm(up - tp) / self.usv_speed
                # 粗略估计后续队列时间
                remaining += max(0, len(self.queues[i]) - ptr - 1) * 15.0
                remaining_times[i] = remaining

        # 构建 makespan 代价矩阵
        n_cols = max(num_usvs, n_valid)
        cost = np.full((num_usvs, n_cols), 1e6, dtype=np.float64)

        for i in range(num_usvs):
            up = get_usv_pos(obs_dict[f"usv_{i}"]) * scale
            for j, t_idx in enumerate(valid_indices):
                tp = get_target_pos(obs_dict[f"usv_{i}"], t_idx) * scale
                travel = np.linalg.norm(up - tp) / self.usv_speed
                time_with = remaining_times[i] + travel
                others_max = max(
                    remaining_times[k] for k in range(num_usvs) if k != i
                ) if num_usvs > 1 else 0.0
                cost[i, j] = max(time_with, others_max)

        row_ind, col_ind = linear_sum_assignment(cost)

        for r, c in zip(row_ind, col_ind):
            if c < n_valid:
                t_idx = valid_indices[c]
                current_target_idx[r] = t_idx
                target_pos_norm[r] = get_target_pos(obs_dict[f"usv_{r}"], t_idx)
                target_pos_abs[r] = target_pos_norm[r] * scale

    # ==========================================================================
    # 辅助方法
    # ==========================================================================

    def _build_result(self, obs_dict, num_usvs, scale):
        """从队列构建初始分配结果。"""
        target_idx = []
        target_norm = []
        target_abs = []

        for i in range(num_usvs):
            if self.queue_ptr[i] < len(self.queues[i]):
                t_idx = self.queues[i][self.queue_ptr[i]]
            else:
                t_idx = -1
            if t_idx < 0:
                # fallback
                target_idx.append(-1)
                target_norm.append(np.zeros(2, dtype=np.float32))
                target_abs.append(np.zeros(2, dtype=np.float32))
            else:
                t_norm = get_target_pos(obs_dict[f"usv_{i}"], t_idx)
                target_idx.append(t_idx)
                target_norm.append(t_norm)
                target_abs.append(t_norm * scale)

        # 修复 fallback
        for i in range(num_usvs):
            if target_idx[i] < 0:
                up = get_usv_pos(obs_dict[f"usv_{i}"]) * scale
                dists = []
                n_targets = len(obs_dict[f"usv_{i}"]["targets"]) // 3
                for j in range(n_targets):
                    tp = get_target_pos(obs_dict[f"usv_{i}"], j) * scale
                    dists.append(np.linalg.norm(up - tp))
                t_idx = int(np.argmin(dists))
                t_norm = get_target_pos(obs_dict[f"usv_{i}"], t_idx)
                target_idx[i] = t_idx
                target_norm[i] = t_norm
                target_abs[i] = t_norm * scale

        return target_idx, target_norm, target_abs, None

    def _fallback(self, obs_dict, num_usvs, valid_mask, scale):
        """无有效目标时的回退分配。"""
        target_idx = []
        target_norm = []
        target_abs = []
        for i in range(num_usvs):
            up = get_usv_pos(obs_dict[f"usv_{i}"]) * scale
            n_t = len(obs_dict[f"usv_{i}"]["targets"]) // 3
            dists = [np.linalg.norm(up - get_target_pos(obs_dict[f"usv_{i}"], j) * scale)
                     for j in range(n_t)]
            t_idx = int(np.argmin(dists))
            t_norm = get_target_pos(obs_dict[f"usv_{i}"], t_idx)
            target_idx.append(t_idx)
            target_norm.append(t_norm)
            target_abs.append(t_norm * scale)
        return target_idx, target_norm, target_abs, None
