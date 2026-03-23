"""
SOM-based task scheduling for multi-USV mission assignment.

This module implements a practical bi-layer decision component inspired by
Zhao et al. (2024): use SOM to organize target visiting order and then split
that global order to multiple USVs with short travel distance.
"""

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class SomScheduleResult:
    """Result container of SOM scheduling."""

    sequence_indices_per_usv: List[List[int]]
    sequence_points_per_usv: List[np.ndarray]
    global_tour_indices: List[int]
    estimated_total_distance: float


class RingSomPlanner:
    """Ring-SOM planner for 2D target ordering."""

    def __init__(self,
                 neuron_ratio: int = 8,
                 n_iterations: int = 1500,
                 init_lr: float = 0.6,
                 final_lr: float = 0.05,
                 init_sigma_ratio: float = 0.35,
                 final_sigma_ratio: float = 0.03,
                 seed: int = 42):
        self.neuron_ratio = max(3, int(neuron_ratio))
        self.n_iterations = max(200, int(n_iterations))
        self.init_lr = float(init_lr)
        self.final_lr = float(final_lr)
        self.init_sigma_ratio = float(init_sigma_ratio)
        self.final_sigma_ratio = float(final_sigma_ratio)
        self.seed = int(seed)

    @staticmethod
    def _normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        center = np.mean(points, axis=0)
        shifted = points - center
        scale = float(np.max(np.linalg.norm(shifted, axis=1)))
        scale = max(scale, 1e-6)
        return shifted / scale, center, scale

    @staticmethod
    def _denormalize_points(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
        return points * scale + center

    def _init_ring(self, n_neurons: int, rng: np.random.RandomState) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, n_neurons, endpoint=False)
        radius = 1.0
        ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        ring += rng.normal(0.0, 0.03, size=ring.shape)
        return ring.astype(np.float32)

    @staticmethod
    def _ring_distance_matrix(n_neurons: int) -> np.ndarray:
        idx = np.arange(n_neurons)
        dist = np.abs(idx[:, None] - idx[None, :])
        return np.minimum(dist, n_neurons - dist).astype(np.float32)

    def fit_tour(self, target_points: Sequence[Sequence[float]]) -> List[int]:
        points = np.asarray(target_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("target_points must be array-like with shape (N, 2)")

        n_targets = points.shape[0]
        if n_targets == 0:
            return []
        if n_targets == 1:
            return [0]

        p_norm, center, scale = self._normalize_points(points)
        rng = np.random.RandomState(self.seed)

        n_neurons = max(3 * n_targets, self.neuron_ratio * n_targets)
        neurons = self._init_ring(n_neurons, rng)
        ring_dist = self._ring_distance_matrix(n_neurons)

        sigma0 = max(1.0, self.init_sigma_ratio * n_neurons)
        sigma1 = max(0.5, self.final_sigma_ratio * n_neurons)

        for it in range(self.n_iterations):
            frac = it / max(1, self.n_iterations - 1)
            lr = self.init_lr + (self.final_lr - self.init_lr) * frac
            sigma = sigma0 + (sigma1 - sigma0) * frac

            sample = p_norm[rng.randint(0, n_targets)]
            d = np.linalg.norm(neurons - sample[None, :], axis=1)
            winner = int(np.argmin(d))

            h = np.exp(-0.5 * (ring_dist[winner] / max(1e-6, sigma)) ** 2)
            neurons += lr * h[:, None] * (sample[None, :] - neurons)

        # Map each target to its closest neuron index
        winner_ids = []
        for i in range(n_targets):
            di = np.linalg.norm(neurons - p_norm[i][None, :], axis=1)
            winner_ids.append((i, int(np.argmin(di))))

        winner_ids.sort(key=lambda x: x[1])
        ordered = [idx for idx, _ in winner_ids]

        # Remove possible duplicates by preserving first occurrence only
        seen = set()
        unique_ordered = []
        for idx in ordered:
            if idx not in seen:
                unique_ordered.append(idx)
                seen.add(idx)

        # If anything missing due to ties, append by nearest insertion
        if len(unique_ordered) < n_targets:
            missing = [i for i in range(n_targets) if i not in seen]
            for m in missing:
                if not unique_ordered:
                    unique_ordered.append(m)
                    continue
                best_pos = 0
                best_delta = float("inf")
                for pos in range(len(unique_ordered) + 1):
                    prev_idx = unique_ordered[pos - 1] if pos > 0 else unique_ordered[-1]
                    next_idx = unique_ordered[pos] if pos < len(unique_ordered) else unique_ordered[0]
                    prev_p = points[prev_idx]
                    next_p = points[next_idx]
                    m_p = points[m]
                    delta = (
                        np.linalg.norm(prev_p - m_p)
                        + np.linalg.norm(m_p - next_p)
                        - np.linalg.norm(prev_p - next_p)
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = pos
                unique_ordered.insert(best_pos, m)

        # Rotate to start from the point nearest centroid direction to stabilize output
        cen = np.mean(points, axis=0)
        d0 = [np.linalg.norm(points[idx] - cen) for idx in unique_ordered]
        start = int(np.argmin(d0))
        tour = unique_ordered[start:] + unique_ordered[:start]

        # Keep callsite deterministic with same inputs
        _ = self._denormalize_points(neurons, center, scale)
        return tour


def _path_length(start: np.ndarray, sequence: Sequence[np.ndarray]) -> float:
    if len(sequence) == 0:
        return 0.0
    total = float(np.linalg.norm(start - sequence[0]))
    for i in range(1, len(sequence)):
        total += float(np.linalg.norm(sequence[i] - sequence[i - 1]))
    return total


def _two_opt_with_fixed_start(start: np.ndarray, sequence: List[np.ndarray]) -> List[np.ndarray]:
    if len(sequence) <= 3:
        return sequence

    best = sequence[:]
    improved = True

    def seq_len(seq: List[np.ndarray]) -> float:
        return _path_length(start, seq)

    best_len = seq_len(best)

    while improved:
        improved = False
        n = len(best)
        for i in range(0, n - 1):
            for k in range(i + 1, n):
                if k - i == 1:
                    continue
                candidate = best[:i] + list(reversed(best[i:k])) + best[k:]
                cand_len = seq_len(candidate)
                if cand_len + 1e-8 < best_len:
                    best = candidate
                    best_len = cand_len
                    improved = True
        if not improved:
            break

    return best


def split_tour_for_two_usvs(global_tour_indices: Sequence[int],
                            target_points: Sequence[Sequence[float]],
                            usv_starts: Sequence[Sequence[float]]) -> Tuple[List[int], List[int], float]:
    """Split a global tour into 2 sequences minimizing total travel length."""
    if len(usv_starts) != 2:
        raise ValueError("usv_starts must contain exactly 2 start points")

    points = np.asarray(target_points, dtype=np.float32)
    s0 = np.asarray(usv_starts[0], dtype=np.float32)
    s1 = np.asarray(usv_starts[1], dtype=np.float32)

    tour = list(global_tour_indices)
    n = len(tour)
    if n == 0:
        return [], [], 0.0

    best_cost = float("inf")
    best_pair = ([], [])

    candidates = [tour, list(reversed(tour))]

    for ordered in candidates:
        # Consider all cyclic rotations
        for rot in range(n):
            rotated = ordered[rot:] + ordered[:rot]

            # split at k: usv0 gets [0:k], usv1 gets [k:n]
            for k in range(n + 1):
                a_idx = rotated[:k]
                b_idx = rotated[k:]

                a_pts = [points[i] for i in a_idx]
                b_pts = [points[i] for i in b_idx]

                cost_direct = _path_length(s0, a_pts) + _path_length(s1, b_pts)
                cost_swap = _path_length(s0, b_pts) + _path_length(s1, a_pts)

                if cost_direct <= cost_swap:
                    if cost_direct < best_cost:
                        best_cost = cost_direct
                        best_pair = (a_idx, b_idx)
                else:
                    if cost_swap < best_cost:
                        best_cost = cost_swap
                        best_pair = (b_idx, a_idx)

    return best_pair[0], best_pair[1], float(best_cost)


def schedule_targets_for_two_usvs_som(target_points: Sequence[Sequence[float]],
                                      usv_starts: Sequence[Sequence[float]],
                                      seed: int = 42,
                                      n_iterations: int = 1500,
                                      apply_two_opt: bool = True) -> SomScheduleResult:
    """
    SOM scheduling entry point.

    Returns target index sequence and coordinate sequence for each USV.
    """
    points = np.asarray(target_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("target_points must be array-like with shape (N, 2)")

    planner = RingSomPlanner(seed=seed, n_iterations=n_iterations)
    global_tour = planner.fit_tour(points)

    seq0_idx, seq1_idx, cost = split_tour_for_two_usvs(global_tour, points, usv_starts)

    seq0_pts = [points[i].copy() for i in seq0_idx]
    seq1_pts = [points[i].copy() for i in seq1_idx]

    if apply_two_opt:
        s0 = np.asarray(usv_starts[0], dtype=np.float32)
        s1 = np.asarray(usv_starts[1], dtype=np.float32)
        seq0_pts = _two_opt_with_fixed_start(s0, seq0_pts)
        seq1_pts = _two_opt_with_fixed_start(s1, seq1_pts)

        # Rebuild index sequence after 2-opt by nearest exact match (safe for generated points)
        seq0_idx = [int(np.where((points == p).all(axis=1))[0][0]) for p in seq0_pts]
        seq1_idx = [int(np.where((points == p).all(axis=1))[0][0]) for p in seq1_pts]
        cost = _path_length(np.asarray(usv_starts[0], dtype=np.float32), seq0_pts) + _path_length(
            np.asarray(usv_starts[1], dtype=np.float32), seq1_pts
        )

    return SomScheduleResult(
        sequence_indices_per_usv=[seq0_idx, seq1_idx],
        sequence_points_per_usv=[np.asarray(seq0_pts, dtype=np.float32), np.asarray(seq1_pts, dtype=np.float32)],
        global_tour_indices=global_tour,
        estimated_total_distance=float(cost),
    )


if __name__ == "__main__":
    demo_targets = np.array([
        [15.0, 20.0],
        [35.0, 75.0],
        [60.0, 40.0],
        [80.0, 70.0],
        [22.0, 55.0],
        [72.0, 18.0],
    ], dtype=np.float32)
    starts = np.array([[8.0, 8.0], [8.0, 92.0]], dtype=np.float32)

    result = schedule_targets_for_two_usvs_som(demo_targets, starts, seed=2026)
    print("Global tour:", result.global_tour_indices)
    print("USV0 sequence:", result.sequence_indices_per_usv[0])
    print("USV1 sequence:", result.sequence_indices_per_usv[1])
    print("Estimated total distance:", round(result.estimated_total_distance, 3))
