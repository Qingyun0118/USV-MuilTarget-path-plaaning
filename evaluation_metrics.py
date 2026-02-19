"""
USV 多目标路径规划 — 综合评估指标模块

参考文献:
  Ma et al., "Multi-objective path planning for unmanned surface vehicle
  with currents effects", ISA Transactions, 2018.

包含六项评估指标:
  1. 路径长度指标 (Shortest Path Metric, SPM)
  2. 路径平滑度指标 (Smoothest Path Metric, SmPM)
  3. 经济成本与能效指标 (Minimum Economic Cost Metric, MECM)
  4. 路径安全度指标 (Safest Path Metric, SaPM)
  5. 样本效率 (Sample Efficiency, SE)
  6. 任务成功率 (Success Rate, SR)
"""

import numpy as np
import csv
import os
from typing import List, Dict, Optional, Tuple


# ============================================================================
# 1. 路径长度指标 (Shortest Path Metric)
# ============================================================================

def compute_path_length(path_points: np.ndarray) -> float:
    """
    计算路径总长度 (所有相邻路径点的欧氏距离之和)

    L = Σ_{i=0}^{N-2} ||P_{i+1} - P_i||_2

    Args:
        path_points: 路径点坐标, shape=(N, 2), 每行为 (x, y)

    Returns:
        路径总长度 (单位与坐标相同)
    """
    if path_points.shape[0] < 2:
        return 0.0
    diffs = np.diff(path_points, axis=0)  # (N-1, 2)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(segment_lengths))


def compute_path_length_ratio(path_points: np.ndarray,
                              target_positions: List[np.ndarray],
                              visit_order: List[int]) -> float:
    """
    计算路径长度比 (SPM): 实际路径长度 / 理想最短直线距离

    SPM = L_actual / L_ideal
    其中 L_ideal 为按访问顺序连接起点和各目标点的直线距离之和

    值越接近 1.0 说明路径越短 (越优)。

    Args:
        path_points:      路径点坐标, shape=(N, 2)
        target_positions: 各目标点全局坐标列表
        visit_order:      目标点的访问顺序 (索引列表)

    Returns:
        路径长度比 SPM (>= 1.0)
    """
    actual_length = compute_path_length(path_points)

    if len(visit_order) == 0 or actual_length < 1e-9:
        return float('inf')

    # 理想直线距离 (起点 -> 第一个目标 -> 第二个目标 -> ...)
    waypoints = [path_points[0]]
    for idx in visit_order:
        if idx < len(target_positions):
            waypoints.append(np.array(target_positions[idx]))
    waypoints = np.array(waypoints)

    ideal_length = compute_path_length(waypoints)
    if ideal_length < 1e-9:
        return float('inf')

    return actual_length / ideal_length


# ============================================================================
# 2. 路径平滑度指标 (Smoothest Path Metric)
# ============================================================================

def compute_smoothness(path_points: np.ndarray,
                       headings: Optional[np.ndarray] = None) -> float:
    """
    计算路径平滑度指标 (SmPM): 相邻航段转向角的绝对值之和

    SmPM = Σ_{i=1}^{N-2} |Δθ_i|

    其中 Δθ_i 为第 i 段与第 i-1 段之间的航向差, 归一化到 [-π, π]

    值越小表示路径越平滑。

    Args:
        path_points: 路径点坐标, shape=(N, 2)
        headings:    对应航向角 (弧度), shape=(N,), 可选
                     若提供则直接使用; 否则由相邻点差分计算

    Returns:
        转向角绝对值之和 (弧度)
    """
    N = path_points.shape[0]
    if N < 3:
        return 0.0

    if headings is not None and len(headings) >= N:
        # 使用提供的航向角
        angles = headings
    else:
        # 由路径点差分计算
        diffs = np.diff(path_points, axis=0)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])  # (N-1,)
        # 补齐使长度 = N-1; 取相邻差
        delta_angles = np.diff(angles)
    
    if headings is not None and len(headings) >= N:
        delta_angles = np.diff(angles[:N])

    # 归一化到 [-π, π]
    delta_angles = (delta_angles + np.pi) % (2 * np.pi) - np.pi

    return float(np.sum(np.abs(delta_angles)))


def compute_smoothness_normalized(path_points: np.ndarray,
                                  headings: Optional[np.ndarray] = None) -> float:
    """
    归一化平滑度指标: 平均每步转向角

    SmPM_norm = SmPM / (N - 2)

    值越小越平滑, 0 表示完全直线。

    Args:
        path_points: shape=(N, 2)
        headings:    shape=(N,), 可选

    Returns:
        平均转向角 (弧度/步)
    """
    N = path_points.shape[0]
    if N < 3:
        return 0.0
    total = compute_smoothness(path_points, headings)
    return total / (N - 2)


# ============================================================================
# 3. 经济成本与能效指标 (Minimum Economic Cost Metric)
# ============================================================================

def compute_energy_cost(path_points: np.ndarray,
                        headings: np.ndarray,
                        flow_velocities: Optional[np.ndarray] = None,
                        dt: float = 0.05,
                        usv_mass: float = 20.0,
                        k_drag: float = 2.0) -> Dict[str, float]:
    """
    计算经济成本与能效指标 (MECM)

    参考 Ma et al. Eq.(13):
        E = Σ_{i=0}^{N-2} F_i · d_i
    其中 F_i 为第 i 段所需推力, d_i 为位移距离。

    简化估算:
        F_i = m * a_i + k_drag * v_i   (惯性力 + 阻力)
        E_total = Σ F_i · Δd_i         (推力做功)

    额外输出:
        - 海流助力/阻力分解
        - 平均单位距离能耗

    Args:
        path_points:     shape=(N, 2)
        headings:        shape=(N,), 各点航向 (弧度)
        flow_velocities: shape=(N, 2), 各点海流速度 (u, v); 若为 None 忽略海流
        dt:              时间步长 (秒)
        usv_mass:        USV 质量 (kg)
        k_drag:          阻力系数

    Returns:
        字典包含:
          - 'total_energy':         总能耗 (J)
          - 'energy_per_meter':     单位距离能耗 (J/m)
          - 'avg_power':            平均功率 (W)
          - 'current_assist_ratio': 海流助力比 (0~1, 越大说明海流越有利)
    """
    N = path_points.shape[0]
    if N < 2:
        return {
            'total_energy': 0.0,
            'energy_per_meter': 0.0,
            'avg_power': 0.0,
            'current_assist_ratio': 0.0,
        }

    diffs = np.diff(path_points, axis=0)  # (N-1, 2)
    segment_lengths = np.linalg.norm(diffs, axis=1)  # (N-1,)
    velocities = diffs / dt  # (N-1, 2) 实际速度向量

    total_energy = 0.0
    current_assist_work = 0.0
    total_work_magnitude = 0.0

    for i in range(N - 1):
        v_actual = velocities[i]
        v_mag = np.linalg.norm(v_actual)

        # 海流速度
        if flow_velocities is not None and i < len(flow_velocities):
            v_current = flow_velocities[i]
        else:
            v_current = np.array([0.0, 0.0])

        # USV 相对水的速度 (推进速度)
        v_propulsion = v_actual - v_current
        v_prop_mag = np.linalg.norm(v_propulsion)

        # 阻力 = k_drag * v_propulsion_magnitude
        F_drag = k_drag * v_prop_mag

        # 加速度估算 (有限差分)
        if i < N - 2:
            v_next = velocities[i + 1]
            a = np.linalg.norm(v_next - v_actual) / dt
        else:
            a = 0.0

        # 推力 = 惯性力 + 阻力
        F_thrust = usv_mass * a + F_drag

        # 该段做功 = 推力 × 位移
        work = F_thrust * segment_lengths[i]
        total_energy += work
        total_work_magnitude += abs(work)

        # 海流助力: 海流速度在航行方向上的投影
        if v_mag > 1e-6:
            direction = v_actual / v_mag
            current_proj = np.dot(v_current, direction)
            if current_proj > 0:
                current_assist_work += current_proj * segment_lengths[i]

    path_length = float(np.sum(segment_lengths))
    total_time = (N - 1) * dt

    energy_per_meter = total_energy / max(path_length, 1e-9)
    avg_power = total_energy / max(total_time, 1e-9)
    current_ratio = current_assist_work / max(total_work_magnitude, 1e-9)

    return {
        'total_energy': float(total_energy),
        'energy_per_meter': float(energy_per_meter),
        'avg_power': float(avg_power),
        'current_assist_ratio': float(np.clip(current_ratio, 0.0, 1.0)),
    }


# ============================================================================
# 4. 路径安全度指标 (Safest Path Metric)
# ============================================================================

def compute_safety_metric(path_points: np.ndarray,
                          obstacles: list,
                          boundary: Tuple[float, float, float, float] = (0, 0, 100, 100)
                          ) -> Dict[str, float]:
    """
    计算路径安全度指标 (SaPM)

    参考 Ma et al. Eq.(10-11):
        SaPM = min_{i} d_obs(P_i)

    其中 d_obs(P_i) 为路径点 P_i 到最近障碍物的距离。

    额外输出:
        - 平均安全距离
        - 进入危险区域的次数
        - 碰撞次数
        - 最近边界距离

    Args:
        path_points: shape=(N, 2)
        obstacles:   Obstacle 对象列表 (具有 geometry, danger_zone 属性)
        boundary:    环境边界 (x_min, y_min, x_max, y_max)

    Returns:
        字典包含:
          - 'min_obstacle_distance':  路径到障碍物的最小距离
          - 'avg_obstacle_distance':  路径到障碍物的平均最小距离
          - 'danger_zone_entries':    进入危险区域的次数
          - 'collision_count':        碰撞次数 (进入障碍物几何体)
          - 'min_boundary_distance':  到边界的最小距离
          - 'safety_score':           综合安全评分 (0-1, 越大越安全)
    """
    from shapely.geometry import Point as ShapelyPoint

    N = path_points.shape[0]
    if N == 0 or len(obstacles) == 0:
        return {
            'min_obstacle_distance': float('inf'),
            'avg_obstacle_distance': float('inf'),
            'danger_zone_entries': 0,
            'collision_count': 0,
            'min_boundary_distance': 0.0,
            'safety_score': 1.0,
        }

    x_min, y_min, x_max, y_max = boundary
    min_obs_dists = []
    danger_count = 0
    collision_count = 0
    min_boundary_dist = float('inf')

    for i in range(N):
        pt = ShapelyPoint(path_points[i, 0], path_points[i, 1])

        # 到各障碍物的最小距离
        min_d = float('inf')
        in_danger = False
        in_collision = False
        for obs in obstacles:
            d = obs.geometry.distance(pt)
            if d < min_d:
                min_d = d
            if obs.danger_zone.contains(pt):
                in_danger = True
            if obs.geometry.contains(pt):
                in_collision = True

        min_obs_dists.append(min_d)
        if in_danger:
            danger_count += 1
        if in_collision:
            collision_count += 1

        # 到边界距离
        bd = min(
            path_points[i, 0] - x_min,
            x_max - path_points[i, 0],
            path_points[i, 1] - y_min,
            y_max - path_points[i, 1]
        )
        if bd < min_boundary_dist:
            min_boundary_dist = bd

    min_obs_dists = np.array(min_obs_dists)
    min_obstacle_distance = float(np.min(min_obs_dists))
    avg_obstacle_distance = float(np.mean(min_obs_dists))

    # 综合安全评分: 基于最小距离和危险区域比例
    # 安全距离参考值 (rou_c + rou_s, 通常 ≈ 3~5)
    safe_ref = 5.0
    dist_score = np.clip(min_obstacle_distance / safe_ref, 0.0, 1.0)
    danger_ratio = danger_count / max(N, 1)
    danger_score = 1.0 - np.clip(danger_ratio * 5.0, 0.0, 1.0)  # 放大惩罚
    collision_score = 1.0 if collision_count == 0 else 0.0
    safety_score = 0.4 * dist_score + 0.3 * danger_score + 0.3 * collision_score

    return {
        'min_obstacle_distance': min_obstacle_distance,
        'avg_obstacle_distance': avg_obstacle_distance,
        'danger_zone_entries': danger_count,
        'collision_count': collision_count,
        'min_boundary_distance': float(min_boundary_dist),
        'safety_score': float(np.clip(safety_score, 0.0, 1.0)),
    }


# ============================================================================
# 5. 样本效率 (Sample Efficiency)
# ============================================================================

def compute_sample_efficiency(episode_steps: List[int],
                              episode_successes: List[bool],
                              target_success_rate: float = 0.95,
                              window_size: int = 100) -> Dict[str, float]:
    """
    计算样本效率 (SE): 达到预设成功率所需的环境交互步数

    N_steps = Σ_{e=1}^{E_conv} T_e

    其中 E_conv 为首次达到目标成功率时的回合数, T_e 为第 e 回合的步数。

    Args:
        episode_steps:       每个 episode 的步数列表
        episode_successes:   每个 episode 是否成功的列表
        target_success_rate: 目标成功率 (默认 0.95)
        window_size:         滑动窗口大小 (计算成功率)

    Returns:
        字典包含:
          - 'convergence_episode':   收敛时的回合数 (E_conv), 未收敛则为 -1
          - 'total_steps_to_converge': 收敛所需的总交互步数 (N_steps)
          - 'final_success_rate':     最终滑动窗口成功率
          - 'converged':             是否达到收敛
    """
    E = len(episode_steps)
    if E == 0:
        return {
            'convergence_episode': -1,
            'total_steps_to_converge': 0,
            'final_success_rate': 0.0,
            'converged': False,
        }

    convergence_episode = -1
    cumulative_steps = 0

    for e in range(E):
        cumulative_steps += episode_steps[e]

        # 滑动窗口成功率
        start = max(0, e - window_size + 1)
        window_successes = episode_successes[start:e + 1]
        sr = sum(window_successes) / len(window_successes)

        if sr >= target_success_rate and convergence_episode < 0:
            convergence_episode = e + 1  # 1-indexed
            break

    if convergence_episode < 0:
        # 未收敛
        total_steps = sum(episode_steps)
        final_window = episode_successes[max(0, E - window_size):]
        final_sr = sum(final_window) / max(len(final_window), 1)
    else:
        total_steps = cumulative_steps
        final_sr = target_success_rate

    return {
        'convergence_episode': convergence_episode,
        'total_steps_to_converge': total_steps,
        'final_success_rate': float(final_sr),
        'converged': convergence_episode > 0,
    }


# ============================================================================
# 6. 任务成功率 (Success Rate)
# ============================================================================

def compute_success_rate(eval_results: List[Dict]) -> Dict[str, float]:
    """
    计算任务成功率 (SR):

    SR = N_success / N_total × 100%

    成功定义: USV 成功遍历所有目标点且无碰撞。

    Args:
        eval_results: 评估结果列表, 每项需包含 'success' 和 'collision' 键

    Returns:
        字典包含:
          - 'success_rate':     成功率 (0~1)
          - 'success_count':    成功次数
          - 'collision_rate':   碰撞率 (0~1)
          - 'total_episodes':   总测试回合数
    """
    N = len(eval_results)
    if N == 0:
        return {
            'success_rate': 0.0,
            'success_count': 0,
            'collision_rate': 0.0,
            'total_episodes': 0,
        }

    success_count = sum(1 for r in eval_results if r.get('success', False))
    collision_count = sum(1 for r in eval_results if r.get('collision', False))

    return {
        'success_rate': success_count / N,
        'success_count': success_count,
        'collision_rate': collision_count / N,
        'total_episodes': N,
    }


# ============================================================================
# 综合评估函数
# ============================================================================

def load_path_from_csv(csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 CSV 文件加载路径数据

    Args:
        csv_file: CSV 文件路径 (列: x, y, heading)

    Returns:
        path_points: shape=(N, 2)
        headings:    shape=(N,)
    """
    xs, ys, hs = [], [], []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    xs.append(float(row[0]))
                    ys.append(float(row[1]))
                    hs.append(float(row[2]) if len(row) >= 3 else 0.0)
                except ValueError:
                    continue
    path_points = np.column_stack([xs, ys]) if len(xs) > 0 else np.zeros((0, 2))
    headings = np.array(hs)
    return path_points, headings


def evaluate_single_episode(csv_file: str,
                            env,
                            result_dict: Dict,
                            flow_fn=None,
                            dt: float = 0.05) -> Dict[str, float]:
    """
    对单个 episode 执行全部 6 项评估指标计算

    Args:
        csv_file:    路径 CSV 文件
        env:         MyUsvEnv 实例 (用于获取障碍物、目标等信息)
        result_dict: evaluate_mpc_hrl 函数返回的单 episode 结果字典,
                     需包含: 'success', 'collision', 'total_distance',
                             'target_order', 'total_steps'
        flow_fn:     海流速度函数 flow_fn(x, y) -> (u, v), 可选
        dt:          时间步长

    Returns:
        包含所有指标的字典
    """
    metrics = {}

    # 加载路径
    if os.path.exists(csv_file):
        path_points, headings = load_path_from_csv(csv_file)
    else:
        path_points = np.zeros((0, 2))
        headings = np.array([])

    # --- 1. 路径长度指标 ---
    metrics['path_length'] = compute_path_length(path_points)

    target_positions = [np.array([t.x, t.y]) for t in env.targets]
    visit_order = result_dict.get('target_order', [])
    metrics['path_length_ratio'] = compute_path_length_ratio(
        path_points, target_positions, visit_order)

    # --- 2. 路径平滑度指标 ---
    metrics['smoothness_total'] = compute_smoothness(path_points, headings)
    metrics['smoothness_avg'] = compute_smoothness_normalized(path_points, headings)

    # --- 3. 经济成本与能效指标 ---
    # 获取海流数据
    flow_velocities = None
    if flow_fn is not None and path_points.shape[0] > 0:
        fvs = []
        for i in range(path_points.shape[0]):
            fu, fv = flow_fn(path_points[i, 0], path_points[i, 1])
            fvs.append([fu, fv])
        flow_velocities = np.array(fvs)

    energy_metrics = compute_energy_cost(
        path_points, headings, flow_velocities, dt=dt,
        usv_mass=getattr(env.usv, 'mass', 20.0),
        k_drag=2.0)
    metrics.update({f'energy_{k}': v for k, v in energy_metrics.items()})

    # --- 4. 路径安全度指标 ---
    safety_metrics = compute_safety_metric(path_points, env.obstacles)
    metrics.update({f'safety_{k}': v for k, v in safety_metrics.items()})

    # --- 5 & 6. 成功与碰撞 (从 result_dict 提取) ---
    metrics['success'] = result_dict.get('success', False)
    metrics['collision'] = result_dict.get('collision', False)
    metrics['targets_reached'] = result_dict.get('targets_reached', 0)
    metrics['total_steps'] = result_dict.get('total_steps', 0)
    metrics['total_distance'] = result_dict.get('total_distance', 0.0)

    return metrics


def evaluate_all_episodes(eval_results: List[Dict],
                          csv_dir: str,
                          env,
                          eval_mode: str = 'rl',
                          flow_fn=None,
                          dt: float = 0.05,
                          episode_steps_train: Optional[List[int]] = None,
                          episode_successes_train: Optional[List[bool]] = None,
                          target_success_rate: float = 0.95) -> Dict:
    """
    对所有评估 episode 执行综合评估

    Args:
        eval_results:          评估结果列表
        csv_dir:               CSV 文件目录
        env:                   MyUsvEnv 实例
        eval_mode:             评估模式 ('rl' 或 'mpc')
        flow_fn:               海流速度函数
        dt:                    时间步长
        episode_steps_train:   训练过程中每 episode 步数 (用于样本效率)
        episode_successes_train: 训练过程中每 episode 成功标志
        target_success_rate:   目标成功率

    Returns:
        综合报告字典
    """
    all_episode_metrics = []

    for ep_result in eval_results:
        ep_idx = ep_result['episode']
        csv_file = os.path.join(csv_dir, f"mpc_hrl_path_{eval_mode}_ep_{ep_idx}.csv")

        ep_metrics = evaluate_single_episode(
            csv_file, env, ep_result, flow_fn=flow_fn, dt=dt)
        ep_metrics['episode'] = ep_idx
        all_episode_metrics.append(ep_metrics)

    # --- 汇总统计 ---
    report = {}

    # 1. 路径长度
    lengths = [m['path_length'] for m in all_episode_metrics if m['success']]
    ratios = [m['path_length_ratio'] for m in all_episode_metrics
              if m['success'] and np.isfinite(m['path_length_ratio'])]
    report['path_length'] = {
        'mean': float(np.mean(lengths)) if lengths else 0.0,
        'std': float(np.std(lengths)) if lengths else 0.0,
        'min': float(np.min(lengths)) if lengths else 0.0,
        'ratio_mean': float(np.mean(ratios)) if ratios else float('inf'),
    }

    # 2. 路径平滑度
    smooth_vals = [m['smoothness_avg'] for m in all_episode_metrics if m['success']]
    report['smoothness'] = {
        'mean': float(np.mean(smooth_vals)) if smooth_vals else 0.0,
        'std': float(np.std(smooth_vals)) if smooth_vals else 0.0,
    }

    # 3. 经济成本
    energies = [m['energy_total_energy'] for m in all_episode_metrics if m['success']]
    epm = [m['energy_energy_per_meter'] for m in all_episode_metrics if m['success']]
    report['energy'] = {
        'total_mean': float(np.mean(energies)) if energies else 0.0,
        'per_meter_mean': float(np.mean(epm)) if epm else 0.0,
    }

    # 4. 路径安全度
    safety_scores = [m['safety_safety_score'] for m in all_episode_metrics]
    min_dists = [m['safety_min_obstacle_distance'] for m in all_episode_metrics]
    report['safety'] = {
        'score_mean': float(np.mean(safety_scores)) if safety_scores else 0.0,
        'min_dist_mean': float(np.mean(min_dists)) if min_dists else 0.0,
        'min_dist_min': float(np.min(min_dists)) if min_dists else 0.0,
    }

    # 5. 样本效率
    if episode_steps_train is not None and episode_successes_train is not None:
        se = compute_sample_efficiency(
            episode_steps_train, episode_successes_train,
            target_success_rate=target_success_rate)
        report['sample_efficiency'] = se
    else:
        report['sample_efficiency'] = {
            'convergence_episode': -1,
            'total_steps_to_converge': 0,
            'final_success_rate': 0.0,
            'converged': False,
        }

    # 6. 任务成功率
    sr = compute_success_rate(eval_results)
    report['success_rate'] = sr

    # 原始数据
    report['episode_metrics'] = all_episode_metrics

    return report


def print_evaluation_report(report: Dict, algorithm_name: str = "Algorithm"):
    """
    格式化打印评估报告

    Args:
        report:         evaluate_all_episodes 返回的报告
        algorithm_name: 算法名称
    """
    print(f"\n{'=' * 70}")
    print(f"  综合评估报告 — {algorithm_name}")
    print(f"{'=' * 70}")

    sr = report['success_rate']
    print(f"\n  [1] 任务成功率 (SR)")
    print(f"      成功率: {sr['success_rate'] * 100:.1f}%  "
          f"({sr['success_count']}/{sr['total_episodes']})")
    print(f"      碰撞率: {sr['collision_rate'] * 100:.1f}%")

    pl = report['path_length']
    print(f"\n  [2] 路径长度指标 (SPM)")
    print(f"      平均路径长度: {pl['mean']:.2f} ± {pl['std']:.2f}")
    print(f"      最短路径长度: {pl['min']:.2f}")
    print(f"      路径长度比: {pl['ratio_mean']:.3f} (1.0 = 理想)")

    sm = report['smoothness']
    print(f"\n  [3] 路径平滑度指标 (SmPM)")
    print(f"      平均转向角: {np.degrees(sm['mean']):.2f}° ± {np.degrees(sm['std']):.2f}° /步")

    en = report['energy']
    print(f"\n  [4] 经济成本与能效指标 (MECM)")
    print(f"      平均总能耗: {en['total_mean']:.1f} J")
    print(f"      单位距离能耗: {en['per_meter_mean']:.2f} J/m")

    sa = report['safety']
    print(f"\n  [5] 路径安全度指标 (SaPM)")
    print(f"      安全评分: {sa['score_mean']:.3f} (0~1)")
    print(f"      平均最近障碍距离: {sa['min_dist_mean']:.2f}")
    print(f"      全局最近障碍距离: {sa['min_dist_min']:.2f}")

    se = report['sample_efficiency']
    print(f"\n  [6] 样本效率 (SE)")
    if se['converged']:
        print(f"      收敛回合: {se['convergence_episode']}")
        print(f"      总交互步数: {se['total_steps_to_converge']:,}")
    else:
        print(f"      未收敛 (最终成功率: {se['final_success_rate'] * 100:.1f}%)")

    print(f"\n{'=' * 70}")


def get_normalized_metrics_for_radar(report: Dict) -> Dict[str, float]:
    """
    从评估报告中提取归一化的六维指标, 用于雷达图绘制

    所有指标归一化到 [0, 1], 值越大越优。

    Returns:
        字典, 键为指标中文名, 值为 0~1 评分
    """
    metrics = {}

    # 1. 路径长度: 路径长度比的倒数 (比值越接近1越好)
    ratio = report['path_length'].get('ratio_mean', float('inf'))
    if np.isfinite(ratio) and ratio > 0:
        metrics['路径长度'] = float(np.clip(1.0 / ratio, 0.0, 1.0))
    else:
        metrics['路径长度'] = 0.0

    # 2. 路径平滑度: 平均转向角越小越好
    # 用指数衰减映射, 参考值 10° ≈ 0.175 rad
    avg_turn = report['smoothness'].get('mean', np.pi)
    metrics['路径平滑度'] = float(np.clip(np.exp(-avg_turn / 0.175), 0.0, 1.0))

    # 3. 经济成本: 单位距离能耗越低越好
    # 用指数衰减, 参考值 100 J/m
    epm = report['energy'].get('per_meter_mean', 1000.0)
    metrics['能效'] = float(np.clip(np.exp(-epm / 100.0), 0.0, 1.0))

    # 4. 路径安全度: 直接使用安全评分
    metrics['路径安全度'] = float(report['safety'].get('score_mean', 0.0))

    # 5. 样本效率: 收敛步数越少越好 (对数映射)
    se = report['sample_efficiency']
    if se['converged']:
        steps = se['total_steps_to_converge']
        # 参考: 10万步=优秀(1.0), 100万步=一般(0.5), 未收敛=0
        metrics['样本效率'] = float(np.clip(1.0 - np.log10(max(steps, 1)) / 7.0, 0.0, 1.0))
    else:
        metrics['样本效率'] = 0.0

    # 6. 任务成功率: 直接使用
    metrics['任务成功率'] = float(report['success_rate'].get('success_rate', 0.0))

    return metrics
