"""
Multi-USV Cooperative Navigation with SOM Task Allocation

架构:
  SOM Scheduler (全局目标分配)
    → 为每个 USV 生成有序目标序列
    → 每个 USV 使用训练好的 SAC Actuator 网络导航到各自目标
    → 所有 USV 完成各自序列 → 任务成功
    → 碰撞 (USV-障碍物 / USV-USV) 或超步数 → 任务终止

参考: Zhao et al. (2024) 双层决策方法
"""

import os
import numpy as np
import torch

from my_USV_Env import MultiUsvEnv, X_MAX, Y_MAX
from USV import USV
from som_scheduler import SOMScheduler
from sac_agent import SACActuatorAgent
from hrl_helpers import get_actuator_state
from hrl_config import device, SCALE


def load_actuator(model_dir: str,
                  state_dim: int = 26,
                  goal_dim: int = 2,
                  action_dim: int = 2) -> SACActuatorAgent:
    """
    加载训练好的 SAC Actuator 网络。

    Args:
        model_dir: 包含 best_actuator_actor.pth / best_actuator_critic.pth 的目录
        state_dim: 状态维度 (7 USV + 15 sensor + 4 current = 26)
        goal_dim: 目标维度 (2: relative x, y)
        action_dim: 动作维度 (2: throttle, rudder)
    """
    agent = SACActuatorAgent(state_dim, goal_dim, action_dim, max_action=1.0)

    actor_path = os.path.join(model_dir, "best_actuator_actor.pth")
    critic_path = os.path.join(model_dir, "best_actuator_critic.pth")

    if os.path.exists(actor_path):
        agent.actor.load_state_dict(
            torch.load(actor_path, map_location=device, weights_only=True))
        print(f"  Loaded actor from {actor_path}")
    else:
        raise FileNotFoundError(f"Actor model not found: {actor_path}")

    if os.path.exists(critic_path):
        agent.critic.load_state_dict(
            torch.load(critic_path, map_location=device, weights_only=True))
        print(f"  Loaded critic from {critic_path}")

    agent.actor.eval()
    agent.critic.eval()
    return agent


def get_single_usv_obs(multi_obs: dict, usv_idx: int) -> dict:
    """从 MultiUsvEnv 的观测中提取单个 USV 的观测字典。"""
    return multi_obs[f"usv_{usv_idx}"]


def run_som_multi_usv(
    num_usvs: int = 3,
    num_targets: int = 6,
    num_obstacles: int = 5,
    model_dir: str = "expert_hrl_sac_output",
    seed: int = 42,
    max_steps: int = 3000,
    per_usv_max_steps: int = 2000,
    som_iterations: int = 200,
    som_balance: bool = True,
    max_imbalance: int = 2,
    target_radius: float = 3.0,
    verbose: bool = True,
    fixed_start_positions=None,
    fixed_targets=None,
    fixed_obstacles=None,
):
    """
    使用 SOM 调度 + 训练好的 Actuator 执行多 USV 协同任务。

    Returns:
        result: dict 包含任务结果信息
    """
    # ---- 1. 初始化环境 ----
    env = MultiUsvEnv(
        num_usvs=num_usvs,
        num_targets=num_targets,
        num_obstacles=num_obstacles,
        seed=seed,
        per_usv_max_steps=per_usv_max_steps,
        current_type="generated",
        target_radius=target_radius,
        fixed_start_positions=fixed_start_positions,
        fixed_targets=fixed_targets,
        fixed_obstacles=fixed_obstacles,
    )
    obs = env.reset(seed=seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-USV SOM Navigation")
        print(f"  USVs: {num_usvs}, Targets: {num_targets}, Obstacles: {num_obstacles}")
        print(f"  Seed: {seed}, Max steps: {max_steps}")
        print(f"{'='*60}")

    # ---- 2. 加载 Actuator ----
    if verbose:
        print(f"\nLoading actuator from: {model_dir}")
    actuator = load_actuator(model_dir)

    # ---- 3. SOM 目标分配 ----
    usv_positions = np.array(env.start_positions, dtype=np.float64)
    target_positions = np.array([[t.x, t.y] for t in env.targets], dtype=np.float64)

    scheduler = SOMScheduler(num_iterations=som_iterations, seed=seed)
    if som_balance:
        sequences = scheduler.schedule_with_balance(
            usv_positions, target_positions, max_imbalance=max_imbalance)
    else:
        sequences = scheduler.schedule(usv_positions, target_positions)

    if verbose:
        print(f"\nSOM Target Assignment:")
        for i, seq in enumerate(sequences):
            seq_coords = [(f"T{j}({target_positions[j][0]:.1f},{target_positions[j][1]:.1f})")
                          for j in seq]
            print(f"  USV {i}: {' → '.join(seq_coords) if seq_coords else '(no targets)'}")

    # ---- 4. 初始化追踪状态 ----
    # 每个 USV 在自己序列中的当前索引
    seq_cursors = [0] * num_usvs
    # USV 是否还有未完成的目标
    usv_active = [len(sequences[i]) > 0 for i in range(num_usvs)]
    # 记录每个 USV 的路径、航向、传感器读数
    paths = [[] for _ in range(num_usvs)]
    headings = [[] for _ in range(num_usvs)]
    sensor_data = [[] for _ in range(num_usvs)]
    # 每个 USV 各自到达的目标
    usv_reached = [[] for _ in range(num_usvs)]

    # ---- 5. 主循环 ----
    step_count = 0
    termination_reason = "running"

    for step in range(max_steps):
        step_count = step + 1

        # 构建各 USV 的动作和目标
        actions = np.zeros((num_usvs, 2), dtype=np.float32)
        goal_positions = []
        active_mask = np.array(usv_active, dtype=bool)

        for i in range(num_usvs):
            usv_obs = get_single_usv_obs(obs, i)

            if usv_active[i]:
                # 当前要去的目标
                target_idx = sequences[i][seq_cursors[i]]
                target_global = target_positions[target_idx]

                # 获取 actuator 状态
                act_state = get_actuator_state(usv_obs)

                # 计算相对目标 (归一化)
                usv_pos_norm = usv_obs['o_usv'][:2]
                target_norm = target_global / SCALE
                relative_goal = target_norm - usv_pos_norm

                # 使用训练好的 actuator 获取动作 (确定性)
                action = actuator.select_action(act_state, relative_goal, evaluate=True)
                actions[i] = action
                goal_positions.append(target_global.astype(np.float32))
            else:
                # 已完成序列, 保持静止
                actions[i] = np.array([0.0, 0.0], dtype=np.float32)
                # 给一个虚拟目标 (不影响逻辑)
                goal_positions.append(
                    np.array([env.usvs[i].x, env.usvs[i].y], dtype=np.float32))

        # 执行环境步进
        obs, done, crush, truncated, info = env.step(
            actions, goal_positions, usv_active_mask=active_mask)

        # 记录路径、航向、传感器
        for i in range(num_usvs):
            paths[i].append(info['usv_positions'][i].copy())
            headings[i].append(env.usvs[i].heading)
            sensor_data[i].append(env.usvs[i].sensor_readings.copy())

        # 检查各 USV 是否到达当前子目标
        for i in range(num_usvs):
            if not usv_active[i]:
                continue
            target_idx = sequences[i][seq_cursors[i]]
            target_pos = target_positions[target_idx]
            usv_pos = info['usv_positions'][i]
            dist = np.linalg.norm(usv_pos - target_pos)

            if dist < target_radius:
                usv_reached[i].append(target_idx)
                seq_cursors[i] += 1
                if verbose:
                    print(f"  Step {step_count}: USV {i} reached target T{target_idx} "
                          f"({seq_cursors[i]}/{len(sequences[i])})")

                # 检查该 USV 是否完成全部序列
                if seq_cursors[i] >= len(sequences[i]):
                    usv_active[i] = False
                    if verbose:
                        print(f"  ★ USV {i} completed all assigned targets!")

        # 检查终止条件
        if crush:
            termination_reason = info.get('termination_reason', 'collision')
            if verbose:
                print(f"\n  ✖ Collision detected at step {step_count}!")
                if info.get('usv_collision_pairs'):
                    print(f"    USV-USV collisions: {info['usv_collision_pairs']}")
                if any(info.get('obstacle_collisions', [])):
                    collided = [i for i, c in enumerate(info['obstacle_collisions']) if c]
                    print(f"    USV-Obstacle collisions: USV {collided}")
            break

        if not any(usv_active):
            termination_reason = "all_completed"
            if verbose:
                print(f"\n  ★ All USVs completed their target sequences at step {step_count}!")
            break

        if done or truncated:
            termination_reason = info.get('termination_reason', 'timeout')
            if verbose:
                print(f"\n  Environment terminated: {termination_reason} at step {step_count}")
            break

    else:
        termination_reason = "max_steps_exceeded"
        if verbose:
            print(f"\n  Max steps ({max_steps}) exceeded")

    # ---- 6. 结果汇总 ----
    total_assigned = sum(len(s) for s in sequences)
    total_reached = sum(len(r) for r in usv_reached)
    success = termination_reason == "all_completed"

    result = {
        "success": success,
        "termination_reason": termination_reason,
        "total_steps": step_count,
        "sequences": sequences,
        "usv_reached": usv_reached,
        "total_assigned": total_assigned,
        "total_reached": total_reached,
        "completion_rate": total_reached / max(total_assigned, 1),
        "paths": paths,
        "headings": headings,
        "sensor_data": sensor_data,
        "env": env,
        "target_positions": target_positions,
        "usv_start_positions": usv_positions,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Result Summary:")
        print(f"  Success: {success}")
        print(f"  Termination: {termination_reason}")
        print(f"  Steps: {step_count}")
        print(f"  Targets: {total_reached}/{total_assigned} "
              f"({result['completion_rate']*100:.1f}%)")
        for i in range(num_usvs):
            print(f"  USV {i}: reached {len(usv_reached[i])}/{len(sequences[i])} targets")
        print(f"{'='*60}")

    return result


def visualize_result(result: dict, save_path: str = None,
                     show_usv_shape: bool = True, usv_interval: int = 80,
                     show_sensor: bool = True):
    """
    可视化 SOM 调度 + 多 USV 导航结果。

    参考单 USV 环境的 draw_path，增加 USV 船体形状与传感器扇形显示。

    Args:
        result: run_som_multi_usv 返回的结果字典
        save_path: 保存路径，为 None 则不保存
        show_usv_shape: 是否在路径上间隔绘制 USV 形状
        usv_interval: 每隔多少步绘制一个 USV 形状
        show_sensor: 是否在起点/终点 USV 上显示传感器
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon as MplPolygon
    from matplotlib.lines import Line2D

    env = result['env']
    paths = result['paths']
    headings_data = result.get('headings', None)
    sensor_data = result.get('sensor_data', None)
    sequences = result['sequences']
    num_usvs = len(paths)

    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'mathtext.fontset': 'cm'})
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 海流场
    U, V = env.U, env.V
    step_q = 4
    Xs, Ys = env.X[::step_q, ::step_q], env.Y[::step_q, ::step_q]
    Us = U[:100, :100].T[::step_q, ::step_q]
    Vs = V[:100, :100].T[::step_q, ::step_q]
    ax.quiver(Xs, Ys, Us, Vs, color='#7FB3D8', alpha=0.3,
              width=0.002, scale=25, zorder=1)

    # 障碍物
    for obs in env.obstacles:
        obs.show_obstacle(ax)

    # 目标 (带 SOM 分配颜色标注)
    usv_colors = plt.cm.tab10(np.linspace(0, 1, max(num_usvs, 1)))
    target_positions = result['target_positions']
    target_to_usv = {}
    for i, seq in enumerate(sequences):
        for order, t_idx in enumerate(seq):
            target_to_usv[t_idx] = (i, order)

    for j in range(len(env.targets)):
        t = env.targets[j]
        if j in target_to_usv:
            usv_i, order = target_to_usv[j]
            color = usv_colors[usv_i]
        else:
            color = 'gray'
        circle_t = plt.Circle((t.x, t.y), t.radius_t,
                              facecolor='none', edgecolor=color,
                              linestyle='--', linewidth=1.0, zorder=4)
        ax.add_patch(circle_t)
        ax.plot(t.x, t.y, marker='*', color=color,
                markersize=12, markeredgewidth=0.5, zorder=5)
        if j in target_to_usv:
            usv_i, order = target_to_usv[j]
            ax.text(t.x + t.radius_t + 0.5, t.y + t.radius_t + 0.5,
                    f"U{usv_i}-{order+1}", fontsize=8, color=color,
                    ha='left', va='bottom', zorder=6,
                    fontweight='bold')

    obs_geoms = [obs.geometry for obs in env.obstacles]

    # USV 路径、形状与传感器
    for i in range(num_usvs):
        color = usv_colors[i]
        path_pts = paths[i]
        if len(path_pts) == 0:
            continue

        path = np.array(path_pts)
        h_list = headings_data[i] if headings_data else None
        s_list = sensor_data[i] if sensor_data else None

        # 如果没有记录的航向，从路径差分推断
        if h_list is None or len(h_list) == 0:
            h_list = []
            for k in range(len(path)):
                if k < len(path) - 1:
                    h_list.append(np.arctan2(path[k+1, 1] - path[k, 1],
                                             path[k+1, 0] - path[k, 0]))
                else:
                    h_list.append(h_list[-1] if h_list else 0.0)

        # 路径线
        ax.plot(path[:, 0], path[:, 1], '-', color=color,
                linewidth=1.5, alpha=0.7, zorder=5)

        # 沿路径间隔绘制 USV 形状（缩小版，不带传感器）
        if show_usv_shape and len(path) > 1:
            for k in range(0, len(path), usv_interval):
                usv_m = USV(path[k, 0], path[k, 1], heading=h_list[k],
                            length=env.usv_length * 0.6,
                            width=env.usv_width * 0.6)
                verts = usv_m.get_vertices()
                if verts:
                    ax.add_patch(MplPolygon(verts, closed=True, facecolor=color,
                                           edgecolor='k', alpha=0.4,
                                           linewidth=0.4, zorder=6))

        # 起点 USV（全尺寸 + 传感器）
        start_usv = USV(path[0, 0], path[0, 1], heading=h_list[0],
                        length=env.usv_length, width=env.usv_width)
        start_usv.detect_obstacles(obs_geoms)
        if s_list and len(s_list) > 0:
            start_usv.sensor_readings = np.array(s_list[0])
        start_usv.show_usv(ax, color=color, alpha=0.8, show_sensor=show_sensor)
        ax.text(path[0, 0] + 1.5, path[0, 1] + 1.5, f"USV{i}",
                fontsize=9, color=color, fontweight='bold', zorder=9)

        # 终点 USV（全尺寸 + 传感器）
        end_usv = USV(path[-1, 0], path[-1, 1], heading=h_list[-1],
                      length=env.usv_length, width=env.usv_width)
        end_usv.detect_obstacles(obs_geoms)
        if s_list and len(s_list) > 0:
            end_usv.sensor_readings = np.array(s_list[-1])
        end_usv.show_usv(ax, color=color, alpha=0.6, show_sensor=show_sensor)

    # 图例
    legend_elements = [
        Patch(facecolor='#A0A0A0', edgecolor='#404040', label='Obstacle'),
        Patch(facecolor='none', edgecolor='#D62728',
              linestyle='--', label='Danger zone'),
        Line2D([0], [0], color='#7FB3D8', alpha=0.4,
               linewidth=1.5, label='Current'),
    ]
    for i in range(num_usvs):
        legend_elements.append(
            Line2D([0], [0], color=usv_colors[i], linewidth=2,
                   marker='s', markersize=6, label=f'USV {i}'))

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.04), ncol=min(4, len(legend_elements)),
              fontsize=9, frameon=True, edgecolor='gray',
              framealpha=1.0, handletextpad=0.4, columnspacing=1.0)

    status = "SUCCESS" if result['success'] else result['termination_reason'].upper()
    ax.set_title(f"SOM Multi-USV Navigation — {status} "
                 f"(steps={result['total_steps']}, "
                 f"targets={result['total_reached']}/{result['total_assigned']})",
                 fontsize=11, pad=8)
    ax.set_xlabel('$x$ / m')
    ax.set_ylabel('$y$ / m')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.tick_params(direction='in', length=3)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def animate_result(result: dict, save_path: str = "multi_usv_animation.gif",
                   interval: int = 50, speed: int = 5,
                   trail_length: int = 0, show_sensor: bool = True):
    """
    动态绘制多 USV 协同导航过程。

    参考单 USV 环境的 animate 方法，扩展为同时回放多个 USV 的运动轨迹，
    包括 USV 船体形状、传感器扇形、时变海流与尾迹。

    Args:
        result: run_som_multi_usv 返回的结果字典
        save_path: 保存路径（.gif），为 None 则直接显示
        interval: 帧间隔（毫秒）
        speed: 每帧前进的步数（加速回放）
        trail_length: 尾迹保留长度（0 = 保留全部已走过轨迹）
        show_sensor: 是否显示传感器
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon as MplPolygon
    from matplotlib.lines import Line2D
    from matplotlib.animation import FuncAnimation

    env = result['env']
    paths = result['paths']
    headings_data = result.get('headings', None)
    sensor_data = result.get('sensor_data', None)
    sequences = result['sequences']
    num_usvs = len(paths)

    # 补算缺失航向
    all_headings = []
    for i in range(num_usvs):
        path = np.array(paths[i]) if len(paths[i]) > 0 else np.empty((0, 2))
        if headings_data and len(headings_data[i]) == len(path):
            all_headings.append(headings_data[i])
        else:
            h = []
            for k in range(len(path)):
                if k < len(path) - 1:
                    h.append(np.arctan2(path[k+1, 1] - path[k, 1],
                                        path[k+1, 0] - path[k, 0]))
                else:
                    h.append(h[-1] if h else 0.0)
            all_headings.append(h)

    max_len = max(len(p) for p in paths) if paths else 0
    total_frames = (max_len - 1) // speed + 1 if max_len > 1 else 1

    # 画布
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'mathtext.fontset': 'cm'})
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 静态: 障碍物、目标
    for obs in env.obstacles:
        obs.show_obstacle(ax)
    obs_geoms = [obs.geometry for obs in env.obstacles]

    usv_colors = plt.cm.tab10(np.linspace(0, 1, max(num_usvs, 1)))
    target_positions = result['target_positions']
    target_to_usv = {}
    for i, seq in enumerate(sequences):
        for order, t_idx in enumerate(seq):
            target_to_usv[t_idx] = (i, order)

    for j in range(len(env.targets)):
        t = env.targets[j]
        if j in target_to_usv:
            usv_i, _ = target_to_usv[j]
            color = usv_colors[usv_i]
        else:
            color = 'gray'
        circle_t = plt.Circle((t.x, t.y), t.radius_t,
                              facecolor='none', edgecolor=color,
                              linestyle='--', linewidth=1.0, zorder=4)
        ax.add_patch(circle_t)
        ax.plot(t.x, t.y, marker='*', color=color,
                markersize=12, markeredgewidth=0.5, zorder=5)
        if j in target_to_usv:
            usv_i, order = target_to_usv[j]
            ax.text(t.x + t.radius_t + 0.5, t.y + t.radius_t + 0.5,
                    f"U{usv_i}-{order+1}", fontsize=8, color=color,
                    ha='left', va='bottom', zorder=6, fontweight='bold')

    # 海流 Quiver（时变）
    step_q = 4
    Xs, Ys = env.X[::step_q, ::step_q], env.Y[::step_q, ::step_q]
    if env.current_type == "generated":
        U0, V0 = env._generate_current_data(t=0)
    else:
        U0, V0 = env.U, env.V
    Us0 = U0[:100, :100].T[::step_q, ::step_q]
    Vs0 = V0[:100, :100].T[::step_q, ::step_q]
    quiver = ax.quiver(Xs, Ys, Us0, Vs0,
                       width=0.002, scale=25, color='#7FB3D8', alpha=0.4, zorder=1)

    # 尾迹线
    trail_lines = []
    for i in range(num_usvs):
        line, = ax.plot([], [], '-', color=usv_colors[i],
                        linewidth=1.5, alpha=0.7, zorder=5)
        trail_lines.append(line)

    title = ax.set_title('', fontsize=11, pad=8)
    ax.set_xlabel('$x$ / m')
    ax.set_ylabel('$y$ / m')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.tick_params(direction='in', length=3)

    # 记录静态 patches/lines 数量，用于每帧清除动态 USV artists
    n_static_patches = len(ax.patches)
    n_static_lines = len(ax.lines)
    _prev_artists = []

    def _update(frame):
        nonlocal _prev_artists

        # 清除上一帧的 USV artists
        for a in _prev_artists:
            try:
                a.remove()
            except ValueError:
                pass
        _prev_artists.clear()

        idx = min(frame * speed, max_len - 1) if max_len > 0 else 0
        t = idx * env.dt

        # 更新时变海流
        if env.current_type == "generated":
            Ut, Vt = env._generate_current_data(t=t)
            Us = Ut[:100, :100].T[::step_q, ::step_q]
            Vs = Vt[:100, :100].T[::step_q, ::step_q]
            quiver.set_UVC(Us, Vs)

        # 更新每个 USV
        for i in range(num_usvs):
            path = paths[i]
            if len(path) == 0:
                continue
            path_arr = np.array(path)
            h_list = all_headings[i]
            s_list = sensor_data[i] if sensor_data else None

            ki = min(idx, len(path) - 1)

            # 尾迹
            s = max(0, ki - trail_length) if trail_length > 0 else 0
            trail_lines[i].set_data(path_arr[s:ki+1, 0], path_arr[s:ki+1, 1])

            # 记录当前 patches/lines 数量
            np_before = len(ax.patches)
            nl_before = len(ax.lines)

            # 绘制 USV 形状 + 传感器
            usv_now = USV(path_arr[ki, 0], path_arr[ki, 1],
                          heading=h_list[ki],
                          length=env.usv_length, width=env.usv_width)
            if s_list and ki < len(s_list):
                usv_now.sensor_readings = np.array(s_list[ki])
            else:
                usv_now.detect_obstacles(obs_geoms)
            usv_now.show_usv(ax, color=usv_colors[i], alpha=0.85,
                             show_sensor=show_sensor)

            # 收集新增的 dynamic artists
            _prev_artists.extend(ax.patches[np_before:])
            _prev_artists.extend(ax.lines[nl_before:])

        title.set_text(f'Step {idx}/{max_len-1}  $t={t:.1f}$ s')
        return [quiver] + trail_lines

    anim = FuncAnimation(fig, _update, frames=total_frames,
                         interval=interval, blit=False, repeat=False)

    if save_path is not None:
        try:
            anim.save(save_path, writer='pillow')
            print(f"Animation saved to {save_path}  ({total_frames} frames)")
        except Exception as e:
            print(f"Failed to save animation: {e}")
        plt.close()
    else:
        plt.show()

    return anim


if __name__ == "__main__":
    result = run_som_multi_usv(
        num_usvs=3,
        num_targets=6,
        num_obstacles=5,
        model_dir="expert_hrl_sac_output_310",
        seed=2026,
        max_steps=3000,
        verbose=True,
    )
    visualize_result(result, save_path="som_multi_usv_result.png")
    animate_result(result, save_path="som_multi_usv_anim.gif", speed=10)
