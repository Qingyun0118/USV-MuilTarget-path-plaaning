import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as MplPolygon, Circle, Wedge
from matplotlib.lines import Line2D
from my_USV_Env import Obstacle, Target, ObstacleShape
from USV import USV


def draw_scene(save_path="scene_demo.png"):
    """
    绘制 USV 工作场景演示图 (论文插图风格):
      - 1 艘 USV (船体较大比例，带航向标注)
      - 3 个不同形状障碍物 (圆、矩形、多边形)
      - 2 个目标点 (去掉 1 号)
      - 150° FOV 传感器，15 rays，检测到更多障碍物交线
      - 局部海流场
    """
    plt.rcParams.update({
        'font.family': ['SimHei', 'serif'],
        'font.size': 13,
        'mathtext.fontset': 'cm'
    })

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ---------- 1. 固定障碍物 (3个，不同形状) ----------
    obstacles = [
        Obstacle(58.0, 58.0, radius_o=3.5, rou_c=1.5, rou_s=2.0,
                 shape=ObstacleShape.CIRCLE),
        Obstacle(38.0, 64.0, rou_c=1.5, rou_s=2.0,
                 shape=ObstacleShape.RECTANGLE,
                 width=9.0, height=5.0, rotation=np.radians(20)),
        Obstacle(60.0, 45.0, rou_c=1.5, rou_s=2.0,
                 shape=ObstacleShape.POLYGON,
                 vertices=[(0, -4), (3.5, -1.5), (4.5, 2.5), (0, 5), (-3.5, 1.5)],
                 rotation=np.radians(-20)),
    ]

    # ---------- 2. 固定目标点 (去掉 1 号，3 号移远) ----------
    target_positions = [
        (55.0, 65.0),   # 原 2 号
        (50.0, 60.0),   # 原 3 号，移远
    ]
    targets = [Target(x, y, radius_t=2.5) for x, y in target_positions]

    # ---------- 3. USV (船体放大，更靠近障碍物以增加交线) ----------
    usv = USV(x=48.0, y=46.0, heading=np.radians(50.0),
              length=5.0, width=2.5)
    usv.sensor_range = 18.0
    usv.sensor_fov = np.radians(150)
    usv.num_detection_lines = 15
    usv.sensor_readings = [usv.sensor_range] * usv.num_detection_lines

    obs_geoms = [obs.geometry for obs in obstacles]
    usv.detect_obstacles(obs_geoms)

    # ---------- 4. 局部海流场 ----------
    def _calc_current(xx, yy, tt=0.0):
        scale_factor_x = 0.2
        scale_factor_y = 0.04
        offset = 50.0
        x_mapped = (xx - offset) * scale_factor_x
        y_mapped = (yy - offset) * scale_factor_y
        B0 = 1.2
        epsilon = 0.3
        omega = 0.4
        beta = np.pi / 2
        k = 0.84
        c = 0.12
        Bt = B0 + epsilon * np.cos(omega * tt + beta)
        X_term = k * (x_mapped - c * tt)
        Y_term = y_mapped - Bt * np.cos(X_term)
        sin_X = np.sin(X_term)
        cos_X = np.cos(X_term)
        D_sq = 1 + k**2 * Bt**2 * sin_X**2
        D = np.sqrt(D_sq)
        A = Y_term / D
        try:
            cosh_A = np.cosh(A)
            sech2_A = 1.0 / (cosh_A**2)
        except OverflowError:
            sech2_A = 0.0
        uu = sech2_A / D
        dN_dx = k * Bt * sin_X
        dD_dx = (k**3 * Bt**2 * sin_X * cos_X) / D
        dA_dx = (dN_dx * D - Y_term * dD_dx) / D_sq
        vv = -sech2_A * dA_dx
        return uu, vv

    step = 2
    x_grid = np.arange(26, 74, step)
    y_grid = np.arange(26, 74, step)
    Xs, Ys = np.meshgrid(x_grid, y_grid)
    Us = np.zeros_like(Xs, dtype=float)
    Vs = np.zeros_like(Ys, dtype=float)

    for i in range(Xs.shape[0]):
        for j in range(Xs.shape[1]):
            xx, yy = Xs[i, j], Ys[i, j]
            in_obs = False
            for obs in obstacles:
                if obs.is_point_in_obstacle([xx, yy]):
                    in_obs = True
                    break
            if not in_obs:
                uu, vv = _calc_current(xx, yy, tt=0.0)
                Us[i, j] = uu
                Vs[i, j] = vv

    ax.quiver(Xs, Ys, Us, Vs, color='#7FB3D8', alpha=0.45,
              width=0.0025, scale=25, zorder=1)

    # ---------- 5. 绘制障碍物 ----------
    obs_face = '#A0A0A0'
    obs_edge = '#404040'
    danger_edge = '#D62728'

    for obs in obstacles:
        if obs.shape == ObstacleShape.CIRCLE:
            ax.add_patch(plt.Circle((obs.x, obs.y), obs.radius_o + obs.rou_c,
                                    facecolor='none', edgecolor=danger_edge,
                                    linestyle='--', linewidth=0.9, zorder=2))
            ax.add_patch(plt.Circle((obs.x, obs.y), obs.radius_o,
                                    facecolor=obs_face, edgecolor=obs_edge,
                                    linewidth=0.9, zorder=3))
        else:
            dv = list(obs.danger_zone.exterior.coords)[:-1]
            ax.add_patch(MplPolygon(dv, closed=True, facecolor='none',
                                    edgecolor=danger_edge, linestyle='--',
                                    linewidth=0.9, zorder=2))
            verts = list(obs.geometry.exterior.coords)[:-1]
            ax.add_patch(MplPolygon(verts, closed=True, facecolor=obs_face,
                                    edgecolor=obs_edge, linewidth=0.9, zorder=3))

    # ---------- 6. 绘制目标点 ----------
    for idx, target in enumerate(targets):
        circle_t = plt.Circle((target.x, target.y), target.radius_t,
                              facecolor='none', edgecolor='#2CA02C',
                              linestyle='--', linewidth=0.9, zorder=4)
        ax.add_patch(circle_t)
        ax.plot(target.x, target.y, marker='*', color='#2CA02C',
                markersize=11, markeredgewidth=0.5, zorder=5)
        ax.text(target.x + target.radius_t + 0.5, target.y + target.radius_t + 0.5,
                str(idx + 1), fontsize=12, color='#2CA02C',
                fontweight='bold', ha='left', va='bottom', zorder=6)

    # ---------- 7. 自定义绘制 USV + 传感器 ----------
    # 7.1 传感器扇形背景
    wedge = Wedge((usv.x, usv.y), usv.sensor_range,
                  np.degrees(usv.heading - usv.sensor_fov / 2),
                  np.degrees(usv.heading + usv.sensor_fov / 2),
                  facecolor='#AED6F1', edgecolor='none', alpha=0.25, zorder=3)
    ax.add_patch(wedge)

    # 7.2 传感器射线
    angles = np.linspace(
        usv.heading - usv.sensor_fov / 2,
        usv.heading + usv.sensor_fov / 2,
        usv.num_detection_lines
    )
    for i, dist in enumerate(usv.sensor_readings):
        angle = angles[i]
        end_x = usv.x + dist * np.cos(angle)
        end_y = usv.y + dist * np.sin(angle)

        if dist < usv.sensor_range - 0.1:
            ax.plot([usv.x, end_x], [usv.y, end_y],
                    color='#C0392B', alpha=0.55, linewidth=1.0, zorder=4)
            ax.plot(end_x, end_y, 'o', color='#C0392B', markersize=3.5, zorder=5)
        else:
            ax.plot([usv.x, end_x], [usv.y, end_y],
                    color='#5D6D7E', alpha=0.35, linewidth=0.7,
                    linestyle='--', zorder=4)

    # 7.3 USV 本体
    vertices = usv.get_vertices()
    patch = MplPolygon(vertices, closed=True, facecolor='#2E86AB',
                       edgecolor='black', alpha=0.9, linewidth=1.0, zorder=6)
    ax.add_patch(patch)

    # 7.4 航向箭头
    arrow_len = usv.length * 1.0
    ax.annotate('', xy=(usv.x + arrow_len * np.cos(usv.heading),
                        usv.y + arrow_len * np.sin(usv.heading)),
                xytext=(usv.x, usv.y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                zorder=7)

    # 航向标注 (heading ψ) — 黑体加大
    label_offset = 5.0
    ax.text(usv.x - label_offset, usv.y + label_offset,
            r'$\psi$', fontsize=17, color='black',
            fontweight='bold', ha='center', va='center')

    # FOV 角度标注弧线 + 文字
    arc_radius = usv.sensor_range * 0.32
    arc_angles = np.linspace(
        usv.heading - usv.sensor_fov / 2,
        usv.heading + usv.sensor_fov / 2, 60)
    arc_x = usv.x + arc_radius * np.cos(arc_angles)
    arc_y = usv.y + arc_radius * np.sin(arc_angles)
    ax.plot(arc_x, arc_y, 'k--', linewidth=0.7, alpha=0.5, zorder=5)
    mid_angle = usv.heading
    ax.text(usv.x + (arc_radius + 3.0) * np.cos(mid_angle) - 1.2,
            usv.y + (arc_radius + 3.0) * np.sin(mid_angle) - 1.2,
            r'$150^{\circ}$ FOV', fontsize=12, color='black',
            fontweight='bold', ha='center', va='center')

    # 15 rays 文字标注
    ax.text(usv.x + usv.sensor_range * 0.70 * np.cos(usv.heading + 0.22),
            usv.y + usv.sensor_range * 0.70 * np.sin(usv.heading + 0.22),
            '15 rays', fontsize=11, color='#2C3E50',
            fontweight='bold', ha='center', va='center')

    # ---------- 8. 图例与排版 ----------
    legend_elements = [
        Line2D([0], [0], marker='*', color='#2CA02C', markersize=11,
               label='Target', linewidth=0),
        Patch(facecolor='#A0A0A0', edgecolor='#404040', label='Obstacle'),
        Patch(facecolor='none', edgecolor='#D62728',
              linestyle='--', label='Danger zone'),
        MplPolygon([(0, 0)], closed=True, facecolor='#2E86AB',
                   edgecolor='black', alpha=0.9, label='USV'),
        Line2D([0], [0], color='#5D6D7E', alpha=0.4, linewidth=1.0,
               linestyle='--', label='Sensor (free)'),
        Line2D([0], [0], color='#C0392B', alpha=0.5, linewidth=1.0,
               label='Sensor (hit)'),
        Patch(facecolor='#AED6F1', alpha=0.25, label='Sensor FOV'),
        Line2D([0], [0], color='#7FB3D8', alpha=0.45,
               linewidth=1.5, label='Current'),
    ]
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.04), ncol=4,
              fontsize=10, frameon=True, edgecolor='gray',
              framealpha=1.0, handletextpad=0.4, columnspacing=1.0)

    ax.set_xlabel('$x$ / m', fontsize=14, fontweight='bold')
    ax.set_ylabel('$y$ / m', fontsize=14, fontweight='bold')
    ax.set_title('USV workspace scene with sonar detection',
                 fontsize=15, fontweight='bold', pad=10)

    ax.set_xlim(26, 74)
    ax.set_ylim(26, 74)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.tick_params(direction='in', length=4)
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Scene saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    draw_scene("usv_scene_demo.png")
