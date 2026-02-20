"""
USV 多目标路径规划 — 雷达图 (Radar Chart) 绘制模块

用于可视化 6 项综合评估指标:
  1. 路径长度 (SPM)
  2. 路径平滑度 (SmPM)
  3. 能效 (MECM)
  4. 路径安全度 (SaPM)
  5. 样本效率 (SE)
  6. 任务成功率 (SR)

支持单算法和多算法对比雷达图。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Optional, Tuple
import os

# 字体设置: 支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# 指标标签 (中英文)
METRIC_LABELS_CN = [
    '路径长度',
    '路径平滑度',
    '能效',
    '路径安全度',
    '样本效率',
    '任务成功率',
]

METRIC_LABELS_EN = [
    'Path Length\n(SPM)',
    'Smoothness\n(SmPM)',
    'Energy\nEfficiency\n(MECM)',
    'Safety\n(SaPM)',
    'Sample\nEfficiency\n(SE)',
    'Success\nRate\n(SR)',
]


def _create_radar_axes(fig, rect=None):
    """
    创建雷达图的极坐标轴

    Args:
        fig:  matplotlib Figure
        rect: 子图位置 [left, bottom, width, height]

    Returns:
        ax: 极坐标 Axes
        angles: 各指标对应的角度数组 (含闭合点)
    """
    N = len(METRIC_LABELS_CN)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    if rect is not None:
        ax = fig.add_axes(rect, polar=True)
    else:
        ax = fig.add_subplot(111, polar=True)

    # 起始角度从顶部开始, 顺时针排列
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    return ax, angles


def plot_single_radar(metrics: Dict[str, float],
                      algorithm_name: str = "Algorithm",
                      title: str = None,
                      save_path: Optional[str] = None,
                      show: bool = True,
                      use_english: bool = False,
                      color: str = '#2196F3',
                      fig_size: Tuple[float, float] = (8, 8)):
    """
    绘制单个算法的雷达图

    Args:
        metrics:        字典 {指标名(中文): 0~1 评分}
        algorithm_name: 算法名称
        title:          图标题, None 则自动生成
        save_path:      保存路径, None 则不保存
        show:           是否显示图形
        use_english:    是否使用英文标签
        color:          填充颜色
        fig_size:       图形尺寸
    """
    labels = METRIC_LABELS_EN if use_english else METRIC_LABELS_CN
    N = len(labels)

    # 提取值 (按标签顺序)
    values = []
    for cn_label in METRIC_LABELS_CN:
        values.append(metrics.get(cn_label, 0.0))
    values_closed = values + values[:1]

    fig = plt.figure(figsize=fig_size, facecolor='white')
    ax, angles = _create_radar_axes(fig)

    # 绘制背景网格
    _draw_radar_background(ax, N, labels)

    # 绘制数据
    ax.plot(angles, values_closed, 'o-', linewidth=2.5, color=color,
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=color, zorder=5)
    ax.fill(angles, values_closed, alpha=0.25, color=color, zorder=4)

    # 标注数值
    for i, (angle, val) in enumerate(zip(angles[:-1], values)):
        offset = 0.12
        ax.text(angle, val + offset, f'{val:.2f}',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.8))

    if title is None:
        title = f'综合评估雷达图 — {algorithm_name}'
    ax.set_title(title, size=16, fontweight='bold', pad=30)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"雷达图已保存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_multi_radar(metrics_dict: Dict[str, Dict[str, float]],
                     title: str = "多算法综合评估对比",
                     save_path: Optional[str] = None,
                     show: bool = True,
                     use_english: bool = False,
                     colors: Optional[List[str]] = None,
                     fig_size: Tuple[float, float] = (10, 10)):
    """
    绘制多个算法的对比雷达图

    Args:
        metrics_dict: {算法名: {指标名: 评分}} 的嵌套字典
        title:        图标题
        save_path:    保存路径
        show:         是否显示
        use_english:  是否使用英文标签
        colors:       各算法颜色列表
        fig_size:     图形尺寸
    """
    labels = METRIC_LABELS_EN if use_english else METRIC_LABELS_CN
    N = len(labels)

    # 默认配色方案
    default_colors = [
        '#2196F3',  # 蓝色
        '#FF5722',  # 橙红
        '#4CAF50',  # 绿色
        '#9C27B0',  # 紫色
        '#FF9800',  # 橙色
        '#00BCD4',  # 青色
        '#E91E63',  # 粉色
        '#795548',  # 棕色
    ]
    if colors is None:
        colors = default_colors

    fig = plt.figure(figsize=fig_size, facecolor='white')
    ax, angles = _create_radar_axes(fig)

    # 绘制背景
    _draw_radar_background(ax, N, labels)

    # 逐算法绘制
    for idx, (algo_name, metrics) in enumerate(metrics_dict.items()):
        color = colors[idx % len(colors)]

        values = []
        for cn_label in METRIC_LABELS_CN:
            values.append(metrics.get(cn_label, 0.0))
        values_closed = values + values[:1]

        ax.plot(angles, values_closed, 'o-', linewidth=2.5, color=color,
                markersize=7, markerfacecolor='white', markeredgewidth=2,
                markeredgecolor=color, label=algo_name, zorder=5 + idx)
        ax.fill(angles, values_closed, alpha=0.12, color=color, zorder=4)

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              fontsize=11, framealpha=0.9, edgecolor='gray',
              title='算法', title_fontsize=12)

    ax.set_title(title, size=16, fontweight='bold', pad=30)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"对比雷达图已保存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def _draw_radar_background(ax, N, labels):
    """
    绘制雷达图背景 (同心圆、网格线、标签)

    Args:
        ax:     极坐标 Axes
        N:      指标数量
        labels: 标签列表
    """
    angles_ticks = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # 同心圆
    rings = [0.2, 0.4, 0.6, 0.8, 1.0]
    for r in rings:
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(theta, [r] * 100, color='#E0E0E0', linewidth=0.8, zorder=1)
        # 数值标注 (仅在第一条轴线旁标注)
        ax.text(angles_ticks[0], r + 0.02, f'{r:.1f}',
                ha='center', va='bottom', fontsize=8, color='#999999')

    # 轴线
    for angle in angles_ticks:
        ax.plot([angle, angle], [0, 1.0], color='#CCCCCC',
                linewidth=0.8, zorder=1)

    # 标签
    ax.set_xticks(angles_ticks)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')

    # 范围
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)


def plot_metrics_bar_chart(metrics_dict: Dict[str, Dict[str, float]],
                           title: str = "各指标对比柱状图",
                           save_path: Optional[str] = None,
                           show: bool = True,
                           fig_size: Tuple[float, float] = (14, 6)):
    """
    绘制各指标的分组柱状图 (补充雷达图)

    Args:
        metrics_dict: {算法名: {指标名: 评分}}
        title:        图标题
        save_path:    保存路径
        show:         是否显示
        fig_size:     图形尺寸
    """
    algo_names = list(metrics_dict.keys())
    n_algos = len(algo_names)
    n_metrics = len(METRIC_LABELS_CN)

    default_colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0',
                      '#FF9800', '#00BCD4', '#E91E63', '#795548']

    fig, ax = plt.subplots(figsize=fig_size, facecolor='white')

    bar_width = 0.8 / n_algos
    x = np.arange(n_metrics)

    for i, algo_name in enumerate(algo_names):
        metrics = metrics_dict[algo_name]
        values = [metrics.get(label, 0.0) for label in METRIC_LABELS_CN]
        offset = (i - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width * 0.9,
                      label=algo_name,
                      color=default_colors[i % len(default_colors)],
                      alpha=0.85, edgecolor='white', linewidth=0.5)

        # 数值标注
        for bar, val in zip(bars, values):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                        fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS_CN, fontsize=11, fontweight='bold')
    ax.set_ylabel('归一化评分', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"柱状图已保存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# ============================================================================
# 演示 / 测试入口
# ============================================================================

if __name__ == '__main__':
    # 示例数据: 模拟三种算法的评估结果
    demo_metrics = {
        'MPC-HRL (Ours)': {
            '路径长度': 0.82,
            '路径平滑度': 0.75,
            '能效': 0.68,
            '路径安全度': 0.90,
            '样本效率': 0.70,
            '任务成功率': 0.85,
        },
        'Pure RL (TD3)': {
            '路径长度': 0.65,
            '路径平滑度': 0.55,
            '能效': 0.50,
            '路径安全度': 0.60,
            '样本效率': 0.40,
            '任务成功率': 0.60,
        },
        'Pure MPC': {
            '路径长度': 0.70,
            '路径平滑度': 0.85,
            '能效': 0.72,
            '路径安全度': 0.95,
            '样本效率': 0.95,
            '任务成功率': 0.50,
        },
    }

    print("绘制单算法雷达图...")
    plot_single_radar(
        demo_metrics['MPC-HRL (Ours)'],
        algorithm_name='MPC-HRL (Ours)',
        save_path='radar_single_demo.png',
        show=False
    )

    print("\n绘制多算法对比雷达图...")
    plot_multi_radar(
        demo_metrics,
        title='多算法综合评估对比',
        save_path='radar_multi_demo.png',
        show=False
    )

    print("\n绘制柱状图...")
    plot_metrics_bar_chart(
        demo_metrics,
        title='各指标对比柱状图',
        save_path='bar_chart_demo.png',
        show=False
    )

    print("\n演示完成！")
