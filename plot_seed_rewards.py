"""
多算法 × 多随机种子 训练曲线对比绘制脚本
支持从 TensorBoard event 文件或 CSV 文件读取数据
支持按 episode 或全局步数 (global steps) 作为横轴
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import json
import numpy as np
import matplotlib.pyplot as plt

# ============ 配置区 ============
# 数据源: "tensorboard" 直接读取 event 文件, "csv" 从导出的 CSV 文件读取
DATA_SOURCE = "tensorboard"  # "tensorboard" | "csv"
# DATA_SOURCE = "csv"  # "tensorboard" | "csv"
# 横轴: "episode" 按 episode 编号, "global_steps" 按累计环境步数（从 training_history.json）
X_AXIS = "episodes"  # "episodes" | "global_steps"
# X_AXIS = "global_steps"  # "episodes" | "global_steps"
# 每个算法：名称 -> 该算法各种子的日志目录列表
# DATA_SOURCE="tensorboard" 时: 填 TensorBoard event 目录 (含 events.out.tfevents.* 的目录)
# DATA_SOURCE="csv" 时: 填 CSV 文件所在目录, CSV 文件名需与 tag 一致 (如 Reward_Actuator.csv)
#
# 注意: X_AXIS="global_steps" 时, 脚本会自动从日志目录向上两级查找 training_history.json
#       (即 <output_dir>/runs/<run_name>/ -> <output_dir>/training_history.json)
#       如果目录结构不同, 可在 HISTORY_OVERRIDE 中手动指定
ALGORITHMS = {
    # "Expert-HRL-TD7-HardQ": [
    #     r"1_Result\trainingResult\TD7\seed2026\Expert_HRL_TD7_HardQ2026",
    #     r"1_Result\trainingResult\TD7\seed2025\Expert_HRL_TD7_HardQ2025",
    #     r"1_Result\trainingResult\TD7\seed2024\Expert_HRL_TD7_HardQ2024",
    #     r"1_Result\trainingResult\TD7\seed2023\Expert_HRL_TD7_HardQ2023",
    #     r"1_Result\trainingResult\TD7\seed2022\Expert_HRL_TD7_HardQ2022",
    #     r"1_Result\trainingResult\TD7\seed2021\Expert_HRL_TD7_HardQ2021",
    # ],
    # "Expert-HRL-TD7-SoftQ": [
    #     r"1_Result\trainingResult\TD7\seed2026\Expert_HRL_TD7_SoftQ2026",
    #     r"1_Result\trainingResult\TD7\seed2025\Expert_HRL_TD7_SoftQ2025",
    #     r"1_Result\trainingResult\TD7\seed2024\Expert_HRL_TD7_SoftQ2024",
    #     r"1_Result\trainingResult\TD7\seed2023\Expert_HRL_TD7_SoftQ2023",
    #     r"1_Result\trainingResult\TD7\seed2022\Expert_HRL_TD7_SoftQ2022",
    #     r"1_Result\trainingResult\TD7\seed2021\Expert_HRL_TD7_SoftQ2021",
    # ],
    "Expert-HRL-TD3-HardQ": [
        r"1_Result\trainingResult\TD3\seed2022\Expert_HRL_TD3_HardQ2022",
        r"1_Result\trainingResult\TD3\seed2025\Expert_HRL_TD3_HardQ2025",
        r"1_Result\trainingResult\TD3\seed2024\Expert_HRL_TD3_HardQ2024",
        r"1_Result\trainingResult\TD3\seed2023\Expert_HRL_TD3_HardQ2023",
        r"1_Result\trainingResult\TD3\seed2026\Expert_HRL_TD3_HardQ2026",
    ],
    "Expert-HRL-TD3-SoftQ": [
        r"1_Result\trainingResult\TD3\seed2026\Expert_HRL_TD3_SoftQ2026",
        r"1_Result\trainingResult\TD3\seed2025\Expert_HRL_TD3_SoftQ2025",
        r"1_Result\trainingResult\TD3\seed2024\Expert_HRL_TD3_SoftQ2024",
        r"1_Result\trainingResult\TD3\seed2023\Expert_HRL_TD3_SoftQ2023",
        r"1_Result\trainingResult\TD3\seed2022\Expert_HRL_TD3_SoftQ2022",
    ],
    # "Expert-HRL-SAC-HardQ": [
    #     r"1_Result\trainingResult\SAC\seed2022\Expert_HRL_SAC_HardQ2022",
    #     r"1_Result\trainingResult\SAC\seed2023\Expert_HRL_SAC_HardQ2023",
    #     r"1_Result\trainingResult\SAC\seed2024\Expert_HRL_SAC_HardQ2024",
    #     r"1_Result\trainingResult\SAC\seed2025\Expert_HRL_SAC_HardQ2025",
    # ],
    # "Expert-HRL-SAC-SoftQ": [
    #     r"1_Result\trainingResult\SAC\seed2025\Expert_HRL_SAC_SoftQ2025",
    #     r"1_Result\trainingResult\SAC\seed2024\Expert_HRL_SAC_SoftQ2024",
    #     r"1_Result\trainingResult\SAC\seed2023\Expert_HRL_SAC_SoftQ2023",
    #     r"1_Result\trainingResult\SAC\seed2022\Expert_HRL_SAC_SoftQ2022",
    # ],

}

# 要绘制的 tag（TensorBoard 中的标量名）
TAGS = [
    "Reward/Actuator",
    "Reward/Selector",
    "Metrics/TotalDistance",
    "Metrics/SuccessRate",
    # ---- Soft vs Hard Q-filter 对比分析 ----
    # "Training/BC_AvgSoftWeight",   # Soft: sigmoid 平均权重; Hard: 等于 hit rate
    # "Training/BC_HitRate",         # Q_expert > Q_policy 的比例 (BC 信号通过率)
    # "Metrics/Collision",           # 碰撞率 (平滑梯度是否降低早期碰撞)
]

# 平滑窗口大小（设为 1 则不平滑）
SMOOTH_WINDOW = 50

# 输出图片路径（None 则只显示不保存）
SAVE_PATH = None

# 手动指定 training_history.json 路径 (可选, 覆盖自动查找)
# 格式: { "日志目录路径": "training_history.json 完整路径" }
HISTORY_OVERRIDE = {}

# 颜色列表（按算法顺序分配）
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
# ================================


def read_scalar_tb(log_dir: str, tag: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        print(f"[WARN] tag '{tag}' not found in {log_dir}")
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def read_scalar_csv(csv_dir: str, tag: str):
    import csv
    fname = tag.replace("/", "_") + ".csv"
    fpath = os.path.join(csv_dir, fname)
    if not os.path.exists(fpath):
        print(f"[WARN] CSV file not found: {fpath}")
        return None, None
    steps, values = [], []
    with open(fpath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["Step"]))
            values.append(float(row["Value"]))
    return np.array(steps), np.array(values)


def read_scalar(log_dir: str, tag: str, data_source: str = "tensorboard"):
    if data_source == "csv":
        return read_scalar_csv(log_dir, tag)
    return read_scalar_tb(log_dir, tag)


def find_training_history(log_dir: str):
    if log_dir in HISTORY_OVERRIDE:
        return HISTORY_OVERRIDE[log_dir]
    # 自动查找: <output_dir>/runs/<run_name>/ -> <output_dir>/training_history.json
    candidate = os.path.join(log_dir, "..", "..", "training_history.json")
    candidate = os.path.normpath(candidate)
    if os.path.exists(candidate):
        return candidate
    # 再向上一级尝试
    candidate2 = os.path.join(log_dir, "..", "training_history.json")
    candidate2 = os.path.normpath(candidate2)
    if os.path.exists(candidate2):
        return candidate2
    return None


def load_global_steps(log_dir: str):
    hist_path = find_training_history(log_dir)
    if hist_path is None:
        print(f"[WARN] training_history.json not found for {log_dir}")
        return None
    with open(hist_path, "r") as f:
        history = json.load(f)
    episode_steps = np.array(history["episode_steps"])
    return np.cumsum(episode_steps)


def smooth(values: np.ndarray, window: int):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = len(values) - len(smoothed)
    return np.concatenate([values[:pad], smoothed])


def align_and_aggregate(all_steps, all_values):
    min_len = min(len(v) for v in all_values)
    aligned = np.array([v[:min_len] for v in all_values])
    mean = aligned.mean(axis=0)
    std = aligned.std(axis=0)
    steps = all_steps[0][:min_len]
    return steps, mean, std


def interpolate_and_aggregate(all_x, all_values, n_points=1000):
    x_min = max(x[0] for x in all_x)
    x_max = min(x[-1] for x in all_x)
    x_common = np.linspace(x_min, x_max, n_points)
    interpolated = []
    for x, v in zip(all_x, all_values):
        interpolated.append(np.interp(x_common, x, v))
    interpolated = np.array(interpolated)
    mean = interpolated.mean(axis=0)
    std = interpolated.std(axis=0)
    return x_common, mean, std


def plot_algo_compare(algorithms, tags, smooth_window=50, save_path=None,
                      data_source="tensorboard", x_axis="episodes"):
    n_tags = len(tags)
    n_cols = (n_tags + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10), squeeze=False)
    axes = axes.flatten()

    x_label = "global steps" if x_axis == "global_steps" else "episodes"

    for idx, tag in enumerate(tags):
        ax = axes[idx]

        for algo_idx, (algo_name, seed_dirs) in enumerate(algorithms.items()):
            color = COLORS[algo_idx % len(COLORS)]
            all_x = []
            all_values = []

            for d in seed_dirs:
                steps, values = read_scalar(d, tag, data_source)
                if steps is None:
                    continue

                if x_axis == "global_steps":
                    global_steps = load_global_steps(d)
                    if global_steps is None:
                        continue
                    n = min(len(values), len(global_steps))
                    values = values[:n]
                    x = global_steps[:n]
                else:
                    x = steps

                values = smooth(values, smooth_window)
                all_x.append(x)
                all_values.append(values)

            if len(all_values) == 0:
                continue

            if len(all_values) == 1:
                ax.plot(all_x[0], all_values[0], linewidth=1.2,
                        color=color, label=f"{algo_name} (1 seed)")
            else:
                if x_axis == "global_steps":
                    x_interp, mean, std = interpolate_and_aggregate(all_x, all_values)
                else:
                    x_interp, mean, std = align_and_aggregate(all_x, all_values)
                ax.plot(x_interp, mean, linewidth=1.5, color=color,
                        label=f"{algo_name} ")
                ax.fill_between(x_interp, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_xlabel(x_label)
        y_label = tag.split("/")[-1] + " Reward" if tag.startswith("Reward/") else tag.split("/")[-1]
        ax.set_ylabel(y_label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for idx in range(len(tags), len(axes)):
        axes[idx].set_visible(False)

    # fig.suptitle(f"Algorithm Comparison", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    if not ALGORITHMS or all(len(v) == 0 for v in ALGORITHMS.values()):
        print("请在脚本顶部 ALGORITHMS 字典中填入各算法的种子日志目录")
        print("格式:")
        print('  ALGORITHMS = {')
        print('      "算法名": [r"seed1日志目录", r"seed2日志目录"],')
        print('      "算法名2": [r"seed1日志目录", r"seed2日志目录"],')
        print('  }')
        exit(1)

    plot_algo_compare(ALGORITHMS, TAGS, SMOOTH_WINDOW, SAVE_PATH, DATA_SOURCE, X_AXIS)
