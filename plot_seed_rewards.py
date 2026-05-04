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
import pandas as pd

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
    ######################## TD7-HardQ VS TD7-SoftQ ########################
    # "Expert-HRL-TD7-HardQ": [
    #     r"1_Result\trainingResult\TD7\seed2025\Expert_HRL_TD7_HardQ2025",
    #     r"1_Result\trainingResult\TD7\seed2024\Expert_HRL_TD7_HardQ2024",
    #     r"1_Result\trainingResult\TD7\seed2023\Expert_HRL_TD7_HardQ2023",
    #     r"1_Result\trainingResult\TD7\seed2022\Expert_HRL_TD7_HardQ2022",
    #     r"1_Result\trainingResult\TD7\seed2021\Expert_HRL_TD7_HardQ2021",
    # ],
    # "Expert-HRL-TD7-SoftQ": [
    #     r"1_Result\trainingResult\TD7\seed2025\Expert_HRL_TD7_SoftQ2025",
    #     r"1_Result\trainingResult\TD7\seed2024\Expert_HRL_TD7_SoftQ2024",
    #     r"1_Result\trainingResult\TD7\seed2023\Expert_HRL_TD7_SoftQ2023",
    #     r"1_Result\trainingResult\TD7\seed2022\Expert_HRL_TD7_SoftQ2022",
    #     r"1_Result\trainingResult\TD7\seed2021\Expert_HRL_TD7_SoftQ2021",
    # ],
    ######################## TD7-HardQ VS TD7-SoftQ END ########################

    ########################  TD7-SoftQ-taudiff ########################
        "Expert-HRL-TD7-HardQ": [
        r"1_Result\trainingResult\TD7\seed2023\Expert_HRL_TD7_HardQ2023",
        r"1_Result\trainingResult\TD7\seed2022\Expert_HRL_TD7_HardQ2022",
        r"1_Result\trainingResult\TD7\seed2021\Expert_HRL_TD7_HardQ2021",
    ],
        "Expert-HRL-TD7-SoftQ(tau=0.001)": [
        r"expert_hrl_td7_0429_seed2021_softQ0.001\runs\Expert_HRL_TD7_20260429_231737",
        r"expert_hrl_td7_0429_seed2022_softQ0.001\runs\Expert_HRL_TD7_20260429_231652",
        r"expert_hrl_td7_0429_seed2023_softQ0.001\runs\Expert_HRL_TD7_20260429_231502",
    ],
    #     "Expert-HRL-TD7-SoftQ(tau=0.1)": [
    #     r"expert_hrl_td7_0427_seed2021_softQ0.1\runs\Expert_HRL_TD7_20260428_182840",
    #     r"expert_hrl_td7_0429_seed2022_softQ0.1\runs\Expert_HRL_TD7_20260429_092922",
    #     r"expert_hrl_td7_0429_seed2023_softQ0.1\runs\Expert_HRL_TD7_20260429_093437",
    # ],
    #     "Expert-HRL-TD7-SoftQ(tau=0.3)": [
    #     r"expert_hrl_td7_0427_seed2021_softQ0.3\runs\Expert_HRL_TD7_20260428_184046",
    #     r"expert_hrl_td7_0429_seed2022_softQ0.3\runs\Expert_HRL_TD7_20260429_093032",
    #     r"expert_hrl_td7_0429_seed2023_softQ0.3\runs\Expert_HRL_TD7_20260429_093548",
        
    # ],

        "Expert-HRL-TD7-SoftQ(tau=0.5)": [
        r"1_Result\trainingResult\TD7\seed2023\Expert_HRL_TD7_SoftQ2023",
        r"1_Result\trainingResult\TD7\seed2022\Expert_HRL_TD7_SoftQ2022",
        r"1_Result\trainingResult\TD7\seed2021\Expert_HRL_TD7_SoftQ2021",
    ],

    #     "Expert-HRL-TD7-SoftQ(tau=0.7)": [
    #     r"expert_hrl_td7_0427_seed2021_softQ0.7\runs\Expert_HRL_TD7_20260428_184127",
    #     r"expert_hrl_td7_0429_seed2022_softQ0.7\runs\Expert_HRL_TD7_20260429_093109",
    #     r"expert_hrl_td7_0429_seed2023_softQ0.7\runs\Expert_HRL_TD7_20260429_093625",

    # ],

    #     "Expert-HRL-TD7-SoftQ(tau=1.0)": [
    #     r"expert_hrl_td7_0427_seed2021_softQ1.0\runs\Expert_HRL_TD7_20260428_183017",
    #     r"expert_hrl_td7_0429_seed2022_softQ1.0\runs\Expert_HRL_TD7_20260429_093157",
    #     r"expert_hrl_td7_0429_seed2023_softQ1.0\runs\Expert_HRL_TD7_20260429_093722",

    # ],

        "Expert-HRL-TD7-SoftQ(tau=5.0)": [
        r"expert_hrl_td7_0429_seed2021_softQ5.0\runs\Expert_HRL_TD7_20260429_231940",
        r"expert_hrl_td7_0429_seed2022_softQ5.0\runs\Expert_HRL_TD7_20260429_232007",
        r"expert_hrl_td7_0429_seed2023_softQ5.0\runs\Expert_HRL_TD7_20260429_232047",

    ],
    ########################  TD7-SoftQ-taudiff END########################

    ######################## TD3-HardQ VS TD3-SoftQ ########################
    # "Expert-HRL-TD3-HardQ": [
    #     r"1_Result\trainingResult\TD3\seed2021\Expert_HRL_TD3_HardQ2021",
    #     r"1_Result\trainingResult\TD3\seed2022\Expert_HRL_TD3_HardQ2022",
    #     r"1_Result\trainingResult\TD3\seed2025\Expert_HRL_TD3_HardQ2025",
    #     r"1_Result\trainingResult\TD3\seed2024\Expert_HRL_TD3_HardQ2024",
    #     r"1_Result\trainingResult\TD3\seed2023\Expert_HRL_TD3_HardQ2023",
    # ],
    # "Expert-HRL-TD3-SoftQ": [
    #     r"1_Result\trainingResult\TD3\seed2025\Expert_HRL_TD3_SoftQ2025",
    #     r"1_Result\trainingResult\TD3\seed2024\Expert_HRL_TD3_SoftQ2024",
    #     r"1_Result\trainingResult\TD3\seed2023\Expert_HRL_TD3_SoftQ2023",
    #     r"1_Result\trainingResult\TD3\seed2022\Expert_HRL_TD3_SoftQ2022",
    #     r"1_Result\trainingResult\TD3\seed2021\Expert_HRL_TD3_SoftQ2021",
    # ],
    ######################## TD3-HardQ VS TD3-SoftQ END ########################

    ######################## SAC-HardQ VS SAC-SoftQ ########################
    # "Expert-HRL-SAC-HardQ": [
    #     r"expert_hrl_sac_0426_seed2021_hardQ\runs\Expert_HRL_SAC_20260502_164334",
    #     r"1_Result\trainingResult\SAC\seed2022\Expert_HRL_SAC_HardQ2022",
    #     r"1_Result\trainingResult\SAC\seed2023\Expert_HRL_SAC_HardQ2023",
    #     r"1_Result\trainingResult\SAC\seed2024\Expert_HRL_SAC_HardQ2024",
    #     r"1_Result\trainingResult\SAC\seed2025\Expert_HRL_SAC_HardQ2025",
    # ],
    # "Expert-HRL-SAC-SoftQ": [
    #     r"expert_hrl_sac_0426_seed2021_softQ\runs\Expert_HRL_SAC_20260502_164223",
    #     r"1_Result\trainingResult\SAC\seed2025\Expert_HRL_SAC_SoftQ2025",
    #     r"1_Result\trainingResult\SAC\seed2024\Expert_HRL_SAC_SoftQ2024",
    #     r"1_Result\trainingResult\SAC\seed2023\Expert_HRL_SAC_SoftQ2023",
    #     r"1_Result\trainingResult\SAC\seed2022\Expert_HRL_SAC_SoftQ2022",
    # ],
    ########################### SAC-HardQ VS SAC-SoftQ END ########################

    ######################## SAC-expert VS SAC-noexpert ########################
    #     "Expert-HRL-SAC": [
    #     r"1_Result\trainingResult\SAC\seed2025\Expert_HRL_SAC_SoftQ2025",
    #     r"1_Result\trainingResult\SAC\seed2024\Expert_HRL_SAC_SoftQ2024",
    #     r"1_Result\trainingResult\SAC\seed2023\Expert_HRL_SAC_SoftQ2023",
    # ],
    #     "NoExpert-HRL-SAC": [
    #     r"expert_hrl_sac_430_seed2024_noexpert\runs\Expert_HRL_SAC_20260430_005355",
    #     r"expert_hrl_sac_430_seed2025_noexpert\runs\Expert_HRL_SAC_20260430_004442",
    #     r"expert_hrl_sac_430_seed2026_noexpert\runs\Expert_HRL_SAC_20260430_004343",
    # ],
    ######################## SAC-expert VS SAC-noexpert END########################

    ########################## TD3-expert VS TD3-noexpert ########################
    #     "Expert-HRL-TD3": [
    #     r"1_Result\trainingResult\TD3\seed2026\Expert_HRL_TD3_SoftQ2026",
    #     r"1_Result\trainingResult\TD3\seed2025\Expert_HRL_TD3_SoftQ2025",
    #     r"1_Result\trainingResult\TD3\seed2024\Expert_HRL_TD3_SoftQ2024",
    # ],
    #     "NoExpert-HRL-TD3": [
    #     r"expert_hrl_td3_430_seed2024_noexpert\runs\Expert_HRL_TD3_20260430_101843",
    #     r"expert_hrl_td3_430_seed2025_noexpert\runs\Expert_HRL_TD3_20260501_135422",
    #     r"expert_hrl_td3_430_seed2026_noexpert\runs\Expert_HRL_TD3_20260501_140702",
    # ],
    ########################## TD3-expert VS TD3-noexpert END ########################

    ############################ TD7-expert VS TD7-noexpert ########################
    #     "Expert-HRL-TD7": [
    #     r"1_Result\trainingResult\TD7\seed2026\Expert_HRL_TD7_SoftQ2026",
    #     r"1_Result\trainingResult\TD7\seed2025\Expert_HRL_TD7_SoftQ2025",
    #     r"1_Result\trainingResult\TD7\seed2024\Expert_HRL_TD7_SoftQ2024",
    # ],
    #     "NoExpert-HRL-TD7": [
    #     r"expert_hrl_td7_430_seed2024_noexpert\runs\Expert_HRL_TD7_20260430_101649",
    #     r"expert_hrl_td7_430_seed2025_noexpert\runs\Expert_HRL_TD7_20260501_135345",
    #     r"expert_hrl_td7_430_seed2026_noexpert\runs\Expert_HRL_TD7_20260501_140625",
    # ],
    ############################ TD7-expert VS TD7-noexpert END ########################

}

# 要绘制的 tag（TensorBoard 中的标量名）
TAGS = [
    # "Reward/Actuator",
    # "Reward/Selector",
    # "Metrics/Collision", 
    # "Metrics/SuccessRate",
    # ---- Soft vs Hard Q-filter 对比分析 ----
    # "Training/BC_AvgSoftWeight",   # Soft: sigmoid 平均权重; Hard: 等于 hit rate
    # "Training/BC_HitRate",         # Q_expert > Q_policy 的比例 (BC 信号通过率)
    
]

# 平滑方法: "uniform" 均匀滑动窗口, "ema" 指数移动平均 (与 TensorBoard 一致), "both" 同时绘制两者对比
SMOOTH_METHOD = "ema"  # "uniform" | "ema" | "both"
# 平滑窗口大小（设为 1 则不平滑；对 EMA 则通过 alpha = 2/(window+1) 近似等效）
SMOOTH_WINDOW = 50
# EMA 的 alpha 参数（仅 SMOOTH_METHOD="ema" 或 "both" 时生效，设为 None 则自动用 2/(SMOOTH_WINDOW+1)）
EMA_ALPHA = None

# 输出图片路径（None 则只显示不保存）
SAVE_PATH = None

# 手动指定 training_history.json 路径 (可选, 覆盖自动查找)
# 格式: { "日志目录路径": "training_history.json 完整路径" }
HISTORY_OVERRIDE = {}

# 颜色列表（按算法顺序分配）
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
          , "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354", "#756bb1", "#020202"]
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
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    half = window // 2
    padded = np.pad(values, (half, window - half - 1), mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def smooth_ema(values: np.ndarray, window: int, alpha: float = None):
    """EMA 平滑，与 TensorBoard 指数移动平均一致。"""
    if window <= 1 or len(values) < window:
        return values
    if alpha is None:
        alpha = 2.0 / (window + 1)  # 与 uniform window 近似等效
    s = pd.Series(values)
    smoothed = s.ewm(alpha=alpha, adjust=False).mean()
    return smoothed.values


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


def _aggregate_and_draw(ax, all_x, all_values, x_axis, color, label, linestyle="-"):
    """将多个种子的数据聚合为 mean±std 并绘制。"""
    if len(all_values) == 1:
        ax.plot(all_x[0], all_values[0], linewidth=1.2, linestyle=linestyle,
                color=color, label=f"{label} (1 seed)")
    else:
        if x_axis == "global_steps":
            x_interp, mean, std = interpolate_and_aggregate(all_x, all_values)
        else:
            x_interp, mean, std = align_and_aggregate(all_x, all_values)
        ax.plot(x_interp, mean, linewidth=1.5, color=color, linestyle=linestyle,
                label=label)
        ax.fill_between(x_interp, mean - std, mean + std, alpha=0.12, color=color)


def plot_algo_compare(algorithms, tags, smooth_window=50, save_path=None,
                      data_source="tensorboard", x_axis="episodes",
                      smooth_method="uniform", ema_alpha=None):
    n_tags = len(tags)
    if n_tags <= 2:
        n_rows, n_cols = 1, max(n_tags, 1)
        figsize = (6 * n_cols, 5)
    else:
        n_cols = (n_tags + 1) // 2
        n_rows = 2
        figsize = (6 * n_cols, 10)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    x_label = "global steps" if x_axis == "global_steps" else "episodes"
    plot_both = (smooth_method == "both")

    for idx, tag in enumerate(tags):
        ax = axes[idx]

        for algo_idx, (algo_name, seed_dirs) in enumerate(algorithms.items()):
            color = COLORS[algo_idx % len(COLORS)]

            # --- 收集各 seed 的原始数据 ---
            all_x_seeds = []
            all_values_raw = []  # 原始数据，每个 seed 一份

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

                all_x_seeds.append(x)
                all_values_raw.append(values)

            if len(all_values_raw) == 0:
                continue

            # --- 根据 smooth_method 绘制 ---
            if plot_both:
                # Uniform 平滑
                all_val_u = [smooth(v.copy(), smooth_window) for v in all_values_raw]
                suffix_u = f"{algo_name} [uniform]"
                _aggregate_and_draw(ax, all_x_seeds, all_val_u, x_axis, color, suffix_u, "-")

                # EMA 平滑
                all_val_e = [smooth_ema(v, smooth_window, ema_alpha) for v in all_values_raw]
                suffix_e = f"{algo_name} [EMA]"
                _aggregate_and_draw(ax, all_x_seeds, all_val_e, x_axis, color, suffix_e, "--")
            elif smooth_method == "ema":
                all_val = [smooth_ema(v, smooth_window, ema_alpha) for v in all_values_raw]
                _aggregate_and_draw(ax, all_x_seeds, all_val, x_axis, color, algo_name, "-")
            else:
                all_val = [smooth(v, smooth_window) for v in all_values_raw]
                _aggregate_and_draw(ax, all_x_seeds, all_val, x_axis, color, algo_name, "-")

        ax.set_xlabel(x_label)
        y_label = tag.split("/")[-1] + " Reward" if tag.startswith("Reward/") else tag.split("/")[-1]
        ax.set_ylabel(y_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for idx in range(len(tags), len(axes)):
        axes[idx].set_visible(False)

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

    plot_algo_compare(ALGORITHMS, TAGS, SMOOTH_WINDOW, SAVE_PATH, DATA_SOURCE, X_AXIS,
                      SMOOTH_METHOD, EMA_ALPHA)
