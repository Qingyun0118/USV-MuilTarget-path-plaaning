"""
箱线图绘制脚本 — 按算法和 Q-filter 类型分组对比成功率分布

从各模型目录下的 batch_eval_results.txt 中提取成功率均值，
按 (算法, Q-filter类型, 场景) 分组绘制箱线图。

用法:
    # 指定模型目录，自动检测所有场景分别出图
    python plot_boxplot.py --model_dirs expert_hrl_td7_0426_seed2022_softQ expert_hrl_td7_0426_seed2022_hardQ ...

    # 只绘制指定场景
    python plot_boxplot.py --model_dirs ... --eval_targets 3 --eval_obstacles 6

    # 自动发现当前目录下所有 expert_hrl_* 模型目录
    python plot_boxplot.py --auto-discover

    # 保存到指定目录
    python plot_boxplot.py --model_dirs ... --output_dir ./figures
"""

import argparse
import ast
import glob
import math
import os
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ============================================================================
# 默认场景列表（与 run_missing_evals.py 保持一致）
# ============================================================================
DEFAULT_SCENARIO_ORDER = [
    (3, 6), 
    # (5, 6), (7, 6), (9, 6),
    # (5, 10), (5, 14), (7, 12), (9, 12),
]


# ============================================================================
# 目录名解析
# ============================================================================

def detect_algo(dir_name):
    """从目录名检测算法类型。"""
    name = os.path.basename(os.path.normpath(dir_name)).lower()
    if 'td7' in name:
        return 'TD7'
    elif 'td3' in name:
        return 'TD3'
    elif 'sac' in name:
        return 'SAC'
    return 'Unknown'


def detect_qfilter(dir_name):
    """从目录名检测 Q-filter 类型。"""
    name = os.path.basename(os.path.normpath(dir_name)).lower()
    if 'softq' in name or 'soft' in name:
        return 'SoftQ'
    elif 'hardq' in name or 'hard' in name:
        return 'HardQ'
    return 'Unknown'


# ============================================================================
# 结果解析
# ============================================================================

def parse_batch_eval_results(model_dir):
    """
    解析单个模型目录下的 batch_eval_results.txt。

    Returns:
        list of dict: [{
            'targets': int, 'obstacles': int,
            'mean_rate': float, 'std_rate': float,
            'algo': str, 'qfilter': str,
            'model_dir': str,
        }, ...]
    """
    result_path = os.path.join(model_dir, "batch_eval_results.txt")
    if not os.path.exists(result_path):
        return []

    dir_name = os.path.basename(os.path.normpath(model_dir))
    algo = detect_algo(dir_name)
    qfilter = detect_qfilter(dir_name)

    with open(result_path, "r", encoding="utf-8") as f:
        content = f.read()

    results = []
    blocks = content.split("批量评估结果")

    for block in blocks:
        if not block.strip():
            continue

        scene_match = re.search(r"环境:\s*(\d+)T\s*/\s*(\d+)O", block)
        if not scene_match:
            continue
        targets = int(scene_match.group(1))
        obstacles = int(scene_match.group(2))

        mean_match = re.search(r"成功率均值\s*\(Mean\):\s*([\d.]+)", block)
        if not mean_match:
            continue
        mean_rate = float(mean_match.group(1))

        std_match = re.search(r"成功率均方差\s*\(Std\):\s*([\d.]+)", block)
        std_rate = float(std_match.group(1)) if std_match else 0.0

        results.append({
            'targets': targets,
            'obstacles': obstacles,
            'mean_rate': mean_rate,
            'std_rate': std_rate,
            'algo': algo,
            'qfilter': qfilter,
            'model_dir': model_dir,
        })

    return results


# ============================================================================
# 数据分组
# ============================================================================

def group_results(all_results, scenario_filter=None):
    """
    将解析结果按场景和分组标签组织。

    Args:
        all_results: parse_batch_eval_results 返回的所有结果列表的拼接
        scenario_filter: (targets, obstacles) 或 None（不过滤）

    Returns:
        dict: {(targets, obstacles): {group_label: [mean_rate, ...]}}
        list: 分组标签列表（固定顺序）
    """
    data = {}          # {(T, O): {label: [rates]}}
    all_labels = set()

    for r in all_results:
        t, o = r['targets'], r['obstacles']

        if scenario_filter is not None and (t, o) != scenario_filter:
            continue

        label = f"{r['algo']}\n{r['qfilter']}"
        all_labels.add(label)

        if (t, o) not in data:
            data[(t, o)] = {}
        if label not in data[(t, o)]:
            data[(t, o)][label] = []

        data[(t, o)][label].append(r['mean_rate'])

    # 固定分组顺序: 按算法再按 Q-filter
    algo_order = ['TD7', 'TD3', 'SAC']
    qf_order = ['HardQ', 'SoftQ']
    sorted_labels = []
    for algo in algo_order:
        for qf in qf_order:
            label = f"{algo}\n{qf}"
            if label in all_labels:
                sorted_labels.append(label)
    # 兜底: 把未匹配到的也加上
    for label in sorted(all_labels):
        if label not in sorted_labels:
            sorted_labels.append(label)

    return data, sorted_labels


# ============================================================================
# 绘图
# ============================================================================

def plot_scenario_boxplot(scenario_data, labels, targets, obstacles, output_dir=None):
    """
    为单个场景绘制箱线图。

    Args:
        scenario_data: {label: [rates]}
        labels: 分组标签顺序列表
        targets, obstacles: 场景参数
        output_dir: 保存目录，None 则只显示不保存
    """
    # 只保留有数据的标签
    active_labels = [l for l in labels if l in scenario_data and len(scenario_data[l]) > 0]
    if len(active_labels) == 0:
        print(f"  [SKIP] 场景 {targets}T{obstacles}O 无有效数据")
        return

    data_vals = [scenario_data[l] for l in active_labels]
    n_groups = len(active_labels)

    fig, ax = plt.subplots(figsize=(max(4, n_groups * 1.6), 5.5))

    # 配色: HardQ 用蓝色系，SoftQ 用橙色系（按算法深浅区分）
    colors = []
    for label in active_labels:
        algo, qf = label.split('\n')
        if 'HardQ' in qf:
            if 'TD7' in algo:
                colors.append('#2166AC')
            elif 'TD3' in algo:
                colors.append('#4393C3')
            else:
                colors.append('#92C5DE')
        else:
            if 'TD7' in algo:
                colors.append('#D6604D')
            elif 'TD3' in algo:
                colors.append('#F4A582')
            else:
                colors.append('#FDDBC7')

    bp = ax.boxplot(data_vals, patch_artist=True, widths=0.55,
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    whiskerprops={'linewidth': 1.2},
                    capprops={'linewidth': 1.2},
                    boxprops={'linewidth': 1.2},
                    flierprops={'marker': 'o', 'markerfacecolor': 'red',
                                'markersize': 5, 'alpha': 0.7})

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(active_labels, fontsize=10)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title(f'Success Rate Distribution — {targets}T{obstacles}O', fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加均值点
    for i, vals in enumerate(data_vals):
        mean_val = np.mean(vals)
        ax.plot(i + 1, mean_val, 'D', color='black', markersize=6, zorder=5)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"boxplot_{targets}T{obstacles}O.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  已保存: {path}")

    plt.show()


# ============================================================================
# 汇总文本输出
# ============================================================================

def print_summary_table(data, sorted_labels):
    """打印按场景的汇总统计表。"""
    print("\n" + "=" * 90)
    print("  各场景各分组统计")
    print("=" * 90)

    scenarios = sorted(data.keys(), key=lambda x: (x[1], x[0]))  # 按障碍数再按目标数

    for t, o in scenarios:
        print(f"\n  ── {t}T{o}O ──")
        header = f"  {'Group':<20} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}"
        print(header)
        print(f"  {'─' * 58}")
        for label in sorted_labels:
            vals = data[(t, o)].get(label, [])
            if len(vals) == 0:
                continue
            arr = np.array(vals)
            display_label = label.replace('\n', '+')
            print(f"  {display_label:<20} {len(vals):>4} {np.mean(arr):>8.4f} "
                  f"{np.std(arr, ddof=1) if len(vals) > 1 else 0.0:>8.4f} "
                  f"{np.min(arr):>8.4f} {np.max(arr):>8.4f}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="箱线图绘制 — 按算法和 Q-filter 类型对比成功率分布")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_dirs", nargs="+", default=None,
                       help="模型目录列表")
    group.add_argument("--auto-discover", action="store_true",
                       help="自动发现当前目录下所有 expert_hrl_*_seed* 目录")

    parser.add_argument("--eval_targets", type=int, default=None,
                        help="只绘制指定目标数的场景（默认自动检测所有场景）")
    parser.add_argument("--eval_obstacles", type=int, default=None,
                        help="只绘制指定障碍物数的场景")
    parser.add_argument("--output_dir", type=str, default="./1_Result",
                        help="图片保存目录 (默认: ./1_Result)")
    parser.add_argument("--no-show", action="store_true",
                        help="不弹出显示图片，仅保存")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存图片，仅显示")

    args = parser.parse_args()

    # --- 确定模型目录 ---
    if args.auto_discover:
        model_dirs = []
        for d in glob.glob("expert_hrl_*_seed*"):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "batch_eval_results.txt")):
                model_dirs.append(d)
        model_dirs.sort()
        if not model_dirs:
            print("[ERROR] 未找到包含 batch_eval_results.txt 的模型目录")
            sys.exit(1)
        print(f"[INFO] 自动发现 {len(model_dirs)} 个有评估记录的模型目录")
    else:
        model_dirs = args.model_dirs
        missing = [d for d in model_dirs if not os.path.isdir(d)]
        if missing:
            print(f"[ERROR] 以下目录不存在: {missing}")
            sys.exit(1)

    # --- 解析所有结果 ---
    all_results = []
    dirs_with_data = 0
    for md in model_dirs:
        records = parse_batch_eval_results(md)
        if records:
            dirs_with_data += 1
            all_results.extend(records)
        else:
            print(f"[WARN] 无评估记录，跳过: {os.path.basename(os.path.normpath(md))}")

    if not all_results:
        print("[ERROR] 所有目录均无有效评估记录")
        sys.exit(1)

    print(f"[INFO] 从 {dirs_with_data}/{len(model_dirs)} 个目录中共解析 "
          f"{len(all_results)} 条评估记录")

    # --- 场景过滤 ---
    if args.eval_targets is not None and args.eval_obstacles is not None:
        scenario_filter = (args.eval_targets, args.eval_obstacles)
    elif args.eval_targets is not None or args.eval_obstacles is not None:
        parser.error("--eval_targets 和 --eval_obstacles 必须同时指定")
    else:
        scenario_filter = None

    # --- 分组 ---
    data, sorted_labels = group_results(all_results, scenario_filter)

    if not data:
        print("[ERROR] 过滤后无数据")
        sys.exit(1)

    print(f"[INFO] 发现 {len(data)} 个场景, {len(sorted_labels)} 个分组:")
    for label in sorted_labels:
        print(f"       - {label.replace(chr(10), '+')}")
    for t, o in sorted(data.keys()):
        counts = {l: len(data[(t, o)].get(l, [])) for l in sorted_labels}
        print(f"       场景 {t}T{o}O: {counts}")

    # --- 打印汇总表 ---
    print_summary_table(data, sorted_labels)

    # --- 逐场景绘图 ---
    scenarios = sorted(data.keys(), key=lambda x: (x[1], x[0]))
    output_dir = None if args.no_save else args.output_dir

    for t, o in scenarios:
        print(f"\n{'─' * 60}")
        print(f"  绘制: {t}T{o}O")
        # 使用非交互式后端以支持 no-show
        if args.no_show:
            plt.switch_backend('Agg')

        plot_scenario_boxplot(
            data[(t, o)], sorted_labels, t, o,
            output_dir=output_dir,
        )

    print(f"\n[DONE] 共绘制 {len(scenarios)} 张箱线图")


if __name__ == "__main__":
    main()
