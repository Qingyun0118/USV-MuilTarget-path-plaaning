"""
泛化能力评估脚本 — 多场景 × 多训练种子方法级统计

对 TD7 hardQ 和 softQ 两种方法，在多个场景下批量评估泛化能力。
每种方法 5个训练种子 × 5个评估种子 × 100 episode/种子。
输出方法×场景的汇总表格。

用法:
    python eval_generalization.py

    # 自定义场景
    python eval_generalization.py --scenarios "5,6" "7,6" "9,6" "5,10" "5,14" "7,12" "9,12"

    # 预览（不实际执行）
    python eval_generalization.py --dry-run

    # 强制重新评估
    python eval_generalization.py --no-skip-existing
"""

import argparse
import glob
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

# ============================================================================
# 配置
# ============================================================================

DEFAULT_SCENARIOS = [
    (3, 6),
    (5, 6),
    (7, 6),
    (9, 6),
    # (5, 10),
    # (5, 14),
    # (7, 12),
    # (9, 12),
]

TRAINING_SEEDS = [2021, 2022, 2023, 2024, 2025]
METHODS = ['hardQ', 'softQ']
METHOD_LABELS = {'hardQ': 'TD7 hardQ', 'softQ': 'TD7 softQ'}

# 评估种子默认值（与 batch_eval_success_rate.py 一致）
DEFAULT_EVAL_SEEDS = [42, 123, 456, 789, 1024]

# ============================================================================
# 障碍物半径自动调整（与 batch_eval_success_rate.py 一致）
# ============================================================================

def compute_obstacle_radius_max(num_obstacles, map_size=100.0, rou_c=1.5, rou_s=1.5,
                                base_n=6, base_rmax=10.0, r_min=3.0):
    if num_obstacles <= base_n:
        return base_rmax

    base_margin = base_rmax + rou_c + rou_s + 5
    base_avail = (map_size - 2 * base_margin) ** 2
    base_avg_r = (r_min + base_rmax) / 2
    base_safe_r = base_avg_r + rou_c + rou_s
    base_total_area = base_n * math.pi * base_safe_r ** 2
    target_ratio = base_avail / base_total_area

    margin_const = rou_c + rou_s + 5
    A = map_size - 2 * margin_const
    c = r_min / 2 + rou_c + rou_s
    B = target_ratio * num_obstacles * math.pi

    sqrt_B = math.sqrt(B)
    numerator = A - sqrt_B * c
    denominator = 2 + sqrt_B / 2

    if denominator <= 0 or numerator <= 0:
        return r_min

    rmax = numerator / denominator
    return max(r_min, min(base_rmax, rmax))


# ============================================================================
# 模型目录发现
# ============================================================================

def discover_method_dirs(method):
    """为指定 method (hardQ / softQ) 按训练种子发现模型目录。"""
    dirs = []
    for seed in TRAINING_SEEDS:
        pattern = f"expert_hrl_td7_*_seed{seed}_{method}"
        matches = glob.glob(pattern)
        matches = [d for d in matches if os.path.isdir(d)]
        if matches:
            dirs.append(matches[0])
    return dirs


# ============================================================================
# 跳过检测
# ============================================================================

def method_scenario_already_done(method_dirs, targets, obstacles, eval_episodes, eval_seeds):
    """检查同方法的所有训练种子目录是否都已有该场景的评估记录。"""
    if not method_dirs:
        return True  # 没有目录就算"已处理"

    seeds_tuple = tuple(eval_seeds)

    for d in method_dirs:
        result_path = os.path.join(d, "batch_eval_results.txt")
        if not os.path.exists(result_path):
            return False

        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()

        key_pattern = rf"环境:\s*{targets}T\s*/\s*{obstacles}O"
        if not re.search(key_pattern, content):
            return False

        seeds_pattern = rf"评估种子列表:\s*\[{seeds_tuple[0]},\s*{seeds_tuple[1]},\s*{seeds_tuple[2]},\s*{seeds_tuple[3]},\s*{seeds_tuple[4]}\]"
        if not re.search(seeds_pattern, content):
            return False

        ep_pattern = rf"每种子评估回合数:\s*{eval_episodes}"
        if not re.search(ep_pattern, content):
            return False

    return True


def parse_existing_method_stats(method_dirs, targets, obstacles, metric_key='success_rate'):
    """从已有结果中解析指定指标的方法级统计。

    metric_key: 'success_rate' | 'collision_rate' | 'steps_success'
    """
    metric_patterns = {
        'success_rate': r"成功率均值\s*\(Mean\):\s*([\d.]+)",
        'collision_rate': r"碰撞率均值\s*\(Mean\):\s*([\d.]+)",
        'steps_success': r"成功步数均值\s*\(Mean\):\s*([\d.]+)",
    }
    mean_pat = metric_patterns[metric_key]

    per_dir_means = []
    for d in method_dirs:
        result_path = os.path.join(d, "batch_eval_results.txt")
        if not os.path.exists(result_path):
            return None, None
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()

        best_pos = -1
        for m in re.finditer(rf"环境:\s*{targets}T\s*/\s*{obstacles}O", content):
            best_pos = m.start()

        if best_pos < 0:
            return None, None

        block = content[best_pos:]
        mean_match = re.search(mean_pat, block)
        if not mean_match:
            return None, None
        per_dir_means.append(float(mean_match.group(1)))

    if len(per_dir_means) < 2:
        return None, None

    import numpy as np
    arr = np.array(per_dir_means)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


# ============================================================================
# 单场景评估
# ============================================================================

def run_scene_eval(method_dirs, targets, obstacles, eval_episodes, eval_seeds,
                   max_steps_per_target, dry_run=False):
    """调用 batch_eval_success_rate.py 并提取方法级 Mean ± Std。"""
    adjusted_rmax = compute_obstacle_radius_max(obstacles)

    cmd = [
        sys.executable,
        "batch_eval_success_rate.py",
        "--model_dirs",
    ]
    cmd.extend(method_dirs)
    cmd.extend([
        "--eval_targets", str(targets),
        "--eval_obstacles", str(obstacles),
        "--eval_episodes", str(eval_episodes),
        "--max_steps_per_target", str(max_steps_per_target),
        "--obstacle_radius_max", f"{adjusted_rmax:.2f}",
        "--seeds",
    ])
    cmd.extend([str(s) for s in eval_seeds])

    rmax_hint = f" (r_max={adjusted_rmax:.2f})" if adjusted_rmax != 10.0 else ""
    print(f"\n{'=' * 70}")
    print(f"[SCENE] {targets}T{obstacles}O {len(method_dirs)} 个训练种子{rmax_hint}")
    print(f"[CMD]  {' '.join(cmd)}")
    print(f"{'=' * 70}")

    if dry_run:
        print("[DRY-RUN] 跳过实际执行")
        return None, None

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=28800)
        elapsed = time.time() - start
        print(f"[DONE] 耗时 {elapsed:.1f}s，返回码: {result.returncode}")

        if result.returncode != 0:
            print(f"[STDERR] {result.stderr[-500:]}")
            return None, None

        # 解析方法级 3 个指标: 成功率、碰撞率、成功步数
        output = result.stdout

        def parse_metric(mean_re, std_re):
            m = re.search(mean_re, output)
            s = re.search(std_re, output)
            if m and s:
                return float(m.group(1)), float(s.group(1))
            return None, None

        stats = {
            'success_rate': parse_metric(
                r"成功率均值\s*\(Mean\):\s+([\d.]+)",
                r"成功率均方差\s*\(Std\):\s+([\d.]+)",
            ),
            'collision_rate': parse_metric(
                r"碰撞率均值\s*\(Mean\):\s+([\d.]+)",
                r"碰撞率均方差\s*\(Std\):\s+([\d.]+)",
            ),
            'steps_success': parse_metric(
                r"成功步数均值\s*\(Mean\):\s+([\d.]+)",
                r"成功步数均方差\s*\(Std\):\s+([\d.]+)",
            ),
        }

        if stats['success_rate'][0] is not None:
            sr_mean, sr_std = stats['success_rate']
            cr_mean, cr_std = stats['collision_rate']
            ss_mean, ss_std = stats['steps_success']
            print(f"[RESULT] SR={sr_mean:.4f}±{sr_std:.4f}  "
                  f"CR={cr_mean:.4f}±{cr_std:.4f}  "
                  f"SS={ss_mean:.1f}±{ss_std:.1f}")
            return stats

        print("[WARN] 无法从输出解析方法级统计")

        # 打印输出尾部帮助调试
        tail_lines = output.strip().split("\n")[-15:]
        print("[OUTPUT TAIL]")
        for line in tail_lines:
            print(f"  {line}")

        return None

    except subprocess.TimeoutExpired:
        print(f"[ERROR] 执行超时 (>28800s)")
        return None
    except Exception as e:
        print(f"[ERROR] 执行失败: {e}")
        return None


# ============================================================================
# 汇总表
# ============================================================================

def write_summary_table(results, scenarios, summary_path):
    """
    results: dict {(method, targets, obstacles): {'success_rate': (m,s), ...}}
    输出三个指标的汇总表格：成功率、碰撞率、成功步数。
    """
    header_col_width = 16
    col_width = 16

    def _build_table(metric_key, fmt_str):
        """构建单指标表格。"""
        lines = []
        header = f"{'方法':<{header_col_width}}"
        for t, o in scenarios:
            header += f" {f'{o}O{t}T':>{col_width}}"
        lines.append(header)
        lines.append("-" * len(header))

        for method in METHODS:
            label = METHOD_LABELS.get(method, method)
            row = f"{label:<{header_col_width}}"
            for scene in scenarios:
                key = (method, scene[0], scene[1])
                if key in results and metric_key in results[key]:
                    mean_val, std_val = results[key][metric_key]
                    cell = fmt_str.format(mean_val, std_val)
                else:
                    cell = "—"
                row += f" {cell:>{col_width}}"
            lines.append(row)
        return lines

    lines = []
    lines.append("=" * 80)
    lines.append("泛化能力评估汇总表")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"评估: 5 训练种子 × 5 评估种子 × 100 episode/种子")
    lines.append("=" * 80)
    lines.append("")

    # 成功率
    lines.append("【成功率 Success Rate】")
    lines.extend(_build_table('success_rate', "{0:.4f}±{1:.4f}"))
    lines.append("")

    # 碰撞率
    lines.append("【碰撞率 Collision Rate】")
    lines.extend(_build_table('collision_rate', "{0:.4f}±{1:.4f}"))
    lines.append("")

    # 成功步数
    lines.append("【成功步数 Steps to Success】")
    lines.extend(_build_table('steps_success', "{0:.0f}±{1:.0f}"))
    lines.append("")
    lines.append("=" * 80)

    summary_text = "\n".join(lines)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n[SUMMARY] 汇总表已保存至: {summary_path} (已覆盖)")
    return summary_text


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="泛化能力评估 — 多场景 × 多训练种子方法级统计"
    )

    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="自定义场景列表，格式: '5,6' '7,6' '9,6'")
    parser.add_argument("--eval_episodes", type=int, default=100,
                        help="每个评估种子的 episode 数 (默认: 100)")
    parser.add_argument("--max_steps_per_target", type=int, default=1000,
                        help="每个目标的最大导航步数 (默认: 1000)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="自定义评估种子列表，默认 [42, 123, 456, 789, 1024]")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="即使已有评估记录，也强制重新评估")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印要执行的命令，不实际运行")
    parser.add_argument("--summary", type=str, default="generalization_summary.txt",
                        help="汇总表文件名 (默认: generalization_summary.txt)")

    args = parser.parse_args()

    # --- 场景列表 ---
    if args.scenarios:
        scenarios = []
        for s in args.scenarios:
            parts = s.split(",")
            if len(parts) != 2:
                print(f"[ERROR] 场景格式错误: '{s}'，应为 '目标数,障碍物数'")
                sys.exit(1)
            scenarios.append((int(parts[0]), int(parts[1])))
    else:
        scenarios = DEFAULT_SCENARIOS

    print(f"[INFO] 场景列表 ({len(scenarios)} 个):")
    for t, o in scenarios:
        print(f"       {o}O{t}T")
    print()

    # --- 评估种子 ---
    eval_seeds = args.seeds if args.seeds is not None else DEFAULT_EVAL_SEEDS
    print(f"[INFO] 评估种子: {eval_seeds}")
    print(f"[INFO] 每种子 {args.eval_episodes} episode")
    print(f"[INFO] 总评估: {len(eval_seeds)} 种子 × {args.eval_episodes} ep = {len(eval_seeds) * args.eval_episodes} ep/场景/训练种子")
    print()

    # --- 发现模型目录 ---
    print(f"[INFO] 目标训练种子: {TRAINING_SEEDS}")
    method_dirs_map = {}
    for method in METHODS:
        dirs = discover_method_dirs(method)
        method_dirs_map[method] = dirs
        if len(dirs) < len(TRAINING_SEEDS):
            found_seeds = set()
            for d in dirs:
                m = re.search(r'seed(\d{4})', d)
                if m:
                    found_seeds.add(int(m.group(1)))
            missing = set(TRAINING_SEEDS) - found_seeds
            print(f"[WARN] {METHOD_LABELS[method]}: 缺少种子 {sorted(missing)} 的模型目录")
        print(f"[INFO] {METHOD_LABELS[method]}: {len(dirs)} 个目录")
        for d in dirs:
            print(f"       {d}")
    print()

    # --- 主评估循环 ---
    total = len(scenarios) * len(METHODS)
    completed = 0
    skipped = 0
    failed = 0
    results = {}  # {(method, targets, obstacles): {'success_rate': (m,s), 'collision_rate': (m,s), 'steps_success': (m,s)}}

    print(f"{'=' * 70}")
    print(f"[START] 总任务数: {len(scenarios)} 场景 × {len(METHODS)} 方法 = {total}")
    print(f"[START] 跳过已有: {'否' if args.no_skip_existing else '是'}")
    print(f"{'=' * 70}")

    for t, o in scenarios:
        for method in METHODS:
            dirs = method_dirs_map[method]
            if not dirs:
                print(f"\n[SKIP] {METHOD_LABELS[method]} {o}O{t}T — 无模型目录")
                failed += 1
                continue

            scene_label = f"{METHOD_LABELS[method]} {o}O{t}T"

            # 跳过检测
            if not args.no_skip_existing and method_scenario_already_done(
                    dirs, t, o, args.eval_episodes, eval_seeds):
                print(f"\n[SKIP] {scene_label} — 所有种子已有评估记录，跳过")
                stats = {}
                for key in ['success_rate', 'collision_rate', 'steps_success']:
                    m, s = parse_existing_method_stats(dirs, t, o, metric_key=key)
                    if m is not None:
                        stats[key] = (m, s)
                if stats.get('success_rate'):
                    results[(method, t, o)] = stats
                    sr = stats['success_rate']
                    print(f"       (复用) SR={sr[0]:.4f}±{sr[1]:.4f}")
                skipped += 1
                continue

            stats = run_scene_eval(
                dirs, t, o,
                args.eval_episodes, eval_seeds,
                args.max_steps_per_target,
                dry_run=args.dry_run,
            )

            if stats is not None and stats.get('success_rate', (None, None))[0] is not None:
                results[(method, t, o)] = stats
                sr = stats['success_rate']
                print(f"[RESULT] {scene_label} SR={sr[0]:.4f}±{sr[1]:.4f}")
                completed += 1
            else:
                failed += 1
                if not args.dry_run:
                    # 失败时尝试从已有文件恢复
                    print(f"[INFO] 尝试从已有文件恢复 {scene_label} 的结果...")
                    stats = {}
                    for key in ['success_rate', 'collision_rate', 'steps_success']:
                        m, s = parse_existing_method_stats(dirs, t, o, metric_key=key)
                        if m is not None:
                            stats[key] = (m, s)
                    if stats.get('success_rate'):
                        results[(method, t, o)] = stats
                        sr = stats['success_rate']
                        print(f"[RECOVER] 从文件恢复 SR={sr[0]:.4f}±{sr[1]:.4f}")

    # --- 结束汇总 ---
    print(f"\n{'=' * 70}")
    print(f"[FINISH] 泛化能力评估完成")
    print(f"[FINISH] 成功: {completed} | 跳过: {skipped} | 失败: {failed} | 总计: {total}")
    print(f"{'=' * 70}")

    # 生成汇总表格
    if not args.dry_run and results:
        summary = write_summary_table(results, scenarios, args.summary)
        print("\n" + summary)
    elif args.dry_run:
        print("\n[DRY-RUN] 未生成汇总表（dry-run 模式）")
    else:
        print("\n[WARN] 没有可用的评估结果，未生成汇总表")


if __name__ == "__main__":
    main()
