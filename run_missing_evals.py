"""
批量评估缺失场景脚本

自动对指定模型目录批量运行多场景评估，支持跳过已完成的组合，
最后输出统一汇总表格。

用法:
    # 自动发现所有 expert_hrl_* 模型目录，跑默认场景列表
    python run_missing_evals.py --auto-discover

    # 指定特定模型目录
    python run_missing_evals.py --model_dirs expert_hrl_td7_0426_seed2022_softQ expert_hrl_sac_0426_seed2022_softQ

    # 自定义场景和评估回合数
    python run_missing_evals.py --auto-discover --scenarios 5,10 7,12 9,12 9,6 5,14 --eval_episodes 100

    # 强制重新跑（不跳过已有结果）
    python run_missing_evals.py --auto-discover --no-skip-existing
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
# 默认场景配置（目标数, 障碍物数）
# ============================================================================
DEFAULT_SCENARIOS = [
    (3, 6),   # 基准训练场景
    (5, 6),   # 固定障碍，增目标
    (7, 6),   # 固定障碍，增目标
    (9, 6),   # 固定障碍，目标极限（建议补充）
    (5, 10),  # 固定目标，增障碍
    (5, 14),  # 固定目标，障碍极限（建议补充）
    (7, 12),  # 联合缩放
    (9, 12),  # 联合缩放
]

# ============================================================================
# 工具函数
# ============================================================================

def compute_obstacle_radius_max(num_obstacles, map_size=100.0, rou_c=1.5, rou_s=1.5,
                                base_n=6, base_rmax=10.0, r_min=3.0):
    """
    根据障碍物数量自动调整最大半径，保持空间密度大致恒定。
    与 batch_eval_success_rate.py 中的 auto_obstacle_radius_max 保持一致。
    """
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


def discover_model_dirs():
    """自动发现当前目录下所有 expert_hrl_* 模型目录。"""
    dirs = []
    for d in glob.glob("expert_hrl_*_seed*"):
        if os.path.isdir(d):
            # 检查目录内是否有模型文件（至少要有 selector 或 actor）
            has_model = any(
                os.path.exists(os.path.join(d, f))
                for f in ["selector.pth", "best_selector.pth", "actuator_actor.pth", "best_actuator_actor.pth"]
            )
            if has_model:
                dirs.append(d)
    dirs.sort()
    return dirs


def parse_results_txt(model_dir):
    """
    读取模型目录下的 batch_eval_results.txt，提取已经评估过的场景。
    返回: set of (targets, obstacles) tuples
    """
    result_path = os.path.join(model_dir, "batch_eval_results.txt")
    evaluated = set()
    if not os.path.exists(result_path):
        return evaluated

    with open(result_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 匹配 "环境: 3T / 6O" 或 "环境: 3 目标 / 6 障碍物" 等格式
    patterns = [
        r"环境:\s*(\d+)T\s*/\s*(\d+)O",
        r"环境:\s*(\d+)\s*目标\s*/\s*(\d+)\s*障碍",
        r"环境:\s*(\d+)T\s*/\s*(\d+)T",  # 兼容可能的笔误
    ]
    for pat in patterns:
        for m in re.finditer(pat, content):
            t, o = int(m.group(1)), int(m.group(2))
            evaluated.add((t, o))

    return evaluated


def run_eval(model_dir, targets, obstacles, eval_episodes, max_steps_per_target, dry_run=False):
    """
    调用 batch_eval_success_rate.py 执行单个场景评估。
    """
    # 根据障碍物数量自动调整最大半径
    adjusted_rmax = compute_obstacle_radius_max(obstacles)

    cmd = [
        sys.executable,
        "batch_eval_success_rate.py",
        "--model_dir", model_dir,
        "--eval_targets", str(targets),
        "--eval_obstacles", str(obstacles),
        "--eval_episodes", str(eval_episodes),
        "--max_steps_per_target", str(max_steps_per_target),
        "--obstacle_radius_max", f"{adjusted_rmax:.2f}",
    ]

    cmd_str = " ".join(cmd)
    rmax_hint = f" (r_max={adjusted_rmax:.2f})" if adjusted_rmax != 10.0 else ""
    print(f"\n{'='*70}")
    print(f"[RUN] {model_dir} | Scene: {targets}T{obstacles}O{rmax_hint}")
    print(f"[CMD] {cmd_str}")
    print(f"{'='*70}")

    if dry_run:
        print("[DRY-RUN] 跳过实际执行")
        return True

    start = time.time()
    try:
        ret = subprocess.run(cmd, check=False)
        elapsed = time.time() - start
        print(f"[DONE] 耗时 {elapsed:.1f}s，返回码: {ret.returncode}")
        return ret.returncode == 0
    except Exception as e:
        print(f"[ERROR] 执行失败: {e}")
        return False


def generate_summary(model_dirs, scenarios, summary_path):
    """
    从各模型的 batch_eval_results.txt 中提取数据，生成汇总表格。
    """
    # 结构: { (targets, obstacles): { model_dir: (mean, std) } }
    data = defaultdict(dict)

    for model_dir in model_dirs:
        result_path = os.path.join(model_dir, "batch_eval_results.txt")
        if not os.path.exists(result_path):
            continue

        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按评估块分割（每个块以 "批量评估结果" 开头）
        blocks = content.split("批量评估结果")

        for block in blocks:
            if not block.strip():
                continue

            # 提取场景
            scene_match = re.search(r"环境:\s*(\d+)T?\s*/\s*(\d+)O?", block)
            if not scene_match:
                continue
            t, o = int(scene_match.group(1)), int(scene_match.group(2))

            # 提取 Mean ± Std
            mean_std_match = re.search(r"Mean\s*±\s*Std:\s*([\d.]+)\s*±\s*([\d.]+)", block)
            if mean_std_match:
                mean_val = float(mean_std_match.group(1))
                std_val = float(mean_std_match.group(2))
                data[(t, o)][model_dir] = (mean_val, std_val)

    # 生成 Markdown 表格
    lines = []
    lines.append("# 批量评估汇总表\n")
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"> 评估回合数: 见各模型目录下 batch_eval_results.txt\n\n")

    # 表头
    header = "| 算法 \\ 场景 |"
    separator = "|:---|"
    for t, o in scenarios:
        header += f" {t}T{o}O |"
        separator += ":---:|"

    lines.append(header)
    lines.append(separator)

    # 按算法分组排序
    for model_dir in sorted(model_dirs):
        row = f"| {model_dir} |"
        for scene in scenarios:
            if scene in data and model_dir in data[scene]:
                mean_val, std_val = data[scene][model_dir]
                row += f" {mean_val:.4f}±{std_val:.4f} |"
            else:
                row += " — |"
        lines.append(row)

    summary_text = "\n".join(lines)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n[SUMMARY] 汇总表已保存至: {summary_path}")
    return summary_text


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="批量评估缺失场景 — 多模型 × 多场景自动扫描与执行"
    )

    # 模型目录选择
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_dirs", nargs="+", default=None,
                       help="指定一个或多个模型目录")
    group.add_argument("--auto-discover", action="store_true",
                       help="自动发现当前目录下所有 expert_hrl_* 模型目录")

    # 场景配置
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="自定义场景列表，格式: '5,10' '7,12' '9,12'（默认使用预设列表）")
    parser.add_argument("--eval_episodes", type=int, default=100,
                        help="每个种子的评估 episode 数 (默认: 100)")
    parser.add_argument("--max_steps_per_target", type=int, default=1000,
                        help="每个目标的最大导航步数 (默认: 1000)")

    # 执行控制
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="默认会跳过该模型已有结果的的场景，加此参数强制重新跑")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印要执行的命令，不实际运行")
    parser.add_argument("--summary", type=str, default="eval_summary_table.md",
                        help="最后生成的汇总表文件名 (默认: eval_summary_table.md)")

    args = parser.parse_args()

    # --- 解析模型目录 ---
    if args.auto_discover:
        model_dirs = discover_model_dirs()
        if not model_dirs:
            print("[ERROR] 未找到任何 expert_hrl_* 模型目录，请检查当前路径。")
            sys.exit(1)
        print(f"[INFO] 自动发现 {len(model_dirs)} 个模型目录:")
        for d in model_dirs:
            print(f"       - {d}")
    else:
        model_dirs = args.model_dirs
        # 过滤不存在的目录
        missing = [d for d in model_dirs if not os.path.isdir(d)]
        if missing:
            print(f"[ERROR] 以下目录不存在: {missing}")
            sys.exit(1)
        print(f"[INFO] 指定 {len(model_dirs)} 个模型目录")

    # --- 解析场景列表 ---
    if args.scenarios:
        scenarios = []
        for s in args.scenarios:
            try:
                t, o = map(int, s.split(","))
                scenarios.append((t, o))
            except ValueError:
                print(f"[ERROR] 场景格式错误: '{s}'，应为 '目标数,障碍物数'（如 5,10）")
                sys.exit(1)
    else:
        scenarios = DEFAULT_SCENARIOS

    print(f"[INFO] 评估场景列表 ({len(scenarios)} 个):")
    for t, o in scenarios:
        print(f"       - {t}T{o}O")

    # --- 主循环 ---
    skip_existing = not args.no_skip_existing
    total_tasks = len(model_dirs) * len(scenarios)
    completed = 0
    skipped = 0
    failed = 0

    print(f"\n{'='*70}")
    print(f"[START] 总任务数: {len(model_dirs)} 模型 × {len(scenarios)} 场景 = {total_tasks}")
    print(f"[START] 评估回合: {args.eval_episodes} / 场景")
    print(f"[START] 跳过已有: {'是' if skip_existing else '否'}")
    print(f"{'='*70}\n")

    for idx, model_dir in enumerate(model_dirs, 1):
        print(f"\n[{idx}/{len(model_dirs)}] 处理模型: {model_dir}")

        # 读取该模型已评估过的场景
        existing = parse_results_txt(model_dir) if skip_existing else set()
        if existing:
            print(f"       已发现结果: {', '.join(f'{t}T{o}O' for t, o in sorted(existing))}")

        for t, o in scenarios:
            scene_key = f"{t}T{o}O"
            if (t, o) in existing:
                print(f"       [SKIP] {scene_key} — 结果已存在")
                skipped += 1
                continue

            success = run_eval(
                model_dir, t, o,
                args.eval_episodes,
                args.max_steps_per_target,
                dry_run=args.dry_run,
            )
            if success:
                completed += 1
            else:
                failed += 1

    # --- 结束汇总 ---
    print(f"\n{'='*70}")
    print(f"[FINISH] 批量评估完成")
    print(f"[FINISH] 成功: {completed} | 跳过: {skipped} | 失败: {failed} | 总计: {total_tasks}")
    print(f"{'='*70}")

    # 生成汇总表格
    if not args.dry_run:
        summary = generate_summary(model_dirs, scenarios, args.summary)
        print("\n" + summary)


if __name__ == "__main__":
    main()
