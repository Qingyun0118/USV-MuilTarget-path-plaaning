"""
轨迹回放模块 — 支持单USV / 多USV 轨迹动画回放

支持两种环境重建方式:
  1. 自动模式: 从评估报告 (evaluation_report_*.txt) 解析环境参数
  2. 手动模式: 通过命令行参数指定环境参数 (--manual)

单USV回放:
    # 自动从报告读取环境参数
    python replay_trajectory.py --output_dir expert_hrl_sac_output --mode rl --episode 1 2 3

    # 手动指定环境参数
    python replay_trajectory.py --manual --num_targets 3 --num_obstacles 6 \\
        --eval_seed 42 --episode 1 --csv_dir expert_hrl_sac_output/evaluation_results/eval_rl \\
        --csv_pattern "expert_hrl_sac_path_rl_ep_{ep}.csv"

多USV回放 (CTDE):
    # 手动指定环境参数 (多USV必须指定)
    python replay_trajectory.py --multi_usv --num_usvs 2 --manual \\
        --num_targets 3 --num_obstacles 6 --eval_seed 42 --episode 1 \\
        --csv_dir ctde_mappo_output/evaluation_results \\
        --csv_prefix "ctde_ep_1"

    # 自动: 若评估时写入了报告
    python replay_trajectory.py --multi_usv --num_usvs 2 \\
        --output_dir ctde_mappo_output --mode rl --episode 1

仅海流:
    python replay_trajectory.py --currents_only --eval_seed 42 --duration 15
"""

import os
import re
import argparse

from my_USV_Env import MyUsvEnv, MultiUsvEnv


# ============================================================================
# 报告解析
# ============================================================================

def parse_report(report_path: str) -> dict:
    """
    从 evaluation_report_{mode}.txt 中解析环境参数。

    Returns:
        dict 包含: eval_seed, num_targets, num_obstacles,
                   usv_length, usv_width, target_radius,
                   obstacle_radius_min, obstacle_radius_max,
                   use_random_shapes
    """
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"报告文件不存在: {report_path}")

    text = open(report_path, 'r', encoding='utf-8').read()

    def _search(pattern, cast=str, default=None):
        """在报告中搜索最后一次出现的匹配（追加模式下取最新记录）"""
        matches = re.findall(pattern, text)
        if matches:
            return cast(matches[-1])
        return default

    params = {
        'eval_seed':          _search(r'评估种子:\s*(\d+)',         int, None),
        'num_targets':        _search(r'目标数量:\s*(\d+)',         int, 3),
        'num_obstacles':      _search(r'障碍物数量:\s*(\d+)',       int, 6),
        'usv_length':         _search(r'USV 尺寸:\s*([\d.]+)',     float, 3.0),
        'usv_width':          _search(r'USV 尺寸:\s*[\d.]+ x ([\d.]+)', float, 1.5),
        'target_radius':      _search(r'目标半径:\s*([\d.]+)',     float, 3.0),
        'obstacle_radius_min': _search(r'障碍物半径:\s*([\d.]+)',   float, 3.0),
        'obstacle_radius_max': _search(r'障碍物半径:\s*[\d.]+~([\d.]+)', float, 10.0),
        'use_random_shapes':  _search(r'随机形状:\s*(\w+)',        str, 'True'),
    }

    # 将 'True'/'False' 字符串转为 bool
    params['use_random_shapes'] = params['use_random_shapes'] in ('True', 'true', '1')

    return params


def get_env_params(args) -> dict:
    """根据 --manual 标志决定从报告解析还是从命令行获取环境参数。

    Returns:
        dict 包含环境重建所需的全部参数
    """
    if args.manual:
        # 手动模式: 从命令行参数构建
        if args.eval_seed is None:
            raise ValueError("手动模式下必须通过 --eval_seed 指定评估种子")
        params = {
            'eval_seed':           args.eval_seed,
            'num_targets':         args.num_targets,
            'num_obstacles':       args.num_obstacles,
            'usv_length':          args.usv_length,
            'usv_width':           args.usv_width,
            'target_radius':       args.target_radius,
            'obstacle_radius_min': args.obstacle_radius_min,
            'obstacle_radius_max': args.obstacle_radius_max,
            'use_random_shapes':   args.use_random_shapes,
        }
        return params
    else:
        # 自动模式: 从评估报告解析
        eval_dir = os.path.join(args.output_dir, "evaluation_results",
                                f"eval_{args.mode}")
        report_path = os.path.join(eval_dir,
                                   f"evaluation_report_{args.mode}.txt")
        params = parse_report(report_path)
        if args.eval_seed is not None:
            params['eval_seed'] = args.eval_seed
        if params['eval_seed'] is None:
            raise ValueError(
                f"无法从报告中读取评估种子，请通过 --eval_seed 手动指定\n"
                f"报告路径: {report_path}")
        return params


def _build_single_env(params, ep_seed):
    """根据参数字典创建单USV环境。"""
    env = MyUsvEnv(
        num_targets=params['num_targets'],
        num_obstacles=params['num_obstacles'],
        seed=params['eval_seed'],
        record_flag=False,
        usv_length=params['usv_length'],
        usv_width=params['usv_width'],
        target_radius=params['target_radius'],
        obstacle_radius_range=(params['obstacle_radius_min'],
                               params['obstacle_radius_max']),
        use_random_shapes=params['use_random_shapes'],
        current_type="generated",
    )
    env.reset(seed=ep_seed)
    return env


def _build_multi_env(params, num_usvs, ep_seed):
    """根据参数字典创建多USV环境。"""
    env = MultiUsvEnv(
        num_usvs=num_usvs,
        num_targets=params['num_targets'],
        num_obstacles=params['num_obstacles'],
        seed=params['eval_seed'],
        record_flag=False,
        usv_length=params['usv_length'],
        usv_width=params['usv_width'],
        target_radius=params['target_radius'],
        obstacle_radius_range=(params['obstacle_radius_min'],
                               params['obstacle_radius_max']),
        use_random_shapes=params['use_random_shapes'],
        current_type="generated",
    )
    env.reset(seed=ep_seed)
    return env


# ============================================================================
# 单USV回放
# ============================================================================

def replay(args, episode: int, params: dict):
    """
    回放单USV episode 轨迹并保存 GIF。
    """
    # 定位 CSV
    if args.csv_dir and args.csv_pattern:
        csv_file = os.path.join(
            args.csv_dir,
            args.csv_pattern.format(ep=episode, mode=args.mode))
    else:
        eval_dir = os.path.join(args.output_dir, "evaluation_results",
                                f"eval_{args.mode}")
        csv_file = os.path.join(
            eval_dir,
            f"expert_hrl_sac_path_{args.mode}_ep_{episode}.csv")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_file}")

    # GIF 保存路径
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(
            args.save_dir,
            f"replay_{args.mode}_ep_{episode}.gif")
    else:
        save_path = os.path.splitext(csv_file)[0] + ".gif"

    ep_seed = params['eval_seed'] + episode - 1
    print(f"[replay] single-USV | mode={args.mode}, episode={episode}, "
          f"ep_seed={ep_seed}")
    print(f"[replay] T={params['num_targets']}, O={params['num_obstacles']}")
    print(f"[replay] csv={csv_file}")

    env = _build_single_env(params, ep_seed)

    env.animate(
        csv_file=csv_file,
        interval=args.interval,
        save_path=save_path,
        trail_length=args.trail_length,
        speed=args.speed,
    )

    size_kb = os.path.getsize(save_path) / 1024 if os.path.exists(save_path) else 0
    print(f"[replay] GIF 已保存: {save_path}  ({size_kb:.0f} KB)")
    return save_path


# ============================================================================
# 多USV回放
# ============================================================================

def replay_multi(args, episode: int, params: dict):
    """
    回放多USV episode 轨迹并保存 GIF。

    CSV 文件命名规范: ``{csv_prefix}_usv_{i}.csv`` (与 MultiUsvEnv.record_points 一致)
    """
    csv_dir = args.csv_dir
    csv_prefix = args.csv_prefix

    if csv_dir is None:
        csv_dir = os.path.join(args.output_dir, "evaluation_results")
    if csv_prefix is None:
        csv_prefix = f"ctde_ep_{episode}"

    # 检查至少一个 USV CSV 存在
    found_any = False
    for i in range(args.num_usvs):
        fpath = os.path.join(csv_dir, f"{csv_prefix}_usv_{i}.csv")
        if os.path.exists(fpath):
            found_any = True
        else:
            print(f"[warn] 未找到 USV {i} 的 CSV: {fpath}")
    if not found_any:
        raise FileNotFoundError(
            f"在 {csv_dir} 下未找到任何 {csv_prefix}_usv_*.csv 文件")

    # GIF 保存路径
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(
            args.save_dir,
            f"replay_multi_{args.num_usvs}usv_ep_{episode}.gif")
    else:
        save_path = os.path.join(
            csv_dir,
            f"replay_multi_{args.num_usvs}usv_ep_{episode}.gif")

    ep_seed = params['eval_seed'] + episode
    print(f"[replay] multi-USV ({args.num_usvs}) | episode={episode}, "
          f"ep_seed={ep_seed}")
    print(f"[replay] T={params['num_targets']}, O={params['num_obstacles']}")
    print(f"[replay] csv_dir={csv_dir}, prefix={csv_prefix}")

    env = _build_multi_env(params, args.num_usvs, ep_seed)

    env.animate(
        csv_prefix=csv_prefix,
        output_dir=csv_dir,
        interval=args.interval,
        save_path=save_path,
        trail_length=args.trail_length,
        speed=args.speed,
    )

    size_kb = os.path.getsize(save_path) / 1024 if os.path.exists(save_path) else 0
    print(f"[replay] GIF 已保存: {save_path}  ({size_kb:.0f} KB)")
    return save_path


# ============================================================================
# 仅海流回放
# ============================================================================

def replay_currents_only(args, params: dict):
    """仅播放时变海流动画（无轨迹）。"""
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "currents_animation.gif")
    else:
        save_path = "currents_animation.gif"

    if args.multi_usv:
        env = _build_multi_env(params, args.num_usvs,
                               params['eval_seed'])
    else:
        env = _build_single_env(params, params['eval_seed'])

    env.animate(
        csv_file=None,
        duration=args.duration,
        interval=args.interval,
        save_path=save_path,
    ) if not args.multi_usv else env.animate(
        csv_prefix="__nonexistent__",
        output_dir=".",
        duration=args.duration,
        interval=args.interval,
        save_path=save_path,
    )

    size_kb = os.path.getsize(save_path) / 1024 if os.path.exists(save_path) else 0
    print(f"[replay] 海流动画已保存: {save_path}  ({size_kb:.0f} KB)")
    return save_path


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="USV 轨迹回放 — 支持单USV/多USV, 自动/手动环境重建",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单USV, 自动从报告读取环境参数
  python replay_trajectory.py --output_dir expert_hrl_sac_output --mode rl --episode 1

  # 单USV, 手动指定环境参数
  python replay_trajectory.py --manual --num_targets 3 --num_obstacles 6 \\
      --eval_seed 42 --episode 1 \\
      --csv_dir expert_hrl_sac_output/evaluation_results/eval_rl \\
      --csv_pattern "expert_hrl_sac_path_rl_ep_{ep}.csv"

  # 多USV, 手动参数
  python replay_trajectory.py --multi_usv --num_usvs 2 --manual \\
      --num_targets 3 --num_obstacles 6 --eval_seed 42 --episode 1 \\
      --csv_dir ctde_mappo_output/eval_records --csv_prefix "ctde_ep_1"
""")

    # ---- 模式选择 ----
    parser.add_argument("--multi_usv", action="store_true",
                        help="多USV模式 (使用 MultiUsvEnv)")
    parser.add_argument("--num_usvs", type=int, default=2,
                        help="USV 数量 (仅 --multi_usv 有效)")
    parser.add_argument("--currents_only", action="store_true",
                        help="仅显示时变海流动画")

    # ---- 环境重建方式 ----
    parser.add_argument("--manual", action="store_true",
                        help="手动指定环境参数 (否则从评估报告自动读取)")

    # 自动模式参数
    parser.add_argument("--output_dir", type=str, default="expert_hrl_sac_output",
                        help="训练输出目录 (自动模式)")
    parser.add_argument("--mode", type=str, default="rl",
                        choices=["expert", "rl"],
                        help="评估模式 (自动模式, 用于定位报告和 CSV)")

    # 手动模式 / 通用环境参数
    parser.add_argument("--num_targets", type=int, default=3,
                        help="目标数量")
    parser.add_argument("--num_obstacles", type=int, default=6,
                        help="障碍物数量")
    parser.add_argument("--eval_seed", type=int, default=None,
                        help="评估种子 (手动模式必填; 自动模式可覆盖报告值)")
    parser.add_argument("--usv_length", type=float, default=3.0)
    parser.add_argument("--usv_width", type=float, default=1.5)
    parser.add_argument("--target_radius", type=float, default=5.0)
    parser.add_argument("--obstacle_radius_min", type=float, default=3.0)
    parser.add_argument("--obstacle_radius_max", type=float, default=10.0)
    parser.add_argument("--use_random_shapes", action="store_true",
                        default=True,
                        help="障碍物使用随机形状")
    parser.add_argument("--no_random_shapes", dest="use_random_shapes",
                        action="store_false")

    # ---- Episode 选择 ----
    parser.add_argument("--episode", type=int, nargs='*', default=None,
                        help="要回放的 episode 编号 (可多个)")

    # ---- CSV 路径 (手动定位) ----
    parser.add_argument("--csv_dir", type=str, default=None,
                        help="CSV 文件所在目录 (手动指定)")
    parser.add_argument("--csv_pattern", type=str, default=None,
                        help="单USV CSV 文件名模板, 用 {ep} 代替 episode, "
                             "如 'path_ep_{ep}.csv'")
    parser.add_argument("--csv_prefix", type=str, default=None,
                        help="多USV CSV 文件名前缀 "
                             "(实际文件为 {prefix}_usv_{i}.csv)")

    # ---- 动画参数 ----
    parser.add_argument("--interval", type=int, default=40,
                        help="帧间隔 (毫秒)")
    parser.add_argument("--speed", type=int, default=2,
                        help="每帧前进的点数")
    parser.add_argument("--trail_length", type=int, default=0,
                        help="尾迹保留长度 (0 = 全部)")
    parser.add_argument("--duration", type=float, default=20,
                        help="仅海流模式的动画时长")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="GIF 保存目录")

    args = parser.parse_args()

    # ---- 获取环境参数 ----
    params = get_env_params(args)
    print(f"[env params] seed={params['eval_seed']}, "
          f"T={params['num_targets']}, O={params['num_obstacles']}, "
          f"usv={params['usv_length']}x{params['usv_width']}, "
          f"target_r={params['target_radius']}")

    # ---- 执行回放 ----
    if args.currents_only:
        replay_currents_only(args, params)
    elif args.episode:
        for ep in args.episode:
            if args.multi_usv:
                replay_multi(args, ep, params)
            else:
                replay(args, ep, params)
    else:
        parser.print_help()
        print("\n请指定 --episode 或 --currents_only")
