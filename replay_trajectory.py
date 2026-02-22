"""
轨迹回放模块 — 根据评估报告自动重建环境并生成动态 GIF

用法:
    # 回放 mpc_hrl_output 目录下 MPC 模式 episode 3 的轨迹
    python replay_trajectory.py --output_dir mpc_hrl_output --mode mpc --episode 3

    # 回放 RL 模式 episode 1~5, 加速 10 倍
    python replay_trajectory.py --output_dir mpc_hrl_output --mode rl --episode 1 2 3 4 5 --speed 10

    # 仅显示海流动画（不传 --episode）
    python replay_trajectory.py --output_dir mpc_hrl_output --mode mpc --currents_only --duration 15

    # 自定义覆盖 seed（不从报告读取）
    python replay_trajectory.py --output_dir mpc_hrl_output --mode mpc --episode 3 --eval_seed 42
"""

import os
import re
import argparse

from my_USV_Env import MyUsvEnv


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


# ============================================================================
# 核心回放函数
# ============================================================================

def replay(output_dir: str, mode: str, episode: int, *,
           eval_seed: int = None,
           interval: int = 40,
           speed: int = 5,
           trail_length: int = 0,
           save_path: str = None):
    """
    回放指定 episode 的运动轨迹并保存 GIF。

    Args:
        output_dir:   训练输出目录 (如 'mpc_hrl_output')
        mode:         评估模式 ('mpc' 或 'rl')
        episode:      episode 编号 (1-based)
        eval_seed:    评估种子 (若为 None 则从报告自动读取)
        interval:     帧间隔 (ms)
        speed:        每帧跳跃点数
        trail_length: 尾迹保留长度 (0 = 全部)
        save_path:    GIF 保存路径 (None 则自动放在 CSV 同目录)
    """
    eval_dir = os.path.join(output_dir, "evaluation_results", f"eval_{mode}")
    report_path = os.path.join(eval_dir, f"evaluation_report_{mode}.txt")

    # 1. 解析报告获取环境参数
    params = parse_report(report_path)
    if eval_seed is not None:
        params['eval_seed'] = eval_seed

    if params['eval_seed'] is None:
        raise ValueError(
            f"无法从报告中读取评估种子，请通过 --eval_seed 手动指定\n"
            f"报告路径: {report_path}")

    # 2. 定位 CSV 文件
    csv_file = os.path.join(eval_dir, f"mpc_hrl_path_{mode}_ep_{episode}.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_file}")

    # 3. 确定 GIF 保存路径
    if save_path is None:
        save_path = os.path.join(eval_dir,
                                 f"mpc_hrl_path_{mode}_ep_{episode}.gif")

    # 4. 重建环境 (与评估时完全一致)
    ep_seed = params['eval_seed'] + episode - 1
    print(f"[replay] mode={mode}, episode={episode}, "
          f"eval_seed={params['eval_seed']}, ep_seed={ep_seed}")
    print(f"[replay] targets={params['num_targets']}, "
          f"obstacles={params['num_obstacles']}")
    print(f"[replay] csv={csv_file}")

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
        current_type="generated"
    )
    env.reset(seed=ep_seed)

    # 5. 生成动画
    env.animate(
        csv_file=csv_file,
        interval=interval,
        save_path=save_path,
        trail_length=trail_length,
        speed=speed,
    )

    size_kb = os.path.getsize(save_path) / 1024 if os.path.exists(save_path) else 0
    print(f"[replay] GIF 已保存: {save_path}  ({size_kb:.0f} KB)")
    return save_path


def replay_currents_only(output_dir: str, mode: str, *,
                         eval_seed: int = None,
                         duration: float = 20,
                         interval: int = 100,
                         save_path: str = None):
    """
    仅播放时变海流动画（无轨迹）。

    Args:
        output_dir:   训练输出目录
        mode:         评估模式 (用于定位报告)
        eval_seed:    评估种子 (若为 None 则从报告读取)
        duration:     动画时长（仿真时间单位）
        interval:     帧间隔 (ms)
        save_path:    GIF 保存路径
    """
    eval_dir = os.path.join(output_dir, "evaluation_results", f"eval_{mode}")
    report_path = os.path.join(eval_dir, f"evaluation_report_{mode}.txt")

    params = parse_report(report_path)
    if eval_seed is not None:
        params['eval_seed'] = eval_seed
    if params['eval_seed'] is None:
        raise ValueError("无法从报告中读取评估种子，请通过 --eval_seed 手动指定")

    if save_path is None:
        save_path = os.path.join(eval_dir, "currents_animation.gif")

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
        current_type="generated"
    )
    env.reset(seed=params['eval_seed'])

    env.animate(
        csv_file=None,
        duration=duration,
        interval=interval,
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
        description="USV 轨迹回放 — 从评估报告自动重建环境并生成 GIF 动画")

    parser.add_argument("--output_dir", type=str, default="mpc_hrl_output",
                        help="训练输出目录 (如 mpc_hrl_output)")
    parser.add_argument("--mode", type=str, default="mpc",
                        choices=["mpc", "rl"],
                        help="评估模式: mpc 或 rl")
    parser.add_argument("--episode", type=int, nargs='*', default=None,
                        help="要回放的 episode 编号 (可多个, 如 1 2 3)")

    # 绘制参数
    parser.add_argument("--interval", type=int, default=40,
                        help="帧间隔 (毫秒), 默认 40ms ≈ 25fps")
    parser.add_argument("--speed", type=int, default=5,
                        help="每帧前进的点数 (加速回放)")
    parser.add_argument("--trail_length", type=int, default=0,
                        help="尾迹保留长度 (0 = 保留全部)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="GIF 保存目录 (默认与 CSV 同目录)")

    # 覆盖参数
    parser.add_argument("--eval_seed", type=int, default=None,
                        help="手动指定评估种子 (覆盖报告中的值)")

    # 仅海流模式
    parser.add_argument("--currents_only", action="store_true",
                        help="仅显示时变海流动画 (不需要 --episode)")
    parser.add_argument("--duration", type=float, default=20,
                        help="海流动画时长 (仅 --currents_only 有效)")

    args = parser.parse_args()

    if args.currents_only:
        save = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save = os.path.join(args.save_dir, "currents_animation.gif")
        replay_currents_only(
            output_dir=args.output_dir,
            mode=args.mode,
            eval_seed=args.eval_seed,
            duration=args.duration,
            interval=args.interval,
            save_path=save,
        )
    elif args.episode:
        for ep in args.episode:
            save = None
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                save = os.path.join(args.save_dir,
                                    f"mpc_hrl_path_{args.mode}_ep_{ep}.gif")
            replay(
                output_dir=args.output_dir,
                mode=args.mode,
                episode=ep,
                eval_seed=args.eval_seed,
                interval=args.interval,
                speed=args.speed,
                trail_length=args.trail_length,
                save_path=save,
            )
    else:
        parser.print_help()
        print("\n请指定 --episode 或 --currents_only")
