"""
批量测试 MPC 控制器在多个随机环境中的鲁棒性
6 障碍物 + 3 目标, 测试 10 个不同种子
"""
import numpy as np
import os
from my_USV_Env import MyUsvEnv, X_MAX, Y_MAX
from MPC_HRL_td3_main import MPCController

SCALE = np.array([X_MAX, Y_MAX])
mpc = MPCController()

seeds = [2025, 42, 100, 123, 200, 333, 456, 789, 1000, 2024]
results = []

for seed in range(2026+80, 2026+80+920):
    env = MyUsvEnv(
        num_targets=3, num_obstacles=6, seed=seed,
        record_flag=False, usv_length=3.0, usv_width=1.5,
        target_radius=3.0, use_random_shapes=True, current_type='generated'
    )
    obs = env.reset()

    total_distance = 0.0
    prev_pos = np.array([env.usv.x, env.usv.y])
    total_steps = 0
    targets_reached = 0
    crush = False

    for tidx in range(3):
        if crush:
            break
        target = env.targets[tidx]
        goal_abs = np.array([target.x, target.y])

        for step in range(1500):
            total_steps += 1
            action = mpc.compute_action(
                env.usv.x, env.usv.y, env.usv.heading,
                goal_abs[0], goal_abs[1],
                obs['o_detector'], usv_u=env.usv.u
            )
            obs, sub_done, crush, truncated, step_info = env.step(action, goal_abs)

            curr_pos = np.array([env.usv.x, env.usv.y])
            total_distance += np.linalg.norm(curr_pos - prev_pos)
            prev_pos = curr_pos.copy()

            reached = step_info.get('reached_targets', [])
            if len(reached) > 0:
                for r in reached:
                    targets_reached += 1
                if tidx in reached or sub_done:
                    break
            if crush:
                break

    env.close_writer()
    success = targets_reached >= 3 and not crush
    results.append({
        'seed': seed, 'targets': targets_reached,
        'steps': total_steps, 'dist': total_distance,
        'crush': crush, 'success': success
    })
    status = "OK" if success else ("CRASH" if crush else "TIMEOUT")
    print(f"Seed {seed:5d}: {status} | EP {seed-2026:5d} | "
          f"Targets: {targets_reached}/3 | "
          f"Steps: {total_steps:5d} | "
          f"Distance: {total_distance:7.1f}")

# 汇总
print("\n" + "=" * 60)
n_success = sum(1 for r in results if r['success'])
n_crash = sum(1 for r in results if r['crush'])
n_timeout = sum(1 for r in results if not r['success'] and not r['crush'])
avg_dist = np.mean([r['dist'] for r in results if r['success']]) if n_success > 0 else 0
print(f"成功: {n_success}/{len(seeds)} ({100*n_success/len(seeds):.0f}%)")
print(f"碰撞: {n_crash}")
print(f"超时: {n_timeout}")
if n_success > 0:
    print(f"成功平均距离: {avg_dist:.1f}")
