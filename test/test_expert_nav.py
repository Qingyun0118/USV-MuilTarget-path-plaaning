"""
测试 Expert 控制器在m障碍物 + n目标的环境中能否顺序导航到所有目标
"""
import numpy as np
import os
from my_USV_Env import MyUsvEnv, X_MAX, Y_MAX
from Expert_HRL_td3_main import ReactiveController

SCALE = np.array([X_MAX, Y_MAX])
num_t=5
num_o=7
# 创建环境
env = MyUsvEnv(
    num_targets=num_t, num_obstacles=num_o, seed=42,
    record_flag=False, usv_length=3.0, usv_width=1.5,
    target_radius=3.0, use_random_shapes=True, current_type='generated'
)
obs = env.reset()

# 打印环境信息
info = env.get_environment_info()
print('=' * 60)
print('Expert 导航测试: 6障碍物 + 3目标')
print('=' * 60)
start = info['start_pos']
print(f"起始位置: ({start[0]:.1f}, {start[1]:.1f})")
for i, t in enumerate(info['targets']):
    print(f"目标{i}: ({t['x']:.1f}, {t['y']:.1f})")
for i, o in enumerate(info['obstacles']):
    print(f"障碍{i}: center=({o['x']:.1f}, {o['y']:.1f}), shape={o['shape']}")

# Expert控制器
expert = ReactiveController()

# 路径记录
csv_file = 'expert_test_path.csv'
if os.path.exists(csv_file):
    os.remove(csv_file)

total_distance = 0.0
prev_pos = np.array([env.usv.x, env.usv.y])
total_steps = 0
targets_reached = 0
crush = False

target_order = list(range(num_t))
print(f"\n访问顺序: {target_order}")

for tidx in target_order:
    if crush:
        break
    target = env.targets[tidx]
    goal_abs = np.array([target.x, target.y])
    print(f"\n--- 导航到目标 {tidx}: ({goal_abs[0]:.1f}, {goal_abs[1]:.1f}) ---")

    for step in range(1500):
        total_steps += 1
        # Expert 动作
        action = expert.compute_action(
            env.usv.x, env.usv.y, env.usv.heading,
            goal_abs[0], goal_abs[1],
            obs['o_detector'], usv_u=env.usv.u
        )
        obs, sub_done, crush, truncated, step_info = env.step(action, goal_abs)
        env.record_points(np.array([env.usv.x, env.usv.y]), csv_file, heading=env.usv.heading)

        # 距离
        curr_pos = np.array([env.usv.x, env.usv.y])
        total_distance += np.linalg.norm(curr_pos - prev_pos)
        prev_pos = curr_pos.copy()

        # 每200步打印状态
        if (step + 1) % 200 == 0:
            dist_to_goal = np.linalg.norm(curr_pos - goal_abs)
            min_sensor = float(np.min(obs['o_detector']))
            print(f"  步数 {step+1}: pos=({curr_pos[0]:.1f},{curr_pos[1]:.1f}) "
                  f"dist={dist_to_goal:.1f} min_sensor={min_sensor:.2f}")

        # 检查到达
        reached = step_info.get('reached_targets', [])
        if len(reached) > 0:
            for r in reached:
                targets_reached += 1
                print(f"  步数 {step+1}: 到达目标 {r}!")
            if tidx in reached or sub_done:
                break

        if crush:
            print(f"  步数 {step+1}: 碰撞! pos=({curr_pos[0]:.1f},{curr_pos[1]:.1f})")
            break
    else:
        dist_to_goal = np.linalg.norm(curr_pos - goal_abs)
        print(f"  超过1500步未到达目标 {tidx}, 剩余距离: {dist_to_goal:.1f}")

env.close_writer()

print(f"\n{'=' * 60}")
print("结果汇总:")
print(f"  到达目标: {targets_reached}/3")
print(f"  总步数: {total_steps}")
print(f"  总距离: {total_distance:.1f}")
print(f"  碰撞: {crush}")
print(f"  成功: {targets_reached >= 3 and not crush}")
print('=' * 60)

# 绘制路径
if os.path.exists(csv_file):
    env.draw_path(csv_file, ifshow=True, ifsave=True, savePath='.',
                  max_episode=1, show_usv_shape=True, usv_interval=100)
