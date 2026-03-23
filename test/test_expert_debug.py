"""诊断 Expert 碰撞时的传感器读数和障碍物位置关系"""
import numpy as np
from my_USV_Env import MyUsvEnv, X_MAX, Y_MAX
from reactive_controller import ReactiveController

crash_seeds = [42, 200, 1000]
expert = ReactiveController()

for seed in crash_seeds:
    env = MyUsvEnv(num_targets=3, num_obstacles=6, seed=seed,
                   record_flag=False, target_radius=3.0,
                   use_random_shapes=True, current_type='generated')
    obs = env.reset()

    print(f"\n{'='*60}")
    print(f"Seed {seed}: USV start=({env.usv.x:.1f},{env.usv.y:.1f}) heading={np.degrees(env.usv.heading):.0f}°")
    
    # 打印起始点附近的障碍物
    for i, o in enumerate(env.obstacles):
        d = np.sqrt((o.x - env.usv.x)**2 + (o.y - env.usv.y)**2)
        bounding = o.get_bounding_radius()
        if d < 30:  # 30m 以内
            print(f"  近障碍{i}: center=({o.x:.1f},{o.y:.1f}) dist={d:.1f} "
                  f"bounding_r={bounding:.1f} shape={o.shape.value}")

    target = env.targets[0]
    goal_abs = np.array([target.x, target.y])
    goal_dir = np.degrees(np.arctan2(goal_abs[1]-env.usv.y, goal_abs[0]-env.usv.x))
    print(f"  目标0: ({goal_abs[0]:.1f},{goal_abs[1]:.1f}) 方向={goal_dir:.0f}°")

    # 跑到碰撞, 记录最后30步
    history = []
    for step in range(500):
        min_sensor = float(np.min(obs['o_detector']))
        action = expert.compute_action(env.usv.x, env.usv.y, env.usv.heading,
                                    goal_abs[0], goal_abs[1], obs['o_detector'],
                                    usv_u=env.usv.u)
        
        history.append({
            'step': step, 'x': env.usv.x, 'y': env.usv.y,
            'heading': np.degrees(env.usv.heading),
            'min_sensor': min_sensor, 'action': action.copy(),
            'u': env.usv.u, 'v': env.usv.v, 'r': env.usv.r,
            'sensors': obs['o_detector'].copy()
        })
        if len(history) > 30:
            history.pop(0)

        obs, sub_done, crush, _, info = env.step(action, goal_abs)
        
        if crush:
            print(f"  >>> CRASH at step {step+1}")
            # 打印最后15步
            for h in history[-15:]:
                print(f"    step {h['step']:3d}: pos=({h['x']:.1f},{h['y']:.1f}) "
                      f"hdg={h['heading']:6.1f}° u={h['u']:5.2f} v={h['v']:5.2f} r={h['r']:5.2f} "
                      f"min_s={h['min_sensor']:.3f} act=[{h['action'][0]:+.3f},{h['action'][1]:+.3f}] "
                      f"sensors=[{','.join(f'{s:.2f}' for s in h['sensors'])}]")
            # 碰撞障碍物
            for i, o in enumerate(env.obstacles):
                if env.usv.is_colliding_with(o.geometry):
                    dx = o.x - env.usv.x
                    dy = o.y - env.usv.y
                    angle_to_obs = np.degrees(np.arctan2(dy, dx)) - np.degrees(env.usv.heading)
                    print(f"    碰撞障碍{i}: ({o.x:.1f},{o.y:.1f}) shape={o.shape.value} "
                          f"相对角度={angle_to_obs:.0f}°")
            break
    env.close_writer()
