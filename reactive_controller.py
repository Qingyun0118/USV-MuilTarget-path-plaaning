"""
Expert 控制器: 比例导航 + 反应式避障

参考 Dawood et al. 2023 的思路, 在训练初期提供 "专家" 示范:
1. 比例导航: 计算期望航向, 按比例施加油门和舵角
2. 反应式避障: 利用传感器读数检测障碍物, 进行转向规避
3. 将 Expert 动作存入 Replay Buffer, 作为执行器的学习依据

Note: Expert 不需要完美, 只需要偶尔成功到达目标即可提供正向经验。
"""

import numpy as np


class ReactiveController:
    """
    Expert 控制器: 比例导航 + 反应式避障
    """

    def __init__(self, kp_heading=2.0, kp_throttle=0.0006,
                 obs_avoid_threshold=0.8, obs_critical_threshold=0.45,
                 min_throttle_for_turn=0.03, max_throttle=0.04):
        """
        Expert 控制器参数

        注意: USV 动力学中阻力极低 (a12=-0.00125), 因此:
        - max_throttle 必须很小 (~0.04) 以限制巡航速度 ~5-6 m/s
        - 检测到障碍物时必须使用负推力主动制动, 不能仅降低油门
        - 制动响应: 从 6m/s 全力制动到停止约需 6 步 (~0.8m)

        Args:
            kp_heading: 航向比例增益
            kp_throttle: 油门比例增益 (极小, 因为 USV 阻力低)
            obs_avoid_threshold: 避障触发阈值 (~12m)
            obs_critical_threshold: 紧急避障阈值 (~6.75m)
            min_throttle_for_turn: 避障转向时最低正油门 (维持舵效)
            max_throttle: 巡航最大油门 (限制 USV 最高速度)
        """
        self.kp_heading = kp_heading
        self.kp_throttle = kp_throttle
        self.obs_avoid_threshold = obs_avoid_threshold
        self.obs_critical_threshold = obs_critical_threshold
        self.min_throttle_for_turn = min_throttle_for_turn
        self.max_throttle = max_throttle

    def compute_action(self, usv_x, usv_y, usv_heading,
                       goal_x, goal_y, sensor_readings=None, usv_u=0.0):
        """
        计算导航动作 (比例导航 + 反应式避障)

        关键设计:
        - USV 动力学: 转向力矩 ∝ T_thrust * sin(δ), 推力为 0 则无法转向
        - 因此即使在避障时也保持最低油门以维持舵效
        - 仅在极度危险(即将碰撞)时才使用倒车

        传感器布局: N 根探测线均匀分布在 150° 扇形 FOV (前向),
        索引 0 = 右侧边缘, 索引 N-1 = 左侧边缘, 中间 = 正前方。

        Returns: np.array([throttle, rudder]) ∈ [-1, 1]²

        *** 关键物理约束 ***
        USV 偏航力矩 = b31 * T_thrust * sin(δ)
        当 T_thrust < 0 (倒车) 时, 舵角方向效果反转!
        → 必须在负油门时反转 rudder 符号以保持预期转向方向。
        """
        # ---- 1. 比例导航 ----
        dx = goal_x - usv_x
        dy = goal_y - usv_y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        desired_heading = np.arctan2(dy, dx)

        heading_error = desired_heading - usv_heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # 基础油门 (限制最大值以控制惯性, USV 阻力系数低易超速)
        throttle = np.clip(self.kp_throttle * dist, 0.005, self.max_throttle)
        if dist < 5.0:
            throttle *= max(0.2, dist / 5.0)

        # 舵角
        rudder = np.clip(self.kp_heading * heading_error / np.pi, -1.0, 1.0)

        # ---- 2. 反应式避障 ----
        if sensor_readings is not None and len(sensor_readings) > 0:
            readings = np.array(sensor_readings)
            min_reading = float(np.min(readings))
            n = len(readings)

            if min_reading < self.obs_avoid_threshold:
                # 5 分区
                n5 = max(n // 5, 1)
                zones = [
                    readings[:n5],              # far_right
                    readings[n5:2 * n5],        # right
                    readings[2 * n5:3 * n5],    # center
                    readings[3 * n5:4 * n5],    # left
                    readings[4 * n5:],           # far_left
                ]
                zone_mins = [float(np.min(z)) for z in zones]
                fr_min, r_min, c_min, l_min, fl_min = zone_mins

                left_openness = float(np.mean(np.concatenate([zones[3], zones[4]])))
                right_openness = float(np.mean(np.concatenate([zones[0], zones[1]])))
                avoid_left = (left_openness >= right_openness)

                # ---- Level 0: 极度危险 (<1.8m) — 全力制动 + 转向 ----
                if min_reading < 0.12:
                    throttle = -1.0
                    rudder = 1.0 if avoid_left else -1.0

                # ---- Level 1: 非常近 (<3.75m) — 强力制动 + 急转 ----
                elif min_reading < 0.25:
                    throttle = -0.7
                    rudder = 0.95 if avoid_left else -0.95

                # ---- Level 2: 接近 (<6.75m) — 中等制动 + 大幅转向 ----
                elif min_reading < self.obs_critical_threshold:
                    if c_min < self.obs_critical_threshold:
                        throttle = -0.35
                        rudder = 0.9 if avoid_left else -0.9
                    elif min(r_min, fr_min) < self.obs_critical_threshold:
                        throttle = -0.2
                        rudder = 0.85
                    elif min(l_min, fl_min) < self.obs_critical_threshold:
                        throttle = -0.2
                        rudder = -0.85
                    else:
                        throttle = -0.15
                        rudder = 0.75 if avoid_left else -0.75

                # ---- Level 3: 警戒 (<12m) — 正油门转向 (保留舵效) ----
                else:
                    # 渐进减速但保持正油门 (正油门下舵角方向正常)
                    brake_ratio = 1.0 - (min_reading - self.obs_critical_threshold) / (
                            self.obs_avoid_threshold - self.obs_critical_threshold)
                    throttle = self.max_throttle * max(0.15, 1.0 - 0.7 * brake_ratio)

                    # 方向偏转
                    if c_min < 0.6:
                        rudder = 0.7 if avoid_left else -0.7
                    elif min(r_min, fr_min) < 0.6:
                        rudder = max(rudder, 0.5)
                    elif min(l_min, fl_min) < 0.6:
                        rudder = min(rudder, -0.5)
                    else:
                        intensity = 0.3 * brake_ratio
                        if avoid_left:
                            rudder = np.clip(rudder + intensity, -1.0, 1.0)
                        else:
                            rudder = np.clip(rudder - intensity, -1.0, 1.0)

        # ============================================================
        # 修正1: 防止倒车 (传感器只朝前, 倒车时完全盲区)
        # 当 u < 0: 覆盖为正油门, 恢复前进
        # 当 u 很小且油门为负: 改为滑行, 防止速度穿越零点
        # ============================================================
        if usv_u < -0.3:
            # 正在倒退, 传感器盲区 — 全力向前刹停倒车
            throttle = self.max_throttle
            rudder = np.clip(self.kp_heading * heading_error / np.pi, -1.0, 1.0)
        elif usv_u < 1.5 and throttle < -0.01:
            # 速度很低时不再主动制动, 用小正油门维持舵效
            throttle = self.min_throttle_for_turn

        # ============================================================
        # 修正2: USV 偏航耦合补偿
        # 偏航力矩 = b31 * T_thrust * sin(δ)
        # 当 T_thrust < 0, sin(δ) 贡献反号
        # → 反转 rudder 以维持预期转向方向
        # ============================================================
        if throttle < -0.01:
            rudder = -rudder

        return np.array([
            np.clip(throttle, -1.0, 1.0),
            np.clip(rudder, -1.0, 1.0)
        ], dtype=np.float32)
