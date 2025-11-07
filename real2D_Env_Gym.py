"""
Reinforcement learning maze example.
"""

from dis import dis
import numpy as np
import numpy
import sys
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas
from mpl_toolkits.mplot3d import Axes3D
import math,logging
import csv,os
from scipy.interpolate import griddata
from itertools import groupby
import gymnasium as gym
from gymnasium.spaces import Discrete,Box
from netCDF4 import Dataset

# 全局常量定义
# GOAL_POINT = np.array([1.0, 1.0])  # 目标点坐标
# 多目标点定义（按顺序访问）
GOAL_POINTS = [
    # np.array([35.0, 35.0]),  # 第一个目标点
    # np.array([30.0, 25.0]),  # 第二个目标点  
    # np.array([15.0, 15.0]),  # 第三个目标点
    # np.array([5.0, 5.0]),    # 第四个目标点
    np.array([1.0, 1.0])     # 最终目标点
]
START_POINT = np.array([40.0, 40.0])  # 起始点坐标

MIN = 1  # 环境最小边界
MAX = 40  # 环境最大边界

# 历史记录目录
record_dir = r'./history_point'
if not os.path.exists(record_dir):
    os.mkdir(record_dir)

class SeaEnv(gym.Env):
    """
    海洋航行强化学习环境类 - 多目标点版本
    模拟AUV在有海流和障碍物的海洋中按顺序访问多个目标点
    """
    
    def __init__(self):
        # """初始化环境"""
        # 状态空间：6维连续空间 [x, y, u, v, goal_x, goal_y] - 位置、海流速度和当前目标点
        # self.observation_space = Box(low=0.0, high=19.0, shape=(6,))
        self.observation_space = Box(low=np.array([0.0, 0.0, -10.0, -10.0, 0.0, 0.0]), high=np.array([40.0, 40.0, 10.0, 10.0, 40.0, 40.0]), dtype=np.float32)
        # 动作空间：8个离散方向
        self.action_space = Discrete(8)

        # 环境状态初始化
        self.current_point = START_POINT  # 当前位置
        self.current_goal_index = 0  # 当前目标点索引
        self.goal_points = GOAL_POINTS  # 所有目标点列表
        self.goal = self.goal_points[self.current_goal_index]  # 当前目标点
        self.current_pointwithflow = START_POINT.copy()  # 当前位置+海流信息
        # self.goal = GOAL_POINT  # 目标点
        self.start = START_POINT  # 起始点
        # self.start_togoal = GOAL_POINT-START_POINT  # 起始点到目标的向量
        self.step_counter = 0  # 步数计数器
        self.visited_goals = []  # 已访问的目标点
        
        # 物理参数
        self.steplength = 1  # 每步移动长度
        self.speed = 1.2  # 船舶基础速度
        self.width = 3  # 局部感知宽度
        self.height = 3  # 局部感知高度
        self.t = 0.4  # 时间参数
        self.time = 0.5  # 时间步长

        # 环境网格和海流数据
        self.X, self.Y = np.meshgrid(np.arange(0, 40, 1), np.arange(0, 40, 1))
        self.U, self.V = self.current_data()  # 海流速度场
        self.obs = self.obs_handle_ciecle()  # 障碍物
        self.circlelist = self.getcircles()  # 圆形障碍物列表
        self.r = 2  # 圆形障碍物半径

    @staticmethod
    def getcircles():
        """
        获取圆形障碍物位置列表
        
        Returns:
            list: 圆形障碍物中心坐标列表
        """
        obs_circle = [
            [3, 8], [12, 7], [26, 25], [18, 33], [26, 6],
            [16, 20], [25, 15], [16, 11], [32, 32], [32, 20],
            [35, 5], [5, 35], [10, 29], [7, 20], [19, 25],
            [30, 5], [6.5, 25], [20.5, 30], [10, 15], [23, 20],
        ]
        return obs_circle

    def iscollision(self):
        """
        检测是否与障碍物发生碰撞
        
        Returns:
            bool: 如果碰撞返回True，否则返回False
        """
        x = self.current_point[0]
        y = self.current_point[1]
        flag = False
        for pos in self.circlelist:  # 遍历所有障碍物
            dis1 = np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)  # 计算当前位置到障碍物的距离
            if dis1 < self.r:   # 如果距离太近，返回True
                return True
        return flag

    def obs_handle_ciecle(self):
        """
        处理圆形障碍物
        
        Returns:
            list: 障碍物列表
        """
        obstacle_x = []
        obstacle_y = []
        obslist=[]
        for i in range(self.U.shape[0]):
            for j in range(self.U.shape[1]):
                aa = str(type(self.U[i][j]))
                if aa != '<class \'numpy.float32\'>':
                    obstacle_x.append(i)
                    obstacle_y.append(j)
                    self.U[i][j] = 0
                    self.V[i][j] = 0
        return obslist

    @staticmethod
    def current_data():
        """
        从NetCDF文件加载海流数据
        
        Returns:
            tuple: (U, V) - 经度和纬度方向的海流速度场
        """
        nc_obj = Dataset('./data/chinasea_201801_c.nc')
        uu = np.transpose(nc_obj.variables['u'][1, :, :])  # 经度方向速度
        vv = np.transpose(nc_obj.variables['v'][1, :, :])  # 纬度方向速度
        uu = uu[60:101, 40:81]  # 截取北部分量
        vv = vv[60:101, 40:81]  # 截取东部分量
        return uu, vv

    def searchflow(self, k=0.09):
        """
        通过双线性插值获取当前位置的海流速度
        
        Args:
            k: 参数（未使用）
            
        Returns:
            numpy.array: [u, v] - 当前位置的海流速度向量
        """
        x = self.current_point[0]
        y = self.current_point[1]

        x1 = math.floor(x)  # 向下取整
        y1 = math.floor(y)
        if x >= MAX or y >= MAX:  # 如果起点超出范围
            u = np.array([self.U[x1][y1], self.V[x1][y1]])
            return u
            
        x2 = math.floor(x) + 1
        y2 = math.floor(y) + 1
        point_grid = self.current_point.copy()
        points = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

        # 四个相邻网格点的海流值
        values_u = np.array([self.U[x1][y1], self.U[x1][y2],self.U[x2][y1], self.U[x2][y2]])
        values_v = np.array([self.V[x1][y1], self.V[x1][y2], self.V[x2][y1], self.V[x2][y2]])
        
        # 双线性插值
        u = griddata(points, values_u, point_grid, method='linear')
        v = griddata(points, values_v, point_grid, method='linear')
        
        uv = np.concatenate((u, v))
        return uv
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_counter = 0
        self.current_point = START_POINT.copy()
        self.current_goal_index = 0
        self.goal = self.goal_points[self.current_goal_index]
        self.visited_goals = []
        
        # 构建状态向量并确保数据类型为float32
        self.current_pointwithflow_goal = np.concatenate((
            self.current_point, 
            self.searchflow(), 
            self.goal
        )).astype(np.float32)  # 关键：转换为float32
        
        state = np.array(self.current_pointwithflow_goal, dtype=np.float32)
        
        # 验证状态是否在观察空间内
        assert self.observation_space.contains(state), \
            f"State {state} not in observation space {self.observation_space}"
        
        info = {
            'goal_reached': False,
            'current_goal_index': self.current_goal_index,
            'visited_goals': self.visited_goals.copy()
        }
        
        return state, info

    def pos_toimg(self):
        """
        将当前位置周围区域转换为局部感知图像
        
        Returns:
            numpy.array: 3x3的局部感知矩阵，1表示可行，0表示障碍
        """
        img = np.ones((self.width, self.height))
        centerx_index = 1  # 中心点x索引
        centery_index = 1  # 中心点y索引
        
        for i in range(self.width):
            for j in range(self.height):
                # 检查每个网格点是否可行
                if_obstacle = self.iscollision()
                if if_obstacle:  # 与障碍物碰撞
                    img[i][j] = 0
                elif (self.current_point[0]-(centerx_index-i)*self.steplength) > MAX or \
                     (self.current_point[0]-(centerx_index-i)*self.steplength) < MIN or\
                     (self.current_point[1]-(centery_index-j)*self.steplength) > MAX or \
                     (self.current_point[1]-(centery_index-j)*self.steplength) < MIN:   # 与边界碰撞
                    img[i][j] = 0
                else:
                    img[i][j] = 1
        return img

    def step(self, action):
        """
        执行一个时间步的动作
        
        Args:
            action: 要执行的动作
            
        Returns:
            tuple: (state, reward, done, info) - 新状态、奖励、是否结束、额外信息
        """
        action = self.is_action_valid(action)  # 验证动作有效性
        old_flow = self.searchflow()  # 当前位置的海流
        old_pos, position, act_spd = self.caculate_position(action, old_flow)  # 计算新位置
        position = self.is_position_valid(position)  # 验证位置有效性

        self.current_point = position  # 更新当前位置
        
        # 检查是否到达当前目标点
        goal_reached = self.is_goal_reached()
        if goal_reached and self.current_goal_index < len(self.goal_points) - 1:
            # 到达目标点但不是最后一个，切换到下一个目标点
            self.visited_goals.append(self.current_goal_index)
            self.current_goal_index += 1
            self.goal = self.goal_points[self.current_goal_index]
        
        # 构建状态向量
        current_flow = self.searchflow()
        state = np.concatenate((self.current_point, current_flow, self.goal))
        
        # done = self.is_done()  # 检查是否完成所有目标
        terminated=self.is_done()
        # truncated=False
        
        reward = self.caculate_reward(old_pos, goal_reached, terminated, act_spd)  # 计算奖励

        self.step_counter = self.step_counter + 1  # 步数增加
        if self.step_counter > 1000:
            truncated=True
        else:
            truncated=False
        
        # 额外信息
        info = {
            'goal_reached': goal_reached,
            'current_goal_index': self.current_goal_index,
            'visited_goals': self.visited_goals.copy()
        }
        
        return state, reward, terminated, truncated,info

    def is_action_valid(self, action):
        """
        验证动作是否在有效范围内
        
        Args:
            action: 要验证的动作
            
        Returns:
            int: 有效动作
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        return action

    def caculate_position(self, action, old_flow):
        """
        根据动作和海流计算新位置
        
        Args:
            action: 执行的动作
            old_flow: 当前位置的海流速度
            
        Returns:
            tuple: (old_pos, new_pos, actual_speed) - 旧位置、新位置、实际速度
        """
        position = self.current_point.copy()
        old_pos = position.copy()
        basespeed = np.array([0.0, 0.0])  # 基础速度向量
        
        # 根据动作方向设置基础速度
        if action == 0:  # up
            basespeed[0] = basespeed[0] - self.speed
        elif action == 1:  # down
            basespeed[0] = basespeed[0] + self.speed
        elif action == 2:  # left
            basespeed[1] = basespeed[1] - self.speed
        elif action == 3:  # right
            basespeed[1] = basespeed[1] + self.speed
        elif action == 4:  # 左上
            basespeed[0] = basespeed[0] - self.speed/np.sqrt(2)
            basespeed[1] = basespeed[1] - self.speed/np.sqrt(2)
        elif action == 5:  # 右上
            basespeed[0] = basespeed[0] - self.speed/np.sqrt(2)
            basespeed[1] = basespeed[1] + self.speed/np.sqrt(2)
        elif action == 6:  # 左下
            basespeed[0] = basespeed[0] + self.speed/np.sqrt(2)
            basespeed[1] = basespeed[1] - self.speed/np.sqrt(2)
        else:  # 右下
            basespeed[0] = basespeed[0] + self.speed/np.sqrt(2)
            basespeed[1] = basespeed[1] + self.speed/np.sqrt(2)
            
        # 计算新位置：位置 + 时间 × (基础速度 + 海流速度)
        position = position + self.time * (basespeed+old_flow)
        actual_spd = basespeed + old_flow  # 实际合成速度

        return old_pos, position, actual_spd

    def is_position_valid(self, position):
        """
        确保位置在环境边界内
        
        Args:
            position: 要验证的位置
            
        Returns:
            numpy.array: 修正后的位置
        """
        for i in range(2):
            if position[i] > MAX:
                position[i] = MAX
            if position[i] < MIN:
                position[i] = MIN
        correct_position = position
        return correct_position

    def get_cosvalue(self, old_flow, real_act):
        """
        计算两个向量之间的夹角
        
        Args:
            old_flow: 海流速度向量
            real_act: 实际动作向量
            
        Returns:
            float: 两个向量的夹角（弧度）
        """
        if np.dot(old_flow,real_act) != 0:
            cos_value = np.dot(old_flow, real_act) / (np.linalg.norm(real_act) * np.linalg.norm(old_flow))
            cos_value = round(cos_value,10)
            np.clip(cos_value,-1,1)
            theta = math.acos(cos_value)
        else:
            theta = np.pi/2  # 90度夹角

        return theta

    def getrelative(self, obs):
        """
        将观测转换为相对于目标点的坐标
        
        Args:
            obs: 原始观测值
            
        Returns:
            numpy.array: 相对坐标观测值
        """
        kk = obs.copy()
        slice1 = kk[:2]  # 位置信息
        slice2 = kk[2:4]  # 海流信息
        goal_info = kk[4:6]  # 目标点信息
        # # 计算相对于目标点的位置，海流信息放大5倍
        # rela_obs = np.concatenate((-GOAL_POINT[0:1]+slice1[0:1],-GOAL_POINT[1:2]+slice1[1:2],slice2*5))
        # 计算相对于目标点的位置，海流信息放大5倍
        rela_obs = np.concatenate((-goal_info[0:1]+slice1[0:1], -goal_info[1:2]+slice1[1:2], 
                                   slice2*5, goal_info))
        return rela_obs

    def caculate_reward(self, old_pos, goal_reached, done, act_spd):
        """
        计算奖励函数 - 多目标点版本
        
        Args:
            old_pos: 移动前的位置
            goal_reached: 是否到达当前目标点
            done: 是否完成所有目标
            act_spd: 实际移动速度
            
        Returns:
            float: 奖励值
        """
        if done:
            # 完成所有目标，给予最高奖励
            reward = 200 + 50 * len(self.goal_points)  # 基础奖励 + 每个目标点的额外奖励
        elif goal_reached:
            # 到达当前目标点，给予高额奖励
            reward = 100 + 20 * (self.current_goal_index + 1)  # 基础奖励 + 目标点序号奖励
        elif all(self.current_point == old_pos):  # 位置不变
            reward = -10  # 停滞惩罚
        else:
            # 正常移动情况
            actual_speed = np.linalg.norm(act_spd)  # 实际速度大小
            old_dis = np.sqrt(np.sum((old_pos - self.goal) ** 2))  # 旧位置到当前目标的距离
            new_dis = np.sqrt(np.sum((self.current_point - self.goal) ** 2))  # 新位置到当前目标的距离

            # 距离奖励：实际距离减少量与理论最大减少量的差值
            distance_reward = (old_dis - new_dis - self.time * self.speed) * 10
            distance_reward = np.arctan(distance_reward)  # 利用反正切函数限制距离奖励范围
            
            # 流场奖励：鼓励利用海流提高速度
            flow_reward = np.arctan(10 * actual_speed/self.speed - 10)  # Rc(k2=10)

            # 奖励组合
            lamda1 = 2  # 距离奖励权重
            lamda2 = 1  # 流场奖励权重
            reward = lamda1 * distance_reward + lamda2 * flow_reward
            
            # 步数惩罚：步数较多时对正奖励进行衰减
            reward = reward/(0.05*self.step_counter+1) if reward > 0 and self.step_counter > 50 else reward
 
        # 碰撞惩罚
        if self.iscollision():  # 与障碍物碰撞
            reward = reward - 50
            
        return reward
    
    def is_goal_reached(self):
        """
        检查是否到达当前目标点
        
        Returns:
            bool: 如果到达当前目标返回True，否则返回False
        """
        if np.sqrt(np.sum((self.current_point - self.goal) ** 2)) < 0.5:
            return True
        else:
            return False
        
    def is_done(self):
        """
        检查是否完成所有目标点
        
        Returns:
            bool: 如果完成所有目标返回True，否则返回False
        """
        # 检查是否到达最后一个目标点
        if self.current_goal_index == len(self.goal_points) - 1 and self.is_goal_reached():
            print("All goals reached!")
            return True
        else:
            return False

    def get_actualtime(self, real_act, old_flow, getleast=False):
        """
        计算实际航行速度（考虑海流影响）
        
        Args:
            real_act: 实际动作向量
            old_flow: 海流速度向量
            getleast: 是否获取最小速度
            
        Returns:
            float: 实际速度大小
        """
        a = np.linalg.norm(old_flow)  # 海流速度大小
        b = self.speed  # 船舶基础速度
        theta = 0 if getleast else self.get_cosvalue(old_flow, real_act)  # 夹角
        
        # 计算合成速度（向量合成公式）
        speed = np.sqrt(a**2 + b**2 - 2*(a*np.sin(theta))**2 + 
                       (1 if theta < np.pi/2 else -1)*2*a*np.sqrt((b**2-(a*np.sin(theta))**2)*np.cos(theta)**2))
        return speed

    def finalflow(self, X, Y, k=1.44):
        """
        获取指定位置的海流速度（简化版本）
        
        Args:
            X, Y: 位置坐标
            k: 参数
            
        Returns:
            tuple: (u, v) - 海流速度分量
        """
        x = int(X)
        y = int(Y)
        x1= math.floor(x)
        y1 = math.floor(y)
        u = self.U[x1][y1]  # 直接取网格点值，不插值
        v = self.V[x1][y1]
        return u, v

    def final(self, nresult_list):
        """
        最终结果可视化 - 多目标点版本
        
        Args:
            nresult_list: 路径结果列表
        """
        # 处理结果列表，去除空元素
        result = [list(g) for k, g in groupby(nresult_list, lambda x: len(x) == 0) if not k]
        kkk = result[len(result)-1]  # 取最后一轮的路径点
        
        # 提取路径坐标
        m, n = np.zeros(len(kkk)), np.zeros(len(kkk))
        for i in range(len(kkk)):
            m[i] = kkk[i][0]  # x坐标
            n[i] = kkk[i][1]  # y坐标
            
        # 创建可视化图形
        plt.rcParams['figure.figsize'] = (10.0, 10.0)
        fig, ax = plt.subplots()
        
        # 绘制海流场
        x, y = np.meshgrid(np.arange(0, self.U.shape[0], 1),
                           np.arange(0, self.U.shape[1], 1))
        ax.quiver(x, y, np.transpose(self.U), np.transpose(self.V), width=0.001,color='black')
        
        # 计算路径总距离
        distlist = []
        for i in range(len(m) - 1):
            action = np.array([m[i + 1] - m[i], n[i + 1] - n[i]])  # 移动向量
            dist = np.linalg.norm(action)*5500  # 转换为实际距离
            distlist.append(dist)

        # 绘制路径
        plt.plot(m, n, 'b-', linewidth=2, label='AUV Path')
        # plt.plot(m, n, 'ro', markersize=2)  # 路径点
        
        # 绘制障碍物
        for circle_para in self.circlelist:
            ax.add_patch(plt.Circle(circle_para, self.r, alpha=1,label='Obstacle' if circle_para == self.circlelist[0] else ""))

        plt.xlabel('x(*0.5°)')
        plt.ylabel('y(*0.5°)')

        # 标记起点和所有目标点
        ax.plot(self.start[0], self.start[1], 'go', markersize=10, markeredgecolor='k', label='Start')
        
        # 绘制所有目标点，用不同颜色和标记区分
        colors = ['orange', 'cyan', 'magenta', 'yellow', 'red']
        markers = ['s', 'D', '^', 'v', '*']
        for i, goal in enumerate(self.goal_points):
            ax.plot(goal[0], goal[1], markers[i], color=colors[i], markersize=8, 
                   markeredgecolor='k', label=f'Goal {i+1}')

        # 输出结果信息
        print("Path coordinates (x):", m.tolist())
        print("Path coordinates (y):", n.tolist())
        print("Total path distance:", sum(distlist), "meters")
        print("Straight line distance:", len(distlist) * 5500 * 0.5, "meters")
        print(f"Visited {len(self.visited_goals)} goals: {self.visited_goals}")

        plt.legend()
        plt.title('Multi-Goal AUV Navigation Path')
        plt.savefig('multi_goal_path.png', dpi=300, bbox_inches='tight')

    def visualization(self):
        """
        环境可视化 - 多目标点版本
        
        Returns:
            tuple: (fig, ax) - 图形和坐标轴对象
        """
        plt.rcParams['figure.figsize'] = (12.0, 12.0)
        fig, ax = plt.subplots()
        
        # 创建与环境范围匹配的网格
        X, Y = np.meshgrid(np.arange(0, 40, 1), 
                        np.arange(0, 40, 1))
        
        # 绘制海流场，注意转置数据以匹配网格
        q = ax.quiver(X, Y, self.U[:40, :40].T, self.V[:40, :40].T, 
                    width=0.001, color='black')

        # 绘制圆形障碍物
        for circle_para in self.circlelist:
            ax.add_patch(
                plt.Circle(
                    (circle_para[0], circle_para[1]),
                    self.r,
                )
            )
        
        # 标记起点和所有目标点
        ax.plot(self.start[0], self.start[1], 'go', markersize=12, markeredgecolor='k', label='Start')
        
        # 绘制所有目标点，用不同颜色和标记区分
        colors = ['orange', 'cyan', 'magenta', 'yellow', 'red']
        markers = ['s', 'D', '^', 'v', '*']
        for i, goal in enumerate(self.goal_points):
            ax.plot(goal[0], goal[1], markers[i], color=colors[i], markersize=10, 
                   markeredgecolor='k', label=f'Goal {i+1}')
        
        # 标记当前目标点
        current_goal = self.goal_points[self.current_goal_index]
        ax.plot(current_goal[0], current_goal[1], markers[self.current_goal_index], 
               color=colors[self.current_goal_index], markersize=15, 
               markeredgecolor='k', markeredgewidth=2, label='Current Goal')
        
        # 添加图例和标签
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Multi-Goal AUV Navigation Environment')
        plt.legend()
        
        # 设置坐标轴范围
        plt.xlim(0, 41)
        plt.ylim(0, 41)
        plt.grid(True, alpha=0.3)
        
        return fig, ax

    def refresh(self, fig, ax):
        """
        刷新显示当前状态
        
        Args:
            fig: 图形对象
            ax: 坐标轴对象
            
        Returns:
            matplotlib.axes: 更新后的坐标轴
        """
        ax2 = ax.twinx()
        # 绘制局部感知区域
        lines = ax2.add_patch(
            plt.Rectangle(
                (self.current_point[0]-5*self.steplength/6-self.steplength,
                    self.current_point[1]-5*self.steplength/6-self.steplength),
                self.width*self.steplength/3,
                self.height*self.steplength/3,
                color='black',
                alpha =0.2
            )
        )

        # 标记当前位置
        ax2.plot(self.current_point[0], self.current_point[1], 'go', markersize=7, markeredgecolor='k')
        plt.axis([-10,10,-10,10])
        plt.axis('off')
        plt.pause(0.01)  # 暂停用于动画显示
        lines.remove()  # 移除临时图形
        return ax2

    def clearax(self):
        """
        清除并重新创建图形
        
        Returns:
            matplotlib.axes: 新的坐标轴对象
        """
        plt.close()
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.arange(0, 40, 1), np.arange(0, 40, 1))
        
        # 重新绘制海流场
        q = ax.quiver(X, Y, self.U[:40, :40].T, self.V[:40, :40].T, 
                     width=0.002, scale=20, color='blue', alpha=0.7)

        # 重新绘制障碍物
        for circle_para in self.circlelist:
            ax.add_patch(
                plt.Circle(
                    (circle_para[0], circle_para[1]),
                    self.r,
                    color='red',
                    alpha=0.7,
                    fill=True
                )
            )
            
        # 重新标记起点和所有目标点
        ax.plot(self.start[0], self.start[1], 'go', markersize=12, markeredgecolor='k', label='Start')
        
        colors = ['orange', 'cyan', 'magenta', 'yellow', 'red']
        markers = ['s', 'D', '^', 'v', '*']
        for i, goal in enumerate(self.goal_points):
            ax.plot(goal[0], goal[1], markers[i], color=colors[i], markersize=10, 
                   markeredgecolor='k', label=f'Goal {i+1}')
                   
        # 标记当前目标点
        current_goal = self.goal_points[self.current_goal_index]
        ax.plot(current_goal[0], current_goal[1], markers[self.current_goal_index], 
               color=colors[self.current_goal_index], markersize=15, 
               markeredgecolor='k', markeredgewidth=2, label='Current Goal')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Multi-Goal AUV Navigation Environment')
        plt.legend()
        plt.xlim(0, 41)
        plt.ylim(0, 41)
        plt.grid(True, alpha=0.3)
        
        return ax