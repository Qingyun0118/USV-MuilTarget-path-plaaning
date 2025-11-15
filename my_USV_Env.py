import matplotlib.pyplot as plt
import numpy as np
from math import * 
import gymnasium as gym
from gymnasium.spaces import Box,Dict,MultiBinary,Discrete
from netCDF4 import Dataset
import random
import csv,os
import math
from scipy.interpolate import griddata
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# define the obstacles class
class Obstacle():
    def __init__(self, x, y, radius_o,rou_c,rou_s):
        self.x = x
        self.y = y
        self.radius_o = radius_o
        self.rou_c = rou_c
        self.rou_s = rou_s
        
    
    # Detects whether the area of two obstacle clss is overlapping
    def is_overlapping(self, other):
        distance = ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
        if distance <= (self.radius_o +self.rou_c+self.rou_s)+ (other.radius_o+other.rou_c+other.rou_s):
            return True
        else:
            return False
    
    
    # draw the obstacle on the environment
    def show_obstacle(self, ax,label_obstacle=None, label_core=None, label_safety=None):
        circle_ob = plt.Circle((self.x, self.y), self.radius_o, color='red', fill=True,label=label_obstacle)
        ax.add_patch(circle_ob)
        circle_c= plt.Circle((self.x, self.y), self.radius_o+self.rou_c, color='red',linestyle='--',fill=False,label=label_core)
        ax.add_patch(circle_c)
        circle_s= plt.Circle((self.x, self.y), self.radius_o+self.rou_c+self.rou_s, color='brown',linestyle='--',fill=False,label=label_safety)
        ax.add_patch(circle_s)
        

# define the target area class
class Target():
    def __init__(self, x, y, radius_t):
        self.x = x
        self.y = y
        self.radius_t = radius_t
        
    # Detects whether the area of the target and the obstacle is overlapping
    def is_overlappingwith_obstacle(self, obstacle):
        distance = ((self.x - obstacle.x)**2 + (self.y - obstacle.y)**2)**0.5
        if distance <= (self.radius_t + obstacle.radius_o + obstacle.rou_c + obstacle.rou_s):
            return True
        else:
            return False
        
    def is_overlapping(self, other):
        distance = ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
        if distance <= self.radius_t + other.radius_t:
            return True
        else:
            return False
        
    # draw the target on the environment
    def show_target(self, ax, prio_num=None,label=None ):
        plt.plot(self.x,self.y,'g*',label=label)
        ax.text(self.x+self.radius_t, self.y+self.radius_t, str(prio_num), color='green', fontsize=12, ha='center', va='center')
        circle_t = plt.Circle((self.x, self.y), self.radius_t, color='green', linestyle='--', fill=False)
        ax.add_patch(circle_t)


#define the environment class for the USV

START_POS = np.array([5.0,5.0]) # starting position of the USV

# define the target positions
OBSTACLE_POS_R=[
    np.array([7.0,16.0,4.5]),
    np.array([60.0,10.0,10.0]),
    np.array([90.0,10.0,10.0]),
    np.array([40.0,4.0,4.0]),
    np.array([90.0,90.0,5.0]),
    np.array([45.0,65.0,5.0]),
    np.array([42.0,47.0,4.0]),
    np.array([92.0,64.0,4.0]),
    np.array([86.0,35.0,4.0]),
]

OBSTACLE_POS=[
    np.array([20.0,70.0]),
    np.array([40.0,30.0]),
    np.array([60.0,70.0]),
    np.array([80.0,20.0]),
    np.array([90.0,90.0]),
]

ROU_C=1.5
ROU_S=3.5

# define the obstacle positions
TARGET_POS=[
    np.array([80.0,50.0]),
    np.array([40.0,90.0]),
    np.array([10.0,50.0]),
]

# define the boundary of the environment
MIN=0.0
MAX=100.0
X_MIN=0.0
Y_MIN=0.0
X_MAX=100.0
Y_MAX=100.0
ROU_MAX=25.0

class MyUsvEnv():
    def __init__(self,record_flag=False):
        """
        define the action space:
        rou:the relative distance between the current waypoint and the next waypoint
        # beta:relative pitch angle between the current waypoint and the next waypoint  for 3D environment,now we only consider 2D environment
        alpha: relative yaw angle between the current waypoint and the next waypoint
        """
        # rou in [-ROU_MAX, ROU_MAX], alpha in [-pi, pi]
        self.action_space = Box(low=np.array([-ROU_MAX, -pi]), high=np.array([ROU_MAX, pi]))
        """
        the state value have to be normalized,be mapped to [-1,1]
        x_norm:normalized x coordinate of the USV
        y_norm:normalized y coordinate of the USV
        Rou_o_norm: normalized distance vector between the USV and the obstacles
        Alpha_o_norm: normalized yaw angle vector between the USV and the obstacles
        P:Hierarchical Priority vector 
        rou_t_norm: normalized distance between the USV and the target deploy on the P
        alpha_t_norm: normalized yaw angle between the USV and the targets
        """
        self.observation_space = {
            "x_norm":Box(low=0.0, high=1.0, shape=(1,)),#x_norm
            "y_norm":Box(low=0.0, high=1.0, shape=(1,)),#y_norm
            "Rou_o_norm":Box(low=0.0, high=1.0, shape=(len(OBSTACLE_POS_R),)),  # Rou_o_norm
            "Alpha_o_norm":Box(low=0.0, high=1.0, shape=(len(OBSTACLE_POS_R),2)),  # Alpha_o_norm
            "Priority_vector":MultiBinary(len(TARGET_POS)-1),  # P
            "rou_t_norm":Box(low=0.0, high=1.0, shape=(1,)),  # rou_t_norm
            "alpha_t_norm":Box(low=0.0, high=1.0, shape=(1,2)),  # alpha_t_norm
        }
    
        self.step_counter = 0# the number of steps
        self.start_pos = START_POS # starting position of the USV
        self.target_r=3.0 # radius of the target area
        self.P=np.zeros(len(TARGET_POS)-1)  # 
        self.target_index=0
        
        # Added file object properties
        self.csv_file = None
        self.csv_writer = None
        self.record_flag = record_flag
        self.X, self.Y = np.meshgrid(np.arange(0,100, 1), np.arange(0, 100, 1))
        self.U, self.V = self.current_data()
        self.timestep=0.5
    
    @staticmethod
    def current_data():
        """
        从NetCDF文件加载海流数据
        
        Returns:
            tuple: (U, V) - 经度和纬度方向的海流速度场
        """
        nc_obj = Dataset('data\globalsea_20250901.nc')
        uu = nc_obj.variables['uo'][0, 0,:,:][1248:1348, 3600:3700]   # 经度方向速度
        vv = nc_obj.variables['vo'][0, 0,:,:][1248:1348, 3600:3700]   # 纬度方向速度
        
        # 创建网格坐标
        x_coords = np.arange(0, 100, 1.0)
        y_coords = np.arange(0, 100, 1.0)
        
        # 遍历所有网格点
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                # 检查当前点是否在任一障碍物的影响范围内
                for obstacle in OBSTACLE_POS_R:
                    obs_x, obs_y, obs_radius = obstacle[0], obstacle[1], obstacle[2]
                    # 计算到障碍物中心的距离
                    distance = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                    # 如果在障碍物半径内，将海流速度置零
                    if distance <= (obs_radius):
                        uu[i, j] = 0.0  
                        vv[i, j] = 0.0
                        break  # 找到一个障碍物匹配即可，无需检查其他障碍物
            
        return uu, vv

    #define the reset function for the environment
    def reset(self):
        self.step_counter = 0
        self.current_pos = START_POS.copy()
        
        # Use the current point as a reference to calculate all relative relationships.
        x_norm = self.normalize_x_y(self.current_pos[0])
        y_norm = self.normalize_x_y(self.current_pos[1])
        
        # Correction: Pass the current point as the reference
        Rou_o_norm = np.array([self.calculate_normalize_Rou(self.current_pos, obs) for obs in OBSTACLE_POS_R])
        Alpha_o_norm = np.array([self.calculate_normalize_alpha(self.current_pos, obs) for obs in OBSTACLE_POS_R])
        
        # P = np.zeros(len(TARGET_POS))  # 
        self.P,self.target_index=self.update_priority_vector(self.current_pos)
        
        # Current Objective (First Objective)
        rou_t_norm = np.array([self.calculate_normalize_Rou(self.current_pos, TARGET_POS[self.target_index])])
        alpha_t_norm = np.array([self.calculate_normalize_alpha(self.current_pos, TARGET_POS[self.target_index])])
        
        init_state = {
            "x_norm": np.array([x_norm]),
            "y_norm": np.array([y_norm]),
            "Rou_o_norm": Rou_o_norm,
            "Alpha_o_norm": Alpha_o_norm,
            "Priority_vector": self.P,
            "rou_t_norm": rou_t_norm,
            "alpha_t_norm": alpha_t_norm,
        }
        # return init_state
        return init_state
    
    # define the reward function for the environment
    def calculate_reward(self,current_pos,old_pos):
        #(1)Safe obstacle avoidance
        reward_oi=0.0
        for obstacle_pos in OBSTACLE_POS_R:
            rou_oi=np.sqrt((obstacle_pos[0]-current_pos[0])**2+(obstacle_pos[1]-current_pos[1])**2)
            if rou_oi>obstacle_pos[2]+ROU_C+ROU_S:
                reward_oi+=0.0
            else:
                reward_oi += -self.calculate_normalize_Rou(current_pos, obstacle_pos)
        
        #(2)Target reaching
        reward_t=0.0
        done=False
        if np.array_equal(self.P,np.array([0.0,0.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[0][0])**2+(current_pos[1]-TARGET_POS[0][1])**2)
            if rou_t<self.target_r:
                self.P[0]=1.0
                print("Target{} Reached!".format(self.target_index))
                self.target_index+=1
            reward_t=-self.calculate_normalize_Rou(current_pos,TARGET_POS[0])+self.sgn(self.target_r-rou_t)
        elif np.array_equal(self.P,np.array([1.0,0.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[1][0])**2+(current_pos[1]-TARGET_POS[1][1])**2)
            if rou_t<self.target_r:
                self.P[1]=1.0
                print("Target{} Reached!".format(self.target_index))
                self.target_index+=1
            reward_t=-self.calculate_normalize_Rou(current_pos,TARGET_POS[1])+self.sgn(self.target_r-rou_t)
        elif np.array_equal(self.P,np.array([1.0,1.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[2][0])**2+(current_pos[1]-TARGET_POS[2][1])**2)
            if rou_t<self.target_r:####
                print("Target{} Reached!".format(self.target_index))
                # pass
                done=True
            reward_t=-self.calculate_normalize_Rou(current_pos,TARGET_POS[2])+self.sgn(self.target_r-rou_t)
            
        # (3)Shortest path
        reward_p=0.0
        reward_p = -np.linalg.norm(current_pos - old_pos)

        # (4) Reward for approaching the target point
        reward_t_approach=0.0
        old_dis=np.sqrt((old_pos[0]-TARGET_POS[self.target_index][0])**2+(old_pos[1]-TARGET_POS[self.target_index][1])**2)
        new_dis=np.sqrt((current_pos[0]-TARGET_POS[self.target_index][0])**2+(current_pos[1]-TARGET_POS[self.target_index][1])**2)
        reward_t_approach=np.arctan(old_dis-new_dis-10)
        
        #weigth the reward
        K_oi=5.0
        K_t=0.1
        K_p=0.005
        K_t_approach=0.5
        reward=K_oi*reward_oi+K_t*reward_t+K_p*reward_p+K_t_approach*reward_t_approach

        reward = reward/(0.05*self.step_counter+1) if reward > 0 and self.step_counter > 50 else reward
        # print("reward_oi:",reward_oi,"reward_t:",reward_t,"reward_p:",reward_p)
        return reward,done
    
    #define the step function for the environment
    def step(self, action):
        self.step_counter += 1
        old_pos = self.current_pos.copy()
        self.current_pos = self.calculate_next_pos(self.current_pos, action)
        self.check_pos(self.current_pos)
        
        reward,terminated=self.calculate_reward(self.current_pos,old_pos)#calculate the reward before update the priority vector
        
        # self.P,self.target_index=self.update_priority_vector(self.current_pos)
        
        # done=self.is_done(self.current_pos)
        # terminated=self.is_done(self.current_pos)
        
        if self.step_counter >= 1000:
            truncated=True
        else:
            truncated=False
        # truncated=False
        state={
            "x_norm": np.array([self.normalize_x_y(self.current_pos[0])]),
            "y_norm": np.array([self.normalize_x_y(self.current_pos[1])]),
            "Rou_o_norm": np.array([self.calculate_normalize_Rou(self.current_pos, obs) for obs in OBSTACLE_POS_R]),
            "Alpha_o_norm": np.array([self.calculate_normalize_alpha(self.current_pos, obs) for obs in OBSTACLE_POS_R]),
            "Priority_vector":self.P,
            "rou_t_norm": np.array([self.calculate_normalize_Rou(self.current_pos, TARGET_POS[self.target_index])]),
            "alpha_t_norm": np.array([self.calculate_normalize_alpha(self.current_pos, TARGET_POS[self.target_index])]),
        }
        
        if self.record_flag:
            self.record_points(self.current_pos,"usv_path_points_1.csv")
        
        # return state, reward, done
        return state, reward, terminated, truncated, {}

    #define the normalize function for the state value
    def normalize_x_y(self, x):
        return x/MAX
    
    def calculate_normalize_Rou(self, current_pos, target_pos):
        """Calculate the normalized distance from current_pos to target_pos"""
        # Calculate the actual distance
        if isinstance(target_pos, list) and len(target_pos) > 2:
            # If it is an obstacle [x, y, r], only take the coordinates
            actual_distance = np.sqrt((target_pos[0]-current_pos[0])**2 + (target_pos[1]-current_pos[1])**2)
        else:
            actual_distance = np.sqrt((target_pos[0]-current_pos[0])**2 + (target_pos[1]-current_pos[1])**2)
        
        # Maximum Possible Distance (Environmental Diagonal)
        max_possible_distance = np.sqrt(X_MAX**2 + Y_MAX**2)
        return actual_distance / max_possible_distance
    
    def calculate_normalize_alpha(self, current_pos, target_pos):
        """Calculate the angle (sine and cosine) from current_pos to target_pos"""
        if isinstance(target_pos, list) and len(target_pos) > 2:
            # If it is an obstacle [x, y, r], only take the coordinates
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
        else:
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
        
        alpha = np.arctan2(dy, dx)
        return np.array([np.sin(alpha), np.cos(alpha)])  # shape (2,)
    
    def sgn(self, x):
        return 1 if x >= 0 else 0
    
    # caculate the position of the USV after taking an action
    def calculate_next_pos(self, current_pos, action):
        rou=action[0]
        alpha=action[1]
        # current_pos[0]+=rou*np.cos(alpha)
        # current_pos[1]+=rou*np.sin(alpha)
        
        # 获取当前点的海流速度
        flow_u, flow_v = self.get_flow_velocity(current_pos)
        
        current_pos[0] += rou*np.cos(alpha)+ flow_u*self.timestep
        current_pos[1] += rou*np.sin(alpha)+ flow_v*self.timestep
        
        return current_pos
    
    # get the flow velocity at the current position
    def get_flow_velocity(self, position):
        x = position[0]
        y = position[1]
        
        # 使用双线性插值获取海流速度
        x1 = math.floor(x)
        y1 = math.floor(y)
        
        # 边界检查
        if x1 >= 99 or y1 >= 99 or x1 < 0 or y1 < 0:
            return 0.0, 0.0
        
        x2 = x1 + 1
        y2 = y1 + 1
        
        # 获取四个网格点的海流速度
        points = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
        values_u = np.array([
            self.U[y1, x1], self.U[y2, x1], 
            self.U[y1, x2], self.U[y2, x2]
        ])
        values_v = np.array([
            self.V[y1, x1], self.V[y2, x1],
            self.V[y1, x2], self.V[y2, x2]
        ])
        
        # 双线性插值
        u = griddata(points, values_u, (x, y), method='linear')
        v = griddata(points, values_v, (x, y), method='linear')
        
        # 处理可能的NaN值
        if np.isnan(u):
            u = 0.0
        if np.isnan(v):
            v = 0.0
            
        return u, v
    
    # update the priority vector
    def update_priority_vector(self, current_pos):
        if np.sqrt((current_pos[0]-TARGET_POS[0][0])**2+(current_pos[1]-TARGET_POS[0][1])**2)<=self.target_r:
            self.P[0]=1.0
            self.target_index+=1
        elif np.sqrt((current_pos[0]-TARGET_POS[1][0])**2+(current_pos[1]-TARGET_POS[1][1])**2)<=self.target_r:
            self.P[1]=1.0
            self.target_index+=1
        elif np.sqrt((current_pos[0]-TARGET_POS[2][0])**2+(current_pos[1]-TARGET_POS[2][1])**2)<=self.target_r:
            # self.P[2]=1.0
            pass
        else:
            self.P=np.zeros(len(TARGET_POS)-1)
            self.target_index=0
        return self.P,self.target_index

    #check the position of the USV is within the boundary
    def check_pos(self, current_pos):
        if current_pos[0]<X_MIN:
            current_pos[0]=X_MIN
        elif current_pos[0]>X_MAX:
            current_pos[0]=X_MAX
        if current_pos[1]<Y_MIN:
            current_pos[1]=Y_MIN
        elif current_pos[1]>Y_MAX:
            current_pos[1]=Y_MAX
        return current_pos
    
    #check finish all targets
    def is_done(self,current_pos):
        rou_t=np.sqrt((current_pos[0]-TARGET_POS[2][0])**2+(current_pos[1]-TARGET_POS[2][1])**2)####
        if np.array_equal(self.P,np.array([1.0,1.0])) and rou_t<=self.target_r:
            return True
        else:
            return False
    
    # visualize the environment
    def visualize_env(self):
        # set the figure size
        plt.rcParams['figure.figsize'] = (6.0, 7.0)
        fig,ax=plt.subplots()
        
        # 设置背景色
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F5F5F5')  # 浅灰色背景
        
        quiver=ax.quiver(self.X, self.Y, self.U[:100, :100].T, self.V[:100, :100].T, 
            width=0.002, scale=20, color='blue', alpha=0.3)
        
        # add the starting position of the USV to the environment
        start_point = ax.plot(self.start_pos[0],self.start_pos[1],'r*',markersize=10,zorder=5)
        
        # add the targets to the environment
        target_points = []
        for i in range(len(TARGET_POS)):
            t=Target(TARGET_POS[i][0],TARGET_POS[i][1],3)
            target_point=t.show_target(ax,i)
            target_points.append(target_point)
        
        # add the obstacles to the environment
        obstacle_patches = []
        for i in range(len(OBSTACLE_POS_R)):
            o = Obstacle(OBSTACLE_POS_R[i][0], OBSTACLE_POS_R[i][1], 
                        OBSTACLE_POS_R[i][2], ROU_C, ROU_S)
            obstacle_patch = o.show_obstacle(ax)
            obstacle_patches.append(obstacle_patch)
            
        # 定义图例元素
        legend_elements = [
            Line2D([0], [0], marker='*', color='red', markeredgecolor='red',
                   markersize=10, label='Start Point', linewidth=0),
            Patch(facecolor='none', edgecolor='green', linestyle='--', label='Target Area'),
            Patch(facecolor='red', edgecolor='red', label='Obstacle'),
            Patch(facecolor='none', edgecolor='red', linestyle='--', label='Danger Area'),
            Patch(facecolor='none', edgecolor='brown', linestyle='--', label='Safety Area'),
            Line2D([0], [0], color='blue', alpha=0.3, label='Ocean Current')
        ]
        
        # 将图例放在图像下方
        ax.legend(handles=legend_elements, loc='upper center', 
                 bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
        #add labeles to the environment
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Multi-Target USV Path Planning Environment')

        
        plt.xlim(X_MIN,X_MAX)
        plt.ylim(Y_MIN,Y_MAX)
        plt.grid(True, alpha=0.3)
        # 调整布局，为底部图例腾出空间
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()
    
    # record the points of path,for drawing the path
    def record_points(self, point, csv_file="usv_path_points_1.csv"):
        if len(point) != 2:
            raise ValueError("Point should be a tuple/list with (x,y) coordinates")
        
        try:
            # 初始化文件写入器
            if self.record_flag and self.csv_file is None:
                file_exists = os.path.isfile(csv_file)
                self.csv_file = open(csv_file, 'a', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                if not file_exists or os.stat(csv_file).st_size == 0:
                    self.csv_writer.writerow(['x', 'y'])
                    self.csv_writer.writerow(self.start_pos)
                    self.csv_file.flush()
            
            # 写入数据
            if self.record_flag:
                self.csv_writer.writerow(point)
                self.csv_file.flush()  # 立即刷新缓冲区
                
        except Exception as e:
            print(f"Error recording point: {e}")
            # 发生错误时关闭文件
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None

    def close_writer(self):  # 新增关闭方法
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
    
    
    # draw the path of the USV after done
    def draw_path(self, csv_file="usv_path_points.csv"):
        # set the figure size
        plt.rcParams['figure.figsize'] = (6.0, 7.0)
        fig,ax=plt.subplots()
        
        # 设置背景色
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F5F5F5')  # 浅灰色背景
        
        quiver=ax.quiver(self.X, self.Y, self.U[:100, :100].T, self.V[:100, :100].T, 
            width=0.002, scale=20, color='blue', alpha=0.3)
        
        # add the starting position of the USV to the environment
        start_point = ax.plot(self.start_pos[0],self.start_pos[1],'r*',markersize=10,zorder=5)
        
        # add the targets to the environment
        target_points = []
        for i in range(len(TARGET_POS)):
            t=Target(TARGET_POS[i][0],TARGET_POS[i][1],3)
            target_point=t.show_target(ax,i)
            target_points.append(target_point)
        
        # add the obstacles to the environment
        obstacle_patches = []
        for i in range(len(OBSTACLE_POS_R)):
            o = Obstacle(OBSTACLE_POS_R[i][0], OBSTACLE_POS_R[i][1], 
                        OBSTACLE_POS_R[i][2], ROU_C, ROU_S)
            obstacle_patch = o.show_obstacle(ax)
            obstacle_patches.append(obstacle_patch)
            
        try:
            # Check if the file is empty
            if not os.path.exists(csv_file):
                print("Path file is empty")
                return

            # Read data and filter invalid points
            x_points = []
            y_points = []
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2 and row[0] and row[1]:  # Make sure valid coordinates exist
                        try:
                            x = float(row[0])
                            y = float(row[1])
                            if X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX:
                                x_points.append(x)
                                y_points.append(y)
                        except ValueError:
                            continue

            # Check valid points
            if len(x_points) < 2:
                print(f"Only {len(x_points)} valid points, need at least 2")
                return
            
            # Draw path details
            ax.plot(x_points, y_points, 
                color='#FF6B35',     # 鲜艳的橙色
                linewidth=1.0,        # 中等线宽
                linestyle='-',        # 实线
                marker='o',           # 圆形标记点
                markersize=4,         # 小标记点
                markerfacecolor='white',  # 标记点内部白色
                markeredgecolor='#FF6B35', # 标记点边缘与路径同色
                markeredgewidth=1,    # 标记点边缘宽度
                zorder=4)             # 确保路径在障碍物之上
            
            plt.plot(x_points[-1], y_points[-1], 'rx', markersize=10,zorder=5)   # 终点
            
                    # 定义图例元素
            legend_elements = [
            Line2D([0], [0], marker='*', color='red', markeredgecolor='red',
                   markersize=10, label='Start Point', linewidth=0),
            Patch(facecolor='none', edgecolor='green', linestyle='--', label='Target Area'),
            Patch(facecolor='red', edgecolor='red', label='Obstacle'),
            Patch(facecolor='none', edgecolor='red', linestyle='--', label='Danger Area'),
            Patch(facecolor='none', edgecolor='brown', linestyle='--', label='Safety Area'),
            Line2D([0], [0], color='blue', alpha=0.3, label='Ocean Current'),
            Line2D([0], [0], color='#FF6B35', label='USV Path', linewidth=1.0, linestyle='-')
            ]
            
            # 将图例放在图像下方
            ax.legend(handles=legend_elements, loc='upper center', 
                 bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9)
            
            plt.title('Multi-Target path-planning USV')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True, alpha=0.3)
            plt.xlim(X_MIN, X_MAX)
            plt.ylim(Y_MIN, Y_MAX)
            # 调整布局，为底部图例腾出空间
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.show()

        except Exception as e:
            print(f"Error drawing path: {str(e)}")


# define the gym environment for the USV
class MyUsvEnv_Gym(gym.Env):
    def __init__(self, record_flag=False):
        """
        define the action space:
        rou:the relative distance between the current waypoint and the next waypoint
        # beta:relative pitch angle between the current waypoint and the next waypoint  for 3D environment,now we only consider 2D environment
        alpha: relative yaw angle between the current waypoint and the next waypoint
        """
        # rou in [-ROU_MAX, ROU_MAX], alpha in [-pi, pi]
        self.action_space = Box(low=np.array([-ROU_MAX, -pi]), high=np.array([ROU_MAX, pi]))
        """
        the state value have to be normalized,be mapped to [-1,1]
        x_norm:normalized x coordinate of the USV
        y_norm:normalized y coordinate of the USV
        Rou_o_norm: normalized distance vector between the USV and the obstacles
        Alpha_o_norm: normalized yaw angle vector between the USV and the obstacles
        P:Hierarchical Priority vector 
        rou_t_norm: normalized distance between the USV and the target deploy on the P
        alpha_t_norm: normalized yaw angle between the USV and the targets
        """
        self.observation_space = Dict({
            "x_norm":Box(low=0.0, high=1.0, shape=(1,)),#x_norm
            "y_norm":Box(low=0.0, high=1.0, shape=(1,)),#y_norm
            "Rou_o_norm":Box(low=0.0, high=1.0, shape=(len(OBSTACLE_POS_R),)),  # Rou_o_norm
            "Alpha_o_norm":Box(low=0.0, high=1.0, shape=(len(OBSTACLE_POS_R),2)),  # Alpha_o_norm
            "Priority_vector":MultiBinary(len(TARGET_POS)-1),  # P
            "rou_t_norm":Box(low=0.0, high=1.0, shape=(1,)),  # rou_t_norm
            "alpha_t_norm":Box(low=0.0, high=1.0, shape=(1,2)),  # alpha_t_norm
        })
    
        self.step_counter = 0# the number of steps
        self.start_pos = START_POS # starting position of the USV
        self.target_r=3.0 # radius of the target area
        self.P=np.zeros(len(TARGET_POS)-1)  # 
        self.target_index=0
        
        # Added file object properties
        self.csv_file = None
        self.csv_writer = None
        self.record_flag = record_flag
    
    #define the reset function for the environment
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.step_counter = 0
        self.current_pos = START_POS.copy()
        
        # Use the current point as a reference to calculate all relative relationships.
        x_norm = np.array([self.normalize_x_y(self.current_pos[0])])
        y_norm = np.array([self.normalize_x_y(self.current_pos[1])])
        
        # Correction: Pass the current point as the reference
        Rou_o_norm = np.array([self.calculate_normalize_Rou(self.current_pos, obs) for obs in OBSTACLE_POS_R])
        Alpha_o_norm = np.array([self.calculate_normalize_alpha(self.current_pos, obs) for obs in OBSTACLE_POS_R])
        
        # P = np.zeros(len(TARGET_POS))  # 
        self.P,self.target_index=self.update_priority_vector(self.current_pos)
        
        # Current Objective (First Objective)
        rou_t_norm = np.array([self.calculate_normalize_Rou(self.current_pos, TARGET_POS[self.target_index])])
        alpha_t_norm = np.array([self.calculate_normalize_alpha(self.current_pos, TARGET_POS[self.target_index])])
        
        init_state = {
            "x_norm": x_norm,
            "y_norm": y_norm,
            "Rou_o_norm": Rou_o_norm,
            "Alpha_o_norm": Alpha_o_norm,
            "Priority_vector": self.P,
            "rou_t_norm": rou_t_norm,
            "alpha_t_norm": alpha_t_norm,
        }
        # return init_state
        info = {}
        return init_state, info
    
    # define the reward function for the environment
    def calculate_reward(self,current_pos,old_pos):
        #(1)Safe obstacle avoidance
        reward_oi=0.0
        for obstacle_pos in OBSTACLE_POS_R:
            rou_oi=np.sqrt((obstacle_pos[0]-current_pos[0])**2+(obstacle_pos[1]-current_pos[1])**2)
            if rou_oi>obstacle_pos[2]+ROU_C+ROU_S:
                reward_oi+=0.0
            else:
                reward_oi += -self.calculate_normalize_Rou(current_pos, obstacle_pos)
        
        #(2)Target reaching
        reward_t=0.0
        done=False
        if np.array_equal(self.P,np.array([0.0,0.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[0][0])**2+(current_pos[1]-TARGET_POS[0][1])**2)
            if rou_t<self.target_r:
                self.P[0]=1.0
                print("Target{} Reached!".format(self.target_index))
                self.target_index+=1
            reward_t=-self.calculate_normalize_Rou(current_pos,TARGET_POS[0])+self.sgn(self.target_r-rou_t)
        elif np.array_equal(self.P,np.array([1.0,0.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[1][0])**2+(current_pos[1]-TARGET_POS[1][1])**2)
            if rou_t<self.target_r:
                self.P[1]=1.0
                print("Target{} Reached!".format(self.target_index))
                self.target_index+=1
            reward_t=-self.calculate_normalize_Rou(current_pos,TARGET_POS[1])+self.sgn(self.target_r-rou_t)
        elif np.array_equal(self.P,np.array([1.0,1.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[2][0])**2+(current_pos[1]-TARGET_POS[2][1])**2)
            if rou_t<self.target_r:####
                print("Target{} Reached!".format(self.target_index))
                # pass
                done=True
            reward_t=-self.calculate_normalize_Rou(current_pos,TARGET_POS[2])+self.sgn(self.target_r-rou_t)+1.0
            
        # (3)Shortest path
        reward_p=0.0
        reward_p = -np.linalg.norm(current_pos - old_pos)
        
        #weigth the reward
        K_oi=0.2
        K_t=0.1
        K_p=0.01
        reward=K_oi*reward_oi+K_t*reward_t+K_p*reward_p
        # print("reward_oi:",reward_oi,"reward_t:",reward_t,"reward_p:",reward_p)
        return reward,done
    
    #define the step function for the environment
    def step(self, action):
        self.step_counter += 1
        old_pos = self.current_pos.copy()
        self.current_pos = self.calculate_next_pos(self.current_pos, action)
        self.check_pos(self.current_pos)
        
        reward,terminated=self.calculate_reward(self.current_pos,old_pos)#calculate the reward before update the priority vector
        
        # self.P,self.target_index=self.update_priority_vector(self.current_pos)
        
        # done=self.is_done(self.current_pos)
        # terminated=self.is_done(self.current_pos)
        
        if self.step_counter >= 2000:
            truncated=True
        else:
            truncated=False
        # truncated=False
        state={
            "x_norm": np.array([self.normalize_x_y(self.current_pos[0])]),
            "y_norm": np.array([self.normalize_x_y(self.current_pos[1])]),
            "Rou_o_norm": np.array([self.calculate_normalize_Rou(self.current_pos, obs) for obs in OBSTACLE_POS_R]),
            "Alpha_o_norm": np.array([self.calculate_normalize_alpha(self.current_pos, obs) for obs in OBSTACLE_POS_R]),
            "Priority_vector":self.P,
            "rou_t_norm": np.array([self.calculate_normalize_Rou(self.current_pos, TARGET_POS[self.target_index])]),
            "alpha_t_norm": np.array([self.calculate_normalize_alpha(self.current_pos, TARGET_POS[self.target_index])]),
        }
        
        info={}
        
        # return state, reward, done
        return state, reward, terminated, truncated,info

    #define the normalize function for the state value
    def normalize_x_y(self, x):
        return x/MAX
    
    def calculate_normalize_Rou(self, current_pos, target_pos):
        """Calculate the normalized distance from current_pos to target_pos"""
        # Calculate the actual distance
        if isinstance(target_pos, list) and len(target_pos) > 2:
            # If it is an obstacle [x, y, r], only take the coordinates
            actual_distance = np.sqrt((target_pos[0]-current_pos[0])**2 + (target_pos[1]-current_pos[1])**2)
        else:
            actual_distance = np.sqrt((target_pos[0]-current_pos[0])**2 + (target_pos[1]-current_pos[1])**2)
        
        # Maximum Possible Distance (Environmental Diagonal)
        max_possible_distance = np.sqrt(X_MAX**2 + Y_MAX**2)
        return actual_distance / max_possible_distance
    
    def calculate_normalize_alpha(self, current_pos, target_pos):
        """Calculate the angle (sine and cosine) from current_pos to target_pos"""
        if isinstance(target_pos, list) and len(target_pos) > 2:
            # If it is an obstacle [x, y, r], only take the coordinates
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
        else:
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
        
        alpha = np.arctan2(dy, dx)
        return np.array([np.sin(alpha), np.cos(alpha)])  # shape (2,)
    
    def sgn(self, x):
        return 1 if x >= 0 else 0
    
    # caculate the position of the USV after taking an action
    def calculate_next_pos(self, current_pos, action):
        rou=action[0]
        alpha=action[1]
        current_pos[0]+=rou*np.cos(alpha)
        current_pos[1]+=rou*np.sin(alpha)
        return current_pos
    
    # update the priority vector
    def update_priority_vector(self, current_pos):
        if np.sqrt((current_pos[0]-TARGET_POS[0][0])**2+(current_pos[1]-TARGET_POS[0][1])**2)<=self.target_r:
            self.P[0]=1.0
            self.target_index+=1
        elif np.sqrt((current_pos[0]-TARGET_POS[1][0])**2+(current_pos[1]-TARGET_POS[1][1])**2)<=self.target_r:
            self.P[1]=1.0
            self.target_index+=1
        elif np.sqrt((current_pos[0]-TARGET_POS[2][0])**2+(current_pos[1]-TARGET_POS[2][1])**2)<=self.target_r:
            # self.P[2]=1.0
            pass
        else:
            self.P=np.zeros(len(TARGET_POS)-1)
            self.target_index=0
        return self.P,self.target_index

    #check the position of the USV is within the boundary
    def check_pos(self, current_pos):
        if current_pos[0]<X_MIN:
            current_pos[0]=X_MIN
        elif current_pos[0]>X_MAX:
            current_pos[0]=X_MAX
        if current_pos[1]<Y_MIN:
            current_pos[1]=Y_MIN
        elif current_pos[1]>Y_MAX:
            current_pos[1]=Y_MAX
        return current_pos
    
    #check finish all targets
    def is_done(self,current_pos):
        rou_t=np.sqrt((current_pos[0]-TARGET_POS[2][0])**2+(current_pos[1]-TARGET_POS[2][1])**2)####
        if np.array_equal(self.P,np.array([1.0,1.0])) and rou_t<=self.target_r:
            return True
        else:
            return False
    
    # visualize the environment
    def visualize_env(self):
        # set the figure size
        plt.rcParams['figure.figsize'] = (6.0, 6.0)
        fig,ax=plt.subplots()
        
        # add the starting position of the USV to the environment
        plt.plot(self.start_pos[0],self.start_pos[1],'bx')
        
        # add the targets to the environment
        for i in range(len(TARGET_POS)):
            t=Target(TARGET_POS[i][0],TARGET_POS[i][1],3)
            t.show_target(ax,i)
        
        # add the obstacles to the environment
        for i in range(len(OBSTACLE_POS_R)):
            o=Obstacle(OBSTACLE_POS_R[i][0],OBSTACLE_POS_R[i][1],OBSTACLE_POS_R[i][2],ROU_C,ROU_S)
            o.show_obstacle(ax)
        
        #add labeles to the environment
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Multi-Target USV Path Planning Environment')
        plt.legend()
        
        plt.xlim(X_MIN,X_MAX)
        plt.ylim(Y_MIN,Y_MAX)
        plt.grid(True, alpha=0.3)
        
        plt.show()
    
    # record the points of path,for drawing the path
    def record_points(self, point, csv_file="usv_path_points_1.csv"):
        if len(point) != 2:
            raise ValueError("Point should be a tuple/list with (x,y) coordinates")
        
        try:
            # 初始化文件写入器
            if self.record_flag and self.csv_file is None:
                file_exists = os.path.isfile(csv_file)
                self.csv_file = open(csv_file, 'a', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                if not file_exists or os.stat(csv_file).st_size == 0:
                    self.csv_writer.writerow(['x', 'y'])
                    self.csv_writer.writerow(self.start_pos)
                    self.csv_file.flush()
            
            # 写入数据
            if self.record_flag:
                self.csv_writer.writerow(point)
                self.csv_file.flush()  # 立即刷新缓冲区
                
        except Exception as e:
            print(f"Error recording point: {e}")
            # 发生错误时关闭文件
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None

    def close_writer(self):  # 新增关闭方法
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
    
    
    # draw the path of the USV after done
    def draw_path(self, csv_file="usv_path_points.csv"):
        plt.rcParams['figure.figsize'] = (6.0, 6.0)
        fig,ax=plt.subplots()
        
        # add the starting position of the USV to the environment
        plt.plot(self.start_pos[0],self.start_pos[1],'bx',label='Start')
        
        # add the targets to the environment
        for i in range(len(TARGET_POS)):
            t=Target(TARGET_POS[i][0],TARGET_POS[i][1],3)
            t.show_target(ax,i)
        
        # add the obstacles to the environment
        for i in range(len(OBSTACLE_POS_R)):
            o=Obstacle(OBSTACLE_POS_R[i][0],OBSTACLE_POS_R[i][1],OBSTACLE_POS_R[i][2],ROU_C,ROU_S)
            o.show_obstacle(ax)
            
        try:
            # Check if the file is empty
            if not os.path.exists(csv_file):
                print("Path file is empty")
                return

            # Read data and filter invalid points
            x_points = []
            y_points = []
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2 and row[0] and row[1]:  # Make sure valid coordinates exist
                        try:
                            x = float(row[0])
                            y = float(row[1])
                            if X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX:
                                x_points.append(x)
                                y_points.append(y)
                        except ValueError:
                            continue

            # Check valid points
            if len(x_points) < 2:
                print(f"Only {len(x_points)} valid points, need at least 2")
                return
            
            # Draw path details
            plt.plot(x_points, y_points, 'b-', linewidth=1, alpha=0.7)
            plt.plot(x_points[-1], y_points[-1], 'rx', label='End')   # 终点
            
            # Draw small arrows at the end of each line segment
            # for i in range(len(x_points)-1):
            #     plt.arrow(x_points[i], y_points[i], x_points[i+1]-x_points[i], y_points[i+1]-y_points[i], color='r', alpha=0.5, width=0.1, head_width=0.3, head_length=0.3)
            
            plt.title('USV Navigation Path with Environment')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.xlim(X_MIN, X_MAX)
            plt.ylim(Y_MIN, Y_MAX)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error drawing path: {str(e)}")


# test
if __name__ == '__main__':
    env=MyUsvEnv(record_flag=True)
    env.visualize_env()
    i=0
    obs=env.reset()
    # print(obs)
    while True:
        i+=1
        action=env.action_space.sample()
        obs,reward,terminated,truncated,_=env.step(action)
        print("step:",i)
        # print("action:",action)
        # position=np.array([obs['x_norm']*MAX,obs['y_norm']*MAX])
        # print("position:",position)
        # print("obs",obs)
        print("reward:",reward)
        if terminated:
            break
        # if i==1000:
        #     break
    env.draw_path("usv_path_points_1.csv")
    env.close_writer()
