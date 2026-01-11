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
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle, Circle,Wedge
from matplotlib.lines import Line2D
from shapely.geometry import Polygon, LineString, Point, box as shapely_box
from shapely.affinity import rotate, translate
from enum import Enum 
from USV import USV

# 障碍物形状枚举
class ObstacleShape(Enum):
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"


# define the obstacles class with multiple shapes
class Obstacle():
    def __init__(self, x, y, radius_o=None, rou_c=1.5, rou_s=3.5, 
                 shape=ObstacleShape.CIRCLE, width=None, height=None, 
                 vertices=None, rotation=0.0):
        """
        初始化障碍物
        Args:
            x: 障碍物中心x坐标
            y: 障碍物中心y坐标
            radius_o: 圆形障碍物半径（仅用于圆形）
            rou_c: 核心危险区域半径
            rou_s: 安全区域半径
            shape: 障碍物形状（ObstacleShape枚举）
            width: 矩形宽度（仅用于矩形）
            height: 矩形高度（仅用于矩形）
            vertices: 多边形顶点列表（仅用于多边形）
            rotation: 旋转角度（弧度）
        """
        self.x = x
        self.y = y
        self.radius_o = radius_o
        self.rou_c = rou_c
        self.rou_s = rou_s
        self.shape = shape
        self.width = width
        self.height = height
        self.vertices = vertices
        self.rotation = rotation
        
        # 创建几何形状
        self._create_geometry()
    
    def _create_geometry(self):
        """创建障碍物的几何形状"""
        if self.shape == ObstacleShape.CIRCLE:
            self.geometry = Point(self.x, self.y).buffer(self.radius_o)
            self.danger_zone = Point(self.x, self.y).buffer(self.radius_o + self.rou_c)
            self.safety_zone = Point(self.x, self.y).buffer(self.radius_o + self.rou_c + self.rou_s)
        
        elif self.shape == ObstacleShape.RECTANGLE:
            # 创建矩形
            half_w = self.width / 2
            half_h = self.height / 2
            rect = shapely_box(self.x - half_w, self.y - half_h, 
                              self.x + half_w, self.y + half_h)
            self.geometry = rotate(rect, self.rotation, origin=(self.x, self.y), use_radians=True)
            self.danger_zone = self.geometry.buffer(self.rou_c)
            self.safety_zone = self.geometry.buffer(self.rou_c + self.rou_s)
        
        elif self.shape == ObstacleShape.POLYGON:
            # 创建多边形
            poly = Polygon(self.vertices)
            poly = translate(poly, self.x, self.y)
            self.geometry = rotate(poly, self.rotation, origin=(self.x, self.y), use_radians=True)
            self.danger_zone = self.geometry.buffer(self.rou_c)
            self.safety_zone = self.geometry.buffer(self.rou_c + self.rou_s)
    
    def get_bounding_radius(self):
        """获取包围圆半径"""
        if self.shape == ObstacleShape.CIRCLE:
            return self.radius_o + self.rou_c + self.rou_s
        else:
            # 对于非圆形，计算边界框的对角线一半
            bounds = self.safety_zone.bounds
            dx = bounds[2] - bounds[0]
            dy = bounds[3] - bounds[1]
            return np.sqrt(dx**2 + dy**2) / 2
    
    def get_center(self):
        """获取障碍物中心"""
        return (self.x, self.y)
    
    # Detects whether the area of two obstacle class is overlapping
    def is_overlapping(self, other):
        """检测两个障碍物的安全区域是否重叠"""
        return self.safety_zone.intersects(other.safety_zone)
    
    def is_point_in_obstacle(self, point):
        """检测点是否在障碍物内"""
        p = Point(point[0], point[1])
        return self.geometry.contains(p)
    
    def is_point_in_danger_zone(self, point):
        """检测点是否在危险区域内"""
        p = Point(point[0], point[1])
        return self.danger_zone.contains(p)
    
    def is_point_in_safety_zone(self, point):
        """检测点是否在安全区域内"""
        p = Point(point[0], point[1])
        return self.safety_zone.contains(p)
    
    def distance_to_point(self, point):
        """计算点到障碍物的距离"""
        p = Point(point[0], point[1])
        return self.geometry.distance(p)
    
    # draw the obstacle on the environment
    def show_obstacle(self, ax, label_obstacle=None, label_core=None, label_safety=None):
        """绘制障碍物"""
        if self.shape == ObstacleShape.CIRCLE:
            circle_ob = plt.Circle((self.x, self.y), self.radius_o, 
                                   color='gray', fill=True, label=label_obstacle, zorder=3)
            ax.add_patch(circle_ob)
            circle_c = plt.Circle((self.x, self.y), self.radius_o + self.rou_c, 
                                  color='red', linestyle='--', fill=False, 
                                  label=label_core, zorder=2)
            ax.add_patch(circle_c)
            circle_s = plt.Circle((self.x, self.y), self.radius_o + self.rou_c + self.rou_s, 
                                  color='brown', linestyle='--', fill=False, 
                                  label=label_safety, zorder=2)
            ax.add_patch(circle_s)
        
        elif self.shape == ObstacleShape.RECTANGLE:
            # 绘制矩形障碍物
            vertices = list(self.geometry.exterior.coords)[:-1]
            rect_patch = MplPolygon(vertices, closed=True, 
                                    facecolor='gray',
                                    label=label_obstacle, zorder=3)
            ax.add_patch(rect_patch)
            
            # 绘制危险区域
            danger_vertices = list(self.danger_zone.exterior.coords)[:-1]
            danger_patch = MplPolygon(danger_vertices, closed=True,
                                      facecolor='none', edgecolor='red',
                                      linestyle='--', label=label_core, zorder=2)
            ax.add_patch(danger_patch)
            
            # 绘制安全区域
            safety_vertices = list(self.safety_zone.exterior.coords)[:-1]
            safety_patch = MplPolygon(safety_vertices, closed=True,
                                      facecolor='none', edgecolor='brown',
                                      linestyle='--', label=label_safety, zorder=2)
            ax.add_patch(safety_patch)
        
        elif self.shape == ObstacleShape.POLYGON:
            # 绘制多边形障碍物
            vertices = list(self.geometry.exterior.coords)[:-1]
            poly_patch = MplPolygon(vertices, closed=True,
                                    facecolor='gray',
                                    label=label_obstacle, zorder=3)
            ax.add_patch(poly_patch)
            
            # 绘制危险区域
            danger_vertices = list(self.danger_zone.exterior.coords)[:-1]
            danger_patch = MplPolygon(danger_vertices, closed=True,
                                      facecolor='none', edgecolor='red',
                                      linestyle='--', label=label_core, zorder=2)
            ax.add_patch(danger_patch)
            
            # 绘制安全区域
            safety_vertices = list(self.safety_zone.exterior.coords)[:-1]
            safety_patch = MplPolygon(safety_vertices, closed=True,
                                      facecolor='none', edgecolor='brown',
                                      linestyle='--', label=label_safety, zorder=2)
            ax.add_patch(safety_patch)


# define the target area class
class Target():
    def __init__(self, x, y, radius_t):
        self.x = x
        self.y = y
        self.radius_t = radius_t
        self.geometry = Point(x, y).buffer(radius_t)
        
    # Detects whether the area of the target and the obstacle is overlapping
    def is_overlappingwith_obstacle(self, obstacle):
        """检测目标区域是否与障碍物安全区域重叠"""
        return self.geometry.intersects(obstacle.safety_zone)
        
    def is_overlapping(self, other):
        """检测两个目标区域是否重叠"""
        return self.geometry.intersects(other.geometry)
    
    def is_point_inside(self, point):
        """检测点是否在目标区域内"""
        p = Point(point[0], point[1])
        return self.geometry.contains(p)
        
    # draw the target on the environment
    def show_target(self, ax, prio_num=None, label=None):
        plt.plot(self.x, self.y, 'g*', label=label, markersize=10, zorder=5)
        ax.text(self.x + self.radius_t, self.y + self.radius_t, str(prio_num), 
                color='green', fontsize=12, ha='center', va='center')
        circle_t = plt.Circle((self.x, self.y), self.radius_t, 
                              color='green', linestyle='--', fill=False, zorder=4)
        ax.add_patch(circle_t)


# define the boundary of the environment
MIN = 0.0
MAX = 100.0
X_MIN = 0.0
Y_MIN = 0.0
X_MAX = 100.0
Y_MAX = 100.0
ROU_MAX = 25.0

# 默认参数
DEFAULT_ROU_C = 1.5
DEFAULT_ROU_S = 3.5


class MyUsvEnv():
    def __init__(self, 
                 num_targets=3, 
                 num_obstacles=5,
                 seed=None,
                 record_flag=False,
                 usv_length=3.0,
                 usv_width=1.5,
                 target_radius=3.0,
                 obstacle_radius_range=(3.0, 10.0),
                 rou_c=DEFAULT_ROU_C,
                 rou_s=DEFAULT_ROU_S,
                 use_random_shapes=True,
                 fixed_start_pos=None,
                 fixed_targets=None,
                 fixed_obstacles=None,
                 current_type="generated"):
        """
        初始化USV环境
        
        Args:
            num_targets: 目标数量
            num_obstacles: 障碍物数量
            seed: 随机种子，用于生成可重复的随机场景
            record_flag: 是否记录路径点
            usv_length: USV长度
            usv_width: USV宽度
            target_radius: 目标区域半径
            obstacle_radius_range: 障碍物半径范围 (min, max)
            rou_c: 核心危险区域半径
            rou_s: 安全区域半径
            use_random_shapes: 是否使用随机形状的障碍物
            fixed_start_pos: 固定起始位置 (可选)
            fixed_targets: 固定目标位置列表 (可选)
            fixed_obstacles: 固定障碍物配置列表 (可选)
            current_type: 洋流生成方式 ("file" 或 "generated")
        """
        # 保存参数
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        self.seed = seed
        self.record_flag = record_flag
        self.usv_length = usv_length
        self.usv_width = usv_width
        self.target_radius = target_radius
        self.obstacle_radius_range = obstacle_radius_range
        self.rou_c = rou_c
        self.rou_s = rou_s
        self.use_random_shapes = use_random_shapes
        self.current_type = current_type
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 初始化随机数生成器
        self.rng = np.random.RandomState(seed)
        
        # 生成或使用固定的位置
        self.fixed_start_pos = fixed_start_pos
        self.fixed_targets = fixed_targets
        self.fixed_obstacles = fixed_obstacles
        
        # 生成环境配置
        self._generate_environment()
        
        # 定义动作空间
        # Action: [Throttle, Rudder]  
        # Throttle: [0, 1] -> 映射到网络输出 [-1, 1]  
        # Rudder: [-pi/4, pi/4] -> 映射到网络输出 [-1, 1] 
        self.action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # 定义观察空间
        # 自身信息 (o_usv): [x_norm, y_norm, u_norm, v_norm, r_norm, heading_sin, heading_cos]  
        # x, y 归一化到 [0, 1]  
        # u, v, r 归一化 (假设最大速度 10m/s, 最大角速度 pi/3 rad/s)  
        # self_state_dim = 7   
          
        # # 相对目标信息 (o_relative): [dx_norm, dy_norm, dist_norm]   
        # rel_target_dim = 3  
          
        # # 海流信息 (o_current): [flow_u_norm, flow_v_norm, flow_angle_sin, flow_angle_cos]  
        # current_info_dim = 4  
          
        # # 探测器信息 (o_detector): [d_1, ..., d_Nd] (归一化)  
        # detector_dim = self.usv.num_detection_lines  

        # self.observation_space = Dict({  
        #     "o_usv":      Box(low=-1.0, high=1.0, shape=(self_state_dim,), dtype=np.float32),  
        #     "o_relative": Box(low=-1.0, high=1.0, shape=(rel_target_dim,), dtype=np.float32),  
        #     "o_current":  Box(low=-np.inf, high=np.inf, shape=(current_info_dim,), dtype=np.float32),  
        #     "o_detector": Box(low=0.0,  high=1.0, shape=(detector_dim,), dtype=np.float32) 
        # }) 

        total_dim = 7 + 3 + 4 + self.usv.num_detection_lines  
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32) 
        
        self.step_counter = 0
        self.P = np.zeros(max(1, self.num_targets - 1))
        self.target_index = 0
        
        # 文件记录相关
        self.csv_file = None
        self.csv_writer = None
        
        # 海流数据
        self.X, self.Y = np.meshgrid(np.arange(0, 100, 1), np.arange(0, 100, 1))
        self.U, self.V = self._load_current_data()
        self.dt = 0.05
        self.current_time = 0.0

    def _get_observation(self):  
            """  
            计算观测值  
            """  
            # --- 1. 获取基础数据 ---  
            # 自身位置与航向  
            x, y = self.usv.x, self.usv.y  
            heading = self.usv.heading  
            
            # 动力学状态 (体坐标系)  
            u, v, r = self.usv.u, self.usv.v, self.usv.r  
            
            # 当前目标  
            if self.target_index < len(self.targets):  
                target = self.targets[self.target_index]  
                tx, ty = target.x, target.y  
                t_radius = target.radius_t # 通信半径/到达半径  
            else:  
                # 所有目标完成，设为当前位置或 0  
                tx, ty = x, y  
                t_radius = 0.0  
    
            # 获取海流 (全局坐标系)  
            flow_u, flow_v = self.get_flow_velocity(x, y, self.current_time)  
            
            # --- 2. 构建观测向量 (归一化处理) ---  
            
            # Part 1: USV 自身信息 (o_usv)  
            # 范围归一化：位置 / 100, 速度 / max_speed  
            o_usv = np.array([  
                x / X_MAX,  
                y / Y_MAX,  
                u / self.usv.max_u,  
                v / self.usv.max_v,  
                r / self.usv.max_r,  
                np.sin(heading),  
                np.cos(heading)  
            ], dtype=np.float32)  
            
            # Part 2: 相对目标信息 (o_relative)  
            dx = tx - x  
            dy = ty - y  
            dist = np.sqrt(dx**2 + dy**2)  
            
            # # 通信标志位 (论文中提到 o_comm takes 0 or 1)  
            # # 此处假设 target.radius_t 即为通信范围  
            # comm_flag = 1.0 if dist <= t_radius else 0.0  
            
            # 归一化距离 (用对角线长度)  
            max_dist = np.sqrt(X_MAX**2 + Y_MAX**2)  
            
            o_relative = np.array([  
                dx / max_dist,  
                dy / max_dist,  
                dist / max_dist,   
            ], dtype=np.float32)  
            
            # Part 3: 海流信息 (o_current)  
            # 计算流速大小和流向角度  
            flow_speed = np.sqrt(flow_u**2 + flow_v**2)  
            flow_angle = np.arctan2(flow_v, flow_u)   
            
            o_current = np.array([  
                flow_u , # 也可以只给 u, v  
                flow_v ,  
                np.sin(flow_angle),      
                np.cos(flow_angle)  
            ], dtype=np.float32)  
            
            # Part 4: 探测器信息 (o_detector)  
            # 直接使用 USV 的传感器读数，归一化到 [0, 1]  
            # self.usv.sensor_readings 存储的是实际距离  
            readings = np.array(self.usv.sensor_readings, dtype=np.float32)  
            o_detector = readings / self.usv.sensor_range  
            
            # # --- 3. 返回字典 ---  
            # return {  
            #     "o_usv": o_usv,  
            #     "o_relative": o_relative,  
            #     "o_current": o_current,  
            #     "o_detector": o_detector  
            # }
            return np.concatenate([o_usv, o_relative, o_current, o_detector])#扁平化输出


    def _generate_environment(self):
        """生成环境配置（起始位置、目标、障碍物）"""
        # 首先生成障碍物（因为起始位置需要避开障碍物）
        if self.fixed_obstacles is not None:
            self.obstacles = self._create_fixed_obstacles(self.fixed_obstacles)
        else:
            self.obstacles = self._generate_obstacles()
        
        # 生成起始位置（需要避开障碍物）
        if self.fixed_start_pos is not None:
            self.start_pos = np.array(self.fixed_start_pos)
        else:
            self.start_pos = self._generate_start_position()
        
        # 创建USV对象
        self.usv = USV(self.start_pos[0], self.start_pos[1],
                       heading=self.rng.uniform(0.0,360.0), length=self.usv_length, width=self.usv_width)
        
        # 生成目标（需要避开障碍物和起始位置）
        if self.fixed_targets is not None:
            self.targets = self._create_fixed_targets(self.fixed_targets)
        else:
            self.targets = self._generate_targets()
        
        # 更新当前位置
        self.current_pos = self.start_pos.copy()
    
    def _generate_start_position(self):
        """生成不与障碍物冲突的起始位置"""
        # 使用USV的预估包围半径
        usv_bounding_radius = max(self.usv_length, self.usv_width) / 2 * 1.2
        margin = usv_bounding_radius + self.rou_c + self.rou_s + 5
        max_attempts = 100
        
        for _ in range(max_attempts):
            x = self.rng.uniform(margin, X_MAX - margin)
            y = self.rng.uniform(margin, Y_MAX - margin)
            
            # 检查是否与已有障碍物冲突
            valid = True
            for obs in self.obstacles:
                if obs.is_point_in_safety_zone([x, y]):
                    valid = False
                    break
            
            if valid:
                return np.array([x, y])
        
        # 如果无法找到有效位置，返回默认位置
        return np.array([5.0, 5.0])
    
    def _generate_obstacles(self):
        """生成随机障碍物"""
        obstacles = []
        max_attempts = 100
        
        for i in range(self.num_obstacles):
            for attempt in range(max_attempts):
                # 随机选择形状
                if self.use_random_shapes:
                    shape = self.rng.choice([ObstacleShape.CIRCLE,
                                            ObstacleShape.RECTANGLE,
                                            ObstacleShape.POLYGON])
                else:
                    shape = ObstacleShape.CIRCLE
                
                # 随机生成位置
                margin = self.obstacle_radius_range[1] + self.rou_c + self.rou_s + 5
                x = self.rng.uniform(margin, X_MAX - margin)
                y = self.rng.uniform(margin, Y_MAX - margin)
                
                # 根据形状生成障碍物
                if shape == ObstacleShape.CIRCLE:
                    radius = self.rng.uniform(*self.obstacle_radius_range)
                    obs = Obstacle(x, y, radius_o=radius, rou_c=self.rou_c, rou_s=self.rou_s,
                                  shape=ObstacleShape.CIRCLE)
                
                elif shape == ObstacleShape.RECTANGLE:
                    width = self.rng.uniform(*self.obstacle_radius_range) * 2
                    height = self.rng.uniform(*self.obstacle_radius_range)
                    rotation = self.rng.uniform(0, 2 * np.pi)
                    obs = Obstacle(x, y, rou_c=self.rou_c, rou_s=self.rou_s,
                                  shape=ObstacleShape.RECTANGLE,
                                  width=width, height=height, rotation=rotation)
                
                elif shape == ObstacleShape.POLYGON:
                    # 生成随机多边形（3-6边）
                    num_vertices = self.rng.randint(3, 7)
                    angles = np.sort(self.rng.uniform(0, 2 * np.pi, num_vertices))
                    radii = self.rng.uniform(self.obstacle_radius_range[0],
                                            self.obstacle_radius_range[1], num_vertices)
                    vertices = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles)]
                    rotation = self.rng.uniform(0, 2 * np.pi)
                    obs = Obstacle(x, y, rou_c=self.rou_c, rou_s=self.rou_s,
                                  shape=ObstacleShape.POLYGON,
                                  vertices=vertices, rotation=rotation)
                
                # 检查是否与已有障碍物冲突
                valid = True
                for existing_obs in obstacles:
                    if obs.is_overlapping(existing_obs):
                        valid = False
                        break
                
                if valid:
                    obstacles.append(obs)
                    break
            
            if len(obstacles) <= i:
                print(f"警告：无法生成第 {i+1} 个障碍物，跳过")
        
        return obstacles
    
    def _generate_targets(self):
        """生成随机目标位置"""
        targets = []
        max_attempts = 100
        
        for i in range(self.num_targets):
            for attempt in range(max_attempts):
                margin = self.target_radius + 5
                x = self.rng.uniform(margin, X_MAX - margin)
                y = self.rng.uniform(margin, Y_MAX - margin)
                
                target = Target(x, y, self.target_radius)
                
                # 检查是否与障碍物冲突
                valid = True
                for obs in self.obstacles:
                    if target.is_overlappingwith_obstacle(obs):
                        valid = False
                        break
                
                # 检查是否与已有目标冲突
                if valid:
                    for existing_target in targets:
                        if target.is_overlapping(existing_target):
                            valid = False
                            break
                
                # 检查是否与起始位置太近
                if valid:
                    dist_to_start = np.sqrt((x - self.start_pos[0])**2 + (y - self.start_pos[1])**2)
                    if dist_to_start < self.target_radius + self.usv.get_bounding_radius() + 5:
                        valid = False
                
                if valid:
                    targets.append(target)
                    break
            
            if len(targets) <= i:
                print(f"警告：无法生成第 {i+1} 个目标，跳过")
        
        return targets
    
    def _create_fixed_obstacles(self, obstacle_configs):
        """根据配置创建固定障碍物"""
        obstacles = []
        for config in obstacle_configs:
            if isinstance(config, np.ndarray) and len(config) == 3:
                # 兼容旧格式 [x, y, radius]
                obs = Obstacle(config[0], config[1], radius_o=config[2],
                              rou_c=self.rou_c, rou_s=self.rou_s,
                              shape=ObstacleShape.CIRCLE)
            elif isinstance(config, dict):
                # 新格式字典配置
                obs = Obstacle(**config, rou_c=self.rou_c, rou_s=self.rou_s)
            else:
                continue
            obstacles.append(obs)
        return obstacles
    
    def _create_fixed_targets(self, target_configs):
        """根据配置创建固定目标"""
        targets = []
        for config in target_configs:
            if isinstance(config, np.ndarray) and len(config) >= 2:
                target = Target(config[0], config[1], self.target_radius)
            elif isinstance(config, dict):
                target = Target(config['x'], config['y'], 
                               config.get('radius', self.target_radius))
            else:
                continue
            targets.append(target)
        return targets
    
    def _load_current_data(self):
        """加载或生成海流数据"""
        if self.current_type == "generated":
            return self._generate_current_data()
        
        # 默认为 "file" 模式
        try:
            nc_obj = Dataset('data/globalsea_20250901.nc')
            uu = nc_obj.variables['uo'][0, 0, :, :][1248:1348, 3600:3700]
            vv = nc_obj.variables['vo'][0, 0, :, :][1248:1348, 3600:3700]
        except:
            # 如果无法加载数据，使用零海流
            print("警告：无法加载海流数据，使用零海流")
            uu = np.zeros((100, 100))
            vv = np.zeros((100, 100))
            return uu, vv
        
        # 在障碍物区域内将海流速度置零
        x_coords = np.arange(0, 100, 1.0)
        y_coords = np.arange(0, 100, 1.0)
        
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                for obs in self.obstacles:
                    if obs.is_point_in_obstacle([x, y]):
                        uu[i, j] = 0.0
                        vv[i, j] = 0.0
                        break
        
        return uu, vv

    def _generate_current_data(self, t=0):
        """根据公式生成海流数据"""
        x_coords = np.arange(0, 100, 1.0)
        y_coords = np.arange(0, 100, 1.0)
        
        uu = np.zeros((100, 100))
        vv = np.zeros((100, 100))
        
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                # 检查障碍物
                in_obstacle = False
                for obs in self.obstacles:
                    if obs.is_point_in_obstacle([x, y]):
                        in_obstacle = True
                        break
                
                if in_obstacle:
                    uu[i, j] = 0.0
                    vv[i, j] = 0.0
                else:
                    u, v = self._calculate_analytic_current(x, y, t)
                    uu[i, j] = u
                    vv[i, j] = v
                    
        return uu, vv

    def _calculate_analytic_current(self, x, y, t):
        """计算解析海流速度"""
        # 将环境坐标 (0-100) 映射到论文公式的有效范围
        # x方向映射到 [-10, 10]
        # y方向映射到 [-2, 2] 以覆盖整个区域
        scale_factor_x = 0.2
        scale_factor_y = 0.04  # 减小y方向的缩放因子，使流场在y方向上更宽
        offset = 50.0
        
        x_mapped = (x - offset) * scale_factor_x
        y_mapped = (y - offset) * scale_factor_y

        # 参数设置 (参考 Formula 5)
        B0 = 1.2
        epsilon = 0.3
        omega = 0.4
        beta = np.pi / 2
        k = 0.84
        c = 0.12
        
        # 计算 B(t)
        Bt = B0 + epsilon * np.cos(omega * t + beta)
        
        # 中间变量 (使用映射后的坐标)
        X_term = k * (x_mapped - c * t)
        Y_term = y_mapped - Bt * np.cos(X_term)
        
        sin_X = np.sin(X_term)
        cos_X = np.cos(X_term)
        
        D_sq = 1 + k**2 * Bt**2 * sin_X**2
        D = np.sqrt(D_sq)
        
        A = Y_term / D
        
        # 计算 sech^2(A)
        # sech(x) = 1/cosh(x)
        # 防止溢出
        try:
            cosh_A = np.cosh(A)
            sech2_A = 1.0 / (cosh_A**2)
        except OverflowError:
            sech2_A = 0.0
            
        # u = sech^2(A) * (1/D)
        # 论文公式 (6): v_cx = - dPsi/dy = sech^2(A)/D
        u = sech2_A / D
        
        # v calculation
        # dN/dx = k * Bt * sin(X_term)
        # dD/dx = (k^3 * Bt^2 * sin(X_term) * cos(X_term)) / D
        # dA/dx = (dN/dx * D - N * dD/dx) / D^2
        
        dN_dx = k * Bt * sin_X
        dD_dx = (k**3 * Bt**2 * sin_X * cos_X) / D
        dA_dx = (dN_dx * D - Y_term * dD_dx) / D_sq
        
        # 论文公式 (6): v_cy = dPsi/dx = -sech^2(A) * dA/dx
        v = -sech2_A * dA_dx
        
        return u, v
    
    def reset(self, seed=None):
        """重置环境"""
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
            self.rng = np.random.RandomState(seed)
        
        self.step_counter = 0
        self.current_time = 0.0

        self._generate_environment()
        self.U, self.V = self._load_current_data()

        self.target_index = 0
        
        return self._get_observation()
    
    def calculate_reward(self, old_pos, action):
        """计算奖励"""
        reward = 0.0  
        terminated = False  
        self.collision_flag = False  
          
        current_pos = np.array([self.usv.x, self.usv.y])  
          
        # A. 碰撞惩罚 (使用 Shapely 检测)  
        for obs in self.obstacles:  
            # 安全区域检测  
            if self.usv.is_colliding_with(obs.safety_zone):  
                # 进入警戒区  
                reward -= 0.5  
                if self.usv.is_colliding_with(obs.geometry):  
                    # 发生碰撞  
                    reward -= 100.0  
                    self.collision_flag = True  
                    terminated = True # 碰撞结束  
                    break  
          
        # B. 目标奖励  
        if self.target_index < len(self.targets):  
            target = self.targets[self.target_index]  
            dist = np.linalg.norm(current_pos - np.array([target.x, target.y]))  
              
            # 到达判定 (USV进入目标半径)  
            if dist < target.radius_t:  
                reward += 50.0  
                print(f"Target {self.target_index} Reached!")  
                self.target_index += 1  
                if self.target_index >= len(self.targets):  
                    terminated = True # 全部完成  
                    reward += 100.0  
            else:  
                # 引导奖励 (Shaping Reward)  
                # 1. 距离递减奖励  
                old_dist = np.linalg.norm(old_pos - np.array([target.x, target.y]))  
                reward += (old_dist - dist) * 10.0   
                  
                # 2. 惩罚时间消耗  
                reward -= 0.1  
                  
        # C. 动作惩罚 (平滑控制)  
        # 惩罚过大的舵角变化或激烈的动作  
        reward -= 0.01 * (action[1]**2)  
          
        return reward, terminated 
    
    def step(self, action):
        """执行一步动作"""
        self.step_counter += 1
        self.current_time += self.dt

         # 1. 动作映射 (Action Mapping)  
        # 将 [-1, 1] 映射到物理范围  
        # 油门: [-1, 1] -> [0, 1]  
        throttle = float((action[0] + 1.0) / 2.0)  
        throttle = np.clip(throttle, 0.0, 1.0)  
          
        # 舵角: [-1, 1] -> [-45deg, 45deg] (弧度)  
        max_rudder = np.radians(45)  
        rudder = float(action[1] * max_rudder) 

        # 2. 动力学更新 (USV Dynamics)  
        # 注意：USV.step 内部计算了推力产生的位移  
        old_pos = np.array([self.usv.x, self.usv.y])  
        self.usv.step(throttle, rudder) 
        
        # 3. 叠加海流影响 (Environment Effect)  
        # USV模型计算的是相对水的速度，需要加上水流速度  
        flow_u, flow_v = self.get_flow_velocity(self.usv.x, self.usv.y, self.current_time)  
        # 直接更新位置 (Euler 积分)  
        self.usv.x += flow_u * self.dt  
        self.usv.y += flow_v * self.dt
        # # 强制更新几何体 (因为手动修改了x,y)  
        # self.usv.update_position(self.usv.x, self.usv.y) 
        # 边界限制
        self.usv.x = np.clip(self.usv.x, X_MIN, X_MAX)
        self.usv.y = np.clip(self.usv.y, Y_MIN, Y_MAX)
        
        # 更新当前位置
        self.current_pos = np.array([self.usv.x, self.usv.y])

        # 4. 传感器探测 (Sensing)
        # 提取障碍物的几何体列表供 USV 探测  
        obs_geoms = [obs.geometry for obs in self.obstacles]  
        self.usv.detect_obstacles(obs_geoms)
         # 5. 奖励计算与状态判断  
        reward, terminated = self.calculate_reward(old_pos, action)  
        truncated = self.step_counter >= 2000  
          
        info = {  
            "collision": self.collision_flag,  
            "target_reached": self.target_index,  
            "flow": (flow_u, flow_v)  
        }  
        
        if self.record_flag:
            self.record_points(self.current_pos, "usv_path_points_test.csv", heading=self.usv.heading)
        
        return self._get_observation(), reward, terminated, truncated, info 
    
    def get_flow_velocity(self, x, y, t=None):
        """获取指定位置的海流速度"""
        if t is None:
            t = self.current_time

        if self.current_type == "generated":
            # 使用解析公式计算当前时间的海流
            return self._calculate_analytic_current(x, y, t)
            
        x1 = math.floor(x)
        y1 = math.floor(y)
        
        if x1 >= 99 or y1 >= 99 or x1 < 0 or y1 < 0:
            return 0.0, 0.0
        
        x2 = x1 + 1
        y2 = y1 + 1
        
        points = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
        values_u = np.array([
            self.U[y1, x1], self.U[y2, x1],
            self.U[y1, x2], self.U[y2, x2]
        ])
        values_v = np.array([
            self.V[y1, x1], self.V[y2, x1],
            self.V[y1, x2], self.V[y2, x2]
        ])
        
        u = griddata(points, values_u, (x, y), method='linear')
        v = griddata(points, values_v, (x, y), method='linear')
        
        if np.isnan(u):
            u = 0.0
        if np.isnan(v):
            v = 0.0
        
        return u, v
    
    def visualize_env(self, time_step=None):
        """可视化环境"""
        plt.rcParams['figure.figsize'] = (8.0, 9.0)
        fig, ax = plt.subplots()
        
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F5F5F5')
        
        # 如果指定了时间步，重新计算该时刻的海流
        if time_step is not None and self.current_type == "generated":
            U, V = self._generate_current_data(t=time_step)
        else:
            U, V = self.U, self.V
            
        # 绘制海流
        quiver = ax.quiver(self.X, self.Y, U[:100, :100].T, V[:100, :100].T,
                          width=0.002, scale=20, color='blue', alpha=0.3)
        
        # 绘制起始位置
        ax.plot(self.start_pos[0], self.start_pos[1], 'r*', markersize=15, zorder=5)
        
        # 绘制USV
        self.usv.show_usv(ax, color='blue', alpha=0.7)
        
        # 绘制目标
        for i, target in enumerate(self.targets):
            target.show_target(ax, i)
        
        # 绘制障碍物
        for i, obs in enumerate(self.obstacles):
            obs.show_obstacle(ax)
        
        # 图例
        legend_elements = [
            Line2D([0], [0], marker='*', color='red', markeredgecolor='red',
                   markersize=10, label='Start Point', linewidth=0),
            MplPolygon([(0, 0)], closed=True, facecolor='blue', edgecolor='darkblue',
                       alpha=0.7, label='USV'),
            Patch(facecolor='none', edgecolor='green', linestyle='--', label='Target Area'),
            Patch(facecolor='red', edgecolor='red', label='Obstacle'),
            Patch(facecolor='none', edgecolor='red', linestyle='--', label='Danger Area'),
            Patch(facecolor='none', edgecolor='brown', linestyle='--', label='Safety Area'),
            Line2D([0], [0], color='blue', alpha=0.3, label='Ocean Current')
        ]
        
        ax.legend(handles=legend_elements, loc='upper center',
                 bbox_to_anchor=(0.5, -0.08), ncol=3, framealpha=0.9)
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Multi-Target USV Path Planning Environment\n'
                 f'(Targets: {self.num_targets}, Obstacles: {self.num_obstacles}, Seed: {self.seed})')
        
        plt.xlim(X_MIN, X_MAX)
        plt.ylim(Y_MIN, Y_MAX)
        plt.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.show()

    def animate_currents(self, duration=20, interval=200, save_path="currents_animation.gif"):
        """
        以动画形式可视化海流随时间的变化
        
        Args:
            duration: 动画持续时间（仿真时间单位）
            interval: 帧间隔（毫秒）
            save_path: 保存路径
        """
        from matplotlib.animation import FuncAnimation
        
        if self.current_type != "generated":
            print("警告：仅在 'generated' 模式下支持海流动画")
            return

        plt.rcParams['figure.figsize'] = (8.0, 9.0)
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F5F5F5')
        
        # 初始化绘图元素
        # 绘制障碍物
        for obs in self.obstacles:
            obs.show_obstacle(ax)
            
        # 绘制目标
        for i, target in enumerate(self.targets):
            target.show_target(ax, i)
            
        # 绘制起始位置
        ax.plot(self.start_pos[0], self.start_pos[1], 'r*', markersize=15, zorder=5)
        
        # 初始化海流 Quiver
        # 先计算初始时刻的海流
        U0, V0 = self._generate_current_data(t=0)
        quiver = ax.quiver(self.X, self.Y, U0[:100, :100].T, V0[:100, :100].T,
                          width=0.002, scale=20, color='blue', alpha=0.3)
        
        # 设置标题和标签
        title = ax.set_title(f'Time-Varying Currents (t=0.00)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.grid(True, alpha=0.3)
        
        # 动画更新函数
        def update(frame):
            t = frame * self.dt
            U, V = self._generate_current_data(t=t)
            
            # 更新 Quiver 数据
            # quiver.set_UVC 需要展平的数组或者与 X, Y 形状匹配的数组
            # 注意：matplotlib quiver 的 U, V 通常对应 X, Y 的网格
            # 这里 self.X, self.Y 是 meshgrid 生成的
            # self.U, self.V 是 (100, 100)
            # 需要注意转置问题，与 visualize_env 保持一致
            quiver.set_UVC(U[:100, :100].T, V[:100, :100].T)
            
            title.set_text(f'Time-Varying Currents (t={t:.2f})')
            return quiver, title
            
        # 计算帧数
        frames = int(duration / self.dt)
        
        anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
        
        try:
            anim.save(save_path, writer='pillow')
            print(f"动画已保存至 {save_path}")
        except Exception as e:
            print(f"保存动画失败: {e}")
            print("请确保安装了 pillow (pip install pillow) 或其他动画写入器")
            
        plt.close()
    
    def record_points(self, point, csv_file="usv_path_points.csv", heading=None):
        """记录路径点和航向
        
        Args:
            point: (x, y) 坐标
            csv_file: CSV文件路径
            heading: USV航向角（弧度），如果为None则使用当前USV航向
        """
        if len(point) != 2:
            raise ValueError("Point should be a tuple/list with (x,y) coordinates")
        
        if heading is None:
            heading = self.usv.heading
        
        try:
            if self.csv_file is None:
                file_exists = os.path.isfile(csv_file)
                self.csv_file = open(csv_file, 'a', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                if not file_exists or os.stat(csv_file).st_size == 0:
                    self.csv_writer.writerow(['x', 'y', 'heading'])
                    self.csv_writer.writerow([self.start_pos[0], self.start_pos[1], self.usv.heading])
                    self.csv_file.flush()
            
            self.csv_writer.writerow([point[0], point[1], heading])
            self.csv_file.flush()
        except Exception as e:
            print(f"Error recording point: {e}")
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
    
    def close_writer(self):
        """关闭文件写入器"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
    
    def draw_path(self, csv_file="usv_path_points.csv", ifshow=True, ifsave=False,
                  savePath=None, max_episode=1, show_usv_shape=False, usv_interval=50):
        """绘制USV路径
        
        Args:
            csv_file: CSV文件路径
            ifshow: 是否显示图像
            ifsave: 是否保存图像
            savePath: 保存路径
            max_episode: 最大episode数量
            show_usv_shape: 是否使用USV形状绘制轨迹节点
            usv_interval: 每隔多少个点绘制一个USV形状（避免过于密集）
        """
        plt.rcParams['figure.figsize'] = (10.0, 11.0)
        fig, ax = plt.subplots()
        
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F5F5F5')
        
        # 绘制海流
        quiver = ax.quiver(self.X, self.Y, self.U[:100, :100].T, self.V[:100, :100].T,
                          width=0.002, scale=20, color='blue', alpha=0.3)
        
        # 绘制目标
        for i, target in enumerate(self.targets):
            target.show_target(ax, i)
        
        # 绘制障碍物
        for obs in self.obstacles:
            obs.show_obstacle(ax)
        
        try:
            colors = ['#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7DC6F',
                     '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA', '#F1948A']
            
            paths_drawn = 0
            
            for episode in range(1, max_episode + 1):
                if max_episode > 1:
                    base_dir = os.path.dirname(csv_file)
                    base_name = os.path.basename(csv_file)
                    if '_episode_' in base_name:
                        base_parts = base_name.split('_episode_')
                        base_prefix = base_parts[0]
                        episode_file = os.path.join(base_dir, f"{base_prefix}_episode_{episode}.csv")
                    else:
                        base_without_ext = os.path.splitext(base_name)[0]
                        episode_file = os.path.join(base_dir, f"{base_without_ext}_{episode}.csv")
                else:
                    episode_file = csv_file
                
                if not os.path.exists(episode_file):
                    print(f"Path file {episode_file} does not exist")
                    continue
                
                x_points = []
                y_points = []
                headings = []
                with open(episode_file, 'r') as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                        has_heading = len(header) >= 3 and 'heading' in header[2].lower()
                    except StopIteration:
                        print(f"File {episode_file} is empty")
                        continue
                    
                    for row in reader:
                        if len(row) >= 2 and row[0] and row[1]:
                            try:
                                x = float(row[0])
                                y = float(row[1])
                                if X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX:
                                    x_points.append(x)
                                    y_points.append(y)
                                    # 读取航向，如果没有则计算
                                    if len(row) >= 3 and row[2]:
                                        headings.append(float(row[2]))
                                    else:
                                        headings.append(0.0)
                            except ValueError:
                                continue
                
                if len(x_points) < 2:
                    print(f"Only {len(x_points)} valid points in {episode_file}, need at least 2")
                    continue
                
                # 如果没有航向数据，根据相邻点计算航向
                if not has_heading or all(h == 0.0 for h in headings):
                    for i in range(len(x_points)):
                        if i < len(x_points) - 1:
                            dx = x_points[i+1] - x_points[i]
                            dy = y_points[i+1] - y_points[i]
                            headings[i] = np.arctan2(dy, dx)
                        else:
                            headings[i] = headings[i-1] if i > 0 else 0.0
                
                color = colors[(episode - 1) % len(colors)] if max_episode > 1 else '#FF6B35'
                
                # 绘制路径线
                label = f'Episode {episode}' if max_episode > 1 else 'USV Path'
                ax.plot(x_points, y_points,
                       color=color,
                       linewidth=1.5,
                       linestyle='--',
                       alpha=0.7,
                       zorder=4,
                       label=label)
                
                # 绘制USV形状或圆点
                if show_usv_shape:
                    # 使用USV形状绘制轨迹节点
                    for i in range(0, len(x_points), usv_interval):
                        usv_marker = USV(x_points[i], y_points[i], heading=headings[i],
                                        length=self.usv_length * 0.8,
                                        width=self.usv_width * 0.8)
                        # 使用渐变透明度
                        alpha = 0.3 + 0.5 * (i / len(x_points))
                        vertices = usv_marker.get_vertices()
                        polygon_patch = MplPolygon(vertices, closed=True,
                                                   facecolor=color, edgecolor='darkblue',
                                                   alpha=alpha, linewidth=0.5, zorder=5)
                        ax.add_patch(polygon_patch)
                else:
                    # 使用圆点标记
                    ax.scatter(x_points[::usv_interval], y_points[::usv_interval],
                              c=color, s=20, marker='o', zorder=5,
                              edgecolors='white', linewidths=0.5)
                
                # 提取障碍物几何体供探测使用
                obs_geoms = [obs.geometry for obs in self.obstacles]

                # 绘制起点（使用USV形状，绿色）
                start_usv = USV(x_points[0], y_points[0], heading=headings[0],
                               length=self.usv_length, width=self.usv_width)
                start_usv.detect_obstacles(obs_geoms)
                start_usv.show_usv(ax, color='green', alpha=0.8)

                # 绘制终点（使用USV形状，红色）
                end_usv = USV(x_points[-1], y_points[-1], heading=headings[-1],
                             length=self.usv_length, width=self.usv_width)
                end_usv.detect_obstacles(obs_geoms)
                end_usv.show_usv(ax, color='red', alpha=0.9)
                
                paths_drawn += 1
            
            if paths_drawn == 0:
                print("No valid paths to draw")
                return
            
            legend_elements = [
                MplPolygon([(0, 0)], closed=True, facecolor='green', edgecolor='darkblue',
                          alpha=0.8, label='Start USV'),
                Patch(facecolor='none', edgecolor='green', linestyle='--', label='Target Area'),
                Patch(facecolor='gray', edgecolor='gray', label='Obstacle'),
                Patch(facecolor='none', edgecolor='red', linestyle='--', label='Danger Area'),
                Patch(facecolor='none', edgecolor='brown', linestyle='--', label='Safety Area'),
                Line2D([0], [0], color='blue', alpha=0.3, label='Ocean Current'),
                Line2D([0], [0], color='#FF6B35', label='USV Path', linewidth=1.5, linestyle='--'),
                MplPolygon([(0, 0)], closed=True, facecolor='red', edgecolor='darkblue',
                          alpha=0.9, label='End USV'),
            ]
            
            ax.legend(handles=legend_elements, loc='upper center',
                     bbox_to_anchor=(0.5, -0.08 if max_episode == 1 else -0.12),
                     ncol=4, framealpha=0.9)
            
            plt.title(f'Multi-Target USV Path Planning \n'
                     f'(Targets: {self.num_targets}, Obstacles: {self.num_obstacles}, Seed: {self.seed})')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True, alpha=0.3)
            plt.xlim(X_MIN, X_MAX)
            plt.ylim(Y_MIN, Y_MAX)
            
            bottom_margin = 0.06 if max_episode == 1 else 0.1
            plt.tight_layout(rect=[0, bottom_margin, 1, 0.95])
            
            if ifsave:
                if max_episode > 1:
                    save_filename = f"all_episodes_paths.png"
                else:
                    save_filename = f"{os.path.basename(csv_file).split('.')[0]}.png"
                plt.savefig(os.path.join(savePath, save_filename), dpi=300, bbox_inches='tight')
                print(f"Image saved to {os.path.join(savePath, save_filename)}")
            if ifshow:
                plt.show()
        except Exception as e:
            print(f"Error drawing path: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_environment_info(self):
        """获取环境信息"""
        info = {
            "num_targets": self.num_targets,
            "num_obstacles": self.num_obstacles,
            "seed": self.seed,
            "start_pos": self.start_pos.tolist(),
            "usv_size": {"length": self.usv_length, "width": self.usv_width},
            "targets": [{"x": t.x, "y": t.y, "radius": t.radius_t} for t in self.targets],
            "obstacles": []
        }
        
        for obs in self.obstacles:
            obs_info = {
                "x": obs.x,
                "y": obs.y,
                "shape": obs.shape.value,
                "rou_c": obs.rou_c,
                "rou_s": obs.rou_s
            }
            if obs.shape == ObstacleShape.CIRCLE:
                obs_info["radius"] = obs.radius_o
            elif obs.shape == ObstacleShape.RECTANGLE:
                obs_info["width"] = obs.width
                obs_info["height"] = obs.height
                obs_info["rotation"] = obs.rotation
            elif obs.shape == ObstacleShape.POLYGON:
                obs_info["vertices"] = obs.vertices
                obs_info["rotation"] = obs.rotation
            info["obstacles"].append(obs_info)
        
        return info


# 兼容性：保留旧的全局变量定义
START_POS = np.array([5.0, 5.0])
OBSTACLE_POS_R = [
    np.array([7.0, 16.0, 4.5]),
    np.array([60.0, 10.0, 10.0]),
    np.array([90.0, 10.0, 10.0]),
    np.array([40.0, 4.0, 4.0]),
    np.array([90.0, 90.0, 5.0]),
    np.array([45.0, 65.0, 5.0]),
    np.array([42.0, 47.0, 4.0]),
    np.array([92.0, 64.0, 4.0]),
    np.array([86.0, 35.0, 4.0]),
]
TARGET_POS = [
    np.array([80.0, 50.0]),
    np.array([40.0, 90.0]),
    np.array([10.0, 50.0]),
]
ROU_C = 1.5
ROU_S = 3.5


def test_usv_running_and_plot():
    """
    测试函数：测试USV在环境中的运行情况，并绘制路线图
    """
    print("\n" + "=" * 60)
    print("测试: USV运行与路径绘制")
    print("=" * 60)
    
    # 路径记录文件 (与 step 函数中硬编码的文件名一致)
    path_file = "usv_path_points_test.csv"
    
    # 清理旧的路径文件
    if os.path.exists(path_file):
        try:
            os.remove(path_file)
        except Exception:
            pass

    # 1. 初始化环境，开启记录模式
    env = MyUsvEnv(
        num_targets=3,
        num_obstacles=5,
        seed=2024,
        record_flag=True,
        use_random_shapes=True,
        current_type="generated"
    )
    
    # 2. 重置环境
    obs = env.reset()
    print("环境初始化完成")
    # env.visualize_env()
    # 3. 运行仿真
    max_steps = 2000
    total_reward = 0
    
    print(f"开始运行仿真 (最大步数: {max_steps})...")
    
    for step in range(max_steps):
        # 随机动作: 油门[0,1], 舵角[-1,1]
        # action = np.array([
        #     np.random.uniform(0.0, 1.0),
        #     np.random.uniform(-1.0, 1.0)
        # ], dtype=np.float32)
        action=env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"仿真结束 (步数: {step+1})")
            break
            
    print(f"总奖励: {total_reward:.2f}")
    
    # 4. 关闭记录器
    env.close_writer()
    
    # 5. 绘制路径
    if os.path.exists(path_file):
        print(f"正在绘制路径图...")
        env.draw_path(csv_file=path_file, ifshow=True, ifsave=True, savePath=".")
    else:
        print(f"错误: 未找到路径文件 {path_file}")


# test
if __name__ == '__main__':
    test_usv_running_and_plot()
    # print("=" * 60)
    # print("测试1: 使用随机种子生成环境")
    # print("=" * 60)
    
    # # 创建带有随机种子的环境
    # env = MyUsvEnv(
    #     num_targets=4,
    #     num_obstacles=6,
    #     seed=42,
    #     record_flag=False,
    #     usv_length=3.0,
    #     usv_width=1.5,
    #     use_random_shapes=True
    # )
    
    # # 显示环境信息
    # info = env.get_environment_info()
    # print(f"目标数量: {info['num_targets']}")
    # print(f"障碍物数量: {info['num_obstacles']}")
    # print(f"随机种子: {info['seed']}")
    # print(f"起始位置: {info['start_pos']}")
    # print(f"USV尺寸: {info['usv_size']}")
    # print("\n目标位置:")
    # for i, t in enumerate(info['targets']):
    #     print(f"  目标{i}: ({t['x']:.1f}, {t['y']:.1f}), 半径={t['radius']}")
    # print("\n障碍物信息:")
    # for i, o in enumerate(info['obstacles']):
    #     print(f"  障碍物{i}: 形状={o['shape']}, 位置=({o['x']:.1f}, {o['y']:.1f})")
    
    # # 可视化环境
    # env.visualize_env()
    
    # print("\n" + "=" * 60)
    # print("测试2: 使用相同种子重新生成环境（应该相同）")
    # print("=" * 60)
    
    # env2 = MyUsvEnv(
    #     num_targets=4,
    #     num_obstacles=6,
    #     seed=42,
    #     use_random_shapes=True
    # )
    
    # info2 = env2.get_environment_info()
    # print(f"起始位置: {info2['start_pos']}")
    # print("目标位置:")
    # for i, t in enumerate(info2['targets']):
    #     print(f"  目标{i}: ({t['x']:.1f}, {t['y']:.1f})")
    
    # print("\n" + "=" * 60)
    # print("测试3: 使用不同种子生成环境")
    # print("=" * 60)
    
    # env3 = MyUsvEnv(
    #     num_targets=3,
    #     num_obstacles=5,
    #     seed=123,
    #     use_random_shapes=True
    # )
    
    # info3 = env3.get_environment_info()
    # print(f"起始位置: {info3['start_pos']}")
    # print("目标位置:")
    # for i, t in enumerate(info3['targets']):
    #     print(f"  目标{i}: ({t['x']:.1f}, {t['y']:.1f})")
    
    # env3.visualize_env()
    
    # print("\n" + "=" * 60)
    # print("测试4: 运行简单的仿真")
    # print("=" * 60)
    
    # env4 = MyUsvEnv(
    #     num_targets=3,
    #     num_obstacles=9,
    #     seed=123,
    #     record_flag=True,
    #     use_random_shapes=True  
    # )
    
    # obs = env4.reset()
    # print(f"初始状态: x={obs['x_norm'][0]:.3f}, y={obs['y_norm'][0]:.3f}")
    # env4.visualize_env()
    # total_reward = 0
    # for i in range(5000):
    #     action = env4.action_space.sample()
    #     obs, reward, terminated, truncated, info = env4.step(action)
    #     total_reward += reward
        
    #     if terminated:
    #         print(f"在第 {i+1} 步完成所有目标!")
    #         break
    #     if truncated:
    #         print(f"在第 {i+1} 步超时")
    #         break
    
    # print(f"总奖励: {total_reward:.2f}")
    # env4.close_writer()
    
    # # 绘制路径
    # if os.path.exists("usv_path_points_test.csv"):
    #     env4.draw_path("usv_path_points_test.csv")

    # print("\n" + "=" * 60)
    # print("测试5: 测试海流随时间变化的可视化")
    # print("=" * 60)
    
    # env5 = MyUsvEnv(
    #     num_targets=3,
    #     num_obstacles=9,
    #     seed=123,
    #     current_type="generated"
    # )
    
    # # 生成海流变化动画
    # print("正在生成海流变化动画...")
    # env5.animate_currents(duration=20, save_path="currents_animation.gif")
    # 创建环境 (障碍物)  
