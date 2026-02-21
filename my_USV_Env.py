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
        obs_face = '#A0A0A0'
        obs_edge = '#404040'
        danger_edge = '#D62728'
        
        def _draw_zone_and_body(geometry, danger_zone):
            # 危险区域（仅虚线边框）
            dv = list(danger_zone.exterior.coords)[:-1]
            ax.add_patch(MplPolygon(dv, closed=True, facecolor='none',
                                    edgecolor=danger_edge, linestyle='--',
                                    linewidth=0.8, label=label_core, zorder=2))
            # 障碍物本体
            verts = list(geometry.exterior.coords)[:-1]
            ax.add_patch(MplPolygon(verts, closed=True, facecolor=obs_face,
                                    edgecolor=obs_edge, linewidth=0.8,
                                    label=label_obstacle, zorder=3))
        
        if self.shape == ObstacleShape.CIRCLE:
            ax.add_patch(plt.Circle((self.x, self.y), self.radius_o + self.rou_c,
                                    facecolor='none', edgecolor=danger_edge,
                                    linestyle='--', linewidth=0.8,
                                    label=label_core, zorder=2))
            ax.add_patch(plt.Circle((self.x, self.y), self.radius_o,
                                    facecolor=obs_face, edgecolor=obs_edge,
                                    linewidth=0.8, label=label_obstacle, zorder=3))
        else:
            _draw_zone_and_body(self.geometry, self.danger_zone)


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
        """绘制目标"""
        # 目标区域虚线圆
        circle_t = plt.Circle((self.x, self.y), self.radius_t,
                              facecolor='none', edgecolor='#2CA02C',
                              linestyle='--', linewidth=0.9, zorder=4)
        ax.add_patch(circle_t)
        # 目标中心标记
        ax.plot(self.x, self.y, marker='*', color='#2CA02C',
                markersize=10, markeredgewidth=0.5, label=label, zorder=5)
        # 序号
        if prio_num is not None:
            ax.text(self.x + self.radius_t + 0.5, self.y + self.radius_t + 0.5,
                    str(prio_num), fontsize=9, color='#2CA02C',
                    ha='left', va='bottom', zorder=6)


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
DEFAULT_ROU_S = 1.5


class MyUsvEnv():
    def __init__(self, 
                 num_targets=3, 
                 num_obstacles=5,
                 dt=0.05,
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
        self_state_dim = 7
          
        # 目标信息 (targets): [tx_norm, ty_norm, status] * num_targets
        # status: 0 (unvisited), 1 (visited)
        target_dim = 3 * self.num_targets
          
        # 海流信息 (o_current): [flow_u_norm, flow_v_norm, flow_angle_sin, flow_angle_cos]
        current_info_dim = 4
          
        # 探测器信息 (o_detector): [d_1, ..., d_Nd] (归一化)
        detector_dim = self.usv.num_detection_lines

        self.observation_space = Dict({
            "o_usv":      Box(low=-np.inf, high=np.inf, shape=(self_state_dim,), dtype=np.float32),
            "targets":    Box(low=-np.inf, high=np.inf, shape=(target_dim,), dtype=np.float32),
            "o_current":  Box(low=-np.inf, high=np.inf, shape=(current_info_dim,), dtype=np.float32),
            "o_detector": Box(low=0.0,  high=1.0, shape=(detector_dim,), dtype=np.float32)
        })
        
        self.step_counter = 0
        self.target_status = np.zeros(self.num_targets, dtype=np.float32) # 0: unvisited, 1: visited
        
        # 文件记录相关
        self.csv_file = None
        self.csv_writer = None
        
        # 海流数据
        self.X, self.Y = np.meshgrid(np.arange(0, 100, 1), np.arange(0, 100, 1))
        self.U, self.V = self._load_current_data()
        self.dt = dt
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
            
            # Part 2: 所有目标信息 (targets)
            # [tx_norm, ty_norm, status] * num_targets
            targets_info = []
            for i, target in enumerate(self.targets):
                targets_info.extend([
                    target.x / X_MAX,
                    target.y / Y_MAX,
                    self.target_status[i]
                ])
            o_targets = np.array(targets_info, dtype=np.float32)
            
            # Part 3: 海流信息 (o_current)
            # 计算流速大小和流向角度
            flow_speed = np.sqrt(flow_u**2 + flow_v**2)
            flow_angle = np.arctan2(flow_v, flow_u)
            
            o_current = np.array([
                flow_u / flow_speed, # 也可以只给 u, v
                flow_v / flow_speed,
                np.sin(flow_angle),
                np.cos(flow_angle)
            ], dtype=np.float32)
            
            # Part 4: 探测器信息 (o_detector)
            # 直接使用 USV 的传感器读数，归一化到 [0, 1]
            # self.usv.sensor_readings 存储的是实际距离
            readings = np.array(self.usv.sensor_readings, dtype=np.float32)
            o_detector = readings / self.usv.sensor_range
            
            # --- 3. 返回字典 ---
            return {
                "o_usv": o_usv,
                "targets": o_targets,
                "o_current": o_current,
                "o_detector": o_detector
            }


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

        self.target_status = np.zeros(self.num_targets, dtype=np.float32)
        
        return self._get_observation()
    
    def check_state(self, position):
        """
        只负责检查当前状态是否碰撞或到达目标
        """
        done = False
        self.collision_flag = False
        self.reached_targets = []
        # 碰撞惩罚 (使用 Shapely 检测)
        for obs in self.obstacles:
            if self.usv.is_colliding_with(obs.geometry):
                # 发生碰撞
                self.collision_flag = True
                break
        
        #更新全局目标状态
        for i, target in enumerate(self.targets):
            if self.target_status[i] == 0: # 未访问
                dist = np.linalg.norm(self.current_pos - np.array([target.x, target.y]))
                
                if dist < target.radius_t:
                    self.target_status[i] = 1.0
                    self.reached_targets.append(i)

        dist = np.linalg.norm(self.current_pos - position)  #靠近子目标的距离惩罚
        if dist < self.target_radius:  
            done = True

        return done, self.collision_flag
    
    def step(self, action, goal_position_global):
        """执行一步动作"""
        self.step_counter += 1
        self.current_time += self.dt

        # 1. 动作映射 (Action Mapping)   
        # 油门: [-1, 1]
        throttle = action[0]  
        # throttle = np.clip(throttle, 0.0, 1.0)  
        
        # 舵角: [-1, 1] -> [-60deg, 60deg] (弧度)  
        max_rudder = np.radians(60)  
        rudder = float(action[1] * max_rudder) 

        # 2. 动力学更新 (USV Dynamics)  
        # 注意：USV.step 内部计算了推力产生的位移  
        # old_pos = np.array([self.usv.x, self.usv.y])  
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
        # 5. 
        done, crush = self.check_state(goal_position_global)
        # truncated = self.step_counter >= 2000
        truncated=False
        
        info = {
            "collision": self.collision_flag,
            "reached_targets": self.reached_targets,
            "flow": (flow_u, flow_v),
            "target_status": self.target_status.copy()
        }
        
        if self.record_flag:
            self.record_points(self.current_pos, "usv_path_points_test.csv", heading=self.usv.heading)
        
        return self._get_observation(), done, crush, truncated, info 
    
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
        plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                             'mathtext.fontset': 'cm'})
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # 海流场
        if time_step is not None and self.current_type == "generated":
            U, V = self._generate_current_data(t=time_step)
        else:
            U, V = self.U, self.V
        # 稀疏采样，每隔3格取一个箭头
        step = 3
        Xs, Ys = self.X[::step, ::step], self.Y[::step, ::step]
        Us, Vs = U[:100, :100].T[::step, ::step], V[:100, :100].T[::step, ::step]
        ax.quiver(Xs, Ys, Us, Vs, color='#7FB3D8', alpha=0.5,
                  width=0.002, scale=25, zorder=1)
        
        # 障碍物
        for obs in self.obstacles:
            obs.show_obstacle(ax)
        
        # 目标
        for i, target in enumerate(self.targets):
            target.show_target(ax, i)
        
        # 起始位置
        ax.plot(self.start_pos[0], self.start_pos[1],
                'r*', markersize=12, zorder=8)
        
        # USV
        self.usv.show_usv(ax, color='#1F77B4', alpha=0.8)
        
        # 图例
        legend_elements = [
            Line2D([0], [0], marker='*', color='r', markersize=8,
                   label='Start', linewidth=0),
            MplPolygon([(0,0)], closed=True, facecolor='#1F77B4',
                       edgecolor='k', alpha=0.8, label='USV'),
            Line2D([0], [0], marker='*', color='#2CA02C', markersize=8,
                   label='Target', linewidth=0),
            Patch(facecolor='#A0A0A0', edgecolor='#404040', label='Obstacle'),
            Patch(facecolor='none', edgecolor='#D62728',
                  linestyle='--', label='Danger zone'),
            Line2D([0], [0], color='#7FB3D8', alpha=0.5,
                   linewidth=1.5, label='Current'),
        ]
        ax.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, -0.04), ncol=3,
                  fontsize=9, frameon=True, edgecolor='gray',
                  framealpha=1.0, handletextpad=0.4, columnspacing=1.0)
        
        ax.set_xlabel('$x$ / m')
        ax.set_ylabel('$y$ / m')
        ax.set_title(f'Environment  (targets={self.num_targets}, '
                     f'obstacles={self.num_obstacles}, seed={self.seed})',
                     fontsize=11, pad=8)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_aspect('equal')
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.tick_params(direction='in', length=3)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()

    def animate(self, csv_file=None, duration=20, interval=100,
                save_path="animation.gif", trail_length=0, speed=1):
        """
        动态可视化动画（海流 + 可选的 USV 轨迹回放）

        当提供 csv_file 时，同时播放时变海流与 USV 运动轨迹；
        当 csv_file 为 None 时，仅播放时变海流。

        Args:
            csv_file:     路径点 CSV 文件（列: x, y, heading），为 None 则仅显示海流
            duration:     仅海流模式下的动画时长（仿真时间单位），有 CSV 时自动按轨迹长度决定
            interval:     帧间隔（毫秒）
            save_path:    保存路径，为 None 则直接显示
            trail_length: 尾迹保留长度（0 = 保留全部已走过轨迹），仅轨迹模式有效
            speed:        每帧前进的点数（加速回放），仅轨迹模式有效
        """
        from matplotlib.animation import FuncAnimation

        if self.current_type != "generated":
            print("警告：仅在 'generated' 模式下支持时变海流动画")
            return

        # ---------- 读取 CSV（若有） ----------
        has_path = csv_file is not None
        x_pts, y_pts, headings = [], [], []

        if has_path:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                    has_heading = len(header) >= 3 and 'heading' in header[2].lower()
                except StopIteration:
                    print("CSV 文件为空"); return
                for row in reader:
                    if len(row) >= 2 and row[0] and row[1]:
                        try:
                            x, y = float(row[0]), float(row[1])
                            x_pts.append(x); y_pts.append(y)
                            headings.append(float(row[2]) if len(row) >= 3 and row[2] else 0.0)
                        except ValueError:
                            continue
            if len(x_pts) < 2:
                print("有效路径点不足"); return

            # 补算航向
            if not has_heading or all(h == 0.0 for h in headings):
                for i in range(len(x_pts)):
                    if i < len(x_pts) - 1:
                        headings[i] = np.arctan2(y_pts[i+1] - y_pts[i],
                                                 x_pts[i+1] - x_pts[i])
                    else:
                        headings[i] = headings[i-1] if i > 0 else 0.0

        # ---------- 画布 ----------
        plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                             'mathtext.fontset': 'cm'})
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # 静态元素: 障碍物、目标
        for obs in self.obstacles:
            obs.show_obstacle(ax)
        for i, target in enumerate(self.targets):
            target.show_target(ax, i)

        # 起始位置
        start_x = x_pts[0] if has_path else self.start_pos[0]
        start_y = y_pts[0] if has_path else self.start_pos[1]
        ax.plot(start_x, start_y, 'r*', markersize=12, zorder=8)

        # 海流 Quiver（稀疏采样，时变更新）
        step = 3
        Xs, Ys = self.X[::step, ::step], self.Y[::step, ::step]
        U0, V0 = self._generate_current_data(t=0)
        Us0 = U0[:100, :100].T[::step, ::step]
        Vs0 = V0[:100, :100].T[::step, ::step]
        quiver = ax.quiver(Xs, Ys, Us0, Vs0,
                           width=0.002, scale=25, color='#7FB3D8', alpha=0.5, zorder=1)

        # 轨迹相关动态元素
        trail_line = None
        _prev_artists = []
        obs_geoms = []
        if has_path:
            trail_line, = ax.plot([], [], color='#D62728', linewidth=1.2,
                                  alpha=0.85, zorder=5)
            obs_geoms = [ob.geometry for ob in self.obstacles]

        # 坐标轴设置
        title = ax.set_title('', fontsize=11, pad=8)
        ax.set_xlabel('$x$ / m')
        ax.set_ylabel('$y$ / m')
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_aspect('equal')
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.tick_params(direction='in', length=3)

        # ---------- 帧数计算 ----------
        if has_path:
            total_frames = (len(x_pts) - 1) // speed + 1
        else:
            total_frames = int(duration / self.dt)

        # ---------- 更新函数 ----------
        def _update(frame):
            nonlocal _prev_artists

            # 1) 更新时变海流
            if has_path:
                t = frame * speed * self.dt
            else:
                t = frame * self.dt
            U, V = self._generate_current_data(t=t)
            Us = U[:100, :100].T[::step, ::step]
            Vs = V[:100, :100].T[::step, ::step]
            quiver.set_UVC(Us, Vs)

            # 2) 更新轨迹 & USV
            if has_path:
                idx = min(frame * speed, len(x_pts) - 1)

                # 尾迹
                s = max(0, idx - trail_length) if trail_length > 0 else 0
                trail_line.set_data(x_pts[s:idx+1], y_pts[s:idx+1])

                # 移除上一帧 USV artists
                for a in _prev_artists:
                    a.remove()
                _prev_artists.clear()

                n_patches = len(ax.patches)
                n_lines = len(ax.lines)

                usv_now = USV(x_pts[idx], y_pts[idx], heading=headings[idx],
                              length=self.usv_length, width=self.usv_width)
                usv_now.detect_obstacles(obs_geoms)
                usv_now.show_usv(ax, color='#1F77B4', alpha=0.85)

                _prev_artists.extend(ax.patches[n_patches:])
                _prev_artists.extend(ax.lines[n_lines:])

                title.set_text(f'Step {idx}/{len(x_pts)-1}  '
                               f'$t={t:.1f}$  pos=({x_pts[idx]:.1f}, {y_pts[idx]:.1f})')
            else:
                title.set_text(f'Time-varying currents ($t={t:.2f}$)')

            return quiver,

        anim = FuncAnimation(fig, _update, frames=total_frames,
                             interval=interval, blit=False, repeat=False)

        if save_path is not None:
            try:
                anim.save(save_path, writer='pillow')
                print(f"动画已保存至 {save_path}  ({total_frames} 帧)")
            except Exception as e:
                print(f"保存动画失败: {e}")
            plt.close()
        else:
            plt.show()
    
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
            usv_interval: 每隔多少个点绘制一个USV形状
        """
        plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                             'mathtext.fontset': 'cm'})
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # 海流场（稀疏箭头）
        step = 3
        Xs, Ys = self.X[::step, ::step], self.Y[::step, ::step]
        Us = self.U[:100, :100].T[::step, ::step]
        Vs = self.V[:100, :100].T[::step, ::step]
        ax.quiver(Xs, Ys, Us, Vs, color='#7FB3D8', alpha=0.4,
                  width=0.002, scale=25, zorder=1)
        
        # 障碍物 & 目标
        for obs in self.obstacles:
            obs.show_obstacle(ax)
        for i, target in enumerate(self.targets):
            target.show_target(ax, i)
        
        try:
            path_colors = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD',
                           '#8C564B', '#E377C2', '#17BECF', '#BCBD22', '#7F7F7F']
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
                
                x_points, y_points, headings = [], [], []
                with open(episode_file, 'r') as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                        has_heading = len(header) >= 3 and 'heading' in header[2].lower()
                    except StopIteration:
                        continue
                    for row in reader:
                        if len(row) >= 2 and row[0] and row[1]:
                            try:
                                x, y = float(row[0]), float(row[1])
                                if X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX:
                                    x_points.append(x)
                                    y_points.append(y)
                                    headings.append(float(row[2]) if len(row) >= 3 and row[2] else 0.0)
                            except ValueError:
                                continue
                
                if len(x_points) < 2:
                    continue
                
                if not has_heading or all(h == 0.0 for h in headings):
                    for i in range(len(x_points)):
                        if i < len(x_points) - 1:
                            headings[i] = np.arctan2(y_points[i+1]-y_points[i],
                                                     x_points[i+1]-x_points[i])
                        else:
                            headings[i] = headings[i-1] if i > 0 else 0.0
                
                color = path_colors[(episode-1) % len(path_colors)] if max_episode > 1 else '#D62728'
                label = f'Episode {episode}' if max_episode > 1 else 'Path'
                
                # 路径线
                ax.plot(x_points, y_points, color=color, linewidth=1.2,
                        alpha=0.85, zorder=5, label=label)
                
                # USV形状标记
                if show_usv_shape:
                    for i in range(0, len(x_points), usv_interval):
                        usv_m = USV(x_points[i], y_points[i], heading=headings[i],
                                    length=self.usv_length*0.6, width=self.usv_width*0.6)
                        verts = usv_m.get_vertices()
                        ax.add_patch(MplPolygon(verts, closed=True, facecolor=color,
                                               edgecolor='k', alpha=0.5,
                                               linewidth=0.4, zorder=6))
                
                obs_geoms = [obs.geometry for obs in self.obstacles]
                
                # 起点
                start_usv = USV(x_points[0], y_points[0], heading=headings[0],
                               length=self.usv_length, width=self.usv_width)
                start_usv.detect_obstacles(obs_geoms)
                start_usv.show_usv(ax, color='#2CA02C', alpha=0.8)
                
                # 终点
                end_usv = USV(x_points[-1], y_points[-1], heading=headings[-1],
                             length=self.usv_length, width=self.usv_width)
                end_usv.detect_obstacles(obs_geoms)
                end_usv.show_usv(ax, color='#D62728', alpha=0.8)
                
                paths_drawn += 1
            
            if paths_drawn == 0:
                print("No valid paths to draw")
                return
            
            legend_elements = [
                MplPolygon([(0,0)], closed=True, facecolor='#2CA02C',
                           edgecolor='k', alpha=0.8, label='Start'),
                MplPolygon([(0,0)], closed=True, facecolor='#D62728',
                           edgecolor='k', alpha=0.8, label='End'),
                Line2D([0],[0], marker='*', color='#2CA02C', markersize=8,
                       label='Target', linewidth=0),
                Patch(facecolor='#A0A0A0', edgecolor='#404040', label='Obstacle'),
                Patch(facecolor='none', edgecolor='#D62728',
                      linestyle='--', label='Danger zone'),
                Line2D([0],[0], color='#7FB3D8', alpha=0.5,
                       linewidth=1.5, label='Current'),
                Line2D([0],[0], color=path_colors[0], linewidth=1.2,
                       label='Path'),
            ]
            ax.legend(handles=legend_elements, loc='upper center',
                      bbox_to_anchor=(0.5, -0.04), ncol=4,
                      fontsize=9, frameon=True, edgecolor='gray',
                      framealpha=1.0, handletextpad=0.4, columnspacing=1.0)
            
            ax.set_title(f'Path planning  (targets={self.num_targets}, '
                         f'obstacles={self.num_obstacles}, seed={self.seed})',
                         fontsize=11, pad=8)
            ax.set_xlabel('$x$ / m')
            ax.set_ylabel('$y$ / m')
            ax.grid(True, linewidth=0.3, alpha=0.4)
            ax.set_xlim(X_MIN, X_MAX)
            ax.set_ylim(Y_MIN, Y_MAX)
            ax.set_aspect('equal')
            ax.tick_params(direction='in', length=3)
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            
            if ifsave:
                save_filename = (f"all_episodes_paths.png" if max_episode > 1
                                 else f"{os.path.basename(csv_file).split('.')[0]}.png")
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

    # ==================== HGG 辅助接口 ====================
    # 以下方法为 HGG (Hindsight Goal Generation) 算法提供支持
    # 参考: Ren et al., "Exploration via Hindsight Goal Generation", NeurIPS 2019

    @staticmethod
    def phi(state):
        """
        状态到目标空间的映射函数 φ: S → G
        HGG 论文 Eq.(1) 中的 φ(s)
        
        对于 USV 导航任务，目标空间为归一化的 2D 位置 [0,1]²
        从观测状态中提取归一化的 (x, y) 坐标
        
        Args:
            state: 观测状态向量 (o_usv 部分)，前两维为 x/X_MAX, y/Y_MAX
        
        Returns:
            np.ndarray: 归一化的 [x, y] 位置，shape=(2,)
        """
        return np.array(state[:2], dtype=np.float32)

    @staticmethod
    def goal_distance(g1, g2):
        """
        计算两个目标之间的欧氏距离（归一化空间）
        用于 HGG 的 Wasserstein 距离计算
        
        Args:
            g1: 目标1，shape=(2,)
            g2: 目标2，shape=(2,)
        
        Returns:
            float: L2 距离
        """
        return np.linalg.norm(np.array(g1) - np.array(g2))

    def sample_task(self, target_idx=None):
        """
        从任务分布 T* 中采样一个 (s0, g) 对
        用于 HGG Algorithm 1 的 Step 4: Sample {(ŝ₀ⁱ, ĝⁱ)} ~ T*
        
        在 USV 多目标导航场景中:
        - s0 = 当前 USV 的归一化起始位置
        - g  = 某个目标点的归一化位置
        
        Args:
            target_idx: 指定目标索引。若为 None，则随机选择一个未访问的目标
        
        Returns:
            s0: 归一化的初始位置 np.ndarray, shape=(2,)
            g:  归一化的目标位置 np.ndarray, shape=(2,)
        """
        # 初始位置：当前 USV 归一化坐标
        s0 = np.array([self.usv.x / X_MAX, self.usv.y / Y_MAX], dtype=np.float32)
        
        if target_idx is not None:
            # 使用指定目标
            target = self.targets[target_idx]
        else:
            # 随机选择一个未访问的目标
            unvisited = [i for i in range(self.num_targets) if self.target_status[i] == 0]
            if len(unvisited) > 0:
                idx = np.random.choice(unvisited)
                target = self.targets[idx]
            else:
                # 所有目标已访问，随机选一个
                target = self.targets[np.random.randint(0, self.num_targets)]
        
        g = np.array([target.x / X_MAX, target.y / Y_MAX], dtype=np.float32)
        return s0, g

    def sample_tasks_batch(self, K, target_idx=None):
        """
        批量采样 K 个 (s0, g) 对，用于 HGG 的二部图匹配
        
        在 USV 场景中，同一个 episode 内初始位置相同，
        但可以对目标位置添加微小扰动以增加多样性
        
        Args:
            K: 采样数量
            target_idx: 指定目标索引（用于 Upper 层 HGG）
        
        Returns:
            s0_batch: shape=(K, 2), 归一化初始位置
            g_batch:  shape=(K, 2), 归一化目标位置
        """
        s0_list = []
        g_list = []
        for _ in range(K):
            s0, g = self.sample_task(target_idx)
            s0_list.append(s0)
            g_list.append(g)
        return np.array(s0_list, dtype=np.float32), np.array(g_list, dtype=np.float32)


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
    # test_usv_running_and_plot()
    print("=" * 60)
    print("测试1: 使用随机种子生成环境")
    print("=" * 60)
    
    # 创建带有随机种子的环境
    env = MyUsvEnv(
        num_targets=5,
        num_obstacles=9,
        seed=42,
        record_flag=False,
        usv_length=3.0,
        usv_width=1.5,
        use_random_shapes=True
    )
    
    # 显示环境信息
    info = env.get_environment_info()
    print(f"目标数量: {info['num_targets']}")
    print(f"障碍物数量: {info['num_obstacles']}")
    print(f"随机种子: {info['seed']}")
    print(f"起始位置: {info['start_pos']}")
    print(f"USV尺寸: {info['usv_size']}")
    print("\n目标位置:")
    for i, t in enumerate(info['targets']):
        print(f"  目标{i}: ({t['x']:.1f}, {t['y']:.1f}), 半径={t['radius']}")
    print("\n障碍物信息:")
    for i, o in enumerate(info['obstacles']):
        print(f"  障碍物{i}: 形状={o['shape']}, 位置=({o['x']:.1f}, {o['y']:.1f})")
    
    # 可视化环境
    env.visualize_env()
    
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
