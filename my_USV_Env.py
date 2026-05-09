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
from colregs_module import (
    EncounterType, NUM_ENCOUNTER_TYPES, OTHER_USV_OBS_DIM,
    ShipDomain, compute_full_colregs_assessment,
    compute_colregs_penalty, compute_colregs_reward,
    get_other_usv_obs_features, normalize_angle
)

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
            
            if flow_speed > 1e-8:
                o_current = np.array([
                    flow_u ,
                    flow_v ,
                    np.sin(flow_angle),
                    np.cos(flow_angle)
                ], dtype=np.float32)
            else:
                o_current = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            
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
                   heading=self.rng.uniform(0.0, 2 * np.pi), length=self.usv_length, width=self.usv_width)
        
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
        # 保存 clip 前的归一化位置 (用于边界惩罚计算)
        usv_pos_raw_norm = np.array([self.usv.x / X_MAX, self.usv.y / Y_MAX])
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
            "target_status": self.target_status.copy(),
            "usv_pos_raw_norm": usv_pos_raw_norm
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
    
    def visualize_env(self, time_step=None, save_path=None):
        """可视化环境

        Args:
            time_step: 指定时间步用于时变海流，None 则用稳态海流
            save_path: 保存路径，None 则仅显示不保存
        """
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 10,
            'mathtext.fontset': 'cm',
        })
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
                  bbox_to_anchor=(0.5, -0.08), ncol=6,
                  fontsize=8.5, frameon=True, edgecolor='gray',
                  framealpha=1.0, handletextpad=0.3, columnspacing=0.8)
        
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
        fig.subplots_adjust(bottom=0.14)   # 手动预留底部空间给图例
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            base, _ = os.path.splitext(save_path)
            pdf_path = base + ".pdf"
            fig.savefig(pdf_path, format="pdf")
            print(f"Scene saved: {save_path}")
            print(f"Scene saved: {pdf_path}")
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
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 10,
            'mathtext.fontset': 'cm',
        })
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
                anim.save(save_path, writer='pillow', dpi=300)
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
                self.csv_file = open(csv_file, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
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
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 10,
            'mathtext.fontset': 'cm',
        })
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
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            
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


class MultiUsvEnv(MyUsvEnv):
    """多USV环境: 继承单USV环境并扩展为并行多艇仿真。"""

    def __init__(self,
                 num_usvs=2,
                 min_start_separation=8.0,
                 usv_collision_buffer=0.5,
                 per_usv_max_steps=2000,
                 fixed_start_positions=None,
                 colregs_k_C=5.0,
                 colregs_k_R=1.0,
                 starboard_threshold=-0.05,
                 **kwargs):
        self.num_usvs = int(max(1, num_usvs))
        self.min_start_separation = float(max(0.0, min_start_separation))
        self.usv_collision_buffer = float(max(0.0, usv_collision_buffer))
        self.per_usv_max_steps = int(max(1, per_usv_max_steps))
        self.fixed_start_positions = fixed_start_positions

        self.usvs = []
        self.start_positions = []
        self.current_positions = None
        self.per_usv_steps = np.zeros(self.num_usvs, dtype=np.int32)

        super().__init__(**kwargs)

        # --- COLREGs Module ---
        usv_bounding = max(self.usv_length, self.usv_width) / 2 * 1.5
        self.ship_domain = ShipDomain(r_min=max(4.0, usv_bounding * 2))
        self.colregs_matrix = {}          # (i,j) → colregs assessment dict
        self.colregs_k_C = float(colregs_k_C)
        self.colregs_k_R = float(colregs_k_R)
        self.starboard_threshold = float(starboard_threshold)
        self._other_usv_obs_dim = OTHER_USV_OBS_DIM  # 16 per other USV

        # Build observation space (extended with other-USV features)
        self.single_observation_space = self.observation_space
        other_usvs_dim = max(0, self.num_usvs - 1) * self._other_usv_obs_dim
        if other_usvs_dim > 0:
            spaces = dict(self.single_observation_space.spaces)
            spaces["o_other_usvs"] = Box(
                low=-np.inf, high=np.inf,
                shape=(other_usvs_dim,), dtype=np.float32)
            self.single_observation_space = Dict(spaces)

        self.observation_space = Dict({
            f"usv_{i}": self.single_observation_space for i in range(self.num_usvs)
        })
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_usvs, 2),
            dtype=np.float32
        )

    def _generate_start_positions(self):
        """生成多USV起点, 要求避开障碍并满足艇间最小间距。"""
        if self.fixed_start_positions is not None:
            if len(self.fixed_start_positions) != self.num_usvs:
                raise ValueError("fixed_start_positions length must equal num_usvs")
            return [np.array(p, dtype=np.float32) for p in self.fixed_start_positions]

        usv_bounding_radius = max(self.usv_length, self.usv_width) / 2 * 1.2
        margin = usv_bounding_radius + self.rou_c + self.rou_s + 5
        max_attempts = 400
        starts = []

        for _ in range(self.num_usvs):
            found = False
            for _ in range(max_attempts):
                x = self.rng.uniform(margin, X_MAX - margin)
                y = self.rng.uniform(margin, Y_MAX - margin)
                candidate = np.array([x, y], dtype=np.float32)

                valid = True
                for obs in self.obstacles:
                    if obs.is_point_in_safety_zone(candidate):
                        valid = False
                        break

                if valid:
                    for existing in starts:
                        if np.linalg.norm(candidate - existing) < self.min_start_separation:
                            valid = False
                            break

                if valid:
                    starts.append(candidate)
                    found = True
                    break

            if not found:
                starts.append(np.array([5.0 + len(starts) * 3.0, 5.0], dtype=np.float32))

        return starts

    def _generate_environment(self):
        """生成多USV环境配置（共享障碍物和目标，多起点）。"""
        if self.fixed_obstacles is not None:
            self.obstacles = self._create_fixed_obstacles(self.fixed_obstacles)
        else:
            self.obstacles = self._generate_obstacles()

        self.start_positions = self._generate_start_positions()
        self.start_pos = self.start_positions[0].copy()  # 兼容旧接口

        self.usvs = []
        for pos in self.start_positions:
            usv = USV(
                pos[0], pos[1],
                heading=self.rng.uniform(0.0, 2 * np.pi),
                length=self.usv_length,
                width=self.usv_width
            )
            self.usvs.append(usv)

        self.usv = self.usvs[0]  # 兼容旧接口

        if self.fixed_targets is not None:
            self.targets = self._create_fixed_targets(self.fixed_targets)
        else:
            self.targets = self._generate_targets()

        self.current_positions = np.array(self.start_positions, dtype=np.float32)
        self.current_pos = self.current_positions[0].copy()  # 兼容旧接口
        self.target_status = np.zeros(self.num_targets, dtype=np.float32)
        self.per_usv_steps = np.zeros(self.num_usvs, dtype=np.int32)

    def _get_single_observation(self, usv_idx):
        usv = self.usvs[usv_idx]

        x, y = usv.x, usv.y
        heading = usv.heading
        u, v, r = usv.u, usv.v, usv.r
        flow_u, flow_v = self.get_flow_velocity(x, y, self.current_time)

        o_usv = np.array([
            x / X_MAX,
            y / Y_MAX,
            u / usv.max_u,
            v / usv.max_v,
            r / usv.max_r,
            np.sin(heading),
            np.cos(heading)
        ], dtype=np.float32)

        targets_info = []
        for i, target in enumerate(self.targets):
            targets_info.extend([
                target.x / X_MAX,
                target.y / Y_MAX,
                self.target_status[i]
            ])
        o_targets = np.array(targets_info, dtype=np.float32)

        flow_speed = np.sqrt(flow_u ** 2 + flow_v ** 2)
        flow_angle = np.arctan2(flow_v, flow_u)
        if flow_speed > 1e-8:
            o_current = np.array([
                flow_u / flow_speed,
                flow_v / flow_speed,
                np.sin(flow_angle),
                np.cos(flow_angle)
            ], dtype=np.float32)
        else:
            o_current = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        readings = np.array(usv.sensor_readings, dtype=np.float32)
        o_detector = readings / usv.sensor_range

        obs = {
            "o_usv": o_usv,
            "targets": o_targets,
            "o_current": o_current,
            "o_detector": o_detector
        }

        # 多 USV: 添加他船 COLREGs 观测
        if self.num_usvs > 1:
            obs["o_other_usvs"] = self._get_other_usvs_obs(usv_idx)

        return obs

    def _get_observation(self):
        return {f"usv_{i}": self._get_single_observation(i) for i in range(self.num_usvs)}

    def _update_sensors(self):
        static_obstacles = [obs.geometry for obs in self.obstacles]
        for i, usv in enumerate(self.usvs):
            dynamic_obstacles = []
            for j, other in enumerate(self.usvs):
                if i == j or other.polygon is None:
                    continue
                dynamic_obstacles.append(other.polygon.buffer(self.usv_collision_buffer))
            usv.detect_obstacles(static_obstacles + dynamic_obstacles)

    def _update_colregs_matrix(self):
        """计算所有 USV 两两之间的 COLREGs 评估 (CRI / 会遇类型等)。"""
        self.colregs_matrix = {}
        for i in range(self.num_usvs):
            for j in range(self.num_usvs):
                if i == j:
                    continue
                self.colregs_matrix[(i, j)] = compute_full_colregs_assessment(
                    self.usvs[i], self.usvs[j], self.ship_domain)

    def _get_other_usvs_obs(self, usv_idx):
        """为 usv_idx 构建所有他船的 COLREGs 观测特征向量。"""
        features = []
        own = self.usvs[usv_idx]
        for j in range(self.num_usvs):
            if j == usv_idx:
                continue
            ci = self.colregs_matrix.get((usv_idx, j))
            if ci is None:
                # 尚未计算 (不应发生), 用零向量填充
                features.append(np.zeros(self._other_usv_obs_dim, dtype=np.float32))
            else:
                features.append(get_other_usv_obs_features(
                    own, self.usvs[j], ci, x_max=X_MAX, y_max=Y_MAX))
        if len(features) == 0:
            return np.array([], dtype=np.float32)
        return np.concatenate(features)

    def _update_global_target_status(self):
        for i, target in enumerate(self.targets):
            if self.target_status[i] == 1.0:
                continue
            target_pos = np.array([target.x, target.y], dtype=np.float32)
            for pos in self.current_positions:
                if np.linalg.norm(pos - target_pos) < target.radius_t:
                    self.target_status[i] = 1.0
                    break

    def _check_collisions(self):
        obstacle_collisions = [False] * self.num_usvs
        usv_collision_pairs = []

        for i, usv in enumerate(self.usvs):
            for obs in self.obstacles:
                if usv.is_colliding_with(obs.geometry):
                    obstacle_collisions[i] = True
                    break

        for i in range(self.num_usvs):
            for j in range(i + 1, self.num_usvs):
                ui = self.usvs[i]
                uj = self.usvs[j]
                d = np.linalg.norm(np.array([ui.x - uj.x, ui.y - uj.y], dtype=np.float32))
                thresh = ui.get_bounding_radius() + uj.get_bounding_radius() + self.usv_collision_buffer
                inter = False
                if ui.polygon is not None and uj.polygon is not None:
                    inter = ui.polygon.intersects(uj.polygon)
                if inter or d <= thresh:
                    usv_collision_pairs.append((i, j))

        return obstacle_collisions, usv_collision_pairs

    def reset(self, seed=None):
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

        self._update_sensors()
        self._update_colregs_matrix()

        # 趋近奖励归一化: 记录每艘 USV 到当前目标的初始距离
        self._goal_initial_dists = np.zeros(self.num_usvs, dtype=np.float32)
        self._prev_goal_positions = [None] * self.num_usvs

        return self._get_observation()

    def step(self, actions, goal_positions_global, usv_active_mask=None):
        """
        多USV并行步进。

        Args:
            actions: shape=(num_usvs,2) 的动作数组
            goal_positions_global: 每艘USV当前目标点（全局坐标）列表
            usv_active_mask: 可选激活掩码。False 表示该USV已完成序列，不再推进
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_usvs, 2):
            raise ValueError(f"actions shape must be ({self.num_usvs}, 2)")
        if len(goal_positions_global) != self.num_usvs:
            raise ValueError("goal_positions_global length must equal num_usvs")

        if usv_active_mask is None:
            active_mask = np.ones(self.num_usvs, dtype=bool)
        else:
            active_mask = np.asarray(usv_active_mask, dtype=bool)
            if active_mask.shape != (self.num_usvs,):
                raise ValueError(f"usv_active_mask shape must be ({self.num_usvs},)")

        self.step_counter += 1
        self.current_time += self.dt

        usv_pos_raw_norm = np.zeros((self.num_usvs, 2), dtype=np.float32)
        reached_goals = [False] * self.num_usvs

        for i, usv in enumerate(self.usvs):
            if not active_mask[i]:
                # 非活跃USV：不执行动力学推进，但仍受海流漂移影响
                flow_u, flow_v = self.get_flow_velocity(usv.x, usv.y, self.current_time)
                usv.x += flow_u * self.dt
                usv.y += flow_v * self.dt
                usv.x = np.clip(usv.x, X_MIN, X_MAX)
                usv.y = np.clip(usv.y, Y_MIN, Y_MAX)
                usv._update_polygon()
                self.current_positions[i] = np.array([usv.x, usv.y], dtype=np.float32)
                continue

            action = actions[i]
            throttle = float(action[0])
            rudder = float(action[1] * np.radians(60))

            usv.step(throttle, rudder)

            flow_u, flow_v = self.get_flow_velocity(usv.x, usv.y, self.current_time)
            usv.x += flow_u * self.dt
            usv.y += flow_v * self.dt

            usv_pos_raw_norm[i] = np.array([usv.x / X_MAX, usv.y / Y_MAX], dtype=np.float32)

            usv.x = np.clip(usv.x, X_MIN, X_MAX)
            usv.y = np.clip(usv.y, Y_MIN, Y_MAX)
            usv._update_polygon()

            self.current_positions[i] = np.array([usv.x, usv.y], dtype=np.float32)
            self.per_usv_steps[i] += 1

            goal = np.array(goal_positions_global[i], dtype=np.float32)
            reached_goals[i] = np.linalg.norm(self.current_positions[i] - goal) < self.target_radius

        self.current_pos = self.current_positions[0].copy()
        self.usv = self.usvs[0]

        self._update_global_target_status()
        self._update_sensors()
        self._update_colregs_matrix()
        obstacle_collisions, usv_collision_pairs = self._check_collisions()

        crush = any(obstacle_collisions) or len(usv_collision_pairs) > 0

        all_sequence_done = np.all(~active_mask)

        active_indices = np.where(active_mask)[0]
        if len(active_indices) == 0:
            all_active_timeout = False
        else:
            all_active_timeout = all(self.per_usv_steps[i] >= self.per_usv_max_steps for i in active_indices)

        done = crush or all_sequence_done or all_active_timeout
        truncated = bool(all_active_timeout and not crush)

        if crush:
            termination_reason = "collision"
        elif all_sequence_done:
            termination_reason = "all_sequence_completed"
        elif all_active_timeout:
            termination_reason = "all_active_timeout"
        else:
            termination_reason = "running"

        info = {
            "obstacle_collisions": obstacle_collisions,
            "usv_collision_pairs": usv_collision_pairs,
            "reached_goals": reached_goals,
            "target_status": self.target_status.copy(),
            "per_usv_steps": self.per_usv_steps.copy(),
            "usv_positions": self.current_positions.copy(),
            "usv_pos_raw_norm": usv_pos_raw_norm,
            "termination_reason": termination_reason
        }

        return self._get_observation(), done, crush, truncated, info

    # ================================================================
    # ==================== CTDE (MARL) Interface =====================
    # ================================================================

    def get_global_state(self):
        """
        获取全局状态向量, 供 Centralized Critic 使用 (MAPPO / QMIX / MADDPG 等).

        Layout:
          - 所有 USV 状态: 7 × num_usvs
          - 所有目标信息:  3 × num_targets
          - 海流信息:      4
          - 两两 CRI:      num_usvs × (num_usvs - 1)

        Returns:
            np.ndarray, shape = (7*N + 3*T + 4 + N*(N-1),)
        """
        parts = []

        # 所有 USV 归一化状态
        for usv in self.usvs:
            parts.append(np.array([
                usv.x / X_MAX, usv.y / Y_MAX,
                usv.u / usv.max_u, usv.v / usv.max_v, usv.r / usv.max_r,
                np.sin(usv.heading), np.cos(usv.heading)
            ], dtype=np.float32))

        # 所有目标信息
        for i, target in enumerate(self.targets):
            parts.append(np.array([
                target.x / X_MAX, target.y / Y_MAX, self.target_status[i]
            ], dtype=np.float32))

        # 代表性海流信息 (取第一艘 USV 位置)
        flow_u, flow_v = self.get_flow_velocity(
            self.usvs[0].x, self.usvs[0].y, self.current_time)
        flow_speed = np.sqrt(flow_u ** 2 + flow_v ** 2)
        flow_angle = np.arctan2(flow_v, flow_u)
        if flow_speed > 1e-8:
            parts.append(np.array([
                flow_u / flow_speed, flow_v / flow_speed,
                np.sin(flow_angle), np.cos(flow_angle)
            ], dtype=np.float32))
        else:
            parts.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        # 两两 CRI 矩阵 (i→j, 含方向性)
        for i in range(self.num_usvs):
            for j in range(self.num_usvs):
                if i != j:
                    ci = self.colregs_matrix.get((i, j), {})
                    parts.append(np.array([ci.get('cri', 0.0)], dtype=np.float32))

        return np.concatenate(parts)

    def get_global_state_dim(self):
        """返回全局状态维度 (方便外部初始化 Critic)。"""
        N, T = self.num_usvs, self.num_targets
        return 7 * N + 3 * T + 4 + N * (N - 1)

    def step_marl(self, actions, goal_positions_global, usv_active_mask=None):
        """
        CTDE 兼容的多 USV 步进, 返回 per-agent reward 和 per-agent done.

        相比 step(), 新增:
          1) per-agent reward (导航 + 避障 + COLREGs 合规)
          2) per-agent done / truncated
          3) info 中包含 COLREGs 详情

        Args:
            actions:               shape=(num_usvs, 2), 每艘 USV 的 (throttle, rudder)
            goal_positions_global: list/array, 每艘 USV 当前子目标的全局坐标
            usv_active_mask:       可选 bool 数组, False 表示该 USV 已完成

        Returns:
            obs_dict:       {"usv_0": {...}, ...}
            rewards_dict:   {"usv_0": float, ...}
            dones_dict:     {"usv_0": bool, ..., "__all__": bool}
            truncated_dict: {"usv_0": bool, ..., "__all__": bool}
            info:           dict (扩展了 colregs 和 reward 细节)
        """
        actions = np.asarray(actions, dtype=np.float32)
        goal_positions_global = [
            np.asarray(g, dtype=np.float32) for g in goal_positions_global]

        # ---------- 1. 步前计算: 各 USV 到目标距离 ----------
        prev_dists = np.zeros(self.num_usvs, dtype=np.float32)
        for i in range(self.num_usvs):
            prev_dists[i] = np.linalg.norm(
                self.current_positions[i] - goal_positions_global[i])

            # 检测目标是否变更 → 记录初始距离 (用于趋近奖励归一化)
            goal_changed = (self._prev_goal_positions[i] is None or
                            not np.array_equal(self._prev_goal_positions[i],
                                               goal_positions_global[i]))
            if goal_changed:
                self._goal_initial_dists[i] = max(prev_dists[i], 1.0)
                self._prev_goal_positions[i] = goal_positions_global[i].copy()

        # ---------- 2. 执行物理步进 (复用 step) ----------
        obs, done, crush, truncated, info = self.step(
            actions, goal_positions_global, usv_active_mask)

        # ---------- 3. 步后: per-agent reward ----------
        rewards_dict = {}
        reward_details = {}

        for i in range(self.num_usvs):
            r_i, details_i = self._compute_per_agent_reward(
                i, actions[i], goal_positions_global[i],
                prev_dists[i], self._goal_initial_dists[i], info, obs)
            rewards_dict[f"usv_{i}"] = r_i
            reward_details[i] = details_i

        # ---------- 4. per-agent done / truncated ----------
        dones_dict = {}
        truncated_dict = {}

        for i in range(self.num_usvs):
            # 碰撞 → 全体结束
            agent_done = crush
            # 也可选: 该 USV 自身的障碍碰撞
            # agent_done = info['obstacle_collisions'][i] or crush
            agent_truncated = truncated

            dones_dict[f"usv_{i}"] = agent_done
            truncated_dict[f"usv_{i}"] = agent_truncated

        dones_dict["__all__"] = done
        truncated_dict["__all__"] = truncated

        # ---------- 5. 扩展 info ----------
        info["rewards_dict"] = rewards_dict
        info["reward_details"] = reward_details
        info["colregs_matrix"] = {
            k: {kk: vv for kk, vv in v.items() if kk != 'encounter_type'}
            | {'encounter_type': int(v['encounter_type'])}
            for k, v in self.colregs_matrix.items()
        }
        info["global_state"] = self.get_global_state()

        return obs, rewards_dict, dones_dict, truncated_dict, info

    def _compute_per_agent_reward(self, usv_idx, action, goal_global,
                                  prev_dist, init_dist, info, obs):
        """
        计算单个 USV 的 CTDE reward.

        Components:
          r_approach:    归一化趋近子目标奖励 (Δdist / d_0 × R_max)
          r_reach:       到达子目标奖励
          r_obs_col:     静态障碍物碰撞惩罚
          r_usv_col:     USV 间碰撞惩罚
          r_colregs:     COLREGs 合规惩罚 (R_C = k_C × u_CRI)
          r_colregs_reward: COLREGs 合规正奖励
          r_proximity:   传感器接近惩罚
          r_boundary:    越界惩罚
          r_step:        时间步惩罚

        Returns:
            total_reward: float
            details: dict 各分项
        """
        # --- 归一化趋近奖励 ---
        # Δdist / d_0 × R_max: 无论距离远近, 完整趋近一个目标的总奖励恒为 R_max
        R_max = 5.0
        new_pos = info['usv_positions'][usv_idx]
        new_dist = np.linalg.norm(new_pos - goal_global)
        r_approach = (prev_dist - new_dist) / max(init_dist, 1.0) * R_max

        # --- 到达奖励 ---
        r_reach = 5.0 if info['reached_goals'][usv_idx] else 0.0

        # --- 静态障碍碰撞 ---
        r_obs_col = -50.0 if info['obstacle_collisions'][usv_idx] else 0.0

        # --- USV 间碰撞 ---
        usv_collided = any(
            usv_idx in pair for pair in info['usv_collision_pairs'])
        r_usv_col = -50.0 if usv_collided else 0.0

        # --- COLREGs 惩罚 + 合规奖励 ---
        r_colregs = 0.0
        r_colregs_reward = 0.0
        rudder_action = float(action[1])
        for j in range(self.num_usvs):
            if j == usv_idx:
                continue
            ci = self.colregs_matrix.get((usv_idx, j))
            if ci is None:
                continue
            r_colregs += compute_colregs_penalty(
                ci['encounter_type'], ci['cri'], rudder_action,
                k_C=self.colregs_k_C,
                starboard_threshold=self.starboard_threshold)
            r_colregs_reward += compute_colregs_reward(
                ci['encounter_type'], ci['cri'], rudder_action,
                k_R=self.colregs_k_R,
                starboard_threshold=self.starboard_threshold)

        # --- 传感器近距惩罚 ---
        obs_i = obs[f'usv_{usv_idx}']
        min_sensor = float(np.min(obs_i['o_detector']))
        r_proximity = -max(0.0, 0.3 - min_sensor) * 5.0

        # --- 越界惩罚 ---
        pos_norm = info['usv_pos_raw_norm'][usv_idx]
        r_boundary = 0.0
        for coord in pos_norm:
            if coord < 0.0:
                r_boundary -= abs(coord) * 10.0
            elif coord > 1.0:
                r_boundary -= (coord - 1.0) * 10.0

        # --- 时间步惩罚 ---
        r_step = -0.01

        total = (r_approach + r_reach + r_obs_col + r_usv_col +
                 r_colregs + r_colregs_reward + r_proximity +
                 r_boundary + r_step)

        details = {
            'approach': r_approach,
            'reach': r_reach,
            'obs_collision': r_obs_col,
            'usv_collision': r_usv_col,
            'colregs': r_colregs,
            'colregs_reward': r_colregs_reward,
            'proximity': r_proximity,
            'boundary': r_boundary,
            'step': r_step,
            'total': total,
        }
        return total, details

    # ================================================================
    # ==================== 轨迹记录与绘制 ============================
    # ================================================================

    def record_points(self, usv_idx=None, csv_prefix="multi_usv_path",
                      output_dir="."):
        """记录各USV当前位置和航向到独立的CSV文件。

        每个USV对应一个CSV: ``{output_dir}/{csv_prefix}_usv_{i}.csv``

        Args:
            usv_idx: 记录指定USV (int) 或 None 表示记录全部
            csv_prefix: 文件名前缀
            output_dir: 输出目录
        """
        if not hasattr(self, '_multi_csv_files'):
            self._multi_csv_files = {}    # usv_idx → file handle
            self._multi_csv_writers = {}  # usv_idx → csv.writer

        os.makedirs(output_dir, exist_ok=True)

        indices = range(self.num_usvs) if usv_idx is None else [usv_idx]

        for i in indices:
            usv = self.usvs[i]
            try:
                if i not in self._multi_csv_files:
                    fpath = os.path.join(output_dir,
                                         f"{csv_prefix}_usv_{i}.csv")
                    fh = open(fpath, 'w', newline='')
                    writer = csv.writer(fh)
                    writer.writerow(['x', 'y', 'heading'])
                    writer.writerow([self.start_positions[i][0],
                                     self.start_positions[i][1],
                                     usv.heading])
                    fh.flush()
                    self._multi_csv_files[i] = fh
                    self._multi_csv_writers[i] = writer

                self._multi_csv_writers[i].writerow(
                    [usv.x, usv.y, usv.heading])
                self._multi_csv_files[i].flush()
            except Exception as e:
                print(f"Error recording point for USV {i}: {e}")
                if i in self._multi_csv_files:
                    self._multi_csv_files[i].close()
                    del self._multi_csv_files[i]
                    del self._multi_csv_writers[i]

    def close_writers(self):
        """关闭所有USV的CSV文件写入器。"""
        if not hasattr(self, '_multi_csv_files'):
            return
        for i, fh in list(self._multi_csv_files.items()):
            fh.close()
        self._multi_csv_files.clear()
        self._multi_csv_writers.clear()

    def draw_path(self, csv_prefix="multi_usv_path", output_dir=".",
                  ifshow=True, ifsave=False, savePath=None,
                  show_usv_shape=False, usv_interval=50):
        """绘制多USV路径（所有USV共享同一张图）。

        Args:
            csv_prefix: 与 record_points 中一致的文件名前缀
            output_dir: CSV 所在目录
            ifshow: 是否显示图像
            ifsave: 是否保存图像
            savePath: 保存目录（默认使用 output_dir）
            show_usv_shape: 是否沿轨迹绘制USV形状
            usv_interval: USV形状的间隔步数
        """
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 10,
            'mathtext.fontset': 'cm',
        })
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # 海流场
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

        usv_colors = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD',
                       '#8C564B', '#E377C2', '#17BECF', '#BCBD22', '#7F7F7F']
        obs_geoms = [obs.geometry for obs in self.obstacles]
        paths_drawn = 0

        for idx in range(self.num_usvs):
            csv_file = os.path.join(output_dir,
                                    f"{csv_prefix}_usv_{idx}.csv")
            if not os.path.exists(csv_file):
                print(f"Path file {csv_file} does not exist")
                continue

            x_pts, y_pts, headings = [], [], []
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                        has_heading = (len(header) >= 3
                                       and 'heading' in header[2].lower())
                    except StopIteration:
                        continue
                    for row in reader:
                        if len(row) >= 2 and row[0] and row[1]:
                            try:
                                x, y = float(row[0]), float(row[1])
                                if X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX:
                                    x_pts.append(x)
                                    y_pts.append(y)
                                    headings.append(
                                        float(row[2])
                                        if len(row) >= 3 and row[2] else 0.0)
                            except ValueError:
                                continue
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue

            if len(x_pts) < 2:
                continue
            if not has_heading or all(h == 0.0 for h in headings):
                for i in range(len(x_pts)):
                    if i < len(x_pts) - 1:
                        headings[i] = np.arctan2(
                            y_pts[i + 1] - y_pts[i],
                            x_pts[i + 1] - x_pts[i])
                    else:
                        headings[i] = headings[i - 1] if i > 0 else 0.0

            color = usv_colors[idx % len(usv_colors)]
            label = f'USV {idx}'

            # 路径线
            ax.plot(x_pts, y_pts, color=color, linewidth=1.2,
                    alpha=0.85, zorder=5, label=label)

            # USV形状标记
            if show_usv_shape:
                for i in range(0, len(x_pts), usv_interval):
                    usv_m = USV(x_pts[i], y_pts[i], heading=headings[i],
                                length=self.usv_length * 0.6,
                                width=self.usv_width * 0.6)
                    verts = usv_m.get_vertices()
                    ax.add_patch(MplPolygon(verts, closed=True,
                                            facecolor=color, edgecolor='k',
                                            alpha=0.5, linewidth=0.4,
                                            zorder=6))

            # 起点
            start_usv = USV(x_pts[0], y_pts[0], heading=headings[0],
                            length=self.usv_length, width=self.usv_width)
            start_usv.detect_obstacles(obs_geoms)
            start_usv.show_usv(ax, color=color, alpha=0.8)
            ax.annotate(f'S{idx}', (x_pts[0], y_pts[0]),
                        fontsize=8, fontweight='bold', color=color,
                        textcoords="offset points", xytext=(5, 5),
                        zorder=10)

            # 终点
            end_usv = USV(x_pts[-1], y_pts[-1], heading=headings[-1],
                          length=self.usv_length, width=self.usv_width)
            end_usv.detect_obstacles(obs_geoms)
            end_usv.show_usv(ax, color=color, alpha=0.5)
            ax.annotate(f'E{idx}', (x_pts[-1], y_pts[-1]),
                        fontsize=8, fontweight='bold', color=color,
                        textcoords="offset points", xytext=(5, -10),
                        zorder=10)

            paths_drawn += 1

        if paths_drawn == 0:
            print("No valid USV paths to draw")
            plt.close(fig)
            return

        # 图例
        legend_elements = [
            Line2D([0], [0], color=usv_colors[i % len(usv_colors)],
                   linewidth=1.5, label=f'USV {i}')
            for i in range(min(paths_drawn, self.num_usvs))
        ] + [
            Line2D([0], [0], marker='*', color='#2CA02C', markersize=8,
                   label='Target', linewidth=0),
            Patch(facecolor='#A0A0A0', edgecolor='#404040', label='Obstacle'),
            Patch(facecolor='none', edgecolor='#D62728',
                  linestyle='--', label='Danger zone'),
            Line2D([0], [0], color='#7FB3D8', alpha=0.5,
                   linewidth=1.5, label='Current'),
        ]
        ax.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, -0.04), ncol=4,
                  fontsize=9, frameon=True, edgecolor='gray',
                  framealpha=1.0, handletextpad=0.4, columnspacing=1.0)

        ax.set_title(f'Multi-USV Path ({self.num_usvs} USVs, '
                     f'{self.num_targets}T, {self.num_obstacles}O, '
                     f'seed={self.seed})', fontsize=11, pad=8)
        ax.set_xlabel('$x$ / m')
        ax.set_ylabel('$y$ / m')
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_aspect('equal')
        ax.tick_params(direction='in', length=3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        if ifsave:
            save_dir = savePath or output_dir
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(
                save_dir,
                f"{csv_prefix}_{self.num_usvs}usv_path.png")
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Image saved to {save_file}")
        if ifshow:
            plt.show()
        else:
            plt.close(fig)

    def animate(self, csv_prefix="multi_usv_path", output_dir=".",
                duration=20, interval=100, save_path=None,
                trail_length=0, speed=5):
        """多USV轨迹动画回放（时变海流 + 多艇同步运动）。

        每个USV的CSV文件命名: ``{output_dir}/{csv_prefix}_usv_{i}.csv``

        Args:
            csv_prefix:   CSV 文件名前缀 (与 record_points 一致)
            output_dir:   CSV 所在目录
            duration:     仅海流模式下的动画时长
            interval:     帧间隔 (ms)
            save_path:    GIF 保存路径, None 则直接显示
            trail_length: 尾迹保留长度 (0 = 全部)
            speed:        每帧前进的点数 (加速回放)
        """
        from matplotlib.animation import FuncAnimation

        if self.current_type != "generated":
            print("警告：仅在 'generated' 模式下支持时变海流动画")
            return

        usv_colors = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD',
                       '#8C564B', '#E377C2', '#17BECF', '#BCBD22', '#7F7F7F']

        # ---------- 读取各 USV 的 CSV ----------
        all_pts = {}  # idx -> (x_pts, y_pts, headings)
        for idx in range(self.num_usvs):
            csv_file = os.path.join(output_dir,
                                    f"{csv_prefix}_usv_{idx}.csv")
            if not os.path.exists(csv_file):
                print(f"CSV 文件不存在: {csv_file}")
                continue
            x_pts, y_pts, headings = [], [], []
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                    has_heading = (len(header) >= 3
                                   and 'heading' in header[2].lower())
                except StopIteration:
                    continue
                for row in reader:
                    if len(row) >= 2 and row[0] and row[1]:
                        try:
                            x, y = float(row[0]), float(row[1])
                            x_pts.append(x)
                            y_pts.append(y)
                            headings.append(
                                float(row[2])
                                if len(row) >= 3 and row[2] else 0.0)
                        except ValueError:
                            continue
            if len(x_pts) < 2:
                continue
            if not has_heading or all(h == 0.0 for h in headings):
                for i in range(len(x_pts)):
                    if i < len(x_pts) - 1:
                        headings[i] = np.arctan2(
                            y_pts[i + 1] - y_pts[i],
                            x_pts[i + 1] - x_pts[i])
                    else:
                        headings[i] = headings[i - 1] if i > 0 else 0.0
            all_pts[idx] = (x_pts, y_pts, headings)

        has_path = len(all_pts) > 0
        if not has_path:
            print("未找到任何有效的 USV 轨迹 CSV, 仅播放海流")

        # ---------- 画布 ----------
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 10,
            'mathtext.fontset': 'cm',
        })
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        for obs in self.obstacles:
            obs.show_obstacle(ax)
        for i, target in enumerate(self.targets):
            target.show_target(ax, i)

        # 起始位置标注
        for idx in all_pts:
            x0, y0 = all_pts[idx][0][0], all_pts[idx][1][0]
            color = usv_colors[idx % len(usv_colors)]
            ax.plot(x0, y0, '*', color=color, markersize=10, zorder=8)
            ax.annotate(f'S{idx}', (x0, y0), fontsize=7,
                        fontweight='bold', color=color,
                        textcoords="offset points", xytext=(4, 4),
                        zorder=10)

        # 海流 Quiver
        qstep = 3
        Xs, Ys = self.X[::qstep, ::qstep], self.Y[::qstep, ::qstep]
        U0, V0 = self._generate_current_data(t=0)
        Us0 = U0[:100, :100].T[::qstep, ::qstep]
        Vs0 = V0[:100, :100].T[::qstep, ::qstep]
        quiver = ax.quiver(Xs, Ys, Us0, Vs0,
                           width=0.002, scale=25, color='#7FB3D8',
                           alpha=0.5, zorder=1)

        # 轨迹线 & USV 动态元素
        trail_lines = {}
        obs_geoms = [ob.geometry for ob in self.obstacles]
        if has_path:
            for idx in all_pts:
                color = usv_colors[idx % len(usv_colors)]
                line, = ax.plot([], [], color=color, linewidth=1.2,
                                alpha=0.85, zorder=5)
                trail_lines[idx] = line

        title = ax.set_title('', fontsize=11, pad=8)
        ax.set_xlabel('$x$ / m')
        ax.set_ylabel('$y$ / m')
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_aspect('equal')
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.tick_params(direction='in', length=3)

        # ---------- 帧数 ----------
        if has_path:
            max_len = max(len(pts[0]) for pts in all_pts.values())
            total_frames = (max_len - 1) // speed + 1
        else:
            total_frames = int(duration / self.dt)

        _prev_artists = []

        def _update(frame):
            nonlocal _prev_artists

            t = frame * speed * self.dt if has_path else frame * self.dt
            U, V = self._generate_current_data(t=t)
            Us = U[:100, :100].T[::qstep, ::qstep]
            Vs = V[:100, :100].T[::qstep, ::qstep]
            quiver.set_UVC(Us, Vs)

            if has_path:
                for a in _prev_artists:
                    a.remove()
                _prev_artists.clear()

                for idx, (xp, yp, hp) in all_pts.items():
                    pidx = min(frame * speed, len(xp) - 1)
                    s = (max(0, pidx - trail_length)
                         if trail_length > 0 else 0)
                    trail_lines[idx].set_data(xp[s:pidx + 1],
                                              yp[s:pidx + 1])

                    n_patches = len(ax.patches)
                    n_lines = len(ax.lines)

                    color = usv_colors[idx % len(usv_colors)]
                    usv_now = USV(xp[pidx], yp[pidx],
                                  heading=hp[pidx],
                                  length=self.usv_length,
                                  width=self.usv_width)
                    usv_now.detect_obstacles(obs_geoms)
                    usv_now.show_usv(ax, color=color, alpha=0.85)

                    _prev_artists.extend(ax.patches[n_patches:])
                    _prev_artists.extend(ax.lines[n_lines:])

                ref_idx = next(iter(all_pts))
                ref_pidx = min(frame * speed,
                               len(all_pts[ref_idx][0]) - 1)
                title.set_text(
                    f'Step {ref_pidx}  $t={t:.1f}$  '
                    f'({len(all_pts)} USVs)')
            else:
                title.set_text(
                    f'Time-varying currents ($t={t:.2f}$)')

            return quiver,

        anim = FuncAnimation(fig, _update, frames=total_frames,
                             interval=interval, blit=False, repeat=False)

        if save_path is not None:
            try:
                anim.save(save_path, writer='pillow', dpi=300)
                print(f"动画已保存至 {save_path}  ({total_frames} 帧)")
            except Exception as e:
                print(f"保存动画失败: {e}")
            plt.close()
        else:
            plt.show()


# ==================== COLREGs 专项测试场景 ====================

def _deg2rad(deg):
    return float(np.deg2rad(deg))


COLREGS_TEST_SCENARIOS = {
    # E1: 本船追越他船，本船让路（左/右均可）
    "E1": {
        "description": "追越-让路：本船追越他船，本船应主动规避。",
        "expected_encounter": EncounterType.OVERTAKING_GIVE_WAY,
        "expected_behavior": "本船应改变航向（左/右均可），避免直航硬追。",
        "start_positions": [[20.0, 50.0], [36.0, 50.0]],
        "headings": [_deg2rad(0.0), _deg2rad(0.0)],
        "initial_speeds": [2.6, 1.2],
        "targets": [{"x": 92.0, "y": 50.0}, {"x": 96.0, "y": 50.0}],
        "obstacles": [
            {"x": 85.0, "y": 20.0, "shape": ObstacleShape.CIRCLE, "radius_o": 4.0}
        ],
        "rule_violation_pairs": []
    },
    # E2: 他船追越本船，本船直航保持
    "E2": {
        "description": "追越-直航：他船追越本船，本船应保持航向。",
        "expected_encounter": EncounterType.OVERTAKING_STAND_ON,
        "expected_behavior": "本船应保持航向，避免不必要避让。",
        "start_positions": [[30.0, 50.0], [20.0, 50.0]],
        "headings": [_deg2rad(0.0), _deg2rad(0.0)],
        "initial_speeds": [1.2, 2.6],
        "targets": [{"x": 84.0, "y": 50.0}, {"x": 88.0, "y": 50.0}],
        "obstacles": [
            {"x": 85.0, "y": 20.0, "shape": ObstacleShape.CIRCLE, "radius_o": 4.0}
        ],
        "rule_violation_pairs": []
    },
    # E3: 对遇，双方应右转
    "E3": {
        "description": "对遇：两船迎面相遇，双方均应向右转。",
        "expected_encounter": EncounterType.HEAD_ON,
        "expected_behavior": "本船应右转避让（rudder<0）。",
        "start_positions": [[42.0, 50.0], [58.0, 50.0]],
        "headings": [_deg2rad(0.0), _deg2rad(180.0)],
        "initial_speeds": [2.2, 2.2],
        "targets": [{"x": 90.0, "y": 50.0}, {"x": 10.0, "y": 50.0}],
        "obstacles": [
            {"x": 50.0, "y": 82.0, "shape": ObstacleShape.CIRCLE, "radius_o": 4.0}
        ],
        "rule_violation_pairs": []
    },
    # E4: 右舷交叉，本船让路（右转）
    "E4": {
        "description": "交叉-让路：他船位于本船右舷交叉，本船应向右转让路。",
        "expected_encounter": EncounterType.CROSSING_GIVE_WAY,
        "expected_behavior": "本船应右转避让（rudder<0）。",
        "start_positions": [[40.0, 50.0], [50.0, 45.0]],
        "headings": [_deg2rad(0.0), _deg2rad(90.0)],
        "initial_speeds": [2.3, 2.0],
        "targets": [{"x": 90.0, "y": 50.0}, {"x": 50.0, "y": 90.0}],
        "obstacles": [
            {"x": 14.0, "y": 15.0, "shape": ObstacleShape.CIRCLE, "radius_o": 3.5}
        ],
        "rule_violation_pairs": []
    },
    # E5: 左舷交叉，本船直航保持
    "E5": {
        "description": "交叉-直航：他船位于本船左舷交叉，本船应保持航向。",
        "expected_encounter": EncounterType.CROSSING_STAND_ON,
        "expected_behavior": "本船应保持航向，避免过度规避。",
        "start_positions": [[40.0, 50.0], [50.0, 55.0]],
        "headings": [_deg2rad(0.0), _deg2rad(-90.0)],
        "initial_speeds": [2.3, 2.0],
        "targets": [{"x": 90.0, "y": 50.0}, {"x": 50.0, "y": 10.0}],
        "obstacles": [
            {"x": 14.0, "y": 85.0, "shape": ObstacleShape.CIRCLE, "radius_o": 3.5}
        ],
        "rule_violation_pairs": []
    },
    # E6: 他船违规/失控，本船主动让路（通过 override 注入）
    "E6": {
        "description": "违规或失控：他船违反规则或失去机动性，本船应主动让路。",
        "expected_encounter": EncounterType.RULE_VIOLATION,
        "expected_behavior": "本船应主动规避（通常右转更安全）。",
        "start_positions": [[40.0, 50.0], [50.0, 45.0]],
        "headings": [_deg2rad(0.0), _deg2rad(110.0)],
        "initial_speeds": [2.4, 1.8],
        "targets": [{"x": 90.0, "y": 50.0}, {"x": 50.0, "y": 90.0}],
        "obstacles": [
            {"x": 82.0, "y": 82.0, "shape": ObstacleShape.CIRCLE, "radius_o": 4.0}
        ],
        "rule_violation_pairs": [(0, 1)]
    }
}


def get_colregs_test_case_config(case_name):
    """获取 COLREGs 测试场景配置（E1-E6）。"""
    key = str(case_name).upper().strip()
    if key not in COLREGS_TEST_SCENARIOS:
        valid = ", ".join(sorted(COLREGS_TEST_SCENARIOS.keys()))
        raise ValueError(f"Unknown COLREGs case '{case_name}', valid cases: {valid}")

    base = COLREGS_TEST_SCENARIOS[key]
    return {
        "case_id": key,
        "description": base["description"],
        "expected_encounter": base["expected_encounter"],
        "expected_behavior": base["expected_behavior"],
        "start_positions": [np.array(p, dtype=np.float32) for p in base["start_positions"]],
        "headings": [float(h) for h in base["headings"]],
        "initial_speeds": [float(s) for s in base["initial_speeds"]],
        "targets": [dict(t) for t in base["targets"]],
        "obstacles": [dict(o) for o in base["obstacles"]],
        "rule_violation_pairs": list(base.get("rule_violation_pairs", []))
    }


def list_colregs_test_cases():
    """列出所有可用 COLREGs 测试场景及说明。"""
    return {
        case_id: cfg["description"]
        for case_id, cfg in COLREGS_TEST_SCENARIOS.items()
    }


class ColregsTestMultiUsvEnv(MultiUsvEnv):
    """
    COLREGs 专项测试环境（硬编码场景）。

    - 固定 USV 起点、初始航向、初始速度
    - 固定目标点与障碍物
    - 支持 E6 场景的 encounter override（RULE_VIOLATION）
    """

    def __init__(self, case_name="E3", **kwargs):
        self.colregs_case = get_colregs_test_case_config(case_name)
        self.colregs_rule_violation_pairs = set(
            tuple(p) for p in self.colregs_case.get("rule_violation_pairs", []))

        kwargs.setdefault("num_usvs", 2)
        kwargs.setdefault("num_targets", len(self.colregs_case["targets"]))
        kwargs.setdefault("num_obstacles", len(self.colregs_case["obstacles"]))
        kwargs.setdefault("fixed_start_positions", self.colregs_case["start_positions"])
        kwargs.setdefault("fixed_targets", self.colregs_case["targets"])
        kwargs.setdefault("fixed_obstacles", self.colregs_case["obstacles"])
        kwargs.setdefault("use_random_shapes", False)
        kwargs.setdefault("current_type", "generated")

        super().__init__(**kwargs)

    def _generate_environment(self):
        """先按父类生成，再强制注入场景初始航向/速度。"""
        super()._generate_environment()

        headings = self.colregs_case["headings"]
        speeds = self.colregs_case["initial_speeds"]
        for i, usv in enumerate(self.usvs):
            if i < len(headings):
                usv.heading = float(headings[i])
            if i < len(speeds):
                usv.u = float(speeds[i])
            usv.v = 0.0
            usv.r = 0.0
            usv._update_polygon()

        self.current_positions = np.array(
            [[u.x, u.y] for u in self.usvs], dtype=np.float32)
        self.current_pos = self.current_positions[0].copy()
        self.usv = self.usvs[0]

    def _update_colregs_matrix(self):
        """常规计算后，按测试需求覆盖特定 pair 的会遇类型。"""
        super()._update_colregs_matrix()
        for pair in self.colregs_rule_violation_pairs:
            if pair in self.colregs_matrix:
                self.colregs_matrix[pair]["encounter_type"] = EncounterType.RULE_VIOLATION

    def get_case_summary(self):
        """返回当前测试场景的摘要信息。"""
        return {
            "case_id": self.colregs_case["case_id"],
            "description": self.colregs_case["description"],
            "expected_encounter": int(self.colregs_case["expected_encounter"]),
            "expected_encounter_name": self.colregs_case["expected_encounter"].name,
            "expected_behavior": self.colregs_case["expected_behavior"]
        }

    def get_current_encounter(self, own_idx=0, target_idx=1):
        """读取当前 pair 的会遇类型与风险指标。"""
        ci = self.colregs_matrix.get((own_idx, target_idx))
        if ci is None:
            return None
        et = EncounterType(ci["encounter_type"])
        return {
            "pair": (own_idx, target_idx),
            "encounter_type": int(et),
            "encounter_name": et.name,
            "cri": float(ci.get("cri", 0.0)),
            "dcpa": float(ci.get("dcpa", 0.0)),
            "tcpa": float(ci.get("tcpa", 0.0)),
            "bearing_rel": float(ci.get("bearing_rel", 0.0))
        }


def create_colregs_test_env(case_name, **kwargs):
    """工厂函数：创建一个硬编码 COLREGs 测试环境。"""
    return ColregsTestMultiUsvEnv(case_name=case_name, **kwargs)


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
