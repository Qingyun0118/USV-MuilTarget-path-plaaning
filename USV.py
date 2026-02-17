import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib.patches import Polygon as MplPolygon  
from matplotlib.patches import Wedge  
from shapely.geometry import Polygon, LineString, Point  
from shapely.affinity import rotate, translate  
from torch.utils.tensorboard import SummaryWriter
import datetime

class USV:  
    """  
    USV类  
    基于 qLPV 结构的非线性动力学模型（经数值稳定性调整）  
    以及扇形多线传感器模型。  
    """  
    def __init__(self, x, y, heading=0.0, 
                mass=20.0,       # 质量 (kg)  
                inertia=2.0,     # 转动惯性 (kg·m^2) 
                dt=0.05,
                length=3.0, width=1.5):  
        
        # 几何参数  
        self.x = x  
        self.y = y  
        self.heading = heading # 弧度  
        self.length = length  
        self.width = width  
        
        # 动力学状态  
        self.u = 0.0  # 纵向速度 (m/s)  
        self.v = 0.0  # 横向速度 (m/s)  
        self.r = 0.0  # 艏摇角速度 (rad/s)  
        
        # 物理限制 (防止数值爆炸)  
        self.max_u = 10.0       # 最大纵向速度  
        self.max_v = 10.0        # 最大横向漂移速度  
        self.max_r = np.pi/3.0     # 最大转速 (rad/s)  
        
        self.mass = mass  
        self.inertia = inertia  
        # 基础阻力系数 (假设值)  
        # 阻力 F_drag = -k * u  
        # 离散方程: u_new = u + dt * (F_thrust - k*u) / m  
        # u_new = (1 - dt*k/m) * u + (dt/m) * F_thrust  
        
        self.dt = dt
        k_drag_lin = 2.0  # 线性阻力系数  
        k_drag_quad = 0.5 # 二次阻力系数 

        # 调整后的 qLPV 参数 (Stable Tuned)  
        # 相比原文献，降低了耦合项权重，增加了阻尼，保证仿真不发散  
        # self.params = {  
        #     # --- 纵向 (Surge) ---  
        #     'a11': 0.95,    # 惯性 (1.0 = 无阻力)  
        #     'a12': -0.01,   # 二次阻力系数  
        #     'a13': -0.1,    # v*r 耦合项 (科里奥利力影响)  
        #     'a14': -0.05,   # r^2 耦合项  
        #     'b11': 0.5,     # 推力增益  
              
        #     # --- 横向 (Sway) ---  
        #     'a21': -0.05,   # u*r 耦合项  
        #     'a22': 0.80,    # 横向阻尼 (较大，船很难横向滑行)  
        #     'a23': -0.2,    # |v|v 阻力  
        #     'a24': 0.0,     # r 耦合  
        #     'a25': 0.0,     # |r|r  
        #     'b21': 0.01,    # 推力产生的微小横向分量  
              
        #     # --- 转向 (Yaw) ---  
        #     'a31': -0.05,   # u*r 阻尼  
        #     'a32': -0.05,   # v 影响  
        #     'a33': 0.0,     # |v|v  
        #     'a34': 0.90,    # 转动惯性  
        #     'a35': -0.1,    # |r|r 阻力  
        #     'b31': 0.8      # 舵效 (转向力矩)  
        # }  
        
        # --- 动态计算 qLPV 参数 ---  
        # 这里的公式展示了质量(m)和惯性(I)如何影响参数 a_ij  
          
        self.params = {  
            # Surge (纵向): 受 mass 影响  
            # a11 = 1 - (阻力 * dt) / mass  
            # 质量越大 -> dt/mass 越小 -> a11 越接近 1.0 -> 惯性越大  
            'a11': 1.0 - (k_drag_lin * self.dt) / self.mass,   
              
            'a12': - (k_drag_quad * self.dt) / self.mass,  
            'a13': -0.1, # 简化的耦合项  
            'a14': -0.05,  
              
            # 推力影响: b = dt / mass  
            # 质量越大，推力产生的加速度越小  
            'b11': self.dt / self.mass * 100.0, # *100是推力放大系数  
              
            # Sway (横向): 受 mass 影响 (通常横向阻力极大)  
            'a21': -0.05,  
            'a22': 1.0 - (k_drag_lin * 5.0 * self.dt) / self.mass, # 横向阻力大5倍  
            'a23': -0.2,  
            'b21': (self.dt / self.mass) * 2.0,  
              
            # Yaw (转向): 受 inertia 影响  
            # a34 = 1 - (转动阻尼 * dt) / inertia  
            # inertia 越大 -> a34 越接近 1.0 -> 越难停下转动  
            'a31': -0.05,  
            'a32': -0.05,  
            'a34': 1.0 - (k_drag_lin * self.dt) / self.inertia,   
            'a35': -0.1,  
              
            # 舵效: 力矩 / inertia  
            'b31': (self.dt / self.inertia) * 50.0   
        }  

        # 传感器参数  
        self.sensor_range = 15.0  
        self.sensor_fov = np.radians(150)  
        self.num_detection_lines = 15  
        self.sensor_readings = [self.sensor_range] * self.num_detection_lines  
          
        self.polygon = None  
        self._update_polygon()  
  
    def _update_polygon(self):  
        """更新USV几何形状，增加NaN检查防止报错"""  
        # 安全检查：如果坐标发散，重置或报错  
        if np.isnan(self.x) or np.isinf(self.x):  
            self.x = 0  
            self.y = 0  
            self.u = 0  
            print("Warning: USV state reset due to NaN")  
  
        half_length = self.length / 2  
        half_width = self.width / 2  
          
        vertices = [  
            (half_length, 0),  
            (half_length * 0.3, half_width),  
            (-half_length, half_width * 0.8),  
            (-half_length, -half_width * 0.8),  
            (half_length * 0.3, -half_width),  
        ]  
          
        poly = Polygon(vertices)  
        # 这里的 rotate 和 translate 必须保证输入是有效数值  
        try:  
            poly = rotate(poly, self.heading, origin=(0, 0), use_radians=True)  
            poly = translate(poly, self.x, self.y)  
            self.polygon = poly  
        except Exception as e:  
            print(f"Error updating polygon: {e}")  
            # 创建一个默认的小框防止程序完全崩溃  
            self.polygon = Polygon([(self.x,self.y), (self.x+1, self.y), (self.x, self.y+1)])  

    def get_bounding_radius(self):
        """获取包围圆半径"""
        return max(self.length, self.width) / 2 * 1.2

    def is_colliding_with(self, geometry):
        """检测USV是否与给定几何体碰撞"""
        if self.polygon is None:
            return False
        return self.polygon.intersects(geometry)
    
    def step(self, throttle, rudder_angle):
        """  
        动力学步进  
        """  
        p = self.params  
          
        # 计算推力  
        T_thrust = throttle * 10.0 # 假设最大推力  
          
        # 三角函数  
        cos_delta = np.cos(rudder_angle)  
        sin_delta = np.sin(rudder_angle)  
          
        u_k, v_k, r_k = self.u, self.v, self.r  
        abs_u, abs_v, abs_r = abs(u_k), abs(v_k), abs(r_k)  
          
        # --- 1. Surge (纵向) ---  
        # 增加阻尼项和惯性，降低耦合项权重  
        u_next = (p['a11'] * u_k +   
                  p['a12'] * abs_u * u_k +   
                  p['a13'] * v_k * r_k +   
                  p['a14'] * r_k**2 +   
                  p['b11'] * T_thrust * cos_delta)  
          
        # --- 2. Sway (横向) ---  
        # 横向阻力通常很大  
        v_next = (p['a21'] * u_k * r_k +   
                  p['a22'] * v_k +   
                  p['a23'] * abs_v * v_k +   
                  p['b21'] * T_thrust * sin_delta)   
          
        # --- 3. Yaw (转向) ---  
        r_next = (p['a31'] * u_k * r_k +   
                  p['a32'] * v_k +   
                  p['a34'] * r_k +   
                  p['a35'] * abs_r * r_k +   
                  p['b31'] * T_thrust * sin_delta) # 舵效与推力相关  
  
        # --- 关键修复：数值限幅 (Clipping) ---  
        # 物理世界中船速不可能无限增加，强制截断以防止溢出  
        self.u = np.clip(u_next, -self.max_u, self.max_u) # 倒车速度慢一点  
        self.v = np.clip(v_next, -self.max_v, self.max_v)  
        self.r = np.clip(r_next, -self.max_r, self.max_r)  
        # self.u = u_next 
        # self.v = v_next
        # self.r = r_next
          
        # --- 运动学更新 ---  
        # 将局部速度转换到全局坐标系  
        x_dot = self.u * np.cos(self.heading) - self.v * np.sin(self.heading)  
        y_dot = self.u * np.sin(self.heading) + self.v * np.cos(self.heading)  
          
        self.x += x_dot * self.dt  
        self.y += y_dot * self.dt  
        self.heading += self.r * self.dt  
          
        # 归一化航向 [-pi, pi]  
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi  
          
        self._update_polygon()  
  
    def detect_obstacles(self, obstacles):  
        """传感器探测逻辑"""  
        self.sensor_readings = []  
          
        angles = np.linspace(  
            self.heading - self.sensor_fov / 2,  
            self.heading + self.sensor_fov / 2,  
            self.num_detection_lines  
        )  
          
        start_point = Point(self.x, self.y)  
          
        for angle in angles:  
            end_x = self.x + self.sensor_range * np.cos(angle)  
            end_y = self.y + self.sensor_range * np.sin(angle)  
            ray = LineString([(self.x, self.y), (end_x, end_y)])  
              
            min_dist = self.sensor_range  
              
            for obs in obstacles:
                try:
                    if ray.intersects(obs):
                        intersection = ray.intersection(obs)
                        # 处理不同类型的交集结果
                        if intersection.is_empty:
                            continue
                        
                        dist = self.sensor_range # 默认最大
                        
                        if intersection.geom_type == 'Point':
                            dist = start_point.distance(intersection)
                        elif intersection.geom_type == 'MultiPoint':
                            # 找最近的点
                            dists = [start_point.distance(p) for p in intersection.geoms]
                            dist = min(dists)
                        elif intersection.geom_type in ['LineString', 'MultiLineString', 'GeometryCollection']:
                            dist = start_point.distance(intersection)
                        
                        if dist < min_dist:
                            min_dist = dist
                except Exception as e:
                    # print(f"Warning: Geometry error in sensor detection: {e}")
                    continue
              
            self.sensor_readings.append(min_dist)  
              
        return self.sensor_readings  
  
    def get_vertices(self):  
        if self.polygon is None: return []  
        return list(self.polygon.exterior.coords)[:-1]  
  
    def show_usv(self, ax, color='blue', alpha=0.7, label=None, show_sensor=True):  
        if self.polygon is None: return  
  
        # 1. 绘制传感器  
        if show_sensor:  
            # 背景扇形
            wedge = Wedge((self.x, self.y), self.sensor_range,
                          np.degrees(self.heading - self.sensor_fov/2),
                          np.degrees(self.heading + self.sensor_fov/2),
                          color='cyan', alpha=0.1, zorder=4)
            ax.add_patch(wedge)
              
            # 探测线
            angles = np.linspace(
                self.heading - self.sensor_fov / 2,
                self.heading + self.sensor_fov / 2,
                self.num_detection_lines
            )
            for i, dist in enumerate(self.sensor_readings):
                angle = angles[i]
                end_x = self.x + dist * np.cos(angle)
                end_y = self.y + dist * np.sin(angle)
                  
                line_color = 'red' if dist < self.sensor_range - 0.1 else 'green'
                ax.plot([self.x, end_x], [self.y, end_y], color=line_color, alpha=0.3, linewidth=1, zorder=4)
                  
                if dist < self.sensor_range - 0.1:
                    ax.plot(end_x, end_y, 'ro', markersize=2, zorder=5)
  
        # 2. 绘制USV本体  
        vertices = self.get_vertices()  
        patch = MplPolygon(vertices, closed=True, facecolor=color,   
                           edgecolor='darkblue', alpha=alpha, label=label, zorder=6)  
        ax.add_patch(patch)  
          
        # 3. 绘制船头方向  
        arrow_len = self.length * 0.8  
        ax.arrow(self.x, self.y,   
                 arrow_len * np.cos(self.heading),   
                 arrow_len * np.sin(self.heading),  
                 head_width=0.5, fc='darkblue', ec='darkblue', zorder=7)
        
if __name__ == '__main__':
    obstacle1 = Point(15, 5).buffer(2.0) # 圆形障碍物  
    obstacle2 = Polygon([(20, 10), (25, 10), (25, 15), (20, 15)]) # 矩形障碍物  
    env_obstacles = [obstacle1, obstacle2]  
    writer=SummaryWriter('usvTest/usvTest_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
      
    # 初始化USV  
    usv = USV(x=0, y=0, heading=np.radians(30),mass=50,inertia=5,dt=0.05)  
      
    # 模拟循环  
    fig, ax = plt.subplots(figsize=(10, 8))  
      
    # 简单的开环控制输入序列  
    # 时间步 0-50: 加速  
    # 时间步 50-100: 左转  
      
    path_x, path_y = [], []  
      
    for t in range(100):  
        # 控制输入  
        throttle = 0.2
        rudder = 0.0  
        if t > 50:  
            # throttle = 0.0
            rudder = np.radians(30) # 向左打舵 (根据模型定义，正值可能向左或向右，需根据坐标系确认)  
              
        # 1. 动力学步进  
        usv.step(throttle, rudder)  
        writer.add_scalar('u',usv.u,t)
        writer.add_scalar('v',usv.v,t)
        writer.add_scalar('r',usv.r,t)
        writer.add_scalar('throttle',throttle,t)
        writer.add_scalar('rudder',rudder,t)

        # 2. 传感器探测  
        readings = usv.detect_obstacles(env_obstacles)  
          
        path_x.append(usv.x)  
        path_y.append(usv.y)  
      
    # 绘图  
    ax.set_xlim(-5, 100)  
    ax.set_ylim(-5, 100)  
    ax.set_aspect('equal')  
      
    # 绘制障碍物  
    from shapely.plotting import plot_polygon  
    for obs in env_obstacles:  
        plot_polygon(obs, ax=ax, facecolor='gray', edgecolor='black')  
      
    # 绘制轨迹  
    ax.plot(path_x, path_y, 'k--', label='Trajectory')  
      
    # 绘制最终状态的USV和传感器  
    usv.show_usv(ax, label='USV')  
      
    plt.title("USV Simulation: Nonlinear Dynamics & Sector Sensor")  
    plt.legend()  
    plt.grid(True)  
    plt.show()