import matplotlib.pyplot as plt
import numpy as np
from math import * 
import gymnasium as gym
from gymnasium.spaces import Box,Dict,MultiBinary,Discrete
from netCDF4 import Dataset
import random

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
    def show_obstacle(self, ax):
        circle_ob = plt.Circle((self.x, self.y), self.radius_o, color='black', fill=False)
        ax.add_patch(circle_ob)
        circle_c= plt.Circle((self.x, self.y), self.radius_o+self.rou_c, color='red',linestyle='--',fill=False)
        ax.add_patch(circle_c)
        circle_s= plt.Circle((self.x, self.y), self.radius_o+self.rou_c+self.rou_s, color='brown',linestyle='--',fill=False)
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
    def show_target(self, ax, prio_num=None):
        plt.plot(self.x,self.y,'g*')
        ax.text(self.x+self.radius_t, self.y+self.radius_t, str(prio_num), color='green', fontsize=12, ha='center', va='center')
        circle_t = plt.Circle((self.x, self.y), self.radius_t, color='green', linestyle='--', fill=False)
        ax.add_patch(circle_t)


#define the environment class for the USV

START_POS = np.array([5.0,5.0]) # starting position of the USV

# define the target positions
OBSTACLE_POS_R=[
    np.array([20.0,70.0,5.0]),
    np.array([40.0,30.0,15.0]),
    np.array([60.0,70.0,15.0]),
    np.array([80.0,20.0,2.5]),
    np.array([90.0,90.0,5.0]),
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
    def __init__(self):
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
            "Priority_vector":MultiBinary(len(TARGET_POS)),  # P
            "rou_t_norm":Box(low=0.0, high=1.0, shape=(1,)),  # rou_t_norm
            "alpha_t_norm":Box(low=0.0, high=1.0, shape=(1,2)),  # alpha_t_norm
        })
    
        self.step_counter = 0# the number of steps
        self.start_pos = START_POS # starting position of the USV
        self.target_r=3.0 # radius of the target area
        self.P=np.zeros(len(TARGET_POS)-1)  # 
        self.target_index=0
        self.render_flag=False # whether to render the environment
        
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
        if np.array_equal(self.P,np.array([0.0,0.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[0][0])**2+(current_pos[1]-TARGET_POS[0][1])**2)
            if rou_t<self.target_r:
                self.P[0]=1.0
                print("Target{} Reached!".format(self.target_index))
                self.target_index+=1
            reward_t+=-self.calculate_normalize_Rou(current_pos,TARGET_POS[0])+self.sgn(self.target_r-rou_t)
        elif np.array_equal(self.P,np.array([1.0,0.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[1][0])**2+(current_pos[1]-TARGET_POS[1][1])**2)
            if rou_t<self.target_r:
                self.P[1]=1.0
                print("Target{} Reached!".format(self.target_index))
                self.target_index+=1
            reward_t+=-self.calculate_normalize_Rou(current_pos,TARGET_POS[1])+self.sgn(self.target_r-rou_t)
        elif np.array_equal(self.P,np.array([1.0,1.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[2][0])**2+(current_pos[1]-TARGET_POS[2][1])**2)
            if rou_t<self.target_r:
                print("Target{} Reached!".format(self.target_index))
            reward_t+=-self.calculate_normalize_Rou(current_pos,TARGET_POS[2])+self.sgn(self.target_r-rou_t)
            
        # (3)Shortest path
        reward_p=0.0
        reward_p = -np.linalg.norm(current_pos - old_pos)
        
        #weigth the reward
        K_oi=0.2
        K_t=0.1
        K_p=0.01
        reward=K_oi*reward_oi+K_t*reward_t+K_p*reward_p
        # print("reward_oi:",reward_oi,"reward_t:",reward_t,"reward_p:",reward_p)
        return reward
    
    #define the step function for the environment
    def step(self, action):
        self.step_counter += 1
        old_pos = self.current_pos.copy()
        self.current_pos = self.calculate_next_pos(self.current_pos, action)
        self.check_pos(self.current_pos)
        
        # if np.sqrt((self.current_pos[0]-TARGET_POS[0][0])**2+(self.current_pos[1]-TARGET_POS[0][1])**2)<self.target_r:
        #     P=np.array([1.0,0.0])
        # elif np.sqrt((self.current_pos[0]-TARGET_POS[1][0])**2+(self.current_pos[1]-TARGET_POS[1][1])**2)<self.target_r:
        #     P=np.array([1.0,1.0])
        
        reward=self.calculate_reward(self.current_pos,old_pos)#calculate the reward before update the priority vector
        
        # self.P,self.target_index=self.update_priority_vector(self.current_pos)
        
        done=self.is_done()
        
        # reward=self.calculate_reward(self.current_pos,old_pos,P)
        
        state={
            "x_norm": np.array([self.normalize_x_y(self.current_pos[0])]),
            "y_norm": np.array([self.normalize_x_y(self.current_pos[1])]),
            "Rou_o_norm": np.array([self.calculate_normalize_Rou(self.current_pos, obs) for obs in OBSTACLE_POS_R]),
            "Alpha_o_norm": np.array([self.calculate_normalize_alpha(self.current_pos, obs) for obs in OBSTACLE_POS_R]),
            "Priority_vector":self.P,
            "rou_t_norm": np.array([self.calculate_normalize_Rou(self.current_pos, TARGET_POS[self.target_index])]),
            "alpha_t_norm": np.array([self.calculate_normalize_alpha(self.current_pos, TARGET_POS[self.target_index])]),
        }
        
        
        return state, reward, done

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
    def is_done(self):
        if np.array_equal(self.P,np.array([1.0,0.0])):
            return True
        else:
            return False
    
    # visualize the environment
    def visualize(self):
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


# define the gym environment for the USV
class MyUsvEnv_Gym(gym.Env):
    def __init__(self):
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
            "Priority_vector":MultiBinary(len(TARGET_POS)),  # P
            "rou_t_norm":Box(low=0.0, high=1.0, shape=(1,)),  # rou_t_norm
            "alpha_t_norm":Box(low=0.0, high=1.0, shape=(1,2)),  # alpha_t_norm
        })
    
        self.step_counter = 0# the number of steps
        self.start_pos = START_POS # starting position of the USV
        self.target_r=3.0 # radius of the target area
        self.P=np.zeros(len(TARGET_POS)-1)  # 
        self.target_index=0
        self.render_flag=False # whether to render the environment
        
    #define the reset function for the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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
        if np.array_equal(self.P,np.array([0.0,0.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[0][0])**2+(current_pos[1]-TARGET_POS[0][1])**2)
            if rou_t<self.target_r:
                self.P[0]=1.0
                print("Target{} Reached!".format(self.target_index))
                self.target_index+=1
            reward_t+=-self.calculate_normalize_Rou(current_pos,TARGET_POS[0])+self.sgn(self.target_r-rou_t)
        elif np.array_equal(self.P,np.array([1.0,0.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[1][0])**2+(current_pos[1]-TARGET_POS[1][1])**2)
            if rou_t<self.target_r:
                self.P[1]=1.0
                print("Target{} Reached!".format(self.target_index))
                self.target_index+=1
            reward_t+=-self.calculate_normalize_Rou(current_pos,TARGET_POS[1])+self.sgn(self.target_r-rou_t)
        elif np.array_equal(self.P,np.array([1.0,1.0])):
            rou_t=np.sqrt((current_pos[0]-TARGET_POS[2][0])**2+(current_pos[1]-TARGET_POS[2][1])**2)
            if rou_t<self.target_r:
                print("Target{} Reached!".format(self.target_index))
            reward_t+=-self.calculate_normalize_Rou(current_pos,TARGET_POS[2])+self.sgn(self.target_r-rou_t)
            
        # (3)Shortest path
        reward_p=0.0
        reward_p = -np.linalg.norm(current_pos - old_pos)
        
        #weigth the reward
        K_oi=0.2
        K_t=0.1
        K_p=0.01
        reward=K_oi*reward_oi+K_t*reward_t+K_p*reward_p
        # print("reward_oi:",reward_oi,"reward_t:",reward_t,"reward_p:",reward_p)
        return reward
    
    #define the step function for the environment
    def step(self, action):
        self.step_counter += 1
        old_pos = self.current_pos.copy()
        self.current_pos = self.calculate_next_pos(self.current_pos, action)
        self.check_pos(self.current_pos)
        
        # if np.sqrt((self.current_pos[0]-TARGET_POS[0][0])**2+(self.current_pos[1]-TARGET_POS[0][1])**2)<self.target_r:
        #     P=np.array([1.0,0.0])
        # elif np.sqrt((self.current_pos[0]-TARGET_POS[1][0])**2+(self.current_pos[1]-TARGET_POS[1][1])**2)<self.target_r:
        #     P=np.array([1.0,1.0])
        
        reward=self.calculate_reward(self.current_pos,old_pos)#calculate the reward before update the priority vector
        
        # self.P,self.target_index=self.update_priority_vector(self.current_pos)
        
        # done=self.is_done()
        terminated=self.is_done()
        truncated=False
        # reward=self.calculate_reward(self.current_pos,old_pos,P)
        
        info = {}
        
        state={
            "x_norm": np.array([self.normalize_x_y(self.current_pos[0])]),
            "y_norm": np.array([self.normalize_x_y(self.current_pos[1])]),
            "Rou_o_norm": np.array([self.calculate_normalize_Rou(self.current_pos, obs) for obs in OBSTACLE_POS_R]),
            "Alpha_o_norm": np.array([self.calculate_normalize_alpha(self.current_pos, obs) for obs in OBSTACLE_POS_R]),
            "Priority_vector":self.P,
            "rou_t_norm": np.array([self.calculate_normalize_Rou(self.current_pos, TARGET_POS[self.target_index])]),
            "alpha_t_norm": np.array([self.calculate_normalize_alpha(self.current_pos, TARGET_POS[self.target_index])]),
        }
        
        
        return state, reward, terminated,truncated, info

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
    def is_done(self):
        if np.array_equal(self.P,np.array([1.0,0.0])):
            return True
        else:
            return False
    
    # visualize the environment
    def visualize(self):
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



# test
# if __name__ == '__main__':
#     env=MyUsvEnv()
#     env.visualize()
#     i=0
#     obs=env.reset()
#     # print(obs)
#     while True:
#         i+=1
#         action=env.action_space.sample()
#         obs,reward,done=env.step(action)
#         print("step:",i)
#         print("action:",action)
#         position=np.array([obs['x_norm']*MAX,obs['y_norm']*MAX])
#         print("position:",position)
#         print("obs",obs)
#         print("reward:",reward)
#         if done:
#             break
        # if i==100:
        #     break
    # env.observation_space.sample()
    # print(env.observation_space.sample())
    # obs=env.reset()
    # current_pos=np.array([obs['x_norm']*MAX,obs['y_norm']*MAX])
    # old_pos=current_pos.copy()
    # P=obs['Priority_vector']
    # print(obs)
    # print(env.calculate_reward(current_pos,old_pos,P))
    # action=env.action_space.sample()
#     fig,ax=plt.subplots(figsize=(5,5))
#     plt.xlim(0,100)
#     plt.ylim(0,100)
#     # ob1 = Obstacle(10, 10, 5, 1.5, 3.5)
#     # ob2 = Obstacle(30, 20, 5, 1.5, 3.5)
#     # ob3 = Obstacle(40, 50, 5, 1.5, 3.5)
    
#     # t1=Target(90,90,3)
#     # t1.show_target(ax)
#     # if ob1.is_overlapping(ob2):print("ob1 and ob2 are overlapping")
#     # if ob1.is_overlapping(ob3):print("ob1 and ob3 are overlapping")
#     # if ob2.is_overlapping(ob3):print("ob2 and ob3 are overlapping")
#     # ob1.show_obstacle(ax)
#     # ob2.show_obstacle(ax)
#     # ob3.show_obstacle(ax)
#     plt.plot(START_POS[0],START_POS[1],'bo')
    
#     for i in range(len(TARGET_POS)):
#         t=Target(TARGET_POS[i][0],TARGET_POS[i][1],3)
#         t.show_target(ax,i)
    
#     for i in range(len(OBSTACLE_POS_R)):
#         o=Obstacle(OBSTACLE_POS_R[i][0],OBSTACLE_POS_R[i][1],OBSTACLE_POS_R[i][2],ROU_C,ROU_S)
#         o.show_obstacle(ax)
    
#     plt.show()
