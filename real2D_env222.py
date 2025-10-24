
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
from gym.spaces import Discrete,Box
from netCDF4 import Dataset

GOAL_POINT = np.array([1.0, 1.0])
START_POINT = np.array([40.0, 40.0])

MIN = 1
MAX = 40
record_dir = r'./history_point'
if not os.path.exists(record_dir):
    os.mkdir(record_dir)
class Env():
    def __init__(self):
        self.observation_space = Box(low=0.0, high=19.0, shape=(4,))
        self.action_space = Discrete(8)

        self.current_point = START_POINT
        self.current_pointwithflow = START_POINT.copy()
        self.goal = GOAL_POINT
        self.start = START_POINT
        self.start_togoal = GOAL_POINT-START_POINT
        self.step_counter = 0
        self.steplength = 1  # lucheng per time
        self.speed = 1.2
        self.width = 3
        self.height = 3
        self.t = 0.4
        self.time = 0.5


        self.X, self.Y = np.meshgrid(np.arange(0, 40, 1), np.arange(0, 40, 1))
        self.U, self.V = self.current_data()
        self.obs = self.obs_rectangle()
        self.circlelist = self.getcircles()
        self.r = 2
    @staticmethod
    def getcircles():
        obs_rectangle = [
            [3, 8],
            [12, 7],
            [26, 25],
            [18, 33],
            [26, 6],
            [16, 20],
            [25, 15],
            [16, 11],
            [32, 32],
            [32, 20],
            [35, 5],
            [5, 35],
            [10, 29],
            [7, 20],
            [19, 25],
            
            [30, 5],
            [6.5, 25],
            [20.5, 30],
            [10, 15],
            [23, 20],
        ]
        return obs_rectangle

    def iscollision(self):
        x = self.current_point[0]
        y = self.current_point[1]
        flag = False
        for pos in self.circlelist:  # 障碍物
            dis1 = np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)  # 计算当前位置到障碍物的距离
            if dis1 < self.r:   # 如果距离太近，返回True
                return True
        return flag

    def obs_rectangle(self):
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
        nc_obj = Dataset('./data/chinasea_201801_c.nc')
        uu = np.transpose(nc_obj.variables['u'][1, :, :])
        vv = np.transpose(nc_obj.variables['v'][1, :, :])
        uu = uu[60:101, 40:81]  # North component of velocity
        vv = vv[60:101, 40:81]  # East component of velocity
        return uu, vv

    def searchflow(self, k=0.09):
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

        values_u = np.array([self.U[x1][y1], self.U[x1][y2],self.U[x2][y1], self.U[x2][y2]])
        values_v = np.array([self.V[x1][y1], self.V[x1][y2], self.V[x2][y1], self.V[x2][y2]])
        u = griddata(points, values_u, point_grid, method='linear')
        v = griddata(points, values_v, point_grid, method='linear')
        # 获取current point这个位置的海流数据

        uv = np.concatenate((u, v))
        return uv
    
    def reset(self):
        self.step_counter = 0
        self.current_point = START_POINT.copy()
        self.current_pointwithflow = np.concatenate((self.current_point, self.searchflow()))
        state = np.array(self.current_pointwithflow)
        return state

    def changestart(self):
        self.step_counter = 0
        self.current_point = np.array([8.0,4.0])
        self.goal = GOAL_POINT
        self.start = self.current_point.copy()
        self.start_togoal = GOAL_POINT - self.start
        local_area = self.pos_toimg().flatten()
        self.current_pointwithflow = np.concatenate((self.current_point,self.searchflow(),local_area))
        state = np.array(self.current_pointwithflow)
        return state


    def pos_toimg(self):
        img = np.ones((self.width, self.height))
        centerx_index = 1
        centery_index = 1
        for i in range(self.width):
            for j in range(self.height):
                if_obstacle = self.iscollision()
                if if_obstacle:  # 与障碍物碰撞
                    img[i][j] = 0
                elif (self.current_point[0]-(centerx_index-i)*self.steplength) > MAX or (self.current_point[0]-(centerx_index-i)*self.steplength) < MIN or\
                        (self.current_point[1]-(centery_index-j)*self.steplength) > MAX or (self.current_point[1]-(centery_index-j)*self.steplength) < MIN:   # 与边界碰撞
                    img[i][j] = 0
                else:
                    img[i][j] = 1
        # img[centerx_index][centery_index] = 1
        return img

    def step(self, action):
        action = self.is_action_valid(action)
        old_flow = self.searchflow()
        old_pos, position, act_spd = self.caculate_position(action, old_flow)
        position = self.is_position_valid(position)

        self.current_point = position
        local_area = self.pos_toimg().flatten()  # 1表示是可行的通路，0表示不可行
        self.current_pointwithflow = np.concatenate((self.current_point, self.searchflow()))
        state = np.array(self.current_pointwithflow)
        done = self.is_done()  # 是否到达目的地
        reward = self.caculate_reward(old_pos, done, act_spd)

        self.step_counter = self.step_counter + 1
        return state, reward, done

    def is_action_valid(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        return action



    def caculate_position(self, action, old_flow):
        '''
        add ocean current information
        '''

        position = self.current_point.copy()
        old_pos = position.copy()
        basespeed = np.array([0.0, 0.0])
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
        position = position + self.time * (basespeed+old_flow)
        actual_spd = basespeed + old_flow

        return old_pos, position, actual_spd

    def is_position_valid(self, position):
        for i in range(2):
            if position[i] > MAX:
                position[i] = MAX
            if position[i] < MIN:
                position[i] = MIN
        correct_position = position
        return correct_position

    def get_actualtime(self,real_act,old_flow,getleast=False):
        a = np.linalg.norm(old_flow)
        #print(a)
        b = self.speed
        theta = 0 if getleast else self.get_cosvalue(old_flow,real_act)
        speed = np.sqrt(a**2+ b**2- 2*(a*np.sin(theta))**2+\
                 (1 if theta<np.pi/2 else -1)*2*a*np.sqrt((b**2-(a*np.sin(theta))**2)*np.cos(theta)**2))
        #print(np.linalg.norm(real_act))
        #time = np.linalg.norm(real_act)/cc
        return speed

    def get_cosvalue(self,old_flow,real_act):
        if np.dot(old_flow,real_act) != 0:
            cos_value = np.dot(old_flow, real_act) / (np.linalg.norm(real_act) * np.linalg.norm(old_flow))
            cos_value = round(cos_value,10)
            np.clip(cos_value,-1,1)
            #print(real_act, old_flow,cos_value)
            theta = math.acos(cos_value)
        else:
            theta = np.pi/2

        return theta
    def getrelative(self, obs):
        kk = obs.copy()
        slice1 = kk[:2]
        slice2 = kk[2:]
        rela_obs = np.concatenate((-GOAL_POINT[0:1]+slice1[0:1],-GOAL_POINT[1:2]+slice1[1:2],slice2*5))
        return rela_obs

    def caculate_reward(self, old_pos, done, act_spd):
        if done:
            reward = 100  # +200/(self.step_counter)
        elif all(self.current_point == old_pos):  # 位置不变
            reward = -10
        else:
            actual_speed = np.linalg.norm(act_spd)
            old_dis = np.sqrt(np.sum((old_pos - GOAL_POINT) ** 2))
            new_dis = np.sqrt(np.sum((self.current_point - GOAL_POINT) ** 2))

            distance_reward = (old_dis - new_dis - self.time * self.speed) * 10
            distance_reward = np.arctan(distance_reward)  # 利用反正切函数限制距离奖励范围
            flow_reward = np.arctan(10 * actual_speed/self.speed - 10)  # Rc(k2=10)

            lamda1 = 2  # 1
            lamda2 = 1  # 1.5
            reward = lamda1 * distance_reward + lamda2 * flow_reward
            reward = reward/(0.05*self.step_counter+1) if reward > 0 and self.step_counter > 50 else reward
 
        if self.iscollision():  # local_sense[60]<0:  #move to obstacle
            reward = reward-50
        return reward

    def is_done(self):
        if np.sqrt(np.sum((self.current_point - GOAL_POINT) ** 2)) < 0.5:
            print("goal")
            return True
        else:
            return False

    def get_actualtime(self, real_act, old_flow, getleast=False):
        a = np.linalg.norm(old_flow)
        b = self.speed
        theta = 0 if getleast else self.get_cosvalue(old_flow, real_act)
        speed = np.sqrt(a**2 + b**2 - 2*(a*np.sin(theta))**2 + (1 if theta < np.pi/2 else -1)*2*a*np.sqrt((b**2-(a*np.sin(theta))**2)*np.cos(theta)**2))
        #print(np.linalg.norm(real_act))
        #time = np.linalg.norm(real_act)/cc
        return speed
    
    def finalflow(self, X, Y, k=1.44):

        x = int(X)#self.current_point[0]
        y = int(Y)#self.current_point[1]
        x1= math.floor(x)
        y1 = math.floor(y)
        x2 = math.floor(x)+1
        y2 = math.floor(y) + 1
        point_grid = self.current_point.copy()
        points = np.array([[x1,y1],[x1,y2],[x2,y1],[x2,y2]])
        values_u = np.array([self.U[x1][y1],self.U[x1][y2],self.U[x2][y1],self.U[x2][y2]])
        values_v = np.array([self.V[x1][y1], self.V[x1][y2], self.V[x2][y1], self.V[x2][y2]])
        u = griddata(points,values_u,point_grid,method='linear')
        v = griddata(points, values_v, point_grid, method='linear')
        u = self.U[x1][y1]
        v = self.V[x1][y1]
        uv = np.concatenate((u, v))
        #print(u,v)
        return u,v

    def final(self, nresult_list):
        result = [list(g) for k, g in groupby(nresult_list, lambda x: len(x) == 0) if not k]   # groupby第一个参数是迭代对象，第二个参数是分组依据
        # if not k排除了k==false的情况，即排除了x为空的情况
        kkk = result[len(result)-1]  # 取最后一轮的路径点
        m, n = np.zeros(len(kkk)), np.zeros(len(kkk))
        for i in range(len(kkk)):
            m[i] = kkk[i][0]  # 表示第i个路径点的横坐标
            n[i] = kkk[i][1]  # 表示第i个路径点的纵坐标
        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        fig, ax = plt.subplots()
        x, y = np.meshgrid(np.arange(0, self.U.shape[0], 1),
                           np.arange(0, self.U.shape[1], 1)
                           )
        ax.quiver(x, y, np.transpose(self.U), np.transpose(self.V),
                  width=0.001)
        # U表示经度方向的海流速度，V表示纬度方向的海流速度
        distlist = []
        for i in range(len(m) - 1):
            action = np.array([m[i + 1] - m[i], n[i + 1] - n[i]])  # 第i+1个路径点到第i个路径点的距离
            dist = np.linalg.norm(action)*5500

            distlist.append(dist)

        goal = self.goal
        plt.plot(m, n)
        for rectangle_para in self.circlelist:
            ax.add_patch(
                plt.Circle(rectangle_para, self.r))

        plt.xlabel('x(*0.5°)')
        plt.ylabel('y(*0.5°)')

        ax.plot(self.start[0:1], self.start[1:2], 'go', markersize=10, markeredgecolor='k')
        ax.plot(goal[0:1], goal[1:2], 'r*', markersize=10, markeredgecolor='k')

        print(m.tolist())
        print(n.tolist())
        print(sum(distlist))
        print(len(distlist)*5500*0.5)

        plt.savefig('path' + '.png')
        
    def visualization(self):
        plt.rcParams['figure.figsize'] = (12.0, 12.0)
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.arange(-10, 10, 1), np.arange(-10,10, 1))
        q = ax.quiver(X, Y, self.u, self.v)

        for rectangle_para in self.obs_rectangle:
            ax.add_patch(
                plt.Rectangle(
                    (rectangle_para[0], rectangle_para[1]),
                    rectangle_para[2],
                    rectangle_para[3],
                    color='black',
                    alpha=1
                )
            )
        ax.plot(self.start[0:1], self.start[1:2], 'bo', markersize=12, markeredgecolor='k')
        goal = self.goal
        ax.plot(goal[0:1], goal[1:2], 'ro', markersize=7, markeredgecolor='k')
        return fig,ax
    
    def refresh(self,fig,ax):
        #fig, ax = plt.subplots()
        ax2 = ax.twinx()
        lines = ax2.add_patch(
            plt.Rectangle(
                (self.current_point[0]-5*self.steplength/6-self.steplength,self.current_point[1]-5*self.steplength/6-self.steplength),
                self.width*self.steplength/3,
                self.height*self.steplength/3,
                color='black',
                alpha =0.2
            )
        )
        # print(self.current_point,self.current_point[0] - 5 * self.steplength / 6 - self.steplength, self.current_point[
        #     1] - 5 * self.steplength / 6 - self.steplength)


        ax2.plot(self.current_point[0:1], self.current_point[1:2], 'go', markersize=7, markeredgecolor='k')
        #ax.plot(self.start[0:1], self.start[1:2], 'go', markersize=7, markeredgecolor='k')
        #ax.plot(self.obstacle1[0:1], self.obstacle1[1:2], 'bo', markersize=7, markeredgecolor='k')
        #ax.plot(GOAL_POINT[0:1], GOAL_POINT[1:2], GOAL_POINT[2:], 'r*', markersize=7, markeredgecolor='k')
        plt.axis([-10,10,-10,10])
        #plt.show()
        plt.axis('off')
        plt.pause(0.01)
        lines.remove()
        return ax2

    def clearax(self):
        plt.close()
        fig,ax = plt.subplots()
        X, Y = np.meshgrid(np.arange(-10, 10, 1), np.arange(-10, 10, 1))
        q = ax.quiver(X, Y, self.u, self.v)

        for rectangle_para in self.obs_rectangle:
            ax.add_patch(
                plt.Rectangle(
                    (rectangle_para[0], rectangle_para[1]),
                    rectangle_para[2],
                    rectangle_para[3],
                    color='black',
                    alpha=1
                )
            )
        ax.plot(self.start[0:1], self.start[1:2], 'bo', markersize=12, markeredgecolor='k')
        goal = self.goal
        ax.plot(goal[0:1], goal[1:2], 'ro', markersize=7, markeredgecolor='k')
        return ax



