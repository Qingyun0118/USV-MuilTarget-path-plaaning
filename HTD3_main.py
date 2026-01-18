"""
HRL-MG-TD3实现。多目标TD3的分层强化学习。
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime
from collections import deque
from my_USV_Env import MyUsvEnv,X_MAX,Y_MAX

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Replay Buffers ---

class ReplayBuffer:
    """Standard Replay Buffer for Selector (DQN)"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.LongTensor(np.array(action)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

class HER_ReplayBuffer:  
    def __init__(self, capacity, max_episode_len, k_future=4):  
        """  
        :param capacity: 缓冲区最大容量（以 episode 为单位）  
        :param max_episode_len: 每个 episode 的最大步数  
        :param k_future: HER 采样比例 (future strategy 4:1)  
        """  
        self.capacity = capacity  
        self.max_episode_len = max_episode_len  
        self.k_future = k_future  
        
        # 存储 buffer，列表的列表。 buffer[i] 是第 i 个 episode 的所有 transition  
        self.buffer = deque(maxlen=capacity)   
        
    def store_episode(self, episode_transitions):  
        """  
        存储完整的一条轨迹  
        episode_transitions: list of (state, action, next_state, achieved_goal, desired_goal, info)  
        注意：我们这里不存 reward，因为 reward 是采样时算的  
        """  
        self.buffer.append(episode_transitions)  

    def sample(self, batch_size):  
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []  
        goal_batch = [] # desired_goal  
        
        # 1. 随机选择 batch_size 个 episode  
        # 假如 buffer 里不够 batch_size，就允许重复采样  
        num_episodes = len(self.buffer)  
        episode_indices = np.random.randint(0, num_episodes, batch_size)  
        
        for ep_idx in episode_indices:  
            episode = self.buffer[ep_idx]  
            episode_len = len(episode)  
            
            # 2. 在该 episode 中随机选择一个时间步 t  
            t = np.random.randint(0, episode_len)  
            transition = episode[t]  
            
            # 解包 transition  
            # 格式: (state, action, next_state, achieved_goal(step), desired_goal, actual_accumulated_pos)  
            # 注意：对于 Lower Agent，achieved_goal 通常是单步位移或累积位移，这里我们需要存绝对位置或相对起点的位移  
            s, a, ns, ag, dg, info = transition  
            
            # 3. HER Logic: 决定是否重标记目标  
            # future strategy: 以概率 1 - 1/(1+k) 采样未来的状态作为目标  
            # 简单写法：80% 概率用 future  
            her_bool = (np.random.random() < 0.8)  
            
            if her_bool:  
                # 采样未来时刻 t_future (t <= t_future < len)  
                t_future = np.random.randint(t, episode_len)  
                # 使用未来时刻的 "achieved_goal" 作为新的 "desired_goal"  
                # 注意：我们需要确保数据结构里存了 consistent 的 achieved_goal  
                # 这里的 ag_future 应该是相对于当前时刻 t 的状态  
                
                # === 针对 HRL 的特殊处理 ===  
                # Lower Agent 的目标通常是 "相对位移"。  
                # 如果 t 时刻在位置 P_t，t_future 时刻在 P_future。  
                # 原始目标是 P_sub - P_t。  
                # HER 目标应该是 P_future - P_t。  
                
                future_transition = episode[t_future]  
                # 我们假设存储时，info 里存了 'usv_pos' (绝对坐标或归一化坐标)  
                pos_t = info['current_pos']  
                pos_future = future_transition[5]['next_pos'] # info is index 5  
                
                new_goal = pos_future - pos_t # 新的相对目标  
                
                # 重算奖励  
                r = compute_reward(np.array([new_goal]), np.array([new_goal])) # 完美达成，距离为0  
                g = new_goal  
            else:  
                # 使用原始目标  
                r = compute_reward(np.array([ag]), np.array([dg]))  
                g = dg  
            
            # 4. 收集数据  
            state_batch.append(s)  
            action_batch.append(a)  
            reward_batch.append(r)  
            next_state_batch.append(ns)  
            done_batch.append(0.0) # HER 通常即使到了目标也不设 done=True，或者根据距离设  
            goal_batch.append(g)  

        # 转换为 Tensor  
        return (  
            torch.FloatTensor(np.array(state_batch)).to(device),  
            torch.FloatTensor(np.array(action_batch)).to(device),  
            torch.FloatTensor(np.array(reward_batch)).reshape(-1, 1).to(device),  
            torch.FloatTensor(np.array(next_state_batch)).to(device),  
            torch.FloatTensor(np.array(done_batch)).unsqueeze(1).to(device),  
            torch.FloatTensor(np.array(goal_batch)).to(device), # desired_goal input to actor  
            None # achieved_goal (not used in standard TD3 input, only for reward)  
        )  
  
    def __len__(self):  
        return len(self.buffer)  


# --- Neural Networks ---
class SelectorQNet(nn.Module):
    """DQN for Selector: State -> Q-values for each target"""
    def __init__(self, state_dim, num_actions):
        super(SelectorQNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Actor(nn.Module):
    """TD3 Actor: (State + Goal) -> Action"""
    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """TD3 Critic: (State + Goal + Action) -> Q-value"""
    def __init__(self, state_dim, goal_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 128)
        self.l6 = nn.Linear(128, 1)

    def forward(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

# --- Agents ---

class SelectorAgent:
    def __init__(self, state_dim, num_targets, lr=1e-4, gamma=0.99, epsilon=0.3):
        self.q_net = SelectorQNet(state_dim, num_targets).to(device)
        self.target_q_net = SelectorQNet(state_dim, num_targets).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_targets = num_targets

    def select_action(self, state, valid_mask=None, eval=False):
        """
        状态：numpy 数组
        有效掩码：布尔数组（如果目标可用/未访问则为 True）
        """
        if not eval and random.random() < self.epsilon:
            if valid_mask is not None:
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    return np.random.choice(valid_indices)
            return np.random.randint(0, self.num_targets)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state_t).cpu().numpy().flatten()
        
        if valid_mask is not None:
            # 用 -inf 屏蔽无效（已访问）的目标
            q_values[~valid_mask.astype(bool)] = -1e9
            
        return np.argmax(q_values)

    def train(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_q = self.target_q_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (1 - done) * self.gamma * next_q

        current_q = self.q_net(state).gather(1, action)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self, tau=0.005):
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class TD3Agent:
    def __init__(self, state_dim, goal_dim, action_dim, max_action, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, tau=0.05, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = Actor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state, goal, noise=0.0):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action = self.actor(state_t, goal_t).cpu().numpy().flatten()
        
        if noise > 0:
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(-self.max_action, self.max_action)
        
        return action

    def train(self, replay_buffer, batch_size=100):
        if len(replay_buffer) < batch_size:
            return 0, 0

        self.total_it += 1

        # Sample replay buffer 
        state, action, reward, next_state, done, desired_goal, achieved_goal = replay_buffer.sample(batch_size)

        # 使用 desired_goal 进行训练（标准 HER 逻辑使用 desired_goal，但我们也可以将 achieved_goal 作为成功示例混合使用）
        # 这里我们假设缓冲区已经包含经过 HER 处理的转换（重新标记的目标），或者我们可以在运行时处理。
        # 为了简化，我们假设缓冲区存储 (s, a, r, s', d, g, ag) 并使用 g。
        # 注意：在标准 HER 中，我们会采样一个批次，并在某些比例上将 g 替换为 ag 并重新计算奖励。
        # 这里我们将实现一个简单版本：缓冲区存储经历。
        # 如果我们想要 HER，应该在推入缓冲区之前进行重新标记，或者在这里进行。
        # 为了在这个实现中简化，我们假设在推入缓冲区之前进行重新标记。
        
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (self.actor_target(next_state, desired_goal) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, desired_goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, desired_goal, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = 0
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            actor_loss = -self.critic.Q1(state, desired_goal, self.actor(state, desired_goal)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss if isinstance(actor_loss, int) else actor_loss.item()

# --- Helper Functions ---

def get_selector_state(obs_dict):
    """Extract state for Selector: USV pos info + Targets info"""
    # o_usv pos: 2, targets: 3*N
    return np.concatenate([obs_dict['o_usv'][0:2], obs_dict['targets']])

def get_upper_state(obs_dict):
    """Extract state for Upper Actuator: USV pos info + Lidar"""
    # o_usv: 2, o_detector: 13
    return np.concatenate([obs_dict['o_usv'][0:2],obs_dict['o_detector']])

def get_lower_state(obs_dict):
    """Extract state for Lower Actuator: USV velocities"""
    # o_usv: [x, y, u, v, r, sin, cos]
    # We need [u, v, r] which are indices 2, 3, 4
    return np.concatenate([obs_dict['o_usv'], obs_dict['o_detector'], obs_dict['o_current']])

def get_target_pos(obs_dict, target_idx, num_targets):
    """Get global position of the selected target"""
    # 目标: [x1, y1, s1, x2, y2, s2, ...]  
    # 如果它们已经归一化，我们需要进行反归一化。   
    # 在 my_USV_Env 中，它们是通过 X_MAX, Y_MAX 进行归一化的。  
    # 但在这里，我们可能希望使用归一化值或者反归一化值。  
    # 对于神经网络输入，我们保持使用归一化值，但计算距离时可能需要真实值。  
    # 环境在 'targets' 中返回归一化值。
    
    # Extract from obs_dict['targets']
    base_idx = target_idx * 3
    tx = obs_dict['targets'][base_idx]
    ty = obs_dict['targets'][base_idx+1]
    return np.array([tx, ty])

def get_usv_pos(obs_dict):
    """Get normalized USV position"""
    return obs_dict['o_usv'][0:2]

def check_subgoal_collision(usv_pos, sub_goal_rel, obstacles, margin=2.0):  
    """  
    检查生成的子目标是否落在障碍物膨胀区域内 (用于 Upper Reward 计算)  
    usv_pos: [x, y]  
    sub_goal_rel: [dx, dy]  
    """  
    # 计算子目标绝对坐标  
    sub_goal_global = usv_pos + sub_goal_rel  
    
    # 简单的点碰撞检测 (可以根据 Obstacle 类的方法优化)  
    # 这里假设 obstacles 是 MyUsvEnv 中的对象列表  
    for obs in obstacles:  
        # 使用障碍物的安全区域 (safety_zone) 进行判断  
        if obs.is_point_in_safety_zone(sub_goal_global):  
            return True  
    return False  

def calc_upper_reward(usv_pos, sub_goal_rel, g_des_global, obstacles):  
    """  
    计算上层执行器奖励 (Eq 2-12)  
    R_up = -alpha * dist(current + sub, final) - beta * Collision  
    """  
    alpha = 1.0  
    beta = 10.0  
    
    # 预测的下一步位置 (假设完美执行)  
    predicted_pos = usv_pos + sub_goal_rel  
    
    # 距离惩罚：子目标是否指向最终目标  
    dist_to_final = np.linalg.norm(predicted_pos - g_des_global)  
    
    # 碰撞惩罚  
    is_collision = check_subgoal_collision(usv_pos, sub_goal_rel, obstacles)  
    
    reward = -alpha * dist_to_final  
    if is_collision:  
        reward -= beta  
        
    return reward  

# def calc_lower_reward(current_pos, target_sub_global, action):  
#     """  
#     计算下层执行器奖励 
#     """  
#     dist = np.linalg.norm(current_pos - target_sub_global)  #靠近子目标的距离惩罚
#     if dist < args.target_radius/5.0:  
#         dist = 0.0  # 达到子目标附近时不再惩罚距离
#     return -dist  
def compute_reward(achieved_goal, desired_goal, info=None):  
    """  
    计算奖励函数 (支持批量计算)  
    :param achieved_goal: 实际达到的相对位移/位置 [batch_size, dim]  
    :param desired_goal: 期望的相对位移/位置 [batch_size, dim]  
    :return: 奖励 [batch_size, 1]  
    """  
    # 计算欧氏距离  
    # 注意：输入通常是 Tensor 或 Numpy  
    if isinstance(achieved_goal, torch.Tensor):  
        dist = torch.norm(achieved_goal - desired_goal, dim=1, keepdim=True)  
        # 奖励设计：距离越近奖励越大，最大为0。基于距离的密集奖励。  
        # 系数 10.0 是为了让梯度更明显  
        return -1.0 * dist   
    else:  
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1, keepdims=True)  
        return -1.0 * dist  


# --- Main HRL Loop ---

def train_hrl(args):  
    # 1. 环境初始化  
    env = MyUsvEnv(  
        num_targets=args.num_targets,  
        num_obstacles=args.num_obstacles,  
        seed=args.seed,  
        record_flag=False,  
        usv_length=args.usv_length,  
        usv_width=args.usv_width,  
        target_radius=args.target_radius,  
        obstacle_radius_range=(args.obstacle_radius_min, args.obstacle_radius_max),  
        use_random_shapes=args.use_random_shapes  
    )  

    # 获取维度  
    obs = env.reset()  
    selector_state_dim = get_selector_state(obs).shape[0]  
    upper_state_dim = get_upper_state(obs).shape[0]  
    lower_state_dim = get_lower_state(obs).shape[0]  
    
    # 动作与目标维度  
    selector_action_dim = args.num_targets  
    upper_action_dim = 2 # (dx, dy)  
    lower_action_dim = 2 # (throttle, rudder)  
    upper_goal_dim = 2   # 相对目标  
    lower_goal_dim = 2   # 相对子目标  
    
    # 2. 智能体初始化  
    selector = SelectorAgent(selector_state_dim, selector_action_dim, epsilon=args.epsilon_start)  
    # 注意：max_action 限制了子目标范围  
    upper_agent = TD3Agent(upper_state_dim, upper_goal_dim, upper_action_dim, max_action=args.subgoal_range, lr_actor=1e-4, lr_critic=1e-3)  
    lower_agent = TD3Agent(lower_state_dim, lower_goal_dim, lower_action_dim, max_action=1.0, lr_actor=1e-4, lr_critic=1e-3)  
    
    # 3. 缓冲区设置 (使用 HER Buffer)  
    selector_buffer = ReplayBuffer(args.selector_buffer_size)  
    # Capacity 是 Episode 数量  
    upper_buffer = HER_ReplayBuffer(capacity=5000, max_episode_len=args.upper_steps)  
    lower_buffer = HER_ReplayBuffer(capacity=10000, max_episode_len=args.lower_steps)  

    writer = SummaryWriter(f'htd3_output/runs/HRL_HER_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')  
    
    global_step = 0  
    SCALE = np.array([X_MAX, Y_MAX]) # 坐标反归一化因子 [100, 100]  

    for episode in range(args.max_episodes):    
        obs_dict = env.reset()    
        usv_pos = get_usv_pos(obs_dict) # 归一化坐标 [0, 1]  
        
        target_status = obs_dict['targets'][2::3] # 0: unvisited    
        valid_mask = (target_status == 0)    
        
        # 记录累积奖励用于显示  
        episode_lower_reward = 0    
        episode_upper_reward = 0  
        episode_selector_reward = 0  
        
        done = False    
        attempts = 0  
        crush = False  
        
        # ==================== Selector Loop ====================  
        while not (done or attempts >= args.num_targets or crush):    
            attempts += 1   
            selector_state = get_selector_state(obs_dict)    
            selector_start_pos = usv_pos  
            selector_start_state = selector_state  
            
            # Selector 选择一个全局目标  
            target_idx = selector.select_action(selector_state, valid_mask)    
            g_des_global_norm = get_target_pos(obs_dict, target_idx, args.num_targets) # 归一化的全局目标  
            
            print(f'EP:{episode} | Selector Target: {target_idx} | Pos: {g_des_global_norm*100}')  
            
            selector_step_reward = 0    
            selector_done = False    
            
            # 临时存储 Upper 轨迹 (用于 HER)  
            upper_trajectory = []  
            
            # ==================== Upper Agent Loop ====================  
            u_step = 0  
            while not (done or selector_done or crush or u_step >= args.upper_steps):    
                u_step += 1  
                upper_state = get_upper_state(obs_dict)    
                usv_pos_start_upper = get_usv_pos(obs_dict)    
                
                # Upper Goal: 最终目标 - 当前位置 (相对向量)  
                g_des_rel = g_des_global_norm - usv_pos_start_upper    
                
                # Upper Action: 生成子目标 (相对偏移)  
                g_sub_rel = upper_agent.select_action(upper_state, g_des_rel, noise=args.exploration_noise)    
                
                # 计算子目标绝对位置 (归一化)  
                g_sub_global_norm = usv_pos_start_upper + g_sub_rel  
                g_sub_global_norm = np.clip(g_sub_global_norm, 0, 1) # 限制在地图内  
                
                # 临时存储 Lower 轨迹 (用于 HER)  
                lower_trajectory = []  
                sub_done = False  
                
                # ==================== Lower Agent Loop ====================  
                l_step = 0  
                # 强制 Lower 步数较小，例如 20  
                while not (done or selector_done or sub_done or crush or l_step >= args.lower_steps):    
                    l_step += 1  
                    lower_state = get_lower_state(obs_dict)    
                    curr_usv_pos = get_usv_pos(obs_dict)    
                    
                    # Lower Goal: 子目标 - 当前位置  
                    curr_g_sub_rel = g_sub_global_norm - curr_usv_pos    
                    
                    # Lower Action: [Throttle, Rudder]  
                    lower_action = lower_agent.select_action(lower_state, curr_g_sub_rel, noise=args.exploration_noise)    
                    
                    # --- 关键：将坐标转为绝对坐标传给环境 ---  
                    g_sub_global_abs = g_sub_global_norm * SCALE  
                    next_obs_dict, _, sub_done, crush, truncated, info = env.step(lower_action, g_sub_global_abs)    
                    
                    next_usv_pos = get_usv_pos(next_obs_dict)    
                    
                    # 计算实际位移 (Achieved Goal)  
                    actual_move = next_usv_pos - curr_usv_pos  
                    
                    # 计算 Dense Reward 用于日志显示 (Buffer里存原始数据由HER重算)  
                    r_low_dense = compute_reward(np.array([actual_move]), np.array([curr_g_sub_rel]))[0][0]  
                    episode_lower_reward += r_low_dense  

                    # 存储到 Lower 轨迹  
                    # info 必须包含 'current_pos' 和 'next_pos' 供 HER 使用  
                    step_info = {'current_pos': curr_usv_pos, 'next_pos': next_usv_pos}  
                    
                    lower_trajectory.append(  
                        (lower_state, lower_action, get_lower_state(next_obs_dict),   
                        actual_move, curr_g_sub_rel, step_info)  
                    )  
                    
                    obs_dict = next_obs_dict  
                    usv_pos = next_usv_pos  
                    global_step += 1    
                    
                    # >>>>>>>>>>>>>>>>>> 动态目标检测 (DGT) 开始 <<<<<<<<<<<<<<<<<<  
                    reached_indices = info.get('reached_targets', [])  
                    if len(reached_indices) > 0:  
                        next_selector_state = get_selector_state(next_obs_dict)  
                        # 检查当前是否所有任务已完成 (用于判断 Fake Done)  
                        current_valid_mask = (next_obs_dict['targets'][2::3] == 0)  
                        is_all_done = not np.any(current_valid_mask)  

                        for r_idx in reached_indices:  
                            if r_idx == target_idx:  
                                # Case 1: 到达了预定目标  
                                print(f"  >>> SUCCESS: Reached Intended Target {target_idx}")  
                                selector_step_reward = 1.0  
                                selector_done = True   
                            else:  
                                # Case 2: 意外到达了其他有效目标  
                                print(f"  >>> DGT HIT: Reached Target {r_idx} (aiming for {target_idx})")  
                                
                                # 构造虚拟经验：假设 Selector 当初选的就是 r_idx  
                                # Action: r_idx  
                                # Reward: 1.0 (成功)  
                                # Next State: next_selector_state  
                                # Done: is_all_done (如果这是最后一个目标，则Done=True)  
                                selector_buffer.push(  
                                    selector_start_state,  
                                    r_idx, # Fake Action  
                                    1.0,   # Fake Reward  
                                    next_selector_state,  
                                    is_all_done  
                                )  
                                
                                # 只要碰到任何目标，我们都认为当前规划需要重置  
                                # 因为当前位置变了，未访问目标列表也变了，继续执行旧策略不如重新决策  
                                selector_done = True  
                    # >>>>>>>>>>>>>>>>>> 动态目标检测 (DGT) 结束 <<<<<<<<<<<<<<<<<<  

                # --- End of Lower Loop ---  
                
                # 1. 存储 Lower 轨迹到 HER Buffer  
                lower_buffer.store_episode(lower_trajectory)  
                
                # 2. 训练 Lower Agent  
                # 只有当数据足够多时才训练  
                if len(lower_buffer) > args.actuator_batch_size:  
                    for _ in range(l_step): # 每步训练一次  
                        lower_agent.train(lower_buffer, args.actuator_batch_size)  

                # 3. 记录 Upper Transition  
                next_upper_state = get_upper_state(obs_dict)  
                usv_pos_end_upper = get_usv_pos(obs_dict)  
                
                # Upper Achieved Goal: 这一阶段的总位移  
                actual_upper_displacement = usv_pos_end_upper - usv_pos_start_upper  
                
                # Upper Reward (Dense): 越接近 Global Target 越好  
                r_up_dense = compute_reward(np.array([actual_upper_displacement]), np.array([g_des_rel]))[0][0]  
                episode_upper_reward += r_up_dense  
                
                step_info_upper = {'current_pos': usv_pos_start_upper, 'next_pos': usv_pos_end_upper}  
                
                upper_trajectory.append(  
                    (upper_state, g_sub_rel, next_upper_state,   
                    actual_upper_displacement, g_des_rel, step_info_upper)  
                )  

            # --- End of Upper Loop ---  
            
            # 1. 存储 Upper 轨迹到 HER Buffer  
            upper_buffer.store_episode(upper_trajectory)  
            
            # 2. 训练 Upper Agent  
            if len(upper_buffer) > args.actuator_batch_size:  
                for _ in range(u_step):  
                    upper_agent.train(upper_buffer, args.actuator_batch_size)  
            
            # 3. Selector Update  
            target_status = obs_dict['targets'][2::3]    
            valid_mask = (target_status == 0)    
            done = not np.any(valid_mask)   

            if done:  
                selector_step_reward = 2.0 # 完成所有任务的大奖励  

            final_selector_reward = selector_step_reward if selector_done else -0.1 # 步数惩罚  
            
            selector_buffer.push(    
                selector_start_state, target_idx, final_selector_reward,     
                get_selector_state(obs_dict), done    
            )  
            
            if len(selector_buffer) > args.selector_batch_size:    
                selector.train(selector_buffer, args.selector_batch_size)    
            
            episode_selector_reward += final_selector_reward    
            
        # --- End of Selector Loop ---    
        
        # 衰减 Epsilon  
        selector.epsilon = max(args.epsilon_end, selector.epsilon * args.epsilon_decay)  
        
        print(f"Episode {episode} End | R_low: {episode_lower_reward:.1f} | R_up: {episode_upper_reward:.1f} | R_sel: {episode_selector_reward:.1f} | Steps: {global_step}")    
        
        writer.add_scalar('Reward/Lower', episode_lower_reward, episode)  
        writer.add_scalar('Reward/Upper', episode_upper_reward, episode)  
        writer.add_scalar('Reward/Selector', episode_selector_reward, episode)  

    # Save Models  
    os.makedirs(args.output_dir, exist_ok=True)  
    torch.save(selector.q_net.state_dict(), os.path.join(args.output_dir, "selector.pth"))  
    torch.save(upper_agent.actor.state_dict(), os.path.join(args.output_dir, "upper_actor.pth"))  
    torch.save(lower_agent.actor.state_dict(), os.path.join(args.output_dir, "lower_actor.pth"))  
    print("Training Complete.")  


def evaluate_hrl(args):
    print("Starting Evaluation...")
    # 1. Environment Setup
    env = MyUsvEnv(
        num_targets=args.num_targets,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        record_flag=False, # We will manually record
        usv_length=args.usv_length,
        usv_width=args.usv_width,
        target_radius=args.target_radius,
        obstacle_radius_range=(args.obstacle_radius_min, args.obstacle_radius_max),
        use_random_shapes=args.use_random_shapes
    )
    
    # Dimensions
    obs = env.reset()
    selector_state_dim = get_selector_state(obs).shape[0]
    upper_state_dim = get_upper_state(obs).shape[0]
    lower_state_dim = get_lower_state(obs).shape[0]
    
    # Action Dims
    selector_action_dim = args.num_targets
    upper_action_dim = 2
    lower_action_dim = 2
    
    # Goal Dims
    upper_goal_dim = 2
    lower_goal_dim = 2
    
    # 2. Agents Setup & Load Models
    selector = SelectorAgent(selector_state_dim, selector_action_dim)
    upper_agent = TD3Agent(upper_state_dim, upper_goal_dim, upper_action_dim, max_action=args.subgoal_range)
    lower_agent = TD3Agent(lower_state_dim, lower_goal_dim, lower_action_dim, max_action=1.0)
    
    model_dir = args.output_dir
    if os.path.exists(os.path.join(model_dir, "selector.pth")):
        selector.q_net.load_state_dict(torch.load(os.path.join(model_dir, "selector.pth"), map_location=device))
        upper_agent.actor.load_state_dict(torch.load(os.path.join(model_dir, "upper_actor.pth"), map_location=device))
        lower_agent.actor.load_state_dict(torch.load(os.path.join(model_dir, "lower_actor.pth"), map_location=device))
        print("Models loaded successfully.")
    else:
        print("No models found, running with random weights.")

    eval_dir = os.path.join(args.output_dir, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)

    for episode in range(args.eval_episodes):
        obs_dict = env.reset()
        done = False
        truncated=False
        crush = False
        episode_lower_reward = 0
        attempts=0
        csv_file = os.path.join(eval_dir, f"hrl_path_episode_{episode+1}.csv")
        if os.path.exists(csv_file):
            os.remove(csv_file)
        
        # Selector chooses a high-level goal
        while not (done or attempts >= args.num_targets or crush):
            attempts+=1
            selector_state = get_selector_state(obs_dict)
            target_status = obs_dict['targets'][2::3]
            valid_mask = (target_status == 0)
            
            # if not np.any(valid_mask):
            #     continue
                
            target_idx = selector.select_action(selector_state, valid_mask, eval=True)
            g_des_global = get_target_pos(obs_dict, target_idx, args.num_targets)
            
            print(f"Episode {episode}: Selector chose Target {target_idx}")
            
            step_count = 0
            while not (done or crush or truncated):
                upper_state = get_upper_state(obs_dict)
                usv_pos = get_usv_pos(obs_dict)
                g_des_rel = g_des_global - usv_pos
                
                # Upper Action
                g_sub_rel = upper_agent.select_action(upper_state, g_des_rel, noise=0.0)
                
                # Lower Loop
                g_sub_global = usv_pos + g_sub_rel
                sub_done=False
                while not (done or sub_done or crush or truncated):
                    lower_state = get_lower_state(obs_dict)
                    curr_usv_pos = get_usv_pos(obs_dict)
                    curr_g_sub_rel = g_sub_global - curr_usv_pos
                    
                    lower_action = lower_agent.select_action(lower_state, curr_g_sub_rel, noise=0.0)
                    
                    next_obs_dict, r_low, sub_done, crush, truncated, info = env.step(lower_action,g_sub_global)
                    
                    

                    # Record Path
                    env.record_points(np.array([env.usv.x, env.usv.y]), csv_file)
                    
                    obs_dict = next_obs_dict
                    episode_lower_reward += r_low
                    
                    # Check if target reached
                    is_reached = (obs_dict['targets'][target_idx*3 + 2] == 1.0)
                    
                    if is_reached:
                        print(f"  Target {target_idx} reached!")
                        target_status = obs_dict['targets'][2::3]
                        valid_mask = (target_status == 0)
                        if np.any(valid_mask):
                            selector_state = get_selector_state(obs_dict)
                            target_idx = selector.select_action(selector_state, valid_mask, eval=True)
                            g_des_global = get_target_pos(obs_dict, target_idx, args.num_targets)
                            print(f"  Selector chose new Target {target_idx}")
                            # Break lower loop to let Upper generate new subgoal for new target
                            break
                        else:
                            done = True
                            break
                    
                    # if terminated or truncated:
                    #     done = True
                    #     break
                
                step_count += 1
            
        print(f"Episode {episode}: lower Reward {episode_lower_reward:.2f} Steps {step_count}, Collisions {info.get('collision', )}")  
        env.close_writer()
        
        # Draw path
        env.draw_path(csv_file, ifshow=False, ifsave=True, savePath=eval_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_targets", default=2, type=int)
    parser.add_argument("--num_obstacles", default=5, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--usv_length", default=3.0, type=float)
    parser.add_argument("--usv_width", default=1.5, type=float)
    parser.add_argument("--target_radius", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_min", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_max", default=10.0, type=float)
    parser.add_argument("--use_random_shapes", action="store_true", default=True)
    
    parser.add_argument("--max_episodes", default=500, type=int)
    # parser.add_argument("--max_steps", default=100000, type=int, help="Global max steps for training")
    parser.add_argument("--upper_steps", default=20, type=int, help="Upper steps per Selector step")
    parser.add_argument("--lower_steps", default=50, type=int, help="Lower steps per Upper step")
    parser.add_argument("--selector_batch_size", default=64, type=int)
    parser.add_argument("--actuator_batch_size", default=256, type=int)
    parser.add_argument("--selector_buffer_size", default=50000, type=int)
    parser.add_argument("--actuator_buffer_size", default=1000000, type=int)
    parser.add_argument("--warmup_steps", default=10000, type=int)
    
    parser.add_argument("--epsilon_start", default=1.0, type=float)
    parser.add_argument("--epsilon_end", default=0.1, type=float)
    parser.add_argument("--epsilon_decay", default=0.995, type=float)
    parser.add_argument("--exploration_noise", default=0.1, type=float)
    
    parser.add_argument("--subgoal_range", default=0.2, type=float, help="Range of sub-goal (normalized)")
    
    parser.add_argument("--output_dir", default="hrl_output", type=str)
    
    # Mode
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_episodes", default=5, type=int)
    
    args = parser.parse_args()
    
    if args.do_train:
        train_hrl(args)
    
    if args.do_eval:
        evaluate_hrl(args)
