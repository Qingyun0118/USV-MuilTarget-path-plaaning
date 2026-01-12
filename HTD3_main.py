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
from my_USV_Env import MyUsvEnv

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
    """Replay Buffer with Hindsight Experience Replay for Actuators"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, desired_goal, achieved_goal):
        self.buffer.append((state, action, reward, next_state, done, desired_goal, achieved_goal))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, desired_goal, achieved_goal = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(desired_goal)).to(device),
            torch.FloatTensor(np.array(achieved_goal)).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

# --- Neural Networks ---
class SelectorQNet(nn.Module):
    """DQN for Selector: State -> Q-values for each target"""
    def __init__(self, state_dim, num_actions):
        super(SelectorQNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Actor(nn.Module):
    """TD3 Actor: (State + Goal) -> Action"""
    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
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
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

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
    def __init__(self, state_dim, num_targets, lr=1e-3, gamma=0.99, epsilon=0.1):
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
    def __init__(self, state_dim, goal_dim, action_dim, max_action, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
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
    """Extract state for Selector: USV info + Targets info"""
    # o_usv: 7, targets: 3*N
    return np.concatenate([obs_dict['o_usv'], obs_dict['targets']])

def get_upper_state(obs_dict):
    """Extract state for Upper Actuator: USV info + Lidar"""
    # o_usv: 7, o_detector: 13
    return np.concatenate([obs_dict['o_usv'], obs_dict['o_detector']])

def get_lower_state(obs_dict):
    """Extract state for Lower Actuator: USV velocities"""
    # o_usv: [x, y, u, v, r, sin, cos]
    # We need [u, v, r] which are indices 2, 3, 4
    return obs_dict['o_usv'][2:5]

def get_target_pos(obs_dict, target_idx, num_targets):
    """Get global position of the selected target"""
    # targets: [x1, y1, s1, x2, y2, s2, ...]
    # We need to denormalize if they are normalized. 
    # In my_USV_Env, they are normalized by X_MAX, Y_MAX.
    # But here we might want to work with normalized values or denormalized.
    # Let's stick to normalized values for NN inputs, but we might need real values for distance calc.
    # The env returns normalized values in 'targets'.
    
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

def calc_lower_reward(current_pos, target_sub_global, action):  
    """  
    计算下层执行器奖励 (Eq 2-13)  
    R_low = -||p - g_sub|| - lambda * ||action||^2  
    """  
    dist = np.linalg.norm(current_pos - target_sub_global)  
    # 动作平滑惩罚 (lambda)  
    action_penalty = 0.1 * np.linalg.norm(action)**2  
    
    # 这里的奖励是密集的距离奖励  
    return -dist - action_penalty  

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

    # 显示环境信息
    inf = env.get_environment_info()
    print(f"目标数量: {inf['num_targets']}")
    print(f"障碍物数量: {inf['num_obstacles']}")
    print(f"随机种子: {inf['seed']}")
    print(f"起始位置: {inf['start_pos']}")
    print(f"USV尺寸: {inf['usv_size']}")
    print("\n目标位置:")
    for i, t in enumerate(inf['targets']):
        print(f"  目标{i}: ({t['x']:.1f}, {t['y']:.1f}), 半径={t['radius']}")
    print("\n障碍物信息:")
    for i, o in enumerate(inf['obstacles']):
        print(f"  障碍物{i}: 形状={o['shape']}, 位置=({o['x']:.1f}, {o['y']:.1f})")

    # 状态维度
    obs = env.reset()
    selector_state_dim = get_selector_state(obs).shape[0]
    upper_state_dim = get_upper_state(obs).shape[0]
    lower_state_dim = get_lower_state(obs).shape[0]
    print(obs)
    # 动作维度
    selector_action_dim = args.num_targets
    upper_action_dim = 2 # (dx, dy) 子目标
    lower_action_dim = 2 # (throttle, rudder)
    
    # 目标维度
    upper_goal_dim = 2 # 相对目标位置（dx, dy）
    lower_goal_dim = 2 # 相对子目标位置 (dx, dy)
    
    # 2. 智能体初始化
    selector = SelectorAgent(selector_state_dim, selector_action_dim, epsilon=args.epsilon_start)
    upper_agent = TD3Agent(upper_state_dim, upper_goal_dim, upper_action_dim, max_action=args.subgoal_range)
    lower_agent = TD3Agent(lower_state_dim, lower_goal_dim, lower_action_dim, max_action=1.0)
    
    # 3. 缓冲区设置
    selector_buffer = ReplayBuffer(args.buffer_size)
    upper_buffer = HER_ReplayBuffer(args.buffer_size)
    lower_buffer = HER_ReplayBuffer(args.buffer_size)
    
    writer = SummaryWriter(f'htd3_output/runs/HRL_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    global_step = 0
    
    for episode in range(args.max_episodes):  
        obs_dict = env.reset()  
        usv_pos = get_usv_pos(obs_dict)  
        
        # 记录 Episode 开始时的未访问目标  
        target_status = obs_dict['targets'][2::3] # 0: unvisited  
        valid_mask = (target_status == 0)  
        print('目标掩码：',valid_mask)
        episode_reward = 0  
        done = False  
        
        # -------------------------------------------------  
        # Outer Loop: Selector (选择目标)  
        # -------------------------------------------------  
        while not done and np.any(valid_mask):  
            selector_state = get_selector_state(obs_dict)  
            
            # Selector 选择一个目标索引 k  
            target_idx = selector.select_action(selector_state, valid_mask)  
            g_des_global = get_target_pos(obs_dict, target_idx, args.num_targets)  
            print('决策器选择的目标：{}，目标位置：{}'.format(target_idx, g_des_global*100))
            # Selector 的单步奖励累积（直到达到这个目标或超时）  
            selector_step_reward = 0  
            selector_done = False  
            
            # 记录 Selector 开始时的状态用于存储  
            selector_start_state = selector_state  
            selector_start_pos = usv_pos  
            
            # -------------------------------------------------  
            # Middle Loop: Upper Actuator (生成子目标)  
            # -------------------------------------------------  
            # 限制 Upper Actuator 在一定步数内尝试到达 Selector 的目标  
            upper_steps_limit = args.max_steps // args.num_targets   
            
            for u_step in range(upper_steps_limit):  
                upper_state = get_upper_state(obs_dict)  
                usv_pos = get_usv_pos(obs_dict)  
                
                # 计算相对最终目标的向量 (Obs for Upper)  
                g_des_rel = g_des_global - usv_pos  
                
                # Upper 选择动作：相对子目标 (dx, dy)  
                g_sub_rel = upper_agent.select_action(upper_state, g_des_rel, noise=args.exploration_noise)  
                # print('执行器上层选择的子目标相对usv位置：{}'.format(g_sub_rel))
                # 计算 Upper 的奖励 (基于生成的子目标质量，不需要等下层执行完)  
                r_up = calc_upper_reward(usv_pos, g_sub_rel, g_des_global, env.obstacles)  
                
                # 子目标的全局坐标 (用于给 Lower 追踪)  
                g_sub_global = usv_pos + g_sub_rel  
                print('执行器上层选择的子目标绝对位置：{}'.format(g_sub_global))  
                # -------------------------------------------------  
                # Inner Loop: Lower Actuator (动力学控制)  
                # -------------------------------------------------  
                # 下层尝试在 k 步内通过底层控制到达 g_sub_global  
                for l_step in range(args.lower_steps):  
                    lower_state = get_lower_state(obs_dict)  
                    curr_usv_pos = get_usv_pos(obs_dict)  
                    
                    # Lower 的目标是相对于当前位置的子目标向量  
                    curr_g_sub_rel = g_sub_global - curr_usv_pos  
                    
                    # Lower 选择动作：[推力, 舵角]  
                    lower_action = lower_agent.select_action(lower_state, curr_g_sub_rel, noise=args.exploration_noise)  
                    
                    # 环境交互  
                    next_obs_dict, env_reward, terminated, truncated, info = env.step(lower_action)  
                    next_usv_pos = get_usv_pos(next_obs_dict)  
                    
                    # Lower 奖励计算  
                    r_low = calc_lower_reward(curr_usv_pos, g_sub_global, lower_action)  
                    
                    # 存储 Lower 经验 (HER 风格: 状态, 动作, 奖励, 下一状态, Done, 期望目标, 实际达成)  
                    # 注意：Lower 的 Goal 是相对位移  
                    actual_move = next_usv_pos - curr_usv_pos  
                    lower_buffer.push(  
                        lower_state, lower_action, r_low, get_lower_state(next_obs_dict),   
                        terminated, curr_g_sub_rel, actual_move  
                    )  
                    
                    # 状态更新  
                    obs_dict = next_obs_dict  
                    usv_pos = next_usv_pos  
                    episode_reward += env_reward  
                    global_step += 1  
                    
                    # === 关键：动态目标检测 (Algorithm 2.2) ===  
                    reached_indices = info.get('reached_targets', [])  
                    
                    if len(reached_indices) > 0:  
                        for reached_idx in reached_indices:  
                            # 1. 正常情况：到达了 Selector 指定的目标  
                            if reached_idx == target_idx:  
                                print(f"Selector Success: Reached target {target_idx}")  
                                selector_step_reward = 1.0  
                                selector_done = True # Selector 的这步决策完成了  
                            
                            # 2. 意外惊喜 (Relabeling)：到达了其他未访问的目标  
                            elif obs_dict['targets'][reached_idx*3 + 2] == 1.0: # 确保是刚变成1的  
                                print(f"**Relabeling**: Accidentally reached {reached_idx} while aiming for {target_idx}")  
                                # 构造一条虚拟经验：就好像 Selector 当初选的是 reached_idx  
                                # State: selector_start_state  
                                # Action: reached_idx  
                                # Reward: 1.0 (成功)  
                                # Next State: get_selector_state(next_obs_dict)  
                                selector_buffer.push(  
                                    selector_start_state, reached_idx, 1.0,   
                                    get_selector_state(next_obs_dict), False  
                                )  
                                # 注意：如果是意外到达，不结束当前对 target_idx 的追求，  
                                # 除非 target_idx 也在 reached_indices 里，或者策略要求重新规划  
                                # 这里我们选择让 Selector 重新规划（跳出 Upper/Lower 循环）  
                                selector_done = True   
                    
                    # 训练 Lower Agent  
                    if len(lower_buffer) > args.batch_size:  
                        lower_agent.train(lower_buffer, args.batch_size)  

                    if terminated or truncated or selector_done:  
                        break  
                # --- End of Lower Loop ---  
                
                # 存储 Upper 经验  
                next_upper_state = get_upper_state(obs_dict)  
                # Upper 的实际达成是：最终位置 - 初始位置  
                actual_upper_displacement = usv_pos - selector_start_pos # 近似  
                
                upper_buffer.push(  
                    upper_state, g_sub_rel, r_up, next_upper_state,   
                    selector_done, g_des_rel, actual_upper_displacement  
                )  
                
                # 训练 Upper Agent  
                if len(upper_buffer) > args.batch_size:  
                    upper_agent.train(upper_buffer, args.batch_size)  
                
                if terminated or truncated or selector_done:  
                    break  
            # --- End of Upper Loop ---  
            
            # 存储 Selector 经验 (正常的)  
            # 如果没成功到达，Reward 为 -1 (或者 0，取决于稀疏奖励设计)  
            final_selector_reward = selector_step_reward if selector_done else -1.0  
            next_selector_state = get_selector_state(obs_dict)  
            
            # 判断整个 Episode 是否结束（所有目标都访问了）  
            target_status = obs_dict['targets'][2::3]  
            valid_mask = (target_status == 0)  
            all_done = not np.any(valid_mask)  
              
            selector_buffer.push(  
                selector_start_state, target_idx, final_selector_reward,   
                next_selector_state, all_done  
            )  
              
            # 训练 Selector  
            if len(selector_buffer) > args.batch_size:  
                selector.train(selector_buffer, args.batch_size)  
              
            if terminated or truncated:  
                done = True  
                break  
                  
        # --- End of Selector Loop ---  
          
        # 衰减 Epsilon  
        selector.epsilon = max(args.epsilon_end, selector.epsilon * args.epsilon_decay)  
          
        print(f"Episode {episode}: Total Reward {episode_reward:.2f}, Steps {global_step}, Collisions {info.get('collision', )}")  
        writer.add_scalar('Reward/Episode', episode_reward, episode)

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
        episode_reward = 0
        
        csv_file = os.path.join(eval_dir, f"hrl_path_episode_{episode+1}.csv")
        if os.path.exists(csv_file):
            os.remove(csv_file)
        
        # Selector chooses a high-level goal
        selector_state = get_selector_state(obs_dict)
        target_status = obs_dict['targets'][2::3]
        valid_mask = (target_status == 0)
        
        if not np.any(valid_mask):
            continue
            
        target_idx = selector.select_action(selector_state, valid_mask, eval=True)
        g_des_global = get_target_pos(obs_dict, target_idx, args.num_targets)
        
        print(f"Episode {episode}: Selector chose Target {target_idx}")
        
        step_count = 0
        while not done and step_count < args.max_steps:
            upper_state = get_upper_state(obs_dict)
            usv_pos = get_usv_pos(obs_dict)
            g_des_rel = g_des_global - usv_pos
            
            # Upper Action
            g_sub_rel = upper_agent.select_action(upper_state, g_des_rel, noise=0.0)
            
            # Lower Loop
            for k in range(args.lower_steps):
                if k == 0:
                    g_sub_global = usv_pos + g_sub_rel
                
                lower_state = get_lower_state(obs_dict)
                curr_usv_pos = get_usv_pos(obs_dict)
                curr_g_sub_rel = g_sub_global - curr_usv_pos
                
                lower_action = lower_agent.select_action(lower_state, curr_g_sub_rel, noise=0.0)
                
                next_obs_dict, env_reward, terminated, truncated, info = env.step(lower_action)
                
                # Record Path
                env.record_points(np.array([env.usv.x, env.usv.y]), csv_file)
                
                obs_dict = next_obs_dict
                episode_reward += env_reward
                
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
                
                if terminated or truncated:
                    done = True
                    break
            
            step_count += 1
            
        print(f"Episode {episode} finished. Reward: {episode_reward:.2f}")
        env.close_writer()
        
        # Draw path
        env.draw_path(csv_file, ifshow=False, ifsave=True, savePath=eval_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_targets", default=3, type=int)
    parser.add_argument("--num_obstacles", default=5, type=int)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--usv_length", default=3.0, type=float)
    parser.add_argument("--usv_width", default=1.5, type=float)
    parser.add_argument("--target_radius", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_min", default=3.0, type=float)
    parser.add_argument("--obstacle_radius_max", default=10.0, type=float)
    parser.add_argument("--use_random_shapes", action="store_true", default=True)
    
    parser.add_argument("--max_episodes", default=1000, type=int)
    parser.add_argument("--max_steps", default=200, type=int, help="Max Upper steps per episode")
    parser.add_argument("--lower_steps", default=2000, type=int, help="Lower steps per Upper step")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--buffer_size", default=50000, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    
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
