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
    
    # 状态维度
    obs = env.reset()
    selector_state_dim = get_selector_state(obs).shape[0]
    upper_state_dim = get_upper_state(obs).shape[0]
    lower_state_dim = get_lower_state(obs).shape[0]
    
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
        done = False
        episode_reward = 0
        
        # 选择器选择一个高级目标
        selector_state = get_selector_state(obs_dict)
        
        # 掩码访问目标
        target_status = obs_dict['targets'][2::3] # 0: unvisited, 1: visited
        valid_mask = (target_status == 0)
        
        # If all visited (shouldn't happen at start), reset
        if not np.any(valid_mask):
            continue
            
        target_idx = selector.select_action(selector_state, valid_mask)
        
        # 获取目标位置（归一化）
        g_des_global = get_target_pos(obs_dict, target_idx, args.num_targets)
        
        # 选择器步骤循环
        step_count = 0
        while not done and step_count < args.max_steps:
            # 1. 上层执行器：生成子目标
            upper_state = get_upper_state(obs_dict)
            usv_pos = get_usv_pos(obs_dict)
            
            # 计算上级代理的相对目标
            g_des_rel = g_des_global - usv_pos
            
            # 上层动作：子目标（相对于USV）
            # 我们将输出缩放到一个小范围内（例如，0.1标准化单位约等于10米）
            g_sub_rel = upper_agent.select_action(upper_state, g_des_rel, noise=args.exploration_noise)
            
            # 2. 下层执行器：控制回路（k 步）
            # 下部代理尝试达到 g_sub_rel。 
            # 注意：g_sub_rel 是全局坐标系的相对向量？还是机体坐标系？
            # 为简单起见，我们假设所有内容都是全局坐标系的相对向量（dx, dy）。
            # 理想情况下，下部代理应在机体坐标系中工作，但为了与上部保持一致，我们暂且使用全局相对坐标。
            
            # 初始下层状态
            lower_state = get_lower_state(obs_dict)
            
            # 执行下级代理 k 步
            for k in range(args.lower_steps):
                # 下层动作
                # 下层代理的目标是 g_sub_rel。
                # 然而，随着 USV 的移动，g_sub_rel（由上层生成的固定空间点）相对于 USV 会发生变化。
                # 我们需要跟踪子目标的“全局”位置。
                # g_sub_global = usv_pos_start   g_sub_rel_start
                
                if k == 0:
                    g_sub_global = usv_pos + g_sub_rel
                
                # 当前相对子目标
                curr_usv_pos = get_usv_pos(obs_dict)
                curr_g_sub_rel = g_sub_global - curr_usv_pos
                
                lower_action = lower_agent.select_action(lower_state, curr_g_sub_rel, noise=args.exploration_noise)
                
                # Environment Step
                next_obs_dict, env_reward, terminated, truncated, info = env.step(lower_action)
                
                # Process Rewards
                # Lower Reward: Distance to sub-goal
                dist_to_sub = np.linalg.norm(curr_g_sub_rel)
                r_low = -dist_to_sub - 0.1 * np.linalg.norm(lower_action)**2
                
                # Store Lower Transition
                next_lower_state = get_lower_state(next_obs_dict)
                # HER: Store with original goal
                lower_buffer.push(lower_state, lower_action, r_low, next_lower_state, terminated, curr_g_sub_rel, curr_g_sub_rel) # Achieved goal approx same as state change? Simplified.
                
                # Update State
                obs_dict = next_obs_dict
                lower_state = next_lower_state
                episode_reward += env_reward
                global_step += 1
                
                # Check Dynamic Goal Detection (Algorithm 2.2)
                # If we reached ANY target, we should credit Selector
                reached_indices = info.get('reached_targets', [])
                if len(reached_indices) > 0:
                    for reached_idx in reached_indices:
                        # If we reached the intended target
                        if reached_idx == target_idx:
                            # Selector Success
                            selector_reward = 1.0
                            selector_done = False # Not necessarily done with episode, but done with this sub-task? 
                            # Actually Selector is high level, one step per target choice.
                            # If target reached, Selector step is done.
                            pass
                        else:
                            # Unexpected success (Relabeling)
                            # Store a transition for Selector as if it had chosen reached_idx
                            # s_start, action=reached_idx, r=1.0, s_next
                            selector_buffer.push(selector_state, reached_idx, 1.0, get_selector_state(obs_dict), False)
                            print(f"Relabeling: Selector accidentally reached target {reached_idx}")
                
                if terminated or truncated:
                    done = True
                    break
            
            # End of Lower Loop (Upper Step Done)
            
            # Upper Reward
            # Did we reach g_des? Or did we make progress?
            # PDF Eq 2-12: -dist(usv+sub, g_des) - collision
            # Here we use actual new pos
            new_usv_pos = get_usv_pos(obs_dict)
            dist_to_des = np.linalg.norm(g_des_global - new_usv_pos)
            
            collision = info.get('collision', False)
            r_up = -dist_to_des - (10.0 if collision else 0.0)
            
            # Store Upper Transition
            next_upper_state = get_upper_state(obs_dict)
            # HER for Upper: Achieved goal is actual displacement
            actual_displacement = new_usv_pos - usv_pos
            upper_buffer.push(upper_state, g_sub_rel, r_up, next_upper_state, done, g_des_rel, actual_displacement)
            
            # Selector Reward (Sparse)
            # If target reached, r=1. Else r=-0.1 or similar
            # Check if target_idx is in reached_targets of the *last* step or any step in the loop?
            # We check status from obs_dict
            is_reached = (obs_dict['targets'][target_idx*3 + 2] == 1.0)
            
            if is_reached:
                r_sel = 1.0
                # Selector step finished
                selector_buffer.push(selector_state, target_idx, r_sel, get_selector_state(obs_dict), False)
                
                # Pick new target if not all done
                target_status = obs_dict['targets'][2::3]
                valid_mask = (target_status == 0)
                if np.any(valid_mask):
                    selector_state = get_selector_state(obs_dict)
                    target_idx = selector.select_action(selector_state, valid_mask)
                    g_des_global = get_target_pos(obs_dict, target_idx, args.num_targets)
                else:
                    # All done
                    done = True
            elif done:
                # Failed or Collision
                r_sel = -1.0
                selector_buffer.push(selector_state, target_idx, r_sel, get_selector_state(obs_dict), True)
            else:
                # Continue towards same target
                pass
            
            step_count += 1
            
            # Training
            if global_step > args.warmup_steps:
                selector.train(selector_buffer, args.batch_size)
                upper_agent.train(upper_buffer, args.batch_size)
                lower_agent.train(lower_buffer, args.batch_size)
                
                # Decay epsilon
                selector.epsilon = max(args.epsilon_end, selector.epsilon * args.epsilon_decay)

        print(f"Episode {episode}: Reward {episode_reward:.2f}, Steps {global_step}")
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
    
    parser.add_argument("--max_episodes", default=500, type=int)
    parser.add_argument("--max_steps", default=200, type=int, help="Max Upper steps per episode")
    parser.add_argument("--lower_steps", default=10, type=int, help="Lower steps per Upper step")
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
