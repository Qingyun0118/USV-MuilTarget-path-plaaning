"""HTD3算法实现 - 基于分层TD3的多目标路径规划"""
import argparse
from collections import defaultdict
import os
import random
from dataclasses import dataclass, field
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class QNet(nn.Module):
    """Q-network (critic) for HTD3 algorithm."""

    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state + dim_action, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], -1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class PolicyNet(nn.Module):
    """Policy Network for HTD3 algorithm."""

    def __init__(self, dim_state, dim_action, max_action=2.0):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(dim_state, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, dim_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.max_action * torch.tanh(self.fc4(x))
        return x


class HTD3(nn.Module):
    """Hierarchical Twin Delayed Deep Deterministic Policy Gradient (HTD3)"""
    
    def __init__(self, dim_state, dim_action, max_action, num_targets=3):
        super().__init__()

        self.max_action = max_action
        self.num_targets = num_targets
        
        # Initialize networks
        self.Q1 = QNet(dim_state, dim_action)
        self.Q2 = QNet(dim_state, dim_action)
        self.Mu = PolicyNet(dim_state, dim_action, max_action)
        
        # Target networks
        self.target_Q1 = QNet(dim_state, dim_action)
        self.target_Q2 = QNet(dim_state, dim_action)
        self.target_Mu = PolicyNet(dim_state, dim_action, max_action)
        
        # Initialize target networks with same weights
        self.target_Q1.load_state_dict(self.Q1.state_dict())
        self.target_Q2.load_state_dict(self.Q2.state_dict())
        self.target_Mu.load_state_dict(self.Mu.state_dict())

    def get_action(self, state, explore=False, exploration_noise=0.1):
        """Get action from policy with optional exploration noise"""
        with torch.no_grad():
            action = self.Mu(state)
            if explore:
                noise = torch.randn_like(action) * exploration_noise
                action = torch.clamp(action + noise, -self.max_action, self.max_action)
            return action

    def compute_value_loss(self, args, s_batch, a_batch, r_batch, d_batch, next_s_batch):
        """Compute the value loss for the two Q networks with target policy smoothing."""

        with torch.no_grad():
            # Target policy smoothing - add noise to target actions
            next_actions = self.target_Mu(next_s_batch)
            noise = torch.clamp(
                torch.randn_like(next_actions) * args.policy_noise,
                -args.noise_clip,
                args.noise_clip,
            )
            next_actions = torch.clamp(next_actions + noise, 
                                     min=-self.max_action, max=self.max_action)

            # Clipped double Q-learning - take minimum of two target Q values
            target_Q1 = self.target_Q1(next_s_batch, next_actions).squeeze()
            target_Q2 = self.target_Q2(next_s_batch, next_actions).squeeze()
            target_Q = torch.min(target_Q1, target_Q2)
            
            # TD target
            y = r_batch + args.gamma * target_Q * (1 - d_batch)

        # Current Q values
        current_Q1 = self.Q1(s_batch, a_batch).squeeze()
        current_Q2 = self.Q2(s_batch, a_batch).squeeze()
        
        # MSE losses
        value_loss1 = F.mse_loss(current_Q1, y)
        value_loss2 = F.mse_loss(current_Q2, y)
        
        return value_loss1, value_loss2

    def compute_policy_loss(self, s_batch):
        """Compute policy loss using deterministic policy gradient"""
        actions = self.Mu(s_batch)
        policy_loss = -self.Q1(s_batch, actions).mean()
        return policy_loss

    def soft_update(self, tau=0.001):
        """Soft update of target networks with parameter tau"""
        for target_param, param in zip(self.target_Q1.parameters(), self.Q1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_Q2.parameters(), self.Q2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_Mu.parameters(), self.Mu.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


@dataclass
class ReplayBuffer:
    """Experience replay buffer for HTD3"""
    maxsize: int
    size: int = 0
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    done: list = field(default_factory=list)
    next_state: list = field(default_factory=list)

    def push(self, state, action, reward, done, next_state):
        """Add experience to buffer"""
        if self.size < self.maxsize:
            self.state.append(state)
            self.action.append(action)
            self.reward.append(reward)
            self.done.append(done)
            self.next_state.append(next_state)
        else:
            position = self.size % self.maxsize
            self.state[position] = state
            self.action[position] = action
            self.reward[position] = reward
            self.done[position] = done
            self.next_state[position] = next_state
        self.size += 1

    def sample(self, n):
        """Sample batch of experiences"""
        total_number = min(self.size, self.maxsize)
        indices = np.random.randint(total_number, size=n)
        
        state = [self.state[i] for i in indices]
        action = [self.action[i] for i in indices]
        reward = [self.reward[i] for i in indices]
        done = [self.done[i] for i in indices]
        next_state = [self.next_state[i] for i in indices]
        
        return state, action, reward, done, next_state


class TrainingInfo:
    """Track training statistics"""
    def __init__(self):
        self.log = defaultdict(list)
        self.episode_length = 0
        self.episode_reward = 0
        self.max_episode_reward = -float("inf")

    def update(self, reward, done):
        """Update episode statistics"""
        self.episode_length += 1
        self.episode_reward += reward
        
        if done:
            self.log["episode_length"].append(self.episode_length)
            self.log["episode_reward"].append(self.episode_reward)
            
            if self.episode_reward > self.max_episode_reward:
                self.max_episode_reward = self.episode_reward
            
            # Reset for next episode
            self.episode_length = 0
            self.episode_reward = 0


def dict_state_to_tensor(state_dict, device):
    """Convert state dictionary to tensor for neural network input"""
    # Extract and flatten all components
    components = []
    
    # Normalized position
    components.extend(state_dict["x_norm"])
    components.extend(state_dict["y_norm"])
    
    # Obstacle information
    components.extend(state_dict["Rou_o_norm"].flatten())
    components.extend(state_dict["Alpha_o_norm"].flatten())
    
    # Priority vector
    components.extend(state_dict["Priority_vector"])
    
    # Target information
    components.extend(state_dict["rou_t_norm"])
    components.extend(state_dict["alpha_t_norm"].flatten())
    
    return torch.tensor(components, dtype=torch.float32).to(device)


def train_htd3(args, env, agent: HTD3):
    """Train the HTD3 agent for multi-objective path planning"""

    # Optimizers with different learning rates for actor and critic
    Q1_optimizer = torch.optim.Adam(agent.Q1.parameters(), lr=args.lr_critic)
    Q2_optimizer = torch.optim.Adam(agent.Q2.parameters(), lr=args.lr_critic)
    Mu_optimizer = torch.optim.Adam(agent.Mu.parameters(), lr=args.lr_actor)
    
    # Experience replay buffer
    replay_buffer = ReplayBuffer(maxsize=args.buffer_size)
    training_info = TrainingInfo()
    
    # Training loop
    state_dict= env.reset()
    state = dict_state_to_tensor(state_dict, args.device)
    
    for step in range(args.max_steps):
        # Action selection with exploration
        if step < args.warmup_steps:
            # Random exploration during warmup
            action = env.action_space.sample()
        else:
            # Policy-based action with exploration noise
            with torch.no_grad():
                action_tensor = agent.get_action(state.unsqueeze(0), explore=True, 
                                               exploration_noise=args.exploration_noise)
                action = action_tensor.cpu().numpy()[0]
        
        # Environment step
        next_state_dict, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = dict_state_to_tensor(next_state_dict, args.device)
        
        # Store experience (convert to CPU for storage)
        replay_buffer.push(state.cpu().numpy(), action, reward, done, next_state.cpu().numpy())
        
        # Update state and training info
        state = next_state
        training_info.update(reward, done)
        
        # Reset environment if episode done
        if done:
            state_dict= env.reset()#
            state = dict_state_to_tensor(state_dict, args.device)
            
            # Logging
            episode_reward = training_info.log["episode_reward"][-1]
            episode_length = training_info.log["episode_length"][-1]
            
            value_loss = training_info.log["value_loss1"][-1] if training_info.log["value_loss1"] else 0
            print(f"Step {step}: Reward={episode_reward:.1f}, Length={episode_length}, "
                  f"MaxReward={training_info.max_episode_reward:.1f}, ValueLoss={value_loss:.4f}")
            
            # Save best model
            if episode_reward >= training_info.max_episode_reward:
                save_path = os.path.join(args.output_dir, "best_model.pth")
                # torch.save({
                #     'actor': agent.Mu.state_dict(),
                #     'critic1': agent.Q1.state_dict(), 
                #     'critic2': agent.Q2.state_dict(),
                #     'episode_reward': episode_reward
                # }, save_path)
                torch.save(agent.Mu.state_dict(), save_path)
        
        # Training after warmup
        if step > args.warmup_steps and step % args.update_frequency == 0:
            # Sample batch
            s_batch, a_batch, r_batch, d_batch, ns_batch = replay_buffer.sample(args.batch_size)
            
            # Convert to tensors and move to device
            s_batch = torch.tensor(np.array(s_batch), dtype=torch.float32).to(args.device)
            a_batch = torch.tensor(np.array(a_batch), dtype=torch.float32).to(args.device)
            r_batch = torch.tensor(np.array(r_batch), dtype=torch.float32).to(args.device)
            d_batch = torch.tensor(np.array(d_batch), dtype=torch.float32).to(args.device)
            ns_batch = torch.tensor(np.array(ns_batch), dtype=torch.float32).to(args.device)
            
            # Update critics
            value_loss1, value_loss2 = agent.compute_value_loss(args, s_batch, a_batch, 
                                                              r_batch, d_batch, ns_batch)
            
            Q1_optimizer.zero_grad()
            value_loss1.backward()
            Q1_optimizer.step()
            
            Q2_optimizer.zero_grad()
            value_loss2.backward()
            Q2_optimizer.step()
            
            training_info.log["value_loss1"].append(value_loss1.item())
            training_info.log["value_loss2"].append(value_loss2.item())
            
            # Delayed policy updates
            if step % args.policy_update_freq == 0:
                policy_loss = agent.compute_policy_loss(s_batch)
                Mu_optimizer.zero_grad()
                policy_loss.backward()
                Mu_optimizer.step()
                
                # Soft update target networks
                agent.soft_update(args.tau)
                
                training_info.log["policy_loss"].append(policy_loss.item())
        
        # Logging and visualization
        if step % args.log_interval == 0 and step > args.warmup_steps:
            # Plot losses
            if training_info.log["value_loss1"]:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(training_info.log["value_loss1"], label='Q1 Loss')
                plt.plot(training_info.log["value_loss2"], label='Q2 Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Value Loss')
                plt.legend()
                plt.title('Critic Losses')
                
                plt.subplot(1, 2, 2)
                plt.plot(training_info.log["episode_reward"])
                plt.xlabel('Episode')
                plt.ylabel('Episode Reward')
                plt.title('Training Progress')
                
                plt.tight_layout()
                plt.savefig(f"{args.output_dir}/training_progress.png", dpi=150, bbox_inches="tight")
                plt.close()
    
    # Save final model
    final_save_path = os.path.join(args.output_dir, "final_model.pth")
    # torch.save({
    #     'actor': agent.Mu.state_dict(),
    #     'critic1': agent.Q1.state_dict(),
    #     'critic2': agent.Q2.state_dict(),
    #     'training_info': training_info.log
    # }, final_save_path)
    torch.save(agent.Mu.state_dict(), final_save_path)
    
    print(f"Training completed. Models saved to {args.output_dir}")


def evaluate_htd3(args, env, agent):
    """Evaluate the trained HTD3 agent"""
    
    # Load trained model
    model_path = os.path.join(args.output_dir, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.output_dir, "final_model.pth")
    # model_path = os.path.join(args.output_dir, "final_model.pth")
    # checkpoint = torch.load(model_path)
    # agent.Mu.load_state_dict(checkpoint['actor'])
    state_dict = torch.load(model_path,map_location=args.device)
    agent.Mu.load_state_dict(state_dict)
    
    print("Loaded trained model for evaluation")
    
    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(args.eval_episodes):
        state_dict= env.reset()
        state = dict_state_to_tensor(state_dict, args.device)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < args.max_episode_steps:
            with torch.no_grad():
                action_tensor = agent.get_action(state.unsqueeze(0), explore=True,exploration_noise=0.0)
                action = action_tensor.cpu().numpy()[0]
            
            next_state_dict, reward, terminated, truncated, _ = env.step(action)
            
            # points=(float(next_state_dict["x_norm"]*MAX),float(next_state_dict["y_norm"]*MAX))
            # print(points)
            # env.record_points(points)
            
            done = terminated or truncated
            
            state = dict_state_to_tensor(next_state_dict, args.device)
            episode_reward += reward
            episode_length += 1
            
            # Render if needed
            # if args.render:
            #     env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Evaluation Episode {episode + 1}: Reward={episode_reward:.1f}, Length={episode_length}")
    
    # Print evaluation results
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    # success_rate = np.mean([r > 0 for r in episode_rewards]) * 100
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Length: {avg_length:.2f}")
    # print(f"Success Rate: {success_rate:.1f}%")
    env.draw_path("usv_path_points_1.csv")


def main():
    """Main function for HTD3 training and evaluation"""
    parser = argparse.ArgumentParser(description="HTD3 for Multi-objective Path Planning")
    
    # Environment parameters
    parser.add_argument("--env", default="MyUsvEnv", type=str, help="Environment name")
    parser.add_argument("--dim_state", default=22, type=int, help="State dimension")  # Based on USV env
    parser.add_argument("--dim_action", default=2, type=int, help="Action dimension")  # [rou, alpha]
    parser.add_argument("--max_action", default=25.0, type=float, help="Maximum action value")
    
    # Training parameters
    parser.add_argument("--max_steps", default=100000, type=int, help="Total training steps")
    parser.add_argument("--warmup_steps", default=5000, type=int, help="Random exploration steps")
    parser.add_argument("--batch_size", default=64, type=int, help="Training batch size")
    parser.add_argument("--buffer_size", default=100000, type=int, help="Replay buffer size")
    parser.add_argument("--update_frequency", default=1, type=int, help="Training frequency")
    parser.add_argument("--policy_update_freq", default=2, type=int, help="Policy update frequency")
    
    # Learning rates
    parser.add_argument("--lr_actor", default=0.0005, type=float, help="Actor learning rate")
    parser.add_argument("--lr_critic", default=0.0003, type=float, help="Critic learning rate")
    
    # Algorithm parameters
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--tau", default=0.001, type=float, help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, type=float, help="Policy noise")
    parser.add_argument("--noise_clip", default=0.5, type=float, help="Noise clipping")
    parser.add_argument("--exploration_noise", default=0.1, type=float, help="Exploration noise")
    
    # Evaluation parameters
    parser.add_argument("--eval_episodes", default=10, type=int, help="Evaluation episodes")
    parser.add_argument("--max_episode_steps", default=1000, type=int, help="Max steps per episode")
    
    # Logging and output
    parser.add_argument("--log_interval", default=1000, type=int, help="Logging interval")
    parser.add_argument("--output_dir", default="htd3_output", type=str, help="Output directory")
    # parser.add_argument("--seed", default=42, type=int, help="Random seed")
    
    # Mode selection
    parser.add_argument("--do_train", action="store_true", help="Enable training")
    parser.add_argument("--do_eval", action="store_true", help="Enable evaluation")
    # parser.add_argument("--render", action="store_true", help="Render during evaluation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    
    # Initialize environment
    from my_USV_Env import MyUsvEnv,MyUsvEnv_Gym
    # env_train = MyUsvEnv(record_flag=False)
    # env_eval = MyUsvEnv(record_flag=True)#eval_episodes must be 1 for recording points
    env_train = MyUsvEnv_Gym(record_flag=False)
    env_eval = MyUsvEnv_Gym(record_flag=True)#eval_episodes must be 1 for recording points

    # Initialize agent
    agent = HTD3(
        dim_state=args.dim_state,
        dim_action=args.dim_action, 
        max_action=args.max_action,
        num_targets=3  # Based on your USV environment
    ).to(args.device)  # Now this should work since HTD3 inherits from nn.Module
    
    # Training
    if args.do_train:
        print("Starting HTD3 training...")
        train_htd3(args, env_train, agent)
    
    # Evaluation
    if args.do_eval:
        print("Starting HTD3 evaluation...")
        evaluate_htd3(args, env_eval, agent)
    
    # env.close()


if __name__ == "__main__":
    main()