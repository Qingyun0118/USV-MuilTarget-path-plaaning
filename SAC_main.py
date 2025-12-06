
import os
import random
import numpy as np
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
import argparse
import datetime
import gymnasium as gym
import itertools
from torch.utils.tensorboard import SummaryWriter
from my_USV_Env import MyUsvEnv,MAX
from collections import defaultdict

# Constants
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Utility functions
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# Neural Network Architectures
# class ValueNetwork(nn.Module):
#     def __init__(self, num_inputs, hidden_dim):
#         super(ValueNetwork, self).__init__()
#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear3 = nn.Linear(hidden_dim, 1)
#         self.apply(weights_init_)

#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         x = self.linear3(x)
#         return x

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))
        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))
        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


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


# Soft Actor-Critic Agent
class SAC(nn.Module):
    def __init__(self, num_inputs, action_space, args):
        super().__init__()

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g., -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

# State flattening function
def flatten_state(state_dict):
    """
    Convert the dictionary state from MyUsvEnv into a flat numpy array.
    """
    # Order of components: x_norm, y_norm, Rou_o_norm, Alpha_o_norm, Priority_vector, rou_t_norm, alpha_t_norm
    # Ensure they are numpy arrays
    x = state_dict['x_norm'].flatten()
    y = state_dict['y_norm'].flatten()
    rou_o = state_dict['Rou_o_norm'].flatten()
    alpha_o = state_dict['Alpha_o_norm'].flatten()

    priority = state_dict['Priority_vector'].flatten()
    rou_t = state_dict['rou_t_norm'].flatten()
    alpha_t = state_dict['alpha_t_norm'].flatten()
    # Concatenate all components
    flat = np.concatenate([x, y, rou_o, alpha_o, priority, rou_t, alpha_t])
    return flat.astype(np.float32)

# # Environment wrapper to flatten states
# class FlattenEnv:
#     def __init__(self, env):
#         self.env = env
#         self.action_space = env.action_space
#         # Compute observation space dimension
#         sample_state = env.reset()
#         flat_dim = len(flatten_state(sample_state))
#         self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
#         # Set max episode steps if not defined
#         self._max_episode_steps = getattr(env, '_max_episode_steps', 1000)

#     def reset(self):
#         state_dict = self.env.reset()
#         return flatten_state(state_dict)

#     def step(self, action):
#         state_dict, reward, terminated, truncated, info = self.env.step(action)
#         flat_state = flatten_state(state_dict)
#         return flat_state, reward, terminated, truncated, info

#     def seed(self, seed):
#         # Seed the action space for reproducible sampling
#         if hasattr(self.env.action_space, 'seed'):
#             self.env.action_space.seed(seed)
#         # If the environment has a seed method, call it
#         if hasattr(self.env, 'seed'):
#             self.env.seed(seed)
#         # Also seed numpy and random for any internal randomness
#         np.random.seed(seed)
#         random.seed(seed)

#     def render(self):
#         return self.env.render()

#     def close(self):
#         if hasattr(self.env, 'close'):
#             return self.env.close()
#         # else do nothing

def train_sac(args,env,agent:SAC):

    # TensorBoard writer
    writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                              args.policy, "autotune" if args.automatic_entropy_tuning else ""))
    print(f"TensorBoard logs saved to "+'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                              args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    episode_rewards = []
    training_info = TrainingInfo()

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state_dict= env.reset()
        state = flatten_state(state_dict)

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for _ in range(args.updates_per_step):
                    # Update parameters of all the networks
                    qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    
                    # Log losses to TensorBoard
                    writer.add_scalar('loss/qf1', qf1_loss, updates)
                    writer.add_scalar('loss/qf2', qf2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/alpha', alpha_loss, updates)
                    writer.add_scalar('alpha/value', alpha, updates)
                    
                    updates += 1

            next_state_dict, reward, terminated, truncated, _ = env.step(action) # Step
            done = terminated or truncated
            next_state = flatten_state(next_state_dict)

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps ==  args.max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            training_info.update(reward, done)


        if total_numsteps > args.max_steps:
            break

        episode_rewards.append(episode_reward)
        # Log episode reward
        writer.add_scalar('reward/episode', episode_reward, i_episode)
        
        print("Episode: {}, total numsteps: {}, episode steps: {}, Current reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if episode_reward>=training_info.max_episode_reward:
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            agent.save_checkpoint(args.env_name, suffix='best', ckpt_path=best_model_path)
        
        # Plot training progress periodically
        # if i_episode % args.log_interval == 0:
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(episode_rewards, label='Episode Reward')
        #     plt.xlabel('Episode')
        #     plt.ylabel('Reward')
        #     plt.title('Training Rewards')
        #     plt.legend()
        #     plt.grid(True)
        #     plot_path = os.path.join(args.output_dir, f'training_progress.png')
        #     plt.savefig(plot_path)
        #     plt.close()
    
    # Save final model
    final_save_path = os.path.join(args.output_dir, "final_model.pth")
    agent.save_checkpoint(args.env_name, suffix='best', ckpt_path=final_save_path)
    
    writer.close()
    print(f"Training completed. Models saved to {args.output_dir}")


def evaluate_sac(args,env, agent):
    """
    Evaluate SAC agent and draw trajectory.
    Creates a new MyUsvEnv with record_flag=True for evaluation.
    """
    # Load best model if exists
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        agent.load_checkpoint(best_model_path, evaluate=True)
        print(f"Loaded best model from {best_model_path}")
    else:
        print("No best model found, using current agent weights.")

    eval_dir = os.path.join(args.output_dir, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)

    # # Create evaluation environment with recording enabled
    # raw_env = MyUsvEnv(record_flag=True)
    # # No need to wrap with FlattenEnv because we will flatten states manually
    # # but we can use FlattenEnv for consistency with training.
    # env = FlattenEnv(raw_env)
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)

    episode_rewards = []
    episode_lengths = []

    for episode in range(args.eval_episodes):
        csv_file = os.path.join(eval_dir, f"usv_path_episode_{episode+1}.csv")
        print(f"Recording points to {csv_file}")
        if os.path.exists(csv_file):
            os.remove(csv_file)

        state_dict = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < args.max_episode_steps:
            state = flatten_state(state_dict)
            action = agent.select_action(state, evaluate=True)
            next_state_dict, reward, terminated, truncated, _ = env.step(action)

            current_position = np.array([next_state_dict["x_norm"][0] * MAX, next_state_dict["y_norm"][0] * MAX])
            env.record_points(current_position,csv_file)

            done = terminated or truncated
            state=flatten_state(next_state_dict)
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Evaluation Episode {episode + 1}: Reward={episode_reward:.1f}, Length={episode_length}")

    # Draw path for the last episode (or all episodes)
    if args.eval_episodes == 1:
        csv_file = os.path.join(eval_dir, f"usv_path_episode_1.csv")
    else:
        csv_file = os.path.join(eval_dir, "usv_path_episode_1.csv")  # base name
    env.draw_path(csv_file, ifshow=True, ifsave=True, savePath=eval_dir, max_episode=args.eval_episodes)

    if args.eval_episodes > 1:
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        print(f"\nEvaluation Results over {args.eval_episodes} episodes:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Length: {avg_length:.2f}")

    env.close_writer()


def main():
    """Main function for SAC training and evaluation"""
    parser = argparse.ArgumentParser(description='SAC for Multi-objective Path Planning')
    
    # Environment parameters
    parser.add_argument('--env-name', default="MyUsvEnv",
                        help='Environment name (default: MyUsvEnv)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    
    parser.add_argument('--eval', type=bool, default=False,
                        help='Evaluates a policy a policy every 10 episode (default: False)')
    
    # Algorithm parameters
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')

    # Learning rates
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--max_steps', type=int, default=500000, metavar='N',
                        help='maximum number of steps (default: 500000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    
    # Evaluation parameters
    parser.add_argument('--max_episode_steps', type=int, default=1000,
                        help='Maximum steps per episode for evaluation (default: 1000)')
    parser.add_argument('--output_dir', default='sac_output', type=str,
                        help='Output directory for models and plots (default: sac_output)')
    parser.add_argument('--eval_episodes', type=int, default=1,
                        help='Number of episodes for final evaluation (default: 1)')
    
    # Logging and output
    # parser.add_argument('--log_interval', type=int, default=1,
    #                     help='Logging interval for plotting (default: 1)')
    
    # Mode selection
    parser.add_argument('--do_train', action='store_true',
                        help='Enable training ')
    parser.add_argument('--do_eval', action='store_true',
                        help='Enable evaluation only')
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Environment
    env = MyUsvEnv(record_flag=False)
    # env = FlattenEnv(raw_env)
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    # Agent
    sample_state = env.reset()
    flat_dim = len(flatten_state(sample_state))
    # print(flatten_state(sample_state)) 
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
    agent = SAC(observation_space.shape[0], env.action_space, args)
    
    # Training
    if args.do_train:
        print("Starting SAC training...")
        train_sac(args, env, agent)
    
    # Evaluation
    if args.do_eval:
        print("Starting SAC evaluation...")
        evaluate_sac(args,env, agent)

# Main training function
if __name__ == '__main__':
    main()
    