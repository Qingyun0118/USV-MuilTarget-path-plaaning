"""10.4节TD3算法实现。
"""
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

import matplotlib.pyplot as plt


class QNet(nn.Module):
    """Q-network (critic) for TD3 algorithm.

    Args:
        nn (torch.nn.Module): Base class for all neural network modules
    
    Implements the Q-function approximator that estimates state-action values.
    Typically consists of fully connected layers with ReLU activations.
    Initialization parameters:
        - dim_state: Dimension of state space
        - dim_action: Dimension of action space
    """

    def __init__(self, dim_state, dim_action):
        super().__init__()
        # self.fc1 = nn.Linear(dim_state + dim_action, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, 1)
        self.fc1 = nn.Linear(dim_state + dim_action, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], -1)
        # x = F.relu(self.fc1(sa))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class PolicyNet(nn.Module):
    """Policy Network for TD3 algorithm.

    Implements a neural network that maps states to actions. The network outputs
    continuous actions bounded by max_action. This is the actor network in the
    TD3 (Twin Delayed DDPG) architecture.

    Args:
        dim_state (int): Dimension of the input state space
        dim_action (int): Dimension of the output action space
        max_action (float, optional): Maximum value for action outputs. Defaults to 2.0.
    """
    def __init__(self, dim_state, dim_action, max_action=2.0):
        super().__init__()
        self.max_action = max_action
        # self.fc1 = nn.Linear(dim_state, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, dim_action)
        self.fc1 = nn.Linear(dim_state, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, dim_action)

    def forward(self, state):
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # x = self.max_action * torch.tanh(self.fc3(x))
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.max_action * torch.tanh(self.fc4(x))
        return x


class TD3:
    def __init__(self, dim_state, dim_action, max_action):
        super().__init__()

        self.max_action = max_action
        self.Q1 = QNet(dim_state, dim_action)
        self.Q2 = QNet(dim_state, dim_action)
        self.Mu = PolicyNet(dim_state, dim_action, max_action)
        self.target_Q1 = QNet(dim_state, dim_action)
        self.target_Q2 = QNet(dim_state, dim_action)
        self.target_Mu = PolicyNet(dim_state, dim_action, max_action)
        self.target_Q1.load_state_dict(self.Q1.state_dict())
        self.target_Q2.load_state_dict(self.Q2.state_dict())
        self.target_Mu.load_state_dict(self.Mu.state_dict())

    def get_action(self, state):
        action = self.Mu(state)
        return action

    def compute_value_loss(self, args, s_batch, a_batch, r_batch, d_batch, next_s_batch):
        """Compute the value loss for the two Q networks.

        Args:
            args (argparse.Namespace): Training arguments.
            s_batch (torch.Tensor): Batch of current states, shape (batch_size, dim_state).
            a_batch (torch.Tensor): Batch of current actions, shape (batch_size, dim_action).
            r_batch (torch.Tensor): Batch of rewards, shape (batch_size,).
            d_batch (torch.Tensor): Batch of done flags, shape (batch_size,), values 0 or 1.
            next_s_batch (torch.Tensor): Batch of next states, shape (batch_size, dim_state).

        Returns:
            value_loss1 (torch.Tensor): Loss value for Q1 network.
            value_loss2 (torch.Tensor): Loss value for Q2 network.
        """

        with torch.no_grad():
            # Have the target policy network make predictions.
            a = self.target_Mu(next_s_batch)
            noise = torch.clamp(
                torch.randn_like(a) * args.policy_noise,
                -args.noise_clip,
                args.noise_clip,
            )
            a = torch.clamp(a + noise, min=-self.max_action, max=self.max_action)

            # Let the two target value networks make predictions.
            q1 = self.target_Q1(next_s_batch, a).squeeze()
            q2 = self.target_Q2(next_s_batch, a).squeeze()

            # Calculate the TD target.
            y = r_batch + args.gamma * torch.min(q1, q2) * (1 - d_batch)

        # Have the two value networks make predictions.
        qvals1 = self.Q1(s_batch, a_batch).squeeze()
        qvals2 = self.Q2(s_batch, a_batch).squeeze()
        value_loss1 = F.mse_loss(y, qvals1)
        value_loss2 = F.mse_loss(y, qvals2)
        return value_loss1, value_loss2

    def compute_policy_loss(self, s_batch):
        """Compute the policy loss for the given state batch.

        Args:
            s_batch (torch.Tensor): A batch of states for which the policy loss will be computed.

        Returns:
            torch.Tensor: The computed policy loss.
        """

        a = self.Mu(s_batch)
        policy_loss = -self.Q1(s_batch, a).mean()
        return policy_loss

    def soft_update(self, tau=0.01):
        """Perform soft update of target network parameters.

        Args:
            tau (float, optional): Interpolation parameter for updating target networks. 
                                 Controls how much of the online network parameters are mixed into 
                                 the target network parameters. Defaults to 0.01.
        """

        def soft_update_(target, source, tau_=0.01):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau_) + param.data * tau_)

        soft_update_(self.target_Q1, self.Q1, tau)
        soft_update_(self.target_Q2, self.Q2, tau)
        soft_update_(self.target_Mu, self.Mu, tau)


@dataclass
class ReplayBuffer:
    maxsize: int
    size: int = 0
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    done: list = field(default_factory=list)
    next_state: list = field(default_factory=list)

    def push(self, state, action, reward, done, next_state):
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
        total_number = self.size if self.size < self.maxsize else self.maxsize
        indices = np.random.randint(total_number, size=n)
        state = [self.state[i] for i in indices]
        action = [self.action[i] for i in indices]
        reward = [self.reward[i] for i in indices]
        done = [self.done[i] for i in indices]
        next_state = [self.next_state[i] for i in indices]
        return state, action, reward, done, next_state


class INFO:
    def __init__(self):
        self.log = defaultdict(list)
        self.episode_length = 0
        self.episode_reward = 0
        self.max_episode_reward = -float("inf")

    def put(self, done, reward):
        """Update episode statistics when an episode is done.

        Args:
            done (bool): Flag indicating whether the episode has terminated.
            reward (float): The reward obtained in the current step.
        
        Notes:
            When done is True, accumulates the episode length and reward,
            and records them in the training log.
        """
        if done is True:
            self.episode_length += 1
            self.episode_reward += reward
            self.log["episode_length"].append(self.episode_length)
            self.log["episode_reward"].append(self.episode_reward)

            if self.episode_reward > self.max_episode_reward:
                self.max_episode_reward = self.episode_reward

            self.episode_length = 0
            self.episode_reward = 0
        else:
            self.episode_length += 1
            self.episode_reward += reward


def train(args, env, agent: TD3):
    """Train the TD3 agent using the specified environment and parameters.

    Args:
        args (argparse.Namespace): Training configuration containing:
            - max_steps: Total training steps
            - warmup_steps: Initial random exploration steps
            - lr: Learning rate
            - batch_size: Training batch size
            - K: Policy update frequency
            - seed: Random seed
            - output_dir: Output directory for saving models
        env (gym.Env): The training environment
        agent (TD3): The TD3 agent instance to be trained

    Training process:
        1. Initializes optimizers for Q networks and policy network
        2. Sets up experience replay buffer
        3. Runs training loop for max_steps:
            - Collects experience using exploration policy
            - Stores transitions in replay buffer
            - Samples batches for training
            - Updates Q networks and policy network
            - Performs soft updates of target networks
        4. Logs training statistics and saves models periodically
        5. Visualizes training progress with plots

    Saves:
        - Best policy model (when achieving new high score)
        - Training loss plots (value_loss.png)
        - Episode reward plots (episode_reward.png)
    """
    Q1_optimizer = torch.optim.Adam(agent.Q1.parameters(), lr=args.lr)
    Q2_optimizer = torch.optim.Adam(agent.Q2.parameters(), lr=args.lr)
    Mu_optimizer = torch.optim.Adam(agent.Mu.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(maxsize=100_000)
    info = INFO()

    state, _ = env.reset(seed=args.seed)
    for step in range(args.max_steps):
        if step < args.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(torch.from_numpy(state))
            action = action.cpu().data.numpy()
            action_noise = np.clip(np.random.randn(args.dim_action), -args.max_action, args.max_action)
            action = np.clip(action + action_noise, -args.max_action, args.max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, done, next_state)
        state = next_state
        info.put(done, reward)

        if done is True:
            # Print information.
            episode_reward = info.log["episode_reward"][-1]
            episode_length = info.log["episode_length"][-1]
            value_loss = info.log["value_loss1"][-1] if len(info.log["value_loss1"]) > 0 else 0
            print(f"step={step}, reward={episode_reward:.0f}, length={episode_length}, max_reward={info.max_episode_reward}, value_loss={value_loss:.2f}")

            # If the score is higher, save the model.
            if episode_reward == info.max_episode_reward:
                save_path = os.path.join(args.output_dir, "model.bin")
                torch.save(agent.Mu.state_dict(), save_path)

            state, _ = env.reset()

        if step > args.warmup_steps:
            s_batch, a_batch, r_batch, d_batch, ns_batch = replay_buffer.sample(n=args.batch_size)

            s_batch = np.array(s_batch)
            a_batch = np.array(a_batch)
            r_batch = np.array(r_batch)
            d_batch = np.array(d_batch)
            ns_batch = np.array(ns_batch)

            s_batch = torch.tensor(s_batch, dtype=torch.float32)
            a_batch = torch.tensor(a_batch, dtype=torch.float32)
            r_batch = torch.tensor(r_batch, dtype=torch.float32)
            d_batch = torch.tensor(d_batch, dtype=torch.float32)
            ns_batch = torch.tensor(ns_batch, dtype=torch.float32)

            value_loss1, value_loss2 = agent.compute_value_loss(args, s_batch, a_batch, r_batch, d_batch, ns_batch)

            Q1_optimizer.zero_grad()
            value_loss1.backward(retain_graph=True)
            Q1_optimizer.step()

            Q2_optimizer.zero_grad()
            value_loss2.backward()
            Q2_optimizer.step()

            info.log["value_loss1"].append(value_loss1.item())
            info.log["value_loss2"].append(value_loss2.item())

            if step % args.K == 0:
                policy_loss = agent.compute_policy_loss(s_batch)
                Mu_optimizer.zero_grad()
                policy_loss.backward()
                Mu_optimizer.step()
                agent.soft_update()

                info.log["policy_loss"].append(policy_loss.item())

            if step % 10000 == 0:
                # draw figure.
                plt.plot(info.log["value_loss1"], label="loss1")
                plt.plot(info.log["value_loss2"], label="loss2")
                plt.legend()
                plt.savefig(f"{args.output_dir}/value_loss.png", bbox_inches="tight")
                plt.close()

                plt.plot(info.log["episode_reward"])
                plt.savefig(f"{args.output_dir}/episode_reward.png", bbox_inches="tight")
                plt.close()


def eval(args, env, agent):
    """Evaluate the trained TD3 agent in the given environment.

    Args:
        args (argparse.Namespace): Evaluation configuration containing:
            - dim_state (int): Dimension of state space
            - dim_action (int): Dimension of action space  
            - max_action (float): Maximum action value
            - output_dir (str): Directory containing saved model
        env (gym.Env): The environment to evaluate the agent in
        agent (TD3): The TD3 agent instance to be evaluated

    Loads the trained policy network and runs evaluation episodes:
        - Loads model weights from model.bin
        - Runs for 5000 steps total across episodes
        - Renders the environment during evaluation
        - Prints episode reward and length when episode terminates
        - Resets statistics after each episode
    """

    agent = TD3(args.dim_state, args.dim_action, args.max_action)
    model_path = os.path.join(args.output_dir, "model.bin")
    agent.Mu.load_state_dict(torch.load(model_path))

    episode_length = 0
    episode_reward = 0
    state, _ = env.reset()
    for i in range(5000):
        episode_length += 1
        action = agent.get_action(torch.from_numpy(state)).cpu().data.numpy()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        episode_reward += reward

        state = next_state
        if done is True:
            print(f"episode reward={episode_reward}, length={episode_length}")
            state, _ = env.reset()
            episode_length = 0
            episode_reward = 0


def main():
    """Main function to run TD3 training and evaluation.

    Configures and runs the TD3 algorithm with command line arguments:
        - Environment configuration:
            --env: Gym environment name (default: Pendulum-v1)
            --dim_state: State space dimension (default: 3)
            --dim_action: Action space dimension (default: 1)
            --max_action: Action value range [-max,max] (default: 2.0)
            --gamma: Discount factor (default: 0.99)
        
        - Training parameters:
            --max_steps: Total training steps (default: 100,000)
            --warmup_steps: Initial random exploration steps (default: 10,000)
            --lr: Learning rate (default: 1e-3)
            --batch_size: Training batch size (default: 32)
            --K: Policy and target network update frequency (default: 2)
            --policy_noise: Noise added to target policy (default: 0.2)
            --noise_clip: Noise clipping range (default: 0.5)
        
        - Runtime options:
            --no_cuda: Disable CUDA if available
            --seed: Random seed (default: 42)
            --output_dir: Output directory (default: 'output')
            --do_train: Enable training mode
            --do_eval: Enable evaluation mode
    
    Initializes environment, agent, and runs training/evaluation as specified.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1", type=str, help="Environment name.")
    parser.add_argument("--dim_state", default=3, type=int, help="Dimension of observation.")
    parser.add_argument("--dim_action", default=1, type=int, help="Number of actions.")
    parser.add_argument("--max_action", default=2.0, type=float, help="Action scale, [-max, max].")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount coefficient.")

    parser.add_argument("--max_steps", default=100_000, type=int, help="Maximum steps for interaction.")
    parser.add_argument("--warmup_steps", default=10_000, type=int, help="Warmup steps without training.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--K", default=2, type=int, help="Delay K steps to update policy and target network.")
    parser.add_argument("--policy_noise", default=0.2, type=float, help="Policy noise.")
    parser.add_argument("--noise_clip", default=0.5, type=float, help="Policy noise.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--output_dir", default="output", type=str, help="Output directory.")
    parser.add_argument("--do_train", action="store_true", help="Train policy.")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate policy.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # 初始化环境。
    env = gym.make(args.env)

    agent = TD3(dim_state=args.dim_state, dim_action=args.dim_action, max_action=args.max_action)

    if args.do_train:
        train(args, env, agent)

    if args.do_eval:
        eval(args, env, agent)


if __name__ == "__main__":
    main()
