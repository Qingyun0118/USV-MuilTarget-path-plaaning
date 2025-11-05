import numpy as np
import torch
import gymnasium as gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
from my_USV_Env import MyUsvEnv_Gym  # 导入自定义环境

# 状态展平函数
def flatten_state(state_dict):
    """将字典状态展平为向量"""
    x_norm = state_dict['x_norm']
    y_norm = state_dict['y_norm']
    rou_o_norm = state_dict['Rou_o_norm']
    alpha_o_norm = state_dict['Alpha_o_norm']
    priority_vector = state_dict['Priority_vector']
    rou_t_norm = state_dict['rou_t_norm']
    alpha_t_norm = state_dict['alpha_t_norm']
    
    # 展平所有状态组件
    flattened = np.concatenate([
        x_norm.flatten(),
        y_norm.flatten(), 
        rou_o_norm.flatten(),
        alpha_o_norm.flatten(),
        priority_vector.flatten(),
        rou_t_norm.flatten(),
        alpha_t_norm.flatten()
    ])
    return flattened

# 计算展平状态维度
def get_flattened_state_dim(env_name):
    """获取展平后的状态维度"""
    if env_name == "MyUsvEnv-Gym":
        env = MyUsvEnv_Gym(record_flag=False)
        sample_state, _ = env.reset()
        flat_state = flatten_state(sample_state)
        env.close()
        return flat_state.shape[0]
    else:
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        env.close()
        return state_dim

# 修改eval_policy函数
def eval_policy(policy, env_name, seed, eval_episodes=10):
    if env_name == "MyUsvEnv-Gym":
        eval_env = MyUsvEnv_Gym(record_flag=False)
    else:
        eval_env = gym.make(env_name)
        eval_env.reset(seed=seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        if env_name == "MyUsvEnv-Gym":
            state, _ = eval_env.reset()
        else:
            state = eval_env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # 状态预处理
            if env_name == "MyUsvEnv-Gym":
                state_flat = flatten_state(state)
            else:
                state_flat = state
                
            action = policy.select_action(state_flat)
            
            if env_name == "MyUsvEnv-Gym":
                state, reward, terminated, truncated, _ = eval_env.step(action)
            else:
                state, reward, terminated, truncated, _ = eval_env.step(action)
                
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="MyUsvEnv-Gym")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # 环境创建
    if args.env == "MyUsvEnv-Gym":
        env = MyUsvEnv_Gym(record_flag=False)
    else:
        env = gym.make(args.env)

    # 计算状态和动作维度
    if args.env == "MyUsvEnv-Gym":
        state_dim = get_flattened_state_dim(args.env)
    else:
        state_dim = env.observation_space.shape[0]
        
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    # 重置环境
    if args.env == "MyUsvEnv-Gym":
        state, _ = env.reset()
    else:
        state, _ = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            # 状态预处理
            if args.env == "MyUsvEnv-Gym":
                state_flat = flatten_state(state)
            else:
                state_flat = state
                
            action = (
                policy.select_action(state_flat)
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        if args.env == "MyUsvEnv-Gym":
            next_state, reward, terminated, truncated, _ = env.step(action)
            done_bool = float(terminated) if episode_timesteps < 2000 else 0  # 使用环境的最大步数
        else:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done_bool = float(terminated) if episode_timesteps < env._max_episode_steps else 0

        # 状态预处理用于存储
        if args.env == "MyUsvEnv-Gym":
            store_state = flatten_state(state)
            store_next_state = flatten_state(next_state)
        else:
            store_state = state
            store_next_state = next_state

        # Store data in replay buffer
        replay_buffer.add(store_state, action, store_next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if terminated or truncated: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            if args.env == "MyUsvEnv-Gym":
                state, _ = env.reset()
            else:
                state, _ = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")