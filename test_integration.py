import sys
sys.path.insert(0, '.')
import argparse
import torch
import numpy as np
import os
from my_USV_Env import MyUsvEnv
from HTD3_main import HTD3, evaluate_htd3, dict_state_to_tensor

def test_evaluation():
    # Use existing trained model
    output_dir = "htd3_output"
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist. Please train a model first.")
        return
    
    # Create a simple argument namespace using argparse.Namespace
    from argparse import Namespace
    args = Namespace(
        output_dir=output_dir,
        eval_episodes=1,
        max_episode_steps=10,  # short episode for quick test
        device=torch.device("cpu")
    )
    
    # Create environment
    env = MyUsvEnv(record_flag=False)
    
    # Initialize agent (architecture must match saved model)
    sample_state = env.reset()
    flat_dim = len(dict_state_to_tensor(sample_state, args.device))
    agent = HTD3(
        dim_state=flat_dim,
        dim_action=env.action_space.shape[0],
        max_action=25.0,
        num_targets=3
    ).to(args.device)
    
    # Run evaluation (should compute multi-objective metrics)
    evaluate_htd3(args, env, agent)
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    test_evaluation()