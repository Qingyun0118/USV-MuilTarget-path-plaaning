import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from my_USV_Env import MyUsvEnv_Gym
from main import flatten_state
import torch

def plot_eval_path(model_dir='./models', output_dir='./plots', bestModel=True, 
                   policy_name="TD3", env_name="MyUsvEnv-Gym", seed=0,episode_num=5):
    """
    加载训练好的模型并在环境中运行，绘制USV的航行路径
    
    参数:
        model_dir: 模型文件目录
        output_dir: 输出图片目录
        bestModel: 是否使用最佳模型，否则使用最新模型
        policy_name: 策略名称 (TD3, DDPG 或 OurDDPG)
        env_name: 环境名称
        seed: 随机种子
    """
    import TD3
    import DDPG
    import OurDDPG
    
    # 创建环境
    env = MyUsvEnv_Gym(record_flag=True)
    
    # 计算状态和动作维度
    from main import get_flattened_state_dim
    state_dim = get_flattened_state_dim(env_name)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # 根据策略名称初始化策略
    if policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action)
    elif policy_name == "OurDDPG":
        policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action)
    else:
        raise ValueError(f"未知的策略名称: {policy_name}")
    
    # 构建模型文件名
    file_name = f"{policy_name}_{env_name}_{seed}"
    
    # 加载模型
    if bestModel:
        model_path = f"{model_dir}/{file_name}_best"
        print(f"加载最佳模型: {model_path}")
    else:
        model_path = f"{model_dir}/{file_name}_latest"
        print(f"加载最新模型: {model_path}")
    
    # 加载模型参数
    policy.load(model_path)
    print("模型加载完成!")
    
    # 评估模型
    print("开始评估训练好的模型...")
    
    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episode_num):  # 运行5个回合
        state_dict, _ = env.reset()
        state = flatten_state(state_dict)
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # 将状态转换为tensor
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            
            # 选择动作（无探索噪声）
            action = policy.select_action(state_tensor.cpu().data.numpy().flatten())
            
            # 执行动作
            next_state_dict, reward, terminated, truncated, _ = env.step(action)
            
            # 更新状态
            state = flatten_state(next_state_dict)
            episode_reward += reward
            episode_length += 1
            
            # 防止无限循环
            if episode_length > 2000:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"评估回合 {episode + 1}: 奖励={episode_reward:.1f}, 步数={episode_length}")
    
    # 打印评估结果
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n评估结果:")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_length:.2f}")
    print(f"奖励范围: {np.min(episode_rewards):.2f} ~ {np.max(episode_rewards):.2f}")
    
    # 绘制路径
    env.draw_path("usv_path_points_1.csv")
    
    # 保存评估结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存评估统计数据
    eval_results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_reward': avg_reward,
        'avg_length': avg_length
    }
    
    model_type = "best" if bestModel else "latest"
    np.save(f"{output_dir}/{file_name}_{model_type}_eval_results.npy", eval_results)
    print(f"评估结果已保存: {output_dir}/{file_name}_{model_type}_eval_results.npy")


def plot_reward_curves(results_dir='./results', output_dir='./plots'):
    """
    加载results文件夹下的.npy数据文件，绘制奖励曲线并保存为PNG图片
    
    参数:
        results_dir: 包含.npy文件的目录路径
        output_dir: 保存PNG图片的输出目录
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 查找所有的.npy文件
    npy_files = glob.glob(os.path.join(results_dir, '*.npy'))
    
    if not npy_files:
        print(f"在目录 {results_dir} 中没有找到.npy文件")
        return
    
    print(f"找到 {len(npy_files)} 个.npy文件")
    
    # 处理每个.npy文件
    for file_path in npy_files:
        try:
            # 加载数据
            data = np.load(file_path)
            
            # 获取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 绘制奖励曲线
            plt.plot(data, linewidth=2)
            
            # 添加标题和标签
            plt.title(f'Training Progress: {filename}', fontsize=14, fontweight='bold')
            plt.xlabel('Evaluation Episode', fontsize=12)
            plt.ylabel('Average Reward', fontsize=12)
            
            # 添加网格
            plt.grid(True, alpha=0.3)
            
            # 添加统计信息
            max_reward = np.max(data)
            final_reward = data[-1]
            avg_last_10 = np.mean(data[-10:]) if len(data) >= 10 else np.mean(data)
            
            # 在图表上添加文本信息
            text_str = f'Final: {final_reward:.2f}\nMax: {max_reward:.2f}\nAvg(last 10): {avg_last_10:.2f}'
            plt.annotate(text_str, xy=(0.02, 0.98), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        verticalalignment='top', fontsize=10)
            
            # 保存图表
            output_path = os.path.join(output_dir, f'{filename}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图表以释放内存
            
            print(f"已生成: {output_path}")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    print(f"\n所有图表已保存到: {output_dir}")

def plot_comparison(results_dir='./results', output_dir='./plots'):
    """
    可选功能：在同一张图上比较多个算法的性能
    """
    npy_files = glob.glob(os.path.join(results_dir, '*.npy'))
    
    if len(npy_files) < 2:
        print("需要至少2个文件进行比较")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 为不同算法分配颜色
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, file_path in enumerate(npy_files):
        try:
            data = np.load(file_path)
            filename = os.path.splitext(os.path.basename(file_path))[0]
            color = colors[i % len(colors)]
            
            # 提取算法名称（假设文件名格式为: 算法_环境_种子）
            parts = filename.split('_')
            algo_name = parts[0] if len(parts) > 0 else filename
            
            plt.plot(data, label=algo_name, color=color, linewidth=2)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    plt.title('Algorithm Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Evaluation Episode', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存比较图
    output_path = os.path.join(output_dir, 'algorithm_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"比较图已保存: {output_path}")

if __name__ == "__main__":
    # 设置matplotlib中文字体（如果需要显示中文）
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 绘制单个算法的奖励曲线
    plot_reward_curves(results_dir='./results', output_dir='./plots')
    
    # 可选：绘制算法比较图
    plot_comparison(results_dir='./results', output_dir='./plots')
    
    print("所有图表生成完成！")