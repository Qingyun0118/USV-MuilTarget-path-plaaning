import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

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