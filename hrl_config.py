"""
全局配置与公共工具

包含:
- device: PyTorch 计算设备
- SCALE: 坐标反归一化因子
- set_global_seed: 全局随机种子固定
"""

import os
import random
import numpy as np
import torch

from my_USV_Env import X_MAX, Y_MAX

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 坐标反归一化因子
SCALE = np.array([X_MAX, Y_MAX])


def set_global_seed(seed: int):
    """
    固定所有随机源, 确保训练可复现

    覆盖:
    - Python random
    - NumPy
    - PyTorch CPU / CUDA
    - cuDNN 确定性模式
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  #会降低GPU性能, 但保证完全可复现
    os.environ['PYTHONHASHSEED'] = str(seed)
