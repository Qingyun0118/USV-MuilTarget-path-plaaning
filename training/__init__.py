"""
training — CTDE MAPPO 训练与评估模块
"""

from training.config import parse_args
from training.train_loop import train_ctde_mappo
from training.eval_loop import evaluate_ctde_mappo

__all__ = ['parse_args', 'train_ctde_mappo', 'evaluate_ctde_mappo']
