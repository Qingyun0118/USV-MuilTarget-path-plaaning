# USV 多目标路径规划系统

基于分层强化学习（HRL）与专家引导的无人艇（USV）多目标路径规划系统，同时支持多USV协同导航（CTDE-MAPPO + COLREGs）。

## 项目简介

本项目针对USV在复杂海洋环境中的多目标访问任务，提出了**两层决策架构**：

- **Selector（选择器）**：基于 DQN + DeepSets，从剩余目标中选择下一个访问目标
- **Actuator（执行器）**：基于目标条件的 SAC/TD3/TD7，输出舵角和油门控制指令

训练过程中引入**专家引导**（基于比例导航和五区避障），通过Q-filter行为克隆加速收敛。多USV场景下，采用CTDE-MAPPO框架实现协同任务分配与导航，并集成COLREGs海上避碰规则。

## 依赖安装

```bash
pip install torch numpy scipy gymnasium netCDF4 shapely matplotlib tensorboard
```

## 代码文件说明

### 训练入口

| 文件 | 说明 |
|------|------|
| `Expert_HRL_SAC_main.py` | HRL 训练主程序（执行器=SAC），含 Selector+Actuator 联合训练、评估、模型保存 |
| `Expert_HRL_TD3_main.py` | HRL 训练主程序（执行器=TD3），结构同 SAC 版本 |
| `Expert_HRL_TD7_main.py` | HRL 训练主程序（执行器=TD7），结构同 SAC 版本 |
| `CTDE_MAPPO_train.py` | 多USV协同训练入口，支持三种任务分配策略，委托 `training/` 包执行 |

### 智能体

| 文件 | 说明 |
|------|------|
| `selector_agent.py` | DQN + DeepSets 目标选择器，目标数量无关的 Q 网络架构 |
| `sac_agent.py` | SAC 执行器，目标条件策略，含 SensorCNN 声纳编码器 |
| `td3_agent.py` | TD3 执行器，双Q网络 + 延迟策略更新 |
| `td7_agent.py` | TD7 执行器（TD3 增强版），含 SDE 状态相关探索 |
| `mappo_agent.py` | MAPPO 多智能体 Actor-Critic，共享参数的预训练执行器 + 中心化 Critic |

### 环境与动力学

| 文件 | 说明 |
|------|------|
| `my_USV_Env.py` | 单USV环境（`MyUsvEnv`，Gym接口）与多USV环境（`MultiUsvEnv`），含障碍物、海流、声纳探测器 |
| `USV.py` | USV qLPV 动力学模型（dt=0.05s），推力-舵角到速度/航向的动力学解算 |
| `colregs_module.py` | COLREGs 海上避碰规则模块：会遇局面分类、碰撞风险指数（CRI）计算、合规惩罚 |
| `reactive_controller.py` | 基于规则的专家控制器：比例导航 + 五区反应式避障 |

### 训练与HR基础设施

| 文件 | 说明 |
|------|------|
| `hrl_config.py` | 全局配置：设备选择、坐标缩放因子、随机种子设置 |
| `hrl_helpers.py` | HRL 辅助函数：Selector/Actuator 状态提取、奖励计算 |
| `buffers.py` | 经验回放缓冲区：`ReplayBuffer`（Selector用）、`GoalConditionedReplayBuffer`（执行器用，含PER/SumTree） |
| `expert_scheduler.py` | 专家调度器：Dawood 指数衰减控制专家示范概率 |

### 多USV协同（CTDE-MAPPO）

| 文件 | 说明 |
|------|------|
| `training/config.py` | CTDE-MAPPO 参数解析 |
| `training/train_loop.py` | CTDE-MAPPO 训练主循环 |
| `training/eval_loop.py` | CTDE-MAPPO 评估循环 |
| `task_assignment/base.py` | 任务分配器抽象基类 |
| `task_assignment/factory.py` | 任务分配器工厂（根据 `--assign_method` 创建实例） |
| `task_assignment/hungarian.py` | 匈牙利算法全局最优二分匹配（Baseline 2） |
| `task_assignment/hungarian_makespan.py` | Makespan 最小化匈牙利分配（Proposed 1） |
| `task_assignment/selector_ppo_makespan.py` | PPO 学习的 DeepSets 分配器，与 MAPPO 联合训练（Proposed 2） |

### 评估与分析

| 文件 | 说明 |
|------|------|
| `batch_eval_success_rate.py` | 批量评估模型 checkpoint，计算成功率与碰撞率 |
| `eval_generalization.py` | 泛化性评估：在不同环境布局下测试模型 |
| `run_missing_evals.py` | 补充未评估的实验记录 |
| `colregs_batch_policy_test.py` | COLREGs 会遇场景策略批量测试（对遇/交叉/追越） |
| `replay_trajectory.py` | 轨迹回放：从CSV日志读取并渲染USV航行动画 |
| `generate_trajectory_animation.py` | 轨迹动画生成工具 |

### 可视化

| 文件 | 说明 |
|------|------|
| `plot_seed_rewards.py` | 多种子训练曲线对比（含置信区间） |
| `plot_boxplot.py` | 成功率/性能箱线图（按算法/Q-filter类型分组） |
| `draw_scene_demo.py` | 绘制场景布局图（障碍物、目标点、USV位置） |

### 测试

| 文件 | 说明 |
|------|------|
| `test/test_expert_batch.py` | 批量测试专家控制器 |
| `test/test_expert_nav.py` | 单次专家导航测试 |
| `test/test_expert_debug.py` | 专家调试脚本 |
| `test/testEnv.ipynb` | 环境交互测试 Notebook |
| `test/test_multi_usv_env_full.ipynb` | 多USV环境完整测试 Notebook |
| `test/test_usv_control.ipynb` | USV控制仿真测试 Notebook |

### 参考实现（不修改）

| 目录 | 说明 |
|------|------|
| `MAPPO/` | MAPPO 上游 on-policy 参考代码 |
| `TD3/` | TD3 论文参考实现 |
| `TD7/` | TD7 论文参考实现 |

## 使用方法

### 训练（单USV，专家引导 HRL）

```bash
# SAC 执行器
python Expert_HRL_SAC_main.py --do_train --seed 2026 --max_episodes 1000

# TD3 执行器
python Expert_HRL_TD3_main.py --do_train --seed 2026 --max_episodes 1000

# TD7 执行器
python Expert_HRL_TD7_main.py --do_train --seed 2026 --max_episodes 1000
```

### 训练（多USV 协同）

```bash
# 匈牙利全局最优匹配
python CTDE_MAPPO_train.py --do_train --assign_method hungarian

# 匈牙利 Makespan 最小化
python CTDE_MAPPO_train.py --do_train --assign_method hungarian_makespan

# PPO 学习分配策略
python CTDE_MAPPO_train.py --do_train --assign_method mappo_makespan
```

### 评估

```bash
# 单USV评估（在各自主程序中）
python Expert_HRL_SAC_main.py --do_eval --load_model path/to/model.pth

# 多USV评估
python CTDE_MAPPO_train.py --do_eval --assign_method hungarian --eval_episodes 5

# 轨迹回放
python replay_trajectory.py --output_dir <output_dir> --mode expert --episode 3
```

### 监控

```bash
tensorboard --logdir runs/
```

## 运行测试

```bash
python test/test_expert_batch.py
python test/test_expert_nav.py
```

## 输出目录说明

- `expert_hrl_{algo}_output*/` — 单USV HRL训练输出（模型、历史记录）
- `ctde_mappo_output_{method}/` — 多USV协同训练输出
- `runs/` — TensorBoard 日志
- 以上目录均已 gitignore
