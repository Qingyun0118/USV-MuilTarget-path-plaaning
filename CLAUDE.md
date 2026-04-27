# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

USV multi-target path planning system using hierarchical reinforcement learning (HRL) with expert guidance. Two-layer decision architecture: **Selector** (target selection via DQN+DeepSets) + **Actuator** (navigation control via SAC/TD3/TD7). Also supports multi-USV cooperative navigation via CTDE-MAPPO with COLREGs compliance.

The codebase is written entirely in Chinese-commented Python. All docstrings, comments, and commit messages are in Chinese.

## Commands

### Training (single USV, expert-guided HRL)
```bash
python Expert_HRL_SAC_main.py --do_train --seed 2026 --max_episodes 1000
python Expert_HRL_TD3_main.py --do_train --seed 2026 --max_episodes 1000
python Expert_HRL_TD7_main.py --do_train --seed 2026 --max_episodes 1000
```

### Training (multi-USV CTDE-MAPPO, requires pretrained actuator)
```bash
python CTDE_MAPPO_train.py --do_train --assign_method hungarian
python CTDE_MAPPO_train.py --do_train --assign_method hungarian_makespan
python CTDE_MAPPO_train.py --do_train --assign_method mappo_makespan
```

### Evaluation
```bash
python CTDE_MAPPO_train.py --do_eval --assign_method hungarian --eval_episodes 5
python replay_trajectory.py --output_dir <output_dir> --mode expert --episode 3
```

### Monitoring
```bash
tensorboard --logdir runs/
```

### Tests
```bash
python test/test_expert_batch.py
python test/test_expert_nav.py
```

### Dependencies
```bash
pip install torch torchvision numpy scipy gymnasium netCDF4 shapely matplotlib tensorboard
```

## Architecture

### Two-layer HRL pipeline (single USV)

The three training entry points (`Expert_HRL_SAC_main.py`, `Expert_HRL_TD3_main.py`, `Expert_HRL_TD7_main.py`) share identical structure — they differ only in which Actuator agent they use. The training loop lives in `train_expert_hrl()` inside each file.

```
Selector (selector_agent.py, DQN+DeepSets) → picks next target index
  ↓
Actuator (sac_agent.py / td3_agent.py / td7_agent.py, goal-conditioned) → [throttle, rudder]
  ↓
Environment (my_USV_Env.py:MyUsvEnv) → qLPV dynamics, sonar, currents, obstacles
```

Shared modules used by all three training scripts:
- `hrl_config.py` — device, coordinate scale, global seed setter
- `hrl_helpers.py` — state extraction (`get_selector_state`, `get_actuator_state`) and reward computation
- `buffers.py` — `ReplayBuffer` (Selector), `GoalConditionedReplayBuffer` with PER/SumTree (Actuator)
- `expert_scheduler.py` — Dawood exponential decay for expert demonstration probability
- `reactive_controller.py` — rule-based expert (proportional nav + 5-zone reactive avoidance)

### Multi-USV cooperative (CTDE-MAPPO)

`CTDE_MAPPO_train.py` is a thin entry point that delegates to the `training/` package:

```
MultiUsvEnv (my_USV_Env.py) — manages N independent MyUsvEnv sub-environments
  ↑
MAPPOTrainer (mappo_agent.py) — shared-parameter Actor (pretrained SAC) + CentralizedCritic
  ↑
Task Assignment (task_assignment/) — pluggable target allocation strategies
```

The `training/` package provides:
- `training/config.py` — argparse parameter parsing (used by CTDE_MAPPO_train.py)
- `training/train_loop.py` — CTDE-MAPPO training main loop
- `training/eval_loop.py` — CTDE-MAPPO evaluation loop

The `task_assignment/` package provides 3 assignment methods selectable via `--assign_method`:
- `hungarian` — global optimal bipartite matching via scipy (Baseline 2)
- `hungarian_makespan` — Makespan-minimizing cost matrix (Proposed 1)
- `mappo_makespan` — PPO-learned DeepSets assigner, jointly trained with MAPPO (Proposed 2)

### Environment internals

`my_USV_Env.py` contains both `MyUsvEnv` (single USV, Gymnasium interface) and `MultiUsvEnv` (multi-agent wrapper). Key dimensions:
- Observation: `o_usv(7) + o_detector(15) + o_current(4) = 26D` for actuator; `usv_pos(2) + targets(3N)` for selector
- Action: `Box([-1,-1], [1,1])` — throttle and rudder
- `USV.py` — qLPV dynamics model (dt=0.05s)
- `colregs_module.py` — COLREGs encounter classification, CRI computation, compliance penalties (used by MultiUsvEnv)

### Neural network patterns

All agents use a shared `SensorCNN` (Conv1d on 15 sonar beams → 64D embedding). The sensor slice in the actuator state is at indices [7:22]. State layout: `[o_usv(7) | o_detector(15) | o_current(4)]`.

Selector uses DeepSets architecture (shared phi-encoder per target, mean-aggregation, per-target Q-head) — decoupled from target count.

### Reference implementations

The `MAPPO/`, `TD3/`, and `TD7/` directories contain upstream reference code (MAPPO on-policy library, TD3/TD7 papers). The active project code — agents, buffers, training loops — lives at the repo root and in `training/`/`task_assignment/`. Do not modify files under `MAPPO/`, `TD3/`, `TD7/` unless explicitly asked.

### Analysis & visualization scripts

- `batch_eval_success_rate.py` — batch evaluate model checkpoints and compute success rate
- `plot_seed_rewards.py` — multi-seed training reward comparison curves
- `draw_scene_demo.py` — render obstacle/target/USV scene layout
- `evaluation_metrics.py` — six-dim metric calculation (SPM, SmPM, MECM, SaPM, SE, SR)
- `radar_chart.py` — radar charts for multi-algorithm comparison
- `replay_trajectory.py` — trajectory replay from CSV logs, renders animation
- `colregs_batch_policy_test.py` — COLREGs encounter scenario policy testing

## Key conventions

- Coordinates are normalized by `X_MAX`/`Y_MAX` (from `my_USV_Env.py`) in observations; use `SCALE` from `hrl_config.py` to denormalize
- First `fixed_env_episodes` (default 0) use a deterministic environment layout for fast convergence; after that, environments are randomized
- Expert probability decays via `ExpertScheduler` using Dawood exponential schedule (p0=0.9 → p_min=0.01)
- Behavior cloning loss is Q-filtered: only imitates expert when expert action has higher Q-value than current policy
- Collision triggers retroactive reward penalty on the N preceding steps in the replay buffer
- Output directories follow the pattern `expert_hrl_{algo}_output*` and `ctde_mappo_output_{method}/` — these are gitignored
- Model checkpoints are `.pth` files (gitignored); training history is `training_history.json` (gitignored)
- TensorBoard logs go to `runs/` (gitignored)
