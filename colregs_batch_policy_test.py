"""
COLREGs 批量策略测试脚本 (E1-E6)

用途:
1. 使用硬编码 COLREGs 测试场景 (E1-E6)
2. 对两套控制策略批量评估:
   - CTDE MAPPO Actor (ctde_mappo_output_hungarian)
   - Expert HRL SAC Actuator (expert_hrl_sac_output_329)
3. 输出规则合规率报告:
   - 右转率 (right-turn rate)
   - 直航率 (straight-going rate)
   - 违规率 (violation rate)
    - 严格合规率 (strict compliance rate)
    - 结果合规率 (outcome compliance rate)

说明:
- 该脚本重点评估本船(usv_0)在会遇状态下的动作合规性。
- 统计窗口仅包含 "CRI >= 阈值 且会遇类型匹配场景预期" 的时间步。
"""

import argparse
import datetime
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch

from colregs_module import EncounterType
from hrl_config import device, set_global_seed
from hrl_helpers import get_actuator_state, get_usv_pos
from my_USV_Env import X_MAX, Y_MAX, create_colregs_test_env
from sac_agent import GaussianActor, SACActuatorAgent


def _safe_torch_load(path: str):
    """兼容不同 torch 版本的加载接口。"""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _pick_model_path(model_dir: str, candidates: List[str]) -> str:
    for name in candidates:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model file found in {model_dir}, candidates={candidates}")


class CTDEHungarianPolicy:
    """CTDE-MAPPO 控制策略封装（仅使用 Actor）。"""

    def __init__(self, model_dir: str, state_dim: int = 26, goal_dim: int = 2, action_dim: int = 2):
        actor_path = _pick_model_path(model_dir, ["best_actor.pth", "actor.pth"])
        self.actor = GaussianActor(state_dim, goal_dim, action_dim, max_action=1.0).to(device)
        self.actor.load_state_dict(_safe_torch_load(actor_path))
        self.actor.eval()
        self.model_path = actor_path

    @torch.no_grad()
    def act(self, usv_obs: Dict[str, np.ndarray], goal_norm: np.ndarray) -> np.ndarray:
        s = get_actuator_state(usv_obs)
        rel_goal = goal_norm - get_usv_pos(usv_obs)
        s_t = torch.FloatTensor(s).unsqueeze(0).to(device)
        g_t = torch.FloatTensor(rel_goal).unsqueeze(0).to(device)
        a_t = self.actor.get_deterministic_action(s_t, g_t)
        return a_t.cpu().numpy().flatten().astype(np.float32)


class ExpertHRLSACPolicy:
    """Expert-HRL-SAC 控制策略封装（仅使用 SAC Actuator Actor）。"""

    def __init__(self, model_dir: str, state_dim: int = 26, goal_dim: int = 2, action_dim: int = 2):
        actor_path = _pick_model_path(model_dir, ["best_actuator_actor.pth", "actuator_actor.pth"])
        self.agent = SACActuatorAgent(state_dim, goal_dim, action_dim, max_action=1.0)
        self.agent.actor.load_state_dict(_safe_torch_load(actor_path))
        self.agent.actor.eval()
        self.model_path = actor_path

    def act(self, usv_obs: Dict[str, np.ndarray], goal_norm: np.ndarray) -> np.ndarray:
        s = get_actuator_state(usv_obs)
        rel_goal = goal_norm - get_usv_pos(usv_obs)
        a = self.agent.select_action(s, rel_goal, evaluate=True)
        return np.asarray(a, dtype=np.float32)


def _prepare_goals(env) -> List[np.ndarray]:
    """按 USV 索引分配固定目标（i -> target_i）。"""
    goals_norm = []
    for i in range(env.num_usvs):
        t_idx = min(i, len(env.targets) - 1)
        tx = env.targets[t_idx].x
        ty = env.targets[t_idx].y
        goals_norm.append(np.array([tx / X_MAX, ty / Y_MAX], dtype=np.float32))
    return goals_norm


def _judge_step(case_id: str, rudder: float,
                starboard_threshold: float,
                straight_threshold: float,
                evasive_threshold: float):
    """根据规则定义单步动作判定。"""
    right_turn = rudder <= starboard_threshold
    straight = abs(rudder) < straight_threshold

    if case_id in ("E3", "E4"):
        strict_compliant = right_turn
    elif case_id in ("E2", "E5"):
        strict_compliant = straight
    elif case_id in ("E1", "E6"):
        strict_compliant = abs(rudder) >= evasive_threshold
    else:
        strict_compliant = True

    violation = not strict_compliant

    return right_turn, straight, violation, strict_compliant


def _is_outcome_compliant(pre_cri: float,
                         post_cri: float,
                         post_enc: int,
                         expected_enc: int,
                         cri_threshold: float,
                         outcome_cri_margin: float) -> bool:
    """
    结果合规判定:
    1) 风险下降: post_cri <= pre_cri - margin
    2) 风险解除: post_cri < cri_threshold 或会遇退出到 SAFE
    """
    risk_decreased = post_cri <= (pre_cri - outcome_cri_margin)
    risk_cleared = (post_cri < cri_threshold) or (post_enc == int(EncounterType.SAFE))
    # 若仍处于同类会遇但风险降低, 也视为结果合规
    same_enc = (post_enc == int(expected_enc))
    return risk_cleared or (same_enc and risk_decreased)


def _new_counter():
    return {
        "sample_steps": 0,
        "right_turn_steps": 0,
        "straight_steps": 0,
        "violation_steps": 0,
        "strict_compliant_steps": 0,
        "outcome_compliant_steps": 0,
    }


def _finalize_counter(counter: Dict[str, float]) -> Dict[str, float]:
    n = max(1, counter["sample_steps"])
    out = dict(counter)
    out["right_turn_rate"] = counter["right_turn_steps"] / n
    out["straight_rate"] = counter["straight_steps"] / n
    out["violation_rate"] = counter["violation_steps"] / n
    out["strict_compliance_rate"] = counter["strict_compliant_steps"] / n
    out["outcome_compliance_rate"] = counter["outcome_compliant_steps"] / n
    return out


def evaluate_policy(policy,
                    policy_name: str,
                    cases: List[str],
                    episodes_per_case: int,
                    max_steps: int,
                    base_seed: int,
                    cri_threshold: float,
                    starboard_threshold: float,
                    straight_threshold: float,
                    evasive_threshold: float,
                    outcome_cri_margin: float):
    """批量评估一个策略在 E1-E6 的合规率。"""
    report = {
        "policy": policy_name,
        "cases": {},
    }

    total = _new_counter()

    for ci, case_id in enumerate(cases):
        cstat = _new_counter()

        for ep in range(episodes_per_case):
            seed = base_seed + ci * 1000 + ep
            env = create_colregs_test_env(case_name=case_id, seed=seed,
                                          rou_c=1.5, rou_s=1.5)
            obs = env.reset(seed=seed)
            goals_norm = _prepare_goals(env)
            expected_enc = env.get_case_summary()["expected_encounter"]

            for _ in range(max_steps):
                # 读取当前会遇态势（动作执行前）
                ci_pair = env.colregs_matrix.get((0, 1), None)
                pre_sample_hit = False
                pre_cri = 0.0
                pre_enc = int(EncounterType.SAFE)
                pre_strict_ok = False

                actions = np.zeros((env.num_usvs, 2), dtype=np.float32)
                for i in range(env.num_usvs):
                    actions[i] = policy.act(obs[f"usv_{i}"], goals_norm[i])

                # 仅统计“风险有效 + 场景匹配”的样本
                if ci_pair is not None:
                    enc = int(ci_pair.get("encounter_type", 0))
                    cri = float(ci_pair.get("cri", 0.0))
                    if enc == int(expected_enc) and cri >= cri_threshold:
                        pre_sample_hit = True
                        pre_cri = cri
                        pre_enc = enc
                        rudder = float(actions[0][1])
                        r_turn, straight, viol, strict_ok = _judge_step(
                            case_id, rudder,
                            starboard_threshold=starboard_threshold,
                            straight_threshold=straight_threshold,
                            evasive_threshold=evasive_threshold,
                        )
                        pre_strict_ok = strict_ok

                        cstat["sample_steps"] += 1
                        total["sample_steps"] += 1

                        if r_turn:
                            cstat["right_turn_steps"] += 1
                            total["right_turn_steps"] += 1
                        if straight:
                            cstat["straight_steps"] += 1
                            total["straight_steps"] += 1
                        if viol:
                            cstat["violation_steps"] += 1
                            total["violation_steps"] += 1
                        if strict_ok:
                            cstat["strict_compliant_steps"] += 1
                            total["strict_compliant_steps"] += 1

                goal_abs = [g * np.array([X_MAX, Y_MAX], dtype=np.float32) for g in goals_norm]
                obs, _, dn, trunc, info = env.step_marl(actions, goal_abs)

                # 结果合规: 动作严格合规 + 下一步风险改善
                if pre_sample_hit and pre_strict_ok:
                    post_ci = info.get("colregs_matrix", {}).get((0, 1), None)
                    if post_ci is not None:
                        post_cri = float(post_ci.get("cri", pre_cri))
                        post_enc = int(post_ci.get("encounter_type", pre_enc))
                    else:
                        post_cri = pre_cri
                        post_enc = pre_enc

                    if _is_outcome_compliant(
                        pre_cri=pre_cri,
                        post_cri=post_cri,
                        post_enc=post_enc,
                        expected_enc=int(expected_enc),
                        cri_threshold=cri_threshold,
                        outcome_cri_margin=outcome_cri_margin,
                    ):
                        cstat["outcome_compliant_steps"] += 1
                        total["outcome_compliant_steps"] += 1

                if dn["__all__"] or trunc["__all__"]:
                    break

        report["cases"][case_id] = _finalize_counter(cstat)

    report["overall"] = _finalize_counter(total)
    return report


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def print_report(all_reports: List[Dict]):
    print("\n" + "=" * 122)
    print("COLREGs 合规率报告（右转率 / 直航率 / 违规率 / 严格合规率 / 结果合规率）")
    print("=" * 122)

    for rep in all_reports:
        print(f"\n策略: {rep['policy']}")
        print("-" * 122)
        print(
            f"{'Case':<6}{'Samples':>10}{'RightTurn':>14}{'Straight':>14}"
            f"{'Violation':>14}{'StrictOK':>14}{'OutcomeOK':>14}"
        )
        for case_id, st in rep["cases"].items():
            print(
                f"{case_id:<6}{int(st['sample_steps']):>10}"
                f"{_fmt_pct(st['right_turn_rate']):>14}"
                f"{_fmt_pct(st['straight_rate']):>14}"
                f"{_fmt_pct(st['violation_rate']):>14}"
                f"{_fmt_pct(st['strict_compliance_rate']):>14}"
                f"{_fmt_pct(st['outcome_compliance_rate']):>14}"
            )
        ov = rep["overall"]
        print("-" * 122)
        print(
            f"{'ALL':<6}{int(ov['sample_steps']):>10}"
            f"{_fmt_pct(ov['right_turn_rate']):>14}"
            f"{_fmt_pct(ov['straight_rate']):>14}"
            f"{_fmt_pct(ov['violation_rate']):>14}"
            f"{_fmt_pct(ov['strict_compliance_rate']):>14}"
            f"{_fmt_pct(ov['outcome_compliance_rate']):>14}"
        )


def build_argparser():
    p = argparse.ArgumentParser(description="COLREGs E1-E6 批量策略合规测试")
    p.add_argument("--ctde_model_dir", type=str,
                   default="ctde_mappo_output_hungarian",
                   help="CTDE MAPPO 模型目录（含 best_actor.pth / actor.pth）")
    p.add_argument("--expert_model_dir", type=str,
                   default="expert_hrl_sac_output_329",
                   help="Expert HRL SAC 模型目录（含 best_actuator_actor.pth / actuator_actor.pth）")
    p.add_argument("--episodes_per_case", type=int, default=5,
                   help="每个场景评测 episode 数")
    p.add_argument("--max_steps", type=int, default=80,
                   help="每个 episode 最大步数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cri_threshold", type=float, default=0.05,
                   help="仅统计 CRI >= 该阈值的样本")
    p.add_argument("--starboard_threshold", type=float, default=-0.05,
                   help="右转判定阈值（rudder <= threshold）")
    p.add_argument("--straight_threshold", type=float, default=0.10,
                   help="直航判定阈值（|rudder| < threshold）")
    p.add_argument("--evasive_threshold", type=float, default=0.10,
                   help="规避动作阈值（|rudder| >= threshold）")
    p.add_argument("--outcome_cri_margin", type=float, default=1e-3,
                   help="结果合规判定中 CRI 最小下降幅度")
    p.add_argument("--report_out", type=str, default="colregs_compliance_report.json",
                   help="报告输出 JSON 文件")
    return p


def main():
    args = build_argparser().parse_args()

    set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cases = ["E1", "E2", "E3", "E4", "E5", "E6"]

    # 1) CTDE MAPPO (Hungarian)
    ctde_policy = CTDEHungarianPolicy(args.ctde_model_dir)
    ctde_report = evaluate_policy(
        ctde_policy,
        policy_name="CTDE_MAPPO_HUNGARIAN",
        cases=cases,
        episodes_per_case=args.episodes_per_case,
        max_steps=args.max_steps,
        base_seed=args.seed,
        cri_threshold=args.cri_threshold,
        starboard_threshold=args.starboard_threshold,
        straight_threshold=args.straight_threshold,
        evasive_threshold=args.evasive_threshold,
        outcome_cri_margin=args.outcome_cri_margin,
    )

    # 2) Expert HRL SAC
    expert_policy = ExpertHRLSACPolicy(args.expert_model_dir)
    expert_report = evaluate_policy(
        expert_policy,
        policy_name="EXPERT_HRL_SAC_329",
        cases=cases,
        episodes_per_case=args.episodes_per_case,
        max_steps=args.max_steps,
        base_seed=args.seed + 10000,
        cri_threshold=args.cri_threshold,
        starboard_threshold=args.starboard_threshold,
        straight_threshold=args.straight_threshold,
        evasive_threshold=args.evasive_threshold,
        outcome_cri_margin=args.outcome_cri_margin,
    )

    all_reports = [ctde_report, expert_report]
    print_report(all_reports)

    out = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "episodes_per_case": args.episodes_per_case,
        "max_steps": args.max_steps,
        "cri_threshold": args.cri_threshold,
        "thresholds": {
            "starboard_threshold": args.starboard_threshold,
            "straight_threshold": args.straight_threshold,
            "evasive_threshold": args.evasive_threshold,
            "outcome_cri_margin": args.outcome_cri_margin,
        },
        "model_paths": {
            "ctde": ctde_policy.model_path,
            "expert_hrl_sac": expert_policy.model_path,
        },
        "reports": all_reports,
    }

    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {args.report_out}")


if __name__ == "__main__":
    main()
