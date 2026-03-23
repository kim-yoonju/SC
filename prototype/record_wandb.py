"""
prototype/record_wandb.py

WandB 학습 기록 유틸리티.

사용:
    from record_wandb import WandbLogger
    wlogger = WandbLogger(config={...}, project="sc-ppo")
    wlogger.set_val_problems(all_problems)

    # 매 iteration:
    wlogger.log_rollout(trajectories, iteration, sub_iters)
    wlogger.log_train(stats, iteration)

    # 4 iteration마다 자동으로 validation 실행:
    wlogger.maybe_log_validation(model, tokenizer, iteration)

    wlogger.finish()
"""

import random
import statistics
from collections import defaultdict
from typing import Dict, List, Optional

import wandb

from utils import (
    Trajectory,
    solve_problem,
    TOKEN_CORRECT,
    TOKEN_SOLVE,
)

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

VAL_INTERVAL            = 4    # 몇 iteration마다 validation 실행
VAL_SEED                = 42   # 고정 시드 (validation set 불변)
VAL_SAMPLES_PER_DIFF    = 2    # 난이도별 validation 문제 수


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(vals: list) -> float:
    return statistics.mean(vals) if vals else 0.0


def _sample_val_problems(
    problems: list,
    seed: int = VAL_SEED,
    n_per_diff: int = VAL_SAMPLES_PER_DIFF,
) -> list:
    """난이도별 n개씩 고정 시드로 샘플링 (validation set 불변)."""
    rng = random.Random(seed)
    by_diff: Dict[str, list] = defaultdict(list)
    for p in problems:
        d = p.get("difficulty")
        key = str(d) if d is not None else "unknown"
        by_diff[key].append(p)

    sampled = []
    for d in sorted(by_diff.keys()):
        pool = by_diff[d]
        sampled.extend(rng.sample(pool, min(n_per_diff, len(pool))))
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# 롤아웃 지표 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_rollout_metrics(trajectories: List[Trajectory], prefix: str = "rollout") -> dict:
    """Trajectory 배치에서 모든 롤아웃 지표를 계산합니다.

    Self-correction 분석:
      - generator correct: is_generator_step=True, action="correct"
      - patcher correct  : is_generator_step=False, action="correct"
      - 보정 전후 reward 비교로 correction 개선도를 계산

    Args:
        trajectories: 분석할 Trajectory 리스트
        prefix: wandb 키 prefix (train 시 "rollout", validation 시 "val")
    """
    if not trajectories:
        return {}

    n_trajs = len(trajectories)

    # ── 스텝 수 ──────────────────────────────────────────────────────────────
    all_n_steps     = []
    solve_counts    = []
    correct_counts  = []
    gen_step_counts = []

    # ── 리워드 ───────────────────────────────────────────────────────────────
    all_rewards      = []
    solve_rewards    = []
    correct_rewards  = []
    gen_rewards      = []

    # ── 액션 예측 정확도 ──────────────────────────────────────────────────────
    action_correct_n = 0
    action_total_n   = 0

    # ── patcher 관련 ─────────────────────────────────────────────────────────
    n_patcher_wrong  = 0
    n_patcher_called = 0   # patcher가 실제로 호출된 trajectory 수

    # ── self-correction 분석 ──────────────────────────────────────────────────
    # generator correct 성공 / 실패 리워드
    gen_corr_success_rewards = []
    gen_corr_fail_rewards    = []

    # patcher correct 성공 시 reward
    patcher_success_rewards  = []

    # 보정 개선도: (correct 스텝 reward) - (직전 스텝 reward)
    gen_corr_improvements    = []   # generator가 직접 고쳤을 때
    patcher_improvements     = []   # patcher가 고쳤을 때

    # correct 스텝 연속 발생 횟수 (loop 감지)
    consecutive_correct_counts = []

    for traj in trajectories:
        steps = traj.steps

        # 스텝 수 집계
        n_total   = len(steps)
        n_solve   = sum(1 for s in steps if s.action == "solve")
        n_correct = sum(1 for s in steps if s.action == "correct")
        n_gen     = sum(1 for s in steps if s.is_generator_step)

        all_n_steps.append(n_total)
        solve_counts.append(n_solve)
        correct_counts.append(n_correct)
        gen_step_counts.append(n_gen)

        if traj.patcher_wrong:
            n_patcher_wrong += 1

        # patcher_wrong 이거나 patcher correct step이 있으면 patcher 호출된 것
        has_patcher_step = any(not s.is_generator_step for s in steps)
        if has_patcher_step or traj.patcher_wrong:
            n_patcher_called += 1

        # consecutive correct 길이 추적
        _consecutive = 0
        _max_consec  = 0
        for s in steps:
            if s.action == "correct":
                _consecutive += 1
                _max_consec = max(_max_consec, _consecutive)
            else:
                _consecutive = 0
        if n_correct > 0:
            consecutive_correct_counts.append(_max_consec)

        for s in steps:
            all_rewards.append(s.reward)
            if s.action == "solve":
                solve_rewards.append(s.reward)
            if s.action == "correct":
                correct_rewards.append(s.reward)
            if s.is_generator_step:
                gen_rewards.append(s.reward)

            # 액션 예측 정확도 (generator step에서만 의미 있음)
            if s.is_generator_step:
                action_total_n += 1
                if s.predicted_next_action == s.ground_truth_next_action:
                    action_correct_n += 1

        # ── self-correction 분석 ─────────────────────────────────────────────
        for i, s in enumerate(steps):
            if s.action != "correct":
                continue

            prev_reward = steps[i - 1].reward if i > 0 else 0.0

            if s.is_generator_step:
                # generator correct 스텝
                if s.reward > 0.1:
                    gen_corr_success_rewards.append(s.reward)
                    gen_corr_improvements.append(s.reward - prev_reward)
                else:
                    gen_corr_fail_rewards.append(s.reward)
            else:
                # patcher correct 스텝
                if s.reward > 0.1:
                    patcher_success_rewards.append(s.reward)
                    patcher_improvements.append(s.reward - prev_reward)

    n_gen_corr_total  = len(gen_corr_success_rewards) + len(gen_corr_fail_rewards)
    n_patcher_success = len(patcher_success_rewards)
    # patcher_wrong 수 = patcher 실패 횟수
    n_patcher_fail    = n_patcher_wrong

    p = prefix
    metrics = {
        # ── 스텝 수 ──────────────────────────────────────────────────────────
        f"{p}/avg_steps":                   _safe_mean(all_n_steps),
        f"{p}/avg_solve_steps":             _safe_mean(solve_counts),
        f"{p}/avg_correct_steps":           _safe_mean(correct_counts),
        f"{p}/avg_gen_steps":               _safe_mean(gen_step_counts),
        f"{p}/min_steps":                   min(all_n_steps) if all_n_steps else 0,
        f"{p}/max_steps":                   max(all_n_steps) if all_n_steps else 0,
        f"{p}/correct_to_solve_ratio":      _safe_mean(correct_counts) / max(_safe_mean(solve_counts), 1e-9),
        f"{p}/avg_max_consecutive_correct": _safe_mean(consecutive_correct_counts),

        # ── 리워드 ───────────────────────────────────────────────────────────
        f"{p}/avg_reward":          _safe_mean(all_rewards),
        f"{p}/avg_solve_reward":    _safe_mean(solve_rewards),
        f"{p}/avg_correct_reward":  _safe_mean(correct_rewards),
        f"{p}/avg_gen_reward":      _safe_mean(gen_rewards),

        # ── 액션 예측 정확도 ──────────────────────────────────────────────────
        f"{p}/action_accuracy": (
            action_correct_n / action_total_n if action_total_n else 0.0
        ),

        # ── patcher 관련 ─────────────────────────────────────────────────────
        f"{p}/n_patcher_wrong":      n_patcher_wrong,
        f"{p}/patcher_wrong_rate":   n_patcher_wrong  / n_trajs,
        f"{p}/patcher_called_rate":  n_patcher_called / n_trajs,

        # ── self-correction: generator ────────────────────────────────────────
        # generator correct 성공률 (호출 대비 reward > 0.1 비율)
        f"{p}/gen_correct_success_rate": (
            len(gen_corr_success_rewards) / n_gen_corr_total if n_gen_corr_total else 0.0
        ),
        # generator correct 성공 시 평균 reward
        f"{p}/gen_correct_reward_when_success": _safe_mean(gen_corr_success_rewards),
        # generator correct 실패 시 평균 reward
        f"{p}/gen_correct_reward_when_fail":    _safe_mean(gen_corr_fail_rewards),
        # generator correct 성공 시 reward 개선량 (correct_reward - prev_step_reward)
        f"{p}/gen_correct_reward_improvement":  _safe_mean(gen_corr_improvements),
        # generator correct 총 호출 수 (trajectory 평균)
        f"{p}/avg_gen_correct_calls":           n_gen_corr_total / n_trajs,

        # ── self-correction: patcher ──────────────────────────────────────────
        # patcher 성공률 (호출 대비 성공 비율)
        f"{p}/patcher_success_rate": (
            n_patcher_success / (n_patcher_success + n_patcher_fail)
            if (n_patcher_success + n_patcher_fail) > 0 else 0.0
        ),
        # patcher 성공 시 평균 reward
        f"{p}/patcher_reward_when_success": _safe_mean(patcher_success_rewards),
        # patcher 성공 시 reward 개선량 (patcher_reward - gen_fail_reward)
        f"{p}/patcher_reward_improvement":  _safe_mean(patcher_improvements),

        # ── n_trajs (확인용) ──────────────────────────────────────────────────
        f"{p}/n_trajectories": n_trajs,
    }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Validation 지표 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_validation_metrics(
    model,
    tokenizer,
    val_problems: list,
) -> dict:
    """val_problems를 현재 모델로 풀어 validation 지표를 반환.

    compute_rollout_metrics를 val prefix로 재활용 +
    정확도 / 난이도별 정확도 추가.
    """
    trajs: List[Trajectory] = []
    for item in val_problems:
        traj = solve_problem(
            model, tokenizer,
            problem=item["problem"],
            answer=item["answer"],
            problem_id=str(item.get("problem_id", "val")),
            rollout_path=None,
            difficulty=item.get("difficulty"),
        )
        if traj is not None:
            trajs.append(traj)

    if not trajs:
        return {}

    n        = len(trajs)
    n_solved = sum(1 for t in trajs if t.is_answer)

    # 난이도별 정확도
    by_diff: Dict[str, Dict] = defaultdict(lambda: {"solved": 0, "total": 0})
    for t in trajs:
        d = str(t.difficulty) if t.difficulty is not None else "unknown"
        by_diff[d]["total"]  += 1
        by_diff[d]["solved"] += int(t.is_answer)

    metrics: dict = {
        "val/accuracy":  n_solved / n,
        "val/n_solved":  n_solved,
        "val/n_total":   n,
    }

    # 난이도별 정확도
    for d in sorted(by_diff.keys()):
        r = by_diff[d]
        metrics[f"val/accuracy_diff_{d}"] = r["solved"] / r["total"] if r["total"] else 0.0

    # rollout 지표를 val prefix로 재활용
    rollout_m = compute_rollout_metrics(trajs, prefix="val")
    metrics.update(rollout_m)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# WandbLogger
# ─────────────────────────────────────────────────────────────────────────────

class WandbLogger:
    """PPO self-correction 학습용 WandB 로거.

    Example::

        wlogger = WandbLogger(config={...}, project="sc-ppo", run_name="run_001")
        wlogger.set_val_problems(all_problems)

        for iteration in range(MAX_ITERATIONS):
            trajs, sub_iters = collect(...)
            stats = trainer.update(trajs)

            wlogger.log_rollout(trajs, iteration, sub_iters)
            wlogger.log_train(stats, iteration)
            wlogger.maybe_log_validation(trainer.model, trainer.tokenizer, iteration)

        wlogger.finish()
    """

    def __init__(
        self,
        config: dict,
        project: str = "sc-ppo",
        run_name: Optional[str] = None,
        val_interval: int = VAL_INTERVAL,
    ):
        self.val_interval  = val_interval
        self._val_problems: Optional[list] = None

        wandb.init(project=project, name=run_name, config=config)

    # ── 셋업 ─────────────────────────────────────────────────────────────────

    def set_val_problems(
        self,
        problems: list,
        n_per_diff: int = VAL_SAMPLES_PER_DIFF,
    ):
        """전체 문제 pool에서 고정 시드로 validation set을 구성합니다."""
        self._val_problems = _sample_val_problems(
            problems, seed=VAL_SEED, n_per_diff=n_per_diff
        )

    # ── 로깅 ─────────────────────────────────────────────────────────────────

    def log_train(self, stats: dict, iteration: int):
        """PPO update() 반환값을 기록합니다.

        stats 키: loss, pg_loss, kl, entropy
        """
        wandb.log(
            {
                "train/loss":    stats.get("loss",    0.0),
                "train/pg_loss": stats.get("pg_loss", 0.0),
                "train/kl":      stats.get("kl",      0.0),
                "train/entropy": stats.get("entropy", 0.0),
            },
            step=iteration,
        )

    def log_rollout(
        self,
        trajectories: List[Trajectory],
        iteration: int,
        sub_iters: int,
        all_trajs: Optional[List[Trajectory]] = None,
    ):
        """Trajectory 배치의 롤아웃 / self-correction 지표를 기록합니다.

        all_trajs: 서브이터에서 시도한 전체 trajectory (boxed 필터링 이전).
                   지정 시 have_answer_rate / correct_rate 를 전체 기준으로 계산.
        """
        metrics = compute_rollout_metrics(trajectories, prefix="rollout")
        metrics["rollout/sub_iters"] = sub_iters
        # 수집 효율: sub_iter 1회당 확보한 answer trajectory 수
        metrics["rollout/answer_per_sub_iter"] = (
            len(trajectories) / max(sub_iters, 1)
        )

        # 전체 시도 기준 정답 출현율 / 정답 일치율
        pool = all_trajs if all_trajs is not None else trajectories
        n_all = len(pool)
        if n_all > 0:
            metrics["rollout/have_answer_rate"] = (
                sum(1 for t in pool if t.have_boxed) / n_all
            )
            metrics["rollout/correct_rate"] = (
                sum(1 for t in pool if t.is_answer) / n_all
            )

        wandb.log(metrics, step=iteration)

    def log_validation(self, metrics: dict, iteration: int):
        """미리 계산된 validation 지표를 wandb에 기록합니다."""
        if metrics:
            wandb.log(metrics, step=iteration)

    def finish(self):
        wandb.finish()
