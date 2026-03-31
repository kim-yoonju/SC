"""
train_grpo.py

GRPO (Group Relative Policy Optimization) 학습 스크립트
train_ppo.py를 기반으로 PPO 대신 GRPO 알고리즘 적용

PPO와의 차이:
  - PPO: critic/value network으로 baseline 추정, TD returns 사용
  - GRPO: 동일 문제에서 group_size개 trajectory 샘플링,
           그룹 내 상대적 보상으로 advantage 계산
           A_i = (r_i - mean(group_rewards)) / (std(group_rewards) + eps)

파이프라인 (iteration 단위):
  1. 각 문제마다 group_size개 trajectory 병렬 생성
  2. problem_id 기준으로 그룹 묶기
  3. 그룹 내 보상 정규화 → group-relative advantage
  4. PPO-clip + KL 패널티로 policy 업데이트
  5. 업데이트된 weights → 워커에 동기화
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


def _parse_args():
    p = argparse.ArgumentParser(description="GRPO Online Training")
    # GPU 설정
    p.add_argument("--rollout_gpus", type=str, help="Rollout GPU IDs, comma-separated (e.g. '2,3,4,5')")
    p.add_argument("--train_gpus",   type=str, help="Train GPU IDs, comma-separated (e.g. '6')")
    # 체크포인트
    p.add_argument("--resume_checkpoint", type=str, default=None)
    # 학습 설정
    p.add_argument("--max_iterations",    type=int)
    p.add_argument("--problems_per_iter", type=int)
    p.add_argument("--train_batch_size",  type=int)
    p.add_argument("--dataset",           type=str)
    # 하이퍼파라미터
    p.add_argument("--lr",          type=float)
    p.add_argument("--clip_eps",    type=float)
    p.add_argument("--kl_coef",     type=float)
    p.add_argument("--max_seq_len", type=int)
    # GRPO 전용
    p.add_argument("--group_size",  type=int,   help="문제당 샘플링할 trajectory 수 (GRPO group size G)")
    p.add_argument("--grpo_eps",    type=float, help="group advantage 정규화 epsilon (default: 1e-8)")
    return p.parse_args()

_args = _parse_args()

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    CONF,
    DATASET_PATH,
    KL_COEF,
    LENGTH_PENALTY_COEF,
    MAX_SEQ_LEN,
    MAX_STEPS,
    PPO_CLIP_EPS,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    SAVE_DIR,
    GENERATOR_MODEL_ID,
    GENERATOR_MAX_NEW_TOKENS,
    TRUNCATE_TOKEN_LIMIT,
    Trajectory,
    create_rollout_file,
    load_generator,
    load_math500,
    load_problems,
    save_trajectory,
    validate_math500,
)
from generate_trajectory import solve_problems_batch
from record_wandb import WandbLogger

# ─────────────────────────────────────────────────────────────────────────────
# config + CLI 인자로 실행 설정 결정 (CLI 인자가 config보다 우선)
# ─────────────────────────────────────────────────────────────────────────────
_ppo = CONF["ppo"]

if _args.rollout_gpus    is not None: _ppo["rollout_gpus"]    = [int(g) for g in _args.rollout_gpus.split(",")]
if _args.train_gpus      is not None: _ppo["train_gpus"]      = [int(g) for g in _args.train_gpus.split(",")]
if _args.max_iterations  is not None: _ppo["max_iterations"]  = _args.max_iterations
if _args.problems_per_iter is not None: _ppo["problems_per_iter"] = _args.problems_per_iter
if _args.train_batch_size  is not None: _ppo["train_batch_size"]  = _args.train_batch_size
if _args.lr        is not None: PPO_LR       = _args.lr
if _args.clip_eps  is not None: PPO_CLIP_EPS = _args.clip_eps
if _args.kl_coef   is not None: KL_COEF      = _args.kl_coef
if _args.max_seq_len is not None: MAX_SEQ_LEN = _args.max_seq_len
if _args.dataset   is not None: DATASET_PATH = str(Path(__file__).resolve().parent.parent / _args.dataset)
if _args.resume_checkpoint is not None: _ppo["resume_checkpoint"] = _args.resume_checkpoint

# GRPO 전용 하이퍼파라미터
GRPO_GROUP_SIZE = _args.group_size if _args.group_size is not None else 4
GRPO_EPS        = _args.grpo_eps   if _args.grpo_eps   is not None else 1e-8

ROLLOUT_GPUS      = _ppo["rollout_gpus"]
TRAIN_GPUS        = _ppo["train_gpus"]
RESUME_CHECKPOINT = _ppo.get("resume_checkpoint")
NUM_WORKERS       = len(ROLLOUT_GPUS)
TRAINER_DEVICE    = "cuda:0"   # CUDA_VISIBLE_DEVICES = TRAIN_GPUS → 항상 cuda:0
MAX_ITERATIONS    = _ppo["max_iterations"]
PROBLEMS_PER_ITER = _ppo["problems_per_iter"]
TRAIN_BATCH_SIZE  = _ppo["train_batch_size"]
CHECKPOINT_BASE   = str(Path(__file__).resolve().parent.parent / CONF["checkpoint"]["ppo_checkpoint_base"])
SAVE_DIR          = str(Path(__file__).resolve().parent.parent / CONF["output_path"]["ppo"])

# GPU 설정은 torch/cuda import 전에 해야 함
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in TRAIN_GPUS)

import ray
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def _finished_without_timeout(traj: Trajectory) -> bool:
    """max_steps 도달 없이 스스로 종료한 trajectory이면 True."""
    return bool(traj.steps) and len(traj.steps) < MAX_STEPS


def _trajectory_total_reward(traj: Trajectory) -> float:
    """trajectory 전체 스텝 보상의 합 (length 패널티 포함)."""
    total = 0.0
    for step in traj.steps:
        n = step.response_ids.shape[1]
        length_penalty = -max(0, n - 512) * 0.0002 if not traj.is_answer else 0.0
        total += step.reward + length_penalty
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Worker (Ray actor) — train_ppo.py와 동일
# ─────────────────────────────────────────────────────────────────────────────

@ray.remote
class RolloutWorker:
    """한 GPU에서 trajectory를 생성하는 Ray Actor."""

    def __init__(self, worker_id: int, rollout_path: str, log_path: str):
        import logging
        self.worker_id    = worker_id
        self.rollout_path = rollout_path

        _wlogger = logging.getLogger()
        _wlogger.setLevel(logging.INFO)
        for h in _wlogger.handlers[:]:
            _wlogger.removeHandler(h)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(f"%(asctime)s [Worker{worker_id}] %(message)s"))
        _wlogger.addHandler(fh)

        self.model, self.tokenizer = load_generator(device_map="auto")
        create_rollout_file(rollout_path)
        logging.info(f"준비 완료  rollout → {rollout_path}  log → {log_path}")

    def generate_trajectories(self, problems_batch: List[dict]) -> List[Trajectory]:
        """problems_batch를 배치 GPU 추론 + 병렬 API 호출로 동시 처리."""
        return solve_problems_batch(
            self.model, self.tokenizer, problems_batch, rollout_path=self.rollout_path,
        )

    def load_state_dict(self, state_dict: dict):
        """GRPOTrainer에서 업데이트된 weights를 동기화."""
        self.model.load_state_dict(state_dict)
        self.model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# GRPO Trainer (main process)
# ─────────────────────────────────────────────────────────────────────────────

class GRPOTrainer:
    """Group Relative Policy Optimization (GRPO) 트레이너.

    각 문제에 대해 group_size개의 trajectory를 샘플링하고,
    그룹 내 상대적 보상으로 advantage를 계산합니다.

    Advantage:
      A_i = (R_i - mean({R_j})) / (std({R_j}) + eps)
      여기서 R_i = trajectory i의 전체 스텝 보상 합

    GRPO objective (per token):
      L = E[ min(ratio * A, clip(ratio, 1±ε) * A) ] - KL_COEF * KL(π_θ ‖ π_ref)

    PPO 대비 차이:
      - critic/value network 없음 (그룹 정규화로 baseline 대체)
      - γ-discount returns 대신 trajectory 전체 스칼라 보상 사용
      - advantage가 그룹 내에서만 정규화됨
    """

    def __init__(self, device: str, training_path: str, resume_checkpoint: str | None = None):
        self.device        = device
        self.training_path = training_path

        model_path = resume_checkpoint if resume_checkpoint else None
        self.model, self.tokenizer = load_generator(device_map={"": device}, model_path=model_path)
        self.model.train()

        # reference model: frozen, KL 계산용
        self.ref_model, _ = load_generator(device_map={"": device}, model_path=resume_checkpoint)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        import bitsandbytes as bnb
        self.optimizer = bnb.optim.AdamW8bit(
            [p for p in self.model.parameters() if p.requires_grad], lr=PPO_LR,
        )
        if resume_checkpoint:
            logger.info(f"GRPOTrainer 체크포인트 로드: {resume_checkpoint}")
        logger.info(f"GRPOTrainer 초기화 완료 ({device})  training → {training_path}")

    # ── GRPO advantage 계산 ───────────────────────────────────────────────────

    @staticmethod
    def _group_relative_advantages(
        trajectories: List[Trajectory],
    ) -> Dict[str, float]:
        """problem_id별 그룹을 구성하고, 그룹 내 상대적 advantage를 반환.

        Returns:
            {traj 객체의 id(메모리 주소) → advantage 스칼라} 매핑
        """
        # problem_id별 그룹화
        groups: Dict[str, List[Trajectory]] = defaultdict(list)
        for traj in trajectories:
            groups[traj.problem_id].append(traj)

        advantages: Dict[int, float] = {}
        for problem_id, group in groups.items():
            rewards = [_trajectory_total_reward(t) for t in group]
            mean_r  = sum(rewards) / len(rewards)
            var_r   = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r   = var_r ** 0.5

            for traj, r in zip(group, rewards):
                advantages[id(traj)] = (r - mean_r) / (std_r + GRPO_EPS)

        return advantages

    # ── GRPO 업데이트 ─────────────────────────────────────────────────────────

    def update(self, trajectories: List[Trajectory]) -> dict:
        """전체 trajectories에 GRPO 업데이트를 적용.

        1. 문제별 그룹 상대적 advantage 계산
        2. 각 trajectory의 모든 토큰에 동일 advantage 적용
        3. gradient accumulation 후 optimizer.step() 1회 수행
        """
        # 그룹 상대적 advantage 계산
        traj_advantages = self._group_relative_advantages(trajectories)

        # 유효한 trajectory만 추려서 학습 데이터 구성
        traj_data = []
        for traj in trajectories:
            if not traj.steps:
                continue
            advantage_val = traj_advantages[id(traj)]
            inp_list, resp_list, lp_old_list = [], [], []
            for step in traj.steps:
                inp_list.append(step.input_ids.to(self.device))
                resp_list.append(step.response_ids.to(self.device))
                lp_old_list.append(step.log_probs_old.to(self.device))
            if inp_list:
                traj_data.append((inp_list, resp_list, lp_old_list, advantage_val))

        if not traj_data:
            return {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "entropy": 0.0}

        n_traj        = len(traj_data)
        total_pg = total_kl = total_entropy = 0.0

        self.model.train()
        self.optimizer.zero_grad()

        for inp_list, resp_list, lp_old_list, advantage_val in traj_data:
            n_steps_in_traj = len(inp_list)
            advantage = torch.tensor(advantage_val, device=self.device)
            traj_pg_sum = traj_kl_sum = traj_entropy_sum = 0.0

            for j in range(n_steps_in_traj):
                inp_ids   = inp_list[j]
                resp_ids  = resp_list[j]
                lp_old    = lp_old_list[j].detach()

                total_len = inp_ids.shape[1] + resp_ids.shape[1]
                if total_len > MAX_SEQ_LEN:
                    logger.warning(f"[update] seq_len={total_len} > MAX_SEQ_LEN={MAX_SEQ_LEN}, skip step")
                    n_steps_in_traj = max(n_steps_in_traj - 1, 1)
                    continue

                split = inp_ids.shape[1] - 1
                with torch.no_grad():
                    policy_prefix_kv = (
                        self.model(inp_ids[:, :split], use_cache=True).past_key_values if split > 0 else None
                    )

                grad_input = torch.cat([inp_ids[:, -1:], resp_ids], dim=1)
                logits = self.model(grad_input, past_key_values=policy_prefix_kv).logits
                lp_new = (
                    F.log_softmax(logits[:, :-1, :], dim=-1)
                    .gather(-1, resp_ids.unsqueeze(-1))
                    .squeeze(-1).squeeze(0)
                )

                with torch.no_grad():
                    ref_prefix_kv = (
                        self.ref_model(inp_ids[:, :split], use_cache=True).past_key_values if split > 0 else None
                    )
                    ref_logits = self.ref_model(grad_input, past_key_values=ref_prefix_kv).logits
                    lp_ref = (
                        F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                        .gather(-1, resp_ids.unsqueeze(-1))
                        .squeeze(-1).squeeze(0)
                    )

                # GRPO: 그룹 정규화된 스칼라 advantage를 모든 토큰에 동일하게 적용
                ratio   = torch.exp(lp_new - lp_old)
                pg_loss = -torch.min(
                    ratio * advantage,
                    torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * advantage,
                ).mean()
                kl      = (torch.exp(lp_new) * (lp_new - lp_ref)).mean()
                entropy = -lp_new.mean()

                step_loss = (pg_loss + KL_COEF * kl) / n_steps_in_traj / n_traj
                step_loss.backward()

                traj_pg_sum      += pg_loss.item()
                traj_kl_sum      += kl.item()
                traj_entropy_sum += entropy.item()

            total_pg      += traj_pg_sum      / n_steps_in_traj
            total_kl      += traj_kl_sum      / n_steps_in_traj
            total_entropy += traj_entropy_sum / n_steps_in_traj

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), PPO_MAX_GRAD_NORM)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.eval()

        stats = {
            "loss":    (total_pg + KL_COEF * total_kl) / n_traj,
            "pg_loss": total_pg      / n_traj,
            "kl":      total_kl      / n_traj,
            "entropy": total_entropy / n_traj,
        }
        for traj in trajectories:
            save_trajectory(traj, self.training_path)
        return stats

    def get_state_dict(self) -> dict:
        """워커에 동기화할 CPU state dict 반환."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}


# ─────────────────────────────────────────────────────────────────────────────
# 메인 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SAVE_DIR, ts)
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_dir = os.path.join(CHECKPOINT_BASE, ts)
    os.makedirs(checkpoint_dir, exist_ok=True)

    rollout_paths    = [os.path.join(run_dir, f"rollouts_worker_{i}.jsonl") for i in range(NUM_WORKERS)]
    worker_log_paths = [os.path.join(run_dir, f"log_worker_{i}.log")        for i in range(NUM_WORKERS)]
    training_path    = os.path.join(run_dir, "training.jsonl")
    log_path         = os.path.join(run_dir, "run.log")

    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(fh)
    logger.info(f"로그 파일: {log_path}")

    ray.init(include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)

    problems     = load_problems(DATASET_PATH)
    n            = len(problems)
    val_problems = load_math500()

    workers = [
        RolloutWorker.options(
            runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": str(ROLLOUT_GPUS[i])}}
        ).remote(worker_id=i, rollout_path=rollout_paths[i], log_path=worker_log_paths[i])
        for i in range(NUM_WORKERS)
    ]

    trainer = GRPOTrainer(device=TRAINER_DEVICE, training_path=training_path, resume_checkpoint=RESUME_CHECKPOINT)

    logger.info(f"문제 {n}개  |  ts={ts}  |  group_size={GRPO_GROUP_SIZE}")
    for i, rp in enumerate(rollout_paths):
        logger.info(f"  rollout  → {rp}")
    logger.info(f"  training → {training_path}")

    wlogger = WandbLogger(
        config={
            "model":                GENERATOR_MODEL_ID,
            "grpo_lr":              PPO_LR,
            "grpo_clip_eps":        PPO_CLIP_EPS,
            "grpo_group_size":      GRPO_GROUP_SIZE,
            "grpo_eps":             GRPO_EPS,
            "train_batch_size":     TRAIN_BATCH_SIZE,
            "grpo_max_grad_norm":   PPO_MAX_GRAD_NORM,
            "kl_coef":              KL_COEF,
            "length_penalty_coef":  LENGTH_PENALTY_COEF,
            "max_steps":            MAX_STEPS,
            "max_seq_len":          MAX_SEQ_LEN,
            "truncate_token_limit": TRUNCATE_TOKEN_LIMIT,
            "max_new_tokens":       GENERATOR_MAX_NEW_TOKENS,
            "num_workers":          NUM_WORKERS,
            "problems_per_iter":    PROBLEMS_PER_ITER,
            "max_iterations":       MAX_ITERATIONS,
        },
        project="sc-grpo",
        run_name=ts,
    )
    wlogger.set_val_problems(problems)

    answer_total   = 0
    problem_cursor = 0

    # GRPO는 그룹 단위로 수집: TRAIN_BATCH_SIZE // group_size 개의 완성된 그룹 필요
    groups_needed = max(TRAIN_BATCH_SIZE // GRPO_GROUP_SIZE, 1)

    iter_bar = tqdm(range(MAX_ITERATIONS), desc="GRPO", unit="iter", dynamic_ncols=True)
    for iteration in iter_bar:

        all_trajs   = []
        # problem_id → 유효한 trajectory 목록
        group_buffer: Dict[str, List[Trajectory]] = defaultdict(list)

        collect_bar = tqdm(total=groups_needed, desc="Collecting Groups", unit="group", leave=False, dynamic_ncols=True)

        while len(group_buffer) < groups_needed:
            start = problem_cursor % n
            # 각 문제를 group_size번 반복해서 배치 구성
            unique_batch = (problems + problems)[start: start + PROBLEMS_PER_ITER]
            repeated_batch = [p for p in unique_batch for _ in range(GRPO_GROUP_SIZE)]

            chunk = len(repeated_batch) // NUM_WORKERS
            futures   = [workers[i].generate_trajectories.remote(repeated_batch[i*chunk:(i+1)*chunk]) for i in range(NUM_WORKERS)]
            new_trajs = [t for tl in ray.get(futures) for t in tl]
            all_trajs.extend(new_trajs)

            prev_groups = len(group_buffer)
            for traj in new_trajs:
                if _finished_without_timeout(traj):
                    group_buffer[traj.problem_id].append(traj)

            # group_size 이상 모인 그룹 수 계산 (중복 카운트 방지)
            complete_groups = sum(1 for trajs in group_buffer.values() if len(trajs) >= GRPO_GROUP_SIZE)
            if complete_groups > prev_groups:
                collect_bar.update(complete_groups - prev_groups)

            problem_cursor += PROBLEMS_PER_ITER

        collect_bar.close()

        # 각 그룹에서 최대 group_size개만 사용
        train_groups = {pid: trajs[:GRPO_GROUP_SIZE] for pid, trajs in group_buffer.items() if len(trajs) >= GRPO_GROUP_SIZE}
        train_groups = dict(list(train_groups.items())[:groups_needed])
        train_trajs  = [t for trajs in train_groups.values() for t in trajs]

        answer_total += len(train_trajs)
        n_correct     = sum(1 for t in train_trajs if t.is_answer)
        n_boxed       = sum(1 for t in train_trajs if t.have_boxed)
        n_timeout     = len(all_trajs) - len(train_trajs)
        logger.info(
            f"[iter {iteration:4d}]  generated={len(all_trajs)}  "
            f"groups={len(train_groups)}  grpo_usable={len(train_trajs)}  timeout={n_timeout}  "
            f"correct={n_correct}  boxed={n_boxed}  total_grpo={answer_total}"
        )

        wlogger.log_rollout(train_trajs, iteration, 1, all_trajs=all_trajs)

        stats = trainer.update(train_trajs)
        logger.info(
            f"           GRPO → loss={stats['loss']:.4f}  "
            f"pg={stats['pg_loss']:.4f}  kl={stats['kl']:.6f}  "
            f"entropy={stats['entropy']:.4f}"
        )

        wlogger.log_train(stats, iteration)

        logger.info("           validation 시작 (MATH-500)...")
        val_metrics = validate_math500(trainer.model, trainer.tokenizer, val_problems)
        wlogger.log_validation(val_metrics, iteration)
        logger.info(
            f"           val accuracy={val_metrics.get('val/accuracy', 0):.4f}"
            f"  ({val_metrics.get('val/n_correct', 0)}/{val_metrics.get('val/n_total', 0)})"
        )

        weights = trainer.get_state_dict()
        ray.get([w.load_state_dict.remote(weights) for w in workers])

        ckpt_dir = os.path.join(checkpoint_dir, f"iter_{iteration:04d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        trainer.model.save_pretrained(ckpt_dir)
        trainer.tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"           ckpt → {ckpt_dir}")

        iter_bar.set_postfix(
            loss=f"{stats['loss']:.4f}",
            kl=f"{stats['kl']:.5f}",
            correct=f"{n_correct}/{len(train_trajs)}",
            val_acc=f"{val_metrics.get('val/accuracy', 0):.3f}",
        )

    logger.info("학습 완료.")
    wlogger.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
