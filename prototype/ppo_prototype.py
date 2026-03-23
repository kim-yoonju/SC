"""
prototype/ppo_prototype.py

실행: python ppo_prototype.py

GPU 구성
  GPU 3, 5, 6, 7  → RolloutWorker (데이터 생성, Ray actor, 4개)
  GPU 4           → PPOTrainer    (모델 학습, main process)

파이프라인 (iteration 단위):
  1. 네 워커가 문제를 16개씩 병렬로 trajectory 생성 (총 64개/서브이터)
     → 각 워커는 datasets/prototype/rollouts_{ts}_workerN.jsonl 에 실시간 저장
  2. train_trajs 64개 모이면 즉시 PPO 1 step (8-bit AdamW + gradient accumulation)
  3. 업데이트된 weights → 워커에 동기화
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from tqdm import tqdm

# ← GPU 설정은 torch/cuda import 전에 해야 함
# 워커(Ray): logical cuda:0~3 = physical GPU 3,5,6,7
# 트레이너:  logical cuda:4   = physical GPU 4  (Ray 풀 밖)
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6,7,4"

import ray
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DATASET_PATH,
    KL_COEF,
    MAX_SEQ_LEN,
    MAX_STEPS,
    PPO_CLIP_EPS,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    SAVE_DIR,
    Trajectory,
    create_rollout_file,
    has_boxed,
    load_generator,
    load_math500,
    load_problems,
    save_trajectory,
    solve_problems_batch,
    validate_math500,
)
from record_wandb import WandbLogger


def _finished_without_timeout(traj: Trajectory) -> bool:
    """모델이 max_steps에 도달하지 않고 스스로 종료한 trajectory이면 True.
    have_boxed 여부와 무관하게 수집 대상 — 형식 페널티는 reward에서 처리."""
    return bool(traj.steps) and len(traj.steps) < MAX_STEPS

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

# CUDA_VISIBLE_DEVICES=3,5,6,7,4 → logical: cuda:0~3=GPU3,5,6,7, cuda:4=GPU4
NUM_WORKERS       = 4
TRAINER_DEVICE    = "cuda:4"   # physical GPU 4
MAX_ITERATIONS    = 1000
PROBLEMS_PER_ITER = 64   # 한 sub-iteration당 워커에 넘길 문제 수 (워커 4개 × 16문제)
TRAIN_BATCH_SIZE  = 64   # 이 수만큼 train_traj 모이면 즉시 PPO 1 step

# SFT 체크포인트 (초기 policy 모델로 사용)
SFT_CHECKPOINT    = "/mnt/yoonju/SC/checkpoints/sft/20260322_202515/epoch3"

# PPO 체크포인트 저장 경로: checkpoints/prototype/{ts}/
CHECKPOINT_BASE   = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "prototype")

# 이어서 학습할 체크포인트 경로 (None이면 기본 모델에서 시작)
RESUME_CHECKPOINT: str | None = "" #"/mnt/yoonju/SC/checkpoints/prototype/20260322_224527/iter_0000"

# ─────────────────────────────────────────────────────────────────────────────
# Rollout Worker (Ray actor, GPU 4 or 5)
# ─────────────────────────────────────────────────────────────────────────────

@ray.remote(num_gpus=1)
class RolloutWorker:
    """한 GPU에서 trajectory를 생성하는 Ray Actor.

    Ray가 num_gpus=1 액터에게 CUDA_VISIBLE_DEVICES를 GPU 하나로 좁혀주므로
    내부에서는 항상 device_map="auto" (→ cuda:0) 사용.
    """

    def __init__(self, worker_id: int, rollout_path: str, log_path: str):
        import logging
        self.worker_id    = worker_id
        self.rollout_path = rollout_path

        worker_log_path = log_path
        _wlogger = logging.getLogger()
        _wlogger.setLevel(logging.INFO)
        for h in _wlogger.handlers[:]:
            _wlogger.removeHandler(h)
        fh = logging.FileHandler(worker_log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            f"%(asctime)s [Worker{worker_id}] %(message)s"
        ))
        _wlogger.addHandler(fh)

        self.model, self.tokenizer = load_generator(device_map="auto")
        # 추론 시작 시 파일 미리 생성
        create_rollout_file(rollout_path)
        logging.info(f"준비 완료  rollout → {rollout_path}  log → {worker_log_path}")

    def generate_trajectories(self, problems_batch: List[dict]) -> List[Trajectory]:
        """problems_batch를 배치 GPU 추론 + 병렬 API 호출로 동시 처리.

        성공/실패 모두 JSONL에 실시간 저장.
        """
        return solve_problems_batch(
            self.model,
            self.tokenizer,
            problems_batch,
            rollout_path=self.rollout_path,
        )

    def load_state_dict(self, state_dict: dict):
        """PPOTrainer에서 업데이트된 weights를 동기화."""
        self.model.load_state_dict(state_dict)
        self.model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# PPO Trainer (main process, GPU 6)
# ─────────────────────────────────────────────────────────────────────────────

class PPOTrainer:
    """TD 방식 per-token 리워드 분배 + PPO-clip + KL 패널티.

    리워드 분배:
      스텝 t의 리워드 r_t, 응답 토큰 수 n_t →
      해당 스텝의 각 토큰에 r_t / n_t 동일 분배

    PPO objective:
      L = E[ min(ratio * A, clip(ratio, 1±ε) * A) ] - KL_COEF * KL(π_θ ‖ π_ref)
    """

    def __init__(self, device: str, training_path: str, resume_checkpoint: str | None = None):
        self.device        = device
        self.training_path = training_path

        # policy model: 학습 대상
        model_path = resume_checkpoint if resume_checkpoint else None
        self.model, self.tokenizer = load_generator(device_map={"": device}, model_path=model_path)
        self.model.train()

        # reference model: frozen, KL 계산용 (resume 시 resume_checkpoint 기준, 아니면 SFT 기준)
        # 참고: 7B bfloat16 × 2 = ~28 GB. GPU 4가 충분한 VRAM을 가져야 함
        self.ref_model, _ = load_generator(device_map={"": device}, model_path=resume_checkpoint)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        import bitsandbytes as bnb
        self.optimizer = bnb.optim.AdamW8bit(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=PPO_LR,
        )
        if resume_checkpoint:
            logger.info(f"PPOTrainer 체크포인트 로드: {resume_checkpoint}")
        logger.info(f"PPOTrainer 초기화 완료 ({device})  training → {training_path}")

    # ── 리워드 / 리턴 계산 ──────────────────────────────────────────────────

    @staticmethod
    def _per_token_rewards(traj: Trajectory) -> List[torch.Tensor]:
        """스텝 리워드를 토큰 수로 나눠 각 토큰에 균등 분배.

        중간 스텝: per-step reward (o3 평가)
        마지막 스텝: per-step reward + outcome 보너스/페널티
          정답(is_answer=True)  → +1.0
          오답(is_answer=False) → -1.0
        """
        rewards = []
        gen_steps = [s for s in traj.steps if s.is_generator_step]
        for i, step in enumerate(gen_steps):
            r = step.reward
            if i == len(gen_steps) - 1:
                if traj.is_answer:
                    r += 1.0           # 정답
                elif traj.have_boxed:
                    r += -0.5          # 오답이지만 \boxed{} 형식은 맞춤
                else:
                    r += -1.0          # 형식도 미준수
            n = step.response_ids.shape[1]
            rewards.append(torch.full((n,), r / max(n, 1)))
        return rewards

    @staticmethod
    def _compute_returns(per_token_rewards: List[torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
        """G_i = Σ_{j≥i} γ^{j-i} r_j"""
        flat    = torch.cat(per_token_rewards)
        returns = torch.zeros_like(flat)
        running = 0.0
        for i in reversed(range(flat.shape[0])):
            running   = flat[i].item() + gamma * running
            returns[i] = running
        return returns

    # ── PPO 업데이트 ─────────────────────────────────────────────────────────

    def update(self, trajectories: List[Trajectory]) -> dict:
        """전체 trajectories를 gradient accumulation 후 optimizer.step() 1회 수행. teacher 스텝 제외.

        각 trajectory 내 step들의 손실을 step 수로 평균 → 전체 trajectory 수(n_traj)로 나눠 backward.
        """
        # ── 전체 return 기준선 계산 (baseline) ────────────────────────────────
        all_ret_flat: List[torch.Tensor] = []
        # trajectory별 (inp, resp, lp_old, ret) 리스트 준비
        traj_data = []
        for traj in trajectories:
            per_tok = self._per_token_rewards(traj)
            if not per_tok:
                continue
            returns = self._compute_returns(per_tok)
            inp_list, resp_list, lp_old_list, ret_list = [], [], [], []
            idx = 0
            for step in traj.steps:
                if not step.is_generator_step:
                    continue
                n = step.response_ids.shape[1]
                inp_list.append(step.input_ids.to(self.device))
                resp_list.append(step.response_ids.to(self.device))
                lp_old_list.append(step.log_probs_old.to(self.device))
                ret_list.append(returns[idx: idx + n].to(self.device))
                idx += n
            if inp_list:
                traj_data.append((inp_list, resp_list, lp_old_list, ret_list))
                all_ret_flat.extend(ret_list)

        if not traj_data:
            return {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "entropy": 0.0}

        baseline = torch.cat(all_ret_flat).mean()

        total_pg      = 0.0
        total_kl      = 0.0
        total_entropy = 0.0
        n_traj        = len(traj_data)

        self.model.train()
        self.optimizer.zero_grad()

        # ── trajectory 단위 mini-batch, gradient accumulation ─────────────────
        for traj_idx, (inp_list, resp_list, lp_old_list, ret_list) in enumerate(traj_data):
            n_steps_in_traj  = len(inp_list)
            traj_pg_sum      = 0.0
            traj_kl_sum      = 0.0
            traj_entropy_sum = 0.0

            for j in range(n_steps_in_traj):
                inp_ids   = inp_list[j]
                resp_ids  = resp_list[j]
                lp_old    = lp_old_list[j].detach()
                advantage = (ret_list[j] - baseline).detach()

                # OOM 방지: 시퀀스 길이 초과 스텝 skip
                total_len = inp_ids.shape[1] + resp_ids.shape[1]
                if total_len > MAX_SEQ_LEN:
                    logger.warning(f"[update] seq_len={total_len} > MAX_SEQ_LEN={MAX_SEQ_LEN}, skip step")
                    n_steps_in_traj = max(n_steps_in_traj - 1, 1)
                    continue

                # ── prefix-to-next-step: prefix는 no_grad + KV cache ─────────
                N     = inp_ids.shape[1]
                split = N - 1

                with torch.no_grad():
                    policy_prefix_kv = (
                        self.model(inp_ids[:, :split], use_cache=True).past_key_values
                        if split > 0 else None
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
                        self.ref_model(inp_ids[:, :split], use_cache=True).past_key_values
                        if split > 0 else None
                    )
                    ref_logits = self.ref_model(grad_input, past_key_values=ref_prefix_kv).logits
                    lp_ref = (
                        F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                        .gather(-1, resp_ids.unsqueeze(-1))
                        .squeeze(-1).squeeze(0)
                    )

                ratio   = torch.exp(lp_new - lp_old)
                pg_loss = -torch.min(
                    ratio * advantage,
                    torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * advantage,
                ).mean()
                kl      = (torch.exp(lp_new) * (lp_new - lp_ref)).mean()
                entropy = -lp_new.mean()

                # 스텝마다 즉시 backward → computation graph 즉시 해제 (OOM 방지)
                # backward(A+B+C) == backward(A) + backward(B) + backward(C) 수학적으로 동일
                # 전체 n_traj개를 gradient accumulation 후 마지막에 optimizer.step() 1회
                step_loss = (pg_loss + KL_COEF * kl) / n_steps_in_traj / n_traj
                step_loss.backward()

                traj_pg_sum      += pg_loss.item()
                traj_kl_sum      += kl.item()
                traj_entropy_sum += entropy.item()

            total_pg      += traj_pg_sum      / n_steps_in_traj
            total_kl      += traj_kl_sum      / n_steps_in_traj
            total_entropy += traj_entropy_sum / n_steps_in_traj

        # 모든 trajectory gradient accumulation 후 optimizer.step() 1회
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
    # 실행 타임스탬프 (모든 파일명에 공유)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SAVE_DIR, ts)
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_dir = os.path.join(CHECKPOINT_BASE, ts)
    os.makedirs(checkpoint_dir, exist_ok=True)

    rollout_paths    = [os.path.join(run_dir, f"rollouts_worker_{i}.jsonl") for i in range(NUM_WORKERS)]
    worker_log_paths = [os.path.join(run_dir, f"log_worker_{i}.log")        for i in range(NUM_WORKERS)]
    training_path    = os.path.join(run_dir, "training.jsonl")
    log_path         = os.path.join(run_dir, "run.log")

    # logger → 파일에만 기록 (터미널 출력 없음)
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(file_handler)
    logger.info(f"로그 파일: {log_path}")

    # Ray 초기화 (logical GPU 0~3=physical GPU3,5,6,7; 트레이너 logical cuda:4=physical GPU4는 pool 밖)
    # log_to_driver=False: worker 로그가 터미널에 출력되지 않음
    ray.init(num_gpus=NUM_WORKERS, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)  # logical GPU 0~3 = physical GPU 3,5,6,7

    # 데이터셋 로드
    problems      = load_problems(DATASET_PATH)
    n             = len(problems)
    val_problems  = load_math500()

    workers = [RolloutWorker.remote(worker_id=i, rollout_path=rollout_paths[i], log_path=worker_log_paths[i]) for i in range(NUM_WORKERS)]

    # 트레이너 (physical GPU 4 = logical cuda:4)
    trainer = PPOTrainer(device=TRAINER_DEVICE, training_path=training_path, resume_checkpoint=RESUME_CHECKPOINT)

    logger.info(f"문제 {n}개  |  ts={ts}")
    for i, rp in enumerate(rollout_paths):
        logger.info(f"  rollout  → {rp}")
    logger.info(f"  training → {training_path}")

    # ── WandB 초기화 ──────────────────────────────────────────────────────────
    from utils import (
        PPO_CLIP_EPS, PPO_LR, PPO_MAX_GRAD_NORM,
        KL_COEF, GAMMA, MAX_STEPS, MAX_SEQ_LEN, TRUNCATE_TOKEN_LIMIT,
        GENERATOR_MODEL_ID, GENERATOR_MAX_NEW_TOKENS,
    )
    wlogger = WandbLogger(
        config={
            "model":                  GENERATOR_MODEL_ID,
            "ppo_lr":                 PPO_LR,
            "ppo_clip_eps":           PPO_CLIP_EPS,
            "train_batch_size":       TRAIN_BATCH_SIZE,
            "ppo_max_grad_norm":      PPO_MAX_GRAD_NORM,
            "kl_coef":                KL_COEF,
            "gamma":                  GAMMA,
            "max_steps":              MAX_STEPS,
            "max_seq_len":            MAX_SEQ_LEN,
            "truncate_token_limit":   TRUNCATE_TOKEN_LIMIT,
            "max_new_tokens":         GENERATOR_MAX_NEW_TOKENS,
            "num_workers":            NUM_WORKERS,
            "problems_per_iter":      PROBLEMS_PER_ITER,
            "max_iterations":         MAX_ITERATIONS,
        },
        project="sc-ppo",
        run_name=ts,
    )
    wlogger.set_val_problems(problems)

    answer_total   = 0
    problem_cursor = 0

    iter_bar = tqdm(range(MAX_ITERATIONS), desc="PPO", unit="iter", dynamic_ncols=True)
    for iteration in iter_bar:

        # ── TRAIN_BATCH_SIZE개 모일 때까지 배치 반복 생성 ─────────────────
        # 수집 기준: max_steps 도달 없이 스스로 종료한 trajectory
        all_trajs   = []
        train_trajs = []
        collect_bar = tqdm(
            total=TRAIN_BATCH_SIZE, desc="Collecting",
            unit="traj", leave=False, dynamic_ncols=True,
        )
        while len(train_trajs) < TRAIN_BATCH_SIZE:
            start = problem_cursor % n
            batch = (problems + problems)[start: start + PROBLEMS_PER_ITER]
            chunk = PROBLEMS_PER_ITER // NUM_WORKERS

            futures   = [workers[i].generate_trajectories.remote(batch[i*chunk:(i+1)*chunk]) for i in range(NUM_WORKERS)]
            new_trajs = [t for tl in ray.get(futures) for t in tl]
            all_trajs.extend(new_trajs)

            prev_len = len(train_trajs)
            train_trajs.extend(t for t in new_trajs if _finished_without_timeout(t))
            added = min(len(train_trajs), TRAIN_BATCH_SIZE) - prev_len
            if added > 0:
                collect_bar.update(added)

            problem_cursor += PROBLEMS_PER_ITER

        collect_bar.close()
        train_trajs = train_trajs[:TRAIN_BATCH_SIZE]
        answer_total  += len(train_trajs)
        n_correct      = sum(1 for t in train_trajs if t.is_answer)
        n_boxed        = sum(1 for t in train_trajs if t.have_boxed)
        n_timeout      = len(all_trajs) - len(train_trajs)
        logger.info(
            f"[iter {iteration:4d}]  generated={len(all_trajs)}  "
            f"ppo_usable={len(train_trajs)}/{TRAIN_BATCH_SIZE}  timeout={n_timeout}  "
            f"correct={n_correct}  boxed={n_boxed}  total_ppo={answer_total}"
        )

        # ── 롤아웃 지표 기록 ──────────────────────────────────────────────
        wlogger.log_rollout(train_trajs, iteration, 1, all_trajs=all_trajs)

        # ── PPO 업데이트 ──────────────────────────────────────────────────
        stats = trainer.update(train_trajs)
        logger.info(
            f"           PPO → loss={stats['loss']:.4f}  "
            f"pg={stats['pg_loss']:.4f}  kl={stats['kl']:.6f}  "
            f"entropy={stats['entropy']:.4f}"
        )

        # ── 학습 지표 기록 ────────────────────────────────────────────────
        wlogger.log_train(stats, iteration)

        # ── Validation (MATH-500, 매 iteration) ──────────────────────────
        logger.info("           validation 시작 (MATH-500)...")
        val_metrics = validate_math500(trainer.model, trainer.tokenizer, val_problems)
        wlogger.log_validation(val_metrics, iteration)
        logger.info(
            f"           val accuracy={val_metrics.get('val/accuracy', 0):.4f}"
            f"  ({val_metrics.get('val/n_correct', 0)}/{val_metrics.get('val/n_total', 0)})"
        )

        # ── weights 동기화 ────────────────────────────────────────────────
        weights = trainer.get_state_dict()
        ray.get([w.load_state_dict.remote(weights) for w in workers])

        # ── checkpoint 저장 ───────────────────────────────────────────────
        ckpt_dir = os.path.join(checkpoint_dir, f"iter_{iteration:04d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        trainer.model.save_pretrained(ckpt_dir)
        trainer.tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"           ckpt → {ckpt_dir}")

        # ── iter_bar postfix 업데이트 ─────────────────────────────────────
        iter_bar.set_postfix(
            loss=f"{stats['loss']:.4f}",
            kl=f"{stats['kl']:.5f}",
            correct=f"{n_correct}/{TRAIN_BATCH_SIZE}",
            val_acc=f"{val_metrics.get('val/accuracy', 0):.3f}",
        )

    logger.info("학습 완료.")
    wlogger.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
