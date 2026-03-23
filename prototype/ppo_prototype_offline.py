"""
prototype/ppo_prototype_offline.py

오프라인 PPO: 사전 생성된 rollout JSONL로 PPO 학습.

왜 off-policy인가:
  JSONL에는 text/reward만 저장되어 있고, 데이터를 생성한 모델과
  현재 SFT 모델이 다르다. 따라서 lp_old를 SFT 모델로 재계산해서
  behavior policy 삼아 PPO를 돌린다.
  → ratio = exp(lp_new - lp_old) 에서 초기에 ratio ≈ 1이 되므로
    clip이 크게 발동하지 않고 안정적으로 시작할 수 있다.

제약:
  - "correct" 스텝의 correct_reason은 JSONL에 저장되지 않아
    "The previous step was INCORRECT." 로 근사한다.
    (patcher 스텝은 is_generator_step=False이므로 PPO에서 제외됨)

GPU 구성:
  CUDA_VISIBLE_DEVICES=4 → logical cuda:0 = physical GPU 4
  policy + ref 모델 모두 cuda:0에 올림 (bfloat16 7B × 2 ≈ 28 GB)

실행:
  cd prototype && python ppo_prototype_offline.py \\
      --data_dir ../output/prototype/20260322_085538
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DATASET_PATH,
    KL_COEF,
    MAX_SEQ_LEN,
    MAX_STEPS,
    PPO_CLIP_EPS,
    PPO_COLLECT_SIZE,
    PPO_MINI_BATCH_SIZE,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    SAVE_DIR,
    SYSTEM_CORRECT,
    SYSTEM_SOLVE,
    Trajectory,
    StepRecord,
    _compute_log_probs,
    _correct_user,
    _solve_user,
    build_chat_prompt,
    format_final_step,
    load_generator,
    save_trajectory,
)
from record_wandb import WandbLogger

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

TRAINER_DEVICE  = "cuda:0"   # physical GPU 4
PPO_EPOCHS      = 3          # 오프라인이므로 데이터를 여러 번 재사용

SFT_CHECKPOINT  = "/mnt/yoonju/SC/checkpoints/sft/20260322_061039/epoch2"
CHECKPOINT_BASE = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "offline_ppo")

# ─────────────────────────────────────────────────────────────────────────────
# JSONL → Trajectory 재구성
# ─────────────────────────────────────────────────────────────────────────────

CORRECT_REASON_DEFAULT = "The previous step was INCORRECT."


def build_trajectory_from_record(
    record: dict,
    model,
    tokenizer,
    device: str,
) -> Optional[Trajectory]:
    """JSONL 레코드 1개를 Trajectory로 재구성.

    텍스트만 저장된 JSONL에서 prompt를 재조립하고,
    현재 model로 input_ids / response_ids / lp_old를 계산한다.

    correct_reason 근사:
      - "correct" action 스텝 → "The previous step was INCORRECT."
      - JSONL에 실제 reason이 없으므로 고정값을 사용한다.
      - patcher 스텝(is_generator_step=False)은 PPO에서 어차피 제외.
    """
    problem    = record["problem"]
    answer     = record["answer"]
    history: List[str] = []

    traj = Trajectory(
        problem_id = str(record.get("problem_id", "")),
        problem    = problem,
        answer     = answer,
        difficulty = record.get("difficulty"),
        have_boxed = record.get("have_boxed", False),
        is_answer  = record.get("is_answer",  False),
        patcher_wrong = record.get("patcher_wrong", False),
    )

    for step_data in record["steps"]:
        action  = step_data["action"]          # "solve" | "correct"
        text    = step_data["text"]
        reward  = step_data["reward"]
        gt_next = step_data["ground_truth_next_action"]
        is_gen  = step_data["is_generator_step"]

        # ── 프롬프트 재구성 ───────────────────────────────────────────────
        if action == "solve":
            prompt = build_chat_prompt(tokenizer, SYSTEM_SOLVE, _solve_user(problem, history))
        else:
            prompt = build_chat_prompt(
                tokenizer, SYSTEM_CORRECT,
                _correct_user(problem, history, CORRECT_REASON_DEFAULT),
            )

        # ── 토큰화 + lp_old 계산 ─────────────────────────────────────────
        inp_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        # response = 추론 텍스트 + GT 액션 토큰 (온라인 build_gt_response와 동일)
        resp_text = text + gt_next
        resp_ids  = tokenizer(
            resp_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)

        with torch.no_grad():
            lp_old = _compute_log_probs(model, inp_ids, resp_ids)

        traj.steps.append(StepRecord(
            step_idx               = step_data["step_idx"],
            action                 = action,
            text                   = text,
            reward                 = reward,
            predicted_next_action  = step_data.get("predicted_next_action",  gt_next),
            ground_truth_next_action = gt_next,
            input_ids              = inp_ids.cpu(),
            response_ids           = resp_ids.cpu(),
            log_probs_old          = lp_old.cpu(),
            is_generator_step      = is_gen,
        ))

        # ── history 누적 (온라인 solve 루프와 동일한 규칙) ─────────────────
        # "solve" 제너레이터 스텝만 format_final_step 적용
        if action == "solve" and is_gen:
            history.append(format_final_step(text))
        else:
            history.append(text)

    return traj if traj.steps else None


def load_offline_data(
    jsonl_paths: List[str],
    model,
    tokenizer,
    device: str,
) -> List[Trajectory]:
    """여러 JSONL 파일을 로드해 Trajectory 리스트로 변환.

    _finished_without_timeout 필터 적용:
      max_steps에 도달하지 않고 스스로 종료한 trajectory만 수집.
    """
    trajs: List[Trajectory] = []
    total_records = 0

    for path in jsonl_paths:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        logger.info(f"로드: {path}  ({len(lines)}건)")

        for line in tqdm(lines, desc=f"재구성 {Path(path).name}", leave=False):
            line = line.strip()
            if not line:
                continue
            total_records += 1
            record = json.loads(line)
            traj = build_trajectory_from_record(record, model, tokenizer, device)
            if traj is None:
                continue
            gen_steps = [s for s in traj.steps if s.is_generator_step]
            # _finished_without_timeout: step이 있고 max_steps 미도달
            if gen_steps and len(traj.steps) < MAX_STEPS:
                trajs.append(traj)

    logger.info(
        f"총 {total_records}건 → 학습 가능 {len(trajs)}건 "
        f"({len(trajs)/max(total_records,1)*100:.1f}%)"
    )
    return trajs


# ─────────────────────────────────────────────────────────────────────────────
# PPO Trainer (ppo_prototype.py와 동일한 로직, 단일 파일로 복사)
# ─────────────────────────────────────────────────────────────────────────────

class PPOTrainer:
    """TD 방식 per-token 리워드 분배 + PPO-clip + KL 패널티."""

    def __init__(self, device: str, training_path: str, resume_checkpoint: str | None = None):
        self.device        = device
        self.training_path = training_path

        model_path = resume_checkpoint if resume_checkpoint else None
        self.model, self.tokenizer = load_generator(device_map={"": device}, model_path=model_path)
        self.model.train()

        # reference model: frozen (항상 SFT 체크포인트 기준)
        self.ref_model, _ = load_generator(device_map={"": device})
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        import bitsandbytes as bnb
        self.optimizer = bnb.optim.AdamW8bit(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=PPO_LR,
        )
        logger.info(f"PPOTrainer 초기화 완료 ({device})  training → {training_path}")

    @staticmethod
    def _per_token_rewards(traj: Trajectory) -> List[torch.Tensor]:
        rewards = []
        gen_steps = [s for s in traj.steps if s.is_generator_step]
        for i, step in enumerate(gen_steps):
            r = step.reward
            if i == len(gen_steps) - 1:
                if traj.is_answer:
                    r += 1.0
                elif traj.have_boxed:
                    r += -0.5
                else:
                    r += -1.0
            n = step.response_ids.shape[1]
            rewards.append(torch.full((n,), r / max(n, 1)))
        return rewards

    @staticmethod
    def _compute_returns(per_token_rewards: List[torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
        flat    = torch.cat(per_token_rewards)
        returns = torch.zeros_like(flat)
        running = 0.0
        for i in reversed(range(flat.shape[0])):
            running    = flat[i].item() + gamma * running
            returns[i] = running
        return returns

    def update(self, trajectories: List[Trajectory]) -> dict:
        all_ret_flat: List[torch.Tensor] = []
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

        total_pg = total_kl = total_entropy = 0.0
        n_traj = len(traj_data)

        self.model.train()
        self.optimizer.zero_grad()

        for traj_idx, (inp_list, resp_list, lp_old_list, ret_list) in enumerate(traj_data):
            n_steps_in_traj  = len(inp_list)
            traj_pg_sum = traj_kl_sum = traj_entropy_sum = 0.0

            for j in range(n_steps_in_traj):
                inp_ids   = inp_list[j]
                resp_ids  = resp_list[j]
                lp_old    = lp_old_list[j].detach()
                advantage = (ret_list[j] - baseline).detach()

                total_len = inp_ids.shape[1] + resp_ids.shape[1]
                if total_len > MAX_SEQ_LEN:
                    logger.warning(f"seq_len={total_len} > MAX_SEQ_LEN={MAX_SEQ_LEN}, skip")
                    n_steps_in_traj = max(n_steps_in_traj - 1, 1)
                    continue

                N     = inp_ids.shape[1]
                split = N - 1

                with torch.no_grad():
                    policy_prefix_kv = (
                        self.model(inp_ids[:, :split], use_cache=True).past_key_values
                        if split > 0 else None
                    )

                grad_input = torch.cat([inp_ids[:, -1:], resp_ids], dim=1)
                logits     = self.model(grad_input, past_key_values=policy_prefix_kv).logits
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

                step_loss = (pg_loss + KL_COEF * kl) / n_steps_in_traj / PPO_MINI_BATCH_SIZE
                step_loss.backward()

                traj_pg_sum      += pg_loss.item()
                traj_kl_sum      += kl.item()
                traj_entropy_sum += entropy.item()

            total_pg      += traj_pg_sum      / n_steps_in_traj
            total_kl      += traj_kl_sum      / n_steps_in_traj
            total_entropy += traj_entropy_sum / n_steps_in_traj

            if (traj_idx + 1) % PPO_MINI_BATCH_SIZE == 0 or traj_idx == n_traj - 1:
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
        return {k: v.cpu() for k, v in self.model.state_dict().items()}


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True,
        help="rollouts_worker_*.jsonl 이 있는 디렉터리",
    )
    parser.add_argument(
        "--batch_size", type=int, default=PPO_COLLECT_SIZE,
        help=f"한 번에 PPO 업데이트할 trajectory 수 (기본 {PPO_COLLECT_SIZE})",
    )
    parser.add_argument(
        "--epochs", type=int, default=PPO_EPOCHS,
        help=f"데이터 반복 epoch 수 (기본 {PPO_EPOCHS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(SAVE_DIR) / f"offline_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(CHECKPOINT_BASE) / ts
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_path = str(run_dir / "training.jsonl")
    log_path      = str(run_dir / "run.log")

    # 파일 로거
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(fh)
    logger.info(f"로그: {log_path}")

    # ── JSONL 파일 수집 ────────────────────────────────────────────────────
    jsonl_paths = sorted(data_dir.glob("rollouts_worker_*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"rollouts_worker_*.jsonl 없음: {data_dir}")
    logger.info(f"JSONL 파일: {[str(p) for p in jsonl_paths]}")

    # ── PPOTrainer 초기화 (policy + ref 모델 로드) ─────────────────────────
    trainer = PPOTrainer(
        device            = TRAINER_DEVICE,
        training_path     = training_path,
        resume_checkpoint = SFT_CHECKPOINT,
    )

    # ── 오프라인 데이터 로드 + lp_old 재계산 ──────────────────────────────
    # lp_old는 초기 SFT 모델 기준으로 한 번만 계산한다.
    # → 이후 PPO epoch에서 lp_new가 달라져도 lp_old는 고정.
    logger.info("오프라인 데이터 로드 + lp_old 계산 시작...")
    trainer.model.eval()
    all_trajs = load_offline_data(
        jsonl_paths=[str(p) for p in jsonl_paths],
        model     = trainer.model,
        tokenizer = trainer.tokenizer,
        device    = TRAINER_DEVICE,
    )
    if not all_trajs:
        raise RuntimeError("학습 가능한 trajectory가 없습니다.")
    logger.info(f"총 학습 trajectory: {len(all_trajs)}")

    # ── WandB ─────────────────────────────────────────────────────────────
    from utils import (
        PPO_CLIP_EPS, PPO_LR, PPO_MAX_GRAD_NORM, PPO_MINI_BATCH_SIZE,
        KL_COEF, GAMMA, MAX_STEPS, MAX_SEQ_LEN, TRUNCATE_TOKEN_LIMIT,
        GENERATOR_MODEL_ID, GENERATOR_MAX_NEW_TOKENS,
    )
    wlogger = WandbLogger(
        config={
            "mode":                   "offline_ppo",
            "data_dir":               str(data_dir),
            "n_trajectories":         len(all_trajs),
            "model":                  GENERATOR_MODEL_ID,
            "ppo_lr":                 PPO_LR,
            "ppo_clip_eps":           PPO_CLIP_EPS,
            "ppo_batch_size":         args.batch_size,
            "ppo_mini_batch_size":    PPO_MINI_BATCH_SIZE,
            "ppo_max_grad_norm":      PPO_MAX_GRAD_NORM,
            "ppo_epochs":             args.epochs,
            "kl_coef":                KL_COEF,
            "gamma":                  GAMMA,
            "max_steps":              MAX_STEPS,
            "max_seq_len":            MAX_SEQ_LEN,
        },
        project="sc-ppo",
        run_name=f"offline_{ts}",
    )

    # ── PPO epoch 루프 ─────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(args.epochs):
        import random
        random.shuffle(all_trajs)

        epoch_bar = tqdm(
            range(0, len(all_trajs), args.batch_size),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            unit="batch",
        )
        for batch_start in epoch_bar:
            batch = all_trajs[batch_start: batch_start + args.batch_size]

            stats = trainer.update(batch)
            logger.info(
                f"[epoch {epoch+1}][step {global_step}]  "
                f"loss={stats['loss']:.4f}  pg={stats['pg_loss']:.4f}  "
                f"kl={stats['kl']:.6f}  entropy={stats['entropy']:.4f}"
            )
            wlogger.log_train(stats, global_step)

            epoch_bar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                kl=f"{stats['kl']:.5f}",
            )
            global_step += 1

        # epoch 체크포인트
        ckpt_dir = checkpoint_dir / f"epoch_{epoch+1:02d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(ckpt_dir))
        trainer.tokenizer.save_pretrained(str(ckpt_dir))
        logger.info(f"ckpt → {ckpt_dir}")

    logger.info("오프라인 PPO 완료.")
    wlogger.finish()


if __name__ == "__main__":
    main()
