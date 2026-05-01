"""
prototype/ppo_prototype.py

실행: python train_ppo.py

GPU 구성 (config/config.yaml의 ppo.rollout_gpus / ppo.train_gpus 에서 설정)
  rollout_gpus → RolloutWorker (데이터 생성, Ray actor)
  train_gpus   → PPOTrainer    (모델 학습, main process)

파이프라인 (iteration 단위):
  1. 네 워커가 문제를 16개씩 병렬로 trajectory 생성 (총 64개/서브이터)
     → 각 워커는 datasets/prototype/rollouts_{ts}_workerN.jsonl 에 실시간 저장
  2. train_trajs 64개 모이면 즉시 PPO 1 step (8-bit AdamW + gradient accumulation)
  3. 업데이트된 weights → 워커에 동기화
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from tqdm import tqdm


def _parse_args():
    p = argparse.ArgumentParser(description="PPO Online Training")
    # GPU 설정
    p.add_argument("--rollout_gpus", type=str, help="Rollout GPU IDs, comma-separated (e.g. '2,3,4,5')")
    p.add_argument("--train_gpus",   type=str, help="Train GPU IDs, comma-separated (e.g. '6')")
    # 체크포인트
    p.add_argument("--resume_checkpoint", type=str, default=None)
    # 학습 설정
    p.add_argument("--max_iterations",   type=int)
    p.add_argument("--problems_per_gpu",  type=int)
    p.add_argument("--train_batch_size",  type=int)
    p.add_argument("--dataset",           type=str)
    # 하이퍼파라미터
    p.add_argument("--lr",          type=float)
    p.add_argument("--clip_eps",    type=float)
    p.add_argument("--kl_coef",     type=float)
    p.add_argument("--gamma",       type=float)
    p.add_argument("--max_seq_len", type=int)
    p.add_argument("--run_ts",     type=str, default=None,
                   help="실행 타임스탬프 (쉘에서 주입). 없으면 Python이 자체 생성")
    return p.parse_args()

_args = _parse_args()

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    CONF,
    DATASET_PATH,
    GAMMA,
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
    VLLM_MAX_MODEL_LEN,
    Trajectory,
    create_rollout_file,
    load_generator,
    load_problems,
    save_trajectory,
)
from generate_trajectory import solve_problems_batch_vllm
from utils import solve_problems_batch
from record_wandb import WandbLogger

# ─────────────────────────────────────────────────────────────────────────────
# config + CLI 인자로 실행 설정 결정 (CLI 인자가 config보다 우선)
# ─────────────────────────────────────────────────────────────────────────────
_ppo = CONF["ppo"]

if _args.rollout_gpus   is not None: _ppo["rollout_gpus"]    = [int(g) for g in _args.rollout_gpus.split(",")]
if _args.train_gpus     is not None: _ppo["train_gpus"]      = [int(g) for g in _args.train_gpus.split(",")]
if _args.max_iterations  is not None: _ppo["max_iterations"]  = _args.max_iterations
if _args.problems_per_gpu  is not None: _ppo["problems_per_gpu"]  = _args.problems_per_gpu
if _args.train_batch_size  is not None: _ppo["train_batch_size"]  = _args.train_batch_size
if _args.lr        is not None: PPO_LR      = _args.lr
if _args.clip_eps  is not None: PPO_CLIP_EPS = _args.clip_eps
if _args.kl_coef   is not None: KL_COEF     = _args.kl_coef
if _args.gamma     is not None: GAMMA       = _args.gamma
if _args.max_seq_len is not None: MAX_SEQ_LEN = _args.max_seq_len
if _args.dataset   is not None: DATASET_PATH = str(Path(__file__).resolve().parent.parent / _args.dataset)
if _args.resume_checkpoint is not None: _ppo["resume_checkpoint"] = _args.resume_checkpoint

ROLLOUT_GPUS      = _ppo["rollout_gpus"]
TRAIN_GPUS        = _ppo["train_gpus"]
RESUME_CHECKPOINT = _ppo.get("resume_checkpoint")
NUM_WORKERS       = len(ROLLOUT_GPUS)
TRAINER_DEVICE    = "cuda:0"  # main 프로세스는 TRAIN_GPUS만 노출되므로 항상 cuda:0
MAX_ITERATIONS    = _ppo["max_iterations"]
PROBLEMS_PER_ITER = _ppo["problems_per_gpu"] * NUM_WORKERS
TRAIN_BATCH_SIZE  = _ppo["train_batch_size"]
CHECKPOINT_BASE   = str(Path(__file__).resolve().parent.parent / CONF["checkpoint"]["ppo_checkpoint_base"])
SAVE_DIR          = str(Path(__file__).resolve().parent.parent / CONF["output_path"]["ppo"])

# main 프로세스(PPOTrainer)는 학습 GPU만 노출
# RolloutWorker는 runtime_env로 각자 CUDA_VISIBLE_DEVICES를 설정하므로 여기서 불필요
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in TRAIN_GPUS)
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import ray
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def _finished_without_timeout(traj: Trajectory) -> bool:
    """max_steps 도달 없이 스스로 종료한 trajectory이면 True."""
    return bool(traj.steps) and len(traj.steps) < MAX_STEPS

# ─────────────────────────────────────────────────────────────────────────────
# Rollout Worker (Ray actor)
# ─────────────────────────────────────────────────────────────────────────────

@ray.remote
class RolloutWorker:
    """한 GPU에서 vLLM AsyncLLMEngine으로 trajectory를 생성하는 Ray Actor.

    runtime_env로 CUDA_VISIBLE_DEVICES=ROLLOUT_GPUS[i]가 주입되므로
    내부에서는 항상 cuda:0 를 사용한다.

    vLLM continuous batching: N개의 문제를 asyncio.gather로 동시에 처리하며,
    각 문제는 다른 문제의 스텝 완료나 API 호출을 기다리지 않고 독립적으로 진행한다.
    """

    def __init__(self, worker_id: int, rollout_path: str, log_path: str):
        import asyncio
        import logging
        import os
        import sys

        self.worker_id    = worker_id
        self.rollout_path = rollout_path

        _wlogger = logging.getLogger()
        _wlogger.setLevel(logging.INFO)
        for h in _wlogger.handlers[:]:
            _wlogger.removeHandler(h)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(f"%(asctime)s [Worker{worker_id}] %(message)s"))
        _wlogger.addHandler(fh)

        # source/ 디렉토리를 경로에 추가 (Ray actor 프로세스는 sys.path가 다를 수 있음)
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        if _src_dir not in sys.path:
            sys.path.insert(0, _src_dir)

        from utils import (
            ACTION_TOKENS, GENERATOR_CACHE_DIR, GENERATOR_MODEL_ID, SFT_CHECKPOINT,
        )
        from transformers import AutoTokenizer

        model_path = SFT_CHECKPOINT if SFT_CHECKPOINT else GENERATOR_MODEL_ID

        # tokenizer: 특수 액션 토큰 포함 (load_generator 와 동일 설정)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=GENERATOR_CACHE_DIR, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})

        # 영속 이벤트 루프 생성 (vLLM asyncio 태스크가 루프 수명 동안 유지됨)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # vLLM AsyncLLMEngine 초기화
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            enable_sleep_mode=True,
            enforce_eager=False,
            max_model_len=VLLM_MAX_MODEL_LEN,
            download_dir=GENERATOR_CACHE_DIR,
        )

        async def _init_engine():
            return AsyncLLMEngine.from_engine_args(engine_args)

        self.engine = self._loop.run_until_complete(_init_engine())

        create_rollout_file(rollout_path)
        logging.info(f"준비 완료 (vLLM)  rollout → {rollout_path}  log → {log_path}")

    def generate_trajectories(self, problems_batch: List[dict]) -> List[Trajectory]:
        """problems_batch를 vLLM continuous batching으로 동시 처리."""
        return solve_problems_batch_vllm(
            self.engine, self.tokenizer, problems_batch,
            rollout_path=self.rollout_path, loop=self._loop,
        )

    def load_state_dict(self, state_dict: dict):
        """PPOTrainer에서 업데이트된 weights를 vLLM 엔진에 동기화.

        state_dict → 공유 메모리(/dev/shm) 임시 파일 → vLLM 엔진 코어에서 직접 로드.
        ZMQ를 통한 대용량 텐서 직렬화를 피하기 위해 파일 경로만 IPC로 전달한다.
        """
        import logging
        import os
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(dir="/dev/shm", suffix=".pt", delete=False) as f:
                tmp_path = f.name
        except (OSError, FileNotFoundError):
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                tmp_path = f.name

        torch.save(state_dict, tmp_path)

        def _load_weights(wrapper, path):
            import torch as _t
            import os as _os
            sd = _t.load(path, weights_only=True, map_location="cpu")
            worker = wrapper.worker
            model  = worker.model_runner.model
            for name, tensor in sd.items():
                try:
                    p = model.get_parameter(name)
                    if p.shape != tensor.shape:
                        continue  # skip resized embeddings (ACTION_TOKENS 추가로 크기 불일치)
                    p.data.copy_(tensor.to(p.device))
                except Exception:
                    pass
            _os.unlink(path)

        self._loop.run_until_complete(
            self.engine.collective_rpc(_load_weights, kwargs={"path": tmp_path})
        )
        logging.info(f"[Worker{self.worker_id}] weights 동기화 완료")

# ─────────────────────────────────────────────────────────────────────────────
# PPO Trainer (main process)
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
            logger.info(f"PPOTrainer 체크포인트 로드: {resume_checkpoint}")
        logger.info(f"PPOTrainer 초기화 완료 ({device})  training → {training_path}")

    # ── 리워드 / 리턴 계산 ──────────────────────────────────────────────────

    @staticmethod
    def _per_token_rewards(traj: Trajectory) -> List[torch.Tensor]:
        """스텝 리워드를 토큰 수로 나눠 각 토큰에 균등 분배.

        reward는 generate_trajectory.py의 score_step에서 이미 계산됨:
          - llm_reward (0.0~0.5): 매 스텝 풀이 품질
          - format_reward: 마지막 스텝 정답 +0.5 / 오답 0.0, 중간 스텝 boxed{} 있으면 -0.1
          - 총합 최대 1.0
        length 페널티: 틀린 trajectory에서만, 스텝 토큰 수가 512 초과 시 초과 토큰당 -0.0002
        """
        rewards = []
        for step in traj.steps:
            n = step.response_ids.shape[1]
            length_penalty = -max(0, n - 512) * 0.0002 if not traj.is_answer else 0.0
            r = step.final_reward + length_penalty
            rewards.append(torch.full((n,), r / max(n, 1)))
        return rewards

    @staticmethod
    def _compute_returns(per_token_rewards: List[torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
        """G_i = Σ_{j≥i} γ^{j-i} r_j"""
        flat    = torch.cat(per_token_rewards)
        returns = torch.zeros_like(flat)
        running = 0.0
        for i in reversed(range(flat.shape[0])):
            running    = flat[i].item() + gamma * running
            returns[i] = running
        return returns

    # ── PPO 업데이트 ─────────────────────────────────────────────────────────

    def update(self, trajectories: List[Trajectory]) -> dict:
        """전체 trajectories를 gradient accumulation 후 optimizer.step() 1회 수행."""
        all_ret_flat: List[torch.Tensor] = []
        traj_data = []
        for traj in trajectories:
            per_tok = self._per_token_rewards(traj)
            if not per_tok:
                continue
            returns = self._compute_returns(per_tok, gamma=GAMMA)
            inp_list, resp_list, lp_old_list, ret_list = [], [], [], []
            idx = 0
            for step in traj.steps:
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
        n_traj   = len(traj_data)

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

                max_inp_len = MAX_SEQ_LEN - resp_ids.shape[1]
                if max_inp_len <= 0:
                    logger.warning(f"[update] resp_len={resp_ids.shape[1]} >= MAX_SEQ_LEN={MAX_SEQ_LEN}, skip step")
                    n_steps_in_traj = max(n_steps_in_traj - 1, 1)
                    continue
                if inp_ids.shape[1] > max_inp_len:
                    logger.warning(f"[update] inp truncate {inp_ids.shape[1]} → {max_inp_len}")
                    inp_ids = inp_ids[:, -max_inp_len:]

                N, split = inp_ids.shape[1], inp_ids.shape[1] - 1
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
        """워커에 동기화할 CPU state dict 반환.

        vLLM 워커는 ACTION_TOKENS 추가 전 원본 vocab 크기로 로드되므로,
        embedding/lm_head처럼 vocab 축에서 크기가 달라진 파라미터는 제외한다.
        """
        vocab_size = self.ref_model.config.vocab_size  # 원본 vocab 크기
        sd = {}
        for k, v in self.model.state_dict().items():
            if v.shape[0] != vocab_size and any(tag in k for tag in ("embed_tokens", "lm_head")):
                continue
            sd[k] = v.cpu()
        return sd

# ─────────────────────────────────────────────────────────────────────────────
# 메인 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ts      = _args.run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
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
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()   # stdout → shell.log(tee)에도 찍힘
    sh.setFormatter(fmt)
    root_logger.addHandler(fh)
    root_logger.addHandler(sh)
    logger.info(f"로그 파일: {log_path}")

    ray.init(include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)

    problems = load_problems(DATASET_PATH)
    n        = len(problems)

    workers = [
        RolloutWorker.options(
            runtime_env={"env_vars": {
                "CUDA_VISIBLE_DEVICES": str(ROLLOUT_GPUS[i]),
                "MASTER_PORT": str(36000 + i),
            }}
        ).remote(worker_id=i, rollout_path=rollout_paths[i], log_path=worker_log_paths[i])
        for i in range(NUM_WORKERS)
    ]

    trainer = PPOTrainer(device=TRAINER_DEVICE, training_path=training_path, resume_checkpoint=RESUME_CHECKPOINT)

    logger.info(f"문제 {n}개  |  ts={ts}")
    for i, rp in enumerate(rollout_paths):
        logger.info(f"  rollout  → {rp}")
    logger.info(f"  training → {training_path}")

    wlogger = WandbLogger(
        config={
            "model":                GENERATOR_MODEL_ID,
            "ppo_lr":               PPO_LR,
            "ppo_clip_eps":         PPO_CLIP_EPS,
            "train_batch_size":     TRAIN_BATCH_SIZE,
            "ppo_max_grad_norm":    PPO_MAX_GRAD_NORM,
            "kl_coef":              KL_COEF,
            "gamma":                GAMMA,
            "length_penalty_coef":  LENGTH_PENALTY_COEF,
            "max_steps":            MAX_STEPS,
            "max_seq_len":          MAX_SEQ_LEN,
            "truncate_token_limit": TRUNCATE_TOKEN_LIMIT,
            "max_new_tokens":       GENERATOR_MAX_NEW_TOKENS,
            "num_workers":          NUM_WORKERS,
            "problems_per_iter":    PROBLEMS_PER_ITER,
            "max_iterations":       MAX_ITERATIONS,
        },
        project="sc-ppo",
        run_name=ts,
    )
    wlogger.set_val_problems([])

    answer_total   = 0
    problem_cursor = 0

    iter_bar = tqdm(range(MAX_ITERATIONS), desc="PPO", unit="iter", dynamic_ncols=True)
    for iteration in iter_bar:

        all_trajs   = []
        train_trajs = []
        collect_bar = tqdm(total=TRAIN_BATCH_SIZE, desc="Collecting", unit="traj", leave=False, dynamic_ncols=True)

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
        train_trajs  = train_trajs[:TRAIN_BATCH_SIZE]
        answer_total += len(train_trajs)
        n_correct     = sum(1 for t in train_trajs if t.is_answer)
        n_boxed       = sum(1 for t in train_trajs if t.have_boxed)
        n_timeout     = len(all_trajs) - len(train_trajs)
        logger.info(
            f"[iter {iteration:4d}]  generated={len(all_trajs)}  "
            f"ppo_usable={len(train_trajs)}/{TRAIN_BATCH_SIZE}  timeout={n_timeout}  "
            f"correct={n_correct}  boxed={n_boxed}  total_ppo={answer_total}"
        )

        wlogger.log_rollout(train_trajs, iteration, 1, all_trajs=all_trajs)

        stats = trainer.update(train_trajs)
        logger.info(
            f"           PPO → loss={stats['loss']:.4f}  "
            f"pg={stats['pg_loss']:.4f}  kl={stats['kl']:.6f}  "
            f"entropy={stats['entropy']:.4f}"
        )

        wlogger.log_train(stats, iteration)

        val_metrics = {}

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
            correct=f"{n_correct}/{TRAIN_BATCH_SIZE}",
            val_acc=f"{val_metrics.get('val/accuracy', 0):.3f}",
        )

    logger.info("학습 완료.")
    wlogger.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
