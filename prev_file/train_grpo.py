"""
train_grpo.py — On-Policy GRPO Training Loop

데이터 생성: generate_trajectory.py의 generate_batch() 직접 사용
학습: TRAIN_BATCH개 trajectory 누적 후 GRPO update

per-iteration:
  1. generate_batch()로 N_PROBLEMS개 trajectory 생성 (vLLM + PRM)
  2. is_right 기반 reward 계산
  3. 배치 내 GRPO advantage: (r - mean) / std
  4. 각 step에 대해 GRPO policy gradient + KL update
  5. vLLM weights 동기화 → 반복

GPU 구성 (config.grpo):
  rollout_gpus: [A, B]  # A = step_manager, B = vLLM generator
  train_gpus:   [C]     # GRPO 학습 전용

Usage:
    python source/train_grpo.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent
_ROOT    = _SRC_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

from utils import CONF, GENERATOR_MODEL_ID, SFT_CHECKPOINT

# ─────────────────────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────────────────────

_grpo = CONF["grpo"]

GRPO_LR         = _grpo["lr"]
GRPO_CLIP_EPS   = _grpo["clip_eps"]
GRPO_MAX_GRAD_N = _grpo["max_grad_norm"]
KL_COEF         = _grpo["kl_coef"]
MAX_SEQ_LEN     = _grpo["max_seq_len"]
MAX_ITERATIONS  = _grpo["max_iterations"]
ROLLOUT_GPUS    = _grpo.get("rollout_gpus", _grpo["train_gpus"][:1])
TRAIN_GPUS      = _grpo["train_gpus"]
TRAINER_GPU     = TRAIN_GPUS[-1]
RESUME          = _grpo.get("resume_checkpoint")

N_PROBLEMS      = _grpo.get("problems_per_gpu", 32)   # 이터당 병렬 처리 문제 수

RL_DATA_PATH    = str(_ROOT / CONF["data_path"]["rl_data"])
CKPT_BASE       = str(_ROOT / "checkpoints" / "grpo")
LOG_BASE        = str(_ROOT / "output" / "train_grpo")


def _load_system_prompt() -> str:
    """generate_trajectory.py와 동일한 방식으로 gen_solve_R 프롬프트 로드."""
    import json as _json
    prm_cfg      = CONF.get("PRM", {})
    prompts_cfg  = CONF.get("prompts", {})
    prompts_file = _ROOT / prompts_cfg.get("file", "prompts/action_prompts.json")
    rubric_rel   = prm_cfg.get("rubric") or prompts_cfg.get("rubric_file")
    if not rubric_rel:
        raise KeyError("config.PRM.rubric 설정이 없습니다")
    rubric_file  = Path(rubric_rel) if Path(rubric_rel).is_absolute() else _ROOT / rubric_rel
    rubric_lines = []
    with open(rubric_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = _json.loads(line)
                rubric_lines.append(f'{e["name"]}: [correct/incorrect — {e["criterion"]}]')
    rubric_str = "\n".join(rubric_lines)
    prompts = {p["name"]: p["content"] for p in _json.load(open(prompts_file))}
    return prompts["gen_solve_R"].replace("{{rubric}}", rubric_str)


def _load_problems() -> list[dict]:
    problems, seen = [], set()
    if RL_DATA_PATH.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(RL_DATA_PATH)
        for _, row in df.iterrows():
            pid  = str(row.get("problem_id", row.get("id", _)))
            prob = row.get("problem", row.get("question", ""))
            gold = row.get("gold_answer", row.get("answer", row.get("final_answer", "")))
            if not gold and isinstance(row.get("reward_model"), dict):
                gold = row["reward_model"].get("ground_truth", "")
            if not gold and isinstance(row.get("extra_info"), dict):
                gold = row["extra_info"].get("gold_answer", "")
            if not prob and row.get("prompt"):
                prompt = row["prompt"]
                if isinstance(prompt, list) and prompt:
                    prob = prompt[-1].get("content", "")
            if pid not in seen and prob:
                seen.add(pid)
                problems.append({"problem_id": pid, "problem": prob, "gold_answer": str(gold)})
    else:
        with open(RL_DATA_PATH, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                pid = d["problem_id"]
                if pid not in seen:
                    seen.add(pid)
                    problems.append({"problem_id": pid, "problem": d["problem"], "gold_answer": d["gold_answer"]})
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Worker  (ROLLOUT_GPUS: [step_manager, vLLM])
# generate_trajectory.py의 generate_batch() 직접 사용
# ─────────────────────────────────────────────────────────────────────────────

import ray


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸리티 (RolloutWorker / Trainer 양쪽에서 사용)
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt_ids(tokenizer, system_prompt: str, problem: str, history: list[str]) -> list[int]:
    step_k = len(history)
    lines  = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Previous steps]")
        for idx, h in enumerate(history, 1):
            lines.append(f"Step {idx}: {h}")
    lines.append(f"\nWrite Step {step_k + 1}.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "\n".join(lines)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def _extract_raw_samples(rollout_groups: list[dict]) -> list[dict]:
    """rollout_groups → [{problem, history, response_text}, ...].

    _compute_advantages와 동일한 순서로 순회해서 인덱스가 1:1로 매칭됨.
    """
    raw = []
    for group in rollout_groups:
        problem = group.get("problem", "")
        for rollout in group["rollouts"]:
            history = []
            for step in rollout["steps"]:
                if step.get("is_error"):
                    history = []
                    continue
                response_text = step.get("inference") or step.get("text", "")
                raw.append({
                    "problem":       problem,
                    "history":       list(history),
                    "response_text": response_text,
                })
                does = step.get("does") or response_text[:80]
                history.append(does)
    return raw


@ray.remote
class GRPORolloutWorker:
    """generate_batch()로 trajectory를 생성하는 Ray Actor.

    ROLLOUT_N번 generate_batch를 호출해 문제당 ROLLOUT_N개 trajectory 수집.
    CUDA_VISIBLE_DEVICES = "A,B" (A=step_manager GPU, B=vLLM GPU) 로 실행.
    """

    def __init__(self, log_path: str):
        import os, sys, logging as _log
        from pathlib import Path as _P

        _src = str(_P(__file__).resolve().parent)
        sys.path.insert(0, _src)

        # ── CUDA 초기화 전 가장 먼저 GPU 설정 ─────────────────────────────────
        # Ray 상속 CUDA_VISIBLE_DEVICES를 덮어쓰기 (CUDA lazy init 이용)
        from utils import CONF as _conf
        _rollout_gpus = _conf["grpo"].get("rollout_gpus", [0, 0])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in _rollout_gpus)
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

        _log.basicConfig(
            level=_log.INFO,
            format="%(asctime)s [RolloutWorker] %(message)s",
            handlers=[
                _log.FileHandler(log_path, encoding="utf-8"),
                _log.StreamHandler(),
            ],
        )

        from utils import load_generator_vllm, load_step_manager, GENERATOR_MODEL_ID, SFT_CHECKPOINT
        from PRM import ApiPrm, ApiPrmTwoStage, load_rubrics, load_fast_rubric

        _root    = _P(__file__).resolve().parent.parent
        gt_cfg   = _conf["generate_trajectory"]
        prm_cfg  = _conf["PRM"]
        rollout_gpus = _rollout_gpus

        model_path = SFT_CHECKPOINT or GENERATOR_MODEL_ID

        # GPU 0 (= 물리 rollout_gpus[0]) 에 step_manager 로드
        self.sm_model, self.sm_tok = load_step_manager(gpu_id=0)
        _log.getLogger().info(f"Step Manager 로드 완료 (physical GPU {rollout_gpus[0]})")

        # vLLM은 rollout_gpus[-1] 만 보이도록 제한 후 로드
        gen_gpu = rollout_gpus[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gen_gpu)
        self.llm, self.tokenizer = load_generator_vllm(
            model_path=model_path, rollout_gpus=[0]
        )
        self.generators = [(self.llm, self.tokenizer, None)]
        _log.getLogger().info(f"vLLM Generator 로드 완료 (physical GPU {gen_gpu})")

        # Rubrics & PRM
        rubric_path = str(_root / prm_cfg["rubric"])
        self.rubrics = load_rubrics(rubric_path)
        fast_rubric_path = prm_cfg.get("fast_rubric")
        if fast_rubric_path:
            if not _P(fast_rubric_path).is_absolute():
                fast_rubric_path = str(_root / fast_rubric_path)
            fast_rubric = load_fast_rubric(_P(fast_rubric_path))
            self.prm_model = ApiPrmTwoStage(
                prm_cfg["model_id"], fast_rubric, self.rubrics,
                max_workers=prm_cfg.get("batch_per_gpu", 8),
            )
        else:
            self.prm_model = ApiPrm(
                prm_cfg["model_id"],
                max_workers=prm_cfg.get("batch_per_gpu", 8),
            )

        self.n_parallel = gt_cfg.get("batch_per_gpu", 4)
        _log.getLogger().info("GRPORolloutWorker 초기화 완료")

    def generate_rollouts(self, problems: list[dict], traj_save_path: str) -> tuple[list[dict], list[str]]:
        """generate_batch()로 trajectory 생성 → GRPO group format 반환.

        trajectory 완성 즉시 traj_save_path에 JSONL 저장 + 터미널 요약 출력.

        Returns:
            (groups, summaries)
            groups:    [{problem_id, gold_answer, problem, rollouts:[{steps, reward}]}]
            summaries: 각 trajectory 요약 문자열 리스트 (main에서 출력)
        """
        import json as _json
        import logging as _log

        from generate_trajectory import generate_batch

        items = [
            {"id": p["problem_id"], "problem": p["problem"], "gold_answer": p["gold_answer"]}
            for p in problems
        ]

        groups: dict[str, dict] = {}
        summaries: list[str] = []
        _traj_file = open(traj_save_path, "a", encoding="utf-8")

        def _fmt_traj(traj: dict) -> str:
            mark = "✓" if traj.get("is_right") else "✗"
            pid  = traj["problem_id"]
            parts = []
            for i, s in enumerate(traj.get("steps", []), 1):
                src = s.get("source", "gen")
                if src == "patcher":
                    parts.append(f"P*_{i:02d}")
                elif s.get("state") == "rethink":
                    parts.append(f"G+_{i:02d}")
                else:
                    parts.append(f"G_{i:02d}")
            result = "correct" if traj.get("is_right") else "wrong"
            return f"{mark}  [id={pid}]  {'  '.join(parts)}  → {result} ({traj.get('traj_type','?')})"

        def _save_fn(traj: dict):
            _traj_file.write(_json.dumps(traj, ensure_ascii=False) + "\n")
            _traj_file.flush()
            summary = _fmt_traj(traj)
            summaries.append(summary)
            _log.getLogger("rollout").info("\n" + summary)

            pid = traj["problem_id"]
            if pid not in groups:
                groups[pid] = {
                    "problem_id": pid,
                    "gold_answer": traj["gold_answer"],
                    "problem":     traj["problem"],
                    "rollouts":    [],
                }
            groups[pid]["rollouts"].append({
                "steps":  traj["steps"],
                "reward": 1.0 if traj.get("is_right") else 0.0,
            })

        generate_batch(
            items=items,
            generators=self.generators,
            prm_model=self.prm_model,
            rubrics=self.rubrics,
            n_parallel=N_PROBLEMS,
            save_fn=_save_fn,
            step_manager_model=self.sm_model,
            step_manager_tok=self.sm_tok,
        )

        _traj_file.close()
        return list(groups.values()), summaries

    def compute_ref_logprobs(self, raw_samples: list[dict], system_prompt: str) -> list[list[float]]:
        """step_manager(= ref model)로 각 step의 log prob 계산.

        _extract_raw_samples와 동일한 순서의 raw_samples를 받아,
        trainer.update()에 넘길 ref log prob 리스트를 반환.
        빈 response나 MAX_SEQ_LEN 초과 시 해당 인덱스는 빈 리스트로 채움.
        """
        import torch as _t
        import torch.nn.functional as _F

        device = next(self.sm_model.parameters()).device
        results: list[list[float]] = []

        for sample in raw_samples:
            prompt_ids   = _build_prompt_ids(self.sm_tok, system_prompt, sample["problem"], sample["history"])
            response_ids = self.sm_tok(sample["response_text"], add_special_tokens=False)["input_ids"]
            if not response_ids:
                results.append([])
                continue

            inp_ids  = _t.tensor([prompt_ids],   dtype=_t.long, device=device)
            resp_ids = _t.tensor([response_ids], dtype=_t.long, device=device)

            max_inp = MAX_SEQ_LEN - resp_ids.shape[1]
            if max_inp <= 0:
                results.append([])
                continue
            if inp_ids.shape[1] > max_inp:
                inp_ids = inp_ids[:, -max_inp:]

            split = inp_ids.shape[1] - 1
            with _t.no_grad():
                pfx_kv = (
                    self.sm_model(inp_ids[:, :split], use_cache=True).past_key_values
                    if split > 0 else None
                )
                grad_input = _t.cat([inp_ids[:, -1:], resp_ids], dim=1)
                logits     = self.sm_model(grad_input, past_key_values=pfx_kv).logits
                lp = (
                    _F.log_softmax(logits[:, :-1, :], dim=-1)
                    .gather(-1, resp_ids.unsqueeze(-1))
                    .squeeze(-1).squeeze(0)
                )
            results.append(lp.cpu().tolist())

        return results

    def load_state_dict(self, state_dict: dict):
        """학습된 weight를 vLLM LLM 내부 모델에 직접 주입."""
        import torch as _t

        model = None
        for path in [
            lambda: self.llm.llm_engine.model_executor.driver_worker.model_runner.model,
            lambda: self.llm.llm_engine.driver_worker.model_runner.model,
        ]:
            try:
                model = path()
                break
            except AttributeError:
                continue

        if model is None:
            import logging as _log
            _log.getLogger().warning("weight sync 경로를 찾지 못했습니다. 이전 weights 사용.")
            return

        for name, tensor in state_dict.items():
            try:
                p = model.get_parameter(name)
                if p.shape == tensor.shape:
                    p.data.copy_(tensor.to(p.device))
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# GRPO Trainer
# ─────────────────────────────────────────────────────────────────────────────

class GRPOTrainer:
    """GRPO advantage + policy gradient + KL penalty.

    generate_batch() 출력(text 기반 steps)을 받아 on-policy 업데이트.
    log_probs는 현재 모델로 실시간 계산 (PPO clip 대신 순수 policy gradient).
    """

    def __init__(self, device: str, system_prompt: str, resume_checkpoint: str | None = None):
        self.device        = device
        self.system_prompt = system_prompt

        from utils import load_generator
        model_path = resume_checkpoint or SFT_CHECKPOINT or GENERATOR_MODEL_ID
        self.model, self.tokenizer = load_generator(
            device_map={"": device}, model_path=model_path
        )
        self.model.train()

        import bitsandbytes as bnb
        self.optimizer = bnb.optim.AdamW8bit(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(GRPO_LR),
        )
        logging.info(f"GRPOTrainer 초기화 완료 ({device})")

    # ── 프롬프트 재구성 + 토크나이징 ─────────────────────────────────────────

    def _build_prompt_ids(self, problem: str, history: list[str]) -> list[int]:
        return _build_prompt_ids(self.tokenizer, self.system_prompt, problem, history)

    # ── GRPO advantage 계산 (문제 그룹 내) ──────────────────────────────────

    @staticmethod
    def _compute_advantages(rollout_groups: list[dict]) -> list[dict]:
        """배치 전체 기준 (r - mean) / std — 문제당 rollout이 1개이므로 batch-level 정규화."""
        raw: list[dict] = []
        for group in rollout_groups:
            problem = group.get("problem", "")
            for rollout in group["rollouts"]:
                reward  = rollout["reward"]
                history = []
                for step in rollout["steps"]:
                    if step.get("is_error"):
                        history = []
                        continue
                    response_text = step.get("inference") or step.get("text", "")
                    raw.append({
                        "problem":       problem,
                        "history":       list(history),
                        "response_text": response_text,
                        "reward":        reward,
                    })
                    does = step.get("does") or response_text[:80]
                    history.append(does)

        if not raw:
            return []
        rewards = [s["reward"] for s in raw]
        mean_r  = sum(rewards) / len(rewards)
        std_r   = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
        for s in raw:
            s["advantage"] = (s.pop("reward") - mean_r) / std_r
        return raw

    # ── GRPO update ──────────────────────────────────────────────────────────

    def update(self, rollout_groups: list[dict], ref_logprobs: list[list[float]]) -> dict:
        """GRPO 업데이트.

        ref_logprobs: _extract_raw_samples와 동일 순서의 step별 log prob 리스트.
                      GRPORolloutWorker.compute_ref_logprobs()의 반환값.
        """
        samples = self._compute_advantages(rollout_groups)
        if not samples:
            return {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0}

        self.model.train()
        self.optimizer.zero_grad()

        n = len(samples)
        total_pg = total_kl = 0.0

        for i, sample in enumerate(samples):
            lp_ref_list = ref_logprobs[i] if i < len(ref_logprobs) else []
            if not lp_ref_list:
                continue

            prompt_ids   = self._build_prompt_ids(sample["problem"], sample["history"])
            response_ids = self.tokenizer(
                sample["response_text"], add_special_tokens=False
            )["input_ids"]
            if not response_ids:
                continue

            inp_ids  = torch.tensor([prompt_ids],   dtype=torch.long, device=self.device)
            resp_ids = torch.tensor([response_ids], dtype=torch.long, device=self.device)
            advantage = float(sample["advantage"])

            max_inp = MAX_SEQ_LEN - resp_ids.shape[1]
            if max_inp <= 0:
                continue
            if inp_ids.shape[1] > max_inp:
                inp_ids = inp_ids[:, -max_inp:]

            split = inp_ids.shape[1] - 1
            with torch.no_grad():
                pfx_kv = (
                    self.model(inp_ids[:, :split], use_cache=True).past_key_values
                    if split > 0 else None
                )
            grad_input = torch.cat([inp_ids[:, -1:], resp_ids], dim=1)
            logits = self.model(grad_input, past_key_values=pfx_kv).logits
            lp_new = (
                F.log_softmax(logits[:, :-1, :], dim=-1)
                .gather(-1, resp_ids.unsqueeze(-1))
                .squeeze(-1).squeeze(0)
            )

            lp_ref = torch.tensor(lp_ref_list, dtype=torch.float32, device=self.device)
            # ref_logprobs는 truncation 전 기준일 수 있으므로 길이 맞춤
            min_len = min(lp_new.shape[0], lp_ref.shape[0])
            lp_new  = lp_new[:min_len]
            lp_ref  = lp_ref[:min_len]

            # On-policy policy gradient (PPO clip 없음 — ratio = 1)
            pg_loss = -(lp_new * advantage).mean()
            kl      = (torch.exp(lp_new) * (lp_new - lp_ref)).mean()

            ((pg_loss + KL_COEF * kl) / n).backward()

            total_pg += pg_loss.item()
            total_kl += kl.item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRPO_MAX_GRAD_N)
        self.optimizer.step()

        return {
            "loss":    (total_pg + KL_COEF * total_kl) / n,
            "pg_loss": total_pg / n,
            "kl":      total_kl / n,
        }

    def get_state_dict(self) -> dict:
        vocab_size = self.model.config.vocab_size
        return {
            k: v.cpu() for k, v in self.model.state_dict().items()
            if not (v.shape[0] != vocab_size and any(t in k for t in ("embed_tokens", "lm_head")))
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = os.path.join(LOG_BASE, ts)
    ckpt_dir = os.path.join(CKPT_BASE, ts)
    os.makedirs(run_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Ray init 전에는 CUDA_VISIBLE_DEVICES를 건드리지 않음
    # (worker 프로세스가 CUDA init 전에 직접 설정)
    ray.init(include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)

    system_prompt = _load_system_prompt()
    problems      = _load_problems()
    logger.info(f"문제 {len(problems)}개 로드  |  ts={ts}")
    logger.info(
        f"N_PROBLEMS={N_PROBLEMS}  rollout_gpus={ROLLOUT_GPUS}  train_gpu={TRAINER_GPU}"
    )

    # ── Rollout Worker (ROLLOUT_GPUS) ────────────────────────────────────────
    # GPU 설정은 worker __init__ 내부에서 CUDA init 전에 직접 수행
    worker = GRPORolloutWorker.options(
        runtime_env={"env_vars": {
            "VLLM_WORKER_MULTIPROC_METHOD": "fork",
            "VLLM_ATTENTION_BACKEND":       "XFORMERS",
        }}
    ).remote(log_path=os.path.join(run_dir, "worker.log"))
    logger.info(f"GRPORolloutWorker → GPUs {ROLLOUT_GPUS}")

    # ── Trainer (TRAINER_GPU) ────────────────────────────────────────────────
    # worker 생성 후 main process GPU 제한 (CUDA lazy init 활용)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(TRAINER_GPU)
    trainer = GRPOTrainer(
        device="cuda:0",
        system_prompt=system_prompt,
        resume_checkpoint=RESUME,
    )
    logger.info(f"GRPOTrainer → GPU {TRAINER_GPU}")

    traj_jsonl = os.path.join(run_dir, "traj_live.jsonl")   # 실시간 저장 파일
    cursor = 0
    n      = len(problems)

    for iteration in tqdm(range(MAX_ITERATIONS), desc="GRPO", unit="iter"):

        # 문제 샘플링 (순환)
        batch = [(problems + problems)[cursor % n + i] for i in range(N_PROBLEMS)]
        cursor += N_PROBLEMS

        # generate_batch()로 N_PROBLEMS개 병렬 생성, 완성 즉시 JSONL 저장
        try:
            rollout_groups, summaries = ray.get(
                worker.generate_rollouts.remote(batch, traj_jsonl)
            )
        except Exception as e:
            logger.error(f"[iter {iteration:4d}]  rollout 실패, iteration skip: {e}")
            continue

        # trajectory 요약 출력 (배치 완료 후)
        print(f"\n[iter {iteration:4d}]")
        for s in summaries:
            print(s)

        rewards   = [r["reward"] for g in rollout_groups for r in g["rollouts"]]
        n_correct = sum(1 for r in rewards if r > 0.5)
        logger.info(
            f"[iter {iteration:4d}]  trajectories={len(rewards)}  correct={n_correct}  "
            f"mean_reward={sum(rewards)/max(len(rewards),1):.3f}  "
            f"traj_jsonl={traj_jsonl}"
        )

        if not rollout_groups:
            logger.warning(f"[iter {iteration:4d}]  trajectory 없음, skip")
            continue

        # step_manager(= ref model)로 ref log prob 계산
        try:
            raw_samples  = _extract_raw_samples(rollout_groups)
            ref_logprobs = ray.get(worker.compute_ref_logprobs.remote(raw_samples, system_prompt))
        except Exception as e:
            logger.error(f"[iter {iteration:4d}]  ref_logprobs 계산 실패, skip: {e}")
            continue

        # GRPO 업데이트
        stats = trainer.update(rollout_groups, ref_logprobs)
        logger.info(
            f"           GRPO → loss={stats['loss']:.4f}  "
            f"pg={stats['pg_loss']:.4f}  kl={stats['kl']:.6f}"
        )

        # weight 동기화
        sd = trainer.get_state_dict()
        ray.get(worker.load_state_dict.remote(sd))

        # 체크포인트
        save_path = os.path.join(ckpt_dir, f"iter_{iteration:04d}")
        trainer.save(save_path)
        logger.info(f"           ckpt → {save_path}")


if __name__ == "__main__":
    main()
