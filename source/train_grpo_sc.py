"""
train_grpo_sc.py — GRPO training for SC classification model

Policy (학습 대상): cls_model — step 평가 + next action 결정
Frozen: base_model (추론 생성), ref_cls_model (KL 기준)

알고리즘:
  1. base_model이 problem → step inference 생성 (greedy)
  2. cls_model이 inference 평가 → cls_output (fast/deep critic + fail rubrics)
  3. next action 결정:
       - fail rubrics 있음              → <|rethink|>
       - fail rubrics 없고 boxed 있음   → <|end|>
       - fail rubrics 없고 boxed 없음   → <|solve|>
  4. <|rethink|>일 때만 GRPO rollout:
       a. base_model이 G개 rethought inference 생성 (temperature > 0)
       b. cls_model이 각 inference 평가 → G개 cls_output + old_log_probs 수집
       c. (α>0이면) PRM API로 각 rollout reward 계산
       d. 가장 좋은 rollout → history 추가 후 다음 step 진행
  5. trajectory 종료 → outcome reward 판정 → GRPO 업데이트
     - advantage: trajectory 내 전체 rollout에 대해 cross-normalization
     - reward_i = α·PRM_i + β·outcome  (α=0이면 PRM API 호출 없음)

실행:
  CUDA_VISIBLE_DEVICES=4,5 python source/train_grpo_sc.py
"""

import copy
import datetime
import json
import logging
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils_sft import (
    CONF, setup_tokenizer,
    build_messages_inference, build_messages_classification, build_chat_prompt,
    TOKEN_SOLVE, TOKEN_RETHINK, TOKEN_END,
)
from utils_math import check_solved, extract_boxed

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
_SC = CONF.get("grpo_sc", {})

INF_CKPT        = (_SC.get("inf_checkpoint") or
                   CONF["checkpoint"].get("sft_checkpoint") or
                   CONF["checkpoint"]["base"])
CLS_CKPT        = (_SC.get("cls_checkpoint") or
                   CONF["checkpoint"].get("sft_checkpoint") or
                   CONF["checkpoint"]["base"])
CACHE_DIR       = CONF["checkpoint"].get("cache_dir")

TRAIN_GPUS      = _SC.get("train_gpus", [0])
ROLLOUT_N       = _SC.get("rollout_n", 4)
LR              = _SC.get("lr", 1e-6)
KL_COEF         = _SC.get("kl_coef", 0.04)
CLIP_EPS        = _SC.get("clip_eps", 0.2)
GRAD_CLIP       = _SC.get("grad_clip", 1.0)
PRM_COEF        = float(_SC.get("prm_coef", 0.0))      # α
OUTCOME_COEF    = float(_SC.get("outcome_coef", 1.0))  # β
MAX_STEPS       = _SC.get("max_steps_per_problem", 20)
INF_MAX_NEW     = _SC.get("inf_max_new_tokens", 1024)
CLS_MAX_NEW     = _SC.get("cls_max_new_tokens", 512)
SUMMARY_MAX_NEW = _SC.get("summary_max_new_tokens", 128)
RETHINK_TEMP    = _SC.get("rethink_temperature", 0.7)
MAX_SEQ_LEN     = _SC.get("max_seq_len", 4096)
TOTAL_PROBLEMS  = _SC.get("total_problems", 5000)
SAVE_STEPS      = _SC.get("save_steps", 100)
WANDB_PROJECT   = _SC.get("wandb_project", "sc-grpo")
RESUME          = _SC.get("resume_from")
RL_DATA         = str(_ROOT / CONF["data_path"]["rl_data"])
CKPT_BASE       = str(_ROOT / "checkpoints" / "grpo_sc")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────
def _load_prompts() -> dict[str, str]:
    path = _ROOT / CONF["prompts"]["file"]
    return {d["name"]: d["content"] for d in json.loads(path.read_text())}

PROMPTS = _load_prompts()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_model(ckpt: str, device: str, trainable: bool = False) -> AutoModelForCausalLM:
    kwargs = dict(torch_dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True)
    if ckpt.startswith("/") or ckpt.startswith("."):
        kwargs["local_files_only"] = True
    else:
        kwargs["cache_dir"] = CACHE_DIR
    model = AutoModelForCausalLM.from_pretrained(ckpt, **kwargs)
    if trainable:
        model.train()
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def setup_models_and_tokenizer():
    dev_base = f"cuda:{TRAIN_GPUS[0]}"
    dev_cls  = f"cuda:{TRAIN_GPUS[-1]}"

    log.info(f"base_model : {INF_CKPT} → {dev_base} (frozen)")
    tokenizer  = setup_tokenizer(INF_CKPT, cache_dir=CACHE_DIR)
    base_model = _load_model(INF_CKPT, dev_base, trainable=False)
    base_model.resize_token_embeddings(len(tokenizer))

    log.info(f"cls_model  : {CLS_CKPT} → {dev_cls} (trainable)")
    cls_model = _load_model(CLS_CKPT, dev_cls, trainable=True)
    cls_model.resize_token_embeddings(len(tokenizer))

    log.info(f"ref_cls    : deepcopy of cls_model → {dev_cls} (frozen)")
    ref_cls = copy.deepcopy(cls_model)
    ref_cls.eval()
    for p in ref_cls.parameters():
        p.requires_grad_(False)

    return base_model, cls_model, ref_cls, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Problem loading
# ─────────────────────────────────────────────────────────────────────────────
def load_problems() -> list[dict]:
    problems, seen = [], set()
    with open(RL_DATA, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pid = d["problem_id"]
            if pid not in seen:
                seen.add(pid)
                problems.append({
                    "problem_id":  pid,
                    "problem":     d["problem"],
                    "gold_answer": d["gold_answer"],
                })
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# Prompt building
# ─────────────────────────────────────────────────────────────────────────────
def _tokenize(tokenizer, system: str, user: str) -> list[int]:
    text = build_chat_prompt(tokenizer, system, user)
    return tokenizer.encode(text, add_special_tokens=False)


# ─────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_step(model, tokenizer, prompt_ids: list[int],
                  max_new: int, temperature: float, device: str) -> tuple[str, list[int]]:
    """단일 inference 생성 (greedy or sampled, no grad)."""
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    do_sample = temperature > 0
    out = model.generate(
        inp,
        max_new_tokens=max_new,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    resp_ids = out[0, len(prompt_ids):].tolist()
    return tokenizer.decode(resp_ids, skip_special_tokens=False), resp_ids


@torch.no_grad()
def generate_step_batch(model, tokenizer, prompt_ids: list[int],
                        n: int, temperature: float,
                        max_new: int, device: str) -> tuple[list[str], list[list[int]]]:
    """rethink rollout: 동일 prompt에서 n개 inference 생성 (no grad)."""
    inp = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)
    out = model.generate(
        inp,
        max_new_tokens=max_new,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    texts, ids_list = [], []
    for i in range(n):
        resp = out[i, len(prompt_ids):].tolist()
        ids_list.append(resp)
        texts.append(tokenizer.decode(resp, skip_special_tokens=False))
    return texts, ids_list


@torch.no_grad()
def generate_does(model, tokenizer, inference: str,
                  system_summary: str, device: str) -> str:
    """base_model으로 step 한 줄 요약 생성 (history context용)."""
    prompt_ids = _tokenize(tokenizer, system_summary, inference)
    text, _ = generate_step(model, tokenizer, prompt_ids, SUMMARY_MAX_NEW, 0.0, device)
    return text.strip()


@torch.no_grad()
def generate_cls_output(model, tokenizer, prompt_ids: list[int],
                        max_new: int, device: str) -> tuple[str, list[int]]:
    """cls_model greedy 생성 (rollout 단계, no grad)."""
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        inp,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    resp_ids = out[0, len(prompt_ids):].tolist()
    return tokenizer.decode(resp_ids, skip_special_tokens=False), resp_ids


def cls_forward_logprobs(model, prompt_ids: list[int], response_ids: list[int],
                         device: str, no_grad: bool = False) -> torch.Tensor:
    """prompt_ids 조건부 response_ids의 token-level log probs.

    no_grad=False: 학습 forward (gradient 추적, GRPO 업데이트용).
    no_grad=True : rollout / ref 계산 (gradient 불필요).
    """
    P, R = len(prompt_ids), len(response_ids)
    if R == 0:
        return torch.zeros(0, device=device)

    input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)

    if no_grad:
        with torch.no_grad():
            logits = model(input_ids).logits[0]
    else:
        logits = model(input_ids).logits[0]

    # logits[P-1 .. P+R-2] → distribution for response tokens [0..R-1]
    resp_logits = logits[P - 1 : P + R - 1].float()
    resp_ids_t  = torch.tensor(response_ids, dtype=torch.long, device=device)
    return F.log_softmax(resp_logits, dim=-1).gather(-1, resp_ids_t.unsqueeze(-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Action parsing
# ─────────────────────────────────────────────────────────────────────────────
_FAIL_RB_RE    = re.compile(r"Fail rubrics:\n(.*?)(?=\n\n|\Z)", re.DOTALL)
_RUBRIC_TOKENS = set(CONF["model"].get("special_tokens", []))


def parse_action(cls_output: str, inference: str) -> tuple[list[str], str]:
    """cls_output과 inference에서 (fail_rubrics, action_token) 결정.

    규칙:
      fail_rubrics 있음             → <|rethink|>
      fail_rubrics 없음 + boxed     → <|end|>
      fail_rubrics 없음 + no boxed  → <|solve|>
    """
    fail_rubrics: list[str] = []
    m = _FAIL_RB_RE.search(cls_output)
    if m:
        for tok in m.group(1).strip().splitlines():
            tok = tok.strip()
            if tok and tok in _RUBRIC_TOKENS:
                fail_rubrics.append(tok)

    has_boxed = extract_boxed(inference) is not None

    if fail_rubrics:
        return fail_rubrics, TOKEN_RETHINK
    if has_boxed:
        return fail_rubrics, TOKEN_END
    return fail_rubrics, TOKEN_SOLVE


# ─────────────────────────────────────────────────────────────────────────────
# PRM reward  (PRM_COEF > 0 일 때만 호출)
# ─────────────────────────────────────────────────────────────────────────────
_prm_batch_inst = None


def _get_prm_batch():
    global _prm_batch_inst
    if _prm_batch_inst is None:
        from PRM import ApiPrmBatch, load_fast_rubric
        prm_cfg   = CONF.get("PRM", {})
        fast_path = prm_cfg.get("fast_rubric", "")
        if not Path(fast_path).is_absolute():
            fast_path = str(_ROOT / fast_path)
        fast_rubric   = load_fast_rubric(Path(fast_path))
        _prm_batch_inst = ApiPrmBatch(prm_cfg["model_id"], fast_rubric, max_workers=8)
    return _prm_batch_inst


def compute_prm_rewards(problem: str, history: list[dict],
                        inferences: list[str]) -> list[float]:
    """G개 inference의 PRM stage-1 reward 계산.
    PRM_COEF == 0 이면 API 호출 없이 [0.0, ...] 반환.
    reward = (pass 루브릭 수) / (전체 루브릭 수)  ∈ [0, 1]
    """
    if PRM_COEF == 0.0:
        return [0.0] * len(inferences)

    prm = _get_prm_batch()
    prev_text = " ".join(s.get("does") or s.get("inference", "") for s in history)
    rewards: list[float] = []
    for inf in inferences:
        try:
            results  = prm.evaluate_batch(
                questions=[problem], prev_steps=[prev_text],
                now_steps=[inf], max_new_tokens=512,
            )
            n_pass   = sum(1 for r in results[0] if r.get("pred") == "correct")
            rewards.append(n_pass / max(len(results[0]), 1))
        except Exception as e:
            log.warning(f"PRM call failed: {e}")
            rewards.append(0.0)
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# GRPO update
# ─────────────────────────────────────────────────────────────────────────────
def grpo_update(
    cls_model,
    ref_cls,
    optimizer: torch.optim.Optimizer,
    rethink_records: list[dict],
    outcome_reward: float,
    device: str,
) -> float:
    """
    rethink_records: trajectory 내 모든 rollout 데이터.
      각 항목: {prompt_ids, response_ids, old_log_probs, prm_reward}

    reward_i  = α·prm_i + β·outcome
    advantage = cross-rethink normalization (trajectory 내 전체 G×K 샘플)

    note: PRM_COEF=0이면 모든 reward가 동일해 advantage=0 → gradient 없음.
          PRM을 쓰거나 trajectory별 다른 outcome을 사용해야 학습 신호가 생김.
    """
    if not rethink_records:
        return 0.0

    rewards = torch.tensor(
        [PRM_COEF * r["prm_reward"] + OUTCOME_COEF * outcome_reward
         for r in rethink_records],
        dtype=torch.float32,
    )

    if rewards.std() < 1e-8:
        log.debug("All rewards identical — no gradient")
        return 0.0

    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss = 0.0
    n = len(rethink_records)

    for idx, rec in enumerate(rethink_records):
        if not rec["response_ids"]:
            continue

        advantage = advantages[idx].item()
        old_lp    = torch.tensor(rec["old_log_probs"], device=device)

        policy_lp = cls_forward_logprobs(
            cls_model, rec["prompt_ids"], rec["response_ids"], device, no_grad=False
        )
        ref_lp = cls_forward_logprobs(
            ref_cls, rec["prompt_ids"], rec["response_ids"], device, no_grad=True
        )

        ratio      = (policy_lp - old_lp).exp()
        clip_ratio = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        pg_loss    = -torch.min(ratio * advantage, clip_ratio * advantage).mean()
        kl_loss    = (policy_lp - ref_lp.to(device)).mean()

        # /n: 각 record별 backward로 gradient 누적, graph 즉시 해제 (메모리 효율)
        step_loss = (pg_loss + KL_COEF * kl_loss) / n
        step_loss.backward()
        total_loss += step_loss.item()

    torch.nn.utils.clip_grad_norm_(cls_model.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory runner
# ─────────────────────────────────────────────────────────────────────────────
def run_trajectory(
    problem_item: dict,
    base_model,
    cls_model,
    ref_cls,
    tokenizer,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """한 problem에 대한 trajectory 생성 + GRPO 업데이트.
    반환: {loss, outcome, n_rethinks, n_steps}
    """
    problem     = problem_item["problem"]
    gold_answer = problem_item["gold_answer"]

    dev_base = next(base_model.parameters()).device
    dev_cls  = next(cls_model.parameters()).device

    system_inf     = PROMPTS.get("gen_inference",       PROMPTS.get("system_solve", ""))
    system_rethink = PROMPTS.get("gen_rethink_inference", "")
    system_cls     = PROMPTS.get("gen_classification",  "")
    system_summary = PROMPTS.get("step_summary_system", "")

    history: list[dict]         = []  # {inference, does, is_error=False}
    rethink_records: list[dict] = []
    n_rethinks  = 0
    final_text  = ""

    optimizer.zero_grad()

    for _step in range(MAX_STEPS):
        # ── 1. base_model: inference 생성 (greedy) ─────────────────────────
        sys_i, usr_i   = build_messages_inference(problem, history, len(history), system_inf)
        prompt_inf     = _tokenize(tokenizer, sys_i, usr_i)
        inference, _   = generate_step(base_model, tokenizer, prompt_inf,
                                       INF_MAX_NEW, 0.0, dev_base)

        # ── 2. cls_model: inference 평가 ────────────────────────────────────
        tmp_steps      = history + [{"inference": inference, "is_error": False}]
        sys_c, usr_c   = build_messages_classification(problem, tmp_steps, len(history), system_cls)
        prompt_cls     = _tokenize(tokenizer, sys_c, usr_c)
        cls_out, _     = generate_cls_output(cls_model, tokenizer, prompt_cls, CLS_MAX_NEW, dev_cls)

        fail_rubrics, action = parse_action(cls_out, inference)
        log.debug(f"  step={_step} action={action} n_fails={len(fail_rubrics)}")

        # ── 3. action 분기 ──────────────────────────────────────────────────
        if action == TOKEN_END:
            final_text = inference
            does = generate_does(base_model, tokenizer, inference, system_summary, dev_base)
            history.append({"inference": inference, "does": does, "is_error": False})
            break

        if action == TOKEN_SOLVE:
            does = generate_does(base_model, tokenizer, inference, system_summary, dev_base)
            history.append({"inference": inference, "does": does, "is_error": False})
            continue

        # ── 4. RETHINK: G개 rollout 생성 + cls_model 평가 ──────────────────
        n_rethinks += 1

        sys_r, usr_r   = build_messages_inference(problem, history, len(history), system_rethink)
        prompt_ret     = _tokenize(tokenizer, sys_r, usr_r)
        rollout_texts, rollout_ids = generate_step_batch(
            base_model, tokenizer, prompt_ret, ROLLOUT_N, RETHINK_TEMP, INF_MAX_NEW, dev_base,
        )

        prm_rewards = compute_prm_rewards(problem, history, rollout_texts)

        for r_inf, r_ids, prm_r in zip(rollout_texts, rollout_ids, prm_rewards):
            r_steps  = history + [{"inference": r_inf, "is_error": False}]
            s_c, u_c = build_messages_classification(problem, r_steps, len(history), system_cls)
            r_prompt = _tokenize(tokenizer, s_c, u_c)

            r_cls_out, r_cls_ids = generate_cls_output(
                cls_model, tokenizer, r_prompt, CLS_MAX_NEW, dev_cls
            )
            old_lps = cls_forward_logprobs(
                cls_model, r_prompt, r_cls_ids, dev_cls, no_grad=True
            ).tolist()

            rethink_records.append({
                "prompt_ids":    r_prompt,
                "response_ids":  r_cls_ids,
                "old_log_probs": old_lps,
                "prm_reward":    prm_r,
            })

        # 가장 좋은 rollout 선택 (PRM score 기준, PRM_COEF=0이면 첫 번째)
        best_idx = int(torch.tensor(prm_rewards).argmax()) if any(r > 0 for r in prm_rewards) else 0
        best_inf = rollout_texts[best_idx]
        does     = generate_does(base_model, tokenizer, best_inf, system_summary, dev_base)
        history.append({"inference": best_inf, "does": does, "is_error": False})

    else:
        final_text = history[-1]["inference"] if history else ""

    # ── 5. outcome reward + GRPO 업데이트 ───────────────────────────────────
    outcome  = 1.0 if (final_text and check_solved(final_text, gold_answer)) else 0.0
    loss_val = grpo_update(cls_model, ref_cls, optimizer,
                           rethink_records, outcome, str(dev_cls))

    return {"loss": loss_val, "outcome": outcome,
            "n_rethinks": n_rethinks, "n_steps": len(history)}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(cls_model, tokenizer, optimizer, step: int, ts: str):
    save_dir = Path(CKPT_BASE) / ts / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    cls_model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
    log.info(f"Checkpoint saved: {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    """config.yaml 값을 CLI에서 선택적으로 override."""
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--inf_checkpoint",  default=None)
    p.add_argument("--cls_checkpoint",  default=None)
    p.add_argument("--resume_from",     default=None)
    p.add_argument("--prm_coef",        type=float, default=None)
    p.add_argument("--outcome_coef",    type=float, default=None)
    args, _ = p.parse_known_args()

    global INF_CKPT, CLS_CKPT, RESUME, PRM_COEF, OUTCOME_COEF
    if args.inf_checkpoint:  INF_CKPT     = args.inf_checkpoint
    if args.cls_checkpoint:  CLS_CKPT     = args.cls_checkpoint
    if args.resume_from:     RESUME       = args.resume_from
    if args.prm_coef   is not None: PRM_COEF     = args.prm_coef
    if args.outcome_coef is not None: OUTCOME_COEF = args.outcome_coef


def main():
    _parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    base_model, cls_model, ref_cls, tokenizer = setup_models_and_tokenizer()
    problems = load_problems()
    log.info(
        f"Loaded {len(problems)} problems | "
        f"ROLLOUT_N={ROLLOUT_N} PRM_COEF={PRM_COEF} OUTCOME_COEF={OUTCOME_COEF}"
    )

    optimizer = torch.optim.AdamW(
        [p for p in cls_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(project=WANDB_PROJECT, name=ts, config=dict(_SC))
    except Exception:
        log.warning("W&B 초기화 실패 — 로컬 로그만 사용")

    global_step = 0
    start_idx   = 0

    if RESUME:
        ckpt_path = Path(RESUME)
        opt_path  = ckpt_path / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        if ckpt_path.name.startswith("step_"):
            start_idx   = int(ckpt_path.name.split("_")[1])
            global_step = start_idx
        log.info(f"Resumed from {RESUME}, step={global_step}")

    end_idx = min(start_idx + TOTAL_PROBLEMS, len(problems))

    for prob in problems[start_idx:end_idx]:
        result      = run_trajectory(prob, base_model, cls_model, ref_cls, tokenizer, optimizer)
        global_step += 1

        if global_step % 10 == 0:
            log.info(
                f"[{global_step}] loss={result['loss']:.4f}  "
                f"outcome={result['outcome']:.0f}  "
                f"rethinks={result['n_rethinks']}  "
                f"traj_steps={result['n_steps']}"
            )

        if wandb_run:
            wandb_run.log({**result, "global_step": global_step})

        if global_step % SAVE_STEPS == 0:
            save_checkpoint(cls_model, tokenizer, optimizer, global_step, ts)

    save_checkpoint(cls_model, tokenizer, optimizer, global_step, ts)
    if wandb_run:
        wandb_run.finish()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
