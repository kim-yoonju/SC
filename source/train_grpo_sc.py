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
       b. 각 rollout에 대해 greedy completion (rethink 금지) → 개별 outcome_i 측정
       c. cls_model이 각 rollout inference 평가 → G개 cls_output + old_log_probs 수집
       d. (α>0이면) PRM API로 각 rollout reward 계산
       e. rollout 중 랜덤 1개 선택 → history 추가 후 다음 step 진행
  5. GRPO 업데이트
     - advantage: trajectory 내 전체 rollout에 대해 cross-normalization
     - reward_i = α·PRM_i + β·outcome_i  (각 rollout의 개별 outcome 사용)

실행:
  CUDA_VISIBLE_DEVICES=4,5 python source/train_grpo_sc.py
"""

import copy
import datetime
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

# --gpus must be applied before CUDA initializes (torch import triggers it)
_gpus_idx = next((i for i, a in enumerate(sys.argv) if a == "--gpus"), None)
if _gpus_idx is not None and _gpus_idx + 1 < len(sys.argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_gpus_idx + 1]

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / "utils"))

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
INF_GPU_COUNT   = _SC.get("inf_gpu_count", max(1, len(_SC.get("train_gpus", [0])) // 2))
ROLLOUT_N       = _SC.get("rollout_n", 4)
LR              = float(_SC.get("lr", 1e-6))
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
MIN_RECORDS          = _SC.get("min_records_per_update", 64)
MAX_COMPLETION_STEPS = _SC.get("max_completion_steps", 30)  # rollout 완성용 최대 추가 스텝
WANDB_PROJECT   = _SC.get("wandb_project", "sc-grpo")
PROBLEM_BATCH_SIZE = _SC.get("problem_batch_size", 64)
MAX_GEN_BATCH_SIZE = _SC.get("max_gen_batch_size", 256)
RESUME          = _SC.get("resume_from")
RL_DATA         = str(_ROOT / CONF["data_path"]["rl_data"])
CKPT_BASE       = str(_ROOT / "checkpoints" / "grpo_sc")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEBUG = False


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
def _first_device(model) -> torch.device:
    return next(model.parameters()).device


def _load_model(ckpt: str, gpu_ids: list[int], trainable: bool = False) -> AutoModelForCausalLM:
    """gpu_ids에 해당하는 GPU들에 모델을 pipeline parallel로 분산 로드."""
    max_memory = {i: "40GiB" for i in gpu_ids}
    kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
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
    # inf_gpu_count개 GPU → base_model, 나머지 → cls_model
    n_inf = min(INF_GPU_COUNT, len(TRAIN_GPUS) - 1) or 1
    inf_gpus = TRAIN_GPUS[:n_inf]
    cls_gpus = TRAIN_GPUS[n_inf:] or TRAIN_GPUS[-1:]

    log.info(f"base_model : {INF_CKPT} → cuda:{inf_gpus} (frozen)")
    tokenizer  = setup_tokenizer(INF_CKPT, cache_dir=CACHE_DIR)
    base_model = _load_model(INF_CKPT, inf_gpus, trainable=False)
    base_model.resize_token_embeddings(len(tokenizer))

    log.info(f"cls_model  : {CLS_CKPT} → cuda:{cls_gpus} (trainable)")
    cls_model = _load_model(CLS_CKPT, cls_gpus, trainable=True)
    cls_model.resize_token_embeddings(len(tokenizer))

    log.info(f"ref_cls    : deepcopy of cls_model → cuda:{cls_gpus} (frozen)")
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
            pid = d["id"]
            if pid not in seen:
                seen.add(pid)
                problems.append({
                    "problem_id":  pid,
                    "problem":     d["problem"],
                    "gold_answer": d["answer"],
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
                  max_new: int, temperature: float) -> tuple[str, list[int]]:
    """단일 inference 생성 (greedy or sampled, no grad)."""
    device = _first_device(model)
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
                        max_new: int) -> tuple[list[str], list[list[int]]]:
    """rethink rollout: 동일 prompt에서 n개 inference 생성 (no grad)."""
    device = _first_device(model)
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
def generate_does(model, tokenizer, inference: str, system_summary: str) -> str:
    """base_model으로 step 한 줄 요약 생성 (history context용)."""
    prompt_ids = _tokenize(tokenizer, system_summary, inference)
    text, _ = generate_step(model, tokenizer, prompt_ids, SUMMARY_MAX_NEW, 0.0)
    return text.strip()


@torch.no_grad()
def generate_cls_output(model, tokenizer, prompt_ids: list[int],
                        max_new: int) -> tuple[str, list[int]]:
    """cls_model greedy 생성 (rollout 단계, no grad)."""
    device = _first_device(model)
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        inp,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    resp_ids = out[0, len(prompt_ids):].tolist()
    return tokenizer.decode(resp_ids, skip_special_tokens=False), resp_ids


@torch.no_grad()
def generate_batched(
    model, tokenizer,
    prompt_ids_list: list[list[int]],
    max_new: int,
    temperature: float,
) -> list[tuple[str, list[int]]]:
    """여러 프롬프트를 left-padding으로 묶어 배치 생성."""
    if not prompt_ids_list:
        return []
    device = _first_device(model)
    do_sample = temperature > 0
    results = []

    for start in range(0, len(prompt_ids_list), MAX_GEN_BATCH_SIZE):
        sub = prompt_ids_list[start : start + MAX_GEN_BATCH_SIZE]
        max_len = max(len(p) for p in sub)
        pad_id  = tokenizer.pad_token_id

        input_ids      = torch.full((len(sub), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(sub), max_len,           dtype=torch.long, device=device)
        for i, p in enumerate(sub):
            offset = max_len - len(p)
            input_ids[i, offset:]      = torch.tensor(p, dtype=torch.long, device=device)
            attention_mask[i, offset:] = 1

        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        for i in range(len(sub)):
            resp = out[i, max_len:].tolist()
            # strip trailing pad/eos
            while resp and resp[-1] in (pad_id, tokenizer.eos_token_id):
                resp.pop()
            results.append((tokenizer.decode(resp, skip_special_tokens=False), resp))
    return results


@torch.no_grad()
def generate_does_batched(model, tokenizer, inferences: list[str], system_summary: str) -> list[str]:
    """여러 inference에 대해 does를 배치 생성."""
    prompts = [_tokenize(tokenizer, system_summary, inf) for inf in inferences]
    results = generate_batched(model, tokenizer, prompts, SUMMARY_MAX_NEW, 0.0)
    return [text.strip() for text, _ in results]


@torch.no_grad()
def complete_rollout_greedy(
    problem: str,
    gold_answer: str,
    history_before_rethink: list[dict],
    rollout_inference: str,
    base_model, tokenizer,
    system_inf: str, system_summary: str,
) -> float:
    """rethink rollout 이후 base_model만으로 greedy completion → outcome(0/1).

    cls 호출 없이 MAX_COMPLETION_STEPS 이내에서 boxed answer 감지로 판정.
    does는 다음 스텝 context를 위해 정상 생성.
    """
    does = generate_does(base_model, tokenizer, rollout_inference, system_summary)
    local_history = history_before_rethink + [
        {"inference": rollout_inference, "does": does, "is_error": False}
    ]

    for _ in range(MAX_COMPLETION_STEPS):
        sys_i, usr_i = build_messages_inference(
            problem, local_history, len(local_history), system_inf
        )
        prompt_inf = _tokenize(tokenizer, sys_i, usr_i)
        inference, _ = generate_step(base_model, tokenizer, prompt_inf, INF_MAX_NEW, 0.0)

        does = generate_does(base_model, tokenizer, inference, system_summary)
        local_history.append({"inference": inference, "does": does, "is_error": False})

        if extract_boxed(inference) is not None:
            return 1.0 if check_solved(inference, gold_answer, problem=problem) else 0.0

    final = local_history[-1]["inference"]
    return 1.0 if check_solved(final, gold_answer, problem=problem) else 0.0


def cls_forward_logprobs(model, prompt_ids: list[int], response_ids: list[int],
                         no_grad: bool = False) -> torch.Tensor:
    """prompt_ids 조건부 response_ids의 token-level log probs.

    no_grad=False: 학습 forward (gradient 추적, GRPO 업데이트용).
    no_grad=True : rollout / ref 계산 (gradient 불필요).
    반환 텐서는 모델의 마지막 레이어 device에 위치.
    """
    P, R = len(prompt_ids), len(response_ids)
    device = _first_device(model)
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
    resp_ids_t  = torch.tensor(response_ids, dtype=torch.long, device=resp_logits.device)
    return F.log_softmax(resp_logits, dim=-1).gather(-1, resp_ids_t.unsqueeze(-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Action parsing
# ─────────────────────────────────────────────────────────────────────────────
_FAIL_RB_RE      = re.compile(r"Fail rubrics:\n(.*?)(?=\n\n|\Z)", re.DOTALL)
_DEEP_CRITIC_RE  = re.compile(r"Deep critic:\s*\n(.*?)(?:\n\n|\Z)", re.DOTALL)
_RUBRIC_TOKENS   = set(CONF["model"].get("special_tokens", []))


def _first_line(text: str, maxlen: int = 120) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s[:maxlen] + ("…" if len(s) > maxlen else "")
    return ""


def _deep_critic_line(cls_out: str, maxlen: int = 150) -> str:
    m = _DEEP_CRITIC_RE.search(cls_out)
    if not m:
        return "(none)"
    for line in m.group(1).splitlines():
        s = line.strip()
        if s:
            return s[:maxlen] + ("…" if len(s) > maxlen else "")
    return "(none)"


def _debug_print_trajectory(
    problem: str,
    gold_answer: str,
    step_infos: list,
    extracted,
    outcome: float,
    rewards_list: list,
    loss_val: float,
):
    W = 72
    print(f"\n{'='*W}")
    print(f"Problem : {problem[:200]}")
    print(f"Gold    : {gold_answer}")

    traj_groups: list[str] = []  # e.g. "G+_00 G_01 G_02 G_03"

    for info in step_infos:
        print(f"{'-'*W}")
        print(f"[step {info['step']}] → {info['action']}")
        print(f"  Inf   : {info['inf_line']}")
        print(f"  Critic: {info['deep_critic']}")
        print(f"  Fails : {info['fail_rubrics']}")

        if info.get("rollouts") is not None:
            r_idx = info["rethink_idx"]
            print(f"  Rollouts (rethink #{r_idx + 1}):")
            parts: list[str] = []
            for n, roll in enumerate(info["rollouts"]):
                marker = "G+" if roll["best"] else " G"
                label  = f"{marker}_{r_idx}{n}"
                outcome_mark = "✓" if roll["outcome"] == 1.0 else "✗"
                print(f"    {label}: {roll['text']}  (outcome={outcome_mark} PRM={roll['prm']:.3f})")
                parts.append(f"{'G+' if roll['best'] else 'G'}_{r_idx}{n}")
            traj_groups.append(" ".join(parts))

    print(f"{'-'*W}")
    last_action = step_infos[-1]["action"] if step_infos else ""
    terminal = "END" if last_action == TOKEN_END else ("SOLVE" if last_action == TOKEN_SOLVE else "TIMEOUT")
    traj_str  = " | ".join(traj_groups) if traj_groups else "(no rethinks)"
    outcome_str = "CORRECT ✓" if outcome == 1.0 else "WRONG ✗"

    _action_abbr = {TOKEN_END: "end", TOKEN_SOLVE: "solve", TOKEN_RETHINK: "rethink"}
    step_seq = " → ".join(
        f"{_action_abbr.get(s['action'], s['action'])}"
        + (f"[{','.join(r.strip('<|>') for r in s['fail_rubrics'])}]" if s['fail_rubrics'] else "")
        for s in step_infos
    )
    print(f"Steps      : {step_seq}")
    print(f"Trajectory : {traj_str} → {terminal}")
    print(f"Outcome    : {outcome_str}  (extracted={extracted}, gold={gold_answer})")
    if rewards_list:
        print(f"Rewards    : {[round(r, 4) for r in rewards_list]}")
    print(f"Loss       : {loss_val:.6f}")
    print("=" * W, flush=True)


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
) -> float:
    """
    rethink_records: 배치 내 모든 trajectory의 rollout 데이터.
      각 항목: {prompt_ids, response_ids, old_log_probs, prm_reward, outcome}

    reward_i  = α·prm_i + β·outcome_i  (rollout별 개별 outcome)
    advantage = 배치 전체에 대해 cross-normalization
    """
    if not rethink_records:
        return 0.0

    rewards = torch.tensor(
        [PRM_COEF * r["prm_reward"] + OUTCOME_COEF * r["outcome"]
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
        policy_lp = cls_forward_logprobs(
            cls_model, rec["prompt_ids"], rec["response_ids"], no_grad=False
        )
        ref_lp = cls_forward_logprobs(
            ref_cls, rec["prompt_ids"], rec["response_ids"], no_grad=True
        )
        old_lp = torch.tensor(rec["old_log_probs"]).to(policy_lp.device)

        ratio      = (policy_lp - old_lp).exp()
        clip_ratio = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        pg_loss    = -torch.min(ratio * advantage, clip_ratio * advantage).mean()
        kl_loss    = (policy_lp - ref_lp.to(policy_lp.device)).mean()

        step_loss = (pg_loss + KL_COEF * kl_loss) / n
        step_loss.backward()
        total_loss += step_loss.item()

    torch.nn.utils.clip_grad_norm_(cls_model.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Batched trajectory runner
# ─────────────────────────────────────────────────────────────────────────────
def _process_rethinks_batched(
    rethink_batch: list,   # list of (state_dict, inf_text, step_info)
    base_model, cls_model, tokenizer,
    system_inf: str, system_rethink: str, system_cls: str, system_summary: str,
) -> None:
    """rethink 발생 states에 대해 rollout + completion + cls 평가를 배치 처리."""
    R = len(rethink_batch)

    # ── 6a. 배치 rollout 생성 ─────────────────────────────────────────────
    rollout_prompts = []
    for s, inf, step_info in rethink_batch:
        prompt = _tokenize(tokenizer, *build_messages_inference(
            s["problem"], s["history"], len(s["history"]), system_rethink
        ))
        rollout_prompts.extend([prompt] * ROLLOUT_N)

    rollout_results  = generate_batched(base_model, tokenizer, rollout_prompts, INF_MAX_NEW, RETHINK_TEMP)
    rollout_texts    = [t for t, _ in rollout_results]
    rollout_ids_list = [ids for _, ids in rollout_results]
    # layout: [s0_r0, s0_r1, ..., s0_rN, s1_r0, ...]

    # ── 6b. 배치 completion ───────────────────────────────────────────────
    # initial does for all rollout inferences
    init_does = generate_does_batched(base_model, tokenizer, rollout_texts, system_summary)

    total = R * ROLLOUT_N
    local_histories = []
    gold_answers    = []
    problems        = []
    for idx, (s, inf, step_info) in enumerate(rethink_batch):
        for j in range(ROLLOUT_N):
            fi = idx * ROLLOUT_N + j
            local_histories.append(
                s["history"] + [{"inference": rollout_texts[fi], "does": init_does[fi], "is_error": False}]
            )
            gold_answers.append(s["gold_answer"])
            problems.append(s["problem"])

    done_flags = [False] * total
    outcomes   = [None]  * total

    for _ in range(MAX_COMPLETION_STEPS):
        active = [i for i, d in enumerate(done_flags) if not d]
        if not active:
            break

        inf_prompts = [
            _tokenize(tokenizer, *build_messages_inference(
                problems[i], local_histories[i], len(local_histories[i]), system_inf
            ))
            for i in active
        ]
        inf_results = generate_batched(base_model, tokenizer, inf_prompts, INF_MAX_NEW, 0.0)
        active_texts = [t for t, _ in inf_results]

        does_list = generate_does_batched(base_model, tokenizer, active_texts, system_summary)

        for ri, (rollout_idx, text, does) in enumerate(zip(active, active_texts, does_list)):
            local_histories[rollout_idx].append({"inference": text, "does": does, "is_error": False})
            if extract_boxed(text) is not None:
                outcomes[rollout_idx]   = 1.0 if check_solved(text, gold_answers[rollout_idx], problem=problems[rollout_idx]) else 0.0
                done_flags[rollout_idx] = True

    for i in range(total):
        if outcomes[i] is None:
            final = local_histories[i][-1]["inference"]
            outcomes[i] = 1.0 if check_solved(final, gold_answers[i], problem=problems[i]) else 0.0

    # ── 6c. 배치 cls 평가 (old_log_probs 수집) ───────────────────────────
    cls_prompts = []
    for idx, (s, inf, step_info) in enumerate(rethink_batch):
        for j in range(ROLLOUT_N):
            fi = idx * ROLLOUT_N + j
            r_steps = s["history"] + [{"inference": rollout_texts[fi], "is_error": False}]
            cls_prompts.append(_tokenize(tokenizer, *build_messages_classification(
                s["problem"], r_steps, len(s["history"]), system_cls
            )))

    cls_results = generate_batched(cls_model, tokenizer, cls_prompts, CLS_MAX_NEW, 0.0)

    for fi, ((text, ids), prompt) in enumerate(zip(cls_results, cls_prompts)):
        old_lps = cls_forward_logprobs(cls_model, prompt, ids, no_grad=True).tolist() if ids else []
        idx = fi // ROLLOUT_N
        s, inf, step_info = rethink_batch[idx]
        s["rethink_records"].append({
            "prompt_ids":    prompt,
            "response_ids":  ids,
            "old_log_probs": old_lps,
            "prm_reward":    0.0,
            "outcome":       outcomes[fi],
        })

    # ── 6d. 랜덤 rollout 선택 → history 업데이트 ─────────────────────────
    for idx, (s, inf, step_info) in enumerate(rethink_batch):
        s["n_rethinks"] += 1
        step_info["rethink_idx"] = s["n_rethinks"] - 1

        best_j   = random.randrange(ROLLOUT_N)
        best_fi  = idx * ROLLOUT_N + best_j
        best_inf = rollout_texts[best_fi]
        best_does = init_does[best_fi]
        s["history"].append({"inference": best_inf, "does": best_does, "is_error": False})

        rollout_outcome_list = [outcomes[idx * ROLLOUT_N + j] for j in range(ROLLOUT_N)]
        # local_histories[fi] 길이에서 rethink 이전 history 길이(line 697로 +1된 것 보정)를 빼면
        # 해당 rollout이 rethink 이후 정답에 도달하기까지 걸린 스텝 수
        base_len = len(s["history"]) - 1
        rollout_step_list = [
            len(local_histories[idx * ROLLOUT_N + j]) - base_len
            for j in range(ROLLOUT_N)
        ]
        step_info["rollouts"] = [
            {
                "text":    _first_line(rollout_texts[idx * ROLLOUT_N + j]),
                "prm":     0.0,
                "outcome": rollout_outcome_list[j],
                "n_steps": rollout_step_list[j],
                "best":    j == best_j,
            }
            for j in range(ROLLOUT_N)
        ]
        s["step_infos"].append(step_info)

        if DEBUG:
            tags = "  ".join(
                f"R{j}{'*' if j == best_j else ''}:"
                f"{'✓' if rollout_outcome_list[j] == 1.0 else '✗'}"
                f"({rollout_step_list[j]})"
                for j in range(ROLLOUT_N)
            )
            print(f"{s['prob_idx']}_{step_info['step']} rethink  {tags}", flush=True)


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _finalize_state(s: dict, all_path: Path) -> None:
    """outcome 계산 후 traj_all.jsonl에 기록."""
    final = s["final_text"]
    s["outcome"]   = 1.0 if (final and check_solved(final, s["gold_answer"], problem=s.get("problem", ""))) else 0.0
    s["extracted"] = extract_boxed(final) if final else None
    s["n_steps"]   = len(s["history"])
    _append_jsonl(all_path, {
        "problem_id":      s["prob_idx"],
        "problem":         s["problem"],
        "gold_answer":     s["gold_answer"],
        "outcome":         s["outcome"],
        "n_rethinks":      s["n_rethinks"],
        "n_steps":         s["n_steps"],
        "steps":           s["step_infos"],
        "rethink_records": s["rethink_records"],
    })


def _record_step(s: dict, step_info: dict, cache_path: Path) -> None:
    """스텝 완료 시 traj_cache.jsonl에 기록."""
    _append_jsonl(cache_path, {
        "problem_id":       s["prob_idx"],
        "step_idx":         step_info["step"],
        "action":           step_info["action"],
        "inf_line":         step_info["inf_line"],
        "fail_rubrics":     step_info["fail_rubrics"],
        "rollout_outcomes": [r["outcome"] for r in (step_info["rollouts"] or [])],
    })


def generate_trajectories_pool(
    all_problems: list[dict],
    base_model, cls_model, tokenizer,
    out_dir: Path,
):
    """Pool 크기를 PROBLEM_BATCH_SIZE로 유지하며 trajectory 생성 (generator).

    - 완료된 trajectory가 생기면 즉시 yield → 큐에서 새 문제 보충
    - 각 스텝 완료 시 traj_cache.jsonl 기록
    - trajectory 완료 + outcome 측정 시 traj_all.jsonl 기록
    """
    system_inf     = PROMPTS.get("gen_inference",        PROMPTS.get("system_solve", ""))
    system_rethink = PROMPTS.get("gen_rethink_inference", "")
    system_cls     = PROMPTS.get("gen_classification",   "")
    system_summary = PROMPTS.get("step_summary_system",  "")

    cache_path = out_dir / "traj_cache.jsonl"
    all_path   = out_dir / "traj_all.jsonl"

    def _new_state(p: dict) -> dict:
        return {
            "prob_idx":        p.get("problem_id", "?"),
            "problem":         p["problem"],
            "gold_answer":     p["gold_answer"],
            "history":         [],
            "rethink_records": [],
            "step_infos":      [],
            "n_rethinks":      0,
            "final_text":      "",
            "done":            False,
            "step_count":      0,
        }

    queue: list[dict] = list(all_problems)
    pool:  list[dict] = [_new_state(queue.pop(0))
                         for _ in range(min(PROBLEM_BATCH_SIZE, len(queue)))]
    if not pool and queue:
        pool.append(_new_state(queue.pop(0)))

    action_short = {TOKEN_END: "end", TOKEN_SOLVE: "solve", TOKEN_RETHINK: "rethink"}

    while pool:
        active = [s for s in pool if not s["done"]]
        if not active:
            break

        # ── 1. Batch inference ─────────────────────────────────────────────
        inf_prompts = [
            _tokenize(tokenizer, *build_messages_inference(
                s["problem"], s["history"], len(s["history"]), system_inf
            ))
            for s in active
        ]
        inf_results = generate_batched(base_model, tokenizer, inf_prompts, INF_MAX_NEW, 0.0)
        inferences  = [t for t, _ in inf_results]

        # ── 2. Batch cls evaluation ────────────────────────────────────────
        cls_prompts = [
            _tokenize(tokenizer, *build_messages_classification(
                s["problem"],
                s["history"] + [{"inference": inf, "is_error": False}],
                len(s["history"]),
                system_cls,
            ))
            for s, inf in zip(active, inferences)
        ]
        cls_results = generate_batched(cls_model, tokenizer, cls_prompts, CLS_MAX_NEW, 0.0)
        cls_outputs = [t for t, _ in cls_results]

        # ── 3. Parse & categorize ──────────────────────────────────────────
        end_batch, solve_batch, rethink_batch = [], [], []
        for s, inf, cls_out in zip(active, inferences, cls_outputs):
            fail_rubrics, action = parse_action(cls_out, inf)
            step_info = {
                "step":         s["step_count"],
                "action":       action,
                "inf_line":     _first_line(inf),
                "deep_critic":  _deep_critic_line(cls_out),
                "fail_rubrics": fail_rubrics,
                "rollouts":     None,
            }
            if action == TOKEN_END:
                end_batch.append((s, inf, step_info))
            elif action == TOKEN_SOLVE:
                solve_batch.append((s, inf, step_info))
            else:
                rethink_batch.append((s, inf, step_info))

        if DEBUG:
            for s, inf, step_info in end_batch + solve_batch:
                print(f"{s['prob_idx']}_{s['step_count']} {action_short.get(step_info['action'], step_info['action'])}", flush=True)
            # rethink는 rollout 결과와 함께 _process_rethinks_batched에서 출력

        # ── 4. END ────────────────────────────────────────────────────────
        if end_batch:
            does_list = generate_does_batched(base_model, tokenizer,
                                              [inf for s, inf, _ in end_batch], system_summary)
            for (s, inf, step_info), does in zip(end_batch, does_list):
                s["final_text"] = inf
                s["history"].append({"inference": inf, "does": does, "is_error": False})
                s["step_infos"].append(step_info)
                s["step_count"] += 1
                s["done"] = True
                _record_step(s, step_info, cache_path)

        # ── 5. SOLVE ──────────────────────────────────────────────────────
        if solve_batch:
            does_list = generate_does_batched(base_model, tokenizer,
                                              [inf for s, inf, _ in solve_batch], system_summary)
            for (s, inf, step_info), does in zip(solve_batch, does_list):
                s["history"].append({"inference": inf, "does": does, "is_error": False})
                s["step_infos"].append(step_info)
                s["step_count"] += 1
                _record_step(s, step_info, cache_path)

        # ── 6. RETHINK ────────────────────────────────────────────────────
        if rethink_batch:
            _process_rethinks_batched(
                rethink_batch, base_model, cls_model, tokenizer,
                system_inf, system_rethink, system_cls, system_summary,
            )
            for s, inf, step_info in rethink_batch:
                s["step_count"] += 1
                _record_step(s, step_info, cache_path)

        # ── 7. MAX_STEPS 초과 강제 종료 ───────────────────────────────────
        for s in active:
            if not s["done"] and s["step_count"] >= MAX_STEPS:
                s["final_text"] = s["history"][-1]["inference"] if s["history"] else ""
                s["done"] = True

        # ── 8. 완료된 문제 처리 → traj_all 기록, 큐에서 보충 ──────────────
        still_active = []
        for s in pool:
            if s["done"]:
                _finalize_state(s, all_path)
                yield s
                if queue:
                    still_active.append(_new_state(queue.pop(0)))
            else:
                still_active.append(s)
        pool = still_active


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
    p.add_argument("--gpus",               default=None)
    p.add_argument("--inf_checkpoint",     default=None)
    p.add_argument("--cls_checkpoint",     default=None)
    p.add_argument("--resume_from",        default=None)
    p.add_argument("--prm_coef",           type=float, default=None)
    p.add_argument("--outcome_coef",       type=float, default=None)
    p.add_argument("--min_records",        type=int,   default=None)
    p.add_argument("--inf_gpu_count",      type=int,   default=None)
    p.add_argument("--problem_batch_size", type=int,   default=None)
    p.add_argument("--max_gen_batch_size", type=int,   default=None)
    p.add_argument("--debug", action="store_true",
                   help="스텝별 live 출력 (문제번호_스텝번호 + rethink rollout 결과)")
    args, _ = p.parse_known_args()

    global INF_CKPT, CLS_CKPT, RESUME, PRM_COEF, OUTCOME_COEF, TRAIN_GPUS, DEBUG
    global INF_GPU_COUNT, MIN_RECORDS, PROBLEM_BATCH_SIZE, MAX_GEN_BATCH_SIZE
    if args.gpus:
        TRAIN_GPUS = list(range(len(args.gpus.split(","))))
    if args.inf_checkpoint:              INF_CKPT       = args.inf_checkpoint
    if args.cls_checkpoint:              CLS_CKPT       = args.cls_checkpoint
    if args.resume_from:                 RESUME         = args.resume_from
    if args.prm_coef        is not None: PRM_COEF       = args.prm_coef
    if args.outcome_coef    is not None: OUTCOME_COEF   = args.outcome_coef
    if args.min_records     is not None: MIN_RECORDS    = args.min_records
    if args.inf_gpu_count   is not None: INF_GPU_COUNT  = args.inf_gpu_count
    if args.problem_batch_size is not None: PROBLEM_BATCH_SIZE = args.problem_batch_size
    if args.max_gen_batch_size is not None: MAX_GEN_BATCH_SIZE = args.max_gen_batch_size
    if args.debug:
        DEBUG = True


def main():
    _parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    base_model, cls_model, ref_cls, tokenizer = setup_models_and_tokenizer()
    problems = load_problems()
    log.info(
        f"Loaded {len(problems)} problems | "
        f"ROLLOUT_N={ROLLOUT_N} MIN_RECORDS={MIN_RECORDS} "
        f"PRM_COEF={PRM_COEF} OUTCOME_COEF={OUTCOME_COEF}"
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

    # 출력 디렉터리
    out_dir = _ROOT / "output" / "GRPO" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {out_dir}")

    # records가 MIN_RECORDS 이상 쌓이면 한 번 학습 (1 iteration)
    all_records:   list[dict]  = []
    iter_outcomes: list[float] = []
    iter_rethinks: list[int]   = []
    iter_steps:    list[int]   = []
    n_trajs = 0

    for s in generate_trajectories_pool(
        problems[start_idx:end_idx], base_model, cls_model, tokenizer, out_dir
    ):
        all_records.extend(s["rethink_records"])
        iter_outcomes.append(s["outcome"])
        iter_rethinks.append(s["n_rethinks"])
        iter_steps.append(s["n_steps"])
        n_trajs += 1

        if len(all_records) < MIN_RECORDS:
            continue

        # GRPO update
        loss_val     = grpo_update(cls_model, ref_cls, optimizer, all_records)
        global_step += 1

        avg_outcome  = sum(iter_outcomes) / len(iter_outcomes)
        avg_rethinks = sum(iter_rethinks) / len(iter_rethinks)
        avg_steps    = sum(iter_steps)    / len(iter_steps)
        n_records    = len(all_records)

        log.info(
            f"[{global_step}] loss={loss_val:.4f}  outcome={avg_outcome:.2f}  "
            f"rethinks={avg_rethinks:.1f}  traj_steps={avg_steps:.1f}  "
            f"records={n_records}  trajs={n_trajs}"
        )
        if wandb_run:
            wandb_run.log({
                "loss": loss_val, "outcome": avg_outcome,
                "n_rethinks": avg_rethinks, "n_steps": avg_steps,
                "n_records": n_records, "n_trajs": n_trajs,
                "global_step": global_step,
            })
        save_checkpoint(cls_model, tokenizer, optimizer, global_step, ts)

        all_records = []; iter_outcomes = []; iter_rethinks = []; iter_steps = []
        n_trajs = 0

    if wandb_run:
        wandb_run.finish()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
