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
elif "CUDA_VISIBLE_DEVICES" not in os.environ:
    # --gpus 미지정 시 config.yaml의 train_gpus로 CUDA_VISIBLE_DEVICES 설정 (torch import 전)
    try:
        import yaml as _yaml
        _cfg_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        with open(_cfg_path) as _f:
            _early_cfg = _yaml.safe_load(_f)
        _train_gpus = _early_cfg.get("grpo_sc", {}).get("train_gpus", [])
        if _train_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in _train_gpus)
    except Exception:
        pass

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

TRAIN_GPUS      = list(range(len(_SC.get("train_gpus", [0]))))  # always relative (CUDA_VISIBLE_DEVICES handles physical mapping)
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
CLS_MAX_NEW     = _SC.get("cls_max_new_tokens", 4096)
SUMMARY_MAX_NEW = _SC.get("summary_max_new_tokens", 128)
RETHINK_TEMP    = _SC.get("rethink_temperature", 1.0)
MAX_SEQ_LEN     = _SC.get("max_seq_len", 4096)
TOTAL_PROBLEMS  = _SC.get("total_problems", 5000)
MIN_RECORDS          = _SC.get("min_records_per_update", 64)
ITER_PROBLEMS        = _SC.get("iter_problems", 64)
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
        attn_implementation="sdpa",
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


def _load_inf_vllm(ckpt: str, n_inf: int):
    """vLLM으로 inference 모델 로드. 처음 n_inf개의 visible GPU를 사용."""
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    from vllm import LLM
    return LLM(
        model=ckpt,
        dtype="bfloat16",
        tensor_parallel_size=n_inf,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=True,
    )


def _load_cls_vllm(ckpt: str, gpu_id: str, all_visible: str):
    """cls generation용 vLLM. CUDA_VISIBLE_DEVICES 조작으로 특정 GPU에만 로드.

    gpu_id: physical GPU ID 문자열 (CUDA_VISIBLE_DEVICES 기준 변환 후 전달).
    vLLM worker는 별도 프로세스로 spawn되므로 spawn 직전 env 변경이 반영된다.
    main process의 CUDA 컨텍스트(이미 초기화)는 영향받지 않는다.
    """
    from vllm import LLM
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        llm = LLM(
            model=ckpt,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.70,
            trust_remote_code=True,
            enforce_eager=True,
        )
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = all_visible
    return llm


def sync_cls_llm(cls_llm, cls_model) -> None:
    """GRPO 업데이트 후 cls_model의 최신 가중치를 cls_llm(vLLM)에 반영.

    vLLM 내부 API로 in-place 업데이트를 시도한다.
    실패 시 경고만 출력하고 넘어간다 (다음 iteration에서 약간 off-policy).
    """
    try:
        state_dict = [(k, v.detach().cpu()) for k, v in cls_model.named_parameters()]
        executor = cls_llm.llm_engine.model_executor
        worker = (getattr(executor, "driver_worker", None)
                  or getattr(executor, "_driver_worker", None))
        if worker is None:
            raise AttributeError("driver_worker not found")
        worker.model_runner.model.load_weights(state_dict)
        log.info("cls_llm weights synced (in-place)")
    except Exception as e:
        log.warning(f"cls_llm weight sync failed: {e} — weights stale this iteration")


def setup_models_and_tokenizer():
    """
    GPU 레이아웃:
      TRAIN_GPUS[0 .. n_inf-1]  → inf_llm  (vLLM, frozen)
      TRAIN_GPUS[n_inf]         → cls_llm  (vLLM, generation only, synced each iter)
      TRAIN_GPUS[n_inf+1 ..]    → cls_model + ref_cls (HF, pipeline parallel, grad)
    """
    # inf_llm에 최대 n_inf개, cls_llm에 1개, 나머지 cls HF에 할당
    n_inf        = min(INF_GPU_COUNT, len(TRAIN_GPUS) - 2) or 1
    cls_vllm_idx = TRAIN_GPUS[n_inf]          # cls_llm용 GPU (relative index)
    cls_hf_gpus  = TRAIN_GPUS[n_inf + 1:] or TRAIN_GPUS[-1:]  # HF용 GPU들

    all_visible = os.environ.get("CUDA_VISIBLE_DEVICES",
                                 ",".join(str(g) for g in TRAIN_GPUS))
    # relative index → physical GPU ID (_load_cls_vllm은 CUDA_VISIBLE_DEVICES를 직접 조작하므로 physical ID 필요)
    _vis_list = [s.strip() for s in all_visible.split(",") if s.strip()]
    physical_cls_gpu = _vis_list[cls_vllm_idx] if cls_vllm_idx < len(_vis_list) else str(cls_vllm_idx)

    log.info(f"inf_llm  (vLLM): {INF_CKPT} → GPU {TRAIN_GPUS[:n_inf]}")
    tokenizer = setup_tokenizer(INF_CKPT, cache_dir=CACHE_DIR)
    inf_llm   = _load_inf_vllm(INF_CKPT, n_inf)

    log.info(f"cls_llm  (vLLM): {CLS_CKPT} → GPU {cls_vllm_idx} (physical {physical_cls_gpu})")
    cls_llm = _load_cls_vllm(CLS_CKPT, physical_cls_gpu, all_visible)

    log.info(f"cls_model  (HF): {CLS_CKPT} → GPUs {cls_hf_gpus} (trainable)")
    cls_model = _load_model(CLS_CKPT, cls_hf_gpus, trainable=True)
    cls_model.resize_token_embeddings(len(tokenizer))

    log.info(f"ref_cls    (HF): deepcopy → GPUs {cls_hf_gpus} (frozen)")
    ref_cls = copy.deepcopy(cls_model)
    ref_cls.eval()
    for p in ref_cls.parameters():
        p.requires_grad_(False)

    return inf_llm, cls_llm, cls_model, ref_cls, tokenizer


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
    """여러 프롬프트를 left-padding으로 묶어 배치 생성.

    길이 내림차순 정렬 후 처리해 padding 낭비를 최소화한다.
    반환 순서는 입력 순서와 동일하게 복원된다.
    """
    if not prompt_ids_list:
        return []
    device = _first_device(model)
    do_sample = temperature > 0

    # 길이 내림차순으로 정렬 (padding 낭비 최소화), 원래 인덱스 보존
    order = sorted(range(len(prompt_ids_list)), key=lambda i: len(prompt_ids_list[i]), reverse=True)
    sorted_prompts = [prompt_ids_list[i] for i in order]

    raw_results: list[tuple[str, list[int]]] = []
    pad_id = tokenizer.pad_token_id

    for start in range(0, len(sorted_prompts), MAX_GEN_BATCH_SIZE):
        sub = sorted_prompts[start : start + MAX_GEN_BATCH_SIZE]
        max_len = len(sub[0])  # 내림차순 정렬이므로 첫 번째가 최장
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
            while resp and resp[-1] in (pad_id, tokenizer.eos_token_id):
                resp.pop()
            raw_results.append((tokenizer.decode(resp, skip_special_tokens=False), resp))

    # 원래 순서로 복원
    results: list[tuple[str, list[int]] | None] = [None] * len(prompt_ids_list)
    for rank, orig_idx in enumerate(order):
        results[orig_idx] = raw_results[rank]
    return results  # type: ignore[return-value]


@torch.no_grad()
def generate_does_batched(model, tokenizer, inferences: list[str], system_summary: str) -> list[str]:
    """여러 inference에 대해 does를 배치 생성."""
    prompts = [_tokenize(tokenizer, system_summary, inf) for inf in inferences]
    results = generate_batched(model, tokenizer, prompts, SUMMARY_MAX_NEW, 0.0)
    return [text.strip() for text, _ in results]


# ─────────────────────────────────────────────────────────────────────────────
# vLLM generation helpers (inference model, frozen)
# ─────────────────────────────────────────────────────────────────────────────

def vllm_generate_batched(
    llm,
    prompt_ids_list: list[list[int]],
    max_new: int,
    temperature: float,
    return_logprobs: bool = False,
) -> list[tuple[str, list[int]]] | list[tuple[str, list[int], list[float]]]:
    """vLLM 배치 생성. MAX_GEN_BATCH_SIZE 단위로 청킹해 KV 캐시 과부하 방지.

    return_logprobs=True: (text, token_ids, log_probs) 반환.
    greedy (temperature=0) 생성 시 top-1 = 선택 토큰이 보장되므로 logprobs=1로 충분.
    """
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=temperature,
        max_tokens=max_new,
        skip_special_tokens=False,
        logprobs=1 if return_logprobs else None,
    )
    results: list = []
    for start in range(0, len(prompt_ids_list), MAX_GEN_BATCH_SIZE):
        chunk = prompt_ids_list[start : start + MAX_GEN_BATCH_SIZE]
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in chunk], sampling_params=sp)
        for o in outputs:
            out = o.outputs[0]
            text     = out.text
            tok_ids  = list(out.token_ids)
            if return_logprobs:
                lps = [
                    out.logprobs[i][tid].logprob
                    if (out.logprobs and i < len(out.logprobs) and tid in out.logprobs[i])
                    else 0.0
                    for i, tid in enumerate(tok_ids)
                ]
                results.append((text, tok_ids, lps))
            else:
                results.append((text, tok_ids))
    return results


def vllm_generate_does_batched(llm, tokenizer, inferences: list[str], system_summary: str) -> list[str]:
    """여러 inference에 대해 does를 vLLM으로 배치 생성."""
    prompts = [_tokenize(tokenizer, system_summary, inf) for inf in inferences]
    return [text.strip() for text, _ in vllm_generate_batched(llm, prompts, SUMMARY_MAX_NEW, 0.0)]


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
        {"inference": rollout_inference, "does": does, "is_fail": False}
    ]

    for _ in range(MAX_COMPLETION_STEPS):
        sys_i, usr_i = build_messages_inference(
            problem, local_history, len(local_history), system_inf
        )
        prompt_inf = _tokenize(tokenizer, sys_i, usr_i)
        inference, _ = generate_step(base_model, tokenizer, prompt_inf, INF_MAX_NEW, 0.0)

        does = generate_does(base_model, tokenizer, inference, system_summary)
        local_history.append({"inference": inference, "does": does, "is_fail": False})

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


@torch.no_grad()
def cls_forward_logprobs_batched(
    model,
    pairs: list[tuple[list[int], list[int]]],
    batch_size: int = 4,
) -> list[list[float]]:
    """여러 (prompt_ids, response_ids) 쌍의 log probs를 mini-batch forward로 계산.

    left-padding으로 배치를 구성해 sequential 호출 대비 ~batch_size배 빠르게 처리.
    """
    if not pairs:
        return []

    results: list[list[float]] = []
    device = _first_device(model)

    for start in range(0, len(pairs), batch_size):
        sub = pairs[start : start + batch_size]
        full_seqs = [p + r for p, r in sub]
        max_len = max(len(s) for s in full_seqs)

        input_ids      = torch.zeros(len(sub), max_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(sub), max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(full_seqs):
            offset = max_len - len(seq)
            input_ids[i, offset:]      = torch.tensor(seq, dtype=torch.long, device=device)
            attention_mask[i, offset:] = 1

        logits = model(input_ids, attention_mask=attention_mask).logits  # (B, L, V)

        for i, (prompt_ids, response_ids) in enumerate(sub):
            R = len(response_ids)
            if R == 0:
                results.append([])
                continue
            P = len(prompt_ids)
            offset = max_len - len(full_seqs[i])
            resp_logits = logits[i, offset + P - 1 : offset + P + R - 1].float()
            resp_ids_t  = torch.tensor(response_ids, dtype=torch.long, device=resp_logits.device)
            lps = F.log_softmax(resp_logits, dim=-1).gather(-1, resp_ids_t.unsqueeze(-1)).squeeze(-1)
            results.append(lps.tolist())

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Action parsing
# ─────────────────────────────────────────────────────────────────────────────
_FAIL_RB_RE      = re.compile(r"Fail rubrics:\n(.*?)(?=\n\n|\Z)", re.DOTALL)
_DEEP_CRITIC_RE  = re.compile(r"Deep critic:\s*\n(.*?)(?:\n\n|\Z)", re.DOTALL)
_RUBRIC_TOKENS   = set(CONF["model"].get("special_tokens", []))
_ACTION_TOKENS   = {TOKEN_SOLVE, TOKEN_RETHINK, TOKEN_END, "<|none|>"}
RUBRIC_ORDER     = [t for t in CONF["model"].get("special_tokens", []) if t not in _ACTION_TOKENS]

_RUBRIC_NAME_TO_TOKEN = {
    "Algebraic Manipulation":                 "<|algebraic_manipulation|>",
    "Abstract and Linear Algebra Operations": "<|abstract_and_linear_algebra_operations|>",
    "Calculus Computation":                   "<|calculus_computation|>",
    "Function and Limit Analysis":            "<|function_and_limit_analysis|>",
    "Geometric Reasoning":                    "<|geometric_reasoning|>",
    "Counting and Probability":               "<|counting_and_probability|>",
    "Number Theoretic Reasoning":             "<|number_theoretic_reasoning|>",
    "Logical and Discrete Reasoning":         "<|logical_and_discrete_reasoning|>",
    "Differential Equations":                 "<|differential_equations|>",
    "Progress and Non-Repetition":            "<|progress_and_non-repetition|>",
    "Atomicity":                              "<|atomicity|>",
}
_RUBRIC_SPLIT_RE = re.compile(
    r"\n  (" + "|".join(re.escape(n) for n in _RUBRIC_NAME_TO_TOKEN) + r"):"
)


def _rubric_scores(cls_output: str) -> list[int | None]:
    """cls_output에서 루브릭별 Verdict 파싱 → correct=0, incorrect=1, 미등장=None."""
    token_to_score: dict[str, int] = {}
    parts = _RUBRIC_SPLIT_RE.split(cls_output)
    # parts: [pre, name1, text1, name2, text2, ...]
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        text = parts[i + 1] if i + 1 < len(parts) else ""
        token = _RUBRIC_NAME_TO_TOKEN.get(name)
        if not token:
            continue
        m = re.search(r"Verdict:\s*(correct|incorrect)", text, re.IGNORECASE)
        if m:
            token_to_score[token] = 0 if m.group(1).lower() == "correct" else 1
    return [token_to_score.get(tok) for tok in RUBRIC_ORDER]


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

    cls 모델 출력 포맷: Deep critic 섹션에 각 루브릭별 "Verdict: incorrect/correct".
    "Verdict: incorrect" 루브릭을 fail_rubrics로 추출한다.
    """
    fail_rubrics: list[str] = []

    parts = _RUBRIC_SPLIT_RE.split(cls_output)
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        text = parts[i + 1] if i + 1 < len(parts) else ""
        token = _RUBRIC_NAME_TO_TOKEN.get(name)
        if not token:
            continue
        if re.search(r"Verdict:\s*incorrect", text, re.IGNORECASE):
            fail_rubrics.append(token)

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
    rethink_records: 배치 내 모든 cls 기록 (main step + rollout).
      각 항목: {prompt_ids, response_ids, old_log_probs, reward}

    reward = backward propagation으로 미리 계산된 future-value
    advantage = 배치 전체에 대해 cross-normalization
    """
    if not rethink_records:
        return 0.0

    rewards = torch.tensor(
        [r["reward"] for r in rethink_records],
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
def _fmt_step_sequence(step_infos: list[dict]) -> str:
    """step_infos → 스텝 구성 문자열.
    solve/end: G,  rethink: G+_XX(k/N)
    """
    parts = []
    for info in step_infos:
        if info["action"] == TOKEN_RETHINK:
            rollouts = info.get("rollouts") or []
            k = sum(1 for r in rollouts if r["outcome"] == 1.0)
            n = len(rollouts)
            parts.append(f"G+_{info['step']:02d}({k}/{n})")
        else:
            parts.append("G")
    return " ".join(parts)


def _process_rethinks_batched(
    rethink_batch: list,   # list of (state_dict, inf_text, step_info)
    inf_llm, cls_llm, cls_model, tokenizer,
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

    rollout_results  = vllm_generate_batched(inf_llm, rollout_prompts, INF_MAX_NEW, RETHINK_TEMP)
    rollout_texts    = [t for t, _ in rollout_results]
    rollout_ids_list = [ids for _, ids in rollout_results]
    # layout: [s0_r0, s0_r1, ..., s0_rN, s1_r0, ...]

    # ── 6b. 배치 completion ───────────────────────────────────────────────
    # initial does for all rollout inferences
    init_does = vllm_generate_does_batched(inf_llm, tokenizer, rollout_texts, system_summary)

    total = R * ROLLOUT_N
    local_histories = []
    gold_answers    = []
    problems        = []
    for idx, (s, inf, step_info) in enumerate(rethink_batch):
        for j in range(ROLLOUT_N):
            fi = idx * ROLLOUT_N + j
            local_histories.append(
                s["history"] + [{"inference": rollout_texts[fi], "does": init_does[fi], "is_fail": False}]
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
        inf_results = vllm_generate_batched(inf_llm, inf_prompts, INF_MAX_NEW, 0.0)
        active_texts = [t for t, _ in inf_results]

        does_list = vllm_generate_does_batched(inf_llm, tokenizer, active_texts, system_summary)

        for ri, (rollout_idx, text, does) in enumerate(zip(active, active_texts, does_list)):
            local_histories[rollout_idx].append({"inference": text, "does": does, "is_fail": False})
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
            r_steps = s["history"] + [{"inference": rollout_texts[fi], "is_fail": False}]
            cls_prompts.append(_tokenize(tokenizer, *build_messages_classification(
                s["problem"], r_steps, len(s["history"]), system_cls
            )))

    cls_results       = vllm_generate_batched(cls_llm, cls_prompts, CLS_MAX_NEW, 0.0, return_logprobs=True)
    rollout_cls_texts = [t   for t, _, _   in cls_results]
    rollout_old_lps   = [lps for _, _, lps in cls_results]

    all_rollout_cls_records = [None] * (len(rethink_batch) * ROLLOUT_N)
    for fi, ((text, ids, _), prompt, old_lps) in enumerate(zip(cls_results, cls_prompts, rollout_old_lps)):
        idx = fi // ROLLOUT_N
        s, inf, step_info = rethink_batch[idx]
        rec = {
            "prompt_ids":    prompt,
            "response_ids":  ids,
            "old_log_probs": old_lps,
            "prm_reward":    0.0,
            "outcome":       outcomes[fi],
        }
        s["rethink_records"].append(rec)
        all_rollout_cls_records[fi] = rec

    # ── 6d. rollout 결과 처리 → 항상 랜덤 선택 (전부 실패 시 is_fail=True로 마킹 후 계속) ──
    for idx, (s, inf, step_info) in enumerate(rethink_batch):
        s["n_rethinks"] += 1
        step_info["rethink_idx"] = s["n_rethinks"] - 1

        rollout_outcome_list = [outcomes[idx * ROLLOUT_N + j] for j in range(ROLLOUT_N)]
        base_len = len(s["history"])   # rethink rollout 추가 전 history 길이
        rollout_step_list = [
            len(local_histories[idx * ROLLOUT_N + j]) - base_len
            for j in range(ROLLOUT_N)
        ]
        rollout_recs = [all_rollout_cls_records[idx * ROLLOUT_N + j] for j in range(ROLLOUT_N)]
        rollout_avg  = sum(r["outcome"] for r in rollout_recs) / ROLLOUT_N

        # rollout 성공 여부에 관계없이 랜덤 선택 후 history에 추가 → cls가 다시 rethink 판단
        best_j  = random.randrange(ROLLOUT_N)
        best_fi = idx * ROLLOUT_N + best_j
        all_failed = rollout_avg == 0.0
        s["history"].append({
            "inference": rollout_texts[best_fi],
            "does":      init_does[best_fi],
            "is_fail":   all_failed,
        })
        if all_failed:
            s["fail_reason"] = "consecutive_all_rollouts_failed"

        step_info["rollouts"] = [
            {
                "text":       _first_line(rollout_texts[idx * ROLLOUT_N + j]),
                "cls_output": rollout_cls_texts[idx * ROLLOUT_N + j],
                "prm":        0.0,
                "outcome":    rollout_outcome_list[j],
                "n_steps":    rollout_step_list[j],
                "best":       j == best_j,
            }
            for j in range(ROLLOUT_N)
        ]
        step_info["cls_rollout_records"] = rollout_recs
        step_info["rollout_avg_outcome"] = rollout_avg
        s["step_infos"].append(step_info)


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _finalize_state(s: dict, all_path: Path, cls_all_path: Path) -> None:
    """outcome 계산 후 inference_all.jsonl과 cls_all.jsonl에 기록."""
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
        "actions":         [info["action"] for info in s["step_infos"]],
        "rethink_records": s["rethink_records"],
        "fail_reason":     s.get("fail_reason"),
    })
    _append_jsonl(cls_all_path, {
        "problem_id":  s["prob_idx"],
        "problem":     s["problem"],
        "gold_answer": s["gold_answer"],
        "outcome":     s["outcome"],
        "steps": [
            {
                "step_idx":            info["step"],
                "action":              info["action"],
                "cls_output":          info.get("cls_output", ""),
                "fail_rubrics":        info["fail_rubrics"],
                "deep_critic":         info["deep_critic"],
                "rubric_scores":       _rubric_scores(info.get("cls_output", "")),
                "rollout_cls_outputs": [r.get("cls_output", "") for r in (info.get("rollouts") or [])],
            }
            for info in s["step_infos"]
        ],
    })


def _record_step(s: dict, step_info: dict, cache_path: Path, cls_cache_path: Path) -> None:
    """스텝 완료 시 inference_cache.jsonl과 cls_cache.jsonl에 기록."""
    _append_jsonl(cache_path, {
        "problem_id": s["prob_idx"],
        "step_idx":   step_info["step"],
        "action":     step_info["action"],
    })
    _append_jsonl(cls_cache_path, {
        "problem_id":    s["prob_idx"],
        "step_idx":      step_info["step"],
        "cls_output":    step_info.get("cls_output", ""),
        "action":        step_info["action"],
        "fail_rubrics":  step_info["fail_rubrics"],
        "deep_critic":   step_info["deep_critic"],
        "rubric_scores": _rubric_scores(step_info.get("cls_output", "")),
    })


def generate_trajectories_pool(
    all_problems: list[dict],
    inf_llm, cls_llm, cls_model, tokenizer,
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

    cache_path     = out_dir / "inference_cache.jsonl"
    all_path       = out_dir / "inference_all.jsonl"
    cls_cache_path = out_dir / "cls_cache.jsonl"
    cls_all_path   = out_dir / "cls_all.jsonl"

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
            "fail_reason":     None,
        }

    queue: list[dict] = list(all_problems)
    pool:  list[dict] = [_new_state(queue.pop(0))
                         for _ in range(min(PROBLEM_BATCH_SIZE, len(queue)))]
    if not pool and queue:
        pool.append(_new_state(queue.pop(0)))

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
        inf_results = vllm_generate_batched(inf_llm, inf_prompts, INF_MAX_NEW, 0.0)
        inferences  = [t for t, _ in inf_results]

        # ── 2. Batch cls evaluation ────────────────────────────────────────
        cls_prompts = [
            _tokenize(tokenizer, *build_messages_classification(
                s["problem"],
                s["history"] + [{"inference": inf, "is_fail": False}],
                len(s["history"]),
                system_cls,
            ))
            for s, inf in zip(active, inferences)
        ]
        cls_results      = vllm_generate_batched(cls_llm, cls_prompts, CLS_MAX_NEW, 0.0, return_logprobs=True)
        cls_outputs      = [t   for t, _, _   in cls_results]
        cls_ids_list     = [ids for _, ids, _ in cls_results]
        cls_old_lps_list = [lps for _, _, lps in cls_results]

        # ── 3. Parse & categorize ──────────────────────────────────────────
        end_batch, solve_batch, rethink_batch = [], [], []
        for i, (s, inf, cls_out) in enumerate(zip(active, inferences, cls_outputs)):
            fail_rubrics, action = parse_action(cls_out, inf)
            step_info = {
                "step":              s["step_count"],
                "action":            action,
                "inf_line":          _first_line(inf),
                "deep_critic":       _deep_critic_line(cls_out),
                "fail_rubrics":      fail_rubrics,
                "rollouts":          None,
                "cls_output":        cls_out,
                "cls_prompt_ids":    cls_prompts[i],
                "cls_response_ids":  cls_ids_list[i],
                "cls_old_log_probs": cls_old_lps_list[i],
            }
            if action == TOKEN_END:
                end_batch.append((s, inf, step_info))
            elif action == TOKEN_SOLVE:
                solve_batch.append((s, inf, step_info))
            else:
                rethink_batch.append((s, inf, step_info))

        # ── 4. END ────────────────────────────────────────────────────────
        if end_batch:
            does_list = vllm_generate_does_batched(inf_llm, tokenizer,
                                                   [inf for s, inf, _ in end_batch], system_summary)
            for (s, inf, step_info), does in zip(end_batch, does_list):
                s["final_text"] = inf
                s["history"].append({"inference": inf, "does": does, "is_fail": False})
                s["step_infos"].append(step_info)
                s["step_count"] += 1
                s["done"] = True
                _record_step(s, step_info, cache_path, cls_cache_path)

        # ── 5. SOLVE ──────────────────────────────────────────────────────
        if solve_batch:
            does_list = vllm_generate_does_batched(inf_llm, tokenizer,
                                                   [inf for s, inf, _ in solve_batch], system_summary)
            for (s, inf, step_info), does in zip(solve_batch, does_list):
                s["history"].append({"inference": inf, "does": does, "is_fail": False})
                s["step_infos"].append(step_info)
                s["step_count"] += 1
                _record_step(s, step_info, cache_path, cls_cache_path)

        # ── 6. RETHINK ────────────────────────────────────────────────────
        if rethink_batch:
            _process_rethinks_batched(
                rethink_batch, inf_llm, cls_llm, cls_model, tokenizer,
                system_inf, system_rethink, system_cls, system_summary,
            )
            for s, inf, step_info in rethink_batch:
                s["step_count"] += 1
                _record_step(s, step_info, cache_path, cls_cache_path)

        # ── 7. MAX_STEPS 초과 강제 종료 ───────────────────────────────────
        for s in active:
            if not s["done"] and s["step_count"] >= MAX_STEPS:
                s["final_text"] = s["history"][-1]["inference"] if s["history"] else ""
                s["done"] = True

        # ── 8. 완료된 문제 처리 → traj_all 기록, 큐에서 보충 ──────────────
        still_active = []
        for s in pool:
            if s["done"]:
                _finalize_state(s, all_path, cls_all_path)
                seq = _fmt_step_sequence(s["step_infos"])
                outcome_str = "✓" if s["outcome"] == 1.0 else "✗"
                fail_tag    = f"  [{s['fail_reason']}]" if s.get("fail_reason") else ""
                print(
                    f"[traj] {s['prob_idx']} {outcome_str}  steps={s['n_steps']}  rethinks={s['n_rethinks']}  [{seq}]{fail_tag}",
                    flush=True,
                )
                yield s
                if queue:
                    still_active.append(_new_state(queue.pop(0)))
            else:
                still_active.append(s)
        pool = still_active


# ─────────────────────────────────────────────────────────────────────────────
# Reward computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_trajectory_records(traj: dict) -> list[dict]:
    """trajectory의 모든 cls 기록에 backward reward propagation으로 reward 할당.

    carry = trajectory_outcome 으로 시작해 역방향으로 전파:
      - rethink 스텝: main_reward = rollout_avg × carry; carry 갱신
      - solve/end 스텝: main_reward = carry; carry 유지

    rollout 기록(rethink 시 각 rollout 평가): reward = rollout_outcome_i
    """
    step_infos = traj.get("step_infos", [])
    carry = traj.get("outcome", 0.0)

    records = []
    for step_info in reversed(step_infos):
        if step_info["action"] == TOKEN_RETHINK:
            rollout_avg = step_info.get("rollout_avg_outcome", 0.0)
            main_reward = rollout_avg * carry
            carry = main_reward
        else:
            main_reward = carry

        # main step cls record
        prompt_ids   = step_info.get("cls_prompt_ids",   [])
        response_ids = step_info.get("cls_response_ids", [])
        if prompt_ids and response_ids:
            records.append({
                "prompt_ids":    prompt_ids,
                "response_ids":  response_ids,
                "old_log_probs": step_info.get("cls_old_log_probs", []),
                "reward":        main_reward,
            })

        # rollout cls records (rethink 스텝에만 존재)
        for rec in step_info.get("cls_rollout_records", []):
            records.append({
                "prompt_ids":    rec["prompt_ids"],
                "response_ids":  rec["response_ids"],
                "old_log_probs": rec["old_log_probs"],
                "reward":        rec["outcome"],
            })

    return records


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
    p.add_argument("--iter_problems",      type=int,   default=None)
    p.add_argument("--inf_gpu_count",      type=int,   default=None)
    p.add_argument("--problem_batch_size", type=int,   default=None)
    p.add_argument("--max_gen_batch_size", type=int,   default=None)
    p.add_argument("--debug", action="store_true",
                   help="스텝별 live 출력 (문제번호_스텝번호 + rethink rollout 결과)")
    args, _ = p.parse_known_args()

    global INF_CKPT, CLS_CKPT, RESUME, PRM_COEF, OUTCOME_COEF, TRAIN_GPUS, DEBUG
    global INF_GPU_COUNT, MIN_RECORDS, ITER_PROBLEMS, PROBLEM_BATCH_SIZE, MAX_GEN_BATCH_SIZE
    if args.gpus:
        TRAIN_GPUS = list(range(len(args.gpus.split(","))))
    if args.inf_checkpoint:              INF_CKPT       = args.inf_checkpoint
    if args.cls_checkpoint:              CLS_CKPT       = args.cls_checkpoint
    if args.resume_from:                 RESUME         = args.resume_from
    if args.prm_coef        is not None: PRM_COEF       = args.prm_coef
    if args.outcome_coef    is not None: OUTCOME_COEF   = args.outcome_coef
    if args.min_records     is not None: MIN_RECORDS    = args.min_records
    if args.iter_problems   is not None: ITER_PROBLEMS  = args.iter_problems
    if args.inf_gpu_count   is not None: INF_GPU_COUNT  = args.inf_gpu_count
    if args.problem_batch_size is not None: PROBLEM_BATCH_SIZE = args.problem_batch_size
    if args.max_gen_batch_size is not None: MAX_GEN_BATCH_SIZE = args.max_gen_batch_size
    if args.debug:
        DEBUG = True


def main():
    _parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    inf_llm, cls_llm, cls_model, ref_cls, tokenizer = setup_models_and_tokenizer()
    problems = load_problems()
    log.info(
        f"Loaded {len(problems)} problems | "
        f"ROLLOUT_N={ROLLOUT_N} ITER_PROBLEMS={ITER_PROBLEMS} "
        f"TOTAL_PROBLEMS={TOTAL_PROBLEMS}"
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

    # 출력 디렉터리
    out_dir = _ROOT / "output" / "GRPO_SC" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {out_dir}")

    start_dt  = datetime.datetime.now()
    meta_path = out_dir / "run_meta.json"
    run_meta  = {
        "ts":                   ts,
        "start_time":           start_dt.isoformat(),
        "end_time":             None,
        "elapsed_seconds":      None,
        "total_steps":          None,
        "inference_model":      INF_CKPT,
        "cls_model":            CLS_CKPT,
        "gpus":                 TRAIN_GPUS,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "config": {
            "rollout_n":          ROLLOUT_N,
            "lr":                 LR,
            "kl_coef":            KL_COEF,
            "clip_eps":           CLIP_EPS,
            "iter_problems":      ITER_PROBLEMS,
            "total_problems":     TOTAL_PROBLEMS,
            "problem_batch_size": PROBLEM_BATCH_SIZE,
            "max_gen_batch_size": MAX_GEN_BATCH_SIZE,
        },
    }
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))

    # ITER_PROBLEMS개 문제를 랜덤 샘플 → 전체 trajectory 완료 → backward reward → GRPO update
    n_iterations = TOTAL_PROBLEMS // max(ITER_PROBLEMS, 1)
    log.info(f"총 {n_iterations} iterations (ITER_PROBLEMS={ITER_PROBLEMS})")

    for iter_idx in range(global_step, n_iterations):
        batch = random.sample(problems, min(ITER_PROBLEMS, len(problems)))
        log.info(f"[iter {iter_idx + 1}/{n_iterations}] {len(batch)} problems sampled")

        # 모든 trajectory 완료
        trajectories = list(generate_trajectories_pool(batch, inf_llm, cls_llm, cls_model, tokenizer, out_dir))

        # backward reward propagation → 전체 cls records 구성
        all_records = []
        for traj in trajectories:
            all_records.extend(compute_trajectory_records(traj))

        if not all_records:
            log.warning("No records collected — skip GRPO update")
            continue

        # GRPO update
        loss_val     = grpo_update(cls_model, ref_cls, optimizer, all_records)
        global_step += 1
        sync_cls_llm(cls_llm, cls_model)

        iter_outcomes = [t["outcome"]    for t in trajectories]
        iter_rethinks = [t["n_rethinks"] for t in trajectories]
        iter_steps    = [t["n_steps"]    for t in trajectories]
        avg_outcome   = sum(iter_outcomes) / len(iter_outcomes)
        avg_rethinks  = sum(iter_rethinks) / len(iter_rethinks)
        avg_steps     = sum(iter_steps)    / len(iter_steps)
        n_records     = len(all_records)
        n_trajs       = len(trajectories)

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

    if wandb_run:
        wandb_run.finish()

    end_dt = datetime.datetime.now()
    run_meta["end_time"]        = end_dt.isoformat()
    run_meta["elapsed_seconds"] = (end_dt - start_dt).total_seconds()
    run_meta["total_steps"]     = global_step
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    log.info("Training complete.")


if __name__ == "__main__":
    main()
