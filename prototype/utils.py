"""
prototype/utils.py
공통 유틸리티: 하이퍼파라미터, 데이터 구조, 모델 로드, GPT 헬퍼, solve 루프
"""

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from openai import OpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터
# ─────────────────────────────────────────────────────────────────────────────

# 생성 모델
GENERATOR_MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"
GENERATOR_CACHE_DIR      = "/mnt/.cache/huggingface"
SFT_CHECKPOINT           = "/mnt/yoonju/SC/checkpoints/sft/20260322_202515/epoch3"
GENERATOR_TEMPERATURE    = 0.8
GENERATOR_MAX_NEW_TOKENS = 512   # 1024→512: 어차피 TRUNCATE_TOKEN_LIMIT으로 잘리고, 생성 자체를 짧게

# GPT API
GPT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TRUNCATOR   = "gpt-5.4-nano"   # 트런케이션 전용
REWARD      = "o3"             # 스텝 리워드 평가
PATCHER     = "o3-mini"        # teacher fallback (correction 생성)

# Solve 루프
MAX_STEPS            = 30
TRUNCATE_TOKEN_LIMIT = 384  # 512→384: inp_ids 누적 억제. 스텝 10개 × 384 = 3.8K max context

# PPO
PPO_EPOCHS          = 1
PPO_LR              = 5e-6  # 1e-5→5e-6: gradient accumulation(×16) 고려, 안정성
PPO_CLIP_EPS        = 0.2
PPO_COLLECT_SIZE    = 512   # PPO 업데이트 전 모을 trajectory 수
PPO_MINI_BATCH_SIZE = 4     # trajectory 단위 gradient accumulation (64/4 = 16 optimizer steps)
PPO_MAX_GRAD_NORM   = 1.0
KL_COEF             = 0.05  # 0.01→0.05: step 수 줄었지만 policy collapse 방지, reward dilution 보완
GAMMA               = 0.95  # 1.0→0.95: discount로 credit assignment 개선, variance 억제
MAX_SEQ_LEN         = 4096  # PPO forward pass 최대 시퀀스 길이 (초과 스텝 skip)

# GPU 설정 (physical GPU 번호)
GPU_WORKERS = [4, 5]   # 데이터 생성 워커
GPU_TRAINER = 6        # PPO 학습

# 데이터셋
DATASET_PATH    = "../datasets/deepmath_16k.parquet"
MATH500_PATH    = "../datasets/math500.parquet"
SAVE_DIR        = "../output/prototype"
VAL_BATCH_SIZE  = 32   # validation 배치 GPU 추론 크기

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """한 스텝의 데이터. 텐서는 CPU에 저장 (Ray 직렬화 호환)."""
    step_idx: int
    action: str                        # "solve" | "correct"  (이번 스텝의 역할)
    text: str                          # 추론 텍스트 (액션 토큰 미포함)
    reward: float                      # Gemini 리워드 [0, 1]
    predicted_next_action: str         # 모델이 예측한 다음 액션 토큰
    ground_truth_next_action: str      # Gemini 리워드 기반 실제 다음 액션 토큰
    input_ids: torch.Tensor            # (1, L_in)   - CPU
    response_ids: torch.Tensor         # (1, L_resp+1) reasoning + GT 액션 토큰 - CPU
    log_probs_old: torch.Tensor        # (L_resp+1,) - CPU
    is_generator_step: bool            # False면 teacher 스텝 → PPO 제외


@dataclass
class Trajectory:
    problem_id: str
    problem: str
    answer: str
    difficulty: Optional[float] = None
    steps: List[StepRecord] = field(default_factory=list)
    have_boxed: bool = False   # 풀이 과정에서 \boxed{}가 등장했는지
    is_answer: bool = False    # 등장한 답이 맞았는지 (reward > 0.1로 검증)
    patcher_wrong: bool = False

# ─────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────────────────────────────────────

# 프롬프트는 source/prompts.json에서 로드 (모든 스크립트 공유)
import pathlib as _pathlib
_PROMPTS_PATH = _pathlib.Path(__file__).resolve().parent.parent / "source" / "prompts.json"
with open(_PROMPTS_PATH) as _f:
    _PROMPTS = json.load(_f)

SYSTEM_SOLVE   = _PROMPTS["system_solve"]
SYSTEM_CORRECT = _PROMPTS["system_correct"]

# ─────────────────────────────────────────────────────────────────────────────
# 스페셜 액션 토큰
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_SOLVE   = "<|solve|>"
TOKEN_CORRECT = "<|correct|>"
TOKEN_END     = "<|end|>"    # 커스텀 종료 토큰 (<|im_end|>와 달리 Qwen 기본 동작 없음)
ACTION_TOKENS = [TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END]

# ─────────────────────────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_generator(
    device_map: str = "auto",
    model_path: str | None = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """SFT 체크포인트(기본값: SFT_CHECKPOINT)를 로드.

    ACTION_TOKENS(<|solve|>, <|correct|>, <|end|>)를 스페셜 토큰으로 등록하고
    모델 임베딩을 리사이즈한다.

    model_path: 체크포인트 경로를 지정하면 해당 경로의 weights를 로드.
                None이면 SFT_CHECKPOINT를 로드.
    """
    load_path = model_path if model_path else SFT_CHECKPOINT
    logger.info(f"Generator 로드 중: {load_path}")

    # SFT 체크포인트는 토크나이저도 함께 저장되어 있음
    tokenizer = AutoTokenizer.from_pretrained(
        load_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 액션 스페셜 토큰 등록 (<|end|>은 커스텀 토큰이므로 TOKEN_END도 포함)
    tokens_to_add = [TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END]
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    if added:
        logger.info(f"  스페셜 토큰 {added}개 추가됨: {tokens_to_add}")

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    if added:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"  임베딩 리사이즈: {len(tokenizer)} vocab")
    model.eval()
    logger.info("Generator 로드 완료.")
    return model, tokenizer

# ─────────────────────────────────────────────────────────────────────────────
# 생성 헬퍼
# ─────────────────────────────────────────────────────────────────────────────


class StopOnActionToken(StoppingCriteria):
    """액션 스페셜 토큰이 생성되면 즉시 종료."""
    def __init__(self, tokenizer, input_length: int):
        self._input_length = input_length
        self._action_ids = set(
            tid for tid in tokenizer.convert_tokens_to_ids(ACTION_TOKENS)
            if tid != tokenizer.unk_token_id
        )

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        if input_ids.shape[1] > self._input_length:
            return input_ids[0, -1].item() in self._action_ids
        return False


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def _pick_action_by_prob(model, tokenizer, full_ids: torch.Tensor) -> str:
    """full_ids 마지막 위치에서 세 액션 토큰 중 가장 높은 로짓의 토큰을 반환."""
    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ACTION_TOKENS]
    with torch.no_grad():
        logits = model(full_ids).logits  # (1, seq_len, vocab)
    last_logits = logits[0, -1, :]
    best_idx = max(range(len(token_ids)), key=lambda i: last_logits[token_ids[i]])
    return ACTION_TOKENS[best_idx]


def generate_one_step(
    model,
    tokenizer,
    prompt_text: str,
) -> Tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor]:
    """한 스텝 생성.

    Returns:
        reasoning_text     : 액션 토큰을 제외한 추론 텍스트
        predicted_action   : 모델이 예측한 다음 액션 ("<|solve|>" 또는 "<|correct|>")
        input_ids_cpu      : (1, L_in)
        response_ids_cpu   : (1, L_reasoning)  — 액션 토큰 미포함
        log_probs_cpu      : (L_reasoning,)
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_ids   = inputs["input_ids"]
    input_length = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=GENERATOR_MAX_NEW_TOKENS,
            temperature=GENERATOR_TEMPERATURE if GENERATOR_TEMPERATURE > 0.0 else None,
            do_sample=GENERATOR_TEMPERATURE > 0.0,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([
                StopOnActionToken(tokenizer, input_length)
            ]),
        )

    # 생성된 토큰에서 추론 텍스트 / 액션 토큰 분리
    response_ids    = output_ids[:, input_length:]
    generated_text  = tokenizer.decode(response_ids[0], skip_special_tokens=False)

    predicted_action = None
    reasoning_text   = generated_text
    for tok in ACTION_TOKENS:
        if tok in generated_text:
            idx = generated_text.rfind(tok)
            reasoning_text   = generated_text[:idx].rstrip()
            predicted_action = tok
            break

    # 액션 토큰이 없으면 확률로 선택 (룰 기반)
    if predicted_action is None:
        predicted_action = _pick_action_by_prob(model, tokenizer, output_ids)

    # 추론 텍스트만 response_ids로 재토크나이즈
    reasoning_ids = tokenizer(
        reasoning_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(model.device)
    log_probs = _compute_log_probs(model, input_ids, reasoning_ids)

    return (
        reasoning_text,
        predicted_action,
        input_ids.cpu(),
        reasoning_ids.cpu(),
        log_probs.cpu(),
    )


def _compute_log_probs(
    model,
    input_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """response_ids 각 토큰의 log prob 반환. shape: (L_resp,)"""
    full_ids = torch.cat([input_ids, response_ids], dim=1)
    with torch.no_grad():
        logits = model(full_ids).logits
    resp_logits = logits[:, input_ids.size(1) - 1 : -1, :]
    lp = F.log_softmax(resp_logits, dim=-1)
    return lp.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)


def build_gt_response(
    model,
    tokenizer,
    reasoning_text: str,
    input_ids_cpu: torch.Tensor,
    gt_action: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """추론 텍스트 + GT 액션 토큰을 합친 타겟 시퀀스의 (response_ids, log_probs) 반환.

    PPO 학습 타겟으로 사용된다. GT 액션 토큰은 응답 시퀀스 끝에 붙는다.
    """
    target_text = reasoning_text + gt_action
    resp_ids = tokenizer(
        target_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(model.device)
    lp = _compute_log_probs(model, input_ids_cpu.to(model.device), resp_ids)
    return resp_ids.cpu(), lp.cpu()


def _normalize_latex(s: str) -> str:
    """수학적으로 동등한 LaTeX 표현을 정규화."""
    s = s.strip().replace(" ", "")
    s = re.sub(r"\\dfrac", r"\\frac", s)          # \dfrac → \frac
    s = re.sub(r"\\tfrac", r"\\frac", s)           # \tfrac → \frac
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)     # \text{X} → X
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)   # \mathrm{X} → X
    s = re.sub(r"\\left|\\right", "", s)            # \left, \right 제거
    s = re.sub(r"[(){}]", "", s)                   # 괄호 제거 (구조 비교 전처리)
    return s


def check_solved(step_text: str, gold_answer: str) -> bool:
    """스텝 텍스트에 \\boxed{정답}이 있는지 확인.
    LaTeX 형식 차이 (\\dfrac vs \\frac, \\text{X} vs X 등)는 정규화 후 비교.
    """
    m = re.search(r"\\boxed\{(.+?)\}", step_text, re.DOTALL)
    if not m:
        return False
    pred_raw = m.group(1).strip().replace(" ", "")
    gold_raw = gold_answer.strip().replace(" ", "")

    # 1) 원본 문자열 일치
    if pred_raw == gold_raw:
        return True
    # 2) 숫자 비교
    try:
        if abs(float(pred_raw) - float(gold_raw)) < 1e-6:
            return True
    except ValueError:
        pass
    # 3) LaTeX 정규화 후 비교
    if _normalize_latex(pred_raw) == _normalize_latex(gold_raw):
        return True
    return False


def has_boxed(text: str) -> bool:
    return bool(re.search(r"\\boxed\{", text))


def extract_boxed(text: str) -> str | None:
    """텍스트에서 마지막 \\boxed{...} 내용을 추출한다 (중첩 괄호 처리)."""
    marker = r"\boxed{"
    pos = text.rfind(marker)
    if pos == -1:
        return None
    start = pos + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def format_final_step(text: str) -> str:
    """마지막 스텝 텍스트에 boxed{}가 있고 'Therefore' 접두사가 없으면
    'Therefore, the final answer is \\boxed{...}.<|end|>' 포맷을 concat해서 반환."""
    if not has_boxed(text) or "Therefore" in text:
        return text
    boxed = extract_boxed(text)
    if boxed is None:
        return text
    return f"{text}\nTherefore, the final answer is \\boxed{{{boxed}}}.<|end|>"

# ─────────────────────────────────────────────────────────────────────────────
# GPT 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _action_label(token: str) -> str:
    """액션 토큰을 프롬프트용 레이블로 변환."""
    return "SOLVE" if token == TOKEN_SOLVE else "CORRECT"


def _gpt(model_name: str, messages: list, max_completion_tokens: int = None) -> str:
    """GPT API 호출 후 응답 텍스트 반환.

    messages: OpenAI chat messages 형식 리스트.
    max_completion_tokens: 출력 토큰 제한 (reward 등 짧은 응답에 사용).
    """
    client = OpenAI(api_key=GPT_API_KEY)
    kwargs: dict = {"model": model_name, "messages": messages}
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def truncate_to_512_tokens(text: str, tokenizer) -> str:
    """스텝 텍스트를 TRUNCATE_TOKEN_LIMIT 이내로 자연스럽게 트런케이션.

    요약/수정 없이 자르기만 — 최대한 길게, 문장·수식 경계에서 끊음.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= TRUNCATE_TOKEN_LIMIT:
        return text

    # 첫 TRUNCATE_TOKEN_LIMIT 토큰까지의 텍스트
    prefix_ids  = token_ids[:TRUNCATE_TOKEN_LIMIT]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)

    # system: 지시문 (모든 호출 공통 → prompt caching)
    # user:   자를 텍스트
    # 출력:   마지막 유효 문장/수식 하나만 → Python rfind 로 잘라냄 (출력 토큰 최소화)
    system_msg = (
        "You are given a math reasoning text that is too long.\n"
        f"Target: keep at most {TRUNCATE_TOKEN_LIMIT} Qwen tokens.\n\n"
        "Task: find the LAST valid cut point in the text. A valid cut point is:\n"
        "  - End of a complete sentence (period, ?, etc.)\n"
        "  - End of a complete math expression (\\boxed{...}, \\], $$,\n"
        "    'Therefore/Thus/Hence/So ...' conclusion)\n"
        "  - NOT mid-sentence or mid-equation\n\n"
        "Output ONLY the final sentence or expression verbatim (the one to end on).\n"
        "Do NOT output the full text. Just that one last sentence/expression."
    )
    try:
        last_sentence = _gpt(
            TRUNCATOR,
            [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": prefix_text},
            ],
        ).strip()

        # 원본에서 해당 문장의 마지막 위치를 찾아 그 뒤에서 자름
        idx = prefix_text.rfind(last_sentence)
        if idx != -1:
            result = prefix_text[: idx + len(last_sentence)]
            if len(tokenizer.encode(result, add_special_tokens=False)) <= TRUNCATE_TOKEN_LIMIT:
                return result
    except Exception as e:
        logger.warning(f"[truncate] GPT 실패 ({e}), hard truncation 사용")

    return prefix_text


_reward_sample_logged = False  # 최초 1회만 샘플 로그 출력


def compute_reward(
    problem: str,
    prev_steps: List[str],
    current_step: str,
    prev_rewards: List[float] = None,
    prev_actions: List[str] = None,
    current_action: str = None,
) -> float:
    """현재 스텝만 평가해 [0, 1] 리워드 반환. 실수/오류 있으면 반드시 0.0.

    멀티턴 messages 구조로 prompt caching 활용:
      system       : 평가 지침 (모든 호출 공통 — 완전 캐싱)
      user         : 문제 (같은 문제 내 모든 호출 공통 — 캐싱)
      asst         : "Understood."
      user/asst    : [ACTION]\n이전 스텝 & 점수 쌍 반복 (step N 호출 시 step N-1까지 캐싱)
      user         : [ACTION]\n현재 스텝 (매 호출마다 새로 추가)

    prev_actions: 이전 스텝별 액션 토큰 목록 (history와 1:1 대응)
    current_action: 현재 스텝의 액션 토큰
    """
    system_msg = (
        "You are a strict mathematical reasoning evaluator.\n"
        "Evaluate ONLY the single CURRENT STEP — not the whole solution.\n\n"
        "Scoring rules (return a single float, nothing else):\n"
        "  0.0  — ANY error: wrong arithmetic, wrong algebra, incorrect logic,\n"
        "         or contradicts previous steps.\n"
        "  0.1  — correct but trivial (restates something already said).\n"
        "  0.5  — correct and makes meaningful progress.\n"
        "  1.0  — correct, significant progress, or reaches the right final answer.\n\n"
        "IMPORTANT: If the current step merely repeats or rephrases what was already\n"
        "established, assign at most 0.1 even if correct.\n"
        "Be strict. Return ONLY the numeric score."
    )
    messages = [
        {"role": "system",    "content": system_msg},
        {"role": "user",      "content": f"[Problem]\n{problem}"},
        {"role": "assistant", "content": "Understood. Show me each step to evaluate."},
    ]

    # 이전 스텝 & 점수를 [ACTION]\nstep_text / score 쌍으로 추가 (캐싱 대상)
    if prev_rewards is not None:
        for i, (step_text, score) in enumerate(zip(prev_steps, prev_rewards)):
            action = prev_actions[i] if prev_actions and i < len(prev_actions) else TOKEN_SOLVE
            messages.append({"role": "user",      "content": f"[{_action_label(action)}]\n{step_text}"})
            messages.append({"role": "assistant", "content": str(score)})
    elif prev_steps:
        for i, step_text in enumerate(prev_steps):
            action = prev_actions[i] if prev_actions and i < len(prev_actions) else TOKEN_SOLVE
            messages.append({"role": "user",      "content": f"[{_action_label(action)}]\n{step_text}"})
            messages.append({"role": "assistant", "content": "Understood."})

    # 평가할 현재 스텝 (매 호출 새로 추가)
    cur_label = _action_label(current_action) if current_action else "SOLVE"
    messages.append({"role": "user", "content": f"[{cur_label}]\n{current_step}"})

    try:
        # o3는 reasoning token도 max_completion_tokens에서 소비하므로 충분히 확보
        raw = _gpt(REWARD, messages, max_completion_tokens=1024).strip()

        global _reward_sample_logged
        if not _reward_sample_logged:
            _reward_sample_logged = True
            logger.info(
                f"[reward][SAMPLE] messages=\n{json.dumps(messages, ensure_ascii=False, indent=2)}"
                f"\n[reward][SAMPLE] raw_response={repr(raw)}"
            )

        # 직접 float 변환 먼저 시도, 실패 시 regex로 첫 번째 숫자 추출
        try:
            score = float(raw)
        except ValueError:
            m = re.search(r"[-+]?\d+\.?\d*", raw)
            if m is None:
                logger.warning(f"[reward] 숫자 없는 응답: {repr(raw)!r}, 0.0 반환")
                return 0.0
            score = float(m.group())
        return float(max(0.0, min(1.0, score)))
    except Exception as e:
        logger.warning(f"[reward] GPT 실패 ({e}), 0.0 반환")
        return 0.0


def teacher_correct_step(
    problem: str,
    prev_steps: List[str],
    correct_reason: str,
    prev_actions: List[str] = None,
    messages_prefix: Optional[list] = None,
    new_committed: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[Optional[str], list]:
    """Patcher가 generator 대신 correction을 한 단계 수행.

    messages_prefix가 주어지면 이전 PATCHER 호출 상태(system+problem+이전 스텝+이전 응답)를
    그대로 이어받아 new_committed 스텝만 추가한 뒤 현재 실패 스텝을 붙인다.
    → 이전 호출과 prefix가 동일하므로 OpenAI prompt cache hit 보장.

    messages_prefix가 None이면 처음부터 구성 (첫 PATCHER 호출).

    Returns: (patcher_text, messages_sent)
      patcher_text  : 생성된 correction 텍스트 (실패 시 None)
      messages_sent : 이번 호출에 사용된 전체 messages 리스트
    """
    failed_step = prev_steps[-1] if prev_steps else ""

    system_msg = (
        "You are an expert mathematician correcting a flawed reasoning step.\n"
        "The student attempted a correction but failed. You must provide the correct version.\n"
        "Write exactly ONE corrected step. Use proper LaTeX. No extra text."
    )

    if messages_prefix is not None:
        # 이전 PATCHER 호출 상태에서 이어받기 (prefix 동일 → cache hit)
        messages = list(messages_prefix)
        for step_text, action_tok in (new_committed or []):
            messages.append({"role": "user",      "content": f"[{_action_label(action_tok)}]\n{step_text}"})
            messages.append({"role": "assistant", "content": "OK"})
    else:
        # 처음부터 구성
        committed = prev_steps[:-1] if len(prev_steps) > 1 else []
        committed_actions = prev_actions[:-1] if prev_actions else []
        messages = [
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": f"[Problem]\n{problem}"},
            {"role": "assistant", "content": "Understood. Show me the steps."},
        ]
        for i, step_text in enumerate(committed):
            action = committed_actions[i] if i < len(committed_actions) else TOKEN_SOLVE
            messages.append({"role": "user",      "content": f"[{_action_label(action)}]\n{step_text}"})
            messages.append({"role": "assistant", "content": "OK"})

    # 실패한 correction 스텝 + 수정 요청 (매 호출 새로 추가 — uncached)
    messages.append({
        "role": "user",
        "content": f"[CORRECT]\n[Failed attempt]\n{failed_step}\n\n[Issue to fix]\n{correct_reason}",
    })

    try:
        return _gpt(PATCHER, messages).strip(), messages
    except Exception as e:
        logger.warning(f"[teacher_correct] GPT 실패 ({e})")
        return None, messages

# ─────────────────────────────────────────────────────────────────────────────
# Solve 루프 프롬프트 빌더
# ─────────────────────────────────────────────────────────────────────────────

def _solve_user(problem: str, history: List[str]) -> str:
    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Steps so far]")
        for i, s in enumerate(history, 1):
            lines.append(f"Step {i}: {s}")
    lines.append("\nWrite the next step.")
    return "\n".join(lines)


def _correct_user(problem: str, history: List[str], reason: str) -> str:
    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Steps so far]")
        for i, s in enumerate(history, 1):
            lines.append(f"Step {i}: {s}")
    lines.append(f"\n[Error to fix]\n{reason}")
    return "\n".join(lines)



# ─────────────────────────────────────────────────────────────────────────────
# Rollout 저장
# ─────────────────────────────────────────────────────────────────────────────

def create_rollout_file(path: str):
    """추론 시작 시 빈 JSONL 파일을 미리 생성."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        open(path, "w").close()
    logger.info(f"Rollout 파일 생성: {path}")


def save_trajectory(traj: Trajectory, path: Optional[str]):
    """Trajectory를 JSONL 파일에 한 줄로 append. path=None이면 스킵."""
    if not path:
        return
    record = {
        "problem_id":    traj.problem_id,
        "problem":       traj.problem,
        "answer":        traj.answer,
        "difficulty":    traj.difficulty,
        "have_boxed":   traj.have_boxed,
        "is_answer":     traj.is_answer,
        "patcher_wrong": traj.patcher_wrong,
        "n_steps":       len(traj.steps),
        "steps": [
            {
                "step_idx":                  s.step_idx,
                "action":                    s.action,
                "text":                      s.text,
                "reward":                    s.reward,
                "predicted_next_action":     s.predicted_next_action,
                "ground_truth_next_action":  s.ground_truth_next_action,
                "is_generator_step":         s.is_generator_step,
            }
            for s in traj.steps
        ],
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Solve 루프
# ─────────────────────────────────────────────────────────────────────────────

def _pfx(pid, step: int = None) -> str:
    """로그 prefix 생성: [P00076] 또는 [P00076] [S03]"""
    try:
        p = f"[P{int(pid):05d}]"
    except (ValueError, TypeError):
        p = f"[P{str(pid):>5s}]"
    return p if step is None else f"{p} [S{step:02d}]"


def _log_traj(traj: "Trajectory", problem_id: str, status: str):
    """Trajectory 요약 로그 출력.

    gt[i]  = step i에 진입할 때 사용된 액션 (step 0은 항상 <|solve|>)
    pred[i] = 같은 방식의 predicted 액션
    is_gen / rewards 는 각 스텝과 1:1 대응.
    """
    steps = traj.steps
    gt_in   = [TOKEN_SOLVE] + [s.ground_truth_next_action for s in steps[:-1]]
    pred_in = [TOKEN_SOLVE] + [s.predicted_next_action    for s in steps[:-1]]
    is_gen  = [s.is_generator_step for s in steps]
    rewards = [round(s.reward, 3)  for s in steps]
    logger.info(
        f"{'─'*60}\n"
        f"[{status}] problem_id={problem_id}  steps={len(steps)}\n"
        f"  pred    : {pred_in}\n"
        f"  gt      : {gt_in}\n"
        f"  is_gen  : {is_gen}\n"
        f"  rewards : {rewards}"
    )


def solve_problem(
    model,
    tokenizer,
    problem: str,
    answer: str,
    problem_id: str = "0",
    rollout_path: Optional[str] = None,
    difficulty: Optional[float] = None,
) -> Optional[Trajectory]:
    """한 문제를 최대 MAX_STEPS 스텝까지 풀어 Trajectory를 반환.

    흐름:
      1. 첫 번째 액션은 항상 solve
      2. 모델이 추론 텍스트 생성 → predicted_action 기록 (라우팅에 미사용)
         (특수 토큰 미생성 시 로짓 확률로 predicted_action 강제)
      3. REWARD 평가 → 룰 기반으로 다음 액션 결정 (boxed 무관, reward만 사용):
           reward > 0.1  → TOKEN_SOLVE (계속 풀기)
           reward <= 0.1 → TOKEN_CORRECT:
             generator 시도 → 여전히 낮으면 patcher → 그래도 낮으면 종료

    저장: rollout_path가 주어지면 trajectory를 JSONL에 저장.
    """
    traj = Trajectory(problem_id=problem_id, problem=problem, answer=answer, difficulty=difficulty)
    history: List[str]       = []
    reward_history: List[float] = []
    action_history: List[str]   = []

    # PATCHER messages 캐시
    patcher_messages_cache: Optional[list] = None
    patcher_history_len: int = 0

    current_action = TOKEN_SOLVE   # 첫 번째 액션은 항상 solve
    correct_reason = ""

    for step_idx in range(MAX_STEPS):

        # ── 프롬프트 구성 ────────────────────────────────────────────────
        if current_action == TOKEN_SOLVE:
            prompt       = build_chat_prompt(tokenizer, SYSTEM_SOLVE, _solve_user(problem, history))
            action_label = "solve"
        else:  # TOKEN_CORRECT
            prompt       = build_chat_prompt(tokenizer, SYSTEM_CORRECT, _correct_user(problem, history, correct_reason))
            action_label = "correct"

        # ── 생성 ────────────────────────────────────────────────────────
        reasoning, predicted_action, inp_ids, _, _ = generate_one_step(model, tokenizer, prompt)
        reasoning = truncate_to_512_tokens(reasoning, tokenizer)

        # ── 리워드 계산 ──────────────────────────────────────────────────
        reward = compute_reward(
            problem, history, reasoning,
            prev_rewards=reward_history,
            prev_actions=action_history,
            current_action=current_action,
        )

        # ── [분기] generator correct 실패 → patcher 호출 ────────────────
        if current_action == TOKEN_CORRECT and reward <= 0.1:

            # generator 실패 스텝 기록
            resp_ids_fail, lp_fail = build_gt_response(
                model, tokenizer, reasoning, inp_ids, TOKEN_CORRECT)
            traj.steps.append(StepRecord(
                step_idx=len(traj.steps), action="correct", text=reasoning, reward=reward,
                predicted_next_action=predicted_action, ground_truth_next_action=TOKEN_CORRECT,
                input_ids=inp_ids, response_ids=resp_ids_fail, log_probs_old=lp_fail,
                is_generator_step=True,
            ))
            history.append(reasoning)
            reward_history.append(reward)
            action_history.append(TOKEN_CORRECT)

            # patcher 호출 (이전 messages 캐시 재사용)
            new_committed = (
                list(zip(history[patcher_history_len:-1], action_history[patcher_history_len:-1]))
                if patcher_messages_cache is not None else None
            )
            patcher_text, patcher_messages = teacher_correct_step(
                problem, history, correct_reason,
                prev_actions=action_history,
                messages_prefix=patcher_messages_cache,
                new_committed=new_committed,
            )
            if patcher_text is None:
                logger.warning(f"{_pfx(problem_id, step_idx)} patcher=None, 종료")
                break

            patcher_text   = truncate_to_512_tokens(patcher_text, tokenizer)
            patcher_reward = compute_reward(
                problem, history, patcher_text,
                prev_rewards=reward_history,
                prev_actions=action_history,
                current_action=TOKEN_CORRECT,
            )

            patcher_prompt  = build_chat_prompt(tokenizer, SYSTEM_CORRECT, _correct_user(problem, history, correct_reason))
            patcher_inp_ids = tokenizer(patcher_prompt, return_tensors="pt")["input_ids"].cpu()

            # patcher도 실패 → 기록 후 종료
            if patcher_reward <= 0.1:
                resp_ids_p, lp_p = build_gt_response(
                    model, tokenizer, patcher_text, patcher_inp_ids, TOKEN_END)
                traj.steps.append(StepRecord(
                    step_idx=len(traj.steps), action="correct", text=patcher_text, reward=patcher_reward,
                    predicted_next_action=TOKEN_CORRECT, ground_truth_next_action=TOKEN_END,
                    input_ids=patcher_inp_ids, response_ids=resp_ids_p, log_probs_old=lp_p,
                    is_generator_step=False,
                ))
                traj.patcher_wrong = True
                _log_traj(traj, problem_id, "PATCHER_WRONG")
                save_trajectory(traj, rollout_path)
                return traj

            # patcher 성공 → 기록 후 solve 재개
            resp_ids_p, lp_p = build_gt_response(
                model, tokenizer, patcher_text, patcher_inp_ids, TOKEN_SOLVE)
            traj.steps.append(StepRecord(
                step_idx=len(traj.steps), action="correct", text=patcher_text, reward=patcher_reward,
                predicted_next_action=TOKEN_SOLVE, ground_truth_next_action=TOKEN_SOLVE,
                input_ids=patcher_inp_ids, response_ids=resp_ids_p, log_probs_old=lp_p,
                is_generator_step=False,
            ))
            history.append(patcher_text)
            reward_history.append(patcher_reward)
            action_history.append(TOKEN_CORRECT)
            patcher_messages_cache = patcher_messages + [{"role": "assistant", "content": patcher_text}]
            patcher_history_len    = len(history)

            current_action = TOKEN_SOLVE
            correct_reason = ""
            continue

        # ── 룰 기반 gt_next_action 결정 (reward만 사용, boxed 무관) ──────
        if reward > 0.1:
            gt_next_action = TOKEN_SOLVE
            correct_reason = ""
        else:
            gt_next_action = TOKEN_CORRECT
            correct_reason = "The previous step was INCORRECT."

        # ── 스텝 기록 ────────────────────────────────────────────────────
        resp_ids, lp = build_gt_response(model, tokenizer, reasoning, inp_ids, gt_next_action)
        traj.steps.append(StepRecord(
            step_idx=len(traj.steps),
            action=action_label,
            text=reasoning,
            reward=reward,
            predicted_next_action=predicted_action,
            ground_truth_next_action=gt_next_action,
            input_ids=inp_ids,
            response_ids=resp_ids,
            log_probs_old=lp,
            is_generator_step=True,
        ))
        history.append(format_final_step(reasoning))
        reward_history.append(reward)
        action_history.append(current_action)

        current_action = gt_next_action

    # ── 최종 집계: 모든 스텝 통틀어 판정 ────────────────────────────────
    for step in traj.steps:
        if has_boxed(step.text):
            traj.have_boxed = True
        if check_solved(step.text, answer):
            traj.is_answer = True

    # ── 요약 로그 ─────────────────────────────────────────────────────────
    status = "ANSWER" if traj.is_answer else ("PATCHER_WRONG" if traj.patcher_wrong else "FAILED/MAX_STEPS")
    _log_traj(traj, problem_id, status)

    save_trajectory(traj, rollout_path)

    return traj

# ─────────────────────────────────────────────────────────────────────────────
# 배치 생성 / 배치 log_probs / 배치 solve
# ─────────────────────────────────────────────────────────────────────────────

def generate_steps_batched(
    model,
    tokenizer,
    prompt_texts: List[str],
    max_new_tokens: Optional[int] = None,
) -> List[Tuple[str, str, torch.Tensor]]:
    """여러 프롬프트를 한 번의 model.generate 호출로 처리.

    Returns: List of (reasoning_text, predicted_action, input_ids_cpu)
    """
    if not prompt_texts:
        return []

    _max_new_tokens = max_new_tokens if max_new_tokens is not None else GENERATOR_MAX_NEW_TOKENS

    # generation 시 left-padding 필수
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompt_texts, return_tensors="pt", padding=True, truncation=False
    ).to(model.device)
    tokenizer.padding_side = orig_side

    max_input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=_max_new_tokens,
            temperature=GENERATOR_TEMPERATURE if GENERATOR_TEMPERATURE > 0.0 else None,
            do_sample=GENERATOR_TEMPERATURE > 0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    eos_id = tokenizer.eos_token_id
    for i, prompt_text in enumerate(prompt_texts):
        # left-padding: 생성 토큰은 항상 max_input_len 이후
        resp_list = output_ids[i, max_input_len:].tolist()
        if eos_id in resp_list:
            resp_list = resp_list[: resp_list.index(eos_id) + 1]

        generated_text = tokenizer.decode(resp_list, skip_special_tokens=False)

        predicted_action = None
        reasoning_text = generated_text
        for tok in ACTION_TOKENS:
            if tok in generated_text:
                idx = generated_text.rfind(tok)
                reasoning_text = generated_text[:idx].rstrip()
                predicted_action = tok
                break
        if predicted_action is None:
            predicted_action = TOKEN_SOLVE  # fallback (greedy decode에서 드묾)

        inp_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].cpu()
        results.append((reasoning_text, predicted_action, inp_ids))

    return results


def build_gt_responses_batched(
    model,
    tokenizer,
    items: List[Tuple[str, torch.Tensor, str]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """(reasoning_text, inp_ids_cpu, gt_action) 목록을 배치 forward pass 로 log_probs 계산.

    Returns: List of (response_ids_cpu, log_probs_cpu)
    """
    if not items:
        return []

    device = model.device
    all_full_seqs, all_resp_ids, all_inp_lens = [], [], []
    for reasoning_text, inp_ids_cpu, gt_action in items:
        target_text = reasoning_text + gt_action
        resp_ids = tokenizer(
            target_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        full_seq = torch.cat([inp_ids_cpu, resp_ids], dim=1).squeeze(0)
        all_full_seqs.append(full_seq)
        all_resp_ids.append(resp_ids.cpu())
        all_inp_lens.append(inp_ids_cpu.shape[1])

    # right-padding으로 batched forward pass
    max_len = max(s.shape[0] for s in all_full_seqs)
    batch_ids = torch.full(
        (len(all_full_seqs), max_len), tokenizer.pad_token_id, dtype=torch.long
    ).to(device)
    attn_mask = torch.zeros_like(batch_ids)
    for i, seq in enumerate(all_full_seqs):
        batch_ids[i, : seq.shape[0]] = seq.to(device)
        attn_mask[i, : seq.shape[0]] = 1

    with torch.no_grad():
        logits = model(input_ids=batch_ids, attention_mask=attn_mask).logits

    results = []
    for i, (resp_ids, inp_len) in enumerate(zip(all_resp_ids, all_inp_lens)):
        resp_len = resp_ids.shape[1]
        resp_logits = logits[i, inp_len - 1 : inp_len + resp_len - 1, :]
        lp = F.log_softmax(resp_logits, dim=-1)
        lp_gathered = (
            lp.gather(-1, resp_ids.squeeze(0).to(device).unsqueeze(-1)).squeeze(-1)
        )
        results.append((resp_ids, lp_gathered.cpu()))

    return results


def solve_problems_batch(
    model,
    tokenizer,
    problems_batch: List[dict],
    rollout_path: Optional[str] = None,
) -> List[Trajectory]:
    """여러 문제를 파이프라인 GPU 추론 + 비동기 API 호출로 처리.

    각 문제는 독립적으로 진행:
      generate → (truncate+reward) API async → generate → ...
    API 결과가 오는 대로 그 문제만 즉시 다음 generate 로 진행.
    다른 문제가 API 기다리는 동안 GPU는 ready 상태인 문제들을 배치 generate.

    phase 전이:
      "generate" → GPU generate → API 제출 → "api"
      "api" → API 완료 → log_probs 계산 → "generate" or "patcher" or "done"
      "patcher" → patcher 완료 → log_probs 계산 → "generate" or "done"
    """

    # Fast Tokenizer(Rust RefCell) 동시 접근 방지 lock
    _tok_lock = threading.Lock()

    # ── 헬퍼: truncate + reward API 를 하나의 Future 로 묶기 ─────────────────
    def _trunc_reward(problem, history, reward_history, raw_text, action_history, current_action):
        with _tok_lock:
            truncated = truncate_to_512_tokens(raw_text, tokenizer)
        reward    = compute_reward(
            problem, history, truncated,
            prev_rewards=reward_history,
            prev_actions=action_history,
            current_action=current_action,
        )
        return truncated, reward

    # ── 상태 초기화 ──────────────────────────────────────────────────────────
    states: List[Dict] = []
    for item in problems_batch:
        states.append({
            "traj":                    Trajectory(
                                           problem_id=item["problem_id"],
                                           problem=item["problem"],
                                           answer=item["answer"],
                                           difficulty=item.get("difficulty"),
                                       ),
            "problem":                 item["problem"],
            "answer":                  item["answer"],
            "problem_id":              item["problem_id"],
            "history":                 [],
            "action_history":          [],   # history 와 1:1 대응하는 액션 토큰 (prompt caching 용)
            "reward_history":          [],   # history 와 1:1 대응하는 reward 값 (prompt caching 용)
            "prev_gt_action":          TOKEN_SOLVE,
            "correct_reason":          "",
            "correct_is_boxed_verify": False,
            # phase: "generate" | "api" | "patcher" | "done"
            "phase":                   "generate",
            "api_future":              None,   # Future → (truncated, reward)
            "gen_output":              None,   # (raw_text, predicted_action, inp_ids)
            "patcher_future":          None,
            "patcher_hist":            None,
            "patcher_prev_actions":    None,
            "patcher_correct_reason":  "",
        })

    executor = ThreadPoolExecutor(max_workers=len(problems_batch) * 4)
    batch_t0 = time.time()
    iter_idx = 0

    while True:
        # ── A. API 완료된 문제 수확 ───────────────────────────────────────────
        api_done = [s for s in states if s["phase"] == "api" and s["api_future"].done()]

        # ── B. Patcher 완료된 문제 수확 ──────────────────────────────────────
        patcher_done_list = [
            s for s in states if s["phase"] == "patcher" and s["patcher_future"].done()
        ]

        # ── C. lp_items 구성 ─────────────────────────────────────────────────
        lp_items: List[Tuple[str, torch.Tensor, str]] = []
        lp_meta:  List                                = []

        for s in api_done:
            truncated, reward = s["api_future"].result()
            s["api_future"] = None
            raw_text, predicted_act, inp_ids = s["gen_output"]
            doing_correct = s["prev_gt_action"] == TOKEN_CORRECT

            if doing_correct and reward <= 0.1:
                # Branch A: patcher 비동기 제출, 이 문제는 patcher phase 로 이동
                hist_with_fail              = s["history"] + [truncated]
                patcher_prev_actions        = s["action_history"] + [TOKEN_CORRECT]
                s["patcher_hist"]           = hist_with_fail
                s["patcher_prev_actions"]   = patcher_prev_actions
                s["patcher_correct_reason"] = s["correct_reason"]
                s["patcher_future"]         = executor.submit(
                    teacher_correct_step,
                    s["problem"], hist_with_fail, s["correct_reason"], patcher_prev_actions,
                )
                s["phase"] = "patcher"
                logger.info(
                    f"{_pfx(s['problem_id'], len(s['traj'].steps))} PATCHER_SUBMIT  reward={reward:.2f}"
                )
                lp_items.append((truncated, inp_ids, TOKEN_CORRECT))
                lp_meta.append(("A_fail", s, truncated, reward, predicted_act, inp_ids))
            else:
                # Branch B: 다음 액션 결정
                if doing_correct and reward > 0.1:
                    gt_next = TOKEN_SOLVE
                elif not doing_correct and has_boxed(truncated):
                    gt_next = TOKEN_CORRECT
                elif reward <= 0.1:
                    gt_next = TOKEN_CORRECT
                else:
                    gt_next = TOKEN_SOLVE
                lp_items.append((truncated, inp_ids, gt_next))
                lp_meta.append(("B", s, truncated, reward, predicted_act, inp_ids, gt_next))

        for s in patcher_done_list:
            patcher_text_raw, _ = s["patcher_future"].result()
            s["patcher_future"] = None
            if patcher_text_raw is None:
                logger.warning(f"{_pfx(s['problem_id'])} teacher=None → skipped")
                s["phase"] = "done"
                if rollout_path:
                    save_trajectory(s["traj"], rollout_path)
                continue
            with _tok_lock:
                pt        = truncate_to_512_tokens(patcher_text_raw, tokenizer)
            pr        = compute_reward(
                s["problem"], s["patcher_hist"], pt,
                prev_rewards=s["reward_history"],
                prev_actions=s["patcher_prev_actions"],
                current_action=TOKEN_CORRECT,
            )
            p_prompt  = build_chat_prompt(
                tokenizer, SYSTEM_CORRECT,
                _correct_user(s["problem"], s["patcher_hist"], s["patcher_correct_reason"]),
            )
            p_inp_ids = tokenizer(p_prompt, return_tensors="pt")["input_ids"].cpu()
            p_gt      = TOKEN_CORRECT if pr <= 0.1 else TOKEN_SOLVE
            lp_items.append((pt, p_inp_ids, p_gt))
            lp_meta.append(("A_patcher", s, pt, pr, p_inp_ids, p_gt))
            logger.info(
                f"{_pfx(s['problem_id'], len(s['traj'].steps))} PATCHER_ARRIVED"
                f"  reward={pr:.2f}  gt={p_gt.replace('<|','').replace('|>','')}"
            )

        # ── D. log_probs 배치 계산 + 상태 업데이트 ───────────────────────────
        if lp_items:
            t0 = time.time()
            lp_results = build_gt_responses_batched(model, tokenizer, lp_items)
            logger.info(
                f"[batch] [I{iter_idx:03d}] log_probs  n={len(lp_items)}  elapsed={time.time()-t0:.2f}s"
            )

            for meta, (resp_ids, lp) in zip(lp_meta, lp_results):
                kind = meta[0]
                s    = meta[1]

                if kind == "B":
                    _, _, truncated, reward, predicted_act, inp_ids, gt_next = meta
                    doing_correct = s["prev_gt_action"] == TOKEN_CORRECT
                    action_label  = "correct" if doing_correct else "solve"
                    step_num      = len(s["traj"].steps)

                    if doing_correct and reward > 0.1:
                        is_answer_now                = s["correct_is_boxed_verify"]
                        s["correct_reason"]          = ""
                        s["correct_is_boxed_verify"] = False
                    elif not doing_correct and has_boxed(truncated):
                        s["traj"].have_boxed        = True
                        is_answer_now                = False
                        s["correct_is_boxed_verify"] = True
                        s["correct_reason"]          = "The previous step contained a boxed answer — verify it."
                    elif reward <= 0.1:
                        is_answer_now                = False
                        s["correct_is_boxed_verify"] = False
                        s["correct_reason"]          = f"The previous step was INCORRECT:\n{truncated}"
                    else:
                        is_answer_now                = False
                        s["correct_is_boxed_verify"] = False
                        s["correct_reason"]          = ""

                    s["traj"].steps.append(StepRecord(
                        step_idx=step_num, action=action_label,
                        text=truncated, reward=reward,
                        predicted_next_action=predicted_act,
                        ground_truth_next_action=gt_next,
                        input_ids=inp_ids, response_ids=resp_ids, log_probs_old=lp,
                        is_generator_step=True,
                    ))
                    s["history"].append(format_final_step(truncated))
                    s["action_history"].append(s["prev_gt_action"])
                    s["reward_history"].append(reward)
                    s["prev_gt_action"] = gt_next
                    logger.info(
                        f"{_pfx(s['problem_id'], step_num)} {action_label.upper()}"
                        f"  reward={reward:.2f}  next={gt_next.replace('<|','').replace('|>','')}"
                        f"  answer={is_answer_now}"
                    )
                    if is_answer_now:
                        s["traj"].is_answer = True
                        s["phase"] = "done"
                        logger.info(
                            f"{_pfx(s['problem_id'])} DONE  status=ANSWER"
                            f"  total_steps={len(s['traj'].steps)}"
                        )
                        if rollout_path:
                            save_trajectory(s["traj"], rollout_path)
                    elif len(s["traj"].steps) >= MAX_STEPS:
                        s["phase"] = "done"
                    else:
                        s["phase"] = "generate"

                elif kind == "A_fail":
                    _, _, truncated, reward, predicted_act, inp_ids = meta
                    step_num = len(s["traj"].steps)
                    s["traj"].steps.append(StepRecord(
                        step_idx=step_num, action="correct",
                        text=truncated, reward=reward,
                        predicted_next_action=predicted_act,
                        ground_truth_next_action=TOKEN_CORRECT,
                        input_ids=inp_ids, response_ids=resp_ids, log_probs_old=lp,
                        is_generator_step=True,
                    ))
                    s["history"].append(truncated)
                    s["action_history"].append(TOKEN_CORRECT)
                    s["reward_history"].append(reward)
                    logger.info(
                        f"{_pfx(s['problem_id'], step_num)} CORRECT(fail)"
                        f"  reward={reward:.2f}  → patcher"
                    )
                    # phase 는 이미 "patcher" 로 설정됨

                elif kind == "A_patcher":
                    _, _, pt, pr, p_inp_ids, p_gt = meta
                    step_num          = len(s["traj"].steps)
                    patcher_is_answer = s["correct_is_boxed_verify"] or has_boxed(pt)

                    s["traj"].steps.append(StepRecord(
                        step_idx=step_num, action="correct",
                        text=pt, reward=pr,
                        predicted_next_action=p_gt,
                        ground_truth_next_action=p_gt,
                        input_ids=p_inp_ids, response_ids=resp_ids, log_probs_old=lp,
                        is_generator_step=False,
                    ))

                    if pr <= 0.1:
                        if s["correct_is_boxed_verify"] or has_boxed(pt):
                            s["traj"].have_boxed = True
                        s["traj"].patcher_wrong = True
                        logger.info(
                            f"{_pfx(s['problem_id'], step_num)} PATCHER"
                            f"  reward={pr:.2f}  → WRONG  total={len(s['traj'].steps)}"
                        )
                        _log_traj(s["traj"], s["problem_id"], "PATCHER_WRONG")
                        s["phase"] = "done"
                        if rollout_path:
                            save_trajectory(s["traj"], rollout_path)
                    else:
                        s["history"].append(pt)
                        s["action_history"].append(TOKEN_CORRECT)
                        s["reward_history"].append(pr)
                        logger.info(
                            f"{_pfx(s['problem_id'], step_num)} PATCHER"
                            f"  reward={pr:.2f}  answer={patcher_is_answer}"
                        )
                        if patcher_is_answer:
                            s["traj"].have_boxed = True
                            s["traj"].is_answer = True
                            s["phase"] = "done"
                            logger.info(
                                f"{_pfx(s['problem_id'])} DONE  status=ANSWER"
                                f"  total_steps={len(s['traj'].steps)}"
                            )
                            if rollout_path:
                                save_trajectory(s["traj"], rollout_path)
                        elif len(s["traj"].steps) >= MAX_STEPS:
                            s["phase"] = "done"
                        else:
                            s["prev_gt_action"]          = TOKEN_SOLVE
                            s["correct_is_boxed_verify"] = False
                            s["correct_reason"]          = ""
                            s["phase"] = "generate"

        # ── E. GPU generate: phase=="generate" 인 문제들 ─────────────────────
        gen_ready = [s for s in states if s["phase"] == "generate"]
        if gen_ready:
            prompts = []
            for s in gen_ready:
                doing_correct = s["prev_gt_action"] == TOKEN_CORRECT
                if doing_correct:
                    prompt = build_chat_prompt(
                        tokenizer, SYSTEM_CORRECT,
                        _correct_user(s["problem"], s["history"], s["correct_reason"]),
                    )
                else:
                    prompt = build_chat_prompt(
                        tokenizer, SYSTEM_SOLVE,
                        _solve_user(s["problem"], s["history"]),
                    )
                prompts.append(prompt)

            pids = [s["problem_id"] for s in gen_ready]
            t0   = time.time()
            with _tok_lock:
                gen_outputs = generate_steps_batched(model, tokenizer, prompts)
            logger.info(
                f"[batch] [I{iter_idx:03d}] GPU_GEN  batch={len(prompts)}  elapsed={time.time()-t0:.2f}s"
                f"  pids={pids}"
            )

            # API 즉시 비동기 제출 → GPU 는 다음 loop 에서 바로 다시 사용 가능
            for s, gen_out in zip(gen_ready, gen_outputs):
                s["gen_output"]  = gen_out
                s["api_future"]  = executor.submit(
                    _trunc_reward,
                    s["problem"], s["history"], s["reward_history"], gen_out[0],
                    s["action_history"], s["prev_gt_action"],
                )
                s["phase"] = "api"

        # ── F. 종료/대기 판단 ─────────────────────────────────────────────────
        if all(s["phase"] == "done" for s in states):
            break

        nothing_new = not gen_ready and not api_done and not patcher_done_list
        if nothing_new:
            # 완료 대기: CPU spin 없이 첫 Future 가 끝날 때까지 block
            pending_futures = (
                [s["api_future"]     for s in states if s["phase"] == "api"]
                + [s["patcher_future"] for s in states if s["phase"] == "patcher"]
            )
            if pending_futures:
                _futures_wait(pending_futures, return_when=FIRST_COMPLETED)
            else:
                break

        # iter 요약
        n_done    = sum(1 for s in states if s["phase"] == "done")
        n_gen     = sum(1 for s in states if s["phase"] == "generate")
        n_api     = sum(1 for s in states if s["phase"] == "api")
        n_patcher = sum(1 for s in states if s["phase"] == "patcher")
        logger.info(
            f"[batch] [I{iter_idx:03d}] done={n_done}  gen={n_gen}"
            f"  api={n_api}  patch={n_patcher}  elapsed={time.time()-batch_t0:.1f}s"
        )
        iter_idx += 1

    executor.shutdown(wait=False)

    # MAX_STEPS 도달 미완료 저장
    for s in states:
        if s["phase"] != "done":
            logger.info(
                f"{_pfx(s['problem_id'])} DONE  status=MAX_STEPS"
                f"  total_steps={len(s['traj'].steps)}"
            )
            if rollout_path:
                save_trajectory(s["traj"], rollout_path)

    n_answer    = sum(1 for s in states if s["traj"].is_answer)
    total_steps = sum(len(s["traj"].steps) for s in states)
    logger.info(
        f"[batch] BATCH_DONE  problems={len(states)}  answered={n_answer}"
        f"  total_steps={total_steps}  iters={iter_idx}"
        f"  elapsed={time.time()-batch_t0:.1f}s"
    )

    return [s["traj"] for s in states]


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 로드
# ─────────────────────────────────────────────────────────────────────────────

# 데이터셋 prompt 컬럼 끝의 지시문 제거 (모델 시스템 프롬프트로 대체)
_TRAILING_INSTRUCTION = re.compile(
    r"\s*Please reason step by step,?\s*and put your final answer within \\boxed\{\}\.\s*$",
    re.IGNORECASE,
)


def _extract_problem(ex: dict) -> str:
    """데이터 레코드에서 문제 텍스트를 추출.

    지원 형식:
      - prompt: [{"role": "user", "content": "..."}]  ← deepmath 형식
      - problem / question 컬럼
    trailing 지시문("Please reason step by step...")은 제거.
    """
    text = ""
    if "prompt" in ex and ex["prompt"]:
        msgs = ex["prompt"]
        # user 메시지 content 사용
        for msg in msgs:
            if msg.get("role") == "user":
                text = msg["content"]
                break
        if not text:
            text = msgs[0].get("content", "")
    elif "problem" in ex:
        text = ex["problem"]
    elif "question" in ex:
        text = ex["question"]

    return _TRAILING_INSTRUCTION.sub("", text).strip()


def _extract_answer(ex: dict) -> str:
    """데이터 레코드에서 정답을 추출."""
    for key in ("final_answer", "answer", "ground_truth"):
        val = ex.get(key)
        if val:
            return str(val).strip()
    # reward_model 딕셔너리 안에 있는 경우
    rm = ex.get("reward_model")
    if isinstance(rm, dict):
        for key in ("ground_truth", "answer"):
            if rm.get(key):
                return str(rm[key]).strip()
    return ""


def load_problems(parquet_path: str = DATASET_PATH) -> List[dict]:
    """parquet에서 {problem_id, problem, answer} 목록을 반환."""
    from datasets import load_dataset as hf_load
    ds = hf_load("parquet", data_files=parquet_path, split="train")

    problems = []
    for i, ex in enumerate(ds):
        problem = _extract_problem(ex)
        answer  = _extract_answer(ex)
        if not problem:
            continue
        problems.append({
            "problem_id": str(ex.get("problem_id", ex.get("extra_info", {}).get("index", i))),
            "problem":    problem,
            "answer":     answer,
        })

    logger.info(f"[load_problems] {len(problems)}개 로드 ({parquet_path})")
    return problems


def load_math500(parquet_path: str = MATH500_PATH) -> List[dict]:
    """MATH-500 parquet에서 {problem_id, problem, answer, subject, level} 목록 반환."""
    from datasets import load_dataset as hf_load
    ds = hf_load("parquet", data_files=parquet_path, split="train")

    problems = []
    for i, ex in enumerate(ds):
        problem = _extract_problem(ex)
        answer  = _extract_answer(ex)
        if not problem:
            continue
        problems.append({
            "problem_id": str(ex.get("unique_id", i)),
            "problem":    problem,
            "answer":     answer,
            "subject":    ex.get("subject", "unknown"),
            "level":      str(ex.get("level", "unknown")),
        })

    logger.info(f"[load_math500] {len(problems)}개 로드 ({parquet_path})")
    return problems


def validate_math500(
    model,
    tokenizer,
    val_problems: List[dict],
    batch_size: int = VAL_BATCH_SIZE,
) -> dict:
    """MATH-500 validation. 외부 API 없이 generator만으로 step-by-step 풀기.

    각 문제를 배치 GPU 추론으로 처리:
      - 매 라운드: phase=="generate" 문제들을 batch_size 단위로 generate
      - \\boxed{} 포함 step 생성 시 → check_solved 로 정답 확인 후 완료
      - MAX_STEPS 도달 시 오답 처리
    """
    from collections import defaultdict

    states = [{
        "problem":  p["problem"],
        "answer":   p["answer"],
        "subject":  p.get("subject", "unknown"),
        "level":    p.get("level",   "unknown"),
        "history":  [],
        "done":     False,
        "correct":  False,
    } for p in val_problems]

    was_training = model.training
    model.eval()

    try:
        for _step in range(MAX_STEPS):
            active = [s for s in states if not s["done"]]
            if not active:
                break

            # batch_size 단위로 나눠 GPU generate
            for i in range(0, len(active), batch_size):
                mini = active[i: i + batch_size]
                prompts = [
                    build_chat_prompt(
                        tokenizer, SYSTEM_SOLVE,
                        _solve_user(s["problem"], s["history"]),
                    )
                    for s in mini
                ]
                with torch.no_grad():
                    gen_outputs = generate_steps_batched(model, tokenizer, prompts)

                for s, (step_text, _, _) in zip(mini, gen_outputs):
                    s["history"].append(step_text)
                    if has_boxed(step_text):
                        s["done"]    = True
                        s["correct"] = check_solved(step_text, s["answer"])

    finally:
        if was_training:
            model.train()

    # ── 지표 계산 ────────────────────────────────────────────────────────────
    n          = len(states)
    n_correct  = sum(1 for s in states if s["correct"])

    by_level:   dict = defaultdict(lambda: {"c": 0, "t": 0})
    by_subject: dict = defaultdict(lambda: {"c": 0, "t": 0})
    for s in states:
        by_level[s["level"]]["t"]       += 1
        by_level[s["level"]]["c"]       += int(s["correct"])
        by_subject[s["subject"]]["t"]   += 1
        by_subject[s["subject"]]["c"]   += int(s["correct"])

    metrics: dict = {
        "val/accuracy": n_correct / n if n else 0.0,
        "val/n_correct": n_correct,
        "val/n_total":   n,
    }
    for lvl, d in sorted(by_level.items()):
        metrics[f"val/acc_level_{lvl}"] = d["c"] / d["t"] if d["t"] else 0.0
    for subj, d in sorted(by_subject.items()):
        safe = subj.lower().replace(" ", "_")
        metrics[f"val/acc_{safe}"] = d["c"] / d["t"] if d["t"] else 0.0

    logger.info(
        f"[validate_math500] accuracy={n_correct}/{n}"
        f" ({100*n_correct/n:.1f}%)" if n else "[validate_math500] n=0"
    )
    return metrics
