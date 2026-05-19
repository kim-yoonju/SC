"""
rollout_sft.py
두 모델(inference + classification)을 사용한 trajectory 생성.

  Inference model  : 수학 풀이 스텝 생성 (추론만)
  Classification model : 스텝 평가 → fast/deep critique, fail rubrics, next action 판별

흐름:
  1. Inference 모델이 한 스텝 생성
  2. Classification 모델이 평가 → next_action 결정
  3. next_action에 따라:
       <|solve|>   → history 추가, 다음 스텝 생성
       <|rethink|> → gen_rethink_inference로 1회 재시도
                     재시도 실패 시 substep 분해 (최대 MAX_SUBSTEP_DEPTH)
       <|end|>     → trajectory 완성

실행 예시:
  python source/rollout_sft.py \
    --inference_model checkpoints/sft/20260518_170909_sft_inference/epoch3 \
    --cls_model checkpoints/sft/20260518_172258_sft_classification/epoch3 \
    --data_path datasets/deepmath_16k/single_reasoning_wrong_16k.jsonl \
    --inf_gpu 0 --cls_gpu 1
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import (
    get_inference_prompts,
    get_rethink_inference_prompt,
    get_classification_prompt,
    RUBRIC_TOKENS,
)
from utils_sft import (
    build_messages_classification,
    setup_tokenizer,
    TOKEN_SOLVE, TOKEN_RETHINK, TOKEN_END, TOKEN_NONE,
    CONF,
)
from utils_math import extract_boxed, has_boxed, check_solved

_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

_GT_CFG           = CONF.get("generate_trajectory", {})
BATCH_PER_GPU     = _GT_CFG.get("batch_per_gpu", 64)
MAX_STEPS         = _GT_CFG.get("max_steps", 20)
MAX_SUBSTEP_DEPTH = _GT_CFG.get("max_substep_depth", 2)
INF_MAX_NEW_TOKENS = _GT_CFG.get("max_new_tokens", 2048)
CLS_MAX_NEW_TOKENS = 512

# rubric token → rubric name 역매핑
_TOKEN_TO_RUBRIC: dict[str, str] = {v: k for k, v in RUBRIC_TOKENS.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 모델 로딩
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, gpu_id: int):
    from transformers import AutoModelForCausalLM
    tokenizer = setup_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{gpu_id}"},
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_step_manager(gpu_id: int | None = None):
    """step_manager 모델 로드 (substep 분해용). 경로 없으면 None 반환."""
    from utils import STEP_MANAGER_GPU, STEP_MANAGER_PATH
    if not STEP_MANAGER_PATH:
        return None, None
    _gpu = gpu_id if gpu_id is not None else STEP_MANAGER_GPU
    tokenizer = setup_tokenizer(STEP_MANAGER_PATH)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        STEP_MANAGER_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{_gpu}"},
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 문제별 상태
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProblemState:
    item:             dict
    history:          list = field(default_factory=list)  # classification 통과된 스텝
    all_steps:        list = field(default_factory=list)  # 오류 포함 전체 스텝
    is_rethink:       bool = False   # 현재 rethink 모드
    rethink_tried:    bool = False   # rethink 1회 소진
    substep_tried:    bool = False   # substep 분해 시도 여부
    in_substep_mode:  bool = False   # substep 풀이 중
    substep_queue:    list = field(default_factory=list)  # [{goal, depth}, ...]
    last_wrong_step:  str  = ""      # rethink 전 원본 틀린 스텝 텍스트
    last_cls_out:     dict = field(default_factory=dict)  # rethink context용 cls 출력
    done:             bool = False
    fail_reason:      str  = ""      # "rethink_fail" | "substep_fail" | "max_steps"


# ─────────────────────────────────────────────────────────────────────────────
# 배치 생성
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    stop_token_ids: list[int] | None = None,
    max_new_tokens: int = 2048,
) -> list[str]:
    """prompts 배치를 생성해 텍스트 리스트 반환. stop token이 있으면 텍스트에 append."""
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]

    stop_ids = list({tokenizer.eos_token_id, *(stop_token_ids or [])} - {None})

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=stop_ids,
    )
    resp_all = out[:, input_len:]

    stop_set = set(stop_ids)
    id_to_tok = {tokenizer.convert_tokens_to_ids(t): t
                 for t in [TOKEN_SOLVE, TOKEN_RETHINK, TOKEN_END]
                 if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id}

    results = []
    for i in range(len(prompts)):
        resp = resp_all[i]
        trim = resp.shape[0]
        stopped_action = None
        for pos, tid in enumerate(resp.tolist()):
            if tid in stop_set:
                trim = pos
                stopped_action = id_to_tok.get(tid)
                break
        text = tokenizer.decode(resp[:trim], skip_special_tokens=False).strip()
        if stopped_action:
            text = text + "\n" + stopped_action
        results.append(text)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Classification 출력 파싱
# ─────────────────────────────────────────────────────────────────────────────

_FAST_RE      = re.compile(r"Fast\s+critic\s*:(.*?)(?=\n\nDeep\s+critic\s*:|$)", re.DOTALL | re.I)
_DEEP_RE      = re.compile(r"Deep\s+critic\s*:(.*?)(?=\n\nFail\s+rubrics\s*:|$)", re.DOTALL | re.I)
_FAIL_RE      = re.compile(r"Fail\s+rubrics\s*:\s*\n(.+?)(?=\n\nNext\s+action\s*:|$)", re.DOTALL | re.I)
_NEXT_ACT_RE  = re.compile(r"Next\s+action\s*:", re.I)
_FAST_LINE_RE = re.compile(r"^\s*(.+?):\s*(correct|incorrect)(?:\s*[—–-]\s*(.+))?$", re.I)
_DEEP_LINE_RE = re.compile(r"^\s*(.+?):\s*(.*?)\s*Verdict\s*:\s*(correct|incorrect)", re.I)


def parse_fail_rubrics(text: str) -> list[str]:
    m = _FAIL_RE.search(text)
    if not m:
        return []
    section = m.group(1).strip()
    if not section or section.lower().rstrip(".") in ("none", TOKEN_NONE.lower()):
        return []
    tokens = re.findall(r"<\|[^|>]+\|>", section)
    result = []
    for tok in tokens:
        if tok == TOKEN_NONE:
            continue
        name = _TOKEN_TO_RUBRIC.get(tok, tok)
        result.append(name)
    return result


def parse_next_action(text: str) -> str:
    # Next action: 섹션 내 action token 우선
    for tok in [TOKEN_END, TOKEN_RETHINK, TOKEN_SOLVE]:
        m = re.search(r"Next\s+action\s*:.*?" + re.escape(tok), text, re.DOTALL | re.I)
        if m:
            return tok
    # 텍스트 끝 3줄에서 탐색 (stop token으로 잘린 경우)
    tail = "\n".join(text.splitlines()[-3:])
    for tok in [TOKEN_END, TOKEN_RETHINK, TOKEN_SOLVE]:
        if tok in tail:
            return tok
    return TOKEN_SOLVE  # fallback


def parse_fast_critique(text: str) -> dict:
    m = _FAST_RE.search(text)
    if not m:
        return {}
    result = {}
    for line in m.group(1).splitlines():
        lm = _FAST_LINE_RE.match(line)
        if lm:
            rubric   = lm.group(1).strip()
            verdict  = lm.group(2).strip().lower()
            critique = (lm.group(3) or "").strip()
            result[rubric] = {"verdict": verdict, "critique": critique}
    return result


def parse_deep_critique(text: str) -> list:
    m = _DEEP_RE.search(text)
    if not m:
        return []
    result = []
    for line in m.group(1).splitlines():
        lm = _DEEP_LINE_RE.match(line)
        if lm:
            rubric   = lm.group(1).strip()
            critique = lm.group(2).strip()
            verdict  = lm.group(3).strip().lower()
            result.append({"rubric": rubric, "critique": critique, "verdict": verdict})
    return result


def parse_cls_output(text: str) -> dict:
    return {
        "fast_critique": parse_fast_critique(text),
        "deep_critique": parse_deep_critique(text),
        "fail_rubrics":  parse_fail_rubrics(text),
        "next_action":   parse_next_action(text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트 빌더
# ─────────────────────────────────────────────────────────────────────────────

_STEP_PREFIX_RE = re.compile(r"^Step\s+\d+[:.]\s*", re.I)


def _history_lines(all_steps: list) -> list[str]:
    result = []
    for s in all_steps:
        if s.get("is_error", False):
            continue
        inf = s.get("inference") or ""
        inf = _STEP_PREFIX_RE.sub("", inf.strip())
        if inf:
            result.append(inf)
    return result


def build_inf_prompt(tok, system: str, problem: str, all_steps: list,
                     substep_goal: str | None = None) -> str:
    """일반 풀이 or substep 풀이용 inference 프롬프트."""
    history = _history_lines(all_steps)
    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Previous steps]")
        lines.extend(history)
    if substep_goal:
        lines.append(f"\n[Current goal]\n{substep_goal}")
        lines.append("\nWrite EXACTLY ONE reasoning step to achieve this goal.")
    else:
        lines.append("\nWrite the next step.")
    user_msg = "\n".join(lines)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_rethink_prompt(tok, system: str, problem: str, all_steps: list,
                         wrong_step: str, cls_out: dict) -> str:
    """rethink inference 프롬프트 — cls 출력의 오류 컨텍스트 포함."""
    history = _history_lines(all_steps)
    fail_rubrics  = cls_out.get("fail_rubrics") or []
    fast_critique = cls_out.get("fast_critique") or {}
    deep_critique = cls_out.get("deep_critique") or []

    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Previous steps]")
        lines.extend(history)
    lines.append(f"\n[Failed step]\n{wrong_step}")

    error_lines = []
    if fail_rubrics:
        error_lines.append(f"Failed rubrics: {', '.join(fail_rubrics)}")
    # fast critique에서 incorrect인 rubric의 critique 추가
    for rubric, data in fast_critique.items():
        if data.get("verdict") == "incorrect" and data.get("critique"):
            error_lines.append(f"- {rubric}: {data['critique']}")
    # deep critique verdict 추가
    for entry in deep_critique:
        if entry.get("verdict") == "incorrect" and entry.get("critique"):
            error_lines.append(f"  → {entry['rubric']}: {entry['critique']}")
    if error_lines:
        lines.append("\n[Why it failed]")
        lines.extend(error_lines)

    lines.append("\nWrite a corrected step using a completely different approach.")
    user_msg = "\n".join(lines)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_cls_prompt(tok, system: str, problem: str, all_steps: list, step_k: int) -> str:
    """classification 프롬프트."""
    sys_str, user_str = build_messages_classification(problem, all_steps, step_k, system)
    msgs = [{"role": "system", "content": sys_str}, {"role": "user", "content": user_str}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ─────────────────────────────────────────────────────────────────────────────
# Substep 분해
# ─────────────────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = (
    "You are a math tutor. Determine if the given reasoning step is atomic "
    "(cannot be split further) or can be decomposed into two independent sub-steps. "
    "Output:\n"
    "Verdict: correct  (if atomic)\n"
    "Verdict: incorrect  (if decomposable)\n"
    "Sub-step A: <goal of first sub-step>\n"
    "Sub-step B: <goal of second sub-step>"
)


def decompose_step(
    sm_model, sm_tok,
    problem: str,
    history_texts: list[str],
    wrong_step: str,
) -> list[dict] | None:
    """step_manager로 substep 분해 시도. 분해 불가(atomic)면 None 반환."""
    if sm_model is None:
        return None

    from utils_sft import build_chat_prompt as _build_chat
    prev = "\n\n".join(history_texts) if history_texts else "(none)"
    user_msg = (
        f"Problem:\n{problem}\n\n"
        f"Previous steps (correct):\n{prev}\n\n"
        f"Current Step:\n{wrong_step}"
    )
    prompt = _build_chat(sm_tok, _DECOMPOSE_SYSTEM, user_msg)
    inputs = sm_tok(prompt, return_tensors="pt").to(sm_model.device)
    with torch.no_grad():
        out = sm_model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=sm_tok.pad_token_id,
        )
    resp = sm_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    if not re.search(r"Verdict:\s*incorrect", resp, re.I):
        return None  # atomic

    m_a = re.search(r"Sub-step A:\s*(.*?)(?=Sub-step B:|$)", resp, re.DOTALL | re.I)
    m_b = re.search(r"Sub-step B:\s*(.*?)$", resp, re.DOTALL | re.I)
    sub1 = m_a.group(1).strip() if m_a else ""
    sub2 = m_b.group(1).strip() if m_b else ""
    if not sub1 or not sub2:
        return None
    return [{"goal": sub1, "depth": 0}, {"goal": sub2, "depth": 0}]


# ─────────────────────────────────────────────────────────────────────────────
# Classification 결과 → State 업데이트
# ─────────────────────────────────────────────────────────────────────────────

def _process_cls(
    state: ProblemState,
    cls_out: dict,
    sm_model,
    sm_tok,
) -> None:
    next_action = cls_out["next_action"]
    step = state.all_steps[-1]

    # ── <|end|> ──────────────────────────────────────────────────────────────
    if next_action == TOKEN_END:
        step["is_error"] = False
        state.history.append(step)
        state.in_substep_mode = False
        state.substep_queue   = []
        state.done            = True
        return

    # ── substep 모드 ─────────────────────────────────────────────────────────
    if state.in_substep_mode:
        cur_depth = state.substep_queue[0].get("depth", 0) if state.substep_queue else 0

        if next_action == TOKEN_SOLVE:
            step["is_error"] = False
            state.history.append(step)
            state.substep_queue.pop(0)
            state.is_rethink = False

            if not state.substep_queue:
                # 모든 substep 통과 → 일반 모드로 복귀
                state.in_substep_mode = False
                state.substep_tried   = False
            return

        # substep <|rethink|> → 더 쪼개거나 종료
        step["is_error"] = True
        if cur_depth < MAX_SUBSTEP_DEPTH and state.substep_queue:
            history_texts = [s["inference"] for s in state.history]
            sub_substeps = decompose_step(
                sm_model, sm_tok,
                state.item.get("problem", ""),
                history_texts,
                step["inference"],
            )
            if sub_substeps:
                for s in sub_substeps:
                    s["depth"] = cur_depth + 1
                state.substep_queue = sub_substeps + state.substep_queue[1:]
                state.is_rethink    = True
                logger.debug(f"substep 재분해 depth={cur_depth+1}")
                return

        # 더 못 쪼갬 → 종료
        state.in_substep_mode = False
        state.substep_queue   = []
        state.done            = True
        state.fail_reason     = "substep_fail"
        return

    # ── 일반 모드 ─────────────────────────────────────────────────────────────
    if next_action == TOKEN_SOLVE:
        step["is_error"] = False
        state.history.append(step)
        state.is_rethink    = False
        state.rethink_tried = False
        state.substep_tried = False
        return

    # <|rethink|>
    step["is_error"] = True

    if not state.rethink_tried:
        # 1차 실패 → rethink 1회
        state.is_rethink       = True
        state.rethink_tried    = True
        state.last_wrong_step  = step["inference"]
        state.last_cls_out     = cls_out
        return

    if not state.substep_tried:
        # rethink도 실패 → substep 분해 시도
        state.substep_tried = True
        state.is_rethink    = False
        history_texts = [s["inference"] for s in state.history]
        substeps = decompose_step(
            sm_model, sm_tok,
            state.item.get("problem", ""),
            history_texts,
            state.last_wrong_step,
        )
        if substeps:
            state.substep_queue   = substeps
            state.in_substep_mode = True
            state.is_rethink      = True
            logger.debug(f"substep 분해 성공: {[s['goal'][:40] for s in substeps]}")
        else:
            # atomic → 종료
            state.done        = True
            state.fail_reason = "rethink_fail"
        return

    # substep도 실패 → 종료
    state.done        = True
    state.fail_reason = "rethink_fail"


# ─────────────────────────────────────────────────────────────────────────────
# 메인 배치 루프
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch(
    problems:     list[dict],
    inf_model,    inf_tok,
    cls_model,    cls_tok,
    sm_model,     sm_tok,
    batch_size:   int = BATCH_PER_GPU,
    max_steps:    int = MAX_STEPS,
) -> list[dict]:
    system_gen     = get_inference_prompts()[0]   # gen_inference
    system_rethink = get_rethink_inference_prompt()
    system_cls     = get_classification_prompt()

    # classification model의 action stop token IDs
    cls_stop_tids = []
    for tok in [TOKEN_SOLVE, TOKEN_RETHINK, TOKEN_END]:
        tid = cls_tok.convert_tokens_to_ids(tok)
        if tid and tid != cls_tok.unk_token_id:
            cls_stop_tids.append(tid)

    states = [ProblemState(item=p) for p in problems]

    for round_idx in range(max_steps):
        active = [s for s in states if not s.done]
        if not active:
            break

        # ── Inference 프롬프트 빌드 ───────────────────────────────────────────
        inf_prompts = []
        for state in active:
            problem = state.item["problem"]
            if state.in_substep_mode:
                goal   = state.substep_queue[0]["goal"]
                prompt = build_inf_prompt(inf_tok, system_gen, problem,
                                          state.all_steps, substep_goal=goal)
            elif state.is_rethink:
                prompt = build_rethink_prompt(inf_tok, system_rethink, problem,
                                              state.all_steps,
                                              state.last_wrong_step,
                                              state.last_cls_out)
            else:
                prompt = build_inf_prompt(inf_tok, system_gen, problem, state.all_steps)
            inf_prompts.append(prompt)

        # ── Inference 배치 생성 (서브배치 분할) ──────────────────────────────
        inf_outputs = []
        for i in range(0, len(inf_prompts), batch_size):
            inf_outputs.extend(
                batch_generate(inf_model, inf_tok, inf_prompts[i:i + batch_size],
                               stop_token_ids=None,
                               max_new_tokens=INF_MAX_NEW_TOKENS)
            )

        # ── State에 pending step 등록 ─────────────────────────────────────────
        for state, inf_text in zip(active, inf_outputs):
            if state.in_substep_mode:
                step_state = "gen_substep"
            elif state.is_rethink:
                step_state = "gen_rethink"
            else:
                step_state = "gen_solve"
            state.all_steps.append({
                "step_idx":  len(state.all_steps),
                "inference": inf_text,
                "state":     step_state,
                "is_error":  False,  # cls가 결정 (임시 False)
            })

        # ── Classification 프롬프트 빌드 ─────────────────────────────────────
        cls_prompts = []
        for state in active:
            step_k  = len(state.all_steps) - 1
            problem = state.item["problem"]
            cls_prompts.append(
                build_cls_prompt(cls_tok, system_cls, problem, state.all_steps, step_k)
            )

        # ── Classification 배치 생성 ──────────────────────────────────────────
        cls_raw_outputs = []
        for i in range(0, len(cls_prompts), batch_size):
            cls_raw_outputs.extend(
                batch_generate(cls_model, cls_tok, cls_prompts[i:i + batch_size],
                               stop_token_ids=cls_stop_tids,
                               max_new_tokens=CLS_MAX_NEW_TOKENS)
            )

        # ── 결과 파싱 + state 업데이트 ────────────────────────────────────────
        for state, cls_raw in zip(active, cls_raw_outputs):
            cls_out = parse_cls_output(cls_raw)
            step    = state.all_steps[-1]
            step["cls_raw"]       = cls_raw
            step["fast_critique"] = cls_out["fast_critique"]
            step["deep_critique"] = cls_out["deep_critique"]
            step["fail_rubrics"]  = cls_out["fail_rubrics"]
            step["next_action"]   = cls_out["next_action"]
            _process_cls(state, cls_out, sm_model, sm_tok)

    # max_steps 초과로 미완성 상태 처리
    for state in states:
        if not state.done:
            state.done        = True
            state.fail_reason = "max_steps"

    return [_build_traj(s) for s in states]


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory 빌드
# ─────────────────────────────────────────────────────────────────────────────

def _build_traj(state: ProblemState) -> dict:
    item        = state.item
    problem     = item.get("problem", "")
    gold_answer = item.get("answer") or item.get("gold_answer", "")

    # 마지막 통과 스텝에서 boxed 답 추출
    pred_answer = ""
    for step in reversed(state.history):
        boxed = extract_boxed(step.get("inference", ""))
        if boxed:
            pred_answer = boxed
            break

    is_right = bool(pred_answer and check_solved(pred_answer, gold_answer))

    traj_type = "correct" if is_right else (state.fail_reason or "fail")

    return {
        "problem_id":  item.get("problem_id") or item.get("id", ""),
        "problem":     problem,
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "is_right":    is_right,
        "traj_type":   traj_type,
        "fail_reason": state.fail_reason,
        "steps":       state.all_steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────────────────────────────────────

def load_problems(data_path: str, num_start: int = 0, num_end: int | None = None) -> list[dict]:
    problems = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    problems = problems[num_start:num_end]
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dual-model SFT trajectory 생성")
    parser.add_argument("--inference_model", required=True,  help="inference 모델 경로")
    parser.add_argument("--cls_model",       required=True,  help="classification 모델 경로")
    parser.add_argument("--data_path",       default=_GT_CFG.get("base_problems"), help="문제 JSONL 경로")
    parser.add_argument("--inf_gpu",  type=int, default=0,   help="inference 모델 GPU 번호")
    parser.add_argument("--cls_gpu",  type=int, default=1,   help="classification 모델 GPU 번호")
    parser.add_argument("--sm_gpu",   type=int, default=None, help="step_manager GPU 번호 (선택)")
    parser.add_argument("--batch_size", type=int, default=BATCH_PER_GPU, help="배치 크기")
    parser.add_argument("--max_steps",  type=int, default=MAX_STEPS,     help="스텝 최대 횟수")
    parser.add_argument("--num_start",  type=int, default=_GT_CFG.get("num_start", 0))
    parser.add_argument("--num_end",    type=int, default=_GT_CFG.get("num_end"),
                        help="처리할 문제 범위 끝 인덱스 (None=전체)")
    parser.add_argument("--output", type=str, default=None, help="출력 폴더 (기본: output/rollout_sft/{ts})")
    args = parser.parse_args()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else (_ROOT / "output" / "rollout_sft" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "rollout.log", encoding="utf-8"),
        ],
    )

    print(f"[{ts}]  출력: {out_dir}")
    print(f"Inference model : {args.inference_model}  (GPU {args.inf_gpu})")
    print(f"Classification  : {args.cls_model}  (GPU {args.cls_gpu})")

    # 모델 로드
    print("모델 로딩 중...")
    inf_model, inf_tok = load_model(args.inference_model, args.inf_gpu)
    cls_model, cls_tok = load_model(args.cls_model, args.cls_gpu)
    sm_model,  sm_tok  = load_step_manager(args.sm_gpu)
    if sm_model:
        print(f"Step manager 로드 완료 (GPU {args.sm_gpu})")
    else:
        print("Step manager 없음 — substep 분해 비활성화")

    # 문제 로드
    problems = load_problems(args.data_path, args.num_start, args.num_end)
    print(f"문제 수: {len(problems)}")

    # 출력 파일
    traj_file = open(out_dir / "traj_all.jsonl", "w", encoding="utf-8")

    # 배치 처리
    total = len(problems)
    solved = fail = 0

    for batch_start in tqdm(range(0, total, args.batch_size), desc="rollout"):
        batch = problems[batch_start: batch_start + args.batch_size]
        trajs = generate_batch(
            batch,
            inf_model, inf_tok,
            cls_model, cls_tok,
            sm_model,  sm_tok,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
        )
        for traj in trajs:
            traj_file.write(json.dumps(traj, ensure_ascii=False) + "\n")
            traj_file.flush()
            if traj["is_right"]:
                solved += 1
            else:
                fail += 1

    traj_file.close()

    print(f"\n완료: 정답={solved}  실패={fail}  총={total}")
    print(f"정확도: {solved / max(total, 1):.4f}")
    print(f"출력 저장: {out_dir / 'traj_all.jsonl'}")


if __name__ == "__main__":
    main()
