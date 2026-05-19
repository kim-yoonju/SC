"""
generate_sft_trajectory.py
base_problems JSONL에서 generator → genPRM 반복으로 trajectory SFT 데이터 생성.

PRM -> local
PATCHER -> api 
쓰도록 하려했는데 local prm을 쓰기 위해 프롬프트 최적화하는게 오래걸리고 힘들어서
16k만 사용하니 그냥 모두 api 쓰자고 결정


흐름:
  1. Generator (SFT checkpoint)가 스텝별로 풀이
  2. GenPRM이 각 스텝을 순서대로 평가 → 처음 틀린 스텝을 찾음
  3. 오류 직전까지만 history에 추가 → Generator가 오류 스텝부터 재시도
  4. patcher가 실패하면 patcher_fail로 종료

병렬 처리:
  - n_parallel = PRM.batch_size 개의 문제를 동시에 처리
  - 각 라운드: 모든 활성 문제에 대해 generator 배치 추론 후 GenPRM 배치 평가
  - GenPRM은 루브릭을 하나씩 평가하되, 판정 확정된 문제는 조기 종료(early stopping)로 제외
  - patcher는 generator와 동일한 배치 추론 (gen_solve_R, 히스토리 없이)

출력 파일 (output/sft_trajectory/{timestamp}/):
  traj_gen.jsonl   generator 단독 정답 (genPRM 오류 미발견)
  traj_mix.jsonl   gen-genPRM 혼합, generator가 최종 정답
  traj_all.jsonl   위 두 가지 전부

스텝 state / next_gold_action:
  일반 gen 스텝      : state=solve       / →<|solve|>
  오류 gen 스텝      : state=solve       / →<|rethink|>
  오류 직후 첫 gen   : state=rethink_pat / →<|solve|>
  마지막 스텝        : (위와 동일)      / →<|end|>
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    CONF,
    TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END,
    extract_boxed, has_boxed, check_solved,
    load_generator, build_chat_prompt,
    _gpt, PATCHER, PATCHER_MAX_NEW_TOKENS,
)

from generate_utils import load_dataset_file

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts"

def _load_action_prompts() -> dict[str, str]:
    rubric_lines = []
    with open(_PROMPTS_PATH / "action_prompts_rubric.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                rubric_lines.append(f"{e['id']}. {e['name']}: [correct/incorrect — {e['description']}]")
    rubric_str = "\n".join(rubric_lines)
    prompts: dict[str, str] = {}
    with open(_PROMPTS_PATH / "action_prompts.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                prompts[e["name"]] = e["content"].replace("{{rubric}}", rubric_str)
    return prompts

_ACTION_PROMPTS            = _load_action_prompts()
GEN_SOLVE_PROMPT           = _ACTION_PROMPTS["gen_solve_R"]
GEN_RETHINK_PROMPT         = _ACTION_PROMPTS["gen_rethink_R"]
_WRONG_STEP_SUMMARY_PROMPT  = _ACTION_PROMPTS["step_summary"]
_CRITIQUE_SUMMARY_PROMPT    = _ACTION_PROMPTS["critique_summary"]

from local_PRM import LocalLlama, load_rubrics, build_system_prompt, PASS_N


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_GT_CFG             = CONF.get("generate_trajectory", {})
TRAJ_MAX_NEW_TOKENS = _GT_CFG.get("max_new_tokens", 4096)
PRM_MAX_NEW_TOKENS  = CONF.get("PRM", {}).get("max_new_tokens", 1024)


def _extract_verdicts_from_text(sc_text: str) -> tuple[int, int]:
    """self-check 텍스트에서 correct/incorrect 카운트 추출.

    1차: \\boxed{correct}, \\boxed{\\text{correct}} 등 boxed 패턴
    2차: ': correct' / ': incorrect' 평문 패턴
    """
    boxed = re.findall(
        r"\\boxed\{(?:\\text\{)?\s*(correct|incorrect)\s*\}+",
        sc_text, re.I,
    )
    if boxed:
        c = sum(1 for m in boxed if m.lower() == "correct")
        i = sum(1 for m in boxed if m.lower() == "incorrect")
        return c, i

    c = len(re.findall(r":\s*correct\b", sc_text, re.I))
    i = len(re.findall(r":\s*incorrect\b", sc_text, re.I))
    return c, i


def _probe_verdict_logprob(
    model, tokenizer, input_device,
    sc_text: str,
    correct_id: int | None,
    incorrect_id: int | None,
) -> tuple[int, int]:
    """텍스트 추출 실패 시 single forward pass로 correct/incorrect logit 비교.

    self_check_text 뒤에 '\\nOverall verdict: '를 붙인 뒤 마지막 토큰의
    correct vs incorrect logit을 비교해 판정 1개를 반환.
    """
    if correct_id is None or incorrect_id is None or not sc_text.strip():
        return 0, 0
    probe = sc_text.strip() + "\nOverall verdict: "
    ids = tokenizer.encode(probe, return_tensors="pt", add_special_tokens=False).to(input_device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]          # (vocab_size,)
    lp_c = logits[correct_id].item()
    lp_i = logits[incorrect_id].item()
    return (1, 0) if lp_c >= lp_i else (0, 1)
PRM_BATCH_PER_GPU   = CONF.get("PRM", {}).get("batch_per_gpu", 128)

# Step-type-aware rubric selection
# SETUP      → Logical Derivation + Calculation Correctness only
# INTERMEDIATE → all rubrics except "Step Role Appropriateness"
# CONCLUDING → all rubrics
_SETUP_RUBRIC_NAMES    = frozenset({"Logical Derivation", "Calculation Correctness"})
_STEP_ROLE_RUBRIC_NAME = "Step Role Appropriateness"

W    = 88
SEP2 = "━" * W


# ─────────────────────────────────────────────────────────────────────────────
# 문제별 상태
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProblemState:
    item:               dict
    history:            list = field(default_factory=list)   # GenPRM 검증된 정답 스텝
    all_steps:          list = field(default_factory=list)   # 오류 포함 전체 스텝
    is_rethink:         bool = False
    step_rethink_tried: bool = False
    step_patcher_tried: bool = False
    patcher_count:      int  = 0
    last_wrong_rubrics:      list = field(default_factory=list)
    last_wrong_step_text:    str  = ""
    last_wrong_step_summary: str  = ""
    use_patcher:             bool = False
    rethink_round:      int  = 0
    traj_idx:           int  = 1
    traj_gen_list:      list = field(default_factory=list)
    traj_mix_list:      list = field(default_factory=list)
    prm_records:        list = field(default_factory=list)
    done:               bool = False
    pending_step:       dict = None


# ─────────────────────────────────────────────────────────────────────────────
# Generator: 여러 문제를 배치로 한 스텝씩 생성
# ─────────────────────────────────────────────────────────────────────────────

def _batch_run_generator(
    model, tokenizer, input_device,
    states: list[ProblemState],
) -> None:
    """배치로 한 스텝씩 생성. 결과를 state.pending_step에 저장."""
    prompts = []
    for state in states:
        step_number = len(state.history) + 1
        lines = [f"[Problem]\n{state.item['problem']}"]
        if state.history:
            lines.append("\n[Previous steps]")
            for i, s in enumerate(state.history, 1):
                step_ctx = (s.get("summary") or {}).get("does") or s["text"]
                lines.append(f"Step {i}: {step_ctx}")
        lines.append(f"\nWrite Step {step_number}.")
        if state.is_rethink:
            error_explanation = state.last_wrong_step_summary or "the previous step contained an error"
            system_prompt = GEN_RETHINK_PROMPT.replace("{{error_explanation}}", error_explanation)
        else:
            system_prompt = GEN_SOLVE_PROMPT
        prompts.append(build_chat_prompt(tokenizer, system_prompt, "\n".join(lines)))

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # correct/incorrect 단일 토큰 ID (공백 prefix 포함 변형까지 시도)
    def _find_token_id(word: str) -> int | None:
        for w in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            ids = tokenizer.encode(w, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        return None
    _correct_id   = _find_token_id("correct")
    _incorrect_id = _find_token_id("incorrect")

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(input_device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=TRAJ_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    resp_all = out[:, input_len:]

    for j, state in enumerate(states):
        step_number = len(state.history) + 1
        resp = resp_all[j]

        # pad / im_end 에서 자름
        trim = resp.shape[0]
        for pos, tid in enumerate(resp.tolist()):
            if tid in (tokenizer.pad_token_id, im_end_id):
                trim = pos
                break

        full_text = tokenizer.decode(resp[:trim], skip_special_tokens=False).strip()

        # step_text / self-check 분리
        sc_idx = full_text.find("\nSelf-check:")
        if sc_idx != -1:
            step_text       = full_text[:sc_idx].strip()
            self_check_text = full_text[sc_idx:]
        else:
            step_text       = full_text.strip()
            self_check_text = ""

        # correct/incorrect 집계 → 룰 기반 액션 결정
        # 1차: 텍스트 추출 (\boxed{} 및 평문 모두 지원)
        correct_count, incorrect_count = _extract_verdicts_from_text(self_check_text)
        # 2차: 추출 실패 시 forward pass logit probe
        if correct_count == 0 and incorrect_count == 0 and self_check_text:
            correct_count, incorrect_count = _probe_verdict_logprob(
                model, tokenizer, input_device, self_check_text,
                _correct_id, _incorrect_id,
            )
        if incorrect_count > correct_count:
            pred_action = TOKEN_CORRECT
        elif has_boxed(step_text):
            pred_action = TOKEN_END
        else:
            pred_action = TOKEN_SOLVE

        # hallucination 감지
        _TOOL_CALL_MARKERS = ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")
        has_tool_call = any(m in full_text for m in _TOOL_CALL_MARKERS)

        # 프롬프트 플레이스홀더를 그대로 에코한 경우 (모델이 템플릿을 채우지 못한 것)
        _TEMPLATE_PLACEHOLDERS = ("[Your one reasoning step", "[Your one corrected reasoning step")
        has_placeholder = any(step_text.strip().startswith(p) for p in _TEMPLATE_PLACEHOLDERS)

        is_error = has_tool_call or has_placeholder
        error_reason = (
            "tool_call token hallucination" if has_tool_call  else
            "template placeholder echo"     if has_placeholder else
            None
        )

        logger.info(
            f"[Generator] id={state.item.get('id')} step={step_number} "
            f"rethink={state.is_rethink} pred_action={pred_action} "
            f"correct={correct_count} incorrect={incorrect_count}"
            + (f" [{error_reason}]" if error_reason else "")
        )

        state.pending_step = {
            "text":         step_text,
            "full_text":    full_text,
            "pred_action":  pred_action,
            "source":       "gen",
            "is_error":     is_error,
            "is_first_pat": state.is_rethink and not is_error,
            "summary":      (
                {"step_analysis": f"{error_reason} — skipped genPRM"}
                if is_error else None
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Patcher: API 모델(config.API_model.PATCHER)로 스텝 생성
# ─────────────────────────────────────────────────────────────────────────────

def _run_patcher_api(states: list[ProblemState]) -> None:
    """Patcher 상태 문제들을 API로 처리. 결과를 state.pending_step에 저장."""

    def _call_one(state: ProblemState) -> None:
        step_number = len(state.history) + 1
        problem_id  = state.item.get("id", "?")
        user_msg    = f"[Problem]\n{state.item['problem']}\n\nWrite Step {step_number}."
        messages    = [
            {"role": "system", "content": GEN_SOLVE_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        logger.info(f"[Patcher API] id={problem_id} step={step_number} model={PATCHER}")
        try:
            text = _gpt(PATCHER, messages, max_completion_tokens=PATCHER_MAX_NEW_TOKENS)
            text = text.strip()
        except Exception as e:
            logger.warning(f"[Patcher API] id={problem_id} 호출 실패: {e}")
            text = ""

        is_error    = not text
        pred_action = TOKEN_END if has_boxed(text) else TOKEN_SOLVE

        logger.info(
            f"[Patcher API] id={problem_id} step={step_number} "
            f"pred_action={pred_action} len={len(text)}"
            + (" [empty response]" if is_error else "")
        )

        state.pending_step = {
            "text":         text,
            "full_text":    text,
            "pred_action":  pred_action,
            "source":       "patcher",
            "is_error":     is_error,
            "is_first_pat": not is_error,
            "summary":      (
                {"step_analysis": "patcher API empty response — skipped genPRM"}
                if is_error else None
            ),
        }
        state.use_patcher = False

    with ThreadPoolExecutor(max_workers=max(1, len(states))) as ex:
        list(ex.map(_call_one, states))


# ─────────────────────────────────────────────────────────────────────────────
# Step 요약: generator로 현재 스텝 추론을 한 줄로 요약
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARIZE_SYSTEM = (
    "You are a math tutor. "
    "In one concise sentence, describe what the following reasoning step does mathematically. "
    "Do not evaluate correctness — only describe the action taken."
)
_SUMMARIZE_MAX_TOKENS         = 64
_CRITIQUE_SUMMARY_MAX_TOKENS  = 128


def _batch_summarize_steps(
    model, tokenizer, input_device,
    states: list[ProblemState],
) -> None:
    """pending_step의 실제 수학 추론을 generator로 한 줄 요약해 does_summary에 저장.
    tool_call 오류 스텝은 건너뜀."""
    to_summarize = [s for s in states if not s.pending_step.get("is_error")]
    for s in states:
        if s.pending_step.get("is_error"):
            s.pending_step["does_summary"] = None
    if not to_summarize:
        return

    prompts = []
    for state in to_summarize:
        step_text = (state.pending_step.get("text") or "")[:1200]
        prompts.append(build_chat_prompt(tokenizer, _SUMMARIZE_SYSTEM, f"Step:\n{step_text}"))

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(input_device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=_SUMMARIZE_MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    for j, state in enumerate(to_summarize):
        raw = tokenizer.decode(out[j, input_len:], skip_special_tokens=True).strip()
        # 첫 문장만 사용
        summary = re.split(r"[\n.]", raw)[0].strip()
        state.pending_step["does_summary"] = summary or None


def _batch_summarize_wrong_steps(
    model, tokenizer, input_device,
    states: list[ProblemState],
) -> None:
    """rethink 상태인 states의 last_wrong_step_text를 generator로 한 줄 오류 요약."""
    to_summarize = [s for s in states if s.is_rethink and s.last_wrong_step_text and not s.last_wrong_step_summary]
    if not to_summarize:
        return

    prompts = []
    for state in to_summarize:
        step_text = state.last_wrong_step_text[:1200]
        prompts.append(build_chat_prompt(tokenizer, _WRONG_STEP_SUMMARY_PROMPT, f"Step:\n{step_text}"))

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(input_device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=_SUMMARIZE_MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    for j, state in enumerate(to_summarize):
        raw = tokenizer.decode(out[j, input_len:], skip_special_tokens=True).strip()
        summary = re.split(r"[\n.]", raw)[0].strip()
        state.last_wrong_step_summary = summary or ""
        logger.info(
            f"[WrongSummary] id={state.item.get('id')} "
            f"summary={state.last_wrong_step_summary!r}"
        )


def _batch_summarize_critique(
    model, tokenizer, input_device,
    states: list[ProblemState],
    prm_results_map: dict,
) -> None:
    """PRM wrong 판정 스텝에 대해 generator로 critique 요약을 생성해 pending_step에 저장.
    - 올바른 스텝(PRM right) 또는 tool_call 오류 스텝은 critique_summary=None 으로 설정.
    """
    to_summarize = []
    for state in states:
        result = prm_results_map.get(id(state), {})
        # 이미 is_error=True(tool_call 할루시네이션)이거나 PRM이 right 판정이면 건너뜀
        if result.get("result") != "wrong" or state.pending_step.get("is_error"):
            state.pending_step["critique_summary"] = None
        else:
            to_summarize.append(state)

    if not to_summarize:
        return

    prompts = []
    for state in to_summarize:
        result    = prm_results_map[id(state)]
        details   = result.get("details", {})
        votes     = result.get("votes", {})
        step_text = (state.pending_step.get("text") or "")[:800]

        wrong_lines = []
        for rn, verdict in votes.items():
            if verdict == "wrong":
                reasoning = (details.get(rn) or {}).get("reasoning", "")
                if reasoning:
                    wrong_lines.append(f"- {rn}: {reasoning[:300]}")

        critique = "\n".join(wrong_lines) if wrong_lines else "(no detailed reasoning available)"
        user_msg = f"Step:\n{step_text}\n\nFailed criteria:\n{critique}"
        prompts.append(build_chat_prompt(tokenizer, _CRITIQUE_SUMMARY_PROMPT, user_msg))

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(input_device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=_CRITIQUE_SUMMARY_MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    for j, state in enumerate(to_summarize):
        raw = tokenizer.decode(out[j, input_len:], skip_special_tokens=True).strip()
        state.pending_step["critique_summary"] = raw or None
        logger.info(
            f"[CritiqueSummary] id={state.item.get('id')} "
            f"critique={state.pending_step['critique_summary']!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory 조립
# ─────────────────────────────────────────────────────────────────────────────

def _compute_labels(steps: list[dict], first_pat_pos: int = 0) -> list[str]:
    labels         = []
    pos            = 0
    in_rethink_run = False
    for s in steps:
        is_rethink = s.get("is_first_pat") or s["source"] == "patcher"
        if is_rethink:
            if not in_rethink_run:
                if first_pat_pos > 0 and pos < first_pat_pos:
                    pos = first_pat_pos
                in_rethink_run = True
            else:
                pos += 1
            labels.append(f"P_{pos:02d}")
        else:
            pos += 1
            in_rethink_run = False
            labels.append(f"G_{pos:02d}")
    return labels


def _build_traj(
    problem_id, problem, gold_answer,
    steps: list[dict],
    is_right: bool,
    traj_type: str,
    first_pat_pos: int = 0,
    traj_idx: int = 0,
    fail_reason: str = None,
) -> dict:
    pred_answer = None
    for s in reversed(steps):
        raw = extract_boxed(s.get("full_text") or s["text"])
        if raw:
            pred_answer = raw
            break

    labels = _compute_labels(steps, first_pat_pos)
    last   = len(steps) - 1
    step_dicts = []

    for i, (s, label) in enumerate(zip(steps, labels)):
        is_last = (i == last)
        if s["is_error"]:
            state, next_action = "solve", TOKEN_CORRECT
        elif s["is_first_pat"]:
            state, next_action = "rethink_pat", TOKEN_END if is_last else TOKEN_SOLVE
        else:
            state, next_action = "solve", TOKEN_END if is_last else TOKEN_SOLVE

        summ = s.get("summary") or {}
        if isinstance(summ, dict):
            does             = summ.get("does") or summ.get("step_analysis") or None
            rubric_votes     = summ.get("rubric_votes") or None
            rubric_text      = summ.get("rubric_text")  or None
            critique_summary = summ.get("critique_summary")
        else:
            does = rubric_votes = rubric_text = critique_summary = None

        step_dicts.append({
            "step_idx":         i,
            "step":             label,
            "inference":        s.get("full_text") or s["text"],
            "source":           s["source"],
            "is_error":         s["is_error"],
            "state":            state,
            "next_gold_action": next_action,
            "does":             does,
            "rubric_votes":     rubric_votes,
            "rubric_text":      rubric_text,
            "critique_summary": critique_summary,
        })

    return {
        "traj_id":     f"{problem_id}_{traj_idx:02d}",
        "problem_id":  str(problem_id),
        "problem":     problem,
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "is_right":    is_right,
        "traj_type":   traj_type,
        "fail_reason": fail_reason,
        "steps":       step_dicts,
    }


def _fmt(steps: list[dict]) -> str:
    parts, n = [], 0
    for s in steps:
        n += 1
        if s.get("is_error"):
            parts.append(f"[E_{n:02d}]")
        elif s.get("is_first_pat"):
            parts.append(f"R_{n:02d}")
        else:
            parts.append(f"G_{n:02d}")
    return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# PRM 결과 처리 (단일 문제)
# ─────────────────────────────────────────────────────────────────────────────

def _process_prm_result(
    state: ProblemState,
    result: dict,
    step_rubrics: list[dict],
    save_fn,
    save_intermediate_fn,
    prm_save_fn,
) -> None:
    """PRM 결과로 state 업데이트. 완료 시 state.done = True."""
    step        = state.pending_step
    step_number = len(state.history) + 1
    problem_id  = state.item.get("id", "?")
    problem     = state.item["problem"]
    gold_answer = state.item["answer"]

    wrong_count   = result["wrong_count"]
    total         = result["total"]
    votes         = result["votes"]
    details       = result["details"]
    wrong_rubrics = [r for r, v in votes.items() if v == "wrong"]
    rubric_votes  = [{"rubric": rn, "verdict": votes.get(rn, "right")} for rn in votes]
    rubric_text   = [
        {
            "rubric":       rn,
            "reasoning":    details[rn].get("reasoning"),
            "verdict_text": details[rn].get("verdict_text"),
        }
        for rn in votes if rn in details
    ]

    # PRM 레코드 수집
    prev_lines    = [f"Step {i+1}: {s['text']}" for i, s in enumerate(state.history)]
    now_step_text = f"Step {step_number}: {step['text']}"
    for rubric in step_rubrics:
        rname = rubric["name"]
        d     = details.get(rname, {})
        rec   = {
            "problem_id":        problem_id,
            "step_global_idx":   step_number,
            "question":          problem,
            "previous_steps":    "\n".join(prev_lines),
            "now_step":          now_step_text,
            "rubric_name":       rname,
            "pred":              votes.get(rname),
            "prob_correct":      d.get("prob_correct"),
            "prob_incorrect":    d.get("prob_incorrect"),
            "reasoning":         d.get("reasoning"),
            "verdict_text":      d.get("verdict_text"),
            "full_response":     d.get("full_response"),
            "method":            d.get("method"),
            "step_final_result": result["result"],
            "step_wrong_count":  wrong_count,
            "step_total":        total,
        }
        state.prm_records.append(rec)
        if prm_save_fn:
            prm_save_fn(rec)

    does_summary = step.get("does_summary")

    def _apply_wrong(reason_suffix: str = ""):
        """step을 wrong으로 처리하고 rethink/patcher/abort 분기."""
        step["is_error"]     = True
        step["is_first_pat"] = False
        step["summary"]      = {
            "does":             does_summary,
            "critique_summary": step.get("critique_summary"),
            "step_analysis":    (
                f"LocalPRM: {wrong_count}/{total} rubrics flagged wrong"
                f" ({', '.join(wrong_rubrics)})"
                + (f" [{reason_suffix}]" if reason_suffix else "")
            ),
            "rubric_votes": rubric_votes,
            "rubric_text":  rubric_text,
            "votes":        votes,
            "details":      details,
        }
        state.all_steps.append(step)

        # 스텝 구성 출력
        for _i, _s in enumerate(state.all_steps, 1):
            _lbl = "[E]" if _s.get("is_error") else ("R" if _s.get("is_first_pat") else "G")
            _src = _s.get("source", "gen")
            _does = ((_s.get("summary") or {}).get("does") or "—")[:80]
            print(f"    {_i:2d}. [{_lbl}|{_src}] {_does}")

        if save_intermediate_fn:
            save_intermediate_fn(
                _build_traj(problem_id, problem, gold_answer,
                            state.all_steps, False, "mix_intermediate",
                            traj_idx=state.traj_idx)
            )
            state.traj_idx += 1

        if not state.step_rethink_tried:
            # 1차 실패 → rethink
            state.is_rethink              = True
            state.step_rethink_tried      = True
            state.last_wrong_step_text    = step["text"]
            state.last_wrong_step_summary = ""
            print(f"  [id={problem_id}]  step={step_number}  {_fmt(state.all_steps)}  → rethink")
        elif not state.step_patcher_tried:
            # rethink도 실패 → patcher 1회
            state.use_patcher          = True
            state.is_rethink           = False
            state.step_patcher_tried   = True
            state.patcher_count       += 1
            state.last_wrong_rubrics   = wrong_rubrics
            state.last_wrong_step_text = step["text"]   # rethink step 텍스트로 갱신
            print(f"  [id={problem_id}]  step={step_number}  {_fmt(state.all_steps)}  → patcher")
        else:
            # patcher도 실패 → 종료
            print(f"  [id={problem_id}]  step={step_number}  {_fmt(state.all_steps)}  → patcher_fail")
            traj = _build_traj(problem_id, problem, gold_answer,
                               state.all_steps, False, "mix",
                               fail_reason="patcher_fail", traj_idx=state.traj_idx)
            state.traj_mix_list.append(traj)
            if save_fn:
                save_fn(traj, "mix")
            state.traj_idx += 1
            state.done = True
            return
        state.rethink_round += 1

    # ── 오류 있음 ─────────────────────────────────────────────────────────────
    if result["result"] == "wrong":
        logger.info(
            f"[LocalPRM] 오류 id={problem_id} step={step_number} "
            f"wrong={wrong_count}/{total} rubrics={wrong_rubrics}"
        )
        # patcher_miss: patcher가 정답을 생성했지만 PRM이 wrong으로 잘못 판정한 경우
        if (step.get("source") == "patcher" and
                has_boxed(step["text"]) and
                check_solved(step["text"], gold_answer)):
            logger.info(f"[id={problem_id}] patcher_miss: patcher correct but PRM wrong")
            print(f"  [id={problem_id}]  step={step_number}  {_fmt(state.all_steps)}  → patcher_miss")
            step["is_error"]     = True
            step["is_first_pat"] = False
            state.all_steps.append(step)
            traj = _build_traj(problem_id, problem, gold_answer,
                               state.all_steps, False, "mix",
                               fail_reason="patcher_miss", traj_idx=state.traj_idx)
            state.traj_mix_list.append(traj)
            if save_fn:
                save_fn(traj, "mix")
            state.traj_idx += 1
            state.done = True
            return
        _apply_wrong()

    # ── 오류 없음 ─────────────────────────────────────────────────────────────
    else:
        logger.info(f"[LocalPRM] id={problem_id} step={step_number}: correct wrong={wrong_count}/{total}")

        # boxed 정답이 있으면 gold_answer와 직접 비교해 종료 여부 결정
        if has_boxed(step["text"]):
            is_right = check_solved(step["text"], gold_answer)
            if not is_right:
                # genPRM 오탐: pred ≠ gold_answer → wrong으로 처리
                logger.info(
                    f"[LocalPRM] id={problem_id} step={step_number}: "
                    f"approved but pred≠gold_answer → force wrong"
                )
                print(f"  [id={problem_id}]  step={step_number}  → gold mismatch, force wrong")
                _apply_wrong("pred≠gold_answer")
                return

        # 정상 정답 스텝: history에 추가
        step["summary"] = {
            "does":             does_summary,
            "critique_summary": None,
            "rubric_votes":     rubric_votes,
            "rubric_text":      rubric_text,
        }
        state.all_steps.append(step)
        state.history.append(step)
        state.is_rethink         = False
        state.step_rethink_tried = False
        state.step_patcher_tried = False

        if has_boxed(step["text"]):
            # is_right=True (위에서 check_solved 통과)
            if state.rethink_round == 0:
                print(f"  ✓  [id={problem_id}]  {_fmt(state.all_steps)}  → correct (gen only)")
                traj = _build_traj(problem_id, problem, gold_answer,
                                   state.all_steps, True, "gen", traj_idx=state.traj_idx)
                state.traj_gen_list.append(traj)
                if save_fn:
                    save_fn(traj, "gen")
            else:
                print(f"  ✓  [id={problem_id}]  {_fmt(state.all_steps)}  → correct (mix)")
                traj = _build_traj(problem_id, problem, gold_answer,
                                   state.all_steps, True, "mix", traj_idx=state.traj_idx)
                state.traj_mix_list.append(traj)
                if save_fn:
                    save_fn(traj, "mix")
            state.traj_idx += 1
            state.done = True


# ─────────────────────────────────────────────────────────────────────────────
# 병렬 배치 생성 루프
# ─────────────────────────────────────────────────────────────────────────────

def _parallel_gen(fn, generators: list, states: list) -> None:
    """states를 generators에 round-robin 분배하고 GPU별 병렬 실행."""
    n = len(generators)
    splits = [states[i::n] for i in range(n)]
    def _run(i):
        if splits[i]:
            model, tokenizer, device = generators[i]
            fn(model, tokenizer, device, splits[i])
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_run, range(n)))


def generate_batch(
    items: list[dict],
    generators: list,
    prm_model: LocalLlama,
    rubrics: list[dict],
    n_parallel: int,
    save_fn=None,
    save_intermediate_fn=None,
    prm_save_fn=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    n_parallel개 문제를 동시에 처리.
    Returns: (all_traj_gen, all_traj_mix, all_prm_records)
    """
    queue  = list(items)
    active: list[ProblemState] = []
    all_traj_gen:    list[dict] = []
    all_traj_mix:    list[dict] = []
    all_prm_records: list[dict] = []

    # Step-type-aware rubric subsets (computed once from the loaded rubric list)
    rubric_sets = {
        "SETUP":        [r for r in rubrics if r["name"] in _SETUP_RUBRIC_NAMES],
        "INTERMEDIATE": [r for r in rubrics if r["name"] != _STEP_ROLE_RUBRIC_NAME],
        "CONCLUDING":   rubrics,
    }

    pbar = tqdm(total=len(items), desc="generating", unit="prob")

    def _flush_done():
        nonlocal active
        newly_done = [s for s in active if s.done]
        for s in newly_done:
            all_traj_gen.extend(s.traj_gen_list)
            all_traj_mix.extend(s.traj_mix_list)
            all_prm_records.extend(s.prm_records)
            pbar.update(1)
        active = [s for s in active if not s.done]
        while len(active) < n_parallel and queue:
            active.append(ProblemState(item=queue.pop(0)))

    # 초기 active 채우기
    while len(active) < n_parallel and queue:
        active.append(ProblemState(item=queue.pop(0)))

    while active:
        _flush_done()
        if not active:
            break

        # ── Patcher API 추론 / Generator 배치 추론 ──────────────────────────────
        patcher_states = [s for s in active if s.use_patcher]
        gen_states     = [s for s in active if not s.use_patcher]
        logger.info(
            f"[Generator batch] n={len(gen_states)}  "
            f"[Patcher API] n={len(patcher_states)}"
        )
        if patcher_states:
            _run_patcher_api(patcher_states)
        if gen_states:
            _parallel_gen(_batch_run_generator, generators, gen_states)

        # ── 스텝 요약 (generator로 현재 스텝 추론 한 줄 요약) ─────────────────────
        logger.info(f"[Summarize] n={len(active)}")
        _parallel_gen(_batch_summarize_steps, generators, active)

        # ── GenPRM 배치 평가 (tool_call hallucination 스텝은 제외) ───────────────
        prm_states, skip_states = [], []
        for state in active:
            if state.pending_step.get("is_error"):
                skip_states.append(state)
            else:
                prm_states.append(state)

        prm_results_map:    dict[int, dict]       = {}
        state_step_rubrics: dict[int, list[dict]] = {}

        if prm_states:
            # Classify step type and select rubric subset per state
            for state in prm_states:
                text = state.pending_step["text"]
                if has_boxed(text):
                    step_type = "CONCLUDING"
                elif len(state.history) == 0:
                    step_type = "SETUP"
                else:
                    step_type = "INTERMEDIATE"
                state.pending_step["step_type"] = step_type
                state_step_rubrics[id(state)]   = rubric_sets[step_type]

            rubrics_per_step = [state_step_rubrics[id(s)] for s in prm_states]

            questions, prev_steps_list, now_steps_list = [], [], []
            for state in prm_states:
                step_number = len(state.history) + 1
                prev_lines  = [f"Step {i+1}: {s['text']}" for i, s in enumerate(state.history)]
                now_text    = f"Step {step_number}: {state.pending_step['text']}"
                questions.append(state.item["problem"])
                prev_steps_list.append("\n".join(prev_lines))
                now_steps_list.append(now_text)

            type_counts = {}
            for s in prm_states:
                t = s.pending_step.get("step_type", "?")
                type_counts[t] = type_counts.get(t, 0) + 1
            logger.info(
                f"[LocalPRM batch] n_problems={len(prm_states)}  "
                f"setup={type_counts.get('SETUP',0)}  "
                f"intermediate={type_counts.get('INTERMEDIATE',0)}  "
                f"concluding={type_counts.get('CONCLUDING',0)}  "
                f"skipped(tool_call)={len(skip_states)}"
            )
            prm_results = prm_model.evaluate_steps_batch(
                rubrics_per_step,
                questions, prev_steps_list, now_steps_list,
                max_new_tokens=PRM_MAX_NEW_TOKENS,
            )
            for state, result in zip(prm_states, prm_results):
                prm_results_map[id(state)] = result

        # tool_call hallucination → per-state wrong 결과로 직접 처리
        for state in skip_states:
            text = state.pending_step["text"]
            if has_boxed(text):
                step_type = "CONCLUDING"
            elif len(state.history) == 0:
                step_type = "SETUP"
            else:
                step_type = "INTERMEDIATE"
            state.pending_step["step_type"] = step_type
            step_rubs = rubric_sets[step_type]
            state_step_rubrics[id(state)] = step_rubs
            n_rubs = len(step_rubs)
            pid = state.item.get("id", "?")
            logger.info(f"[Generator] id={pid} tool_call hallucination → force wrong ({step_type})")
            prm_results_map[id(state)] = {
                "result":      "wrong",
                "wrong_count": n_rubs,
                "total":       n_rubs,
                "threshold":   n_rubs // 2 + 1,
                "votes":       {r["name"]: "wrong" for r in step_rubs},
                "details":     {},
            }

        # ── Critique 요약 (PRM wrong 판정 스텝에 대해 generator로 요약) ──────────
        logger.info(f"[CritiqueSummary] n={len(active)}")
        _parallel_gen(
            lambda m, t, d, sts: _batch_summarize_critique(m, t, d, sts, prm_results_map),
            generators, active,
        )

        # ── 결과 처리 ─────────────────────────────────────────────────────────
        for state in active:
            result    = prm_results_map[id(state)]
            step_rubs = state_step_rubrics.get(id(state), rubrics)
            _process_prm_result(state, result, step_rubs, save_fn, save_intermediate_fn, prm_save_fn)

        # ── rethink 대상 wrong step 오류 요약 ────────────────────────────────
        rethink_states = [s for s in active if s.is_rethink and not s.done]
        if rethink_states:
            logger.info(f"[WrongSummary] n={len(rethink_states)}")
            _parallel_gen(_batch_summarize_wrong_steps, generators, rethink_states)

        _flush_done()

    pbar.close()
    logger.info(
        f"완료 → gen={len(all_traj_gen)}  mix={len(all_traj_mix)}  "
        f"prm_records={len(all_prm_records)}"
    )
    return all_traj_gen, all_traj_mix, all_prm_records


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trajectory SFT 데이터 생성")
    parser.add_argument("--num_data",    type=int, default=None)
    parser.add_argument("--offset",      type=int, default=0)
    parser.add_argument("--output",      type=str, default=None,
                        help="출력 폴더 경로 (기본: output/sft_trajectory/{timestamp})")
    parser.add_argument("--rubric_file", type=str, default=None,
                        help="루브릭 jsonl 경로")
    parser.add_argument("--n_parallel",  type=int, default=None,
                        help="동시 처리 문제 수 (기본: PRM.batch_size // n_rubrics)")
    args = parser.parse_args()

    root   = Path(__file__).resolve().parent.parent
    gt_cfg = CONF.get("generate_trajectory", {})

    dataset_path = (
        gt_cfg.get("base_problems")
        or str(root / CONF["data_path"]["deepmath_16k"])
    )

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir     = Path(args.output) if args.output else (root / "output" / "sft_trajectory" / ts)
    prm_out_dir = root / "output" / "genPRM" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    prm_out_dir.mkdir(parents=True, exist_ok=True)

    # ── 로깅 설정 ────────────────────────────────────────────────────────────
    import sys as _sys
    log_path    = out_dir / "run.log"
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    root_logger.addHandler(file_handler)

    class _Tee:
        def __init__(self, *streams): self._streams = streams
        def write(self, data):
            for s in self._streams: s.write(data)
        def flush(self):
            for s in self._streams: s.flush()

    _log_file   = open(log_path, "a", encoding="utf-8")
    _sys.stdout = _Tee(_sys.__stdout__, _log_file)

    # ── 출력 파일 ────────────────────────────────────────────────────────────
    files = {
        k: open(out_dir / f"traj_{k}.jsonl", "w", encoding="utf-8")
        for k in ("gen", "mix", "all")
    }
    prm_eval_file = open(prm_out_dir / "prm_evals.jsonl", "w", encoding="utf-8")

    num_data = args.num_data or gt_cfg.get("num_data", 1)

    # ── GPU 설정 ─────────────────────────────────────────────────────────────
    rollout_gpus = gt_cfg.get("rollout_gpus", [0])
    prm_gpu_ids    = CONF.get("PRM", {}).get("gpu_id", [1])
    n_prm_gpus     = len(prm_gpu_ids)
    PRM_BATCH_SIZE = PRM_BATCH_PER_GPU * n_prm_gpus
    logger.info(f"prm_gpu_ids={prm_gpu_ids}  rollout_gpus={rollout_gpus}")

    # ── 데이터 & 루브릭 로드 ─────────────────────────────────────────────────
    items = load_dataset_file(dataset_path)
    items = items[args.offset:] if num_data == -1 else items[args.offset: args.offset + num_data]
    logger.info(f"로드된 문제 수: {len(items)}")

    rubric_path = args.rubric_file or CONF.get("PRM", {}).get("rubric_file")
    if not rubric_path:
        raise ValueError("루브릭 파일 경로를 config.PRM.rubric_file 또는 --rubric_file 인수로 지정해 주세요.")
    if not Path(rubric_path).is_absolute():
        rubric_path = str(root / rubric_path)
    rubrics    = load_rubrics(rubric_path)
    n_rubrics  = len(rubrics)
    # GenPRM이 루브릭을 하나씩 순차 평가(조기 종료)하므로 한 번에 n_problems개만 처리.
    # 이전에는 n_problems × n_rubrics를 한 배치로 처리해 PRM_BATCH_SIZE // n_rubrics로 제한했으나,
    # 이제 루브릭당 최대 n_parallel개 아이템만 올라가므로 PRM_BATCH_SIZE까지 가능.
    n_parallel = args.n_parallel or PRM_BATCH_SIZE

    logger.info(
        f"데이터셋={dataset_path}  sft출력={out_dir}  prm출력={prm_out_dir}  "
        f"num_data={num_data}  offset={args.offset}  "
        f"PRM_BATCH_SIZE={PRM_BATCH_SIZE}  n_rubrics={n_rubrics}  n_parallel={n_parallel}"
    )

    # ── Generator 로드 (generate_trajectory.rollout_gpus의 각 GPU에 1개씩) ──
    base_model_id = CONF["checkpoint"]["base"]
    generators = []
    for gpu_id in rollout_gpus:
        device_map = {"": f"cuda:{gpu_id}"}
        logger.info(f"Generator 로딩 중: {base_model_id}  device_map={device_map}")
        model, tokenizer = load_generator(model_path=base_model_id, device_map=device_map)
        generators.append((model, tokenizer, next(model.parameters()).device))
        logger.info(f"Generator 로드 완료 (device={generators[-1][2]})")
    logger.info(f"Generator {len(generators)}개 로드 완료")

    # ── Local PRM 로드 (PRM.gpu_id로 CUDA_VISIBLE_DEVICES 설정 후 vLLM 초기화) ──
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in prm_gpu_ids)
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (prm only)")
    prm_model_id = CONF.get("PRM", {}).get("model_id")
    if not prm_model_id:
        raise ValueError("config.yaml에 PRM.model_id가 없습니다.")
    prm_cache_dir = CONF.get("checkpoint", {}).get("cache_dir", "/tmp")
    logger.info(f"LocalPRM 로딩 중: {prm_model_id}  gpu={prm_gpu_ids}  batch_size={PRM_BATCH_SIZE}")
    prm_model = LocalLlama(
        model_path=prm_model_id,
        cache_dir=prm_cache_dir,
        batch_size=PRM_BATCH_SIZE,
        tensor_parallel_size=n_prm_gpus,
    )
    logger.info("LocalPRM 로드 완료")
    logger.info(f"루브릭 로드 완료: {n_rubrics}개  ({rubric_path})")

    # ── 저장 함수 ─────────────────────────────────────────────────────────────
    counts = {"gen": 0, "mix": 0}

    def _save(traj: dict, traj_type: str):
        line = json.dumps(traj, ensure_ascii=False) + "\n"
        files[traj_type].write(line); files[traj_type].flush()
        files["all"].write(line);     files["all"].flush()
        counts[traj_type] += 1

    def _save_intermediate(traj: dict):
        line = json.dumps(traj, ensure_ascii=False) + "\n"
        files["all"].write(line); files["all"].flush()

    def _save_prm_record(rec: dict):
        prm_eval_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        prm_eval_file.flush()

    t_start = time.time()

    try:
        generate_batch(
            items,
            generators,
            prm_model, rubrics,
            n_parallel=n_parallel,
            save_fn=_save,
            save_intermediate_fn=_save_intermediate,
            prm_save_fn=_save_prm_record,
        )
    finally:
        for f in files.values():
            f.close()
        prm_eval_file.close()
        _sys.stdout = _sys.__stdout__
        _log_file.close()

    elapsed_min = (time.time() - t_start) / 60
    total_traj  = sum(counts.values())
    logger.info(
        f"완료: {len(items)}개 문제 / {total_traj}개 trajectory  "
        f"gen={counts['gen']}  mix={counts['mix']}  소요={elapsed_min:.1f}분  "
        f"sft출력={out_dir}  prm출력={prm_out_dir}"
    )


if __name__ == "__main__":
    main()
