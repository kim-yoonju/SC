"""
generate_sft_trajectory.py
base_problems JSONL에서 generator → PRM_log 반복으로 trajectory SFT 데이터 생성.

PRM -> api (config.API_model.PRM)
PATCHER -> api (config.API_model.PATCHER)
GENERATOR -> local (config.checkpoint.base)


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
  traj_all.jsonl   모든 완결 trajectory (gen 단독 + gen-PRM_log 혼합)

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
import threading
import time
import torch
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    CONF,
    TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END,
    load_generator, load_generator_vllm, load_step_manager, build_chat_prompt,
    _call_llm, PATCHER, PATCHER_MAX_NEW_TOKENS, PATCHER_THINKING_BUDGET,
    _record_usage, _print_cost_summary, set_run_log, set_call_role,
    set_problem_context, run_log_direct,
    STEP_MANAGER_GPU, STEP_MANAGER_PATH,
)
from utils_math import extract_boxed, has_boxed, check_solved

from generate_utils import (
    load_dataset_file,
    _prm_is_fail, _extract_verdicts_from_text,
)

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts"
_ROOT_PATH    = Path(__file__).resolve().parent.parent

def _load_action_prompts() -> dict[str, str]:
    rubric_lines = []
    _prm_sect    = CONF.get("PRM")
    _rubric_rel  = _prm_sect.get("rubric")
    if not _rubric_rel:
        raise KeyError("config.PRM.rubric 설정이 없습니다")
    rubric_file  = Path(_rubric_rel) if Path(_rubric_rel).is_absolute() else _ROOT_PATH / _rubric_rel
    with open(rubric_file, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                e = json.loads(line)
                rubric_lines.append(f"{e['name']}: [correct/incorrect — {e['criterion']}]")
    rubric_str = "\n".join(rubric_lines)
    prompts: dict[str, str] = {}
    with open(_PROMPTS_PATH / "action_prompts.json", encoding="utf-8") as f:
        for e in json.load(f):
            prompts[e["name"]] = e["content"].replace("{{rubric}}", rubric_str)
    return prompts

_ACTION_PROMPTS            = _load_action_prompts()
GEN_SOLVE_PROMPT           = _ACTION_PROMPTS["gen_solve_R"]
GEN_RETHINK_PROMPT         = _ACTION_PROMPTS["gen_rethink_R"]
PAT_SOLVE_PROMPT           = _ACTION_PROMPTS["pat_solve"]
_CRITIQUE_REVIEW_PROMPT    = _ACTION_PROMPTS["critique_review"]
_SUMMARIZE_SYSTEM          = _ACTION_PROMPTS["step_summary_system"]
_WRONG_STEP_SUMMARIZE_SYSTEM = _ACTION_PROMPTS["wrong_step_summary_system"]
_RUBRIC_DOES_SYSTEM        = _ACTION_PROMPTS["rubric_does_system"]
_CRITIQUE_REVIEW_SYSTEM    = _ACTION_PROMPTS["critique_review_system"]

from PRM import (
    ApiPrm, ApiPrmBatch, ApiPrmTwoStage,
    evaluate_step, load_rubrics, load_fast_rubric,
)

_PRM_API       = CONF.get("PRM", {}).get("model_id")


def _setup_logging(log_path=None):
    """로그를 콘솔 + 파일에 동시 기록. log_path=None이면 콘솔만."""
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

_setup_logging()   # 실행 즉시 콘솔 로깅 시작
logger = logging.getLogger(__name__)

_GT_CFG = CONF.get("generate_trajectory")
TRAJ_MAX_NEW_TOKENS  = _GT_CFG["max_new_tokens"]
TRAJ_MAX_STEPS       = _GT_CFG.get("max_steps")
MAX_SUBSTEP_DEPTH    = _GT_CFG.get("max_substep_depth", 2)

# ── 비교 실험용 오버라이드 ───────────────────────────────────────────────────
# None  → config.yaml의 use_vllm 설정을 그대로 사용
# True  → vLLM 강제 (GPU 2개를 tensor_parallel=2로 묶어 단일 인스턴스)
# False → HF 강제 (GPU마다 독립 모델: GPU[0]=gen, GPU[1]=sum, lock 없이 병렬)
_FORCE_VLLM: "bool | None" = None
# ────────────────────────────────────────────────────────────────────────────

USE_VLLM = _GT_CFG["use_vllm"] if _FORCE_VLLM is None else _FORCE_VLLM
_PRM_CFG = CONF.get("PRM")
PRM_MAX_NEW_TOKENS        = _PRM_CFG["max_new_tokens"]
PRM_STAGE1_MAX_NEW_TOKENS = _PRM_CFG["stage1_max_new_tokens"]

PRM_BATCH_PER_GPU   = _PRM_CFG["batch_per_gpu"]


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
    last_wrong_rubrics:           list = field(default_factory=list)
    last_wrong_step_text:         str  = ""
    last_wrong_rubric_details:    dict = field(default_factory=dict)  # {rubric_name: prm_response}
    last_wrong_does:              str  = ""
    last_wrong_gen_deep_critique: dict = field(default_factory=dict)  # {rubric_name: {verdict, critique}}
    last_wrong_prm_deep_critique: list = field(default_factory=list)  # [{rubric, verdict, critique}]
    last_wrong_answer:            str  = ""   # pred≠gold_answer일 때 모델이 낸 틀린 답
    use_patcher:             bool = False
    rethink_round:      int  = 0
    traj_list:          list = field(default_factory=list)
    prm_records:        list = field(default_factory=list)
    done:               bool = False
    pending_step:       dict = field(default_factory=dict)
    # ── substep 분해 ────────────────────────────────────────────────────────────
    step_substep_tried: bool = False       # substep 분해 시도 여부
    in_substep_mode:    bool = False       # 현재 서브스텝 풀이 중
    substep_queue:      list = field(default_factory=list)   # [{goal, depth}, ...]
    substep_passed:     list = field(default_factory=list)   # 통과된 서브스텝 텍스트
    substep_depth:      int  = 0           # 현재 분해 깊이


# ─────────────────────────────────────────────────────────────────────────────
# Generator: 여러 문제를 배치로 한 스텝씩 생성
# ─────────────────────────────────────────────────────────────────────────────

def _build_rethink_explanation(state: "ProblemState") -> str:
    """rethink 프롬프트용 오류 설명 — 틀린 스텝 does + PRM critique + guidance (Atomicity 제외)."""
    parts = []

    if state.last_wrong_does:
        parts.append(f"[What was attempted]\n{state.last_wrong_does}")

    prm_deep = state.last_wrong_prm_deep_critique or []
    fail_entries = [e for e in prm_deep if e.get("verdict") == "incorrect" and e.get("rubric") != "Atomicity"]

    if not fail_entries and state.last_wrong_answer:
        parts.append(
            f"[Why it failed]\n"
            f"Your answer \\boxed{{{state.last_wrong_answer}}} was verified to be incorrect. "
            f"All rubrics passed, meaning the reasoning steps looked valid, "
            f"but the final answer is wrong — the approach itself is flawed. "
            f"Abandon this method entirely and try a fundamentally different approach."
        )

    if fail_entries:
        parts.append("[Fail rubrics]\n" + "\n".join(f"- {e['rubric']}" for e in fail_entries))

        why_lines = [
            f"- {e['rubric']}: {e['critique'].strip()}"
            for e in fail_entries if e.get("critique")
        ]
        if why_lines:
            parts.append("[Why it failed]")
            parts.extend(why_lines)

        how_lines = [
            f"- {e['rubric']}: {g}"
            for e in fail_entries
            if (g := _RETHINK_GUIDANCE.get(e["rubric"], ""))
        ]
        if how_lines:
            parts.append("[How to fix]")
            parts.extend(how_lines)

    if state.in_substep_mode and state.substep_queue:
        current = state.substep_queue[0]
        parts.append(f"[Focus ONLY on]: {current['goal']}")

    return "\n".join(parts) if parts else "the previous step contained an error — try a completely different approach"


# ── Atomicity 기반 서브스텝 분해 ──────────────────────────────────────────────

def _load_atomicity_system_prompt() -> str:
    _prm_cfg = CONF.get("PRM", {})
    _prompts_cfg = CONF.get("prompts", {})
    _rubric_rel = _prm_cfg.get("rubric", _prompts_cfg.get("rubric_file", "prompts/prm_rubric_v6.1.jsonl"))
    path = Path(__file__).resolve().parent.parent / _rubric_rel
    if not path.exists():
        return ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("name") == "Atomicity":
                return entry.get("system_prompt", "")
    return ""

_DECOMPOSE_SYSTEM = _load_atomicity_system_prompt()


def _load_rethink_guidance() -> dict[str, str]:
    _prm_cfg = CONF.get("PRM", {})
    _prompts_cfg = CONF.get("prompts", {})
    _rubric_rel = _prm_cfg.get("rubric", _prompts_cfg.get("rubric_file", "prompts/prm_rubric_v6.1.jsonl"))
    path = Path(__file__).resolve().parent.parent / _rubric_rel
    if not path.exists():
        return {}
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            name = entry.get("name")
            guidance = entry.get("rethink_guidance")
            if name and guidance:
                result[name] = guidance
    return result

_RETHINK_GUIDANCE: dict[str, str] = _load_rethink_guidance()


def _decompose_with_atomicity(
    problem: str,
    history: list[str],
    wrong_step: str,
    sm_model,
    sm_tok,
) -> list[dict] | None:
    """step_manager로 Atomicity 기준 분해 시도.
    반환: [{"goal": ...}, {"goal": ...}] 또는 None (atomic이면 분해 불가)
    """
    if sm_model is None:
        return None

    prev = "\n\n".join(history) if history else "(none)"
    user_msg = (
        f"Problem:\n{problem}\n\n"
        f"Previous steps (correct):\n{prev}\n\n"
        f"Current Step:\n{wrong_step}"
    )
    prompt = build_chat_prompt(sm_tok, _DECOMPOSE_SYSTEM, user_msg)
    inputs = sm_tok(prompt, return_tensors="pt").to(sm_model.device)
    with torch.no_grad():
        out = sm_model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=sm_tok.pad_token_id,
        )
    resp = sm_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # v7.7 출력 형식: CoT 후 "Verdict: incorrect/correct" + "Sub-step A/B:" 섹션
    if not re.search(r"Verdict:\s*incorrect", resp, re.I):
        return None   # atomic → patcher 호출

    m_a = re.search(r"Sub-step A:\s*(.*?)(?=Sub-step B:|Independence:|Verdict:|$)", resp, re.DOTALL | re.I)
    m_b = re.search(r"Sub-step B:\s*(.*?)(?=Independence:|Verdict:|$)", resp, re.DOTALL | re.I)
    sub1 = m_a.group(1).strip() if m_a else ""
    sub2 = m_b.group(1).strip() if m_b else ""
    if not sub1 or not sub2:
        return None
    return [{"goal": sub1, "depth": 0}, {"goal": sub2, "depth": 0}]



# ─────────────────────────────────────────────────────────────────────────────
# vLLM 공통 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _vllm_texts(llm, prompts: list[str], max_new_tokens: int,
                stop_token_ids: list[int] | None = None) -> list[str]:
    """vLLM batch generate → decoded text 리스트 반환."""
    from vllm import SamplingParams
    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        stop_token_ids=stop_token_ids or None,
    )
    return [o.outputs[0].text for o in llm.generate(prompts, params, use_tqdm=False)]


def _batch_run_generator_vllm(
    llm, tokenizer, _device,
    states: list["ProblemState"],
) -> None:
    """vLLM로 한 스텝씩 생성. 결과를 state.pending_step에 저장."""
    prompts = []
    messages_list = []
    for state in states:
        step_number = len(state.history) + 1
        lines = [f"[Problem]\n{state.item['problem']}"]
        if state.is_rethink:
            system_prompt = GEN_RETHINK_PROMPT
            if state.history:
                if state.last_wrong_answer:
                    # pred≠gold_answer: 이전 스텝들도 틀린 접근법의 일부
                    lines.append(
                        "\n[Previous approach — produced a wrong final answer, DO NOT continue from this]"
                    )
                    for i, s in enumerate(state.history, 1):
                        step_ctx = (s.get("summary") or {}).get("does") or s.get("inference") or s["text"]
                        lines.append(f"Step {i}: {step_ctx}")
                    lines.append("⚠️ The approach above is WRONG. Start over with a different method.")
                else:
                    lines.append("\n[Already completed — do NOT repeat or restate any of these]")
                    for i, s in enumerate(state.history, 1):
                        step_ctx = (s.get("summary") or {}).get("does") or s.get("inference") or s["text"]
                        lines.append(f"Step {i}: {step_ctx}")
            lines.append(f"\n[Previous step attempt — INCORRECT]\n{_build_rethink_explanation(state)}")
        else:
            system_prompt = GEN_SOLVE_PROMPT
            if state.history:
                lines.append("\n[Previous steps]")
                for i, s in enumerate(state.history, 1):
                    step_ctx = (s.get("summary") or {}).get("does") or s.get("inference") or s["text"]
                    lines.append(f"Step {i}: {step_ctx}")
        lines.append(f"\nWrite Step {step_number}.")
        user_content = "\n".join(lines)
        messages_list.append([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ])
        prompts.append(build_chat_prompt(tokenizer, system_prompt, user_content))

    _action_ids = [
        tokenizer.convert_tokens_to_ids(t)
        for t in [TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END]
    ]
    _action_ids = [tid for tid in _action_ids if tid is not None and tid != tokenizer.unk_token_id]
    _eos_ids    = list({tokenizer.eos_token_id, *_action_ids} - {None})

    _tok_to_action = {
        tokenizer.convert_tokens_to_ids(TOKEN_CORRECT): TOKEN_CORRECT,
        tokenizer.convert_tokens_to_ids(TOKEN_END):     TOKEN_END,
        tokenizer.convert_tokens_to_ids(TOKEN_SOLVE):   TOKEN_SOLVE,
    }

    from vllm import SamplingParams
    params = SamplingParams(
        max_tokens=TRAJ_MAX_NEW_TOKENS,
        temperature=0,
        stop_token_ids=_eos_ids,
    )
    outputs = llm.generate(prompts, params, use_tqdm=False)

    for i, (output, state) in enumerate(zip(outputs, states)):
        step_number = len(state.history) + 1
        full_text   = output.outputs[0].text

        # step_text / self-correction 분리
        _sc_m = re.search(
            r"\n(?:Self-correction:|(?:\d+\.\s*)?\*{0,2}(?:Fast\s+critic|fast_critique)\*{0,2}\s*:|Algebraic\s+Manipulation\s*:)",
            full_text, re.IGNORECASE,
        )
        if _sc_m:
            step_text       = full_text[:_sc_m.start()].strip()
            self_check_text = full_text[_sc_m.start():]
        else:
            step_text       = full_text.strip()
            self_check_text = ""

        # stop_reason으로 액션 결정 (int 이면 stop_token_ids 중 하나)
        stop_reason = output.outputs[0].stop_reason
        if isinstance(stop_reason, int) and stop_reason in _tok_to_action:
            pred_action = _tok_to_action[stop_reason]
        else:
            correct_count, incorrect_count, _ = _extract_verdicts_from_text(self_check_text)
            if incorrect_count > correct_count:
                pred_action = TOKEN_CORRECT
            elif has_boxed(step_text):
                pred_action = TOKEN_END
            else:
                pred_action = TOKEN_SOLVE

        _TOOL_CALL_MARKERS    = ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")
        _TEMPLATE_PLACEHOLDERS = ("[Your one reasoning step", "[Your one corrected reasoning step")
        has_tool_call   = any(m in full_text for m in _TOOL_CALL_MARKERS)
        has_placeholder = any(step_text.strip().startswith(p) for p in _TEMPLATE_PLACEHOLDERS)
        is_error        = has_tool_call or has_placeholder
        error_reason    = (
            "tool_call token hallucination" if has_tool_call else
            "template placeholder echo"     if has_placeholder else None
        )

        logger.info(
            f"[Generator vLLM] id={state.item.get('id')} step={step_number} "
            f"rethink={state.is_rethink} pred_action={pred_action}"
            + (f" [{error_reason}]" if error_reason else "")
        )

        run_log_direct({
            "ts":         datetime.now().isoformat(timespec="seconds"),
            "role":       "rethink" if state.is_rethink else "generator",
            "model":      CONF["checkpoint"]["base"],
            "problem_id": str(state.item.get("id", "?")),
            "step":       step_number,
            "in_tok":     len(output.prompt_token_ids) if output.prompt_token_ids else None,
            "out_tok":    len(output.outputs[0].token_ids) if output.outputs else None,
            "messages":   messages_list[i],
            "output":     full_text,
        })

        state.pending_step = {
            "text":             step_text,
            "inference":        step_text,
            "full_text":        full_text,
            "pred_action":      pred_action,
            "next_pred_action": _parse_next_pred_action(self_check_text),
            "source":           "gen",
            "role":             "rethink" if state.is_rethink else "gen",
            "was_rethink":      state.is_rethink,
            "is_error":         is_error,
            "is_first_pat":     state.is_rethink and not is_error,
            "summary":          (
                {"step_analysis": f"{error_reason} — skipped PRM_log"} if is_error else None
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
        lines       = [f"[Problem]\n{state.item['problem']}"]
        if state.history:
            lines.append("\n[Previous steps]")
            for i, s in enumerate(state.history, 1):
                step_ctx = (s.get("summary") or {}).get("does") or s.get("inference") or s["text"]
                lines.append(f"Step {i}: {step_ctx}")
        explanation = _build_rethink_explanation(state)
        if explanation and explanation != "the previous step contained an error":
            lines.append(f"\n[Note: previous attempts at Step {step_number} were rejected]\n{explanation}")
        lines.append(f"\nWrite Step {step_number}.")
        messages    = [
            {"role": "system", "content": PAT_SOLVE_PROMPT},
            {"role": "user",   "content": "\n".join(lines)},
        ]
        logger.info(f"[Patcher API] id={problem_id} step={step_number} model={PATCHER}")
        try:
            set_problem_context(problem_id, step_number)
            set_call_role("patcher")
            text = _call_llm(PATCHER, messages, max_completion_tokens=PATCHER_MAX_NEW_TOKENS,
                             thinking_budget=PATCHER_THINKING_BUDGET)
            text = (text or "").strip()
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

        _sc_m = re.search(
            r"\n(?:Self-correction:|(?:\d+\.\s*)?\*{0,2}(?:Fast\s+critic|fast_critique)\*{0,2}\s*:|Algebraic\s+Manipulation\s*:)",
            text, re.IGNORECASE,
        )
        step_text = text[:_sc_m.start()].strip() if _sc_m else text.strip()

        state.pending_step = {
            "text":             text,
            "inference":        step_text,
            "full_text":        text,
            "pred_action":      pred_action,
            "next_pred_action": _parse_next_pred_action(text),
            "source":           "patcher",
            "role":             "patcher",
            "was_rethink":      True,
            "is_error":         is_error,
            "is_first_pat":     not is_error,
            "summary":          (
                {"step_analysis": "patcher API empty response — skipped PRM_log"}
                if is_error else None
            ),
        }
        state.use_patcher = False

    with ThreadPoolExecutor(max_workers=max(1, len(states))) as ex:
        list(ex.map(_call_one, states))


# ─────────────────────────────────────────────────────────────────────────────



def _run_prm_batch(
    prm_model: "ApiPrm | ApiPrmBatch",
    prm_states: list,
    rubrics_per_step: list[list[dict]],
    prm_stats: dict = None,
) -> dict[int, dict]:
    """prm_states 각각을 평가해 prm_results_map 반환.
    ApiPrmBatch 모드: 상태당 1번 API 호출.
    ApiPrm 모드: 루브릭당 1번 API 호출 (기존 방식).
    """
    if not prm_states:
        return {}

    if isinstance(prm_model, ApiPrmTwoStage):
        # ── Stage 1: batch 평가 ────────────────────────────────────────────────
        prev_steps_list, step_numbers = [], []
        for state in prm_states:
            prev_lines = [f"Step {i+1}: {s['summary']['does']}"
                          for i, s in enumerate(state.history)
                          if not s.get("is_error") and (s.get("summary") or {}).get("does")]
            prev_steps_list.append("\n".join(prev_lines))
            step_numbers.append(len(state.history) + 1)

        _t_s1 = time.time()
        s1_verdicts_list = prm_model.stage1.evaluate_batch(
            questions  = [state.item["problem"] for state in prm_states],
            prev_steps = prev_steps_list,
            now_steps  = [state.pending_step.get("inference") or state.pending_step["text"]
                          for state in prm_states],
            max_new_tokens = PRM_STAGE1_MAX_NEW_TOKENS,
            problem_ids  = [str(state.item.get("id", "?")) for state in prm_states],
            step_numbers = step_numbers,
        )
        logger.info(f"[TIMING] PRM stage1={time.time()-_t_s1:.1f}s  n={len(prm_states)}")

        # Stage 1 결과 집계
        s1_results = {}
        for state, verdicts in zip(prm_states, s1_verdicts_list):
            votes = {name: ("correct" if v["pred"] == "correct" else "incorrect")
                     for name, v in zip(prm_model.stage1.rubric_names, verdicts)}
            n_wrong = sum(1 for v in votes.values() if v == "incorrect")
            s1_results[id(state)] = {
                "result":      "incorrect" if n_wrong >= 1 else "correct",  # stage2 라우팅용: 1개라도 fail이면 재평가
                "wrong_count": n_wrong,
                "total":       len(votes),
                "threshold":   1,
                "votes":       votes,
                "fast_critiques": {name: v.get("critique")
                                   for name, v in zip(prm_model.stage1.rubric_names, verdicts)},
                "details":     {name: {"response": v.get("response"),
                                       "verdict_text": v.get("verdict_text", ""),
                                       "full_response": v.get("full_response"),
                                       "prob_correct": v.get("prob_correct"),
                                       "prob_incorrect": v.get("prob_incorrect"),
                                       "method": "api_batch_stage1"}
                                for name, v in zip(prm_model.stage1.rubric_names, verdicts)},
                "prm_n":       1,
            }

        # ── Stage 2: Stage 1 fail인 경우에만 개별 루브릭 재평가 ────────────────
        fail_states = [s for s in prm_states if s1_results[id(s)]["result"] == "incorrect"]

        # 샘플별 stage1 결과 로깅
        for state in prm_states:
            r = s1_results[id(state)]
            pid = state.item.get("id", "?")
            step_n = len(state.history) + 1
            n_fail = r["wrong_count"]
            n_total = r["total"]
            if n_fail > 0:
                failed = [name for name, v in r["votes"].items() if v == "incorrect"]
                logger.info(
                    f"[Stage1] id={pid} step={step_n}  "
                    f"batch {n_fail}/{n_total} fail ({', '.join(failed)}) "
                    f"→ stage2 {n_fail}개 루브릭 호출"
                )
            else:
                logger.info(
                    f"[Stage1] id={pid} step={step_n}  "
                    f"batch {n_fail}/{n_total} pass → stage2 없음"
                )

        logger.info(
            f"[2-Stage PRM] stage1: {len(prm_states)}개 중 "
            f"fail={len(fail_states)}, pass={len(prm_states)-len(fail_states)}"
        )

        s2_results = {}
        if fail_states:
            def _eval_stage2(state):
                sid = id(state)
                step_number = len(state.history) + 1
                problem_id  = str(state.item.get("id", "?"))
                prev_lines  = [f"Step {i+1}: {s['summary']['does']}"
                               for i, s in enumerate(state.history)
                               if not s.get("is_error") and (s.get("summary") or {}).get("does")]
                # stage1에서 fail인 루브릭만 재평가
                failed_names   = {n for n, v in s1_results[sid]["votes"].items() if v == "incorrect"}
                stage2_rubrics = [r for r in prm_model.rubrics if r["name"] in failed_names]
                now_step = state.pending_step.get("inference") or state.pending_step["text"]
                if state.pending_step.get("source") == "patcher":
                    now_step += (
                        "\n\n[Evaluator note: This step was generated by the patcher "
                        "and is guaranteed correct. Analyze each rubric for informational "
                        "purposes only, and output Verdict: correct for every rubric.]"
                    )
                verdict, detail = evaluate_step(
                    question     = state.item["problem"],
                    prev_steps   = "\n".join(prev_lines),
                    now_step     = now_step,
                    rubrics      = stage2_rubrics,
                    model        = prm_model.stage2,
                    fail_k       = 1,
                    max_new_tokens = PRM_MAX_NEW_TOKENS,
                    cot          = True,
                    problem_id   = problem_id,
                    step_number  = step_number,
                )
                s2_votes = {name: ("correct" if res["pred"] == "correct" else "incorrect")
                            for name, res in detail.items()}
                s2_dets  = {name: {"response":       res.get("response"),
                                   "verdict_text":   res.get("verdict_text", ""),
                                   "full_response":  res.get("full_response"),
                                   "prob_correct":   res.get("prob_correct"),
                                   "prob_incorrect": res.get("prob_incorrect"),
                                   "method":         res.get("method", "api_stage2")}
                            for name, res in detail.items()}
                # stage1 pass 루브릭 + stage2 재평가 루브릭 합산
                all_votes = {**s1_results[sid]["votes"], **s2_votes}
                all_dets  = {**s1_results[sid]["details"], **s2_dets}
                n_wrong   = sum(1 for v in all_votes.values() if v == "incorrect")
                return id(state), {
                    "result":      "incorrect" if _prm_is_fail(all_votes) else "correct",
                    "wrong_count": n_wrong,
                    "total":       len(all_votes),
                    "threshold":   "core>=2|extra>=1",
                    "votes":       all_votes,
                    "details":     all_dets,
                    "prm_n":       len(s2_votes),
                    "_s2_rubric_count": len(stage2_rubrics),
                }

            _t_s2 = time.time()
            with ThreadPoolExecutor(max_workers=max(1, len(fail_states))) as ex:
                for state_id, res in ex.map(_eval_stage2, fail_states):
                    s2_results[state_id] = res
            logger.info(
                f"[TIMING] PRM stage2={time.time()-_t_s2:.1f}s  "
                f"n_fail={len(fail_states)}  avg_rubrics="
                f"{sum(r.get('_s2_rubric_count',0) for r in s2_results.values())/max(1,len(s2_results)):.1f}"
            )

        # ── 최종 결과: pass → stage1, fail → stage2 ───────────────────────────
        final = {}
        for state in prm_states:
            sid = id(state)
            if sid in s2_results:
                entry = dict(s2_results[sid])
                entry["prm_filter"] = s1_results[sid]  # stage1 결과 보존
                final[sid] = entry
            else:
                final[sid] = s1_results[sid]
            final[sid]["fast_rubric"] = {
                name: {"verdict": pred, "critique": s1_results[sid]["fast_critiques"].get(name)}
                for name, pred in s1_results[sid]["votes"].items()
            }
        if prm_stats is not None:
            prm_stats["fast_rubric_calls"] += len(prm_states)
            prm_stats["rubric_calls"] += sum(
                s2_results[id(s)].get("_s2_rubric_count", 0) for s in fail_states
            )
        return final

    if isinstance(prm_model, ApiPrmBatch):
        prev_steps_list, step_numbers = [], []
        for state in prm_states:
            prev_lines = [f"Step {i+1}: {s['summary']['does']}"
                          for i, s in enumerate(state.history)
                          if not s.get("is_error") and (s.get("summary") or {}).get("does")]
            prev_steps_list.append("\n".join(prev_lines))
            step_numbers.append(len(state.history) + 1)

        verdicts_list = prm_model.evaluate_batch(
            questions  = [state.item["problem"] for state in prm_states],
            prev_steps = prev_steps_list,
            now_steps  = [state.pending_step.get("inference") or state.pending_step["text"]
                          for state in prm_states],
            max_new_tokens = PRM_MAX_NEW_TOKENS,
            problem_ids  = [str(state.item.get("id", "?")) for state in prm_states],
            step_numbers = step_numbers,
        )

        results = {}
        for state, verdicts in zip(prm_states, verdicts_list):
            votes = {
                name: ("correct" if v["pred"] == "correct" else "incorrect")
                for name, v in zip(prm_model.rubric_names, verdicts)
            }
            dets = {
                name: {
                    "response":       v.get("response"),
                    "verdict_text":   v.get("verdict_text", ""),
                    "full_response":  v.get("full_response"),
                    "prob_correct":   v.get("prob_correct"),
                    "prob_incorrect": v.get("prob_incorrect"),
                    "method":         v.get("method", "api_batch"),
                }
                for name, v in zip(prm_model.rubric_names, verdicts)
            }
            n_wrong = sum(1 for v in votes.values() if v == "incorrect")
            results[id(state)] = {
                "result":      "incorrect" if _prm_is_fail(votes) else "correct",
                "wrong_count": n_wrong,
                "total":       len(votes),
                "threshold":   "core>=2|extra>=1",
                "votes":       votes,
                "details":     dets,
                "prm_n":       1,
            }
        if prm_stats is not None:
            prm_stats["fast_rubric_calls"] += len(prm_states)
        return results

    # ── 기존 ApiPrm 모드: 루브릭당 1번 호출 ─────────────────────────────────
    def _eval_one(args):
        state, step_rubrics = args
        if not step_rubrics:
            logger.warning(f"[PRM] id={state.item.get('id','?')} 루브릭 없음 → pass 처리")
            return id(state), {"result": "correct", "wrong_count": 0, "total": 0, "threshold": 1, "votes": {}, "details": {}, "prm_n": 0}
        step_number = len(state.history) + 1
        problem_id  = str(state.item.get("id", "?"))
        prev_lines  = [f"Step {i+1}: {s['summary']['does']}"
                       for i, s in enumerate(state.history)
                       if not s.get("is_error") and (s.get("summary") or {}).get("does")]
        verdict, detail = evaluate_step(
            question     = state.item["problem"],
            prev_steps   = "\n".join(prev_lines),
            now_step     = state.pending_step.get("inference") or state.pending_step["text"],
            rubrics      = step_rubrics,
            model        = prm_model,
            fail_k       = 1,
            max_new_tokens = PRM_MAX_NEW_TOKENS,
            cot          = True,
            problem_id   = problem_id,
            step_number  = step_number,
        )        
        votes = {name: ("correct" if res["pred"] == "correct" else "incorrect") for name, res in detail.items()}
        dets  = {
            name: {
                "response":       res.get("response"),
                "verdict_text":   res.get("verdict_text", ""),
                "full_response":  res.get("full_response"),
                "prob_correct":   res.get("prob_correct"),
                "prob_incorrect": res.get("prob_incorrect"),
                "method":         res.get("method", "api"),
            }
            for name, res in detail.items()
        }
        n_wrong = sum(1 for v in votes.values() if v == "incorrect")
        return id(state), {
            "result":      "incorrect" if _prm_is_fail(votes) else "correct",
            "wrong_count": n_wrong,
            "total":       len(votes),
            "threshold":   "core>=2|extra>=1",
            "votes":       votes,
            "details":     dets,
            "prm_n":       len(votes),
        }

    with ThreadPoolExecutor(max_workers=max(1, len(prm_states))) as ex:
        pairs = list(ex.map(_eval_one, zip(prm_states, rubrics_per_step)))
    if prm_stats is not None:
        prm_stats["rubric_calls"] += sum(len(r) for r in rubrics_per_step)
    return dict(pairs)


_SUMMARIZE_MAX_TOKENS        = 128
_WRONG_STEP_SUMMARIZE_MAX_TOKENS = 384
_CRITIQUE_REVIEW_MAX_TOKENS = 256


def _parse_rubric_lines(section: str, rubric_names: list[str]) -> dict[str, dict]:
    """섹션 텍스트에서 루브릭별 verdict + critique 파싱."""
    results = {}
    for line in section.split("\n"):
        line = line.strip()
        if not line:
            continue
        for rubric in rubric_names:
            if rubric.lower() in line.lower():
                vm = re.search(r"Verdict\s*:\s*(correct|incorrect)", line, re.IGNORECASE)
                if vm:
                    verdict = vm.group(1).lower()
                elif "incorrect" in line.lower():
                    verdict = "incorrect"
                elif "correct" in line.lower():
                    verdict = "correct"
                else:
                    verdict = None
                rp = re.search(rf"{re.escape(rubric)}\s*[:\.]?\s*(.+?)(?:\s+Verdict\s*:.*)?$", line, re.IGNORECASE)
                critique = None
                if rp:
                    cand = re.sub(r"\s*Verdict\s*:.*$", "", rp.group(1), flags=re.IGNORECASE).strip()
                    critique = cand if cand else None
                results[rubric] = {"verdict": verdict, "critique": critique}
                break
    return results


def _parse_generator_self_check(text: str, rubric_names: list[str]) -> dict[str, dict]:
    """Fast critic 섹션 파싱.
    지원 포맷: "Fast critic:", "**Fast critic:**", "1. **Fast critic:**", "fast_critique:"
    반환: {rubric_name: {"verdict", "critique"}}"""
    _FC = r"(?:\d+\.\s*)?\*{0,2}(?:Fast\s+critic|fast_critique)\*{0,2}"
    _DC = r"(?:\d+\.\s*)?\*{0,2}(?:Deep\s+critic|deep_critique)\*{0,2}"
    m = re.search(
        _FC + r"\s*:\*{0,2}\s*\n(.*?)(?=" + _DC + r"\s*:\*{0,2}\s*\n|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    if m:
        return _parse_rubric_lines(m.group(1), rubric_names)
    m = re.search(r"Self-correct(?:ion)?\s*[:\s]*\n(.*)", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return {}
    return _parse_rubric_lines(m.group(1), rubric_names)


def _parse_generator_deep_check(text: str, rubric_names: list[str]) -> dict[str, dict]:
    """Deep critic 섹션 파싱.
    지원 포맷: "Deep critic:", "**Deep critic:**", "2. **Deep critic:**", "deep_critique:"
    반환: {rubric_name: {"verdict", "critique"}}

    지원 포맷:
      - [rubric]: [reasoning] Verdict: correct/incorrect
      - [rubric]: Verdict: incorrect — [reasoning after]
      - 멀티라인 블록
    """
    _DC = r"(?:\d+\.\s*)?\*{0,2}(?:Deep\s+critic|deep_critique)\*{0,2}"
    m = re.search(
        _DC + r"\s*:\*{0,2}\s*\n(.*?)$",
        text, re.DOTALL | re.IGNORECASE
    )
    if not m:
        return {}
    section = m.group(1)

    # "Fail rubrics:" 이후는 gen 포맷 구조물이므로 제거
    fail_m = re.search(r"^\s*Fail\s+rubrics\s*:", section, re.IGNORECASE | re.MULTILINE)
    if fail_m:
        section = section[:fail_m.start()]

    # 첫 번째 비어있지 않은 줄이 "none"이면 딥 크리틱 없음 (이후 내용 무시)
    first_line = next((l.strip() for l in section.split("\n") if l.strip()), "")
    if first_line.lower() == "none":
        return {}

    results = {}
    for i, rubric in enumerate(rubric_names):
        # 다음 루브릭 시작 전까지의 블록 캡처
        later = [re.escape(r) for r in rubric_names[i + 1:]]
        end_pat = rf"(?:[-*•\s]*(?:{'|'.join(later)})\s*[:\.])" if later else r"\Z"
        pat = re.search(
            rf"(?:[-*•]\s*)?{re.escape(rubric)}\s*[:\.]?\s*(.*?)(?={end_pat}|\Z)",
            section, re.DOTALL | re.IGNORECASE
        )
        if not pat:
            continue
        block = pat.group(1).strip()
        if not block:
            continue

        # 뒤따르는 display-mode LaTeX 블록(\[...\])은 step 내용 유출이므로 제거
        paras = block.split("\n\n")
        paras = [p for p in paras if not p.strip().startswith("\\[")]
        block = "\n\n".join(paras).strip()
        if not block:
            continue

        # verdict 파싱
        vm = re.search(r"Verdict\s*:\s*(correct|incorrect)", block, re.IGNORECASE)
        if vm:
            verdict = vm.group(1).lower()
        elif "incorrect" in block.lower():
            verdict = "incorrect"
        elif "correct" in block.lower():
            verdict = "correct"
        else:
            verdict = None

        # reasoning: verdict 앞 텍스트 + verdict 뒤 텍스트 모두 포함
        if vm:
            before = block[:vm.start()].strip()
            after  = re.sub(r"^[\s—\-\.]+", "", block[vm.end():]).strip()
            reasoning = " ".join(filter(None, [before, after])) or None
        else:
            reasoning = block or None

        results[rubric] = {"verdict": verdict, "critique": reasoning}

    return results


def _parse_next_pred_action(text: str) -> str | None:
    """Self-correction 블록에서 'Next action: solve/rethink/end' 를 추출.
    'Next action:\\n solve' 처럼 줄바꿈 뒤에 오는 경우도 지원."""
    m = re.search(r"Next\s+action\s*:\s*\n?\s*(solve|rethink|end)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def _batch_run_step_summary_only(
    model, tokenizer, input_device,
    states: list[ProblemState],
) -> None:
    """PRM 평가와 병렬로 실행 가능한 step summary만 처리. prm_results_map 불필요."""
    prompts: list[str] = []
    to_summarize: list = []

    for state in states:
        step = state.pending_step
        if step.get("is_error"):
            step["does_summary"] = None
            continue
        step_text = (step.get("inference") or step.get("text") or "")[:1200]
        prompts.append(build_chat_prompt(tokenizer, _SUMMARIZE_SYSTEM, f"Step:\n{step_text}"))
        to_summarize.append(state)

    if not prompts:
        return

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
        state.pending_step["does_summary"] = re.split(r"\n", raw)[0].strip() or None



def _generate_texts(model, tokenizer, device, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Batch generate text completions. device=None → vLLM, otherwise HF."""
    if device is None:
        return _vllm_texts(model, prompts, max_new_tokens)
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return [tokenizer.decode(out[j, input_len:], skip_special_tokens=True)
            for j in range(len(prompts))]


def _batch_run_all_summaries_and_gen_critique(
    model, tokenizer, device,
    states: list[ProblemState],
    prm_results_map: dict,
) -> None:
    """③ all_summaries + ⑤ gen_critique_review를 한 번의 generate로 처리.
    device=None → vLLM, otherwise HF."""
    from collections import defaultdict

    prompts: list[str] = []
    tasks:   list      = []  # (type, state, extra)

    for state in states:
        step   = state.pending_step
        result = prm_results_map.get(id(state), {})

        if not step.get("is_error"):
            step_text = (step.get("inference") or step.get("text") or "")[:1200]

            if "does_summary" not in step:
                prompts.append(build_chat_prompt(tokenizer, _SUMMARIZE_SYSTEM, f"Step:\n{step_text}"))
                tasks.append(("step_summary", state, None))

            # PRM fail 스텝 → 상세 wrong step 요약 생성 (rethink 프롬프트용)
            if result.get("result") == "incorrect":
                prompts.append(build_chat_prompt(tokenizer, _WRONG_STEP_SUMMARIZE_SYSTEM, f"Step:\n{step_text}"))
                tasks.append(("wrong_step_summary", state, None))
            else:
                step["wrong_step_summary"] = None

            votes   = result.get("votes", {})
            details = result.get("details", {})
            step_short = (step.get("inference") or step.get("text") or "")[:600]
            for rn in votes:
                if (details.get(rn) or {}).get("method") == "api_batch_stage1":
                    continue
                response = (details.get(rn) or {}).get("response") or ""
                user_msg  = (
                    f"Rubric: {rn} (verdict: {votes[rn]})\n"
                    f"Rubric reasoning:\n{response}\n\n"
                    f"Step:\n{step_short}"
                )
                prompts.append(build_chat_prompt(tokenizer, _CRITIQUE_REVIEW_SYSTEM, user_msg))
                tasks.append(("critique", state, rn))
        else:
            step["does_summary"]     = None
            step["critique_review"] = None
            step["wrong_step_summary"] = None

    # ⑤ gen_critique_review 프롬프트 추가
    for state in states:
        step    = state.pending_step
        entries = [(rn, v["critique"])
                   for rn, v in (step.get("gen_deep_critique") or {}).items()
                   if v.get("critique")]
        if not entries:
            step["gen_critique_review"] = None
            continue
        combined  = "\n\n".join(f"[{label}]\n{critique}" for label, critique in entries)
        step_text = (step.get("inference") or step.get("text") or "")[:400]
        user_msg  = f"Step:\n{step_text}\n\nAnalyses:\n{combined}"
        prompts.append(build_chat_prompt(tokenizer, _CRITIQUE_PARA_SUMMARY_SYSTEM, user_msg))
        tasks.append(("gen_critique_review", state, None))

    if not prompts:
        return

    max_tokens_per_task = [
        _WRONG_STEP_SUMMARIZE_MAX_TOKENS if t == "wrong_step_summary" else 256
        for t, _, _ in tasks
    ]
    max_new_tokens = max(max_tokens_per_task)
    texts = _generate_texts(model, tokenizer, device, prompts, max_new_tokens)
    critique_by_state: dict = defaultdict(list)

    for (task_type, state, extra), raw in zip(tasks, texts):
        raw = raw.strip()

        if task_type == "step_summary":
            state.pending_step["does_summary"] = re.split(r"\n", raw)[0].strip() or None

        elif task_type == "wrong_step_summary":
            state.pending_step["wrong_step_summary"] = raw or None

        elif task_type == "critique":
            result = prm_results_map.get(id(state), {})
            critique_by_state[id(state)].append({
                "rubric":   extra,
                "verdict":  result.get("votes", {}).get(extra, "correct"),
                "critique": raw or None,
            })

        elif task_type == "gen_critique_review":
            state.pending_step["gen_critique_review"] = raw or None

    for state in states:
        if not state.pending_step.get("is_error"):
            result = prm_results_map.get(id(state), {})
            votes  = result.get("votes", {})
            filled = {e["rubric"]: e for e in critique_by_state.get(id(state), [])}
            state.pending_step["critique_review"] = [
                filled.get(rn, {"rubric": rn, "verdict": None, "critique": None})
                for rn in votes
            ] or None


def _batch_generate_critique_review(
    model, tokenizer, device, states: list[ProblemState],
    *, source: str,  # "prm" or "gen"
) -> None:
    output_key = "prm_critique_review" if source == "prm" else "gen_critique_review"
    pairs = []
    for state in states:
        step = state.pending_step
        if source == "prm":
            entries = [(e["rubric"], e["critique"])
                       for e in (step.get("critique_review") or [])
                       if e.get("verdict") in ("incorrect", "incorrect") and e.get("critique")]
        else:
            entries = [(rn, v["critique"])
                       for rn, v in (step.get("gen_deep_critique") or {}).items()
                       if v.get("critique")]
        if not entries:
            step[output_key] = None
            continue
        combined  = "\n\n".join(f"[{label}]\n{critique}" for label, critique in entries)
        step_text = (step.get("inference") or step.get("text") or "")[:400]
        user_msg  = f"Step:\n{step_text}\n\nAnalyses:\n{combined}"
        pairs.append((state, build_chat_prompt(tokenizer, _CRITIQUE_PARA_SUMMARY_SYSTEM, user_msg)))

    if not pairs:
        return
    texts = _generate_texts(model, tokenizer, device, [p[1] for p in pairs], 256)
    for (state, _), raw in zip(pairs, texts):
        raw = raw.strip()
        if output_key == "prm_critique_review":
            raw = re.sub(r"\n*\d*\.?\s*Failed rubrics\s*:.*", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
        state.pending_step[output_key] = raw or None


_GEN_DEEP_CRITIQUE_SYSTEM = (
    "You are a math student re-examining your own solution step against ONE specific rubric.\n\n"
    "TASK: Check whether this step violates the rubric criterion.\n"
    "  1. State whether the criterion APPLIES to this step (Yes/No).\n"
    "  2. If Yes: perform the actual check (expand, substitute, verify the specific condition).\n"
    "  3. Conclude: Verdict: correct  OR  Verdict: incorrect\n\n"
    "BIAS: If the check applies and you are uncertain → Verdict: incorrect.\n"
    "      If the check is N/A → Verdict: correct.\n"
    "Be concise but explicit about what you computed or verified."
)

_CRITIQUE_PARA_SUMMARY_SYSTEM = (
    "Given rubric-specific error analyses of a math solution step, write "
    "one concise paragraph summarizing what went wrong mathematically. "
    "Be specific: reference actual expressions or values from the step. "
    "Do NOT list rubric names or add any extra sections."
)


def _extract_gen_fast_critique(states: list[ProblemState], rubric_names: list[str]) -> None:
    """inference의 fast_critique + deep_critique 섹션을 파싱해 두 필드 모두 설정."""
    for state in states:
        step = state.pending_step
        text = (step.get("full_text") or step.get("text") or "")
        fast = _parse_generator_self_check(text, rubric_names)
        step["gen_fast_critique"] = {
            rn: {"verdict": fast.get(rn, {}).get("verdict"),
                 "critique": fast.get(rn, {}).get("critique")}
            for rn in rubric_names
        } or None
        deep = _parse_generator_deep_check(text, rubric_names)
        step["gen_deep_critique"] = deep if deep else {}

# ─────────────────────────────────────────────────────────────────────────────
# Trajectory 조립
# ─────────────────────────────────────────────────────────────────────────────

def _compute_labels(steps: list[dict], first_pat_pos: int = 0) -> list[str]:
    labels = []
    pos    = 0
    for s in steps:
        role = s.get("role", "gen")
        pos += 1
        if role == "patcher":
            labels.append(f"P*_{pos:02d}")
        elif role == "rethink":
            labels.append(f"G+_{pos:02d}")
        else:
            labels.append(f"G_{pos:02d}")
    return labels


def _parse_pred_fail_rubrics(gen_critique_review: str | None) -> list[str]:
    """gen_critique_review에서 모델이 직접 출력한 'Failed rubrics:' 줄을 파싱."""
    if not gen_critique_review:
        return []
    m = re.search(r"Failed rubrics\s*:\s*(.+)", gen_critique_review, re.I)
    if not m:
        return []
    raw = m.group(1).strip()
    if raw.lower() in ("none", "none.", "n/a"):
        return []
    return [r.strip().rstrip(".") for r in raw.split(",") if r.strip()]


def _compute_step_action(s: dict) -> tuple[list[str], str]:
    """prm_deep_critique 기반으로 fail_rubrics와 next_gold_action을 계산."""
    summ = s.get("summary") or {}
    deep = summ.get("prm_deep_critique") if isinstance(summ, dict) else None
    if deep is None:
        deep = s.get("critique_review") or []

    fail_rubrics = [
        e["rubric"] for e in (deep or [])
        if e.get("verdict") in ("incorrect", "incorrect")
    ]

    if fail_rubrics:
        action = TOKEN_CORRECT
    elif has_boxed(s.get("full_text") or s.get("text", "")):
        action = TOKEN_END
    else:
        action = TOKEN_SOLVE

    return fail_rubrics, action


def _build_traj(
    problem_id, problem, gold_answer,
    steps: list[dict],
    is_right: bool,
    traj_type: str,
    first_pat_pos: int = 0,
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
        step_src = s.get("source", "gen")
        _is_last_correct = (is_last and not s["is_error"])
        _has_box = has_boxed(s.get("full_text") or s.get("text", ""))
        if _is_last_correct and _has_box:
            state = "end"
        elif s.get("was_rethink") or step_src == "patcher":
            state = "rethink"
        else:
            state = "solve"

        fail_rubrics, next_action = _compute_step_action(s)

        summ = s.get("summary") or {}
        if isinstance(summ, dict):
            does                 = summ.get("does") or summ.get("step_analysis") or None
            prm_fast_critique    = summ.get("prm_fast_critique") or None
            prm_deep_critique    = summ.get("prm_deep_critique") or None
            prm_critique_review = summ.get("prm_critique_review") or None
            gen_fast_critique    = summ.get("gen_fast_critique") or None
            gen_deep_critique    = summ.get("gen_deep_critique") or None
            gen_critique_review = summ.get("gen_critique_review") or None
        else:
            does = prm_fast_critique = prm_deep_critique = prm_critique_review = None
            gen_fast_critique = gen_deep_critique = gen_critique_review = None

        pred_fail_rubrics = _parse_pred_fail_rubrics(gen_critique_review)

        step_dicts.append({
            "step_idx":             i,
            "step":                 label,
            "text":                 s.get("full_text") or s["text"],
            "inference":            s["text"],
            "source":               s["source"],
            "is_error":             s["is_error"],
            "state":                state,
            "gold_fail_rubrics":    fail_rubrics,
            "pred_fail_rubrics":    pred_fail_rubrics,
            "next_gold_action":     next_action,
            "next_pred_action":     s.get("next_pred_action"),
            "does":                 does,
            "prm_fast_critique":    prm_fast_critique,
            "prm_deep_critique":    prm_deep_critique,
            "prm_critique_review": prm_critique_review,
            "gen_fast_critique":    gen_fast_critique,
            "gen_deep_critique":    gen_deep_critique,
            "gen_critique_review": gen_critique_review,
        })

    return {
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
        role = s.get("role", "gen")
        if role == "patcher":
            parts.append(f"P*_{n:02d}")
        elif role == "rethink":
            parts.append(f"G+_{n:02d}")
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
    prm_save_fn,
    rubric_text_save_fn=None,
    prm_filter_save_fn=None,
    step_manager_model=None,
    step_manager_tok=None,
) -> None:
    """PRM 결과로 state 업데이트. 완료 시 state.done = True."""
    step        = state.pending_step
    step_number = len(state.history) + 1
    problem_id  = state.item.get("id", "?")
    problem     = state.item["problem"]
    gold_answer = state.item.get("gold_answer") or state.item["answer"]

    step["prm_n"] = result.get("prm_n")

    wrong_count   = result["wrong_count"]
    total         = result["total"]
    votes         = result["votes"]
    details       = result["details"]
    wrong_rubrics = [r for r, v in votes.items() if v == "incorrect"]
    rubric_text   = [
        {
            "rubric":       rn,
            "response":     details[rn].get("response"),
            "verdict_text": details[rn].get("verdict_text"),
        }
        for rn in votes if rn in details
    ]

    # 루브릭별 full text 별도 저장
    if rubric_text_save_fn:
        for rt in rubric_text:
            rubric_text_save_fn({
                "problem_id":   problem_id,
                "step_idx":     step_number,
                "rubric":       rt["rubric"],
                "verdict":      votes.get(rt["rubric"]),
                "response":     rt["response"],
                "verdict_text": rt["verdict_text"],
            })

    # _batch_run_all_summaries에서 로컬 generator가 이미 계산한 critique 사용
    prm_deep_critique    = step.get("critique_review")
    prm_fast_critique    = result.get("fast_rubric")
    prm_critique_review = step.get("prm_critique_review")
    gen_fast_critique    = step.get("gen_fast_critique")
    gen_deep_critique    = step.get("gen_deep_critique")
    gen_critique_review = step.get("gen_critique_review")

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
            "response":          d.get("response"),
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

    # prm_filter(stage1) 결과 저장 — stage2까지 간 step에만 존재
    if prm_filter_save_fn and result.get("prm_filter"):
        f1 = result["prm_filter"]
        for rname, d in f1.get("details", {}).items():
            filter_rec = {
                "problem_id":        problem_id,
                "step_global_idx":   step_number,
                "question":          problem,
                "previous_steps":    "\n".join(prev_lines),
                "now_step":          now_step_text,
                "rubric_name":       rname,
                "pred":              f1["votes"].get(rname),
                "prob_correct":      d.get("prob_correct"),
                "prob_incorrect":    d.get("prob_incorrect"),
                "response":          d.get("response"),
                "verdict_text":      d.get("verdict_text"),
                "full_response":     d.get("full_response"),
                "method":            d.get("method"),
                "filter_result":     f1["result"],
                "filter_wrong_count": f1["wrong_count"],
                "filter_total":      f1["total"],
            }
            prm_filter_save_fn(filter_rec)


    does_summary  = step.get("does_summary")

    def _apply_wrong(reason_suffix: str = ""):
        """step을 wrong으로 처리하고 rethink/patcher/abort 분기."""
        step["is_error"]     = True
        step["is_first_pat"] = False
        # 오류 스텝에도 substep_meta 기록
        if state.in_substep_mode and state.substep_queue:
            cur = state.substep_queue[0]
            step["substep_meta"] = {
                "is_substep":    True,
                "depth":         cur.get("depth", 0),
                "goal":          cur.get("goal", ""),
                "substep_index": len(state.substep_passed),
            }
        else:
            step["substep_meta"] = {"is_substep": False}
        step["summary"]      = {
            "does":                 does_summary,
            "prm_fast_critique":    prm_fast_critique,
            "step_analysis":        (
                f"API_PRM_checklist: {wrong_count}/{total} rubrics flagged wrong"
                f" ({', '.join(wrong_rubrics)})"
                + (f" [{reason_suffix}]" if reason_suffix else "")
            ),
            "prm_deep_critique":    prm_deep_critique,
            "prm_critique_review": prm_critique_review,
            "gen_fast_critique":    gen_fast_critique,
            "gen_deep_critique":    gen_deep_critique,
            "gen_critique_review": gen_critique_review,
            "votes":                votes,
            "details":              details,
        }
        state.all_steps.append(step)

        if TRAJ_MAX_STEPS and len(state.all_steps) >= TRAJ_MAX_STEPS:
            print(f"  [id={problem_id}]  {_fmt(state.all_steps)}  → max_steps")
            traj = _build_traj(problem_id, problem, gold_answer,
                               state.all_steps, False, "all",
                               fail_reason="max_steps")
            if save_fn:
                save_fn(traj)
            state.done = True
            return

        _rubric_details = {
            rn: (details.get(rn, {}).get("response") or details.get(rn, {}).get("verdict_text") or "")
            for rn in wrong_rubrics
            if details.get(rn, {}).get("response") or details.get(rn, {}).get("verdict_text")
        }
        state.last_wrong_prm_deep_critique = prm_deep_critique or []
        if state.in_substep_mode:
            # ── 서브스텝 rethink 실패 → 더 쪼개거나 patcher ─────────────────
            cur_depth = state.substep_queue[0].get("depth", 0) if state.substep_queue else 0
            decomposed = False
            if cur_depth < MAX_SUBSTEP_DEPTH and state.substep_queue:
                sub_substeps = _decompose_with_atomicity(
                    state.item.get("problem", ""),
                    [s["text"] for s in state.all_steps if not s.get("is_error")],
                    step["text"],   # 방금 실패한 서브스텝 rethink 결과
                    step_manager_model, step_manager_tok,
                )
                if sub_substeps:
                    for s in sub_substeps:
                        s["depth"] = cur_depth + 1
                    # 현재 서브스텝 대신 2개 sub-substep으로 교체
                    state.substep_queue = sub_substeps + state.substep_queue[1:]
                    state.is_rethink    = True
                    logger.info(f"[id={problem_id}] 서브스텝 재분해 성공 depth={cur_depth+1}: "
                                f"{[s['goal'][:40] for s in sub_substeps]}")
                    decomposed = True
            if not decomposed:
                # 더 못 쪼갬 (atomic or max_depth) → patcher
                logger.info(f"[id={problem_id}] 서브스텝 분해 한계(depth={cur_depth}) → patcher")
                state.in_substep_mode    = False
                state.substep_queue      = []
                state.is_rethink         = False
                state.use_patcher        = True
                state.step_patcher_tried = True
                state.patcher_count     += 1
                state.last_wrong_rubrics = wrong_rubrics
        elif not state.step_rethink_tried:
            # 1차 실패 → rethink
            state.is_rethink                    = True
            state.step_rethink_tried            = True
            state.last_wrong_step_text          = step["text"]
            state.last_wrong_rubric_details     = _rubric_details
            state.last_wrong_does               = step.get("wrong_step_summary") or (step.get("summary") or {}).get("does") or ""
            state.last_wrong_gen_deep_critique  = step.get("gen_deep_critique") or {}
        elif not state.step_substep_tried:
            # rethink 실패 → Atomicity 기준으로 서브스텝 분해 시도
            state.step_substep_tried = True
            substeps = _decompose_with_atomicity(
                state.item.get("problem", ""),
                [s["text"] for s in state.all_steps if not s.get("is_error")],
                state.last_wrong_step_text,
                step_manager_model, step_manager_tok,
            )
            if substeps:
                # NON-ATOMIC → 서브스텝 큐 설정, rethink 모드로 진입
                state.substep_queue   = substeps
                state.substep_passed  = []
                state.in_substep_mode = True
                state.is_rethink      = True
                state.last_wrong_rubric_details = _rubric_details
                logger.info(f"[id={problem_id}] substep 분해 성공: {[s['goal'][:50] for s in substeps]}")
            else:
                # ATOMIC → 바로 patcher
                state.use_patcher        = True
                state.is_rethink         = False
                state.step_patcher_tried = True
                state.patcher_count     += 1
                state.last_wrong_rubrics = wrong_rubrics
                state.last_wrong_rubric_details = _rubric_details
                logger.info(f"[id={problem_id}] substep 분해 불가(atomic) → patcher")
        elif not state.step_patcher_tried:
            # substep도 실패 → patcher 1회
            state.use_patcher               = True
            state.is_rethink                = False
            state.step_patcher_tried        = True
            state.patcher_count            += 1
            state.last_wrong_rubrics        = wrong_rubrics
            state.last_wrong_step_text      = step["text"]
            state.last_wrong_rubric_details = _rubric_details
        else:
            # patcher도 실패 → 종료
            print(f"  [id={problem_id}]  {_fmt(state.all_steps)}  → patcher_fail")
            traj = _build_traj(problem_id, problem, gold_answer,
                               state.all_steps, False, "all",
                               fail_reason="patcher_fail")
            state.traj_list.append(traj)
            if save_fn:
                save_fn(traj)
            state.done = True
            return
        state.rethink_round += 1

    # ── patcher step 특별 처리 ────────────────────────────────────────────────
    if step.get("source") == "patcher":
        if step.get("is_error"):
            # 빈 응답 → patcher_fail
            logger.info(f"[id={problem_id}] patcher empty response → patcher_fail")
            print(f"  [id={problem_id}]  {_fmt(state.all_steps)}  → patcher_fail")
            traj = _build_traj(problem_id, problem, gold_answer,
                               state.all_steps, False, "all",
                               fail_reason="patcher_fail")
            state.traj_list.append(traj)
            if save_fn:
                save_fn(traj)
            state.done = True
            return
        if has_boxed(step["text"]):
            if check_solved(step["text"], gold_answer):
                # patcher가 정답 → PRM 판단 무관하게 correct로 처리
                logger.info(f"[id={problem_id}] patcher solved correctly → treat as correct")
                result = {**result, "result": "correct", "wrong_count": 0}
            else:
                # patcher가 틀린 답 제출 → patcher_fail
                logger.info(f"[id={problem_id}] patcher wrong final answer → patcher_fail")
                state.all_steps.append(step)
                print(f"  [id={problem_id}]  {_fmt(state.all_steps)}  → patcher_fail")
                traj = _build_traj(problem_id, problem, gold_answer,
                                   state.all_steps, False, "all",
                                   fail_reason="patcher_fail")
                state.traj_list.append(traj)
                if save_fn:
                    save_fn(traj)
                state.done = True
                return
        elif result["result"] == "incorrect":
            # patcher 중간 step (boxed 없음): PRM wrong 무시하고 correct로 처리
            logger.info(f"[id={problem_id}] patcher mid-step PRM wrong → override to correct")
            result = {**result, "result": "correct", "wrong_count": 0}

    # ── 오류 있음 ─────────────────────────────────────────────────────────────
    if result["result"] == "incorrect":
        logger.info(
            f"[PRM→fail] id={problem_id} step={step_number} "
            f"wrong={wrong_count}/{total} rubrics={wrong_rubrics}"
        )
        _apply_wrong()

    # ── 오류 없음 ─────────────────────────────────────────────────────────────
    else:
        logger.info(f"[API_PRM_checklist] id={problem_id} step={step_number}: correct wrong={wrong_count}/{total}")

        # boxed 정답이 있으면 gold_answer와 직접 비교해 종료 여부 결정
        if has_boxed(step["text"]):
            is_right = check_solved(step["text"], gold_answer)
            if not is_right:
                # PRM_log 오탐: pred ≠ gold_answer → wrong으로 처리
                logger.info(
                    f"[API_PRM_checklist] id={problem_id} step={step_number}: "
                    f"approved but pred≠gold_answer → force wrong"
                )
                state.last_wrong_answer = extract_boxed(step["text"]) or ""
                _apply_wrong("pred≠gold_answer")
                return

        # 정상 정답 스텝: history에 추가
        step["summary"] = {
            "does":                 does_summary,
            "prm_fast_critique":    prm_fast_critique,
            "prm_deep_critique":    prm_deep_critique,
            "prm_critique_review": prm_critique_review,
            "gen_fast_critique":    gen_fast_critique,
            "gen_deep_critique":    gen_deep_critique,
            "gen_critique_review": gen_critique_review,
        }
        state.all_steps.append(step)
        state.history.append(step)

        # ── substep_meta: 쪼개진 스텝인지 기록 ───────────────────────────────
        if state.in_substep_mode and state.substep_queue:
            cur = state.substep_queue[0]
            step["substep_meta"] = {
                "is_substep":    True,
                "depth":         cur.get("depth", 0),
                "goal":          cur.get("goal", ""),
                "substep_index": len(state.substep_passed),  # 0-based
            }
        else:
            step["substep_meta"] = {"is_substep": False}

        # ── 서브스텝 모드: 통과된 서브스텝 처리 ──────────────────────────────
        if state.in_substep_mode and state.substep_queue:
            state.substep_passed.append(step["text"])
            state.substep_queue.pop(0)  # 완료된 서브스텝 제거

            if state.substep_queue:
                # 아직 남은 서브스텝 → 다음 서브스텝 rethink 계속
                state.is_rethink = True
                logger.info(f"[id={problem_id}] 서브스텝 완료, 다음 서브스텝 진행: {state.substep_queue[0]['goal'][:50]}")
            else:
                # 모든 서브스텝 통과 → 정상 궤도 복귀
                state.in_substep_mode    = False
                state.substep_passed     = []
                state.is_rethink         = False
                state.step_rethink_tried = False
                state.step_substep_tried = False
                state.step_patcher_tried = False
                logger.info(f"[id={problem_id}] 모든 서브스텝 통과 → 정상 궤도 복귀")
            return  # 아래 상태 리셋 건너뜀

        state.is_rethink         = False
        state.step_rethink_tried = False
        state.step_substep_tried = False
        state.step_patcher_tried = False

        if has_boxed(step["text"]):
            # is_right=True (위에서 check_solved 통과)
            if state.rethink_round == 0:
                print(f"  ✓  [id={problem_id}]  {_fmt(state.all_steps)}  → correct (gen only)")
            else:
                print(f"  ✓  [id={problem_id}]  {_fmt(state.all_steps)}  → correct (mix)")
            traj = _build_traj(problem_id, problem, gold_answer,
                               state.all_steps, True, "all")
            state.traj_list.append(traj)
            if save_fn:
                save_fn(traj)
            state.done = True
        elif TRAJ_MAX_STEPS and len(state.all_steps) >= TRAJ_MAX_STEPS:
            print(f"  [id={problem_id}]  {_fmt(state.all_steps)}  → max_steps")
            traj = _build_traj(problem_id, problem, gold_answer,
                               state.all_steps, False, "all",
                               fail_reason="max_steps")
            if save_fn:
                save_fn(traj)
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


class _GPUBatchWorker:
    """GPU worker that collects pending states and runs them as a batch.
    A single background thread processes batches so the GPU is never
    called concurrently. Exceptions are propagated to the caller."""

    def __init__(self, fn, model, tokenizer, device,
                 batch_wait: float = 0.15, max_batch: int = 32):
        self._fn        = fn        # fn(model, tokenizer, device, states)
        self._model     = model
        self._tok       = tokenizer
        self._device    = device
        self._wait      = batch_wait
        self._max_batch = max_batch
        self._q: list   = []
        self._q_lock    = threading.Lock()
        self._trigger   = threading.Event()
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, state: "ProblemState") -> "concurrent.futures.Future":
        """Queue a state. Returns a Future resolved when batch completes."""
        fut = concurrent.futures.Future()
        with self._q_lock:
            self._q.append((state, fut))
        self._trigger.set()
        return fut

    def _run(self) -> None:
        while not self._stop.is_set():
            self._trigger.wait(timeout=self._wait)
            self._trigger.clear()
            with self._q_lock:
                if not self._q:
                    continue
                # max_batch 제한: 남은 건 다음 배치로
                batch = self._q[:self._max_batch]
                self._q = self._q[self._max_batch:]
                if self._q:
                    self._trigger.set()  # 남은 항목 즉시 처리
            states = [s for s, _ in batch]
            futs   = [f for _, f in batch]
            try:
                self._fn(self._model, self._tok, self._device, states)
                for f in futs:
                    f.set_result(None)
            except Exception as exc:
                import traceback as _tb
                logger.warning(f"[GPUBatchWorker] batch error: {exc}\n{_tb.format_exc()}")
                for f in futs:
                    if not f.done():
                        f.set_exception(exc)

    def shutdown(self) -> None:
        self._stop.set()
        self._trigger.set()
        self._thread.join()


def generate_batch(
    items: list[dict],
    generators: list,
    prm_model: ApiPrm,
    rubrics: list[dict],
    n_parallel: int,
    save_fn=None,
    prm_save_fn=None,
    rubric_text_save_fn=None,
    prm_filter_save_fn=None,
    step_manager_model=None,
    step_manager_tok=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    n_parallel개 문제를 동시에 처리.
    Returns: (all_traj, all_prm_records)
    """
    all_traj:        list[dict] = []
    all_prm_records: list[dict] = []
    prm_stats  = {"total_steps": 0, "fast_rubric_calls": 0, "rubric_calls": 0}
    _lock      = threading.Lock()
    _save_lock = threading.Lock()
    rubric_names = [r["name"] for r in rubrics]

    pbar = tqdm(total=len(items), desc="generating", unit="prob")

    # ── thread-safe save 래퍼 ────────────────────────────────────────────────
    def _ts(fn):
        if fn is None:
            return None
        def _w(*a, **kw):
            with _save_lock:
                return fn(*a, **kw)
        return _w

    ts_save     = _ts(save_fn)
    ts_prm      = _ts(prm_save_fn)
    ts_rubric   = _ts(rubric_text_save_fn)
    ts_prm_filt = _ts(prm_filter_save_fn)

    # ── PRM API: 단일 state 평가 ─────────────────────────────────────────────
    def _run_prm_one(state: ProblemState) -> dict:
        if state.pending_step.get("is_error"):
            n = len(rubrics)
            logger.info(f"[Generator] id={state.item.get('id','?')} tool_call hallucination → force wrong")
            return {"result": "incorrect", "wrong_count": n, "total": n,
                    "threshold": 1,
                    "votes": {r["name"]: "incorrect" for r in rubrics},
                    "details": {}}
        with _lock:
            prm_stats["total_steps"] += 1
        local = {"total_steps": 0, "fast_rubric_calls": 0, "rubric_calls": 0}
        rm = _run_prm_batch(prm_model, [state], [rubrics], prm_stats=local)
        with _lock:
            prm_stats["fast_rubric_calls"] += local["fast_rubric_calls"]
            prm_stats["rubric_calls"]       += local["rubric_calls"]
        return rm[id(state)]

    # ── GPU worker 설정 ───────────────────────────────────────────────────────
    gen_worker: "_GPUBatchWorker | None" = None
    sum_worker: "_GPUBatchWorker | None" = None
    _gpu_batch = _GT_CFG["batch_per_gpu"]

    if USE_VLLM and generators:
        # ── vLLM 모드: 단일 LLM 인스턴스, gen + sum 공유 (직렬화 lock으로 보호) ──
        _vllm_lock = threading.Lock()
        llm_v, tok_v = generators[0][0], generators[0][1]

        def _gen_fn_vllm(llm, tok, _, states):
            with _vllm_lock:
                _batch_run_generator_vllm(llm, tok, _, states)

        def _sum_fn_vllm(llm, tok, _, states):
            prm_map = {id(s): s._prm_result for s in states}
            with _vllm_lock:
                _batch_run_all_summaries_and_gen_critique(llm, tok, None, states, prm_map)
                _batch_generate_critique_review(llm, tok, None, states, source="prm")

        gen_worker = _GPUBatchWorker(_gen_fn_vllm, llm_v, tok_v, None, max_batch=_gpu_batch)
        sum_worker = _GPUBatchWorker(_sum_fn_vllm, llm_v, tok_v, None, max_batch=_gpu_batch)

    elif len(generators) >= 1:
        # ── HF 모드: gen worker(rollout GPU) + summary worker(step_manager GPU 0) ──
        gen_m, gen_t, gen_d = generators[0]
        gen_worker = _GPUBatchWorker(_batch_run_generator, gen_m, gen_t, gen_d,
                                     max_batch=_gpu_batch)

        def _sum_fn(model, tokenizer, device, states):
            prm_map = {id(s): s._prm_result for s in states}
            _batch_run_step_summary_only(model, tokenizer, device, states)
            _batch_run_all_summaries_and_gen_critique(model, tokenizer, device, states, prm_map)
            _batch_generate_critique_review(model, tokenizer, device, states, source="prm")

        # step_manager가 로드된 경우 요약 전용 GPU(0) 사용, 아니면 gen GPU 공유
        if step_manager_model is not None:
            sm_device = next(step_manager_model.parameters()).device
            sum_worker = _GPUBatchWorker(_sum_fn, step_manager_model, step_manager_tok,
                                         sm_device, max_batch=_gpu_batch)
            logger.info(f"Summary worker → step_manager (GPU {STEP_MANAGER_GPU})")
        elif len(generators) >= 2:
            sm, st, sd = generators[1]
            sum_worker = _GPUBatchWorker(_sum_fn, sm, st, sd, max_batch=_gpu_batch)
            logger.info("Summary worker → generators[1]")

    # ── 문제별 독립 파이프라인 (GPU worker 모드) ──────────────────────────────
    def _run_problem_pipeline(item: dict) -> None:
        """문제 하나를 완주. gen_worker / sum_worker가 있으면 dynamic batching 사용."""
        state = ProblemState(item=item)

        while not state.done:
            # ① Generation (gen/rethink/patcher)
            if state.use_patcher:
                _run_patcher_api([state])
            elif gen_worker:
                gen_worker.submit(state).result()
            else:
                raise RuntimeError("gen_worker가 없을 때 pipeline 모드 진입")

            # ② PRM API (다른 문제들의 GPU 작업과 동시 실행)
            result = _run_prm_one(state)
            state._prm_result = result

            # ③ CPU 파싱
            _extract_gen_fast_critique([state], rubric_names)

            # ④ GPU summary (sum_worker에 제출 → 다른 문제들과 배치)
            if sum_worker:
                sum_worker.submit(state).result()
            elif step_manager_model is not None:
                prm_map = {id(state): state._prm_result}
                sm_device = next(step_manager_model.parameters()).device
                _batch_run_step_summary_only(step_manager_model, step_manager_tok, sm_device, [state])
                _batch_run_all_summaries_and_gen_critique(step_manager_model, step_manager_tok, sm_device, [state], prm_map)
                _batch_generate_critique_review(step_manager_model, step_manager_tok, sm_device, [state], source="prm")

            # ⑤ 결과 처리
            _process_prm_result(
                state, result, rubrics,
                ts_save, ts_prm, ts_rubric, ts_prm_filt,
                step_manager_model, step_manager_tok,
            )

        with _lock:
            all_traj.extend(state.traj_list)
            all_prm_records.extend(state.prm_records)
        pbar.update(1)

    # ── eval cycle (1-GPU 순차 배치 모드) ────────────────────────────────────
    def _run_gpu_summaries_batch(states: list, prm_results_map: dict) -> None:
        if not states:
            return
        if not generators and step_manager_model is None:
            return
        if USE_VLLM:
            llm_s, tok_s = generators[0][0], generators[0][1]
            _batch_run_all_summaries_and_gen_critique(llm_s, tok_s, None, states, prm_results_map)
            _batch_generate_critique_review(llm_s, tok_s, None, states, source="prm")
        elif generators:
            _parallel_gen(_batch_run_step_summary_only, generators, states)
            _prm_snap = prm_results_map
            _parallel_gen(
                lambda m, t, d, sts: _batch_run_all_summaries_and_gen_critique(m, t, d, sts, _prm_snap),
                generators, states,
            )
            _parallel_gen(
                lambda m, t, d, sts: _batch_generate_critique_review(m, t, d, sts, source="prm"),
                generators, states,
            )
        elif step_manager_model is not None:
            sm_device = next(step_manager_model.parameters()).device
            _batch_run_step_summary_only(step_manager_model, step_manager_tok, sm_device, states)
            _batch_run_all_summaries_and_gen_critique(step_manager_model, step_manager_tok, sm_device, states, prm_results_map)
            _batch_generate_critique_review(step_manager_model, step_manager_tok, sm_device, states, source="prm")

    def _eval_cycle(states: list) -> list:
        prm_results_map: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(states))) as ex:
            state_futs = {ex.submit(_run_prm_one, s): s for s in states}
            for fut in as_completed(state_futs):
                s = state_futs[fut]
                prm_results_map[id(s)] = fut.result()
        _extract_gen_fast_critique(states, rubric_names)
        _run_gpu_summaries_batch(states, prm_results_map)
        for state in states:
            _process_prm_result(
                state, prm_results_map[id(state)], rubrics,
                save_fn, prm_save_fn,
                rubric_text_save_fn, prm_filter_save_fn,
                step_manager_model, step_manager_tok,
            )
        return [s for s in states if s.use_patcher and not s.done]

    # ── 실행 ─────────────────────────────────────────────────────────────────
    try:
        if gen_worker and sum_worker:
            # ── Dynamic batching 모드 (GPU 2개 이상) ─────────────────────────
            logger.info(f"[generate_batch] dynamic batching 모드  n={len(items)}  n_parallel={n_parallel}")
            with ThreadPoolExecutor(max_workers=n_parallel) as ex:
                futs = [ex.submit(_run_problem_pipeline, item) for item in items]
                for fut in as_completed(futs):
                    fut.result()
        else:
            # ── 순차 배치 모드 (GPU 1개 또는 API 모드) ───────────────────────
            queue  = list(items)
            active: list[ProblemState] = []

            def _flush_done():
                nonlocal active
                for s in [s for s in active if s.done]:
                    all_traj.extend(s.traj_list)
                    all_prm_records.extend(s.prm_records)
                    pbar.update(1)
                active = [s for s in active if not s.done]
                while len(active) < n_parallel and queue:
                    active.append(ProblemState(item=queue.pop(0)))

            while len(active) < n_parallel and queue:
                active.append(ProblemState(item=queue.pop(0)))

            _round_idx = 0
            while active:
                _flush_done()
                if not active:
                    break
                _round_idx += 1
                _t_round = time.time()

                logger.info(f"[Round {_round_idx}] active={len(active)}")
                if USE_VLLM and generators:
                    llm_r, tok_r = generators[0][0], generators[0][1]
                    _batch_run_generator_vllm(llm_r, tok_r, None, active)
                else:
                    _parallel_gen(_batch_run_generator, generators, active)

                to_eval = list(active)
                while to_eval:
                    patcher_states = _eval_cycle(to_eval)
                    if not patcher_states:
                        break
                    _run_patcher_api(patcher_states)
                    to_eval = patcher_states

                logger.info(f"[TIMING] round={_round_idx}  total={time.time()-_t_round:.1f}s")
                _flush_done()
    finally:
        if gen_worker:
            gen_worker.shutdown()
        if sum_worker:
            sum_worker.shutdown()

    pbar.close()
    logger.info(
        f"PRM 통계: total_steps={prm_stats['total_steps']}  "
        f"fast_rubric_calls={prm_stats['fast_rubric_calls']}  "
        f"rubric_calls={prm_stats['rubric_calls']}"
    )
    logger.info(
        f"완료 → traj={len(all_traj)}  prm_records={len(all_prm_records)}"
    )
    return all_traj, all_prm_records


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trajectory SFT 데이터 생성")
    parser.add_argument("--num_data",    type=int, default=None)
    parser.add_argument("--num_start",   type=int, default=None)
    parser.add_argument("--output",      type=str, default=None,
                        help="출력 폴더 경로 (기본: output/sft_trajectory/{timestamp})")
    parser.add_argument("--rubric_file", type=str, default=None,
                        help="루브릭 jsonl 경로 (루브릭별 1회 호출)")
    parser.add_argument("--fast_rubric_file", type=str, default=None,
                        help="fast 루브릭 JSON 경로 (샘플당 1회 호출). 예: prompts/prm_rubric_v6.0_batch.json")
    parser.add_argument("--n_parallel",  type=int, default=None,
                        help="동시 처리 문제 수 (기본: PRM.batch_size // n_rubrics)")
    parser.add_argument("--debug",       type=str, default=None,
                        help="디버그용 문제 ID 파일 (각 줄 첫 단어가 problem_id). 지정 시 해당 문제만 실행")
    parser.add_argument("--resume_folder", type=str, default=None,
                        help="이전 실행 폴더. 해당 폴더의 traj_all.jsonl에 있는 problem_id는 건너뜀")
    args = parser.parse_args()

    root   = Path(__file__).resolve().parent.parent
    gt_cfg = _GT_CFG

    if "base_problems" not in gt_cfg:
        raise KeyError("config.generate_trajectory.base_problems 설정이 없습니다")
    dataset_path = gt_cfg["base_problems"]

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir     = Path(args.output) if args.output else (root / "output" / "sft_trajectory" / ts)
    prm_out_dir = out_dir / "prm"
    out_dir.mkdir(parents=True, exist_ok=True)
    prm_out_dir.mkdir(parents=True, exist_ok=True)

    # ── 로깅 설정 ────────────────────────────────────────────────────────────
    log_path = out_dir / "run.log"
    _setup_logging(log_path)   # 콘솔 + 파일 동시 기록

    # ── 출력 파일 ────────────────────────────────────────────────────────────
    traj_all_file = open(out_dir / "traj_all.jsonl", "w", encoding="utf-8")
    prm_eval_file    = open(prm_out_dir / "prm_evals.jsonl",  "w", encoding="utf-8")
    prm_filter_file  = open(prm_out_dir / "prm_filter.jsonl", "w", encoding="utf-8")
    _run_jsonl_file  = open(out_dir / "run.jsonl", "w", encoding="utf-8")
    _run_jsonl_lock  = __import__("threading").Lock()
    def _run_log(record: dict):
        line = json.dumps(record, ensure_ascii=False)
        with _run_jsonl_lock:
            _run_jsonl_file.write(line + "\n")
            _run_jsonl_file.flush()
    set_run_log(_run_log)

    num_data  = args.num_data  if args.num_data  is not None else gt_cfg.get("num_data", -1)
    num_start = args.num_start if args.num_start is not None else gt_cfg.get("num_start", 0)
    num_end   = gt_cfg.get("num_end")

    # ── GPU 설정 ─────────────────────────────────────────────────────────────
    rollout_gpus = gt_cfg["rollout_gpus"]
    # rollout_gpus만 노출 → GPU 0 등 다른 GPU 접근 차단 (OOM 방지)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in rollout_gpus)
    prm_gpu_ids  = _PRM_CFG["gpu_id"]
    n_prm_gpus     = len(prm_gpu_ids)
    PRM_BATCH_SIZE = PRM_BATCH_PER_GPU * n_prm_gpus
    logger.info(f"prm_gpu_ids={prm_gpu_ids}  rollout_gpus={rollout_gpus}  CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # ── 데이터 & 루브릭 로드 ─────────────────────────────────────────────────
    items = load_dataset_file(dataset_path)
    if args.debug:
        debug_path = Path(args.debug) if Path(args.debug).is_absolute() else root / args.debug
        with open(debug_path, encoding="utf-8") as f:
            debug_ids = {line.split()[0] for line in f if line.strip()}
        items = [it for it in items if str(it["id"]) in debug_ids]
        logger.info(f"[debug] {debug_path} 에서 {len(debug_ids)}개 ID 로드 → 매칭 문제: {len(items)}개")
    else:
        if num_end is not None:
            items = items[num_start:num_end]
        elif num_data == -1:
            items = items[num_start:]
        else:
            items = items[num_start: num_start + num_data]

    if args.resume_folder:
        resume_path = Path(args.resume_folder) / "traj_all.jsonl"
        done_ids: set[str] = set()
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as _rf:
                for _line in _rf:
                    _line = _line.strip()
                    if _line:
                        _rec = json.loads(_line)
                        pid = _rec.get("problem_id")
                        if pid:
                            done_ids.add(str(pid))
            before = len(items)
            items = [it for it in items if str(it.get("id", "")) not in done_ids]
            logger.warning(
                f"[resume] {resume_path} 에서 완료 ID {len(done_ids)}개 로드 "
                f"→ {before - len(items)}개 건너뜀, {len(items)}개 처리 예정"
            )
        else:
            logger.warning(f"[resume] {resume_path} 파일 없음 → 건너뜀 없이 전체 실행")

    logger.info(f"로드된 문제 수: {len(items)}")

    rubric_path = args.rubric_file or _PRM_CFG.get("rubric")
    if not rubric_path:
        raise ValueError("루브릭 파일 경로를 지정해 주세요: --rubric_file 혹은 config.PRM.rubric 설정")
    if not Path(rubric_path).is_absolute():
        rubric_path = str(root / rubric_path)

    rubrics      = load_rubrics(rubric_path)
    fast_rubric  = None

    n_rubrics  = len(rubrics)
    n_parallel = args.n_parallel or gt_cfg["batch_per_gpu"] * len(rollout_gpus)

    logger.info(
        f"데이터셋={dataset_path}  sft출력={out_dir}  prm출력={prm_out_dir}  "
        f"num_start={num_start}  num_end={num_end}  num_data={num_data}  "
        f"PRM_BATCH_SIZE={PRM_BATCH_SIZE}  n_rubrics={n_rubrics}  n_parallel={n_parallel}"
    )

    # ── Generator 로드 ────────────────────────────────────────────────────────
    base_model_id = CONF["checkpoint"]["base"]
    generators = []

    # CUDA_VISIBLE_DEVICES = "3,4" → 상대 인덱스: 0=rollout_gpus[0], 1=rollout_gpus[1], ...
    # rollout_gpus[0] → step_manager (상대 인덱스 0)
    # rollout_gpus[1:] → vLLM generation (physical ids for vLLM worker subprocess)
    sm_gpu   = 0  # 상대 인덱스 0 = physical rollout_gpus[0]
    gen_gpus = rollout_gpus[1:] if len(rollout_gpus) > 1 else rollout_gpus
    logger.info(f"GPU 배분: step_manager=cuda:{sm_gpu}(physical {rollout_gpus[0]})  generator={gen_gpus}")

    # ── Step Manager 로드 (요약·분해 전용) ────────────────────────────────────
    step_manager_model = None
    step_manager_tok   = None
    sm_model, sm_tok = load_step_manager(gpu_id=sm_gpu)
    step_manager_model = sm_model
    step_manager_tok   = sm_tok
    logger.info(f"Step Manager 로드 완료: cuda:{sm_gpu}(physical {rollout_gpus[0]})  path={STEP_MANAGER_PATH}")

    # ── Generator 로드 ────────────────────────────────────────────────────────
    if USE_VLLM:
        if len(gen_gpus) == 0:
            raise ValueError(
                f"vLLM 사용 시 rollout_gpus에 최소 2개 GPU 필요 "
                f"(현재: {rollout_gpus}, step_manager=cuda:{sm_gpu})"
            )
        # vLLM worker subprocess는 physical gen_gpus만 노출
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gen_gpus)
        logger.info(
            f"Generator vLLM 모드: {base_model_id}  "
            f"tensor_parallel={len(gen_gpus)}  gpus={gen_gpus}"
        )
        llm, tokenizer = load_generator_vllm(
            model_path=base_model_id,
            rollout_gpus=gen_gpus,
        )
        generators.append((llm, tokenizer, None))
        logger.info("Generator vLLM 로드 완료")
    else:
        for i, _gpu in enumerate(gen_gpus):
            device_map = {"": f"cuda:{i + 1}"}  # 상대 인덱스: sm=0, gen=1,2,...
            logger.info(f"Generator 로딩 중: {base_model_id}  device_map={device_map}(physical {_gpu})")
            model, tokenizer = load_generator(model_path=base_model_id, device_map=device_map)
            generators.append((model, tokenizer, next(model.parameters()).device))
            logger.info(f"Generator 로드 완료 (device={generators[-1][2]})")
        logger.info(f"Generator {len(generators)}개 로드 완료")

    # ── PRM 로드 ──────────────────────────────────────────────────────────────
    if not _PRM_API:
        raise ValueError("config.yaml의 PRM.model_id에 모델 이름을 설정해 주세요.")
    fast_rubric_path = args.fast_rubric_file or _PRM_CFG.get("fast_rubric")
    if fast_rubric_path:
        if not Path(fast_rubric_path).is_absolute():
            fast_rubric_path = str(root / fast_rubric_path)
        fast_rubric = load_fast_rubric(Path(fast_rubric_path))
        prm_model = ApiPrmTwoStage(_PRM_API, fast_rubric, rubrics, max_workers=n_parallel)
        logger.info(
            f"PRM 2-Stage 모드: {_PRM_API}  "
            f"stage1(fast)={fast_rubric_path}  stage2(rubric)={rubric_path}"
        )
    else:
        prm_model = ApiPrm(_PRM_API, max_workers=n_parallel)
        logger.info(f"PRM API 모드: {_PRM_API}  루브릭={n_rubrics}개  ({rubric_path})")

    # ── 실행 메타 기록 ────────────────────────────────────────────────────────
    run_meta = {
        "timestamp":        ts,
        "rubric_file":      rubric_path,
        "fast_rubric_file": fast_rubric_path or None,
        "dataset":          dataset_path,
        "num_start":        num_start,
        "num_data":         num_data,
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as _f:
        json.dump(run_meta, _f, indent=2, ensure_ascii=False)
    logger.info(f"run_meta.json 저장: rubric_file={Path(rubric_path).name}")

    prm_summary = {
        "timestamp":        ts,
        "rubric_file":      rubric_path,
        "rubric_version":   Path(rubric_path).stem,
        "fast_rubric_file": fast_rubric_path or None,
        "fast_rubric_version": Path(fast_rubric_path).stem if fast_rubric_path else None,
        "n_rubrics":        n_rubrics,
    }
    with open(prm_out_dir / "prm_summary.json", "w", encoding="utf-8") as _f:
        json.dump(prm_summary, _f, indent=2, ensure_ascii=False)
    logger.info(f"prm_summary.json 저장: {prm_out_dir / 'prm_summary.json'}")

    # ── 저장 함수 ─────────────────────────────────────────────────────────────
    count = 0

    def _save(traj: dict):
        nonlocal count
        line = json.dumps(traj, ensure_ascii=False) + "\n"
        traj_all_file.write(line); traj_all_file.flush()
        count += 1

    def _save_prm_record(rec: dict):
        prm_eval_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        prm_eval_file.flush()

    def _save_prm_filter(rec: dict):
        prm_filter_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        prm_filter_file.flush()

    t_start = time.time()

    try:
        generate_batch(
            items,
            generators,
            prm_model, rubrics,
            n_parallel=n_parallel,
            save_fn=_save,
            prm_save_fn=_save_prm_record,
            rubric_text_save_fn=None,
            prm_filter_save_fn=_save_prm_filter,
        )
    finally:
        traj_all_file.close()
        prm_eval_file.close()
        prm_filter_file.close()
        _run_jsonl_file.close()
        set_run_log(None)
        _print_cost_summary()

    elapsed_min = (time.time() - t_start) / 60
    logger.info(
        f"완료: {len(items)}개 문제 / {count}개 trajectory  "
        f"소요={elapsed_min:.1f}분  sft출력={out_dir}  prm출력={prm_out_dir}"
    )


if __name__ == "__main__":
    main()
