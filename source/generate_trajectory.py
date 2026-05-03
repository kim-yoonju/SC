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
  traj_gen.jsonl   generator 단독 정답 (PRM_log 오류 미발견)
  traj_mix.jsonl   gen-PRM_log 혼합, generator가 최종 정답
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
    _call_llm, PATCHER, PATCHER_MAX_NEW_TOKENS,
    _record_usage, _print_cost_summary, set_run_log, set_call_role,
)

from generate_utils import load_dataset_file

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts"
_ROOT_PATH    = Path(__file__).resolve().parent.parent

def _load_action_prompts() -> dict[str, str]:
    rubric_lines = []
    _rubric_rel  = CONF.get("PRM", {}).get("rubric", "prompts/prm_rubric_v6.2.jsonl")
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
_WRONG_STEP_SUMMARY_PROMPT  = _ACTION_PROMPTS["step_summary"]
_CRITIQUE_SUMMARY_PROMPT    = _ACTION_PROMPTS["critique_summary"]

from PRM import (
    ApiPrm, ApiPrmBatch, ApiPrmTwoStage,
    evaluate_step, load_rubrics, load_fast_rubric,
    build_system_prompt, build_user_message,
)

_GENERATOR_API = CONF.get("API_model", {}).get("GENERATOR")
_SUMMARY_API   = CONF.get("API_model", {}).get("SUMMARY") or _GENERATOR_API
_PRM_API       = CONF.get("PRM", {}).get("model_id")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_GT_CFG             = CONF.get("generate_trajectory", {})
TRAJ_MAX_NEW_TOKENS = _GT_CFG.get("max_new_tokens", 4096)
TRAJ_MAX_STEPS      = _GT_CFG.get("max_steps", None)
PRM_MAX_NEW_TOKENS  = CONF.get("PRM", {}).get("max_new_tokens", 1024)


def _extract_verdicts_from_text(sc_text: str) -> tuple[int, int]:
    """self-check 텍스트에서 correct/incorrect 카운트 추출.

    1차: \\boxed{correct}, \\boxed{\\text{correct}} 등 boxed 패턴
    2차: ': correct' / ': incorrect' 평문 패턴
    """
    na = len(re.findall(r":\s*(?:not applicable|n/a)\b", sc_text, re.I))

    boxed = re.findall(
        r"\\boxed\{(?:\\text\{)?\s*(correct|incorrect)\s*\}+",
        sc_text, re.I,
    )
    if boxed:
        c = sum(1 for m in boxed if m.lower() == "correct")
        i = sum(1 for m in boxed if m.lower() == "incorrect")
        return c, i, na

    c = len(re.findall(r":\s*correct\b", sc_text, re.I))
    i = len(re.findall(r":\s*incorrect\b", sc_text, re.I))
    return c, i, na


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
    last_wrong_rubrics:       list = field(default_factory=list)
    last_wrong_step_text:     str  = ""
    last_wrong_step_summary:  str  = ""
    last_wrong_rubric_details: dict = field(default_factory=dict)  # {rubric_name: reasoning}
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

def _build_rethink_explanation(state: "ProblemState") -> str:
    """rethink 프롬프트용 오류 설명 — 요약 + 실패 루브릭 reasoning 포함."""
    parts = []
    if state.last_wrong_step_summary:
        parts.append(state.last_wrong_step_summary)
    if state.last_wrong_rubric_details:
        parts.append("Failed evaluation criteria:")
        for rubric_name, reasoning in state.last_wrong_rubric_details.items():
            # reasoning은 PRM_critique(요약) 우선, 없으면 루브릭 이름만
            if reasoning and reasoning.strip():
                parts.append(f"- {rubric_name}: {reasoning.strip()}")
            else:
                parts.append(f"- {rubric_name}")
    return "\n".join(parts) if parts else "the previous step contained an error"


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
            system_prompt = GEN_RETHINK_PROMPT.replace("{{error_explanation}}", _build_rethink_explanation(state))
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

        # step_text / self-correction 분리
        sc_idx = full_text.find("\nSelf-correction:")
        if sc_idx != -1:
            step_text       = full_text[:sc_idx].strip()
            self_check_text = full_text[sc_idx:]
        else:
            step_text       = full_text.strip()
            self_check_text = ""

        # correct/incorrect 집계 → 룰 기반 액션 결정
        # 1차: 텍스트 추출 (\boxed{} 및 평문 모두 지원)
        correct_count, incorrect_count, na_count = _extract_verdicts_from_text(self_check_text)
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
            f"correct={correct_count} incorrect={incorrect_count} na={na_count}"
            + (f" [{error_reason}]" if error_reason else "")
        )

        state.pending_step = {
            "text":             step_text,
            "full_text":        full_text,
            "pred_action":      pred_action,
            "next_pred_action": _parse_next_pred_action(self_check_text),
            "source":           "gen",
            "role":             "rethink" if state.is_rethink else "gen",
            "is_error":         is_error,
            "is_first_pat":     state.is_rethink and not is_error,
            "summary":          (
                {"step_analysis": f"{error_reason} — skipped PRM_log"}
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
        lines       = [f"[Problem]\n{state.item['problem']}"]
        if state.history:
            lines.append("\n[Previous steps]")
            for i, s in enumerate(state.history, 1):
                step_ctx = (s.get("summary") or {}).get("does") or s["text"]
                lines.append(f"Step {i}: {step_ctx}")
        explanation = _build_rethink_explanation(state)
        if explanation and explanation != "the previous step contained an error":
            lines.append(f"\n[Note: previous attempts at Step {step_number} were rejected]\n{explanation}")
        lines.append(f"\nWrite Step {step_number}.")
        messages    = [
            {"role": "system", "content": GEN_SOLVE_PROMPT},
            {"role": "user",   "content": "\n".join(lines)},
        ]
        logger.info(f"[Patcher API] id={problem_id} step={step_number} model={PATCHER}")
        try:
            set_call_role("patcher")
            text = _call_llm(PATCHER, messages, max_completion_tokens=PATCHER_MAX_NEW_TOKENS)
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

        state.pending_step = {
            "text":             text,
            "full_text":        text,
            "pred_action":      pred_action,
            "next_pred_action": _parse_next_pred_action(text),
            "source":           "patcher",
            "role":             "patcher",
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
# Generator API: config.API_model.GENERATOR 모델로 스텝 생성
# ─────────────────────────────────────────────────────────────────────────────

def _run_generator_api(states: list[ProblemState]) -> None:
    """API 모델(config.API_model.GENERATOR)로 스텝 생성. 결과를 state.pending_step에 저장."""

    def _call_one(state: ProblemState) -> None:
        step_number = len(state.history) + 1
        problem_id  = state.item.get("id", "?")
        lines       = [f"[Problem]\n{state.item['problem']}"]
        if state.history:
            lines.append("\n[Previous steps]")
            for i, s in enumerate(state.history, 1):
                step_ctx = (s.get("summary") or {}).get("does") or s["text"]
                lines.append(f"Step {i}: {step_ctx}")
        lines.append(f"\nWrite Step {step_number}.")
        if state.is_rethink:
            system_prompt = GEN_RETHINK_PROMPT.replace("{{error_explanation}}", _build_rethink_explanation(state))
        else:
            system_prompt = GEN_SOLVE_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": "\n".join(lines)},
        ]
        logger.info(f"[Generator API] id={problem_id} step={step_number} rethink={state.is_rethink} model={_GENERATOR_API}")
        try:
            set_call_role("generator")
            text = _call_llm(_GENERATOR_API, messages, max_completion_tokens=TRAJ_MAX_NEW_TOKENS)
            text = (text or "").strip()
        except Exception as e:
            logger.warning(f"[Generator API] id={problem_id} 호출 실패: {e}")
            text = ""

        is_error    = not text
        pred_action = TOKEN_END if has_boxed(text) else TOKEN_SOLVE

        state.pending_step = {
            "text":             text,
            "full_text":        text,
            "pred_action":      pred_action,
            "next_pred_action": _parse_next_pred_action(text),
            "source":           "gen",
            "role":             "rethink" if state.is_rethink else "gen",
            "is_error":         is_error,
            "is_first_pat":     state.is_rethink and not is_error,
            "summary":          (
                {"step_analysis": "generator API empty response"}
                if is_error else None
            ),
        }

    with ThreadPoolExecutor(max_workers=max(1, len(states))) as ex:
        list(ex.map(_call_one, states))


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
            prev_lines = [f"Step {i+1}: {s['text']}" for i, s in enumerate(state.history)]
            prev_steps_list.append("\n".join(prev_lines))
            step_numbers.append(len(state.history) + 1)

        s1_verdicts_list = prm_model.stage1.evaluate_batch(
            questions  = [state.item["problem"] for state in prm_states],
            prev_steps = prev_steps_list,
            now_steps  = [f"Step {n}: {state.pending_step['text']}"
                          for n, state in zip(step_numbers, prm_states)],
            max_new_tokens = PRM_MAX_NEW_TOKENS,
        )

        # Stage 1 결과 집계
        s1_results = {}
        for state, verdicts in zip(prm_states, s1_verdicts_list):
            votes = {name: ("pass" if v["pred"] == "pass" else "fail")
                     for name, v in zip(prm_model.stage1.rubric_names, verdicts)}
            n_wrong = sum(1 for v in votes.values() if v == "fail")
            s1_results[id(state)] = {
                "result":      "fail" if n_wrong >= 1 else "pass",
                "wrong_count": n_wrong,
                "total":       len(votes),
                "threshold":   1,
                "votes":       votes,
                "fast_critiques": {name: v.get("critique")
                                   for name, v in zip(prm_model.stage1.rubric_names, verdicts)},
                "details":     {name: {"reasoning": v.get("reasoning"),
                                       "verdict_text": v.get("verdict_text", ""),
                                       "full_response": v.get("full_response"),
                                       "prob_correct": v.get("prob_correct"),
                                       "prob_incorrect": v.get("prob_incorrect"),
                                       "method": "api_batch_stage1"}
                                for name, v in zip(prm_model.stage1.rubric_names, verdicts)},
                "prm_n":       1,
            }

        # ── Stage 2: Stage 1 fail인 경우에만 개별 루브릭 재평가 ────────────────
        fail_states = [s for s in prm_states if s1_results[id(s)]["result"] == "fail"]

        # 샘플별 stage1 결과 로깅
        for state in prm_states:
            r = s1_results[id(state)]
            pid = state.item.get("id", "?")
            step_n = len(state.history) + 1
            n_fail = r["wrong_count"]
            n_total = r["total"]
            if n_fail > 0:
                failed = [name for name, v in r["votes"].items() if v == "fail"]
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
                prev_lines  = [f"Step {i+1}: {s['text']}" for i, s in enumerate(state.history)]
                # stage1에서 fail인 루브릭만 재평가
                failed_names   = {n for n, v in s1_results[sid]["votes"].items() if v == "fail"}
                stage2_rubrics = [r for r in prm_model.rubrics if r["name"] in failed_names]
                now_step = f"Step {step_number}: {state.pending_step['text']}"
                if state.pending_step.get("source") == "patcher":
                    now_step += (
                        "\n\n[Evaluator note: This step was generated by the patcher "
                        "and is guaranteed correct. Analyze each rubric for informational "
                        "purposes only, and output Verdict: correct for every rubric.]"
                    )
                verdict, detail = evaluate_step(
                    question   = state.item["problem"],
                    prev_steps = "\n".join(prev_lines),
                    now_step   = now_step,
                    rubrics    = stage2_rubrics,
                    model      = prm_model.stage2,
                    fail_k     = 1,
                    max_new_tokens = PRM_MAX_NEW_TOKENS,
                    cot        = True,
                )
                s2_votes = {name: ("pass" if res["pred"] == "pass" else "fail")
                            for name, res in detail.items()}
                s2_dets  = {name: {"reasoning":      res.get("reasoning"),
                                   "verdict_text":   res.get("verdict_text", ""),
                                   "full_response":  res.get("full_response"),
                                   "prob_correct":   res.get("prob_correct"),
                                   "prob_incorrect": res.get("prob_incorrect"),
                                   "method":         res.get("method", "api_stage2")}
                            for name, res in detail.items()}
                # stage1 pass 루브릭 + stage2 재평가 루브릭 합산
                all_votes = {**s1_results[sid]["votes"], **s2_votes}
                all_dets  = {**s1_results[sid]["details"], **s2_dets}
                n_wrong   = sum(1 for v in all_votes.values() if v == "fail")
                return id(state), {
                    "result":      "fail" if n_wrong >= 1 else "pass",
                    "wrong_count": n_wrong,
                    "total":       len(all_votes),
                    "threshold":   1,
                    "votes":       all_votes,
                    "details":     all_dets,
                    "prm_n":       len(s2_votes),
                    "_s2_rubric_count": len(stage2_rubrics),
                }

            with ThreadPoolExecutor(max_workers=max(1, len(fail_states))) as ex:
                for state_id, res in ex.map(_eval_stage2, fail_states):
                    s2_results[state_id] = res

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
            prev_lines = [f"Step {i+1}: {s['text']}" for i, s in enumerate(state.history)]
            prev_steps_list.append("\n".join(prev_lines))
            step_numbers.append(len(state.history) + 1)

        verdicts_list = prm_model.evaluate_batch(
            questions  = [state.item["problem"] for state in prm_states],
            prev_steps = prev_steps_list,
            now_steps  = [f"Step {n}: {state.pending_step['text']}"
                          for n, state in zip(step_numbers, prm_states)],
            max_new_tokens = PRM_MAX_NEW_TOKENS,
        )

        results = {}
        for state, verdicts in zip(prm_states, verdicts_list):
            votes = {
                name: ("pass" if v["pred"] == "pass" else "fail")
                for name, v in zip(prm_model.rubric_names, verdicts)
            }
            dets = {
                name: {
                    "reasoning":      v.get("reasoning"),
                    "verdict_text":   v.get("verdict_text", ""),
                    "full_response":  v.get("full_response"),
                    "prob_correct":   v.get("prob_correct"),
                    "prob_incorrect": v.get("prob_incorrect"),
                    "method":         v.get("method", "api_batch"),
                }
                for name, v in zip(prm_model.rubric_names, verdicts)
            }
            n_wrong = sum(1 for v in votes.values() if v == "fail")
            results[id(state)] = {
                "result":      "fail" if n_wrong >= 1 else "pass",
                "wrong_count": n_wrong,
                "total":       len(votes),
                "threshold":   1,
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
            return id(state), {"result": "pass", "wrong_count": 0, "total": 0, "threshold": 1, "votes": {}, "details": {}, "prm_n": 0}
        step_number = len(state.history) + 1
        prev_lines  = [f"Step {i+1}: {s['text']}" for i, s in enumerate(state.history)]
        verdict, detail = evaluate_step(
            question   = state.item["problem"],
            prev_steps = "\n".join(prev_lines),
            now_step   = f"Step {step_number}: {state.pending_step['text']}",
            rubrics    = step_rubrics,
            model      = prm_model,
            fail_k     = 1,
            max_new_tokens = PRM_MAX_NEW_TOKENS,
            cot        = True,
        )
        votes = {name: ("pass" if res["pred"] == "pass" else "fail") for name, res in detail.items()}
        dets  = {
            name: {
                "reasoning":      res.get("reasoning"),
                "verdict_text":   res.get("verdict_text", ""),
                "full_response":  res.get("full_response"),
                "prob_correct":   res.get("prob_correct"),
                "prob_incorrect": res.get("prob_incorrect"),
                "method":         res.get("method", "api"),
            }
            for name, res in detail.items()
        }
        n_wrong = sum(1 for v in votes.values() if v == "fail")
        return id(state), {
            "result":      "fail" if verdict == "fail" else "pass",
            "wrong_count": n_wrong,
            "total":       len(votes),
            "threshold":   1,
            "votes":       votes,
            "details":     dets,
            "prm_n":       len(votes),
        }

    with ThreadPoolExecutor(max_workers=max(1, len(prm_states))) as ex:
        pairs = list(ex.map(_eval_one, zip(prm_states, rubrics_per_step)))
    if prm_stats is not None:
        prm_stats["rubric_calls"] += sum(len(r) for r in rubrics_per_step)
    return dict(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Step 요약: generator로 현재 스텝 추론을 한 줄로 요약
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARIZE_SYSTEM = (
    "You are a math tutor. "
    "In one concise sentence, describe what the following reasoning step does mathematically. "
    "Focus on the key equations, substitutions, or results — include the actual mathematical expressions (e.g., 'substitutes u=1/y² to obtain du/dx=−2u/(e^x·u+1)'). "
    "Do not evaluate correctness — only describe the action taken."
)
_RUBRIC_DOES_SYSTEM = (
    "You are a math tutor. "
    "In one concise sentence, summarize what this rubric criterion observed about the reasoning step. "
    "Do not repeat the verdict — only describe what was noted."
)
_SUMMARIZE_MAX_TOKENS         = 128
_CRITIQUE_SUMMARY_MAX_TOKENS  = 256


def _generate_rubric_does(rubric_text: list[dict]) -> list[dict]:
    """각 루브릭의 reasoning을 generator API로 한 줄 요약."""
    if not _SUMMARY_API or not rubric_text:
        return [{"rubric": rt["rubric"], "does": None} for rt in rubric_text]

    def _one(rt):
        reasoning = (rt.get("reasoning") or "")[:500]
        if not reasoning:
            return {"rubric": rt["rubric"], "does": None}
        try:
            msgs = [
                {"role": "system", "content": _RUBRIC_DOES_SYSTEM},
                {"role": "user",   "content": f"Rubric: {rt['rubric']}\n\nReasoning:\n{reasoning}"},
            ]
            set_call_role("rubric_does")
            raw = _call_llm(_SUMMARY_API, msgs, max_completion_tokens=64) or ""
            summary = re.split(r"[\n.]", raw.strip())[0].strip()
        except Exception:
            summary = None
        return {"rubric": rt["rubric"], "does": summary or None}

    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=min(9, len(rubric_text))) as ex:
        return list(ex.map(_one, rubric_text))


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
        summary = re.split(r"\n", raw)[0].strip()
        state.pending_step["does_summary"] = summary or None


def _api_summarize_steps(states: list[ProblemState]) -> None:
    """API 모드용 does_summary 생성."""
    def _one(state):
        if state.pending_step.get("is_error"):
            state.pending_step["does_summary"] = None
            return
        step_text = (state.pending_step.get("text") or "")[:1200]
        try:
            msgs = [
                {"role": "system", "content": _SUMMARIZE_SYSTEM},
                {"role": "user",   "content": f"Step:\n{step_text}"},
            ]
            set_call_role("does")
            raw = _call_llm(_SUMMARY_API, msgs, max_completion_tokens=_SUMMARIZE_MAX_TOKENS) or ""
            summary = re.split(r"\n", raw.strip())[0].strip()
            state.pending_step["does_summary"] = summary or None
        except Exception:
            state.pending_step["does_summary"] = None

    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=min(16, len(states))) as ex:
        list(ex.map(_one, states))


def _api_summarize_wrong_steps(states: list[ProblemState]) -> None:
    """API 모드: 직전 wrong step을 generator API로 요약 → last_wrong_step_summary 저장."""
    if not _GENERATOR_API:
        return
    to_summarize = [s for s in states if s.last_wrong_step_text and not s.last_wrong_step_summary]
    if not to_summarize:
        return

    def _one(state):
        try:
            msgs = [
                {"role": "system", "content": _WRONG_STEP_SUMMARY_PROMPT},
                {"role": "user",   "content": f"Step:\n{state.last_wrong_step_text[:1200]}"},
            ]
            set_call_role("step_summary")
            raw = _call_llm(_SUMMARY_API, msgs, max_completion_tokens=_SUMMARIZE_MAX_TOKENS) or ""
            state.last_wrong_step_summary = re.split(r"\n", raw.strip())[0].strip()
        except Exception:
            state.last_wrong_step_summary = ""

    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=min(16, len(to_summarize))) as ex:
        list(ex.map(_one, to_summarize))


def _batch_summarize_wrong_steps(
    model, tokenizer, input_device,
    states: list[ProblemState],
) -> None:
    """rethink 상태인 states의 last_wrong_step_text를 generator로 한 줄 오류 요약."""
    to_summarize = [s for s in states if s.last_wrong_step_text and not s.last_wrong_step_summary]
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


_CRITIQUE_SUMMARY_SYSTEM = (
    "You are a math evaluator summarizing a rubric judgment on a solution step. "
    "Write 2–3 sentences focused specifically on THIS rubric.\n\n"
    "If verdict is FAIL:\n"
    "  • Name the exact error this rubric detected (wrong formula, sign error, missing term, incorrect cancellation, etc.)\n"
    "  • Quote or reference the specific expression or claim in the step that is wrong\n"
    "  • State what the correct value, form, or reasoning should be\n\n"
    "If verdict is PASS:\n"
    "  • State what this rubric checked and what it confirmed, OR write 'N/A — this rubric does not apply to this step'\n\n"
    "Be concrete: name specific formulas, values, or expressions from the step. "
    "Avoid vague statements like 'the step is correct' or 'no error was found'."
)



def _parse_rubric_lines(section: str, rubric_names: list[str]) -> dict[str, dict]:
    """섹션 텍스트에서 루브릭별 verdict + critique 파싱."""
    import re as _re
    results = {}
    for line in section.split("\n"):
        line = line.strip()
        if not line:
            continue
        for rubric in rubric_names:
            if rubric.lower() in line.lower():
                vm = _re.search(r"Verdict\s*:\s*(correct|incorrect)", line, _re.IGNORECASE)
                if vm:
                    verdict = "pass" if vm.group(1).lower() == "correct" else "fail"
                elif "incorrect" in line.lower():
                    verdict = "fail"
                elif "correct" in line.lower():
                    verdict = "pass"
                else:
                    verdict = None
                rp = _re.search(rf"{_re.escape(rubric)}\s*[:\.]?\s*(.+?)(?:\s+Verdict\s*:.*)?$", line, _re.IGNORECASE)
                critique = None
                if rp:
                    cand = _re.sub(r"\s*Verdict\s*:.*$", "", rp.group(1), flags=_re.IGNORECASE).strip()
                    critique = cand if cand else None
                results[rubric] = {"verdict": verdict, "critique": critique}
                break
    return results


def _parse_generator_self_check(text: str, rubric_names: list[str]) -> dict[str, dict]:
    """Fast critic 섹션 파싱 (Fast critic: / fast_critique: 모두 지원).
    반환: {rubric_name: {"verdict", "critique"}}"""
    import re as _re
    m = _re.search(
        r"(?:Fast\s+critic|fast_critique)\s*:\s*\n(.*?)(?=(?:Deep\s+critic|deep_critique)\s*:|$)",
        text, _re.DOTALL | _re.IGNORECASE
    )
    if m:
        return _parse_rubric_lines(m.group(1), rubric_names)
    m = _re.search(r"Self-correct(?:ion)?\s*[:\s]*\n(.*)", text, _re.DOTALL | _re.IGNORECASE)
    if not m:
        return {}
    return _parse_rubric_lines(m.group(1), rubric_names)


def _parse_generator_deep_check(text: str, rubric_names: list[str]) -> dict[str, dict]:
    """Deep critic 섹션 파싱 (Deep critic: / deep_critique: 모두 지원).
    반환: {rubric_name: {"verdict", "critique"}}

    지원 포맷:
      - [rubric]: [reasoning] Verdict: correct/incorrect
      - [rubric]: Verdict: incorrect — [reasoning after]
      - 멀티라인 블록
    """
    import re as _re
    m = _re.search(
        r"(?:Deep\s+critic|deep_critique)\s*:\s*\n(.*?)$",
        text, _re.DOTALL | _re.IGNORECASE
    )
    if not m:
        return {}
    section = m.group(1)

    # 첫 번째 비어있지 않은 줄이 "none"이면 딥 크리틱 없음 (이후 내용 무시)
    first_line = next((l.strip() for l in section.split("\n") if l.strip()), "")
    if first_line.lower() == "none":
        return {}

    results = {}
    for i, rubric in enumerate(rubric_names):
        # 다음 루브릭 시작 전까지의 블록 캡처
        later = [_re.escape(r) for r in rubric_names[i + 1:]]
        end_pat = rf"(?:[-*•\s]*(?:{'|'.join(later)})\s*[:\.])" if later else r"\Z"
        pat = _re.search(
            rf"(?:[-*•]\s*)?{_re.escape(rubric)}\s*[:\.]?\s*(.*?)(?={end_pat}|\Z)",
            section, _re.DOTALL | _re.IGNORECASE
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
        vm = _re.search(r"Verdict\s*:\s*(correct|incorrect)", block, _re.IGNORECASE)
        if vm:
            verdict = "pass" if vm.group(1).lower() == "correct" else "fail"
        elif "incorrect" in block.lower():
            verdict = "fail"
        elif "correct" in block.lower():
            verdict = "pass"
        else:
            verdict = None

        # reasoning: verdict 앞 텍스트 + verdict 뒤 텍스트 모두 포함
        if vm:
            before = block[:vm.start()].strip()
            after  = _re.sub(r"^[\s—\-\.]+", "", block[vm.end():]).strip()
            reasoning = " ".join(filter(None, [before, after])) or None
        else:
            reasoning = block or None

        results[rubric] = {"verdict": verdict, "critique": reasoning}

    return results


def _parse_next_pred_action(text: str) -> str | None:
    """Self-correction 블록에서 'Next action: solve/rethink/end' 를 추출."""
    import re as _re
    m = _re.search(r"Next\s+action\s*:\s*(solve|rethink|end)\b", text, _re.IGNORECASE)
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
        step_text = (step.get("text") or "")[:1200]
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


def _batch_run_all_summaries(
    model, tokenizer, input_device,
    states: list[ProblemState],
    prm_results_map: dict,
) -> None:
    """critique summary + wrong step summary를 처리. step summary는 PRM과 병렬로 미리 실행됨."""
    from collections import defaultdict

    prompts: list[str] = []
    tasks:   list      = []  # (type, state, extra)

    for state in states:
        step   = state.pending_step
        result = prm_results_map.get(id(state), {})

        if not step.get("is_error"):
            step_text = (step.get("text") or "")[:1200]

            # ① step summary — PRM과 병렬로 _batch_run_step_summary_only에서 이미 처리됨
            # does_summary가 없는 경우에만 fallback으로 실행
            if "does_summary" not in step:
                prompts.append(build_chat_prompt(tokenizer, _SUMMARIZE_SYSTEM, f"Step:\n{step_text}"))
                tasks.append(("step_summary", state, None))

            # ② critique summary (stage2 평가된 루브릭만)
            votes   = result.get("votes", {})
            details = result.get("details", {})
            step_short = (step.get("text") or "")[:600]
            for rn in votes:
                if (details.get(rn) or {}).get("method") == "api_batch_stage1":
                    continue
                reasoning = (details.get(rn) or {}).get("reasoning", "")[:300]
                user_msg  = (
                    f"Rubric: {rn} (verdict: {votes[rn]})\n"
                    f"Rubric reasoning:\n{reasoning}\n\n"
                    f"Step:\n{step_short}"
                )
                prompts.append(build_chat_prompt(tokenizer, _CRITIQUE_SUMMARY_SYSTEM, user_msg))
                tasks.append(("critique", state, rn))

            # ③ wrong step summary (PRM fail 스텝만)
            if result.get("result") == "fail":
                prompts.append(build_chat_prompt(tokenizer, _WRONG_STEP_SUMMARY_PROMPT, f"Step:\n{step_text}"))
                tasks.append(("wrong_summary", state, None))
        else:
            step["does_summary"]    = None
            step["critique_summary"] = None

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

    critique_by_state: dict = defaultdict(list)

    for j, (task_type, state, extra) in enumerate(tasks):
        raw = tokenizer.decode(out[j, input_len:], skip_special_tokens=True).strip()

        if task_type == "step_summary":
            state.pending_step["does_summary"] = re.split(r"\n", raw)[0].strip() or None

        elif task_type == "critique":
            result = prm_results_map.get(id(state), {})
            critique_by_state[id(state)].append({
                "rubric":       extra,
                "verdict":  result.get("votes", {}).get(extra, "pass"),
                "critique": raw or None,
            })

        elif task_type == "wrong_summary":
            state.pending_step["_wrong_step_summary"] = re.split(r"[\n.]", raw)[0].strip()

    for state in states:
        if not state.pending_step.get("is_error"):
            result = prm_results_map.get(id(state), {})
            votes  = result.get("votes", {})
            filled = {e["rubric"]: e for e in critique_by_state.get(id(state), [])}
            state.pending_step["critique_summary"] = [
                filled.get(rn, {"rubric": rn, "verdict": None, "critique": None})
                for rn in votes
            ] or None
            state.pending_step["gen_fast_critique"] = {
                rn: {"verdict": None, "critique": None} for rn in votes
            } or None


def _batch_summarize_critique(
    model, tokenizer, input_device,
    states: list[ProblemState],
    prm_results_map: dict,
) -> None:
    """모든 스텝에 대해 루브릭별 critique 요약을 생성해 pending_step에 저장.
    로컬 모델 모드: local generator 배치 추론.
    API 모드 (model=None): _GENERATOR_API 병렬 호출.
    tool_call 오류 스텝은 None.
    """
    from collections import defaultdict

    to_summarize = []
    for state in states:
        if state.pending_step.get("is_error"):
            state.pending_step["critique_summary"] = None
        else:
            to_summarize.append(state)

    if not to_summarize:
        return

    # ── API 모드 ─────────────────────────────────────────────────────────────
    if model is None:
        if not _SUMMARY_API:
            for state in to_summarize:
                state.pending_step["critique_summary"] = None
            return

        def _api_one_rubric(args):
            rn, prm_verdict, g_check, reasoning, step_text = args
            g_verdict  = g_check.get("verdict")
            g_critique = g_check.get("critique")
            if not reasoning:
                return ({"rubric": rn, "verdict": prm_verdict, "critique": None},
                        {"verdict": g_verdict, "critique": g_critique})
            try:
                msgs = [
                    {"role": "system", "content": _CRITIQUE_SUMMARY_SYSTEM},
                    {"role": "user",   "content": (
                        f"Rubric: {rn} (verdict: {prm_verdict})\n"
                        f"Rubric reasoning:\n{reasoning[:800]}\n\n"
                        f"Step:\n{step_text[:800]}"
                    )},
                ]
                set_call_role("critique_summary")
                raw = _call_llm(_SUMMARY_API, msgs, max_completion_tokens=_CRITIQUE_SUMMARY_MAX_TOKENS) or ""
                return ({"rubric": rn, "verdict": prm_verdict, "critique": raw.strip() or None},
                        {"verdict": g_verdict, "critique": g_critique})
            except Exception:
                return ({"rubric": rn, "verdict": prm_verdict, "critique": None},
                        {"verdict": g_verdict, "critique": g_critique})

        # 모든 (state, rubric) 쌍을 한 번에 병렬 호출
        # stage1 전용 루브릭(api_batch_stage1)은 stage2 평가 없음 → critique summary 생략
        all_tasks = []  # (state, rn, task_args)
        for state in to_summarize:
            result    = prm_results_map.get(id(state), {})
            votes     = result.get("votes", {})
            details   = result.get("details", {})
            step_text = (state.pending_step.get("text") or "")[:600]
            full_text = state.pending_step.get("full_text") or state.pending_step.get("text") or ""
            g_checks  = _parse_generator_self_check(full_text, list(votes.keys()))
            for rn in votes:
                if (details.get(rn) or {}).get("method") == "api_batch_stage1":
                    continue
                all_tasks.append((
                    state, rn,
                    (rn, votes.get(rn, "pass"), g_checks.get(rn, {}),
                     (details.get(rn) or {}).get("reasoning", ""), step_text)
                ))

        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=min(64, len(all_tasks))) as ex:
            results = list(ex.map(lambda t: _api_one_rubric(t[2]), all_tasks))

        # 결과를 state별로 재조립 (stage2 미평가 루브릭은 null 항목으로 채움)
        from collections import defaultdict
        by_prm:  dict = defaultdict(list)   # {sid: [{rubric, verdict, critique}]}
        by_gen:  dict = defaultdict(dict)   # {sid: {rn: {verdict, critique}}}
        for (state, rn, _), (prm_entry, gen_entry) in zip(all_tasks, results):
            by_prm[id(state)].append(prm_entry)
            by_gen[id(state)][rn] = gen_entry
        for state in to_summarize:
            result = prm_results_map.get(id(state), {})
            votes  = result.get("votes", {})
            filled_prm = {e["rubric"]: e for e in by_prm.get(id(state), [])}
            filled_gen = by_gen.get(id(state), {})
            state.pending_step["critique_summary"] = [
                filled_prm.get(rn, {"rubric": rn, "verdict": None, "critique": None})
                for rn in votes
            ] or None
            state.pending_step["gen_fast_critique"] = {
                rn: filled_gen.get(rn, {"verdict": None, "critique": None})
                for rn in votes
            } or None
        return

    # ── 로컬 모델 모드: (state, rubric) 쌍을 배치로 처리 ──────────────────────
    # stage1 전용 루브릭(api_batch_stage1)은 stage2 평가 없음 → critique summary 생략
    pairs = []  # (state, rn, prm_verdict, g_verdict, prompt)
    for state in to_summarize:
        result     = prm_results_map.get(id(state), {})
        votes      = result.get("votes", {})
        details    = result.get("details", {})
        step_text  = (state.pending_step.get("text") or "")[:600]
        full_text  = state.pending_step.get("full_text") or state.pending_step.get("text") or ""
        g_checks   = _parse_generator_self_check(full_text, list(votes.keys()))
        for rn in votes:
            if (details.get(rn) or {}).get("method") == "api_batch_stage1":
                continue
            prm_verdict = votes[rn]
            g_check     = g_checks.get(rn, {})
            reasoning   = (details.get(rn) or {}).get("reasoning", "")[:800]
            user_msg    = (
                f"Rubric: {rn} (verdict: {prm_verdict})\n"
                f"Rubric reasoning:\n{reasoning}\n\n"
                f"Step:\n{step_text}"
            )
            pairs.append((state, rn, prm_verdict, g_check, build_chat_prompt(tokenizer, _CRITIQUE_SUMMARY_SYSTEM, user_msg)))

    if not pairs:
        return

    prompts   = [p[4] for p in pairs]
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

    results_by_state: dict = defaultdict(list)
    for j, (state, rn, prm_verdict, g_check, _) in enumerate(pairs):
        raw = tokenizer.decode(out[j, input_len:], skip_special_tokens=True).strip()
        results_by_state[id(state)].append({
            "rubric":       rn,
            "verdict":  prm_verdict,
            "critique": raw or None,
            "_verdict":   g_check.get("verdict"),
            "_critique":  g_check.get("critique"),
        })

    state_map = {id(s): s for s in to_summarize}
    for state in to_summarize:
        result = prm_results_map.get(id(state), {})
        votes  = result.get("votes", {})
        filled = {e["rubric"]: e for e in results_by_state.get(id(state), [])}
        state.pending_step["critique_summary"] = [
            {k: v for k, v in filled.get(rn, {"rubric": rn, "verdict": None,
                                               "critique": None}).items()
             if not k.startswith("_")}
            for rn in votes
        ] or None
        state.pending_step["gen_fast_critique"] = {
            rn: {"verdict": (filled.get(rn) or {}).get("_verdict"),
                 "critique": (filled.get(rn) or {}).get("_critique")}
            for rn in votes
        } or None

def _api_summarize_critique(states: list[ProblemState], prm_results_map: dict) -> None:
    """API 모드용 (generators 없을 때): model=None으로 _batch_summarize_critique 호출."""
    _batch_summarize_critique(None, None, None, states, prm_results_map)


def _batch_generate_prm_critique_summary(
    model, tokenizer, input_device, states: list[ProblemState]
) -> None:
    """로컬 generator로 prm_critique_summary 생성 (PRM deep fail 항목 한 단락 요약)."""
    pairs = []
    for state in states:
        step = state.pending_step
        critique_list = step.get("critique_summary") or []
        fail_entries  = [e for e in critique_list
                         if e.get("verdict") == "fail" and e.get("critique")]
        if not fail_entries:
            step["prm_critique_summary"] = None
            continue
        combined  = "\n\n".join(f"[{e['rubric']}]\n{e['critique']}" for e in fail_entries)
        step_text = (step.get("text") or "")[:400]
        user_msg  = f"Step:\n{step_text}\n\nRubric analyses:\n{combined}"
        pairs.append((state, build_chat_prompt(tokenizer, _CRITIQUE_PARA_SUMMARY_SYSTEM, user_msg)))

    if not pairs:
        return
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer([p[1] for p in pairs], return_tensors="pt", padding=True).to(input_device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    for j, (state, _) in enumerate(pairs):
        raw = tokenizer.decode(out[j, input_len:], skip_special_tokens=True).strip()
        state.pending_step["prm_critique_summary"] = raw or None


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
    "Given rubric-specific error analyses of a math solution step, write:\n"
    "1. One concise paragraph summarizing what went wrong mathematically. "
    "Be specific: reference actual expressions or values from the step.\n"
    "2. On a new line, list the rubrics you determined were violated:\n"
    "   Failed rubrics: <comma-separated rubric names, or 'None'>"
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


def _batch_generate_gen_deep_critique(
    model, tokenizer, input_device, states: list[ProblemState], rubrics: list[dict]
) -> None:
    """로컬 generator로 gen_deep_critique 생성 (gen_fast fail 루브릭만 상세 재분석)."""
    rubric_map = {r["name"]: r["criterion"] for r in rubrics}

    # gen_deep_critique 초기화
    for state in states:
        state.pending_step["gen_deep_critique"] = {}

    pairs = []  # (state, rn, prompt)
    for state in states:
        fc = state.pending_step.get("gen_fast_critique") or {}
        for rn, v in fc.items():
            if not (isinstance(v, dict) and v.get("verdict") == "fail"):
                continue
            criterion  = rubric_map.get(rn, rn)
            step_text  = (state.pending_step.get("text") or "")[:600]
            prev_lines = [f"Step {i+1}: {s['text'][:200]}" for i, s in enumerate(state.history)]
            prev_text  = "\n".join(prev_lines) if prev_lines else ""
            user_msg   = (
                f"Criterion: {rn}\n{criterion}\n\n"
                + (f"Previous steps:\n{prev_text}\n\n" if prev_text else "")
                + f"My step:\n{step_text}"
            )
            pairs.append((state, rn,
                          build_chat_prompt(tokenizer, _GEN_DEEP_CRITIQUE_SYSTEM, user_msg)))

    if not pairs:
        return
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer([p[2] for p in pairs], return_tensors="pt", padding=True).to(input_device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    for j, (state, rn, _) in enumerate(pairs):
        raw = tokenizer.decode(out[j, input_len:], skip_special_tokens=True).strip()
        vm  = re.search(r"verdict\s*:\s*(correct|incorrect)", raw, re.I)
        verdict = ("pass" if vm.group(1).lower() == "correct" else "fail") if vm else None
        state.pending_step["gen_deep_critique"][rn] = {"verdict": verdict, "critique": raw}


def _batch_generate_gen_critique_summary(
    model, tokenizer, input_device, states: list[ProblemState]
) -> None:
    """로컬 generator로 gen_critique_summary 생성 (gen_deep fail 항목 한 단락 요약)."""
    pairs = []
    for state in states:
        step    = state.pending_step
        deep    = step.get("gen_deep_critique") or {}
        entries = [(rn, v) for rn, v in deep.items() if v.get("critique")]
        if not entries:
            step["gen_critique_summary"] = None
            continue
        combined  = "\n\n".join(f"[{rn}]\n{v['critique']}" for rn, v in entries)
        step_text = (step.get("text") or "")[:400]
        user_msg  = f"Step:\n{step_text}\n\nAnalyses:\n{combined}"
        pairs.append((state, build_chat_prompt(tokenizer, _CRITIQUE_PARA_SUMMARY_SYSTEM, user_msg)))

    if not pairs:
        return
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer([p[1] for p in pairs], return_tensors="pt", padding=True).to(input_device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    for j, (state, _) in enumerate(pairs):
        raw = tokenizer.decode(out[j, input_len:], skip_special_tokens=True).strip()
        state.pending_step["gen_critique_summary"] = raw or None


def _api_generate_gen_critique_summary(states: list[ProblemState]) -> None:
    """API 모드용 gen_critique_summary 생성 (gen_deep fail 항목 한 단락 요약)."""
    if not _SUMMARY_API:
        for state in states:
            state.pending_step.setdefault("gen_critique_summary", None)
        return

    def _one(state):
        step    = state.pending_step
        deep    = step.get("gen_deep_critique") or {}
        entries = [(rn, v) for rn, v in deep.items() if v.get("critique")]
        if not entries:
            step["gen_critique_summary"] = None
            return
        combined  = "\n\n".join(f"[{rn}]\n{v['critique']}" for rn, v in entries)
        step_text = (step.get("text") or "")[:400]
        user_msg  = f"Step:\n{step_text}\n\nAnalyses:\n{combined}"
        try:
            msgs = [
                {"role": "system", "content": _CRITIQUE_PARA_SUMMARY_SYSTEM},
                {"role": "user",   "content": user_msg},
            ]
            set_call_role("gen_critique_summary")
            raw = _call_llm(_SUMMARY_API, msgs, max_completion_tokens=256) or ""
            step["gen_critique_summary"] = raw.strip() or None
        except Exception:
            step["gen_critique_summary"] = None

    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=min(32, max(1, len(states)))) as ex:
        list(ex.map(_one, states))


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


def _parse_pred_fail_rubrics(gen_critique_summary: str | None) -> list[str]:
    """gen_critique_summary에서 모델이 직접 출력한 'Failed rubrics:' 줄을 파싱."""
    if not gen_critique_summary:
        return []
    m = re.search(r"Failed rubrics\s*:\s*(.+)", gen_critique_summary, re.I)
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
        deep = s.get("critique_summary") or []

    fail_rubrics = [
        e["rubric"] for e in (deep or [])
        if e.get("verdict") == "fail"
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
        step_src = s.get("source", "gen")
        if step_src == "patcher":
            state = "pat_solve"
        elif s["is_first_pat"]:
            state = "gen_rethink"
        else:
            state = "gen_solve"

        fail_rubrics, next_action = _compute_step_action(s)

        summ = s.get("summary") or {}
        if isinstance(summ, dict):
            does                 = summ.get("does") or summ.get("step_analysis") or None
            prm_fast_critique    = summ.get("prm_fast_critique") or None
            prm_deep_critique    = summ.get("prm_deep_critique") or None
            prm_critique_summary = summ.get("prm_critique_summary") or None
            gen_fast_critique    = summ.get("gen_fast_critique") or None
            gen_deep_critique    = summ.get("gen_deep_critique") or None
            gen_critique_summary = summ.get("gen_critique_summary") or None
        else:
            does = prm_fast_critique = prm_deep_critique = prm_critique_summary = None
            gen_fast_critique = gen_deep_critique = gen_critique_summary = None

        pred_fail_rubrics = _parse_pred_fail_rubrics(gen_critique_summary)

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
            "prm_critique_summary": prm_critique_summary,
            "gen_fast_critique":    gen_fast_critique,
            "gen_deep_critique":    gen_deep_critique,
            "gen_critique_summary": gen_critique_summary,
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
    save_intermediate_fn,
    prm_save_fn,
    rubric_text_save_fn=None,
    history_save_fn=None,
    prm_filter_save_fn=None,
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
    wrong_rubrics = [r for r, v in votes.items() if v == "fail"]
    rubric_text   = [
        {
            "rubric":       rn,
            "reasoning":    details[rn].get("reasoning"),
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
                "reasoning":    rt["reasoning"],
                "verdict_text": rt["verdict_text"],
            })

    # _batch_run_all_summaries에서 로컬 generator가 이미 계산한 critique 사용
    prm_deep_critique    = step.get("critique_summary")
    prm_fast_critique    = result.get("fast_rubric")
    prm_critique_summary = step.get("prm_critique_summary")
    gen_fast_critique    = step.get("gen_fast_critique")
    gen_deep_critique    = step.get("gen_deep_critique")
    gen_critique_summary = step.get("gen_critique_summary")

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
                "reasoning":         d.get("reasoning"),
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
        step["summary"]      = {
            "does":                 does_summary,
            "prm_fast_critique":    prm_fast_critique,
            "step_analysis":        (
                f"API_PRM_checklist: {wrong_count}/{total} rubrics flagged wrong"
                f" ({', '.join(wrong_rubrics)})"
                + (f" [{reason_suffix}]" if reason_suffix else "")
            ),
            "prm_deep_critique":    prm_deep_critique,
            "prm_critique_summary": prm_critique_summary,
            "gen_fast_critique":    gen_fast_critique,
            "gen_deep_critique":    gen_deep_critique,
            "gen_critique_summary": gen_critique_summary,
            "votes":                votes,
            "details":              details,
        }
        state.all_steps.append(step)

        if save_intermediate_fn:
            save_intermediate_fn(
                _build_traj(problem_id, problem, gold_answer,
                            state.all_steps, False, "mix_intermediate",
                            traj_idx=state.traj_idx)
            )
            state.traj_idx += 1

        if TRAJ_MAX_STEPS and len(state.all_steps) >= TRAJ_MAX_STEPS:
            print(f"  [id={problem_id}]  {_fmt(state.all_steps)}  → max_steps")
            traj = _build_traj(problem_id, problem, gold_answer,
                               state.all_steps, False, "mix",
                               fail_reason="max_steps", traj_idx=state.traj_idx)
            if save_intermediate_fn:
                save_intermediate_fn(traj)
            state.traj_idx += 1
            state.done = True
            return

        _critique_by_rubric = {
            entry["rubric"]: entry.get("critique")
            for entry in (step.get("critique_summary") or [])
            if entry.get("critique")
        }
        _rubric_details = {
            rn: _critique_by_rubric.get(rn) or ""
            for rn in wrong_rubrics
        }
        _wrong_summary = step.pop("_wrong_step_summary", "")

        if not state.step_rethink_tried:
            # 1차 실패 → rethink
            state.is_rethink                = True
            state.step_rethink_tried        = True
            state.last_wrong_step_text      = step["text"]
            state.last_wrong_step_summary   = _wrong_summary
            state.last_wrong_rubric_details = _rubric_details
        elif not state.step_patcher_tried:
            # rethink도 실패 → patcher 1회
            state.use_patcher               = True
            state.is_rethink                = False
            state.step_patcher_tried        = True
            state.patcher_count            += 1
            state.last_wrong_rubrics        = wrong_rubrics
            state.last_wrong_step_text      = step["text"]
            state.last_wrong_step_summary   = _wrong_summary
            state.last_wrong_rubric_details = _rubric_details  # rethink 실패 기준으로 갱신
        else:
            # patcher도 실패 → 종료
            print(f"  [id={problem_id}]  {_fmt(state.all_steps)}  → patcher_fail")
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

    # ── patcher step 특별 처리 ────────────────────────────────────────────────
    if step.get("source") == "patcher":
        if step.get("is_error"):
            # 빈 응답 → patcher_fail
            logger.info(f"[id={problem_id}] patcher empty response → patcher_fail")
            print(f"  [id={problem_id}]  {_fmt(state.all_steps)}  → patcher_fail")
            traj = _build_traj(problem_id, problem, gold_answer,
                               state.all_steps, False, "mix",
                               fail_reason="patcher_fail", traj_idx=state.traj_idx)
            state.traj_mix_list.append(traj)
            if save_fn:
                save_fn(traj, "mix")
            state.traj_idx += 1
            state.done = True
            return
        if has_boxed(step["text"]):
            if check_solved(step["text"], gold_answer):
                # patcher가 정답 → PRM 판단 무관하게 correct로 처리
                logger.info(f"[id={problem_id}] patcher solved correctly → treat as correct")
                result = {**result, "result": "pass", "wrong_count": 0}
            else:
                # patcher가 틀린 답 제출 → patcher_fail
                logger.info(f"[id={problem_id}] patcher wrong final answer → patcher_fail")
                state.all_steps.append(step)
                print(f"  [id={problem_id}]  {_fmt(state.all_steps)}  → patcher_fail")
                traj = _build_traj(problem_id, problem, gold_answer,
                                   state.all_steps, False, "mix",
                                   fail_reason="patcher_fail", traj_idx=state.traj_idx)
                state.traj_mix_list.append(traj)
                if save_fn:
                    save_fn(traj, "mix")
                state.traj_idx += 1
                state.done = True
                return
        elif result["result"] == "fail":
            # patcher 중간 step (boxed 없음): PRM wrong 무시하고 correct로 처리
            logger.info(f"[id={problem_id}] patcher mid-step PRM wrong → override to correct")
            result = {**result, "result": "pass", "wrong_count": 0}

    # ── 오류 있음 ─────────────────────────────────────────────────────────────
    if result["result"] == "fail":
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
                _apply_wrong("pred≠gold_answer")
                return

        # 정상 정답 스텝: history에 추가
        step["summary"] = {
            "does":                 does_summary,
            "prm_fast_critique":    prm_fast_critique,
            "prm_deep_critique":    prm_deep_critique,
            "prm_critique_summary": prm_critique_summary,
            "gen_fast_critique":    gen_fast_critique,
            "gen_deep_critique":    gen_deep_critique,
            "gen_critique_summary": gen_critique_summary,
        }
        state.all_steps.append(step)
        state.history.append(step)

        if history_save_fn:
            history_save_fn({
                "problem_id":  problem_id,
                "step_idx":    len(state.history),
                "step":        _fmt([step]),
                "source":      step.get("source"),
                "does":                 does_summary,
                "prm_fast_critique":    prm_fast_critique,
                "prm_deep_critique":    prm_deep_critique,
            })
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
        elif TRAJ_MAX_STEPS and len(state.all_steps) >= TRAJ_MAX_STEPS:
            print(f"  [id={problem_id}]  {_fmt(state.all_steps)}  → max_steps")
            traj = _build_traj(problem_id, problem, gold_answer,
                               state.all_steps, False, "mix",
                               fail_reason="max_steps", traj_idx=state.traj_idx)
            if save_intermediate_fn:
                save_intermediate_fn(traj)
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
    prm_model: ApiPrm,
    rubrics: list[dict],
    n_parallel: int,
    save_fn=None,
    save_intermediate_fn=None,
    prm_save_fn=None,
    rubric_text_save_fn=None,
    history_save_fn=None,
    prm_filter_save_fn=None,
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
    prm_stats = {"total_steps": 0, "fast_rubric_calls": 0, "rubric_calls": 0}

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
        def _run_gen():
            if gen_states:
                if _GENERATOR_API:
                    _run_generator_api(gen_states)
                else:
                    _parallel_gen(_batch_run_generator, generators, gen_states)

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_pat = ex.submit(_run_patcher_api, patcher_states) if patcher_states else None
            fut_gen = ex.submit(_run_gen)
            if fut_pat: fut_pat.result()
            fut_gen.result()

        # ── PRM 배치 평가 + step_summary GPU 병렬 실행 ──────────────────────────
        # PRM은 API 호출(GPU 유휴), step_summary는 GPU 사용 → 동시 실행 가능
        prm_states, skip_states = [], []
        for state in active:
            if state.pending_step.get("is_error"):
                skip_states.append(state)
            else:
                prm_states.append(state)

        prm_results_map:    dict[int, dict]       = {}
        state_step_rubrics: dict[int, list[dict]] = {}

        if prm_states:
            for state in prm_states:
                state_step_rubrics[id(state)] = rubrics
            logger.info(
                f"[PRM batch] n_problems={len(prm_states)}  "
                f"n_rubrics={len(rubrics)}  skipped(tool_call)={len(skip_states)}"
            )
            prm_stats["total_steps"] += len(prm_states)
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_prm = ex.submit(
                    _run_prm_batch, prm_model, prm_states, [rubrics] * len(prm_states),
                    prm_stats=prm_stats,
                )
                fut_sum = ex.submit(
                    _parallel_gen, _batch_run_step_summary_only, generators, active,
                )
                prm_results_map.update(fut_prm.result())
                fut_sum.result()
        else:
            _parallel_gen(_batch_run_step_summary_only, generators, active)

        for state in skip_states:
            state_step_rubrics[id(state)] = rubrics
            n_rubs = len(rubrics)
            pid = state.item.get("id", "?")
            logger.info(f"[Generator] id={pid} tool_call hallucination → force wrong")
            prm_results_map[id(state)] = {
                "result":      "fail",
                "wrong_count": n_rubs,
                "total":       n_rubs,
                "threshold":   1,
                "votes":       {r["name"]: "fail" for r in rubrics},
                "details":     {},
            }

        # ── critique + wrong step summary (step summary는 PRM과 병렬로 완료됨) ──
        logger.info(f"[AllSummaries] n={len(active)}")
        _parallel_gen(
            lambda m, t, d, sts: _batch_run_all_summaries(m, t, d, sts, prm_results_map),
            generators, active,
        )

        # ── gen_fast/deep_critique: inference Self-correction 파싱 (전체 루브릭) ─
        rubric_names = [r["name"] for r in rubrics]
        _extract_gen_fast_critique(active, rubric_names)

        # ── pass 2a: prm_critique_summary (GPU 병렬) ────────────────────────
        _parallel_gen(
            lambda m, t, d, sts: _batch_generate_prm_critique_summary(m, t, d, sts),
            generators, active,
        )

        # ── pass 3: gen_critique_summary (GPU 병렬 or API) ───────────────────
        if generators:
            _parallel_gen(
                lambda m, t, d, sts: _batch_generate_gen_critique_summary(m, t, d, sts),
                generators, active,
            )
        else:
            _api_generate_gen_critique_summary(active)

        # ── 결과 처리 ─────────────────────────────────────────────────────────
        for state in active:
            result    = prm_results_map[id(state)]
            step_rubs = state_step_rubrics.get(id(state), rubrics)
            _process_prm_result(state, result, step_rubs, save_fn, save_intermediate_fn, prm_save_fn, rubric_text_save_fn, history_save_fn, prm_filter_save_fn)

        _flush_done()

    pbar.close()
    _w = 60
    print(f"\n{'─' * _w}")
    print(
        f"[PRM 통계]  전체 스텝: {prm_stats['total_steps']}  "
        f"fast_rubric 호출: {prm_stats['fast_rubric_calls']}  "
        f"rubric 호출: {prm_stats['rubric_calls']}"
    )
    print(f"{'─' * _w}")
    logger.info(
        f"PRM 통계: total_steps={prm_stats['total_steps']}  "
        f"fast_rubric_calls={prm_stats['fast_rubric_calls']}  "
        f"rubric_calls={prm_stats['rubric_calls']}"
    )
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
    parser.add_argument("--num_start",   type=int, default=None)
    parser.add_argument("--output",      type=str, default=None,
                        help="출력 폴더 경로 (기본: output/sft_trajectory/{timestamp})")
    parser.add_argument("--rubric_file", type=str, default=None,
                        help="루브릭 jsonl 경로 (루브릭별 1회 호출)")
    parser.add_argument("--fast_rubric_file", type=str, default=None,
                        help="fast 루브릭 JSON 경로 (샘플당 1회 호출). 예: prompts/prm_rubric_v6.0_batch.json")
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
    prm_out_dir = root / "output" / "PRM_log" / ts
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
    prm_eval_file    = open(prm_out_dir / "prm_evals.jsonl",  "w", encoding="utf-8")
    prm_filter_file  = open(prm_out_dir / "prm_filter.jsonl", "w", encoding="utf-8")
    history_file     = open(out_dir / "history.jsonl", "w", encoding="utf-8")

    _run_jsonl_file  = open(out_dir / "run.jsonl", "w", encoding="utf-8")
    _run_jsonl_lock  = __import__("threading").Lock()
    def _run_log(record: dict):
        line = json.dumps(record, ensure_ascii=False)
        with _run_jsonl_lock:
            _run_jsonl_file.write(line + "\n")
            _run_jsonl_file.flush()
    set_run_log(_run_log)
    rubric_text_dir = prm_out_dir / "rubric_texts"
    rubric_text_dir.mkdir(parents=True, exist_ok=True)
    rubric_text_files: dict = {}

    num_data  = args.num_data  or gt_cfg.get("num_data",  1)
    num_start = args.num_start if args.num_start is not None else gt_cfg.get("num_start", 0)

    # ── GPU 설정 ─────────────────────────────────────────────────────────────
    rollout_gpus = gt_cfg.get("rollout_gpus", [0])
    prm_gpu_ids    = CONF.get("PRM", {}).get("gpu_id", [1])
    n_prm_gpus     = len(prm_gpu_ids)
    PRM_BATCH_SIZE = PRM_BATCH_PER_GPU * n_prm_gpus
    logger.info(f"prm_gpu_ids={prm_gpu_ids}  rollout_gpus={rollout_gpus}")

    # ── 데이터 & 루브릭 로드 ─────────────────────────────────────────────────
    items = load_dataset_file(dataset_path)
    items = items[num_start:] if num_data == -1 else items[num_start: num_start + num_data]
    logger.info(f"로드된 문제 수: {len(items)}")

    _prm_conf  = CONF.get("PRM", {})
    rubric_path = args.rubric_file or _prm_conf.get("rubric")
    if not rubric_path:
        raise ValueError("루브릭 파일 경로를 지정해 주세요: --rubric_file 혹은 config.PRM.rubric 설정")
    if not Path(rubric_path).is_absolute():
        rubric_path = str(root / rubric_path)

    rubrics      = load_rubrics(rubric_path)
    fast_rubric  = None
    prm_mode     = "per_rubric"

    n_rubrics  = len(rubrics)
    n_parallel = args.n_parallel or gt_cfg.get("batch_per_gpu", PRM_BATCH_PER_GPU) * len(rollout_gpus)

    logger.info(
        f"데이터셋={dataset_path}  sft출력={out_dir}  prm출력={prm_out_dir}  "
        f"num_start={num_start}  num_data={num_data}  "
        f"PRM_BATCH_SIZE={PRM_BATCH_SIZE}  n_rubrics={n_rubrics}  n_parallel={n_parallel}"
    )

    # ── Generator 로드 ────────────────────────────────────────────────────────
    base_model_id = CONF["checkpoint"]["base"]
    generators = []
    if _GENERATOR_API:
        logger.info(f"Generator API 모드: {_GENERATOR_API}  (로컬 모델 로드 생략)")
    else:
        for gpu_id in rollout_gpus:
            device_map = {"": f"cuda:{gpu_id}"}
            logger.info(f"Generator 로딩 중: {base_model_id}  device_map={device_map}")
            model, tokenizer = load_generator(model_path=base_model_id, device_map=device_map)
            generators.append((model, tokenizer, next(model.parameters()).device))
            logger.info(f"Generator 로드 완료 (device={generators[-1][2]})")
        logger.info(f"Generator {len(generators)}개 로드 완료")

    # ── PRM 로드 ──────────────────────────────────────────────────────────────
    if not _PRM_API:
        raise ValueError("config.yaml의 PRM.model_id에 모델 이름을 설정해 주세요.")
    fast_rubric_path = args.fast_rubric_file or _prm_conf.get("fast_rubric")
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

    # ── 저장 함수 ─────────────────────────────────────────────────────────────
    counts = {"gen": 0, "mix": 0}

    history_buffer: dict = {}  # problem_id → list of step records

    def _save(traj: dict, traj_type: str):
        line = json.dumps(traj, ensure_ascii=False) + "\n"
        files[traj_type].write(line); files[traj_type].flush()
        files["all"].write(line);     files["all"].flush()
        counts[traj_type] += 1
        pid = traj.get("problem_id")
        if pid and pid in history_buffer:
            record = {"problem_id": pid, "history": history_buffer.pop(pid)}
            history_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            history_file.flush()

    def _save_intermediate(traj: dict):
        line = json.dumps(traj, ensure_ascii=False) + "\n"
        files["all"].write(line); files["all"].flush()

    def _save_prm_record(rec: dict):
        prm_eval_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        prm_eval_file.flush()

    def _save_prm_filter(rec: dict):
        prm_filter_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        prm_filter_file.flush()

    def _save_rubric_text(rec: dict):
        rname = rec["rubric"]
        safe  = rname.replace(" ", "_").replace("/", "-")
        if safe not in rubric_text_files:
            rubric_text_files[safe] = open(rubric_text_dir / f"{safe}.jsonl", "w", encoding="utf-8")
        f = rubric_text_files[safe]
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()

    def _save_history(rec: dict):
        pid = rec["problem_id"]
        if pid not in history_buffer:
            history_buffer[pid] = []
        history_buffer[pid].append({k: v for k, v in rec.items() if k != "problem_id"})

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
            rubric_text_save_fn=_save_rubric_text,
            history_save_fn=_save_history,
            prm_filter_save_fn=_save_prm_filter,
        )
    finally:
        for f in files.values():
            f.close()
        prm_eval_file.close()
        prm_filter_file.close()
        history_file.close()
        _run_jsonl_file.close()
        set_run_log(None)
        for f in rubric_text_files.values():
            f.close()
        _record_usage(_PRM_API, [{"input_tokens": prm_model.total_input, "output_tokens": prm_model.total_output}])
        _print_cost_summary()
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
