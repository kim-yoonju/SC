"""
generate_sft_trajectory.py
base_problems JSONL에서 generator → patcher 반복으로 trajectory SFT 데이터 생성.

기존의 PATCHER를 한 번 호출해서 모든걸 생성해달라고해서
SFT 데이터를 효율적으로 생성하고자함

흐름:
  1. Generator (SFT checkpoint)가 스텝별로 풀이
  2. 틀리면 Patcher (API)가 "Step K 이후 오류 찾아 수정해줘"로 호출
  3. patcher 첫 스텝 1개만 history에 추가 → Generator가 이어서 풀이
  4. 정답 맞출 때까지 반복 (MAX_ROUNDS)

출력 파일 (output/sft_trajectory/{timestamp}/):
  traj_gen.jsonl   generator 단독 정답 (patcher 없음)
  traj_pat.jsonl   round 1 patcher 전체 교정 (gen_correct + patcher_all, error step 없음)
  traj_mix.jsonl   gen-patcher 혼합, generator가 최종 정답
  traj_all.jsonl   위 세 가지 전부

스텝 state / next_gold_action:
  일반 gen/patcher   : state=solve     / →<|solve|>
  오류 gen 스텝      : state=solve     / →<|rethink|>
  오류 직후 첫 pat   : state=rethink_pat / →<|solve|>
  마지막 스텝        : (위와 동일)    / →<|end|>
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import torch
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    CONF, PATCHER, PATCHER_MAX_NEW_TOKENS,
    SFT_GENERATOR_PROMPT,
    SFT_PATCHER_ALL_RPRM_PROMPT,
    GENERATOR_TEMPERATURE,
    TOKEN_SOLVE, TOKEN_END,
    _gpt, extract_boxed, check_solved,
    load_generator, build_chat_prompt,
)


from generate_utils import (
    calc_cost, load_dataset_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_GT_CFG             = CONF.get("generate_trajectory", {})
TRAJ_MAX_NEW_TOKENS = _GT_CFG.get("max_new_tokens", 4096)
MAX_STEPS           = _GT_CFG.get("max_steps", 30)
MAX_API             = _GT_CFG.get("max_api", 10)

TOKEN_RETHINK = "<|rethink|>"

W    = 88
SEP2 = "━" * W


# ─────────────────────────────────────────────────────────────────────────────
# 파싱 유틸
# ─────────────────────────────────────────────────────────────────────────────

_STEP_HEADER_RE = re.compile(r"Step\s+(\d+)\s*:", re.IGNORECASE)
_ERROR_STEP_RE  = re.compile(r"First error at step:\s*(\d+)", re.IGNORECASE)

# R-PRM 파싱용
# sft_patcher_all_R-PRM: 각 스텝 뒤 "Verification: Is the step correct (Yes/No)? Yes — ..."
_VERIFY_LINE_RE = re.compile(
    r"\nVerification:\s*Is the step correct[^?]*\?\s*(Yes|No)(?:\s*[—-]\s*([^\n]*))?",
    re.IGNORECASE,
)
# sft_patcher_step_R-PRM: "Analysis of Step N: ..." 블록 (수정 스텝 직전까지)
_ANALYSIS_BLOCK_RE = re.compile(
    r"Analysis of Step \d+:(.*?)(?=\nStep\s+\d+:\s+\S|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# sft_patcher_step_R-PRM: per-step eval block header "Step N:\n- Does: ..."
_EVAL_STEP_HEADER_RE = re.compile(r"Step\s+(\d+):\s*\n", re.IGNORECASE)


def _parse_rprm_analysis(response: str) -> str | None:
    """R-PRM step prompt의 Analysis 블록을 파싱해 summary 문자열로 반환."""
    m = _ANALYSIS_BLOCK_RE.search(response)
    if not m:
        return None
    return m.group(1).strip()


def _parse_per_step_eval_blocks(text: str) -> dict[int, str]:
    """pre-analysis 영역에서 'Step N:\\n- Does:...\\nVerification:...' 블록 파싱.
    Returns {global_step_num: summary_str}
    """
    headers = list(_EVAL_STEP_HEADER_RE.finditer(text))
    result: dict[int, str] = {}
    for i, h in enumerate(headers):
        step_num = int(h.group(1))
        start    = h.end()
        end      = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block    = text[start:end].strip()
        if block:
            result[step_num] = block
    return result


def _parse_steps(response: str, source: str) -> list[dict]:
    """'Step N:' 헤더 기준으로 스텝 분리. source = 'gen' | 'patcher'."""
    headers = list(_STEP_HEADER_RE.finditer(response))
    steps: list[dict] = []

    def _step(text: str, summary: str | None = None) -> dict:
        return {"text": text, "source": source, "is_error": False, "is_first_pat": False, "summary": summary}

    if not headers:
        if response.strip():
            steps.append(_step(response.strip()))
        return steps

    pre = response[:headers[0].start()].strip()
    if pre:
        steps.append(_step(pre))

    for i, h in enumerate(headers):
        start = h.end()
        end   = headers[i + 1].start() if i + 1 < len(headers) else len(response)
        text  = response[start:end].strip()
        if text:
            steps.append(_step(text))

    return steps


def _parse_steps_rprm(response: str, source: str) -> list[dict]:
    """sft_patcher_all_R-PRM 형식 파싱: 각 스텝 뒤 Verification 라인을 summary로 추출."""
    headers = list(_STEP_HEADER_RE.finditer(response))
    steps: list[dict] = []

    for i, h in enumerate(headers):
        start = h.end()
        end   = headers[i + 1].start() if i + 1 < len(headers) else len(response)
        block = response[start:end].strip()

        vm = _VERIFY_LINE_RE.search(block)
        if vm:
            text    = block[:vm.start()].strip()
            verdict = vm.group(1).strip()
            note    = (vm.group(2) or "").strip()
            summary = f"Is the step correct? {verdict}" + (f" — {note}" if note else "")
        else:
            text    = block
            summary = None

        if text:
            steps.append({
                "text": text, "source": source,
                "is_error": False, "is_first_pat": False,
                "summary": summary,
            })

    return steps


def _parse_patcher_response(
    response: str,
    n_all_steps: int,
    check_from: int,
) -> tuple[int, list[dict]]:
    """
    Patcher 응답 파싱.
    Returns (error_step_idx 1-based in all_steps, patcher_steps).
    error_step_idx는 check_from+1 이상으로 클램프.
    """
    m = _ERROR_STEP_RE.search(response)
    error_step_idx = int(m.group(1)) if m else max(check_from + 1, n_all_steps)
    error_step_idx = max(error_step_idx, check_from + 1)  # 클램프

    headers = list(_STEP_HEADER_RE.finditer(response))
    patcher_steps: list[dict] = []
    for i, h in enumerate(headers):
        start = h.end()
        end   = headers[i + 1].start() if i + 1 < len(headers) else len(response)
        text  = response[start:end].strip()
        if text:
            patcher_steps.append({
                "text": text, "source": "patcher",
                "is_error": False, "is_first_pat": False,
            })

    if patcher_steps:
        patcher_steps[0]["is_first_pat"] = True

    return error_step_idx, patcher_steps


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

def _run_generator(model, tokenizer, input_device, problem: str, history: list[dict]) -> list[dict]:
    """history: step dict 목록. Returns 새 gen 스텝 목록."""
    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Steps solved so far]")
        for i, s in enumerate(history, 1):
            lines.append(f"Step {i}: {s['text']}")
        lines.append(f"\nContinue solving from Step {len(history) + 1}.")
    user_msg = "\n".join(lines)

    logger.debug(f"[Generator] history={len(history)} steps, input_len_chars={len(user_msg)}")

    prompt    = build_chat_prompt(tokenizer, SFT_GENERATOR_PROMPT, user_msg)
    enc       = tokenizer(prompt, return_tensors="pt").to(input_device)
    input_len = enc["input_ids"].shape[1]

    logger.debug(f"[Generator] prompt_tokens={input_len}")

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=TRAJ_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_text  = tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()
    steps     = _parse_steps(gen_text, "gen")
    logger.info(f"[Generator] 생성 완료: {len(steps)} steps, output_tokens={out.shape[1] - input_len}")
    for i, s in enumerate(steps):
        preview = s["text"][:120].replace("\n", " ")
        logger.info(f"  gen_step[{i+1}]: {preview}")
    return steps


# ─────────────────────────────────────────────────────────────────────────────
# Patcher
# ─────────────────────────────────────────────────────────────────────────────

def _generate_patcher_traj(
    problem: str,
    gold_answer: str,
) -> tuple[list[dict], float]:
    """
    traj_pat 전용: patcher가 문제를 처음부터 끝까지 독립적으로 풀어 trajectory 생성.
    sft_patcher_all_R-PRM 사용 → 각 스텝에 Verification summary 포함.
    Returns (patcher_steps, cost_usd)
    """
    user_msg = f"[Problem]\n{problem}\n\nExpected answer: {gold_answer}"
    messages = [
        {"role": "system", "content": SFT_PATCHER_ALL_RPRM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    usage_out: list[dict] = []
    logger.info(f"[PatcherAll] API 호출 시작 (model={PATCHER})")
    try:
        response = _gpt(
            PATCHER, messages,
            max_completion_tokens=PATCHER_MAX_NEW_TOKENS,
            usage_out=usage_out,
        )
    except Exception as e:
        logger.error(f"[PatcherAll] API 호출 실패: {e}", exc_info=True)
        return [], 0.0

    logger.info(f"[PatcherAll] 응답 수신: {len(response) if response else 0}자")
    if not response:
        logger.warning("[PatcherAll] 빈 응답")
        return [], 0.0

    patcher_steps = _parse_steps_rprm(response, source="patcher")
    if patcher_steps:
        patcher_steps[0]["is_first_pat"] = True

    u    = usage_out[0] if usage_out else {}
    cost = calc_cost(PATCHER, u.get("input_tokens", 0), u.get("output_tokens", 0))
    is_right = check_solved(patcher_steps[-1]["text"], gold_answer) if patcher_steps else False
    logger.info(f"[PatcherAll] steps={len(patcher_steps)}  {'correct' if is_right else 'wrong'}  in={u.get('input_tokens',0)}  out={u.get('output_tokens',0)}  cost=${cost:.5f}")
    return patcher_steps, cost


def _summary_to_str(summary) -> str:
    """summary dict → does 필드만 반환. 문자열이면 그대로 반환."""
    if not summary:
        return ""
    if isinstance(summary, dict):
        does = summary.get("does") or summary.get("step_analysis") or ""
        return does.strip()
    return str(summary).strip()


def _build_history_context(history: list[dict]) -> str:
    """history 스텝들의 summary를 [Steps to Date] 컨텍스트로 변환."""
    if not history:
        return ""
    lines = ["[Steps to Date]"]
    for i, s in enumerate(history, 1):
        summary_str = _summary_to_str(s.get("summary"))
        if not summary_str:
            summary_str = s.get("text", "")[:150].replace("\n", " ")
        lines.append(f"Step {i}: {summary_str}")
    return "\n".join(lines)


def _run_patcher_step(
    problem: str,
    steps_to_check: list[dict],
    gold_answer: str,
    step_offset: int = 0,
    history: list[dict] | None = None,
) -> tuple[int | None, dict | None, float, str | None, dict[int, str]]:
    """
    딱 한 스텝만 수정하는 patcher (sft_patcher_step_R-PRM, JSON 출력).
    Returns:
      error_step_idx  : global 1-based 에러 스텝 번호, 없으면 None
      rethink_step    : 수정된 스텝 dict | None
      cost_usd        : API 비용
      error_summary   : Analysis 텍스트 → 에러 스텝 summary | None
      per_step_summaries : {global_step_num_1based: summary_str}
    """
    lines = [f"[Problem]\n{problem}\n"]

    history_ctx = _build_history_context(history or [])
    if history_ctx:
        lines.append(history_ctx + "\n")

    lines.append("[Review Steps]")
    for i, s in enumerate(steps_to_check, step_offset + 1):
        lines.append(f"Step {i}: {s['text']}")


    messages = [
        {"role": "system", "content": SFT_PATCHER_STEP_RPRM_PROMPT},
        {"role": "user",   "content": "\n".join(lines)},
    ]

    usage_out: list[dict] = []
    logger.info(f"[PatcherStep] API 호출 시작 (model={PATCHER}, step_offset={step_offset}, steps_to_check={len(steps_to_check)})")
    try:
        response = _gpt(
            PATCHER, messages,
            max_completion_tokens=PATCHER_MAX_NEW_TOKENS,
            usage_out=usage_out,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.error(f"[PatcherStep] API 호출 실패: {e}", exc_info=True)
        return step_offset + 1, None, 0.0, None, {}

    logger.info(f"[PatcherStep] 응답 수신: {len(response) if response else 0}자")
    if not response:
        logger.warning("[PatcherStep] 빈 응답")
        return step_offset + 1, None, 0.0, None, {}

    # JSON 파싱
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(f"[PatcherStep] JSON 파싱 실패: {e}\n응답: {response[:500]}")
        return step_offset + 1, None, 0.0, None, {}

    u    = usage_out[0] if usage_out else {}
    cost = calc_cost(PATCHER, u.get("input_tokens", 0), u.get("output_tokens", 0))

    # error_step_idx
    raw_err        = data.get("first_error_step")
    error_step_idx = int(raw_err) if raw_err is not None else None
    n_total = step_offset + len(steps_to_check)
    if error_step_idx is not None:
        error_step_idx = max(error_step_idx, step_offset + 1)
        error_step_idx = min(error_step_idx, n_total)

    # per-step summaries (에러 이전 스텝들만 — 에러 스텝은 error_analysis에서 처리)
    per_step_summaries: dict[int, dict] = {}
    for ev in data.get("recap_steps", []):
        step_num = ev.get("step")
        raw_summ = ev.get("summary")
        if step_num is None:
            continue
        # 에러 스텝 이상은 무시 (모델이 실수로 포함했을 경우 방어)
        if error_step_idx is not None and int(step_num) >= error_step_idx:
            continue
        if isinstance(raw_summ, dict):
            summary = raw_summ
        elif isinstance(raw_summ, str) and raw_summ.strip():
            summary = {"does": raw_summ.strip()}
        else:
            continue
        per_step_summaries[int(step_num)] = summary

    # 에러 없으면 여기서 반환
    if error_step_idx is None:
        logger.info(f"[PatcherStep] 에러 없음  per_step_summaries={len(per_step_summaries)}  in={u.get('input_tokens',0)}  out={u.get('output_tokens',0)}  cost=${cost:.5f}")
        return None, None, cost, None, per_step_summaries

    # error_summary (error_analysis 구조체)
    raw_err_analysis = data.get("error_analysis")
    if isinstance(raw_err_analysis, dict):
        error_summary = raw_err_analysis if any(raw_err_analysis.values()) else None
    elif isinstance(raw_err_analysis, str) and raw_err_analysis.strip():
        error_summary = {"step_analysis": raw_err_analysis.strip()}
    else:
        error_summary = None

    # rethink step
    corrected = data.get("rethink_step") or {}
    text      = (corrected.get("text") or "").strip()
    step_dict = None
    if text:
        raw_summ = corrected.get("summary")
        if isinstance(raw_summ, dict):
            pat_summary = raw_summ if any(raw_summ.values()) else None
        elif isinstance(raw_summ, str) and raw_summ.strip():
            pat_summary = {"does": raw_summ.strip()}
        else:
            pat_summary = None
        step_dict = {
            "text":         text,
            "source":       "patcher",
            "is_error":     False,
            "is_first_pat": True,
            "summary":      pat_summary,
        }

    logger.info(
        f"[PatcherStep] error_step={error_step_idx}  step_dict={'OK' if step_dict else 'FAIL'}"
        f"  error_summary={'OK' if error_summary else 'NONE'}"
        f"  per_step_summaries={len(per_step_summaries)}"
        f"  pat_summary={'OK' if (step_dict and step_dict.get('summary')) else 'NONE'}"
        f"  in={u.get('input_tokens',0)}  out={u.get('output_tokens',0)}  cost=${cost:.5f}"
    )
    if step_dict:
        logger.info(f"[PatcherStep] 수정 스텝 내용: {step_dict['text'][:200].replace(chr(10), ' ')}")
    else:
        logger.warning(f"[PatcherStep] 파싱 실패. 응답 전문:\n{response}")

    return error_step_idx, step_dict, cost, error_summary, per_step_summaries



# ─────────────────────────────────────────────────────────────────────────────
# Trajectory 조립
# ─────────────────────────────────────────────────────────────────────────────

def _compute_labels(steps: list[dict], first_pat_pos: int = 0) -> list[str]:
    """
    스텝 레이블 계산.
      - gen  : G_{pos:02d}  (순차 증가)
      - patcher 첫 스텝 : first_pat_pos 위치에서 시작 (traj_pat 보정용)
      - patcher 이후    : 순차 증가
      - traj_mix (error step 포함) : first_pat_pos=0 → error step이 pos를 이미 선점하므로 자동

    first_pat_pos: traj_pat 전용. error step이 없으므로 patcher가 시작할 1-based 위치를 명시.
    """
    labels         = []
    pos            = 0
    in_patcher_run = False

    for s in steps:
        if s["source"] == "patcher":
            if not in_patcher_run:
                if first_pat_pos > 0 and pos < first_pat_pos:
                    pos = first_pat_pos  # traj_pat: error step 위치로 점프
                in_patcher_run = True
            else:
                pos += 1
            labels.append(f"P_{pos:02d}")
        else:
            pos += 1
            in_patcher_run = False
            labels.append(f"G_{pos:02d}")

    return labels


def _build_traj(
    problem_id, problem, gold_answer,
    steps: list[dict],
    is_right: bool,
    traj_type: str,
    first_pat_pos: int = 0,
    traj_idx: int = 0,
) -> dict:
    """
    내부 step 포맷 → 저장 포맷 변환.
    steps 각 항목: {text, source, is_error, is_first_pat}
    """
    pred_answer = None
    for s in reversed(steps):
        raw = extract_boxed(s["text"])
        if raw:
            pred_answer = raw
            break

    labels = _compute_labels(steps, first_pat_pos)
    last   = len(steps) - 1
    step_dicts = []

    for i, (s, label) in enumerate(zip(steps, labels)):
        is_last = (i == last)
        if s["is_error"]:
            state, next_action = "solve", TOKEN_RETHINK
        elif s["is_first_pat"]:
            state, next_action = "rethink_pat", TOKEN_END if is_last else TOKEN_SOLVE
        else:
            state, next_action = "solve", TOKEN_END if is_last else TOKEN_SOLVE

        summ = s.get("summary") or {}
        if isinstance(summ, dict):
            # error_analysis 키 → does/consistency/calculation 로 매핑
            does        = summ.get("does") or summ.get("step_analysis") or None
            consistency = summ.get("consistency") or summ.get("consistency_check") or None
            calculation = summ.get("calculation") or summ.get("calculation_check") or None
        else:
            does = consistency = calculation = None

        step_dicts.append({
            "step_idx":         i,
            "step":             label,
            "inference":        s["text"],
            "source":           s["source"],
            "is_error":         s["is_error"],
            "state":            state,
            "next_gold_action": next_action,
            "does":             does,
            "consistency":      consistency,
            "calculation":      calculation,
        })

    return {
        "traj_id":     f"{problem_id}_{traj_idx:02d}",
        "problem_id":  str(problem_id),
        "problem":     problem,
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "is_right":    is_right,
        "traj_type":   traj_type,
        "steps":       step_dicts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 터미널 출력
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(steps: list[dict]) -> str:
    parts, n = [], 0
    for s in steps:
        n += 1
        if s.get("is_error"):
            parts.append(f"[E_{n:02d}]")
        elif s["source"] == "patcher":
            parts.append(f"P_{n:02d}")
        else:
            parts.append(f"G_{n:02d}")
    return "  ".join(parts)


def _fmt_compact(steps: list[dict]) -> str:
    """저장된 traj steps를 압축 표시. G×3  [E]  P  G×2 형태."""
    parts = []
    i = 0
    while i < len(steps):
        s = steps[i]
        if s.get("is_error"):
            parts.append("[E]")
            i += 1
        elif s["source"] == "patcher":
            j = i
            while j < len(steps) and steps[j]["source"] == "patcher" and not steps[j].get("is_error"):
                j += 1
            cnt = j - i
            parts.append(f"P×{cnt}" if cnt > 1 else "P")
            i = j
        else:
            j = i
            while j < len(steps) and steps[j]["source"] == "gen" and not steps[j].get("is_error"):
                j += 1
            cnt = j - i
            parts.append(f"G×{cnt}" if cnt > 1 else "G")
            i = j
    return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 생성 루프
# ─────────────────────────────────────────────────────────────────────────────

def generate_trajectory(
    item: dict,
    model, tokenizer, input_device,
    save_fn=None,              # save_fn(traj, traj_type) → 완성된 trajectory 저장
    save_intermediate_fn=None, # save_intermediate_fn(traj) → patcher 라운드마다 traj_all 저장
) -> tuple[list[dict], list[dict], float]:
    """
    단일 문제 trajectory 생성.
    save_fn이 주어지면 각 trajectory 완성 즉시 저장.
    save_intermediate_fn이 주어지면 patcher 라운드마다 중간 상태를 traj_all에 저장.
    Returns: (traj_gen_list, traj_pat_list, traj_mix_list, total_cost_usd)
    """
    def _emit(traj: dict, traj_type: str, lst: list):
        lst.append(traj)
        if save_fn is not None:
            save_fn(traj, traj_type)

    problem     = item["problem"]
    gold_answer = item["answer"]
    problem_id  = item.get("id", "?")

    logger.info(f"[ID {problem_id}]  정답: {gold_answer!r}")

    traj_gen_list: list[dict] = []
    traj_mix_list: list[dict] = []
    total_cost = 0.0
    traj_idx   = 1

    history: list[dict] = []   # gen 컨텍스트 (history + error_step + pat_step 누적)
    mix_buf: list[dict] = []   # traj_mix 저장용 버퍼
    patcher_round = 0
    api_calls     = 0

    for rnd in range(1, MAX_STEPS + 2):
        # ── Generator ─────────────────────────────────────────────────────────
        gen_steps = _run_generator(model, tokenizer, input_device, problem, history)
        if not gen_steps:
            logger.warning("[Generator] 빈 응답, 중단")
            print(f"  ✗  (empty gen)  → skipped")
            break

        # ── Patcher: 항상 호출 (summary 생성 + 에러 검증) ───────────────────────
        if api_calls >= MAX_API:
            logger.info(f"MAX_API({MAX_API}) 초과, 미완성 저장")
            all_steps_so_far = mix_buf + [dict(s) for s in gen_steps]
            print(f"  ✗  {_fmt(all_steps_so_far)}  → wrong (max_api)")
            _emit(
                _build_traj(problem_id, problem, gold_answer,
                            all_steps_so_far, False, "mix", traj_idx=traj_idx),
                "mix", traj_mix_list,
            )
            traj_idx += 1
            break

        all_steps   = history + gen_steps
        step_offset = len(history)
        error_idx, pat_step, cost, error_summary, per_step_summaries = _run_patcher_step(
            problem, gen_steps, gold_answer, step_offset=step_offset, history=history
        )
        total_cost += cost
        api_calls  += 1

        # 모델이 실제 스텝 수를 초과한 번호를 반환할 경우 클램프
        if error_idx is not None:
            error_idx = min(error_idx, len(all_steps))

        # gen 스텝에 summary 부착
        for j, gs in enumerate(gen_steps):
            global_num = step_offset + j + 1
            if global_num in per_step_summaries:
                gs["summary"] = per_step_summaries[global_num]

        # ── 에러 없음: 정답 확정 ──────────────────────────────────────────────
        if error_idx is None:
            if patcher_round == 0:
                print(f"  ✓  {_fmt(gen_steps)}  → correct (gen only)")
                _emit(
                    _build_traj(problem_id, problem, gold_answer,
                                gen_steps, True, "gen", traj_idx=traj_idx),
                    "gen", traj_gen_list,
                )
            else:
                mix_buf += [dict(s) for s in gen_steps]
                print(f"  ✓  {_fmt(mix_buf)}  → correct (mix)")
                _emit(
                    _build_traj(problem_id, problem, gold_answer,
                                mix_buf, True, "mix", traj_idx=traj_idx),
                    "mix", traj_mix_list,
                )
            traj_idx += 1
            break

        # ── 에러 있음: 수정 후 계속 ───────────────────────────────────────────
        if pat_step is None:
            logger.warning("patcher_step 수정 실패, 중단")
            all_steps_so_far = mix_buf + [dict(s) for s in gen_steps]
            print(f"  ✗  {_fmt(all_steps_so_far) or '(no steps)'}  → wrong (patcher failed)")
            break

        if patcher_round >= MAX_STEPS:
            logger.info(f"MAX_STEPS({MAX_STEPS}) 초과, 미완성 저장")
            all_steps_so_far = mix_buf + [dict(s) for s in gen_steps]
            print(f"  ✗  {_fmt(all_steps_so_far)}  → wrong (max_steps)")
            _emit(
                _build_traj(problem_id, problem, gold_answer,
                            all_steps_so_far, False, "mix", traj_idx=traj_idx),
                "mix", traj_mix_list,
            )
            traj_idx += 1
            break

        err_i           = error_idx - 1
        new_gen_correct = gen_steps[: err_i - len(history)]
        error_step      = dict(all_steps[err_i], is_error=True, is_first_pat=False,
                               summary=error_summary)

        mix_buf  += [dict(s) for s in new_gen_correct] + [error_step, pat_step]
        history   = list(history) + [dict(s) for s in new_gen_correct] + [pat_step]
        patcher_round += 1

        last_text  = mix_buf[-1]["text"]
        has_boxed  = extract_boxed(last_text) is not None
        print(f"  [api={api_calls}/{MAX_API}]  have_boxed={has_boxed}  {_fmt(mix_buf)}  | → round {rnd + 1}")

        # patcher 라운드마다 중간 상태를 traj_all에 기록
        if save_intermediate_fn is not None:
            save_intermediate_fn(
                _build_traj(problem_id, problem, gold_answer,
                            mix_buf, False, "mix_intermediate", traj_idx=traj_idx)
            )
            traj_idx += 1

    logger.info(f"완료 → gen={len(traj_gen_list)}  mix={len(traj_mix_list)}")
    return traj_gen_list, traj_mix_list, total_cost


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trajectory SFT 데이터 생성")
    parser.add_argument("--num_data", type=int, default=None)
    parser.add_argument("--offset",   type=int, default=0)
    parser.add_argument("--output",   type=str, default=None,
                        help="출력 폴더 경로 (기본: output/sft_trajectory/{timestamp})")
    args = parser.parse_args()

    root         = Path(__file__).resolve().parent.parent
    gt_cfg       = CONF.get("generate_trajectory", {})
    dataset_path = (
        gt_cfg.get("base_problems")
        or str(root / CONF["data_path"]["deepmath_16k"])
    )

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else (root / "output" / "sft_trajectory" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 로깅 설정: logger → 파일 전용, print → 터미널+파일 ────────────────────
    import sys as _sys

    log_path = out_dir / "run.log"

    # root logger에서 콘솔 핸들러 제거 후 파일 핸들러만 등록
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    root_logger.addHandler(file_handler)

    # print()는 터미널 + 로그 파일 양쪽에 기록
    class _Tee:
        def __init__(self, *streams): self._streams = streams
        def write(self, data):
            for s in self._streams: s.write(data)
        def flush(self):
            for s in self._streams: s.flush()

    _log_file   = open(log_path, "a", encoding="utf-8")
    _sys.stdout = _Tee(_sys.__stdout__, _log_file)

    files = {
        k: open(out_dir / f"traj_{k}.jsonl", "w", encoding="utf-8")
        for k in ("gen", "mix", "all")
    }

    num_data = args.num_data or gt_cfg.get("num_data", 1)

    logger.info(f"데이터셋={dataset_path}  출력={out_dir}  num_data={num_data}  offset={args.offset}  Patcher={PATCHER}  MAX_STEPS={MAX_STEPS}  MAX_API={MAX_API}")

    rollout_gpus = gt_cfg.get("rollout_gpus", None)
    if rollout_gpus:
        cuda_visible = ",".join(str(g) for g in rollout_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        logger.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}")

    items = load_dataset_file(dataset_path)
    items = items[args.offset:] if num_data == -1 else items[args.offset : args.offset + num_data]
    logger.info(f"로드된 문제 수: {len(items)}")

    base_model_id = CONF["checkpoint"]["base"]
    logger.info(f"Generator 로딩 중: {base_model_id}")
    model, tokenizer = load_generator(model_path=base_model_id)
    input_device     = next(model.parameters()).device
    logger.info(f"Generator 로드 완료 (input_device={input_device})")

    def _save(traj: dict, traj_type: str):
        line = json.dumps(traj, ensure_ascii=False) + "\n"
        files[traj_type].write(line); files[traj_type].flush()
        files["all"].write(line);     files["all"].flush()

    def _save_intermediate(traj: dict):
        line = json.dumps(traj, ensure_ascii=False) + "\n"
        files["all"].write(line); files["all"].flush()

    t_start    = time.time()
    counts     = {"gen": 0, "mix": 0}
    total_cost = 0.0

    def _save_and_count(traj: dict, traj_type: str):
        _save(traj, traj_type)
        counts[traj_type] += 1

    try:
        pbar = tqdm(items, total=len(items), desc="generating", unit="prob")
        for item in pbar:
            gen_list, mix_list, cost = generate_trajectory(
                item, model, tokenizer, input_device,
                save_fn=_save_and_count,
                save_intermediate_fn=_save_intermediate,
            )
            total_cost += cost
            pbar.set_postfix(gen=counts["gen"], mix=counts["mix"])
            logger.info(f"누적 → gen={counts['gen']}  mix={counts['mix']}")
    finally:
        for f in files.values():
            f.close()
        _sys.stdout = _sys.__stdout__
        _log_file.close()

    elapsed_min = (time.time() - t_start) / 60
    total_traj  = sum(counts.values())

    logger.info(f"완료: {len(items)}개 문제 / {total_traj}개 trajectory  gen={counts['gen']}  mix={counts['mix']}  소요={elapsed_min:.1f}분  비용=${total_cost:.4f}  출력={out_dir}")


if __name__ == "__main__":
    main()
