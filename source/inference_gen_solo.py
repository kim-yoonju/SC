"""
evaluate_gen_solo.py
PRM 없이 gen의 self-check(critique + next_action)만으로 궤적을 구동하여
gen 단독 성능을 평가.

흐름:
  1. Generator가 스텝별로 풀이 + self-check (critique + next_action 포함)
  2. Gen의 pred_action에 따라:
     - TOKEN_SOLVE  → 스텝 수락, 다음 스텝으로
     - TOKEN_CORRECT → gen의 deep critique를 error_explanation으로 rethink (1회까지)
     - TOKEN_END    → 최종 답 추출 후 gold_answer와 비교
  3. max_steps 초과 시 fail로 처리

출력 (output/eval_gen_solo/{timestamp}/):
  results.jsonl  문제별 결과 (is_right, n_steps, fail_reason, ...)
  summary.json   전체 정확도 요약
  run.log        실행 로그
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    CONF,
    TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END, ACTION_TOKENS,
    load_generator, load_generator_vllm, load_step_manager, build_chat_prompt,
    run_log_direct, set_run_log,
)
from utils_math import extract_boxed, has_boxed, check_solved
from generate_utils import load_dataset_file

_ROOT_PATH    = Path(__file__).resolve().parent.parent
_PROMPTS_PATH = _ROOT_PATH / "prompts"

logger = logging.getLogger(__name__)


def _setup_logging(log_path=None):
    fmt  = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
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

_setup_logging()


# ── Config ────────────────────────────────────────────────────────────────────
_GT_CFG             = CONF.get("generate_trajectory", {})
TRAJ_MAX_NEW_TOKENS = _GT_CFG.get("max_new_tokens", 4096)
TRAJ_MAX_STEPS      = _GT_CFG.get("max_steps", None)
MAX_SUBSTEP_DEPTH   = _GT_CFG.get("max_substep_depth", 2)
USE_VLLM            = _GT_CFG.get("use_vllm", False)
_GENERATOR_API      = CONF.get("API_model", {}).get("GENERATOR")


# ── Prompts & Rubrics ─────────────────────────────────────────────────────────
_RUBRIC_NAMES: list[str] = []


def _load_prompts() -> tuple[str, str]:
    """(GEN_SOLVE_PROMPT, GEN_RETHINK_PROMPT) 로드 및 _RUBRIC_NAMES 채우기."""
    rubric_rel  = CONF.get("PRM", {}).get("rubric", "prompts/prm_rubric_v6.2.jsonl")
    rubric_file = Path(rubric_rel) if Path(rubric_rel).is_absolute() else _ROOT_PATH / rubric_rel
    rubric_lines = []
    with open(rubric_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                _RUBRIC_NAMES.append(e["name"])
                rubric_lines.append(f"{e['name']}: [correct/incorrect — {e['criterion']}]")
    rubric_str = "\n".join(rubric_lines)
    prompts: dict[str, str] = {}
    with open(_PROMPTS_PATH / "action_prompts.json", encoding="utf-8") as f:
        for e in json.load(f):
            prompts[e["name"]] = e["content"].replace("{{rubric}}", rubric_str)
    return prompts["gen_solve_R"], prompts["gen_rethink_R"]


GEN_SOLVE_PROMPT, GEN_RETHINK_PROMPT = _load_prompts()


# ── Step Manager: Atomicity 분해 ──────────────────────────────────────────────

def _load_atomicity_system_prompt() -> str:
    rubric_rel  = CONF.get("PRM", {}).get("rubric", "prompts/prm_rubric_v6.2.jsonl")
    path = _ROOT_PATH / rubric_rel if not Path(rubric_rel).is_absolute() else Path(rubric_rel)
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
    rubric_rel  = CONF.get("PRM", {}).get("rubric", "prompts/prm_rubric_v6.2.jsonl")
    path = _ROOT_PATH / rubric_rel if not Path(rubric_rel).is_absolute() else Path(rubric_rel)
    result = {}
    if not path.exists():
        return result
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
    반환: [{"goal": ..., "depth": 0}, {"goal": ..., "depth": 0}] 또는 None (atomic이면 분해 불가)
    """
    if sm_model is None or not _DECOMPOSE_SYSTEM:
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

    if not re.search(r"Verdict:\s*incorrect", resp, re.I):
        return None  # atomic

    m_a = re.search(r"Sub-step A:\s*(.*?)(?=Sub-step B:|Independence:|Verdict:|$)", resp, re.DOTALL | re.I)
    m_b = re.search(r"Sub-step B:\s*(.*?)(?=Independence:|Verdict:|$)", resp, re.DOTALL | re.I)
    sub1 = m_a.group(1).strip() if m_a else ""
    sub2 = m_b.group(1).strip() if m_b else ""
    if not sub1 or not sub2:
        return None
    return [{"goal": sub1, "depth": 0}, {"goal": sub2, "depth": 0}]


# ── Trajectory summary ───────────────────────────────────────────────────────
def _traj_summary(state: "ProblemState", is_right: bool, fail_reason: str | None) -> str:
    _labels = {TOKEN_SOLVE: "solve", TOKEN_CORRECT: "rethink", TOKEN_END: "END"}
    actions = [_labels.get(s["pred_action"], s["pred_action"]) for s in state.all_steps]
    outcome = "CORRECT" if is_right else f"FAIL({fail_reason})"
    return (
        f"[id={state.item.get('id', '?')}] "
        f"{' → '.join(actions)} | {outcome} | "
        f"steps={len(state.all_steps)} rethinks={state.n_rethinks}"
    )


# ── ProblemState ──────────────────────────────────────────────────────────────
@dataclass
class ProblemState:
    item:                      dict
    history:                   list = field(default_factory=list)   # 수락된 스텝
    all_steps:                 list = field(default_factory=list)   # 전체 스텝 (오류 포함)
    is_rethink:                bool = False
    step_rethink_tried:        bool = False
    step_substep_tried:        bool = False
    last_wrong_rubric_details: dict = field(default_factory=dict)
    last_wrong_step_text:      str  = ""
    n_rethinks:                int  = 0
    done:                      bool = False
    pending_step:              dict = field(default_factory=dict)
    result:                    dict = field(default_factory=dict)
    # ── substep 분해 ──────────────────────────────────────────────────────────
    in_substep_mode:           bool = False
    substep_queue:             list = field(default_factory=list)
    substep_passed:            list = field(default_factory=list)


# ── Text helpers ──────────────────────────────────────────────────────────────
# 모델이 step 이후 self-check 섹션을 시작하는 마커들
_CRITIQUE_MARKERS = [
    "\nFast critic:",
    "\nSelf-correction:",
    "\nDeep critic:",
    "\nFail rubrics:",
    "\nNext action:",
]


def _split_step_and_critique(full_text: str) -> tuple[str, str]:
    """full_text에서 step_text와 self_check_text를 분리.
    모델 포맷(gen_solve_R vs system_solve)에 무관하게 동작.
    """
    earliest = len(full_text)
    for marker in _CRITIQUE_MARKERS:
        idx = full_text.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    if earliest < len(full_text):
        return full_text[:earliest].strip(), full_text[earliest:]
    return full_text.strip(), ""


def _parse_next_action_text(sc_text: str) -> str | None:
    """self_check_text의 'Next action:' 섹션에서 action을 룰 기반으로 추출.
    명시적 토큰 > \\boxed{} > 빈 칸(None 반환) 순으로 판정.
    """
    m = re.search(r"Next action:\s*\n?(.*?)(?=\n[A-Z]|\Z)", sc_text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    content = m.group(1).strip()

    # 명시적 action 토큰 (텍스트로 출력된 경우)
    if TOKEN_END in content:
        return TOKEN_END
    if TOKEN_CORRECT in content:
        return TOKEN_CORRECT
    if TOKEN_SOLVE in content:
        return TOKEN_SOLVE

    # 모델이 \boxed{answer}를 Next action에 쓰는 경우 → END
    if has_boxed(content):
        return TOKEN_END

    # 빈 칸이면 판정 불가 → fallback으로 넘김
    return None


def _parse_gen_fast_critique(text: str, rubric_names: list[str]) -> dict:
    """Fast critic 섹션 파싱.
    반환: {rubric_name: {"verdict": "correct"/"incorrect", "critique": str|None}}
    """
    m = re.search(
        r"Fast\s+critic\s*:\s*\n(.*?)(?=\nDeep\s+critic\s*:|\nFail\s+rubrics\s*:|\Z)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return {}
    result = {}
    for line in m.group(1).split("\n"):
        line = line.strip().lstrip("-*•0123456789. ")
        if not line:
            continue
        for rn in rubric_names:
            if line.lower().startswith(rn.lower() + ":"):
                rest    = line[len(rn) + 1:].strip()
                verdict = "incorrect" if "incorrect" in rest.lower() else "correct"
                dash_m  = re.search(r"[—\-]\s*(.+)", rest)
                critique = dash_m.group(1).strip() if dash_m else None
                result[rn] = {"verdict": verdict, "critique": critique}
                break
    return result


def _parse_pred_fail_rubrics_from_text(text: str) -> list[str]:
    """Fail rubrics 섹션에서 루브릭 토큰 목록 추출.
    모델이 <|token|> 형태로 출력하거나 'none' 이면 빈 리스트 반환.
    """
    fr_m = re.search(r"Fail\s+rubrics\s*:", text, re.IGNORECASE)
    if not fr_m:
        return []
    rest = text[fr_m.end():]
    # "Next action:" 이전까지만 사용 (바로 이어지는 경우 포함)
    na_m = re.search(r"Next\s+action\s*:", rest, re.IGNORECASE)
    section = rest[: na_m.start()].strip() if na_m else rest.strip()
    if not section or section.lower() in ("none", "none."):
        return []
    tokens = re.findall(r"<\|[^|>]+\|>", section)
    if tokens:
        return tokens
    # fallback: 줄 단위로 수집 (none이나 빈 줄 제외)
    return [p.strip() for p in re.split(r"[,\n]", section)
            if p.strip() and p.strip().lower() not in ("none", "")]


def _action_to_str(action: str | None) -> str | None:
    """TOKEN_* 상수를 "solve"/"rethink"/"end" 문자열로 변환."""
    return {TOKEN_SOLVE: "solve", TOKEN_CORRECT: "rethink", TOKEN_END: "end"}.get(action)


def _extract_verdicts_from_text(sc_text: str) -> tuple[int, int, int]:
    """self-check 텍스트에서 correct/incorrect/na 카운트 추출."""
    na = len(re.findall(r":\s*(?:not applicable|n/a)\b", sc_text, re.I))
    boxed = re.findall(
        r"\\boxed\{(?:\\text\{)?\s*(correct|incorrect)\s*\}+", sc_text, re.I
    )
    if boxed:
        correct   = sum(1 for v in boxed if v.lower() == "correct")
        incorrect = sum(1 for v in boxed if v.lower() == "incorrect")
        return correct, incorrect, na
    plain     = re.findall(r":\s*(correct|incorrect)\b", sc_text, re.I)
    correct   = sum(1 for v in plain if v.lower() == "correct")
    incorrect = sum(1 for v in plain if v.lower() == "incorrect")
    return correct, incorrect, na


def _parse_generator_deep_check(text: str, rubric_names: list[str]) -> dict[str, dict]:
    """Deep critic 섹션 파싱. 반환: {rubric_name: {'verdict', 'critique'}}"""
    m = re.search(
        r"(?:Deep\s+critic|deep_critique)\s*:\s*\n(.*?)$",
        text, re.DOTALL | re.IGNORECASE
    )
    if not m:
        return {}
    section    = m.group(1)
    first_line = next((l.strip() for l in section.split("\n") if l.strip()), "")
    if first_line.lower() == "none":
        return {}

    results = {}
    for i, rubric in enumerate(rubric_names):
        later   = [re.escape(r) for r in rubric_names[i + 1:]]
        end_pat = rf"(?:[-*•\s]*(?:{'|'.join(later)})\s*[:\.])" if later else r"\Z"
        pat     = re.search(
            rf"(?:[-*•]\s*)?{re.escape(rubric)}\s*[:\.]?\s*(.*?)(?={end_pat}|\Z)",
            section, re.DOTALL | re.IGNORECASE
        )
        if not pat:
            continue
        block = pat.group(1).strip()
        if not block:
            continue
        paras = [p for p in block.split("\n\n") if not p.strip().startswith("\\[")]
        block = "\n\n".join(paras).strip()
        if not block:
            continue
        vm = re.search(r"Verdict\s*:\s*(correct|incorrect)", block, re.IGNORECASE)
        if vm:
            verdict   = "pass" if vm.group(1).lower() == "correct" else "fail"
            before    = block[:vm.start()].strip()
            after     = re.sub(r"^[\s—\-\.]+", "", block[vm.end():]).strip()
            reasoning = " ".join(filter(None, [before, after])) or None
        elif "incorrect" in block.lower():
            verdict, reasoning = "fail", block
        elif "correct" in block.lower():
            verdict, reasoning = "pass", block
        else:
            verdict, reasoning = None, block
        results[rubric] = {"verdict": verdict, "critique": reasoning}
    return results


def _build_rethink_explanation(state: ProblemState) -> str:
    """generate_trajectory.py 방식: [What was attempted] + [Fail rubrics] + [Why it failed] + [How to fix]."""
    parts = []

    if state.last_wrong_step_text:
        parts.append(f"[What was attempted]\n{state.last_wrong_step_text[:400]}")

    fail_details = {
        rname: critique
        for rname, critique in state.last_wrong_rubric_details.items()
        if rname != "Atomicity"
    }

    if fail_details:
        parts.append("[Fail rubrics]\n" + "\n".join(f"- {rn}" for rn in fail_details))

        why_lines = [
            f"- {rn}: {critique.strip()}"
            for rn, critique in fail_details.items()
            if critique and critique.strip()
        ]
        if why_lines:
            parts.append("[Why it failed]")
            parts.extend(why_lines)

        how_lines = [
            f"- {rn}: {g}"
            for rn in fail_details
            if (g := _RETHINK_GUIDANCE.get(rn, ""))
        ]
        if how_lines:
            parts.append("[How to fix]")
            parts.extend(how_lines)

    if state.in_substep_mode and state.substep_queue:
        parts.append(f"[Focus ONLY on]: {state.substep_queue[0]['goal']}")

    return "\n".join(parts) if parts else "the previous step contained an error — try a completely different approach"


# ── Logits Processors ─────────────────────────────────────────────────────────
def _get_ws_token_ids(tokenizer) -> set[int]:
    """'\n', ' \n', ' ' 등 공백 계열 토큰 ID 수집."""
    ws_ids: set[int] = set()
    for s in ["\n", " \n", "\r\n", " "]:
        try:
            ws_ids.update(tokenizer.encode(s, add_special_tokens=False))
        except Exception:
            pass
    return ws_ids


_RUBRIC_SPECIAL_TOKEN_STRS = [
    "<|algebraic_manipulation|>",
    "<|abstract_and_linear_algebra_operations|>",
    "<|calculus_computation|>",
    "<|function_and_limit_analysis|>",
    "<|geometric_reasoning|>",
    "<|counting_and_probability|>",
    "<|number_theoretic_reasoning|>",
    "<|logical_and_discrete_reasoning|>",
    "<|differential_equations|>",
    "<|progress_and_non-repetition|>",
    "<|atomicity|>",
]


def _get_fail_rubrics_allowed_ids(tokenizer) -> tuple[set[int], set[int]]:
    """(allowed_ids, next_ids) 반환.
    allowed_ids: Fail rubrics 섹션에서 허용할 토큰 집합
    next_ids:    섹션 종료를 알리는 'Next' 토큰 집합
    """
    ids: set[int] = set()
    for s in _RUBRIC_SPECIAL_TOKEN_STRS:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid != tokenizer.unk_token_id:
            ids.add(tid)
    for s in ["none", "None"]:
        ids.update(tokenizer.encode(s, add_special_tokens=False))
    ids.update(tokenizer.encode("\n", add_special_tokens=False))   # 줄바꿈(구분자)
    next_ids: set[int] = set(tokenizer.encode("Next", add_special_tokens=False))
    ids |= next_ids                                                 # 섹션 전환 허용
    return ids, next_ids


class _FailRubricsLogitsProcessor:
    """'Fail rubrics:' 이후 ~ 'Next action:' 이전 구간을
    rubric special tokens / none / \\n / Next 토큰으로만 제한.

    모델이 '\\n'만 반복하는 대신 명시적으로 실패 루브릭 또는 none을
    선택하도록 강제한다.
    """

    def __init__(self, input_len: int, tokenizer, allowed_ids: set[int], next_ids: set[int]):
        self.input_len = input_len
        self.tokenizer = tokenizer
        self._allowed  = allowed_ids
        self._next_ids = next_ids
        self.active: list[bool] = []
        self.exited: list[bool] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch = input_ids.shape[0]
        if not self.active:
            self.active = [False] * batch
            self.exited = [False] * batch

        for i in range(batch):
            if self.exited[i]:
                continue

            gen_ids = input_ids[i, self.input_len:].tolist()

            if not self.active[i]:
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                if "Fail rubrics:" in gen_text:
                    self.active[i] = True

            if self.active[i]:
                # 직전 토큰이 "Next" → 섹션 종료, 이후는 자유 생성
                if gen_ids and gen_ids[-1] in self._next_ids:
                    self.exited[i] = True
                    continue

                mask = torch.full_like(scores[i], float("-inf"))
                for tid in self._allowed:
                    if tid < mask.shape[0]:
                        mask[tid] = scores[i, tid]
                scores[i] = mask

        return scores


class _NextActionHFLogitsProcessor:
    """HF batch 생성 시 'Next action:' 이후 action token / 공백만 허용."""

    def __init__(self, input_len: int, tokenizer, action_token_ids: list[int]):
        self.input_len = input_len
        self.tokenizer = tokenizer
        self.action_ids = set(action_token_ids)
        self.ws_ids = _get_ws_token_ids(tokenizer)
        self._allowed: set[int] = self.action_ids | self.ws_ids
        self.active: list[bool] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch = input_ids.shape[0]
        if not self.active:
            self.active = [False] * batch

        for i in range(batch):
            if not self.active[i]:
                gen_ids = input_ids[i, self.input_len:].tolist()
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                if "Next action:" in gen_text:
                    self.active[i] = True

            if self.active[i]:
                mask = torch.full_like(scores[i], float("-inf"))
                for tid in self._allowed:
                    if tid < mask.shape[0]:
                        mask[tid] = scores[i, tid]
                scores[i] = mask

        return scores



# ── Generator (HF) ────────────────────────────────────────────────────────────
def _batch_run_generator(model, tokenizer, input_device,
                         states: list[ProblemState]) -> None:
    """HF 모델로 배치 생성. 결과를 state.pending_step에 저장."""
    prompts = []
    for state in states:
        step_number = len(state.history) + 1
        lines = [f"[Problem]\n{state.item['problem']}"]
        if state.history:
            lines.append("\n[Previous steps]")
            for i, s in enumerate(state.history, 1):
                lines.append(f"Step {i}: {s['text']}")
        lines.append(f"\nWrite Step {step_number}.")
        system_prompt = (
            GEN_RETHINK_PROMPT.replace("{{error_explanation}}", _build_rethink_explanation(state))
            if state.is_rethink else GEN_SOLVE_PROMPT
        )
        prompts.append(build_chat_prompt(tokenizer, system_prompt, "\n".join(lines)))

    im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    _action_ids = [
        tokenizer.convert_tokens_to_ids(t)
        for t in [TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END]
    ]
    _action_ids = [tid for tid in _action_ids if tid is not None and tid != tokenizer.unk_token_id]
    _eos_ids    = [tid for tid in {tokenizer.eos_token_id, *_action_ids} if tid is not None]

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(input_device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]

    from transformers import LogitsProcessorList
    _fr_allowed, _fr_next = _get_fail_rubrics_allowed_ids(tokenizer)
    _fr_processor = _FailRubricsLogitsProcessor(input_len, tokenizer, _fr_allowed, _fr_next)
    _na_processor = _NextActionHFLogitsProcessor(input_len, tokenizer, _action_ids)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=TRAJ_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=_eos_ids,
            logits_processor=LogitsProcessorList([_fr_processor, _na_processor]),
        )

    resp_all  = out[:, input_len:]
    _stop_ids = {tokenizer.pad_token_id, im_end_id, *_action_ids}

    _id_to_action = {
        tokenizer.convert_tokens_to_ids(TOKEN_SOLVE):   TOKEN_SOLVE,
        tokenizer.convert_tokens_to_ids(TOKEN_CORRECT): TOKEN_CORRECT,
        tokenizer.convert_tokens_to_ids(TOKEN_END):     TOKEN_END,
    }

    for j, state in enumerate(states):
        resp = resp_all[j]
        trim = resp.shape[0]
        stopped_action = None
        for pos, tid in enumerate(resp.tolist()):
            if tid in _stop_ids:
                trim = pos
                stopped_action = _id_to_action.get(tid)
                break
        full_text = tokenizer.decode(resp[:trim], skip_special_tokens=False).strip()

        step_text, self_check_text = _split_step_and_critique(full_text)

        # stop token으로 action이 확정된 경우 텍스트 파싱 불필요
        pred_action = stopped_action or _parse_next_action_text(self_check_text)
        if pred_action is None:
            correct_count, incorrect_count, _ = _extract_verdicts_from_text(self_check_text)
            if incorrect_count > correct_count:
                pred_action = TOKEN_CORRECT
            elif has_boxed(step_text) or has_boxed(self_check_text):
                pred_action = TOKEN_END
            else:
                pred_action = TOKEN_SOLVE

        # 학습 데이터 불변식: END는 반드시 \boxed{}와 함께 나옴
        if pred_action == TOKEN_END and not has_boxed(full_text):
            logger.info(
                f"[Generator HF] id={state.item.get('id')} step={len(state.history)+1} "
                f"TOKEN_END without \\boxed{{}} → override to TOKEN_SOLVE"
            )
            pred_action = TOKEN_SOLVE

        _TOOL_CALL_MARKERS    = ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")
        _TEMPLATE_PLACEHOLDERS = ("[Your one reasoning step", "[Your one corrected reasoning step")
        is_error = (
            any(m in full_text for m in _TOOL_CALL_MARKERS) or
            any(step_text.strip().startswith(p) for p in _TEMPLATE_PLACEHOLDERS)
        )

        logger.info(
            f"[Generator HF] id={state.item.get('id')} "
            f"step={len(state.history)+1} rethink={state.is_rethink} "
            f"pred_action={pred_action} is_error={is_error}"
        )

        state.pending_step = {
            "text":              step_text,
            "inference":         step_text,
            "full_text":         full_text,
            "pred_action":       pred_action,
            "next_pred_action":  _action_to_str(pred_action),
            "gen_fast_critique": _parse_gen_fast_critique(full_text, _RUBRIC_NAMES),
            "gen_deep_critique": _parse_generator_deep_check(full_text, _RUBRIC_NAMES),
            "pred_fail_rubrics": _parse_pred_fail_rubrics_from_text(full_text),
            "is_error":          is_error,
            "role":              "rethink" if state.is_rethink else "gen",
        }


# ── Generator (vLLM) ──────────────────────────────────────────────────────────
def _batch_run_generator_vllm(llm, tokenizer, _device,
                               states: list[ProblemState]) -> None:
    """vLLM로 배치 생성. 결과를 state.pending_step에 저장."""
    from vllm import SamplingParams

    prompts = []
    for state in states:
        step_number = len(state.history) + 1
        lines = [f"[Problem]\n{state.item['problem']}"]
        if state.history:
            lines.append("\n[Previous steps]")
            for i, s in enumerate(state.history, 1):
                lines.append(f"Step {i}: {s['text']}")
        lines.append(f"\nWrite Step {step_number}.")
        system_prompt = (
            GEN_RETHINK_PROMPT.replace("{{error_explanation}}", _build_rethink_explanation(state))
            if state.is_rethink else GEN_SOLVE_PROMPT
        )
        prompts.append(build_chat_prompt(tokenizer, system_prompt, "\n".join(lines)))

    _action_ids = [
        tokenizer.convert_tokens_to_ids(t)
        for t in [TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END]
    ]
    _action_ids   = [tid for tid in _action_ids if tid is not None and tid != tokenizer.unk_token_id]
    _tok_to_action = {tid: tok for tok, tid in zip(
        [TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END],
        [tokenizer.convert_tokens_to_ids(t) for t in [TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END]]
    )}

    # stop_token_ids: fine-tuned 모델용 / stop: base 모델이 텍스트로 뱉을 때
    # vLLM V1(0.7+)은 per-request logits processor 미지원으로 제거
    params = SamplingParams(
        max_tokens=TRAJ_MAX_NEW_TOKENS,
        temperature=0,
        stop_token_ids=_action_ids,
        stop=ACTION_TOKENS,
    )
    outputs = llm.generate(prompts, params, use_tqdm=False)

    # Phase 2: "Next action:" 이후 action token을 뱉지 않은 요청만 재생성
    _NA_MARKER = "Next action:"
    _second_info: list[tuple[int, str]] = []  # (index, extended_prompt)
    for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
        sr = output.outputs[0].stop_reason
        action_stopped = (
            (isinstance(sr, int) and sr in _tok_to_action) or
            sr in (TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END)
        )
        raw_text = output.outputs[0].text
        if not action_stopped and _NA_MARKER in raw_text:
            na_pos = raw_text.rfind(_NA_MARKER)
            extended = prompt + raw_text[: na_pos + len(_NA_MARKER)]
            _second_info.append((idx, extended))

    _forced_actions: dict[int, str] = {}
    if _second_info:
        sec_params = SamplingParams(
            max_tokens=5,  # "\n" + action token 정도면 충분
            temperature=0,
            stop_token_ids=_action_ids,
            stop=ACTION_TOKENS,
        )
        sec_outputs = llm.generate(
            [ep for _, ep in _second_info], sec_params, use_tqdm=False
        )
        for (orig_idx, _), sec_out in zip(_second_info, sec_outputs):
            sr2 = sec_out.outputs[0].stop_reason
            if isinstance(sr2, int) and sr2 in _tok_to_action:
                _forced_actions[orig_idx] = _tok_to_action[sr2]
            elif sr2 in (TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END):
                _forced_actions[orig_idx] = sr2

    _TOOL_CALL_MARKERS    = ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")
    _TEMPLATE_PLACEHOLDERS = ("[Your one reasoning step", "[Your one corrected reasoning step")

    for idx, (state, output) in enumerate(zip(states, outputs)):
        full_text   = output.outputs[0].text.strip()
        stop_reason = output.outputs[0].stop_reason

        step_text, self_check_text = _split_step_and_critique(full_text)

        # 우선순위: stop token ID > stop string > phase-2 강제 선택 > Next action 텍스트 파싱 > fallback
        if isinstance(stop_reason, int) and stop_reason in _tok_to_action:
            pred_action = _tok_to_action[stop_reason]
        elif stop_reason in (TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END):
            pred_action = stop_reason
        elif idx in _forced_actions:
            pred_action = _forced_actions[idx]
        else:
            pred_action = _parse_next_action_text(self_check_text)
            if pred_action is None:
                correct_count, incorrect_count, _ = _extract_verdicts_from_text(self_check_text)
                if incorrect_count > correct_count:
                    pred_action = TOKEN_CORRECT
                elif has_boxed(step_text) or has_boxed(self_check_text):
                    pred_action = TOKEN_END
                else:
                    pred_action = TOKEN_SOLVE

        # 학습 데이터 불변식: END는 반드시 \boxed{}와 함께 나옴
        if pred_action == TOKEN_END and not has_boxed(full_text):
            logger.info(
                f"[Generator vLLM] id={state.item.get('id')} step={len(state.history)+1} "
                f"TOKEN_END without \\boxed{{}} → override to TOKEN_SOLVE"
            )
            pred_action = TOKEN_SOLVE

        is_error = (
            any(m in full_text for m in _TOOL_CALL_MARKERS) or
            any(step_text.strip().startswith(p) for p in _TEMPLATE_PLACEHOLDERS)
        )

        logger.info(
            f"[Generator vLLM] id={state.item.get('id')} "
            f"step={len(state.history)+1} rethink={state.is_rethink} "
            f"pred_action={pred_action} stop_reason={stop_reason!r} is_error={is_error}"
        )

        state.pending_step = {
            "text":              step_text,
            "inference":         step_text,
            "full_text":         full_text,
            "pred_action":       pred_action,
            "next_pred_action":  _action_to_str(pred_action),
            "gen_fast_critique": _parse_gen_fast_critique(full_text, _RUBRIC_NAMES),
            "gen_deep_critique": _parse_generator_deep_check(full_text, _RUBRIC_NAMES),
            "pred_fail_rubrics": _parse_pred_fail_rubrics_from_text(full_text),
            "is_error":          is_error,
            "role":              "rethink" if state.is_rethink else "gen",
        }


# ── Generator (API) ───────────────────────────────────────────────────────────
def _run_generator_api(states: list[ProblemState]) -> None:
    """API 모델로 생성. 결과를 state.pending_step에 저장."""
    from utils import _call_llm
    api_model = CONF.get("API_model", {}).get("GENERATOR")

    def _one(state: ProblemState):
        step_number = len(state.history) + 1
        lines = [f"[Problem]\n{state.item['problem']}"]
        if state.history:
            lines.append("\n[Previous steps]")
            for i, s in enumerate(state.history, 1):
                lines.append(f"Step {i}: {s['text']}")
        lines.append(f"\nWrite Step {step_number}.")
        system_prompt = (
            GEN_RETHINK_PROMPT.replace("{{error_explanation}}", _build_rethink_explanation(state))
            if state.is_rethink else GEN_SOLVE_PROMPT
        )
        text = _call_llm(
            model_id=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": "\n".join(lines)},
            ],
            max_tokens=TRAJ_MAX_NEW_TOKENS,
        ) or ""

        step_text, self_check_text = _split_step_and_critique(text)

        pred_action = _parse_next_action_text(self_check_text)
        if pred_action is None:
            correct_count, incorrect_count, _ = _extract_verdicts_from_text(self_check_text)
            if incorrect_count > correct_count:
                pred_action = TOKEN_CORRECT
            elif has_boxed(step_text) or has_boxed(self_check_text):
                pred_action = TOKEN_END
            else:
                pred_action = TOKEN_SOLVE

        state.pending_step = {
            "text":              step_text,
            "inference":         step_text,
            "full_text":         text,
            "pred_action":       pred_action,
            "next_pred_action":  _action_to_str(pred_action),
            "gen_fast_critique": _parse_gen_fast_critique(text, _RUBRIC_NAMES),
            "gen_deep_critique": _parse_generator_deep_check(text, _RUBRIC_NAMES),
            "pred_fail_rubrics": _parse_pred_fail_rubrics_from_text(text),
            "is_error":          not text.strip(),
            "role":              "rethink" if state.is_rethink else "gen",
        }

    with ThreadPoolExecutor(max_workers=len(states)) as ex:
        list(ex.map(_one, states))


# ── Verbose Debug Print ───────────────────────────────────────────────────────
_VERBOSE = False  # --verbose 플래그로 활성화


def _print_step_verbose(state: "ProblemState", step: dict) -> None:
    """한 스텝의 생성 → 비평 → 리워드 시뮬레이션을 터미널에 상세 출력."""
    pid      = state.item.get("id", "?")
    step_num = len(state.all_steps) + 1
    W        = 72
    SEP      = "=" * W
    sub_sep  = "-" * W

    print(f"\n{SEP}")
    print(f"  Problem {pid}  |  Step {step_num}  |  rethink={state.is_rethink}")
    print(SEP)

    # ── Full Generated Text ──────────────────────────────────────────────────
    print("\n[FULL TEXT]")
    print(step.get("full_text") or step.get("text", ""))

    # ── Fast Critic ──────────────────────────────────────────────────────────
    fast = step.get("gen_fast_critique") or {}
    if fast:
        print(f"\n{sub_sep}")
        print("[FAST CRITIC]")
        for rub, v in fast.items():
            verdict  = v.get("verdict", "?")
            critique = v.get("critique") or ""
            tag = "✗" if verdict == "incorrect" else "✓"
            print(f"  {tag} {rub}: {verdict}" + (f"  — {critique[:60]}" if critique else ""))

    # ── Deep Critic ──────────────────────────────────────────────────────────
    deep = step.get("gen_deep_critique") or {}
    if deep:
        print(f"\n{sub_sep}")
        print("[DEEP CRITIC]")
        for rub, v in deep.items():
            verdict  = v.get("verdict", "?")
            critique = (v.get("critique") or "")
            na_flag  = "N/A" if "N/A" in critique or "not apply" in critique.lower() else ""
            tag = "✗" if verdict == "fail" else ("?" if verdict is None else "✓")
            summary  = critique[:80].replace("\n", " ")
            print(f"  {tag} {rub}: {verdict or '?'}  {na_flag}  {summary}")

    # ── Pred Fail Rubrics ────────────────────────────────────────────────────
    fail_rubrics = step.get("pred_fail_rubrics") or []
    print(f"\n{sub_sep}")
    print("[PRED FAIL RUBRICS]  (모델이 식별한 실패 루브릭)")
    print("  " + (", ".join(fail_rubrics) if fail_rubrics else "(none)"))

    # ── Action Decision ──────────────────────────────────────────────────────
    action = step.get("pred_action", "?")
    print(f"\n{sub_sep}")
    print(f"[PRED ACTION]  →  {action}")

    # ── Simulated Reward (PRM 없이 모델 자체 판단 기준) ──────────────────────
    deep_fails = [rub for rub, v in deep.items() if v.get("verdict") == "fail"]
    fast_fails = [rub for rub, v in fast.items() if v.get("verdict") == "incorrect"
                  and not any("N/A" in (deep.get(rub, {}).get("critique") or "")
                              or "not apply" in (deep.get(rub, {}).get("critique") or "").lower()
                              for _ in [0])]
    effective_fails = deep_fails or fast_fails

    print(f"\n{sub_sep}")
    print("[REWARD SIMULATION]  (PRM 호출 없이 모델 자체 critique 기준)")
    if effective_fails:
        model_pass = False
        print(f"  effective fail rubrics: {effective_fails}")
    else:
        model_pass = True
        print("  effective fail rubrics: (none) → step PASS")

    if action == "<|rethink|>":
        action_r = 1.0 if not model_pass else 0.0
        label = "✓ correct (rethink when fail)" if not model_pass else "✗ wrong (rethink when pass)"
    elif action == "<|solve|>":
        action_r = 1.0 if model_pass else 0.0
        label = "✓ correct (solve when pass)" if model_pass else "✗ wrong (solve when fail)"
    else:  # <|end|>
        has_boxed = r"\boxed{" in (step.get("full_text") or "")
        action_r = 1.0if has_boxed else 0.0
        label = "✓ has \\boxed{}" if has_boxed else "✗ no \\boxed{}"

    rubric_match = 1.0 if (not effective_fails and not fail_rubrics) else (
        0.0 if (effective_fails and not fail_rubrics) else
        len(set(effective_fails) & set(fail_rubrics)) / len(set(effective_fails) | set(fail_rubrics))
    )

    print(f"  rubric_match (Jaccard): {rubric_match:.2f}")
    print(f"  action_reward:          {action_r:.1f}  ({label})")
    print(f"  simulated total:        {rubric_match + action_r:.2f}")
    print(SEP)


# ── Process Gen Result ────────────────────────────────────────────────────────
def _process_gen_result(state: ProblemState, save_fn=None,
                        step_manager_model=None, step_manager_tok=None,
                        raw_step_save_fn=None) -> None:
    """Gen의 pred_action으로 state 업데이트. 완료 시 state.done = True."""
    step        = state.pending_step
    pred_action = step["pred_action"]
    problem_id  = state.item.get("id", "?")
    gold_answer = state.item.get("gold_answer") or state.item["answer"]

    if _VERBOSE:
        _print_step_verbose(state, step)

    # 스텝별 raw 출력 저장
    if raw_step_save_fn:
        raw_step_save_fn({
            "problem_id":        str(problem_id),
            "step_number":       len(state.all_steps) + 1,
            "is_rethink":        state.is_rethink,
            "role":              step.get("role", "gen"),
            "pred_action":       pred_action,
            "next_pred_action":  step.get("next_pred_action"),
            "step_text":         step["text"],
            "full_text":         step.get("full_text", step["text"]),
            "gen_fast_critique": step.get("gen_fast_critique"),
            "gen_deep_critique": step.get("gen_deep_critique"),
            "pred_fail_rubrics": step.get("pred_fail_rubrics"),
        })

    def _finish(is_right: bool, fail_reason: str = None, pred_answer: str = None):
        result = {
            "problem_id":  str(problem_id),
            "is_right":    is_right,
            "fail_reason": fail_reason,
            "n_steps":     len(state.all_steps),
            "n_rethinks":  state.n_rethinks,
            "pred_answer": pred_answer,
            "gold_answer": gold_answer,
            "steps": [
                {
                    "role":              s["role"],
                    "pred_action":       s["pred_action"],
                    "next_pred_action":  s.get("next_pred_action"),
                    "text":              s["text"],
                    "full_text":         s.get("full_text", s["text"]),
                    "inference":         s.get("inference", s["text"]),
                    "gen_fast_critique": s.get("gen_fast_critique"),
                    "gen_deep_critique": s.get("gen_deep_critique"),
                    "pred_fail_rubrics": s.get("pred_fail_rubrics"),
                }
                for s in state.all_steps
            ],
        }
        state.result = result
        if save_fn:
            save_fn(result)
        summary = _traj_summary(state, is_right, fail_reason)
        print(summary)
        logger.info(f"[traj] {summary}")
        state.done = True

    def _apply_wrong():
        """Gen이 이 스텝을 틀렸다고 판단 → rethink / substep 분해 / 종료."""
        step["is_error"] = True
        state.all_steps.append(step)

        if TRAJ_MAX_STEPS and len(state.all_steps) >= TRAJ_MAX_STEPS:
            _finish(False, fail_reason="max_steps")
            return

        if state.in_substep_mode:
            # 서브스텝 실패 → 종료 (patcher 없음)
            state.in_substep_mode = False
            state.substep_queue   = []
            logger.info(f"[id={problem_id}] 서브스텝 실패 → substep_fail")
            _finish(False, fail_reason="substep_fail")
            return

        if not state.step_rethink_tried:
            # gen의 deep critique를 rethink 설명으로 사용
            deep       = _parse_generator_deep_check(
                step.get("full_text") or step.get("text") or "", _RUBRIC_NAMES
            )
            fail_rubrics = {
                rname: info.get("critique") or ""
                for rname, info in deep.items()
                if info.get("verdict") == "fail"
            }
            # deep critique에 fail이 없으면 critique 있는 것 전체 사용
            if not fail_rubrics:
                fail_rubrics = {
                    rname: info.get("critique") or ""
                    for rname, info in deep.items()
                    if info.get("critique")
                }
            state.last_wrong_rubric_details = fail_rubrics
            state.last_wrong_step_text      = step["text"]
            state.is_rethink                = True
            state.step_rethink_tried        = True
            state.n_rethinks               += 1
            logger.info(f"[id={problem_id}] TOKEN_CORRECT → rethink (fail_rubrics={list(fail_rubrics.keys())})")
        elif not state.step_substep_tried:
            # rethink 후에도 실패 → step_manager로 분해 시도
            state.step_substep_tried = True
            substeps = _decompose_with_atomicity(
                state.item["problem"],
                [s["text"] for s in state.all_steps if not s.get("is_error")],
                state.last_wrong_step_text,
                step_manager_model,
                step_manager_tok,
            )
            if substeps:
                state.substep_queue   = substeps
                state.substep_passed  = []
                state.in_substep_mode = True
                state.is_rethink      = True
                state.n_rethinks     += 1
                logger.info(f"[id={problem_id}] substep 분해 성공: {[s['goal'][:50] for s in substeps]}")
            else:
                logger.info(f"[id={problem_id}] substep 분해 불가(atomic) → rethink_fail")
                _finish(False, fail_reason="rethink_fail")
        else:
            logger.info(f"[id={problem_id}] rethink + substep 모두 실패 → rethink_fail")
            _finish(False, fail_reason="rethink_fail")

    # 할루시네이션 / 빈 응답
    if step.get("is_error"):
        logger.info(f"[id={problem_id}] is_error → apply_wrong")
        _apply_wrong()
        return

    if pred_action == TOKEN_CORRECT:
        _apply_wrong()

    elif pred_action == TOKEN_END:
        state.all_steps.append(step)
        state.history.append(step)
        state.is_rethink         = False
        state.step_rethink_tried = False
        state.step_substep_tried = False
        state.in_substep_mode    = False
        state.substep_queue      = []
        pred_answer = extract_boxed(step.get("full_text") or step["text"])
        is_right    = check_solved(step.get("full_text") or step["text"], gold_answer)
        _finish(is_right, fail_reason=None if is_right else "wrong_answer", pred_answer=pred_answer)

    else:  # TOKEN_SOLVE
        state.all_steps.append(step)
        state.history.append(step)

        if state.in_substep_mode and state.substep_queue:
            # 서브스텝 하나 통과 → 다음 서브스텝 또는 정상 복귀
            state.substep_passed.append(step["text"])
            state.substep_queue.pop(0)
            if state.substep_queue:
                state.is_rethink = True
                logger.info(f"[id={problem_id}] 서브스텝 완료, 다음: {state.substep_queue[0]['goal'][:50]}")
            else:
                state.in_substep_mode    = False
                state.substep_passed     = []
                state.is_rethink         = False
                state.step_rethink_tried = False
                state.step_substep_tried = False
                logger.info(f"[id={problem_id}] 모든 서브스텝 통과 → 정상 복귀")
        else:
            state.is_rethink         = False
            state.step_rethink_tried = False
            state.step_substep_tried = False

        if TRAJ_MAX_STEPS and len(state.all_steps) >= TRAJ_MAX_STEPS:
            _finish(False, fail_reason="max_steps")


# ── Parallel helpers ──────────────────────────────────────────────────────────
def _parallel_gen(fn, generators: list, states: list) -> None:
    n      = len(generators)
    splits = [states[i::n] for i in range(n)]

    def _run(i):
        if splits[i]:
            model, tokenizer, device = generators[i]
            fn(model, tokenizer, device, splits[i])

    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_run, range(n)))


# ── Main Evaluation Loop ──────────────────────────────────────────────────────
def evaluate_gen_batch(
    items: list[dict],
    generators: list,
    n_parallel: int,
    save_fn=None,
    step_manager_model=None,
    step_manager_tok=None,
    raw_step_save_fn=None,
) -> list[dict]:
    """
    n_parallel개 문제를 동시에 처리하는 순차 배치 루프.
    반환: 문제별 result dict 리스트.
    """
    all_results: list[dict] = []
    _lock      = threading.Lock()
    _save_lock = threading.Lock()

    def _ts_save(rec: dict):
        with _save_lock:
            if save_fn:
                save_fn(rec)

    queue:  list[dict]         = list(items)
    active: list[ProblemState] = []
    pbar = tqdm(total=len(items), desc="gen-solo eval", unit="prob")

    def _fill_active():
        while len(active) < n_parallel and queue:
            active.append(ProblemState(item=queue.pop(0)))

    _fill_active()
    _round_idx = 0

    while active:
        # 완료된 state 수집
        done_states = [s for s in active if s.done]
        for s in done_states:
            with _lock:
                all_results.append(s.result)
            pbar.update(1)
        active = [s for s in active if not s.done]
        _fill_active()

        if not active:
            break

        _round_idx += 1
        logger.info(f"[Round {_round_idx}] active={len(active)}")

        # 생성
        if _GENERATOR_API:
            _run_generator_api(active)
        elif USE_VLLM and generators:
            llm_r, tok_r = generators[0][0], generators[0][1]
            _batch_run_generator_vllm(llm_r, tok_r, None, active)
        else:
            _parallel_gen(_batch_run_generator, generators, active)

        # pred_action 기반 상태 업데이트
        for state in active:
            _process_gen_result(state, _ts_save, step_manager_model, step_manager_tok, raw_step_save_fn)

    pbar.close()
    return all_results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gen 단독 성능 평가 (PRM 없음)")
    parser.add_argument("--num_data",   type=int, default=None,
                        help="처리할 문제 수 (-1이면 전체)")
    parser.add_argument("--num_start",  type=int, default=None,
                        help="시작 인덱스")
    parser.add_argument("--output",     type=str, default=None,
                        help="출력 폴더 경로 (기본: output/eval_gen_solo/{timestamp})")
    parser.add_argument("--n_parallel", type=int, default=None,
                        help="동시 처리 문제 수")
    parser.add_argument("--debug",      type=str, default=None,
                        help="디버그용 문제 ID 파일")
    parser.add_argument("--resume_folder", type=str, default=None,
                        help="이전 실행 폴더. results.jsonl의 problem_id는 건너뜀")
    parser.add_argument("--verbose", action="store_true",
                        help="스텝별 full_text, fast/deep critic, 리워드 시뮬레이션을 터미널에 출력")
    args = parser.parse_args()

    global _VERBOSE
    _VERBOSE = args.verbose

    root   = _ROOT_PATH
    gt_cfg  = CONF.get("generate_trajectory", {})
    inf_cfg = CONF.get("inference", {})

    dataset_path = (
        inf_cfg.get("data_path")
        or gt_cfg.get("base_problems")
        or str(root / CONF["data_path"]["deepmath_16k"])
    )

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else (root / "output" / "eval_gen_solo" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir / "run.log")

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    items = load_dataset_file(dataset_path)
    if args.debug:
        debug_path = Path(args.debug) if Path(args.debug).is_absolute() else root / args.debug
        with open(debug_path, encoding="utf-8") as f:
            debug_ids = {line.split()[0] for line in f if line.strip()}
        items = [it for it in items if str(it["id"]) in debug_ids]
        logger.info(f"[debug] {len(debug_ids)}개 ID → 매칭 {len(items)}개")
    else:
        if args.num_start:
            items = items[args.num_start:]
        if args.num_data and args.num_data != -1:
            items = items[:args.num_data]

    if args.resume_folder:
        resume_path = Path(args.resume_folder) / "results.jsonl"
        done_ids: set[str] = set()
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as _rf:
                for _line in _rf:
                    _line = _line.strip()
                    if _line:
                        pid = json.loads(_line).get("problem_id")
                        if pid:
                            done_ids.add(str(pid))
            before = len(items)
            items  = [it for it in items if str(it.get("id", "")) not in done_ids]
            logger.warning(f"[resume] {len(done_ids)}개 완료 ID 제외 → {len(items)}개 처리")

    logger.info(f"문제 수={len(items)}  dataset={dataset_path}")

    rollout_gpus = inf_cfg.get("rollout_gpus") or gt_cfg.get("rollout_gpus", [0])
    n_parallel   = args.n_parallel or inf_cfg.get("batch_per_gpu") or gt_cfg.get("batch_per_gpu", 8) * len(rollout_gpus)

    single_gpu = len(rollout_gpus) == 1
    # GPU 1개: step_manager = generator 공유 / 2개 이상: 첫 번째 GPU가 step_manager 전용
    gen_gpus   = rollout_gpus[1:] if not single_gpu else rollout_gpus
    logger.info(f"GPU 배분: step_manager=cuda:{rollout_gpus[0]}  generator={gen_gpus}  single_gpu={single_gpu}")

    # ── Step Manager 로드 ────────────────────────────────────────────────────
    _sft_ckpt           = CONF["checkpoint"].get("sft_checkpoint", "")
    base_model_id       = _sft_ckpt if _sft_ckpt else CONF["checkpoint"]["base"]
    generators: list    = []
    step_manager_model  = None
    step_manager_tok    = None

    if not _GENERATOR_API and not single_gpu:
        # GPU 2개 이상 → 첫 번째 GPU에 step_manager 별도 로드
        sm_model, sm_tok = load_step_manager(gpu_id=rollout_gpus[0])
        step_manager_model = sm_model
        step_manager_tok   = sm_tok
        logger.info(f"Step Manager 로드 완료 (cuda:{rollout_gpus[0]})")
    else:
        logger.info("rollout_gpus=1 또는 API 모드 → Step Manager 별도 로드 생략")

    # ── Generator 로드 ────────────────────────────────────────────────────────
    if _GENERATOR_API:
        logger.info(f"Generator API 모드: {_GENERATOR_API}")
    elif USE_VLLM:
        if not single_gpu and len(gen_gpus) == 0:
            raise ValueError(
                f"vLLM + step_manager 사용 시 rollout_gpus에 최소 2개 GPU 필요 (현재: {rollout_gpus})"
            )
        _vllm_gpus = gen_gpus if not single_gpu else rollout_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in _vllm_gpus)
        logger.info(f"Generator vLLM 모드: {base_model_id}  gpus={_vllm_gpus}")
        llm, tokenizer = load_generator_vllm(model_path=base_model_id, rollout_gpus=_vllm_gpus)
        generators.append((llm, tokenizer, None))
        logger.info("Generator vLLM 로드 완료")
    else:
        for i, gpu_id in enumerate(gen_gpus):
            # single_gpu: cuda:{gpu_id} / multi_gpu: cuda:{i+1} (cuda:0 is step_manager)
            rel_idx    = i if single_gpu else i + 1
            device_map = {"": f"cuda:{rel_idx}"}
            logger.info(f"Generator HF 로딩: {base_model_id}  device=cuda:{rel_idx}(physical {gpu_id})")
            model, tokenizer = load_generator(model_path=base_model_id, device_map=device_map)
            generators.append((model, tokenizer, next(model.parameters()).device))
        logger.info(f"Generator {len(generators)}개 로드 완료")
        if single_gpu and generators:
            step_manager_model = generators[0][0]
            step_manager_tok   = generators[0][1]
            logger.info("rollout_gpus=1 → step_manager를 generators[0]와 공유")

    # ── 실행 메타 기록 ────────────────────────────────────────────────────────
    run_meta = {
        "timestamp":   ts,
        "dataset":     dataset_path,
        "num_items":   len(items),
        "n_parallel":  n_parallel,
        "max_steps":   TRAJ_MAX_STEPS,
        "max_new_tokens": TRAJ_MAX_NEW_TOKENS,
        "model":       base_model_id if not _GENERATOR_API else _GENERATOR_API,
        "mode":        "api" if _GENERATOR_API else ("vllm" if USE_VLLM else "hf"),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    # ── run.jsonl 설정 ────────────────────────────────────────────────────────
    _run_jsonl_file = open(out_dir / "run.jsonl", "w", encoding="utf-8")
    _run_jsonl_lock = threading.Lock()

    def _run_log(record: dict):
        with _run_jsonl_lock:
            _run_jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            _run_jsonl_file.flush()

    set_run_log(_run_log)

    # ── raw_outputs.jsonl 설정 ────────────────────────────────────────────────
    _raw_file = open(out_dir / "raw_outputs.jsonl", "w", encoding="utf-8")
    _raw_lock = threading.Lock()

    def _raw_save(rec: dict):
        with _raw_lock:
            _raw_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            _raw_file.flush()

    # ── 샘플 문제/정답 미리 출력 ─────────────────────────────────────────────
    if items:
        s = items[0]
        print("\n=== 샘플 (첫 번째 문제) ===")
        print(f"[문제]\n{s.get('problem', '')}\n")
        print(f"[정답]\n{s.get('gold_answer') or s.get('answer', '')}\n")
        print("[Generator 출력] → 첫 결과 수신 시 출력됩니다.\n")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    results_file = open(out_dir / "results.jsonl", "w", encoding="utf-8")
    _first_printed = False

    def _save(rec: dict):
        nonlocal _first_printed
        results_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        results_file.flush()
        if not _first_printed and rec.get("problem_id") == str(items[0].get("id", "?")):
            steps = rec.get("steps", [])
            if steps:
                first = steps[0]
                npa   = first.get("next_pred_action") or first.get("pred_action", "")
                full  = first.get("full_text") or first["text"]
                print(f"\n[Generator 출력 - Step 1]\n{full}\n→ next_action: {npa}\n")
            _first_printed = True

    t_start = time.time()
    try:
        results = evaluate_gen_batch(
            items, generators, n_parallel=n_parallel, save_fn=_save,
            step_manager_model=step_manager_model, step_manager_tok=step_manager_tok,
            raw_step_save_fn=_raw_save,
        )
    finally:
        results_file.close()
        _run_jsonl_file.close()
        _raw_file.close()
        set_run_log(None)

    elapsed_min = (time.time() - t_start) / 60
    n_right     = sum(1 for r in results if r.get("is_right"))
    n_total     = len(results)
    accuracy    = n_right / n_total if n_total else 0.0

    fail_counts: dict[str, int] = {}
    for r in results:
        fr = r.get("fail_reason") or "correct"
        fail_counts[fr] = fail_counts.get(fr, 0) + 1

    avg_steps    = sum(r.get("n_steps", 0) for r in results) / n_total if n_total else 0
    avg_rethinks = sum(r.get("n_rethinks", 0) for r in results) / n_total if n_total else 0

    summary = {
        "timestamp":    ts,
        "dataset":      dataset_path,
        "model":        run_meta["model"],
        "n_total":      n_total,
        "n_correct":    n_right,
        "accuracy":     round(accuracy, 4),
        "fail_reasons": fail_counts,
        "avg_steps":    round(avg_steps, 2),
        "avg_rethinks": round(avg_rethinks, 2),
        "elapsed_min":  round(elapsed_min, 1),
        "max_steps":    TRAJ_MAX_STEPS,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Gen-Solo 평가 결과 ===")
    print(f"정확도: {n_right}/{n_total} = {accuracy:.1%}")
    print(f"실패 원인: {fail_counts}")
    print(f"평균 스텝: {avg_steps:.1f}  평균 rethink: {avg_rethinks:.2f}")
    print(f"소요: {elapsed_min:.1f}분  출력: {out_dir}")
    logger.info(
        f"완료: {n_total}개  accuracy={accuracy:.1%}  "
        f"fail={fail_counts}  elapsed={elapsed_min:.1f}min  out={out_dir}"
    )


if __name__ == "__main__":
    main()
