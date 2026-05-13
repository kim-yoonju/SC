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
    TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END,
    load_generator, load_generator_vllm, build_chat_prompt,
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


# ── ProblemState ──────────────────────────────────────────────────────────────
@dataclass
class ProblemState:
    item:                      dict
    history:                   list = field(default_factory=list)   # 수락된 스텝
    all_steps:                 list = field(default_factory=list)   # 전체 스텝 (오류 포함)
    is_rethink:                bool = False
    step_rethink_tried:        bool = False
    last_wrong_rubric_details: dict = field(default_factory=dict)
    last_wrong_step_text:      str  = ""
    n_rethinks:                int  = 0
    done:                      bool = False
    pending_step:              dict = field(default_factory=dict)
    result:                    dict = field(default_factory=dict)


# ── Text helpers ──────────────────────────────────────────────────────────────
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
    """Gen의 deep critique에서 rethink 설명 구성."""
    parts = []
    if state.last_wrong_rubric_details:
        parts.append("Failed evaluation criteria:")
        for rubric_name, reasoning in state.last_wrong_rubric_details.items():
            if reasoning and reasoning.strip():
                parts.append(f"- {rubric_name}: {reasoning.strip()}")
            else:
                parts.append(f"- {rubric_name}")
    return "\n".join(parts) if parts else "the previous step contained an error"


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

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=TRAJ_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=_eos_ids,
        )

    resp_all  = out[:, input_len:]
    _stop_ids = {tokenizer.pad_token_id, im_end_id, *_action_ids}

    for j, state in enumerate(states):
        resp = resp_all[j]
        trim = resp.shape[0]
        for pos, tid in enumerate(resp.tolist()):
            if tid in _stop_ids:
                trim = pos
                break
        full_text = tokenizer.decode(resp[:trim], skip_special_tokens=False).strip()

        sc_idx = full_text.find("\nSelf-correction:")
        if sc_idx != -1:
            step_text       = full_text[:sc_idx].strip()
            self_check_text = full_text[sc_idx:]
        else:
            step_text       = full_text.strip()
            self_check_text = ""

        correct_count, incorrect_count, _ = _extract_verdicts_from_text(self_check_text)
        if incorrect_count > correct_count:
            pred_action = TOKEN_CORRECT
        elif has_boxed(step_text):
            pred_action = TOKEN_END
        else:
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
            "text":        step_text,
            "full_text":   full_text,
            "pred_action": pred_action,
            "is_error":    is_error,
            "role":        "rethink" if state.is_rethink else "gen",
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

    params  = SamplingParams(max_tokens=TRAJ_MAX_NEW_TOKENS, temperature=0, stop_token_ids=_action_ids)
    outputs = llm.generate(prompts, params, use_tqdm=False)

    _TOOL_CALL_MARKERS    = ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")
    _TEMPLATE_PLACEHOLDERS = ("[Your one reasoning step", "[Your one corrected reasoning step")

    for state, output in zip(states, outputs):
        full_text   = output.outputs[0].text.strip()
        stop_reason = output.outputs[0].stop_reason

        sc_idx = full_text.find("\nSelf-correction:")
        if sc_idx != -1:
            step_text       = full_text[:sc_idx].strip()
            self_check_text = full_text[sc_idx:]
        else:
            step_text       = full_text.strip()
            self_check_text = ""

        # stop_reason이 int면 action token ID
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

        is_error = (
            any(m in full_text for m in _TOOL_CALL_MARKERS) or
            any(step_text.strip().startswith(p) for p in _TEMPLATE_PLACEHOLDERS)
        )

        logger.info(
            f"[Generator vLLM] id={state.item.get('id')} "
            f"step={len(state.history)+1} rethink={state.is_rethink} "
            f"pred_action={pred_action} is_error={is_error}"
        )

        state.pending_step = {
            "text":        step_text,
            "full_text":   full_text,
            "pred_action": pred_action,
            "is_error":    is_error,
            "role":        "rethink" if state.is_rethink else "gen",
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

        sc_idx          = text.find("\nSelf-correction:")
        step_text       = text[:sc_idx].strip() if sc_idx != -1 else text.strip()
        self_check_text = text[sc_idx:] if sc_idx != -1 else ""

        correct_count, incorrect_count, _ = _extract_verdicts_from_text(self_check_text)
        if incorrect_count > correct_count:
            pred_action = TOKEN_CORRECT
        elif has_boxed(step_text):
            pred_action = TOKEN_END
        else:
            pred_action = TOKEN_SOLVE

        state.pending_step = {
            "text":        step_text,
            "full_text":   text,
            "pred_action": pred_action,
            "is_error":    not text.strip(),
            "role":        "rethink" if state.is_rethink else "gen",
        }

    with ThreadPoolExecutor(max_workers=len(states)) as ex:
        list(ex.map(_one, states))


# ── Process Gen Result ────────────────────────────────────────────────────────
def _process_gen_result(state: ProblemState, save_fn=None) -> None:
    """Gen의 pred_action으로 state 업데이트. 완료 시 state.done = True."""
    step        = state.pending_step
    pred_action = step["pred_action"]
    problem_id  = state.item.get("id", "?")
    gold_answer = state.item.get("gold_answer") or state.item["answer"]

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
                {"role": s["role"], "pred_action": s["pred_action"], "text": s["text"][:400]}
                for s in state.all_steps
            ],
        }
        state.result = result
        if save_fn:
            save_fn(result)
        state.done = True

    def _apply_wrong():
        """Gen이 이 스텝을 틀렸다고 판단 → rethink 1회 또는 종료."""
        step["is_error"] = True
        state.all_steps.append(step)

        if TRAJ_MAX_STEPS and len(state.all_steps) >= TRAJ_MAX_STEPS:
            _finish(False, fail_reason="max_steps")
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
        else:
            # rethink 후에도 gen이 틀렸다고 판단 → 종료
            logger.info(f"[id={problem_id}] rethink 후에도 TOKEN_CORRECT → rethink_fail")
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
        pred_answer = extract_boxed(step.get("full_text") or step["text"])
        is_right    = check_solved(step.get("full_text") or step["text"], gold_answer)
        _finish(is_right, fail_reason=None if is_right else "wrong_answer", pred_answer=pred_answer)

    else:  # TOKEN_SOLVE
        state.all_steps.append(step)
        state.history.append(step)
        state.is_rethink         = False
        state.step_rethink_tried = False

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
            _process_gen_result(state, _ts_save)

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
    args = parser.parse_args()

    root   = _ROOT_PATH
    gt_cfg = CONF.get("generate_trajectory", {})

    dataset_path = (
        gt_cfg.get("base_problems")
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
        num_start = args.num_start if args.num_start is not None else gt_cfg.get("num_start", 0)
        num_data  = args.num_data  if args.num_data  is not None else gt_cfg.get("num_data", 1)
        if num_data == -1:
            items = items[num_start:]
        else:
            items = items[num_start: num_start + num_data]

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

    rollout_gpus = gt_cfg.get("rollout_gpus", [0])
    n_parallel   = args.n_parallel or gt_cfg.get("batch_per_gpu", 8) * len(rollout_gpus)

    # ── Generator 로드 ────────────────────────────────────────────────────────
    base_model_id = CONF["checkpoint"]["base"]
    generators: list = []

    if _GENERATOR_API:
        logger.info(f"Generator API 모드: {_GENERATOR_API}")
    elif USE_VLLM:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in rollout_gpus)
        logger.info(f"Generator vLLM 모드: {base_model_id}  gpus={rollout_gpus}")
        llm, tokenizer = load_generator_vllm(model_path=base_model_id, rollout_gpus=rollout_gpus)
        generators.append((llm, tokenizer, None))
        logger.info("Generator vLLM 로드 완료")
    else:
        for gpu_id in rollout_gpus:
            device_map = {"": f"cuda:{gpu_id}"}
            logger.info(f"Generator HF 로딩: {base_model_id}  device={device_map}")
            model, tokenizer = load_generator(model_path=base_model_id, device_map=device_map)
            generators.append((model, tokenizer, next(model.parameters()).device))
        logger.info(f"Generator {len(generators)}개 로드 완료")

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

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    results_file = open(out_dir / "results.jsonl", "w", encoding="utf-8")

    def _save(rec: dict):
        results_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        results_file.flush()

    t_start = time.time()
    try:
        results = evaluate_gen_batch(items, generators, n_parallel=n_parallel, save_fn=_save)
    finally:
        results_file.close()
        _run_jsonl_file.close()
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
