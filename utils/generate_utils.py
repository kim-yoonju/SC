"""
generate.py
generate_sft_data.py 와 generate_rethink_data.py 가 공통으로 사용하는 유틸리티.
"""

import json
import logging
import re
from pathlib import Path
from typing import List

import torch

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def build_solve_user_msg(problem: str, history_steps: list[str]) -> str:
    """[Problem] / [Previous steps] / Write Step N. 포맷 유저 메시지 생성."""
    lines = [f"[Problem]\n{problem}"]
    if history_steps:
        lines.append("\n[Previous steps]")
        for i, step in enumerate(history_steps, 1):
            lines.append(f"\nStep {i}:\n{step}")
    lines.append(f"\nWrite Step {len(history_steps) + 1}.")
    return "\n".join(lines)


def load_prompts() -> dict[str, str]:
    """action_prompts.jsonl을 로드하고 {{rubric}} 등 변수를 치환해 반환."""
    rubric_path = _PROMPTS_DIR / "action_prompts_rubric.jsonl"
    rubric_lines = []
    with open(rubric_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                rubric_lines.append(
                    f"{entry['id']}. {entry['name']}: [pass/fail — {entry['description']}]"
                )
    rubric_str = "\n".join(rubric_lines)

    prompts: dict[str, str] = {}
    with open(_PROMPTS_DIR / "action_prompts.json", encoding="utf-8") as f:
        for entry in json.load(f):
            prompts[entry["name"]] = entry["content"].replace("{{rubric}}", rubric_str)
    return prompts


# ─────────────────────────────────────────────────────────────────────────────
# API 비용 추적
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_PRICING: dict[str, dict] = {
    "o3-mini":      {"input": 1.10,  "output": 4.40},
    "o3":           {"input": 10.00, "output": 40.00},
    "gpt-4o":       {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "gpt-5.4-mini": {"input": 0.15,  "output": 0.60},
}

def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# 정답 추출
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로딩 및 전처리 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _extract_gsm8k_answer(answer_text: str) -> str:
    """gsm8k 정답 텍스트에서 #### 뒤의 숫자만 추출."""
    m = re.search(r"####\s*(.+)", answer_text)
    return m.group(1).strip().replace(",", "") if m else answer_text.strip()


def _extract_problem(ex: dict) -> str:
    text = ex.get("problem") or ex.get("question") or ""
    if not text and "prompt" in ex:
        text = next((m["content"] for m in ex["prompt"] if m["role"] == "user"), "")
    return re.sub(r"\s*Please reason step by step.*$", "", text, flags=re.I).strip()


def _extract_answer(ex: dict) -> str:
    for k in ("answer", "final_answer", "ground_truth"):
        if ex.get(k):
            v = str(ex[k]).strip()
            return _extract_gsm8k_answer(v) if "####" in v else v
    # parquet 포맷: reward_model dict 안의 ground_truth
    rm = ex.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth"):
        return str(rm["ground_truth"]).strip()
    return ""


def _solve_user(problem: str, history: List[str]) -> str:
    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Steps so far]")
        for i, s in enumerate(history, 1):
            lines.append(f"Step {i}: {s}")
    return "\n".join(lines)


def _correct_user(problem: str, history: List[str]) -> str:
    lines = [f"[Problem]\n{problem}"]
    if history:
        steps = history[-10:]
        offset = len(history) - len(steps)
        lines.append("\n[Steps so far]")
        for i, s in enumerate(steps, offset + 1):
            lines.append(f"Step {i}: {s}")
        lines.append(f"\nStep {len(history)} above contains an error.")
    return "\n".join(lines)


def _load_jsonl_eval(p) -> list[dict]:
    """평가/생성용 JSONL → [{id, problem, answer}, ...] 변환."""
    items = []
    with open(p) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            problem = d.get("problem") or d.get("question", "")
            answer  = d.get("answer") or d.get("gold_answer", "")
            if "####" in str(answer):
                answer = _extract_gsm8k_answer(str(answer))
            items.append({"id": str(d.get("id", i)), "problem": problem, "answer": str(answer)})
    return items


def _load_parquet_eval(p) -> list[dict]:
    """평가/생성용 Parquet (math500/deepmath 계열) → [{id, problem, answer}, ...] 변환."""
    import pandas as pd

    df = pd.read_parquet(p)
    items = []

    # 직접 problem/answer 컬럼이 있는 포맷 (base_multi_reasoning 계열 등)
    if "problem" in df.columns and "answer" in df.columns:
        for i, (_, row) in enumerate(df.iterrows()):
            problem = str(row.get("problem", "")).strip()
            answer  = str(row.get("answer", "")).strip()
            if "####" in answer:
                answer = _extract_gsm8k_answer(answer)
            item_id = str(row.get("id", i))
            if not problem:
                continue
            items.append({"id": item_id, "problem": problem, "answer": answer})
        return items

    for i, (_, row) in enumerate(df.iterrows()):
        prompt = row.get("prompt")
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except json.JSONDecodeError:
                prompt = [{"role": "user", "content": prompt}]

        problem = ""
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    text = msg.get("content", "")
                    text = re.sub(r"\s*Please reason step by step,.*$", "", text, flags=re.DOTALL).strip()
                    problem = text
                    break
        elif isinstance(prompt, dict) and prompt.get("role") == "user":
            problem = re.sub(r"\s*Please reason step by step,.*$", prompt.get("content", ""), flags=re.DOTALL).strip()

        if "final_answer" in df.columns:
            answer = str(row.get("final_answer", ""))
        else:
            rm = row.get("reward_model", {})
            if hasattr(rm, "item"):
                rm = rm.item()
            if isinstance(rm, str):
                try:
                    rm = json.loads(rm)
                except json.JSONDecodeError:
                    rm = {}
            answer = str(rm.get("ground_truth", "")) if isinstance(rm, dict) else ""

        extra = row.get("extra_info", {})
        if hasattr(extra, "item"):
            extra = extra.item()
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError:
                extra = {}
        item_id = str(extra.get("index", i)) if isinstance(extra, dict) else str(i)

        if not problem:
            continue
        items.append({"id": item_id, "problem": problem, "answer": answer})
    return items


def load_dataset_file(path: str) -> list[dict]:
    """JSONL / Parquet 파일을 [{id, problem, answer}, ...] 형태로 로드.

    지원 포맷:
      - JSONL: problem + answer/gold_answer 필드
      - Parquet (math500/math7500): reward_model.ground_truth + prompt
      - Parquet (deepmath_*): final_answer + prompt
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"데이터셋 없음: {path}")
    if p.suffix == ".jsonl":
        return _load_jsonl_eval(p)
    elif p.suffix == ".parquet":
        return _load_parquet_eval(p)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {p.suffix}")


def extract_step_content(step: dict) -> tuple[str, str]:
    """step dict에서 (action, clean_text) 추출.
    구 포맷(text 필드)과 신 포맷(content 필드) 모두 처리.
    """
    action = step["action"]
    if "content" in step and step["content"] is not None:
        text = step["content"].strip()
    else:
        text = step.get("text", "")
        text = text.replace("<|end|>", "").replace("<|im_end|>", "").strip()
        for act in ["solve", "correct", "rethink", "end", "review"]:
            m = re.search(rf"<{act}>(.*?)</{act}>", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
                break
    return action, text


def build_target_text(action: str, text: str) -> str:
    """텍스트 + 액션 특수 토큰을 타겟 문자열로 합침.
    예: "reasoning...<|solve|>"
    """
    return f"{text}<|{action}|>"


# ─────────────────────────────────────────────────────────────────────────────
# PRM 판정 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

# Atomicity: PRM 평가에서는 fail 조건으로 쓰지 않음
def _prm_is_fail(votes: dict) -> bool:
    fails = sum(1 for n, v in votes.items() if v == "incorrect" and n != "Atomicity")
    return fails >= 1


def _extract_verdicts_from_text(sc_text: str) -> tuple[int, int, int]:
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
