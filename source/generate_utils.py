"""
generate.py
generate_sft_data.py 와 generate_rethink_data.py 가 공통으로 사용하는 유틸리티.
"""

import json
import logging
import re
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

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

def extract_pred_answer(response: str, extract_boxed_fn, normalize_fn) -> str | None:
    """모델 응답에서 \\boxed{} 안의 정답을 추출해 \\boxed{answer} 형태로 반환."""
    raw = extract_boxed_fn(response)
    if raw is None:
        return None
    normalized = normalize_fn(raw)
    content = normalized if normalized else raw.strip()
    return f"\\boxed{{{content}}}"


# ─────────────────────────────────────────────────────────────────────────────
# 스텝 파싱 공통
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r'([.!?\]$}]|\\newline)\s*$')

def merge_incomplete(parts: list[str]) -> list[str]:
    """문장이 완결되지 않은 단락을 다음 단락과 병합."""
    merged, buf = [], ""
    for part in parts:
        buf = (buf + "\n" + part) if buf else part
        if _SENTENCE_END.search(buf):
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────────────────────

W = 72
SEP  = "─" * W
SEP2 = "━" * W

def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=W, initial_indent=prefix,
                         subsequent_indent=prefix,
                         break_long_words=False, break_on_hyphens=False)

def print_sample(result: dict, extract_boxed_fn) -> None:
    """결과 샘플 1개를 터미널에 출력. solve/rethink 타입 모두 지원."""
    print()
    print(SEP2)
    print("  SAMPLE OUTPUT")
    print(SEP2)

    print(f"\nPROBLEM")
    print(SEP)
    print(_wrap(result["problem"]))

    print(f"\nGOLD ANSWER")
    print(SEP)
    print(f"    {result['gold_answer']}")

    correct_mark = "✓" if result.get("is_right") else "✗"
    pred_raw = extract_boxed_fn(result['pred_answer']) if result.get('pred_answer') else None
    print(f"\nPRED ANSWER  [{correct_mark}]")
    print(SEP)
    print(f"    {pred_raw or '(없음)'}")

    steps = result.get("steps", [])
    print(f"\nSTEPS  ({len(steps)} steps)")
    print(SEP)
    for s in steps:
        step_type = s.get("type", "solve")
        action    = s.get("next_gold_action", "")
        label     = f"[{s['step_idx']}] ({step_type}) → {action}"
        print(f"\n  {label}")
        print(_wrap(s["text"], indent=4))

    print()
    print(SEP2)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 병렬 실행 + 즉시 저장 공통 루프
# ─────────────────────────────────────────────────────────────────────────────

def run_parallel(
    items: list[dict],
    solve_fn,          # solve_fn(item) -> dict | None
    output_path: str,
    model: str,
    workers: int,
    log_interval: int = 50,
    append: bool = False,
) -> list[dict]:
    """
    items를 병렬로 처리하고, 결과를 output_path에 즉시 저장한다.
    완료된 결과 리스트를 반환한다.
    append=True면 기존 파일에 이어 쓴다.
    """
    results    = []
    write_lock = threading.Lock()
    total_in   = total_out = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_path, "a" if append else "w", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(solve_fn, item): item for item in items}
            done = 0
            for fut in as_completed(futures):
                done += 1
                result = fut.result()
                if result is not None:
                    results.append(result)
                    u = result.get("usage", {})
                    with write_lock:
                        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_file.flush()
                        total_in  += u.get("input_tokens", 0)
                        total_out += u.get("output_tokens", 0)
                if done % log_interval == 0 or done == len(items):
                    cost = calc_cost(model, total_in, total_out)
                    logger.info(
                        f"  진행: {done}/{len(items)}  성공: {len(results)}  "
                        f"누적 비용: ${cost:.4f}"
                    )
    finally:
        out_file.close()

    return results


def print_cost_summary(results: list[dict], model: str) -> None:
    """API 비용 요약 출력."""
    total_in  = sum(r.get("usage", {}).get("input_tokens",  0) for r in results)
    total_out = sum(r.get("usage", {}).get("output_tokens", 0) for r in results)
    total_cost = calc_cost(model, total_in, total_out)

    print()
    print(SEP2)
    print("  API 비용 요약")
    print(SEP2)
    print(f"  모델         : {model}")
    print(f"  입력 토큰    : {total_in:,}")
    print(f"  출력 토큰    : {total_out:,}")
    print(f"  총 비용      : ${total_cost:.4f}")
    if results:
        print(f"  문제당 평균  : ${total_cost / len(results):.5f}")
    print(SEP2)

    cost_sorted = sorted(
        [r for r in results if r.get("usage")],
        key=lambda r: r["usage"]["cost_usd"], reverse=True
    )
    if cost_sorted:
        print("\n  [비용 상위 5개 문제]")
        for r in cost_sorted[:5]:
            u   = r["usage"]
            pid = r.get("problem_id", "?")
            print(f"  id={pid:8}  in={u['input_tokens']:5d}  out={u['output_tokens']:5d}  ${u['cost_usd']:.5f}")
    print()


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
