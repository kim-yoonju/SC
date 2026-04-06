"""
generate_rethink_data.py
PATCHER API를 활용해 SFT 학습용 rethink 추론 데이터 생성

스텝 형식: Step N (solve): 또는 Step N (rethink):
  - 앞 스텝이 solve이고 다음 스텝이 rethink이면 → next_gold_action = <|rethink|>
  - 그 외 (다음 스텝이 solve이면)              → next_gold_action = <|solve|>
  - 마지막 스텝                               → next_gold_action = <|end|>

출력 형식 (JSONL):
  {
    "problem_id":  str,
    "problem":     str,
    "pred_answer": str | null,
    "gold_answer": str,
    "is_right":    bool,
    "steps":       [{"step_idx": int, "type": "solve"|"rethink", "text": str, "next_gold_action": str}, ...],
    "usage":       {"input_tokens": int, "output_tokens": int, "cost_usd": float}
  }
"""

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    CONF, PATCHER, PATCHER_MAX_NEW_TOKENS, SYSTEM_RETHINK_API_SFT,
    _gpt, _normalize_latex, extract_boxed, check_solved, load_dataset_file,
)
from generate import (
    calc_cost, extract_pred_answer, merge_incomplete,
    print_sample, run_parallel, print_cost_summary,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 스텝 파싱 (Step N (solve/rethink): 형식)
# ─────────────────────────────────────────────────────────────────────────────

_STEP_RE = re.compile(
    r"Step\s+\d+\s*\(\s*(solve|rethink)\s*\)\s*:\s*",
    re.IGNORECASE,
)


def parse_steps(text: str) -> list[dict]:
    """
    Step N (solve): / Step N (rethink): 헤더 기준 분리.
    헤더 없으면 빈 줄 단락 분리 + 불완전 단락 병합 (모두 solve 타입으로).

    next_gold_action 규칙:
      - solve 스텝 뒤에 rethink가 오면 → <|rethink|>
      - 그 외 (rethink 뒤 또는 마지막이 아닌 solve 뒤 solve) → <|solve|>
      - 마지막 스텝 → <|end|>
    """
    matches = list(_STEP_RE.finditer(text))

    if matches:
        parts = []
        for i, m in enumerate(matches):
            start = m.end()
            end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            raw_type = m.group(1).lower()
            step_type = "rethink" if raw_type == "rethink" else "solve"
            content = text[start:end].strip()
            if content:
                parts.append((step_type, content))
    else:
        # 폴백: 빈 줄 단락 분리
        raw   = [p.strip() for p in text.split("\n\n") if p.strip()]
        texts = merge_incomplete(raw)
        parts = [("solve", t) for t in texts]

    if not parts:
        return []

    steps = []
    for i, (step_type, content) in enumerate(parts):
        steps.append({
            "step_idx": i,
            "type":     step_type,
            "text":     content,
        })

    # next_gold_action 부여
    last = len(steps) - 1
    for i, s in enumerate(steps):
        if i == last:
            s["next_gold_action"] = "<|end|>"
        elif steps[i + 1]["type"] == "rethink":
            s["next_gold_action"] = "<|rethink|>"
        else:
            s["next_gold_action"] = "<|solve|>"

    return steps


# ─────────────────────────────────────────────────────────────────────────────
# 단일 문제 풀이
# ─────────────────────────────────────────────────────────────────────────────

def solve_problem(item: dict) -> dict | None:
    problem     = item["problem"]
    gold_answer = item["answer"]
    problem_id  = item.get("id")

    messages = [
        {"role": "system", "content": SYSTEM_RETHINK_API_SFT},
        {"role": "user",   "content": problem},
    ]
    usage_out = []
    try:
        response = _gpt(PATCHER, messages,
                        max_completion_tokens=PATCHER_MAX_NEW_TOKENS,
                        usage_out=usage_out)
    except Exception as e:
        logger.warning(f"[solve] API 호출 실패 (id={problem_id or '?'}): {e}")
        return None

    if not response:
        return None

    usage       = usage_out[0] if usage_out else {"input_tokens": 0, "output_tokens": 0}
    cost        = calc_cost(PATCHER, usage["input_tokens"], usage["output_tokens"])
    steps       = parse_steps(response)
    pred_answer = extract_pred_answer(response, extract_boxed, _normalize_latex)
    is_right    = check_solved(response, gold_answer) if pred_answer else False

    result = {
        "problem":      problem,
        "pred_answer":  pred_answer,
        "gold_answer":  gold_answer,
        "is_right":     is_right,
        "steps":        steps,
        "usage": {
            "input_tokens":  usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cost_usd":      round(cost, 6),
        },
    }
    if problem_id is not None:
        result = {"problem_id": str(problem_id)} | result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rethink SFT 데이터 생성 (solve/rethink step)")
    parser.add_argument("--num_data", type=int, default=None)
    parser.add_argument("--output",   type=str, default=None)
    parser.add_argument("--workers",  type=int, default=8)
    parser.add_argument("--offset",   type=int, default=0)
    args = parser.parse_args()

    num_data = args.num_data if args.num_data is not None else CONF["sft"]["num_data"]
    root     = Path(__file__).resolve().parent.parent
    dataset_path = str(root / CONF["data_path"]["deepmath_16k"])

    if args.output:
        output_path = args.output
    else:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root / "output" / "sft_data_api"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"rethink_api_{ts}.jsonl")

    logger.info(f"데이터셋: {dataset_path}")
    logger.info(f"샘플 수: {num_data}  offset: {args.offset}  모델: {PATCHER}")
    logger.info(f"출력: {output_path}")

    items = load_dataset_file(dataset_path)
    items = items[args.offset: args.offset + num_data]
    logger.info(f"로드된 문제 수: {len(items)}")

    results = run_parallel(items, solve_problem, output_path,
                           model=PATCHER, workers=args.workers)

    n_right = sum(1 for r in results if r.get("is_right"))
    logger.info(f"저장 완료: {output_path}")
    logger.info(f"  총 {len(results)}개  정답: {n_right}/{len(results)} "
                f"({n_right/max(len(results),1)*100:.1f}%)")

    print_cost_summary(results, PATCHER)

    if results:
        print_sample(results[0], extract_boxed)


if __name__ == "__main__":
    main()
