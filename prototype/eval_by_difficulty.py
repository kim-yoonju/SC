"""
prototype/eval_by_difficulty.py

난이도별 2문제씩 샘플링해 모델 성능 평가.
실행: python eval_by_difficulty.py
"""

import logging
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATASET_PATH, SAVE_DIR, load_generator, solve_problem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

SAMPLES_PER_DIFFICULTY = 2
SEED = 42


def load_problems_with_difficulty(parquet_path: str) -> list[dict]:
    from datasets import load_dataset as hf_load

    ds = hf_load("parquet", data_files=parquet_path, split="train")

    from utils import _extract_problem, _extract_answer, _TRAILING_INSTRUCTION

    problems = []
    for i, ex in enumerate(ds):
        problem = _extract_problem(ex)
        answer  = _extract_answer(ex)
        difficulty = ex.get("difficulty")
        if not problem or difficulty is None:
            continue
        problems.append({
            "problem_id": str(ex.get("problem_id", i)),
            "problem":    problem,
            "answer":     answer,
            "difficulty": difficulty,
        })
    return problems


def sample_by_difficulty(problems: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_diff = defaultdict(list)
    for p in problems:
        by_diff[p["difficulty"]].append(p)

    difficulties = sorted(by_diff.keys())
    logger.info(f"난이도 목록: {difficulties}")
    logger.info(f"난이도별 문제 수: { {d: len(by_diff[d]) for d in difficulties} }")

    sampled = []
    for d in difficulties:
        pool = by_diff[d]
        sampled.extend(rng.sample(pool, min(n, len(pool))))
    return sampled


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(SAVE_DIR, exist_ok=True)
    rollout_path = os.path.join(SAVE_DIR, f"eval_difficulty_{ts}.jsonl")

    model, tokenizer = load_generator(device_map="auto")

    problems = load_problems_with_difficulty(DATASET_PATH)
    sampled  = sample_by_difficulty(problems, SAMPLES_PER_DIFFICULTY, SEED)

    logger.info(f"총 {len(sampled)}개 문제 평가 시작")

    # 난이도별 결과 집계
    results = defaultdict(lambda: {"solved": 0, "total": 0, "patcher_wrong": 0, "steps": []})

    for item in sampled:
        diff = item["difficulty"]
        results[diff]["total"] += 1

        traj = solve_problem(
            model, tokenizer,
            problem=item["problem"],
            answer=item["answer"],
            problem_id=item["problem_id"],
            rollout_path=rollout_path,
            difficulty=item["difficulty"],
        )

        if traj.is_answer:
            results[diff]["solved"] += 1
            results[diff]["steps"].append(len(traj.steps))
        elif traj.patcher_wrong:
            results[diff]["patcher_wrong"] += 1

    # ── 최종 요약 출력 ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'난이도':<10} {'solved':>8} {'total':>8} {'정답률':>8} {'avg_steps':>10} {'patcher_wrong':>14}")
    print(f"{'─'*60}")
    for diff in sorted(results.keys()):
        r = results[diff]
        acc       = r["solved"] / r["total"] if r["total"] else 0
        avg_steps = sum(r["steps"]) / len(r["steps"]) if r["steps"] else 0
        print(
            f"{diff:<10} {r['solved']:>8} {r['total']:>8} "
            f"{acc:>7.0%} {avg_steps:>10.1f} {r['patcher_wrong']:>14}"
        )
    print(f"{'='*60}")
    print(f"저장: {rollout_path}")


if __name__ == "__main__":
    main()
