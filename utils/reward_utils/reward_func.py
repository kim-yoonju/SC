"""
GRPO reward function for SC (Self-Correction) training.

Reward design:
  - 각 스텝: self-PRM score
      Fail rubrics: none  → 1.0 (correct)
      Fail rubrics: <tokens> → 0.0 (incorrect)
  - 마지막 스텝 (<|end|>): self-PRM score + outcome reward (정답 여부 +1)
  - 총 reward = sum(step_rewards)  [max = num_steps + 1]
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "source"))
from utils_math import check_solved

_ACTION_RE = re.compile(r'(<\|solve\|>|<\|rethink\|>|<\|end\|>)')
_FAIL_RUBRICS_RE = re.compile(
    r'Fail rubrics:\s*\n(.*?)(?=\n\nNext action:|\Z)', re.DOTALL
)


def _parse_steps(response: str) -> list[dict]:
    parts = _ACTION_RE.split(response)
    steps = []
    i = 0
    while i < len(parts) - 1:
        steps.append({"text": parts[i], "action": parts[i + 1]})
        i += 2
    # 마지막 action token 이후 남은 텍스트가 있으면 마지막 step에 포함
    if len(parts) % 2 == 0 and parts[-1].strip():
        if steps:
            steps[-1]["text"] += parts[-1]
    return steps


def _self_prm(step_text: str) -> float:
    """모델이 직접 출력한 Fail rubrics 섹션으로 step 정오 판단."""
    m = _FAIL_RUBRICS_RE.search(step_text)
    if not m:
        return 0.0
    return 1.0 if m.group(1).strip().lower() == "none" else 0.0


def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> float:
    """
    verl custom reward function.

    Args:
        data_source:  parquet의 data_source 컬럼 (사용 안 함)
        solution_str: 모델이 생성한 응답 전체
        ground_truth: 정답 문자열 (parquet의 reward_model.ground_truth)
        extra_info:   {"problem_id": ..., "gold_answer": ...}
    Returns:
        float reward
    """
    steps = _parse_steps(solution_str)
    if not steps:
        return 0.0

    step_rewards = [_self_prm(s["text"]) for s in steps]

    # 마지막 스텝에 outcome reward 추가
    outcome = 1.0 if check_solved(solution_str, ground_truth) else 0.0
    step_rewards[-1] += outcome

    return float(sum(step_rewards))
