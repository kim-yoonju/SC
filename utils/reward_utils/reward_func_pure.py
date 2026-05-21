"""
Standard GRPO reward function (pure baseline).

단순 outcome reward: 정답이면 1.0, 틀리면 0.0.
NaiveRewardManager가 마지막 토큰에 배치.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "source"))
from utils_math import check_solved


def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> float:
    problem = (extra_info or {}).get("problem", "")
    return 1.0 if check_solved(solution_str, ground_truth, problem=problem) else 0.0
