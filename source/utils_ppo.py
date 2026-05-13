"""utils_ppo.py — PPO 학습 전용 데이터 구조 및 상수."""

import json
import logging
import pathlib as _pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from utils import CONF
from utils_math import has_boxed, extract_boxed

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PPO 하이퍼파라미터
# ─────────────────────────────────────────────────────────────────────────────

_ppo                = CONF["ppo"]
PPO_LR              = _ppo["lr"]
PPO_CLIP_EPS        = _ppo["clip_eps"]
PPO_MAX_GRAD_NORM   = _ppo["max_grad_norm"]
KL_COEF             = _ppo["kl_coef"]
GAMMA               = _ppo["gamma"]
MAX_SEQ_LEN         = _ppo["max_seq_len"]
LENGTH_PENALTY_COEF = _ppo["length_penalty_coef"]

_ROOT_PATH   = _pathlib.Path(__file__).resolve().parent.parent
DATASET_PATH = str(_ROOT_PATH / CONF["data_path"]["deepmath_16k"])
SAVE_DIR     = str(_ROOT_PATH / CONF["output_path"]["ppo"])

# ─────────────────────────────────────────────────────────────────────────────
# 상태 상수
# ─────────────────────────────────────────────────────────────────────────────

SOLVE       = "solve"
CORRECT_GEN = "correct_gen"
CORRECT_PAT = "correct_pat"
END_MAX     = "end_max"
END_ANSWER  = "end_answer"

ACTIVE_STATES   = {SOLVE, CORRECT_GEN, CORRECT_PAT}
TERMINAL_STATES = {END_MAX, END_ANSWER}

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step_idx: int
    state: str
    action: str
    text: str
    final_reward: float
    llm_reward: float
    format_reward: float
    predicted_next_action: str
    gold_next_action: str
    input_ids: torch.Tensor
    response_ids: torch.Tensor
    log_probs_old: torch.Tensor
    use_patcher: bool

@dataclass
class Trajectory:
    problem_id: str
    problem: str
    answer: str
    difficulty: Optional[float] = None
    steps: List[StepRecord] = field(default_factory=list)
    have_boxed: bool = False
    is_answer: bool = False
    patcher_wrong: bool = False
    end_state: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# Rollout 파일 I/O
# ─────────────────────────────────────────────────────────────────────────────

def create_rollout_file(path: str):
    """빈 rollout JSONL 파일을 생성 (디렉토리 자동 생성 포함)."""
    _pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    open(path, "w").close()


def save_trajectory(traj: Trajectory, path: str):
    """Trajectory를 JSONL에 append 저장."""
    last_boxed_text = next(
        (s.text for s in reversed(traj.steps) if has_boxed(s.text)), ""
    )
    pred_answer = extract_boxed(last_boxed_text) if last_boxed_text else None
    record = {
        "problem_id":    traj.problem_id,
        "problem":       traj.problem,
        "gold_answer":   traj.answer,
        "pred_answer":   pred_answer,
        "have_boxed":    traj.have_boxed,
        "is_right":      traj.is_answer,
        "patcher_wrong": traj.patcher_wrong,
        "end_state":     traj.end_state,
        "steps": [
            {
                "step_idx":                s.step_idx,
                "state":                   s.state,
                "action":                  s.action,
                "text":                    s.text,
                "final_reward":            s.final_reward,
                "llm_reward":              s.llm_reward,
                "format_reward":           s.format_reward,
                "predicted_next_action":   s.predicted_next_action,
                "gold_next_action":        s.gold_next_action,
                "use_patcher":             s.use_patcher,
            }
            for s in traj.steps
        ],
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
