"""
SFT 전용 유틸리티
- utils.py에서 SFT에 필요한 것만 추출 (torch/transformers만 의존)
- 모델 input/output 빌더 포함
"""

import json
import pathlib as _pathlib
from functools import lru_cache

import torch
import yaml
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = _pathlib.Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    if config_path is None:
        config_path = _ROOT / "configs" / "config.yaml"
    config_path = _pathlib.Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config 파일을 찾을 수 없습니다: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"config 파일이 비어 있습니다: {config_path}")
    return cfg


CONF = load_config()

# ─────────────────────────────────────────────────────────────────────────────
# 특수 토큰
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_SOLVE   = CONF["model"]["token_solve"]
TOKEN_RETHINK = CONF["model"]["token_rethink"]
TOKEN_END     = CONF["model"]["token_end"]
ACTION_TOKENS = [TOKEN_SOLVE, TOKEN_RETHINK, TOKEN_END]
SPECIAL_TOKENS = CONF["model"]["special_tokens"]

# rubric name -> token 매핑 (fast/deep critique 출력에 사용)
RUBRIC_TOKENS: dict[str, str] = {
    t[2:-2].replace("_", " ").title(): t
    for t in SPECIAL_TOKENS
    if t not in ACTION_TOKENS
}
# title() 복원이 안 되는 케이스 보정
_RUBRIC_NAME_OVERRIDES = {
    "Abstract And Linear Algebra Operations": "Abstract and Linear Algebra Operations",
    "Function And Limit Analysis":            "Function and Limit Analysis",
    "Counting And Probability":               "Counting and Probability",
    "Number Theoretic Reasoning":             "Number Theoretic Reasoning",
    "Logical And Discrete Reasoning":         "Logical and Discrete Reasoning",
    "Progress And Non-Repetition":            "Progress and Non-Repetition",
}
RUBRIC_TOKENS = {
    _RUBRIC_NAME_OVERRIDES.get(k, k): v
    for k, v in RUBRIC_TOKENS.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# 토크나이저
# ─────────────────────────────────────────────────────────────────────────────

def setup_tokenizer(model_id: str, cache_dir: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tokenizer


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def collate_fn(batch, pad_token_id: int) -> dict:
    """가변 길이 시퀀스를 패딩해 배치로 묶는다."""
    input_ids_list, labels_list = zip(*batch)
    max_len = max(x.size(0) for x in input_ids_list)
    padded_input   = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_labels  = torch.full((len(batch), max_len), -100,         dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len,                dtype=torch.long)
    for i, (inp, lbl) in enumerate(zip(input_ids_list, labels_list)):
        seq_len = inp.size(0)
        padded_input[i, :seq_len]   = inp
        padded_labels[i, :seq_len]  = lbl
        attention_mask[i, :seq_len] = 1
    return {"input_ids": padded_input, "attention_mask": attention_mask, "labels": padded_labels}

# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트 로드
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    path = _ROOT / "prompts" / "action_prompts.json"
    with open(path, encoding="utf-8") as f:
        return {d["name"]: d["content"] for d in json.load(f)}


@lru_cache(maxsize=1)
def _load_rubric_str() -> str:
    path = _ROOT / "prompts" / "prm_rubric_v6.1.jsonl"
    with open(path, encoding="utf-8") as f:
        rubrics = [json.loads(l) for l in f if l.strip()]
    return "\n".join(f"{i}. {r['name']}: [correct/incorrect — {r['criterion']}]"
                     for i, r in enumerate(rubrics, 1))


def get_system_prompts() -> tuple[str, str]:
    """(SYSTEM_SOLVE, SYSTEM_RETHINK) 반환."""
    prompts    = _load_prompts()
    rubric_str = _load_rubric_str()
    system_solve   = prompts["gen_solve_R"].replace("{{rubric}}", rubric_str)
    system_rethink = prompts["gen_rethink_R"].replace("{{rubric}}", rubric_str)
    return system_solve, system_rethink

# ─────────────────────────────────────────────────────────────────────────────
# 모델 Input / Target 빌더
# ─────────────────────────────────────────────────────────────────────────────

def _history_text(steps: list[dict], up_to: int) -> list[str]:
    """steps[:up_to]에서 history용 텍스트 추출 (does 요약 우선, 없으면 inference 앞 300자)."""
    result = []
    for s in steps[:up_to]:
        text = s.get("does") or (s.get("inference") or "")[:300]
        result.append(text)
    return result


def _error_explanation(steps: list[dict], rethink_idx: int) -> str:
    """rethink 스텝 직전 wrong step에서 오류 설명 추출."""
    for i in range(rethink_idx - 1, -1, -1):
        s = steps[i]
        if s.get("is_error"):
            parts = []
            does = s.get("does")
            if does:
                parts.append(does)
            summary = s.get("prm_critique_summary") or s.get("gen_critique_summary")
            if summary:
                parts.append(summary)
            return "\n".join(parts) if parts else "the previous step contained an error"
    return "the previous step contained an error"


def build_messages(problem: str, steps: list[dict], k: int) -> tuple[str, str]:
    """k번째 스텝의 (system, user) 메시지 문자열 반환 (tokenizer 불필요)."""
    system_solve, system_rethink = get_system_prompts()
    history = _history_text(steps, k)
    source  = steps[k].get("source", "gen")

    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Previous steps]")
        for i, h in enumerate(history, 1):
            lines.append(f"Step {i}: {h}")
    lines.append(f"\nWrite Step {k + 1}.")
    user_msg = "\n".join(lines)

    if source == "rethink":
        err_exp = _error_explanation(steps, k)
        system  = system_rethink.replace("{{error_explanation}}", err_exp)
    else:
        system = system_solve

    return system, user_msg


def build_input(problem: str, steps: list[dict], k: int, tokenizer) -> str:
    """generate_trajectory 등 외부 호환용 — chat prompt 문자열 반환."""
    system, user_msg = build_messages(problem, steps, k)
    return build_chat_prompt(tokenizer, system, user_msg)


def build_target(step: dict) -> str:
    """
    스텝에 대한 target 텍스트 생성 (loss 계산 영역).

    [math step] + Fast critic + Deep critic + Summary + [wrong rubric tokens] + [next_action]
    """
    parts = []

    # 1. math step (inference = 순수 풀이만)
    parts.append(step.get("inference") or "")

    # 2. fast critic (prm 기준)
    fast = step.get("prm_fast_critique") or {}
    if fast:
        parts.append("\n\nFast critic:")
        for rubric, data in fast.items():
            verdict  = data.get("verdict", "pass")
            critique = data.get("critique") or ""
            line = f"  {rubric}: {verdict}"
            if verdict.lower() == "fail" and critique:
                line += f" — {critique}"
            parts.append(f"\n{line}")

    # 3. deep critic (non-null verdict만)
    deep = step.get("prm_deep_critique") or []
    deep_items = [d for d in deep if d.get("verdict") is not None]
    if deep_items:
        parts.append("\n\nDeep critic:")
        for d in deep_items:
            rubric   = d.get("rubric", "")
            verdict  = d.get("verdict", "")
            critique = d.get("critique") or ""
            line = f"  {rubric}: {verdict}"
            if critique:
                line += f" — {critique}"
            parts.append(f"\n{line}")

    # 4. summary (prm 우선, gen fallback)
    summary = step.get("prm_critique_summary") or step.get("gen_critique_summary") or ""
    if summary:
        parts.append(f"\n\nSummary: {summary}")

    # 5. wrong rubric special tokens (gold_fail_rubrics 직접 사용)
    wrong_tokens = [
        RUBRIC_TOKENS[r]
        for r in (step.get("gold_fail_rubrics") or [])
        if r in RUBRIC_TOKENS
    ]
    if wrong_tokens:
        parts.append("\n\nWrong rubrics:\n" + " ".join(wrong_tokens))

    # 6. next action
    next_action = step.get("next_gold_action") or TOKEN_SOLVE
    parts.append("\n\nNext action:\n" + next_action)

    return "".join(parts)
