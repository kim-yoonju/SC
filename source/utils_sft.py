"""
SFT 전용 유틸리티
- utils.py에서 SFT에 필요한 것만 추출 (torch/transformers만 의존)
- 모델 input/output 빌더 포함
"""

import json
import os
import pathlib as _pathlib
from functools import lru_cache

import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm
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

_model_cfg = CONF.get("model", {})

TOKEN_SOLVE   = _model_cfg.get("token_solve",   "<|solve|>")
TOKEN_RETHINK = _model_cfg.get("token_rethink", "<|rethink|>")
TOKEN_END     = _model_cfg.get("token_end",     "<|end|>")
ACTION_TOKENS  = [TOKEN_SOLVE, TOKEN_RETHINK, TOKEN_END]
SPECIAL_TOKENS = _model_cfg.get("special_tokens", [])

# ─────────────────────────────────────────────────────────────────────────────
# 토크나이저
# ─────────────────────────────────────────────────────────────────────────────

def setup_tokenizer(model_id: str, cache_dir: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if SPECIAL_TOKENS:
        tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tokenizer

def _build_plain_prompt(system: str, user: str) -> str:
    return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"


def build_chat_prompt(tokenizer, system: str, user: str, model_id: str = None) -> str:
    """model_id가 주어지면 모델명으로 chatML 여부 결정, 없으면 tokenizer.chat_template으로 판단."""
    if model_id is not None:
        _id = model_id.lower()
        use_chatml = "qwen" in _id and "instruct" in _id
    else:
        use_chatml = getattr(tokenizer, "chat_template", None) is not None

    if use_chatml:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return _build_plain_prompt(system, user)


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
# 모델 Input / Target 빌더
# ─────────────────────────────────────────────────────────────────────────────

def _history_text(steps: list[dict], up_to: int) -> list[str]:
    """steps[:up_to]에서 history용 텍스트 추출 (does 요약 우선, 없으면 inference 전체)."""
    result = []
    for s in steps[:up_to]:
        text = s.get("does") or (s.get("inference") or "")
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


def build_messages(problem: str, steps: list[dict], k: int,
                   system_solve: str, system_rethink: str) -> tuple[str, str]:
    """k번째 스텝의 (system, user) 메시지 문자열 반환 (tokenizer 불필요)."""
    history = _history_text(steps, k)

    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Previous steps]")
        for i, h in enumerate(history, 1):
            lines.append(f"Step {i}: {h}")
    lines.append(f"\nWrite Step {k + 1}.")
    user_msg = "\n".join(lines)

    state = steps[k].get("state", "")
    if state == "gen_rethink":
        err_exp = _error_explanation(steps, k)
        system  = system_rethink.replace("{{error_explanation}}", err_exp)
    else:
        system = system_solve

    return system, user_msg


import re as _re
_NUMERIC_VERDICT = _re.compile(r"^\d+\s*:\s*(correct|incorrect)\s*$", _re.I)


def _clean_critique(text: str, max_chars: int = 120) -> str:
    """숫자:correct/incorrect 패턴이면 None 반환, 아니면 max_chars로 잘라 반환."""
    text = text.strip()
    if not text or _NUMERIC_VERDICT.match(text):
        return None
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    dot = cut.rfind(".")
    return (cut[:dot + 1] if dot > 30 else cut).rstrip()


def build_target(step: dict, rubric_tokens: dict | None = None) -> str:
    """
    스텝에 대한 target 텍스트 생성 (loss 계산 영역).

    [math step] + Fast critic + Deep critic + Fail rubrics + Next action
    """
    parts = []

    # 1. math step — gen self-correction이 섞여 있으면 제거 (API gen은 분리 없이 저장됨)
    inference = step.get("inference") or ""
    sc_idx = inference.find("\nSelf-correction:")
    if sc_idx != -1:
        inference = inference[:sc_idx].strip()
    parts.append(inference)

    def _verdict(raw: str) -> str:
        v = (raw or "").lower()
        return "incorrect" if v == "incorrect" else "correct"

    # 2. fast critic
    fast = step.get("prm_fast_critique") or {}
    if fast:
        parts.append("\n\nFast critic:")
        for rubric, data in fast.items():
            raw      = data.get("verdict", "correct")
            critique = data.get("critique") or ""
            line = f"  {rubric}: {_verdict(raw)}"
            if _verdict(raw) == "incorrect":
                critique_text = critique.strip() if critique.strip() and not _NUMERIC_VERDICT.match(critique.strip()) else ""
                if critique_text:
                    line += f" — {critique_text}"
            parts.append(f"\n{line}")

    # 3. deep critic — fast incorrect 받은 루브릭을 재평가
    #    correct(N/A): critique 포함해 길게 출력
    #    incorrect: critique 포함해 길게 출력
    fast_fails = {r for r, d in fast.items() if _verdict(d.get("verdict") or "") == "incorrect"}
    deep = step.get("prm_deep_critique") or []
    deep_items = [
        d for d in deep
        if d.get("rubric") in fast_fails and d.get("verdict") is not None
    ]
    if deep_items:
        parts.append("\n\nDeep critic:")
        for d in deep_items:
            rubric   = d.get("rubric", "")
            raw      = (d.get("verdict") or "")
            critique = d.get("critique") or ""
            # deep critic는 자르지 않음 — 모델이 full reasoning을 학습해야 함
            critique_text = critique.strip() if critique.strip() and not _NUMERIC_VERDICT.match(critique.strip()) else ""
            line = f"  {rubric}: {critique_text} Verdict: {_verdict(raw)}" if critique_text else f"  {rubric}: Verdict: {_verdict(raw)}"
            parts.append(f"\n\n{line}")

    # 4. fail rubric special tokens — 없으면 none (모델이 항상 이 섹션을 출력하도록)
    _rubric_tokens = rubric_tokens or {}
    fail_tokens = [
        _rubric_tokens[r]
        for r in (step.get("gold_fail_rubrics") or [])
        if r in _rubric_tokens
    ]
    parts.append("\n\nFail rubrics:\n" + ("\n".join(fail_tokens) if fail_tokens else "none"))

    # 5. next action
    next_action = step.get("next_gold_action") or TOKEN_SOLVE
    parts.append("\n\nNext action:\n" + next_action)

    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 — generate_trajectory.py 출력 형식
#
# 데이터 형식:
#   {
#     "problem_id": str, "problem": str,
#     "gold_answer": str, "is_right": bool, "traj_type": str,
#     "steps": [
#       {
#         "step_idx": int, "step": str,
#         "inference": str, "source": str,
#         "is_error": bool, "state": str,
#         "next_gold_action": str,
#         "does": str,
#         "PRM_critique_summary": [...],
#       }, ...
#     ]
#   }
# ─────────────────────────────────────────────────────────────────────────────

class TrajDataset(Dataset):
    """
    generate_trajectory.py 출력 JSONL로부터 SFT 학습 샘플 생성.

    각 스텝마다 하나의 학습 샘플:
      input  (loss 제외): build_messages(problem, steps, k)
      target (loss 계산): build_target(steps[k])

    기본적으로 모든 스텝을 학습하되, skip_error=True면 is_error=True 스텝 제외.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 3072,
        skip_error: bool = False,
    ):
        self.max_length = max_length
        self.samples    = []

        raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"[TrajDataset] {len(raw)}개 trajectory 로드, 토크나이징 중...")

        skipped = 0
        for item in tqdm(raw, desc="Tokenizing", disable=(rank != 0)):
            skipped += self._process_item(item, tokenizer, skip_error)

        if rank == 0:
            print(f"[TrajDataset] 학습 샘플: {len(self.samples)}  (제외: {skipped})")

    def _process_item(self, item: dict, tokenizer, skip_error: bool) -> int:
        problem = item["problem"]
        steps   = item["steps"]
        skipped = 0

        for k, step in enumerate(steps):
            is_error = step.get("is_error", False)

            if skip_error and is_error:
                continue

            system_str, user_str = build_messages(problem, steps, k)
            assistant_str = build_target(step)

            full_msgs = [
                {"role": "system",    "content": system_str},
                {"role": "user",      "content": user_str},
                {"role": "assistant", "content": assistant_str},
            ]
            full_str   = tokenizer.apply_chat_template(full_msgs,     tokenize=False, add_generation_prompt=False)
            prefix_str = tokenizer.apply_chat_template(full_msgs[:2], tokenize=False, add_generation_prompt=True)

            full_ids   = tokenizer.encode(full_str,   add_special_tokens=False)
            prefix_len = len(tokenizer.encode(prefix_str, add_special_tokens=False))

            full_ids = full_ids[:self.max_length]

            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels    = torch.full_like(input_ids, -100)
            labels[prefix_len:] = input_ids[prefix_len:]

            if is_error:
                inference_str = step.get("inference", "")
                if inference_str:
                    inf_len = len(tokenizer.encode(inference_str, add_special_tokens=False))
                    inf_end = min(prefix_len + inf_len, len(full_ids))
                    labels[prefix_len:inf_end] = -100

            self.samples.append((input_ids, labels))

        return skipped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 전처리 데이터셋 — preprocess_sft.py 출력 형식
#
# 데이터 형식:
#   {
#     "input":  [{"role": "system", "content": ...}, {"role": "user", "content": ...}],
#     "target": str,
#   }
# ─────────────────────────────────────────────────────────────────────────────

class PreprocessedDataset(Dataset):
    """
    preprocess_sft.py 출력 JSONL로부터 SFT 학습 샘플 생성.

    각 줄의 input(system+user 메시지 리스트)과 target을 토크나이징해 학습 샘플 구성.
    is_error=True 샘플은 skip_error=True 시 제외.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 3072,
        skip_error: bool = False,
    ):
        self.samples = []

        raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"[PreprocessedDataset] {len(raw)}개 샘플 로드, 토크나이징 중...")

        skipped = 0
        for item in tqdm(raw, desc="Tokenizing", disable=(rank != 0)):
            is_error = item.get("is_error", False)

            if skip_error and is_error:
                skipped += 1
                continue

            msgs       = item["input"]  # [{"role": "system", ...}, {"role": "user", ...}]
            target_str = item["target"]

            full_msgs  = msgs + [{"role": "assistant", "content": target_str}]
            full_str   = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
            prefix_str = tokenizer.apply_chat_template(msgs,      tokenize=False, add_generation_prompt=True)

            full_ids   = tokenizer.encode(full_str,   add_special_tokens=False)
            prefix_len = len(tokenizer.encode(prefix_str, add_special_tokens=False))

            full_ids = full_ids[:max_length]

            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels    = torch.full_like(input_ids, -100)
            labels[prefix_len:] = input_ids[prefix_len:]

            # 실패한 스텝(is_error=True)은 inference 토큰을 loss에서 제외.
            # rethink까지 간 generator 스텝, patching까지 간 rethink 스텝이 해당.
            # critic + action 토큰은 그대로 학습.
            if is_error:
                inference_str = item.get("inference", "")
                if inference_str:
                    inf_len = len(tokenizer.encode(inference_str, add_special_tokens=False))
                    inf_end = min(prefix_len + inf_len, len(full_ids))
                    labels[prefix_len:inf_end] = -100

            self.samples.append((input_ids, labels))

        if rank == 0:
            print(f"[PreprocessedDataset] 학습 샘플: {len(self.samples)}  (제외: {skipped})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def debug(data_path: str, tokenizer, n: int = 0):
    """전처리된 JSONL(input/target 포맷)에서 n번째 샘플을 토크나이징해 출력."""
    raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    sep = "─" * 72

    if n >= len(raw):
        print(f"[debug] 인덱스 {n}이 범위를 초과했습니다. (총 {len(raw)}개)")
        return

    item       = raw[n]
    msgs       = item["input"]   # [system, user]
    target_str = item["target"]
    full_msgs  = msgs + [{"role": "assistant", "content": target_str}]

    full_str   = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prefix_str = tokenizer.apply_chat_template(msgs,      tokenize=False, add_generation_prompt=True)

    p_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
    f_ids = tokenizer.encode(full_str,   add_special_tokens=False)
    t_len = len(f_ids) - len(p_ids)

    print(f"\n{'='*72}")
    print(f"[샘플 {n}  state={item.get('state')}  is_error={item.get('is_error')}]")
    print(f"\n[INPUT — {len(p_ids)} tok]\n{sep}")
    print(prefix_str[-800:])
    print(f"\n[TARGET — {t_len} tok]\n{sep}")
    print(target_str)
    print(f"\n토큰: input={len(p_ids)}  target={t_len}  total={len(f_ids)}")


# ─────────────────────────────────────────────────────────────────────────────
# 기본 데이터셋 — jsonl에 "input", "target" 필드만 있는 경우
#
# 데이터 형식:
#   {"input": "<prompt string>", "target": "<response string>"}
#
# 사용 예:
#   dataset = SimpleDataset("data.jsonl", tokenizer, max_length=2048)
# ─────────────────────────────────────────────────────────────────────────────

# class SimpleDataset(Dataset):
#     def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
#         self.samples = []
#         raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
#         skipped = 0
#         for item in raw:
#             input_str  = item["input"]
#             target_str = item["target"]
#             full_str   = input_str + target_str
#             full_ids   = tokenizer.encode(full_str,   add_special_tokens=False)
#             prefix_len = len(tokenizer.encode(input_str, add_special_tokens=False))
#             if len(full_ids) > max_length:
#                 skipped += 1
#                 continue
#             input_ids = torch.tensor(full_ids, dtype=torch.long)
#             labels    = torch.full_like(input_ids, -100)
#             labels[prefix_len:] = input_ids[prefix_len:]
#             self.samples.append((input_ids, labels))
#         print(f"[SimpleDataset] {len(self.samples)}개 로드  (제외: {skipped})")
#
#     def __len__(self):  return len(self.samples)
#     def __getitem__(self, idx): return self.samples[idx]
