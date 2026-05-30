"""
SFT 전용 유틸리티
- utils.py에서 SFT에 필요한 것만 추출 (torch/transformers만 의존)
- 모델 input/output 빌더 포함
"""

import json
import os
import pathlib as _pathlib
import re as _re
from collections import defaultdict
from functools import lru_cache

import torch
import yaml
from torch.utils.data import Dataset, Sampler
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

TOKEN_NONE    = _model_cfg.get("token_none",    "<|none|>")
ACTION_TOKENS  = [TOKEN_SOLVE, TOKEN_RETHINK, TOKEN_END]
SPECIAL_TOKENS = _model_cfg.get("special_tokens", [])

# ─────────────────────────────────────────────────────────────────────────────
# 토크나이저
# ─────────────────────────────────────────────────────────────────────────────

def setup_tokenizer(model_id: str, cache_dir: str = None):
    import os
    kwargs = dict(trust_remote_code=True)
    is_local_path = model_id.startswith("/") or model_id.startswith("./") or model_id.startswith("../")
    if is_local_path:
        if not os.path.isdir(model_id):
            raise FileNotFoundError(f"로컬 모델 경로가 존재하지 않습니다: {model_id}")
        kwargs["local_files_only"] = True
    else:
        kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
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

_STEP_PREFIX      = _re.compile(r"^Step\s+\d+[:.]\s*", _re.I)
_CRITIC_MARKER    = "\n\nFast critic:"
_NEXT_ACTION_RE   = _re.compile(r"Next action:\n(.+)", _re.MULTILINE)


def _inference_end_idx(tokenizer, prefix_str: str, target_str: str) -> int | None:
    """
    is_fail=True 스텝에서 inference 부분의 마지막 토큰 인덱스(exclusive) 반환.
    target_str에서 '\\n\\nFast critic:' 앞까지를 inference로 간주.
    경계를 찾지 못하면 None 반환 (= 마스킹 없이 전체 loss 유지).
    """
    split = target_str.find(_CRITIC_MARKER)
    if split == -1:
        return None  # Fast critic 없음 → 마스킹 불가, 전체 loss 유지
    inference_only = target_str[:split]
    boundary_ids = tokenizer.encode(prefix_str + inference_only, add_special_tokens=False)
    return len(boundary_ids)


def mask_inference_for_error(
    labels: torch.Tensor,
    prefix_len: int,
    tokenizer,
    prefix_str: str,
    target_str: str,
) -> None:
    """
    is_fail=True 스텝에서 inference 부분의 loss를 제거한다.
    - 좋은 inference(is_fail=False)는 전체 시퀀스에 loss.
    - 나쁜 inference(is_fail=True)는 inference를 건너뛰고
      Fast critic ~ Next action 구간만 loss.
    - Fast critic 마커를 찾지 못하면 마스킹 없이 전체 loss 유지.
    """
    inf_end = _inference_end_idx(tokenizer, prefix_str, target_str)
    if inf_end is None:
        return
    labels[prefix_len:min(inf_end, len(labels))] = -100


def _strip_newlines(text: str) -> str:
    return " ".join(text.split())


def _strip_step_prefix(text: str) -> str:
    return _STEP_PREFIX.sub("", text, count=1)


def _history_text(steps: list[dict], up_to: int) -> list[str]:
    """steps[:up_to]에서 is_fail=False 스텝만 history용 텍스트로 추출."""
    result = []
    for s in steps[:up_to]:
        if s.get("is_fail", False):
            continue
        text = s.get("does") or (s.get("inference") or "")
        text = _strip_step_prefix(_strip_newlines(text))
        result.append(text)
    return result


def _error_explanation(steps: list[dict], rethink_idx: int) -> str:
    """rethink 스텝 직전 wrong step에서 오류 설명 추출."""
    for i in range(rethink_idx - 1, -1, -1):
        s = steps[i]
        if s.get("is_fail"):
            parts = []
            does = s.get("does")
            if does:
                parts.append(_strip_newlines(does))
            summary = s.get("prm_critique_summary") or s.get("gen_critique_summary")
            if summary:
                parts.append(_strip_newlines(summary))
            return " ".join(parts) if parts else "the previous step contained an error"
    return "the previous step contained an error"


def build_messages_inference(problem: str, steps: list[dict], k: int,
                             system: str) -> tuple[str, str]:
    """k번째 스텝의 inference model용 (system, user) 메시지 반환.
    gen_rethink 상태의 error_explanation 치환 없이 히스토리만 사용.
    """
    history = _history_text(steps, k)
    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Previous steps]")
        for h in history:
            lines.append(h)
    lines.append("\nWrite the next step.")
    user_msg = "\n".join(lines)
    return system, user_msg


def _extract_inference(step: dict) -> str:
    """스텝에서 inference 텍스트 추출 (self-correction·step prefix 제거)."""
    text = step.get("inference_summary") or step.get("inference") or ""
    sc_idx = text.find("\nSelf-correction:")
    if sc_idx != -1:
        text = text[:sc_idx].strip()
    return _strip_step_prefix(text.strip())


_PREV_MARKER      = "\n\n[Previous steps]\n"
_CUR_MARKER       = "\n\n[Current step]"
_ONELINER_PREFIX  = "approach: "   # _strip_newlines 처리된 단일 줄 스텝 접두어

def _trim_oldest_history_step(user_msg: str) -> str | None:
    """[Previous steps] 섹션에서 가장 오래된 스텝 한 항목 제거. 제거할 게 없으면 None.

    - 단일 줄 스텝("approach: ..."): 한 줄씩 제거
    - 마지막 multiline 스텝(직전 inference 전체): 섹션 통째 제거
    """
    prev_start = user_msg.find(_PREV_MARKER)
    if prev_start == -1:
        return None
    steps_start = prev_start + len(_PREV_MARKER)
    cur_start = user_msg.find(_CUR_MARKER, steps_start)
    if cur_start == -1:
        return None
    steps_section = user_msg[steps_start:cur_start]

    # 단일 줄 스텝이 아니면(= multiline 마지막 스텝만 남음) 섹션 전체 제거
    if not steps_section.startswith(_ONELINER_PREFIX):
        return user_msg[:prev_start] + user_msg[cur_start:]

    # 첫 번째 단일 줄 스텝 제거
    newline_pos = steps_section.find("\n")
    if newline_pos == -1 or not steps_section[newline_pos + 1:]:
        return user_msg[:prev_start] + user_msg[cur_start:]
    return user_msg[:steps_start] + steps_section[newline_pos + 1:] + user_msg[cur_start:]


def build_messages_classification(problem: str, steps: list[dict], k: int,
                                   system: str) -> tuple[str, str]:
    """k번째 스텝의 classification model용 (system, user) 메시지 반환.

    History 포맷:
      - 직전 비오류 스텝 (k-1에 해당): inference 전체 그대로
      - 그 이전 비오류 스텝들: does 요약 (없으면 inference) 한 줄
    """
    history_steps = [s for s in steps[:k] if not s.get("is_fail", False)]

    history_texts = []
    for i, s in enumerate(history_steps):
        if i == len(history_steps) - 1:
            # 직전 스텝: inference 그대로
            history_texts.append(_extract_inference(s))
        else:
            # 그 이전 스텝: does 요약
            text = s.get("does") or (s.get("inference") or "")
            history_texts.append(_strip_step_prefix(_strip_newlines(text)))

    step = steps[k]
    inference = _extract_inference(step)

    lines = [f"[Problem]\n{problem}"]
    if history_texts:
        lines.append("\n[Previous steps]")
        for h in history_texts:
            lines.append(h)
    lines.append(f"\n[Current step]\n{inference}")
    lines.append("\nEvaluate this step.")
    user_msg = "\n".join(lines)
    return system, user_msg


def build_messages(problem: str, steps: list[dict], k: int,
                   system_solve: str, system_rethink: str) -> tuple[str, str]:
    """k번째 스텝의 (system, user) 메시지 문자열 반환 (tokenizer 불필요)."""
    history = _history_text(steps, k)

    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Previous steps]")
        for h in history:
            lines.append(h)
    lines.append("\nReason the next step.")
    user_msg = "\n".join(lines)

    state = steps[k].get("state", "")
    if state == "gen_rethink":
        err_exp = _error_explanation(steps, k)
        system  = system_rethink.replace("{{error_explanation}}", err_exp)
    else:
        system = system_solve

    return system, user_msg


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


_CLS_MARKERS = ("\nDeep critic:", "Deep critic:", "\nDeep critique:", "Deep critique:",
                "\nFast critic:", "Fast critic:", "\nFail rubrics:", "\nNext action:")

def build_target_inference(step: dict) -> str:
    """Inference model target: math step only (no Does summary)."""
    inference = step.get("inference") or ""
    sc_idx = inference.find("\nSelf-correction:")
    if sc_idx != -1:
        inference = inference[:sc_idx]
    for marker in _CLS_MARKERS:
        m_idx = inference.find(marker)
        if m_idx != -1:
            inference = inference[:m_idx]
    return _strip_step_prefix(inference.strip())


_NA_PREFIX   = _re.compile(r"^N/A\s*[—–\-]+\s*", _re.I)
_FAIL_PREFIX = _re.compile(r"^FAIL\s*[:\-]+\s*", _re.I)
# critique 원문 안에 이미 포함된 trailing "Verdict: correct/incorrect" 제거
# (\*{0,2}...\*{0,2} 는 **Verdict: ...** 마크다운 bold 포함)
_TRAILING_VERDICT = _re.compile(
    r'(?:\s*\*{0,2}Verdict:\s*(?:correct|incorrect)\*{0,2})+\s*$', _re.I)


def _clean_deep_critique(text: str) -> str:
    """Deep critic 텍스트 정규화.
    - 'N/A — ' / 'FAIL: ' 접두어 제거
    - 줄바꿈 → 공백 (인라인 텍스트로 변환)
    - 이미 포함된 trailing 'Verdict: correct/incorrect' 제거
      (build_target_classification이 authoritative verdict를 다시 붙이므로)
    """
    text = _strip_newlines(text.strip())
    text = _NA_PREFIX.sub("", text)
    text = _FAIL_PREFIX.sub("", text)
    text = _TRAILING_VERDICT.sub("", text)
    return text.strip()


def _get_incorrect_rubrics(step: dict) -> list[str]:
    """Deep critic 결과에서 incorrect 판정을 받은 루브릭 이름 목록 반환."""
    def _v(raw): return "incorrect" if (raw or "").lower() in ("incorrect", "fail") else "correct"
    dc_list = step.get("prm_deep_critique") or []
    fc_dict = step.get("prm_fast_critique") or {}
    rubric_iter = [d.get("rubric", "") for d in dc_list] if dc_list else list(fc_dict.keys())
    result = []
    for rubric in rubric_iter:
        if not rubric:
            continue
        dc_entry = next((d for d in dc_list if d.get("rubric") == rubric), {})
        if dc_entry.get("verdict") is not None:
            raw = dc_entry.get("verdict") or ""
        else:
            raw = (fc_dict.get(rubric) or {}).get("verdict", "correct")
        if _v(raw) == "incorrect":
            result.append(rubric)
    return result


def build_target_classification(step: dict, rubric_tokens: dict | None = None,
                                include_rubrics: bool = True,
                                include_actions: bool = True,
                                use_summary: bool = False) -> str:
    """Classification model target: Deep critic (전체 루브릭) + Fail rubrics + Next action

    루브릭별 우선순위:
      1. prm_deep_critique verdict != null  → deep critique 텍스트 사용
      2. prm_deep_critique verdict == null  → fast critic에서 correct 받은 것, prm_fast_critique 텍스트로 대체

    use_summary=True: critique 원문 대신 prm_critique_summary dict의 짧은 요약 사용
    """
    parts = []

    def _verdict(raw: str) -> str:
        v = (raw or "").lower()
        return "incorrect" if v in ("incorrect", "fail") else "correct"

    dc_list = step.get("prm_deep_critique") or []
    fc_dict = step.get("prm_fast_critique") or {}
    critique_summary_map = (step.get("prm_critique_summary") or {}) if use_summary else {}

    incorrect_rubrics: list[str] = []
    parts.append("Deep critic:")

    if not dc_list and not fc_dict:
        parts.append("\n  none")
    else:
        # dc_list가 있으면 그 순서 사용 (항상 11개 루브릭 포함),
        # 없으면 fc_dict 순서 사용
        rubric_iter = [d.get("rubric", "") for d in dc_list] if dc_list else list(fc_dict.keys())

        for rubric in rubric_iter:
            if not rubric:
                continue
            dc_entry = next((d for d in dc_list if d.get("rubric") == rubric), {})

            if dc_entry.get("verdict") is not None:
                # deep critique에 실제 판정이 있음
                raw      = dc_entry.get("verdict") or ""
                critique = dc_entry.get("critique") or ""
            else:
                # fast critic correct → prm_fast_critique 텍스트로 대체
                fc = fc_dict.get(rubric) or {}
                raw      = fc.get("verdict", "correct")
                critique = fc.get("critique") or ""

            if use_summary:
                critique = critique_summary_map.get(rubric) or ""

            verdict = _verdict(raw)
            if verdict == "incorrect":
                incorrect_rubrics.append(rubric)

            critique_text = critique.strip()
            if critique_text and not _NUMERIC_VERDICT.match(critique_text):
                critique_clean = _clean_deep_critique(critique_text)
            else:
                critique_clean = ""

            line = (f"  {rubric}: {critique_clean} Verdict: {verdict}"
                    if critique_clean else f"  {rubric}: Verdict: {verdict}")
            parts.append(f"\n{line}")

    # next action — include_actions=True일 때만 포함
    if include_actions:
        next_action = step.get("next_gold_action") or TOKEN_SOLVE
        parts.append("\n\nNext action:\n" + next_action)

    return "".join(parts)


def build_target(step: dict, rubric_tokens: dict | None = None) -> str:
    """
    스텝에 대한 target 텍스트 생성 (loss 계산 영역).

    [math step] + Fast critic + Deep critic + Fail rubrics + Next action
    """
    parts = []

    # 1. math step — gen self-correction 제거, Step N: 접두사 제거
    inference = step.get("inference") or ""
    sc_idx = inference.find("\nSelf-correction:")
    if sc_idx != -1:
        inference = inference[:sc_idx].strip()
    inference = _strip_step_prefix(inference.strip())
    parts.append(inference)

    def _verdict(raw: str) -> str:
        v = (raw or "").lower()
        return "incorrect" if v in ("incorrect", "fail") else "correct"

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
            critique_clean = _strip_newlines(critique) if critique.strip() and not _NUMERIC_VERDICT.match(critique.strip()) else ""
            line = f"  {rubric}: {critique_clean} Verdict: {_verdict(raw)}" if critique_clean else f"  {rubric}: Verdict: {_verdict(raw)}"
            parts.append(f"\n{line}")

    # 4. fail rubric special tokens — 없으면 섹션 헤더만 출력 (TOKEN_NONE 제거)
    _rubric_tokens = rubric_tokens or {}
    _gfr = step.get("gold_fail_rubrics")
    _gfr_list = _gfr if isinstance(_gfr, list) else []
    fail_tokens = [_rubric_tokens[r] for r in _gfr_list if r in _rubric_tokens]
    parts.append("\n\nFail rubrics:\n" + "\n".join(fail_tokens))

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
#         "is_fail": bool, "state": str,
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

    기본적으로 모든 스텝을 학습하되, skip_error=True면 is_fail=True 스텝 제외.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 3072,
        skip_error: bool = False,
    ):
        from preprocess import get_system_prompts
        self._system_solve, self._system_rethink = get_system_prompts()

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
            is_fail = step.get("is_fail", False)

            if skip_error and is_fail:
                continue

            system_str, user_str = build_messages(problem, steps, k,
                                                   self._system_solve, self._system_rethink)
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

            if is_fail:
                mask_inference_for_error(labels, prefix_len, tokenizer, prefix_str, assistant_str)

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

class TrajectoryOrderedSampler(Sampler):
    """
    Trajectory 단위로 순서를 보장하는 분산 학습용 샘플러.

    - 매 에폭마다 trajectory 순서를 shuffle (set_epoch 호출로)
    - trajectory 내 step 순서는 항상 k 오름차순 유지
    - 각 rank는 전체 인덱스의 연속 청크를 담당
    """

    def __init__(self, traj_groups: list[list[int]], num_replicas: int, rank: int, seed: int = 0):
        self.traj_groups  = traj_groups
        self.num_replicas = num_replicas
        self.rank         = rank
        self.seed         = seed
        self.epoch        = 0

        total = sum(len(g) for g in traj_groups)
        self.total_size  = (total // num_replicas) * num_replicas
        self.num_samples = self.total_size // num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        traj_order = torch.randperm(len(self.traj_groups), generator=g).tolist()

        indices = []
        for ti in traj_order:
            indices.extend(self.traj_groups[ti])
        indices = indices[:self.total_size]

        # 각 rank가 연속 청크를 담당 (trajectory 내 순서 보존)
        start = self.rank * self.num_samples
        return iter(indices[start: start + self.num_samples])

    def __len__(self):
        return self.num_samples


class PreprocessedDataset(Dataset):
    """
    preprocess_sft.py 출력 JSONL로부터 SFT 학습 샘플 생성.

    각 줄의 input(system+user 메시지 리스트)과 target을 토크나이징해 학습 샘플 구성.
    is_fail=True 샘플은 skip_error=True 시 제외.
    traj_id 필드가 있으면 TrajectoryOrderedSampler 사용 가능.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 3072,
        skip_error: bool = False,
    ):
        self.samples  = []
        self.traj_ids = []

        raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"[PreprocessedDataset] {len(raw)}개 샘플 로드, 토크나이징 중...")

        skipped = 0
        for item in tqdm(raw, desc="Tokenizing", disable=(rank != 0)):
            is_fail = item.get("is_fail", False)

            if skip_error and is_fail:
                skipped += 1
                continue

            msgs       = item["input"]
            target_str = item["target"]

            full_msgs  = msgs + [{"role": "assistant", "content": target_str}]
            full_str   = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
            prefix_str = tokenizer.apply_chat_template(msgs,      tokenize=False, add_generation_prompt=True)

            full_ids   = tokenizer.encode(full_str,   add_special_tokens=False)
            prefix_len = len(tokenizer.encode(prefix_str, add_special_tokens=False))

            if len(full_ids) > max_length:
                user_content = msgs[1]["content"]
                while len(full_ids) > max_length:
                    trimmed = _trim_oldest_history_step(user_content)
                    if trimmed is None:
                        break
                    user_content = trimmed
                    trimmed_msgs  = [msgs[0], {"role": "user", "content": user_content}]
                    full_str      = tokenizer.apply_chat_template(
                        trimmed_msgs + [{"role": "assistant", "content": target_str}],
                        tokenize=False, add_generation_prompt=False)
                    prefix_str    = tokenizer.apply_chat_template(
                        trimmed_msgs, tokenize=False, add_generation_prompt=True)
                    full_ids   = tokenizer.encode(full_str,   add_special_tokens=False)
                    prefix_len = len(tokenizer.encode(prefix_str, add_special_tokens=False))
                if len(full_ids) > max_length:
                    skipped += 1
                    continue

            full_ids = full_ids[:max_length]

            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels    = torch.full_like(input_ids, -100)
            labels[prefix_len:] = input_ids[prefix_len:]

            if is_fail:
                mask_inference_for_error(labels, prefix_len, tokenizer, prefix_str, target_str)

            self.samples.append((input_ids, labels))
            self.traj_ids.append(item.get("traj_id"))

        if rank == 0:
            print(f"[PreprocessedDataset] 학습 샘플: {len(self.samples)}  (제외: {skipped}, max_length={max_length} 초과 포함)")

    @property
    def has_traj_ids(self) -> bool:
        return any(tid is not None for tid in self.traj_ids)

    def traj_groups(self) -> list[list[int]]:
        """traj_id 별로 샘플 인덱스 그룹핑. TrajectoryOrderedSampler에 전달."""
        groups: dict[int, list[int]] = defaultdict(list)
        for idx, tid in enumerate(self.traj_ids):
            groups[tid if tid is not None else idx].append(idx)
        return list(groups.values())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _build_samples(data_path: str):
    """데이터 파일을 읽어 (idx, msgs, inference, target_str, state, is_fail) 리스트 반환."""
    raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    first = raw[0] if raw else {}
    is_preprocessed = "input" in first and "target" in first

    if is_preprocessed:
        result = []
        for i, item in enumerate(raw):
            result.append((i, item["input"], item.get("inference", ""),
                           item["target"], item.get("state"), item.get("is_fail")))
        return result
    else:
        from preprocess import get_system_prompts, RUBRIC_TOKENS
        system_solve, system_rethink = get_system_prompts()
        result = []
        idx = 0
        for traj in raw:
            problem = traj["problem"]
            steps   = traj["steps"]
            for k, step in enumerate(steps):
                system_str, user_str = build_messages(problem, steps, k, system_solve, system_rethink)
                target_str = build_target(step, RUBRIC_TOKENS)
                msgs = [{"role": "system", "content": system_str},
                        {"role": "user",   "content": user_str}]
                result.append((idx, msgs, step.get("inference", ""),
                               target_str, step.get("state"), step.get("is_fail")))
                idx += 1
        return result


def _generate_critique(model, tokenizer, msgs: list) -> str:
    """model에 msgs를 입력해 critique를 생성하고 반환."""
    import torch
    input_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_str, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(next(model.parameters()).device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=1500,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


def _print_sample(tokenizer, idx, msgs, inference_str, target_str, state, is_fail, model=None):
    sep = "─" * 72

    print(f"\n{'='*72}")
    print(f"[샘플 {idx}  state={state}  is_fail={is_fail}]")

    print(f"\n[INPUT]\n{sep}")
    if isinstance(msgs, list):
        for msg in msgs:
            role    = msg.get("role", "?")
            content = msg.get("content", "")
            print(f"\n<{role}>")
            print(content)
    else:
        print(msgs)

    if model is not None and tokenizer is not None:
        print(f"\n[GENERATION]\n{sep}")
        generated = _generate_critique(model, tokenizer, msgs)
        print(generated)

    print(f"\n[TARGET]\n{sep}")
    print(target_str)

    if tokenizer is not None:
        try:
            full_msgs  = (msgs if isinstance(msgs, list) else [{"role": "user", "content": msgs}]) \
                         + [{"role": "assistant", "content": target_str}]
            full_str   = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
            prefix_str = tokenizer.apply_chat_template(
                msgs if isinstance(msgs, list) else [{"role": "user", "content": msgs}],
                tokenize=False, add_generation_prompt=True)
            p_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
            f_ids = tokenizer.encode(full_str,   add_special_tokens=False)
            t_len = len(f_ids) - len(p_ids)
            print(f"\n토큰: input={len(p_ids)}  target={t_len}  total={len(f_ids)}")
        except Exception as e:
            print(f"\n[토큰 수 계산 실패: {e}]")


def debug(data_path: str, tokenizer, n: int | None = None,
          model_path: str | None = None, cache_dir: str | None = None):
    """
    n=None : is_fail=False 샘플 하나 자동 출력
    n=int  : n번째 샘플 출력
    model_path가 주어지면 모델을 로드해 [GENERATION] 섹션에 실제 inference 출력.
    """
    samples = _build_samples(data_path)

    model = None
    if model_path is not None and tokenizer is not None:
        import torch
        from transformers import AutoModelForCausalLM
        print(f"[debug] 모델 로드 중: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, cache_dir=cache_dir,
            dtype=torch.bfloat16, trust_remote_code=True,
            device_map="auto", attn_implementation="sdpa",
        )
        model.eval()
        print("[debug] 모델 로드 완료")

    if n is not None:
        if n >= len(samples):
            print(f"[debug] 인덱스 {n}이 범위를 초과했습니다. (총 {len(samples)}개)")
            return
        idx, msgs, inference_str, target_str, state, is_fail = samples[n]
        _print_sample(tokenizer, idx, msgs, inference_str, target_str, state, is_fail, model=model)
        return

    # 자동 모드: is_fail=False 첫 번째 샘플 출력
    for idx, msgs, inference_str, target_str, state, is_fail in samples:
        if not is_fail:
            _print_sample(tokenizer, idx, msgs, inference_str, target_str, state, is_fail, model=model)
            return


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
