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

FOCAL_SENTINEL = -1.0   # token_weights에서 이 값 = focal loss 적용 위치
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
    input_ids_list, labels_list, token_weights_list = zip(*batch)
    max_len = max(x.size(0) for x in input_ids_list)
    padded_input   = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_labels  = torch.full((len(batch), max_len), -100,         dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len,                dtype=torch.long)
    for i, (inp, lbl) in enumerate(zip(input_ids_list, labels_list)):
        seq_len = inp.size(0)
        padded_input[i, :seq_len]   = inp
        padded_labels[i, :seq_len]  = lbl
        attention_mask[i, :seq_len] = 1
    result = {"input_ids": padded_input, "attention_mask": attention_mask, "labels": padded_labels}
    if any(tw is not None for tw in token_weights_list):
        padded_weights = torch.ones(len(batch), max_len, dtype=torch.float32)
        for i, tw in enumerate(token_weights_list):
            if tw is not None:
                seq_len = tw.size(0)
                padded_weights[i, :seq_len] = tw
        result["token_weights"] = padded_weights
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 모델 Input / Target 빌더
# ─────────────────────────────────────────────────────────────────────────────

_STEP_PREFIX      = _re.compile(r"^Step\s+\d+[:.]\s*", _re.I)
_CRITIC_MARKER    = "\n\nFast critic:"
_NEXT_ACTION_RE   = _re.compile(r"Next action:\n(.+)", _re.MULTILINE)


def _find_next_action_pos(target_str: str, labels: torch.Tensor, prefix_len: int,
                          tokenizer, action_weight_map: dict) -> tuple[int, int] | None:
    """
    target_str에서 "Next action:\\n" 뒤 스페셜 토큰을 파싱하고,
    labels에서 해당 토큰의 위치와 token_id를 반환. 실패 시 None.
    """
    m = _NEXT_ACTION_RE.search(target_str)
    if not m:
        return None
    tok_str = m.group(1).strip()
    tids = tokenizer.encode(tok_str, add_special_tokens=False)
    if not tids:
        return None
    tid = tids[-1]
    if tid not in action_weight_map:
        return None
    if prefix_len >= len(labels):
        return None
    positions = (labels[prefix_len:] == tid).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        return None
    # Next action은 target 맨 끝 — 마지막 occurrence 사용
    pos = prefix_len + positions[-1].item()
    return pos, tid


def _inference_end_idx(tokenizer, prefix_str: str, target_str: str) -> int | None:
    """
    is_error=True 스텝에서 inference 부분의 마지막 토큰 인덱스(exclusive) 반환.
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
    is_error=True 스텝에서 inference 부분의 loss를 제거한다.
    - 좋은 inference(is_error=False)는 전체 시퀀스에 loss.
    - 나쁜 inference(is_error=True)는 inference를 건너뛰고
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
    """steps[:up_to]에서 is_error=False 스텝만 history용 텍스트로 추출."""
    result = []
    for s in steps[:up_to]:
        if s.get("is_error", False):
            continue
        text = s.get("does") or (s.get("inference") or "")
        text = _strip_step_prefix(_strip_newlines(text))
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


def build_messages_classification(problem: str, steps: list[dict], k: int,
                                   system: str) -> tuple[str, str]:
    """k번째 스텝의 classification model용 (system, user) 메시지 반환.
    유저 메시지에 현재 스텝의 inference + Does 텍스트를 포함해 평가 요청.
    inference 모델 출력 형식([math step]\\n\\nDoes: [summary])과 일치시킴.
    """
    history = _history_text(steps, k)
    step = steps[k]

    inference = step.get("inference_summary") or step.get("inference") or ""
    sc_idx = inference.find("\nSelf-correction:")
    if sc_idx != -1:
        inference = inference[:sc_idx].strip()
    inference = _strip_step_prefix(inference.strip())

    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Previous steps]")
        for h in history:
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


def compute_action_weights(data_path: str, tokenizer) -> dict | None:
    """
    전처리 JSONL의 next action 분포를 자동 계산해 역빈도 가중치 반환.
    - solve를 기준(1.0)으로 정규화: weight_i = count_solve / count_i
    - action token이 없는 데이터(inference 모드 등)면 None 반환.
    """
    import re
    pattern = re.compile(r"Next action:\n(.+)$", re.MULTILINE)
    counts: dict[str, int] = {"solve": 0, "rethink": 0, "end": 0}

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            target = item.get("target", "")
            m = pattern.search(target)
            if not m:
                continue
            tok = m.group(1).strip()
            if TOKEN_SOLVE in tok:
                counts["solve"] += 1
            elif TOKEN_RETHINK in tok:
                counts["rethink"] += 1
            elif TOKEN_END in tok:
                counts["end"] += 1

    if counts["solve"] == 0:
        return None  # inference 모드 등 — action token 없음

    _name2str = {"solve": TOKEN_SOLVE, "rethink": TOKEN_RETHINK, "end": TOKEN_END}
    weight_map: dict[int, float] = {}
    solve_n = counts["solve"]
    for name, n in counts.items():
        if n == 0:
            continue
        tids = tokenizer.encode(_name2str[name], add_special_tokens=False)
        if tids:
            weight_map[tids[-1]] = solve_n / n  # solve=1 기준 역빈도

    total = sum(counts.values())
    print(f"\n[action_weights] Next action 역빈도 가중치 (solve=1.0 기준)")
    print(f"  {'액션':<10} {'count':>7}  {'비율':>6}  {'weight':>8}")
    print(f"  {'─'*38}")
    for name in ("solve", "rethink", "end"):
        n = counts[name]
        if n == 0:
            continue
        tids = tokenizer.encode(_name2str[name], add_special_tokens=False)
        w = weight_map.get(tids[-1], 0.0) if tids else 0.0
        print(f"  {name:<10} {n:>7d}  {n/max(total,1):>5.1%}  {w:>8.3f}")
    print(f"  {'─'*38}\n")

    return weight_map


_FAIL_RUBRICS_RE = _re.compile(r"\n\nFail rubrics:\n(.*?)$", _re.DOTALL)


def compute_rubric_weights(data_path: str, tokenizer, max_weight: float = 10.0) -> dict | None:
    """
    전처리 JSONL의 fail rubric 토큰 분포를 계산해 역빈도 가중치 반환.
    - <|none|>을 제외한 가장 많이 등장한 루브릭을 기준(1.0)으로 정규화.
    - <|none|>은 count_max_rubric / count_none 으로 down-weight (<1.0).
    - 실제 루브릭은 count_max_rubric / count_i (≥1.0), max_weight로 상한 클리핑.
    - multi-label: 한 샘플에 여러 루브릭이 있으면 각각 카운트.
    """
    counts: dict[str, int] = {}   # rubric_token_str → count
    _valid_tokens = set(SPECIAL_TOKENS)  # 유효한 special token만 카운트 (오염 항목 제외)

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            target = item.get("target", "")
            m = _FAIL_RUBRICS_RE.search(target)
            if not m:
                continue
            section = m.group(1).strip()
            for tok in section.split("\n"):
                tok = tok.strip()
                if tok and tok in _valid_tokens:
                    counts[tok] = counts.get(tok, 0) + 1

    if not counts:
        return None

    # <|none|>을 제외한 실제 루브릭 카운트로 기준값 계산
    rubric_counts = {k: v for k, v in counts.items() if k != TOKEN_NONE}
    none_count = counts.get(TOKEN_NONE, 0)

    if not rubric_counts:
        return None  # 실제 루브릭 없음 — 가중치 미적용

    count_max = max(rubric_counts.values())
    weight_map: dict[int, float] = {}

    # 실제 루브릭: 역빈도 원시값 계산
    raw_weights = {tok_str: count_max / n for tok_str, n in rubric_counts.items()}

    # max_weight 초과 시 전체를 비율 유지하며 스케일다운
    actual_max = max(raw_weights.values())
    scale = min(1.0, max_weight / actual_max)

    for tok_str, raw_w in raw_weights.items():
        tids = tokenizer.encode(tok_str, add_special_tokens=False)
        if tids:
            weight_map[tids[-1]] = raw_w * scale

    # <|none|>: 실제 루브릭과 같은 scale 적용
    if none_count:
        tids = tokenizer.encode(TOKEN_NONE, add_special_tokens=False)
        if tids:
            weight_map[tids[-1]] = (count_max / none_count) * scale

    total = sum(counts.values())
    print(f"\n[rubric_focal] Fail rubrics 토큰 분포 (focal loss 위치 감지용 — weight 값은 미사용)")
    print(f"  {'루브릭 토큰':<45} {'count':>7}  {'비율':>6}")
    print(f"  {'─'*60}")
    for tok_str, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {tok_str:<45} {n:>7d}  {n/max(total,1):>5.1%}")
    print(f"  {'─'*60}\n")

    return weight_map


def _find_rubric_positions(
    target_str: str,
    labels: torch.Tensor,
    prefix_len: int,
    tokenizer,
    rubric_weight_map: dict,
) -> list[tuple[int, int, float]] | None:
    """
    target_str의 Fail rubrics 섹션에서 루브릭 특수 토큰 위치와 가중치를 반환.
    반환:
      None  — 섹션 자체가 없거나 루브릭 토큰이 labels에서 발견되지 않은 파싱 실패
      [...]  — 발견된 (pos, token_id, weight) 리스트 (<|none|> 포함)
    """
    m = _FAIL_RUBRICS_RE.search(target_str)
    if not m:
        return None
    section = m.group(1).strip()

    if not section:
        return []   # fail rubrics 없음 — focal 적용 위치 없음 (정상)

    results = []
    for tok_str in section.split("\n"):
        tok_str = tok_str.strip()
        if not tok_str:
            continue
        tids = tokenizer.encode(tok_str, add_special_tokens=False)
        if not tids:
            return None
        tid = tids[-1]
        if tid not in rubric_weight_map:
            return None
        if prefix_len >= len(labels):
            return None
        positions = (labels[prefix_len:] == tid).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            return None
        pos = prefix_len + positions[-1].item()
        results.append((pos, tid, rubric_weight_map[tid]))

    return results


_CLS_MARKERS = ("\nDeep critique:", "Deep critique:", "\nFast critic:", "Fast critic:",
                "\nFail rubrics:", "\nNext action:")

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


def build_target_classification(step: dict, rubric_tokens: dict | None = None) -> str:
    """Classification model target: Fast critic + Deep critic + Fail rubrics.
    Next action은 룰 기반으로 결정되므로 target에서 제외.
    """
    parts = []

    def _verdict(raw: str) -> str:
        v = (raw or "").lower()
        return "incorrect" if v in ("incorrect", "fail") else "correct"

    # fast critic
    fast = step.get("prm_fast_critique") or {}
    parts.append("Fast critic:")
    for rubric, data in fast.items():
        raw = data.get("verdict", "correct")
        critique = data.get("critique") or ""
        line = f"  {rubric}: {_verdict(raw)}"
        if _verdict(raw) == "incorrect":
            critique_text = critique.strip()
            if critique_text and not _NUMERIC_VERDICT.match(critique_text):
                line += f" — {critique_text}"
        parts.append(f"\n{line}")

    # deep critic
    fast_fails = {r for r, d in fast.items() if _verdict(d.get("verdict") or "") == "incorrect"}
    deep = step.get("prm_deep_critique") or []
    deep_items = [d for d in deep if d.get("rubric") in fast_fails and d.get("verdict") is not None]
    parts.append("\n\nDeep critic:")
    if deep_items:
        for d in deep_items:
            rubric = d.get("rubric", "")
            raw = d.get("verdict") or ""
            critique = d.get("critique") or ""
            critique_clean = (_strip_newlines(critique)
                              if critique.strip() and not _NUMERIC_VERDICT.match(critique.strip())
                              else "")
            line = (f"  {rubric}: {critique_clean} Verdict: {_verdict(raw)}"
                    if critique_clean else f"  {rubric}: Verdict: {_verdict(raw)}")
            parts.append(f"\n{line}")
    else:
        parts.append("\n  none")

    # fail rubrics — 없으면 섹션 헤더만 출력 (TOKEN_NONE 제거)
    _rubric_tokens = rubric_tokens or {}
    _gfr = step.get("gold_fail_rubrics")
    _gfr_list = _gfr if isinstance(_gfr, list) else []
    fail_tokens = [_rubric_tokens[r] for r in _gfr_list if r in _rubric_tokens]
    parts.append("\n\nFail rubrics:\n" + "\n".join(fail_tokens))

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
            is_error = step.get("is_error", False)

            if skip_error and is_error:
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

            if is_error:
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
    is_error=True 샘플은 skip_error=True 시 제외.
    traj_id 필드가 있으면 TrajectoryOrderedSampler 사용 가능.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 3072,
        skip_error: bool = False,
        action_weight_map: dict | None = None,
        rubric_weight_map: dict | None = None,
    ):
        self.samples  = []
        self.traj_ids = []   # traj_id per sample (None if not in data)

        raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"[PreprocessedDataset] {len(raw)}개 샘플 로드, 토크나이징 중...")

        skipped = 0
        action_parse_skipped = 0
        rubric_parse_skipped = 0
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

            if len(full_ids) > max_length:
                skipped += 1
                continue

            full_ids = full_ids[:max_length]

            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels    = torch.full_like(input_ids, -100)
            labels[prefix_len:] = input_ids[prefix_len:]

            if is_error:
                mask_inference_for_error(labels, prefix_len, tokenizer, prefix_str, target_str)

            # 역빈도 가중치: action + rubric 동시 적용 가능
            token_weights = None
            if action_weight_map or rubric_weight_map:
                tw = torch.ones(len(full_ids), dtype=torch.float32)

                if action_weight_map:
                    result = _find_next_action_pos(target_str, labels, prefix_len,
                                                   tokenizer, action_weight_map)
                    if result is None:
                        action_parse_skipped += 1
                        # 잘림으로 토큰 위치를 못 찾은 경우 — 샘플은 유효하므로 weight 없이 유지
                    else:
                        pos, tid = result
                        tw[pos] = action_weight_map[tid]

                if rubric_weight_map:
                    rubric_result = _find_rubric_positions(target_str, labels, prefix_len,
                                                           tokenizer, rubric_weight_map)
                    if rubric_result is None:
                        rubric_parse_skipped += 1
                        # 잘림으로 토큰 위치를 못 찾은 경우 — 샘플은 유효하므로 focal 없이 유지
                    else:
                        for pos, tid, _ in rubric_result:
                            tw[pos] = FOCAL_SENTINEL   # focal loss 적용 마킹

                if not tw.eq(1.0).all():
                    token_weights = tw

            self.samples.append((input_ids, labels, token_weights))
            self.traj_ids.append(item.get("traj_id"))   # None if old-format data

        if rank == 0:
            print(f"[PreprocessedDataset] 학습 샘플: {len(self.samples)}  (제외: {skipped}, max_length={max_length} 초과 포함)")
            if action_weight_map:
                print(f"[PreprocessedDataset] Next action 가중치 미적용 (잘림): {action_parse_skipped}개")
            if rubric_weight_map:
                print(f"[PreprocessedDataset] Fail rubrics 가중치 미적용 (잘림): {rubric_parse_skipped}개")

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
    """데이터 파일을 읽어 (idx, msgs, target_str, state, is_error, rubric_tokens) 리스트 반환."""
    raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    first = raw[0] if raw else {}
    is_preprocessed = "input" in first and "target" in first

    if is_preprocessed:
        result = []
        for i, item in enumerate(raw):
            result.append((i, item["input"], item["target"],
                           item.get("state"), item.get("is_error"), {}))
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
                result.append((idx, msgs, target_str,
                               step.get("state"), step.get("is_error"), RUBRIC_TOKENS))
                idx += 1
        return result


def _print_sample(tokenizer, idx, msgs, target_str, state, is_error):
    sep = "─" * 72
    full_msgs  = msgs + [{"role": "assistant", "content": target_str}]
    full_str   = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prefix_str = tokenizer.apply_chat_template(msgs,      tokenize=False, add_generation_prompt=True)

    p_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
    f_ids = tokenizer.encode(full_str,   add_special_tokens=False)
    t_len = len(f_ids) - len(p_ids)

    print(f"\n{'='*72}")
    print(f"[샘플 {idx}  state={state}  is_error={is_error}]")
    print(f"\n[INPUT — {len(p_ids)} tok]\n{sep}")
    for msg in msgs:
        role    = msg["role"]
        content = msg["content"]
        print(f"\n<{role}>")
        print(content)
    print(f"\n[TARGET — {t_len} tok]\n{sep}")
    print(target_str)
    print(f"\n토큰: input={len(p_ids)}  target={t_len}  total={len(f_ids)}")


def debug(data_path: str, tokenizer, n: int | None = None):
    """
    n=None : is_error=False 샘플 하나 자동 출력
    n=int  : n번째 샘플 출력
    """
    samples = _build_samples(data_path)

    if n is not None:
        if n >= len(samples):
            print(f"[debug] 인덱스 {n}이 범위를 초과했습니다. (총 {len(samples)}개)")
            return
        idx, msgs, target_str, state, is_error, _ = samples[n]
        _print_sample(tokenizer, idx, msgs, target_str, state, is_error)
        return

    # 자동 모드: is_error=False 첫 번째 샘플 출력
    for idx, msgs, target_str, state, is_error, _ in samples:
        if not is_error:
            _print_sample(tokenizer, idx, msgs, target_str, state, is_error)
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
