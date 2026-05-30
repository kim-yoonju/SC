"""
evaluate_classification.py

sft_classification 모델을 평가하자

classification model이 inference를 입력으로 받아
fail_rubrics, next_action을 얼마나 잘 예측하는지 평가한다.

실행 예시:

python source/eval_sft_classification.py \
--data_path /mnt/yoonju/SC/output/sft_trajectory/traj_cls_eval_50_with_history.jsonl \
--classification_model /mnt/yoonju/SC/checkpoints/sft/20260522_121600_clss_001/epoch3 --gpus 4,5


출력 (output/eval_classification/{timestamp}/):
    predictions.jsonl  스텝별 gold vs pred 비교
    summary.json       전체 메트릭 요약
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import os

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from preprocess import get_classification_prompt, RUBRIC_TOKENS
from utils_sft import (
    build_messages_classification, setup_tokenizer,
    TOKEN_SOLVE, TOKEN_RETHINK, TOKEN_END, CONF,
)
from utils import has_boxed

_ROOT = Path(__file__).resolve().parent.parent

# token string → rubric name 역매핑
_TOKEN_TO_RUBRIC: dict[str, str] = {v: k for k, v in RUBRIC_TOKENS.items()}

# 모든 action token → 정규 레이블
_ACTION_LABEL = {
    TOKEN_SOLVE:   "solve",
    TOKEN_RETHINK: "rethink",
    TOKEN_END:     "end",
    "<|solve|>":   "solve",
    "<|rethink|>": "rethink",
    "<|end|>":     "end",
    "solve":       "solve",
    "rethink":     "rethink",
    "end":         "end",
}


# ─────────────────────────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, gpu_id: int):
    from transformers import AutoModelForCausalLM
    tokenizer = setup_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{gpu_id}"},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY_RUBRIC_MARKERS = {"None", "<|none|>", "none", ""}


def _normalize_gold_fr(val) -> list[str]:
    """gold_fail_rubrics 값을 list[str]로 정규화. none 마커는 빈 리스트로."""
    if not val:
        return []
    if isinstance(val, str):
        val = [val]
    return [r for r in val if r.strip() not in _EMPTY_RUBRIC_MARKERS]


def load_samples(data_path: str, skip_error: bool = True,
                 num_start: int | None = None, num_end: int | None = None) -> list[dict]:
    """traj_all.jsonl 또는 flat 스텝 레코드(k 필드 포함)에서 평가 샘플 추출."""
    raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    raw = raw[num_start:num_end]

    # history + flat 현재 스텝 형식 감지 (history 필드가 최상위에 있으면)
    if raw and "history" in raw[0] and "steps" not in raw[0]:
        samples = []
        for rec in raw:
            gold_fr = _normalize_gold_fr(rec.get("gold_fail_rubrics"))
            next_action = rec.get("next_gold_action") or TOKEN_SOLVE
            if not gold_fr and next_action == TOKEN_RETHINK:
                continue
            # history(is_fail=False 스텝들) + 현재 스텝으로 steps 재구성
            current_step = {
                "step_idx":          rec["step_idx"],
                "step":              rec.get("step", ""),
                "inference":         rec.get("inference") or rec.get("text") or "",
                "source":            rec.get("source", ""),
                "is_fail":          rec.get("is_fail", False),
                "state":             rec.get("state", ""),
                "gold_fail_rubrics": gold_fr,
                "next_gold_action":  next_action,
                "does":              rec.get("does", ""),
                "prm_deep_critique": rec.get("prm_deep_critique"),
                "prm_critique_summary": rec.get("prm_critique_summary"),
            }
            history = rec.get("history") or []
            steps = history + [current_step]
            k = len(history)
            samples.append({
                "problem":           rec["problem"],
                "steps":             steps,
                "k":                 k,
                "gold_fail_rubrics": gold_fr,
                "next_gold_action":  next_action,
                "problem_id":        rec.get("problem_id", ""),
                "step_idx":          rec["step_idx"],
                "state":             rec.get("state", ""),
                "is_right":          rec.get("is_right", None),
            })
        return samples

    # flat 스텝 레코드 형식 감지 (k 필드가 최상위에 있으면)
    if raw and "k" in raw[0]:
        samples = []
        for rec in raw:
            gold_fr = _normalize_gold_fr(rec.get("gold_fail_rubrics"))
            next_action = rec.get("next_gold_action") or TOKEN_SOLVE
            if not gold_fr and next_action == TOKEN_RETHINK:
                continue
            samples.append({
                "problem":           rec["problem"],
                "steps":             rec["steps"],
                "k":                 rec["k"],
                "gold_fail_rubrics": gold_fr,
                "next_gold_action":  next_action,
                "problem_id":        rec.get("problem_id", ""),
                "step_idx":          rec.get("step_idx", rec["k"]),
                "state":             rec.get("state", ""),
                "is_right":          rec.get("is_right", None),
            })
        return samples

    samples = []
    for traj in raw:
        problem = traj["problem"]
        steps   = traj["steps"]
        for k, step in enumerate(steps):
            if skip_error and step.get("is_fail", False):
                continue
            # gold_fail_rubrics가 없으면 건너뜀
            if "gold_fail_rubrics" not in step:
                continue
            gold_fr = _normalize_gold_fr(step.get("gold_fail_rubrics"))
            next_action = step.get("next_gold_action") or TOKEN_SOLVE
            # fail rubric 없는데 rethink인 노이즈 데이터 제거
            if not gold_fr and next_action == TOKEN_RETHINK:
                continue
            samples.append({
                "problem":          problem,
                "steps":            steps,
                "k":                k,
                "gold_fail_rubrics": gold_fr,
                "next_gold_action":  next_action,
                "problem_id":        traj.get("problem_id", ""),
                "step_idx":          step.get("step_idx", k),
                "state":             step.get("state", ""),
                "is_right":          traj.get("is_right", None),
            })
    return samples


def balance_none_samples(samples: list[dict], seed: int = 42) -> list[dict]:
    """none 샘플(gold_fail_rubrics=[])을 가장 많은 비-none 루브릭 클래스 수로 다운샘플링."""
    none_samples     = [s for s in samples if not s["gold_fail_rubrics"]]
    non_none_samples = [s for s in samples if s["gold_fail_rubrics"]]

    if not non_none_samples or not none_samples:
        return samples

    rubric_counts: Counter = Counter()
    for s in non_none_samples:
        for rubric in s["gold_fail_rubrics"]:
            rubric_counts[rubric] += 1

    max_count = max(rubric_counts.values())
    orig_none = len(none_samples)

    if orig_none > max_count:
        none_samples = random.Random(seed).sample(none_samples, max_count)
        top5 = dict(rubric_counts.most_common(5))
        print(f"none 샘플 다운샘플링: {orig_none} → {max_count}  (최대 루브릭={max_count}, top5={top5})")

    return non_none_samples + none_samples


# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트 빌드
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(tokenizer, system: str, problem: str, steps: list, k: int) -> str:
    sys_str, user_str = build_messages_classification(problem, steps, k, system)
    messages = [
        {"role": "system", "content": sys_str},
        {"role": "user",   "content": user_str},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─────────────────────────────────────────────────────────────────────────────
# 생성
# ─────────────────────────────────────────────────────────────────────────────

def vllm_generate(llm, prompts: list[str], max_new_tokens: int) -> list[str]:
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sp)
    return [o.outputs[0].text.strip() for o in outputs]


@torch.no_grad()
def batch_generate(model, tokenizer, prompts: list[str],
                   max_new_tokens: int = 1024) -> list[str]:
    """배치 생성. 길이 내림차순 정렬 후 처리해 padding 낭비를 최소화한다.
    루브릭/액션 special token이 EOS로 처리되어 잘리는 현상을 방지하기 위해
    custom special token ID를 EOS에서 명시적으로 제외한다."""
    if not prompts:
        return []

    # custom special token ID (루브릭/액션) 를 EOS에서 명시적으로 제외
    _custom_ids = {
        tokenizer.convert_tokens_to_ids(t)
        for t in getattr(tokenizer, "additional_special_tokens", [])
    } - {None}
    _raw_eos = (getattr(model.generation_config, "eos_token_id", None)
                or tokenizer.eos_token_id)
    if isinstance(_raw_eos, int):
        _eos = _raw_eos if _raw_eos not in _custom_ids else tokenizer.eos_token_id
    else:
        _eos = [x for x in _raw_eos if x not in _custom_ids] or tokenizer.eos_token_id

    # 길이 내림차순 정렬 (padding 낭비 최소화), 원래 인덱스 보존
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    sorted_prompts = [prompts[i] for i in order]

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(sorted_prompts, return_tensors="pt", padding=True).to(model.device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=_eos,
        repetition_penalty=1.3,
    )

    # 원래 순서로 복원
    sorted_results = []
    for i in range(len(sorted_prompts)):
        resp = out[i, input_len:]
        text = tokenizer.decode(resp, skip_special_tokens=False).strip()
        sorted_results.append(text)

    results = [None] * len(prompts)
    for rank, orig_idx in enumerate(order):
        results[orig_idx] = sorted_results[rank]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 파싱
# ─────────────────────────────────────────────────────────────────────────────


def _is_real_rubric(name: str) -> bool:
    return name.strip() not in _EMPTY_RUBRIC_MARKERS


def parse_fail_rubrics(text: str) -> list[str]:
    """Fail rubrics 섹션에서 루브릭 이름 목록 추출."""
    m = re.search(r"Fail\s+rubrics\s*:", text, re.IGNORECASE)
    if not m:
        return []
    section = text[m.end():]
    eos = section.find("<|im_end|>")
    if eos != -1:
        section = section[:eos]
    section = section.strip()
    if not section or section.lower().rstrip(".") in ("none", "<|none|>"):
        return []
    tokens = re.findall(r"<\|[^|>]+\|>", section)
    result = []
    for tok in tokens:
        if tok == "<|none|>":
            continue  # none 마커는 건너뜀
        name = _TOKEN_TO_RUBRIC.get(tok)
        if name and _is_real_rubric(name):
            result.append(name)
    return result


def parse_fail_rubrics_from_deep_critique(text: str) -> list[str]:
    """Deep critic 섹션에서 Verdict: incorrect 라인의 루브릭 이름 목록 추출."""
    m = re.search(r"Deep\s+critic\s*:", text, re.IGNORECASE)
    if not m:
        return []
    section = text[m.end():]
    end = re.search(r"\n\n(Fail\s+rubrics|Next\s+action)\s*:", section, re.IGNORECASE)
    if end:
        section = section[:end.start()]
    result = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"verdict:\s*incorrect", stripped, re.IGNORECASE):
            rubric = re.split(r"\s*:", stripped, maxsplit=1)[0].strip()
            if rubric and _is_real_rubric(rubric):
                result.append(rubric)
    return result


def parse_per_rubric_verdicts(text: str) -> dict[str, str]:
    """Deep critic 섹션에서 루브릭별 verdict 파싱. {rubric_name: 'correct'/'incorrect'}"""
    m = re.search(r"Deep\s+critic\s*:", text, re.IGNORECASE)
    if not m:
        return {}
    section = text[m.end():]
    end = re.search(r"\n\n(Fail\s+rubrics|Next\s+action)\s*:", section, re.IGNORECASE)
    if end:
        section = section[:end.start()]
    result = {}
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        verdict_m = re.search(r"verdict:\s*(correct|incorrect)", stripped, re.IGNORECASE)
        if not verdict_m:
            continue
        verdict = verdict_m.group(1).lower()
        rubric = re.split(r"\s*:", stripped, maxsplit=1)[0].strip()
        if rubric and _is_real_rubric(rubric):
            result[rubric] = verdict
    return result


def parse_next_action(text: str) -> str | None:
    """Next action 섹션에서 액션 토큰을 직접 파싱."""
    m = re.search(r"Next\s+action\s*:\n(.+)", text, re.IGNORECASE)
    if not m:
        return None
    return _ACTION_LABEL.get(m.group(1).strip())


def rule_based_action(pred_rubrics: list[str], inference: str) -> str:
    """룰 기반 next action 결정.
    - fail rubrics 있음 → rethink
    - 없음 + boxed{} 있음 → end
    - 없음 + boxed{} 없음 → solve
    """
    effective = [r for r in pred_rubrics if _is_real_rubric(r)]
    if effective:
        return TOKEN_RETHINK
    if has_boxed(inference):
        return TOKEN_END
    return TOKEN_SOLVE


# ─────────────────────────────────────────────────────────────────────────────
# 메트릭 계산
# ─────────────────────────────────────────────────────────────────────────────

def action_label(s: str | None) -> str | None:
    if s is None:
        return None
    return _ACTION_LABEL.get(s.strip())


def jaccard(gold: set, pred: set) -> float:
    if not gold and not pred:
        return 1.0
    union = gold | pred
    inter = gold & pred
    return len(inter) / len(union) if union else 1.0


def compute_metrics(records: list[dict], pred_key: str = "pred_rubrics") -> dict:
    """
    records: [{"gold_action", "pred_action", "gold_rubrics", pred_key, "state"}, ...]
    pred_key: 예측 rubric 리스트가 담긴 필드명
    """
    action_pairs = []
    rubric_exact = []
    rubric_jaccard = []
    rubric_tp = rubric_fp = rubric_fn = 0
    action_none_count = 0

    for r in records:
        ga = action_label(r["gold_action"])
        pa = action_label(r["pred_action"])
        if pa is None:
            action_none_count += 1
            pa = "solve"  # fallback
        action_pairs.append({"gold": ga, "pred": pa, "state": r.get("state", "")})

        gold_set = set(r["gold_rubrics"])
        pred_set = set(r.get(pred_key, []))
        rubric_exact.append(gold_set == pred_set)
        rubric_jaccard.append(jaccard(gold_set, pred_set))

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        rubric_tp += tp
        rubric_fp += fp
        rubric_fn += fn

    # ── Action metrics ────────────────────────────────────────────────────────
    labels = ["solve", "rethink", "end"]
    tp_c = Counter()
    fp_c = Counter()
    fn_c = Counter()
    for p in action_pairs:
        g, pred = p["gold"], p["pred"]
        if pred == g:
            tp_c[g] += 1
        else:
            fn_c[g] += 1
            fp_c[pred] += 1

    action_accuracy = sum(tp_c.values()) / len(action_pairs) if action_pairs else 0.0
    action_rows = []
    for label in labels:
        support = tp_c[label] + fn_c[label]
        prec = tp_c[label] / (tp_c[label] + fp_c[label]) if (tp_c[label] + fp_c[label]) > 0 else 0.0
        rec  = tp_c[label] / (tp_c[label] + fn_c[label]) if (tp_c[label] + fn_c[label]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        action_rows.append({"label": label, "precision": prec, "recall": rec, "f1": f1, "support": support})

    action_macro_f1 = sum(r["f1"] for r in action_rows) / len(action_rows) if action_rows else 0.0

    # ── Rubric metrics ────────────────────────────────────────────────────────
    micro_prec = rubric_tp / (rubric_tp + rubric_fp) if (rubric_tp + rubric_fp) > 0 else 0.0
    micro_rec  = rubric_tp / (rubric_tp + rubric_fn) if (rubric_tp + rubric_fn) > 0 else 0.0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    n_none_gold = sum(1 for r in records if not r["gold_rubrics"])
    n_none_pred = sum(1 for r in records if not r.get(pred_key, []))
    none_match  = sum(1 for r in records if not r["gold_rubrics"] and not r.get(pred_key, []))
    none_acc    = none_match / n_none_gold if n_none_gold else None

    # ── Per-rubric class metrics ──────────────────────────────────────────────
    all_rubric_labels = sorted(set(
        rb for rec in records for rb in (rec["gold_rubrics"] + rec.get(pred_key, []))
    ))
    rubric_tp_c: Counter = Counter()
    rubric_fp_c: Counter = Counter()
    rubric_fn_c: Counter = Counter()
    rubric_support: Counter = Counter()
    for rec in records:
        gold_set = set(rec["gold_rubrics"])
        pred_set = set(rec.get(pred_key, []))
        for lbl in all_rubric_labels:
            in_gold = lbl in gold_set
            in_pred = lbl in pred_set
            if in_gold:
                rubric_support[lbl] += 1
            if in_gold and in_pred:
                rubric_tp_c[lbl] += 1
            elif in_pred and not in_gold:
                rubric_fp_c[lbl] += 1
            elif in_gold and not in_pred:
                rubric_fn_c[lbl] += 1

    rubric_rows = []
    for lbl in all_rubric_labels:
        tp = rubric_tp_c[lbl]
        fp = rubric_fp_c[lbl]
        fn = rubric_fn_c[lbl]
        sup = rubric_support[lbl]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_  = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
        rubric_rows.append({"label": lbl, "precision": prec, "recall": rec_, "f1": f1_, "support": sup})

    rubric_macro_f1 = sum(r["f1"] for r in rubric_rows) / len(rubric_rows) if rubric_rows else 0.0
    supported_rows = [r for r in rubric_rows if r["support"] > 0]
    rubric_macro_f1_supported = (
        sum(r["f1"] for r in supported_rows) / len(supported_rows) if supported_rows else 0.0
    )

    return {
        "n_samples":           len(records),
        "action_none_count":   action_none_count,
        # action
        "action_accuracy":     action_accuracy,
        "action_macro_f1":     action_macro_f1,
        "action_rows":         action_rows,
        "action_label_dist":   {l: sum(1 for p in action_pairs if p["gold"] == l) for l in labels},
        # rubric set-level
        "rubric_exact_match":  sum(rubric_exact) / len(rubric_exact) if rubric_exact else 0.0,
        "rubric_avg_jaccard":  sum(rubric_jaccard) / len(rubric_jaccard) if rubric_jaccard else 0.0,
        "rubric_micro_prec":   micro_prec,
        "rubric_micro_rec":    micro_rec,
        "rubric_micro_f1":     micro_f1,
        "n_none_gold":         n_none_gold,
        "n_none_pred":         n_none_pred,
        "none_step_accuracy":  none_acc,
        # rubric per-class
        "rubric_rows":                rubric_rows,
        "rubric_macro_f1":            rubric_macro_f1,
        "rubric_macro_f1_supported":  rubric_macro_f1_supported,
    }


def compute_per_rubric_binary_metrics(
    records: list[dict],
    all_rubrics: list[str],
    pred_verdicts_key: str = "pred_verdicts",
) -> dict:
    """
    루브릭별 binary classification metrics (incorrect 클래스 기준).
    gold: rubric in gold_rubrics → incorrect, 아니면 → correct.
    pred: pred_verdicts.get(rubric, 'correct').
    """
    rubric_tp: Counter = Counter()
    rubric_fp: Counter = Counter()
    rubric_tn: Counter = Counter()
    rubric_fn: Counter = Counter()
    rubric_missing: Counter = Counter()

    for rec in records:
        gold_fail = set(rec["gold_rubrics"])
        pred_verdicts = rec.get(pred_verdicts_key, {})
        for rubric in all_rubrics:
            gold = "incorrect" if rubric in gold_fail else "correct"
            if rubric in pred_verdicts:
                pred = pred_verdicts[rubric]
            else:
                pred = "correct"
                rubric_missing[rubric] += 1
            if gold == "incorrect" and pred == "incorrect":
                rubric_tp[rubric] += 1
            elif gold == "correct" and pred == "incorrect":
                rubric_fp[rubric] += 1
            elif gold == "correct" and pred == "correct":
                rubric_tn[rubric] += 1
            else:
                rubric_fn[rubric] += 1

    n = len(records)
    rubric_rows = []
    for rubric in all_rubrics:
        tp = rubric_tp[rubric]
        fp = rubric_fp[rubric]
        tn = rubric_tn[rubric]
        fn = rubric_fn[rubric]
        acc  = (tp + tn) / n if n > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
        rubric_rows.append({
            "rubric":            rubric,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy":          acc,
            "precision":         prec,
            "recall":            rec_,
            "f1":                f1,
            "support_incorrect": tp + fn,
            "missing":           rubric_missing[rubric],
            "parsed":            n - rubric_missing[rubric],
        })

    n_total = n * len(all_rubrics)
    overall_acc = sum(r["tp"] + r["tn"] for r in rubric_rows) / n_total if n_total > 0 else 0.0
    macro_f1    = sum(r["f1"] for r in rubric_rows) / len(rubric_rows) if rubric_rows else 0.0
    coverage    = sum(1 - r["missing"] / n for r in rubric_rows) / len(rubric_rows) if rubric_rows and n > 0 else 0.0

    return {
        "n_samples":             n,
        "rubric_binary_rows":    rubric_rows,
        "overall_binary_accuracy": overall_acc,
        "binary_macro_f1":       macro_f1,
        "avg_rubric_coverage":   coverage,
    }


def compute_multiclass_with_none(
    records: list[dict],
    all_rubrics: list[str],
    pred_key: str = "pred_rubrics_critique",
) -> dict:
    """11개 루브릭 + None 클래스 포함 multi-class metrics (one-vs-rest 방식).
    None 클래스: gold_rubrics=[] → gold=None, pred_rubrics=[] → pred=None."""
    classes = ["None"] + list(all_rubrics)
    tp_c: Counter = Counter()
    fp_c: Counter = Counter()
    fn_c: Counter = Counter()
    support: Counter = Counter()

    for rec in records:
        gold_set  = set(rec["gold_rubrics"])
        pred_set  = set(rec.get(pred_key, []))
        gold_none = len(gold_set) == 0
        pred_none = len(pred_set) == 0

        if gold_none:
            support["None"] += 1
        if gold_none and pred_none:
            tp_c["None"] += 1
        elif not gold_none and pred_none:
            fp_c["None"] += 1
        elif gold_none and not pred_none:
            fn_c["None"] += 1

        for rubric in all_rubrics:
            in_gold = rubric in gold_set
            in_pred = rubric in pred_set
            if in_gold:
                support[rubric] += 1
            if in_gold and in_pred:
                tp_c[rubric] += 1
            elif in_pred and not in_gold:
                fp_c[rubric] += 1
            elif in_gold and not in_pred:
                fn_c[rubric] += 1

    rows = []
    for cls in classes:
        t   = tp_c[cls]; f_p = fp_c[cls]; f_n = fn_c[cls]
        sup = support[cls]
        prec = t / (t + f_p) if (t + f_p) > 0 else 0.0
        rec_ = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1   = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
        rows.append({"class": cls, "tp": t, "fp": f_p, "fn": f_n,
                     "precision": prec, "recall": rec_, "f1": f1, "support": sup})

    macro_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    supported = [r for r in rows if r["support"] > 0]
    macro_f1_supported = sum(r["f1"] for r in supported) / len(supported) if supported else 0.0
    return {
        "n_samples": len(records),
        "rows": rows,
        "macro_f1": macro_f1,
        "macro_f1_supported": macro_f1_supported,
    }


def print_multiclass_with_none(mc_metrics: dict, title: str = "Multi-class (None 포함)"):
    sep = "=" * 80
    n = mc_metrics.get("n_samples", 0)
    print(f"\n{sep}")
    print(f"  {title}  (총 샘플: {n})")
    print(sep)
    print(f"  Macro F1             : {mc_metrics['macro_f1']:.4f}")
    print(f"  Macro F1 (support>0) : {mc_metrics['macro_f1_supported']:.4f}")

    rows = mc_metrics["rows"]
    if not rows:
        return
    col = max(max(len(r["class"]) for r in rows), 5)
    print(f"\n  {'class':<{col}}  {'prec':>7}  {'rec':>7}  {'f1':>7}  {'support':>8}  {'tp':>6}  {'fp':>6}  {'fn':>6}")
    print("  " + "-" * (col + 62))
    for r in rows:
        print(f"  {r['class']:<{col}}  {r['precision']:>7.4f}  {r['recall']:>7.4f}  {r['f1']:>7.4f}"
              f"  {r['support']:>8}  {r['tp']:>6}  {r['fp']:>6}  {r['fn']:>6}")


def print_per_rubric_binary_metrics(binary_metrics: dict, title: str = "Per-rubric Binary"):
    sep = "=" * 80
    n = binary_metrics.get("n_samples", 0)
    print(f"\n{sep}")
    print(f"  {title}  (총 샘플: {n})")
    print(sep)
    print(f"  Overall binary accuracy : {binary_metrics['overall_binary_accuracy']:.4f}")
    print(f"  Macro F1 (incorrect cls): {binary_metrics['binary_macro_f1']:.4f}")
    print(f"  파싱 비율 (루브릭 커버리지): {binary_metrics['avg_rubric_coverage']:.4f}")

    rows = binary_metrics["rubric_binary_rows"]
    if not rows:
        return
    col = max(max(len(r["rubric"]) for r in rows), 6)
    print(f"\n  {'rubric':<{col}}  {'acc':>7}  {'prec':>7}  {'rec':>7}  {'f1':>7}  "
          f"{'support':>8}  {'parsed':>8}  {'missing':>8}")
    print("  " + "-" * (col + 68))
    for r in rows:
        print(f"  {r['rubric']:<{col}}  {r['accuracy']:>7.4f}  {r['precision']:>7.4f}"
              f"  {r['recall']:>7.4f}  {r['f1']:>7.4f}  {r['support_incorrect']:>8}"
              f"  {r['parsed']:>8}  {r['missing']:>8}")


def confusion_matrix_str(pairs: list[dict], labels: list[str]) -> str:
    idx = {l: i for i, l in enumerate(labels)}
    n   = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p in pairs:
        gi = idx.get(p["gold"])
        pi = idx.get(p["pred"])
        if gi is not None and pi is not None:
            mat[gi][pi] += 1
    col_w  = max(len(l) for l in labels) + 2
    header = " " * (col_w + 2) + "  ".join(f"{l:>{col_w}}" for l in labels)
    lines  = [header, " " * (col_w + 2) + "  ".join("-" * col_w for _ in labels)]
    for i, gl in enumerate(labels):
        row_str = "  ".join(f"{mat[i][j]:>{col_w}}" for j in range(n))
        lines.append(f"{gl:>{col_w}}  {row_str}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_rubric_comparison_table(m_token: dict, m_critique: dict):
    """두 채점 방식(special token vs deep critique verdict)의 rubric 성능 비교 테이블."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  Fail Rubrics 채점 방식 비교")
    print(sep)

    col = 20
    h1, h2 = "special token", "deep critique"
    print(f"\n  {'':>{col}}  {'prec':>8}  {'rec':>8}  {'f1':>8}  {'exact':>8}  {'jaccard':>8}")
    print("  " + "-" * (col + 48))

    def row(label, m):
        print(f"  {label:>{col}}  "
              f"{m['rubric_micro_prec']:>8.4f}  "
              f"{m['rubric_micro_rec']:>8.4f}  "
              f"{m['rubric_micro_f1']:>8.4f}  "
              f"{m['rubric_exact_match']:>8.4f}  "
              f"{m['rubric_avg_jaccard']:>8.4f}")

    row(h1, m_token)
    row(h2, m_critique)

    # per-rubric 비교
    all_labels = sorted(set(
        r["label"] for r in m_token["rubric_rows"] + m_critique["rubric_rows"]
    ))
    if not all_labels:
        return

    lbl_col = max(max(len(l) for l in all_labels), 5)
    print(f"\n  Per-rubric F1 비교:")
    print(f"  {'rubric':<{lbl_col}}  {'support':>8}  {h1:>14}  {h2:>14}")
    print("  " + "-" * (lbl_col + 42))

    tok_map  = {r["label"]: r for r in m_token["rubric_rows"]}
    crit_map = {r["label"]: r for r in m_critique["rubric_rows"]}
    for lbl in all_labels:
        tok_f1  = tok_map[lbl]["f1"]  if lbl in tok_map  else 0.0
        crit_f1 = crit_map[lbl]["f1"] if lbl in crit_map else 0.0
        sup     = tok_map[lbl]["support"] if lbl in tok_map else crit_map.get(lbl, {}).get("support", 0)
        better  = "▲" if tok_f1 > crit_f1 else ("▼" if tok_f1 < crit_f1 else " ")
        print(f"  {lbl:<{lbl_col}}  {sup:>8}  {tok_f1:>13.4f}{better}  {crit_f1:>14.4f}")

    print(f"\n  Macro F1  →  {h1}: {m_token['rubric_macro_f1']:.4f}  |  {h2}: {m_critique['rubric_macro_f1']:.4f}")
    print(f"  None-step acc  →  {h1}: "
          + (f"{m_token['none_step_accuracy']:.4f}" if m_token['none_step_accuracy'] is not None else "N/A")
          + f"  |  {h2}: "
          + (f"{m_critique['none_step_accuracy']:.4f}" if m_critique['none_step_accuracy'] is not None else "N/A"))


def compute_action_metrics(records: list[dict], pred_action_key: str = "pred_action_verdict") -> dict:
    labels = ["solve", "rethink", "end"]
    tp_c = Counter()
    fp_c = Counter()
    fn_c = Counter()
    for r in records:
        g = action_label(r["gold_action"]) or "solve"
        p = action_label(r.get(pred_action_key)) or "solve"
        if p == g:
            tp_c[g] += 1
        else:
            fn_c[g] += 1
            fp_c[p] += 1
    n = len(records)
    accuracy = sum(tp_c.values()) / n if n else 0.0
    rows = []
    for label in labels:
        support = tp_c[label] + fn_c[label]
        prec = tp_c[label] / (tp_c[label] + fp_c[label]) if (tp_c[label] + fp_c[label]) > 0 else 0.0
        rec  = tp_c[label] / (tp_c[label] + fn_c[label]) if (tp_c[label] + fn_c[label]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({"label": label, "precision": prec, "recall": rec, "f1": f1, "support": support})
    macro_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    return {"n": n, "accuracy": accuracy, "macro_f1": macro_f1, "rows": rows}


def print_action_table(action_metrics: dict, title: str = "Next Action (verdict 기반 룰)"):
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  {title}  (n={action_metrics['n']})")
    print(sep)
    print(f"  Accuracy : {action_metrics['accuracy']:.4f}  "
          f"({int(action_metrics['accuracy'] * action_metrics['n'])}/{action_metrics['n']})")
    print(f"  Macro F1 : {action_metrics['macro_f1']:.4f}")
    col = 8
    print(f"\n  {'label':<{col}}  {'prec':>8}  {'rec':>8}  {'f1':>8}  {'support':>8}")
    print("  " + "-" * (col + 38))
    for r in action_metrics["rows"]:
        print(f"  {r['label']:<{col}}  {r['precision']:>8.4f}  {r['recall']:>8.4f}"
              f"  {r['f1']:>8.4f}  {r['support']:>8}")


def print_metrics(metrics: dict, title: str = "전체"):
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  {title}  (n={metrics['n_samples']})")
    print(sep)
    print(f"  Exact match  : {metrics['rubric_exact_match']:.4f}")
    print(f"  Avg Jaccard  : {metrics['rubric_avg_jaccard']:.4f}")
    print(f"  Macro F1     : {metrics['rubric_macro_f1_supported']:.4f}")




# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classification 모델 평가")
    parser.add_argument("--data_path",            required=True, help="traj_all.jsonl 경로")
    parser.add_argument("--classification_model", required=True, help="모델 경로")
    parser.add_argument("--gpus",        type=str, default="0",  help="GPU 번호 (단일: 0, 다중: 0,1,2,3)")
    parser.add_argument("--batch_size",  type=int, default=8,    help="GPU당 배치 크기")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--output",      type=str, default=None, help="출력 폴더 (기본: output/eval_classification/{ts})")
    parser.add_argument("--skip_error", action="store_true",  help="is_fail=True 스텝 제외")
    parser.add_argument("--by_state",    action="store_true",    help="state별 breakdown 출력")
    parser.add_argument("--limit",       type=int, default=None, help="평가할 최대 샘플 수 (디버그용)")
    parser.add_argument("--num_start",   type=int, default=None, help="시작 인덱스 (inclusive)")
    parser.add_argument("--num_end",     type=int, default=None, help="종료 인덱스 (exclusive)")
    parser.add_argument("--balance_none", action="store_true",
                        help="none 샘플(fail rubric 없는 스텝)을 가장 많은 루브릭 클래스 수로 다운샘플링")
    args = parser.parse_args()

    gpu_list = [int(g.strip()) for g in args.gpus.split(",")]

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else (_ROOT / "output" / "eval_classification" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{ts}] 모델: {args.classification_model}")
    print(f"데이터: {args.data_path}")
    print(f"GPU   : {gpu_list}")
    print(f"출력  : {out_dir}")

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    samples = load_samples(args.data_path, skip_error=args.skip_error,
                           num_start=args.num_start, num_end=args.num_end)
    if args.balance_none:
        samples = balance_none_samples(samples)
    if args.limit:
        samples = samples[:args.limit]
    print(f"평가 샘플 수: {len(samples)}")

    system = get_classification_prompt()

    # ── vLLM 초기화 ──────────────────────────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
    os.environ["NCCL_P2P_DISABLE"]     = "1"
    os.environ["TORCHDYNAMO_DISABLE"]  = "1"

    from vllm import LLM
    llm = LLM(
        model=args.classification_model,
        dtype="bfloat16",
        tensor_parallel_size=len(gpu_list),
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=True,
    )
    tokenizer = setup_tokenizer(args.classification_model)

    # ── 프롬프트 빌드 & 생성 ─────────────────────────────────────────────────
    prompts = [
        build_prompt(tokenizer, system, s["problem"], s["steps"], s["k"])
        for s in tqdm(samples, desc="프롬프트 빌드")
    ]
    generated = vllm_generate(llm, prompts, args.max_new_tokens)

    # ── 결과 수집 ─────────────────────────────────────────────────────────────
    records = []
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as pred_file:
        for s, gen_text in zip(samples, generated):
            pred_verdicts         = parse_per_rubric_verdicts(gen_text)
            pred_rubrics_critique = [r for r, v in pred_verdicts.items() if v == "incorrect"]
            pred_rubrics          = parse_fail_rubrics(gen_text)
            inference             = s["steps"][s["k"]].get("inference") or ""
            direct_action         = parse_next_action(gen_text)
            pred_action           = direct_action if direct_action is not None else rule_based_action(pred_rubrics_critique, inference)
            pred_action_verdict   = rule_based_action(pred_rubrics_critique, inference)
            record = {
                "problem_id":            s["problem_id"],
                "step_idx":              s["step_idx"],
                "state":                 s["state"],
                "is_right":              s["is_right"],
                "gold_rubrics":          s["gold_fail_rubrics"],
                "pred_verdicts":         pred_verdicts,
                "pred_rubrics":          pred_rubrics_critique,
                "pred_rubrics_token":    pred_rubrics,
                "pred_rubrics_critique": pred_rubrics_critique,
                "gold_action":           s["next_gold_action"],
                "pred_action":           pred_action,
                "pred_action_verdict":   pred_action_verdict,
                "generated":             gen_text,
            }
            records.append(record)
            pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── 메트릭 ───────────────────────────────────────────────────────────────
    all_rubrics = sorted(k for k in RUBRIC_TOKENS.keys() if k != "None")
    metrics        = compute_metrics(records, pred_key="pred_rubrics")
    binary_metrics = compute_per_rubric_binary_metrics(records, all_rubrics)
    mc_metrics     = compute_multiclass_with_none(records, all_rubrics)
    action_metrics = compute_action_metrics(records, pred_action_key="pred_action_verdict")
    print_metrics(metrics)
    print_multiclass_with_none(mc_metrics)
    print_action_table(action_metrics)

    if args.by_state:
        by_state: dict[str, list] = defaultdict(list)
        for r in records:
            by_state[r.get("state", "")].append(r)
        for state, recs in sorted(by_state.items()):
            m    = compute_metrics(recs, pred_key="pred_rubrics")
            mc_s = compute_multiclass_with_none(recs, all_rubrics)
            am   = compute_action_metrics(recs, pred_action_key="pred_action_verdict")
            print_metrics(m, title=f"state={state!r}")
            print_multiclass_with_none(mc_s, title=f"Multi-class  state={state!r}")
            print_action_table(am, title=f"Next Action  state={state!r}")

    # ── 요약 저장 ─────────────────────────────────────────────────────────────
    summary = {
        "timestamp":           ts,
        "data_path":           args.data_path,
        "classification_model": args.classification_model,
        "n_samples":           metrics["n_samples"],
        "action_accuracy":     round(metrics["action_accuracy"],    4),
        "action_macro_f1":     round(metrics["action_macro_f1"],    4),
        "rubric_exact_match":  round(metrics["rubric_exact_match"], 4),
        "rubric_avg_jaccard":  round(metrics["rubric_avg_jaccard"], 4),
        "rubric_micro_f1":     round(metrics["rubric_micro_f1"],    4),
        "rubric_micro_prec":   round(metrics["rubric_micro_prec"],  4),
        "rubric_micro_rec":    round(metrics["rubric_micro_rec"],   4),
        "n_none_gold":         metrics["n_none_gold"],
        "action_label_dist":   metrics["action_label_dist"],
        "action_rows": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in metrics["action_rows"]
        ],
        "rubric_macro_f1":     round(metrics["rubric_macro_f1"], 4),
        "rubric_rows": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in metrics["rubric_rows"]
        ],
        # per-rubric binary metrics
        "overall_binary_accuracy":  round(binary_metrics["overall_binary_accuracy"], 4),
        "binary_macro_f1":          round(binary_metrics["binary_macro_f1"], 4),
        "avg_rubric_coverage":      round(binary_metrics["avg_rubric_coverage"], 4),
        "rubric_binary_rows": [
            {
                "rubric":    r["rubric"],
                "accuracy":  round(r["accuracy"],  4),
                "precision": round(r["precision"], 4),
                "recall":    round(r["recall"],    4),
                "f1":        round(r["f1"],        4),
                "support_incorrect": r["support_incorrect"],
                "missing":   r["missing"],
            }
            for r in binary_metrics["rubric_binary_rows"]
        ],
        # multi-class metrics (None 포함)
        "multiclass_macro_f1":           round(mc_metrics["macro_f1"], 4),
        "multiclass_macro_f1_supported": round(mc_metrics["macro_f1_supported"], 4),
        "multiclass_rows": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in mc_metrics["rows"]
        ],
        # action per gold_action
        "action_by_gold_action": {
            act: {
                "n":          sum(1 for r in records if action_label(r["gold_action"]) == act),
                "action_acc": round(
                    sum(1 for r in records
                        if action_label(r["gold_action"]) == act
                        and action_label(r["pred_action"]) == act)
                    / max(1, sum(1 for r in records if action_label(r["gold_action"]) == act)), 4),
            }
            for act in ["solve", "rethink", "end"]
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n출력 저장: {out_dir}")


if __name__ == "__main__":
    main()
