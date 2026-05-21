"""
evaluate_classification.py

sft_classification 모델을 평가하자

classification model이 inference를 입력으로 받아
fail_rubrics, next_action을 얼마나 잘 예측하는지 평가한다.

실행 예시:
    python source/eval_sft_classification.py \
        --data_path /mnt/yoonju/SC/output/sft_trajectory/traj_all_base_400.jsonl \
        --classification_model /mnt/yoonju/SC/checkpoints/sft/20260518_172258_sft_classification/epoch3

출력 (output/eval_classification/{timestamp}/):
    predictions.jsonl  스텝별 gold vs pred 비교
    summary.json       전체 메트릭 요약
"""

import argparse
import json
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.multiprocessing as mp
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
    )
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_samples(data_path: str, skip_error: bool = True,
                 num_start: int | None = None, num_end: int | None = None) -> list[dict]:
    """traj_all.jsonl 또는 flat 스텝 레코드(k 필드 포함)에서 평가 샘플 추출."""
    raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    raw = raw[num_start:num_end]

    # flat 스텝 레코드 형식 감지 (k 필드가 최상위에 있으면)
    if raw and "k" in raw[0]:
        samples = []
        for rec in raw:
            gold_fr = rec.get("gold_fail_rubrics") or []
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
            if skip_error and step.get("is_error", False):
                continue
            # gold_fail_rubrics가 없으면 건너뜀
            if "gold_fail_rubrics" not in step:
                continue
            gold_fr = (
                [step["gold_fail_rubrics"]]
                if isinstance(step.get("gold_fail_rubrics"), str)
                else step.get("gold_fail_rubrics")
            ) or []
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

@torch.no_grad()
def batch_generate(model, tokenizer, prompts: list[str],
                   max_new_tokens: int = 1024) -> list[str]:
    """배치 생성. EOS까지 생성 후 텍스트 리스트 반환.
    루브릭/액션 special token이 EOS로 처리되어 잘리는 현상을 방지하기 위해
    model.generation_config에서 custom special token ID를 제외한 EOS ID만 사용."""
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]

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

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=_eos,
    )
    results = []
    for i in range(len(prompts)):
        resp = out[i, input_len:]
        text = tokenizer.decode(resp, skip_special_tokens=False).strip()
        results.append(text)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 파싱
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY_RUBRIC_MARKERS = {"None", "<|none|>", "none", ""}


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
    has_verdict = False
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"verdict:", stripped, re.IGNORECASE):
            has_verdict = True
            if re.search(r"verdict:\s*incorrect", stripped, re.IGNORECASE):
                rubric = re.split(r"\s*:", stripped, maxsplit=1)[0].strip()
                if rubric and _is_real_rubric(rubric):
                    result.append(rubric)
    return result


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


def print_metrics(metrics: dict, records: list[dict], title: str = "전체"):
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  {title}  (n={metrics['n_samples']})")
    print(sep)

    print(f"\n── Next Action ──────────────────────────────────────────────")
    print(f"  Accuracy   : {metrics['action_accuracy']:.4f}"
          f"  ({int(metrics['action_accuracy'] * metrics['n_samples'])}/{metrics['n_samples']})")
    print(f"  Macro F1   : {metrics['action_macro_f1']:.4f}")
    if metrics["action_none_count"]:
        print(f"  파싱 실패   : {metrics['action_none_count']} 스텝 (solve로 처리)")

    col = 10
    print(f"\n  {'label':<{col}}  {'prec':>8}  {'rec':>8}  {'f1':>8}  {'support':>8}")
    print("  " + "-" * (col + 38))
    for r in metrics["action_rows"]:
        print(f"  {r['label']:<{col}}  {r['precision']:>8.4f}  {r['recall']:>8.4f}"
              f"  {r['f1']:>8.4f}  {r['support']:>8}")

    action_pairs = [
        {"gold": action_label(r["gold_action"]) or "solve",
         "pred": action_label(r["pred_action"]) or "solve",
         "state": r.get("state", "")}
        for r in records
    ]
    print("\n  Confusion matrix  (rows=gold, cols=pred):")
    for line in confusion_matrix_str(action_pairs, ["solve", "rethink", "end"]).splitlines():
        print("    " + line)

    print(f"\n── Fail Rubrics ─────────────────────────────────────────────")
    print(f"  Exact match  : {metrics['rubric_exact_match']:.4f}")
    print(f"  Avg Jaccard  : {metrics['rubric_avg_jaccard']:.4f}")
    print(f"  Micro P/R/F1 : {metrics['rubric_micro_prec']:.4f} / "
          f"{metrics['rubric_micro_rec']:.4f} / {metrics['rubric_micro_f1']:.4f}")
    print(f"  Macro F1 (support>0) : {metrics['rubric_macro_f1_supported']:.4f}")
    print(f"  Gold=none    : {metrics['n_none_gold']}  Pred=none: {metrics['n_none_pred']}"
          + (f"  None-step acc: {metrics['none_step_accuracy']:.4f}"
             if metrics["none_step_accuracy"] is not None else ""))

    if metrics.get("rubric_rows"):
        rubric_col = max(len(r["label"]) for r in metrics["rubric_rows"])
        rubric_col = max(rubric_col, 5)
        print(f"\n  {'label':<{rubric_col}}  {'prec':>8}  {'rec':>8}  {'f1':>8}  {'support':>8}")
        print("  " + "-" * (rubric_col + 38))
        for r in metrics["rubric_rows"]:
            print(f"  {r['label']:<{rubric_col}}  {r['precision']:>8.4f}  {r['recall']:>8.4f}"
                  f"  {r['f1']:>8.4f}  {r['support']:>8}")
        supported = [r for r in metrics["rubric_rows"] if r["support"] > 0]
        print(f"  {'Macro (support>0)':<{rubric_col}}  {'':>8}  {'':>8}  {metrics['rubric_macro_f1_supported']:>8.4f}  {len(supported):>7}cls")


# ─────────────────────────────────────────────────────────────────────────────
# 멀티GPU 워커
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_worker(gpu_id: int, model_path: str, samples: list[dict],
                batch_size: int, max_new_tokens: int, system: str,
                tmp_path: str, show_sample: bool):
    model, tokenizer = load_model(model_path, gpu_id)
    records = []
    for batch_start in tqdm(range(0, len(samples), batch_size),
                            desc=f"GPU {gpu_id}", position=gpu_id):
        batch = samples[batch_start: batch_start + batch_size]
        prompts = [
            build_prompt(tokenizer, system, s["problem"], s["steps"], s["k"])
            for s in batch
        ]
        generated = batch_generate(model, tokenizer, prompts, max_new_tokens)

        if show_sample and batch_start == 0:
            sep = "─" * 64
            lines = [
                f"\n[GPU {gpu_id} 샘플 출력 예시]\n{sep}",
                "[PROMPT]\n" + prompts[0],
                sep,
                "[GENERATED]\n" + generated[0],
                sep + "\n",
            ]
            tqdm.write("\n".join(lines))

        for s, gen_text in zip(batch, generated):
            pred_rubrics          = parse_fail_rubrics(gen_text)
            pred_rubrics_critique = parse_fail_rubrics_from_deep_critique(gen_text)
            inference    = s["steps"][s["k"]].get("inference") or ""
            pred_action  = rule_based_action(pred_rubrics, inference)
            records.append({
                "problem_id":            s["problem_id"],
                "step_idx":              s["step_idx"],
                "state":                 s["state"],
                "is_right":              s["is_right"],
                "gold_rubrics":          s["gold_fail_rubrics"],
                "pred_rubrics":          pred_rubrics,
                "pred_rubrics_critique": pred_rubrics_critique,
                "gold_action":           s["next_gold_action"],
                "pred_action":           pred_action,
                "generated":             gen_text,
            })

    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classification 모델 평가")
    parser.add_argument("--data_path",            required=True, help="traj_all.jsonl 경로")
    parser.add_argument("--classification_model", required=True, help="모델 경로")
    parser.add_argument("--gpus",        type=str, default="0",  help="GPU 번호 (단일: 0, 다중: 0,1,2,3)")
    parser.add_argument("--batch_size",  type=int, default=8,    help="GPU당 배치 크기")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output",      type=str, default=None, help="출력 폴더 (기본: output/eval_classification/{ts})")
    parser.add_argument("--skip_error", action="store_true",  help="is_error=True 스텝 제외")
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

    # ── 멀티GPU 병렬 생성 ────────────────────────────────────────────────────
    n = len(gpu_list)
    chunks   = [samples[i::n] for i in range(n)]
    tmp_files = [tempfile.mktemp(suffix=f"_gpu{g}.jsonl") for g in gpu_list]

    if n == 1:
        _gpu_worker(gpu_list[0], args.classification_model, chunks[0],
                    args.batch_size, args.max_new_tokens, system,
                    tmp_files[0], show_sample=True)
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for i, (gpu_id, chunk, tmp) in enumerate(zip(gpu_list, chunks, tmp_files)):
            p = ctx.Process(target=_gpu_worker,
                            args=(gpu_id, args.classification_model, chunk,
                                  args.batch_size, args.max_new_tokens, system,
                                  tmp, i == 0))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"워커 프로세스 실패 (exitcode={p.exitcode})")

    # ── 결과 수집 ─────────────────────────────────────────────────────────────
    records = []
    pred_file = open(out_dir / "predictions.jsonl", "w", encoding="utf-8")
    for tmp in tmp_files:
        for line in open(tmp, encoding="utf-8"):
            rec = json.loads(line)
            records.append(rec)
            pred_file.write(line)
    pred_file.close()

    # ── 메트릭 ───────────────────────────────────────────────────────────────
    metrics          = compute_metrics(records, pred_key="pred_rubrics")
    metrics_critique = compute_metrics(records, pred_key="pred_rubrics_critique")
    print_metrics(metrics, records)
    print_rubric_comparison_table(metrics, metrics_critique)

    if args.by_state:
        by_state: dict[str, list] = defaultdict(list)
        for r in records:
            by_state[r.get("state", "")].append(r)
        for state, recs in sorted(by_state.items()):
            m  = compute_metrics(recs, pred_key="pred_rubrics")
            mc = compute_metrics(recs, pred_key="pred_rubrics_critique")
            print_metrics(m, recs, title=f"state={state!r}")
            print_rubric_comparison_table(m, mc)

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
        "critique_rubric_exact_match":           round(metrics_critique["rubric_exact_match"],           4),
        "critique_rubric_micro_f1":              round(metrics_critique["rubric_micro_f1"],              4),
        "critique_rubric_micro_prec":            round(metrics_critique["rubric_micro_prec"],            4),
        "critique_rubric_micro_rec":             round(metrics_critique["rubric_micro_rec"],             4),
        "critique_rubric_macro_f1_supported":    round(metrics_critique["rubric_macro_f1_supported"],    4),
        "critique_rubric_rows": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in metrics_critique["rubric_rows"]
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
