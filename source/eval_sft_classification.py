"""
evaluate_classification.py

sft_classification 모델을 평가하자

classification model이 inference를 입력으로 받아
fail_rubrics, next_action을 얼마나 잘 예측하는지 평가한다.

실행 예시:
    python source/evaluate_classification.py \
        --data_path /mnt/yoonju/SC/output/sft_trajectory/traj_all_base_400.jsonl \
        --classification_model /mnt/yoonju/SC/checkpoints/sft/20260518_172258_sft_classification/epoch3

출력 (output/eval_classification/{timestamp}/):
    predictions.jsonl  스텝별 gold vs pred 비교
    summary.json       전체 메트릭 요약
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

def load_samples(data_path: str, skip_error: bool = True) -> list[dict]:
    """traj_all.jsonl에서 평가 샘플 추출."""
    raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
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
            samples.append({
                "problem":          problem,
                "steps":            steps,
                "k":                k,
                "gold_fail_rubrics": step.get("gold_fail_rubrics") or [],
                "next_gold_action":  step.get("next_gold_action") or TOKEN_SOLVE,
                "problem_id":        traj.get("problem_id", ""),
                "step_idx":          step.get("step_idx", k),
                "state":             step.get("state", ""),
                "is_right":          traj.get("is_right", None),
            })
    return samples


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
    """배치 생성. EOS까지 생성 후 텍스트 리스트 반환."""
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    tokenizer.padding_side = orig_side
    input_len = enc["input_ids"].shape[1]

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    results = []
    for i in range(len(prompts)):
        resp = out[i, input_len:]
        text = tokenizer.decode(resp, skip_special_tokens=True).strip()
        results.append(text)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 파싱
# ─────────────────────────────────────────────────────────────────────────────

def parse_fail_rubrics(text: str) -> list[str]:
    """Fail rubrics 섹션에서 루브릭 이름 목록 추출."""
    m = re.search(r"Fail\s+rubrics\s*:", text, re.IGNORECASE)
    if not m:
        return []
    section = text[m.end():].strip()
    if not section or section.lower().rstrip(".") == "none":
        return []
    tokens = re.findall(r"<\|[^|>]+\|>", section)
    result = []
    for tok in tokens:
        name = _TOKEN_TO_RUBRIC.get(tok)
        result.append(name if name else tok)
    return result


def rule_based_action(pred_rubrics: list[str], inference: str) -> str:
    """룰 기반 next action 결정.
    - fail rubrics 있음 → rethink
    - 없음 + boxed{} 있음 → end
    - 없음 + boxed{} 없음 → solve
    """
    if pred_rubrics:
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


def compute_metrics(records: list[dict]) -> dict:
    """
    records: [{"gold_action", "pred_action", "gold_rubrics", "pred_rubrics", "state"}, ...]
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
        pred_set = set(r["pred_rubrics"])
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
    n_none_pred = sum(1 for r in records if not r["pred_rubrics"])
    none_match  = sum(1 for r in records if not r["gold_rubrics"] and not r["pred_rubrics"])
    none_acc    = none_match / n_none_gold if n_none_gold else None

    # ── Per-rubric class metrics ──────────────────────────────────────────────
    all_rubric_labels = sorted(set(
        rb for rec in records for rb in (rec["gold_rubrics"] + rec["pred_rubrics"])
    ))
    rubric_tp_c: Counter = Counter()
    rubric_fp_c: Counter = Counter()
    rubric_fn_c: Counter = Counter()
    rubric_support: Counter = Counter()
    for rec in records:
        gold_set = set(rec["gold_rubrics"])
        pred_set = set(rec["pred_rubrics"])
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
        "rubric_rows":         rubric_rows,
        "rubric_macro_f1":     rubric_macro_f1,
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
    print(f"  Macro F1     : {metrics['rubric_macro_f1']:.4f}")
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classification 모델 평가")
    parser.add_argument("--data_path",            required=True, help="traj_all.jsonl 경로")
    parser.add_argument("--classification_model", required=True, help="모델 경로")
    parser.add_argument("--gpus",        type=int, default=0,    help="GPU 번호")
    parser.add_argument("--batch_size", type=int, default=8,    help="배치 크기")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output",     type=str, default=None, help="출력 폴더 (기본: output/eval_classification/{ts})")
    parser.add_argument("--no_skip_error", action="store_true", help="is_error=True 스텝도 포함")
    parser.add_argument("--by_state",   action="store_true",    help="state별 breakdown 출력")
    parser.add_argument("--limit",      type=int, default=None, help="평가할 최대 샘플 수 (디버그용)")
    args = parser.parse_args()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else (_ROOT / "output" / "eval_classification" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{ts}] 모델: {args.classification_model}")
    print(f"데이터: {args.data_path}")
    print(f"출력  : {out_dir}")

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    samples = load_samples(args.data_path, skip_error=not args.no_skip_error)
    if args.limit:
        samples = samples[:args.limit]
    print(f"평가 샘플 수: {len(samples)}")

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    print("모델 로딩 중...")
    model, tokenizer = load_model(args.classification_model, args.gpus)

    system = get_classification_prompt()

    # ── 배치 생성 ─────────────────────────────────────────────────────────────
    records = []
    pred_file = open(out_dir / "predictions.jsonl", "w", encoding="utf-8")

    for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="generating"):
        batch = samples[batch_start: batch_start + args.batch_size]

        prompts = [
            build_prompt(tokenizer, system, s["problem"], s["steps"], s["k"])
            for s in batch
        ]

        generated = batch_generate(
            model, tokenizer, prompts,
            max_new_tokens=args.max_new_tokens,
        )

        if batch_start == 0:
            sep = "─" * 64
            print(f"\n[샘플 출력 예시]")
            print(sep)
            print("[PROMPT]\n" + prompts[0])
            print(sep)
            print("[GENERATED]\n" + generated[0])
            print(sep + "\n")

        for s, gen_text in zip(batch, generated):
            pred_rubrics = parse_fail_rubrics(gen_text)
            inference    = s["steps"][s["k"]].get("inference") or ""
            pred_action  = rule_based_action(pred_rubrics, inference)

            rec = {
                "problem_id":    s["problem_id"],
                "step_idx":      s["step_idx"],
                "state":         s["state"],
                "is_right":      s["is_right"],
                "gold_rubrics":  s["gold_fail_rubrics"],
                "pred_rubrics":  pred_rubrics,
                "gold_action":   s["next_gold_action"],
                "pred_action":   pred_action,
                "generated":     gen_text,
            }
            records.append(rec)
            pred_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pred_file.flush()

    pred_file.close()

    # ── 메트릭 ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(records)
    print_metrics(metrics, records)

    if args.by_state:
        by_state: dict[str, list] = defaultdict(list)
        for r in records:
            by_state[r.get("state", "")].append(r)
        for state, recs in sorted(by_state.items()):
            m = compute_metrics(recs)
            print_metrics(m, recs, title=f"state={state!r}")

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
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n출력 저장: {out_dir}")


if __name__ == "__main__":
    main()
