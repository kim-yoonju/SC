"""
eval_action_cls.py
trajectory jsonl에서 gold/pred next action을 파싱해 classification 성능을 측정한다.

사용법:
    python source/eval_action_cls.py /mnt/yoonju/SC/output/sft_trajectory/20260515_064458/traj_all.jsonl
    python source/eval_action_cls.py <path> --skip-missing   # pred=None 스텝 제외
    python source/eval_action_cls.py <path> --by-state       # state별 breakdown
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


LABEL_ORDER = ["solve", "rethink", "end"]


def normalize(action: str | None) -> str | None:
    """'<|solve|>' → 'solve', 'solve' → 'solve', None → None"""
    if action is None:
        return None
    return action.strip().strip("<|>").strip("|")


def load_pairs(path: str, skip_missing: bool) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for step in d.get("steps", []):
                gold = normalize(step.get("next_gold_action"))
                pred = normalize(step.get("next_pred_action"))
                state = step.get("state", "")
                if gold is None:
                    continue
                if pred is None and skip_missing:
                    continue
                pairs.append({"gold": gold, "pred": pred, "state": state})
    return pairs


def classification_report(pairs: list[dict]) -> dict:
    labels = sorted({p["gold"] for p in pairs} | {p["pred"] for p in pairs if p["pred"]})
    label_order = [l for l in LABEL_ORDER if l in labels] + [l for l in labels if l not in LABEL_ORDER]

    tp = Counter()
    fp = Counter()
    fn = Counter()
    correct = 0

    for p in pairs:
        g, pred = p["gold"], p["pred"]
        if pred == g:
            tp[g] += 1
            correct += 1
        else:
            fn[g] += 1
            if pred is not None:
                fp[pred] += 1

    rows = []
    for label in label_order:
        support = tp[label] + fn[label]
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        recall    = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({"label": label, "precision": precision, "recall": recall,
                     "f1": f1, "support": support})

    total = len(pairs)
    accuracy = correct / total if total else 0.0

    macro_p  = sum(r["precision"] for r in rows) / len(rows) if rows else 0.0
    macro_r  = sum(r["recall"]    for r in rows) / len(rows) if rows else 0.0
    macro_f1 = sum(r["f1"]        for r in rows) / len(rows) if rows else 0.0

    return {"rows": rows, "accuracy": accuracy, "total": total,
            "correct": correct, "macro_p": macro_p, "macro_r": macro_r,
            "macro_f1": macro_f1, "label_order": label_order}


def confusion_matrix_str(pairs: list[dict], label_order: list[str]) -> str:
    idx = {l: i for i, l in enumerate(label_order)}
    n = len(label_order)
    mat = [[0] * n for _ in range(n)]

    for p in pairs:
        g = p["gold"]
        pred = p["pred"] if p["pred"] is not None else "__missing__"
        if g in idx and pred in idx:
            mat[idx[g]][idx[pred]] += 1
        elif g in idx:
            pass  # pred None or unknown — counted separately

    col_w = max(len(l) for l in label_order) + 2
    header = " " * (col_w + 2) + "  ".join(f"{l:>{col_w}}" for l in label_order)
    lines = [header, " " * (col_w + 2) + "  ".join("-" * col_w for _ in label_order)]
    for i, gl in enumerate(label_order):
        row_str = "  ".join(f"{mat[i][j]:>{col_w}}" for j in range(n))
        lines.append(f"{gl:>{col_w}}  {row_str}")
    return "\n".join(lines)


def print_report(report: dict, pairs: list[dict], title: str = ""):
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    n_missing = sum(1 for p in pairs if p["pred"] is None)

    print(f"\n  Total steps : {report['total']}  (pred=None: {n_missing})")
    print(f"  Accuracy    : {report['accuracy']:.4f}  ({report['correct']}/{report['total']})\n")

    col = 12
    header = f"  {'label':<{col}}  {'precision':>10}  {'recall':>8}  {'f1':>8}  {'support':>8}"
    print(header)
    print("  " + "-" * (col + 44))
    for r in report["rows"]:
        print(f"  {r['label']:<{col}}  {r['precision']:>10.4f}  {r['recall']:>8.4f}  {r['f1']:>8.4f}  {r['support']:>8}")
    print("  " + "-" * (col + 44))
    print(f"  {'macro avg':<{col}}  {report['macro_p']:>10.4f}  {report['macro_r']:>8.4f}  {report['macro_f1']:>8.4f}  {report['total']:>8}")

    print("\n  Confusion matrix  (rows=gold, cols=pred):")
    for line in confusion_matrix_str(pairs, report["label_order"]).splitlines():
        print("    " + line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="traj jsonl 파일 경로")
    parser.add_argument("--skip-missing", action="store_true",
                        help="pred=None인 스텝을 평가에서 제외 (기본: 오답 처리)")
    parser.add_argument("--by-state", action="store_true",
                        help="state 별로 분리해서 결과 출력")
    args = parser.parse_args()

    pairs = load_pairs(args.input, args.skip_missing)
    print(f"\n파일: {args.input}")
    print(f"skip_missing={args.skip_missing}")

    report = classification_report(pairs)
    print_report(report, pairs, title="전체 (all steps)")

    if args.by_state:
        state_groups: dict[str, list] = defaultdict(list)
        for p in pairs:
            state_groups[p["state"]].append(p)

        for state in sorted(state_groups):
            grp = state_groups[state]
            rpt = classification_report(grp)
            print_report(rpt, grp, title=f"state = {state}")


if __name__ == "__main__":
    main()
