"""
결과 jsonl 파일들을 읽어 성능을 분석하는 스크립트.

아래 PATHS 하이퍼파라미터만 수정하고 실행하면 됩니다:
  python test.py
"""

import json
from pathlib import Path

# =============================================================================
# 하이퍼파라미터 — 여기만 수정하세요
# =============================================================================

# 분석할 파일 또는 폴더의 절대 경로 목록.
#   - 폴더: 그 안의 results*.jsonl 파일을 모두 읽음
#   - 파일: 해당 .jsonl 파일만 읽음
#   - 여러 개 입력 시 마지막에 비교표 출력
PATHS = [
    "/mnt/yoonju/SC/output/eval_results/qwen_baseline",
    # "/mnt/yoonju/SC/output/eval_sft/20260322_202515",  # 비교 대상 추가 예시
]

VERBOSE = False   # True 로 바꾸면 오답 목록 출력

# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 파일 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dir(dir_path: Path) -> list[dict]:
    """디렉토리 내 results_*.jsonl 파일을 모두 읽어 합친다."""
    files = sorted(dir_path.glob("results*.jsonl"))
    if not files:
        raise FileNotFoundError(f"{dir_path} 에 results*.jsonl 파일이 없습니다.")
    records = []
    for f in files:
        records.extend(load_jsonl(f))
    # idx 기준 중복 제거 (여러 타임스탬프 버전이 있을 경우)
    seen = set()
    unique = []
    for r in records:
        key = r.get("idx", id(r))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def load_path(p: str) -> tuple[list[dict], str]:
    path = Path(p)
    if path.is_dir():
        records = load_dir(path)
        label = path.name
    elif path.suffix == ".jsonl":
        records = load_jsonl(path)
        label = path.stem
    else:
        raise ValueError(f"jsonl 파일 또는 디렉토리를 지정하세요: {p}")
    return records, label


# ─────────────────────────────────────────────────────────────────────────────
# 분석
# ─────────────────────────────────────────────────────────────────────────────

def summarize(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}

    n_correct = sum(1 for r in records if r.get("correct", False))
    tokens    = [r.get("n_tokens", 0) for r in records]

    # 정답 / 오답 그룹 토큰 통계
    correct_tokens   = [r.get("n_tokens", 0) for r in records if r.get("correct")]
    incorrect_tokens = [r.get("n_tokens", 0) for r in records if not r.get("correct")]

    def _stats(lst):
        if not lst:
            return {"mean": 0, "min": 0, "max": 0}
        return {
            "mean": round(sum(lst) / len(lst), 1),
            "min":  min(lst),
            "max":  max(lst),
        }

    return {
        "n_total":          n,
        "n_correct":        n_correct,
        "accuracy":         round(n_correct / n, 4),
        "tokens":           _stats(tokens),
        "tokens_correct":   _stats(correct_tokens),
        "tokens_incorrect": _stats(incorrect_tokens),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────────────────────

def _bar(ratio: float, width: int = 30) -> str:
    filled = round(ratio * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_summary(label: str, s: dict):
    acc = s["accuracy"]
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  총 문제수   : {s['n_total']:>6}")
    print(f"  정답 수     : {s['n_correct']:>6}  ({acc:.1%})")
    print(f"  Accuracy    : {acc:.4f}  {_bar(acc)}")
    print()
    t = s["tokens"]
    tc = s["tokens_correct"]
    ti = s["tokens_incorrect"]
    print(f"  토큰 수 (전체)   평균 {t['mean']:>7.1f}  min {t['min']:>5}  max {t['max']:>6}")
    print(f"  토큰 수 (정답)   평균 {tc['mean']:>7.1f}  min {tc['min']:>5}  max {tc['max']:>6}")
    print(f"  토큰 수 (오답)   평균 {ti['mean']:>7.1f}  min {ti['min']:>5}  max {ti['max']:>6}")


def print_comparison(labels: list[str], summaries: list[dict]):
    W = 65
    print(f"\n{'='*W}")
    print(f"{'모델 비교':^{W}}")
    print(f"{'='*W}")

    col_w = (W - 22) // len(labels)
    header = f"  {'지표':<20}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print(f"{'─'*W}")

    rows = [
        ("Accuracy",         "accuracy",  lambda v: f"{v:.1%}"),
        ("n_total",          "n_total",   lambda v: str(v)),
        ("n_correct",        "n_correct", lambda v: str(v)),
        ("Avg Tokens",       None,        None),
        ("  (전체)",         ("tokens", "mean"),    lambda v: f"{v:.1f}"),
        ("  (정답)",         ("tokens_correct", "mean"),   lambda v: f"{v:.1f}"),
        ("  (오답)",         ("tokens_incorrect", "mean"), lambda v: f"{v:.1f}"),
    ]

    for row_label, key, fmt in rows:
        if key is None:
            print(f"  {row_label}")
            continue
        vals = []
        for s in summaries:
            if isinstance(key, tuple):
                v = s.get(key[0], {}).get(key[1])
            else:
                v = s.get(key)
            vals.append(fmt(v) if v is not None else "-")
        line = f"  {row_label:<20}" + "".join(f"{v:>{col_w}}" for v in vals)
        print(line)

    print(f"{'='*W}")

    # 기준 대비 delta (첫 번째가 baseline)
    if len(summaries) >= 2:
        print("\n  [베이스라인 대비 차이]")
        base_acc = summaries[0]["accuracy"]
        for label, s in zip(labels[1:], summaries[1:]):
            delta = s["accuracy"] - base_acc
            sign  = "+" if delta >= 0 else ""
            print(f"    {label}: Accuracy {sign}{delta:.1%}  ({sign}{delta*100:.2f}pp)")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_labels    = []
    all_summaries = []

    for p in PATHS:
        records, label = load_path(p)
        s = summarize(records)
        all_labels.append(label)
        all_summaries.append(s)
        print_summary(label, s)

        if VERBOSE:
            wrong = [r for r in records if not r.get("correct")]
            print(f"\n  [오답 {len(wrong)}개]")
            for r in wrong[:10]:
                print(f"    idx={r.get('idx','?')}  gold={r.get('gold_answer','?')}  pred={r.get('predicted','?')}")
            if len(wrong) > 10:
                print(f"    ... ({len(wrong)-10}개 더)")

    if len(all_labels) >= 2:
        print_comparison(all_labels, all_summaries)


if __name__ == "__main__":
    main()
