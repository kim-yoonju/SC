"""
enrich_jsonl.py
deepmath_16k.parquet에서 problem/answer를 가져와 JSONL 파일에 추가.
id == extra_info.index 기준으로 매칭.

사용법:
  python enrich_jsonl.py [파일1.jsonl ...]

인자 없이 실행하면 config의 경로를 기본값으로 사용.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import CONF

DEEPMATH_PATH = "/mnt/yoonju/SC/datasets/deepmath_16k/deepmath_16k.parquet"
DEFAULT_FILES = [
    "/mnt/yoonju/SC/datasets/deepmath_16k/base_single_reasoning_16k.jsonl",
    "/mnt/yoonju/SC/datasets/deepmath_16k/base_single_reasoning_right_16k.jsonl",
    "/mnt/yoonju/SC/datasets/deepmath_16k/base_single_reasoning_wrong_16k.jsonl",
]


def _build_lookup(parquet_path: str) -> dict:
    df = pd.read_parquet(parquet_path)
    lookup = {}
    for _, row in df.iterrows():
        idx = str(row["extra_info"].get("index", ""))
        if not idx:
            continue
        msgs = row["prompt"]
        content = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        problem = re.sub(r"\s*Please reason step by step[^$]*$", "", content, flags=re.DOTALL).strip()
        answer = str(row["final_answer"]).strip()
        lookup[idx] = {"problem": problem, "answer": answer}
    return lookup


def enrich_file(path: str, lookup: dict) -> None:
    p = Path(path)
    if not p.exists():
        print(f"[SKIP] 파일 없음: {path}")
        return

    lines = p.read_text(encoding="utf-8").splitlines()
    enriched = []
    n_found = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("problem") and record.get("answer"):
            enriched.append(record)
            n_found += 1
            continue
        idx = str(record.get("id", ""))
        src = lookup.get(idx, {})
        record["problem"] = src.get("problem", "")
        record["answer"] = src.get("answer", "")
        if src:
            n_found += 1
        enriched.append(record)

    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in enriched) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] {p.name}: {len(enriched)}개 중 {n_found}개 매칭 완료")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FILES
    print(f"deepmath 로딩 중: {DEEPMATH_PATH}")
    lookup = _build_lookup(DEEPMATH_PATH)
    print(f"룩업 테이블 구축 완료: {len(lookup)}개 항목\n")
    for path in targets:
        enrich_file(path, lookup)


if __name__ == "__main__":
    main()
