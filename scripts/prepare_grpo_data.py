"""
rl_data JSONL → verl GRPO용 parquet 변환 스크립트

출력 parquet 컬럼:
  prompt          : [{"role": "system", ...}, {"role": "user", ...}]
  data_source     : "sc-grpo"
  reward_model    : {"ground_truth": gold_answer, "style": "rule"}
  extra_info      : {"problem_id": ..., "gold_answer": ...}

Usage:
    python scripts/prepare_grpo_data.py
    python scripts/prepare_grpo_data.py --input /path/to/data.jsonl --output /path/to/out.parquet
"""

import argparse
import json
import pathlib
import sys

import pandas as pd
import yaml

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "source"))


def load_config():
    with open(_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def build_system_prompt(cfg: dict) -> str:
    prompts_file = _ROOT / cfg["prompts"]["file"]
    rubric_file  = _ROOT / cfg["prompts"].get("rubric_file", "prompts/prm_rubric_v7.7.jsonl")

    prompts = {p["name"]: p["content"] for p in json.load(open(prompts_file))}
    rubric_lines = [json.loads(l) for l in open(rubric_file) if l.strip()]
    rubric_text  = "\n".join(
        f'{r["name"]}: [{r["criterion"]}]' for r in rubric_lines
    )
    return prompts["gen_solve_R"].replace("{{rubric}}", rubric_text)


def convert(input_path: str, output_path: str, system_prompt: str) -> None:
    seen_ids: set = set()
    rows = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pid = d["problem_id"]
            if pid in seen_ids:          # 같은 문제 중복 제거
                continue
            seen_ids.add(pid)

            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"[Problem]\n{d['problem']}\n\nWrite Step 1."},
            ]
            rows.append({
                "prompt":       prompt,
                "data_source":  "sc-grpo",
                "reward_model": {"ground_truth": d["gold_answer"], "style": "rule"},
                "extra_info":   {"problem_id": pid, "gold_answer": d["gold_answer"]},
            })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"저장 완료: {output_path}  ({len(df)}개 문제)")


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=cfg["data_path"]["rl_data"])
    parser.add_argument("--output", default=str(_ROOT / "datasets" / "grpo_train.parquet"))
    args = parser.parse_args()

    system_prompt = build_system_prompt(cfg)
    convert(args.input, args.output, system_prompt)


if __name__ == "__main__":
    main()
