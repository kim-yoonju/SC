import os
import ast
import json
import random
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from utils import load_jsonl, lower_keys

# 로컬 parquet 데이터셋 경로
EVAL_PARQUET_DIR = "/mnt/yoonju/NRL/S2R/data/eval"
EVAL_PARQUET_DATASETS = {
    os.path.splitext(f)[0]: os.path.join(EVAL_PARQUET_DIR, f)
    for f in os.listdir(EVAL_PARQUET_DIR)
    if f.endswith(".parquet")
}  # e.g. {"math500": "/.../math500.parquet", "aime2024": ..., ...}


def _load_parquet_dataset(data_name, parquet_path):
    """로컬 parquet → 평가 공통 포맷 변환.
    모든 파일이 prompt(list-of-dicts repr) + reward_model(ground_truth) 구조를 가짐.
    출력 필드: problem, gt, gt_cot, type, level
    """
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    rows = table.to_pydict()
    n = len(table)

    examples = []
    for i in range(n):
        # ── 문제 추출 (prompt의 user 메시지 우선, 없으면 question/problem 직접 사용) ──
        if "prompt" in rows:
            prompt_raw = rows["prompt"][i]
            try:
                messages = ast.literal_eval(prompt_raw) if isinstance(prompt_raw, str) else prompt_raw
                problem = next(m["content"] for m in messages if m["role"] == "user")
            except Exception:
                problem = str(prompt_raw)
        elif "question" in rows:
            problem = rows["question"][i] or ""
        elif "problem" in rows:
            problem = rows["problem"][i] or ""
        else:
            problem = ""

        # ── 정답 추출 (reward_model.ground_truth 우선, 없으면 answer 직접 사용) ──
        if "reward_model" in rows:
            reward_raw = rows["reward_model"][i]
            try:
                reward = ast.literal_eval(reward_raw) if isinstance(reward_raw, str) else reward_raw
                gt = str(reward.get("ground_truth", ""))
            except Exception:
                gt = ""
        elif "answer" in rows:
            gt = rows["answer"][i] or ""
        else:
            gt = ""

        # solution 필드가 있으면 gt_cot로 사용 (없으면 빈 문자열)
        gt_cot = rows["solution"][i] if "solution" in rows else ""

        examples.append({
            "problem": problem,
            "gt":      gt,
            "gt_cot":  gt_cot,
            "type":    rows.get("type",    [""]*n)[i] or rows.get("subject", [""]*n)[i] or "",
            "level":   rows.get("level",   [""]*n)[i] or "",
        })
    return Dataset.from_list(examples)


def load_data(data_name, split, data_dir="./data"):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        # ── 로컬 eval parquet 우선 처리 ───────────────────────────────
        if data_name in EVAL_PARQUET_DATASETS:
            dataset = _load_parquet_dataset(data_name, EVAL_PARQUET_DATASETS[data_name])

        elif data_name == "math":
            dataset = load_dataset(
                "competition_math",
                split=split,
                name="main",
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "svamp":
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets(
                [dataset, load_dataset("ChilleD/SVAMP", split="test")]
            )
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(lambda x: ";" not in x["answer"])
        elif data_name == "mawps":
            examples = []
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example["type"] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "mmlu_stem":
            dataset = load_dataset("hails/mmlu_no_train", "all", split="test")
            stem_subjects = [
                "abstract_algebra", "astronomy", "college_biology",
                "college_chemistry", "college_computer_science", "college_mathematics",
                "college_physics", "computer_security", "conceptual_physics",
                "electrical_engineering", "elementary_mathematics", "high_school_biology",
                "high_school_chemistry", "high_school_computer_science",
                "high_school_mathematics", "high_school_physics",
                "high_school_statistics", "machine_learning",
            ]
            dataset = dataset.rename_column("subject", "type")
            dataset = dataset.filter(lambda x: x["type"] in stem_subjects)
        elif data_name == "carp_en":
            dataset = load_jsonl(f"{data_dir}/carp_en/test.jsonl")
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # deduplicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples
