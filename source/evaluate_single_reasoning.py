
"""
Qwen2.5-7B-Instruct 평가 스크립트

checkpoint가 있으면 로컬 checkpoint를 불러오고, 없으면 HuggingFace에서 다운로드.

평가 지표:
  - Format 정확도 : \\boxed{} 형식에서 정답 추출 후 비교

실행:
  python source/evaluate_single_reasoning.py
  python source/evaluate_single_reasoning.py --dataset datasets/aime25_test.jsonl
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_config,
    load_dataset_file,
    load_model_and_tokenizer,
    build_chat_prompt,
    generate_batch,
    answers_equal,
    format_correct,
)

GPUS = [4,5]                                          # 사용할 GPU 번호 목록
DATASETS = ["deepmath_16k"] # 평가할 데이터셋 (config data_path의 키)

# ─────────────────────────────────────────────────────────────────────────────
# 평가 메인
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(dataset_path: str, model, tokenizer, device, cfg: dict, action_prompts: dict, out_file=None) -> dict:
    eval_cfg = cfg["single_reasoning"]
    batch_size = eval_cfg["batch_size"]
    max_new_tokens = eval_cfg["max_new_tokens"]
    system_prompt = action_prompts["base_prompt"]

    items = load_dataset_file(dataset_path)
    is_gsm8k = "gsm8k" in Path(dataset_path).name.lower()
    print(f"  문제 수: {len(items)}")

    responses = []
    for b in tqdm(range(0, len(items), batch_size), desc="  생성"):
        batch   = items[b: b + batch_size]
        prompts = [build_chat_prompt(tokenizer, system_prompt, it["problem"]) for it in batch]
        responses.extend(generate_batch(prompts, model, tokenizer, device, max_new_tokens))

    fmt_results = []
    for r, it in tqdm(zip(responses, items), total=len(items), desc="  평가"):
        fc = format_correct(r, it["answer"], is_gsm8k=is_gsm8k)
        fmt_results.append(fc)

        if out_file is not None:
            row = {
                "id":             it["id"],
                "problem":        it["problem"],
                "answer":         it["answer"],
                "response":       r,
                "format_correct": fc,
            }
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_file.flush()

    n = len(items)
    return {
        "dataset":        Path(dataset_path).name,
        "n_total":        n,
        "format_correct": sum(fmt_results),
        "format_acc":     round(sum(fmt_results) / n, 4),
    }


def print_result(result: dict):
    W = 50
    print(f"\n{'='*W}")
    print(f"  데이터셋  : {result['dataset']}")
    print(f"  문제 수   : {result['n_total']}")
    print(f"{'─'*W}")
    print(f"  Format 정확도  : {result['format_acc']:.4f}  ({result['format_correct']}/{result['n_total']})")
    print(f"{'='*W}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=None, help="배치 크기 (config 값 override)")
    p.add_argument("--max_new_tokens", type=int, default=None, help="최대 생성 토큰 수 (config 값 override)")
    p.add_argument("--dataset", default=None, help="단일 데이터셋 경로 (config 값 override)")
    return p.parse_args()


def print_summary(results: list[dict]):
    W = 60
    print(f"\n{'='*W}")
    print(f"{'전체 요약':^{W}}")
    print(f"{'='*W}")
    print(f"  {'데이터셋':<30} {'Format':>8} {'N':>6}")
    print(f"  {'─'*48}")
    for r in results:
        print(f"  {r['dataset']:<30} {r['format_acc']:>7.1%} {r['n_total']:>6}")
    print(f"{'='*W}")


def main():
    args = parse_args()
    cfg = load_config()

    with open(_ROOT / "prompts/action_prompts.json", "r", encoding="utf-8") as f:
        action_prompts = json.load(f)

    eval_cfg = cfg["single_reasoning"]

    # CLI 인자로 config 값 override
    if args.batch_size is not None:
        eval_cfg["batch_size"] = args.batch_size
    if args.max_new_tokens is not None:
        eval_cfg["max_new_tokens"] = args.max_new_tokens

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPUS)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # OpenAI API key 설정
    api_key = cfg.get("API_key", {}).get("gpt", "")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model, tokenizer = load_model_and_tokenizer(device, cfg)

    # 데이터셋 목록 결정
    if args.dataset:
        datasets = {"(CLI)": args.dataset}
    else:
        all_datasets = cfg["data_path"]
        datasets = {name: str(_ROOT / all_datasets[name]) for name in DATASETS}

    # 출력 경로
    output_dir = cfg["output_path"]["Inference"]
    out_file = None
    if output_dir:
        out_dir = _ROOT / output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{timestamp}.jsonl"
        out_file = open(out_path, "w")
        print(f"결과 저장 경로: {out_path}")

    all_results = []
    try:
        for i, (name, dataset_path) in enumerate(datasets.items(), 1):
            if not Path(dataset_path).exists():
                print(f"\n[스킵] 파일 없음: {name} ({dataset_path})")
                continue
            print(f"\n[{i}/{len(datasets)}] {name}")
            result = evaluate(dataset_path, model, tokenizer, device, cfg, action_prompts, out_file)
            print_result(result)
            all_results.append(result)
    finally:
        if out_file:
            out_file.close()

    if len(all_results) > 1:
        print_summary(all_results)


if __name__ == "__main__":
    main()
