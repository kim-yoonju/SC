"""
Qwen2.5-7B-Instruct 베이스라인 평가 스크립트

- 단순 single-pass inference (multi-step 없음)
- GPU 0,1,2,3 멀티프로세스 병렬 실행
"""

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


_ROOT = Path(__file__).resolve().parent.parent

MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR   = "/mnt/.cache/huggingface"
DATASET_PATH = str(_ROOT / "datasets/deepmath_16k.parquet")
OUTPUT_DIR  = str(_ROOT / "output/eval_results/qwen_baseline")
GPUS        = [2]

SYSTEM_PROMPT = """\
You are an expert mathematician. Solve the given math problem step by step.
Show your work clearly. Put your final answer inside \\boxed{}."""


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def parse_boxed(text: str):
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    match = matches[-1]
    start = match.end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def normalize_answer(answer) -> str:
    if answer is None:
        return ""
    answer = str(answer).strip().strip("$").replace(" ", "")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\!", "").replace("\\,", "")
    return answer


def check_correct(pred, gold) -> bool:
    if pred is None:
        return False
    pred_n = normalize_answer(pred)
    gold_n = normalize_answer(gold)
    if pred_n == gold_n:
        return True
    try:
        return abs(float(pred_n) - float(gold_n)) < 1e-6
    except (ValueError, TypeError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 배치 생성
# ─────────────────────────────────────────────────────────────────────────────

def batch_generate(model, tokenizer, problems: list, max_new_tokens: int) -> list[dict]:
    tokenizer.padding_side = "left"

    prompt_texts = []
    for p in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Problem:\n{p['problem']}"},
        ]
        prompt_texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)
    input_length = enc["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    del enc
    torch.cuda.empty_cache()

    results = []
    for i, p in enumerate(problems):
        new_ids = output_ids[i][input_length:]
        eos_id  = tokenizer.eos_token_id
        n_tokens = next(
            (j for j, t in enumerate(new_ids.tolist()) if t == eos_id),
            len(new_ids)
        )
        response    = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        boxed       = parse_boxed(response)
        pred_answer = boxed if boxed else response.strip()
        correct     = check_correct(pred_answer, p["answer"])
        results.append({
            "response":     response,
            "pred_answer":  pred_answer,
            "n_tokens":     n_tokens,
            "correct":      correct,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 워커 프로세스
# ─────────────────────────────────────────────────────────────────────────────

def worker_fn(gpu_id: int, examples: list, output_path: str, args, result_queue: Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] {len(examples)}문제 시작")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    model.eval()

    n_correct = 0
    batches = [examples[s:s + args.batch_size] for s in range(0, len(examples), args.batch_size)]

    with open(output_path, "w") as out_f:
        processed = 0
        for batch in tqdm(batches, desc=f"GPU{gpu_id}", position=gpu_id):
            batch_results = batch_generate(model, tokenizer, batch, args.max_new_tokens)
            for example, res in zip(batch, batch_results):
                if res["correct"]:
                    n_correct += 1
                record = {
                    "idx":         example["_idx"],
                    "problem":     example["problem"],
                    "gold_answer": example["answer"],
                    "predicted":   res["pred_answer"],
                    "correct":     res["correct"],
                    "n_tokens":    res["n_tokens"],
                    "response":    res["response"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
            processed += len(batch)
            tqdm.write(f"  [GPU {gpu_id}] {processed}/{len(examples)}  acc={n_correct/processed:.3f}")

    result_queue.put({"gpu": gpu_id, "n_correct": n_correct, "n_total": len(examples)})
    print(f"[GPU {gpu_id}] 완료: {n_correct}/{len(examples)} = {n_correct/len(examples):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str, default=DATASET_PATH)
    parser.add_argument("--gpus",           type=str, default=",".join(str(g) for g in GPUS))
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_problems",   type=int, default=None)
    args = parser.parse_args()

    gpu_list = [int(g) for g in args.gpus.split(",")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = str(_ROOT / "output" / "eval_results" / f"base_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 데이터셋 로드
    dataset_path = str(Path(args.dataset).resolve())
    if dataset_path.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=dataset_path, split="train")
    else:
        ds = load_dataset(dataset_path, split="test")

    def normalize(ex):
        if not ex.get("problem"):
            prompt = ex.get("prompt", "")
            if isinstance(prompt, list):
                user_msgs = [m["content"] for m in prompt if m.get("role") == "user"]
                ex["problem"] = user_msgs[0] if user_msgs else ""
            else:
                ex["problem"] = ex.get("question", "")
        if not ex.get("answer"):
            ex["answer"] = ex.get("final_answer", "")
        return ex

    ds = ds.map(normalize)
    examples = list(ds)
    examples = examples[-100:]
    for i, ex in enumerate(examples):
        ex["_idx"] = i

    print(f"모델: {MODEL_ID}")
    print(f"평가 문제 수: {len(examples)}, GPU: {gpu_list}, batch_size: {args.batch_size}")

    chunks = [examples[i::len(gpu_list)] for i in range(len(gpu_list))]
    result_queue: Queue = Queue()
    processes, output_paths = [], []

    for n, (gpu_id, chunk) in enumerate(zip(gpu_list, chunks)):
        out_path = os.path.join(args.output_dir, f"worker_{n}.jsonl")
        output_paths.append(out_path)
        p = Process(target=worker_fn, args=(gpu_id, chunk, out_path, args, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 결과 집계
    total_correct, total_problems = 0, 0
    while not result_queue.empty():
        r = result_queue.get()
        total_correct  += r["n_correct"]
        total_problems += r["n_total"]

    all_records = []
    for path in output_paths:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    all_records.append(json.loads(line))
    all_records.sort(key=lambda x: x["idx"])

    correct_list = [r for r in all_records if r["correct"]]
    wrong_list   = [r for r in all_records if not r["correct"]]

    final_acc = total_correct / total_problems if total_problems > 0 else 0.0
    avg_tokens = sum(r["n_tokens"] for r in all_records) / len(all_records) if all_records else 0.0
    W = 70
    print(f"\n{'='*W}")
    print(f"  Qwen2.5-7B-Instruct 베이스라인")
    print(f"  정확도: {final_acc:.4f}  ({total_correct} / {total_problems})")
    print(f"  평균 토큰: {avg_tokens:.1f}")
    print(f"{'='*W}")

    print(f"\n[O] 맞춘 문제 ({len(correct_list)}개)")
    print(f"  idx: {[r['idx'] for r in correct_list]}")

    print(f"\n[X] 틀린 문제 ({len(wrong_list)}개)")
    print(f"  idx: {[r['idx'] for r in wrong_list]}")

    summary = {
        "model":          MODEL_ID,
        "n_total":        total_problems,
        "n_correct":      total_correct,
        "accuracy":       round(final_acc, 4),
        "avg_tokens":     round(avg_tokens, 1),
        "batch_size":     args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "gpus":           gpu_list,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {args.output_dir}")


if __name__ == "__main__":
    main()
