"""
MATH500 평가 스크립트

모델이 문제를 단계별로 풀도록 하고 최종 정답을 채점한다.
결과는 JSONL과 summary.json 으로 저장된다.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    apply_chat_template,
    check_answer_correct,
    extract_first_action,
    format_messages,
    parse_boxed,
    parse_step,
)


# ---------------------------------------------------------------------------
# 문제 풀이
# ---------------------------------------------------------------------------

def solve_problem(
    model,
    tokenizer,
    problem: str,
    max_steps: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """
    한 문제를 <end>에 도달하거나 max_steps 초과할 때까지 단계별로 풀고 결과를 반환한다.
    temperature=0 이면 greedy decoding.
    """
    history = []
    steps = []

    for step_idx in range(max_steps):
        messages = format_messages(problem, history)
        input_text = apply_chat_template(tokenizer, messages)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            if temperature > 0:
                gen_kwargs.update(temperature=temperature, do_sample=True)
            else:
                gen_kwargs["do_sample"] = False

            outputs = model.generate(**gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
        step_text = extract_first_action(generated)
        action, content = parse_step(step_text)

        steps.append({
            "step_idx": step_idx,
            "action": action,
            "content": content,
            "text": step_text,
        })

        if action == "end":
            return {
                "steps": steps,
                "num_steps": step_idx + 1,
                "final_answer": content,
                "terminated": True,
            }

        history.append(step_text)

    return {
        "steps": steps,
        "num_steps": max_steps,
        "final_answer": None,
        "terminated": False,
    }


# ---------------------------------------------------------------------------
# 데이터셋 로드
# ---------------------------------------------------------------------------

def load_eval_dataset(dataset_name: str, split: str):
    if dataset_name.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=dataset_name, split="train")
    else:
        ds = load_dataset(dataset_name, split=split)

    def normalize(example):
        if "problem" not in example:
            prompt = example.get("prompt", [])
            if isinstance(prompt, list):
                user_msgs = [m["content"] for m in prompt if m.get("role") == "user"]
                example["problem"] = user_msgs[0] if user_msgs else ""
            else:
                example["problem"] = example.get("question", "")
        if "answer" not in example:
            if "final_answer" in example:
                example["answer"] = example["final_answer"]
            else:
                reward_model = example.get("reward_model", {})
                if isinstance(reward_model, dict) and "ground_truth" in reward_model:
                    example["answer"] = reward_model["ground_truth"]
                else:
                    solution = example.get("solution", "")
                    example["answer"] = parse_boxed(solution) or solution
        return example

    return ds.map(normalize)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MATH500 평가")
    parser.add_argument("--model_name", type=str, required=True,
                        help="평가할 모델 경로")
    parser.add_argument("--dataset", type=str, default="datasets/math500.parquet",
                        help="평가 데이터셋 이름 또는 로컬 .parquet 경로")
    parser.add_argument("--split", type=str, default="test",
                        help="데이터셋 split (MATH500은 보통 test)")
    parser.add_argument("--output_dir", type=str, default="data/eval_results",
                        help="결과 저장 디렉토리")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="평가할 최대 문제 수 (None이면 전체)")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0이면 greedy decoding")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--tag", type=str, default="",
                        help="결과 파일 구분용 태그 (예: before_train, after_train)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    # 모델 로드
    print(f"[eval] 모델 로드: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 데이터셋
    print(f"[eval] 데이터셋 로드: {args.dataset} ({args.split})")
    dataset = load_eval_dataset(args.dataset, args.split)
    if args.max_problems:
        dataset = dataset.select(range(min(args.max_problems, len(dataset))))
    print(f"[eval] 평가 문제 수: {len(dataset)}")

    # 결과 파일
    tag = f"_{args.tag}" if args.tag else ""
    results_file = os.path.join(args.output_dir, f"results{tag}.jsonl")
    summary_file = os.path.join(args.output_dir, f"summary{tag}.json")

    n_correct = 0
    n_terminated = 0
    action_counts = {"solve": 0, "correct": 0, "end": 0, "invalid": 0}

    with open(results_file, "w") as f:
        for i, example in enumerate(tqdm(dataset, desc="평가")):
            problem = example["problem"]
            gold_answer = example["answer"]

            result = solve_problem(
                model, tokenizer, problem,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            pred_answer = result["final_answer"]
            is_correct = check_answer_correct(pred_answer or "", gold_answer)

            if is_correct:
                n_correct += 1
            if result["terminated"]:
                n_terminated += 1

            for step in result["steps"]:
                act = step.get("action") or "invalid"
                if act in action_counts:
                    action_counts[act] += 1
                else:
                    action_counts["invalid"] += 1

            record = {
                "problem_id": str(i),
                "problem": problem,
                "gold_answer": gold_answer,
                "predicted_answer": pred_answer,
                "correct": is_correct,
                "num_steps": result["num_steps"],
                "terminated": result["terminated"],
                "steps": result["steps"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 50 == 0:
                acc = n_correct / (i + 1)
                print(f"  [{i + 1}/{len(dataset)}] 정확도: {acc:.3f}")

    final_acc = n_correct / len(dataset)
    summary = {
        "model": args.model_name,
        "dataset": args.dataset,
        "split": args.split,
        "n_total": len(dataset),
        "n_correct": n_correct,
        "accuracy": round(final_acc, 4),
        "n_terminated": n_terminated,
        "termination_rate": round(n_terminated / len(dataset), 4),
        "action_counts": action_counts,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[eval] 최종 정확도: {final_acc:.3f} ({n_correct}/{len(dataset)})")
    print(f"[eval] 결과 저장: {results_file}")
    print(f"[eval] 요약 저장: {summary_file}")


if __name__ == "__main__":
    main()
