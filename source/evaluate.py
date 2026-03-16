"""
MATH500 평가 스크립트

매 스텝마다 classifier head로 다음 액션(solve/correct)을 예측하고,
예측된 액션 태그를 prefix로 붙여 모델이 추론을 생성한다.
배치 단위로 병렬 처리한다.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
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
# Classifier 로드
# ---------------------------------------------------------------------------

def load_classifier(cls_head_path: str, device):
    cls_head_path = Path(cls_head_path)
    config_path = cls_head_path.parent / "classifier_config.json"
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "checkpoints/action_cls/best_model/classifier_config.json"

    with open(config_path) as f:
        cls_config = json.load(f)

    classifier = nn.Linear(cls_config["hidden_size"], cls_config["num_labels"])
    state = torch.load(cls_head_path, map_location="cpu")
    classifier.load_state_dict(state)
    classifier = classifier.to(device=device, dtype=torch.bfloat16)
    classifier.eval()

    id2label = cls_config["id2label"]
    print(f"[eval] Classifier head 로드: {cls_head_path}")
    print(f"       ({cls_config['hidden_size']} → {cls_config['num_labels']} classes: {id2label})")
    return classifier, id2label


# ---------------------------------------------------------------------------
# 배치 문제 풀이 (classifier-guided)
# ---------------------------------------------------------------------------

def solve_batch(
    model,
    tokenizer,
    classifier,
    id2label,
    problems: list,
    max_steps: int = 10,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> list:
    """
    문제 배치를 병렬로 풀고 결과 dict 리스트를 반환한다.
    각 스텝에서:
      1) 활성 문제의 프롬프트를 배치 토크나이즈
      2) forward → classifier → 다음 액션 예측 (배치)
      3) 예측 액션 prefix 붙여 배치 generate
      4) 종료된 문제 마킹
    """
    N = len(problems)
    histories = [[] for _ in range(N)]
    all_steps = [[] for _ in range(N)]
    final_answers = [None] * N
    terminated = [False] * N
    num_steps_done = [0] * N

    # left padding for batched generation
    tokenizer.padding_side = "left"

    for step_idx in range(max_steps):
        active = [i for i in range(N) if not terminated[i]]
        if not active:
            break

        # 1) 활성 문제의 프롬프트 구성
        prompt_texts = []
        for i in active:
            messages = format_messages(problems[i], histories[i])
            prompt_texts.append(apply_chat_template(tokenizer, messages))

        prompt_enc = tokenizer(
            prompt_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(model.device)

        # 2) classifier로 액션 예측: 마지막 레이어 hidden state만 추출 후 즉시 해제
        last_hidden_store = {}

        def _hook(module, inp, out):
            # out: (hidden_state, ...) or just hidden_state tensor
            h = out[0] if isinstance(out, tuple) else out
            last_hidden_store["h"] = h[:, -1, :].clone()  # (B, H) left-pad → 마지막 = 실제 마지막 토큰

        # 모델 구조에 따라 마지막 transformer 레이어 이름이 다를 수 있음
        last_layer = model.model.layers[-1]
        hook_handle = last_layer.register_forward_hook(_hook)
        with torch.no_grad():
            model(**prompt_enc)
        hook_handle.remove()

        boundary_hidden = last_hidden_store.pop("h")  # (B, H)
        torch.cuda.empty_cache()

        cls_device = next(classifier.parameters()).device
        cls_dtype = next(classifier.parameters()).dtype
        logits = classifier(boundary_hidden.to(dtype=cls_dtype, device=cls_device))
        pred_ids = logits.argmax(dim=1).tolist()
        predicted_actions = [id2label[str(pid)] for pid in pred_ids]
        del boundary_hidden, logits

        # 3) 예측 액션 prefix 추가 후 배치 generate
        prefixed_texts = [pt + f"<{pa}>" for pt, pa in zip(prompt_texts, predicted_actions)]
        prefixed_enc = tokenizer(
            prefixed_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(model.device)
        input_len = prefixed_enc["input_ids"].shape[1]

        with torch.no_grad():
            gen_kwargs = dict(
                **prefixed_enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            if temperature > 0:
                gen_kwargs.update(temperature=temperature, do_sample=True)
            else:
                gen_kwargs["do_sample"] = False
            outputs = model.generate(**gen_kwargs)
        del prefixed_enc
        torch.cuda.empty_cache()

        # 4) 결과 파싱 및 상태 업데이트
        for j, i in enumerate(active):
            new_tokens = outputs[j][input_len:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

            prefix_text = f"<{predicted_actions[j]}>"
            step_raw = prefix_text + generated if not generated.startswith(prefix_text) else generated
            step_text = extract_first_action(step_raw)
            action, content = parse_step(step_text)

            if action is None:
                action = predicted_actions[j]
                content = generated.strip()
                step_text = f"<{action}>{content}</{action}>"

            # \boxed{} 가 포함된 경우 → end로 처리하고 답 추출
            if action != "end":
                boxed = parse_boxed(content or "")
                if boxed is not None:
                    action = "end"
                    content = boxed
                    step_text = f"<end>{boxed}</end>"

            all_steps[i].append({
                "step_idx": step_idx,
                "action": action,
                "predicted_action": predicted_actions[j],
                "content": content,
                "text": step_text,
            })
            num_steps_done[i] = step_idx + 1

            if action == "end":
                final_answers[i] = content
                terminated[i] = True
            else:
                histories[i].append(step_text)

    results = []
    for i in range(N):
        results.append({
            "steps": all_steps[i],
            "num_steps": num_steps_done[i],
            "final_answer": final_answers[i],
            "terminated": terminated[i],
        })
    return results


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
    parser = argparse.ArgumentParser(description="MATH500 평가 (classifier-guided, batched)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cls_head_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="datasets/math500.parquet")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="data/eval_results")
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--worker_id", type=int, default=0,
                        help="멀티 GPU 분산 평가 시 이 워커의 인덱스 (0-based)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="멀티 GPU 분산 평가 시 전체 워커 수")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    print(f"[eval] 모델 로드: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype_map[args.torch_dtype],
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cls_head_path = args.cls_head_path
    if cls_head_path is None:
        candidate = Path(args.model_name) / "classifier_head.pt"
        cls_head_path = str(candidate) if candidate.exists() else \
            "checkpoints/action_cls/best_model/classifier_head.pt"
    classifier, id2label = load_classifier(cls_head_path, device=next(model.parameters()).device)

    print(f"[eval] 데이터셋 로드: {args.dataset} ({args.split})")
    dataset = load_eval_dataset(args.dataset, args.split)
    if args.max_problems:
        dataset = dataset.select(range(min(args.max_problems, len(dataset))))

    # 멀티 GPU: 데이터셋을 워커 수만큼 분할
    all_examples_full = list(dataset)
    if args.num_workers > 1:
        chunks = [all_examples_full[i::args.num_workers] for i in range(args.num_workers)]
        all_examples = chunks[args.worker_id]
        print(f"[eval] 워커 {args.worker_id}/{args.num_workers}: "
              f"{len(all_examples)}문제 담당 (전체 {len(all_examples_full)})")
    else:
        all_examples = all_examples_full
        print(f"[eval] 평가 문제 수: {len(all_examples)}, 배치 크기: {args.batch_size}")

    tag = f"_{args.tag}" if args.tag else ""
    worker_suffix = f"_worker{args.worker_id}" if args.num_workers > 1 else ""
    results_file = os.path.join(args.output_dir, f"results{tag}{worker_suffix}.jsonl")
    summary_file = os.path.join(args.output_dir, f"summary{tag}{worker_suffix}.json")

    n_correct = 0
    n_terminated = 0
    action_counts = {"solve": 0, "correct": 0, "end": 0, "invalid": 0}
    pred_action_counts = {"solve": 0, "correct": 0}

    batches = [all_examples[s:s + args.batch_size] for s in range(0, len(all_examples), args.batch_size)]

    with open(results_file, "w") as out_f:
        processed = 0
        for batch_examples in tqdm(batches, desc="배치 평가"):
            problems = [e["problem"] for e in batch_examples]
            gold_answers = [e["answer"] for e in batch_examples]

            batch_results = solve_batch(
                model, tokenizer, classifier, id2label, problems,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            for j, (example, result) in enumerate(zip(batch_examples, batch_results)):
                pred_answer = result["final_answer"]
                is_correct = check_answer_correct(pred_answer or "", gold_answers[j])

                if is_correct:
                    n_correct += 1
                if result["terminated"]:
                    n_terminated += 1

                for step in result["steps"]:
                    act = step.get("action") or "invalid"
                    action_counts[act if act in action_counts else "invalid"] += 1
                    pa = step.get("predicted_action")
                    if pa in pred_action_counts:
                        pred_action_counts[pa] += 1

                record = {
                    "problem_id": str(processed + j),
                    "problem": example["problem"],
                    "gold_answer": gold_answers[j],
                    "predicted_answer": pred_answer,
                    "correct": is_correct,
                    "num_steps": result["num_steps"],
                    "terminated": result["terminated"],
                    "steps": result["steps"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            processed += len(batch_examples)
            acc = n_correct / processed
            tqdm.write(f"  [{processed}/{len(all_examples)}] 정확도: {acc:.3f}")

    final_acc = n_correct / len(all_examples)
    summary = {
        "model": args.model_name,
        "cls_head": cls_head_path,
        "dataset": args.dataset,
        "split": args.split,
        "batch_size": args.batch_size,
        "n_total": len(all_examples),
        "n_correct": n_correct,
        "accuracy": round(final_acc, 4),
        "n_terminated": n_terminated,
        "termination_rate": round(n_terminated / len(all_examples), 4),
        "action_counts": action_counts,
        "predicted_action_counts": pred_action_counts,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[eval] 최종 정확도: {final_acc:.3f} ({n_correct}/{len(all_examples)})")
    print(f"[eval] 예측 액션 분포: {pred_action_counts}")
    print(f"[eval] 결과 저장: {results_file}")
    print(f"[eval] 요약 저장: {summary_file}")


if __name__ == "__main__":
    main()
