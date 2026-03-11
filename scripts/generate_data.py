"""
Monte Carlo 롤아웃 데이터 생성 스크립트

흐름:
  문제마다 한 스텝씩 생성 → 해당 스텝 이후 8개 롤아웃 완료 → 정답 도달률로 임시 리워드 계산
  → action 타입에 따라 최종 리워드 계산 → JSONL 저장

리워드 규칙:
  <solve>  : final_reward = temp_reward (MC 정답률)
  <correct>: final_reward = relu(temp_reward - 직전 <solve> 스텝의 final_reward)
  <end>    : 정답이면 (100 - 스텝 수), 틀리면 -100
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
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
# Stopping Criteria: 배치 지원, 닫힘 태그 등장 즉시 생성 중단
# ---------------------------------------------------------------------------

STOP_STRINGS = ["</solve>", "</correct>", "</end>"]


class BatchedActionTagStoppingCriteria(StoppingCriteria):
    """배치 내 모든 시퀀스가 닫힘 태그를 생성하면 중단한다.
    한 시퀀스가 먼저 끝나도 나머지가 끝날 때까지 계속 생성한다.
    """

    def __init__(self, stop_ids_list: list, input_length: int, batch_size: int):
        self.stop_ids_list = stop_ids_list  # list of list[int]
        self.input_length = input_length
        self.done = [False] * batch_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for i in range(len(self.done)):
            if self.done[i]:
                continue
            new_ids = input_ids[i][self.input_length:].tolist()
            for stop in self.stop_ids_list:
                n = len(stop)
                if len(new_ids) >= n and new_ids[-n:] == stop:
                    self.done[i] = True
                    break
        return all(self.done)


def _get_stop_ids(tokenizer) -> list:
    stop_ids = []
    for s in STOP_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_ids.append(ids)
    return stop_ids


# ---------------------------------------------------------------------------
# 배치 생성 함수
# ---------------------------------------------------------------------------

def generate_steps_batch(
    model,
    tokenizer,
    stop_ids: list,
    prompts: list,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
) -> list:
    """여러 프롬프트를 한 번의 배치 호출로 각각 한 스텝씩 생성한다.
    닫힘 태그가 나온 시퀀스는 멈추고, 전체가 끝나면 반환한다.
    """
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList([
        BatchedActionTagStoppingCriteria(stop_ids, input_length, len(prompts))
    ])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    results = []
    for i in range(len(prompts)):
        new_tokens = outputs[i][input_length:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(extract_first_action(generated))
    return results


def generate_one_step(
    model,
    tokenizer,
    problem: str,
    history: list,
    stop_ids: list,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
) -> str:
    """단일 (problem, history) 쌍에 대해 한 스텝을 생성한다."""
    messages = format_messages(problem, history)
    prompt = apply_chat_template(tokenizer, messages)
    return generate_steps_batch(model, tokenizer, stop_ids, [prompt], max_new_tokens, temperature)[0]


# ---------------------------------------------------------------------------
# 배치 MC 롤아웃
# ---------------------------------------------------------------------------

def run_mc_rollouts_batch(
    model,
    tokenizer,
    stop_ids: list,
    problem: str,
    history: list,
    gold_answer: str,
    n_rollouts: int = 8,
    max_steps: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
) -> list:
    """n_rollouts개의 MC 롤아웃을 배치로 병렬 실행한다.
    매 스텝마다 아직 완료되지 않은 롤아웃들을 한 번의 generate 호출로 처리한다.
    """
    histories = [list(history) for _ in range(n_rollouts)]
    results = [{"correct": False, "num_steps": 0} for _ in range(n_rollouts)]
    done = [False] * n_rollouts

    for _ in range(max_steps):
        active = [i for i in range(n_rollouts) if not done[i]]
        if not active:
            break

        # 활성 롤아웃 프롬프트 배치 구성
        prompts = [
            apply_chat_template(tokenizer, format_messages(problem, histories[i]))
            for i in active
        ]
        step_texts = generate_steps_batch(
            model, tokenizer, stop_ids, prompts, max_new_tokens, temperature
        )

        for j, i in enumerate(active):
            step_text = step_texts[j]
            action, content = parse_step(step_text)

            if action == "end":
                is_correct = check_answer_correct(content or step_text, gold_answer)
                results[i] = {
                    "correct": is_correct,
                    "num_steps": len(histories[i]) + 1,
                }
                done[i] = True
            elif action is None:
                results[i]["num_steps"] = len(histories[i]) + 1
                done[i] = True
            else:
                histories[i].append(step_text)
                results[i]["num_steps"] = len(histories[i])

    return results


# ---------------------------------------------------------------------------
# 문제 처리
# ---------------------------------------------------------------------------

def process_problem(
    model,
    tokenizer,
    stop_ids: list,
    problem: str,
    gold_answer: str,
    problem_id: str,
    args,
) -> Optional[dict]:
    """한 문제를 스텝별로 생성하고 MC 롤아웃(배치) 리워드를 계산한다."""
    history = []
    steps_data = []
    prev_solve_reward = 0.0

    print(f"[{problem_id}] 문제 시작", flush=True)

    for step_idx in range(args.max_steps):
        # 1) 한 스텝 생성
        step_text = generate_one_step(
            model, tokenizer, problem, history, stop_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        action, content = parse_step(step_text)

        if action is None:
            print(f"[{problem_id}] step {step_idx}: 유효하지 않은 action → 종료", flush=True)
            break

        print(f"[{problem_id}] step {step_idx}: <{action}>", flush=True)

        step_data = {
            "step_idx": step_idx,
            "action": action,
            "content": content,
            "text": step_text,
            "history_before": list(history),
        }

        # 2) <end> 처리: 직접 리워드 계산
        if action == "end":
            is_correct = check_answer_correct(content or step_text, gold_answer)
            num_steps = step_idx + 1
            final_reward = float(100 - num_steps) if is_correct else -100.0
            step_data["temp_reward"] = 1.0 if is_correct else 0.0
            step_data["mc_rollouts"] = []
            step_data["is_correct"] = is_correct
            step_data["final_reward"] = final_reward
            steps_data.append(step_data)
            print(f"[{problem_id}] <end> 정답={is_correct} reward={final_reward:.1f}", flush=True)
            break

        # 3) MC 롤아웃 배치 실행 (n_rollouts개를 한 번에)
        print(f"[{problem_id}] step {step_idx}: MC 롤아웃 {args.n_rollouts}개 배치 실행", flush=True)
        history_after = history + [step_text]
        rollouts = run_mc_rollouts_batch(
            model, tokenizer, stop_ids, problem, history_after, gold_answer,
            n_rollouts=args.n_rollouts,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        n_correct = sum(1 for r in rollouts if r["correct"])
        temp_reward = n_correct / args.n_rollouts

        step_data["temp_reward"] = temp_reward
        step_data["mc_rollouts"] = [
            {"correct": r["correct"], "num_steps": r["num_steps"]}
            for r in rollouts
        ]

        # 4) 최종 리워드 계산
        if action == "solve":
            final_reward = temp_reward
            prev_solve_reward = final_reward
            step_data["prev_solve_reward"] = None
        elif action == "correct":
            final_reward = max(0.0, temp_reward - prev_solve_reward)
            step_data["prev_solve_reward"] = prev_solve_reward
        else:
            final_reward = 0.0
            step_data["prev_solve_reward"] = None

        step_data["final_reward"] = final_reward
        steps_data.append(step_data)
        print(
            f"[{problem_id}] step {step_idx}: MC {n_correct}/{args.n_rollouts} 정답 "
            f"temp={temp_reward:.3f} final={final_reward:.3f}",
            flush=True,
        )
        history.append(step_text)

    if not steps_data:
        return None

    return {
        "problem_id": problem_id,
        "problem": problem,
        "gold_answer": gold_answer,
        "steps": steps_data,
    }


# ---------------------------------------------------------------------------
# 배치 문제 처리 (여러 문제를 동시에 step 생성)
# ---------------------------------------------------------------------------

def process_problems_batch(model, tokenizer, stop_ids, batch, args):
    """
    여러 문제를 동시에 처리한다.
    같은 step_idx의 메인 스텝 생성을 하나의 배치로 묶고,
    각 문제의 MC 롤아웃은 개별적으로 배치 실행한다.
    """
    n = len(batch)
    global_ids = [str(gi) for gi, _ in batch]
    problems = [ex["problem"] for _, ex in batch]
    gold_answers = [ex["answer"] for _, ex in batch]

    histories = [[] for _ in range(n)]
    steps_data = [[] for _ in range(n)]
    prev_solve_rewards = [0.0] * n
    active = list(range(n))  # 아직 완료되지 않은 문제 인덱스

    for step_idx in range(args.max_steps):
        if not active:
            break

        # 활성 문제들의 프롬프트를 한 배치로 생성
        prompts = [
            apply_chat_template(tokenizer, format_messages(problems[i], histories[i]))
            for i in active
        ]
        step_texts = generate_steps_batch(
            model, tokenizer, stop_ids, prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        still_active = []
        for j, i in enumerate(active):
            step_text = step_texts[j]
            action, content = parse_step(step_text)
            pid = global_ids[i]

            if action is None:
                print(f"[{pid}] step {step_idx}: 유효하지 않은 action → 종료", flush=True)
                continue  # 이 문제는 더 이상 처리하지 않음

            print(f"[{pid}] step {step_idx}: <{action}>", flush=True)

            step_data = {
                "step_idx": step_idx,
                "action": action,
                "content": content,
                "text": step_text,
                "history_before": list(histories[i]),
            }

            if action == "end":
                is_correct = check_answer_correct(content or step_text, gold_answers[i])
                final_reward = float(100 - (step_idx + 1)) if is_correct else -100.0
                step_data.update({
                    "temp_reward": 1.0 if is_correct else 0.0,
                    "mc_rollouts": [],
                    "is_correct": is_correct,
                    "final_reward": final_reward,
                })
                steps_data[i].append(step_data)
                print(f"[{pid}] <end> 정답={is_correct} reward={final_reward:.1f}", flush=True)
                continue  # 완료

            # MC 롤아웃 배치 실행
            history_after = histories[i] + [step_text]
            rollouts = run_mc_rollouts_batch(
                model, tokenizer, stop_ids, problems[i], history_after, gold_answers[i],
                n_rollouts=args.n_rollouts,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            n_correct = sum(1 for r in rollouts if r["correct"])
            temp_reward = n_correct / args.n_rollouts

            step_data["temp_reward"] = temp_reward
            step_data["mc_rollouts"] = [
                {"correct": r["correct"], "num_steps": r["num_steps"]} for r in rollouts
            ]

            if action == "solve":
                final_reward = temp_reward
                prev_solve_rewards[i] = final_reward
                step_data["prev_solve_reward"] = None
            elif action == "correct":
                final_reward = max(0.0, temp_reward - prev_solve_rewards[i])
                step_data["prev_solve_reward"] = prev_solve_rewards[i]
            else:
                final_reward = 0.0
                step_data["prev_solve_reward"] = None

            step_data["final_reward"] = final_reward
            steps_data[i].append(step_data)
            print(
                f"[{pid}] step {step_idx}: MC {n_correct}/{args.n_rollouts} "
                f"temp={temp_reward:.3f} final={final_reward:.3f}", flush=True,
            )
            histories[i].append(step_text)
            still_active.append(i)

        active = still_active

    results = []
    for i in range(n):
        if steps_data[i]:
            results.append({
                "problem_id": global_ids[i],
                "problem": problems[i],
                "gold_answer": gold_answers[i],
                "steps": steps_data[i],
            })
    return results, global_ids


# ---------------------------------------------------------------------------
# 데이터셋 로드
# ---------------------------------------------------------------------------

def load_math_dataset(dataset_name: str, split: str):
    """MATH 계열 데이터셋을 로드하고 problem/answer 필드를 통일한다.
    로컬 .parquet 파일 경로도 지원한다.
    """
    if dataset_name.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=dataset_name, split="train")
    else:
        ds = load_dataset(dataset_name, split=split)

    def normalize(example):
        # problem 필드 통일
        if "problem" not in example:
            prompt = example.get("prompt", [])
            if isinstance(prompt, list):
                user_msgs = [m["content"] for m in prompt if m.get("role") == "user"]
                example["problem"] = user_msgs[0] if user_msgs else ""
            else:
                # DeepMath-103K: "question" 필드
                example["problem"] = example.get("question", "")
        # answer 필드 통일
        if "answer" not in example:
            # DeepMath-103K: "final_answer" 필드
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

    ds = ds.map(normalize)
    return ds


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MC 롤아웃 데이터 생성")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace 모델 이름 또는 로컬 경로")
    parser.add_argument("--output_dir", type=str, default="data/rollouts",
                        help="결과 JSONL 저장 디렉토리")
    parser.add_argument("--dataset", type=str, default="datasets/math7500.parquet",
                        help="HuggingFace 데이터셋 이름 또는 로컬 .parquet 경로")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="처리 시작 인덱스 (분산 처리용)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="처리 종료 인덱스 (None이면 끝까지)")
    parser.add_argument("--n_rollouts", type=int, default=8,
                        help="MC 롤아웃 횟수")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="문제당 최대 스텝 수")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="스텝 생성 시 최대 토큰 수")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="생성 온도")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--problem_batch_size", type=int, default=1,
                        help="메인 스텝 생성 시 동시에 처리할 문제 수 (VRAM이 충분하면 늘릴 것)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 모델 로드
    print(f"[generate] 모델 로드 중: {args.model_name}")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 배치 생성 시 left padding 필수

    # 닫힘 태그 토큰 ID 사전 계산 (매 호출마다 encode 하지 않도록)
    stop_ids = _get_stop_ids(tokenizer)
    print(f"[generate] stop token ids: {stop_ids}", flush=True)

    # 데이터셋 로드
    print(f"[generate] 데이터셋 로드 중: {args.dataset} ({args.split})")
    dataset = load_math_dataset(args.dataset, args.split)

    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    end_idx = min(end_idx, len(dataset))
    dataset = dataset.select(range(args.start_idx, end_idx))
    print(f"[generate] 처리 범위: [{args.start_idx}, {end_idx}) → {len(dataset)}개 문제")

    # 출력 파일 (샤딩 지원, 데이터셋 이름 포함)
    dataset_stem = Path(args.dataset).stem  # e.g. "deepmath103k"
    out_filename = f"rollouts_{dataset_stem}_{args.start_idx}_{end_idx}.jsonl"
    output_file = os.path.join(args.output_dir, out_filename)

    # 이미 처리된 ID 확인 (재개 지원)
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["problem_id"])
                except json.JSONDecodeError:
                    pass
        print(f"[generate] 이미 처리된 문제: {len(processed_ids)}개 → 이어서 진행")

    # 진행 상황 추적 파일
    progress_file = os.path.join(
        args.output_dir, f"progress_{dataset_stem}_{args.start_idx}_{end_idx}.json"
    )
    n_done = len(processed_ids)
    n_total = len(dataset)
    reward_sum = 0.0
    reward_count = 0

    def save_progress(current_id: str, extra: dict = None):
        progress = {
            "start_idx": args.start_idx,
            "end_idx": end_idx,
            "n_total": n_total,
            "n_done": n_done,
            "n_remaining": n_total - n_done,
            "pct_done": round(100 * n_done / n_total, 1),
            "avg_final_reward": round(reward_sum / reward_count, 4) if reward_count else None,
            "current_problem_id": current_id,
        }
        if extra:
            progress.update(extra)
        with open(progress_file, "w") as pf:
            json.dump(progress, pf, indent=2, ensure_ascii=False)

    print(f"[generate] problem_batch_size={args.problem_batch_size}, n_rollouts={args.n_rollouts}", flush=True)

    with open(output_file, "a") as f:
        # 미처리 예제만 추려서 problem_batch_size 단위로 묶음
        pending = [
            (args.start_idx + i, example)
            for i, example in enumerate(dataset)
            if str(args.start_idx + i) not in processed_ids
        ]

        for batch_start in tqdm(range(0, len(pending), args.problem_batch_size), desc="롤아웃 생성"):
            batch = pending[batch_start: batch_start + args.problem_batch_size]

            if args.problem_batch_size == 1:
                # 단일 문제 처리
                global_i, example = batch[0]
                problem_id = str(global_i)
                save_progress(problem_id)
                try:
                    result = process_problem(
                        model, tokenizer, stop_ids,
                        example["problem"], example["answer"], problem_id, args,
                    )
                except Exception as e:
                    print(f"\n[generate] 문제 {problem_id} 오류: {e}", flush=True)
                    result = None

                results = [result] if result is not None else []
                ids = [problem_id]

            else:
                # 배치 문제 처리: 같은 step_idx의 스텝 생성을 배치로 묶음
                results, ids = process_problems_batch(
                    model, tokenizer, stop_ids, batch, args,
                )

            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                for step in result["steps"]:
                    r = step.get("final_reward", 0.0)
                    reward_sum += r
                    reward_count += 1

            n_done += len(batch)
            save_progress(str(batch[-1][0]), {"last_saved_id": str(batch[-1][0])})

    save_progress("done", {"completed": True})
    print(f"[generate] 완료. 저장: {output_file}")


if __name__ == "__main__":
    main()
