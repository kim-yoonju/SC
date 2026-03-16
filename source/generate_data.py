"""
Monte Carlo 롤아웃 데이터 생성 스크립트

흐름:
  문제마다 한 스텝씩 생성 → 해당 스텝 이후 8개 롤아웃 완료 → 정답 도달률로 임시 리워드 계산
  → action 타입에 따라 최종 리워드 계산 → JSONL 저장

리워드 규칙:
  <solve>  : final_reward = temp_reward (MC 정답률)
  <correct>: final_reward = relu(temp_reward - 직전 <solve> 스텝의 final_reward)
  <end>    : 정답이면 (100 - 스텝 수), 틀리면 -100

Teacher injection (--use_teacher 옵션):
  generator가 <correct>를 시도했는데 temp_reward == 0.0 인 경우에만
  teacher model(GPT)이 대신 correction step을 생성해 history에 삽입한다.
  실패한 <correct> 스텝은 steps에 기록되지만 history에는 포함되지 않고
  teacher 스텝이 그 자리를 대체한다.
"""

import argparse
import json
import os
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
    """배치 내 모든 시퀀스가 닫힘 태그를 생성하면 중단한다."""

    def __init__(self, stop_ids_list: list, input_length: int, batch_size: int):
        self.stop_ids_list = stop_ids_list
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
    messages = format_messages(problem, history)
    prompt = apply_chat_template(tokenizer, messages)
    return generate_steps_batch(model, tokenizer, stop_ids, [prompt], max_new_tokens, temperature)[0]


# ---------------------------------------------------------------------------
# Teacher step 생성 (--use_teacher 옵션 활성화 시에만 호출)
# ---------------------------------------------------------------------------

def generate_teacher_step(problem: str, history: list) -> Optional[str]:
    """Teacher model(GPT)로 correction step 하나 생성.
    generator의 <correct> 시도가 temp_reward == 0.0 일 때만 호출된다.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        print("[teacher] openai 패키지가 설치되지 않았습니다.", flush=True)
        return None

    messages = format_messages(problem, history)
    prompt = "".join(f"{m['role']}: {m['content']}\n" for m in messages)

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a math expert who fixes incorrect reasoning. Output exactly one action tag: <correct>...</correct>"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[teacher] generation failed: {e}", flush=True)
        return None


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
    histories = [list(history) for _ in range(n_rollouts)]
    results = [{"correct": False, "num_steps": 0} for _ in range(n_rollouts)]
    done = [False] * n_rollouts

    for _ in range(max_steps):
        active = [i for i in range(n_rollouts) if not done[i]]
        if not active:
            break

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
                results[i] = {"correct": is_correct, "num_steps": len(histories[i]) + 1}
                done[i] = True
            elif action is None:
                results[i]["num_steps"] = len(histories[i]) + 1
                done[i] = True
            else:
                histories[i].append(step_text)
                results[i]["num_steps"] = len(histories[i])

    return results


# ---------------------------------------------------------------------------
# 문제 처리 (단일)
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
    history = []
    steps_data = []
    prev_solve_reward = 0.0

    print(f"[{problem_id}] 문제 시작", flush=True)

    for step_idx in range(args.max_steps):
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

        if action == "end":
            is_correct = check_answer_correct(content or step_text, gold_answer)
            num_steps = step_idx + 1
            final_reward = float(100 - num_steps) if is_correct else -100.0
            step_data.update({
                "temp_reward": 1.0 if is_correct else 0.0,
                "mc_rollouts": [],
                "is_correct": is_correct,
                "final_reward": final_reward,
            })
            steps_data.append(step_data)
            print(f"[{problem_id}] <end> 정답={is_correct} reward={final_reward:.1f}", flush=True)
            break

        # MC 롤아웃
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
            {"correct": r["correct"], "num_steps": r["num_steps"]} for r in rollouts
        ]

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
            f"[{problem_id}] step {step_idx}: MC {n_correct}/{args.n_rollouts} "
            f"temp={temp_reward:.3f} final={final_reward:.3f}",
            flush=True,
        )

        # Teacher injection: generator의 correct 시도가 실패한 경우에만
        if args.use_teacher and action == "correct" and temp_reward == 0.0:
            teacher_text = generate_teacher_step(problem, history)
            if teacher_text is not None:
                _, teacher_content = parse_step(teacher_text)
                teacher_step = {
                    "step_idx": step_idx + 0.5,
                    "action": "teacher_correct",
                    "content": teacher_content,
                    "text": teacher_text,
                    "history_before": list(history),
                    "temp_reward": None,
                    "final_reward": None,
                    "teacher": True,
                }
                steps_data.append(teacher_step)
                print(f"[{problem_id}] step {step_idx}: teacher step injected", flush=True)
                # 실패한 correct 대신 teacher step을 history에 삽입
                history.append(teacher_text)
                continue  # step_text(실패 correct)는 history에 추가하지 않음

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
    n = len(batch)
    global_ids = [str(gi) for gi, _ in batch]
    problems = [ex["problem"] for _, ex in batch]
    gold_answers = [ex["answer"] for _, ex in batch]

    histories = [[] for _ in range(n)]
    steps_data = [[] for _ in range(n)]
    prev_solve_rewards = [0.0] * n
    active = list(range(n))

    for step_idx in range(args.max_steps):
        if not active:
            break

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
                continue

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
                continue  # 완료, still_active에 추가하지 않음

            # MC 롤아웃
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

            # Teacher injection: generator의 correct 시도가 실패한 경우에만
            if args.use_teacher and action == "correct" and temp_reward == 0.0:
                teacher_text = generate_teacher_step(problems[i], histories[i])
                if teacher_text is not None:
                    _, teacher_content = parse_step(teacher_text)
                    teacher_step = {
                        "step_idx": step_idx + 0.5,
                        "action": "teacher_correct",
                        "content": teacher_content,
                        "text": teacher_text,
                        "history_before": list(histories[i]),
                        "temp_reward": None,
                        "final_reward": None,
                        "teacher": True,
                    }
                    steps_data[i].append(teacher_step)
                    print(f"[{pid}] step {step_idx}: teacher step injected", flush=True)
                    # 실패한 correct 대신 teacher step을 history에 삽입
                    histories[i].append(teacher_text)
                    still_active.append(i)
                    continue  # step_text(실패 correct)는 history에 추가하지 않음

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
    parser = argparse.ArgumentParser(description="MC 롤아웃 데이터 생성")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/rollouts")
    parser.add_argument("--dataset", type=str, default="datasets/math7500.parquet")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--n_rollouts", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--problem_batch_size", type=int, default=1)
    parser.add_argument("--use_teacher", action="store_true",
                        help="correct 시도 실패(temp_reward=0) 시 GPT teacher가 대신 correction 생성")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    tokenizer.padding_side = "left"

    stop_ids = _get_stop_ids(tokenizer)
    print(f"[generate] stop token ids: {stop_ids}", flush=True)
    if args.use_teacher:
        print("[generate] teacher injection 활성화 (correct 실패 시 GPT 대체)", flush=True)

    print(f"[generate] 데이터셋 로드 중: {args.dataset} ({args.split})")
    dataset = load_math_dataset(args.dataset, args.split)

    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    end_idx = min(end_idx, len(dataset))
    dataset = dataset.select(range(args.start_idx, end_idx))
    print(f"[generate] 처리 범위: [{args.start_idx}, {end_idx}) → {len(dataset)}개 문제")

    dataset_stem = Path(args.dataset).stem
    out_filename = f"rollouts_{dataset_stem}_{args.start_idx}_{end_idx}.jsonl"
    output_file = os.path.join(args.output_dir, out_filename)

    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["problem_id"])
                except json.JSONDecodeError:
                    pass
        print(f"[generate] 이미 처리된 문제: {len(processed_ids)}개 → 이어서 진행")

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
        pending = [
            (args.start_idx + i, example)
            for i, example in enumerate(dataset)
            if str(args.start_idx + i) not in processed_ids
        ]

        for batch_start in tqdm(range(0, len(pending), args.problem_batch_size), desc="롤아웃 생성"):
            batch = pending[batch_start: batch_start + args.problem_batch_size]

            if args.problem_batch_size == 1:
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
            else:
                results, _ = process_problems_batch(
                    model, tokenizer, stop_ids, batch, args,
                )

            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                for step in result["steps"]:
                    r = step.get("final_reward") or 0.0
                    reward_sum += r
                    reward_count += 1

            n_done += len(batch)
            save_progress(str(batch[-1][0]), {"last_saved_id": str(batch[-1][0])})

    save_progress("done", {"completed": True})
    print(f"[generate] 완료. 저장: {output_file}")


if __name__ == "__main__":
    main()
