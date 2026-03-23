"""
Math-Shepherd PRM 기반 데이터 생성 스크립트

흐름:
  문제마다 한 스텝씩 생성 → PRM으로 리워드 측정
  → 리워드 > threshold: history에 추가 후 계속
  → 리워드 == 0:  correct 힌트로 재생성 시도
  → correct도 리워드 == 0: GPT API로 해당 스텝 대체
  → <end> 태그: 정답 확인 후 종료

PRM: peiyi9979/math-shepherd-mistral-7b-prm (Math-Shepherd, MC 없이 직접 스코어링)
  - 스텝 구분자 ки 위치의 logit(+) / logit(+, -) 로 스텝 리워드 계산
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
    parse_boxed,
    parse_step,
)

# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a mathematical problem solver that reasons step by step.\n\n"
    "At each turn, output EXACTLY ONE action tag:\n\n"
    "<solve>...</solve>  — one reasoning step toward the answer\n"
    "<correct>...</correct>  — correct a wrong previous step\n"
    "<end>\\boxed{final_answer}</end>  — when you are confident of the final answer\n\n"
    "Rules:\n"
    "- ONE tag per response, nothing outside the tag.\n"
    "- Put the answer inside \\boxed{} within <end> when you know it."
)

# 리워드 == 0 일 때 correction을 유도하는 한 줄 힌트
CORRECT_HINT = (
    "The previous step appears to be incorrect or unhelpful. "
    "Please correct it using the <correct>...</correct> tag."
)

# Math-Shepherd PRM 특수 토큰
PRM_STEP_TAG = "ки"
PRM_GOOD = "+"
PRM_BAD = "-"

STOP_STRINGS = ["</solve>", "</correct>", "</end>"]


# ---------------------------------------------------------------------------
# Stopping Criteria
# ---------------------------------------------------------------------------

class ActionTagStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_ids_list: list, input_length: int):
        self.stop_ids_list = stop_ids_list
        self.input_length = input_length
        self.done = False

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        if self.done:
            return True
        new_ids = input_ids[0][self.input_length:].tolist()
        for stop in self.stop_ids_list:
            n = len(stop)
            if len(new_ids) >= n and new_ids[-n:] == stop:
                self.done = True
                return True
        return False


def _get_stop_ids(tokenizer) -> list:
    stop_ids = []
    for s in STOP_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_ids.append(ids)
    return stop_ids


# ---------------------------------------------------------------------------
# 프롬프트 포맷 (utils.format_messages 재구현, 커스텀 system prompt 사용)
# ---------------------------------------------------------------------------

def _format_messages(problem: str, history: list, extra_hint: Optional[str] = None) -> list:
    user_content = f"Problem:\n{problem}"
    if history:
        user_content += "\n\nPrevious steps:\n"
        for i, step in enumerate(history, 1):
            user_content += f"Step {i}: {step}\n"
    user_content += "\nGenerate your next step (one action tag only):"
    if extra_hint:
        user_content += f"\n\n{extra_hint}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# 스텝 생성
# ---------------------------------------------------------------------------

def generate_one_step(
    model,
    tokenizer,
    stop_ids: list,
    problem: str,
    history: list,
    extra_hint: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
) -> str:
    messages = _format_messages(problem, history, extra_hint)
    prompt = apply_chat_template(tokenizer, messages)

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList([
        ActionTagStoppingCriteria(stop_ids, input_length)
    ])

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    new_tokens = output[0][input_length:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_first_action(generated)


# ---------------------------------------------------------------------------
# PRM 리워드 계산 (Math-Shepherd 방식)
# ---------------------------------------------------------------------------

def compute_prm_reward(
    prm_model,
    prm_tokenizer,
    problem: str,
    history: list,
    current_step_text: str,
) -> float:
    """
    현재까지의 스텝 히스토리 + current_step_text 를 Math-Shepherd PRM 포맷으로 변환하여
    마지막 스텝의 리워드(P(good)) 를 반환한다.

    포맷:
        {problem}

        Step 1: {content1} ки
        Step 2: {content2} ки
        ...

    ки 위치의 logit(+ token) / (logit(+) + logit(-)) 로 리워드 계산.
    """
    all_steps = history + [current_step_text]
    step_contents = []
    for s in all_steps:
        _, c = parse_step(s)
        step_contents.append((c or s).strip())

    formatted = problem.strip() + "\n\n"
    for i, c in enumerate(step_contents):
        formatted += f"Step {i + 1}: {c} {PRM_STEP_TAG}\n"

    inputs = prm_tokenizer(formatted, return_tensors="pt").to(prm_model.device)

    # 토큰 ID (공백 포함 인코딩으로 서브워드 분리 문제 완화)
    step_tag_id = prm_tokenizer.encode(f" {PRM_STEP_TAG}", add_special_tokens=False)[-1]
    good_token_id = prm_tokenizer.encode(f" {PRM_GOOD}", add_special_tokens=False)[-1]
    bad_token_id = prm_tokenizer.encode(f" {PRM_BAD}", add_special_tokens=False)[-1]

    with torch.no_grad():
        outputs = prm_model(**inputs)

    logits = outputs.logits[0]        # [seq_len, vocab_size]
    input_ids = inputs["input_ids"][0]

    step_positions = (input_ids == step_tag_id).nonzero(as_tuple=True)[0]

    if len(step_positions) == 0:
        return 0.5  # ки 토큰을 찾지 못하면 중립값

    # 마지막 ки 위치의 logit → 다음 토큰(+ 또는 -)에 대한 확률
    last_pos = step_positions[-1].item()
    gb_logits = logits[last_pos, [good_token_id, bad_token_id]]
    reward = gb_logits.softmax(dim=0)[0].item()  # P(good)
    return reward


# ---------------------------------------------------------------------------
# GPT Teacher 스텝 생성
# ---------------------------------------------------------------------------

def generate_teacher_step(problem: str, history: list) -> Optional[str]:
    """correct 시도도 리워드 == 0 일 때 GPT API로 correction 스텝 대체 생성."""
    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        print("[teacher] openai 패키지가 없습니다.", flush=True)
        return None

    messages = _format_messages(problem, history, extra_hint=CORRECT_HINT)
    prompt_text = "\n\n".join(
        f"{m['role'].upper()}:\n{m['content']}" for m in messages
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a math expert. The previous reasoning step was wrong. "
                        "Output exactly one correction tag: <correct>...</correct>. "
                        "Nothing else."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[teacher] GPT 호출 실패: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# 문제 처리 (단일)
# ---------------------------------------------------------------------------

def process_problem(
    gen_model,
    gen_tokenizer,
    prm_model,
    prm_tokenizer,
    stop_ids: list,
    problem: str,
    gold_answer: str,
    problem_id: str,
    args,
) -> Optional[dict]:
    history = []
    steps_data = []
    prev_solve_reward = 0.0
    recorded_idx = 0  # steps_data 내 인덱스

    print(f"[{problem_id}] 문제 시작", flush=True)

    for solve_iter in range(args.max_steps):
        # ── 1. 스텝 생성 ────────────────────────────────────────────────
        step_text = generate_one_step(
            gen_model, gen_tokenizer, stop_ids,
            problem, history,
            extra_hint=None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        action, content = parse_step(step_text)

        if action is None:
            print(f"[{problem_id}] iter {solve_iter}: 유효하지 않은 action → 종료", flush=True)
            break

        print(f"[{problem_id}] iter {solve_iter}: <{action}>", flush=True)

        # ── 2. <end> 처리 (boxed 정답) ──────────────────────────────────
        if action == "end":
            is_correct = check_answer_correct(content or step_text, gold_answer)
            final_reward = float(100 - (solve_iter + 1)) if is_correct else -100.0
            steps_data.append({
                "step_idx": recorded_idx,
                "action": "end",
                "content": content,
                "text": step_text,
                "history_before": list(history),
                "temp_reward": 1.0 if is_correct else 0.0,
                "mc_rollouts": [],
                "is_correct": is_correct,
                "final_reward": final_reward,
                "teacher": False,
            })
            print(f"[{problem_id}] <end> 정답={is_correct} reward={final_reward:.1f}", flush=True)
            break

        # ── 3. PRM 리워드 계산 ───────────────────────────────────────────
        prm_reward = compute_prm_reward(
            prm_model, prm_tokenizer, problem, history, step_text
        )
        print(f"[{problem_id}] iter {solve_iter}: PRM={prm_reward:.4f}", flush=True)

        # ── 4. 리워드 == 0 → correct 시도 ───────────────────────────────
        if prm_reward <= args.reward_threshold:
            # 원래 스텝 기록 (failed)
            if action == "solve":
                orig_final = prm_reward
                prev_solve_reward = orig_final
            elif action == "correct":
                orig_final = max(0.0, prm_reward - prev_solve_reward)
            else:
                orig_final = 0.0

            steps_data.append({
                "step_idx": recorded_idx,
                "action": action,
                "content": content,
                "text": step_text,
                "history_before": list(history),
                "temp_reward": prm_reward,
                "mc_rollouts": [],
                "prev_solve_reward": prev_solve_reward if action in ("correct",) else None,
                "final_reward": orig_final,
                "teacher": False,
            })
            recorded_idx += 1

            print(f"[{problem_id}] iter {solve_iter}: 리워드 낮음 → correct 시도", flush=True)

            # correct 힌트와 함께 재생성
            correct_text = generate_one_step(
                gen_model, gen_tokenizer, stop_ids,
                problem, history,
                extra_hint=CORRECT_HINT,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            correct_action, correct_content = parse_step(correct_text)
            correct_prm = compute_prm_reward(
                prm_model, prm_tokenizer, problem, history, correct_text
            )
            print(f"[{problem_id}] iter {solve_iter}: correct PRM={correct_prm:.4f}", flush=True)

            if correct_prm <= args.reward_threshold:
                # correct도 리워드 == 0 → GPT 대체
                print(f"[{problem_id}] iter {solve_iter}: correct도 낮음 → GPT teacher", flush=True)

                # 실패한 correct 기록
                steps_data.append({
                    "step_idx": recorded_idx,
                    "action": correct_action or "correct",
                    "content": correct_content,
                    "text": correct_text,
                    "history_before": list(history),
                    "temp_reward": correct_prm,
                    "mc_rollouts": [],
                    "prev_solve_reward": prev_solve_reward,
                    "final_reward": 0.0,
                    "teacher": False,
                })
                recorded_idx += 1

                teacher_text = generate_teacher_step(problem, history)
                if teacher_text is not None:
                    _, teacher_content = parse_step(teacher_text)
                    teacher_prm = compute_prm_reward(
                        prm_model, prm_tokenizer, problem, history, teacher_text
                    )
                    teacher_final = max(0.0, teacher_prm - prev_solve_reward)
                    steps_data.append({
                        "step_idx": recorded_idx,
                        "action": "teacher_correct",
                        "content": teacher_content,
                        "text": teacher_text,
                        "history_before": list(history),
                        "temp_reward": teacher_prm,
                        "mc_rollouts": [],
                        "prev_solve_reward": prev_solve_reward,
                        "final_reward": teacher_final,
                        "teacher": True,
                    })
                    recorded_idx += 1
                    history.append(teacher_text)
                    print(f"[{problem_id}] iter {solve_iter}: teacher step 주입 PRM={teacher_prm:.4f}", flush=True)
                # teacher 실패 시 history에 추가 없이 다음 iter
                continue

            # correct 성공 → correct 스텝 기록 후 history에 추가
            correct_final = max(0.0, correct_prm - prev_solve_reward)
            steps_data.append({
                "step_idx": recorded_idx,
                "action": correct_action or "correct",
                "content": correct_content,
                "text": correct_text,
                "history_before": list(history),
                "temp_reward": correct_prm,
                "mc_rollouts": [],
                "prev_solve_reward": prev_solve_reward,
                "final_reward": correct_final,
                "teacher": False,
            })
            recorded_idx += 1
            history.append(correct_text)
            # correct 후 prev_solve_reward는 유지 (correct는 solve가 아님)
            continue

        # ── 5. 정상 스텝 (리워드 > 0) ──────────────────────────────────
        if action == "solve":
            final_reward = prm_reward
            prev_solve_reward = final_reward
            prev_ref = None
        elif action == "correct":
            final_reward = max(0.0, prm_reward - prev_solve_reward)
            prev_ref = prev_solve_reward
        else:
            final_reward = 0.0
            prev_ref = None

        steps_data.append({
            "step_idx": recorded_idx,
            "action": action,
            "content": content,
            "text": step_text,
            "history_before": list(history),
            "temp_reward": prm_reward,
            "mc_rollouts": [],
            "prev_solve_reward": prev_ref,
            "final_reward": final_reward,
            "teacher": False,
        })
        recorded_idx += 1
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
# 데이터셋 로드
# ---------------------------------------------------------------------------

def load_dataset_normalized(path: str):
    if path.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=path, split="train")
    else:
        ds = load_dataset(path, split="train")

    def normalize(example):
        # problem 필드 정규화
        if "problem" not in example:
            prompt = example.get("prompt", [])
            if isinstance(prompt, list):
                user_msgs = [m["content"] for m in prompt if m.get("role") == "user"]
                example["problem"] = user_msgs[0] if user_msgs else ""
            else:
                example["problem"] = example.get("question", "")
        # answer 필드 정규화
        if "answer" not in example:
            if "final_answer" in example:
                example["answer"] = example["final_answer"]
            else:
                rm = example.get("reward_model", {})
                if isinstance(rm, dict) and "ground_truth" in rm:
                    example["answer"] = rm["ground_truth"]
                else:
                    sol = example.get("solution", "")
                    example["answer"] = parse_boxed(sol) or sol
        return example

    return ds.map(normalize)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Math-Shepherd PRM 기반 데이터 생성")
    parser.add_argument("--gen_model", type=str,
                        default="peiyi9979/math-shepherd-mistral-7b-prm",
                        help="추론 생성 모델")
    parser.add_argument("--prm_model", type=str,
                        default="peiyi9979/math-shepherd-mistral-7b-prm",
                        help="PRM 리워드 모델")
    parser.add_argument("--dataset", type=str,
                        default="datasets/deepmath_1k_cls.parquet")
    parser.add_argument("--output_dir", type=str, default="data/prm_rollouts")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--reward_threshold", type=float, default=0.1,
                        help="이 값 이하이면 리워드 == 0 으로 처리하여 correction 시도")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.torch_dtype]

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    same_model = (args.gen_model == args.prm_model)

    print(f"[init] 생성 모델 로드: {args.gen_model}", flush=True)
    gen_tokenizer = AutoTokenizer.from_pretrained(args.gen_model)
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, torch_dtype=dtype, device_map="auto"
    )
    gen_model.eval()
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.padding_side = "left"

    if same_model:
        print("[init] PRM 모델 = 생성 모델 (공유)", flush=True)
        prm_model = gen_model
        prm_tokenizer = gen_tokenizer
    else:
        print(f"[init] PRM 모델 로드: {args.prm_model}", flush=True)
        prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_model)
        prm_model = AutoModelForCausalLM.from_pretrained(
            args.prm_model, torch_dtype=dtype, device_map="auto"
        )
        prm_model.eval()

    stop_ids = _get_stop_ids(gen_tokenizer)
    print(f"[init] stop token ids: {stop_ids}", flush=True)

    # ── 데이터셋 로드 ──────────────────────────────────────────────────────
    print(f"[init] 데이터셋 로드: {args.dataset}", flush=True)
    dataset = load_dataset_normalized(args.dataset)

    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    end_idx = min(end_idx, len(dataset))
    dataset = dataset.select(range(args.start_idx, end_idx))
    print(f"[init] 처리 범위: [{args.start_idx}, {end_idx}) → {len(dataset)}개", flush=True)

    # ── 출력 파일 ──────────────────────────────────────────────────────────
    stem = Path(args.dataset).stem
    out_file = os.path.join(args.output_dir, f"prm_rollouts_{stem}_{args.start_idx}_{end_idx}.jsonl")

    # 이미 처리된 문제 ID 확인 (재시작 지원)
    processed_ids: set = set()
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["problem_id"])
                except json.JSONDecodeError:
                    pass
        print(f"[init] 이미 처리된 문제: {len(processed_ids)}개 → 이어서 진행", flush=True)

    # ── 생성 루프 ──────────────────────────────────────────────────────────
    pending = [
        (args.start_idx + i, ex)
        for i, ex in enumerate(dataset)
        if str(args.start_idx + i) not in processed_ids
    ]

    with open(out_file, "a") as fout:
        for global_i, example in tqdm(pending, desc="PRM 데이터 생성"):
            problem_id = str(global_i)
            try:
                result = process_problem(
                    gen_model, gen_tokenizer,
                    prm_model, prm_tokenizer,
                    stop_ids,
                    example["problem"],
                    example["answer"],
                    problem_id,
                    args,
                )
            except Exception as e:
                print(f"\n[error] 문제 {problem_id}: {e}", flush=True)
                result = None

            if result is not None:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"[done] 저장 완료: {out_file}", flush=True)


if __name__ == "__main__":
    main()
