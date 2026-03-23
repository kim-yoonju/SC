"""
prototype/evaluate_prototype.py

ppo_prototype.py와 동일한 모델 로딩/추론 방식을 사용해 MATH 데이터셋 평가.

  - load_generator (SFT 체크포인트 + 액션 토큰 등록)
  - SYSTEM_SOLVE + 액션 토큰 기반 step-by-step 추론
  - GPU 2, 3, 5, 6  → multiprocessing 병렬 실행
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue

import torch
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SFT_CHECKPOINT,
    GENERATOR_MODEL_ID,
    MAX_STEPS,
    build_chat_prompt,
    check_solved,
    extract_boxed,
    generate_steps_batched,
    has_boxed,
    load_generator,
    SYSTEM_SOLVE,
    SYSTEM_CORRECT,
    TOKEN_SOLVE,
    TOKEN_CORRECT,
    TOKEN_END,
    _solve_user,
    _correct_user,
    _extract_problem,
    _extract_answer,
)

EVAL_MAX_NEW_TOKENS = 1024  # 평가 시 스텝당 최대 토큰 (학습 시 512와 별개)

# evaluate_prototype.py 는 구 PPO 체크포인트(iter_XXXX) 전용 평가 스크립트.
# utils.py 의 TOKEN_END 는 SFT/PPO 학습 코드와 함께 바뀔 수 있으므로
# 여기서는 평가 대상 모델이 학습될 때 사용된 토큰을 명시적으로 고정한다.
TOKEN_END = "<|end|>"   # iter_0005 이전 PPO 모델은 <|end|> 로 학습됨

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH    = str(_ROOT / "datasets/deepmath_16k.parquet")
OUTPUT_ROOT     = _ROOT / "output" / "eval_results"
GPUS            = [2]
EXTRACTOR_MODEL = "gpt-4.1-nano"  # 마지막 스텝에서 정답 추출용 (LLM fallback)

# ─────────────────────────────────────────────────────────────────────────────
# LLM 정답 추출 (fallback)
# ─────────────────────────────────────────────────────────────────────────────

_openai_client = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def extract_answer_llm(problem: str, last_step_text: str) -> str | None:
    """마지막 스텝 텍스트에서 EXTRACTOR_MODEL로 최종 답을 추출한다."""
    prompt = (
        "You are a math answer extractor. "
        "Given a math problem and the final reasoning step, extract ONLY the final numerical or symbolic answer. "
        "Output ONLY the answer itself with no explanation, no units, no punctuation.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Final reasoning step:\n{last_step_text}\n\n"
        "Final answer:"
    )
    try:
        response = _get_openai_client().chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0,
            timeout=30,
        )
        answer = response.choices[0].message.content.strip()
        return answer if answer else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 배치 문제 풀이
# ─────────────────────────────────────────────────────────────────────────────

def solve_batch(model, tokenizer, problems: list, max_steps: int,
                on_result=None, on_step=None) -> list:
    """
    problems:  [{"problem": str, "answer": str}, ...]
    on_result: callable(i, result) — 문제 i가 끝나는 즉시 호출
    on_step:   callable(step_idx, active, responses, tok_counts, newly_terminated)
    Returns:   [{"steps": [...], "final_answer": str|None, "correct": bool, "n_steps": int}, ...]

    액션 토큰 기반 라우팅:
      TOKEN_END   → 종료
      TOKEN_SOLVE + \\boxed{} → 종료
      TOKEN_SOLVE → SYSTEM_SOLVE 로 다음 스텝 계속
      TOKEN_CORRECT → SYSTEM_CORRECT 로 다음 스텝 교정
    """
    N            = len(problems)
    history      = [[] for _ in range(N)]   # 문제별 스텝 텍스트 누적
    all_steps    = [[] for _ in range(N)]
    all_tokens   = [[] for _ in range(N)]
    terminated   = [False] * N
    next_action  = [TOKEN_SOLVE] * N        # 각 문제의 다음 스텝 액션

    for step_idx in range(max_steps):
        active = [i for i in range(N) if not terminated[i]]
        if not active:
            break

        prompt_texts = []
        for i in active:
            prob = problems[i]["problem"]
            if next_action[i] == TOKEN_CORRECT:
                prompt = build_chat_prompt(tokenizer, SYSTEM_CORRECT,
                                           _correct_user(prob, history[i], ""))
            else:
                prompt = build_chat_prompt(tokenizer, SYSTEM_SOLVE,
                                           _solve_user(prob, history[i]))
            prompt_texts.append(prompt)

        gen_results = generate_steps_batched(model, tokenizer, prompt_texts,
                                             max_new_tokens=EVAL_MAX_NEW_TOKENS)

        newly_terminated = []
        responses  = []
        tok_counts = []

        for j, i in enumerate(active):
            reasoning_text, predicted_action, _ = gen_results[j]
            step_text = reasoning_text + (predicted_action or "")

            n_tok = len(tokenizer.encode(step_text, add_special_tokens=False))
            tok_counts.append(n_tok)
            responses.append(step_text)

            all_steps[i].append({"step_idx": step_idx, "text": step_text, "action": predicted_action})
            all_tokens[i].append(n_tok)
            history[i].append(step_text)

            # 종료 조건: <|end|> 또는 <|solve|>인데 boxed 포함
            if predicted_action == TOKEN_END or (predicted_action == TOKEN_SOLVE and has_boxed(step_text)):
                terminated[i] = True
                newly_terminated.append(i)
                if on_result:
                    on_result(i, _make_result(i, problems, all_steps, all_tokens, terminated))
            else:
                # 다음 스텝 액션 업데이트
                next_action[i] = predicted_action if predicted_action else TOKEN_SOLVE

        if on_step:
            on_step(step_idx, active, responses, tok_counts, newly_terminated)

    # max_steps 초과 미종료 처리
    results = []
    for i in range(N):
        res = _make_result(i, problems, all_steps, all_tokens, terminated)
        if not terminated[i] and on_result:
            on_result(i, res)
        results.append(res)
    return results


def _make_result(i, problems, all_steps, all_tokens, terminated) -> dict:
    last_text = all_steps[i][-1]["text"] if all_steps[i] else ""
    correct   = check_solved(last_text, problems[i]["answer"])
    boxed     = extract_boxed(last_text)
    return {
        "steps":        all_steps[i],
        "token_counts": all_tokens[i],
        "final_answer": boxed,
        "correct":      correct,
        "n_steps":      len(all_steps[i]),
        "terminated":   terminated[i],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 워커 프로세스 (GPU 1개 담당)
# ─────────────────────────────────────────────────────────────────────────────

def worker_fn(gpu_id: int, examples: list, output_path: str, args, result_queue: Queue):
    import traceback

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    out_dir  = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / (Path(output_path).stem + ".log"))

    out_f = open(output_path, "w", buffering=1)
    log_f = open(log_path,    "w", buffering=1)

    def log(msg: str):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}][GPU {gpu_id}] {msg}"
        log_f.write(line + "\n")

    def log_sep(char="─", width=80):
        log_f.write(char * width + "\n")

    log_sep("═")
    log(f"output  : {output_path}")
    log(f"log     : {log_path}")
    log(f"model   : {args.model_path or SFT_CHECKPOINT}")
    log(f"dataset : {args.dataset}")
    log(f"문제 수  : {len(examples)}  |  batch_size={args.batch_size}  max_steps={args.max_steps}")
    log_sep("═")

    try:
        model_path_arg = args.model_path if args.model_path else None
        t0 = time.time()
        log("▶ model/tokenizer 로딩 중 (load_generator)...")
        model, tokenizer = load_generator(device_map="auto", model_path=model_path_arg)
        log(f"✓ 로딩 완료  ({time.time()-t0:.1f}s)")
        log_sep()

        n_correct = 0
        processed = 0
        batches   = [examples[s:s + args.batch_size] for s in range(0, len(examples), args.batch_size)]
        log(f"총 배치 수: {len(batches)}")
        log_sep()

        for batch_idx, batch in enumerate(tqdm(batches, desc=f"GPU{gpu_id}", position=gpu_id,
                                               leave=True, dynamic_ncols=True)):
            t_batch   = time.time()
            first_idx = batch[0]["_idx"]
            last_idx  = batch[-1]["_idx"]
            log_sep("─")
            log(f"BATCH {batch_idx+1}/{len(batches)}  |  idx {first_idx}~{last_idx}  ({len(batch)}문제)")
            log_sep("─")

            for bi, ex in enumerate(batch):
                prob_preview = ex["problem"].replace("\n", " ")[:120]
                log(f"  [{bi:>3}] idx={ex['_idx']}  answer={ex['answer']}  problem: {prob_preview}...")
            log_sep()

            def on_result(local_i, res, _batch=batch):
                nonlocal n_correct
                example = _batch[local_i]

                # LLM fallback 정답 추출
                last_step   = res["steps"][-1]["text"] if res["steps"] else ""
                llm_answer  = extract_answer_llm(example["problem"], last_step)
                llm_correct = check_solved(f"\\boxed{{{llm_answer}}}", example["answer"]) if llm_answer else False

                final_correct = res["correct"] or llm_correct
                if final_correct:
                    n_correct += 1

                record = {
                    "idx":          example["_idx"],
                    "problem":      example["problem"],
                    "gold_answer":  example["answer"],
                    "have_boxed":   res["terminated"],
                    "predicted":    res["final_answer"],
                    "correct":      res["correct"],
                    "llm_answer":   llm_answer,
                    "llm_correct":  llm_correct,
                    "n_steps":      res["n_steps"],
                    "token_counts": res["token_counts"],
                    "steps":        res["steps"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                status = "O" if res["correct"] else ("o" if llm_correct else "X")
                log_sep("·")
                log(f"  DONE [{status}] idx={example['_idx']}  steps={res['n_steps']}  "
                    f"have_boxed={res['terminated']}")
                log(f"         gold    : {example['answer']}")
                log(f"         boxed   : {res['final_answer']}")
                log(f"         llm     : {llm_answer}  (correct={llm_correct})")
                if res["steps"]:
                    last_preview = res["steps"][-1]["text"].replace("\n", " ")[:200]
                    log(f"         last_step: {last_preview}...")

            def on_step(step_idx, active_indices, responses, tok_counts, newly_terminated,
                        _batch=batch):
                elapsed = time.time() - t_batch
                avg_tok = sum(tok_counts) / len(tok_counts) if tok_counts else 0
                max_tok = max(tok_counts) if tok_counts else 0
                log(f"  STEP {step_idx:>2}  |  active={len(active_indices)}  "
                    f"종료={len(newly_terminated)}  "
                    f"avg_tok={avg_tok:.0f}  max_tok={max_tok}  elapsed={elapsed:.1f}s")
                for j, bi in enumerate(active_indices):
                    ex        = _batch[bi]
                    preview   = responses[j].replace("\n", " ")[:150]
                    term_mark = " ← TERMINATED" if bi in newly_terminated else ""
                    log(f"    [{bi:>3}] idx={ex['_idx']}  tok={tok_counts[j]:>4}  | {preview}...{term_mark}")

            solve_batch(
                model, tokenizer, batch,
                max_steps=args.max_steps,
                on_result=on_result,
                on_step=on_step,
            )
            processed += len(batch)
            acc        = n_correct / processed
            t_elapsed  = time.time() - t_batch
            log_sep("─")
            log(f"BATCH {batch_idx+1} 완료  |  누적 {processed}/{len(examples)}  "
                f"acc={acc:.4f}  batch_time={t_elapsed:.1f}s")

    except Exception:
        log("ERROR:\n" + traceback.format_exc())
        raise
    finally:
        out_f.close()
        log_sep("═")
        log("워커 종료")
        log_f.close()

    result_queue.put({"gpu": gpu_id, "n_correct": n_correct, "n_total": len(examples)})


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, default="/mnt/yoonju/SC/checkpoints/prototype/20260322_235727/iter_0001",
                        help="체크포인트 경로. 비워두면 utils.SFT_CHECKPOINT 사용")
    parser.add_argument("--dataset",      type=str, default=DATASET_PATH)
    parser.add_argument("--gpus",         type=str, default=",".join(str(g) for g in GPUS),
                        help="사용할 GPU 번호 (쉼표 구분, 예: 2,3,5,6)")
    parser.add_argument("--batch_size",   type=int, default=128)
    parser.add_argument("--max_steps",    type=int, default=MAX_STEPS)
    args = parser.parse_args()

    gpu_list = [int(g) for g in args.gpus.split(",")]

    if args.model_path:
        args.model_path = str(Path(args.model_path).resolve())
    args.dataset = str(Path(args.dataset).resolve())

    model_tag = Path(args.model_path).name if args.model_path else Path(SFT_CHECKPOINT).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = str(OUTPUT_ROOT / f"{model_tag}_{timestamp}")

    # 데이터셋 로드
    if args.dataset.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=args.dataset, split="train")
    else:
        ds = load_dataset(args.dataset, split="test")

    def normalize(ex):
        ex["problem"] = _extract_problem(ex)
        ex["answer"]  = _extract_answer(ex)
        return ex

    ds       = ds.map(normalize)
    examples = list(ds)
    # 뒤에서 100개 고정 (학습에 사용되지 않은 held-out 샘플)
    examples = examples[-100:]
    random.seed(42)
    random.shuffle(examples)
    for i, ex in enumerate(examples):
        ex["_idx"] = i

    print(f"평가 문제 수: {len(examples)}, GPU: {gpu_list}, batch_size: {args.batch_size}, max_steps: {args.max_steps}")
    print(f"모델: {args.model_path or SFT_CHECKPOINT}")

    # 데이터 분할 및 워커 실행
    chunks       = [examples[i::len(gpu_list)] for i in range(len(gpu_list))]
    result_queue: Queue = Queue()
    processes    = []
    output_paths = []
    for n, (gpu_id, chunk) in enumerate(zip(gpu_list, chunks)):
        out_path = os.path.join(args.output_dir, f"worker_{n}.jsonl")
        output_paths.append(out_path)
        p = Process(target=worker_fn, args=(gpu_id, chunk, out_path, args, result_queue))
        p.start()
        processes.append(p)

    # 결과 집계
    total_correct  = 0
    total_problems = 0
    for _ in processes:
        r = result_queue.get()
        total_correct  += r["n_correct"]
        total_problems += r["n_total"]

    for p in processes:
        p.join()

    all_records = []
    for path in output_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                all_records.append(json.loads(line))
    all_records.sort(key=lambda x: x["idx"])

    boxed_correct = [r for r in all_records if r["correct"]]
    llm_only      = [r for r in all_records if not r["correct"] and r.get("llm_correct")]
    wrong_list    = [r for r in all_records if not r["correct"] and not r.get("llm_correct")]

    n_boxed    = len(boxed_correct)
    n_llm_only = len(llm_only)
    final_acc  = total_correct / total_problems if total_problems > 0 else 0.0
    boxed_acc  = n_boxed / total_problems if total_problems > 0 else 0.0
    W = 70

    print(f"\n{'='*W}")
    print(f"  결과  |  모델: {args.model_path or SFT_CHECKPOINT}")
    print(f"  정확도 (boxed+llm): {final_acc:.4f}  ({total_correct} / {total_problems})")
    print(f"  정확도 (boxed만):   {boxed_acc:.4f}  ({n_boxed} / {total_problems})")
    print(f"  LLM 추출로만 맞춤: {n_llm_only}개")
    print(f"{'='*W}")

    def _fmt(records):
        return {
            "idx":    [r["idx"]          for r in records],
            "steps":  [r["n_steps"]      for r in records],
            "tokens": [r["token_counts"] for r in records],
        }

    print(f"\n[O boxed] {len(boxed_correct)}개")
    for k, v in _fmt(boxed_correct).items():
        print(f"  {k}: {v}")

    print(f"\n[o llm]   {len(llm_only)}개  (boxed 없었으나 llm이 맞춤)")
    for k, v in _fmt(llm_only).items():
        print(f"  {k}: {v}")

    print(f"\n[X]       {len(wrong_list)}개")
    for k, v in _fmt(wrong_list).items():
        print(f"  {k}: {v}")

    summary = {
        "model":           args.model_path or SFT_CHECKPOINT,
        "extractor_model": EXTRACTOR_MODEL,
        "n_total":         total_problems,
        "n_correct":       total_correct,
        "n_correct_boxed": n_boxed,
        "n_correct_llm":   n_llm_only,
        "accuracy":        round(final_acc, 4),
        "accuracy_boxed":  round(boxed_acc, 4),
        "batch_size":      args.batch_size,
        "max_steps":       args.max_steps,
        "gpus":            gpu_list,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {args.output_dir}")


if __name__ == "__main__":
    main()
