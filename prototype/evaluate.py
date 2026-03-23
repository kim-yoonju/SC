"""
prototype/evaluate.py

ppo_prototype.py 학습 루프와 동일한 추론 방식으로 모델 성능을 측정한다.
  - load_generator  : 액션 토큰 등록 포함 (ppo_prototype.py와 동일)
  - generate_steps_batched : batched GPU 추론 (validate_math500과 동일)
  - has_boxed / check_solved : 종료 및 정답 판정 (validate_math500과 동일)

멀티 GPU: multiprocessing 으로 각 GPU가 데이터 분할을 독립적으로 처리

실행 예시:
  # 단일 GPU
  python prototype/evaluate.py --model_path checkpoints/prototype/20260322_000000/iter_0010

  # 멀티 GPU (2,3,5 번 사용)
  python prototype/evaluate.py --model_path checkpoints/prototype/20260322_000000/iter_0010 --gpus 2,3,5

  # deepmath held-out 평가
  python prototype/evaluate.py --model_path checkpoints/prototype/... --dataset datasets/deepmath_16k.parquet --held_out 200
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    MAX_STEPS,
    SFT_CHECKPOINT,
    VAL_BATCH_SIZE,
    MATH500_PATH,
    build_chat_prompt,
    check_solved,
    extract_boxed,
    generate_steps_batched,
    has_boxed,
    load_generator,
    load_math500,
    SYSTEM_SOLVE,
    _solve_user,
    _extract_problem,
    _extract_answer,
)


# ─────────────────────────────────────────────────────────────────────────────
# 핵심 평가 루프 (validate_math500 와 동일, 결과를 세밀하게 기록)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_problems(
    model,
    tokenizer,
    problems: list,
    batch_size: int = VAL_BATCH_SIZE,
    max_steps: int = MAX_STEPS,
    on_done=None,
) -> list:
    """
    problems: [{"problem": str, "answer": str, ...}, ...]

    Returns: [{"problem": str, "answer": str, "steps": [...],
               "final_answer": str|None, "correct": bool,
               "n_steps": int, "terminated": bool}, ...]

    on_done: callable(i, result) — 문제 i 완료 시 호출 (실시간 저장용)
    """
    import torch

    N = len(problems)
    states = [{
        "problem":    p["problem"],
        "answer":     p["answer"],
        **{k: v for k, v in p.items() if k not in ("problem", "answer")},
        "history":    [],
        "steps":      [],
        "done":       False,
        "correct":    False,
        "terminated": False,
    } for p in problems]

    was_training = model.training
    model.eval()

    try:
        for step_idx in range(max_steps):
            active_idx = [i for i in range(N) if not states[i]["done"]]
            if not active_idx:
                break

            for batch_start in range(0, len(active_idx), batch_size):
                mini_idx = active_idx[batch_start: batch_start + batch_size]
                mini     = [states[i] for i in mini_idx]

                prompts = [
                    build_chat_prompt(
                        tokenizer, SYSTEM_SOLVE,
                        _solve_user(s["problem"], s["history"]),
                    )
                    for s in mini
                ]

                with torch.no_grad():
                    gen_outputs = generate_steps_batched(model, tokenizer, prompts)

                for s, (step_text, predicted_action, _) in zip(mini, gen_outputs):
                    s["history"].append(step_text)
                    s["steps"].append({
                        "step_idx":        step_idx,
                        "text":            step_text,
                        "predicted_action": predicted_action,
                    })

                    if has_boxed(step_text):
                        s["done"]       = True
                        s["terminated"] = True
                        s["correct"]    = check_solved(step_text, s["answer"])

    finally:
        if was_training:
            model.train()

    results = []
    for i, s in enumerate(states):
        last_text = s["steps"][-1]["text"] if s["steps"] else ""
        res = {
            "problem":      s["problem"],
            "answer":       s["answer"],
            "steps":        s["steps"],
            "final_answer": extract_boxed(last_text),
            "correct":      s["correct"],
            "n_steps":      len(s["steps"]),
            "terminated":   s["terminated"],
        }
        # 원본 필드 (subject, level 등) 전달
        for k in s:
            if k not in ("problem", "answer", "history", "steps", "done",
                         "correct", "terminated"):
                res[k] = s[k]
        results.append(res)
        if on_done:
            on_done(i, res)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 워커 프로세스 (GPU 1개 담당)
# ─────────────────────────────────────────────────────────────────────────────

def worker_fn(gpu_id: int, worker_idx: int, examples: list, out_path: str,
              model_path: str, args, result_queue: Queue):
    import traceback
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / (Path(out_path).stem + ".log"))

    out_f = open(out_path, "w", buffering=1)
    log_f = open(log_path, "w", buffering=1)

    def log(msg: str):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}][GPU {gpu_id}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log(f"=== 워커 시작  GPU={gpu_id}  문제={len(examples)}  model={model_path} ===")
    log(f"output → {out_path}")

    n_correct = 0
    n_done    = 0

    try:
        t0 = time.time()
        log("모델 로딩 중 (load_generator)...")
        model, tokenizer = load_generator(device_map="auto", model_path=model_path)
        log(f"모델 로딩 완료 ({time.time() - t0:.1f}s)")

        def on_done(local_i, res):
            nonlocal n_correct, n_done
            n_done += 1
            if res["correct"]:
                n_correct += 1
            record = {
                "idx":          examples[local_i].get("_idx", local_i),
                **{k: v for k, v in res.items() if k != "steps"},
                "steps":        res["steps"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            status = "O" if res["correct"] else "X"
            log(f"[{status}] idx={record['idx']}  steps={res['n_steps']}  "
                f"terminated={res['terminated']}  acc={n_correct}/{n_done}")

        batches = [examples[s: s + args.batch_size]
                   for s in range(0, len(examples), args.batch_size)]
        log(f"배치 수: {len(batches)}  batch_size={args.batch_size}  max_steps={args.max_steps}")

        processed = 0
        for batch_idx, batch in enumerate(batches):
            log(f"--- 배치 {batch_idx + 1}/{len(batches)} (문제 {processed}~{processed + len(batch) - 1}) ---")
            t_batch = time.time()
            evaluate_problems(
                model, tokenizer, batch,
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                on_done=lambda li, res, _off=processed: on_done(li + _off, res),
            )
            processed += len(batch)
            log(f"배치 완료 ({time.time() - t_batch:.1f}s)  "
                f"누적 acc={n_correct}/{processed}")

    except Exception:
        log("ERROR:\n" + traceback.format_exc())
        raise
    finally:
        out_f.close()
        log("=== 워커 종료 ===")
        log_f.close()

    result_queue.put({
        "gpu":       gpu_id,
        "n_correct": n_correct,
        "n_total":   len(examples),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_dataset(dataset_path: str, held_out: int) -> list:
    """parquet → [{problem, answer, ...}]  (뒤에서 held_out개, 나머지는 전체)"""
    import random

    if dataset_path.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=dataset_path, split="train")
    else:
        ds = load_dataset(dataset_path, split="test")

    examples = []
    for i, ex in enumerate(ds):
        problem = _extract_problem(ex)
        answer  = _extract_answer(ex)
        if not problem:
            continue
        record = {"problem": problem, "answer": answer}
        for k in ("subject", "level", "problem_id"):
            if k in ex:
                record[k] = ex[k]
        examples.append(record)

    if held_out > 0:
        examples = examples[-held_out:]
        random.seed(42)
        random.shuffle(examples)

    for i, ex in enumerate(examples):
        ex["_idx"] = i

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# 요약 출력 / 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(output_dir: str, all_records: list, args) -> dict:
    from collections import defaultdict

    n_total     = len(all_records)
    n_correct   = sum(1 for r in all_records if r["correct"])
    n_terminated = sum(1 for r in all_records if r["terminated"])
    accuracy    = n_correct / n_total if n_total else 0.0

    by_level:   dict = defaultdict(lambda: {"c": 0, "t": 0})
    by_subject: dict = defaultdict(lambda: {"c": 0, "t": 0})
    for r in all_records:
        lvl  = str(r.get("level",   "unknown"))
        subj = str(r.get("subject", "unknown"))
        by_level[lvl]["t"]    += 1
        by_level[lvl]["c"]    += int(r["correct"])
        by_subject[subj]["t"] += 1
        by_subject[subj]["c"] += int(r["correct"])

    acc_by_level   = {k: round(v["c"] / v["t"], 4) for k, v in sorted(by_level.items()) if v["t"]}
    acc_by_subject = {k: round(v["c"] / v["t"], 4) for k, v in sorted(by_subject.items()) if v["t"]}

    summary = {
        "model":             args.model_path,
        "dataset":           args.dataset,
        "n_total":           n_total,
        "n_correct":         n_correct,
        "accuracy":          round(accuracy, 4),
        "n_terminated":      n_terminated,
        "termination_rate":  round(n_terminated / n_total, 4) if n_total else 0.0,
        "batch_size":        args.batch_size,
        "max_steps":         args.max_steps,
        "gpus":              args.gpus,
        "acc_by_level":      acc_by_level,
        "acc_by_subject":    acc_by_subject,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    W = 70
    print(f"\n{'=' * W}")
    print(f"  모델    : {args.model_path}")
    print(f"  데이터셋: {args.dataset}")
    print(f"  정확도  : {accuracy:.4f}  ({n_correct}/{n_total})")
    print(f"  종료율  : {n_terminated / n_total:.4f}" if n_total else "")
    if acc_by_level:
        print(f"  레벨별  : {acc_by_level}")
    if acc_by_subject and len(acc_by_subject) > 1:
        print(f"  과목별  : {acc_by_subject}")
    print(f"{'=' * W}")
    print(f"결과 저장: {output_dir}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ppo_prototype 체크포인트 평가")
    parser.add_argument("--model_path", type=str, default=SFT_CHECKPOINT,
                        help="평가할 체크포인트 경로 (HF 모델명도 가능)")
    _root = Path(__file__).resolve().parent.parent
    parser.add_argument("--dataset", type=str,
                        default=str(_root / "datasets" / "deepmath_16k.parquet"),
                        help="평가 데이터셋 (.parquet 또는 HF dataset 이름)")
    parser.add_argument("--gpus", type=str, default="2",
                        help="사용할 GPU 번호, 쉼표 구분 (예: 2,3)")
    parser.add_argument("--batch_size", type=int, default=VAL_BATCH_SIZE,
                        help="GPU당 배치 크기 (기본: VAL_BATCH_SIZE)")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS,
                        help="문제당 최대 스텝 수 (기본: MAX_STEPS)")
    parser.add_argument("--held_out", type=int, default=100,
                        help="데이터셋 뒤에서 n개만 사용 (0=전체)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="결과 저장 경로 (기본: output/eval_results/<model>_<ts>)")
    args = parser.parse_args()

    args.model_path = str(Path(args.model_path).resolve())
    args.dataset    = str(Path(args.dataset).resolve()) if Path(args.dataset).exists() else args.dataset
    gpu_list        = [int(g.strip()) for g in args.gpus.split(",")]
    args.gpus       = gpu_list

    # 출력 디렉토리
    if args.output_dir is None:
        model_tag    = Path(args.model_path).name
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path(__file__).resolve().parent.parent
                              / "output" / "eval_results" / f"{model_tag}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 데이터셋 로드
    print(f"[eval] 데이터셋 로드: {args.dataset}" +
          (f"  (held_out={args.held_out})" if args.held_out > 0 else ""))
    if args.dataset == str(Path(MATH500_PATH).resolve()):
        raw = load_math500(args.dataset)
        examples = []
        for i, p in enumerate(raw):
            p["_idx"] = i
            examples.append(p)
    else:
        examples = load_eval_dataset(args.dataset, args.held_out)

    print(f"[eval] 평가 문제 수: {len(examples)}")
    print(f"[eval] 모델         : {args.model_path}")
    print(f"[eval] GPU          : {gpu_list}")
    print(f"[eval] 결과 저장    : {args.output_dir}")

    # 데이터 분할 및 워커 실행
    chunks       = [examples[i::len(gpu_list)] for i in range(len(gpu_list))]
    result_queue = Queue()
    processes    = []
    output_paths = []

    for worker_idx, (gpu_id, chunk) in enumerate(zip(gpu_list, chunks)):
        out_path = os.path.join(args.output_dir, f"worker_{worker_idx}.jsonl")
        output_paths.append(out_path)
        p = Process(
            target=worker_fn,
            args=(gpu_id, worker_idx, chunk, out_path, args.model_path, args, result_queue),
        )
        p.start()
        processes.append(p)

    # 워커 완료 대기
    total_correct  = 0
    total_problems = 0
    for _ in processes:
        r = result_queue.get()
        total_correct  += r["n_correct"]
        total_problems += r["n_total"]
    for p in processes:
        p.join()

    # 결과 집계 및 요약
    all_records = []
    for path in output_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    all_records.sort(key=lambda x: x.get("idx", 0))

    save_summary(args.output_dir, all_records, args)


if __name__ == "__main__":
    main()
