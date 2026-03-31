import argparse
import json
import os
import random
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue

import torch
from datasets import load_dataset
from tqdm import tqdm

# 프로젝트 루트 및 utils 임포트 설정
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SFT_CHECKPOINT,
    GENERATOR_MODEL_ID,
    MAX_STEPS,
    build_chat_prompt,
    check_end,
    check_solved,
    extract_boxed,
    generate_steps_batched,
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

'''
python source/evaluate_step_reasoning.py --gpus 2,3,4,5 --model_path "/mnt/yoonju/SC/checkpoints/ppo/20260329_185123/iter_0016"
'''
# ─────────────────────────────────────────────────────────────────────────────
# 설정 파일 로드 (Config)
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path="config/config.yaml"):
    """설정 파일을 로드합니다."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 실행 시점에 config.yaml을 먼저 읽습니다.
CONF = load_config()

GPUS     = [6]                                      # 사용할 GPU 번호 목록
DATASETS = ["math500", "amc23", "aime24", "aime25"]          # 평가할 데이터셋 (config data_path의 키)

# 평가할 데이터셋 목록 (DATASETS에 지정된 것만)
EVAL_DATASETS = [(name, CONF['data_path'][name]) for name in DATASETS if name in CONF['data_path']]

# Config 기반 변수 할당 (경로 및 모델 설정)
OUTPUT_ROOT     = CONF['output_path']['eval']
EXTRACTOR_MODEL = CONF['API_model']['EXTRACTOR']

# 액션 토큰 설정 (Config에 정의된 값 사용)
TOKEN_SOLVE     = CONF['model']['token_solve']
TOKEN_CORRECT   = CONF['model']['token_correct']
TOKEN_END       = CONF['model']['token_end']

# 추론 설정
EVAL_MAX_NEW_TOKENS = CONF['step_reasoning']['max_new_tokens']

# ─────────────────────────────────────────────────────────────────────────────
# 배치 문제 풀이 및 결과 생성
# ─────────────────────────────────────────────────────────────────────────────

def solve_batch(model, tokenizer, problems: list, max_steps: int,
                on_result=None, on_step=None, greedy=False, ds_name: str = "") -> list:
    N            = len(problems)
    history      = [[] for _ in range(N)]
    all_steps    = [[] for _ in range(N)]
    all_tokens   = [[] for _ in range(N)]
    terminated   = [False] * N
    next_action  = [TOKEN_SOLVE] * N

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
                                             max_new_tokens=EVAL_MAX_NEW_TOKENS, greedy=greedy)

        newly_terminated = []
        for j, i in enumerate(active):
            reasoning_text, predicted_action, _ = gen_results[j]
            step_text = reasoning_text + (predicted_action or "")

            n_tok = len(tokenizer.encode(step_text, add_special_tokens=False))
            all_steps[i].append({"step_idx": step_idx, "text": step_text, "action": predicted_action})
            all_tokens[i].append(n_tok)
            history[i].append(step_text)

            # 종료 조건 확인
            if check_end(step_text, predicted_action):
                terminated[i] = True
                newly_terminated.append(i)
                if on_result:
                    on_result(i, _make_result(i, problems, all_steps, all_tokens, terminated, ds_name))
            else:
                next_action[i] = predicted_action if predicted_action else TOKEN_SOLVE

        if on_step:
            on_step(step_idx, active, [g[0] for g in gen_results], [len(tokenizer.encode(g[0])) for g in gen_results], newly_terminated)

    results = []
    for i in range(N):
        res = _make_result(i, problems, all_steps, all_tokens, terminated, ds_name)
        if not terminated[i] and on_result:
            on_result(i, res)
        results.append(res)
    return results

def _make_result(i, problems, all_steps, all_tokens, terminated, ds_name: str = "") -> dict:
    last_text = all_steps[i][-1]["text"] if all_steps[i] else ""
    all_text  = "\n".join(s["text"] for s in all_steps[i])
    if "gsm8k" in ds_name.lower():
        boxed   = extract_boxed(all_text, is_gsm8k=True)
        correct = check_solved(all_text, problems[i]["answer"], is_gsm8k=True)
    else:
        boxed   = extract_boxed(all_text)
        correct = check_solved(all_text, problems[i]["answer"])
    return {
        "steps":        all_steps[i],
        "token_counts": all_tokens[i],
        "final_answer": boxed,
        "correct":      correct,
        "n_steps":      len(all_steps[i]),
        "terminated":   terminated[i],
    }

# ─────────────────────────────────────────────────────────────────────────────
# 연속 배치 풀 (한 문제가 끝나면 즉시 다음 문제로 교체)
# ─────────────────────────────────────────────────────────────────────────────

def _new_slot(ex: dict) -> dict:
    return {
        "example":     ex,
        "history":     [],
        "steps":       [],
        "tokens":      [],
        "next_action": TOKEN_SOLVE,
        "n_steps":     0,
    }

def _build_slot_prompt(tokenizer, slot: dict) -> str:
    prob = slot["example"]["problem"]
    if slot["next_action"] == TOKEN_CORRECT:
        return build_chat_prompt(tokenizer, SYSTEM_CORRECT,
                                 _correct_user(prob, slot["history"], ""))
    return build_chat_prompt(tokenizer, SYSTEM_SOLVE,
                             _solve_user(prob, slot["history"]))

def solve_continuous(model, tokenizer, examples: list, batch_size: int, max_steps: int,
                     on_result=None, greedy: bool = False, ds_name: str = "",
                     gpu_id: int = 0) -> None:
    """
    batch_size 개의 슬롯을 유지하면서 추론.
    슬롯 하나가 종료되면 대기 큐에서 다음 문제를 즉시 투입해
    항상 batch_size 개의 step이 병렬 처리된다.
    """
    queue = list(examples)
    total = len(queue)

    # 초기 슬롯 채우기
    n_init = min(batch_size, len(queue))
    active = [_new_slot(queue.pop(0)) for _ in range(n_init)]

    pbar = tqdm(total=total, desc=f"GPU{gpu_id}", position=gpu_id)

    while active:
        prompts = [_build_slot_prompt(tokenizer, s) for s in active]
        gen_results = generate_steps_batched(model, tokenizer, prompts,
                                             max_new_tokens=EVAL_MAX_NEW_TOKENS, greedy=greedy)

        next_active = []
        for slot, (reasoning_text, predicted_action, _) in zip(active, gen_results):
            step_text = reasoning_text + (predicted_action or "")
            n_tok = len(tokenizer.encode(step_text, add_special_tokens=False))
            slot["steps"].append({"step_idx": slot["n_steps"], "text": step_text, "action": predicted_action})
            slot["tokens"].append(n_tok)
            slot["history"].append(step_text)
            slot["n_steps"] += 1

            terminated = check_end(step_text, predicted_action) or slot["n_steps"] >= max_steps
            if terminated:
                if on_result:
                    on_result(slot["example"], slot, terminated=check_end(step_text, predicted_action))
                pbar.update(1)
                # 슬롯이 비면 즉시 다음 문제 투입
                if queue:
                    next_active.append(_new_slot(queue.pop(0)))
            else:
                slot["next_action"] = predicted_action or TOKEN_SOLVE
                next_active.append(slot)

        active = next_active

    pbar.close()

# ─────────────────────────────────────────────────────────────────────────────
# 워커 프로세스 (GPU 할당 및 실행)
# ─────────────────────────────────────────────────────────────────────────────

def worker_fn(gpu_id: int, examples: list, output_path: str, args, result_queue: Queue, ds_name: str = ""):
    import traceback
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    out_dir  = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / (Path(output_path).stem + ".log"))

    out_f = open(output_path, "w", buffering=1, encoding="utf-8")
    log_f = open(log_path,    "w", buffering=1, encoding="utf-8")

    def log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_f.write(f"[{ts}][GPU {gpu_id}] {msg}\n")

    try:
        model_path_arg = args.model_path if args.model_path else None
        model, tokenizer = load_generator(device_map="auto", model_path=model_path_arg)
        log("✓ 모델 로드 완료")

        def on_result(example, slot, terminated=False):
            all_text = "\n".join(s["text"] for s in slot["steps"])
            answer   = example["answer"]
            if "gsm8k" in ds_name.lower():
                boxed   = extract_boxed(all_text, is_gsm8k=True)
                correct = check_solved(all_text, answer, is_gsm8k=True)
            else:
                boxed   = extract_boxed(all_text)
                correct = check_solved(all_text, answer)

            record = {
                "idx":          example["_idx"],
                "problem":      example["problem"],
                "gold_answer":  answer,
                "have_boxed":   terminated,
                "predicted":    boxed,
                "correct":      correct,
                "n_steps":      slot["n_steps"],
                "token_counts": slot["tokens"],
                "steps":        slot["steps"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        solve_continuous(model, tokenizer, examples,
                         batch_size=args.batch_size, max_steps=args.max_steps,
                         on_result=on_result, greedy=True, ds_name=ds_name, gpu_id=gpu_id)

    except Exception:
        log(f"ERROR: {traceback.format_exc()}")
        raise
    finally:
        out_f.close()
        log_f.close()

    result_queue.put({"gpu": gpu_id, "n_total": len(examples)})

# ─────────────────────────────────────────────────────────────────────────────
# 메인 (결과 취합 및 상세 출력)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, default='')
    parser.add_argument("--gpus",         type=str, default=None)
    parser.add_argument("--datasets",     type=str, default=None,
                        help="평가할 데이터셋 이름(콤마 구분). 예: gsm8k,aime24. 미지정 시 기본값(DATASETS) 사용.")
    parser.add_argument("--batch_size",   type=int, default=CONF['step_reasoning']['batch_size'])
    parser.add_argument("--max_steps",    type=int, default=CONF['step_reasoning']['max_steps'])
    args = parser.parse_args()

    if args.gpus is None:
        args.gpus = ",".join(str(g) for g in GPUS)

    # --datasets 인자가 있으면 해당 데이터셋만, 없으면 DATASETS 기본값 사용
    if args.datasets:
        ds_names = [d.strip() for d in args.datasets.split(",")]
    else:
        ds_names = [name for name in DATASETS if name in CONF['data_path']]
    eval_datasets = [(name, CONF['data_path'][name]) for name in ds_names if name in CONF['data_path']]

    if not args.model_path:
        args.model_path = "Qwen/Qwen2.5-7B-Instruct"
        print(f"! 모델 경로가 지정되지 않아 기본 모델({args.model_path})을 사용합니다.")

    gpu_list = [int(g) for g in args.gpus.split(",")]
    model_tag = Path(args.model_path).name if "/" not in args.model_path else args.model_path.split("/")[-1]
    run_dir = os.path.join(OUTPUT_ROOT, f"{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    all_summary = {}

    for ds_name, ds_path in eval_datasets:
        print(f"\n{'='*65}")
        print(f" 데이터셋: {ds_name} ({ds_path})")
        print(f"{'='*65}")

        if not os.path.exists(ds_path):
            print(f"! 데이터셋 파일이 없어 건너뜁니다: {ds_path}")
            continue

        # 파일 포맷 자동 감지 (parquet / jsonl)
        ext = Path(ds_path).suffix.lower()
        fmt = "parquet" if ext == ".parquet" else "json"
        ds = load_dataset(fmt, data_files=ds_path, split="train")
        ds = ds.map(lambda ex: {"problem": _extract_problem(ex), "answer": _extract_answer(ex)})
        examples = list(ds)
        for i, ex in enumerate(examples):
            ex["_idx"] = i

        args.output_dir = os.path.join(run_dir, ds_name)
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"▶ 평가 시작: {model_tag} | GPU: {gpu_list} | 문제 수: {len(examples)}")

        # 멀티프로세싱 실행
        chunks = [examples[i::len(gpu_list)] for i in range(len(gpu_list))]
        result_queue, processes, output_paths = Queue(), [], []

        for n, (gpu_id, chunk) in enumerate(zip(gpu_list, chunks)):
            out_path = os.path.join(args.output_dir, f"worker_{n}.jsonl")
            output_paths.append(out_path)
            p = Process(target=worker_fn, args=(gpu_id, chunk, out_path, args, result_queue, ds_name))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # ─────────────────────────────────────────────────────────────────────
        # 결과 집계
        # ─────────────────────────────────────────────────────────────────────
        total_correct = total_problems = 0

        for path in output_paths:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    total_problems += 1
                    if rec["correct"]: total_correct += 1

        acc_rule = total_correct / total_problems if total_problems > 0 else 0

        W = 65
        print(f"\n{'='*W}")
        print(f" [{ds_name}] {model_tag}")
        print(f"{'-'*W}")
        print(f" 전체 문제 수   : {total_problems}")
        print(f" Rule(Boxed)  : {acc_rule:.4f} ({total_correct}/{total_problems})")
        print(f"{'='*W}")

        ds_summary = {
            "dataset": ds_path,
            "metrics": {
                "acc_rule": round(acc_rule, 4),
            },
            "counts": {
                "total":   total_problems,
                "correct": total_correct,
            },
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(ds_summary, f, indent=2, ensure_ascii=False)

        all_summary[ds_name] = ds_summary

    # 전체 run summary 저장
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model_path, "datasets": all_summary, "config": CONF}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()