import argparse
import asyncio
import json
import os
import sys
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue
from typing import List, Optional

import torch
from datasets import load_dataset

# 프로젝트 루트 및 utils 임포트 설정
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    ACTION_TOKENS,
    SFT_CHECKPOINT,
    GENERATOR_MODEL_ID,
    GENERATOR_CACHE_DIR,
    VLLM_MAX_MODEL_LEN,
    MAX_STEPS,
    build_chat_prompt,
    check_end,
    check_solved,
    extract_boxed,
    SYSTEM_SOLVE,
    SYSTEM_SOLVE_SFT,
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
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONF = load_config()

GPUS     = [6]
DATASETS = ["math500", "amc23", "aime24", "aime25"]

EVAL_DATASETS = [(name, CONF['data_path'][name]) for name in DATASETS if name in CONF['data_path']]

OUTPUT_ROOT     = CONF['output_path']['eval']

TOKEN_SOLVE     = CONF['model']['token_solve']
TOKEN_CORRECT   = CONF['model']['token_correct']
TOKEN_END       = CONF['model']['token_end']

EVAL_MAX_NEW_TOKENS = CONF['step_reasoning']['max_new_tokens']
_MAX_HISTORY_TOKENS = 4096

# 상태 상수
SOLVE       = "solve"
CORRECT_GEN = "correct_gen"
CORRECT_PAT = "correct_pat"

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수 (generate_trajectory와 동일한 로직)
# ─────────────────────────────────────────────────────────────────────────────

def _trim_history(history: List[str], tokenizer, max_tokens: int = _MAX_HISTORY_TOKENS) -> List[str]:
    """history를 최근 max_tokens 이내로 trim."""
    if not history:
        return history
    total, keep = 0, []
    for step in reversed(history):
        n = len(tokenizer(step, add_special_tokens=False)["input_ids"])
        if total + n > max_tokens:
            break
        keep.append(step)
        total += n
    return list(reversed(keep))


def _next_state(current_state: str, pred_action: str, text: str) -> Optional[str]:
    """eval 전용: 모델이 생성한 액션 토큰 기반 상태 전환. None이면 종료."""
    if pred_action == TOKEN_END:
        return None
    if pred_action == TOKEN_CORRECT:
        if current_state == CORRECT_GEN:
            return CORRECT_PAT
        if current_state == CORRECT_PAT:
            return None
        return CORRECT_GEN
    return SOLVE  # TOKEN_SOLVE


def _build_eval_prompt(tokenizer, state: str, problem: str, history: List[str]) -> str:
    """상태에 따라 평가용 chat prompt를 생성 (history trim 포함)."""
    trimmed = _trim_history(history, tokenizer)
    if state in (CORRECT_GEN, CORRECT_PAT):
        return build_chat_prompt(tokenizer, SYSTEM_CORRECT, _correct_user(problem, trimmed))
    return build_chat_prompt(tokenizer, SYSTEM_SOLVE_SFT, _solve_user(problem, trimmed))


def _parse_vllm_output(completion, action_token_ids, im_end_id, eos_token_id, tokenizer):
    """vLLM CompletionOutput → (pred_action, text_token_ids)."""
    token_ids   = list(completion.token_ids)
    stop_reason = completion.stop_reason

    if isinstance(stop_reason, int) and stop_reason in action_token_ids:
        pred_action = tokenizer.decode([stop_reason])
        text_tids   = token_ids[:-1] if token_ids and token_ids[-1] == stop_reason else token_ids
    elif isinstance(stop_reason, int) and stop_reason in {eos_token_id, im_end_id}:
        pred_action = TOKEN_END
        text_tids   = token_ids[:-1] if token_ids and token_ids[-1] == stop_reason else token_ids
    else:
        # max_tokens 도달 → 계속 풀기
        pred_action = TOKEN_SOLVE
        text_tids   = token_ids

    return pred_action, text_tids


# ─────────────────────────────────────────────────────────────────────────────
# vLLM 기반 연속 평가 (generate_trajectory와 동일한 async continuous batching)
# ─────────────────────────────────────────────────────────────────────────────

async def _solve_eval_vllm(engine, tokenizer, examples: list, max_steps: int,
                            on_result, ds_name: str, gpu_id: int, log_fn=None):
    """vLLM AsyncLLMEngine 기반 평가.

    asyncio.gather로 모든 문제를 동시 실행. 각 문제는 독립적으로 스텝을 진행.
    patcher 없음 — CORRECT_PAT 상태 도달 시 종료.
    """
    from vllm import SamplingParams

    def log(msg):
        if log_fn:
            log_fn(msg)

    action_token_ids = set(
        tid for tid in tokenizer.convert_tokens_to_ids(ACTION_TOKENS)
        if tid != tokenizer.unk_token_id
    )
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids  = list(action_token_ids) + [tokenizer.eos_token_id, im_end_id]

    total      = len(examples)
    done_count = [0]

    log(f"평가 시작: 총 {total}문제, max_steps={max_steps}")

    async def _process_one(example):
        idx       = example["_idx"]
        problem   = example["problem"]
        ref_steps = example.get("steps", [])  # per-step action annotations (optional)

        history = []
        steps   = []
        tokens  = []
        state   = SOLVE

        for step_idx in range(max_steps):
            prompt = _build_eval_prompt(tokenizer, state, problem, history)

            sp     = SamplingParams(max_tokens=EVAL_MAX_NEW_TOKENS, temperature=0.0,
                                    stop_token_ids=stop_ids)
            req_id = f"eval_{idx}_{step_idx}_{uuid.uuid4().hex[:6]}"
            final  = None
            async for out in engine.generate(prompt, sp, req_id):
                final = out

            completion              = final.outputs[0]
            pred_action, text_tids  = _parse_vllm_output(
                completion, action_token_ids, im_end_id, tokenizer.eos_token_id, tokenizer
            )

            text = tokenizer.decode(text_tids, skip_special_tokens=True)
            for tok in ACTION_TOKENS:
                text = text.replace(tok, "")
            text  = text.strip()
            n_tok = len(text_tids)

            steps.append({"step_idx": step_idx, "text": text, "action": pred_action})
            tokens.append(n_tok)
            history.append(text)

            log(f"  [prob {idx}] step={step_idx}  action={pred_action}  tokens={n_tok}")

            # 다음 상태 결정: ref_steps에 predicted_next_action → gold_next_action → 모델 pred_action 순으로 사용
            if step_idx < len(ref_steps):
                ref = ref_steps[step_idx]
                if "predicted_next_action" in ref:
                    next_action = ref["predicted_next_action"]
                elif "gold_next_action" in ref:
                    next_action = ref["gold_next_action"]
                else:
                    next_action = pred_action
            else:
                next_action = pred_action

            next_s = _next_state(state, next_action, text)
            # None: TOKEN_END → 정상 종료 / CORRECT_PAT: patcher 없으므로 종료
            is_end = (next_s is None) or (next_s == CORRECT_PAT)
            if is_end:
                done_count[0] += 1
                reason = "end" if next_s is None else "no_patcher"
                log(f"  DONE problem_idx={idx}  steps={step_idx+1}  reason={reason}  action={pred_action}  next_action={next_action}")
                slot = {"steps": steps, "tokens": tokens, "n_steps": step_idx + 1}
                if on_result:
                    on_result(example, slot, terminated=(next_s is None))
                return

            state = next_s

        # max_steps 소진
        done_count[0] += 1
        log(f"  DONE problem_idx={idx}  steps={max_steps}  reason=max_steps")
        slot = {"steps": steps, "tokens": tokens, "n_steps": max_steps}
        if on_result:
            on_result(example, slot, terminated=False)

    await asyncio.gather(*[_process_one(ex) for ex in examples])
    log(f"평가 완료: {done_count[0]}/{total}문제")


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
        line = f"[{ts}][GPU {gpu_id}] {msg}\n"
        log_f.write(line)
        log_f.flush()

    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        from transformers import AutoTokenizer

        model_path = args.model_path or SFT_CHECKPOINT or GENERATOR_MODEL_ID

        # 토크나이저 로드 + 스페셜 토큰 추가
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=GENERATOR_CACHE_DIR, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})

        # vLLM 엔진 초기화
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            enforce_eager=False,
            max_model_len=VLLM_MAX_MODEL_LEN,
            download_dir=GENERATOR_CACHE_DIR,
        )

        async def _init():
            return AsyncLLMEngine.from_engine_args(engine_args)

        engine = loop.run_until_complete(_init())
        log("✓ vLLM 엔진 로드 완료")

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
            out_f.flush()

        loop.run_until_complete(
            _solve_eval_vllm(engine, tokenizer, examples, args.max_steps,
                             on_result, ds_name, gpu_id, log_fn=log)
        )

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

    if args.datasets:
        ds_names = [d.strip() for d in args.datasets.split(",")]
    else:
        ds_names = [name for name in DATASETS if name in CONF['data_path']]
    eval_datasets = [(name, CONF['data_path'][name]) for name in ds_names if name in CONF['data_path']]

    if not args.model_path:
        args.model_path = SFT_CHECKPOINT or GENERATOR_MODEL_ID
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

        # 결과 집계
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
            "metrics": {"acc_rule": round(acc_rule, 4)},
            "counts":  {"total": total_problems, "correct": total_correct},
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(ds_summary, f, indent=2, ensure_ascii=False)

        all_summary[ds_name] = ds_summary

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model_path, "datasets": all_summary, "config": CONF}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
