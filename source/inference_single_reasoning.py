
"""
Qwen2.5-7B-Instruct 평가 스크립트

checkpoint가 있으면 로컬 checkpoint를 불러오고, 없으면 HuggingFace에서 다운로드.

평가 지표:
  - Format 정확도 : \\boxed{} 형식에서 정답 추출 후 비교

실행:
  python source/evaluate_single_reasoning.py
  python source/evaluate_single_reasoning.py --resume output/evaluate_single_reasoning/20260419_083305
  """

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

os.environ["HF_HUB_CACHE"] = "/mnt/.cache/huggingface"

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    CONF,
    load_config,
    build_chat_prompt,
    check_solved,
    _call_llm,
    _print_cost_summary,
)
from generate_utils import load_prompts as _load_prompts, load_dataset_file

GPUS = [4,5]
N_SAMPLES = -1     # -1이면 전체 데이터셋 사용, 양수이면 해당 개수만 추출
MODEL = None # None이면 config의 checkpoint.base 사용, 직접 지정 시 해당 경로 사용
VLLM_MAX_MODEL_LEN = CONF["vllm"]["max_model_len"]
API_MAX_WORKERS = 64  # API 병렬 요청 수

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _is_api_model(model_name: str) -> bool:
    m = model_name.lower()
    return any(m.startswith(p) for p in ("gpt-", "o1", "o3", "o4", "deepseek-", "gemini-", "claude-"))


# ─────────────────────────────────────────────────────────────────────────────
# 평가 메인
# ─────────────────────────────────────────────────────────────────────────────

_ID_RE = re.compile(r'"id"\s*:\s*(?:"([^"]*)"|([\d]+))')

def load_done_ids(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                m = _ID_RE.search(line)
                if m:
                    done.add(m.group(1) or m.group(2))
    return done


def evaluate(dataset_path: str, llm, tokenizer, cfg: dict, system_prompt: str, out_file=None, done_ids: set = None, batch_size: int = None) -> dict:
    from vllm import SamplingParams

    inf_cfg = cfg["inference"]
    max_new_tokens = inf_cfg["max_new_tokens"]
    if batch_size is None:
        batch_size = inf_cfg.get("batch_per_gpu", 32) * len(GPUS)

    all_items = load_dataset_file(dataset_path)
    if N_SAMPLES != -1:
        all_items = all_items[:N_SAMPLES]
    is_gsm8k = "gsm8k" in Path(dataset_path).name.lower()

    if done_ids:
        items = [it for it in all_items if str(it.get("id", "?")) not in done_ids]
        print(f"  문제 수: {len(all_items)} (이미 완료: {len(done_ids)}, 남은: {len(items)})")
    else:
        items = all_items
        print(f"  문제 수: {len(items)}")

    prompts = [build_chat_prompt(tokenizer, system_prompt, it["problem"]) for it in items]
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    outputs = []
    for i in range(0, len(prompts), batch_size):
        outputs.extend(llm.generate(prompts[i:i + batch_size], sp))

    fmt_results = []
    for out, it in tqdm(zip(outputs, items), total=len(items), desc="  평가"):
        r  = out.outputs[0].text.strip()
        fc = check_solved(r, it["answer"], is_gsm8k=is_gsm8k)
        fmt_results.append(fc)

        if out_file is not None:
            row = {
                "id":         it.get("id", "?"),
                "problem":    it.get("problem", ""),
                "answer":     it.get("answer", ""),
                "response":   r,
                "is_correct": fc,
            }
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_file.flush()

    n = len(all_items)
    n_done = len(done_ids) if done_ids else 0
    total_correct = sum(fmt_results)
    return {
        "dataset":        Path(dataset_path).name,
        "n_total":        n,
        "n_done":         n_done,
        "format_correct": total_correct,
        "format_acc":     round(total_correct / max(len(items), 1), 4),
    }


def evaluate_api(dataset_path: str, model_name: str, cfg: dict, system_prompt: str, out_file=None, done_ids: set = None) -> dict:
    inf_cfg = cfg["inference"]
    max_new_tokens = inf_cfg["max_new_tokens"]
    if model_name.lower() in ("o3-mini",) or model_name.lower().startswith("claude-"):
        max_new_tokens = 8192

    all_items = load_dataset_file(dataset_path)
    if N_SAMPLES != -1:
        all_items = all_items[:N_SAMPLES]
    is_gsm8k = "gsm8k" in Path(dataset_path).name.lower()

    if done_ids:
        items = [it for it in all_items if str(it.get("id", "?")) not in done_ids]
        print(f"  문제 수: {len(all_items)} (이미 완료: {len(done_ids)}, 남은: {len(items)})")
    else:
        items = all_items
        print(f"  문제 수: {len(items)}")

    lock = __import__("threading").Lock()
    fmt_results = []

    def _call(it):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": it["problem"]},
        ]
        r = _call_llm(model_name, messages, max_completion_tokens=max_new_tokens) or ""
        r = r.strip()
        fc = check_solved(r, it["answer"], is_gsm8k=is_gsm8k)
        row = {"id": it.get("id", "?"), "problem": it.get("problem", ""), "answer": it.get("answer", ""), "response": r, "is_correct": fc}
        if out_file is not None:
            with lock:
                out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_file.flush()
        return fc

    with ThreadPoolExecutor(max_workers=API_MAX_WORKERS) as ex:
        futures = {ex.submit(_call, it): it for it in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  평가"):
            fmt_results.append(fut.result())

    n = len(all_items)
    n_done = len(done_ids) if done_ids else 0
    total_correct = sum(fmt_results)
    return {
        "dataset":        Path(dataset_path).name,
        "n_total":        n,
        "n_done":         n_done,
        "format_correct": total_correct,
        "format_acc":     round(total_correct / max(len(items), 1), 4),
    }


def print_result(result: dict):
    W = 50
    print(f"\n{'='*W}")
    print(f"  데이터셋  : {result['dataset']}")
    print(f"  문제 수   : {result['n_total']}")
    print(f"{'─'*W}")
    print(f"  Format 정확도  : {result['format_acc']:.4f}  ({result['format_correct']}/{result['n_total']})")
    print(f"{'='*W}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_per_gpu", type=int, default=None, help="GPU당 배치 크기 (config 값 override)")
    p.add_argument("--max_new_tokens", type=int, default=None, help="최대 생성 토큰 수 (config 값 override)")
    p.add_argument("--dataset", default=None, help="단일 데이터셋 경로 (config 값 override)")
    p.add_argument("--resume", default=None, help="이어서 실행할 이전 출력 폴더 경로 (예: output/evaluate_single_reasoning/20260419_073753)")
    return p.parse_args()


def print_summary(results: list[dict]):
    W = 60
    print(f"\n{'='*W}")
    print(f"{'전체 요약':^{W}}")
    print(f"{'='*W}")
    print(f"  {'데이터셋':<30} {'Format':>8} {'N':>6}")
    print(f"  {'─'*48}")
    for r in results:
        print(f"  {r['dataset']:<30} {r['format_acc']:>7.1%} {r['n_total']:>6}")
    print(f"{'='*W}")


def main():
    args = parse_args()
    cfg = load_config()

    system_prompt = _load_prompts()["single_step_reasoning"]

    eval_cfg = cfg["inference"]
    if args.batch_per_gpu is not None:
        eval_cfg["batch_per_gpu"] = args.batch_per_gpu
    batch_size = eval_cfg.get("batch_per_gpu", 32) * len(GPUS)
    if args.max_new_tokens is not None:
        eval_cfg["max_new_tokens"] = args.max_new_tokens

    model_path = MODEL if MODEL else cfg["inference"]["model"]
    use_api = _is_api_model(model_path)

    if use_api:
        print(f"API 모드: {model_path}")
        llm, tokenizer = None, None
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPUS)
        os.environ["NCCL_P2P_DISABLE"] = "1"

        from vllm import LLM
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=len(GPUS),
            gpu_memory_utilization=0.90,
            max_model_len=None,
            trust_remote_code=True,
            enforce_eager=True,
        )

    # 데이터셋 목록 결정
    if args.dataset:
        datasets = {"(CLI)": args.dataset}
    else:
        inference_data_path = cfg["inference"]["data_path"]
        datasets = {"inference": inference_data_path}

    # 출력 폴더 결정 (resume 시 기존 폴더 재사용)
    resume_arg = args.resume or eval_cfg.get("resume") or None
    resume_file = None
    if resume_arg:
        resume_path = Path(resume_arg) if Path(resume_arg).is_absolute() else _ROOT / resume_arg
        if resume_path.suffix == ".jsonl" or resume_path.is_file():
            resume_file = resume_path
            out_dir = resume_path.parent
        else:
            out_dir = resume_path
        print(f"이어서 실행: {resume_file or out_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _ROOT / cfg["output_path"]["Inference"] / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 저장 경로: {out_dir}")

    all_results = []
    for i, (name, dataset_path) in enumerate(datasets.items(), 1):
        if not Path(dataset_path).exists():
            print(f"\n[스킵] 파일 없음: {name} ({dataset_path})")
            continue
        print(f"\n[{i}/{len(datasets)}] {name}")
        out_path = resume_file if resume_file else out_dir / f"{name}.jsonl"
        done_ids = load_done_ids(out_path)
        if done_ids:
            print(f"  이전 결과 {len(done_ids)}개 발견, 이어서 실행합니다.")
        write_mode = "a" if done_ids else "w"
        with open(out_path, write_mode, encoding="utf-8") as out_file:
            if use_api:
                result = evaluate_api(dataset_path, model_path, cfg, system_prompt, out_file, done_ids)
            else:
                result = evaluate(dataset_path, llm, tokenizer, cfg, system_prompt, out_file, done_ids, batch_size)
        print_result(result)
        all_results.append(result)

    if len(all_results) > 1:
        print_summary(all_results)

    if use_api:
        _print_cost_summary()


if __name__ == "__main__":
    main()
