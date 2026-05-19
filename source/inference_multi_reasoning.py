"""
inference_multi_reasoning.py
한 스텝씩 생성 → \boxed{} 나올 때까지 반복.

프롬프트 구조 (i번째 스텝 생성 시):
  - [Problem]
  - [Progress so far]  : 1 ~ i-2번째 스텝 (있을 때)
  - [Previous step]    : i-1번째 스텝 (있을 때)
  → 전 스텝 오류 여부 스스로 판단 후 교정 또는 계속 풀기
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    CONF,
    build_chat_prompt,
)
from utils_math import check_solved, has_boxed
from generate_utils import load_prompts as _load_prompts, load_dataset_file

_prompts             = _load_prompts()
SOLVE_PROMPT         = _prompts["multi_step_reasoning"]
STEP_SUMMARY_PROMPT  = _prompts["step_summary"]

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 빠른 설정 (config 값을 덮어씀; 기본값으로 두려면 그대로 유지)
# ─────────────────────────────────────────────────────────────────────────────
GPUS      = []     # 비어있으면 config의 inference.gpus 사용
N_SAMPLES = -1    # -1이면 전체 데이터셋 사용, 양수이면 해당 개수만 추출
MODEL     = ""    # 비어있으면 config의 inference.model 사용

# ─────────────────────────────────────────────────────────────────────────────
_INF_CFG           = CONF["inference"]
GEN_MAX_NEW_TOKENS = _INF_CFG["max_new_tokens"]
MAX_STEPS          = _INF_CFG.get("max_steps", CONF.get("ppo", {}).get("max_steps", 20))
VLLM_MAX_MODEL_LEN = CONF["vllm"]["max_model_len"]
BATCH_PER_GPU      = _INF_CFG.get("batch_per_gpu", 32)


# ─────────────────────────────────────────────────────────────────────────────
# 문제별 상태
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _State:
    item:        dict
    history:     list = field(default_factory=list)  # context용 (summary or full text)
    full_steps:  list = field(default_factory=list)  # 스텝별 전체 추론 텍스트
    summaries:   list = field(default_factory=list)  # 스텝별 한줄 요약 (final step은 None)
    steps_taken: int  = 0
    done:        bool = False


# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트 구성
# ─────────────────────────────────────────────────────────────────────────────

def _build_step_prompt(tokenizer, state: _State) -> str:
    n = len(state.history)          # 지금까지 완료된 스텝 수
    step_number = n + 1

    lines = [f"[Problem]\n{state.item['problem']}"]

    # steps 1 ~ n-1 : 요약 맥락
    if n >= 2:
        lines.append("\n[Progress so far]")
        for i, s in enumerate(state.history[:-1], 1):
            lines.append(f"Step {i}: {s}")

    # step n : 직전 스텝 (오류 검토 대상)
    if n >= 1:
        lines.append(f"\n[Previous step (Step {n})]\n{state.history[-1]}")

    lines.append(f"\nNow write Step {step_number}:")

    return build_chat_prompt(tokenizer, SOLVE_PROMPT, "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 스텝 요약 프롬프트
# ─────────────────────────────────────────────────────────────────────────────

def _build_summary_prompt(tokenizer, problem: str, step_text: str) -> str:
    user = f"[Problem]\n{problem}\n\n[Step]\n{step_text}"
    return build_chat_prompt(tokenizer, STEP_SUMMARY_PROMPT, user)


# ─────────────────────────────────────────────────────────────────────────────
# 스텝 슬라이싱
# ─────────────────────────────────────────────────────────────────────────────

def _extract_step(text: str, step_number: int) -> str:
    """모델 출력에서 step_number 번째 스텝만 추출.
    'Step {N+1}:' 경계로 슬라이싱. 없으면 전체 반환."""
    next_n = step_number + 1
    m = re.search(rf'\bStep\s+{next_n}\s*:', text, re.IGNORECASE)
    return text[:m.start()].strip() if m else text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# resume 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="멀티 스텝 추론 (vLLM)")
    parser.add_argument("--num_data",     type=int, default=None)
    parser.add_argument("--offset",       type=int, default=0)
    parser.add_argument("--output",       type=str, default=None)
    parser.add_argument("--batch_per_gpu", type=int, default=None, help="GPU당 배치 크기 (config 값 override)")
    parser.add_argument("--resume",       type=str, default=_INF_CFG.get("resume"),
                        help="이어서 실행할 기존 JSONL 경로")
    args = parser.parse_args()

    root         = Path(__file__).resolve().parent.parent
    dataset_path = CONF["inference"]["data_path"]

    resume_path = Path(args.resume) if args.resume else None
    done_ids    = load_done_ids(resume_path) if resume_path else set()
    if done_ids:
        print(f"resume: {resume_path}  건너뛸 문제 수={len(done_ids)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = Path(args.output)
    elif resume_path:
        out_path = resume_path
    else:
        out_path = root / "output" / "gen_multi_reasoning" / f"{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items    = load_dataset_file(dataset_path)
    num_data = args.num_data if args.num_data is not None else (N_SAMPLES if N_SAMPLES != -1 else _INF_CFG.get("num_data", -1))
    items    = items[args.offset:] if num_data == -1 else items[args.offset : args.offset + num_data]

    if done_ids:
        before = len(items)
        items  = [it for it in items if str(it.get("id", "?")) not in done_ids]

    rollout_gpus  = GPUS if GPUS else _INF_CFG.get("gpus", CONF.get("generate_trajectory", {}).get("rollout_gpus", [0]))
    base_model_id = MODEL if MODEL else _INF_CFG["model"]
    tp_size       = len(rollout_gpus)
    batch_size    = (args.batch_per_gpu if args.batch_per_gpu is not None else BATCH_PER_GPU) * tp_size

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in rollout_gpus)
    print(f"데이터셋={dataset_path}  문제 수={len(items)}  GPU={rollout_gpus}  출력={out_path}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=base_model_id,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.70,
        max_model_len=VLLM_MAX_MODEL_LEN,
        trust_remote_code=True,
        download_dir=CONF.get("inference", {}).get("cache_dir"),
    )

    sp     = SamplingParams(temperature=0.0, max_tokens=GEN_MAX_NEW_TOKENS)
    sp_sum = SamplingParams(temperature=0.0, max_tokens=128)

    # 슬라이딩 윈도우: 항상 batch_size개 문제를 동시에 처리
    queue   = deque(_State(item=it) for it in items)
    active  = [queue.popleft() for _ in range(min(batch_size, len(queue)))]
    n_correct = 0
    n_saved   = 0
    n_total   = len(items)
    round_idx = 0
    t_start   = time.time()

    file_mode = "a" if resume_path and out_path == resume_path else "w"
    n_batches = (n_total + batch_size - 1) // batch_size

    def _save(fout, state: _State) -> bool:
        full_output = "\n\n".join(state.full_steps)
        is_correct  = check_solved(full_output, state.item["answer"])
        record = {
            "id":          str(state.item.get("id", "?")),
            "problem":     state.item["problem"],
            "answer":      state.item["answer"],
            "total_steps": len(state.full_steps),
            "steps":       [
                {"idx": i + 1, "inference": full, "summary": summ}
                for i, (full, summ) in enumerate(zip(state.full_steps, state.summaries))
            ],
            "is_correct":  is_correct,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        return is_correct

    # ── 스텝별 배치 루프 ──────────────────────────────────────────────────────
    with open(out_path, file_mode, encoding="utf-8") as fout:
        while active:
            batch_idx = n_saved // batch_size + 1
            print(f"[배치 {batch_idx}/{n_batches}] {n_saved}/{n_total} 완료, active={len(active)}", flush=True)
            prompts = [_build_step_prompt(tokenizer, s) for s in active]
            outputs = llm.generate(prompts, sp, use_tqdm=False)
            round_idx += 1

            next_active = []
            all_steps   = []  # (state, step_text, is_done)

            for state, out in zip(active, outputs):
                raw_text  = out.outputs[0].text.strip()
                step_text = _extract_step(raw_text, state.steps_taken + 1)
                state.steps_taken += 1
                is_done = has_boxed(step_text) or state.steps_taken >= MAX_STEPS
                all_steps.append((state, step_text, is_done))

            # 전체 스텝 배치 요약
            sum_prompts = [
                _build_summary_prompt(tokenizer, s.item["problem"], t)
                for s, t, _ in all_steps
            ]
            sum_outputs = llm.generate(sum_prompts, sp_sum, use_tqdm=False)

            for (state, step_text, is_done), sum_out in zip(all_steps, sum_outputs):
                summary = sum_out.outputs[0].text.strip()
                state.full_steps.append(step_text)
                state.summaries.append(summary)
                if is_done:
                    state.history.append(step_text)  # 마지막 스텝은 풀텍스트 보존
                else:
                    state.history.append(summary)

            for state, _, is_done in all_steps:
                if is_done:
                    is_correct = _save(fout, state)
                    if is_correct:
                        n_correct += 1
                    n_saved += 1
                    if queue:
                        next_active.append(queue.popleft())
                else:
                    next_active.append(state)

            active = next_active

    elapsed = (time.time() - t_start) / 60
    print(
        f"완료: {n_total}개 문제 / 정답={n_correct} / 오답={n_total - n_correct} "
        f"({(n_total - n_correct) / max(n_total, 1) * 100:.1f}%) / 소요={elapsed:.1f}분 / 출력={out_path}"
    )


if __name__ == "__main__":
    main()
