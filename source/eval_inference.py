"""
eval_inference.py — CLS 모델 기반 다중 스텝 추론 정확도 평가
모델 하나는 generator
하나는 PRM or cls 올리고 진짜 성능 측정하는걸 짜야함

실행:
  python source/eval_inference.py --gpus 4,5
  python source/eval_inference.py --gpus 4,5 --dataset datasets/aime24_test.jsonl
  python source/eval_inference.py --gpus 4,5 --resume output/eval_inference/20260524_xxx

GPU 할당:
  GPUs[0]: 추론 모델 (config.checkpoint.base)
  GPUs[1]: CLS 모델  (config.checkpoint.sft_checkpoint)

출력 파일 (output/eval_inference/{timestamp}/):
  traj_all.jsonl  — 문제당 full trajectory (inference + cls_output + action per step)
  results.jsonl   — 문제당 요약 (id, is_correct, n_steps, actions, pred/gold)
"""

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / "utils"))

from utils_sft import (
    CONF, setup_tokenizer,
    build_messages_inference, build_messages_classification, build_chat_prompt,
    TOKEN_SOLVE, TOKEN_RETHINK, TOKEN_END,
)
from utils_math import check_solved, extract_boxed

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
_SC  = CONF.get("grpo_sc", {})
_INF = CONF.get("inference", {})

INF_CKPT     = CONF["checkpoint"]["base"]
CLS_CKPT     = CONF["checkpoint"].get("sft_checkpoint") or CONF["checkpoint"]["base"]
VLLM_MAX_LEN = CONF.get("vllm", {}).get("max_model_len", 32768)
INF_MAX_NEW          = _SC.get("inf_max_new_tokens", _INF.get("max_new_tokens", 4096))
CLS_MAX_NEW          = _SC.get("cls_max_new_tokens", 4096)
MAX_STEPS            = _SC.get("max_steps_per_problem", 20)
BATCH_SIZE           = _INF.get("batch_per_gpu", 32)
ROLLOUT_N    = _SC.get("rollout_n", 8)
RETHINK_TEMP = _SC.get("rethink_temperature", 0.7)


def _load_prompts() -> dict[str, str]:
    path = _ROOT / CONF["prompts"]["file"]
    return {d["name"]: d["content"] for d in json.loads(path.read_text())}


PROMPTS        = _load_prompts()
SYSTEM_INF     = PROMPTS.get("gen_inference", "")
SYSTEM_RETHINK = PROMPTS.get("gen_rethink_inference", "")
SYSTEM_CLS     = PROMPTS.get("gen_classification", "")


# ─────────────────────────────────────────────────────────────────────────────
# ID 추출 (id → unique_id → 순번 순으로 fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _get_id(item: dict, idx: int) -> str:
    for key in ("id", "unique_id"):
        v = item.get(key)
        if v is not None and str(v) != "":
            return str(v)
    return str(idx)


# ─────────────────────────────────────────────────────────────────────────────
# CLS 출력 파싱
# ─────────────────────────────────────────────────────────────────────────────
_RUBRIC_NAME_TO_TOKEN = {
    "Algebraic Manipulation":                 "<|algebraic_manipulation|>",
    "Abstract and Linear Algebra Operations": "<|abstract_and_linear_algebra_operations|>",
    "Calculus Computation":                   "<|calculus_computation|>",
    "Function and Limit Analysis":            "<|function_and_limit_analysis|>",
    "Geometric Reasoning":                    "<|geometric_reasoning|>",
    "Counting and Probability":               "<|counting_and_probability|>",
    "Number Theoretic Reasoning":             "<|number_theoretic_reasoning|>",
    "Logical and Discrete Reasoning":         "<|logical_and_discrete_reasoning|>",
    "Differential Equations":                 "<|differential_equations|>",
    "Progress and Non-Repetition":            "<|progress_and_non-repetition|>",
    "Atomicity":                              "<|atomicity|>",
}
_RUBRIC_SPLIT_RE = re.compile(
    r"\n  (" + "|".join(re.escape(n) for n in _RUBRIC_NAME_TO_TOKEN) + r"):"
)


def parse_action(cls_output: str, inference: str) -> tuple[list[str], str]:
    """CLS 출력에서 (fail_rubrics, action_token) 결정.

    우선 순위:
      1. "Next action:" 섹션에서 직접 추출
      2. 루브릭별 "Verdict: incorrect" 유무로 판단
    """
    # 1. Next action: 직접 파싱
    m = re.search(
        r"Next\s+action:\s*\n\s*(?:<\|)?(solve|rethink|end)(?:\|>)?",
        cls_output, re.IGNORECASE,
    )
    if m:
        tok    = m.group(1).lower()
        action = {TOKEN_SOLVE[2:-2]: TOKEN_SOLVE,
                  TOKEN_RETHINK[2:-2]: TOKEN_RETHINK,
                  TOKEN_END[2:-2]: TOKEN_END}.get(tok, TOKEN_SOLVE)
        fail_rubrics: list[str] = []
        if action == TOKEN_RETHINK:
            parts = _RUBRIC_SPLIT_RE.split(cls_output)
            for i in range(1, len(parts), 2):
                name = parts[i].strip()
                text = parts[i + 1] if i + 1 < len(parts) else ""
                token = _RUBRIC_NAME_TO_TOKEN.get(name)
                if token and re.search(r"Verdict:\s*incorrect", text, re.IGNORECASE):
                    fail_rubrics.append(token)
        return fail_rubrics, action

    # 2. 루브릭 기반 fallback
    fail_rubrics = []
    parts = _RUBRIC_SPLIT_RE.split(cls_output)
    for i in range(1, len(parts), 2):
        name  = parts[i].strip()
        text  = parts[i + 1] if i + 1 < len(parts) else ""
        token = _RUBRIC_NAME_TO_TOKEN.get(name)
        if not token:
            continue
        if re.search(r"Verdict:\s*incorrect", text, re.IGNORECASE):
            fail_rubrics.append(token)

    if fail_rubrics:
        return fail_rubrics, TOKEN_RETHINK
    if extract_boxed(inference) is not None:
        return fail_rubrics, TOKEN_END
    return fail_rubrics, TOKEN_SOLVE


# ─────────────────────────────────────────────────────────────────────────────
# vLLM 생성
# ─────────────────────────────────────────────────────────────────────────────

def _vllm_generate(llm, prompts: list[str], max_new_tokens: int,
                   temperature: float = 0.0) -> list[str]:
    from vllm import SamplingParams
    sp      = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


# ─────────────────────────────────────────────────────────────────────────────
# Rollout 선택: rethink 후보 N개를 생성·완성하고 정답 도달율 기준 best 선택
# ─────────────────────────────────────────────────────────────────────────────

def _batch_rollout_select(
    states_todo: list[dict],
    inf_llm,
    tokenizer,
    rollout_n: int,
    rethink_temp: float,
    max_steps: int,
    inf_max_new: int,
) -> list[dict]:
    """각 state에 대해 rollout_n개의 rethink 후보를 생성하고,
    각 후보를 greedy 완성해 정답 도달 여부를 확인한 뒤 best 후보를 반환.

    완성 스텝 한도: 상태별 남은 예산 (max_steps - state["total_steps"]).

    반환: [{"inf_text": str, "outcomes": list[float], "best_j": int}, ...]
    """
    # 1. 후보 생성 (temperature 샘플링으로 다양성 확보)
    rollout_prompts = []
    for s in states_todo:
        sys_m, usr_m = build_messages_inference(
            s["problem"], s["history"], len(s["history"]), SYSTEM_RETHINK
        )
        prompt = build_chat_prompt(tokenizer, sys_m, usr_m)
        rollout_prompts.extend([prompt] * rollout_n)

    rollout_texts = _vllm_generate(inf_llm, rollout_prompts, inf_max_new, rethink_temp)

    # 2. 후보별 완성 히스토리 준비 + 상태별 남은 스텝 계산
    total       = len(states_todo) * rollout_n
    local_hists = []
    gold_list   = []
    prob_list   = []
    remaining   = []   # 후보별 남은 완성 스텝 예산
    for si, s in enumerate(states_todo):
        rem = max(max_steps - s["total_steps"], 1)
        for j in range(rollout_n):
            rt = rollout_texts[si * rollout_n + j]
            local_hists.append(s["history"] + [{"inference": rt, "is_fail": False}])
            gold_list.append(s["gold_answer"])
            prob_list.append(s["problem"])
            remaining.append(rem)

    done_flags   = [False] * total
    outcomes     = [0.0]   * total
    step_counts  = [0]     * total

    # 3. 완성 루프 (배치) — 후보별 남은 예산 소진까지
    for _ in range(max(remaining)):
        active_k = [
            k for k, d in enumerate(done_flags)
            if not d and step_counts[k] < remaining[k]
        ]
        if not active_k:
            break
        comp_prompts = []
        for k in active_k:
            sys_m, usr_m = build_messages_inference(
                prob_list[k], local_hists[k], len(local_hists[k]), SYSTEM_INF
            )
            comp_prompts.append(build_chat_prompt(tokenizer, sys_m, usr_m))
        comp_texts = _vllm_generate(inf_llm, comp_prompts, inf_max_new)
        for k, text in zip(active_k, comp_texts):
            step_counts[k] += 1
            local_hists[k].append({"inference": text, "is_fail": False})
            if extract_boxed(text) is not None:
                outcomes[k]   = 1.0 if check_solved(
                    text, gold_list[k], problem=prob_list[k]
                ) else 0.0
                done_flags[k] = True

    # 4. 미완성 완성본 최종 판정
    for k in range(total):
        if not done_flags[k]:
            last = local_hists[k][-1]["inference"] if local_hists[k] else ""
            if extract_boxed(last) is not None:
                outcomes[k] = 1.0 if check_solved(
                    last, gold_list[k], problem=prob_list[k]
                ) else 0.0

    # 5. 상태별 best 후보 선택 (정답 도달 후보 중 첫 번째; 없으면 index 0)
    results = []
    for si in range(len(states_todo)):
        state_outs = outcomes[si * rollout_n : (si + 1) * rollout_n]
        state_rols = rollout_texts[si * rollout_n : (si + 1) * rollout_n]
        best_j = next((j for j, o in enumerate(state_outs) if o == 1.0), 0)
        results.append({
            "inf_text": state_rols[best_j],
            "outcomes": list(state_outs),
            "best_j":   best_j,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        items = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    if p.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(p).to_dict("records")
    raise ValueError(f"지원하지 않는 파일 형식: {p.suffix}")


# ─────────────────────────────────────────────────────────────────────────────
# 배치 평가 루프
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_batch(
    problems: list[dict],
    global_indices: list[int],
    inf_llm,
    cls_llm,
    tokenizer,
    max_steps: int,
    max_rethinks: int,
    rollout_n: int,
    rethink_temp: float,
    inf_max_new: int,
    cls_max_new: int,
) -> list[dict]:
    """문제 배치를 추론 → CLS 루프로 풀고 결과 반환.

    rollout_n > 0 이면: CLS가 rethink를 결정할 때 rollout_n개 후보를 생성·완성해
    정답 도달율이 가장 높은 스텝을 선택한 뒤 CLS가 재평가한다.
    완성 스텝 한도는 각 상태의 남은 예산(max_steps - total_steps)을 사용한다.

    반환 상태의 steps 필드:
      [{"step_idx", "inference", "cls_output", "action", "is_fail",
        "fail_rubrics", "rollout_outcomes"(optional)}, ...]
    """
    states = [
        {
            "id":            _get_id(p, idx),
            "problem":       p.get("problem", ""),
            "gold_answer":   str(p.get("gold_answer") or p.get("answer", "")),
            "history":       [],
            "steps":         [],
            "is_rethink":    False,
            "rethink_count": 0,
            "total_steps":   0,
            "done":          False,
            "is_correct":    False,
            "pred_answer":   None,
            "actions":       [],
            "_pending":      None,   # rollout으로 미리 선택된 inf
        }
        for p, idx in zip(problems, global_indices)
    ]

    for _round in range(max_steps):
        active = [i for i, s in enumerate(states) if not s["done"]]
        if not active:
            break

        # ── 1a. Rollout 선택: rethink 상태 + rollout_n > 0 ──────────────
        rollout_needed = [
            i for i in active
            if states[i]["is_rethink"] and rollout_n > 0 and states[i]["_pending"] is None
        ]
        if rollout_needed:
            selections = _batch_rollout_select(
                [states[i] for i in rollout_needed],
                inf_llm, tokenizer,
                rollout_n, rethink_temp, max_steps, inf_max_new,
            )
            for i, sel in zip(rollout_needed, selections):
                states[i]["_pending"] = sel

        # ── 1b. 추론 모델: pending 없는 상태만 생성 ─────────────────────
        gen_active = [i for i in active if states[i]["_pending"] is None]
        if gen_active:
            inf_prompts = []
            for i in gen_active:
                s      = states[i]
                system = SYSTEM_RETHINK if s["is_rethink"] else SYSTEM_INF
                sys_m, usr_m = build_messages_inference(
                    s["problem"], s["history"], len(s["history"]), system
                )
                inf_prompts.append(build_chat_prompt(tokenizer, sys_m, usr_m))
            gen_texts = _vllm_generate(inf_llm, inf_prompts, inf_max_new)
        else:
            gen_texts = []

        # ── 1c. inf_texts 조합 (pending 우선) ───────────────────────────
        gen_iter  = iter(gen_texts)
        inf_map   = {}   # state_idx → inf_text
        for i in active:
            s = states[i]
            if s["_pending"] is not None:
                inf_map[i] = s["_pending"]["inf_text"]
            else:
                inf_map[i] = next(gen_iter)
        inf_texts = [inf_map[i] for i in active]

        # ── 2. CLS 모델: 스텝 평가 ──────────────────────────────────────
        cls_prompts = []
        for i, inf_text in zip(active, inf_texts):
            s = states[i]
            temp_steps = s["history"] + [{"inference": inf_text, "is_fail": False}]
            sys_c, usr_c = build_messages_classification(
                s["problem"], temp_steps, len(temp_steps) - 1, SYSTEM_CLS
            )
            cls_prompts.append(build_chat_prompt(tokenizer, sys_c, usr_c))

        cls_texts = _vllm_generate(cls_llm, cls_prompts, cls_max_new)

        # ── 3. 결과 처리 ─────────────────────────────────────────────────
        for i, inf_text, cls_text in zip(active, inf_texts, cls_texts):
            s       = states[i]
            pending = s.pop("_pending")   # None이거나 rollout 결과

            fail_rubrics, action = parse_action(cls_text, inf_text)
            action_str = action.strip("<|>")
            s["actions"].append(action_str)
            s["total_steps"] += 1

            is_fail    = (action == TOKEN_RETHINK)
            step_rec   = {
                "step_idx":     len(s["steps"]),
                "inference":    inf_text,
                "cls_output":   cls_text,
                "action":       action_str,
                "is_fail":      is_fail,
                "fail_rubrics": fail_rubrics,
            }
            if pending is not None:
                step_rec["rollout_outcomes"] = pending["outcomes"]
                step_rec["rollout_best_j"]   = pending["best_j"]
            s["steps"].append(step_rec)
            s["history"].append({"inference": inf_text, "is_fail": is_fail})

            if action == TOKEN_RETHINK:
                s["is_rethink"]    = True
                s["rethink_count"] += 1
                s["_pending"]      = None   # 다음 라운드에서 rollout 재실행
                if s["rethink_count"] >= max_rethinks:
                    pred = extract_boxed(inf_text)
                    s["pred_answer"] = pred
                    s["is_correct"]  = check_solved(
                        inf_text, s["gold_answer"], problem=s["problem"]
                    )
                    s["done"] = True

            elif action == TOKEN_END:
                pred = extract_boxed(inf_text)
                s["pred_answer"]   = pred
                s["is_correct"]    = check_solved(
                    inf_text, s["gold_answer"], problem=s["problem"]
                )
                s["is_rethink"]    = False
                s["rethink_count"] = 0
                s["_pending"]      = None
                s["done"]          = True

            else:  # TOKEN_SOLVE
                s["is_rethink"]    = False
                s["rethink_count"] = 0
                s["_pending"]      = None
                if extract_boxed(inf_text) is not None:
                    pred = extract_boxed(inf_text)
                    s["pred_answer"] = pred
                    s["is_correct"]  = check_solved(
                        inf_text, s["gold_answer"], problem=s["problem"]
                    )
                    s["done"] = True

    # 최대 스텝 초과: 마지막 성공 스텝으로 정답 판정
    for s in states:
        if not s["done"]:
            last_inf = next(
                (st["inference"] for st in reversed(s["history"]) if not st.get("is_fail")),
                (s["history"][-1]["inference"] if s["history"] else ""),
            )
            pred = extract_boxed(last_inf) if last_inf else None
            s["pred_answer"] = pred
            s["is_correct"]  = (
                check_solved(last_inf, s["gold_answer"], problem=s["problem"])
                if last_inf else False
            )
            s["done"] = True

    return states


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default=None,
                   help="GPU 번호 쌍 (예: 4,5). 첫 번째: 추론, 두 번째: CLS")
    p.add_argument("--dataset", default=None,
                   help="평가 데이터셋 경로 (config 기본값 override)")
    p.add_argument("--max_steps",    type=int, default=MAX_STEPS)
    p.add_argument("--max_rethinks", type=int, default=3,
                   help="스텝당 최대 rethink 횟수")
    p.add_argument("--batch_size",   type=int, default=BATCH_SIZE)
    p.add_argument("--inf_max_new",  type=int, default=INF_MAX_NEW)
    p.add_argument("--cls_max_new",  type=int, default=CLS_MAX_NEW)
    p.add_argument("--rollout_n",   type=int,   default=ROLLOUT_N,
                   help="rethink 후보 수 (0이면 rollout 비활성화)")
    p.add_argument("--rethink_temp", type=float, default=RETHINK_TEMP,
                   help="rethink 후보 생성 temperature")
    p.add_argument("--num_start",    type=int, default=None)
    p.add_argument("--num_end",      type=int, default=None)
    p.add_argument("--resume", default=None,
                   help="이전 출력 폴더 경로 (이어서 실행)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── GPU 설정 ──────────────────────────────────────────────────────────
    if args.gpus:
        gpu_list = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        gpu_list = _INF.get("gpus", [0, 1])
    if len(gpu_list) < 2:
        raise ValueError("GPU 2개 이상 필요: --gpus inf_gpu,cls_gpu (예: --gpus 4,5)")

    inf_gpu     = gpu_list[0]
    cls_gpu     = gpu_list[1]
    all_visible = ",".join(str(g) for g in gpu_list)

    print(f"추론 모델: {INF_CKPT}  → GPU {inf_gpu}")
    print(f"CLS  모델: {CLS_CKPT}  → GPU {cls_gpu}")

    # ── 데이터셋 ──────────────────────────────────────────────────────────
    dataset_path = (
        args.dataset
        or _INF.get("data_path")
        or CONF["data_path"].get("math500")
    )
    if not dataset_path:
        raise ValueError(
            "데이터셋 경로 미지정. --dataset 또는 config.inference.data_path 설정 필요"
        )

    print(f"데이터셋: {dataset_path}")
    all_items = load_dataset(dataset_path)

    num_start = args.num_start if args.num_start is not None else (_INF.get("num_start") or 0)
    num_end   = args.num_end   if args.num_end   is not None else _INF.get("num_end")
    if num_end:
        all_items = all_items[num_start:num_end]
    elif num_start:
        all_items = all_items[num_start:]

    # 전역 인덱스를 붙여 ID fallback에 사용
    indexed_items = [(num_start + i, it) for i, it in enumerate(all_items)]

    # ── 출력 경로 ─────────────────────────────────────────────────────────
    done_ids: set[str] = set()
    if args.resume:
        out_dir = Path(args.resume)
        traj_path    = out_dir / "traj_all.jsonl"
        results_path = out_dir / "results.jsonl"
        # resume: traj_all 기준으로 완료된 ID 수집
        if traj_path.exists():
            with open(traj_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        done_ids.add(str(json.loads(line).get("problem_id", "")))
        print(f"이전 결과 {len(done_ids)}개 발견, 이어서 실행합니다.")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _ROOT / "output" / "eval_inference" / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_path    = out_dir / "traj_all.jsonl"
        results_path = out_dir / "results.jsonl"

    # done_ids 제외
    items_todo = [(idx, it) for idx, it in indexed_items
                  if _get_id(it, idx) not in done_ids]
    print(f"문제 수: {len(all_items)}  (남은: {len(items_todo)})")
    print(f"결과 저장: {out_dir}/")

    # ── meta.json 저장 ────────────────────────────────────────────────────
    meta_path = out_dir / "meta.json"
    meta = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "dataset":      str(Path(dataset_path).resolve()),
        "n_total":      len(all_items),
        "inf_model":    INF_CKPT,
        "cls_model":    CLS_CKPT,
        "inf_gpu":      inf_gpu,
        "cls_gpu":      cls_gpu,
        "max_steps":    args.max_steps,
        "max_rethinks": args.max_rethinks,
        "batch_size":   args.batch_size,
        "inf_max_new":           args.inf_max_new,
        "cls_max_new":           args.cls_max_new,
        "rollout_n":    args.rollout_n,
        "rethink_temp": args.rethink_temp,
        "num_start":    num_start,
        "num_end":      num_end,
        "resume":       args.resume,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"meta     : {meta_path}")

    # ── 모델 로드 ─────────────────────────────────────────────────────────
    from vllm import LLM

    tokenizer = setup_tokenizer(INF_CKPT)

    print(f"추론 모델 로드 중 (GPU {inf_gpu})...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(inf_gpu)
    try:
        inf_llm = LLM(
            model=INF_CKPT,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,
            trust_remote_code=True,
            max_model_len=VLLM_MAX_LEN,
        )
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = all_visible
    print("추론 모델 로드 완료.")

    print(f"CLS 모델 로드 중 (GPU {cls_gpu})...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cls_gpu)
    try:
        cls_tokenizer = setup_tokenizer(CLS_CKPT)
        tmp_tok_dir = tempfile.mkdtemp(prefix="sc_eval_cls_tok_")
        cls_tokenizer.save_pretrained(tmp_tok_dir)
        cls_llm = LLM(
            model=CLS_CKPT,
            tokenizer=tmp_tok_dir,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,
            trust_remote_code=True,
            max_model_len=VLLM_MAX_LEN,
        )
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = all_visible
    print("CLS 모델 로드 완료.")

    # ── 평가 루프 ─────────────────────────────────────────────────────────
    from tqdm import tqdm

    n_correct_running = 0
    n_done_running    = len(done_ids)
    write_mode        = "a" if done_ids else "w"

    traj_f    = open(traj_path,    write_mode, encoding="utf-8")
    results_f = open(results_path, write_mode, encoding="utf-8")

    try:
        with tqdm(total=len(items_todo), desc="평가") as pbar:
            for batch_start in range(0, len(items_todo), args.batch_size):
                batch_indexed = items_todo[batch_start : batch_start + args.batch_size]
                indices = [idx for idx, _ in batch_indexed]
                problems = [it for _, it in batch_indexed]

                results = evaluate_batch(
                    problems, indices,
                    inf_llm, cls_llm, tokenizer,
                    max_steps=args.max_steps,
                    max_rethinks=args.max_rethinks,
                    rollout_n=args.rollout_n,
                    rethink_temp=args.rethink_temp,
                    inf_max_new=args.inf_max_new,
                    cls_max_new=args.cls_max_new,
                )

                for r in results:
                    # traj_all: 문제당 trajectory 전체
                    traj_row = {
                        "problem_id":  r["id"],
                        "problem":     r["problem"],
                        "gold_answer": r["gold_answer"],
                        "pred_answer": r["pred_answer"],
                        "is_correct":  r["is_correct"],
                        "n_steps":     r["total_steps"],
                        "actions":     r["actions"],
                        "steps": [
                            {
                                "step_idx":    st["step_idx"],
                                "action":      st["action"],
                                "is_fail":     st["is_fail"],
                                "fail_rubrics": st["fail_rubrics"],
                                "inference":   st["inference"],
                                "cls_output":  st["cls_output"],
                            }
                            for st in r["steps"]
                        ],
                    }
                    traj_f.write(json.dumps(traj_row, ensure_ascii=False) + "\n")
                    traj_f.flush()

                    # results: 요약만
                    result_row = {
                        "problem_id":  r["id"],
                        "problem":     r["problem"],
                        "gold_answer": r["gold_answer"],
                        "pred_answer": r["pred_answer"],
                        "is_correct":  r["is_correct"],
                        "n_steps":     r["total_steps"],
                        "actions":     r["actions"],
                    }
                    results_f.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                    results_f.flush()

                    if r["is_correct"]:
                        n_correct_running += 1
                    n_done_running += 1

                pbar.update(len(batch_indexed))
                acc_now = n_correct_running / max(n_done_running - len(done_ids), 1)
                pbar.set_postfix(acc=f"{acc_now:.3f} ({n_correct_running}/{n_done_running - len(done_ids)})")
    finally:
        traj_f.close()
        results_f.close()

    # ── 요약 출력 ─────────────────────────────────────────────────────────
    n_cur     = n_done_running - len(done_ids)
    acc       = n_correct_running / max(n_cur, 1)

    # 액션 분포: results 파일에서 집계
    action_counts: dict[str, int] = {}
    avg_steps = 0.0
    with open(results_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for row in rows:
        avg_steps += row.get("n_steps", 0)
        for a in row.get("actions", []):
            action_counts[a] = action_counts.get(a, 0) + 1
    avg_steps /= max(len(rows), 1)

    W = 60
    print(f"\n{'='*W}")
    print(f"  데이터셋  : {Path(dataset_path).name}")
    print(f"  문제 수   : {n_cur}")
    print(f"{'─'*W}")
    print(f"  정확도    : {acc:.4f}  ({n_correct_running}/{n_cur})")
    print(f"  평균 스텝 : {avg_steps:.2f}")
    print(f"  액션 분포 : {action_counts}")
    print(f"{'='*W}")
    print(f"  traj_all : {traj_path}")
    print(f"  results  : {results_path}")


if __name__ == "__main__":
    main()
