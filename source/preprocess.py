"""
SFT 전처리 스크립트

generate_trajectory.py 출력 JSONL을 읽어 학습용 전처리 데이터를 생성합니다.
각 스텝마다 하나의 샘플:
  input  : [{"role": "system", ...}, {"role": "user", ...}]
  target : 전체 타겟 텍스트 (inference + critics + action)
  is_error: bool
  state  : str  ("gen_solve" | "gen_rethink" | "pat_solve")

실행 예시:
  python source/preprocess.py \
      --data_path output/sft_trajectory/xxx/traj_mix.jsonl \
      --output_path output/sft_preprocessed/xxx.jsonl

디버그 (샘플 1개 출력):
  python source/preprocess.py --data_path output/sft_trajectory/xxx/traj_mix.jsonl --debug
"""

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_sft import build_messages, build_target, SPECIAL_TOKENS, ACTION_TOKENS, CONF, TOKEN_SOLVE, TOKEN_END

_ROOT = Path(__file__).resolve().parent.parent

DEBUG_SAMPLE_IDX = 15  # --debug 시 출력할 샘플 번호 (0-based)

# ─── config에서 프롬프트 설정 로드 ────────────────────────────────────────────
_prompt_cfg        = CONF.get("prompts", {})
_prm_cfg           = CONF.get("PRM", {})
_prompts_file      = _ROOT / _prompt_cfg.get("file", "prompts/action_prompts.json")
_rubric_file       = _ROOT / _prm_cfg.get("rubric", _prompt_cfg.get("rubric_file", "prompts/prm_rubric_v6.2.jsonl"))
_grpo_rubric_file  = _ROOT / _prm_cfg.get("rubric", "prompts/prm_rubric_v6.4.jsonl")
_solve_key         = _prompt_cfg.get("solve", "gen_solve_R")
_rethink_key       = _prompt_cfg.get("rethink", "gen_rethink_R")


@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    with open(_prompts_file, encoding="utf-8") as f:
        return {d["name"]: d["content"] for d in json.load(f)}


@lru_cache(maxsize=1)
def _load_rubric_str() -> str:
    with open(_rubric_file, encoding="utf-8") as f:
        rubrics = [json.loads(l) for l in f if l.strip()]
    return "\n".join(f"{i}. {r['name']}: [correct/incorrect — {r['criterion']}]"
                     for i, r in enumerate(rubrics, 1))


@lru_cache(maxsize=1)
def _load_grpo_rubric_str() -> str:
    with open(_grpo_rubric_file, encoding="utf-8") as f:
        rubrics = [json.loads(l) for l in f if l.strip()]
    return "\n".join(f'{r["name"]}: [{r["criterion"]}]' for r in rubrics)


def get_system_prompts() -> tuple[str, str]:
    prompts    = _load_prompts()
    rubric_str = _load_rubric_str()
    system_solve   = prompts[_solve_key].replace("{{rubric}}", rubric_str)
    system_rethink = prompts[_rethink_key].replace("{{rubric}}", rubric_str)
    return system_solve, system_rethink


# rubric name → special token 매핑
_RUBRIC_TOKENS: dict[str, str] = {
    t[2:-2].replace("_", " ").title(): t
    for t in SPECIAL_TOKENS
    if t not in ACTION_TOKENS
}
_RUBRIC_NAME_OVERRIDES = {
    "Abstract And Linear Algebra Operations": "Abstract and Linear Algebra Operations",
    "Function And Limit Analysis":            "Function and Limit Analysis",
    "Counting And Probability":               "Counting and Probability",
    "Number Theoretic Reasoning":             "Number Theoretic Reasoning",
    "Logical And Discrete Reasoning":         "Logical and Discrete Reasoning",
    "Progress And Non-Repetition":            "Progress and Non-Repetition",
}
RUBRIC_TOKENS: dict[str, str] = {
    _RUBRIC_NAME_OVERRIDES.get(k, k): v
    for k, v in _RUBRIC_TOKENS.items()
}


def build_sft_sample(
    problem: str,
    steps: list[dict],
    k: int,
    system_solve: str,
    system_rethink: str,
) -> dict:
    """
    trajectory의 k번째 스텝으로부터 SFT 학습 샘플 하나를 생성합니다.

    반환값:
      {
        "input":    [{"role": "system", ...}, {"role": "user", ...}],
        "target":   str,   # inference + critics + next_action
        "inference": str,
        "is_error": bool,
        "state":    str,   # "gen_solve" | "gen_rethink" | "pat_solve"
      }

    데이터 필드 (steps[k]):
      inference           : str   — 모델이 생성한 순수 수학 풀이
      state               : str   — gen_solve | gen_rethink | pat_solve
      is_error            : bool
      gold_fail_rubrics   : list[str]
      next_gold_action    : str   — "<|solve|>" | "<|rethink|>" | "<|end|>"
      prm_fast_critique   : dict  — {rubric: {verdict, critique}}
      prm_deep_critique   : list  — [{rubric, verdict, critique}]
      prm_critique_summary: str
      does                : str   — 히스토리용 한 줄 요약
    """
    step = steps[k]
    system_str, user_str = build_messages(problem, steps, k, system_solve, system_rethink)
    target = build_target(step, RUBRIC_TOKENS)

    return {
        "input": [
            {"role": "system", "content": system_str},
            {"role": "user",   "content": user_str},
        ],
        "target":    target,
        "inference": step.get("inference", ""),
        "is_error":  step.get("is_error", False),
        "state":     step.get("state", "gen_solve"),
    }


def build_sft_sample_gen_only(
    problem: str,
    step: dict,
    system_solve: str,
) -> dict:
    """
    Gen-only path: is_error=False인 스텝 하나를 히스토리 없이 gen_solve 포맷으로 변환.

    self-correction trajectory에서 실제로 loss를 받는 성공 스텝만 추출해
    단순 gen_solve 샘플처럼 만들어 준다.
    (generator 실패 → rethink → patcher 성공 이면 patcher 스텝만 사용,
     generator 실패 → rethink 성공 이면 rethink 스텝만 사용)
    """
    user_str = f"[Problem]\n{problem}\n\nWrite Step 1."
    target   = build_target(step, RUBRIC_TOKENS)

    return {
        "input": [
            {"role": "system", "content": system_solve},
            {"role": "user",   "content": user_str},
        ],
        "target":    target,
        "inference": step.get("inference", ""),
        "is_error":  False,
        "state":     "gen_solve",
    }


def build_grpo_system_prompt() -> str:
    prompts    = _load_prompts()
    rubric_str = _load_grpo_rubric_str()
    return prompts[_solve_key].replace("{{rubric}}", rubric_str)


def prepare_grpo_data(
    input_path: str,
    output_path: str,
    num_start: int | None = None,
    num_end: int | None = None,
) -> str:
    """rl_data (JSONL 또는 parquet) → verl GRPO parquet 변환.

    입력 포맷 자동 감지:
      - .jsonl  : 줄 단위 JSON, problem_id / problem / gold_answer 필드
      - .parquet: 이미 prompt 컬럼이 있으면 그대로 반환, 없으면 변환

    num_start / num_end: 레코드 인덱스 슬라이스 [num_start, num_end)
    """
    import pandas as pd

    inp = Path(input_path)

    # parquet이고 이미 verl 포맷(prompt 컬럼)이면 변환 불필요
    if inp.suffix == ".parquet" and num_start is None and num_end is None:
        df_in = pd.read_parquet(input_path)
        if "prompt" in df_in.columns:
            print(f"[prepare_grpo_data] 이미 verl 포맷 — 변환 생략: {input_path}")
            return input_path
        records = df_in.to_dict("records")
    elif inp.suffix == ".parquet":
        df_in = pd.read_parquet(input_path)
        if "prompt" in df_in.columns:
            records = None  # 이미 verl 포맷이지만 슬라이싱 필요
        else:
            records = df_in.to_dict("records")
    else:
        with open(input_path, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]

    # 슬라이싱 (이미 verl 포맷 parquet인 경우)
    if inp.suffix == ".parquet" and records is None:
        df_in = pd.read_parquet(input_path)
        df_sliced = df_in.iloc[num_start:num_end]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_sliced.to_parquet(output_path, index=False)
        print(f"[prepare_grpo_data] 슬라이스 저장: [{num_start}:{num_end}] → {output_path}  ({len(df_sliced)}개)")
        return str(out)

    # 인덱스 슬라이싱
    if num_start is not None or num_end is not None:
        records = records[num_start:num_end]
        print(f"[prepare_grpo_data] 슬라이스 적용: [{num_start}:{num_end}] → {len(records)}개 레코드")

    system_prompt = build_grpo_system_prompt()
    seen_ids: set = set()
    rows = []
    for d in records:
        pid = d.get("problem_id") or d.get("id", "")
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        gold = d.get("gold_answer") or d.get("answer", "")
        rows.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"[Problem]\n{d['problem']}\n\nWrite Step 1."},
            ],
            "data_source":  "sc-grpo",
            "reward_model": {"ground_truth": gold, "style": "rule"},
            "extra_info":   {"problem_id": pid, "gold_answer": gold},
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"[prepare_grpo_data] 저장 완료: {output_path}  ({len(rows)}개 문제)")
    return str(out)


def debug(preprocessed_path: str, idx: int = 1):
    """저장된 전처리 JSONL에서 idx번째(0-based) 샘플을 읽어 TARGET을 출력한다."""
    sep = "─" * 72
    with open(preprocessed_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx and line.strip():
                sample = json.loads(line)
                break
        else:
            print(f"샘플 {idx}번을 찾을 수 없습니다.")
            return
    print(f"\n{'='*72}")
    print(f"[state={sample.get('state')}  is_error={sample.get('is_error')}]")
    print(f"\n[TARGET]\n{sep}")
    print(sample["target"])


def _write_sc_samples(items: list, out_file, system_solve: str, system_rethink: str) -> int:
    """Self-correction path: 모든 스텝을 그대로 전처리해 out_file에 씁니다."""
    total = 0
    for item in tqdm(items, desc="  SC 전처리"):
        problem = item["problem"]
        steps   = item["steps"]
        for k in range(len(steps)):
            record = build_sft_sample(problem, steps, k, system_solve, system_rethink)
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
    return total


def _write_gen_only_samples(items: list, out_file, system_solve: str) -> int:
    """Gen-only path: is_error=False인 스텝만 뽑아 gen_solve 포맷으로 변환 후 out_file에 씁니다."""
    total = 0
    for item in tqdm(items, desc="  Gen-only 전처리"):
        problem = item["problem"]
        for step in item["steps"]:
            if not step.get("is_error", True):   # 성공한 스텝만
                record = build_sft_sample_gen_only(problem, step, system_solve)
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    return total


def preprocess(
    output_path: str,
    data_path: str,
    seed: int = 42,
    solve_ratio: int = 43,
    rethink_ratio: int = 43,
    end_ratio: int = 14,
) -> str:
    """
    trajectory를 스텝 단위로 분류해 균형 잡힌 SFT 데이터를 생성합니다.

    균형화 전략:
      - is_right=True 필터
      - rethink 전부 사용 (gen + patcher)
      - solve, end는 rethink 수 기준으로 solve_ratio:rethink_ratio:end_ratio 비율에 맞게 샘플링
      - 각 버킷에서 generator 우선, 부족하면 patcher 보충
    """
    import random

    raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    print(f"[preprocess] {len(raw)}개 trajectory 로드: {data_path}")

    system_solve, system_rethink = get_system_prompts()
    rng = random.Random(seed)

    # ── (source × state) 별로 분류 — is_right=False trajectory 제외 ──────────
    gen_solve:   list = []
    gen_rethink: list = []
    gen_end:     list = []
    pat_solve:   list = []
    pat_rethink: list = []
    pat_end:     list = []

    def _norm_state(step: dict) -> str:
        """구 형식(gen_solve/gen_rethink/pat_solve)과 신 형식(solve/rethink/end) 통일."""
        state  = step.get("state", "solve")
        action = step.get("next_gold_action", "")
        if state == "end":
            return "end"
        if state in ("gen_rethink", "rethink"):
            return "rethink"
        if state == "pat_solve":
            return "end" if TOKEN_END in action else "rethink"
        return "end" if TOKEN_END in action else "solve"

    n_traj_skipped = 0
    n_total_steps  = 0
    for item in raw:
        n_total_steps += len(item["steps"])
        if not item.get("is_right", False):
            n_traj_skipped += 1
            continue
        problem = item["problem"]
        steps   = item["steps"]
        for k, step in enumerate(steps):
            src   = step.get("source", "gen")
            state = _norm_state(step)
            entry = (problem, steps, k)
            if state == "end":
                (gen_end     if src == "gen" else pat_end).append(entry)
            elif state == "rethink":
                (gen_rethink if src == "gen" else pat_rethink).append(entry)
            else:
                (gen_solve   if src == "gen" else pat_solve).append(entry)

    n_right_traj = len(raw) - n_traj_skipped
    print(f"[preprocess] trajectory 필터: is_right {n_right_traj}개 사용 / 전체 {len(raw)}개 (제외 {n_traj_skipped}개)")
    print(f"[preprocess] 총 스텝 수: {n_total_steps}개")

    for lst in [gen_solve, gen_rethink, gen_end, pat_solve, pat_rethink, pat_end]:
        rng.shuffle(lst)

    # ── 목표 수 계산 (rethink 전부 기준) ────────────────────────────────────
    all_rethink = gen_rethink + pat_rethink
    n_rethink   = len(all_rethink)
    n_solve_tgt = round(n_rethink * solve_ratio   / rethink_ratio)
    n_end_tgt   = round(n_rethink * end_ratio     / rethink_ratio)

    # ── solve: gen 우선, 부족하면 patcher 보충 ───────────────────────────────
    solve_pool  = gen_solve[:n_solve_tgt]
    if len(solve_pool) < n_solve_tgt:
        solve_pool += pat_solve[:n_solve_tgt - len(solve_pool)]

    # ── end: gen 우선, 부족하면 patcher 보충 ─────────────────────────────────
    end_pool = gen_end[:n_end_tgt]
    if len(end_pool) < n_end_tgt:
        end_pool += pat_end[:n_end_tgt - len(end_pool)]

    # ── 통계 출력 ────────────────────────────────────────────────────────────
    def _count_src(pool):
        ng = sum(1 for _, steps, k in pool if steps[k].get("source", "gen") == "gen")
        return ng, len(pool) - ng

    sg, sp = _count_src(solve_pool)
    rg, rp = _count_src(all_rethink)
    eg, ep = _count_src(end_pool)
    total  = len(solve_pool) + n_rethink + len(end_pool)

    print(f"\n  [ 스텝 비율 (목표 {solve_ratio}:{rethink_ratio}:{end_ratio}) ]")
    print(f"  {'액션':<10} {'수':>6}  {'비율':>6}    {'gen':>6} {'gen%':>6}    {'patcher':>7} {'pat%':>6}")
    print(f"  {'─'*62}")
    for label, n, ng, np in [
        ("solve",   len(solve_pool), sg, sp),
        ("rethink", n_rethink,       rg, rp),
        ("end",     len(end_pool),   eg, ep),
    ]:
        print(f"  {label:<10} {n:6d}  {n/max(total,1):5.1%}    {ng:6d} {ng/max(n,1):5.1%}    {np:7d} {np/max(n,1):5.1%}")
    print(f"  {'─'*62}")
    print(f"  {'합계':<10} {total:6d}")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples = solve_pool + all_rethink + end_pool
    rng.shuffle(all_samples)

    with open(out_path, "w", encoding="utf-8") as f:
        for problem, steps, k in tqdm(all_samples, desc="  전처리"):
            record = build_sft_sample(problem, steps, k, system_solve, system_rethink)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[preprocess] 완료: {len(all_samples)}개  →  {out_path}")
    return str(out_path)


def main():
    p = argparse.ArgumentParser(
        description="전처리 스크립트 (SFT / GRPO)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--data_path",    required=False, default=None,
                   help="trajectory JSONL 경로 (SFT 모드)")
    p.add_argument("--output_path",  default=None,
                   help="출력 경로 (SFT 모드)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--solve_ratio",  type=int, default=43)
    p.add_argument("--rethink_ratio", type=int, default=43)
    p.add_argument("--end_ratio",    type=int, default=14)
    p.add_argument("--debug", action="store_true",
                   help="저장된 전처리 파일에서 샘플 1개 출력")
    p.add_argument("--grpo", action="store_true",
                   help="GRPO 데이터 준비 모드: rl_data JSONL → verl parquet")
    p.add_argument("--grpo_input",  default=None,
                   help="GRPO 입력 JSONL 경로 (기본: config의 data_path.rl_data)")
    p.add_argument("--grpo_output", default=str(_ROOT / "datasets" / "grpo_train.parquet"),
                   help="GRPO 출력 parquet 경로")
    p.add_argument("--num_start", type=int, default=None,
                   help="RL 데이터 슬라이스 시작 인덱스 (포함)")
    p.add_argument("--num_end",   type=int, default=None,
                   help="RL 데이터 슬라이스 끝 인덱스 (미포함)")
    args = p.parse_args()

    if args.grpo:
        grpo_input = args.grpo_input or CONF["data_path"]["rl_data"]
        result_path = prepare_grpo_data(grpo_input, args.grpo_output, args.num_start, args.num_end)
        print(f"TRAIN_FILE={result_path}")
        return

    if args.debug:
        if not args.output_path:
            p.error("--debug 사용 시 --output_path (전처리 완료된 JSONL)를 지정하세요.")
        debug(args.output_path, idx=DEBUG_SAMPLE_IDX)
        return

    if not args.output_path:
        p.error("--output_path 를 지정하세요.")
    if not args.data_path:
        p.error("--data_path 를 지정하세요.")

    preprocess(
        args.output_path,
        data_path=args.data_path,
        seed=args.seed,
        solve_ratio=args.solve_ratio,
        rethink_ratio=args.rethink_ratio,
        end_ratio=args.end_ratio,
    )


if __name__ == "__main__":
    main()
