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
_prompt_cfg   = CONF.get("prompts", {})
_prm_cfg      = CONF.get("PRM", {})
_prompts_file = _ROOT / _prompt_cfg.get("file", "prompts/action_prompts.json")
_rubric_file  = _ROOT / _prm_cfg.get("rubric", _prompt_cfg.get("rubric_file", "prompts/prm_rubric_v6.2.jsonl"))
_solve_key    = _prompt_cfg.get("solve", "gen_solve_R")
_rethink_key  = _prompt_cfg.get("rethink", "gen_rethink_R")


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
    end_ratio: float = 0.13,
) -> str:
    """
    trajectory를 스텝 단위로 분류해 균형 잡힌 SFT 데이터를 생성합니다.

    균형화 전략 (우선순위 순):
      1. action 균형: solve = rethink (동수), end = SC 전체의 end_ratio (~13%, 자연 빈도)
      2. gen-only 50%: SC 총합이 전체의 50%가 되도록 solve·rethink 수 결정
      3. fail rubrics 균형: rethink SC를 1~10개 그룹에 그리디 균등 배분
    """
    import random
    from collections import defaultdict

    raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    print(f"[preprocess] {len(raw)}개 trajectory 로드: {data_path}")

    system_solve, system_rethink = get_system_prompts()
    rng = random.Random(seed)

    # ── 스텝을 action × fail rubric 개수별로 분류 ───────────────────────────
    end_steps:       list = []
    solve_steps:     list = []
    rethink_by_fail: dict[int, list] = defaultdict(list)

    for item in raw:
        problem = item["problem"]
        steps   = item["steps"]
        for k, step in enumerate(steps):
            action = step.get("next_gold_action") or TOKEN_SOLVE
            n_fail = len(step.get("gold_fail_rubrics") or [])
            entry  = (problem, steps, k)
            if TOKEN_END in action:
                end_steps.append(entry)
            elif TOKEN_SOLVE in action:
                solve_steps.append(entry)
            else:
                rethink_by_fail[n_fail].append(entry)

    for lst in [end_steps, solve_steps] + list(rethink_by_fail.values()):
        rng.shuffle(lst)

    # ── SC 수 결정 ───────────────────────────────────────────────────────────
    total_steps = len(end_steps) + len(solve_steps) + sum(len(v) for v in rethink_by_fail.values())
    sc_total    = total_steps // 2                      # gen-only 50%
    sc_end      = int(sc_total * end_ratio)             # end: 자연 빈도 (~13%)
    sc_solve    = (sc_total - sc_end) // 2              # solve = rethink
    sc_rethink_budget = sc_total - sc_end - sc_solve

    # ── rethink: fail rubric 그룹별 그리디 균등 배분 ─────────────────────────
    sorted_fail_keys = sorted(rethink_by_fail.keys(), key=lambda n: len(rethink_by_fail[n]))
    quota: dict[int, int] = {}
    remaining = sc_rethink_budget
    for i, n in enumerate(sorted_fail_keys):
        share    = remaining // (len(sorted_fail_keys) - i)
        take     = min(len(rethink_by_fail[n]), share)
        quota[n] = take
        remaining -= take

    sc_end_lst     = end_steps[:sc_end]
    gen_end_lst    = end_steps[sc_end:]
    sc_solve_lst   = solve_steps[:sc_solve]
    gen_solve_lst  = solve_steps[sc_solve:]
    sc_rethink_lst:  list = []
    gen_rethink_lst: list = []
    for n in rethink_by_fail:
        take = quota.get(n, 0)
        sc_rethink_lst.extend(rethink_by_fail[n][:take])
        gen_rethink_lst.extend(rethink_by_fail[n][take:])

    sc_steps      = sc_end_lst + sc_solve_lst + sc_rethink_lst
    genonly_steps = gen_end_lst + gen_solve_lst + gen_rethink_lst
    sc_rethink_n  = sum(quota.values())

    # ── 분포 출력 ────────────────────────────────────────────────────────────
    n_sc  = len(sc_steps)
    n_gen = len(genonly_steps)
    print(f"\n  SC {n_sc} ({n_sc/total_steps:.1%}) / gen-only {n_gen} ({n_gen/total_steps:.1%})")
    print(f"\n  [ Next action ]")
    print(f"  {'액션':<10} {'전체':>6}  {'SC':>5}  {'SC내':>6}  {'gen-only':>8}")
    print(f"  {'─'*42}")
    for label, total_n, sc_n in [
        ("solve",   len(solve_steps),                        sc_solve),
        ("rethink", sum(len(v) for v in rethink_by_fail.values()), sc_rethink_n),
        ("end",     len(end_steps),                          sc_end),
    ]:
        print(f"  {label:<10} {total_n:6d}  {sc_n:5d}  {sc_n/n_sc:5.1%}  {total_n-sc_n:8d}")

    print(f"\n  [ Fail rubrics (SC 내) ]")
    print(f"  {'개수':>4}  {'SC':>5}  {'비율':>6}  {'gen-only':>8}")
    print(f"  {'─'*34}")
    print(f"  {'0개':>4}  {sc_end+sc_solve:5d}  {(sc_end+sc_solve)/n_sc:5.1%}  {len(gen_end_lst)+len(gen_solve_lst):8d}  ← solve+end")
    for n in sorted(quota):
        c   = quota[n]
        gen = len(rethink_by_fail[n]) - c
        print(f"  {n:3d}개  {c:5d}  {c/n_sc:5.1%}  {gen:8d}")
    print(f"  {'─'*34}")
    print(f"  0개 : 1~10개 = {sc_end+sc_solve} ({(sc_end+sc_solve)/n_sc:.1%}) : {sc_rethink_n} ({sc_rethink_n/n_sc:.1%})")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_sc = total_gen = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for problem, steps, k in tqdm(sc_steps, desc="  SC 전처리"):
            record = build_sft_sample(problem, steps, k, system_solve, system_rethink)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_sc += 1

        for problem, steps, k in tqdm(genonly_steps, desc="  Gen-only 전처리"):
            step = steps[k]
            record = build_sft_sample_gen_only(problem, step, system_solve)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_gen += 1

    print(f"[preprocess] 완료: SC {total_sc}개 + gen-only {total_gen}개 = {total_sc+total_gen}개  →  {out_path}")
    return str(out_path)


def main():
    p = argparse.ArgumentParser(
        description="trajectory JSONL → SFT 전처리 JSONL 변환\n"
                    "데이터를 무작위로 절반은 SC, 절반은 gen-only 포맷으로 변환합니다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--data_path",   required=False, default=None,
                   help="trajectory JSONL 경로\n(절반 SC / 절반 gen-only 로 무작위 분할)")
    p.add_argument("--output_path", default=None,
                   help="출력 전처리 JSONL 경로")
    p.add_argument("--seed",            type=int,   default=42,
                   help="랜덤 시드 (기본: 42)")
    p.add_argument("--end_ratio",       type=float, default=0.13,
                   help="SC 내 end action 비율 (기본: 0.13 = 자연 빈도)")
    p.add_argument("--debug", action="store_true",
                   help="--output_path의 저장된 전처리 파일에서 샘플 1개 출력")
    args = p.parse_args()

    if args.debug:
        if not args.output_path:
            p.error("--debug 사용 시 --output_path (전처리 완료된 JSONL)를 지정하세요.")
        debug(args.output_path, idx=DEBUG_SAMPLE_IDX)
        return

    if not args.output_path:
        p.error("--output_path 를 지정하세요.")
    if not args.data_path:
        p.error("--data_path 를 지정하세요.")

    preprocess(args.output_path, data_path=args.data_path, seed=args.seed, end_ratio=args.end_ratio)


if __name__ == "__main__":
    main()
