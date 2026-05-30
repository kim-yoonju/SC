"""
SFT 전처리 스크립트

generate_trajectory.py 출력 JSONL을 읽어 학습용 전처리 데이터를 생성합니다.
각 스텝마다 하나의 샘플:
  input  : [{"role": "system", ...}, {"role": "user", ...}]
  target : 전체 타겟 텍스트 (inference + critics + action)
  is_fail: bool
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from utils_sft import (build_messages, build_target,
                        build_messages_inference, build_messages_classification,
                        build_target_inference, build_target_classification,
                        _get_incorrect_rubrics,
                        SPECIAL_TOKENS, ACTION_TOKENS, CONF, TOKEN_SOLVE, TOKEN_END, TOKEN_RETHINK, TOKEN_NONE)

_ROOT = Path(__file__).resolve().parent.parent

DEBUG_SAMPLE_IDX = 15  # --debug 시 출력할 샘플 번호 (0-based)

# ─── config에서 프롬프트 설정 로드 ────────────────────────────────────────────
_prompt_cfg        = CONF.get("prompts", {})
_prm_cfg           = CONF.get("PRM", {})
_rubric_file       = _ROOT / _prm_cfg.get("deep_rubric", _prompt_cfg.get("rubric_file", "prompts/deep_rubric_v6.2.json"))
_grpo_rubric_file  = _ROOT / _prm_cfg.get("deep_rubric", "prompts/deep_rubric_v6.4.json")
_solve_key         = _prompt_cfg.get("solve", "gen_solve_R")
_rethink_key       = _prompt_cfg.get("rethink", "gen_rethink_R")

_PROMPT_FILES = ["prompts/generator.json", "prompts/prm.json", "prompts/patcher.json"]


@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    result = {}
    for rel in _PROMPT_FILES:
        with open(_ROOT / rel, encoding="utf-8") as f:
            result.update({d["name"]: d["content"] for d in json.load(f)})
    return result


@lru_cache(maxsize=1)
def _load_rubric_str() -> str:
    with open(_rubric_file, encoding="utf-8") as f:
        rubrics = json.load(f)
    return "\n".join(f"{i}. {r['name']}: [correct/incorrect — {r['criterion']}]"
                     for i, r in enumerate(rubrics, 1))


def _load_rubric_str_simple() -> str:
    """classification용 — 루브릭당 한 줄 (simple_criterion 사용)."""
    with open(_rubric_file, encoding="utf-8") as f:
        rubrics = json.load(f)
    return "\n".join(f"{i}. {r['name']}: {r.get('simple_criterion', r['criterion'][:120])}"
                     for i, r in enumerate(rubrics, 1))


@lru_cache(maxsize=1)
def get_inference_prompts() -> tuple[str, str]:
    """(system_gen_inference, system_pat_inference) 반환."""
    prompts = _load_prompts()
    return prompts["gen_inference"], prompts["pat_inference"]


@lru_cache(maxsize=1)
def get_rethink_inference_prompt() -> str:
    """gen_rethink_inference 프롬프트 반환."""
    return _load_prompts()["gen_rethink_inference"]


@lru_cache(maxsize=1)
def get_classification_prompt() -> str:
    """gen_classification 프롬프트에 rubric을 치환해 반환."""
    prompts = _load_prompts()
    rubric_str = _load_rubric_str_simple()
    return prompts["gen_classification"].replace("{{rubric}}", rubric_str)


@lru_cache(maxsize=1)
def _load_grpo_rubric_str() -> str:
    with open(_grpo_rubric_file, encoding="utf-8") as f:
        rubrics = json.load(f)
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


def _group_and_order_trajectories(all_samples: list, rng) -> list:
    """
    (problem, steps, k) 튜플 리스트를 trajectory 단위로 묶어서:
      - trajectory 순서는 rng로 shuffle (에폭 다양성 확보)
      - trajectory 내 스텝은 k 오름차순 정렬 (인과 순서 보장)
    반환: (traj_id, problem, steps, k) 리스트
    """
    from collections import defaultdict
    by_traj = defaultdict(list)
    for entry in all_samples:
        _, steps, _ = entry
        by_traj[id(steps)].append(entry)

    traj_list = list(by_traj.values())
    rng.shuffle(traj_list)

    ordered = []
    for traj_id, group in enumerate(traj_list):
        group.sort(key=lambda x: x[2])  # step index k 오름차순
        for entry in group:
            ordered.append((traj_id,) + entry)
    return ordered


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
        "is_fail": bool,
        "state":    str,   # "gen_solve" | "gen_rethink" | "pat_solve"
      }

    데이터 필드 (steps[k]):
      inference           : str   — 모델이 생성한 순수 수학 풀이
      state               : str   — gen_solve | gen_rethink | pat_solve
      is_fail            : bool
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
        "is_fail":  step.get("is_fail", False),
        "state":     step.get("state", "gen_solve"),
    }


def build_sft_sample_gen_only(
    problem: str,
    step: dict,
    system_solve: str,
) -> dict:
    """
    Gen-only path: is_fail=False인 스텝 하나를 히스토리 없이 gen_solve 포맷으로 변환.

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
        "is_fail":  False,
        "state":     "gen_solve",
    }


def build_sft_sample_inference(
    problem: str,
    steps: list[dict],
    k: int,
    system_gen: str,
    system_pat: str,
) -> dict:
    """inference model용 샘플 — math step + Does 요약만 target."""
    step = steps[k]
    src = step.get("source", "gen")
    system = system_pat if src == "pat" else system_gen
    system_str, user_str = build_messages_inference(problem, steps, k, system)
    target = build_target_inference(step)
    return {
        "input": [
            {"role": "system", "content": system_str},
            {"role": "user",   "content": user_str},
        ],
        "target":    target,
        "inference": step.get("inference", ""),
        "is_fail":  step.get("is_fail", False),
        "state":     "pat_inference" if src == "pat" else "gen_inference",
    }


def build_sft_sample_classification(
    problem: str,
    steps: list[dict],
    k: int,
    system: str,
    include_rubrics: bool = True,
    include_actions: bool = True,
    use_summary: bool = False,
) -> dict:
    """classification model용 샘플 — math step + Does를 input으로, Fast/Deep critic (+ Fail rubrics? + Next action?)을 target으로."""
    step = steps[k]
    system_str, user_str = build_messages_classification(problem, steps, k, system)
    target = build_target_classification(step, RUBRIC_TOKENS,
                                         include_rubrics=include_rubrics,
                                         include_actions=include_actions,
                                         use_summary=use_summary)
    return {
        "input": [
            {"role": "system", "content": system_str},
            {"role": "user",   "content": user_str},
        ],
        "target":          target,
        "inference":       step.get("inference", ""),
        "is_fail":        False,
        "state":           "gen_classification",
        "incorrect_rubrics": _get_incorrect_rubrics(step),
    }


def preprocess_mode(
    output_path: str,
    data_path: str,
    mode: str,
    seed: int = 42,
    solve_ratio: int = 43,
    rethink_ratio: int = 43,
    end_ratio: int = 14,
    no_balance: bool = False,
    no_filter: bool = False,
    include_rubrics: bool = True,
    include_actions: bool = True,
    max_length: int | None = None,
    max_target_length: int | None = None,
    use_summary: bool = False,
) -> str:
    """
    trajectory를 inference 또는 classification 모드로 전처리합니다.

    mode="inference"      : inference + Does 요약만 target (gen_inference / pat_inference 프롬프트)
    mode="classification" : Fast/Deep critic + fail rubrics + next action만 target (gen_classification 프롬프트)

    balance 로직은 preprocess()와 동일.
    no_filter=True : is_right 필터 없이 전체 trajectory의 모든 스텝 사용
    """
    import random

    raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    print(f"[preprocess_{mode}] {len(raw)}개 trajectory 로드: {data_path}")

    if mode == "inference":
        system_gen, system_pat = get_inference_prompts()
        def _build(problem, steps, k):
            return build_sft_sample_inference(problem, steps, k, system_gen, system_pat)
    elif mode == "classification":
        system_cls = get_classification_prompt()
        def _build(problem, steps, k):
            return build_sft_sample_classification(problem, steps, k, system_cls,
                                                   include_rubrics=include_rubrics,
                                                   include_actions=include_actions,
                                                   use_summary=use_summary)
    else:
        raise ValueError(f"알 수 없는 mode: {mode}. 'inference' 또는 'classification'을 사용하세요.")

    rng = random.Random(seed)

    gen_solve:   list = []
    gen_rethink: list = []
    gen_end:     list = []
    pat_solve:   list = []
    pat_rethink: list = []
    pat_end:     list = []

    def _norm_state(step: dict) -> str:
        state  = step.get("state", "solve")
        action = step.get("next_gold_action", "")
        if state == "end":
            return "end"
        if state in ("gen_rethink", "rethink"):
            return "rethink"
        if state == "pat_solve":
            return "end" if TOKEN_END in action else "rethink"
        return "end" if TOKEN_END in action else "solve"

    # classification 전용: 전체 루브릭 집합 (None 토큰 제외)
    _cls_expected_rubrics = {r for r in RUBRIC_TOKENS.keys() if r != "None"} if mode == "classification" else set()

    n_traj_skipped = 0
    n_total_steps  = 0
    n_action_skipped = 0
    n_patcher_skipped = 0              # classification: patcher step 제외 카운트
    n_rubric_incomplete = 0            # classification: 루브릭 verdict 불완전 제외
    _missing_field_counts: dict = {}   # classification 모드 필드별 누락 집계
    _fail_rubric_counts:   dict = {}   # fail rubric 빈도 집계
    for item in raw:
        n_total_steps += len(item["steps"])
        if not no_filter and not item.get("is_right", False):
            n_traj_skipped += 1
            continue
        problem = item["problem"]
        steps   = item["steps"]
        for k, step in enumerate(steps):
            if mode == "inference" and step.get("is_fail", False):
                continue
            if mode == "classification":
                # patcher 스텝: prm_deep_critique 없음 → 제외
                if step.get("source") == "patcher":
                    n_patcher_skipped += 1
                    continue
                # prm_deep_critique만 필수 (prm_fast_critique는 선택적 fallback)
                _cls_missing = [] if step.get("prm_deep_critique") is not None else ["prm_deep_critique"]
                if not step.get("next_gold_action"):
                    _cls_missing.append("next_gold_action")
                if _cls_missing:
                    n_action_skipped += 1
                    for f in _cls_missing:
                        _missing_field_counts[f] = _missing_field_counts.get(f, 0) + 1
                    continue
                # 루브릭 11개 전부 verdict 있는지 확인
                dc_rubrics = {d.get("rubric", "") for d in (step.get("prm_deep_critique") or [])}
                if not _cls_expected_rubrics.issubset(dc_rubrics):
                    n_rubric_incomplete += 1
                    continue
                # 페일 루브릭 없이 rethink인 스텝은 모순된 시그널 → 제외
                _next_action = step.get("next_gold_action", "")
                _raw_rubrics = step.get("gold_fail_rubrics")
                _rubrics = _raw_rubrics if isinstance(_raw_rubrics, list) else []
                if TOKEN_RETHINK in _next_action and not _rubrics:
                    n_action_skipped += 1
                    _missing_field_counts["none_rubric_rethink"] = _missing_field_counts.get("none_rubric_rethink", 0) + 1
                    continue
            src   = step.get("source", "gen")
            state = _norm_state(step)
            entry = (problem, steps, k)
            if state == "end":
                (gen_end     if src == "gen" else pat_end).append(entry)
            elif state == "rethink":
                (gen_rethink if src == "gen" else pat_rethink).append(entry)
                if mode == "classification":  # rethink로 확정된 스텝에서만 집계
                    for r in (_raw_rubrics if isinstance(_raw_rubrics, list) else []):
                        _fail_rubric_counts[r] = _fail_rubric_counts.get(r, 0) + 1
            else:
                (gen_solve   if src == "gen" else pat_solve).append(entry)

    n_right_traj = len(raw) - n_traj_skipped
    n_active_steps = n_total_steps - sum(
        len(item["steps"]) for item in raw
        if not no_filter and not item.get("is_right", False)
    )
    if mode == "classification":
        none_rubric_rethink_cnt = _missing_field_counts.pop("none_rubric_rethink", 0)
        field_missing_cnt = n_action_skipped - none_rubric_rethink_cnt
        print(
            f"[preprocess_{mode}] patcher step 제외: {n_patcher_skipped}개 / {n_active_steps}개 "
            f"({n_patcher_skipped * 100 / max(n_active_steps, 1):.1f}%)"
        )
        if field_missing_cnt:
            print(
                f"[preprocess_{mode}] 필드 누락으로 건너뜀: {field_missing_cnt}개 / {n_active_steps}개 "
                f"({field_missing_cnt * 100 / max(n_active_steps, 1):.1f}%)"
            )
            for f, cnt in sorted(_missing_field_counts.items(), key=lambda x: -x[1]):
                print(f"  └─ {f}: {cnt}개 누락")
        if n_rubric_incomplete:
            print(
                f"[preprocess_{mode}] 루브릭 verdict 불완전 제외: {n_rubric_incomplete}개 / {n_active_steps}개 "
                f"({n_rubric_incomplete * 100 / max(n_active_steps, 1):.1f}%)"
            )
        print(
            f"[preprocess_{mode}] rubric=none & action=rethink 필터: {none_rubric_rethink_cnt}개 제외 "
            f"({none_rubric_rethink_cnt * 100 / max(n_active_steps, 1):.1f}%)"
        )
    if no_filter:
        print(f"[preprocess_{mode}] trajectory 필터 없음 (no_filter): 전체 {len(raw)}개 사용")
    else:
        print(f"[preprocess_{mode}] trajectory 필터: is_right {n_right_traj}개 / 전체 {len(raw)}개")
    print(f"[preprocess_{mode}] 총 스텝 수: {n_total_steps}개")

    for lst in [gen_solve, gen_rethink, gen_end, pat_solve, pat_rethink, pat_end]:
        rng.shuffle(lst)

    def _print_fail_rubric_stats(rubric_counts: dict, n_rethink_steps: int) -> None:
        if not rubric_counts or mode != "classification":
            return
        total_fails = sum(rubric_counts.values())
        col = max(max(len(r) for r in rubric_counts), 6)
        sep = "─" * (col + 36)
        print(f"\n  [ Fail rubrics 분포 ]")
        print(f"  % of fails   : 해당 루브릭 언급 수 / 전체 fail 언급 수 합계  (합산 = 100%)")
        print(f"  % of rethink : 해당 루브릭 포함 스텝 수 / rethink 스텝 수    (multi-label이라 합산 > 100%)")
        print(f"\n  {'Rubric':<{col}}  {'Count':>6}  {'% of fails':>10}  {'% of rethink':>13}")
        print(f"  {sep}")
        for rubric, cnt in sorted(rubric_counts.items(), key=lambda x: -x[1]):
            pct_of_fails   = cnt / max(total_fails, 1)
            pct_of_rethink = cnt / max(n_rethink_steps, 1)
            print(f"  {rubric:<{col}}  {cnt:>6}  {pct_of_fails:>9.1%}  {pct_of_rethink:>12.1%}")
        print(f"  {sep}")
        print(f"  {'Total (rubric mentions)':<{col}}  {total_fails:>6}  {'100.0%':>10}  "
              f"{'(rethink: ' + str(n_rethink_steps) + '개)':>13}")

    if no_balance:
        solve_pool  = gen_solve   + pat_solve
        all_rethink = gen_rethink + pat_rethink
        end_pool    = gen_end     + pat_end
        total = len(solve_pool) + len(all_rethink) + len(end_pool)
        print(f"\n  [ 스텝 비율 (자연 분포 — no_balance) ]")
        for label, pool in [("solve", solve_pool), ("rethink", all_rethink), ("end", end_pool)]:
            print(f"  {label:<10} {len(pool):6d}  {len(pool)/max(total,1):5.1%}")
        print(f"  {'합계':<10} {total:6d}")
        _print_fail_rubric_stats(_fail_rubric_counts, len(all_rethink))

        all_samples = _group_and_order_trajectories(
            solve_pool + all_rethink + end_pool, rng)
    else:
        all_rethink = gen_rethink + pat_rethink
        n_rethink   = len(all_rethink)
        n_solve_tgt = round(n_rethink * solve_ratio   / rethink_ratio)
        n_end_tgt   = round(n_rethink * end_ratio     / rethink_ratio)

        solve_pool = gen_solve[:n_solve_tgt]
        if len(solve_pool) < n_solve_tgt:
            solve_pool += pat_solve[:n_solve_tgt - len(solve_pool)]

        end_pool = gen_end[:n_end_tgt]
        if len(end_pool) < n_end_tgt:
            end_pool += pat_end[:n_end_tgt - len(end_pool)]

        total = len(solve_pool) + n_rethink + len(end_pool)
        print(f"\n  [ 스텝 비율 (목표 {solve_ratio}:{rethink_ratio}:{end_ratio}) ]")
        for label, pool in [("solve", solve_pool), ("rethink", all_rethink), ("end", end_pool)]:
            print(f"  {label:<10} {len(pool):6d}  {len(pool)/max(total,1):5.1%}")
        print(f"  {'합계':<10} {total:6d}")
        _print_fail_rubric_stats(_fail_rubric_counts, n_rethink)

        all_samples = _group_and_order_trajectories(
            solve_pool + all_rethink + end_pool, rng)

    # max_length 필터링용 토크나이저 (옵션)
    _tokenizer = None
    if max_length:
        from utils_sft import setup_tokenizer
        _cfg = CONF.get("checkpoint", {})
        _model_id = _cfg.get("sft_checkpoint") or _cfg.get("base", "")
        _cache_dir = _cfg.get("cache_dir")
        tgt_msg = f", target≤{max_target_length}" if max_target_length else ""
        print(f"[preprocess_{mode}] max_length 필터 적용 (total≤{max_length}{tgt_msg}) — 토크나이저 로드 중...")
        _tokenizer = setup_tokenizer(_model_id, _cache_dir)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_short_skipped = 0
    n_long_skipped = 0
    written = 0
    seq_lengths: list[int] = []
    with open(out_path, "w", encoding="utf-8") as f:
        for traj_id, problem, steps, k in tqdm(all_samples, desc=f"  {mode} 전처리",
                                               dynamic_ncols=True, leave=True, file=sys.stderr):
            record = _build(problem, steps, k)
            if mode == "inference" and len(record["target"].split()) < 5:
                n_short_skipped += 1
                continue
            if _tokenizer is not None:
                msgs = record["input"] + [{"role": "assistant", "content": record["target"]}]
                full = _tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                n_tok = len(_tokenizer.encode(full, add_special_tokens=False))
                if n_tok > max_length:
                    n_long_skipped += 1
                    continue
                if max_target_length is not None:
                    n_target_tok = len(_tokenizer.encode(record["target"], add_special_tokens=False))
                    if n_target_tok > max_target_length:
                        n_long_skipped += 1
                        continue
                seq_lengths.append(n_tok)
            else:
                seq_lengths.append(len(record["target"].split()))
            record["traj_id"] = traj_id
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    if n_short_skipped:
        print(f"[preprocess_{mode}] 짧은/오염 타겟 제외: {n_short_skipped}개")
    if n_long_skipped:
        tgt_msg = f" 또는 target>{max_target_length}" if max_target_length else ""
        print(f"[preprocess_{mode}] total>{max_length}{tgt_msg} 제외: {n_long_skipped}개")
    print(f"[preprocess_{mode}] 완료: {written}개  →  {out_path}")

    if seq_lengths:
        import statistics
        sl = sorted(seq_lengths)
        n = len(sl)
        unit = "tokens (전체 시퀀스)" if _tokenizer is not None else "words in target (approximate)"
        print(f"\n  [ 시퀀스 길이 분포 — {unit} ]")
        print(f"  {'최솟값':<14} {sl[0]:>8,}")
        print(f"  {'중앙값 (p50)':<14} {int(statistics.median(sl)):>8,}")
        print(f"  {'p75':<14} {sl[int(n * 0.75)]:>8,}")
        print(f"  {'p95 (상위 5%)':<14} {sl[int(n * 0.95)]:>8,}")
        print(f"  {'p99 (상위 1%)':<14} {sl[int(n * 0.99)]:>8,}")
        print(f"  {'최댓값':<14} {sl[-1]:>8,}")
        if _tokenizer is None:
            print(f"  ※ 토크나이저 미로드 — target 단어 수 기준 (참고용)")
            print(f"     정확한 토큰 수는 --max_length 옵션 추가 후 재실행하세요.")
    return str(out_path)


def _parse_verdicts_from_target(target: str) -> dict[str, str]:
    """target 텍스트의 Deep critic 섹션에서 {rubric: verdict} 파싱."""
    import re
    m = re.search(r"Deep\s+critic\s*:", target, re.IGNORECASE)
    if not m:
        return {}
    section = target[m.end():]
    end = re.search(r"\n\n(Fail\s+rubrics|Next\s+action)\s*:", section, re.IGNORECASE)
    if end:
        section = section[:end.start()]
    result = {}
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        vm = re.search(r"verdict:\s*(correct|incorrect)", stripped, re.IGNORECASE)
        if not vm:
            continue
        rubric = re.split(r"\s*:", stripped, maxsplit=1)[0].strip()
        if rubric:
            result[rubric] = vm.group(1).lower()
    return result


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
    """저장된 전처리 JSONL에서 is_fail=False인 샘플 중 idx번째(0-based)를 읽어 TARGET을 출력한다."""
    sep = "─" * 72
    with open(preprocessed_path, encoding="utf-8") as f:
        count = 0
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("is_fail", False):
                continue
            if count == idx:
                break
            count += 1
        else:
            print(f"is_fail=False 샘플 {idx}번을 찾을 수 없습니다.")
            return
    print(f"\n{'='*72}")
    print(f"[state={sample.get('state')}  is_fail={sample.get('is_fail')}]")
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
    """Gen-only path: is_fail=False인 스텝만 뽑아 gen_solve 포맷으로 변환 후 out_file에 씁니다."""
    total = 0
    for item in tqdm(items, desc="  Gen-only 전처리"):
        problem = item["problem"]
        for step in item["steps"]:
            if not step.get("is_fail", True):   # 성공한 스텝만
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
    no_balance: bool = False,
    no_filter: bool = False,
) -> str:
    """
    trajectory를 스텝 단위로 분류해 SFT 데이터를 생성합니다.

    no_balance=False (기본): solve_ratio:rethink_ratio:end_ratio 비율로 샘플링
    no_balance=True        : 전체를 자연 분포 그대로 사용
                             → class weighted loss (--action_weights)와 함께 쓸 것
    no_filter=False (기본) : is_right=True trajectory만 사용
    no_filter=True         : is_right 필터 없이 전체 trajectory 사용

    is_fail=False 스텝 → 전체 시퀀스에 loss (좋은 inference 학습)
    is_fail=True  스텝 → inference 마스킹 후 Fast critic~Next action만 loss
                         (나쁜 inference는 건너뛰고 비판/액션 판단만 학습)
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
        if not no_filter and not item.get("is_right", False):
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
    if no_filter:
        print(f"[preprocess] trajectory 필터 없음 (no_filter): 전체 {len(raw)}개 사용")
    else:
        print(f"[preprocess] trajectory 필터: is_right {n_right_traj}개 사용 / 전체 {len(raw)}개 (제외 {n_traj_skipped}개)")
    print(f"[preprocess] 총 스텝 수: {n_total_steps}개 (전체 사용)")

    for lst in [gen_solve, gen_rethink, gen_end, pat_solve, pat_rethink, pat_end]:
        rng.shuffle(lst)

    def _count_src(pool):
        ng = sum(1 for _, steps, k in pool if steps[k].get("source", "gen") == "gen")
        return ng, len(pool) - ng

    if no_balance:
        # ── 자연 분포: is_right=True 전체 사용 (균형화 없음) ─────────────────
        solve_pool   = gen_solve   + pat_solve
        all_rethink  = gen_rethink + pat_rethink
        end_pool     = gen_end     + pat_end

        total = len(solve_pool) + len(all_rethink) + len(end_pool)
        print(f"\n  [ 스텝 비율 (자연 분포 — no_balance) ]")
        print(f"  {'액션':<10} {'수':>6}  {'비율':>6}")
        print(f"  {'─'*30}")
        for label, pool in [("solve", solve_pool), ("rethink", all_rethink), ("end", end_pool)]:
            print(f"  {label:<10} {len(pool):6d}  {len(pool)/max(total,1):5.1%}")
        print(f"  {'─'*30}")
        print(f"  {'합계':<10} {total:6d}")

        all_samples = _group_and_order_trajectories(
            solve_pool + all_rethink + end_pool, rng)
    else:
        # ── 목표 수 계산 (rethink 전부 기준) ────────────────────────────────
        all_rethink = gen_rethink + pat_rethink
        n_rethink   = len(all_rethink)
        n_solve_tgt = round(n_rethink * solve_ratio   / rethink_ratio)
        n_end_tgt   = round(n_rethink * end_ratio     / rethink_ratio)

        # ── solve: gen 우선, 부족하면 patcher 보충 ───────────────────────────
        solve_pool  = gen_solve[:n_solve_tgt]
        if len(solve_pool) < n_solve_tgt:
            solve_pool += pat_solve[:n_solve_tgt - len(solve_pool)]

        # ── end: gen 우선, 부족하면 patcher 보충 ─────────────────────────────
        end_pool = gen_end[:n_end_tgt]
        if len(end_pool) < n_end_tgt:
            end_pool += pat_end[:n_end_tgt - len(end_pool)]

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

        all_samples = _group_and_order_trajectories(
            solve_pool + all_rethink + end_pool, rng)

    # ── 저장 (trajectory 단위 순서 보장, traj_id 포함) ─────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for traj_id, problem, steps, k in tqdm(all_samples, desc="  전처리"):
            record = build_sft_sample(problem, steps, k, system_solve, system_rethink)
            record["traj_id"] = traj_id
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
    p.add_argument("--mode",         choices=["full", "inference", "classification"],
                   default="full",
                   help="전처리 모드: full(기본)=전체 target, inference=step+Does만, classification=critic+action만")
    p.add_argument("--no-balance",   dest="no_balance", action="store_true",
                   help="균형화 없이 자연 분포 그대로 출력. --action_weights와 함께 사용.")
    p.add_argument("--no-filter",    dest="no_filter",  action="store_true",
                   help="is_right 필터 없이 전체 trajectory의 모든 스텝 사용.")
    p.add_argument("--rubric_weights", action="store_true", default=False,
                   help="classification mode: Fail rubrics 섹션을 target에 포함 (학습 대상에 추가)")
    p.add_argument("--action_weights", action="store_true", default=False,
                   help="classification mode: Next action 섹션을 target에 포함 (학습 대상에 추가)")
    p.add_argument("--use_summary", action="store_true", default=False,
                   help="classification mode: critique 원문 대신 prm_critique_summary 짧은 요약을 target으로 사용")
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
    p.add_argument("--max_length", type=int, default=None,
                   help="전처리 시 input+target 합계 초과 샘플 제외 (토크나이저 로드 필요)")
    p.add_argument("--max_target_length", type=int, default=None,
                   help="전처리 시 target 단독 토큰 수 초과 샘플 제외")
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

    if args.mode in ("inference", "classification"):
        preprocess_mode(
            args.output_path,
            data_path=args.data_path,
            mode=args.mode,
            seed=args.seed,
            solve_ratio=args.solve_ratio,
            rethink_ratio=args.rethink_ratio,
            end_ratio=args.end_ratio,
            no_balance=args.no_balance,
            no_filter=args.no_filter,
            include_rubrics=True,
            include_actions=args.action_weights,
            max_length=args.max_length,
            max_target_length=args.max_target_length,
            use_summary=args.use_summary,
        )
    else:
        preprocess(
            args.output_path,
            data_path=args.data_path,
            seed=args.seed,
            solve_ratio=args.solve_ratio,
            rethink_ratio=args.rethink_ratio,
            end_ratio=args.end_ratio,
            no_balance=args.no_balance,
            no_filter=args.no_filter,
        )


if __name__ == "__main__":
    main()
