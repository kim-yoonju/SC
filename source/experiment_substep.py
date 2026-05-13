"""
experiment_substep.py

가설 검증: rethink 실패 스텝을 재귀적으로 2분할해서
          각 서브스텝만 rethink 시키면 patcher 없이 pass 가능한가?

알고리즘:
  1. 실패한 gen 스텝에 대해 patcher 정답을 2개 서브스텝으로 분해
  2. 각 서브스텝에 대해 rethink
     - pass → 다음 서브스텝으로
     - fail + 아직 분할 가능 → 해당 서브스텝을 다시 2분할 후 재귀
     - fail + 더 못 쪼갬 → patcher 필요로 표시
  3. 모든 서브스텝 pass 여부와 patcher 필요 비율 집계

분해 모델:
  --decompose_model local  → 로컬 SFT 체크포인트 (GPU 3)
  --decompose_model <name> → API 모델 (deepseek-pro 등)

Usage:
  python source/experiment_substep.py \\
    --input output/rethink_fail_cases_20260511_both.jsonl \\
    --decompose_model local --gpu 3 --max_depth 2
"""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (CONF, _call_llm, _print_cost_summary, build_chat_prompt,
                   load_step_manager, load_generator,
                   STEP_MANAGER_GPU, STEP_MANAGER_PATH, STEP_MANAGER_MAX_TOKENS,
                   GENERATOR_MODEL_ID)
from PRM import ApiPrmBatch, load_fast_rubric

ROOT             = Path(__file__).resolve().parent.parent
INPUT_FILE       = ROOT / "output" / "rethink_fail_substep.jsonl"
OUTPUT_FILE      = ROOT / "output" / "substep_experiment.jsonl"
FAST_RUBRIC_PATH = ROOT / CONF.get("PRM", {}).get("fast_rubric", "prompts/batch_prm_rubric_v7.3.json")
PRM_MODEL        = CONF.get("PRM", {}).get("model_id", "deepseek-chat")

_ROLLOUT_GPUS            = CONF.get("generate_trajectory", {}).get("rollout_gpus", [0, 1])
DEFAULT_DECOMPOSE_MODEL  = "local"
DECOMPOSE_GPU            = _ROLLOUT_GPUS[0]          # rollout_gpus[0]
DECOMPOSE_CHECKPOINT     = STEP_MANAGER_PATH
RETHINK_GPU              = _ROLLOUT_GPUS[1] if len(_ROLLOUT_GPUS) > 1 else _ROLLOUT_GPUS[0]
RETHINK_CHECKPOINT       = GENERATOR_MODEL_ID        # checkpoint.base

# ── PRM 판정 기준 ──────────────────────────────────────────────────────────────
_EXTRA_RUBRICS = frozenset({"Progress and Non-Repetition", "Atomicity"})

def _is_fail(verdicts: list[dict]) -> bool:
    extra = sum(1 for v in verdicts if v["pred"] == "incorrect" and v["name"] in _EXTRA_RUBRICS)
    core  = sum(1 for v in verdicts if v["pred"] == "incorrect" and v["name"] not in _EXTRA_RUBRICS)
    return core >= 2 or extra >= 1

def _fail_rubrics(verdicts: list[dict]) -> list[str]:
    return [v["name"] for v in verdicts if v["pred"] == "incorrect"]


# ── Prompts ───────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """\
You are evaluating whether a math reasoning step can be split into two independent sub-steps.

A valid split requires ALL of the following:
- Sub-step A produces a standalone result Ra
- Sub-step B produces a standalone result Rb
- Ra is NOT needed to derive Rb, AND Rb is NOT needed to derive Ra

ALWAYS ATOMIC (never split):
- Sequential algebra chains (a→b→c, each line feeds the next)
- Introduce-then-use patterns (e.g. "Let u=f(x), then du=...")
- Multiple computations that ALL feed into one final synthesis
- Case analysis where all cases establish the same single conclusion

NON-ATOMIC (valid split):
- Two independent approaches each reaching their own conclusion
- Two symmetric cases with independent results (e.g. x=+v and x=−v)

If NON-ATOMIC, output JSON:
{"atomic": false, "sub1": "<goal of first independent sub-step>", "sub2": "<goal of second independent sub-step>"}

If ATOMIC (cannot be meaningfully split), output:
{"atomic": true}

Output ONLY the JSON. No other text."""

_RETHINK_SUBSTEP_SYSTEM = """\
You are correcting part of an incorrect reasoning step.

The previous step was wrong. Your task is to write a corrected version \
that accomplishes ONLY this specific goal: {goal}

RULES:
- Write EXACTLY ONE corrected step for the stated goal.
- ONE OPERATION ONLY: do exactly one computation, substitution, or deduction.
- EXECUTE — show every computation explicitly.
- Do NOT address any other part of the original step.
- Use LaTeX for all math."""


# ── 로컬 모델 ─────────────────────────────────────────────────────────────────

# step_manager (분해 전용, GPU 0)
_decompose_model = None
_decompose_tok   = None

# base model (rethink 전용, GPU 1)
_rethink_model   = None
_rethink_tok     = None


def load_decompose_model(gpu_id: int, checkpoint_path: str):
    global _decompose_model, _decompose_tok
    print(f"[Step Manager] 로드 중 (GPU {gpu_id}): {checkpoint_path}")
    _decompose_model, _decompose_tok = load_step_manager(
        gpu_id=gpu_id, model_path=checkpoint_path
    )
    print(f"[Step Manager] 로드 완료")


def load_rethink_model(gpu_id: int, checkpoint_path: str):
    global _rethink_model, _rethink_tok
    print(f"[Base/Rethink] 로드 중 (GPU {gpu_id}): {checkpoint_path}")
    _rethink_model, _rethink_tok = load_generator(
        model_path=checkpoint_path, device_map=f"cuda:{gpu_id}"
    )
    print(f"[Base/Rethink] 로드 완료")


def _generate_with(model, tok, system: str, user: str, max_new_tokens: int = 1024) -> str | None:
    import torch
    if model is None:
        return None
    prompt = build_chat_prompt(tok, system, user)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()


def _call_decompose(system: str, user: str, decompose_model: str) -> str | None:
    if decompose_model == "local":
        return _generate_with(_decompose_model, _decompose_tok, system, user, max_new_tokens=1024)
    return _call_llm(decompose_model,
                     [{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
                     max_completion_tokens=1024)


# ── Core functions ────────────────────────────────────────────────────────────

def _parse_decompose_resp(resp: str) -> list[dict] | None:
    """Atomicity 기반 분해 응답 파싱. atomic이면 None, 아니면 [{goal}, {goal}]."""
    # LaTeX 역슬래시 수정
    def _fix_bs(s: str) -> str:
        valid = set('"\\bfnrtu/')
        r, i = [], 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s) and s[i+1] not in valid:
                r.append('\\\\')
            else:
                r.append(s[i])
            i += 1
        return ''.join(r)

    m = re.search(r"\{.*\}", resp, re.DOTALL)
    if not m:
        return None
    raw = m.group()
    for candidate in [raw, _fix_bs(raw)]:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            data = None
    if data is None:
        return None

    if data.get("atomic", True):
        return None   # atomic → 쪼갤 수 없음
    sub1 = data.get("sub1", "").strip()
    sub2 = data.get("sub2", "").strip()
    if not sub1 or not sub2:
        return None
    return [{"goal": sub1}, {"goal": sub2}]


def decompose_into_2(problem: str, history_str: str, step_content: str,
                     decompose_model: str) -> list[dict] | None:
    """Atomicity 기준으로 step_content를 2개 독립 서브스텝으로 분해.
    atomic이면 None 반환."""
    user_msg = (
        f"Problem:\n{problem}\n\n"
        f"Previous steps:\n{history_str or '(none)'}\n\n"
        f"Step to evaluate:\n{step_content}"
    )
    resp = _call_decompose(_DECOMPOSE_SYSTEM, user_msg, decompose_model)
    if not resp:
        return None
    return _parse_decompose_resp(resp)


def rethink_substep(problem: str, history_str: str, gen_step: str, goal: str) -> str | None:
    """base 모델(로컬)로 서브스텝 rethink."""
    system = _RETHINK_SUBSTEP_SYSTEM.format(goal=goal)
    user_msg = (
        f"Problem:\n{problem}\n\n"
        f"Previous steps (correct):\n{history_str or '(none)'}\n\n"
        f"Incorrect step to fix:\n{gen_step}\n\n"
        f"Write the corrected version for: {goal}"
    )
    return _generate_with(_rethink_model, _rethink_tok, system, user_msg, max_new_tokens=1024)


def prm_eval(problem: str, prev_str: str, step: str, prm: ApiPrmBatch) -> tuple[bool, list[str]]:
    verdicts = prm.evaluate_batch(
        questions=[problem], prev_steps=[prev_str], now_steps=[step]
    )[0]
    for v, name in zip(verdicts, prm.rubric_names):
        v["name"] = name
    failed = _is_fail(verdicts)
    return not failed, _fail_rubrics(verdicts)


# ── 재귀 알고리즘 ──────────────────────────────────────────────────────────────

def attempt_recursive(
    problem: str,
    base_history_str: str,    # history (고정)
    prev_context: list[str],  # 이전 서브스텝 누적
    gen_step: str,            # 원래 틀린 스텝 (에러 컨텍스트 + 분해 대상)
    target_goal: str,         # 이번 서브스텝 목표 설명
    prm: ApiPrmBatch,
    decompose_model: str,
    depth: int,
    max_depth: int,
) -> dict:
    """
    target_goal을 rethink로 달성 시도. 실패하면 재귀 분해.
    반환: {"passed": bool, "needs_patcher": bool, "steps": [...]}
    """
    prev_str = "\n\n".join([base_history_str] + prev_context).strip()

    # 1. rethink 시도
    generated = rethink_substep(problem, prev_str, gen_step, target_goal)
    if not generated:
        generated = "(generation failed)"

    passed, fail_rbs = prm_eval(problem, prev_str, generated, prm)

    if passed:
        return {
            "passed": True,
            "needs_patcher": False,
            "steps": [{"goal": target_goal, "generated": generated,
                       "fail_rubrics": [], "depth": depth}],
        }

    # 2. 실패 → 더 쪼갤 수 있는지 Atomicity 기준으로 판단
    if depth >= max_depth:
        return {
            "passed": False,
            "needs_patcher": True,
            "steps": [{"goal": target_goal, "generated": generated,
                       "fail_rubrics": fail_rbs, "depth": depth,
                       "note": "max_depth_reached"}],
        }

    # 분해 대상: 지금 rethink가 시도한 내용 (generated) 또는 gen_step
    # rethink 결과가 틀렸으니 그걸 다시 쪼개는 것보다 목표(goal)를 더 작게 쪼갬
    subs = decompose_into_2(problem, prev_str, generated, decompose_model)
    if subs is None:
        return {
            "passed": False,
            "needs_patcher": True,
            "steps": [{"goal": target_goal, "generated": generated,
                       "fail_rubrics": fail_rbs, "depth": depth,
                       "note": "atomic_cannot_split"}],
        }

    # 3. 2개 서브스텝으로 재귀
    all_steps = []
    all_passed = True
    any_needs_patcher = False
    running_context = list(prev_context)

    for sub in subs:
        sub_goal = sub.get("goal", "sub-step")

        sub_result = attempt_recursive(
            problem, base_history_str, running_context,
            gen_step, sub_goal,
            prm, decompose_model, depth + 1, max_depth,
        )

        all_steps.extend(sub_result["steps"])
        if sub_result["passed"]:
            running_context.append(sub_result["steps"][-1]["generated"])
        else:
            all_passed = False
            # 실패한 서브스텝은 컨텍스트에서 빠짐 (이어지는 서브스텝은 독립적이므로 OK)

        if sub_result["needs_patcher"]:
            any_needs_patcher = True

    return {
        "passed": all_passed,
        "needs_patcher": any_needs_patcher,
        "steps": all_steps,
    }


def process_case(case: dict, prm: ApiPrmBatch, decompose_model: str, max_depth: int) -> dict | None:
    problem  = case["problem"]
    history  = case["history"]   # does 요약 리스트 (is_error=False 스텝만)

    # 두 가지 입력 포맷 지원:
    #   신규: current_step + fail_rubrics (rethink_fail_substep.jsonl)
    #   구형: gen.inference + rethink.fail_rubrics (rethink_fail_cases.jsonl)
    wrong_step   = case.get("current_step") or case.get("gen", {}).get("inference", "")
    fail_rubrics = case.get("fail_rubrics") or case.get("rethink", {}).get("fail_rubrics", [])

    history_str = "\n\n".join(history) if history else ""

    # 최초 분해: 틀린 스텝을 Atomicity 기준으로 쪼갬
    subs = decompose_into_2(problem, history_str, wrong_step, decompose_model)
    if subs is None:
        return {
            "traj_id":      case.get("traj_id", "?"),
            "problem_id":   case.get("problem_id", "?"),
            "n_steps_tried":  0,
            "all_passed":     False,
            "needs_patcher":  True,
            "skipped_atomic": True,
            "steps":          [],
            "fail_rubrics":   fail_rubrics,
        }

    all_steps         = []
    all_passed        = True
    any_needs_patcher = False
    running_context   = []

    for sub in subs:
        sub_goal = sub.get("goal", "sub-step")

        result = attempt_recursive(
            problem, history_str, running_context,
            wrong_step, sub_goal,
            prm, decompose_model, depth=0, max_depth=max_depth,
        )

        all_steps.extend(result["steps"])
        if result["passed"]:
            running_context.append(result["steps"][-1]["generated"])
        else:
            all_passed = False

        if result["needs_patcher"]:
            any_needs_patcher = True

    return {
        "traj_id":      case.get("traj_id", "?"),
        "problem_id":   case.get("problem_id", "?"),
        "n_steps_tried":  len(all_steps),
        "all_passed":     all_passed,
        "needs_patcher":  any_needs_patcher,
        "skipped_atomic": False,
        "steps":          all_steps,
        "fail_rubrics":   fail_rubrics,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          type=str, default=str(INPUT_FILE))
    parser.add_argument("--output",         type=str, default=str(OUTPUT_FILE))
    parser.add_argument("--n",              type=int, default=89)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--decompose_model",   type=str, default=DEFAULT_DECOMPOSE_MODEL,
                        help="분해 모델: 'local'(기본) 또는 API 모델명")
    parser.add_argument("--decompose_gpu",     type=int, default=DECOMPOSE_GPU,
                        help=f"step_manager GPU (기본={DECOMPOSE_GPU})")
    parser.add_argument("--decompose_ckpt",    type=str, default=DECOMPOSE_CHECKPOINT,
                        help="step_manager 체크포인트")
    parser.add_argument("--rethink_gpu",       type=int, default=RETHINK_GPU,
                        help=f"base rethink 모델 GPU (기본={RETHINK_GPU})")
    parser.add_argument("--rethink_ckpt",      type=str, default=RETHINK_CHECKPOINT,
                        help="base rethink 체크포인트")
    parser.add_argument("--max_depth",         type=int, default=2,
                        help="최대 재귀 분할 깊이")
    parser.add_argument("--workers",           type=int, default=1,
                        help="로컬 모델 사용 시 1 고정")
    args = parser.parse_args()
    args.workers = 1  # 로컬 모델 2개 동시 사용 → 순차 처리

    # 모델 로드
    load_decompose_model(args.decompose_gpu, args.decompose_ckpt)
    load_rethink_model(args.rethink_gpu, args.rethink_ckpt)

    with open(args.input) as f:
        all_cases = [json.loads(l) for l in f]
    random.seed(args.seed)
    cases = random.sample(all_cases, min(args.n, len(all_cases)))

    print(f"샘플: {len(cases)}개")
    print(f"[GPU] decompose(step_manager)={args.decompose_gpu}  rethink(base)={args.rethink_gpu}  prm={PRM_MODEL}")
    print(f"decompose={args.decompose_model}  max_depth={args.max_depth}")

    fast_rubric = load_fast_rubric(FAST_RUBRIC_PATH)
    prm = ApiPrmBatch(PRM_MODEL, fast_rubric, max_workers=32)

    results, skipped = [], 0

    def _run(case):
        return process_case(case, prm, args.decompose_model, args.max_depth)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run, c): c for c in cases}
        for fut in tqdm(as_completed(futs), total=len(cases), desc="cases"):
            r = fut.result()
            if r is None:
                skipped += 1
            else:
                results.append(r)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("처리된 케이스 없음")
        return

    from collections import Counter

    # 케이스 분류
    n_atomic  = sum(r.get("skipped_atomic", False) for r in results)
    tried     = [r for r in results if not r.get("skipped_atomic")]
    n_tried   = len(tried)

    # 분해 시도한 케이스에서만 집계
    tried_pass    = sum(r["all_passed"]   for r in tried)
    tried_fail    = n_tried - tried_pass

    # 서브스텝 집계 (분해 시도한 케이스의 스텝들만)
    all_steps = [s for r in tried for s in r["steps"]]
    n_steps   = len(all_steps)
    step_pass = sum(not s.get("fail_rubrics") for s in all_steps)
    step_fail = n_steps - step_pass

    # 종료 사유
    note_dist = Counter(s.get("note", "pass") for s in all_steps)

    # 원본 fail rubrics top3
    orig_fail = Counter(rb for r in results for rb in r.get("fail_rubrics", []))

    print(f"\n{'='*60}")
    print(f" 결과 요약  (전체={n}개  skipped={skipped}개)")
    print(f"{'='*60}")
    print(f" atomic → 분해 불가:     {n_atomic}/{n} ({100*n_atomic/n:.0f}%)")
    print(f" 분해 시도:              {n_tried}/{n} ({100*n_tried/n:.0f}%)")
    if n_tried:
        print(f"{'─'*60}")
        print(f" [케이스 수준]  분해 시도 {n_tried}개")
        print(f"   케이스 전체 pass:     {tried_pass}/{n_tried} ({100*tried_pass/n_tried:.0f}%)")
        print(f"   케이스 전체 fail:     {tried_fail}/{n_tried} ({100*tried_fail/n_tried:.0f}%)")
    if all_steps:
        print(f"{'─'*60}")
        print(f" [서브스텝 수준]  총 {n_steps}개 생성")
        print(f"   서브스텝 pass:        {step_pass}/{n_steps} ({100*step_pass/n_steps:.0f}%)")
        print(f"   서브스텝 fail:        {step_fail}/{n_steps} ({100*step_fail/n_steps:.0f}%)")
        print(f"   종료 사유:            {dict(note_dist)}")
    print(f"{'─'*60}")
    print(f" 원본 fail rubrics (top3):")
    for rb, cnt in orig_fail.most_common(3):
        print(f"   {cnt:3d}  {rb}")
    print(f"{'='*60}")
    print(f" 저장: {out_path}")

    _print_cost_summary()


if __name__ == "__main__":
    main()
