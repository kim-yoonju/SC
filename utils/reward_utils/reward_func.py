"""
GRPO reward function for SC (Self-Correction) training.

External PRM으로 각 스텝을 평가해 0~2 per-step reward 계산.

Per-step reward structure:
  - "Fail rubrics:" position: PRM이 해당 스텝을 pass → +1.0
  - action token position:   모델이 선택한 액션이 PRM 판정과 일치 → +1.0
      PRM pass  → <|solve|> or <|end|>(정답 맞음) 이면 correct
      PRM fail  → <|rethink|> 이면 correct

2-stage PRM:
  Stage 1: batch 루브릭 (config.PRM.fast_rubric) — 빠름
  Stage 2: 개별 루브릭 (config.PRM.rubric) — Stage 1 fail인 스텝만 재평가

extra_info["problem"] 에 문제 텍스트가 있어야 함.
"""

import re
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "source"))

try:
    from utils import CONF
    from utils_math import check_solved
    from PRM import ApiPrmBatch, ApiPrm, load_fast_rubric, load_deep_rubrics, build_system_prompt
except Exception as _import_err:
    traceback.print_exc()
    raise

# ── 설정 ──────────────────────────────────────────────────────────────────────
_PRM_CONF = CONF.get("PRM", {})
_MODEL_ID = _PRM_CONF.get("model_id", "deepseek-chat")
_FAST_RUBRIC_REL = _PRM_CONF.get("fast_rubric", "")
_RUBRIC_REL = _PRM_CONF.get("rubric", "")
_STAGE1_MAX_TOK = _PRM_CONF.get("stage1_max_new_tokens", 2048)
_STAGE2_MAX_TOK = _PRM_CONF.get("max_new_tokens", 4096)

# ── 지연 초기화 ───────────────────────────────────────────────────────────────
_prm_batch: "ApiPrmBatch | None" = None
_prm_individual: "ApiPrm | None" = None
_rubrics: "list | None" = None


def _get_prm():
    global _prm_batch, _prm_individual, _rubrics
    if _prm_batch is None:
        def _abs(rel):
            p = Path(rel)
            return p if p.is_absolute() else _ROOT / p

        fast_rubric = load_fast_rubric(_abs(_FAST_RUBRIC_REL))
        _rubrics = load_deep_rubrics(_abs(_RUBRIC_REL))
        _prm_batch = ApiPrmBatch(_MODEL_ID, fast_rubric, max_workers=32)
        _prm_individual = ApiPrm(_MODEL_ID, max_workers=32)
    return _prm_batch, _prm_individual, _rubrics


# ── 정규식 ────────────────────────────────────────────────────────────────────
_ACTION_TOKEN_RE = re.compile(r'(<\|solve\|>|<\|rethink\|>|<\|end\|>)')
_FAST_CRITIC_KW_RE = re.compile(r'Fast critic:')
_FAIL_RUBRICS_KW_RE = re.compile(r'Fail rubrics:')
_NEXT_ACTION_KW_RE = re.compile(r'Next action:')


def _parse_fail_section(text: str) -> set[str]:
    """'Fail rubrics:' 이후 텍스트에서 루브릭 이름 집합 파싱."""
    text = text.strip()
    if not text or text.lower() in ("none", "n/a", "-", "none."):
        return set()
    fails = set()
    for line in text.splitlines():
        line = line.strip().lstrip("-•*").strip()
        if line and line.lower() not in ("none", "n/a"):
            fails.add(line)
    return fails


# ── "Next action:" 텍스트에서 액션 파싱 (inference_gen_solo 동일 로직) ─────────
_ACTION_TEXT_MAP = {
    "<|solve|>":   "<|solve|>",
    "<|rethink|>": "<|rethink|>",
    "<|end|>":     "<|end|>",
    "solve":       "<|solve|>",
    "rethink":     "<|rethink|>",
    "end":         "<|end|>",
}

def _parse_action_from_text(text: str) -> str | None:
    """'Next action:' 이후 텍스트에서 액션 토큰 파싱. inference_gen_solo._parse_next_action_text와 동일."""
    m = re.search(r"Next action:\s*\n?(.*?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    content = m.group(1).strip()
    for key, token in _ACTION_TEXT_MAP.items():
        if key in content:
            return token
    return None


# ── 스텝 파싱 ─────────────────────────────────────────────────────────────────
def _parse_steps(response: str) -> list[dict]:
    """
    response를 액션 토큰 기준으로 분할.
    액션 토큰이 잘린 경우(max_response_length 도달) "Next action:" 텍스트로 폴백.
    각 스텝:
      reasoning_text    : "Fast critic:" 앞 추론 텍스트 (PRM 입력용)
      full_text         : 액션 토큰 이전 전체 텍스트
      action            : <|solve|> / <|rethink|> / <|end|>
      rubric_pos        : response 내 "Fail rubrics:" 시작 char 위치
      model_fail_rubrics: 모델이 나열한 fail 루브릭 집합
    """
    parts = _ACTION_TOKEN_RE.split(response)
    # parts = [text0, action0, text1, action1, ...]  (정상)
    # parts = [text0]                                (액션 토큰 잘린 경우)

    steps = []
    char_offset = 0

    i = 0
    while i < len(parts):
        full_text = parts[i]

        # 액션 토큰이 뒤따르면 사용, 없으면 "Next action:" 텍스트에서 파싱
        if i + 1 < len(parts):
            action = parts[i + 1]
            i += 2
        else:
            # 폴백: max_response_length로 잘린 경우
            action = _parse_action_from_text(full_text)
            if action is None:
                break  # 액션 없음 → 미완성 스텝, 무시
            i += 1

        # "Fail rubrics:" 위치
        rb_match = None
        for m in _FAIL_RUBRICS_KW_RE.finditer(full_text):
            rb_match = m
        rubric_pos = char_offset + rb_match.start() if rb_match else char_offset + max(0, len(full_text) - 1)

        # PRM 입력용 추론 텍스트: "Fast critic:" 이전 (없으면 "Fail rubrics:" 이전)
        fc_match = _FAST_CRITIC_KW_RE.search(full_text)
        if fc_match:
            reasoning_text = full_text[:fc_match.start()].strip()
        elif rb_match:
            reasoning_text = full_text[:rb_match.start()].strip()
        else:
            reasoning_text = full_text.strip()

        # 모델이 생성한 "Fail rubrics:" 목록 파싱
        if rb_match:
            after_rb = full_text[rb_match.end():]
            na_in_after = _NEXT_ACTION_KW_RE.search(after_rb)
            fail_section = after_rb[:na_in_after.start()] if na_in_after else after_rb
            model_fail_rubrics = _parse_fail_section(fail_section)
        else:
            model_fail_rubrics = set()

        steps.append({
            "reasoning_text":    reasoning_text,
            "full_text":         full_text,
            "action":            action,
            "rubric_pos":        rubric_pos,
            "model_fail_rubrics": model_fail_rubrics,
        })

        char_offset += len(full_text) + len(action)

    return steps


# ── 외부 PRM 평가 ──────────────────────────────────────────────────────────────
def _evaluate_steps(question: str, steps: list[dict]) -> list[dict]:
    """
    각 스텝을 2-stage PRM으로 평가.
    반환: [{
        "pass": bool,
        "fail_rubrics": set[str],
        "stage1_verdicts": {rubric_name: "correct"/"incorrect"},
        "stage2_verdicts": {rubric_name: "correct"/"incorrect"} | None,
    }, ...]
    Stage 1: batch 루브릭 (빠름)
    Stage 2: 개별 루브릭 (Stage 1 fail 스텝만)
    """
    prm_batch, prm_individual, rubrics = _get_prm()
    n = len(steps)
    if n == 0:
        return []

    questions = [question] * n
    prev_steps_list: list[str] = []
    now_steps_list: list[str] = []

    cumulative = ""
    for step in steps:
        prev_steps_list.append(cumulative.strip())
        now_steps_list.append(step["reasoning_text"])
        cumulative += step["reasoning_text"] + "\n" + step["action"] + "\n"

    # Stage 1: 배치 루브릭
    batch_results = prm_batch.evaluate_batch(
        questions=questions,
        prev_steps=prev_steps_list,
        now_steps=now_steps_list,
        max_new_tokens=_STAGE1_MAX_TOK,
    )

    step_results: list[dict] = []
    need_individual: list[int] = []
    for i, verdicts in enumerate(batch_results):
        stage1 = {prm_batch.rubric_names[j]: v.get("pred", "incorrect") for j, v in enumerate(verdicts)}
        prm_fails = {name for name, pred in stage1.items() if pred == "incorrect"}
        step_results.append({
            "pass": not bool(prm_fails) or None,
            "fail_rubrics": prm_fails,
            "stage1_verdicts": stage1,
            "stage2_verdicts": None,
        })
        if prm_fails:
            step_results[-1]["pass"] = None
            need_individual.append(i)

    # Stage 2: 개별 루브릭 (fail 스텝만) — 더 정밀한 루브릭별 판정으로 덮어쓰기
    if need_individual and rubrics:
        for step_idx in need_individual:
            ind_results = prm_individual.evaluate_batch(
                questions=[question] * len(rubrics),
                prev_steps=[prev_steps_list[step_idx]] * len(rubrics),
                now_steps=[now_steps_list[step_idx]] * len(rubrics),
                system_prompts=[build_system_prompt(r) for r in rubrics],
                max_new_tokens=_STAGE2_MAX_TOK,
            )
            stage2 = {rubrics[j]["name"]: r.get("pred", "incorrect") for j, r in enumerate(ind_results)}
            ind_fails = {name for name, pred in stage2.items() if pred == "incorrect"}
            step_results[step_idx]["pass"] = len(ind_fails) == 0
            step_results[step_idx]["fail_rubrics"] = ind_fails
            step_results[step_idx]["stage2_verdicts"] = stage2

    # None 남은 경우 fail로 처리
    for r in step_results:
        if r["pass"] is None:
            r["pass"] = False
    return step_results


# ── 메인 reward 함수 ───────────────────────────────────────────────────────────
def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """
    solution_str: skip_special_tokens=False 로 디코딩된 모델 응답.
    extra_info["problem"]: 문제 텍스트 (PRM 호출에 필요).
    """
    steps = _parse_steps(solution_str)
    if not steps:
        return {"score": 0.0, "rubric_rewards": [], "action_rewards": []}

    question = (extra_info or {}).get("problem", "")
    outcome = 1.0 if check_solved(solution_str, ground_truth) else 0.0

    step_results = _evaluate_steps(question, steps)

    rubric_rewards: list[tuple[float, int]] = []
    action_rewards: list[float] = []

    for step, prm_result in zip(steps, step_results):
        prm_pass = prm_result["pass"]
        prm_fails = prm_result["fail_rubrics"]      # PRM이 판정한 fail 루브릭 집합
        model_fails = step["model_fail_rubrics"]    # 모델이 나열한 fail 루브릭 집합

        # ── Fail rubrics 리워드: 모델 목록 vs PRM 목록 Jaccard 유사도 ───
        if not prm_fails and not model_fails:
            rubric_match = 1.0                      # 둘 다 fail 없음 → 완벽 일치
        else:
            inter = len(prm_fails & model_fails)
            union = len(prm_fails | model_fails)
            rubric_match = inter / union if union > 0 else 1.0
        rubric_rewards.append((rubric_match, step["rubric_pos"]))

        # ── Next action 리워드 ────────────────────────────────────
        action = step["action"]
        if action == "<|rethink|>":
            action_r = 1.0 if not prm_pass else 0.0
        elif action == "<|solve|>":
            action_r = 1.0 if prm_pass else 0.0
        else:  # <|end|>: boxed 형식 → 1, 정답 → 2
            has_boxed = r"\\boxed{" in solution_str or r"\boxed{" in solution_str
            if outcome == 1.0:
                action_r = 2.0
            elif has_boxed:
                action_r = 1.0
            else:
                action_r = 0.0

        action_rewards.append(action_r)

    total = sum(r for r, _ in rubric_rewards) + sum(action_rewards)

    _maybe_print_reward_debug(
        question=question,
        solution_str=solution_str,
        ground_truth=ground_truth,
        steps=steps,
        step_results=step_results,
        rubric_rewards=rubric_rewards,
        action_rewards=action_rewards,
        outcome=outcome,
        total=total,
    )

    return {"score": total, "rubric_rewards": rubric_rewards, "action_rewards": action_rewards}


# ── 터미널 디버그 출력 (학습 중 처음 N개 샘플) ────────────────────────────────
_DEBUG_PRINT_MAX = 3   # 전체 클러스터에서 처음 N개 샘플만 출력

import os as _os
import tempfile as _tempfile
import fcntl as _fcntl

_DEBUG_COUNTER_FILE = _os.path.join(_tempfile.gettempdir(), "grpo_reward_debug_count.txt")

def _acquire_debug_slot() -> bool:
    """파일 락으로 전체 워커 통틀어 _DEBUG_PRINT_MAX번까지만 True 반환."""
    try:
        with open(_DEBUG_COUNTER_FILE, "a+") as f:
            _fcntl.flock(f, _fcntl.LOCK_EX)
            f.seek(0)
            count = int(f.read().strip() or "0")
            if count >= _DEBUG_PRINT_MAX:
                _fcntl.flock(f, _fcntl.LOCK_UN)
                return False
            f.seek(0)
            f.truncate()
            f.write(str(count + 1))
            _fcntl.flock(f, _fcntl.LOCK_UN)
            return True
    except Exception:
        return False


def _extract_text_section(text: str, start: str, ends: list[str]) -> str:
    """start 마커 이후 ~ 첫 번째 end 마커 이전 텍스트 추출."""
    m = re.search(start, text, re.IGNORECASE)
    if not m:
        return "(없음)"
    rest = text[m.end():]
    for end in ends:
        em = re.search(end, rest, re.IGNORECASE)
        if em:
            return rest[:em.start()].strip()
    return rest.strip() or "(없음)"


def _maybe_print_reward_debug(
    question, solution_str, ground_truth,
    steps, step_results, rubric_rewards, action_rewards, outcome, total,
):
    if not _acquire_debug_slot():
        return

    W    = 80
    THICK = "=" * W
    THIN  = "-" * W

    def hdr(step_n, label):
        tag = f"[Step {step_n} / {label}]"
        print(f"\n{THIN}")
        print(tag)
        print(THIN)

    # ── 문제 헤더 ──────────────────────────────────────────────────────────────
    print(f"\n{THICK}")
    print(f"GRPO REWARD DEBUG  |  {'✓ CORRECT' if outcome else '✗ WRONG'}  |  total={total:.2f}")
    print(THICK)
    print(f"[문제]  {question}")
    print(f"[정답]  {ground_truth}")

    for i, (step, prm_result, (rub_r, _), act_r) in enumerate(
        zip(steps, step_results, rubric_rewards, action_rewards), 1
    ):
        full_text = step.get("full_text", step["reasoning_text"])

        # ── 1. 모델 full text ──────────────────────────────────────────────
        hdr(i, "모델 full text")
        print(full_text)

        # ── 2. 모델 추론 (풀이 부분만) ────────────────────────────────────
        hdr(i, "모델 추론 (풀이)")
        print(step["reasoning_text"])

        # ── 3. 모델 self-critique ──────────────────────────────────────────
        hdr(i, "모델 self-critique")
        fast_text = _extract_text_section(full_text, r"Fast critic\s*:",
                                          [r"Deep critic\s*:", r"Fail rubrics\s*:"])
        deep_text = _extract_text_section(full_text, r"Deep critic\s*:",
                                          [r"Fail rubrics\s*:", r"Next action\s*:"])
        model_fails = step["model_fail_rubrics"]
        pred_action = step["action"]

        print(f"  Fast critic:\n{_indent(fast_text)}")
        print(f"  Deep critic:\n{_indent(deep_text)}")
        print(f"  Pred fail rubrics : {sorted(model_fails) if model_fails else '(none)'}")
        print(f"  Pred next action  : {pred_action}")

        # ── 4. PRM 평가 결과 ──────────────────────────────────────────────
        hdr(i, "PRM 평가")
        s1 = prm_result.get("stage1_verdicts") or {}
        s2 = prm_result.get("stage2_verdicts")
        prm_fails = prm_result["fail_rubrics"]
        gold_action = "<|rethink|>" if not prm_result["pass"] else (
            "<|end|>" if i == len(steps) else "<|solve|>"
        )

        print("  PRM Stage1 (batch fast):")
        for rub, pred in s1.items():
            mark = "✗" if pred == "incorrect" else "✓"
            print(f"    {mark} {rub}: {pred}")

        if s2:
            print("  PRM Stage2 (deep, Stage1 fail 재평가):")
            for rub, pred in s2.items():
                mark = "✗" if pred == "incorrect" else "✓"
                print(f"    {mark} {rub}: {pred}")
        else:
            print("  PRM Stage2: (생략 — Stage1 전부 pass)")

        print(f"  Gold fail rubrics : {sorted(prm_fails) if prm_fails else '(none)'}")
        print(f"  Gold next action  : {gold_action}  (PRM pass={prm_result['pass']})")

        # ── 5. 리워드 ─────────────────────────────────────────────────────
        hdr(i, "리워드")
        print(f"  rubric jaccard : {rub_r:.2f}  (모델 fail rubrics vs PRM gold fail rubrics)")
        print(f"  action reward  : {act_r:.1f}   (pred={pred_action}  gold={gold_action})")
        print(f"  step 합계      : {rub_r + act_r:.2f}")

    # ── 전체 합산 ──────────────────────────────────────────────────────────────
    print(f"\n{THICK}")
    print(f"[합산]  rubric={sum(r for r,_ in rubric_rewards):.2f}"
          f"  action={sum(action_rewards):.2f}  total={total:.2f}")
    print(THICK)


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())
