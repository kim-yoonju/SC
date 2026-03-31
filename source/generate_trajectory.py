"""
generate_trajectory.py
PPO rollout용 trajectory 생성 로직 (state machine 기반)

State machine — 모델이 생성한 액션 토큰에 의해 전환:
  첫 스텝은 항상 SOLVE 상태에서 시작.

  TOKEN_SOLVE + boxed{}  → 종료 (정답)
  TOKEN_END              → 종료
  TOKEN_CORRECT:
    SOLVE        → CORRECT_GEN
    CORRECT_GEN  → CORRECT_PAT (patcher 호출)
    CORRECT_PAT  → 종료 (실패)
  TOKEN_SOLVE (no boxed) → SOLVE (계속 시도)

  액션 토큰이 명시적으로 생성되지 않은 경우:
    logit 기반으로 가장 확률 높은 액션 토큰을 선택해 위 규칙 적용.
"""

import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from utils import (
    ACTION_TOKENS,
    END_ANSWER,
    END_MAX,
    GENERATOR_CANDIDATE,
    GENERATOR_MAX_NEW_TOKENS,
    GENERATOR_TEMPERATURE,
    MAX_STEPS,
    PATCHER,
    PATCHER_CANDIDATE,
    PATCHER_TEMPERATURE,
    SYSTEM_CORRECT,
    SYSTEM_SOLVE,
    TOKEN_CORRECT,
    TOKEN_END,
    TOKEN_SOLVE,
    StepRecord,
    Trajectory,
    _correct_user,
    _gpt,
    _solve_user,
    build_chat_prompt,
    check_solved,
    has_boxed,
    R_PRM,
    save_trajectory,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 상태 상수
# ─────────────────────────────────────────────────────────────────────────────

SOLVE       = "solve"
CORRECT_GEN = "correct_gen"
CORRECT_PAT = "correct_pat"

# ─────────────────────────────────────────────────────────────────────────────
# Reward 계산
# ─────────────────────────────────────────────────────────────────────────────

def score_step(text: str, answer: str, is_last: bool = False) -> float:
    """스텝 reward R_final을 반환.

    R_final = R_PRM: REWARD 모델이 생성한 0~1 연속값.  state machine 분기 기준 (> 0.5).
    """
    r_prm = R_PRM(text, answer)

    logger.debug(f"  [reward] R_PRM={r_prm:.3f}  is_last={is_last}")
    return r_prm


# ─────────────────────────────────────────────────────────────────────────────
# State machine 전환 로직
# ─────────────────────────────────────────────────────────────────────────────

def _next_state(
    current_state: str,
    pred_action: str,
    text: str,
) -> Optional[str]:
    """eval 전용: 모델이 생성한 액션 토큰 기반 상태 전환. None이면 종료.

      TOKEN_SOLVE + boxed{}  → None (종료)
      TOKEN_END              → None (종료)
      TOKEN_CORRECT:
        SOLVE / CORRECT_GEN  → CORRECT_GEN / CORRECT_PAT (한 단계 깊게)
        CORRECT_PAT          → None (더 이상 패처 없음)
      TOKEN_SOLVE (no boxed) → SOLVE (계속 풀기)
    """
    if pred_action == TOKEN_SOLVE and has_boxed(text):
        return None
    if pred_action == TOKEN_END:
        return None
    if pred_action == TOKEN_CORRECT:
        if current_state == CORRECT_GEN:
            return CORRECT_PAT
        if current_state == CORRECT_PAT:
            return None
        return CORRECT_GEN
    return SOLVE  # TOKEN_SOLVE without boxed


def _next_state_by_reward(
    current_state: str,
    r_prm: float,
    text: str,
) -> Optional[str]:
    """train 전용: llm_reward 기반 ground truth 상태 전환. None이면 종료.

      solve,       boxed{}       → None (end, 정답)
      solve,       r>0.5         → solve
      solve,       r<=0.5        → correct_gen
      correct_gen, r>0.5         → solve
      correct_gen, r<=0.5        → correct_pat
      correct_pat, r>0.5         → solve
      correct_pat, r<=0.5        → None (end, patcher_wrong)
      end,         r>0.5         → None (end, 성공)
      end,         r<=0.5        → correct_gen
    """
    if current_state == SOLVE and has_boxed(text):
        return None
    if r_prm > 0.5:
        if current_state in (SOLVE, CORRECT_GEN, CORRECT_PAT):
            return SOLVE
        return None  # end state, reward OK → 종료
    else:
        if current_state == SOLVE:
            return CORRECT_GEN
        if current_state == CORRECT_GEN:
            return CORRECT_PAT
        if current_state == CORRECT_PAT:
            return None  # patcher_wrong → 종료
        return CORRECT_GEN  # end state, reward 낮음 → correct_gen


def _gt_action_token(next_state: Optional[str]) -> str:
    """ground truth 다음 상태를 액션 토큰으로 변환."""
    if next_state is None:
        return TOKEN_END
    if next_state == SOLVE:
        return TOKEN_SOLVE
    return TOKEN_CORRECT  # CORRECT_GEN or CORRECT_PAT


# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트 빌더
# ─────────────────────────────────────────────────────────────────────────────

def _build_gen_prompt(tokenizer, state: str, problem: str, history: List[str]) -> str:
    """상태에 따라 generator용 chat prompt를 생성."""
    if state == SOLVE:
        return build_chat_prompt(tokenizer, SYSTEM_SOLVE, _solve_user(problem, history))
    else:  # correct_gen, end
        reason = history[-1] if history else ""
        return build_chat_prompt(tokenizer, SYSTEM_CORRECT, _correct_user(problem, history, reason))


def _call_patcher(problem: str, history: List[str], temperature: float = None) -> str:
    """PATCHER API를 호출해 한 스텝 풀이를 반환."""
    reason = history[-1] if history else ""
    messages = [
        {"role": "system", "content": SYSTEM_CORRECT},
        {"role": "user",   "content": _correct_user(problem, history, reason)},
    ]
    logger.info(f"  [patcher] {PATCHER} 호출 중  history_len={len(history)}  temp={temperature}")
    try:
        result = _gpt(PATCHER, messages, max_completion_tokens=GENERATOR_MAX_NEW_TOKENS, temperature=temperature)
        logger.info(f"  [patcher] 응답 {len(result)}자  preview={result[:80].replace(chr(10),' ')!r}")
        return result
    except Exception as e:
        logger.warning(f"  [patcher] 호출 실패: {e}")
        return ""


def _run_generator_rollouts(
    model,
    tokenizer,
    problems: List[dict],
    initial_histories: List[List[str]],
    action_token_ids: set,
    _max: int,
) -> List[bool]:
    """initial_histories에서 SOLVE로 시작해 generator만으로 풀고 정답 여부를 반환.

    평가 전용 - log_probs 계산 없음, patcher 호출 없음.
    CORRECT_PAT에 도달하면 patcher 없이 종료 (실패 처리).
    """
    n         = len(problems)
    answers   = [p["answer"] for p in problems]
    histories = [h[:] for h in initial_histories]
    states    = [SOLVE] * n
    solved    = [False] * n
    last_boxed: Dict[int, str] = {}
    active    = list(range(n))

    model.eval()
    for _ in range(MAX_STEPS):
        if not active:
            break

        # patcher 없이 종료
        pat_stuck = [i for i in active if states[i] == CORRECT_PAT]
        for i in pat_stuck:
            active.remove(i)

        gen_active = [i for i in active if states[i] != CORRECT_PAT]
        if not gen_active:
            break

        prompts = [_build_gen_prompt(tokenizer, states[i], problems[i]["problem"], histories[i]) for i in gen_active]
        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        tokenizer.padding_side = orig_side
        n_in = enc["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=_max,
                temperature=GENERATOR_TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=list(action_token_ids) + [tokenizer.eos_token_id],
            )

        resp_all = out_ids[:, n_in:]
        newly_done = []
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        for j, i in enumerate(gen_active):
            resp = resp_all[j]
            trim = resp.shape[0]
            pred_action = TOKEN_SOLVE
            for pos, tid in enumerate(resp.tolist()):
                if tid in action_token_ids:
                    trim = pos + 1
                    pred_action = tokenizer.decode([tid])
                    break
                if tid == tokenizer.pad_token_id:
                    trim = pos
                    break
                if tid == im_end_id:
                    trim = pos
                    pred_action = TOKEN_END
                    break

            text = tokenizer.decode(resp[:trim], skip_special_tokens=True)
            for tok in ACTION_TOKENS:
                text = text.replace(tok, "")
            text = text.strip()

            if has_boxed(text):
                last_boxed[i] = text
            histories[i].append(text)

            next_s = _next_state(states[i], pred_action, text)

            if next_s is None or next_s == CORRECT_PAT:
                last_text = last_boxed.get(i, "")
                solved[i] = check_solved(last_text, answers[i]) if last_text else False
                newly_done.append(i)
            else:
                states[i] = next_s

        for i in newly_done:
            active.remove(i)

    return solved


def _select_best_patcher_candidate(
    model,
    tokenizer,
    problem: str,
    answer: str,
    history: List[str],
    action_token_ids: set,
    _max: int,
) -> str:
    """patcher_candidate 수만큼 후보 생성 후, 각 후보로부터 generator rollout의
    정답 도달률이 가장 높은 후보를 반환."""

    # 1. patcher candidates 병렬 생성 (낮은 temperature)
    with ThreadPoolExecutor(max_workers=PATCHER_CANDIDATE) as ex:
        candidates = list(ex.map(
            lambda _: _call_patcher(problem, history, temperature=PATCHER_TEMPERATURE),
            range(PATCHER_CANDIDATE),
        ))
    candidates = [c for c in candidates if c]
    if not candidates:
        return ""

    # 2. 전체 (patcher_candidate × generator_candidate) 조합 배치 롤아웃
    combo_problems:  List[dict]      = []
    combo_histories: List[List[str]] = []
    combo_cand_idx:  List[int]       = []

    for ci, cand in enumerate(candidates):
        extended = history + [cand]
        for _ in range(GENERATOR_CANDIDATE):
            combo_problems.append({"problem": problem, "answer": answer})
            combo_histories.append(extended)
            combo_cand_idx.append(ci)

    solved_list = _run_generator_rollouts(model, tokenizer, combo_problems, combo_histories, action_token_ids, _max)

    # 3. 성공률 가장 높은 candidate 선택
    counts = [0] * len(candidates)
    for ci, s in zip(combo_cand_idx, solved_list):
        if s:
            counts[ci] += 1
    best_idx = max(range(len(candidates)), key=lambda k: counts[k])

    logger.info(
        f"  [patcher best-of-{len(candidates)}] "
        + ", ".join(f"cand{k}={counts[k]}/{GENERATOR_CANDIDATE}" for k in range(len(candidates)))
        + f"  → cand{best_idx} 선택"
    )
    return candidates[best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 메인 trajectory 생성
# ─────────────────────────────────────────────────────────────────────────────

def solve_problems_batch(
    model,
    tokenizer,
    problems: List[dict],
    rollout_path: str = None,
    max_new_tokens: int = None,
) -> List[Trajectory]:
    """problems 배치에 대해 state machine 기반으로 Trajectory를 생성.

    generator 스텝: use_patcher=False — PPO 학습 대상
    patcher  스텝: use_patcher=True  — PPO 학습 제외
    """
    _max = max_new_tokens or GENERATOR_MAX_NEW_TOKENS
    action_token_ids = set(
        tid for tid in tokenizer.convert_tokens_to_ids(ACTION_TOKENS)
        if tid != tokenizer.unk_token_id
    )
    # logit fallback용: token_id → token string 매핑
    action_id_to_token = {
        tid: tok
        for tok, tid in zip(ACTION_TOKENS, tokenizer.convert_tokens_to_ids(ACTION_TOKENS))
        if tid != tokenizer.unk_token_id
    }

    trajs = [
        Trajectory(
            problem_id=p.get("problem_id", str(i)),
            problem=p.get("problem", ""),
            answer=p.get("answer", ""),
        )
        for i, p in enumerate(problems)
    ]
    histories:        List[List[str]] = [[] for _ in problems]
    states:           List[str]       = [SOLVE] * len(problems)
    last_boxed_texts: Dict[int, str]  = {}   # 문제별 마지막 boxed{} 포함 스텝 텍스트
    step_counts:      List[int]       = [0] * len(problems)  # 트래젝토리별 step_idx 카운터
    active = list(range(len(problems)))

    logger.info(f"[batch] 시작  n={len(problems)}  rollout={rollout_path}")

    _iter       = 0
    t_batch_start = time.time()

    def _next_label(s: str | None) -> str:
        return {SOLVE: "solve", CORRECT_GEN: "correct", CORRECT_PAT: "patcher"}.get(s, "done") if s else "done"

    def _state_label(s: str) -> str:
        return {SOLVE: "SOLVE", CORRECT_GEN: "CORRECT", CORRECT_PAT: "PATCHER"}.get(s, s)

    def _update_boxed(i: int, text: str):
        if has_boxed(text):
            last_boxed_texts[i] = text

    def _terminate(i: int, reason: str = "done"):
        last_text = last_boxed_texts.get(i, "")
        trajs[i].have_boxed = bool(last_text)
        trajs[i].is_answer  = check_solved(last_text, trajs[i].answer) if last_text else False
        trajs[i].end_state  = END_MAX if reason == "timeout" else END_ANSWER
        status = "ANSWER" if trajs[i].is_answer else ("BOXED" if trajs[i].have_boxed else "FAIL")
        logger.info(
            f"[P{trajs[i].problem_id:>6}] DONE"
            f"  status={status}  total_steps={len(trajs[i].steps)}  reason={reason}"
        )
        if rollout_path:
            save_trajectory(trajs[i], rollout_path)
            logger.info(f"[P{trajs[i].problem_id:>6}] SAVED → {rollout_path}")

    model.eval()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    for step_idx in range(MAX_STEPS):
        if not active:
            break

        newly_done: List[int] = []

        # ── Generator batch (solve / correct_gen) ────────────────────────
        gen_active = [i for i in active if states[i] != CORRECT_PAT]
        if gen_active:
            # ① generate 시작 전: 각 문제가 몇 번째 스텝을 요청하는지 즉시 기록
            for i in gen_active:
                logger.info(
                    f"[P{trajs[i].problem_id:>6}] step={step_idx:02d}  state={_state_label(states[i])}  history={len(histories[i])}  → generating"
                )

            prompts = [
                _build_gen_prompt(tokenizer, states[i], trajs[i].problem, histories[i])
                for i in gen_active
            ]

            orig_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            tokenizer.padding_side = orig_side
            n_in = enc["input_ids"].shape[1]

            # ② 배치 GPU 생성 (병렬) — 각 시퀀스가 액션 토큰 생성 시 독립적으로 중단
            t0 = time.time()
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=_max,
                    temperature=GENERATOR_TEMPERATURE,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=list(action_token_ids) + [tokenizer.eos_token_id],
                )
            logger.info(
                f"[batch] [I{_iter:03d}] GPU_GEN"
                f"  batch={len(gen_active)}  elapsed={time.time()-t0:.2f}s"
            )

            # ③ log_probs: response 위치만 계산
            t1 = time.time()
            with torch.no_grad():
                full_lp = F.log_softmax(
                    model(out_ids[:, :-1]).logits[:, n_in - 1:, :], dim=-1
                )
            resp_all = out_ids[:, n_in:]
            logger.info(f"[batch] [I{_iter:03d}] log_probs  n={len(gen_active)}  elapsed={time.time()-t1:.2f}s")

            # ④ decode (CPU, 순차)
            decoded = []
            for j, i in enumerate(gen_active):
                resp = resp_all[j]
                trim = resp.shape[0]
                pred_action = None  # 명시적 액션 미발견 → 나중에 fallback
                for pos, tid in enumerate(resp.tolist()):
                    if tid in action_token_ids:
                        trim = pos + 1
                        pred_action = tokenizer.decode([tid])
                        break
                    if tid == tokenizer.pad_token_id:
                        trim = pos
                        break
                    if tid == im_end_id:
                        trim = pos
                        pred_action = TOKEN_END
                        break

                # 액션 토큰이 명시적으로 생성되지 않은 경우: logit 기반 fallback
                if pred_action is None:
                    last_pos = min(trim, full_lp.shape[1]) - 1
                    if last_pos >= 0 and action_id_to_token:
                        act_ids = list(action_id_to_token.keys())
                        best_id = act_ids[full_lp[j, last_pos, act_ids].argmax().item()]
                        pred_action = action_id_to_token[best_id]
                        logger.info(
                            f"[P{trajs[i].problem_id:>6}] [S{step_idx:02d}] action fallback → {pred_action}"
                        )
                    else:
                        pred_action = TOKEN_SOLVE

                resp_trim = resp[:trim]
                lp = full_lp[j, :trim].gather(1, resp_trim.unsqueeze(1)).squeeze(1).cpu()

                text = tokenizer.decode(resp_trim, skip_special_tokens=True)
                for tok in ACTION_TOKENS:
                    text = text.replace(tok, "")
                decoded.append((j, i, resp_trim, lp, text.strip(), pred_action))

            # ⑤ R_PRM 병렬 호출 → 완료된 순서대로 즉시 처리 + 로그
            t2 = time.time()
            with ThreadPoolExecutor(max_workers=len(decoded)) as ex:
                future_to_d = {
                    ex.submit(
                        score_step, d[4], trajs[d[1]].answer, is_last=(d[5] == TOKEN_END)
                    ): d
                    for d in decoded
                }
                for fut in as_completed(future_to_d):
                    j, i, resp_trim, lp, text, pred_action = future_to_d[fut]
                    r_prm = fut.result()
                    # ground truth 상태 전환 (reward 기반) — 실제 학습 경로
                    gt_next_s = _next_state_by_reward(states[i], r_prm, text)
                    gt_action = _gt_action_token(gt_next_s)
                    format_reward = 0.1 if (pred_action == TOKEN_END and has_boxed(text)) else 0.0
                    R_final = r_prm + format_reward

                    # patcher_wrong: correct_pat 에서 reward 낮으면 실패 종료
                    if states[i] == CORRECT_PAT and r_prm <= 0.5:
                        trajs[i].patcher_wrong = True

                    _action_name = pred_action.strip("<|>")
                    _gt_name = gt_action.strip("<|>")
                    _is_ans = check_solved(text, trajs[i].answer)
                    logger.info(
                        f"[P{trajs[i].problem_id:>6}] step={step_idx:02d}  state={_state_label(states[i])}"
                        f"  pred={_action_name}  gt={_gt_name}"
                        f"  R_PRM={r_prm:.3f}  format={format_reward:.1f}  R_final={R_final:.3f}"
                        f"  tokens={resp_trim.shape[0]}  next={_next_label(gt_next_s)}"
                        f"  is_answer={_is_ans}"
                    )

                    trajs[i].steps.append(StepRecord(
                        step_idx=step_counts[i],
                        state=states[i],
                        action=pred_action.strip("<|>"),
                        text=text,
                        reward=R_final,
                        llm_reward=r_prm,
                        format_reward=format_reward,
                        predicted_next_action=pred_action,
                        ground_truth_next_action=gt_action,
                        input_ids=enc["input_ids"][j:j+1].cpu(),
                        response_ids=resp_trim.unsqueeze(0).cpu(),
                        log_probs_old=lp,
                        use_patcher=False,
                    ))
                    step_counts[i] += 1
                    histories[i].append(text)
                    _update_boxed(i, text)

                    if gt_next_s is None:
                        _terminate(i, reason="generator")
                        newly_done.append(i)
                    else:
                        states[i] = gt_next_s

        # ── Patcher calls (correct_pat) ───────────────────────────────────
        pat_active = [i for i in active if states[i] == CORRECT_PAT and i not in newly_done]
        if pat_active:
            for i in pat_active:
                logger.info(
                    f"[P{trajs[i].problem_id:>6}] step={step_idx:02d}  PATCHER_SUBMIT  history={len(histories[i])}"
                )
            t_pat = time.time()
            pat_texts = [
                _select_best_patcher_candidate(
                    model, tokenizer,
                    trajs[i].problem, trajs[i].answer, histories[i],
                    action_token_ids, _max,
                )
                for i in pat_active
            ]
            logger.info(
                f"[batch] [I{_iter:03d}] PATCHER  n={len(pat_active)}  elapsed={time.time()-t_pat:.2f}s"
            )

            # R_PRM + patcher log_probs 병렬 처리 → 완료 순으로 즉시 기록
            def _score_and_logprobs(i, text):
                r_prm = score_step(text, trajs[i].answer, is_last=False)
                prompt   = _build_gen_prompt(tokenizer, CORRECT_GEN, trajs[i].problem, histories[i])
                inp_ids  = tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"]
                resp_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)["input_ids"]
                n_in_p   = inp_ids.shape[1]
                n_resp_p = resp_ids.shape[1]
                with torch.no_grad():
                    logits_p = model(torch.cat([inp_ids, resp_ids], dim=1)[:, :-1]).logits
                    lp_p = (
                        F.log_softmax(logits_p, dim=-1)[0, n_in_p - 1: n_in_p - 1 + n_resp_p]
                        .gather(1, resp_ids.squeeze(0).unsqueeze(1))
                        .squeeze(1).cpu()
                    )
                return r_prm, inp_ids.cpu(), resp_ids.cpu(), lp_p

            with ThreadPoolExecutor(max_workers=len(pat_active)) as ex:
                future_to_pi = {
                    ex.submit(_score_and_logprobs, i, text): (i, text)
                    for i, text in zip(pat_active, pat_texts)
                }
                for fut in as_completed(future_to_pi):
                    i, text = future_to_pi[fut]
                    r_prm, inp_ids, resp_ids, lp_p = fut.result()
                    gt_next_s = _next_state_by_reward(CORRECT_PAT, r_prm, text)
                    gt_action = _gt_action_token(gt_next_s)
                    R_final = r_prm

                    if r_prm <= 0.5:
                        trajs[i].patcher_wrong = True

                    _gt_name = gt_action.strip("<|>")
                    _is_ans_pat = check_solved(text, trajs[i].answer)
                    logger.info(
                        f"[P{trajs[i].problem_id:>6}] step={step_idx:02d}  PATCHER_ARRIVED"
                        f"  gt={_gt_name}  patcher_wrong={trajs[i].patcher_wrong}"
                        f"  R_PRM={r_prm:.3f}  R_final={R_final:.3f}"
                        f"  tokens={resp_ids.shape[1]}  next={_next_label(gt_next_s)}"
                        f"  is_answer={_is_ans_pat}"
                    )

                    trajs[i].steps.append(StepRecord(
                        step_idx=step_counts[i],
                        state=CORRECT_PAT,
                        action="correct",
                        text=text,
                        reward=R_final,
                        llm_reward=r_prm,
                        format_reward=0.0,
                        predicted_next_action=TOKEN_CORRECT,
                        ground_truth_next_action=gt_action,
                        input_ids=inp_ids,
                        response_ids=resp_ids,
                        log_probs_old=lp_p,
                        use_patcher=True,
                    ))
                    step_counts[i] += 1
                    histories[i].append(text)
                    _update_boxed(i, text)

                    if gt_next_s is None:
                        _terminate(i, reason="patcher")
                        newly_done.append(i)
                    else:
                        states[i] = gt_next_s

        # 스텝 종료 상태 요약
        done_count  = len(newly_done)
        gen_count   = sum(1 for i in active if i not in newly_done and states[i] != CORRECT_PAT)
        patch_count = sum(1 for i in active if i not in newly_done and states[i] == CORRECT_PAT)
        logger.info(
            f"[batch] [I{_iter:03d}] done={done_count}"
            f"  gen={gen_count}  api=0  patch={patch_count}"
            f"  elapsed={time.time()-t_batch_start:.1f}s"
        )
        _iter += 1

        for i in newly_done:
            active.remove(i)

    # MAX_STEPS 소진 후에도 active에 남은 항목 처리
    if active:
        logger.info(f"[batch] [I{_iter:03d}] TIMEOUT  미완료={len(active)}개")
        for i in active:
            _terminate(i, reason="timeout")

    logger.info(
        f"[batch] 완료  total={len(trajs)}"
        f"  correct={sum(1 for t in trajs if t.is_answer)}"
        f"  boxed={sum(1 for t in trajs if t.have_boxed)}"
        f"  elapsed={time.time()-t_batch_start:.1f}s"
    )
    return trajs


# ─────────────────────────────────────────────────────────────────────────────
# 샘플 trajectory 출력 (디버깅용)
# ─────────────────────────────────────────────────────────────────────────────

def print_sample_trajectories(
    model,
    tokenizer,
    problems: List[dict],
    n: int = 3,
    max_new_tokens: int = None,
) -> None:
    """problems에서 n개를 골라 trajectory를 생성하고 콘솔에 출력.

    Args:
        model: 생성 모델
        tokenizer: 토크나이저
        problems: 전체 문제 리스트 (각 항목에 'problem', 'answer', 선택적으로 'problem_id')
        n: 샘플 수 (기본 3)
        max_new_tokens: 스텝당 최대 생성 토큰 수
    """
    sample = problems[:n]
    print(f"\n{'='*70}")
    print(f"[sample] {len(sample)}개 trajectory 생성 중...")
    print(f"{'='*70}\n")

    trajs = solve_problems_batch(model, tokenizer, sample, rollout_path=None, max_new_tokens=max_new_tokens)

    for traj in trajs:
        print(f"{'─'*70}")
        print(f"problem_id : {traj.problem_id}")
        print(f"problem    : {traj.problem[:120].strip()}")
        print(f"answer     : {traj.answer}")
        print(f"is_answer  : {traj.is_answer}  |  have_boxed: {traj.have_boxed}  |  steps: {len(traj.steps)}")
        for s in traj.steps:
            print(f"\n  [step {s.step_idx}]  action={s.action}  reward={s.reward:.3f}"
                  f"  (llm={s.llm_reward:.3f}, format={s.format_reward:.1f})"
                  f"  use_patcher={s.use_patcher}")
            preview = s.text[:200].replace("\n", " ")
            print(f"  text: {preview!r}")
        print()


def _load_done_ids(*jsonl_paths: str) -> set:
    """이미 처리된 problem_id 집합을 반환.

    여러 jsonl 파일을 받아 problem_id를 수집한다.
    파일이 없거나 비어있으면 조용히 건너뛴다.
    """
    done = set()
    for path in jsonl_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        pid = str(d.get("problem_id", ""))
                        if pid:
                            done.add(pid)
                    except json.JSONDecodeError:
                        pass
    return done


def _save_traj(traj: "Trajectory", path: str):
    """Trajectory를 SFT 학습 포맷으로 JSONL에 append 저장.

    make_sft.py / train_sft.py 와 호환되는 필드명을 사용한다.
    input_ids / log_probs 등 학습 전용 텐서는 포함하지 않는다.
    """
    record = {
        "problem_id": traj.problem_id,
        "problem":    traj.problem,
        "answer":     traj.answer,
        "have_boxed": traj.have_boxed,
        "is_answer":  traj.is_answer,
        "end_state":  traj.end_state,
        "steps": [
            {
                "step_idx":                  s.step_idx,
                "state":                     s.state,
                "action":                    s.action,
                "text":                      s.text,
                "reward":                    s.reward,
                "llm_reward":                s.llm_reward,
                "format_reward":             s.format_reward,
                "predicted_next_action":     s.predicted_next_action,
                "ground_truth_next_action":  s.ground_truth_next_action,
                "use_patcher":               s.use_patcher,
            }
            for s in traj.steps
        ],
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    """SFT 데이터 생성 진입점.

    config의 ppo.rollout_gpus 에 지정된 GPU를 Ray worker로 띄워
    ppo.problems_per_iter 수만큼 문제를 병렬 생성한다.

    sft_ppo.jsonl 에 이미 있는 problem_id는 건너뛰고
    결과는 output/sft_data/{timestamp}/rollouts_worker_N.jsonl 에 저장한다.

    실행 예:
        cd /mnt/yoonju/SC
        python source/generate_trajectory.py
        python source/generate_trajectory.py --checkpoint checkpoints/ppo/iter_0003
    """
    import argparse
    import sys
    from datetime import datetime
    from pathlib import Path

    import ray

    from utils import (
        CONF,
        DATASET_PATH,
        load_generator,
        load_problems,
        create_rollout_file,
    )

    parser = argparse.ArgumentParser(description="SFT 데이터 생성")
    parser.add_argument("--checkpoint", type=str, default='',
                        help="모델 체크포인트 경로 (default: config SFT checkpoint)")
    parser.add_argument("--dataset",    type=str, default=None,
                        help="문제 parquet 경로 (default: config deepmath_16k)")
    parser.add_argument("--skip_file",  type=str, default=None,
                        help="건너뛸 problem_id jsonl (default: datasets/sft_ppo.jsonl)")
    parser.add_argument("--wrong_only", type=str, default=None,
                        help="이 jsonl의 id 목록에 해당하는 문제만 처리 (default: config generate_trajectory.base_wrong_file)")
    args = parser.parse_args()

    _root = Path(__file__).resolve().parent.parent

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # ── config에서 GPU / 배치 크기 읽기 ──────────────────────────────────
    _gt_cfg        = CONF.get("generate_trajectory", {})
    _ppo_cfg       = CONF["ppo"]
    rollout_gpus   = _gt_cfg.get("rollout_gpus") or _ppo_cfg["rollout_gpus"]
    problems_per_iter = _ppo_cfg["problems_per_iter"] # ex) 64
    num_workers    = len(rollout_gpus)
    chunk          = problems_per_iter // num_workers  # 워커 1개당 문제 수

    # ── 출력 경로 설정 ────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _root / "output" / "sft_trajectory" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / "run.log")

    rollout_paths   = [str(out_dir / f"rollouts_worker_{i}.jsonl") for i in range(num_workers)]
    worker_log_paths = [str(out_dir / f"log_worker_{i}.log")       for i in range(num_workers)]

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    # ── 이미 완료된 problem_id 수집 ───────────────────────────────────────
    default_skip = str(_root / "datasets" / "sft_ppo.jsonl")
    skip_file    = args.skip_file or (default_skip if os.path.exists(default_skip) else None)
    # 워커 출력 파일들도 포함 (재실행 시 이어서 처리)
    done_ids = _load_done_ids(skip_file, *rollout_paths)
    logger.info(f"건너뛸 problem_id: {len(done_ids)}개  (from: {skip_file or '없음'})")

    # ── 문제 로드 및 필터링 ────────────────────────────────────────────────
    dataset_path = args.dataset or DATASET_PATH
    all_problems = load_problems(dataset_path)

    # wrong_only 필터: base_wrong_file의 id 목록에 있는 문제만 처리
    wrong_only_path = args.wrong_only or _gt_cfg.get("base_wrong_file")
    if wrong_only_path and os.path.exists(wrong_only_path):
        wrong_ids = set()
        with open(wrong_only_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    _obj = json.loads(_line)
                    _id = _obj.get("id") or _obj.get("problem_id")
                    if _id is not None:
                        wrong_ids.add(str(_id))
        all_problems = [p for p in all_problems if str(p["problem_id"]) in wrong_ids]
        logger.info(f"wrong_only 필터 적용: {wrong_only_path} → {len(wrong_ids)}개 id → {len(all_problems)}개 문제")
    elif wrong_only_path:
        logger.warning(f"wrong_only 파일을 찾을 수 없습니다: {wrong_only_path}")

    todo = [p for p in all_problems if str(p["problem_id"]) not in done_ids]
    logger.info(f"전체 {len(all_problems)}개 문제 → 미완료 {len(todo)}개")

    if not todo:
        logger.info("처리할 문제가 없습니다.")
        return

    logger.info(
        f"GPU: {rollout_gpus}  workers: {num_workers}  "
        f"problems_per_iter: {problems_per_iter}  chunk/worker: {chunk}"
    )

    # ── Ray 워커 생성 ─────────────────────────────────────────────────────
    # GPU 지정: runtime_env env_vars는 PyTorch CUDA 초기화 타이밍에 반영이 불안정하므로
    # worker __init__ 안에서 os.environ으로 직접 설정한 뒤 모델을 로드한다.
    @ray.remote
    class _SFTWorker:
        def __init__(self, worker_id: int, rollout_path: str, log_path: str, checkpoint: str, gpu_id: int):
            import logging
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            self.worker_id    = worker_id
            self.rollout_path = rollout_path

            _wlog = logging.getLogger()
            _wlog.setLevel(logging.INFO)
            for h in _wlog.handlers[:]:
                _wlog.removeHandler(h)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(f"%(asctime)s [W{worker_id}] %(message)s"))
            _wlog.addHandler(fh)

            self.model, self.tokenizer = load_generator(
                device_map={"": "cuda:0"}, model_path=checkpoint or None
            )
            create_rollout_file(rollout_path)
            logging.info(f"준비 완료  GPU={gpu_id}  rollout → {rollout_path}")

        def run(self, problems: list) -> dict:
            """problems 배치를 처리하고 저장 후 간단한 통계를 반환."""
            trajs = solve_problems_batch(
                self.model, self.tokenizer, problems, rollout_path=None
            )
            for traj in trajs:
                _save_traj(traj, self.rollout_path)
            return {
                "n":       len(trajs),
                "correct": sum(1 for t in trajs if t.is_answer),
                "boxed":   sum(1 for t in trajs if t.have_boxed),
            }

    ray.init(include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)

    workers = [
        _SFTWorker.remote(
            worker_id=i,
            rollout_path=rollout_paths[i],
            log_path=worker_log_paths[i],
            checkpoint=args.checkpoint,
            gpu_id=rollout_gpus[i],
        )
        for i in range(num_workers)
    ]

    # ── 라운드 단위 병렬 처리 ─────────────────────────────────────────────
    # 한 라운드 = problems_per_iter 개 문제를 num_workers 개 워커가 나눠 처리
    total_saved = total_correct = total_boxed = 0
    cursor      = 0
    n_todo      = len(todo)
    round_idx   = 0

    while cursor < n_todo:
        batch = todo[cursor: cursor + problems_per_iter]
        cursor += problems_per_iter
        round_idx += 1

        # 워커마다 chunk 개씩 분배 (마지막 라운드는 짧을 수 있음)
        futures = [
            workers[i].run.remote(batch[i * chunk: (i + 1) * chunk])
            for i in range(num_workers)
            if batch[i * chunk: (i + 1) * chunk]   # 빈 슬라이스 제외
        ]
        results = ray.get(futures)

        for r in results:
            total_saved   += r["n"]
            total_correct += r["correct"]
            total_boxed   += r["boxed"]

        logger.info(
            f"[round {round_idx}]  이번={sum(r['n'] for r in results)}개  "
            f"correct={sum(r['correct'] for r in results)}  "
            f"boxed={sum(r['boxed'] for r in results)}  "
            f"누적={total_saved}/{n_todo}"
        )

    logger.info(
        f"전체 완료: {total_saved}개  correct={total_correct}  boxed={total_boxed}"
    )
    for i, p in enumerate(rollout_paths):
        logger.info(f"  worker {i} → {p}")

    ray.shutdown()


if __name__ == "__main__":
    main()
