"""
experiment_patcher_local.py

로컬 DeepSeek-R1-Distill-Llama-70B 모델(vLLM)로 step 정오 예측 정확도를 측정하는 실험 스크립트.

평가 방식:
  1단계(추론 생성): 프롬프트 → 모델이 추론 텍스트 생성
  2단계(verdict 확률): 텍스트 파싱으로 correct/incorrect 판정.
                       파싱 실패 시 "Verdict: " prefix를 붙여 vLLM logprob으로 fallback.

Usage:
    python source/local_experiment.py
    python source/local_experiment.py --model_path casperhansen/deepseek-r1-distill-llama-70b-awq
    python source/local_experiment.py --start 0  --end 25   # train
    python source/local_experiment.py --start 25 --end 50   # test
"""

import argparse
import json
import logging
import math
import re
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import _call_llm, CONF, set_call_role, set_run_log, set_problem_context

_DEFAULT_PRM_MODEL   = CONF.get("PRM", {}).get("model_id_checklist") or CONF.get("PRM", {}).get("model_id_batch")
_DEFAULT_RUBRIC_FILE = CONF.get("PRM", {}).get("rubric_file")

# ─────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────────────────────────────────────
# 클래스별 슬라이스 범위: train=0~25, test=25~50
SAMPLE_START = 0   # 시작 인덱스 (포함)
SAMPLE_END   = 32   # 끝 인덱스 (미포함)

ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "output" / "sft_trajectory" / "traj_all_base_400.jsonl"
CONFIG_PATH = ROOT / "configs" / "config.yaml"

# Verdict 토큰 설정 — 루브릭 프롬프트가 모델에게 출력하도록 지시하는 단어
VERDICT_PASS_WORD = "correct"    # 스텝이 올바를 때 모델이 출력하는 단어
VERDICT_FAIL_WORD = "incorrect"  # 스텝이 틀렸을 때 모델이 출력하는 단어

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 루브릭 파일
# ─────────────────────────────────────────────────────────────────────────────

# 실험할 루브릭 버전 리스트 — 여기에 버전 번호를 추가하면 자동으로 모두 실험
DEEP_RUBRIC_VERSIONS = ["6.5"]  # 예: ["4.0", "4.1", "4.2"]
DEEP_RUBRIC_FILES = [ROOT / "prompts" / f"deep_rubric_v{v}.json" for v in DEEP_RUBRIC_VERSIONS]

# 사용할 루브릭 번호 리스트 (1-indexed). None이면 전체 사용.
# 예: [1, 2, 3] → 1~3번 루브릭만 사용 / None → 전체
DEEP_RUBRIC_INDICES = [11] # range(1,10) # [10,11]



def load_deep_rubrics(path: Path | str | None = None) -> list[dict]:
    """JSON 파일에서 루브릭 목록 로드. 각 항목에 name, criterion, system_prompt 포함."""
    path = Path(path) if path else DEEP_RUBRIC_FILES[0]
    if not path.exists():
        logger.error(f"루브릭 파일 없음: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        rubrics = json.load(f)
    logger.info(f"루브릭 로드: {path.name}  ({len(rubrics)}개)")
    return rubrics


_INDIVIDUAL_OUTPUT_FORMAT = """OUTPUT FORMAT:
Up to 3 lines: which check fired, what you verified, what was wrong or correct.
Silent on non-applicable checks. End with exactly one of:
Verdict: correct
Verdict: incorrect"""

def build_system_prompt(rubric: dict, cot: bool = False) -> str:
    """루브릭 dict의 system_prompt 필드를 반환. OUTPUT FORMAT 섹션을 3줄 형식으로 교체."""
    prompt = rubric["system_prompt"]
    if "OUTPUT FORMAT:" in prompt:
        prompt = prompt[:prompt.index("OUTPUT FORMAT:")].rstrip() + "\n\n" + _INDIVIDUAL_OUTPUT_FORMAT
    else:
        prompt = prompt + "\n\n" + _INDIVIDUAL_OUTPUT_FORMAT
    return prompt

_FIRST_STEP_MARKER = "Since the Now Step is the first step"


def build_user_message(question: str, previous_steps: str, now_step: str) -> str:
    has_prev = bool(previous_steps and _FIRST_STEP_MARKER not in previous_steps)
    parts = [f"Problem:\n{question}"]
    if has_prev:
        parts.append(f"Previous steps (confirmed correct):\n{previous_steps}")
    parts.append(f"Current step to evaluate:\n{now_step}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 모델 (API)
# ─────────────────────────────────────────────────────────────────────────────

import math as _math


def _prob_from_logprobs(token_logprobs: list, idx: int = 0) -> tuple[float | None, float | None]:
    """token_logprobs[idx]의 top_logprobs에서 correct/incorrect 확률을 정규화해 반환."""
    if not token_logprobs or idx >= len(token_logprobs):
        return None, None
    lp_c = lp_i = None
    for alt in (token_logprobs[idx].top_logprobs or []):
        t = (alt.token or "").strip().lower()
        if t == "correct":
            lp_c = alt.logprob
        elif t == "incorrect":
            lp_i = alt.logprob
    if lp_c is None and lp_i is None:
        return None, None
    p_c = _math.exp(lp_c) if lp_c is not None else 0.0
    p_i = _math.exp(lp_i) if lp_i is not None else 0.0
    total = p_c + p_i
    return (p_c / total, p_i / total) if total > 0 else (None, None)


def _prob_from_verdict_token(token_logprobs: list) -> tuple[float | None, float | None]:
    """전체 응답 logprobs에서 'Verdict' 직후 토큰의 correct/incorrect 정규화 확률 추출."""
    for i, tok in enumerate(token_logprobs):
        if "verdict" in (tok.token or "").lower():
            for j in range(i + 1, min(i + 5, len(token_logprobs))):
                if (token_logprobs[j].token or "").strip() not in (":", ""):
                    return _prob_from_logprobs(token_logprobs, j)
    return None, None


def _forced_verdict(model_name: str, messages: list, truncated_resp: str) -> dict:
    """잘린 응답에 '\\nVerdict: ' 붙여 재호출 → correct/incorrect 1토큰 + logprob 추출."""
    fallback_messages = messages + [
        {"role": "assistant", "content": truncated_resp.rstrip() + "\nVerdict: "},
    ]
    logprobs_out = []
    try:
        token = _call_llm(model_name, fallback_messages, max_completion_tokens=3,
                          logprobs_out=logprobs_out) or ""
        token = token.strip().lower()
    except Exception:
        token = ""

    p_c, p_i = _prob_from_logprobs(logprobs_out, 0)
    if p_c is None:
        pred = "correct" if "correct" in token and "incorrect" not in token else "incorrect"
        p_c  = 1.0 if pred == "correct" else 0.0
        p_i  = 1.0 - p_c
    else:
        pred = "correct" if p_c >= p_i else "incorrect"

    logger.debug(f"[forced_verdict] token={token!r}  pred={pred}  p_correct={p_c:.3f}")
    return {
        "verdict_text":   f"Verdict: {token} [forced]",
        "response":       truncated_resp,
        "prob_correct":   p_c,
        "prob_incorrect": p_i,
        "reward":         p_c,
        "pred":           pred,
        "method":         "forced_verdict",
    }


def _parse_verdict(response: str) -> dict:
    """API 응답에서 correct/incorrect verdict 파싱."""
    if "</think>" in response:
        after_think = response.split("</think>", 1)[1].strip()
    else:
        after_think = response

    m = re.search(r"verdict[:\s]+(\w+)", after_think, re.I)
    if m:
        word = m.group(1).lower()
    else:
        words = re.findall(r"\b(correct|incorrect)\b", after_think, re.I)
        word  = words[-1].lower() if words else "incorrect"

    pred           = "correct" if word == "correct" else "incorrect"
    prob_correct   = 1.0 if pred == "correct" else 0.0
    prob_incorrect = 1.0 - prob_correct
    return {
        "verdict_text":   after_think,
        "response":       response,
        "prob_correct":   prob_correct,
        "prob_incorrect": prob_incorrect,
        "reward":         prob_correct,
        "pred":           pred,
        "method":         "api",
    }


# 모델별 1M 토큰당 가격 (USD) — (input, output)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "deepseek-reasoner": (0.55,  2.19),
    "deepseek-chat":     (0.27,  1.10),
    "gpt-4o":            (2.50, 10.00),
    "gpt-4o-mini":       (0.15,  0.60),
    "o1":                (15.0, 60.00),
    "o1-mini":           (3.00, 12.00),
    "o3-mini":           (1.10,  4.40),
    "o3":                (10.0, 40.00),
}


def _calc_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    key = next((k for k in _MODEL_PRICING if k in model_name.lower()), None)
    if key is None:
        return 0.0
    p_in, p_out = _MODEL_PRICING[key]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


class ApiPrm:
    """API 모델(config.PRM.model_id_checklist)로 PRM 평가."""

    def __init__(self, model_name: str, max_workers: int = 16):
        self.model_name    = model_name
        self.max_workers   = max_workers
        self.total_input   = 0
        self.total_output  = 0
        self.total_cached  = 0
        self._log = logging.getLogger("API_PRM")
        self._log.info(f"초기화: {model_name}")

    def evaluate_batch(
        self,
        questions: list[str],
        prev_steps: list[str],
        now_steps: list[str],
        system_prompts: list[str],
        max_new_tokens: int = 4096,
        problem_ids: list[str] = None,
        step_numbers: list[int] = None,
    ) -> list[dict]:
        def _call_one(args):
            q, prev, now, sys_prompt, pid, step = args
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": build_user_message(q, prev, now)},
            ]
            usage_out    = []
            logprobs_out = []
            t0 = time.time()
            try:
                set_problem_context(pid, step)
                set_call_role("critique")
                resp = _call_llm(self.model_name, messages, max_completion_tokens=max_new_tokens,
                                 usage_out=usage_out, logprobs_out=logprobs_out)
                resp = resp.strip() if resp else ""
                if usage_out and usage_out[0].get("finish_reason") == "length":
                    return _forced_verdict(self.model_name, messages, resp), usage_out, time.time() - t0
            except Exception as e:
                self._log.warning(f"호출 실패: {e}")
                resp = ""

            verdict = _parse_verdict(resp)
            # logprobs에서 실제 correct/incorrect 확률 추출해 덮어쓰기
            p_c, p_i = _prob_from_verdict_token(logprobs_out)
            if p_c is not None:
                verdict["prob_correct"]   = p_c
                verdict["prob_incorrect"] = p_i
                verdict["reward"]         = p_c
                verdict["method"]         = "logprob"
            return verdict, usage_out, time.time() - t0

        _pids  = problem_ids  or ["?"] * len(questions)
        _steps = step_numbers or [-1]  * len(questions)
        items  = list(zip(questions, prev_steps, now_steps, system_prompts, _pids, _steps))
        if not items:
            return []
        self._log.info(f"추론 시작: n={len(items)}  model={self.model_name}")
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as ex:
            raw = list(ex.map(_call_one, items))
        elapsed = time.time() - t_start

        results = []
        call_times = []
        for verdict, usage_out, call_time in raw:
            results.append(verdict)
            call_times.append(call_time)
            for u in usage_out:
                self.total_input  += u.get("input_tokens",  0)
                self.total_output += u.get("output_tokens", 0)
                self.total_cached += u.get("cached_tokens", 0)

        avg_call = sum(call_times) / len(call_times)
        self._log.info(f"추론 완료 (wall={elapsed:.1f}s) | 평균 호출 {avg_call:.1f}s")
        return results

    def print_cost(self):
        key      = next((k for k in _MODEL_PRICING if k in self.model_name.lower()), None)
        p_in, p_out = _MODEL_PRICING.get(key, (0.0, 0.0)) if key else (0.0, 0.0)
        # cached 단가: deepseek-reasoner=$0.14, 그 외 50% 할인
        p_cached = (0.14 if "deepseek-reasoner" in self.model_name.lower()
                    else 0.07 if "deepseek-chat" in self.model_name.lower()
                    else p_in * 0.5)
        non_cached  = self.total_input - self.total_cached
        cost_in     = non_cached       / 1_000_000 * p_in
        cost_cached = self.total_cached / 1_000_000 * p_cached
        cost_out    = self.total_output / 1_000_000 * p_out
        saved       = self.total_cached / 1_000_000 * (p_in - p_cached)
        print(
            f"\n[API 비용] model={self.model_name}\n"
            f"  input  {non_cached:>10,} tok  ${cost_in:.4f}\n"
            f"  cached {self.total_cached:>10,} tok  ${cost_cached:.4f}  (절약 ${saved:.4f})\n"
            f"  output {self.total_output:>10,} tok  ${cost_out:.4f}\n"
            f"  총계                    ${cost_in+cost_cached+cost_out:.4f} USD"
        )


def _rebuild_fast_rubric_for_indices(path: Path, indices: list[int]) -> dict:
    """Fast rubric JSON에서 1-indexed 위치의 루브릭만 추출해 시스템 프롬프트를 재조립."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        rubric_items = [item for item in data if item["type"] == "rubric"]
        selected = [rubric_items[i - 1] for i in indices if 1 <= i <= len(rubric_items)]
        selected_ids = {id(item) for item in selected}
        filtered = [item for item in data if item["type"] != "rubric" or id(item) in selected_ids]

        rubric_names = [item["label"] for item in filtered if item["type"] == "rubric"]
        n = len(rubric_names)
        output_format = "\n\n".join(
            f'[RUBRIC {i}] {name}\n<analysis>\nVerdict: correct/incorrect'
            for i, name in enumerate(rubric_names, 1)
        )
        parts = []
        rubric_counter = 0
        for item in filtered:
            text = "\n".join(item["content"]) if isinstance(item["content"], list) else item["content"]
            if item["type"] == "rubric":
                rubric_counter += 1
                text = re.sub(r'\[RUBRIC\s+\d+\]', f'[RUBRIC {rubric_counter}]', text)
            text = text.replace("{{n_rubrics}}", str(n)).replace("{{output_format}}", output_format)
            parts.append(text)
        return {"system_prompt": "\n\n".join(parts), "rubric_names": rubric_names}
    else:
        rubrics = data["rubrics"]
        selected = [rubrics[i - 1] for i in indices if 1 <= i <= len(rubrics)]
        n = len(selected)
        output_format = "\n\n".join(
            f'[RUBRIC {i}] {r["name"]}\n<analysis>\nVerdict: correct'
            for i, r in enumerate(selected, 1)
        )
        rubric_sections = "\n\n".join(
            f'━━━ [RUBRIC {i}] {r["name"]} ━━━\n{r["system_prompt"]}'
            for i, r in enumerate(selected, 1)
        )
        system_prompt = (
            data["shared_prompt"]
            .replace("{{n_rubrics}}", str(n))
            .replace("{{output_format}}", output_format)
            + "\n\n" + rubric_sections
        )
        return {"system_prompt": system_prompt, "rubric_names": [r["name"] for r in selected]}


def load_fast_rubric(path: "Path | str") -> dict:
    """배치 루브릭 JSON 로드. system_prompt_template + rubrics[] 구조를 조립해 system_prompt 생성."""
    path = Path(path)
    if not path.exists():
        logger.error(f"배치 루브릭 파일 없음: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # ── 배열 형식: [{type, no, label, content[]}, ...] ──────────────────────────
    if isinstance(data, list):
        rubric_names = [item["label"] for item in data if item["type"] == "rubric"]
        n = len(rubric_names)
        output_format = "\n\n".join(
            f'[RUBRIC {i}] {name}\n<analysis>\nVerdict: correct/incorrect'
            for i, name in enumerate(rubric_names, 1)
        )
        parts = []
        for item in data:
            text = "\n".join(item["content"]) if isinstance(item["content"], list) else item["content"]
            text = text.replace("{{n_rubrics}}", str(n)).replace("{{output_format}}", output_format)
            parts.append(text)
        result = {"system_prompt": "\n\n".join(parts), "rubric_names": rubric_names}
        logger.info(f"배치 루브릭 로드: {path.name}  ({n}개)")
        return result

    # ── dict 형식: {shared_prompt, rubrics[]} ───────────────────────────────────
    rubrics = data["rubrics"]
    n = len(rubrics)

    output_format = "\n\n".join(
        f'[RUBRIC {i}] {r["name"]}\n<analysis>\nVerdict: correct'
        for i, r in enumerate(rubrics, 1)
    )
    rubric_sections = "\n\n".join(
        f'━━━ [RUBRIC {i}] {r["name"]} ━━━\n{r["system_prompt"]}'
        for i, r in enumerate(rubrics, 1)
    )
    data["system_prompt"] = (
        data["shared_prompt"]
        .replace("{{n_rubrics}}", str(n))
        .replace("{{output_format}}", output_format)
        + "\n\n" + rubric_sections
    )
    data["rubric_names"] = [r["name"] for r in rubrics]

    logger.info(f"배치 루브릭 로드: {path.name}  ({n}개)")
    return data


def _parse_batch_verdict(response: str, rubric_names: list[str]) -> list[dict]:
    """배치 응답에서 루브릭별 verdict 파싱.
    1순위: compact 형식 (N: correct/incorrect)
    2순위: delimiter 형식 (<<<START_RUBRIC_N>>>)
    3순위: [RUBRIC N] 마커 fallback
    """
    n = len(rubric_names)

    # ── compact 형식 감지: 'N: correct' 또는 'N: incorrect' 줄이 과반수 ────────
    compact_hits = sum(
        1 for i in range(1, n + 1)
        if re.search(rf'^\s*{i}\s*:\s*(correct|incorrect)', response, re.I | re.M)
    )
    use_compact = compact_hits >= n // 2 + 1

    results = []
    for i, name in enumerate(rubric_names, 1):
        if use_compact:
            # ── compact: 'N: correct/incorrect' ──────────────────────────────
            m = re.search(rf'^\s*{i}\s*:\s*(correct|incorrect)', response, re.I | re.M)
            block = m.group(0).strip() if m else ""
            word  = m.group(1).lower() if m else "incorrect"
        else:
            # ── delimiter 방식 ────────────────────────────────────────────────
            m = re.search(rf'<<<START_RUBRIC_{i}>>>(.*?)<<<END_RUBRIC_{i}>>>',
                          response, re.DOTALL | re.IGNORECASE)
            if m:
                block = m.group(1).strip()
            else:
                # ── [RUBRIC N] 마커 fallback ──────────────────────────────────
                if i < n:
                    pat = (rf'\[RUBRIC\s+{i}\][^\n]*\n(.*?)'
                           rf'(?=<<<START_RUBRIC_{i+1}>>>|\[RUBRIC\s+{i+1}\])')
                else:
                    pat = rf'\[RUBRIC\s+{i}\][^\n]*\n(.*?)$'
                m2 = re.search(pat, response, re.DOTALL | re.IGNORECASE)
                block = m2.group(1).strip() if m2 else ""

            vm = re.search(r"verdict[:\s]+(\w+)", block, re.I)
            if vm:
                word = vm.group(1).lower()
            else:
                words = re.findall(r"\b(correct|incorrect)\b", block, re.I)
                word  = words[-1].lower() if words else "incorrect"

        # one-line critique: first non-header, non-verdict line in the block
        _critique = None
        _section_header_pat = re.compile(
            r"^\*{0,2}(?:checks|analysis|rubric|gate|step\s*\d+|n/a\s*rule|evidence|additional\s+checks?)[:\.]?\*{0,2}\s*$",
            re.IGNORECASE,
        )
        for _line in block.split("\n"):
            _l = _line.strip()
            if not _l:
                continue
            if re.match(r"\[RUBRIC", _l, re.I) or re.match(r"verdict", _l, re.I):
                continue
            if _section_header_pat.match(_l):
                continue
            if re.match(r"^\d+\s*:\s*(correct|incorrect)\s*$", _l, re.I):
                continue
            _critique = _l
            break

        pred = "correct" if word == "correct" else "incorrect"
        results.append({
            "verdict_text":   block,
            "response":       block,          # 전체 아닌 해당 루브릭 블록만 저장
            "prob_correct":   1.0 if pred == "correct" else 0.0,
            "prob_incorrect": 0.0 if pred == "correct" else 1.0,
            "reward":         0.0 if pred == "correct" else 1.0,
            "pred":           pred,
            "critique":       _critique,
            "method":         "api_batch",
        })
    return results


class ApiPrmBatch:
    """모든 루브릭을 샘플당 1번 API 호출로 처리."""

    def __init__(self, model_name: str, fast_rubric: dict, max_workers: int = 32):
        self.model_name    = model_name
        self.rubric_names  = fast_rubric["rubric_names"]
        self.system_prompt = fast_rubric["system_prompt"]
        self.max_workers   = max_workers
        self.total_input   = 0
        self.total_output  = 0
        self.total_cached  = 0
        self._log = logging.getLogger("API_PRM_Batch")
        self._log.info(f"초기화: {model_name}  루브릭={len(self.rubric_names)}개")

    @property
    def rubric_dicts(self) -> list[dict]:
        """generate_trajectory.py의 rubrics list 형식 호환용 (name만 포함)."""
        return [{"name": n} for n in self.rubric_names]

    def evaluate_batch(
        self,
        questions: list[str],
        prev_steps: list[str],
        now_steps: list[str],
        max_new_tokens: int = 4096,
        problem_ids: list[str] = None,
        step_numbers: list[int] = None,
    ) -> list[list[dict]]:
        """각 샘플에 대해 1번 API 호출. 반환: [[rubric0_verdict, ...], ...]"""
        def _call_one(args):
            q, prev, now, pid, step = args
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": build_user_message(q, prev, now)},
            ]
            usage_out = []
            t0 = time.time()
            try:
                set_problem_context(pid, step)
                set_call_role("fast_rubric")
                resp = _call_llm(self.model_name, messages, max_completion_tokens=max_new_tokens, usage_out=usage_out)
                resp = resp.strip() if resp else ""
            except Exception as e:
                self._log.warning(f"호출 실패: {e}")
                resp = ""
            return _parse_batch_verdict(resp, self.rubric_names), usage_out, time.time() - t0

        _pids  = problem_ids  or ["?"] * len(questions)
        _steps = step_numbers or [-1]  * len(questions)
        items  = list(zip(questions, prev_steps, now_steps, _pids, _steps))
        if not items:
            return []
        self._log.info(f"추론 시작: n={len(items)}  model={self.model_name}")
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as ex:
            raw = list(ex.map(_call_one, items))
        elapsed = time.time() - t_start

        results = []
        call_times = []
        for verdicts, usage_out, call_time in raw:
            results.append(verdicts)
            call_times.append(call_time)
            for u in usage_out:
                self.total_input  += u.get("input_tokens",  0)
                self.total_output += u.get("output_tokens", 0)
                self.total_cached += u.get("cached_tokens", 0)

        avg_call = sum(call_times) / len(call_times)
        self._log.info(f"추론 완료 (wall={elapsed:.1f}s) | 평균 호출 {avg_call:.1f}s")
        return results

    def print_cost(self):
        key = next((k for k in _MODEL_PRICING if k in self.model_name.lower()), None)
        p_in, p_out = _MODEL_PRICING.get(key, (0.0, 0.0)) if key else (0.0, 0.0)
        p_cached = (0.14 if "deepseek-reasoner" in self.model_name.lower()
                    else 0.07 if "deepseek-chat" in self.model_name.lower()
                    else p_in * 0.5)
        non_cached  = self.total_input - self.total_cached
        cost_in     = non_cached        / 1_000_000 * p_in
        cost_cached = self.total_cached / 1_000_000 * p_cached
        cost_out    = self.total_output / 1_000_000 * p_out
        saved       = self.total_cached / 1_000_000 * (p_in - p_cached)
        print(
            f"\n[API 비용] model={self.model_name}\n"
            f"  input  {non_cached:>10,} tok  ${cost_in:.4f}\n"
            f"  cached {self.total_cached:>10,} tok  ${cost_cached:.4f}  (절약 ${saved:.4f})\n"
            f"  output {self.total_output:>10,} tok  ${cost_out:.4f}\n"
            f"  총계                    ${cost_in+cost_cached+cost_out:.4f} USD"
        )


class ApiPrmTwoStage:
    """2-stage PRM.
    Stage 1: batch 루브릭으로 평가 (빠르고 저렴).
    Stage 2: Stage 1이 fail인 경우에만 개별 루브릭 9개로 재평가 (정확).
    최종 verdict = Stage 1 pass → pass / Stage 1 fail → Stage 2 결과.
    """

    def __init__(self, model_name: str, fast_rubric: dict, rubrics: list[dict],
                 max_workers: int = 32):
        self.model_name = model_name
        self.stage1     = ApiPrmBatch(model_name, fast_rubric, max_workers)
        self.stage2     = ApiPrm(model_name, max_workers)
        self.rubrics    = rubrics
        logger.info(
            f"API PRM 2-Stage 초기화: {model_name} | "
            f"stage1=fast({len(fast_rubric['rubric_names'])}개) / "
            f"stage2=개별({len(rubrics)}개 루브릭)"
        )

    @property
    def rubric_dicts(self) -> list[dict]:
        return [{"name": r["name"]} for r in self.rubrics]

    @property
    def total_input(self) -> int:
        return self.stage1.total_input + self.stage2.total_input

    @property
    def total_output(self) -> int:
        return self.stage1.total_output + self.stage2.total_output

    @property
    def total_cached(self) -> int:
        return self.stage1.total_cached + self.stage2.total_cached

    def print_cost(self):
        print("\n[2-Stage PRM 비용]")
        print("  Stage 1 (batch):")
        self.stage1.print_cost()
        print("  Stage 2 (개별 루브릭):")
        self.stage2.print_cost()


# ─────────────────────────────────────────────────────────────────────────────
# 외부 import용 평가 함수
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_step(
    question: str,
    prev_steps: str,
    now_step: str,
    rubrics: list[dict] | None = None,
    model: "ApiPrm | None" = None,
    fail_k: int = 1,
    max_new_tokens: int = 4096,
    cot: bool = False,
    problem_id: str = None,
    step_number: int = None,
) -> tuple[str, dict]:
    """단일 스텝을 루브릭 hard voting으로 평가.

    rubrics, model 미지정 시 config.yaml의 PRM.rubric_file / API_model.PRM 사용.
    fail_k개 이상의 루브릭이 'fail' → "fail" 반환, 아니면 "pass".

    Returns:
        verdict : "pass" or "fail"
        detail  : {rubric_name: full result dict} — pred/verdict_text/response 등 포함
    """
    if model is None:
        if not _DEFAULT_PRM_MODEL:
            raise ValueError("config.yaml의 PRM.model_id_checklist에 모델 이름을 설정해 주세요.")
        model = ApiPrm(_DEFAULT_PRM_MODEL)

    if rubrics is None:
        if not _DEFAULT_RUBRIC_FILE:
            raise ValueError("config.yaml의 PRM.rubric_file에 루브릭 파일 경로를 설정해 주세요.")
        rubric_path = Path(_DEFAULT_RUBRIC_FILE)
        if not rubric_path.is_absolute():
            rubric_path = ROOT / rubric_path
        rubrics = load_deep_rubrics(rubric_path)

    if not rubrics:
        return "pass", {}

    _pid   = str(problem_id) if problem_id is not None else "?"
    _step  = int(step_number) if step_number is not None else -1
    results = model.evaluate_batch(
        questions=[question] * len(rubrics),
        prev_steps=[prev_steps] * len(rubrics),
        now_steps=[now_step] * len(rubrics),
        system_prompts=[build_system_prompt(r, cot=cot) for r in rubrics],
        max_new_tokens=max_new_tokens,
        problem_ids=[_pid]  * len(rubrics),
        step_numbers=[_step] * len(rubrics),
    )
    detail  = {r["name"]: res for r, res in zip(rubrics, results)}
    n_fail  = sum(1 for res in detail.values() if res["pred"] == "incorrect")
    verdict = "fail" if n_fail >= fail_k else "pass"
    return verdict, detail


# ─────────────────────────────────────────────────────────────────────────────
# 메트릭
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """recall=1이 되는 최소 threshold에서 precision 최대화.

    prob_correct(연속값 우선, 없으면 pred에서 0/1 유추)를 기준으로
    실제 fail 샘플 전부를 잡는 가장 높은 threshold를 찾아 적용한다.
    """
    def _prob_c(r):
        v = r.get("prob_correct")
        return v if v is not None else (1.0 if r.get("pred") == "correct" else 0.0)

    valid = [r for r in results if r.get("label") in ("correct", "incorrect")]
    if not valid:
        return {"total": 0, "accuracy": 0.0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "threshold": 0.0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0}

    fail_probs = [_prob_c(r) for r in valid if r["label"] == "incorrect"]
    # fail 샘플이 없으면 threshold=0 (아무것도 fail로 안 잡음)
    threshold = max(fail_probs) if fail_probs else 0.0

    tp = fp = tn = fn = 0
    for r in valid:
        label  = r["label"]
        pred   = "incorrect" if _prob_c(r) <= threshold else "correct"
        if pred == "incorrect" and label == "incorrect":   tp += 1
        elif pred == "incorrect" and label == "correct":   fp += 1
        elif pred == "correct"  and label == "correct":    tn += 1
        else:                                               fn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp)    if (tp + fp) else 0.0
    recall    = tp / (tp + fn)    if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "total":     total,
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "threshold": round(threshold, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp)    if (tp + fp) else 0.0
    recall    = tp / (tp + fn)    if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def compute_ensemble_metrics(
    all_results: dict[str, list[dict]],
    ensemble_rubric_names: list[str] | None = None,
) -> dict:
    """
    Hard voting (k) 및 Soft voting (threshold) 앙상블 성능 계산.

    Hard voting  : k개 이상 루브릭이 'fail' 예측 → fail  (key: ">=k")
    Soft voting  : 전체 루브릭 평균 reward >= threshold → fail
                   (reward 높을수록 fail 가능성 높음, threshold: 0.60 / 0.65 / 0.70)

    ensemble_rubric_names: 앙상블에 사용할 루브릭 이름 목록.
                           None이면 all_results의 전체 루브릭 사용.

    반환 dict 구조:
      hard_voting : {">=k": {accuracy, precision, recall, f1, tp, fp, tn, fn}}
      soft_voting : {"thr=0.30": {accuracy, precision, recall, f1, tp, fp, tn, fn}, ...}
    """
    import numpy as np

    if ensemble_rubric_names is not None:
        rubric_names = [n for n in ensemble_rubric_names if n in all_results]
    else:
        rubric_names = list(all_results.keys())
    n_rubrics = len(rubric_names)

    # 샘플별로 (label, preds, avg_reward) 취합
    by_sample: dict[int, dict] = {}
    for name, results in all_results.items():
        for r in results:
            idx = r["sample_idx"]
            if idx not in by_sample:
                by_sample[idx] = {"label": r["label"], "preds": {}, "rewards": {}}
            by_sample[idx]["preds"][name]   = r["pred"]
            by_sample[idx]["rewards"][name] = r.get("reward", 0.0)

    samples = []
    for idx in sorted(by_sample.keys()):
        s = by_sample[idx]
        rewards = [s["rewards"].get(name, 0.0) for name in rubric_names]
        samples.append({
            "label":       s["label"],
            "n_fail_pred": sum(1 for name in rubric_names if s["preds"].get(name) == "fail"),
            "avg_reward":  float(np.mean(rewards)),
        })

    # ── Hard voting ──────────────────────────────────────────────
    # key: ">=k" — k개 이상 루브릭이 fail 예측 시 fail
    hard_voting: dict[str, dict] = {}
    for k in range(1, n_rubrics + 1):
        tp = fp = tn = fn = 0
        for s in samples:
            pred  = "incorrect" if s["n_fail_pred"] >= k else "correct"
            label = s["label"]
            if label not in ("correct", "incorrect"):
                continue
            if pred == "incorrect" and label == "incorrect":   tp += 1
            elif pred == "incorrect" and label == "correct":   fp += 1
            elif pred == "correct"  and label == "correct":    tn += 1
            else:                                               fn += 1
        hard_voting[f">={k}"] = _metrics_from_counts(tp, fp, tn, fn)

    # ── Soft voting ───────────────────────────────────────────────
    # 고정 threshold: 0.60 / 0.65 / 0.70  (avg_reward >= threshold → fail)
    soft_voting: dict[str, dict] = {}
    for thr in [0.60, 0.65, 0.70]:
        tp = fp = tn = fn = 0
        for s in samples:
            pred  = "incorrect" if s["avg_reward"] >= thr else "correct"
            label = s["label"]
            if label not in ("correct", "incorrect"):
                continue
            if pred == "incorrect" and label == "incorrect":   tp += 1
            elif pred == "incorrect" and label == "correct":   fp += 1
            elif pred == "correct"  and label == "correct":    tn += 1
            else:                                               fn += 1
        soft_voting[f"thr={thr:.2f}"] = _metrics_from_counts(tp, fp, tn, fn)

    return {
        "hard_voting": hard_voting,
        "soft_voting": soft_voting,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드 및 샘플링
# ─────────────────────────────────────────────────────────────────────────────

def load_data(start: int = SAMPLE_START, end: int = SAMPLE_END,
              path: "Path | str | None" = None) -> list[dict]:
    raw = []
    with open(path or DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # history가 리스트면 줄바꿈으로 concat
            if isinstance(d.get("history"), list):
                d["history"] = "\n".join(d["history"])
            raw.append(d)

    # JSONL 형식(inference/history 필드)과 기존 형식(question/now_step/previous_steps) 통일
    for d in raw:
        if "now_step" not in d:
            d["now_step"] = d.get("inference", "")
        if "previous_steps" not in d:
            d["previous_steps"] = d.get("history", "")
        if "question" not in d:
            d["question"] = d.get("problem", d.get("problem_id", ""))

    _PASS_VALS = {"Yes", "correct"}
    _FAIL_VALS = {"No", "incorrect"}
    pass_samples = [d for d in raw if str(d.get("gold_answer", "")).strip() in _PASS_VALS]
    fail_samples = [d for d in raw if str(d.get("gold_answer", "")).strip() in _FAIL_VALS]
    other        = [d for d in raw if d not in pass_samples and d not in fail_samples]

    if pass_samples or fail_samples:
        pass_samples = pass_samples[start:end]
        fail_samples = fail_samples[start:end]
        data = pass_samples + fail_samples
    else:
        data = other[start:end] if (start or end) else other

    random.seed(42)
    random.shuffle(data)
    for i, d in enumerate(data):
        d["sample_idx"] = i
    logger.info(
        f"데이터 로드: pass={len(pass_samples) if pass_samples or fail_samples else '-'}"
        f"  fail={len(fail_samples) if pass_samples or fail_samples else '-'}"
        f"  total={len(data)}  (클래스별 [{start}:{end}])"
    )
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 실험 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    model_path: str,
    cache_dir: str,
    gpu_ids: list[int],
    batch_size: int,
    data: list[dict],
    rubrics: list[dict],
    max_new_tokens: int = 4096,
    model: "ApiPrm" = None,
) -> dict[str, list[dict]]:
    t_inference_start = time.time()

    all_pairs: list[tuple[dict, dict]] = [
        (rubric, item) for rubric in rubrics for item in data
    ]
    total    = len(all_pairs)
    n_batches = math.ceil(total / batch_size)

    logger.info(
        f"전체 {total}쌍 ({len(rubrics)}루브릭 × {len(data)}샘플) "
        f"| batch_size={batch_size}, n_batches={n_batches}"
    )

    all_results: dict[str, list[dict]] = {r["name"]: [] for r in rubrics}

    for batch_idx, batch_start in enumerate(range(0, total, batch_size)):
        batch_pairs   = all_pairs[batch_start : batch_start + batch_size]
        batch_rubrics = [p[0] for p in batch_pairs]
        batch_items   = [p[1] for p in batch_pairs]

        t0 = time.time()
        try:
            outs = model.evaluate_batch(
                questions=[item["question"] for item in batch_items],
                prev_steps=[item.get("previous_steps", "") for item in batch_items],
                now_steps=[item["now_step"] for item in batch_items],
                system_prompts=[build_system_prompt(r) for r in batch_rubrics],
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            logger.error(f"[배치 {batch_idx+1}/{n_batches}] 오류: {e}")
            outs = [
                {"verdict_text": "", "response": None,
                 "prob_correct": None, "prob_incorrect": None, "pred": None, "method": None}
                for _ in batch_pairs
            ]

        elapsed = time.time() - t0
        logger.info(
            f"배치 {batch_idx+1}/{n_batches} 완료 ({elapsed:.1f}s) | "
            f"{len(batch_pairs)}쌍, {elapsed/len(batch_pairs):.1f}s/샘플"
        )

        for (rubric, item), out in zip(batch_pairs, outs):
            label = "correct" if str(item.get("gold_answer", "")).strip() in ("Yes", "correct") else "incorrect"
            pred  = out["pred"]
            all_results[rubric["name"]].append({
                "rubric_name":    rubric["name"],
                "sample_idx":     item.get("sample_idx"),
                "question":       item.get("question", ""),
                "previous_steps": item.get("previous_steps", ""),
                "now_step":       item.get("now_step", ""),
                "gold_answer":    item.get("gold_answer", ""),
                "label":          label,
                "pred":           pred,
                "is_correct":     (pred == label) if pred is not None else None,
                "prob_correct":   out["prob_correct"],
                "prob_incorrect": out["prob_incorrect"],
                "reward":         out.get("reward"),
                "verdict_text":   out.get("verdict_text", ""),
                "response":       out.get("response", ""),
                "method":         out.get("method", "logprob"),
            })

    for name in all_results:
        all_results[name].sort(key=lambda x: (x.get("sample_idx") or 0))

    inference_elapsed = time.time() - t_inference_start
    logger.info(f"총 추론 시간 (모델 로딩 제외): {inference_elapsed:.1f}s")
    return all_results, inference_elapsed


# ─────────────────────────────────────────────────────────────────────────────
# 상관관계 분석
# ─────────────────────────────────────────────────────────────────────────────

def analyze_rubric_correlation(all_results: dict[str, list[dict]]) -> dict:
    """
    루브릭 간 상관관계 분석.

    반환 dict 구조:
      agreement_matrix  : 루브릭 쌍별 예측 일치율 (0~1)
      phi_matrix        : 루브릭 쌍별 phi coefficient (이진 상관계수, -1~1)
      unique_contribution: 루브릭별 단독 정답 기여율
        - unique_hits       : 이 루브릭만 맞추고 나머지는 모두 틀린 샘플 수 (wrong 레이블 기준)
        - shared_hits       : 여러 루브릭이 함께 맞춘 샘플 수 (wrong 레이블 기준)
        - miss_when_others  : 다른 루브릭 평균이 맞출 때 이 루브릭이 틀린 샘플 수
    """
    rubric_names = list(all_results.keys())
    by_sample: dict[int, dict[str, str | None]] = {}
    for name, results in all_results.items():
        for r in results:
            idx = r["sample_idx"]
            if idx not in by_sample:
                by_sample[idx] = {"label": r["label"]}
            by_sample[idx][name] = r["pred"]

    sample_ids = sorted(by_sample.keys())

    agreement_matrix: dict[str, dict[str, float]] = {r: {} for r in rubric_names}
    phi_matrix:       dict[str, dict[str, float]] = {r: {} for r in rubric_names}

    for i, ra in enumerate(rubric_names):
        for j, rb in enumerate(rubric_names):
            if i == j:
                agreement_matrix[ra][rb] = 1.0
                phi_matrix[ra][rb]       = 1.0
                continue

            agree = a = b = c = d = 0
            for idx in sample_ids:
                pa = by_sample[idx].get(ra)
                pb = by_sample[idx].get(rb)
                if pa is None or pb is None:
                    continue
                aw = pa == "incorrect"
                bw = pb == "incorrect"
                if aw and bw:  a += 1
                elif aw:       b += 1
                elif bw:       c += 1
                else:          d += 1
                if pa == pb:
                    agree += 1

            total = a + b + c + d
            agreement_matrix[ra][rb] = round(agree / total, 4) if total else 0.0

            denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            phi = (a * d - b * c) / denom if denom > 0 else 0.0
            phi_matrix[ra][rb] = round(phi, 4)

    unique_contribution: dict[str, dict] = {}
    for target in rubric_names:
        others = [r for r in rubric_names if r != target]
        unique_hits = shared_hits = miss_when_others = 0

        wrong_samples = [idx for idx in sample_ids if by_sample[idx]["label"] == "incorrect"]
        for idx in wrong_samples:
            sample = by_sample[idx]
            target_hit   = sample.get(target) == "incorrect"
            others_hits  = [sample.get(o) == "incorrect" for o in others if sample.get(o) is not None]
            n_others_hit = sum(others_hits)

            if target_hit and n_others_hit == 0:
                unique_hits += 1
            elif target_hit and n_others_hit > 0:
                shared_hits += 1
            elif not target_hit and n_others_hit > 0:
                miss_when_others += 1

        unique_contribution[target] = {
            "unique_hits":        unique_hits,
            "shared_hits":        shared_hits,
            "miss_when_others":   miss_when_others,
            "unique_hit_rate":    round(unique_hits / len(wrong_samples), 4) if wrong_samples else 0.0,
        }

    coverage_dist: dict[int, int] = {}
    for idx in sample_ids:
        if by_sample[idx]["label"] != "incorrect":
            continue
        n_correct = sum(
            1 for r in rubric_names
            if by_sample[idx].get(r) == "incorrect"
        )
        coverage_dist[n_correct] = coverage_dist.get(n_correct, 0) + 1

    return {
        "agreement_matrix":  agreement_matrix,
        "phi_matrix":        phi_matrix,
        "unique_contribution": unique_contribution,
        "fail_sample_coverage_dist": {str(k): v for k, v in sorted(coverage_dist.items())},
    }


def print_correlation(corr: dict, rubric_names: list[str]):
    """상관관계 분석 결과를 콘솔에 출력."""
    W = 75
    short = {r: r[:18] for r in rubric_names}

    print(f"\n{'='*W}")
    print(" [루브릭 예측 일치율 (Agreement Matrix)]  대각선=1.0, 높을수록 유사한 루브릭")
    print(f"{'─'*W}")
    header = f" {'':20}" + "".join(f"{short[r]:>10}" for r in rubric_names)
    print(header)
    for ra in rubric_names:
        row = f" {short[ra]:<20}" + "".join(
            f"{corr['agreement_matrix'][ra][rb]:>10.3f}" for rb in rubric_names
        )
        print(row)

    print(f"\n{'─'*W}")
    print(" [Phi Coefficient Matrix]  +1=완전일치, 0=독립, -1=반대  (positive class=fail)")
    print(f"{'─'*W}")
    print(header)
    for ra in rubric_names:
        row = f" {short[ra]:<20}" + "".join(
            f"{corr['phi_matrix'][ra][rb]:>10.3f}" for rb in rubric_names
        )
        print(row)

    print(f"\n{'─'*W}")
    print(" [Unique Contribution]  fail 샘플 기준 — 해당 루브릭만 맞추는 샘플 비율")
    print(f"{'─'*W}")
    print(f" {'Rubric':<32} {'Unique':>8} {'Shared':>8} {'MissWhenOthers':>16} {'UniqueRate':>12}")
    uc = corr["unique_contribution"]
    for r in rubric_names:
        u = uc[r]
        print(
            f" {r:<32} {u['unique_hits']:>8} {u['shared_hits']:>8} "
            f"{u['miss_when_others']:>16} {u['unique_hit_rate']:>12.4f}"
        )

    print(f"\n{'─'*W}")
    print(" [Fail 샘플 커버리지]  'k개 루브릭이 맞춘 fail 샘플 수'")
    print(f"{'─'*W}")
    for k, cnt in sorted(corr["fail_sample_coverage_dist"].items(), key=lambda x: int(x[0])):
        bar = "█" * cnt
        print(f"  {k:>2}개 루브릭 맞춤: {cnt:>4}개  {bar}")
    print(f"{'='*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────────────────────────────────────

def _compute_binary_metrics(results: list[dict]) -> dict:
    """pred 필드를 직접 사용한 이진 분류 메트릭."""
    valid = [r for r in results if r.get("label") in ("correct", "incorrect")]
    if not valid:
        return {"total": 0, "accuracy": 0.0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    tp = fp = tn = fn = 0
    for r in valid:
        label = r["label"]
        pred  = r.get("pred", "correct")
        if   pred == "incorrect" and label == "incorrect": tp += 1
        elif pred == "incorrect" and label == "correct":   fp += 1
        elif pred == "correct"   and label == "correct":   tn += 1
        else:                                               fn += 1
    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total        if total       else 0.0
    precision = tp / (tp + fp)           if (tp + fp)   else 0.0
    recall    = tp / (tp + fn)           if (tp + fn)   else 0.0
    f1        = 2*precision*recall / (precision + recall) if (precision + recall) else 0.0
    return {"total": total, "accuracy": round(accuracy, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def print_results(all_results: dict[str, list[dict]], model_path: str,
                  eval_names: set | None = None) -> dict:
    """루브릭별 메트릭을 비교표로 출력하고 metrics_by_rubric을 반환.
    eval_names가 주어지면 해당 루브릭만 표시 (반환값은 전체)."""
    W = 83
    print(f"\n{'='*W}")
    print(f" 모델: {Path(model_path).name}")
    print(f"{'='*W}")
    print(f" {'Rubric':<32} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'N':>5}")
    print(f" {'-'*32} {'------':>6} {'------':>6} {'------':>6} {'------':>6} {'----':>4} {'----':>4} {'----':>4} {'----':>4} {'-----':>5}")

    best_prec, best_name = -1.0, ""
    metrics_by_rubric = {}
    for rubric_name, results in all_results.items():
        m = _compute_binary_metrics(results)
        metrics_by_rubric[rubric_name] = m
        if eval_names and rubric_name not in eval_names:
            continue
        if m["precision"] > best_prec:
            best_prec, best_name = m["precision"], rubric_name
        marker = " ←" if rubric_name == best_name else ""
        print(
            f" {rubric_name:<32} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} {m['f1']:>6.4f} "
            f"{m['tp']:>4} {m['fp']:>4} {m['tn']:>4} {m['fn']:>4} {m['total']:>5}{marker}"
        )

    print(f"{'─'*W}")
    print(f" Best Precision: {best_name}  ({best_prec:.4f})")
    print(f"{'='*W}\n")
    return metrics_by_rubric


# ─────────────────────────────────────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    out_dir: Path,
    all_results: dict[str, list[dict]],
    metrics_by_rubric: dict,
    corr: dict,
    model_path: str,
    rubrics: list[dict],
    inference_elapsed: float = 0.0,
    rubric_file: Path | str = "",
):
    """out_dir 안에 루브릭별 jsonl + summary.json + prompts.jsonl 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for rubric_name, results in all_results.items():
        safe_name = rubric_name.replace(" ", "_").replace("/", "-")
        jsonl_path = out_dir / f"{safe_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"  저장: {jsonl_path.name}  ({len(results)}건)")

    prompts_path = out_dir / "prompts.jsonl"
    with open(prompts_path, "w", encoding="utf-8") as f:
        for rubric in rubrics:
            row = {
                "rubric_name":    rubric["name"],
                "prompt_version": rubric.get("prompt_version", ""),
                "system_prompt":  rubric["system_prompt"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"  프롬프트 저장: {prompts_path.name}  ({len(rubrics)}건)")

    # 앙상블은 앞에서 10개 루브릭 사용 (Atomicity 제외)
    ensemble_names = [r["name"] for r in rubrics[:10]]
    ensemble = compute_ensemble_metrics(all_results, ensemble_rubric_names=ensemble_names)

    # unique_find를 per-rubric metrics에 추가 (fail 샘플 중 이 루브릭만 잡는 샘플)
    by_idx: dict[int, dict] = {}
    for rb_name, results in all_results.items():
        for d in results:
            idx = d["sample_idx"]
            if idx not in by_idx:
                by_idx[idx] = {"label": d["label"]}
            by_idx[idx][rb_name] = d["pred"]

    all_rubric_names = list(all_results.keys())
    for rb_name in all_rubric_names:
        unique_samples = []
        for idx, sample in by_idx.items():
            if sample.get("label") != "incorrect":
                continue
            only_this = (sample.get(rb_name) == "incorrect") and all(
                sample.get(o) != "incorrect" for o in all_rubric_names if o != rb_name
            )
            if only_this:
                unique_samples.append(idx)
        unique_samples.sort()
        uc_entry = corr.get("unique_contribution", {}).get(rb_name, {})
        metrics_by_rubric[rb_name]["unique_find"]         = len(unique_samples)
        metrics_by_rubric[rb_name]["unique_find_rate"]    = uc_entry.get("unique_hit_rate", 0.0)
        metrics_by_rubric[rb_name]["unique_find_samples"] = unique_samples

    eval_names = {r["name"] for r in rubrics[:10]}
    summary = {
        "model":               model_path,
        "timestamp":           out_dir.name,
        "rubric_file":         str(rubric_file),
        "n_rubrics":           len(rubrics[:10]),
        "rubrics":             [r["name"] for r in rubrics[:10]],
        "inference_elapsed_s": round(inference_elapsed, 1),
        "metrics":             {k: v for k, v in metrics_by_rubric.items() if k in eval_names},
        "correlation":         corr,
        "ensemble":            ensemble,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"  요약 저장: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def _run_batch_rubric_experiment(
    model: "ApiPrmBatch",
    data: list[dict],
    max_new_tokens: int,
    out_dir: "Path",
    rubric_file: str = "",
):
    """ApiPrmBatch(stage1)로 샘플당 1번 호출.

    저장 구조:
      {out_dir}/
        log/api_calls.jsonl    각 API 호출: model, in_tok, out_tok, sample_idx
        result/results.jsonl   샘플별 input+output 전체
        summary.json           루브릭별/overall accuracy
    """
    rubric_names = model.rubric_names
    questions  = [d["question"]                 for d in data]
    prev_steps = [d.get("previous_steps", "")   for d in data]
    now_steps  = [d["now_step"]                 for d in data]
    # gold_fail_rubrics가 있으면 전체 label은 "fail"(어느 루브릭이든 하나라도 실패)
    # gold_answer 기반 기존 포맷도 병행 지원
    def _overall_label(d):
        gold = str(d.get("gold_answer", "")).strip()
        if gold in ("Yes", "correct"):
            return "correct"
        if d.get("gold_fail_rubrics") is not None:
            _fr = d["gold_fail_rubrics"]
            _has_fail = isinstance(_fr, list) and bool(_fr)
            return "incorrect" if _has_fail else "correct"
        return "incorrect"
    labels = [_overall_label(d) for d in data]

    # ── 출력 디렉토리 준비 ─────────────────────────────────────────────────────
    log_dir    = out_dir / "log"
    result_dir = out_dir / "result"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── API 호출 로그 수집 ────────────────────────────────────────────────────
    _log_file  = open(log_dir / "api_calls.jsonl", "w", encoding="utf-8")
    _log_lock  = __import__("threading").Lock()
    _call_idx  = [0]

    def _log_call(record: dict):
        with _log_lock:
            idx = _call_idx[0]
            _call_idx[0] += 1
            _log_file.write(json.dumps({
                "call_idx":   idx,
                "ts":         record.get("ts"),
                "model":      record.get("model"),
                "role":       record.get("role"),
                "in_tok":     record.get("in_tok"),
                "out_tok":    record.get("out_tok"),
                "sample_idx": None,       # 배치 호출이라 개별 매핑 불가
            }, ensure_ascii=False) + "\n")
            _log_file.flush()

    set_run_log(_log_call)

    # ── 추론 실행 ─────────────────────────────────────────────────────────────
    t0 = time.time()
    verdicts_list = model.evaluate_batch(
        questions=questions,
        prev_steps=prev_steps,
        now_steps=now_steps,
        max_new_tokens=max_new_tokens,
    )
    wall = time.time() - t0

    set_run_log(None)
    _log_file.close()

    # ── result/results.jsonl 저장 ─────────────────────────────────────────────
    result_file = open(result_dir / "results.jsonl", "w", encoding="utf-8")
    per_rubric: dict[str, list[dict]] = {n: [] for n in rubric_names}

    for item, label, verdicts in zip(data, labels, verdicts_list):
        row = {
            "sample_idx":     item.get("sample_idx"),
            "label":          label,
            "gold_answer":    item.get("gold_answer"),
            "question":       item.get("question", ""),
            "previous_steps": item.get("previous_steps", ""),
            "now_step":       item.get("now_step", ""),
            "rubric_verdicts": {},
        }
        _gfr = item.get("gold_fail_rubrics")
        gold_fail_set = set(_gfr if isinstance(_gfr, list) else [])
        for rname, v in zip(rubric_names, verdicts):
            pred = v.get("pred")
            # gold_fail_rubrics가 있으면 루브릭별 label, 없으면 샘플 전체 label 사용
            rubric_label = (
                "incorrect" if rname in gold_fail_set else "correct"
            ) if gold_fail_set or item.get("gold_fail_rubrics") is not None else label
            row["rubric_verdicts"][rname] = {
                "pred":          pred,
                "critique":      v.get("critique"),
                "response":      v.get("response"),
            }
            per_rubric[rname].append({
                "rubric_name": rname,
                "sample_idx":  item.get("sample_idx"),
                "label":       rubric_label,
                "pred":        pred,
                "is_correct":  (pred == rubric_label) if pred is not None else None,
            })
        result_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    result_file.close()

    # ── summary.json 저장 + 출력 ──────────────────────────────────────────────
    per_rubric_metrics = {n: _compute_binary_metrics(per_rubric[n]) for n in rubric_names}

    overall_pairs = []
    for label, verdicts in zip(labels, verdicts_list):
        any_fail = any(v.get("pred") == "incorrect" for v in verdicts)
        overall_pairs.append({"pred": "incorrect" if any_fail else "correct", "label": label})
    overall = _compute_binary_metrics(overall_pairs)

    summary = {
        "model":          model.model_name,
        "rubric_file":    str(rubric_file),
        "max_new_tokens": max_new_tokens,
        "n_samples":      len(data),
        "wall_sec":       round(wall, 2),
        "avg_sec_per_sample": round(wall / max(1, len(data)), 2),
        "overall":        overall,
        "per_rubric":     per_rubric_metrics,
        "cost": {
            "total_input":  model.total_input,
            "total_output": model.total_output,
            "total_cached": model.total_cached,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 콘솔 출력
    W = 72
    print(f"\n{'='*W}")
    print(f" [배치 루브릭 실험]  model={model.model_name}  max_new_tokens={max_new_tokens}")
    print(f" 데이터: {len(data)}샘플  wall={wall:.1f}s  avg={wall/max(1,len(data)):.1f}s/샘플")
    print(f"{'─'*W}")
    print(f" {'Rubric':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print(f" {'─'*35} {'──────':>6} {'──────':>6} {'──────':>6} {'──────':>6} {'─────':>5}")
    for rname, m in per_rubric_metrics.items():
        print(f" {rname:<35} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['total']:>5}")
    print(f" {'─'*35} {'──────':>6} {'──────':>6} {'──────':>6} {'──────':>6} {'─────':>5}")
    print(f" {'OVERALL (any-fail → fail)':<35} {overall['accuracy']:>6.3f} {overall['precision']:>6.3f} "
          f"{overall['recall']:>6.3f} {overall['f1']:>6.3f} {overall['total']:>5}")
    print(f"{'='*W}")
    print(f"\n저장 위치: {out_dir}")
    print(f"  log/api_calls.jsonl  — API 호출 기록 ({_call_idx[0]}건)")
    print(f"  result/results.jsonl — 샘플별 input/output ({len(data)}건)")
    print(f"  summary.json         — 루브릭별/overall 성능")


def main():
    api_model      = CONF.get("PRM", {}).get("model_id") or CONF.get("API_model", {}).get("PRM")
    if not api_model:
        raise ValueError("config.yaml의 PRM.model_id에 모델 이름을 설정해 주세요.")
    max_new_tokens = CONF.get("PRM", {}).get("max_new_tokens", 4096)

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=SAMPLE_START,
                        help="클래스별 슬라이스 시작 인덱스 (포함, 기본: SAMPLE_START)")
    parser.add_argument("--end", type=int, default=SAMPLE_END,
                        help="클래스별 슬라이스 끝 인덱스 (미포함, 기본: SAMPLE_END)")
    parser.add_argument("--deep_rubric_file", type=str, default=None,
                        help="딥 루브릭 json 파일 경로.")
    parser.add_argument("--fast_rubric_file", type=str, default=None,
                        help="배치 루브릭 JSON 경로. 지정 시 ApiPrmBatch(stage1) 모드로 실행. "
                             "예: prompts/fast_rubric_v6.9.json")
    parser.add_argument("--data_file", type=str, default=None,
                        help="평가할 데이터 JSON 파일 경로 (기본: output/deepmath_100.json)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="API 호출 최대 토큰 수 (기본: config PRM.max_new_tokens)")
    parser.add_argument("--rubric_index", type=int, nargs="+", default=None,
                        help="사용할 루브릭 번호 리스트 (1-indexed, 스페이스 구분). "
                             "예: --rubric_index 11  또는  --rubric_index 1 3 10 11")
    args = parser.parse_args()

    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens

    data = load_data(args.start, args.end, path=args.data_file)

    def _slice_rubrics(rubrics: list[dict]) -> list[dict]:
        """루브릭 선택 우선순위: --rubric_index > DEEP_RUBRIC_INDICES > 전체."""
        total = len(rubrics)
        indices = args.rubric_index if args.rubric_index is not None else (
            list(DEEP_RUBRIC_INDICES) if DEEP_RUBRIC_INDICES is not None else None
        )
        if indices is not None:
            selected = [rubrics[i - 1] for i in indices if 1 <= i <= total]
            logger.info(
                f"루브릭 선택 ({indices}): "
                f"{len(selected)}개 / 전체 {total}개  → {[r['name'] for r in selected]}"
            )
            return selected
        return rubrics

    # ── 배치 루브릭 모드 ──────────────────────────────────────────────────────
    if args.fast_rubric_file:
        fast_rubric = load_fast_rubric(Path(args.fast_rubric_file))
        if args.rubric_index:
            fast_rubric = _rebuild_fast_rubric_for_indices(Path(args.fast_rubric_file), args.rubric_index)
            logger.info(
                f"루브릭 선택 (--rubric_index={args.rubric_index}): "
                f"{len(fast_rubric['rubric_names'])}개 → {fast_rubric['rubric_names']}"
            )
        batch_model = ApiPrmBatch(api_model, fast_rubric, max_workers=len(data))
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir     = ROOT / "output" / "PRM" / f"{timestamp}_batch_{Path(args.fast_rubric_file).stem}"
        _run_batch_rubric_experiment(batch_model, data, max_new_tokens, out_dir,
                                     rubric_file=args.fast_rubric_file)
        batch_model.print_cost()
        return

    # ── 딥 루브릭 모드 ────────────────────────────────────────────────────────
    deep_rubric_file_path = Path(args.deep_rubric_file) if args.deep_rubric_file else DEEP_RUBRIC_FILES[0]
    logger.info(f"딥 루브릭 파일: {deep_rubric_file_path.name}")

    shared_model = ApiPrm(api_model)

    rubrics = load_deep_rubrics(deep_rubric_file_path)
    rubrics = _slice_rubrics(rubrics)
    logger.info(f"평가 루브릭 ({len(rubrics)}개): {[r['name'] for r in rubrics]}")

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    deep_rubric_ver = deep_rubric_file_path.stem.replace(".", "_")
    out_dir         = ROOT / "output" / "PRM" / f"{timestamp}_{deep_rubric_ver}"
    logger.info(f"출력 디렉토리: {out_dir}")

    logger.info(f"max_new_tokens: {max_new_tokens}")
    all_results, inference_elapsed = run_experiment(
        model_path=api_model,
        cache_dir="",
        gpu_ids=[],
        batch_size=len(data) * len(rubrics),
        data=data,
        rubrics=rubrics,
        max_new_tokens=max_new_tokens,
        model=shared_model,
    )
    eval_names = {r["name"] for r in rubrics[:10]}
    metrics_by_rubric = print_results(all_results, api_model, eval_names=eval_names)

    if len(all_results) >= 2:
        corr = analyze_rubric_correlation(all_results)
        print_correlation(corr, list(all_results.keys()))
    else:
        corr = {}

    save_results(out_dir, all_results, metrics_by_rubric, corr, api_model, rubrics,
                 inference_elapsed=inference_elapsed, rubric_file=deep_rubric_file_path)
    print(f"\n결과 저장 완료: {out_dir}")

    shared_model.print_cost()


if __name__ == "__main__":
    main()
