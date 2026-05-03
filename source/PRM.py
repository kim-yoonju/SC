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

import os
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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import _call_llm, CONF, set_call_role

_DEFAULT_PRM_MODEL   = CONF.get("PRM", {}).get("model_id_checklist") or CONF.get("PRM", {}).get("model_id_batch")
_DEFAULT_RUBRIC_FILE = CONF.get("PRM", {}).get("rubric_file")

# ─────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────────────────────────────────────
# 클래스별 슬라이스 범위: train=0~25, test=25~50
SAMPLE_START = 0   # 시작 인덱스 (포함)
SAMPLE_END   = 25   # 끝 인덱스 (미포함)

ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "output" / "deepmath_100.json"
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
RUBRIC_VERSIONS = ["6.0",]  # 예: ["4.0", "4.1", "4.2"]
RUBRIC_FILES = [ROOT / "prompts" / f"prm_rubric_v{v}.jsonl" for v in RUBRIC_VERSIONS]



def load_rubrics(path: Path | str | None = None) -> list[dict]:
    """JSONL 파일에서 루브릭 목록 로드. 각 항목에 name, criterion, system_prompt 포함."""
    path = Path(path) if path else RUBRIC_FILES[0]
    if not path.exists():
        logger.error(f"루브릭 파일 없음: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        rubrics = [json.loads(line) for line in f if line.strip()]
    logger.info(f"루브릭 로드: {path.name}  ({len(rubrics)}개)")
    return rubrics


_COT_SUFFIX = """
Before your verdict, explicitly show your reasoning:
1. Which part of the rubric applies to this step (GATE check)
2. What you actually verified or computed
3. What error you found (if any)
Then end with Verdict: correct or Verdict: incorrect."""

def build_system_prompt(rubric: dict, cot: bool = False) -> str:
    """루브릭 dict의 system_prompt 필드를 반환. cot=True면 추론 과정 명시 지시문 추가."""
    prompt = rubric["system_prompt"]
    if cot:
        prompt = prompt + _COT_SUFFIX
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
        pred = "pass" if "correct" in token and "incorrect" not in token else "fail"
        p_c  = 1.0 if pred == "pass" else 0.0
        p_i  = 1.0 - p_c
    else:
        pred = "pass" if p_c >= p_i else "fail"

    logger.debug(f"[forced_verdict] token={token!r}  pred={pred}  p_correct={p_c:.3f}")
    return {
        "reasoning":      truncated_resp,
        "verdict_text":   f"Verdict: {token} [forced]",
        "full_response":  truncated_resp,
        "prob_correct":   p_c,
        "prob_incorrect": p_i,
        "reward":         p_c,
        "pred":           pred,
        "method":         "forced_verdict",
    }


def _parse_verdict(response: str) -> dict:
    """API 응답에서 correct/incorrect verdict 파싱."""
    if "</think>" in response:
        reasoning   = response.split("</think>")[0].replace("<think>", "").strip()
        after_think = response.split("</think>", 1)[1].strip()
    else:
        reasoning   = response
        after_think = response

    m = re.search(r"verdict[:\s]+(\w+)", after_think, re.I)
    if m:
        word = m.group(1).lower()
    else:
        words = re.findall(r"\b(correct|incorrect)\b", after_think, re.I)
        word  = words[-1].lower() if words else "incorrect"

    pred           = "pass" if word == "correct" else "fail"
    prob_correct   = 1.0 if pred == "pass" else 0.0
    prob_incorrect = 1.0 - prob_correct
    return {
        "reasoning":      reasoning,
        "verdict_text":   after_think[:200],
        "full_response":  response,
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
    ) -> list[dict]:
        def _call_one(args):
            q, prev, now, sys_prompt = args
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": build_user_message(q, prev, now)},
            ]
            usage_out    = []
            logprobs_out = []
            t0 = time.time()
            try:
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

        items   = list(zip(questions, prev_steps, now_steps, system_prompts))
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
    delimiter 방식(<<<START_RUBRIC_N>>> ... <<<END_RUBRIC_N>>>)을 우선 시도하고,
    없으면 [RUBRIC N] 마커 방식으로 fallback.
    full_response는 전체 응답이 아닌 해당 루브릭 블록만 저장.
    """
    n = len(rubric_names)
    results = []
    for i, name in enumerate(rubric_names, 1):
        # ── delimiter 방식 우선 ────────────────────────────────────────────────
        delim_pat = rf'<<<START_RUBRIC_{i}>>>(.*?)<<<END_RUBRIC_{i}>>>'
        m = re.search(delim_pat, response, re.DOTALL | re.IGNORECASE)
        if m:
            block = m.group(1).strip()
        else:
            # ── [RUBRIC N] 마커 방식 fallback ─────────────────────────────────
            if i < n:
                pat = rf'\[RUBRIC\s+{i}\][^\n]*\n(.*?)(?=<<<START_RUBRIC_{i+1}>>>|\[RUBRIC\s+{i+1}\])'
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
            _critique = _l
            break

        pred = "pass" if word == "correct" else "fail"
        results.append({
            "reasoning":      block,
            "verdict_text":   block[:200],
            "full_response":  block,          # 전체 아닌 해당 루브릭 블록만 저장
            "prob_correct":   1.0 if pred == "pass" else 0.0,
            "prob_incorrect": 0.0 if pred == "pass" else 1.0,
            "reward":         0.0 if pred == "pass" else 1.0,
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
    ) -> list[list[dict]]:
        """각 샘플에 대해 1번 API 호출. 반환: [[rubric0_verdict, ...], ...]"""
        def _call_one(args):
            q, prev, now = args
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": build_user_message(q, prev, now)},
            ]
            usage_out = []
            t0 = time.time()
            try:
                set_call_role("fast_rubric")
                resp = _call_llm(self.model_name, messages, max_completion_tokens=max_new_tokens, usage_out=usage_out)
                resp = resp.strip() if resp else ""
            except Exception as e:
                self._log.warning(f"호출 실패: {e}")
                resp = ""
            return _parse_batch_verdict(resp, self.rubric_names), usage_out, time.time() - t0

        items   = list(zip(questions, prev_steps, now_steps))
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
) -> tuple[str, dict]:
    """단일 스텝을 루브릭 hard voting으로 평가.

    rubrics, model 미지정 시 config.yaml의 PRM.rubric_file / API_model.PRM 사용.
    fail_k개 이상의 루브릭이 'fail' → "fail" 반환, 아니면 "pass".

    Returns:
        verdict : "pass" or "fail"
        detail  : {rubric_name: full result dict} — pred/reasoning/verdict_text 등 포함
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
        rubrics = load_rubrics(rubric_path)

    if not rubrics:
        return "pass", {}

    results = model.evaluate_batch(
        questions=[question] * len(rubrics),
        prev_steps=[prev_steps] * len(rubrics),
        now_steps=[now_step] * len(rubrics),
        system_prompts=[build_system_prompt(r, cot=cot) for r in rubrics],
        max_new_tokens=max_new_tokens,
    )
    detail  = {r["name"]: res for r, res in zip(rubrics, results)}
    n_fail  = sum(1 for res in detail.values() if res["pred"] == "fail")
    verdict = "fail" if n_fail >= fail_k else "pass"
    return verdict, detail


# ─────────────────────────────────────────────────────────────────────────────
# 메트릭
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """accuracy / precision / recall / F1 계산 (positive class = "fail")."""
    tp = fp = tn = fn = 0
    for r in results:
        pred, label = r["pred"], r["label"]
        if pred is None or label not in ("pass", "fail"):
            continue
        if pred == "fail" and label == "fail":
            tp += 1
        elif pred == "fail" and label == "pass":
            fp += 1
        elif pred == "pass" and label == "pass":
            tn += 1
        elif pred == "pass" and label == "fail":
            fn += 1

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


def compute_ensemble_metrics(all_results: dict[str, list[dict]]) -> dict:
    """
    Hard voting (k) 및 Soft voting (threshold) 앙상블 성능 계산.

    Hard voting  : k개 이상 루브릭이 'fail' 예측 → fail  (key: ">=k")
    Soft voting  : 전체 루브릭 평균 reward >= threshold → fail
                   (reward 높을수록 fail 가능성 높음, threshold: 0.60 / 0.65 / 0.70)

    반환 dict 구조:
      hard_voting : {">=k": {accuracy, precision, recall, f1, tp, fp, tn, fn}}
      soft_voting : {"thr=0.30": {accuracy, precision, recall, f1, tp, fp, tn, fn}, ...}
    """
    import numpy as np

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
            pred  = "fail" if s["n_fail_pred"] >= k else "pass"
            label = s["label"]
            if label not in ("pass", "fail"):
                continue
            if pred == "fail" and label == "fail":   tp += 1
            elif pred == "fail" and label == "pass": fp += 1
            elif pred == "pass" and label == "pass": tn += 1
            else:                                    fn += 1
        hard_voting[f">={k}"] = _metrics_from_counts(tp, fp, tn, fn)

    # ── Soft voting ───────────────────────────────────────────────
    # 고정 threshold: 0.60 / 0.65 / 0.70  (avg_reward >= threshold → fail)
    soft_voting: dict[str, dict] = {}
    for thr in [0.60, 0.65, 0.70]:
        tp = fp = tn = fn = 0
        for s in samples:
            pred  = "fail" if s["avg_reward"] >= thr else "pass"
            label = s["label"]
            if label not in ("pass", "fail"):
                continue
            if pred == "fail" and label == "fail":   tp += 1
            elif pred == "fail" and label == "pass": fp += 1
            elif pred == "pass" and label == "pass": tn += 1
            else:                                    fn += 1
        soft_voting[f"thr={thr:.2f}"] = _metrics_from_counts(tp, fp, tn, fn)

    return {
        "hard_voting": hard_voting,
        "soft_voting": soft_voting,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드 및 샘플링
# ─────────────────────────────────────────────────────────────────────────────

def load_data(start: int = SAMPLE_START, end: int = SAMPLE_END) -> list[dict]:
    with open(DATA_PATH) as f:
        raw = json.load(f)

    pass_samples = [d for d in raw if str(d.get("gold_answer", "")).strip() == "Yes"]
    fail_samples = [d for d in raw if str(d.get("gold_answer", "")).strip() == "No"]

    pass_samples = pass_samples[start:end]
    fail_samples = fail_samples[start:end]

    data = pass_samples + fail_samples
    random.seed(42)
    random.shuffle(data)
    for i, d in enumerate(data):
        d["sample_idx"] = i
    logger.info(
        f"데이터 로드: pass={len(pass_samples)}  fail={len(fail_samples)}  total={len(data)}"
        f"  (클래스별 [{start}:{end}])"
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
                {"reasoning": None, "verdict_text": "", "full_response": None,
                 "prob_correct": None, "prob_incorrect": None, "pred": None, "method": None}
                for _ in batch_pairs
            ]

        elapsed = time.time() - t0
        logger.info(
            f"배치 {batch_idx+1}/{n_batches} 완료 ({elapsed:.1f}s) | "
            f"{len(batch_pairs)}쌍, {elapsed/len(batch_pairs):.1f}s/샘플"
        )

        for (rubric, item), out in zip(batch_pairs, outs):
            label = "pass" if str(item.get("gold_answer", "")).strip() == "Yes" else "fail"
            all_results[rubric["name"]].append({
                "rubric_name":    rubric["name"],
                "sample_idx":     item.get("sample_idx"),
                "question":       item.get("question", ""),
                "previous_steps": item.get("previous_steps", ""),
                "now_step":       item.get("now_step", ""),
                "gold_answer":    item.get("gold_answer", ""),
                "label":          label,
                "pred":           out["pred"],
                "is_correct":     (out["pred"] == label) if out["pred"] is not None else None,
                "prob_correct":   out["prob_correct"],
                "prob_incorrect": out["prob_incorrect"],
                "reward":         out.get("reward"),
                "reasoning":      out["reasoning"],
                "verdict_text":   out.get("verdict_text", ""),
                "full_response":  out.get("full_response", ""),
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
                aw = pa == "fail"
                bw = pb == "fail"
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

        wrong_samples = [idx for idx in sample_ids if by_sample[idx]["label"] == "fail"]
        for idx in wrong_samples:
            sample = by_sample[idx]
            target_hit   = sample.get(target) == "fail"
            others_hits  = [sample.get(o) == "fail" for o in others if sample.get(o) is not None]
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
        if by_sample[idx]["label"] != "fail":
            continue
        n_correct = sum(
            1 for r in rubric_names
            if by_sample[idx].get(r) == "fail"
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

def print_results(all_results: dict[str, list[dict]], model_path: str) -> dict:
    """루브릭별 메트릭을 비교표로 출력하고 metrics_by_rubric을 반환."""
    W = 75
    print(f"\n{'='*W}")
    print(f" 모델: {Path(model_path).name}")
    print(f"{'='*W}")
    print(f" {'Rubric':<32} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print(f" {'-'*32} {'------':>6} {'------':>6} {'------':>6} {'------':>6} {'-----':>5}")

    best_f1, best_name = -1.0, ""
    metrics_by_rubric = {}
    for rubric_name, results in all_results.items():
        m = compute_metrics(results)
        metrics_by_rubric[rubric_name] = m
        if m["f1"] > best_f1:
            best_f1, best_name = m["f1"], rubric_name
        marker = " ←" if rubric_name == best_name else ""
        print(
            f" {rubric_name:<32} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['total']:>5}{marker}"
        )

    print(f"{'─'*W}")
    print(f" Best F1: {best_name}  ({best_f1:.4f})")
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

    ensemble = compute_ensemble_metrics(all_results)

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
            if sample.get("label") != "fail":
                continue
            only_this = (sample.get(rb_name) == "fail") and all(
                sample.get(o) != "fail" for o in all_rubric_names if o != rb_name
            )
            if only_this:
                unique_samples.append(idx)
        unique_samples.sort()
        uc_entry = corr.get("unique_contribution", {}).get(rb_name, {})
        metrics_by_rubric[rb_name]["unique_find"]         = len(unique_samples)
        metrics_by_rubric[rb_name]["unique_find_rate"]    = uc_entry.get("unique_hit_rate", 0.0)
        metrics_by_rubric[rb_name]["unique_find_samples"] = unique_samples

    summary = {
        "model":               model_path,
        "timestamp":           out_dir.name,
        "rubric_file":         str(rubric_file),
        "n_rubrics":           len(rubrics),
        "rubrics":             [r["name"] for r in rubrics],
        "inference_elapsed_s": round(inference_elapsed, 1),
        "metrics":             metrics_by_rubric,
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

def main():
    api_model      = CONF.get("API_model", {}).get("PRM")
    if not api_model:
        raise ValueError("config.yaml의 PRM.model_id_checklist에 모델 이름을 설정해 주세요.")
    max_new_tokens = CONF.get("API_model", {}).get("max_new_tokens", 4096)

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=SAMPLE_START,
                        help="클래스별 슬라이스 시작 인덱스 (포함, 기본: SAMPLE_START)")
    parser.add_argument("--end", type=int, default=SAMPLE_END,
                        help="클래스별 슬라이스 끝 인덱스 (미포함, 기본: SAMPLE_END)")
    parser.add_argument("--rubric_file", type=str, default=None,
                        help="루브릭 jsonl 파일 경로 (단일). --rubric_files와 중복 시 --rubric_files 우선. "
                             "미지정 시 파일 상단 RUBRIC_VERSIONS 목록 전체 실험.")
    parser.add_argument("--rubric_files", type=str, nargs="+", default=None,
                        help="루브릭 jsonl 파일 경로 리스트 (스페이스 구분). "
                             "예: prompts/v1.jsonl prompts/v2.jsonl prompts/v3.jsonl")
    parser.add_argument("--rubrics", type=str, default=None,
                        help="평가할 루브릭 이름 (콤마 구분, 미지정 시 파일 전체). "
                             "예: 'Step-Goal Alignment,Result Range Validity'")
    args = parser.parse_args()

    # 루브릭 파일 리스트 결정 (CLI > 파일 상단 RUBRIC_VERSIONS 순)
    if args.rubric_files:
        rubric_file_list = [Path(p) for p in args.rubric_files]
    elif args.rubric_file:
        rubric_file_list = [Path(args.rubric_file)]
    else:
        rubric_file_list = RUBRIC_FILES

    logger.info(f"실험할 루브릭 파일 {len(rubric_file_list)}개: {[p.name for p in rubric_file_list]}")

    data         = load_data(args.start, args.end)
    shared_model = ApiPrm(api_model)

    for rubric_file_path in rubric_file_list:
        rubrics = load_rubrics(rubric_file_path)
        if args.rubrics:
            selected = {n.strip() for n in args.rubrics.split(",")}
            rubrics = [r for r in rubrics if r["name"] in selected]
            if not rubrics:
                logger.warning(f"[{rubric_file_path.name}] 일치하는 루브릭 없음 — 건너뜀.")
                continue
        logger.info(f"[{rubric_file_path.name}] 평가 루브릭 ({len(rubrics)}개): {[r['name'] for r in rubrics]}")

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        rubric_ver = rubric_file_path.stem.replace(".", "_")
        out_dir    = ROOT / "output" / "PRM" / f"{timestamp}_{rubric_ver}"
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
        metrics_by_rubric = print_results(all_results, api_model)

        if len(all_results) >= 2:
            corr = analyze_rubric_correlation(all_results)
            print_correlation(corr, list(all_results.keys()))
        else:
            corr = {}

        save_results(out_dir, all_results, metrics_by_rubric, corr, api_model, rubrics,
                     inference_elapsed=inference_elapsed, rubric_file=rubric_file_path)
        print(f"\n[{rubric_file_path.name}] 결과 저장 완료: {out_dir}")

    shared_model.print_cost()


if __name__ == "__main__":
    main()
