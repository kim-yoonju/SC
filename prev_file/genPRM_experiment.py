"""
experiment_patcher_local.py

로컬 genPRM 모델로 step 정오 예측 정확도를 측정하는 실험 스크립트.

평가 방식:
  1단계(추론 생성): 프롬프트 → 모델이 추론 텍스트 생성
  2단계(verdict 확률): [프롬프트 + 추론 + "Verdict: "] 입력 후
                       마지막 위치에서 "correct" / "incorrect" 토큰의
                       logprob을 비교해 이진 분류 수행

wrong 레이블 데이터의 경우 now_step이 첫 번째 오류 스텝이므로,
모델이 "incorrect"에 더 높은 확률을 부여하면 올바른 예측으로 간주.

Usage:
    python source/experiment_patcher_local.py
    python source/experiment_patcher_local.py --model_path /path/to/model --gpu 0
    python source/experiment_patcher_local.py --start 0  --end 25   # train
    python source/experiment_patcher_local.py --start 25 --end 50   # test

루브릭 5/10 이상 pass하면 진짜 다음 스텝으로 넘어갈때
f1-score 0.9, recall 0.9정도 나옴
근데 병렬처리해도 시간이 너무 오래걸려서 다른 모델 (deepseek-70b로 대체)
"""

import os

# CUDA_VISIBLE_DEVICES는 torch import 전에 설정해야 하므로 config를 직접 파싱
import yaml as _yaml
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
with open(_CONFIG_PATH) as _f:
    _cfg = _yaml.safe_load(_f)
_gpu_ids = _cfg.get("PRM", {}).get("gpu_id", [0])
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in _gpu_ids)

import argparse
import json
import logging
import math
import multiprocessing as mp
import re
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────────────────────────────────────
# 클래스별 슬라이스 범위: train=0~25, test=25~50
SAMPLE_START = 0   # 시작 인덱스 (포함)
SAMPLE_END   = 25   # 끝 인덱스 (미포함)

ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "output" / "deepmath_100.json"
CONFIG_PATH = ROOT / "config" / "config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 루브릭 파일
# ─────────────────────────────────────────────────────────────────────────────

RUBRIC_FILE = ROOT / "prompts" / "prm_rubric_v3.6.jsonl"


def load_rubrics(path: Path | str | None = None) -> list[dict]:
    """JSONL 파일에서 루브릭 목록 로드. 각 항목에 name, criterion, system_prompt 포함."""
    path = Path(path) if path else RUBRIC_FILE
    if not path.exists():
        logger.error(f"루브릭 파일 없음: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        rubrics = [json.loads(line) for line in f if line.strip()]
    logger.info(f"루브릭 로드: {path.name}  ({len(rubrics)}개)")
    return rubrics


def build_system_prompt(rubric: dict) -> str:
    """루브릭 dict의 system_prompt 필드를 그대로 반환."""
    return rubric["system_prompt"]

_FIRST_STEP_MARKER = "Since the Now Step is the first step"


def build_user_message(question: str, previous_steps: str, now_step: str) -> str:
    has_prev = bool(previous_steps and _FIRST_STEP_MARKER not in previous_steps)
    parts = [f"Problem:\n{question}"]
    if has_prev:
        parts.append(f"Previous steps (confirmed correct):\n{previous_steps}")
    parts.append(f"Current step to evaluate:\n{now_step}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────────────────────────

class LocalGenPRM:
    def __init__(self, model_path: str, cache_dir: str, batch_size: int, gpu_device: int = 0):
        self.device = f"cuda:{gpu_device}"
        self.batch_size = batch_size

        t_start = time.time()
        logger.info(f"[cuda:{gpu_device}] 토크나이저 로드 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        t_tok = time.time()
        logger.info(f"[cuda:{gpu_device}] 토크나이저 로드 완료 ({t_tok - t_start:.1f}s)")

        logger.info(f"[cuda:{gpu_device}] 모델 로드 중: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": gpu_device},
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.model.eval()
        t_model = time.time()
        logger.info(f"[cuda:{gpu_device}] 모델 로드 완료 ({t_model - t_tok:.1f}s)  총 로드: {t_model - t_start:.1f}s")

        self._tok_correct   = self._find_token_id("correct")
        self._tok_incorrect = self._find_token_id("incorrect")
        logger.info(
            f"[cuda:{gpu_device}] verdict 토큰 ID — correct: {self._tok_correct}  incorrect: {self._tok_incorrect}"
        )

    def _find_token_id(self, word: str) -> int:
        """단어에 해당하는 단일 토큰 ID를 반환. 스페이스 prefix 포함 버전도 시도."""
        for candidate in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        # fallback: 첫 번째 서브워드 토큰 사용
        return self.tokenizer.encode(word, add_special_tokens=False)[0]

    def _parse_output(self, full_output: str):
        """full_output을 파싱해 (reasoning, verdict_text, pred, prob_correct, prob_incorrect, method) 반환.
        텍스트로 판단 불가한 경우 pred=None 반환."""
        verify_match = re.search(r"<verify>(.*?)</verify>", full_output, re.DOTALL)
        output_match = re.search(r"<output>(.*?)</output>", full_output, re.DOTALL)

        if verify_match:
            reasoning = verify_match.group(1).strip()
        elif "</think>" in full_output:
            think_part, _ = full_output.split("</think>", 1)
            reasoning = think_part.replace("<think>", "").strip()
        else:
            reasoning = full_output.replace("<think>", "").strip()

        if output_match:
            verdict_text = output_match.group(1).strip()
        elif "</think>" in full_output:
            _, after_think = full_output.split("</think>", 1)
            verdict_text = after_think.strip()
        else:
            verdict_text = ""

        pred = prob_correct = prob_incorrect = method = None

        boxed_match = re.search(r"\\boxed\{([^}]+)\}", verdict_text)
        if boxed_match:
            boxed_val = boxed_match.group(1).strip().lower()
            if boxed_val in ("yes", "correct"):
                pred, prob_correct, prob_incorrect, method = "right", 1.0, 0.0, "text_parse"
            elif boxed_val in ("no", "incorrect"):
                pred, prob_correct, prob_incorrect, method = "wrong", 0.0, 1.0, "text_parse"

        if pred is None:
            verdict_lower = verdict_text.lower()
            if "incorrect" in verdict_lower:
                pred, prob_correct, prob_incorrect, method = "wrong", 0.0, 1.0, "text_parse"
            elif "correct" in verdict_lower:
                pred, prob_correct, prob_incorrect, method = "right", 1.0, 0.0, "text_parse"
            elif "yes" in verdict_lower.split():
                pred, prob_correct, prob_incorrect, method = "right", 1.0, 0.0, "text_parse"
            elif "no" in verdict_lower.split():
                pred, prob_correct, prob_incorrect, method = "wrong", 0.0, 1.0, "text_parse"

        return reasoning, verdict_text, pred, prob_correct, prob_incorrect, method

    def _logits_to_verdict_prob(self, last_logits: torch.Tensor) -> tuple[str, float, float]:
        """correct/incorrect 두 토큰의 logit만 꺼내 합=1로 정규화 후 (pred, prob_correct, prob_incorrect) 반환."""
        lp_c = last_logits[self._tok_correct].item()
        lp_i = last_logits[self._tok_incorrect].item()
        max_lp = max(lp_c, lp_i)
        exp_c  = math.exp(lp_c - max_lp)
        exp_i  = math.exp(lp_i - max_lp)
        total  = exp_c + exp_i
        prob_correct   = round(exp_c / total, 4)
        prob_incorrect = round(exp_i / total, 4)
        pred = "right" if prob_correct > prob_incorrect else "wrong"
        return pred, prob_correct, prob_incorrect

    def _logprob_from_ids(
        self, unpadded_output_ids: torch.Tensor, full_output: str
    ) -> tuple[str, float, float]:
        """unpadded_output_ids 끝에 verdict prefix를 붙여 correct/incorrect 확률 측정.
        truncated 출력(</think> 없음)의 경우 강제로 닫고 verdict 요청."""
        first_device = next(self.model.parameters()).device
        if "</verify>" in full_output:
            verdict_prefix = "</verify>\n<output>\n**Judgement**: $\\boxed{"
        elif "</think>" in full_output:
            verdict_prefix = "</think>\nVerdict: "
        else:
            verdict_prefix = "\n</think>\nVerdict: "
        verdict_ids = self.tokenizer.encode(verdict_prefix, add_special_tokens=False)
        full_ids = torch.cat(
            [unpadded_output_ids.to(first_device),
             torch.tensor(verdict_ids, device=first_device)],
            dim=0,
        ).unsqueeze(0)
        last_logits = self.model(full_ids).logits[0, -1, :]
        return self._logits_to_verdict_prob(last_logits)

    def _verdict_logprob_at_token(
        self, unpadded_ids: torch.Tensor
    ) -> tuple[str | None, float | None, float | None]:
        """시퀀스 안에서 마지막 correct/incorrect 토큰을 찾아,
        그 토큰이 생성되던 시점의 실제 확률을 반환.
        (텍스트 파싱 성공 케이스에서 1.0/0.0 대신 실제 모델 확률을 얻기 위해 사용)"""
        first_device = next(self.model.parameters()).device
        ids_list = unpadded_ids.tolist()
        verdict_pos = None
        for t in range(len(ids_list) - 1, -1, -1):
            if ids_list[t] in (self._tok_correct, self._tok_incorrect):
                verdict_pos = t
                break
        if verdict_pos is None:
            return None, None, None
        # verdict_pos 직전까지를 입력으로 forward → 마지막 logit이 verdict 토큰의 분포
        prefix_ids = unpadded_ids[:verdict_pos].unsqueeze(0).to(first_device)
        last_logits = self.model(prefix_ids).logits[0, -1, :]
        return self._logits_to_verdict_prob(last_logits)

    # ─────────────────────────────────────────────────────────────────────────
    # 2단계 추론
    # ─────────────────────────────────────────────────────────────────────────

    def _build_chat_input(self, user_message: str, system_prompt: str) -> str:
        """chat template을 적용해 입력 문자열 생성."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def evaluate(
        self,
        question: str,
        previous_steps: str,
        now_step: str,
        system_prompt: str,
        max_new_tokens: int = 8196,
    ) -> dict:
        """단일 샘플 평가. 내부적으로 evaluate_batch(batch_size=1)을 호출."""
        return self.evaluate_batch(
            questions=[question],
            prev_steps=[previous_steps],
            now_steps=[now_step],
            system_prompts=[system_prompt],
            max_new_tokens=max_new_tokens,
        )[0]

    @torch.inference_mode()
    def evaluate_batch(
        self,
        questions: list[str],
        prev_steps: list[str],
        now_steps: list[str],
        system_prompts: list[str],
        max_new_tokens: int = 8196,
    ) -> list[dict]:
        """
        여러 샘플을 한 번에 배치 추론. 각 샘플이 서로 다른 system_prompt를 가질 수 있음.

        전략:
          Pass 1: 배치 토크나이징 + 배치 생성 → 텍스트 파싱으로 verdict 결정
          Pass 2: 텍스트 파싱 실패 시 해당 샘플만 logprob fallback

        Returns:
            각 샘플에 대한 dict 리스트 (reasoning, verdict_text, prob_correct, prob_incorrect, pred, method)
        """
        first_device = next(self.model.parameters()).device

        # ── 배치 입력 구성 (샘플별 system_prompt 독립 적용) ──────────────
        chat_inputs = [
            self._build_chat_input(
                build_user_message(q, p, s),
                sp,
            )
            for q, p, s, sp in zip(questions, prev_steps, now_steps, system_prompts)
        ]

        # ── Pass 1: 배치 생성 ─────────────────────────────────────────────
        t_tok_start = time.time()
        encoded = self.tokenizer(
            chat_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(first_device)
        t_tok_end = time.time()

        input_len = encoded.input_ids.shape[1]
        batch_size = encoded.input_ids.shape[0]
        logger.info(
            f"[{first_device}] 토크나이징 완료 ({t_tok_end - t_tok_start:.2f}s) | "
            f"batch={batch_size}, input_seq_len={input_len}, max_new_tokens={max_new_tokens}"
        )

        t_gen_start = time.time()
        output_ids = self.model.generate(
            encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        t_gen_end = time.time()

        total_new_tokens = (output_ids.shape[1] - input_len) * batch_size
        gen_elapsed = t_gen_end - t_gen_start
        logger.info(
            f"[{first_device}] 생성 완료 ({gen_elapsed:.1f}s) | "
            f"output_seq_len={output_ids.shape[1]}, new_tokens/sample≈{output_ids.shape[1]-input_len}, "
            f"처리량≈{total_new_tokens/gen_elapsed:.1f} tok/s"
        )

        results = []
        n_text_parse = 0
        n_logprob_text_fail = 0
        n_logprob_truncated = 0

        pad_id = self.tokenizer.pad_token_id

        for i in range(len(questions)):
            new_ids = output_ids[i, input_len:]

            # 실제 생성 토큰 수 (trailing pad 제외) 및 clean token IDs 구성
            if pad_id is not None:
                non_pad = (new_ids != pad_id).nonzero(as_tuple=True)[0]
                actual_new_len = (non_pad[-1].item() + 1) if len(non_pad) > 0 else 0
            else:
                actual_new_len = new_ids.shape[0]
            clean_new_ids = new_ids[:actual_new_len]

            # left-padding 제거한 실제 입력 토큰
            pad_count = (encoded.attention_mask[i] == 0).sum().item()
            actual_input_i = encoded.input_ids[i, pad_count:]

            # 모든 케이스에서 공통으로 사용할 unpadded_ids (입력 + 생성, pad 없음)
            unpadded_ids = torch.cat([actual_input_i, clean_new_ids])

            full_output = self.tokenizer.decode(clean_new_ids, skip_special_tokens=True).strip()

            # ── 잘린 출력 감지: max_new_tokens 도달 + 종료 태그 없음
            is_truncated = (
                actual_new_len >= max_new_tokens
                and "</think>" not in full_output
                and "</verify>" not in full_output
            )

            reasoning  = full_output
            verdict_text = ""
            pred = prob_correct = prob_incorrect = None
            method = None

            if not is_truncated:
                # 정상 종료: 텍스트 파싱 시도
                reasoning, verdict_text, pred, _pc, _pi, method = self._parse_output(full_output)
                if pred is not None:
                    # 텍스트 파싱 성공 → 실제로 그 토큰을 생성할 때의 확률로 교체
                    n_text_parse += 1
                    _actual_pred, _actual_pc, _actual_pi = self._verdict_logprob_at_token(unpadded_ids)
                    if _actual_pred is not None:
                        pred, prob_correct, prob_incorrect = _actual_pred, _actual_pc, _actual_pi
                        method = "text_parse+logprob"
                    else:
                        # 토큰 탐색 실패 시 기존 1.0/0.0 유지
                        prob_correct, prob_incorrect = _pc, _pi

            if pred is None:
                # logprob fallback: 텍스트 파싱 실패 or truncated
                # truncated → \n</think>\nVerdict: 를 강제로 붙여 판정
                if is_truncated:
                    n_logprob_truncated += 1
                else:
                    n_logprob_text_fail += 1
                pred, prob_correct, prob_incorrect = self._logprob_from_ids(unpadded_ids, full_output)
                method = "logprob_truncated" if is_truncated else "logprob"

            # reward = prob_incorrect (incorrect일 확률 = 스텝이 틀릴 확률)
            reward = prob_incorrect

            results.append({
                "reasoning":      reasoning,
                "verdict_text":   verdict_text,
                "full_response":  full_output,
                "prob_correct":   prob_correct,
                "prob_incorrect": prob_incorrect,
                "reward":         reward,
                "pred":           pred,
                "method":         method,
            })

        n_logprob_total = n_logprob_truncated + n_logprob_text_fail
        logger.info(
            f"[{first_device}] 판정 방법 — text_parse+logprob={n_text_parse}, "
            f"logprob_truncated={n_logprob_truncated}, logprob_text_fail={n_logprob_text_fail}"
        )

        return results

    def _agg_logprob(self, logits: torch.Tensor, primary_id: int) -> float:
        """primary_id의 logit을 반환 (확장 가능한 구조)."""
        return logits[primary_id].item()


# ─────────────────────────────────────────────────────────────────────────────
# 메트릭
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """accuracy / precision / recall / F1 계산 (positive class = "wrong")."""
    tp = fp = tn = fn = 0
    for r in results:
        pred, label = r["pred"], r["label"]
        if pred is None or label not in ("right", "wrong"):
            continue
        if pred == "wrong" and label == "wrong":
            tp += 1
        elif pred == "wrong" and label == "right":
            fp += 1
        elif pred == "right" and label == "right":
            tn += 1
        elif pred == "right" and label == "wrong":
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


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드 및 샘플링
# ─────────────────────────────────────────────────────────────────────────────

def load_data(start: int = SAMPLE_START, end: int = SAMPLE_END) -> list[dict]:
    with open(DATA_PATH) as f:
        raw = json.load(f)

    # raw 파일 순서 기준으로 클래스별 분리 (안정적 슬라이싱)
    right = [d for d in raw if str(d.get("gold_answer", "")).strip() == "Yes"]
    wrong = [d for d in raw if str(d.get("gold_answer", "")).strip() == "No"]

    right = right[start:end]
    wrong = wrong[start:end]

    data = right + wrong
    random.seed(42)
    random.shuffle(data)
    for i, d in enumerate(data):
        d["sample_idx"] = i
    logger.info(
        f"데이터 로드: right={len(right)}  wrong={len(wrong)}  total={len(data)}"
        f"  (클래스별 [{start}:{end}])"
    )
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 실험 실행
# ─────────────────────────────────────────────────────────────────────────────

def _pool_worker(worker_args: tuple) -> list[dict]:
    """GPU당 하나씩 spawn되는 worker. model을 단독 GPU에 로드해 배치 처리."""
    rank, pairs, model_path, cache_dir, batch_size, max_new_tokens = worker_args

    _log = logging.getLogger(f"worker-{rank}")

    t_worker_start = time.time()
    model = LocalGenPRM(model_path, cache_dir=cache_dir, batch_size=batch_size, gpu_device=rank)
    t_load_done = time.time()
    _log.info(
        f"[worker-{rank}] ===== 모델 로드 완료: {t_load_done - t_worker_start:.1f}s | "
        f"처리 예정: {len(pairs)}쌍, batch_size={batch_size}, n_batches={math.ceil(len(pairs)/batch_size)} ====="
    )

    results = []
    n_batches = math.ceil(len(pairs) / batch_size)
    t_infer_start = time.time()
    for batch_idx, batch_start in enumerate(range(0, len(pairs), batch_size)):
        batch_pairs   = pairs[batch_start : batch_start + batch_size]
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
            _log.error(f"[배치 {batch_idx+1}/{n_batches}] 오류: {e}")
            outs = [
                {"reasoning": None, "verdict_text": "", "full_response": None,
                 "prob_correct": None, "prob_incorrect": None, "pred": None, "method": None}
                for _ in batch_pairs
            ]

        elapsed = time.time() - t0
        _log.info(
            f"[worker-{rank}] 배치 {batch_idx+1}/{n_batches} 완료 ({elapsed:.1f}s) | "
            f"{len(batch_pairs)}쌍, {elapsed/len(batch_pairs):.1f}s/샘플"
        )

        for (rubric, item), out in zip(batch_pairs, outs):
            label = "right" if str(item.get("gold_answer", "")).strip() == "Yes" else "wrong"
            results.append({
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

    t_worker_end = time.time()
    t_infer_total = t_worker_end - t_infer_start
    t_total = t_worker_end - t_worker_start
    _log.info(
        f"[worker-{rank}] ===== 완료 ===== "
        f"총={t_total:.1f}s | 모델로드={t_load_done - t_worker_start:.1f}s ({(t_load_done-t_worker_start)/t_total*100:.0f}%) | "
        f"추론={t_infer_total:.1f}s ({t_infer_total/t_total*100:.0f}%) | "
        f"{len(pairs)}쌍 → {t_infer_total/len(pairs):.1f}s/샘플"
    )
    return results


def run_experiment(
    model_path: str,
    cache_dir: str,
    gpu_ids: list[int],
    batch_size: int,
    data: list[dict],
    rubrics: list[dict],
    max_new_tokens: int = 8196,
) -> dict[str, list[dict]]:
    """
    GPU당 모델 1개씩 로드해 data parallelism으로 (루브릭, 샘플) 쌍을 처리.
    pairs를 GPU 수만큼 균등 분할 → 각 worker가 독립적으로 배치 처리.
    """
    all_pairs: list[tuple[dict, dict]] = [
        (rubric, item) for rubric in rubrics for item in data
    ]
    total   = len(all_pairs)
    n_gpus  = len(gpu_ids)
    # GPU 수만큼 균등 분할 (라운드로빈)
    chunks  = [all_pairs[i::n_gpus] for i in range(n_gpus)]

    logger.info(
        f"전체 {total}쌍 ({len(rubrics)}루브릭 × {len(data)}샘플) "
        f"→ {n_gpus}개 GPU에 분산 (GPU당 ~{math.ceil(total/n_gpus)}쌍, batch_size={batch_size})"
    )

    worker_args = [
        (rank, chunks[rank], model_path, cache_dir, batch_size, max_new_tokens)
        for rank in range(n_gpus)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_gpus) as pool:
        chunk_results = pool.map(_pool_worker, worker_args)

    # rubric별로 재조합
    all_results: dict[str, list[dict]] = {r["name"]: [] for r in rubrics}
    for worker_results in chunk_results:
        for row in worker_results:
            all_results[row["rubric_name"]].append(row)

    for name in all_results:
        all_results[name].sort(key=lambda x: (x.get("sample_idx") or 0))

    return all_results


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
    # sample_idx → {rubric_name: pred}
    by_sample: dict[int, dict[str, str | None]] = {}
    for name, results in all_results.items():
        for r in results:
            idx = r["sample_idx"]
            if idx not in by_sample:
                by_sample[idx] = {"label": r["label"]}
            by_sample[idx][name] = r["pred"]

    sample_ids = sorted(by_sample.keys())
    n = len(rubric_names)

    # ── pairwise agreement & phi ──────────────────────────────────────────
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
                # a=both wrong, b=A wrong B right, c=A right B wrong, d=both right
                # (positive class = "wrong")
                aw = pa == "wrong"
                bw = pb == "wrong"
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

    # ── unique contribution (wrong 레이블 샘플에서) ────────────────────────
    unique_contribution: dict[str, dict] = {}
    for target in rubric_names:
        others = [r for r in rubric_names if r != target]
        unique_hits = shared_hits = miss_when_others = 0

        wrong_samples = [idx for idx in sample_ids if by_sample[idx]["label"] == "wrong"]
        for idx in wrong_samples:
            sample = by_sample[idx]
            target_hit   = sample.get(target) == "wrong"   # wrong 레이블을 wrong으로 예측 = 정답
            others_hits  = [sample.get(o) == "wrong" for o in others if sample.get(o) is not None]
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

    # ── 샘플별 루브릭 합의 분포 ────────────────────────────────────────────
    # 각 wrong 샘플을 몇 개 루브릭이 맞췄는지 집계
    coverage_dist: dict[int, int] = {}   # {맞춘_루브릭_수: 샘플_수}
    for idx in sample_ids:
        if by_sample[idx]["label"] != "wrong":
            continue
        n_correct = sum(
            1 for r in rubric_names
            if by_sample[idx].get(r) == "wrong"
        )
        coverage_dist[n_correct] = coverage_dist.get(n_correct, 0) + 1

    return {
        "agreement_matrix":  agreement_matrix,
        "phi_matrix":        phi_matrix,
        "unique_contribution": unique_contribution,
        "wrong_sample_coverage_dist": {str(k): v for k, v in sorted(coverage_dist.items())},
    }


def print_correlation(corr: dict, rubric_names: list[str]):
    """상관관계 분석 결과를 콘솔에 출력."""
    W = 75
    short = {r: r[:18] for r in rubric_names}  # 출력 너비 맞춤

    # agreement matrix
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

    # phi matrix
    print(f"\n{'─'*W}")
    print(" [Phi Coefficient Matrix]  +1=완전일치, 0=독립, -1=반대  (positive class=wrong)")
    print(f"{'─'*W}")
    print(header)
    for ra in rubric_names:
        row = f" {short[ra]:<20}" + "".join(
            f"{corr['phi_matrix'][ra][rb]:>10.3f}" for rb in rubric_names
        )
        print(row)

    # unique contribution
    print(f"\n{'─'*W}")
    print(" [Unique Contribution]  wrong 샘플 기준 — 해당 루브릭만 맞추는 샘플 비율")
    print(f"{'─'*W}")
    print(f" {'Rubric':<32} {'Unique':>8} {'Shared':>8} {'MissWhenOthers':>16} {'UniqueRate':>12}")
    uc = corr["unique_contribution"]
    for r in rubric_names:
        u = uc[r]
        print(
            f" {r:<32} {u['unique_hits']:>8} {u['shared_hits']:>8} "
            f"{u['miss_when_others']:>16} {u['unique_hit_rate']:>12.4f}"
        )

    # coverage dist
    print(f"\n{'─'*W}")
    print(" [Wrong 샘플 커버리지]  'k개 루브릭이 맞춘 wrong 샘플 수'")
    print(f"{'─'*W}")
    for k, cnt in sorted(corr["wrong_sample_coverage_dist"].items(), key=lambda x: int(x[0])):
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
):
    """out_dir 안에 루브릭별 jsonl + summary.json + prompts.jsonl 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 루브릭별 jsonl
    for rubric_name, results in all_results.items():
        safe_name = rubric_name.replace(" ", "_").replace("/", "-")
        jsonl_path = out_dir / f"{safe_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"  저장: {jsonl_path.name}  ({len(results)}건)")

    # prompts.jsonl — 실험에 사용된 시스템 프롬프트 전체 텍스트를 루브릭별로 저장
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

    # summary.json
    summary = {
        "model":          model_path,
        "timestamp":      out_dir.name,
        "rubric_file":    str(RUBRIC_FILE),
        "n_rubrics":      len(rubrics),
        "rubrics":        [r["name"] for r in rubrics],
        "metrics":        metrics_by_rubric,
        "correlation":    corr,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"  요약 저장: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    prm_cfg        = config.get("PRM", {})
    default_model  = prm_cfg.get("model_id", "GenPRM/GenPRM-7B")
    cache_dir      = config.get("checkpoint", {}).get("cache_dir", "/tmp")
    gpu_ids        = prm_cfg.get("gpu_id", [0])
    batch_size     = prm_cfg.get("batch_size", 32)
    max_new_tokens = prm_cfg.get("max_new_tokens", 8196)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=default_model,
                        help="로컬 모델 경로 또는 HuggingFace 모델 ID")
    parser.add_argument("--start", type=int, default=SAMPLE_START,
                        help="클래스별 슬라이스 시작 인덱스 (포함, 기본: SAMPLE_START)")
    parser.add_argument("--end", type=int, default=SAMPLE_END,
                        help="클래스별 슬라이스 끝 인덱스 (미포함, 기본: SAMPLE_END)")
    parser.add_argument("--rubric_file", type=str, default=str(RUBRIC_FILE),
                        help=f"루브릭 jsonl 파일 경로 (기본: {RUBRIC_FILE.name})")
    parser.add_argument("--rubrics", type=str, default=None,
                        help="평가할 루브릭 이름 (콤마 구분, 미지정 시 파일 전체). "
                             "예: 'Step-Goal Alignment,Result Range Validity'")
    args = parser.parse_args()

    sample_start = args.start
    sample_end   = args.end

    # 루브릭 로드
    rubrics = load_rubrics(args.rubric_file)
    if args.rubrics:
        selected = {n.strip() for n in args.rubrics.split(",")}
        rubrics = [r for r in rubrics if r["name"] in selected]
        if not rubrics:
            logger.error(f"일치하는 루브릭 없음.")
            sys.exit(1)
    logger.info(f"평가 루브릭 ({len(rubrics)}개): {[r['name'] for r in rubrics]}")

    # 출력 디렉토리: output/genPRM/{timestamp}/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = ROOT / "output" / "genPRM" / timestamp
    logger.info(f"출력 디렉토리: {out_dir}")

    data = load_data(sample_start, sample_end)

    logger.info(f"max_new_tokens: {max_new_tokens} (config PRM.max_new_tokens)")
    all_results = run_experiment(
        model_path=args.model_path,
        cache_dir=cache_dir,
        gpu_ids=gpu_ids,
        batch_size=batch_size,
        data=data,
        rubrics=rubrics,
        max_new_tokens=max_new_tokens,
    )
    metrics_by_rubric = print_results(all_results, args.model_path)

    # 상관관계 분석 (루브릭 2개 이상일 때만)
    if len(all_results) >= 2:
        corr = analyze_rubric_correlation(all_results)
        print_correlation(corr, list(all_results.keys()))
    else:
        corr = {}

    save_results(out_dir, all_results, metrics_by_rubric, corr, args.model_path, rubrics)
    print(f"\n결과 저장 완료: {out_dir}")


if __name__ == "__main__":
    main()
