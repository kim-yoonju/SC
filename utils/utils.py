"""
prototype/utils.py
공통 유틸리티: 하이퍼파라미터 로드, 모델 로드, GPT 헬퍼, 수학 판정 및 생성 로직
"""

import json
import logging
import os
import pathlib as _pathlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait, FIRST_COMPLETED, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from openai import OpenAI

# 로깅 설정
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 설정 및 하이퍼파라미터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path=None):
    """설정 파일을 로드합니다."""
    if config_path is None:
        config_path = _pathlib.Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    config_path = _pathlib.Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config 파일을 찾을 수 없습니다: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"config 파일이 비어 있습니다: {config_path}")
    return cfg

# 실행 시점에 설정 로드 (utils.py 위치 기준 절대경로)
CONF = load_config()

# API 키 (환경변수 우선, 없으면 config; gpt 키 없으면 KeyError)
GPT_API_KEY      = os.environ.get("OPENAI_API_KEY")   or CONF["API_key"]["gpt"]
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or CONF["API_key"].get("deepseek")

# API 클라이언트
import httpx as _httpx
client          = OpenAI(api_key=GPT_API_KEY)
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    http_client=_httpx.Client(
        headers={"Accept-Encoding": "gzip, deflate"},  # brotli 비활성화
        limits=_httpx.Limits(keepalive_expiry=20),     # 서버 idle timeout(~30s)보다 짧게 설정해 stale connection 방지
    ),
) if DEEPSEEK_API_KEY else None

# ─────────────────────────────────────────────────────────────────────────────
# 글로벌 API 비용 추적기
# ─────────────────────────────────────────────────────────────────────────────

# 가격표 ($/1M tokens): (input_normal, output, input_cached)
_API_PRICING: dict[str, tuple[float, float, float]] = {
    # (input $/1M, output $/1M, cached_input $/1M)
    "o3-mini":             (1.10,  4.40,   0.55),
    "o3":                  (10.0, 40.00,   2.50),
    "gpt-4o":              (2.50, 10.00,   1.25),
    "gpt-4o-mini":         (0.15,  0.60,   0.075),
    "deepseek-reasoner":   (0.435, 0.870,  0.03625),
    "deepseek-chat":       (0.140, 0.280,  0.00280),
    "deepseek-flash":      (0.140, 0.280,  0.00280),   # deepseek-v4-flash
    "deepseek-pro":        (0.435, 0.870,  0.003625),  # deepseek-v4-pro
    "claude":              (3.00, 15.00,   0.30),
    "gemini":              (1.25,  5.00,   0.00),
}

# alias → 실제 API 모델명
_DEEPSEEK_ALIASES: dict[str, str] = {
    "deepseek-flash": "deepseek-v4-flash",
    "deepseek-pro":   "deepseek-v4-pro",
}

_cost_lock   = threading.Lock()
_token_usage: dict[str, dict] = {}   # provider → {model, input, output, cached}


def _resolve_pricing_key(model_name: str) -> str:
    """모델 이름으로 _API_PRICING 키를 반환."""
    m = model_name.lower()
    for key in _API_PRICING:
        if key in m:
            return key
    if m.startswith("claude"):
        return "claude"
    if m.startswith("gemini"):
        return "gemini"
    return m  # 매핑 없으면 그대로 (비용 0)


_run_log_fn = None  # set via set_run_log() to capture all model calls

import contextvars as _cv
_call_role:    _cv.ContextVar[str] = _cv.ContextVar("call_role",    default="unknown")
_problem_id:   _cv.ContextVar[str] = _cv.ContextVar("problem_id",   default="?")
_step_number:  _cv.ContextVar[int] = _cv.ContextVar("step_number",  default=-1)

def set_run_log(fn):
    """콜백 fn(record: dict)을 등록하면 모든 _call_llm 호출이 기록됨."""
    global _run_log_fn
    _run_log_fn = fn

def set_call_role(role: str):
    """현재 컨텍스트의 호출 역할을 설정. _call_llm 로그의 'role' 필드에 기록됨."""
    return _call_role.set(role)

def set_problem_context(pid, step: int):
    """현재 컨텍스트의 문제 ID와 스텝 번호를 설정. _call_llm 로그에 포함됨."""
    _problem_id.set(str(pid))
    _step_number.set(int(step))

def run_log_direct(record: dict):
    """run_log_fn을 직접 호출. vLLM/로컬 등 non-API 생성기에서 직접 기록할 때 사용."""
    if _run_log_fn is not None:
        try:
            _run_log_fn(record)
        except Exception:
            pass


def _record_usage(model_name: str, usage_list: list) -> None:
    """usage 딕셔너리 리스트를 받아 모델별 토큰 집계."""
    if not usage_list:
        return
    key = _resolve_pricing_key(model_name)
    total_in     = sum(u.get("input_tokens",  0) for u in usage_list)
    total_out    = sum(u.get("output_tokens", 0) for u in usage_list)
    total_cached = sum(u.get("cached_tokens", 0) for u in usage_list)
    with _cost_lock:
        rec = _token_usage.setdefault(key, {"model": model_name, "input": 0, "output": 0, "cached": 0})
        rec["input"]  += total_in
        rec["output"] += total_out
        rec["cached"] += total_cached


def _print_cost_summary() -> None:
    """지금까지 누적된 API 비용 요약을 출력."""
    if not _token_usage:
        return
    print(f"\n{'━'*60}")
    print("  API 비용 요약")
    print(f"{'━'*60}")
    total_cost = 0.0
    for provider, rec in _token_usage.items():
        p_in, p_out, p_cached = _API_PRICING.get(provider, (0.0, 0.0, 0.0))

        cached     = rec["cached"]
        non_cached = rec["input"] - cached
        cost_in     = non_cached / 1_000_000 * p_in
        cost_cached = cached     / 1_000_000 * p_cached
        cost_out    = rec["output"] / 1_000_000 * p_out
        cost        = cost_in + cost_cached + cost_out
        saved       = cached / 1_000_000 * (p_in - p_cached)
        total_cost += cost
        print(
            f"  [{provider.upper():8s}] {rec['model']}\n"
            f"    input  {non_cached:>12,} tok  ${cost_in:.4f}\n"
            f"    cached {cached:>12,} tok  ${cost_cached:.4f}  (절약 ${saved:.4f})\n"
            f"    output {rec['output']:>12,} tok  ${cost_out:.4f}\n"
            f"    소계                      ${cost:.4f}"
        )
    print(f"{'─'*60}")
    print(f"  총 비용:                        ${total_cost:.4f}")
    print(f"{'━'*60}\n")


# Generator / checkpoint
GENERATOR_MODEL_ID       = CONF["checkpoint"]["base"]
GENERATOR_CACHE_DIR      = CONF["checkpoint"]["cache_dir"]
SFT_CHECKPOINT           = CONF["checkpoint"]["sft_checkpoint"]

# Step Manager: 요약·분해 전용 모델 (local_inference_model.gpu에 로드)
_LIM_CONF              = CONF.get("local_inference_model", {})
STEP_MANAGER_PATH      = CONF["checkpoint"]["base"]
STEP_MANAGER_GPU       = _LIM_CONF.get("gpu", 0)
STEP_MANAGER_MAX_TOKENS = _LIM_CONF.get("max_new_tokens", 1024)
GENERATOR_TEMPERATURE    = CONF.get("step_reasoning", {}).get("temperature", 1.0)
GENERATOR_MAX_NEW_TOKENS = CONF.get("step_reasoning", {}).get("max_new_tokens", 2048)
PATCHER_MAX_NEW_TOKENS   = CONF.get("PATCHER", {}).get("max_new_tokens", 2048)
PATCHER_THINKING_BUDGET  = CONF.get("PATCHER", {}).get("thinking_budget", None)
MAX_STEPS                = CONF.get("generate_trajectory", {}).get("max_steps") or CONF.get("ppo", {}).get("max_steps") or CONF.get("step_reasoning", {}).get("max_steps", 10)
VLLM_MAX_MODEL_LEN       = CONF.get("vllm", {}).get("max_model_len", 32768)
TRUNCATE_TOKEN_LIMIT     = CONF.get("step_reasoning", {}).get("truncate_token_limit", 4096)

PATCHER = CONF.get("PATCHER", {}).get("model_id")

_ROOT_PROJ = _pathlib.Path(__file__).resolve().parent.parent


def resolve_model_path(model_arg: str) -> tuple[str, str | None]:
    """
    모델 경로 or HuggingFace ID → (model_path, cache_dir) 반환.
    - 절대경로 or 존재하는 로컬 경로 → cache_dir=None
    - 프로젝트 루트 기준 상대경로 → cache_dir=None
    - HuggingFace 모델 ID → config의 cache_dir 사용
    """
    p = _pathlib.Path(model_arg)
    if p.is_absolute() or p.exists():
        return str(p), None
    local = _ROOT_PROJ / model_arg
    if local.exists():
        return str(local), None
    return model_arg, GENERATOR_CACHE_DIR


# ─────────────────────────────────────────────────────────────────────────────
# 액션 토큰
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_SOLVE   = "<|solve|>"
TOKEN_CORRECT = "<|rethink|>"
TOKEN_END     = "<|end|>"
TOKEN_NONE    = "<|none|>"
ACTION_TOKENS = [TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END, TOKEN_NONE]

# ─────────────────────────────────────────────────────────────────────────────
# 모델 로딩 및 생성 로직
# ─────────────────────────────────────────────────────────────────────────────

def load_step_manager(gpu_id: int | None = None, model_path: str | None = None):
    """Step Manager 모델 로드 (요약·분해 전용, local_inference_model.gpu에 배치).

    Returns (model, tokenizer).
    """
    import os as _os
    gpu  = gpu_id if gpu_id is not None else STEP_MANAGER_GPU
    path = model_path or STEP_MANAGER_PATH

    # CUDA_VISIBLE_DEVICES 없이 device_map으로 특정 GPU 지정
    device_map = f"cuda:{gpu}"
    logger.info(f"Step Manager 로드 중 (GPU {gpu}): {path}")
    model, tokenizer = load_generator(model_path=path, device_map=device_map)
    logger.info(f"Step Manager 로드 완료 (GPU {gpu})")
    return model, tokenizer


def load_generator(device_map="auto", model_path=None, load_in_4bit=False, max_memory=None):
    load_path = model_path if model_path else SFT_CHECKPOINT
    logger.info(f"Generator 로드 중: {load_path} (4bit={load_in_4bit})")

    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    added = tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})

    # config.vocab_size 패치: 로컬 체크포인트의 config.json과 실제 가중치 크기가 다를 수 있다.
    # safetensors 파일에서 실제 임베딩 크기를 직접 읽어 config를 패치한다.
    config = AutoConfig.from_pretrained(load_path, trust_remote_code=True)
    is_local = _pathlib.Path(load_path).exists()
    if is_local:
        import glob as _glob
        try:
            from safetensors import safe_open
            embed_key = "model.embed_tokens.weight"
            for shard in sorted(_glob.glob(os.path.join(load_path, "*.safetensors"))):
                with safe_open(shard, framework="pt", device="cpu") as f:
                    if embed_key in f.keys():
                        actual_vocab_size = f.get_slice(embed_key).get_shape()[0]
                        if config.vocab_size != actual_vocab_size:
                            logger.info(
                                f"config.vocab_size({config.vocab_size}) != 실제 임베딩 크기({actual_vocab_size}), "
                                "config 패치 후 로드"
                            )
                            config.vocab_size = actual_vocab_size
                        break
        except Exception as e:
            logger.warning(f"임베딩 크기 사전 확인 실패, config.vocab_size({config.vocab_size}) 그대로 사용: {e}")

    _extra = {"max_memory": max_memory} if max_memory is not None else {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            load_path, config=config, quantization_config=quantization_config,
            device_map=device_map, trust_remote_code=True, **_extra
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            load_path, config=config, torch_dtype=torch.bfloat16,
            device_map=device_map, trust_remote_code=True, **_extra
        )

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model, tokenizer


def load_generator_vllm(model_path=None, rollout_gpus=None):
    """vLLM LLM 로드. action token 등 special token 처리 포함.
    CUDA_VISIBLE_DEVICES는 호출 전에 이미 설정되어 있어야 함 (main()에서 처리).
    Returns (llm, tokenizer) where llm is vLLM LLM instance.
    """
    import tempfile
    from vllm import LLM

    load_path = model_path if model_path else SFT_CHECKPOINT
    logger.info(f"Generator vLLM 로드 중: {load_path}")

    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})

    # vLLM은 내부 tokenizer를 따로 로드하므로 수정된 tokenizer를 임시 디렉터리에 저장
    tmp_tok_dir = tempfile.mkdtemp(prefix="sc_vllm_tok_")
    tokenizer.save_pretrained(tmp_tok_dir)
    logger.info(f"vLLM tokenizer 임시 저장: {tmp_tok_dir}")

    tensor_parallel_size = len(rollout_gpus) if rollout_gpus else 1

    llm = LLM(
        model=load_path,
        tokenizer=tmp_tok_dir,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=0.70,
        seed=42,
        enforce_eager=True,
    )

    logger.info(f"Generator vLLM 로드 완료 (tensor_parallel={tensor_parallel_size})")
    return llm, tokenizer


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"

from utils_math import (  # noqa: E402
    extract_boxed, has_boxed, check_solved,
)

# ─────────────────────────────────────────────────────────────────────────────
# API 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(model_name: str, messages: list, max_completion_tokens: int = None, temperature: float = None,
         usage_out: list = None, response_format: dict = None, logprobs_out: list = None,
         thinking_budget: int = None) -> str:
    """DeepSeek 또는 OpenAI 호환 API를 호출합니다.

    usage_out:       제공 시 {"input_tokens": int, "output_tokens": int, "finish_reason": str} dict를 append.
    logprobs_out:    제공 시 resp.choices[0].logprobs.content (token별 top_logprobs 리스트)를 extend.
    thinking_budget: DeepSeek reasoning 모델의 thinking 토큰 예산. 설정 시 thinking enabled+budget, 미설정 시 disabled.
    """
    import time, re as _re2
    wait = 1.0
    attempt = 0
    while True:
        try:
            if model_name.lower().startswith("deepseek"):
                if deepseek_client is None:
                    raise ValueError("DeepSeek API 키가 config.API_key.deepseek에 없습니다.")
                active_client = deepseek_client
            else:
                active_client = client

            api_model_name = _DEEPSEEK_ALIASES.get(model_name.lower(), model_name)
            kwargs = {"model": api_model_name, "messages": messages}
            if max_completion_tokens:
                kwargs["max_tokens"] = max_completion_tokens
            kwargs["temperature"] = temperature if temperature is not None else 0
            if response_format is not None:
                kwargs["response_format"] = response_format
            if logprobs_out is not None:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 20
            if "deepseek" in model_name.lower() and "reasoner" not in model_name.lower():
                if thinking_budget is not None and thinking_budget > 0:
                    kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": thinking_budget}}
                else:
                    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            resp = active_client.chat.completions.create(**kwargs)

            if not resp.choices:
                logger.warning(f"[_call_llm] model={model_name}  choices=None/empty, returning None")
                return None

            choice        = resp.choices[0]
            finish_reason = choice.finish_reason
            content       = choice.message.content

            if resp.usage:
                details = resp.usage.prompt_tokens_details
                cached  = (getattr(details, "cached_tokens", 0) or 0) if details else 0
                usage_entry = {
                    "input_tokens":  resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                    "cached_tokens": cached,
                    "finish_reason": finish_reason,
                }
                _record_usage(model_name, [usage_entry])
                if usage_out is not None:
                    usage_out.append(usage_entry)

            if logprobs_out is not None and choice.logprobs and choice.logprobs.content:
                logprobs_out.extend(choice.logprobs.content)

            if not content or finish_reason not in ("stop", None):
                _logger = logging.getLogger(__name__)
                _logger.warning(
                    f"[_call_llm] model={model_name}  finish_reason={finish_reason!r}"
                    f"  content_len={len(content) if content else 0}"
                    f"  prompt_tokens={resp.usage.prompt_tokens if resp.usage else '?'}"
                    f"  completion_tokens={resp.usage.completion_tokens if resp.usage else '?'}"
                )
                if hasattr(resp.usage, 'completion_tokens_details'):
                    _logger.warning(f"[_call_llm] completion_tokens_details={resp.usage.completion_tokens_details}")

            if _run_log_fn is not None:
                try:
                    from datetime import datetime as _dt
                    _run_log_fn({
                        "ts":         _dt.now().isoformat(timespec="seconds"),
                        "role":       _call_role.get(),
                        "model":      model_name,
                        "problem_id": _problem_id.get(),
                        "step":       _step_number.get(),
                        "in_tok":     resp.usage.prompt_tokens if resp.usage else None,
                        "out_tok":    resp.usage.completion_tokens if resp.usage else None,
                        "messages":   messages,
                        "output":     content,
                    })
                except Exception:
                    pass

            return content
        except Exception as e:
            err_str = str(e)
            err_type = type(e).__name__

            # openai 예외에서 추가 정보 추출
            extra_parts = [f"type={err_type}"]
            if hasattr(e, "status_code"):
                extra_parts.append(f"status_code={e.status_code}")
            if hasattr(e, "body") and e.body:
                extra_parts.append(f"body={str(e.body)[:300]}")
            if hasattr(e, "message") and e.message:
                extra_parts.append(f"message={e.message}")
            if e.__cause__:
                extra_parts.append(f"cause={type(e.__cause__).__name__}: {e.__cause__}")
            extra_info = "  |  ".join(extra_parts)

            is_quota_exceeded = "insufficient_quota" in err_str or "billing" in err_str.lower()
            is_rate_limit = ("429" in err_str or "rate_limit" in err_str.lower()) and not is_quota_exceeded
            is_connection_error = any(kw in err_str.lower() for kw in ("connection error", "connectionerror", "connect timeout", "remotedisconnected"))
            if is_rate_limit:
                _rl_delays = [0.5, 1.0, 2.0, 4.0]
                if attempt >= len(_rl_delays):
                    logger.error(
                        f"API rate limit 최대 재시도 초과 ({model_name}, {len(_rl_delays)}회)\n"
                        f"  {extra_info}\n  err={err_str}"
                    )
                    raise e
                retry_after = _rl_delays[attempt]
                logger.warning(
                    f"API rate limit ({model_name})  attempt={attempt + 1}/{len(_rl_delays)}"
                    f"  → {retry_after}s 후 재시도\n"
                    f"  {extra_info}\n  err={err_str}"
                )
                time.sleep(retry_after)
                attempt += 1
            elif is_connection_error and attempt < 5:
                t0 = time.time()
                time.sleep(wait)
                elapsed = time.time() - t0
                attempt += 1
                logger.warning(
                    f"API connection error ({model_name}), {elapsed:.1f}s 대기 후 재시도 (attempt {attempt})\n"
                    f"  {extra_info}\n  err={err_str}"
                )
                wait = min(wait * 2, 30.0)
            else:
                logger.error(
                    f"API 호출 실패 ({model_name})\n"
                    f"  {extra_info}\n  err={err_str}"
                )
                raise e





# ─────────────────────────────────────────────────────────────────────────────
# 배치 추론 및 기타 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────


def setup_tokenizer(model_id: str, cache_dir: str = None, special_tokens: list = None):
    """토크나이저 로드 및 특수 토큰 추가.
    special_tokens 미지정 시 ACTION_TOKENS 사용.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = special_tokens if special_tokens is not None else ACTION_TOKENS
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    return tokenizer


def collate_fn(batch, pad_token_id: int) -> dict:
    """가변 길이 시퀀스를 패딩해 배치로 묶는다."""
    input_ids_list, labels_list = zip(*batch)
    max_len = max(x.size(0) for x in input_ids_list)
    padded_input   = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_labels  = torch.full((len(batch), max_len), -100,         dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len,                dtype=torch.long)
    for i, (inp, lbl) in enumerate(zip(input_ids_list, labels_list)):
        seq_len = inp.size(0)
        padded_input[i, :seq_len]   = inp
        padded_labels[i, :seq_len]  = lbl
        attention_mask[i, :seq_len] = 1
    return {"input_ids": padded_input, "attention_mask": attention_mask, "labels": padded_labels}


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로딩 (PPO / 평가 공통)
# ─────────────────────────────────────────────────────────────────────────────

def load_problems(data_path: str, n: int = None) -> List[dict]:
    """parquet 또는 jsonl 파일에서 문제 리스트를 [{problem_id, problem, answer}, ...] 형태로 반환.
    n이 주어지면 마지막 n개만 반환.
    """
    if str(data_path).endswith(".jsonl"):
        items = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                items.append({
                    "problem_id": str(obj.get("id") or obj.get("problem_id", f"problem_{len(items)}")),
                    "problem":    obj.get("problem", ""),
                    "answer":     str(obj.get("answer", "")),
                })
        if n:
            items = items[-n:]
        return items

    import pandas as pd

    df = pd.read_parquet(data_path)
    if n:
        df = df.iloc[-n:]

    items = []
    for i, (_, row) in enumerate(df.iterrows()):
        prompt = row.get("prompt", "")
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        problem_text = ""
        if isinstance(prompt, list):
            for msg in prompt:
                if msg.get("role") == "user":
                    text = msg["content"]
                    text = re.sub(r"\s*Please reason step by step,.*$", "", text, flags=re.DOTALL).strip()
                    problem_text = text
                    break
        else:
            problem_text = str(prompt)

        extra = row.get("extra_info", {})
        if isinstance(extra, str):
            extra = json.loads(extra)
        elif hasattr(extra, "item"):
            extra = extra.item()

        problem_id = str(extra.get("index", f"problem_{i}")) if isinstance(extra, dict) else f"problem_{i}"
        items.append({
            "problem_id": problem_id,
            "problem":    problem_text,
            "answer":     str(row.get("final_answer", row.get("answer", ""))),
        })
    return items


