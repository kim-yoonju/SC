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
    StoppingCriteria,
    StoppingCriteriaList,
)

import anthropic as _anthropic
from openai import OpenAI
from google import genai
from google.genai import types as genai_types

# 로깅 설정
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)
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
GPT_API_KEY       = os.environ.get("OPENAI_API_KEY")     or CONF["API_key"]["gpt"]
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY")     or CONF["API_key"].get("gemini")
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY")   or CONF["API_key"].get("deepseek")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  or CONF["API_key"].get("claude")

# API 클라이언트
client           = OpenAI(api_key=GPT_API_KEY)
deepseek_client  = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None
gemini_client    = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
anthropic_client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

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
_call_role: _cv.ContextVar[str] = _cv.ContextVar("call_role", default="unknown")

def set_run_log(fn):
    """콜백 fn(record: dict)을 등록하면 모든 _call_llm 호출이 기록됨."""
    global _run_log_fn
    _run_log_fn = fn

def set_call_role(role: str):
    """현재 컨텍스트의 호출 역할을 설정. _call_llm 로그의 'role' 필드에 기록됨."""
    return _call_role.set(role)


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
GENERATOR_TEMPERATURE    = CONF.get("step_reasoning", {}).get("temperature", 1.0)
GENERATOR_MAX_NEW_TOKENS = CONF.get("step_reasoning", {}).get("max_new_tokens", 2048)
PATCHER_MAX_NEW_TOKENS   = CONF.get("PATCHER", {}).get("max_new_tokens", 2048)
API_MAX_SEQ_LEN          = CONF.get("API_model", {}).get("max_seq_len", 1500)
TRUNCATE_TOKEN_LIMIT     = CONF.get("step_reasoning", {}).get("truncate_token_limit", 4096)
MAX_STEPS                = CONF.get("generate_trajectory", {}).get("max_steps") or CONF.get("ppo", {}).get("max_steps") or CONF.get("step_reasoning", {}).get("max_steps", 10)
VLLM_MAX_MODEL_LEN       = CONF.get("vllm", {}).get("max_model_len", 32768)

# API 모델
TRUNCATOR     = CONF.get("API_model", {}).get("TRUNCATOR")
REWARD        = CONF.get("API_model", {}).get("REWARD")
PATCHER       = CONF.get("PATCHER", {}).get("model_id")
EXTRACTOR     = CONF.get("API_model", {}).get("EXTRACTOR")

# PPO 하이퍼파라미터
_ppo                = CONF["ppo"]
PPO_LR              = _ppo["lr"]
PPO_CLIP_EPS        = _ppo["clip_eps"]
PPO_MAX_GRAD_NORM   = _ppo["max_grad_norm"]
KL_COEF             = _ppo["kl_coef"]
GAMMA               = _ppo["gamma"]
MAX_SEQ_LEN         = _ppo["max_seq_len"]
LENGTH_PENALTY_COEF = _ppo["length_penalty_coef"]
PATCHER_CANDIDATE   = _ppo["patcher_candidate"]
GENERATOR_CANDIDATE = _ppo["generator_candidate"]
PATCHER_TEMPERATURE = _ppo["patcher_temperature"]

_ROOT_PATH   = _pathlib.Path(__file__).resolve().parent.parent
DATASET_PATH = str(_ROOT_PATH / CONF["data_path"]["deepmath_16k"])
SAVE_DIR     = str(_ROOT_PATH / CONF["output_path"]["ppo"])

# ─────────────────────────────────────────────────────────────────────────────
# 상태 상수
# ─────────────────────────────────────────────────────────────────────────────

SOLVE       = "solve"
CORRECT_GEN = "correct_gen"
CORRECT_PAT = "correct_pat"
END_MAX     = "end_max"      # MAX_STEPS 도달로 강제 종료
END_ANSWER  = "end_answer"   # 정상 흐름으로 종료 (성공/실패 무관)

ACTIVE_STATES   = {SOLVE, CORRECT_GEN, CORRECT_PAT}
TERMINAL_STATES = {END_MAX, END_ANSWER}

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step_idx: int
    state: str           # 이 스텝이 실행된 시점의 상태 (solve/correct_gen/correct_pat)
    action: str
    text: str
    final_reward: float
    llm_reward: float
    format_reward: float
    predicted_next_action: str
    gold_next_action: str
    input_ids: torch.Tensor
    response_ids: torch.Tensor
    log_probs_old: torch.Tensor
    use_patcher: bool

@dataclass
class Trajectory:
    problem_id: str
    problem: str
    answer: str
    difficulty: Optional[float] = None
    steps: List[StepRecord] = field(default_factory=list)
    have_boxed: bool = False
    is_answer: bool = False
    patcher_wrong: bool = False
    end_state: Optional[str] = None   # end_max / end_answer

# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트 및 액션 토큰
# ─────────────────────────────────────────────────────────────────────────────

def _get_prompts() -> Dict[str, str]:
    """action_prompts.jsonl을 지연 로드해 캐싱."""
    if not _get_prompts._cache:
        from generate_utils import load_prompts
        _get_prompts._cache.update(load_prompts())
    return _get_prompts._cache
_get_prompts._cache: Dict[str, str] = {}

TOKEN_SOLVE   = "<|solve|>"
TOKEN_CORRECT = "<|rethink|>"
TOKEN_END     = "<|end|>"
ACTION_TOKENS = [TOKEN_SOLVE, TOKEN_CORRECT, TOKEN_END]

# ─────────────────────────────────────────────────────────────────────────────
# 모델 로딩 및 생성 로직
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(device, cfg: dict):
    """config 기반으로 모델과 토크나이저를 로드.

    로컬 checkpoint가 있으면 사용하고, 없으면 HuggingFace에서 다운로드.
    action 특수 토큰은 추가하지 않으므로 단순 추론/평가에 적합.
    """
    ckpt_cfg = cfg.get("checkpoint", {})
    sft_checkpoint = ckpt_cfg.get("sft_checkpoint", "")
    hf_model_id = ckpt_cfg["base"]
    cache_dir = ckpt_cfg.get("cache_dir", None)

    ckpt = _pathlib.Path(sft_checkpoint) if sft_checkpoint else _pathlib.Path("")
    if sft_checkpoint and ckpt.exists() and (ckpt / "config.json").exists():
        model_id = str(ckpt)
        logger.info(f"로컬 checkpoint 로드: {model_id}")
        load_kwargs = {}
    else:
        model_id = hf_model_id
        logger.info(f"HuggingFace 모델 로드: {model_id}")
        load_kwargs = {"trust_remote_code": True}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # config.vocab_size가 실제 체크포인트 가중치 크기와 다를 수 있으므로 패치.
    # Qwen2.5 계열은 config.vocab_size=152064이지만 실제 임베딩은 151668인 경우 존재.
    config = AutoConfig.from_pretrained(model_id, **load_kwargs)
    if config.vocab_size != len(tokenizer):
        logger.info(f"config.vocab_size({config.vocab_size}) != tokenizer vocab size({len(tokenizer)}), config 패치 후 로드")
        config.vocab_size = len(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16, **load_kwargs
    ).to(device)
    model.eval()
    return model, tokenizer


def load_generator(device_map="auto", model_path=None, load_in_4bit=False):
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
            device_map=device_map, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            load_path, config=config, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True
        )

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model, tokenizer

class StopOnActionToken(StoppingCriteria):
    def __init__(self, tokenizer, input_length: int):
        self._input_length = input_length
        self._action_ids = set(tid for tid in tokenizer.convert_tokens_to_ids(ACTION_TOKENS) if tid != tokenizer.unk_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        if input_ids.shape[1] > self._input_length:
            return input_ids[0, -1].item() in self._action_ids
        return False

def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"

# ─────────────────────────────────────────────────────────────────────────────
# 수학 및 정답 판정 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_latex(s: str) -> str:
    s = s.strip().replace(" ", "")
    s = re.sub(r"\\dfrac|\\tfrac", r"\\frac", s)
    s = re.sub(r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\\(|\\\)", "", s)             # \( \) 인라인 수식 구분자 제거
    s = re.sub(r"\\left|\\right|[()]", "", s)  # {} 는 제거하지 않음 (행렬 환경 보호)
    # 순환군 표기 통일: \mathbb{Z}_n / C_n / Z_n  →  \mathbb{Z}/n\mathbb{Z}
    s = re.sub(r"\\mathbb\{Z\}_\{(\d+)\}", r"\\mathbb{Z}/\1\\mathbb{Z}", s)
    s = re.sub(r"\\mathbb\{Z\}_(\d+)", r"\\mathbb{Z}/\1\\mathbb{Z}", s)
    s = re.sub(r"\bC_\{?(\d+)\}?", r"\\mathbb{Z}/\1\\mathbb{Z}", s)
    s = re.sub(r"\bZ_\{?(\d+)\}?", r"\\mathbb{Z}/\1\\mathbb{Z}", s)
    # \frac 단축 표기 확장: \frac19 → \frac{1}{9}
    s = re.sub(r"\\frac([^{\\])([^{\\])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac([^{\\])\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\frac\{([^}]*)\}([^{\\])", r"\\frac{\1}{\2}", s)
    return s


# True/False ↔ Yes/No 정규화 테이블
_BOOL_NORM: dict[str, str] = {
    "true": "yes", "false": "no",
    "yes": "yes",  "no": "no",
}

def _normalize_bool(s: str) -> str | None:
    """'True'/'False'/'Yes'/'No' 계열 문자열을 'yes'/'no'로 정규화. 해당 없으면 None."""
    return _BOOL_NORM.get(s.strip().lower())


# 도(°) → 라디안 변환 후 비교
_DEG_RE = re.compile(
    r"^([+-]?\d+(?:\.\d+)?)\s*\^?\s*(?:\\circ|°|\\degree)$"
)
_RAD_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d+)?)?)\s*\\?pi(?:\s*/\s*([+-]?\d+(?:\.\d+)?))?$"
    r"|^\\frac\{([+-]?(?:\d+(?:\.\d+)?)?\\?pi)\}\{([+-]?\d+(?:\.\d+)?)\}$"
    r"|^\\frac([+-]?(?:\d+(?:\.\d+)?)?\\?pi)\{([+-]?\d+(?:\.\d+)?)\}$"
)

def _parse_angle_rad(s: str) -> float | None:
    """LaTeX 각도 문자열을 라디안(float)으로 변환. 실패 시 None."""
    import math
    s = s.strip().replace(" ", "")

    # 도(°) 형식
    m = _DEG_RE.match(s)
    if m:
        return float(m.group(1)) * math.pi / 180

    # \frac{N\pi}{D} 또는 \frac{N\pi}\{D\} 형식
    m = re.match(r"^\\frac\{([+-]?\d*(?:\.\d+)?)\\pi\}\{([+-]?\d+(?:\.\d+)?)\}$", s)
    if m:
        num = float(m.group(1)) if m.group(1) not in ("", "+", "-") else (1.0 if m.group(1) != "-" else -1.0)
        return num * math.pi / float(m.group(2))

    m = re.match(r"^\\frac([+-]?\d*(?:\.\d+)?)\\pi\{([+-]?\d+(?:\.\d+)?)\}$", s)
    if m:
        num = float(m.group(1)) if m.group(1) not in ("", "+", "-") else (1.0 if m.group(1) != "-" else -1.0)
        return num * math.pi / float(m.group(2))

    # N\pi/D 또는 N\pi 형식
    m = re.match(r"^([+-]?\d*(?:\.\d+)?)\\pi(?:/([+-]?\d+(?:\.\d+)?))?$", s)
    if m:
        num = float(m.group(1)) if m.group(1) not in ("", "+", "-") else (1.0 if m.group(1) != "-" else -1.0)
        denom = float(m.group(2)) if m.group(2) else 1.0
        return num * math.pi / denom

    return None


def _angle_equal(a: str, b: str) -> bool:
    """두 각도 표현이 도/라디안 변환 후 동치인지 비교."""
    import math
    ra, rb = _parse_angle_rad(a), _parse_angle_rad(b)
    if ra is None or rb is None:
        return False
    return abs(ra - rb) < 1e-9

def extract_boxed(text: str, is_gsm8k: bool = False) -> str | None:
    """정답 추출.

    is_gsm8k=True: GSM8K 형식 (#### 뒤 숫자)만 사용.
    is_gsm8k=False: \\boxed{} → #### (GSM8K) → ### 순으로 탐색.
    """
    if not is_gsm8k:
        # 1. \boxed{}
        marker = r"\boxed{"
        pos = text.rfind(marker)
        if pos != -1:
            start, depth = pos + len(marker), 1
            for i in range(start, len(text)):
                if text[i] == "{": depth += 1
                elif text[i] == "}": depth -= 1
                if depth == 0: return text[start:i].strip()
    # GSM8K: 마지막 #### 뒤 숫자
    m = None
    for match in re.finditer(r"####\s*(.+)", text):
        m = match
    if m:
        return m.group(1).strip().replace(",", "")
    if is_gsm8k:
        return None
    # 마지막 ### 뒤 한 줄
    m = None
    for match in re.finditer(r"###\s*(.+)", text):
        m = match
    if m:
        return m.group(1).strip()
    return None

# 알려진 수학적 동치 표현 → 정규형 매핑 (latex2sympy2가 처리 못하는 케이스)
_KNOWN_EQUIV: dict[str, str] = {
    r"\mathfrak{c}": r"2^{\aleph_0}",
}
_KNOWN_EQUIV.update({v: k for k, v in list(_KNOWN_EQUIV.items())})

def _canonicalize(s: str) -> str:
    for variant, canonical in _KNOWN_EQUIV.items():
        s = s.replace(variant.replace(" ", ""), canonical.replace(" ", ""))
    return s

def _latex2sympy_equal(a: str, b: str) -> bool:
    """latex2sympy2_extended로 두 LaTeX 수식이 동치인지 확인. 실패 시 False."""
    try:
        from latex2sympy2_extended import latex2sympy
        from sympy import simplify, Matrix
        la, lb = latex2sympy(a), latex2sympy(b)
        if isinstance(la, Matrix) or isinstance(lb, Matrix):
            return la == lb
        diff = simplify(la - lb)
        if diff == 0:
            return True
        # Float 연산 오차(e.g. 5.55e-17)는 == 0이 False여도 실질적으로 0
        try:
            return abs(float(diff)) < 1e-9
        except Exception:
            return False
    except Exception:
        return False


def _polynomial_form_equal(a: str, b: str) -> bool:
    """두 식이 x에 대한 다항식이고, 수치 계수는 일치하며 나머지는 자유 상수인 경우 동치.
    예: 2x^2 + Ax + B  ↔  2x^2 + bx + c  (임의 상수 이름 무관)
    최소 하나의 수치 계수가 일치해야 함 (단순 기호 쌍 오매칭 방지)."""
    try:
        from latex2sympy2_extended import latex2sympy
        from sympy import Poly, symbols
        x = symbols('x')
        pa = Poly(latex2sympy(a), x)
        pb = Poly(latex2sympy(b), x)
        if pa.degree() != pb.degree() or pa.degree() < 1:
            return False
        has_numeric_match = False
        for ci, cj in zip(pa.all_coeffs(), pb.all_coeffs()):
            if ci.is_number and cj.is_number:
                if ci != cj:
                    return False
                has_numeric_match = True
            elif ci.is_number != cj.is_number:
                return False
        return has_numeric_match
    except Exception:
        return False


def _numeric_approx_equal(a: str, b: str, rel_tol: float = 2e-3) -> bool:
    """두 수식을 float로 평가해 상대 오차 0.2% 이내이면 True."""
    try:
        from latex2sympy2_extended import latex2sympy
        import sympy
        fa = float(sympy.N(latex2sympy(a), 15))
        fb = float(sympy.N(latex2sympy(b), 15))
        if fa == fb == 0:
            return True
        return abs(fa - fb) / max(abs(fa), abs(fb), 1e-12) < rel_tol
    except Exception:
        return False


def _matrix_elements(s: str) -> list[str] | None:
    """pmatrix/bmatrix/matrix 환경에서 원소 리스트를 평탄하게 추출."""
    m = re.search(r"\\begin\{[pbvBsS]?matrix\}(.*?)\\end\{[pbvBsS]?matrix\}", s, re.DOTALL)
    if not m:
        return None
    inner = m.group(1)
    elements = []
    for row in re.split(r"\\\\", inner):
        for elem in re.split(r"&", row):
            elem = elem.strip()
            if elem:
                elements.append(elem)
    return elements or None


def _matrix_equal(a: str, b: str) -> bool:
    """두 행렬/벡터 LaTeX 표현이 원소별로 수치 동치인지 확인."""
    ea, eb = _matrix_elements(a), _matrix_elements(b)
    if ea is None or eb is None or len(ea) != len(eb):
        return False
    for x, y in zip(ea, eb):
        nx, ny = x.replace(" ", ""), y.replace(" ", "")
        if nx == ny:
            continue
        if _normalize_latex(nx) == _normalize_latex(ny):
            continue
        # 8/3 (plain division) ↔ \frac{8}{3} 등 수치 비교
        if not _numeric_approx_equal(nx, ny):
            return False
    return True


def _extract_approx_value(s: str) -> str | None:
    """'expr \\approx 1.243' 형태에서 근삿값 숫자를 추출."""
    m = re.search(r"\\approx\s*([+-]?\d+(?:\.\d+)?)", s)
    return m.group(1) if m else None


def _normalize_pred(pred: str) -> list[str]:
    """pred에서 비교 가능한 후보 문자열 목록을 반환.
    - 원본
    - \\approx 뒤 숫자 (있으면)
    - = 이후 표현식 추출 (f(x)=expr, numeric 및 symbolic 모두)
    - \\text{Yes/No,...} 앞 boolean 추출
    - \\text{...} 완전 제거 후 남은 수식
    """
    candidates = [pred]

    def _add(s: str) -> None:
        s = s.strip().rstrip(".,")
        if s and s not in candidates:
            candidates.append(s)

    approx = _extract_approx_value(pred)
    if approx:
        _add(approx)

    # f(x)=expr 또는 = value 패턴에서 = 이후 표현식 추출 (numeric + symbolic)
    m = re.search(r"=\s*(.+?)\s*$", pred.replace(" ", ""))
    if m:
        _add(m.group(1))

    # = VALUE \text{qualifier} 또는 = VALUE, \quad qualifier 패턴에서 VALUE만 추출
    # 예: f(x) = 0 \text{ for all x}  →  0
    #     f(x) = 2x^2 + Ax + B, \quad A,B∈ℝ  →  2x^2 + Ax + B
    m = re.search(r"=\s*([^\\,]+?)\s*(?:\\text|,\s*(?:\\text|\\quad))", pred)
    if m:
        _add(m.group(1).strip())

    # \text{Yes/No/True/False, ...} 에서 leading boolean 추출
    m = re.match(r"\\text\{(yes|no|true|false)[^}]*\}", pred, re.IGNORECASE)
    if m:
        _add(m.group(1).capitalize())

    # (SET, operation) 표기에서 SET 추출: (\mathbb{Z},\cdot) → \mathbb{Z}
    m = re.match(r"\((.+?),\s*(?:\\cdot|\\times|\\circ|\+|\-)\s*\)",
                 pred.replace(" ", ""))
    if m:
        _add(m.group(1))

    # 'main_expr, qualifier' 패턴에서 qualifier 이전 주답 추출
    # 예: f(n)=n+c, \quad c∈ℕ  →  f(n)=n+c
    #     P(x)=c, \text{ where } c is palindromic  →  P(x)=c
    m = re.search(r",\s*(?:\\quad|\\text\{\s*(?:where|for|with|and)\b)", pred)
    if m:
        _add(pred[:m.start()])

    # \text{...} 완전 제거 후 남은 수식 (예: 1.242\text{ (approximately)} → 1.242)
    stripped = re.sub(r"\\text\{[^}]*\}", "", pred).strip().rstrip(".,")
    if stripped:
        _add(stripped)

    return candidates


def _normalize_gold(gold: str) -> list[str]:
    """gold에서 비교 가능한 후보 문자열 목록을 반환.
    - 원본
    - \\text{단위} 완전 제거 후 남은 수식 (예: 54\\text{ gallons} → 54)
    """
    candidates = [gold]
    stripped = re.sub(r"\\text\{[^}]*\}", "", gold).strip().rstrip(".,")
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    return candidates


def _times_sorted(s: str) -> str:
    """A×B×C 형태의 직접곱 표현을 성분 정렬해 정규화. (교환법칙 처리)"""
    parts = re.split(r"\\times", s)
    if len(parts) < 2:
        return s
    return r"\times".join(sorted(p.strip() for p in parts))


_INFINITY_RE = re.compile(
    r"^(?:"
    r"\\infty"
    r"|[+]?\\infty"
    r"|\\text\{(?:diverges?|divergent|infinite?|infinity|\\infty)\}"
    r"|diverges?"
    r"|divergent"
    r"|infinite?"
    r"|infinity"
    r")$",
    re.IGNORECASE,
)
_NEG_INFINITY_RE = re.compile(
    r"^(?:-\\infty|\\text\{-\\infty\}|-infinity|-infinite?)$",
    re.IGNORECASE,
)

def _normalize_infinity(s: str) -> str | None:
    """발산/무한대 표현 → '\\infty' 또는 '-\\infty'. 해당 없으면 None."""
    s = s.replace(" ", "")
    if _INFINITY_RE.match(s):
        return "\\infty"
    if _NEG_INFINITY_RE.match(s):
        return "-\\infty"
    return None


def _compare_single(pred: str, gold: str) -> bool:
    """pred, gold 두 문자열이 수학적으로 동치인지 모든 방법으로 확인."""
    pred, gold = pred.replace(" ", ""), gold.replace(" ", "")
    if pred == gold: return True
    if pred.lower() == gold.lower(): return True
    pi, gi = _normalize_infinity(pred), _normalize_infinity(gold)
    if pi is not None and gi is not None and pi == gi: return True
    pb, gb = _normalize_bool(pred), _normalize_bool(gold)
    if pb is not None and gb is not None and pb == gb: return True
    try:
        if abs(float(pred) - float(gold)) < 1e-6: return True
    except ValueError:
        pass
    np_, ng = _normalize_latex(pred), _normalize_latex(gold)
    if np_ == ng: return True
    if _times_sorted(np_) == _times_sorted(ng): return True
    if _canonicalize(pred) == _canonicalize(gold): return True
    if _angle_equal(pred, gold): return True
    if _matrix_equal(pred, gold): return True
    if _latex2sympy_equal(pred, gold): return True
    if _polynomial_form_equal(pred, gold): return True
    return _numeric_approx_equal(pred, gold)


def _extract_mc_options(problem: str) -> dict[str, str]:
    """객관식 문제에서 옵션 파싱. {'A': '2', 'B': '-1', ...}"""
    options = {}
    # 패턴: A) val / (A) val / A. val — 줄 단위로 파싱
    pattern = re.compile(
        r'[\(\[]?([A-F])[\)\]\.]\)?[\s\)]*'   # 옵션 레터
        r'((?:(?![\(\[]?[A-F][\)\]\.]|\Z).)+)',  # 값 (다음 옵션 전까지)
        re.DOTALL
    )
    for m in pattern.finditer(problem):
        key = m.group(1).upper()
        val = m.group(2).strip().rstrip(' \n,;')
        if val:
            options[key] = val
    return options


def check_solved(step_text: str, gold_answer, is_gsm8k: bool = False,
                 problem: str = "") -> bool:
    pred_raw = extract_boxed(step_text, is_gsm8k=is_gsm8k)
    if not pred_raw:
        return False
    gold_str = str(gold_answer).strip()
    if "####" in gold_str:
        gold_str = _extract_gsm8k_answer(gold_str)

    # 객관식: gold가 단일 레터(A-F)이고 문제에 옵션이 있으면 값↔레터 매칭
    if problem and re.fullmatch(r"[A-F]", gold_str):
        options = _extract_mc_options(problem)
        if gold_str in options:
            option_val = options[gold_str].strip()
            for pred_cand in _normalize_pred(pred_raw):
                for gold_cand in _normalize_gold(option_val):
                    if _compare_single(pred_cand, gold_cand):
                        return True

    for pred_cand in _normalize_pred(pred_raw):
        for gold_cand in _normalize_gold(gold_str):
            if _compare_single(pred_cand, gold_cand):
                return True
    return False

def has_boxed(text: str) -> bool:
    return bool(re.search(r"\\boxed\{", text))

def check_end(text: str, action: str | None) -> bool:
    """추론 종료 조건: <|end|> 액션."""
    return action == TOKEN_END

def answers_equal(pred: str, gold: str) -> bool:
    """두 정답 문자열이 동일한지 비교. 숫자는 부동소수점 오차 허용."""
    pred = pred.strip().replace(" ", "")
    gold = gold.strip().replace(" ", "")
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False

def format_correct(response: str, gold: str, is_gsm8k: bool = False,
                   problem: str = "") -> bool:
    """정답 추출 후 비교. GSM8K는 #### 형식 사용."""
    return check_solved(response, gold, is_gsm8k=is_gsm8k, problem=problem)

# ─────────────────────────────────────────────────────────────────────────────
# API 클라이언트 라우터
# ─────────────────────────────────────────────────────────────────────────────

def _is_reasoning_model(model_name: str) -> bool:
    """temperature=0 미지원 / max_completion_tokens 필요한 추론 모델 여부."""
    m = model_name.lower()
    return any(tok in m for tok in ["o1", "o3", "gpt-5", "thinking", "deepseek-reasoner"])


def _call_gemini(model_name: str, messages: list, max_output_tokens: int = None, temperature: float = None) -> str:
    """네이티브 Gemini SDK로 호출. messages는 OpenAI 형식."""
    if not GEMINI_API_KEY:
        raise ValueError(
            "Gemini 모델을 사용하려면 GEMINI_API_KEY 환경변수 또는 "
            "config/config.yaml의 API_key.gemini를 설정하세요."
        )
    # system 메시지 분리
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system_instruction = "\n\n".join(system_parts) if system_parts else None

    # user/assistant 메시지를 Gemini contents 형식으로 변환
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    gen_config = genai_types.GenerateContentConfig(
        max_output_tokens=max_output_tokens,
        temperature=temperature if temperature is not None else 0.0,
        system_instruction=system_instruction,
    )

    resp = gemini_client.models.generate_content(
        model=model_name,
        contents=contents,
        config=gen_config,
    )

    # 전체 응답 객체 로깅
    for i, cand in enumerate(resp.candidates):
        text = cand.content.parts[0].text if cand.content and cand.content.parts else None
        logger.info(
            f"[Gemini raw] cand={i}"
            f"  finish_reason={cand.finish_reason}"
            f"  safety={cand.safety_ratings}"
            f"  text={text!r}"
        )

    if resp.usage_metadata:
        _record_usage(model_name, [{
            "input_tokens":  getattr(resp.usage_metadata, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0) or 0,
            "cached_tokens": getattr(resp.usage_metadata, "cached_content_token_count", 0) or 0,
        }])

    return resp.text


def _call_claude(model_name: str, messages: list, max_tokens: int = None, temperature: float = None,
                 usage_out: list = None) -> str:
    """Anthropic Claude API 호출. messages는 OpenAI 형식."""
    if not anthropic_client:
        raise ValueError(
            "Claude 모델을 사용하려면 ANTHROPIC_API_KEY 환경변수 또는 "
            "config/config.yaml의 API_key.claude를 설정하세요."
        )
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system_text = "\n\n".join(system_parts) if system_parts else None
    user_messages = [m for m in messages if m["role"] != "system"]

    kwargs = {"model": model_name, "messages": user_messages}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = 8192
    if temperature is not None:
        kwargs["temperature"] = temperature
    if system_text:
        kwargs["system"] = system_text

    resp = anthropic_client.messages.create(**kwargs)

    usage_entry = {
        "input_tokens":  resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cached_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
    }
    _record_usage(model_name, [usage_entry])
    if usage_out is not None:
        usage_out.append(usage_entry)

    content = resp.content[0].text if resp.content else ""
    if _run_log_fn is not None:
        try:
            from datetime import datetime as _dt
            _run_log_fn({
                "ts":      _dt.now().isoformat(timespec="seconds"),
                "model":   model_name,
                "in_tok":  resp.usage.input_tokens,
                "out_tok": resp.usage.output_tokens,
                "messages": messages,
                "output":  content,
            })
        except Exception:
            pass
    return content


# ─────────────────────────────────────────────────────────────────────────────
# API 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(model_name: str, messages: list, max_completion_tokens: int = None, temperature: float = None,
         usage_out: list = None, response_format: dict = None, logprobs_out: list = None) -> str:
    """모델 종류에 따라 OpenAI 또는 Gemini API를 호출합니다.

    usage_out:    제공 시 {"input_tokens": int, "output_tokens": int, "finish_reason": str} dict를 append.
    logprobs_out: 제공 시 resp.choices[0].logprobs.content (token별 top_logprobs 리스트)를 extend.
    """
    import time, re as _re2
    wait = 1.0
    attempt = 0
    while True:
        try:
            if model_name.lower().startswith("gemini"):
                return _call_gemini(model_name, messages, max_output_tokens=max_completion_tokens, temperature=temperature)

            if model_name.lower().startswith("claude"):
                return _call_claude(model_name, messages, max_tokens=max_completion_tokens, temperature=temperature, usage_out=usage_out)

            if model_name.lower().startswith("deepseek"):
                if deepseek_client is None:
                    raise ValueError("DeepSeek API 키가 config.API_key.deepseek에 없습니다.")
                active_client = deepseek_client
            else:
                active_client = client

            reasoning = _is_reasoning_model(model_name)

            api_model_name = _DEEPSEEK_ALIASES.get(model_name.lower(), model_name)
            kwargs = {"model": api_model_name, "messages": messages}
            if max_completion_tokens:
                if reasoning:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                else:
                    kwargs["max_tokens"] = max_completion_tokens
            if not reasoning:
                kwargs["temperature"] = temperature if temperature is not None else 0
            if response_format is not None:
                kwargs["response_format"] = response_format
            if logprobs_out is not None and not reasoning:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 20

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
                        "ts":      _dt.now().isoformat(timespec="seconds"),
                        "role":    _call_role.get(),
                        "model":   model_name,
                        "in_tok":  resp.usage.prompt_tokens if resp.usage else None,
                        "out_tok": resp.usage.completion_tokens if resp.usage else None,
                        "messages": messages,
                        "output":  content,
                    })
                except Exception:
                    pass

            return content
        except Exception as e:
            err_str = str(e)
            is_quota_exceeded = "insufficient_quota" in err_str or "billing" in err_str.lower()
            is_rate_limit = ("429" in err_str or "rate_limit" in err_str.lower()) and not is_quota_exceeded
            if is_rate_limit:
                m = _re2.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                retry_after = float(m.group(1)) + 0.5 if m else wait
                t0 = time.time()
                time.sleep(retry_after)
                elapsed = time.time() - t0
                attempt += 1
                logger.warning(f"API rate limit ({model_name}), {elapsed:.1f}s 대기 후 재시도 (attempt {attempt}) | {err_str[:300]}")
                wait = min(wait * 2, 60.0)
            else:
                logger.error(f"API 호출 실패 ({model_name}): {e}")
                raise e

def truncate_step_if_needed(text: str) -> str:
    """스텝 텍스트가 API_MAX_SEQ_LEN 초과이면 TRUNCATOR API로 단축."""
    if len(text) <= API_MAX_SEQ_LEN:
        return text
    logger.info(f"[truncate] {len(text)}자 초과 → TRUNCATOR 호출")
    messages = [
        {"role": "system", "content": SYSTEM_TRUNCATOR},
        {"role": "user",   "content": text},
    ]
    try:
        result = _call_llm(TRUNCATOR, messages, max_completion_tokens=512)
        if result:
            logger.info(f"[truncate] → {len(result)}자")
            return result
    except Exception as e:
        logger.warning(f"[truncate] 실패: {e}")
    return text[:API_MAX_SEQ_LEN]


def R_PRM(response: str, gold: str, model: str = None, history: list = None) -> float:
    """REWARD 모델로 스텝 풀이 품질을 0.0~1.0 연속값으로 채점. OpenAI/Gemini 모델 모두 지원.

    history가 주어지면 LLM_SCORE_SFT_PROMPT(문맥 포함)를 사용한다.
    """
    import re as _re
    if model is None:
        model = REWARD
    # <|...|> 및 |>...|> 형태의 special token 잔재 제거
    clean_response = _re.sub(r"<?<?\|[^|>]*\|>", "", response).strip()
    if history:
        history_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(history))
        prompt = LLM_SCORE_SFT_PROMPT.format(response=clean_response, gold=gold, history=history_text)
    else:
        prompt = LLM_SCORE_PROMPT.format(response=clean_response, gold=gold)
    try:
        _max_tokens = 4096 if _is_reasoning_model(model) else 512
        text = _call_llm(model, [{"role": "user", "content": prompt}], max_completion_tokens=_max_tokens)
        if text is None:
            logger.warning(f"R_PRM None 응답  model={model}")
            return 0.0
        # "Score: 0.7" 형식 우선 파싱
        for line in text.strip().splitlines():
            if line.lower().startswith("score:"):
                val = line.split(":", 1)[1].strip()
                if val:
                    m = _re.search(r"[0-9]*\.?[0-9]+", val)
                    if m:
                        return max(0.0, min(1.0, float(m.group())))
        # 폴백: 응답 전체에서 첫 번째 숫자 추출
        m = _re.search(r"\b([0-9]*\.?[0-9]+)\b", text)
        if m:
            val = float(m.group(1))
            if 0.0 <= val <= 1.0:
                logger.warning(f"R_PRM 폴백 파싱  val={val}  raw={text!r}")
                return val
        logger.warning(f"R_PRM 파싱 실패  model={model}  raw_len={len(text)}\n--- PRM raw response ---\n{text!r}\n--- end ---")
    except Exception as e:
        logger.warning(f"R_PRM 오류: {e}")
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 배치 추론 및 기타 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch(prompts: list[str], model, tokenizer, device, max_new_tokens: int) -> list[str]:
    """여러 프롬프트를 greedy decoding으로 배치 생성. 단순 평가에 적합."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(device)
    padded_len = enc["input_ids"].shape[1]

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0,
            top_p=None,
        )

    results = []
    for i in range(len(prompts)):
        gen_ids  = out[i][padded_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        results.append(gen_text)
    return results


def generate_steps_batched(model, tokenizer, prompt_texts, max_new_tokens=None, greedy=False):
    if not prompt_texts: return []
    _max = max_new_tokens or GENERATOR_MAX_NEW_TOKENS

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)
    tokenizer.padding_side = orig_side

    gen_kwargs = dict(max_new_tokens=_max, pad_token_id=tokenizer.eos_token_id)
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = GENERATOR_TEMPERATURE

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # action 토큰 ID 쌍 (unk 제외)
    action_pairs = [
        (tok, tid)
        for tok, tid in zip(ACTION_TOKENS, tokenizer.convert_tokens_to_ids(ACTION_TOKENS))
        if tid != tokenizer.unk_token_id
    ]

    results = []
    no_action_indices = []  # logits 조회가 필요한 샘플 인덱스
    max_in = inputs["input_ids"].shape[1]

    for i, prompt in enumerate(prompt_texts):
        gen_text = tokenizer.decode(output_ids[i, max_in:], skip_special_tokens=False)
        reasoning, predicted = gen_text, None
        for tok in ACTION_TOKENS:
            if tok in gen_text:
                idx = gen_text.rfind(tok)
                reasoning, predicted = gen_text[:idx].rstrip(), tok
                break
        results.append([reasoning, predicted, tokenizer(prompt, return_tensors="pt")["input_ids"].cpu()])
        if predicted is None:
            no_action_indices.append(i)

    # 액션 토큰이 없는 샘플: 마지막 토큰 logits에서 가장 확률 높은 액션 토큰 선택
    if no_action_indices and action_pairs:
        with torch.no_grad():
            for i in no_action_indices:
                logits = model(output_ids[i:i+1]).logits[0, -1, :]  # (vocab,)
                action_logits = torch.tensor([logits[tid].item() for _, tid in action_pairs])
                best_tok = action_pairs[action_logits.argmax().item()][0]
                results[i][1] = best_tok

    return [(r[0], r[1], r[2]) for r in results]

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로딩 및 전처리 유틸리티 (generate_utils 로 이동, 하위 호환 re-export)
# ─────────────────────────────────────────────────────────────────────────────

from generate_utils import (  # noqa: E402
    _extract_gsm8k_answer,
    _extract_problem,
    _extract_answer,
    _solve_user,
    _correct_user,
    _load_jsonl_eval,
    _load_parquet_eval,
    load_dataset_file,
    extract_step_content,
    build_target_text,
)


def load_raw_data(data_path: str) -> list[dict]:
    """JSONL 파일을 읽어 dict 리스트로 반환."""
    items = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


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


def pick_system(action: str) -> str:
    """액션에 맞는 system 프롬프트 반환."""
    p = _get_prompts()
    return p["system_rethink"] if action in ("correct", "rethink") else p["system_solve"]


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


def load_math500() -> List[dict]:
    """config의 math500 경로에서 문제 리스트를 반환."""
    path = str(_ROOT_PATH / CONF["data_path"]["math500"])
    return load_problems(path)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout 파일 I/O
# ─────────────────────────────────────────────────────────────────────────────

def create_rollout_file(path: str):
    """빈 rollout JSONL 파일을 생성 (디렉토리 자동 생성 포함)."""
    _pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    open(path, "w").close()


def save_trajectory(traj: "Trajectory", path: str):
    """Trajectory를 JSONL에 append 저장."""
    last_boxed_text = next(
        (s.text for s in reversed(traj.steps) if has_boxed(s.text)), ""
    )
    pred_answer = extract_boxed(last_boxed_text) if last_boxed_text else None
    record = {
        "problem_id":    traj.problem_id,
        "problem":       traj.problem,
        "gold_answer":   traj.answer,
        "pred_answer":   pred_answer,
        "have_boxed":    traj.have_boxed,
        "is_right":      traj.is_answer,
        "patcher_wrong": traj.patcher_wrong,
        "end_state":     traj.end_state,
        "steps": [
            {
                "step_idx":                s.step_idx,
                "state":                   s.state,
                "action":                  s.action,
                "text":                    s.text,
                "final_reward":            s.final_reward,
                "llm_reward":              s.llm_reward,
                "format_reward":           s.format_reward,
                "predicted_next_action":   s.predicted_next_action,
                "gold_next_action":        s.gold_next_action,
                "use_patcher":             s.use_patcher,
            }
            for s in traj.steps
        ],
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트 토큰 예산
# ─────────────────────────────────────────────────────────────────────────────

_MAX_PROMPT_TOKENS  = VLLM_MAX_MODEL_LEN - GENERATOR_MAX_NEW_TOKENS
_MAX_HISTORY_TOKENS = 4096


# ─────────────────────────────────────────────────────────────────────────────
# Reward 계산
# ─────────────────────────────────────────────────────────────────────────────

def score_step(text: str, answer: str, is_last: bool = False, history: List[str] = None) -> float:
    """스텝 reward R_final을 반환.

    R_final = R_PRM: REWARD 모델이 생성한 0~1 연속값.  state machine 분기 기준 (> 0.5).
    history가 주어지면 문맥을 포함한 SFT scorer를 사용한다.
    """
    r_prm = R_PRM(text, answer, history=history)

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

      TOKEN_END              → None (종료)
      TOKEN_CORRECT:
        SOLVE / CORRECT_GEN  → CORRECT_GEN / CORRECT_PAT (한 단계 깊게)
        CORRECT_PAT          → None (더 이상 패처 없음)
      TOKEN_SOLVE            → SOLVE (계속 풀기)
    """
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

      boxed + r>0.5              → None (end, 정답 도달)
      solve,       r>0.5         → solve
      solve,       r<=0.5        → correct_gen
      correct_gen, r>0.5         → solve
      correct_gen, r<=0.5        → correct_pat
      correct_pat, r>0.5         → solve
      correct_pat, r<=0.5        → correct_gen (patcher 실패 → generator 재시도, MAX_STEPS까지 반복)
      end,         r>0.5         → None (end, 성공)
      end,         r<=0.5        → correct_gen
    """
    if has_boxed(text) and r_prm > 0.5:
        return None  # boxed 답 + 높은 reward → 종료 (gold: <|end|>)
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
            return CORRECT_GEN  # patcher 실패 → generator로 재시도 (MAX_STEPS까지 반복)
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

def _trim_history(history: List[str], tokenizer, max_tokens: int = _MAX_HISTORY_TOKENS) -> List[str]:
    """history를 최근 max_tokens 토큰 이내로 trim (오래된 스텝 제거)."""
    if not history:
        return history
    total, keep = 0, []
    for step in reversed(history):
        n = len(tokenizer(step, add_special_tokens=False)["input_ids"])
        if total + n > max_tokens:
            break
        keep.append(step)
        total += n
    return list(reversed(keep))


def _build_gen_prompt(tokenizer, state: str, problem: str, history: List[str], sft_mode: bool = False) -> str:
    """상태에 따라 generator용 chat prompt를 생성.

    누적 history가 _MAX_PROMPT_TOKENS를 초과하면 오래된 step부터 제거해 길이를 맞춘다.
    """
    def _make(hist):
        p = _get_prompts()
        if state == SOLVE:
            system = p["system_solve_sft"] if sft_mode else p["system_solve"]
            return build_chat_prompt(tokenizer, system, _solve_user(problem, hist))
        else:
            return build_chat_prompt(tokenizer, p["system_rethink"], _correct_user(problem, hist))

    if not history:
        return _make(history)

    # 빠른 경로: 대부분의 경우 길이가 충분
    prompt = _make(history)
    n_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if n_tokens <= _MAX_PROMPT_TOKENS:
        return prompt

    # 이진 탐색: 최근 step을 최대한 많이 유지하면서 budget 안으로 수렴
    lo, hi = 0, len(history)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        n = len(tokenizer(_make(history[-mid:]), add_special_tokens=False)["input_ids"])
        if n <= _MAX_PROMPT_TOKENS:
            lo = mid
        else:
            hi = mid - 1

    if lo < len(history):
        logger.warning(
            f"[prompt trim] history {len(history)} steps → 최근 {lo}개만 사용 "
            f"(prompt {n_tokens} > {_MAX_PROMPT_TOKENS} tokens)"
        )
    return _make(history[-lo:] if lo > 0 else [])


def _call_patcher(problem: str, history: List[str], temperature: float = None) -> str:
    """PATCHER API를 호출해 한 스텝 풀이를 반환 (action token 없음)."""
    messages = [
        {"role": "system", "content": _get_prompts().get("pat_solve_R", "")},
        {"role": "user",   "content": _correct_user(problem, history)},
    ]
    logger.info(f"  [patcher] {PATCHER} 호출 중  history_len={len(history)}  temp={temperature}")
    try:
        result = _call_llm(PATCHER, messages, max_completion_tokens=PATCHER_MAX_NEW_TOKENS, temperature=temperature)
        result = truncate_step_if_needed(result)
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
    import torch
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
    sft_mode: bool = False,
) -> List["Trajectory"]:
    """problems 배치에 대해 state machine 기반으로 Trajectory를 생성.

    generator 스텝: use_patcher=False — PPO 학습 대상
    patcher  스텝: use_patcher=True  — PPO 학습 제외
    sft_mode=True: SFT 전용 프롬프트 + 문맥 포함 reward scorer 사용
    """
    import torch
    import torch.nn.functional as F

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
                _build_gen_prompt(tokenizer, states[i], trajs[i].problem, histories[i], sft_mode=sft_mode)
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

            # ③ log_probs: 마이크로배치로 쪼개 GPU→CPU 즉시 이동 (OOM 방지)
            t1 = time.time()
            _LP_MB = 4  # 한 번에 처리할 시퀀스 수
            lp_parts = []
            with torch.no_grad():
                for _s in range(0, len(gen_active), _LP_MB):
                    _chunk = out_ids[_s:_s + _LP_MB, :-1]
                    _logits = model(_chunk).logits[:, n_in - 1:, :]
                    lp_parts.append(F.log_softmax(_logits, dim=-1).cpu())
                    del _logits
            full_lp = torch.cat(lp_parts, dim=0)   # CPU tensor
            del lp_parts
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
                lp = full_lp[j, :trim].gather(1, resp_trim.cpu().unsqueeze(1)).squeeze(1)

                text = tokenizer.decode(resp_trim, skip_special_tokens=True)
                for tok in ACTION_TOKENS:
                    text = text.replace(tok, "")
                text = truncate_step_if_needed(text.strip())
                decoded.append((j, i, resp_trim, lp, text, pred_action))

            # ⑤ R_PRM 병렬 호출 → 완료된 순서대로 즉시 처리 + 로그
            t2 = time.time()
            with ThreadPoolExecutor(max_workers=len(decoded)) as ex:
                future_to_d = {
                    ex.submit(
                        score_step, d[4], trajs[d[1]].answer, d[5] == TOKEN_END,
                        histories[d[1]] if sft_mode else None,
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
                        final_reward=R_final,
                        llm_reward=r_prm,
                        format_reward=format_reward,
                        predicted_next_action=pred_action,
                        gold_next_action=gt_action,
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
            pat_submit_times: dict[int, float] = {i: time.time() for i in pat_active}

            def _call_patcher_timed(i):
                text = _select_best_patcher_candidate(
                    model, tokenizer,
                    trajs[i].problem, trajs[i].answer, histories[i],
                    action_token_ids, _max,
                )
                elapsed = time.time() - pat_submit_times[i]
                logger.info(
                    f"[P{trajs[i].problem_id:>6}] step={step_idx:02d}  PATCHER_DONE"
                    f"  wait={elapsed:.1f}s  len={len(text)}"
                )
                return i, text

            with ThreadPoolExecutor(max_workers=len(pat_active)) as ex:
                pat_futures = {ex.submit(_call_patcher_timed, i): i for i in pat_active}
                pat_result_map = {}
                for fut in as_completed(pat_futures):
                    idx, text = fut.result()
                    pat_result_map[idx] = text
            pat_texts = [pat_result_map[i] for i in pat_active]
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
                # generator가 patcher 스텝 이후의 action token 예측
                act_prompt = _build_gen_prompt(tokenizer, SOLVE, trajs[i].problem, histories[i] + [text])
                act_ids = tokenizer(act_prompt, return_tensors="pt").to(model.device)["input_ids"]
                with torch.no_grad():
                    act_logits = model(act_ids).logits[0, -1, :]
                best_action_id = max(action_token_ids, key=lambda tid: act_logits[tid].item())
                pred_action = action_id_to_token.get(best_action_id, TOKEN_SOLVE)
                return r_prm, inp_ids.cpu(), resp_ids.cpu(), lp_p, pred_action

            with ThreadPoolExecutor(max_workers=len(pat_active)) as ex:
                future_to_pi = {
                    ex.submit(_score_and_logprobs, i, text): (i, text)
                    for i, text in zip(pat_active, pat_texts)
                }
                for fut in as_completed(future_to_pi):
                    i, text = future_to_pi[fut]
                    r_prm, inp_ids, resp_ids, lp_p, pred_action = fut.result()
                    gt_next_s = _next_state_by_reward(CORRECT_PAT, r_prm, text)
                    gt_action = _gt_action_token(gt_next_s)
                    R_final = r_prm

                    if r_prm <= 0.5:
                        trajs[i].patcher_wrong = True

                    _gt_name = gt_action.strip("<|>")
                    _is_ans_pat = check_solved(text, trajs[i].answer)
                    logger.info(
                        f"[P{trajs[i].problem_id:>6}] step={step_idx:02d}  PATCHER_ARRIVED"
                        f"  pred={pred_action.strip('<|>')}  gt={_gt_name}"
                        f"  patcher_wrong={trajs[i].patcher_wrong}"
                        f"  R_PRM={r_prm:.3f}  R_final={R_final:.3f}"
                        f"  tokens={resp_ids.shape[1]}  next={_next_label(gt_next_s)}"
                        f"  is_answer={_is_ans_pat}"
                    )

                    trajs[i].steps.append(StepRecord(
                        step_idx=step_counts[i],
                        state=CORRECT_PAT,
                        action=pred_action.strip("<|>"),
                        text=text,
                        final_reward=R_final,
                        llm_reward=r_prm,
                        format_reward=0.0,
                        predicted_next_action=pred_action,
                        gold_next_action=gt_action,
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


def solve_problem(
    model,
    tokenizer,
    problem: str,
    answer: str,
    problem_id: str = "",
    rollout_path=None,
    difficulty=None,
) -> "Optional[Trajectory]":
    """단일 문제에 대해 trajectory를 생성. solve_problems_batch의 단일 문제 래퍼."""
    items = [{"problem_id": problem_id or "val", "problem": problem, "answer": answer}]
    trajs = solve_problems_batch(model, tokenizer, items, rollout_path=rollout_path)
    if trajs:
        traj = trajs[0]
        if difficulty is not None:
            traj.difficulty = difficulty
        return traj
    return None


def validate_math500(model, tokenizer, problems: List[dict], max_new_tokens: int = None) -> dict:
    """MATH-500 문제에 대한 단일 스텝 정답률 검증.

    각 문제에 대해 한 번 추론하고 \\boxed{} 기반으로 정답 여부를 판정한다.
    """
    _max = max_new_tokens or GENERATOR_MAX_NEW_TOKENS
    n_correct = 0

    for prob in problems:
        prompt = build_chat_prompt(tokenizer, _get_prompts()["system_solve"], _solve_user(prob["problem"], []))
        results = generate_steps_batched(model, tokenizer, [prompt], max_new_tokens=_max)
        if results:
            text, _, _ = results[0]
            if check_solved(text, prob["answer"]):
                n_correct += 1

    n_total = len(problems)
    return {
        "val/accuracy": n_correct / n_total if n_total > 0 else 0.0,
        "val/n_correct": n_correct,
        "val/n_total":   n_total,
    }