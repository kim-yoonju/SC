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
from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait, FIRST_COMPLETED
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

def load_config(config_path="config/config.yaml"):
    """설정 파일을 로드합니다."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config 파일을 찾을 수 없습니다: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"config 파일이 비어 있습니다: {config_path}")
    return cfg

# 실행 시점에 설정 로드
CONF = load_config()

# API 키 (환경변수 우선, 없으면 config; gpt 키 없으면 KeyError)
GPT_API_KEY    = os.environ.get("OPENAI_API_KEY") or CONF["API_key"]["gpt"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or CONF["API_key"].get("gemini")

# API 클라이언트
client = OpenAI(api_key=GPT_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Generator / checkpoint
GENERATOR_MODEL_ID       = CONF["checkpoint"]["base"]
GENERATOR_CACHE_DIR      = CONF["checkpoint"]["cache_dir"]
SFT_CHECKPOINT           = CONF["checkpoint"]["sft_checkpoint"]
GENERATOR_TEMPERATURE    = CONF["step_reasoning"]["temperature"]
GENERATOR_MAX_NEW_TOKENS = CONF["step_reasoning"]["max_new_tokens"]
PATCHER_MAX_NEW_TOKENS   = CONF["API_model"].get("max_new_tokens", 2048)
API_MAX_SEQ_LEN          = CONF["API_model"].get("max_seq_len", 1500)
TRUNCATE_TOKEN_LIMIT     = CONF["step_reasoning"]["truncate_token_limit"]
MAX_STEPS                = CONF.get("generate_trajectory", {}).get("max_steps") or CONF["ppo"].get("max_steps", CONF["step_reasoning"]["max_steps"])
VLLM_MAX_MODEL_LEN       = CONF.get("vllm", {}).get("max_model_len", 32768)

# API 모델
TRUNCATOR     = CONF["API_model"]["TRUNCATOR"]
REWARD        = CONF["API_model"]["REWARD"]
PATCHER       = CONF["API_model"]["PATCHER"]
EXTRACTOR     = CONF["API_model"]["EXTRACTOR"]

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

_PROMPTS_PATH = _pathlib.Path(__file__).resolve().parent.parent / "prompts" / "action_prompts.jsonl"
_PROMPTS: Dict[str, str] = {}
with open(_PROMPTS_PATH) as _f:
    for _line in _f:
        _line = _line.strip()
        if _line:
            _entry = json.loads(_line)
            _PROMPTS[_entry["name"]] = _entry["content"]

SYSTEM_SOLVE             = _PROMPTS["system_solve"]
SYSTEM_CORRECT           = _PROMPTS["system_rethink"]
PATCHER_PROMPT           = _PROMPTS["patcher_prompt"]
SYSTEM_SOLVE_SFT         = _PROMPTS["system_solve_sft"]
SYSTEM_SOLVE_API_SFT     = _PROMPTS["system_solve_api_sft"]
SYSTEM_RETHINK_API_SFT   = _PROMPTS["system_rethink_api_sft"]
SFT_GENERATOR_PROMPT     = _PROMPTS["sft_generator"]
SFT_PATCHER_PROMPT       = _PROMPTS["sft_patcher"]
SFT_PATCHER_STEP_PROMPT  = _PROMPTS["sft_patcher_step"]
SFT_PATCHER_ALL_PROMPT   = _PROMPTS["sft_patcher_all"]
LLM_SCORE_PROMPT         = _PROMPTS["llm_score"]
LLM_SCORE_SFT_PROMPT     = _PROMPTS["llm_score_sft"]
SYSTEM_TRUNCATOR         = _PROMPTS["system_truncator"]

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


def load_generator(device_map="auto", model_path=None):
    load_path = model_path if model_path else SFT_CHECKPOINT
    logger.info(f"Generator 로드 중: {load_path}")

    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    s = re.sub(r"\\left|\\right|[()]", "", s)  # {} 는 제거하지 않음 (행렬 환경 보호)
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
        # 행렬은 직접 비교
        if isinstance(la, Matrix) or isinstance(lb, Matrix):
            return la == lb
        return simplify(la - lb) == 0
    except Exception:
        return False


def _numeric_approx_equal(a: str, b: str, rel_tol: float = 1e-3) -> bool:
    """두 수식을 float로 평가해 상대 오차 0.1% 이내이면 True."""
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

def check_solved(step_text: str, gold_answer, is_gsm8k: bool = False) -> bool:
    pred_raw = extract_boxed(step_text, is_gsm8k=is_gsm8k)
    if not pred_raw: return False
    gold_str = str(gold_answer).strip()
    if "####" in gold_str:
        gold_str = _extract_gsm8k_answer(gold_str)
    pred_raw, gold_raw = pred_raw.replace(" ", ""), gold_str.replace(" ", "")
    if pred_raw == gold_raw: return True
    # 대소문자 무시 비교
    if pred_raw.lower() == gold_raw.lower(): return True
    # True/False ↔ Yes/No 동치
    pb, gb = _normalize_bool(pred_raw), _normalize_bool(gold_raw)
    if pb is not None and gb is not None and pb == gb: return True
    try:
        if abs(float(pred_raw) - float(gold_raw)) < 1e-6: return True
    except ValueError: pass
    if _normalize_latex(pred_raw) == _normalize_latex(gold_raw): return True
    if _canonicalize(pred_raw) == _canonicalize(gold_raw): return True
    # 도(°) ↔ 라디안 동치
    if _angle_equal(pred_raw, gold_raw): return True
    if _latex2sympy_equal(pred_raw, gold_raw): return True
    # 소수 근사 비교 (마지막 — 2/ln5 ≈ 1.242 등)
    return _numeric_approx_equal(pred_raw, gold_raw)

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

def format_correct(response: str, gold: str, is_gsm8k: bool = False) -> bool:
    """정답 추출 후 비교. GSM8K는 #### 형식 사용."""
    pred = extract_boxed(response, is_gsm8k=is_gsm8k)
    if pred is None:
        return False
    return answers_equal(pred, gold)

# ─────────────────────────────────────────────────────────────────────────────
# API 클라이언트 라우터
# ─────────────────────────────────────────────────────────────────────────────

def _is_reasoning_model(model_name: str) -> bool:
    """temperature=0 미지원 / max_completion_tokens 필요한 추론 모델 여부."""
    m = model_name.lower()
    return any(tok in m for tok in ["o1", "o3", "gpt-5", "thinking"])


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

    return resp.text


# ─────────────────────────────────────────────────────────────────────────────
# API 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _gpt(model_name: str, messages: list, max_completion_tokens: int = None, temperature: float = None,
         usage_out: list = None) -> str:
    """모델 종류에 따라 OpenAI 또는 Gemini API를 호출합니다.

    usage_out: 제공 시 {"input_tokens": int, "output_tokens": int} dict를 append.
    """
    import time, re as _re2
    wait = 1.0
    attempt = 0
    while True:
        try:
            if model_name.lower().startswith("gemini"):
                return _call_gemini(model_name, messages, max_output_tokens=max_completion_tokens, temperature=temperature)

            reasoning = _is_reasoning_model(model_name)

            kwargs = {"model": model_name, "messages": messages}
            if max_completion_tokens:
                if reasoning:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                else:
                    kwargs["max_tokens"] = max_completion_tokens
            if not reasoning:
                kwargs["temperature"] = temperature if temperature is not None else 0

            resp = client.chat.completions.create(**kwargs)
            if usage_out is not None and resp.usage:
                usage_out.append({
                    "input_tokens":  resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                })

            choice        = resp.choices[0]
            finish_reason = choice.finish_reason
            content       = choice.message.content

            # 응답 상세 로그 (빈 응답이거나 비정상 종료 시 항상 기록)
            if not content or finish_reason not in ("stop", None):
                _logger = logging.getLogger(__name__)
                _logger.warning(
                    f"[_gpt] model={model_name}  finish_reason={finish_reason!r}"
                    f"  content_len={len(content) if content else 0}"
                    f"  prompt_tokens={resp.usage.prompt_tokens if resp.usage else '?'}"
                    f"  completion_tokens={resp.usage.completion_tokens if resp.usage else '?'}"
                )
                if hasattr(resp.usage, 'completion_tokens_details'):
                    _logger.warning(f"[_gpt] completion_tokens_details={resp.usage.completion_tokens_details}")

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
        result = _gpt(TRUNCATOR, messages, max_completion_tokens=512)
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
        text = _gpt(model, [{"role": "user", "content": prompt}], max_completion_tokens=_max_tokens)
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
    return SYSTEM_CORRECT if action in ("correct", "rethink") else SYSTEM_SOLVE


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


# solve_problems_batch는 generate_trajectory.py로 이동
from generate_trajectory import solve_problems_batch  # noqa: F401


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
        prompt = build_chat_prompt(tokenizer, SYSTEM_SOLVE, _solve_user(prob["problem"], []))
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