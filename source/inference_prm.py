"""
MathShepherdPRM: Qwen 모델로 수학 추론 스텝을 채점하는 PRM.

채점 방식: "이 스텝이 올바른가? 1 또는 0으로만 답하라" 프롬프트 후
"1" 토큰과 "0" 토큰의 로짓 확률비로 점수 계산.

  score = P("1") / (P("1") + P("0"))

→ generate() 없이 단일 forward pass로 연속값 0~1 획득.
→ 모델이 숫자를 텍스트로 생성하지 않아도 되므로 파싱 오류 없음.
"""

import os
import re
import json
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ─────────────────────────────────────────────────────────────
# 프롬프트 로드 (action_prompts.jsonl에서 "prm_prompt" 항목 사용)
# ─────────────────────────────────────────────────────────────

_PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "action_prompts.jsonl"

def _load_prm_prompt() -> str:
    with open(_PROMPTS_PATH) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("name") == "prm_prompt":
                return entry["content"]
    raise ValueError("prm_prompt not found in action_prompts.jsonl")

PRM_PROMPT = _load_prm_prompt()

# 액션 토큰 제거용 패턴 (수학 내용만 PRM에 전달)
_ACTION_TOKEN_RE = re.compile(r"<\|(?:solve|rethink|end|correct)\|>")

def _strip_action_tokens(text: str) -> str:
    return _ACTION_TOKEN_RE.sub("", text).strip()


# ─────────────────────────────────────────────────────────────
# PRM 클래스
# ─────────────────────────────────────────────────────────────

class MathShepherdPRM:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        model_id: str = "Qwen/Qwen2.5-Math-72B-Instruct",
        gpu_id: int = 4,
    ):
        self.device = f"cuda:{gpu_id}"

        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.cache_dir = config.get("checkpoint", {}).get("cache_dir", "/tmp")
        print(f"캐시 경로: {self.cache_dir}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"[{model_id}] GPU {gpu_id}번에 로드 중...")
        hf_token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, cache_dir=self.cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": self.device},
            torch_dtype=torch.float16,
            token=hf_token,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.model.eval()
        print("모델 로드 완료")

        # "1" / "0" 토큰 ID 미리 확인
        self._tok1 = self.tokenizer.encode("1", add_special_tokens=False)
        self._tok0 = self.tokenizer.encode("0", add_special_tokens=False)
        assert len(self._tok1) == 1 and len(self._tok0) == 1, \
            f'"1"/"0" must be single tokens, got {self._tok1} / {self._tok0}'
        self._id1 = self._tok1[0]
        self._id0 = self._tok0[0]
        print(f'token IDs → "1":{self._id1}  "0":{self._id0}')

    def get_step_score(
        self,
        problem: str,
        current_step: str,
        gold_answer: str,
        history: list[str] | None = None,
    ) -> float:
        """
        현재 스텝 품질을 0.0~1.0으로 반환.

        단일 forward pass → 마지막 위치에서 "1"/"0" 로짓 확률비 계산.
          score = P("1") / (P("1") + P("0"))
        """
        history_text = "\n".join(
            f"Step {i+1}: {_strip_action_tokens(s)}" for i, s in enumerate(history or [])
        ) or "(none)"

        user_content = PRM_PROMPT.format(
            problem=problem,
            gold=gold_answer,
            history=history_text,
            response=_strip_action_tokens(current_step),
        )

        messages = [{"role": "user", "content": user_content}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :]   # (vocab,)

        p1 = torch.softmax(logits[[self._id1, self._id0]], dim=0)[0].item()
        return p1   # P("1") / (P("1") + P("0"))


# ─────────────────────────────────────────────────────────────
# 전역 인스턴스 (외부 호출용)
# ─────────────────────────────────────────────────────────────

_prm_instance = None


def get_prm_inference(
    problem: str,
    current_step: str,
    gold_answer: str,
    history: list[str] | None = None,
    gpu_num: int = 4,
    config_file: str = "config/config.yaml",
) -> float:
    global _prm_instance
    if _prm_instance is None:
        _prm_instance = MathShepherdPRM(
            config_path=config_file,
            model_id="Qwen/Qwen2.5-Math-72B-Instruct",
            gpu_id=gpu_num,
        )
    return _prm_instance.get_step_score(problem, current_step, gold_answer, history)


# ─────────────────────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prob    = "Solve for x: 3x + 2 = 11"
    gold    = "3"
    history = ["Subtract 2 from both sides: 3x = 9"]
    step    = "Divide both sides by 3: x = 3."

    reward = get_prm_inference(prob, step, gold, history, gpu_num=4,
                               config_file="config/config.yaml")
    print(f'PRM score (correct step): {reward:.4f}  (expected ~1.0)')

    step_wrong = "Multiply both sides by 3: x = 27."
    reward2 = get_prm_inference(prob, step_wrong, gold, history, gpu_num=4,
                                config_file="config/config.yaml")
    print(f'PRM score (wrong step):   {reward2:.4f}  (expected ~0.0)')
