"""
MathShepherdPRM: Qwen2.5-Math-PRM-72B 로 수학 추론 스텝을 채점하는 PRM.

채점 방식:
  - 스텝 구분자 <extra_0> 위치에서 모델이 출력하는 2-클래스 로짓을 softmax
  - score = P("good") = softmax(logits)[0]   at the last <extra_0> position

→ AutoModel (Reward Model) 로드, generate() 없이 단일 forward pass.
→ config/config.yaml 의 PRM 섹션에서 model_id / gpu_id / revision 읽음.
"""

import os
import re
import json
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# 액션 토큰 제거용 패턴
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
        model_id: str | None = None,
        gpu_id: int | None = None,
        revision: str | None = None,
    ):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        prm_cfg = config.get("PRM", {})
        if model_id is None:
            model_id = prm_cfg.get("model_id", "Qwen/Qwen2.5-Math-PRM-72B")
        if gpu_id is None:
            gpu_id = int(prm_cfg.get("gpu_id", 4))
        if revision is None:
            revision = prm_cfg.get("revision", None)

        self.device = f"cuda:{gpu_id}"
        self.cache_dir = config.get("checkpoint", {}).get("cache_dir", "/tmp")
        print(f"캐시 경로: {self.cache_dir}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"[{model_id}] GPU {gpu_id}번에 로드 중... (revision={revision})")
        hf_token = os.getenv("HF_TOKEN")

        # Qwen2.5-Math-PRM-72B 는 Reward Model → AutoModel 사용
        # revision 고정으로 trust_remote_code 보안 위험 최소화
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=self.cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": self.device},
            dtype=torch.float16,
            token=hf_token,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            revision=revision,
        )
        self.model.eval()
        print("모델 로드 완료")

        # <extra_0> 스텝 구분자 토큰 ID
        sep_ids = self.tokenizer.encode("<extra_0>", add_special_tokens=False)
        assert len(sep_ids) == 1, f"<extra_0> must be a single token, got {sep_ids}"
        self._step_sep_id = sep_ids[0]
        print(f"step separator token id: {self._step_sep_id}")

    def get_step_score(
        self,
        problem: str,
        current_step: str,
        gold_answer: str,
        history: list[str] | None = None,
        is_correct: bool | None = None,
    ) -> float:
        """
        현재 스텝 품질을 0.0~1.0으로 반환.

        히스토리 스텝 + 현재 스텝을 <extra_0> 로 이어 붙인 뒤,
        마지막 <extra_0> 위치에서 모델 출력 확률의 index-1 (good) 값을 반환.

        is_correct: 전체 풀이가 맞았는지 여부. None이면 미포함.
        """
        all_steps = list(history or []) + [_strip_action_tokens(current_step)]
        # 각 스텝을 <extra_0> 으로 구분, 마지막에도 붙임
        solution = "<extra_0>".join(_strip_action_tokens(s) for s in all_steps) + "<extra_0>"

        if is_correct is None:
            system_prompt = _SYSTEM_PROMPT
        else:
            outcome = "correct" if is_correct else "incorrect"
            system_prompt = f"{_SYSTEM_PROMPT} Note: this solution reaches a {outcome} final answer."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": problem},
            {"role": "assistant", "content": solution},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(**inputs, use_cache=False)

        # output[0]: (batch, seq_len, 2)  →  [bad_logit, good_logit] per token  (class 0=bad, 1=good)
        logits = output[0]
        token_ids = inputs["input_ids"][0]

        sep_positions = (token_ids == self._step_sep_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) == 0:
            return 0.5   # fallback

        last_pos = sep_positions[-1]
        probs = torch.softmax(logits[0, last_pos], dim=-1)
        return probs[1].item()   # P(good): class 1 = positive/good, class 0 = negative/bad


# ─────────────────────────────────────────────────────────────
# 전역 인스턴스 (외부 호출용)
# ─────────────────────────────────────────────────────────────

_prm_instance = None


def get_prm_inference(
    problem: str,
    current_step: str,
    gold_answer: str,
    history: list[str] | None = None,
    is_correct: bool | None = None,
    gpu_num: int | None = None,
    config_file: str = "config/config.yaml",
) -> float:
    global _prm_instance
    if _prm_instance is None:
        _prm_instance = MathShepherdPRM(
            config_path=config_file,
            model_id=None,   # config에서 읽음
            gpu_id=gpu_num,  # None이면 config에서 읽음
        )
    return _prm_instance.get_step_score(problem, current_step, gold_answer, history, is_correct)


# ─────────────────────────────────────────────────────────────
# 단독 실행 테스트: train_ppo_data.jsonl 에서 5개 샘플 채점
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from collections import defaultdict

    ROOT        = Path(__file__).parent.parent
    DATA_PATH   = ROOT / "output" / "train_ppo_data.jsonl"
    CONFIG_PATH = ROOT / "config" / "config.yaml"
    N_SAMPLES   = 5

    def _infer_state(step):
        if "state" in step:
            return step["state"]
        for v in (step.get("action", ""), step.get("gold_next_action", "") or "",
                  step.get("predicted_next_action", "") or ""):
            if "correct" in v:
                return "correct"
        return step.get("action", "")

    if DATA_PATH.exists():
        samples = []
        steps_by_problem = defaultdict(list)

        with open(DATA_PATH) as f:
            for line in f:
                if len(samples) >= N_SAMPLES * 10:
                    break
                d = json.loads(line)
                gold = str(d.get("gold_answer", ""))
                for step in d["steps"]:
                    if step["text"] == "...":
                        continue
                    entry = {
                        "problem_id":  d["problem_id"],
                        "problem":     d["problem"],
                        "gold_answer": gold,
                        "step_idx":    step["step_idx"],
                        "state":       _infer_state(step),
                        "text":        step["text"],
                        "llm_reward":  step.get("llm_reward", 0.0),
                    }
                    steps_by_problem[d["problem_id"]].append(entry)
                    if len(samples) < N_SAMPLES:
                        samples.append(entry)

        print(f"\n{'='*70}")
        print(f"{'idx':>3}  {'problem_id':>10}  {'step':>4}  {'state':>12}  "
              f"{'prm':>6}  {'llm':>6}  {'diff':>6}")
        print(f"{'-'*70}")

        for i, s in enumerate(samples):
            hist = [e["text"] for e in steps_by_problem[s["problem_id"]]
                    if e["step_idx"] < s["step_idx"]]
            score = get_prm_inference(
                problem=s["problem"],
                current_step=s["text"],
                gold_answer=s["gold_answer"],
                history=hist,
                config_file=str(CONFIG_PATH),
            )
            diff = score - s["llm_reward"]
            flag = "  ← LARGE" if abs(diff) > 0.4 else ""
            print(f"{i:>3}  {s['problem_id']:>10}  {s['step_idx']:>4}  {s['state']:>12}  "
                  f"{score:>6.3f}  {s['llm_reward']:>6.3f}  {diff:>+6.3f}{flag}")
            print(f"     problem : {s['problem'][:80]!r}")
            print(f"     step    : {s['text'][:100]!r}")
            print()

        print(f"{'='*70}")

    else:
        # fallback: 하드코딩 예제
        print(f"[경고] {DATA_PATH} 가 없어 하드코딩 예제로 테스트합니다.")
        prob    = "Solve for x: 3x + 2 = 11"
        gold    = "3"
        history = ["Subtract 2 from both sides: 3x = 9"]

        cases = [
            ("Divide both sides by 3: x = 3.", "correct"),
            ("Multiply both sides by 3: x = 27.", "wrong"),
        ]
        for step_text, label in cases:
            score = get_prm_inference(prob, step_text, gold, history,
                                      config_file=str(CONFIG_PATH))
            print(f"PRM score ({label:>7} step): {score:.4f}")
