"""
SFT 체크포인트 추론 테스트
gen_solve_R 프롬프트로 포맷을 잘 따르는지 확인.

실행:
  python test_sft_inference.py
  python test_sft_inference.py --model checkpoints/sft/20260504_130650/epoch3
  python test_sft_inference.py --gpus 4,5
"""

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_ROOT_PROJ = _ROOT.parent
sys.path.insert(0, str(_ROOT))

os.environ["HF_HUB_CACHE"] = "/mnt/.cache/huggingface"

DEFAULT_CHECKPOINT = "Qwen/Qwen2.5-7B-Instruct"# "/mnt/yoonju/SC/checkpoints/sft/20260505_130300/epoch3"
DEFAULT_GPUS = [2]

from utils import resolve_model_path as _resolve_model_path

# 문자열: 모델이 step + critic 전체 생성
# (problem, wrong_step) 튜플: wrong_step을 prefix로 주입하고 모델은 critic 섹션만 생성
SAMPLE_PROBLEMS = [
    "x=2, y=3, x+y=?",
    (
        "Find the sum of all integers from 1 to 100.",
        "The sum of integers from 1 to 100 is $100 \\times 100 = 10000$.",
    ),
    (
        "Differentiate $f(x) = x^3 \\sin x$.",
        "Using the chain rule: $f'(x) = 3x^2 \\cos x$.",
    ),
    (
        "How many ways can 3 books be chosen from 5 distinct books?",
        "The number of ways is $5 \\times 4 \\times 3 = 60$.",
    ),
]


def build_rubric_str() -> str:
    from utils import CONF
    rubric_rel = CONF.get("PRM", {}).get("rubric", "prompts/prm_rubric_v6.2.jsonl")
    rubric_path = Path(rubric_rel) if Path(rubric_rel).is_absolute() else _ROOT_PROJ / rubric_rel
    lines = []
    with open(rubric_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                lines.append(f"{e['name']}: [correct/incorrect — {e['criterion']}]")
    return "\n".join(lines)


def build_gen_solve_prompt() -> str:
    with open(_ROOT_PROJ / "prompts" / "action_prompts.json", encoding="utf-8") as f:
        for entry in json.load(f):
            if entry["name"] == "gen_solve_R":
                return entry["content"].replace("{{rubric}}", build_rubric_str())
    raise ValueError("gen_solve_R not found in action_prompts.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_CHECKPOINT, help="체크포인트 경로 (절대 또는 프로젝트 루트 상대)")
    parser.add_argument("--gpus", default=",".join(str(g) for g in DEFAULT_GPUS))
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    model_path, cache_dir = _resolve_model_path(args.model)
    gpus = [g.strip() for g in args.gpus.split(",")]

    print(f"모델: {model_path}")
    if cache_dir:
        print(f"캐시: {cache_dir}")
    print(f"GPU : {gpus}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    system_prompt = build_gen_solve_prompt()
    print("\n[gen_solve_R 시스템 프롬프트 (앞 200자)]")
    print(system_prompt[:200], "...\n")

    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    action_tokens = ["<|solve|>", "<|rethink|>", "<|end|>"]
    stop_ids = [
        tokenizer.encode(tok, add_special_tokens=False)[0]
        for tok in action_tokens
        if tokenizer.encode(tok, add_special_tokens=False)
    ]
    print(f"Stop token IDs: {dict(zip(action_tokens, [tokenizer.encode(t, add_special_tokens=False) for t in action_tokens]))}")

    from utils import build_chat_prompt

    SEP = "=" * 70

    def _generate(input_ids):
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature if args.temperature > 0 else None,
                do_sample=args.temperature > 0,
                eos_token_id=stop_ids + [tokenizer.eos_token_id],
                pad_token_id=tokenizer.pad_token_id,
            )
        return out[0][input_ids.shape[1]:]

    def _format_response(prefix: str, generated_ids) -> str:
        tail = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        # 베이스 모델 EOS 토큰 제거 (예: <|im_end|>)
        for eos in ["<|im_end|>", "<|endoftext|>"]:
            tail = tail.replace(eos, "").strip()
        response = (prefix.strip() + "\n\n" + tail).strip() if prefix else tail
        # action token이 response 어딘가에 있으면 그 뒤를 잘라 Next action: 섹션으로 정리
        stop_tok = next((t for t in action_tokens if response.endswith(t)), None)
        if stop_tok:
            response = response[: -len(stop_tok)].rstrip()
            if "Next action:" in response:
                response = response + "\n" + stop_tok
            else:
                response = response + "\n\nNext action:\n" + stop_tok
        return response

    def _print_result(label: str, response: str):
        print(f"{SEP}")
        print(label)
        print(f"{SEP}")
        print(response)
        print()
        print(f"[포맷 체크]")
        print(f"  Fast critic  : {'O' if 'Fast critic:' in response else 'X'}")
        print(f"  Deep critic  : {'O' if 'Deep critic:' in response else 'X'}")
        print(f"  Fail rubrics : {'O' if 'Fail rubrics:' in response else 'X (없음 — 정답 스텝이면 정상)'}")
        print(f"  Next action  : {'O' if 'Next action:' in response else 'X'}")
        print(f"  \\boxed{{}}   : {'O' if chr(92) + 'boxed{' in response else 'X (없음 — 중간 단계면 정상)'}")
        print()

    for i, entry in enumerate(SAMPLE_PROBLEMS, 1):
        if isinstance(entry, tuple):
            problem, wrong_step = entry
            prompt = build_chat_prompt(tokenizer, system_prompt, problem)
            input_ids = tokenizer(prompt + wrong_step, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            generated = _generate(input_ids)
            response = _format_response(wrong_step, generated)
            _print_result(f"[{i}] {problem}\n[주입 step] {wrong_step}", response)
        else:
            problem = entry
            prompt = build_chat_prompt(tokenizer, system_prompt, problem)
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            generated = _generate(input_ids)
            response = _format_response("", generated)
            _print_result(f"[{i}] {problem}", response)


if __name__ == "__main__":
    main()
