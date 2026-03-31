"""
SFT 평가 스크립트 — 베이스 모델과 SFT 모델 동시 비교

train_sft.py / prototype/utils.py 와 동일한 포맷으로 두 모델을 순차 평가하고
결과를 나란히 출력한다.

  베이스: Qwen/Qwen2.5-7B-Instruct (학습 전)
  SFT  : output/sft_checkpoints/<최신 run>/epoch* (학습 후)

  데이터: deepmath_16k 뒤에서 100개 고정 샘플

실행:
  python source/evaluate_sft.py
  python source/evaluate_sft.py --sft_model_id checkpoints/sft/.../epoch2
  python source/evaluate_sft.py --gpus 2,3,4,5
"""

import argparse
import datetime
import json
import re
import sys
import torch.multiprocessing as mp
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_problems,
    SYSTEM_SOLVE,
    TOKEN_SOLVE,
    TOKEN_CORRECT,
    TOKEN_END,
    ACTION_TOKENS,
    check_end,
)

# ─────────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent


BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SFT_MODEL_ID = "/mnt/yoonju/SC/output/sft_checkpoints/20260322_202515/epoch3"

CACHE_DIR     = "/mnt/.cache/huggingface"
DATA_PATH     = str(_ROOT / "datasets/deepmath_16k.parquet")
OUT_DIR       = str(_ROOT / "output/eval_sft")

N_SAMPLES      = 100   # 평가 문제 수 (deepmath_16k 뒤에서 100개 고정)
MAX_STEPS      = 20    # 문제당 최대 스텝
MAX_NEW_TOKENS = 2048  # 스텝당 최대 생성 토큰 (SFT 모델은 스텝당 ~243 토큰 생성)
BATCH_SIZE     = 128    # 배치 크기
DEFAULT_GPUS   = "2,3,4,5,6"

# <|end|>은 커스텀 종료 토큰 → 별도 등록 필요
SPECIAL_TOKENS = ACTION_TOKENS
SYSTEM_PROMPT  = SYSTEM_SOLVE   # 평가 시 기본 solve 프롬프트 사용

# ─────────────────────────────────────────────────────────────────────────────
# 정규식
# ─────────────────────────────────────────────────────────────────────────────

BOXED_RE = re.compile(r"\\boxed\{")


def _detect_action(text: str) -> str | None:
    """생성된 텍스트에서 마지막 액션 토큰을 찾아 반환. 없으면 None."""
    for tok in ACTION_TOKENS:
        if tok in text:
            return tok
    return None


def _strip_action(text: str) -> str:
    """생성된 텍스트에서 액션 토큰과 그 이후를 제거한 순수 추론 텍스트 반환."""
    for tok in ACTION_TOKENS:
        idx = text.rfind(tok)
        if idx != -1:
            return text[:idx].rstrip()
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 정답 판정
# ─────────────────────────────────────────────────────────────────────────────

def _extract_boxed(text: str) -> str | None:
    """정답 추출 우선순위: \\boxed{} → #### (GSM8K) → ### 뒤 텍스트."""
    import re as _re
    # 1. \boxed{}
    marker = r"\boxed{"
    pos = text.rfind(marker)
    if pos != -1:
        start = pos + len(marker)
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start:i - 1].strip()
    # 2. GSM8K: 마지막 #### 뒤 숫자
    m = None
    for match in _re.finditer(r"####\s*(.+)", text):
        m = match
    if m:
        return m.group(1).strip().replace(",", "")
    # 3. 마지막 ### 뒤 한 줄
    m = None
    for match in _re.finditer(r"###\s*(.+)", text):
        m = match
    if m:
        return m.group(1).strip()
    return None


def _check_correct(text: str, gold: str) -> bool:
    pred = _extract_boxed(text)
    if pred is None:
        return False
    pred = pred.replace(" ", "")
    gold = gold.strip().replace(" ", "")
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False



# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트 빌더 (train_sft._build_chat_prefix 와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def build_step_prompt(problem: str, history: list[str], tokenizer) -> str:
    """스텝마다 history를 반영해 프롬프트를 새로 구성."""
    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Steps so far]")
        for i, s in enumerate(history, 1):
            lines.append(f"Step {i}: {s}")
    lines.append("\nWrite the next step.")
    user_msg = "\n".join(lines)

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {SYSTEM_PROMPT}\n\nUser: {user_msg}\n\nAssistant:"


# ─────────────────────────────────────────────────────────────────────────────
# 배치 생성 (한 스텝)
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch_step(
    prompts: list[str],
    model,
    tokenizer,
    device,
    stop_ids: list[int],
    eos_id: int,
) -> list[tuple[str, int]]:
    """
    여러 프롬프트를 한 번에 생성 (left-padding 배치).
    반환: [(생성된 텍스트_with_action_token, 생성 토큰 수), ...]
    """
    all_stop_ids = list(set([eos_id] + stop_ids))

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
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=all_stop_ids,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    results = []
    stop_set = set(all_stop_ids)
    for i in range(len(prompts)):
        gen_ids = out[i][padded_len:]

        # 첫 stop 토큰까지만 사용 (포함)
        stop_pos = len(gen_ids)
        for j, tid in enumerate(gen_ids.tolist()):
            if tid in stop_set:
                stop_pos = j + 1
                break
        gen_ids = gen_ids[:stop_pos]

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        # <|endoftext|>만 제거 (<|end|>은 액션 토큰이므로 유지)
        # <|endoftext|>만 제거 (<|end|>은 액션 토큰이므로 유지)
        gen_text = gen_text.replace("<|endoftext|>", "").strip()
        results.append((gen_text, len(gen_ids)))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 문제 풀기
# ─────────────────────────────────────────────────────────────────────────────

def _build_record(item: dict, steps: list[dict]) -> dict:
    total_steps    = len(steps)
    total_tokens   = sum(s["n_tokens"] for s in steps)
    n_with_action  = sum(1 for s in steps if s["action"] is not None)
    action_rate    = n_with_action / total_steps if total_steps else 0.0
    reached_end    = steps[-1]["action"] == TOKEN_END if steps else False
    # 전체 스텝 중 정답 포함 여부
    correct        = any(s["correct"] for s in steps)
    return {
        "problem_id":   item["problem_id"],
        "problem":      item["problem"],
        "answer":       item["answer"],
        "total_steps":  total_steps,
        "total_tokens": total_tokens,
        "action_rate":  action_rate,   # 스텝이 올바른 액션 토큰으로 끝난 비율
        "reached_end":  reached_end,   # <|end|> 에 도달한 문제
        "correct":      correct,       # \\boxed{정답} 일치 여부
        "steps":        steps,
    }


def solve_all(
    items: list[dict],
    model,
    tokenizer,
    device,
    stop_ids: list[int],
    eos_id: int,
    batch_size: int,
    max_steps: int,
    desc: str = "Steps",
    out_path: str | None = None,
) -> list[dict]:
    """모든 문제를 스텝별 배치로 처리. 스텝마다 history로 프롬프트를 재구성한다."""
    histories  = [[] for _ in items]   # 문제별 추론 텍스트 히스토리
    steps_list = [[] for _ in items]
    done       = [False] * len(items)
    written    = [False] * len(items)

    out_f = open(out_path, "a") if out_path else None

    def _write(i: int):
        if out_f and not written[i]:
            record = _build_record(items[i], steps_list[i])
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            written[i] = True

    for step_idx in tqdm(range(max_steps), desc=desc):
        active_idx = [i for i, d in enumerate(done) if not d]
        if not active_idx:
            break

        for b_start in range(0, len(active_idx), batch_size):
            batch_idx = active_idx[b_start: b_start + batch_size]
            # 스텝마다 각 문제의 현재 history로 프롬프트 재구성
            batch_prompts = [
                build_step_prompt(items[i]["problem"], histories[i], tokenizer)
                for i in batch_idx
            ]

            outputs = generate_batch_step(
                batch_prompts, model, tokenizer, device, stop_ids, eos_id,
            )

            for i, (gen_text, n_tokens) in zip(batch_idx, outputs):
                action        = _detect_action(gen_text)
                reasoning     = _strip_action(gen_text)
                has_boxed     = bool(BOXED_RE.search(gen_text))
                is_correct    = _check_correct(gen_text, items[i]["answer"])

                steps_list[i].append({
                    "step_idx":  step_idx,
                    "action":    action,
                    "text":      reasoning,
                    "n_tokens":  n_tokens,
                    "has_boxed": has_boxed,
                    "correct":   is_correct,
                })

                # history에는 액션 토큰 없는 순수 추론 텍스트만 추가
                histories[i].append(reasoning)

                if check_end(gen_text, action) or is_correct:
                    done[i] = True
                    _write(i)

    for i in range(len(items)):
        _write(i)

    if out_f:
        out_f.close()

    return [_build_record(items[i], steps_list[i]) for i in range(len(items))]


# ─────────────────────────────────────────────────────────────────────────────
# 멀티GPU 워커
# ─────────────────────────────────────────────────────────────────────────────

def gpu_worker(gpu_id: int, items: list[dict], model_id: str, load_kwargs: dict,
               batch_size: int, max_steps: int, out_path: str):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # <|solve|>, <|correct|>, <|end|> 등록 (<|end|>은 커스텀 토큰이므로 등록 필요)
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # config.vocab_size가 실제 체크포인트 가중치 크기와 다를 수 있으므로 패치.
    # 베이스 모델은 special token 추가 전이므로 실제 임베딩 크기 = len(tokenizer) - added.
    config = AutoConfig.from_pretrained(model_id, **load_kwargs)
    actual_vocab_size = len(tokenizer) - added
    if config.vocab_size != actual_vocab_size:
        config.vocab_size = actual_vocab_size

    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16, **load_kwargs,
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # stop ID 수집: <|solve|>, <|correct|>, <|end|>
    stop_ids = []
    for tok in ACTION_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid != tokenizer.unk_token_id:
            stop_ids.append(tid)
    eos_id = tokenizer.eos_token_id

    solve_all(
        items, model, tokenizer, device,
        stop_ids, eos_id,
        batch_size=batch_size,
        max_steps=max_steps,
        desc=f"GPU{gpu_id}",
        out_path=out_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model_id",   default=SFT_MODEL_ID,
                   help="SFT 체크포인트 경로. 기본값: output/sft_checkpoints 최신 epoch")
    p.add_argument("--base_model_id",  default=BASE_MODEL_ID,
                   help="베이스 모델 ID. 기본값: Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data_path",      default=DATA_PATH)
    p.add_argument("--n_samples",      type=int, default=N_SAMPLES)
    p.add_argument("--max_steps",      type=int, default=MAX_STEPS)
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--batch_size",     type=int, default=BATCH_SIZE)
    p.add_argument("--out_dir",        default=OUT_DIR)
    p.add_argument("--gpus",           default=DEFAULT_GPUS,
                   help="쉼표로 구분된 GPU 번호 (예: 2,3,4,5)")
    return p.parse_args()


def _run_eval(model_id: str, items: list[dict], gpu_list: list[int],
              batch_size: int, max_steps: int, out_dir: Path, tag: str) -> list[dict]:
    """한 모델에 대해 멀티GPU 평가를 실행하고 records 리스트를 반환."""
    n_gpus = len(gpu_list)
    worker_paths = [str(out_dir / f"{tag}_worker_{i}.jsonl") for i in range(n_gpus)]
    for wp in worker_paths:
        Path(wp).touch()

    is_local    = model_id.startswith("/") or model_id.startswith(".")
    load_kwargs = {"trust_remote_code": True}
    if not is_local:
        load_kwargs["cache_dir"] = CACHE_DIR

    slices    = [items[i::n_gpus] for i in range(n_gpus)]
    processes = []
    for i, (gpu_id, item_slice, worker_path) in enumerate(zip(gpu_list, slices, worker_paths)):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, item_slice, model_id, load_kwargs,
                  batch_size, max_steps, worker_path),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"[{tag}] 워커 {failed} 실패")

    records = []
    for wp in worker_paths:
        with open(wp) as f:
            for line in f:
                records.append(json.loads(line))
    return records


def _summarize(records: list[dict], model_id: str, gpu_list: list[int]) -> dict:
    n = len(records)
    if n == 0:
        return {}
    return {
        "model_id":        model_id,
        "n_samples":       n,
        "gpus":            gpu_list,
        "accuracy":        round(sum(r["correct"]      for r in records) / n, 4),
        "reach_end_rate":  round(sum(r["reached_end"]  for r in records) / n, 4),
        "action_rate":     round(sum(r["action_rate"]  for r in records) / n, 4),
        "avg_steps":       round(sum(r["total_steps"]  for r in records) / n, 2),
        "avg_tokens":      round(sum(r["total_tokens"] for r in records) / n, 1),
    }


def _print_comparison(base_s: dict, sft_s: dict):
    W = 62

    def _fmt(val, is_pct=False):
        if val is None:
            return "  -   "
        return f"{val:.1%}" if is_pct else str(val)

    def _delta(b, s, is_pct=False):
        if b is None or s is None:
            return ""
        d = s - b
        sign = "+" if d >= 0 else ""
        return f"({sign}{d:.1%})" if is_pct else f"({sign}{d:.2f})"

    print("\n" + "=" * W)
    print(f"{'베이스 vs SFT 비교':^{W}}")
    print("=" * W)
    print(f"{'지표':<22} {'베이스':>12} {'SFT':>12} {'차이':>12}")
    print("-" * W)

    rows = [
        ("Accuracy",       "accuracy",       True),
        ("Reach-end Rate", "reach_end_rate", True),
        ("Action Rate",    "action_rate",    True),
        ("Avg Steps",      "avg_steps",      False),
        ("Avg Tokens",     "avg_tokens",     False),
    ]
    for label, key, is_pct in rows:
        b = base_s.get(key)
        s = sft_s.get(key)
        print(f"  {label:<20} {_fmt(b, is_pct):>12} {_fmt(s, is_pct):>12} {_delta(b, s, is_pct):>12}")

    print("=" * W)
    print(f"  베이스: {base_s.get('model_id', '-')}")
    print(f"  SFT   : {sft_s.get('model_id', '-')}")
    print(f"  n     : {base_s.get('n_samples', '-')} 문제  (deepmath_16k 마지막 {base_s.get('n_samples', '-')}개)")
    print("=" * W)


def main():
    args     = parse_args()
    gpu_list = [int(g) for g in args.gpus.split(",")]

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPUs      : {gpu_list}")
    print(f"베이스    : {args.base_model_id}")
    print(f"SFT       : {args.sft_model_id}")
    print(f"데이터    : {args.data_path}  (마지막 {args.n_samples}개)")
    print(f"결과 경로 : {out_dir}")

    items = load_problems(args.data_path, args.n_samples)
    print(f"로드된 문제 수: {len(items)}\n")

    mp.set_start_method("spawn", force=True)

    # ── 1) 베이스 모델 평가 ───────────────────────────────────────────────────
    print(f"[1/2] 베이스 모델 평가 중: {args.base_model_id}")
    base_records = _run_eval(
        args.base_model_id, items, gpu_list,
        args.batch_size, args.max_steps, out_dir, tag="base",
    )
    base_s = _summarize(base_records, args.base_model_id, gpu_list)
    (out_dir / "base.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in base_records)
    )

    # ── 2) SFT 모델 평가 ─────────────────────────────────────────────────────
    print(f"\n[2/2] SFT 모델 평가 중: {args.sft_model_id}")
    sft_records = _run_eval(
        args.sft_model_id, items, gpu_list,
        args.batch_size, args.max_steps, out_dir, tag="sft",
    )
    sft_s = _summarize(sft_records, args.sft_model_id, gpu_list)
    (out_dir / "sft.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sft_records)
    )

    # ── 3) 비교 출력 & 저장 ──────────────────────────────────────────────────
    _print_comparison(base_s, sft_s)

    summary = {"base": base_s, "sft": sft_s}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n결과 저장: {out_dir}")


if __name__ == "__main__":
    main()
