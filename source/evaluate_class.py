"""
sft_eval.jsonl 샘플 평가 — vLLM 버전 (빠른 병렬 추론).
evaluate_sample.py와 동일한 로직이지만 HuggingFace generate 대신
vLLM을 사용해 멀티-GPU tensor parallel로 전체 스텝을 한 번에 추론.

실행:
  python source/evaluate_sample_vllm.py
  python source/evaluate_sample_vllm.py --model checkpoints/sft/20260505_130300/epoch3
  python source/evaluate_sample_vllm.py --eval_data output/sft_eval.jsonl
  python source/evaluate_sample_vllm.py --gpus 2,3,4,5 --max_samples 30
  python source/evaluate_sample_vllm.py --out output/eval_sample_result.json
  python source/evaluate_sample_vllm.py --gpu_memory_utilization 0.85 --max_model_len 8192
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

DEFAULT_CHECKPOINT = "/mnt/yoonju/SC/checkpoints/sft/20260506_031634/epoch3"
DEFAULT_GPUS = "4,5,6,7"
DEFAULT_EVAL_DATA = str(_ROOT_PROJ / "output" / "sft_eval.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# 출력 파싱
# ─────────────────────────────────────────────────────────────────────────────

def force_action_from_logprobs(
    vllm_out,
    action_token_ids: dict[int, str],
) -> str | None:
    """
    action token이 생성되지 않았을 때, 마지막 생성 위치의 logprobs에서
    action token 중 가장 확률이 높은 것을 반환. 없으면 None.
    """
    logprobs_list = vllm_out.outputs[0].logprobs
    if not logprobs_list:
        return None
    # 마지막 토큰 위치의 logprob dict에서 action 토큰 탐색
    for step_logprobs in reversed(logprobs_list):
        best_tok, best_lp = None, float("-inf")
        for tid, tok_str in action_token_ids.items():
            lp_obj = step_logprobs.get(tid)
            if lp_obj is not None:
                lp = lp_obj.logprob if hasattr(lp_obj, "logprob") else lp_obj
                if lp > best_lp:
                    best_lp, best_tok = lp, tok_str
        if best_tok is not None:
            return best_tok
    return None


def parse_output(
    gen_text: str,
    action_tokens: list[str],
    rubric_token_to_name: dict[str, str],
) -> tuple[str | None, list[str]]:
    """
    모델 생성 텍스트에서 (next_action, fail_rubrics) 추출.
      next_action  : action token 문자열 (없으면 None)
      fail_rubrics : rubric 이름 목록
    """
    next_action = None
    na_pos = gen_text.find("Next action:")
    search_area = gen_text[na_pos:] if na_pos != -1 else gen_text
    for tok in action_tokens:
        if tok in search_area:
            next_action = tok
            break

    fail_rubrics: list[str] = []
    fr_pos = gen_text.find("Fail rubrics:")
    if fr_pos != -1:
        end = na_pos if na_pos != -1 else len(gen_text)
        section = gen_text[fr_pos:end]
        for tok, name in rubric_token_to_name.items():
            if tok in section:
                fail_rubrics.append(name)

    return next_action, fail_rubrics


def clean_gen_text(text: str) -> str:
    """채팅 템플릿 end-of-turn 토큰 이후를 잘라낸다."""
    for sep in ("<|im_end|>", "<|endoftext|>", "</s>"):
        pos = text.find(sep)
        if pos != -1:
            text = text[:pos]
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 메트릭 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_action_metrics(golds: list[str], preds: list[str], classes: list[str]) -> dict:
    from sklearn.metrics import classification_report, accuracy_score
    acc = float(accuracy_score(golds, preds))
    report = classification_report(golds, preds, labels=classes, output_dict=True, zero_division=0)
    report_clean = {
        cls: {k: float(v) if isinstance(v, float) else v for k, v in vals.items()}
        if isinstance(vals, dict) else float(vals)
        for cls, vals in report.items()
    }
    return {"accuracy": acc, "report": report_clean}


def compute_rubric_metrics(
    golds: list[list[str]], preds: list[list[str]], rubric_names: list[str]
) -> dict:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    n, m = len(golds), len(rubric_names)
    idx = {r: i for i, r in enumerate(rubric_names)}

    y_true = np.zeros((n, m), dtype=int)
    y_pred = np.zeros((n, m), dtype=int)
    for i, (g, p) in enumerate(zip(golds, preds)):
        for r in g:
            if r in idx:
                y_true[i, idx[r]] = 1
        for r in p:
            if r in idx:
                y_pred[i, idx[r]] = 1

    per_rubric = {}
    for j, name in enumerate(rubric_names):
        per_rubric[name] = {
            "accuracy":  float(accuracy_score(y_true[:, j], y_pred[:, j])),
            "precision": float(precision_score(y_true[:, j], y_pred[:, j], zero_division=0)),
            "recall":    float(recall_score(y_true[:, j], y_pred[:, j], zero_division=0)),
            "f1":        float(f1_score(y_true[:, j], y_pred[:, j], zero_division=0)),
            "support":   int(y_true[:, j].sum()),
        }

    return {
        "micro_f1":        float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall":    float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_rubric":      per_rubric,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_action_results(metrics: dict, classes: list[str]):
    print("\n" + "=" * 62)
    print("[ Next Action — 3-class Classification ]")
    print("=" * 62)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    rpt = metrics["report"]
    print(f"  Macro F1 : {rpt.get('macro avg', {}).get('f1-score', 0):.4f}")
    print()
    hdr = f"  {'Class':<14}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Sup':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cls in classes:
        c = rpt.get(cls, {})
        print(
            f"  {cls:<14}  {c.get('precision', 0):>6.3f}"
            f"  {c.get('recall', 0):>6.3f}"
            f"  {c.get('f1-score', 0):>6.3f}"
            f"  {int(c.get('support', 0)):>6}"
        )


def print_rubric_results(metrics: dict, rubric_names: list[str]):
    print("\n" + "=" * 68)
    print("[ Fail Rubric — Per-rubric Binary Classification ]")
    print("=" * 68)
    print(f"  Micro F1     : {metrics['micro_f1']:.4f}")
    print(f"  Macro F1     : {metrics['macro_f1']:.4f}")
    print(f"  Micro Prec   : {metrics['micro_precision']:.4f}")
    print(f"  Micro Recall : {metrics['micro_recall']:.4f}")
    print()
    hdr = f"  {'Rubric':<42}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Sup':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    pr = metrics["per_rubric"]
    for name in rubric_names:
        c = pr.get(name, {})
        print(
            f"  {name:<42}  {c.get('accuracy', 0):>6.3f}"
            f"  {c.get('precision', 0):>6.3f}"
            f"  {c.get('recall', 0):>6.3f}"
            f"  {c.get('f1', 0):>6.3f}"
            f"  {c.get('support', 0):>5}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",                  default=DEFAULT_CHECKPOINT)
    parser.add_argument("--eval_data",              default=DEFAULT_EVAL_DATA)
    parser.add_argument("--gpus",                   default=DEFAULT_GPUS)
    parser.add_argument("--max_samples",            type=int,   default=None,  help="최대 trajectory 수 (디버그용)")
    parser.add_argument("--max_tokens",             type=int,   default=1024)
    parser.add_argument("--batch_size",             type=int,   default=32,    help="전체 동시 처리 시퀀스 수 (max_num_seqs). 클수록 메모리 사용 증가")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,  help="vLLM GPU 메모리 사용률 (0~1). 크래시 시 낮춰볼 것")
    parser.add_argument("--max_model_len",          type=int,   default=8192,  help="vLLM 최대 시퀀스 길이. None이면 모델 기본값(32768)으로 KV cache 폭증")
    parser.add_argument("--inference_chunk",        type=int,   default=200,   help="llm.generate에 한 번에 넘길 프롬프트 수 (CPU 메모리 스파이크 방지)")
    parser.add_argument("--out",                    default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    from utils import resolve_model_path as _resolve_model_path
    model_path, cache_dir = _resolve_model_path(args.model)

    gpu_list = args.gpus.split(",")
    n_gpus = len(gpu_list)
    max_num_seqs = args.batch_size  # 전체 동시 처리 수 (GPU 수와 무관)
    print(f"모델     : {model_path}")
    print(f"GPU      : {args.gpus}  ({n_gpus}개, tensor_parallel_size={n_gpus})")
    print(f"배치     : max_num_seqs={max_num_seqs}, gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"데이터   : {args.eval_data}")

    # vLLM은 CUDA_VISIBLE_DEVICES로 사용할 GPU를 제한하고
    # tensor_parallel_size로 그 GPU들에 모델을 분산
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # flash_attn/vllm_flash_attn이 torch 2.9.1과 ABI 불일치로 깨진 환경이므로
    # flashinfer (정상 작동 확인)를 명시적으로 사용
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from utils_sft import (
        ACTION_TOKENS, TOKEN_SOLVE, TOKEN_RETHINK,
        build_chat_prompt, build_messages,
    )
    from preprocess import RUBRIC_TOKENS, get_system_prompts
    from utils_sft import CONF

    rubric_file = CONF.get("PRM", {}).get("rubric", "prompts/prm_rubric_v6.2.jsonl")
    rubric_path = (
        Path(rubric_file) if Path(rubric_file).is_absolute() else _ROOT_PROJ / rubric_file
    )
    rubric_names: list[str] = []
    with open(rubric_path, encoding="utf-8") as rf:
        for ln in rf:
            ln = ln.strip()
            if ln:
                rubric_names.append(json.loads(ln)["name"])

    rubric_token_to_name: dict[str, str] = {v: k for k, v in RUBRIC_TOKENS.items()}
    system_solve, system_rethink = get_system_prompts()

    # 프롬프트 빌드용 토크나이저 (vLLM은 내부에서 별도 토크나이저 사용)
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # action token → token ID 매핑 (special token이 있으면 stop_token_ids로 사용)
    stop_token_ids = [
        tid for tok in ACTION_TOKENS
        if (tid := tokenizer.convert_tokens_to_ids(tok)) not in (None, tokenizer.unk_token_id)
    ]
    stop_token_ids.append(tokenizer.eos_token_id)
    print(f"action stop IDs : {dict(zip(ACTION_TOKENS, stop_token_ids[:-1])) if len(stop_token_ids) > 1 else '없음 (base 모델)'}")

    # vLLM 엔진 초기화
    print("vLLM 모델 로드 중...")
    llm_kwargs = dict(
        model=model_path,
        download_dir=cache_dir,
        tensor_parallel_size=n_gpus,
        max_num_seqs=max_num_seqs,
        max_model_len=args.max_model_len,  # 항상 명시적으로 설정 (None 허용 안 함)
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0,          # greedy (원본 코드와 동일)
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,  # 루브릭 special token이 gen_text에 남도록
        logprobs=20,                # action token 강제 파싱용 (vLLM 최대값)
    )

    # ── 데이터 로드 & 스텝 평탄화 ────────────────────────────────────────────
    trajectories: list[dict] = []
    with open(args.eval_data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    if args.max_samples:
        trajectories = trajectories[: args.max_samples]

    flat: list[dict] = []
    for traj in trajectories:
        problem = traj["problem"]
        steps   = traj["steps"]
        for k, step in enumerate(steps):
            sys_str, user_str = build_messages(problem, steps, k, system_solve, system_rethink)
            inference  = step.get("inference", "")
            full_input = build_chat_prompt(tokenizer, sys_str, user_str) + inference
            flat.append({
                "traj_id":     traj["traj_id"],
                "step_idx":    k,
                "state":       step.get("state"),
                "is_error":    step.get("is_error"),
                "gold_action": step.get("next_gold_action") or TOKEN_SOLVE,
                "gold_fail":   step.get("fail_rubrics") or [],
                "full_input":  full_input,
                "user_str":    user_str,
                "inference":   inference,
            })

    total_steps = len(flat)
    print(f"trajectory {len(trajectories)}개, 총 스텝 {total_steps}개")

    # ── vLLM 청크 단위 추론 ───────────────────────────────────────────────────
    # 전체를 한 번에 넘기면 CPU 메모리 스파이크 발생 → inference_chunk 단위로 분할
    prompts = [s["full_input"] for s in flat]
    chunk_size = args.inference_chunk
    print(f"vLLM 추론 시작 (총 {len(prompts)}개, chunk={chunk_size})...")
    all_outputs = []
    for chunk_start in range(0, len(prompts), chunk_size):
        chunk = prompts[chunk_start : chunk_start + chunk_size]
        print(f"  [{chunk_start+1}–{chunk_start+len(chunk)}/{len(prompts)}] 추론 중...")
        all_outputs.extend(llm.generate(chunk, sampling_params))
    outputs = all_outputs

    # ── 결과 파싱 ─────────────────────────────────────────────────────────────
    gold_actions: list[str] = []
    pred_actions: list[str] = []
    gold_rubrics: list[list[str]] = []
    pred_rubrics: list[list[str]] = []
    samples_out:  list[dict] = []

    # stop_token_id → action token 역매핑 (vLLM은 action token을 생성하면 output에 포함하지 않고 멈춤)
    stop_id_to_action = {
        tokenizer.convert_tokens_to_ids(tok): tok
        for tok in ACTION_TOKENS
        if tokenizer.convert_tokens_to_ids(tok) not in (None, tokenizer.unk_token_id)
    }

    stop_reason_counts: dict = {}
    for i, (s, vllm_out) in enumerate(zip(flat, outputs)):
        gen_text = clean_gen_text(vllm_out.outputs[0].text)
        pred_action, pred_fail = parse_output(gen_text, ACTION_TOKENS, rubric_token_to_name)

        # action token이 stop token으로 소비되어 gen_text에 없는 경우 복원
        stopped_id = vllm_out.outputs[0].stop_reason
        stop_reason_counts[stopped_id] = stop_reason_counts.get(stopped_id, 0) + 1
        if pred_action is None:
            pred_action = stop_id_to_action.get(stopped_id)
        # action token 미생성 시 logprobs에서 강제 파싱
        if pred_action is None:
            pred_action = force_action_from_logprobs(vllm_out, stop_id_to_action)

        gold_actions.append(s["gold_action"])
        pred_actions.append(pred_action if pred_action else TOKEN_RETHINK)
        gold_rubrics.append(s["gold_fail"])
        pred_rubrics.append(pred_fail)
        samples_out.append({
            "traj_id":      s["traj_id"],
            "step_idx":     s["step_idx"],
            "state":        s["state"],
            "is_error":     s["is_error"],
            "gold_action":  s["gold_action"],
            "pred_action":  pred_action,
            "gold_rubrics": s["gold_fail"],
            "pred_rubrics": pred_fail,
            "gen_text":     gen_text,
        })

        if i == 0:
            SEP = "=" * 62
            stopped_id = vllm_out.outputs[0].stop_reason
            finish_reason = vllm_out.outputs[0].finish_reason
            print(f"\n{SEP}")
            print("[ 샘플 #0 디버그 출력 ]")
            print(SEP)
            print(f"  traj_id      : {s['traj_id']}")
            print(f"  step_idx     : {s['step_idx']}  state={s['state']}  is_error={s['is_error']}")
            print(f"  gold_action  : {s['gold_action']}")
            print(f"  pred_action  : {pred_action}  (finish={finish_reason}, stop_id={stopped_id})")
            print(f"  gold_rubrics : {s['gold_fail']}")
            print(f"  pred_rubrics : {pred_fail}")
            print(f"\n--- 입력 (user 메시지) ---")
            print(s["user_str"])
            print(f"\n--- inference (주입된 prefix) ---")
            print(s["inference"][:300] + ("..." if len(s["inference"]) > 300 else ""))
            print(f"\n--- 모델 생성 (critic 섹션) ---")
            print(gen_text)
            if pred_action and pred_action not in gen_text:
                print(f"[stop token으로 생성됨: {pred_action}]")
            print("--- end ---\n")

    # ── stop_reason 분포 출력 (디버그용) ─────────────────────────────────────
    id_to_label = {v: k for k, v in stop_id_to_action.items()}
    id_to_label[tokenizer.eos_token_id] = "<EOS/im_end>"
    print("\n[ stop_reason 분포 ]")
    for sid, cnt in sorted(stop_reason_counts.items(), key=lambda x: -x[1]):
        label = id_to_label.get(sid, str(sid))
        print(f"  {label}: {cnt}")

    # ── 메트릭 계산 & 출력 ────────────────────────────────────────────────────
    action_metrics = compute_action_metrics(gold_actions, pred_actions, ACTION_TOKENS)
    rubric_metrics = compute_rubric_metrics(gold_rubrics, pred_rubrics, rubric_names)

    print_action_results(action_metrics, ACTION_TOKENS)
    print_rubric_results(rubric_metrics, rubric_names)

    out_path = args.out or str(_ROOT_PROJ / "output" / "eval_sample_result.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model":   model_path,
                "n_steps": len(gold_actions),
                "action":  action_metrics,
                "rubric":  rubric_metrics,
                "samples": samples_out,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
