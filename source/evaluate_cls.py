"""
Classifier Head 성능 평가 스크립트 (MCE 리워드 버전)

평가 방식:
  - Qwen2.5-7B-Instruct로 스텝을 하나씩 생성
  - 각 스텝 생성 전에 학습된 classifier head로 다음 액션(solve/correct) 예측
  - 액션 예측 결과에 따라 다른 프롬프트로 LLM 호출:
      solve   → 일반 프롬프트
      correct → CORRECT_HINT가 추가된 프롬프트

Ground truth 결정 (MCE 리워드 기반):
  - 스텝 i를 생성한 뒤, 그 스텝 이후부터 rollout을 n_rollouts번 수행
  - (정답에 도달한 rollout 수) / n_rollouts = MCE_reward(i)
  - 스텝 i+1의 GT:
      MCE_reward(i) == 0.0  →  gt = correct  (스텝 i가 나빴으므로 수정 필요)
      MCE_reward(i)  > 0.0  →  gt = solve
  - 첫 번째 스텝은 항상 gt = solve (직전 MCE 없음)

출력:
  - 문제별로 gt_actions 리스트, pred_actions 리스트 출력
  - 생성된 모든 데이터를 jsonl로 저장

GPU: cuda:4 단일 사용
데이터: DeepMath-103K (16 000문제)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["HF_HOME"] = "/mnt/.cache/huggingface"

# ---------------------------------------------------------------------------
# 하이퍼파라미터 (여기서 수정)
# ---------------------------------------------------------------------------
CLS_HEAD_PATH = "checkpoints/action_cls/20260319_093003/epoch_009/classifier_head.pt"
GPU_ID = 0   # 사용할 GPU 번호 (cuda:GPU_ID)
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    SYSTEM_PROMPT,
    apply_chat_template,
    check_answer_correct,
    extract_first_action,
    format_cls_messages,
    parse_boxed,
    parse_step,
)

# correct 액션 유도 힌트
CORRECT_HINT = (
    "The previous step appears to be incorrect or unhelpful. "
    "Please correct it using the <correct>...</correct> tag."
)


# ---------------------------------------------------------------------------
# 프롬프트 구성
# ---------------------------------------------------------------------------

def format_messages_for_action(problem: str, history: list, predicted_action: str) -> list:
    """predicted_action에 따라 다른 프롬프트를 구성한다."""
    user_content = f"Problem:\n{problem}"
    if history:
        user_content += "\n\nPrevious steps:\n"
        for i, step in enumerate(history, 1):
            user_content += f"Step {i}: {step}\n"
    user_content += "\nGenerate your next step (one action tag only):"
    if predicted_action == "correct":
        user_content += f"\n\n{CORRECT_HINT}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Classifier 로드
# ---------------------------------------------------------------------------

def load_classifier(cls_head_path: str, device: torch.device):
    cls_head_path = Path(cls_head_path)
    config_path = cls_head_path.parent / "classifier_config.json"

    with open(config_path) as f:
        cls_config = json.load(f)

    classifier = nn.Linear(cls_config["hidden_size"], cls_config["num_labels"])
    state = torch.load(cls_head_path, map_location="cpu", weights_only=True)
    classifier.load_state_dict(state)
    classifier = classifier.to(device=device, dtype=torch.bfloat16)
    classifier.eval()

    id2label = cls_config["id2label"]
    print(f"[init] Classifier 로드: {cls_head_path}")
    print(f"       ({cls_config['hidden_size']} → {cls_config['num_labels']} classes: {id2label})")
    return classifier, id2label


# ---------------------------------------------------------------------------
# 스텝 생성
# ---------------------------------------------------------------------------

def generate_step(
    gen_model,
    gen_tokenizer,
    problem: str,
    history: list,
    predicted_action: str,
    max_new_tokens: int = 1024,
) -> str:
    """predicted_action에 맞는 프롬프트를 구성해 LLM을 호출한다."""
    messages = format_messages_for_action(problem, history, predicted_action)
    prompt_text = apply_chat_template(gen_tokenizer, messages)

    gen_tokenizer.padding_side = "left"
    inputs = gen_tokenizer(
        [prompt_text], return_tensors="pt",
        padding=True, truncation=True, max_length=4096,
    ).to(gen_model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=gen_tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    generated = gen_tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_first_action(generated)


# ---------------------------------------------------------------------------
# MCE 리워드 계산
# ---------------------------------------------------------------------------

def rollout_to_end(
    gen_model,
    gen_tokenizer,
    problem: str,
    history: list,
    gold_answer: str,
    max_rollout_steps: int = 10,
    max_new_tokens: int = 512,
) -> bool:
    """현재 history에서 끝까지 rollout하여 정답 도달 여부를 반환한다.

    rollout은 항상 'solve' 프롬프트(일반 추론)를 사용한다.
    """
    h = list(history)
    for _ in range(max_rollout_steps):
        step_text = generate_step(gen_model, gen_tokenizer, problem, h, "solve", max_new_tokens)
        action, content = parse_step(step_text)
        if action is None:
            action = "solve"
            content = step_text.strip()
            step_text = f"<solve>{content}</solve>"
        if action != "end":
            boxed = parse_boxed(content or "")
            if boxed is not None:
                action = "end"
                content = boxed
                step_text = f"<end>{boxed}</end>"
        if action == "end":
            return check_answer_correct(content or "", gold_answer)
        h.append(step_text)
    return False


def compute_mce_reward(
    gen_model,
    gen_tokenizer,
    problem: str,
    history: list,
    gold_answer: str,
    n_rollouts: int = 4,
    max_rollout_steps: int = 10,
    max_new_tokens: int = 512,
) -> float:
    """MCE 리워드: rollout 중 정답에 도달한 비율.

    classification.py의 get_label 규칙 대응:
      MCE_reward == 0.0  →  이 스텝이 나쁨 → 다음 gt = correct
      MCE_reward  > 0.0  →  이 스텝이 좋음 → 다음 gt = solve
    """
    n_correct = sum(
        rollout_to_end(
            gen_model, gen_tokenizer, problem, history, gold_answer,
            max_rollout_steps, max_new_tokens,
        )
        for _ in range(n_rollouts)
    )
    return n_correct / n_rollouts


# ---------------------------------------------------------------------------
# Classifier 예측 (hidden state 추출)
# ---------------------------------------------------------------------------

def predict_action(
    gen_model,
    gen_tokenizer,
    classifier,
    id2label: dict,
    problem: str,
    history: list,
    correct_threshold: float = 0.3,
):
    """현재 history 기반으로 다음 액션 예측. (action_str, probs_dict) 반환.

    correct_threshold: P(correct) >= threshold이면 correct 예측.
    낮출수록 correct recall↑, precision↓.
    """
    messages = format_cls_messages(problem, history)
    prompt_text = apply_chat_template(gen_tokenizer, messages)

    gen_tokenizer.padding_side = "left"
    inputs = gen_tokenizer(
        [prompt_text], return_tensors="pt",
        padding=True, truncation=True, max_length=4096,
    ).to(gen_model.device)

    last_hidden_store = {}

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        last_hidden_store["h"] = h[:, -1, :].clone()

    hook = gen_model.model.layers[-1].register_forward_hook(_hook)
    with torch.no_grad():
        gen_model(**inputs)
    hook.remove()

    hidden = last_hidden_store.pop("h")  # (1, H)
    cls_device = next(classifier.parameters()).device
    cls_dtype = next(classifier.parameters()).dtype

    with torch.no_grad():
        logits = classifier(hidden.to(dtype=cls_dtype, device=cls_device))

    probs = logits.softmax(dim=1)[0].tolist()
    correct_prob = probs[1]  # P(correct)
    predicted_action = "correct" if correct_prob >= correct_threshold else "solve"

    del hidden, logits
    return predicted_action, {"solve": probs[0], "correct": correct_prob}


# ---------------------------------------------------------------------------
# 단일 문제 평가
# ---------------------------------------------------------------------------

def evaluate_problem(
    gen_model,
    gen_tokenizer,
    classifier,
    id2label: dict,
    problem: str,
    gold_answer: str,
    problem_id: str,
    max_steps: int = 15,
    max_new_tokens: int = 1024,
    n_rollouts: int = 4,
    max_rollout_steps: int = 10,
    rollout_max_new_tokens: int = 512,
    correct_threshold: float = 0.3,
) -> dict:
    history = []
    steps_data = []
    prev_mce_reward = None  # 직전 스텝의 MCE 리워드 (첫 스텝은 None)
    final_answer = None

    gt_actions = []    # 문제별 GT 액션 리스트 (end 제외)
    pred_actions = []  # 문제별 예측 액션 리스트 (end 제외)

    for step_idx in range(max_steps):
        # ── 1. Classifier 예측 ────────────────────────────────────────────
        cls_prediction, cls_probs = predict_action(
            gen_model, gen_tokenizer, classifier, id2label, problem, history,
            correct_threshold=correct_threshold,
        )

        # ── 2. Ground truth 결정 (직전 MCE 리워드 기반) ───────────────────
        # classification.py get_label 규칙:
        #   MCE_reward == 0.0  →  correct
        #   MCE_reward  > 0.0  →  solve
        if prev_mce_reward is None:
            gt_action = "solve"   # 첫 스텝: 직전 MCE 없음 → solve
        else:
            gt_action = "correct" if prev_mce_reward == 0.0 else "solve"

        # ── 3. 스텝 생성 (action에 따른 프롬프트 사용) ────────────────────
        step_text = generate_step(
            gen_model, gen_tokenizer,
            problem, history, cls_prediction,
            max_new_tokens=max_new_tokens,
        )
        action, content = parse_step(step_text)

        if action is None:
            action = cls_prediction
            content = step_text.strip()
            step_text = f"<{action}>{content}</{action}>"

        # boxed 정답이 포함된 경우 end로 처리 (cls_prediction 무관)
        if action != "end":
            boxed = parse_boxed(content or "")
            if boxed is not None:
                action = "end"
                content = boxed
                step_text = f"<end>{boxed}</end>"

        # history에 현재 스텝 추가
        history.append(step_text)

        # ── 4. MCE 리워드 계산 (end가 아닌 스텝만) ───────────────────────
        mce_reward = None
        if action != "end":
            mce_reward = compute_mce_reward(
                gen_model, gen_tokenizer,
                problem, history, gold_answer,
                n_rollouts=n_rollouts,
                max_rollout_steps=max_rollout_steps,
                max_new_tokens=rollout_max_new_tokens,
            )

        # ── 5. 레코드 저장 ────────────────────────────────────────────────
        cls_correct = None
        if action != "end":
            cls_correct = (cls_prediction == gt_action)
            gt_actions.append(gt_action)
            pred_actions.append(cls_prediction)

        steps_data.append({
            "step_idx": step_idx,
            "action": action,
            "content": content,
            "text": step_text,
            "history_len": len(history) - 1,
            "mce_reward": mce_reward,
            "cls_prediction": cls_prediction,
            "cls_probs": cls_probs,
            "gt_action": gt_action,
            "cls_correct": cls_correct,
        })

        if action == "end":
            # ── 6. 강제 correct 검증 ─────────────────────────────────────
            # 정답이 나왔더라도 correct 프롬프트로 한 번 더 검증한다.
            # 검증 스텝에 boxed 정답이 있으면 그 값이 최종 정답.
            # 없으면 검증 스텝을 history에 추가하고 루프 계속.
            verify_text = generate_step(
                gen_model, gen_tokenizer,
                problem, history, "correct",
                max_new_tokens=max_new_tokens,
            )
            verify_action, verify_content = parse_step(verify_text)
            if verify_action is None:
                verify_action = "correct"
                verify_content = verify_text.strip()
                verify_text = f"<correct>{verify_content}</correct>"

            # verify 스텝에서도 boxed 확인
            verify_boxed = parse_boxed(verify_content or "")
            if verify_action == "end" or verify_boxed is not None:
                final_answer = verify_content if verify_action == "end" else verify_boxed
                verify_ended = True
            else:
                verify_ended = False

            history.append(verify_text)
            gt_actions.append("verify")
            pred_actions.append(f"verify/{'end' if verify_ended else 'continue'}")
            steps_data.append({
                "step_idx": f"{step_idx}v",
                "action": "end" if verify_ended else verify_action,
                "content": verify_content,
                "text": verify_text,
                "history_len": len(history) - 1,
                "mce_reward": None,
                "cls_prediction": "forced_correct",
                "cls_probs": None,
                "gt_action": "verify",
                "cls_correct": None,
                "forced_verify": True,
            })

            if verify_ended:
                break

            # 검증 스텝이 정답을 내지 못한 경우 → 루프 계속
            prev_mce_reward = None  # verify 스텝은 MCE 없음
            torch.cuda.empty_cache()
            continue

        prev_mce_reward = mce_reward
        torch.cuda.empty_cache()

    is_correct = check_answer_correct(final_answer or "", gold_answer)

    return {
        "problem_id": problem_id,
        "problem": problem,
        "gold_answer": gold_answer,
        "final_answer": final_answer,
        "correct": is_correct,
        "num_steps": len(steps_data),
        "terminated": final_answer is not None,
        "gt_actions": gt_actions,
        "pred_actions": pred_actions,
        "steps": steps_data,
    }


# ---------------------------------------------------------------------------
# 데이터셋 로드
# ---------------------------------------------------------------------------

def load_deepmath(dataset_path: str, max_problems: int = 16000):
    """DeepMath 데이터셋을 로드한다.

    로컬 parquet 파일이 있으면 사용하고, 없으면 HuggingFace의
    zwhe99/DeepMath-103K에서 max_problems개를 로드한다.
    """
    if os.path.exists(dataset_path):
        ds = load_dataset("parquet", data_files=dataset_path, split="train")
    else:
        print("[init] HuggingFace에서 zwhe99/DeepMath-103K 로드...", flush=True)
        ds = load_dataset("zwhe99/DeepMath-103K", split="train")

    if max_problems and len(ds) > max_problems:
        ds = ds.select(range(max_problems))

    def normalize(example):
        if "problem" not in example:
            prompt = example.get("prompt", example.get("question", ""))
            if isinstance(prompt, list):
                user_msgs = [m["content"] for m in prompt if m.get("role") == "user"]
                example["problem"] = user_msgs[0] if user_msgs else ""
            else:
                example["problem"] = prompt
        if isinstance(example.get("problem"), list):
            msgs = example["problem"]
            user_msgs = [m["content"] for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
            example["problem"] = user_msgs[0] if user_msgs else str(msgs)
        if "answer" not in example:
            solution = example.get("solution", example.get("final_answer", ""))
            example["answer"] = parse_boxed(solution) or solution
        return example

    return ds.map(normalize)


# ---------------------------------------------------------------------------
# Per-class 메트릭 계산
# ---------------------------------------------------------------------------

def compute_metrics(cls_by_gt: dict, cls_by_pred: dict) -> dict:
    metrics = {}
    labels = ["solve", "correct"]
    total_all = sum(v["total"] for v in cls_by_gt.values())
    total_correct_all = sum(v["correct"] for v in cls_by_gt.values())

    overall_acc = round(total_correct_all / total_all, 4) if total_all > 0 else 0.0
    metrics["overall_accuracy"] = overall_acc

    for label in labels:
        tp = cls_by_gt[label]["correct"]
        fn = cls_by_gt[label]["total"] - tp
        fp = cls_by_pred[label]["total"] - tp

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        accuracy = round(tp / cls_by_gt[label]["total"], 4) if cls_by_gt[label]["total"] > 0 else 0.0

        metrics[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "support": cls_by_gt[label]["total"],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return metrics


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classifier Head 성능 평가 (MCE 리워드, DeepMath-16K)")
    parser.add_argument("--gen_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cls_head_path", type=str, default=CLS_HEAD_PATH)
    parser.add_argument("--dataset", type=str,
                        default="/mnt/seoyoon/projects/SC-sy/datasets/math_wrong.parquet",
                        help="로컬 parquet 경로")
    parser.add_argument("--output_dir", type=str, default="data/cls_eval_results")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="평가할 최대 문제 수 (None이면 전체)")
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--n_rollouts", type=int, default=4,
                        help="MCE 리워드 계산을 위한 rollout 횟수")
    parser.add_argument("--max_rollout_steps", type=int, default=10,
                        help="각 rollout에서 최대 생성 스텝 수")
    parser.add_argument("--rollout_max_new_tokens", type=int, default=512,
                        help="rollout 스텝 생성 시 최대 토큰 수")
    parser.add_argument("--correct_threshold", type=float, default=0.3,
                        help="P(correct) >= threshold이면 correct 예측 (낮을수록 recall↑ precision↓)")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default=f"cuda:{GPU_ID}")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.torch_dtype]
    device = torch.device(args.device)

    # ── 생성 모델 로드 ──────────────────────────────────────────────────────
    print(f"[init] 생성 모델 로드: {args.gen_model} → {device}", flush=True)
    gen_tokenizer = AutoTokenizer.from_pretrained(args.gen_model)
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, torch_dtype=dtype, device_map={"": device}
    )
    gen_model.eval()
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    # ── Classifier 로드 ─────────────────────────────────────────────────────
    classifier, id2label = load_classifier(args.cls_head_path, device=device)

    # ── 데이터셋 로드 ──────────────────────────────────────────────────────
    print(f"[init] 데이터셋 로드: {args.dataset}", flush=True)
    dataset = load_deepmath(args.dataset, max_problems=args.max_problems)
    all_examples = list(dataset)
    print(f"[init] 평가 문제 수: {len(all_examples)}", flush=True)
    print(f"[init] MCE rollout: {args.n_rollouts}회 / 최대 {args.max_rollout_steps} 스텝", flush=True)

    # ── 출력 파일 ─────────────────────────────────────────────────────────
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    results_file = os.path.join(args.output_dir, f"results{tag}_{run_time}.jsonl")
    summary_file = os.path.join(args.output_dir, f"summary{tag}_{run_time}.json")
    log_file     = os.path.join(args.output_dir, f"log{tag}_{run_time}.txt")
    print(f"[init] 결과 파일: {results_file}", flush=True)
    print(f"[init] 로그 파일: {log_file}", flush=True)
    print(f"[init] 시작 문제 번호: {args.start_idx}", flush=True)

    # ── 집계 통계 초기화 ───────────────────────────────────────────────────
    n_correct_answer = 0
    n_terminated = 0
    cls_by_gt = {
        "solve":   {"total": 0, "correct": 0},
        "correct": {"total": 0, "correct": 0},
    }
    cls_by_pred = {
        "solve":   {"total": 0},
        "correct": {"total": 0},
    }

    # ── 평가 루프 ──────────────────────────────────────────────────────────
    pending = [(i, ex) for i, ex in enumerate(all_examples) if i >= args.start_idx]
    n_done = 0

    with open(results_file, "w") as out_f, open(log_file, "w") as log_f:
        for i, example in tqdm(pending, desc="Classifier 평가"):
            problem_id = str(i)
            try:
                result = evaluate_problem(
                    gen_model, gen_tokenizer,
                    classifier, id2label,
                    example["problem"],
                    example["answer"],
                    problem_id,
                    max_steps=args.max_steps,
                    max_new_tokens=args.max_new_tokens,
                    n_rollouts=args.n_rollouts,
                    max_rollout_steps=args.max_rollout_steps,
                    rollout_max_new_tokens=args.rollout_max_new_tokens,
                    correct_threshold=args.correct_threshold,
                )
            except Exception as e:
                import traceback
                print(f"\n[error] 문제 {problem_id}: {e}", flush=True)
                traceback.print_exc()
                continue

            # 통계 업데이트
            n_done += 1
            if result["correct"]:
                n_correct_answer += 1
            if result["terminated"]:
                n_terminated += 1
            for step in result["steps"]:
                if step["cls_correct"] is not None:
                    gt = step["gt_action"]
                    pred = step["cls_prediction"]
                    if gt in cls_by_gt:
                        cls_by_gt[gt]["total"] += 1
                        if step["cls_correct"]:
                            cls_by_gt[gt]["correct"] += 1
                    if pred in cls_by_pred:
                        cls_by_pred[pred]["total"] += 1

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            # ── 문제별 출력: gt/pred 액션 리스트 + 누적 통계 ──────────────
            ans_mark = "O" if result["correct"] else "X"
            metrics_now = compute_metrics(cls_by_gt, cls_by_pred)
            log_line = (
                f"\n[#{i:>4}] gt_actions   = {result['gt_actions']}\n"
                f"        pred_actions = {result['pred_actions']}\n"
                f"        answer={ans_mark}"
                f"  | 누적 cls_acc={metrics_now['overall_accuracy']:.3f}"
                f"  ans_acc={n_correct_answer / n_done:.3f}"
            )
            tqdm.write(log_line)
            log_f.write(log_line + "\n")
            log_f.flush()

    # ── 최종 요약 ──────────────────────────────────────────────────────────
    metrics = compute_metrics(cls_by_gt, cls_by_pred)
    ans_acc = n_correct_answer / n_done if n_done > 0 else 0.0

    summary = {
        "gen_model": args.gen_model,
        "cls_head": args.cls_head_path,
        "n_rollouts": args.n_rollouts,
        "max_rollout_steps": args.max_rollout_steps,
        "n_total": n_done,
        "n_correct_answer": n_correct_answer,
        "answer_accuracy": round(ans_acc, 4),
        "n_terminated": n_terminated,
        "termination_rate": round(n_terminated / n_done, 4) if n_done > 0 else 0.0,
        "classifier_metrics": metrics,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_lines = [f"\n[done] Classifier 전체 정확도: {metrics['overall_accuracy']:.4f}"]
    for label in ["solve", "correct"]:
        m = metrics[label]
        summary_lines.append(
            f"       [{label:>7}]  support={m['support']:4d}  "
            f"acc={m['accuracy']:.4f}  "
            f"prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  "
            f"f1={m['f1']:.4f}"
        )
    summary_lines.append(f"[done] 최종 답안 정확도: {ans_acc:.4f} ({n_correct_answer}/{n_done})")
    summary_lines.append(f"[done] 결과 저장: {results_file}")
    summary_lines.append(f"[done] 요약 저장: {summary_file}")
    summary_lines.append(f"[done] 로그 저장: {log_file}")
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(log_file, "a") as log_f:
        log_f.write(summary_text + "\n")


if __name__ == "__main__":
    main()
