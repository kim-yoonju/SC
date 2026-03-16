"""
MATH500 단일 문제 inference (디버깅용)
문제 index를 지정해서 한 문제만 추론한다.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    apply_chat_template,
    check_answer_correct,
    extract_first_action,
    format_messages,
    parse_boxed,
    parse_step,
)

# ---- 설정 ----
MODEL_NAME   = "checkpoints/offline_reinforce/epoch-1"
CLS_HEAD_PATH = "checkpoints/action_cls/best_model/classifier_head.pt"
DATASET_PATH = "datasets/math500.parquet"
PROBLEM_IDX  = 158
GPUS         = "3,4"   # 사용할 GPU
MAX_STEPS    = 10
MAX_NEW_TOKENS = 512


def load_classifier(cls_head_path, device):
    cls_head_path = Path(cls_head_path)
    config_path = cls_head_path.parent / "classifier_config.json"
    with open(config_path) as f:
        cls_config = json.load(f)
    classifier = nn.Linear(cls_config["hidden_size"], cls_config["num_labels"])
    classifier.load_state_dict(torch.load(cls_head_path, map_location="cpu"))
    classifier = classifier.to(device=device, dtype=torch.bfloat16)
    classifier.eval()
    return classifier, cls_config["id2label"]


def solve_one(model, tokenizer, classifier, id2label, problem):
    tokenizer.padding_side = "left"
    history = []
    steps = []

    for step_idx in range(MAX_STEPS):
        messages = format_messages(problem, history)
        prompt_text = apply_chat_template(tokenizer, messages)

        prompt_enc = tokenizer(
            [prompt_text], return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(model.device)

        # 마지막 레이어 hidden state hook
        last_hidden_store = {}
        def _hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            last_hidden_store["h"] = h[:, -1, :].clone()
        hook_handle = model.model.layers[-1].register_forward_hook(_hook)
        with torch.no_grad():
            model(**prompt_enc)
        hook_handle.remove()

        boundary_hidden = last_hidden_store.pop("h")
        cls_device = next(classifier.parameters()).device
        cls_dtype  = next(classifier.parameters()).dtype
        logits = classifier(boundary_hidden.to(dtype=cls_dtype, device=cls_device))
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = logits.argmax(dim=1).item()
        predicted_action = id2label[str(pred_id)]
        prob_str = "  ".join(f"{id2label[str(i)]}: {probs[i].item():.4f}" for i in range(len(id2label)))
        print(f"[classifier] {prob_str}  →  {predicted_action}")
        del boundary_hidden, logits, probs
        torch.cuda.empty_cache()

        # generate
        prefixed_text = prompt_text + f"<{predicted_action}>"
        prefixed_enc = tokenizer(
            [prefixed_text], return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(model.device)
        input_len = prefixed_enc["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **prefixed_enc,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        del prefixed_enc
        torch.cuda.empty_cache()

        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        step_raw   = f"<{predicted_action}>" + generated
        step_text  = extract_first_action(step_raw)
        action, content = parse_step(step_text)

        if action is None:
            action   = predicted_action
            content  = generated.strip()
            step_text = f"<{action}>{content}</{action}>"

        if action != "end":
            boxed = parse_boxed(content or "")
            if boxed is not None:
                action    = "end"
                content   = boxed
                step_text = f"<end>{boxed}</end>"

        steps.append({
            "step_idx": step_idx,
            "predicted_action": predicted_action,
            "action": action,
            "content": content,
            "text": step_text,
        })

        print(f"\n{'='*60}")
        print(f"Step {step_idx+1}  |  예측 액션: {predicted_action}  →  실제 액션: {action}")
        print(f"{'='*60}")
        print(step_text)

        if action == "end":
            return content, steps

        history.append(step_text)

    return None, steps


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUS

    problem = "Remmy wants to divide $10$ by $\\frac{2}{3}$, but he cannot remember how to do that.  By what number should he multiply $10$ to get the answer?"
    gold    = ""

    print(f"=== 문제 ===")
    print(problem)
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    classifier, id2label = load_classifier(CLS_HEAD_PATH, device=next(model.parameters()).device)
    print(f"Classifier: {id2label}\n")

    pred_answer, steps = solve_one(model, tokenizer, classifier, id2label, problem)

    print(f"\n=== 결과 ===")
    print(f"예측 답: {pred_answer}")
    print(f"정답:    {gold}")
    correct = check_answer_correct(pred_answer or "", gold)
    print(f"정오:    {'✓' if correct else '✗'}")
    print(f"스텝 수: {len(steps)}")


if __name__ == "__main__":
    main()
