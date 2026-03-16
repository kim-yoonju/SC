"""
Offline REINFORCE 학습 + Classification Head (다음 액션 예측)
"""
import argparse, json, os, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, Adafactor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from utils import apply_chat_template, format_messages, parse_step

# label2id (classifier_config.json 기준)
ACTION_LABEL = {"solve": 0, "correct": 1}


class RolloutStepDataset(Dataset):
    def __init__(self, data_files: list, tokenizer, max_length=2048, reward_threshold=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for file_path in data_files:
            with open(file_path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    problem = d["problem"]
                    # worker 파일 포맷: 한 줄 = 한 step (step_text 키)
                    if "step_text" in d:
                        if d.get("is_teacher"):
                            continue
                        reward = float(d.get("final_reward", 0.0))
                        history = d.get("history_before", [])
                        step_text = d["step_text"]
                        messages = format_messages(problem, history)
                        prompt = apply_chat_template(tokenizer, messages)
                        action, _ = parse_step(step_text)
                        action_label = ACTION_LABEL.get(action, -1)
                        self.examples.append({
                            "prompt": prompt,
                            "response": step_text,
                            "reward": reward,
                            "action_label": action_label,
                        })
                    # 메인 jsonl 포맷: 한 줄 = 한 problem (steps 리스트)
                    else:
                        for step in d.get("steps", []):
                            if step.get("is_teacher"):
                                continue
                            reward = float(step.get("final_reward", 0.0))
                            history = step.get("history_before", [])
                            step_text = step["text"]
                            messages = format_messages(problem, history)
                            prompt = apply_chat_template(tokenizer, messages)
                            action, _ = parse_step(step_text)
                            action_label = ACTION_LABEL.get(action, -1)
                            self.examples.append({
                                "prompt": prompt,
                                "response": step_text,
                                "reward": reward,
                                "action_label": action_label,
                            })

        if reward_threshold is not None:
            before = len(self.examples)
            self.examples = [e for e in self.examples if abs(e["reward"]) > reward_threshold]
            print(f"[train] 리워드 필터링: {before} → {len(self.examples)}")

        print(f"[train] 총 학습 예제: {len(self.examples)}")

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


def make_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        prompts = [e["prompt"] for e in batch]
        responses = [e["response"] for e in batch]
        rewards = torch.tensor([e["reward"] for e in batch], dtype=torch.float32)
        action_labels = torch.tensor([e["action_label"] for e in batch], dtype=torch.long)
        full_texts = [p + r for p, r in zip(prompts, responses)]
        enc = tokenizer(full_texts, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        penc = tokenizer(prompts, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "prompt_lengths": penc["attention_mask"].sum(dim=1),
            "rewards": rewards,
            "action_labels": action_labels,
        }
    return collate_fn


def compute_losses(model, classifier, input_ids, attention_mask, prompt_lengths,
                   rewards, action_labels, normalize=True, cls_coef=0.1):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    # --- REINFORCE loss ---
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    attn = attention_mask[:, 1:]

    token_log_probs = -F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none"
    ).reshape(shift_labels.shape)

    B, T = shift_labels.shape
    resp_mask = torch.zeros(B, T, device=input_ids.device)
    for i, plen in enumerate(prompt_lengths):
        resp_mask[i, plen - 1:] = 1.0
    resp_mask = resp_mask * attn

    if normalize and rewards.std() > 1e-8:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    resp_lp = (token_log_probs * resp_mask).sum(1) / resp_mask.sum(1).clamp(min=1)
    reinforce_loss = -(rewards * resp_lp).mean()

    # --- Classification loss (다음 액션 예측) ---
    # 프롬프트 마지막 토큰 위치의 hidden state를 classifier에 통과
    last_hidden = outputs.hidden_states[-1]  # (B, T, H)
    prompt_last_pos = (prompt_lengths - 1).clamp(max=last_hidden.size(1) - 1)
    hidden_at_boundary = last_hidden[torch.arange(B, device=input_ids.device), prompt_last_pos]  # (B, H)
    cls_logits = classifier(hidden_at_boundary.to(next(classifier.parameters()).dtype))  # (B, num_labels)
    cls_loss = F.cross_entropy(cls_logits, action_labels, ignore_index=-1)

    total_loss = reinforce_loss + cls_coef * cls_loss
    return total_loss, reinforce_loss, cls_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_files", type=str, nargs="+", required=True,
                        help="학습할 jsonl 파일 경로들")
    parser.add_argument("--output_dir", type=str, default="models/reinforce_trained")
    parser.add_argument("--cls_head_path", type=str,
                        default="/mnt/yoonju/SC/checkpoints/action_cls/best_model/classifier_head.pt",
                        help="pretrained classifier head (.pt) 경로")
    parser.add_argument("--cls_coef", type=float, default=0.1,
                        help="classification loss 가중치")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--reward_threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[train] GPU 수: {accelerator.num_processes}")
        print(f"[train] 모델 로드: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Classifier head 로드
    cls_config_path = Path(args.cls_head_path).parent / "classifier_config.json"
    with open(cls_config_path) as f:
        cls_config = json.load(f)
    classifier = nn.Linear(cls_config["hidden_size"], cls_config["num_labels"])
    cls_state = torch.load(args.cls_head_path, map_location="cpu")
    classifier.load_state_dict(cls_state)
    classifier = classifier.to(torch.bfloat16)
    if accelerator.is_main_process:
        print(f"[train] Classifier head 로드: {args.cls_head_path} "
              f"({cls_config['hidden_size']} → {cls_config['num_labels']} classes: {cls_config['id2label']})")

    dataset = RolloutStepDataset(args.data_files, tokenizer,
                                  max_length=args.max_length,
                                  reward_threshold=args.reward_threshold)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=make_collate_fn(tokenizer, args.max_length), drop_last=True)

    params = list(model.parameters()) + list(classifier.parameters())
    optimizer = Adafactor(
        params,
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.01,
    )
    total_steps = len(dataloader) * args.num_epochs // args.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    model, classifier, optimizer, dataloader, scheduler = accelerator.prepare(
        model, classifier, optimizer, dataloader, scheduler
    )
    model.train()
    classifier.train()

    global_step = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}",
                    disable=not accelerator.is_main_process)

        for batch in pbar:
            with accelerator.accumulate(model):
                total_loss, rl_loss, cls_loss = compute_losses(
                    model, classifier,
                    batch["input_ids"], batch["attention_mask"],
                    batch["prompt_lengths"], batch["rewards"],
                    batch["action_labels"],
                    cls_coef=args.cls_coef,
                )
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    all_params = (list(accelerator.unwrap_model(model).parameters()) +
                                  list(accelerator.unwrap_model(classifier).parameters()))
                    accelerator.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += total_loss.item()
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    pbar.set_postfix(
                        loss=f"{total_loss.item():.4f}",
                        rl=f"{rl_loss.item():.4f}",
                        cls=f"{cls_loss.item():.4f}",
                    )

        avg_loss = epoch_loss / len(dataloader)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print(f"\n[train] Epoch {epoch+1} 완료 — 평균 손실: {avg_loss:.4f}")
            epoch_path = os.path.join(args.output_dir, f"epoch-{epoch+1}")
            accelerator.unwrap_model(model).save_pretrained(epoch_path)
            tokenizer.save_pretrained(epoch_path)
            torch.save(
                accelerator.unwrap_model(classifier).state_dict(),
                os.path.join(epoch_path, "classifier_head.pt"),
            )
            print(f"[train] 저장: {epoch_path}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(
            accelerator.unwrap_model(classifier).state_dict(),
            os.path.join(args.output_dir, "classifier_head.pt"),
        )
        print(f"[train] 완료: {args.output_dir}")


if __name__ == "__main__":
    main()
