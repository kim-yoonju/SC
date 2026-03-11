"""
REINFORCE 기반 학습 스크립트 (multi-GPU DDP via accelerate)

generate_data.py 가 생성한 JSONL 파일을 읽어
  loss = -mean( reward_i * log p(step_i | prefix_i) )
로 모델을 학습한다.

실행: torchrun 또는 accelerate launch 를 통해 호출할 것.
      run_train.sh 참고.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import apply_chat_template, format_messages


# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------

class RolloutStepDataset(Dataset):
    """
    생성된 JSONL 파일에서 (prompt, response, reward) 쌍을 만든다.
    각 스텝이 하나의 학습 예제가 된다.
    """

    def __init__(self, data_files: list, tokenizer, max_length: int = 2048,
                 reward_threshold: float = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for file_path in data_files:
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        problem_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._process_problem(problem_data)

        if reward_threshold is not None:
            before = len(self.examples)
            self.examples = [
                ex for ex in self.examples
                if abs(ex["reward"]) > reward_threshold
            ]
            print(f"[train] 리워드 필터링 ({reward_threshold}): {before} → {len(self.examples)}개")

        print(f"[train] 총 학습 예제: {len(self.examples)}개")

    def _process_problem(self, problem_data: dict):
        problem = problem_data["problem"]
        for step in problem_data["steps"]:
            reward = step.get("final_reward", 0.0)
            history = step.get("history_before", [])
            step_text = step["text"]

            messages = format_messages(problem, history)
            prompt = apply_chat_template(self.tokenizer, messages)

            self.examples.append({
                "prompt": prompt,
                "response": step_text,
                "reward": float(reward),
                "action": step.get("action", ""),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def make_collate_fn(tokenizer, max_length: int):
    def collate_fn(batch):
        prompts = [ex["prompt"] for ex in batch]
        responses = [ex["response"] for ex in batch]
        rewards = torch.tensor([ex["reward"] for ex in batch], dtype=torch.float32)

        full_texts = [p + r for p, r in zip(prompts, responses)]
        encodings = tokenizer(
            full_texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        prompt_encodings = tokenizer(
            prompts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1)

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "prompt_lengths": prompt_lengths,
            "rewards": rewards,
        }

    return collate_fn


# ---------------------------------------------------------------------------
# 손실 계산
# ---------------------------------------------------------------------------

def compute_reinforce_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    rewards: torch.Tensor,
    normalize_rewards: bool = True,
) -> torch.Tensor:
    """
    REINFORCE loss: -E[ reward * mean log p(response | prompt) ]
    response 토큰에만 손실을 적용한다.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    attn_mask_shifted = attention_mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    B, T_minus1 = shift_labels.shape
    response_mask = torch.zeros(B, T_minus1, device=input_ids.device)
    for i, plen in enumerate(prompt_lengths):
        response_mask[i, plen - 1:] = 1.0
    response_mask = response_mask * attn_mask_shifted

    if normalize_rewards and rewards.std() > 1e-8:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    response_log_prob = (token_log_probs * response_mask).sum(dim=1) / (
        response_mask.sum(dim=1).clamp(min=1)
    )

    loss = -(rewards * response_log_prob).mean()
    return loss


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="REINFORCE 학습 (multi-GPU)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/rollouts")
    parser.add_argument("--output_dir", type=str, default="models/prm_trained")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="GPU당 배치 크기")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--normalize_rewards", action="store_true", default=True)
    parser.add_argument("--reward_threshold", type=float, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Accelerator 초기화 (DDP 자동 처리)
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[train] GPU 수: {accelerator.num_processes}")
        print(f"[train] 모델 로드: {args.model_name}")

    # 모델 로드 (device_map 없이 - DDP에서는 accelerate가 배치)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 데이터셋
    data_files = sorted(Path(args.data_dir).glob("*.jsonl"))
    if not data_files:
        raise FileNotFoundError(f"JSONL 파일 없음: {args.data_dir}")
    if accelerator.is_main_process:
        print(f"[train] 데이터 파일: {len(data_files)}개")

    dataset = RolloutStepDataset(
        [str(p) for p in data_files],
        tokenizer,
        max_length=args.max_length,
        reward_threshold=args.reward_threshold,
    )

    collate_fn = make_collate_fn(tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # 옵티마이저 & 스케줄러
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_update_steps = (
        len(dataloader) * args.num_epochs // args.grad_accum_steps
    )
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    # accelerate가 DDP / device 배치 / gradient accumulation 모두 처리
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    model.train()
    global_step = 0
    log_history = []

    if accelerator.is_main_process:
        print(f"[train] 학습 시작: {args.num_epochs} 에폭 / {total_update_steps} 업데이트 스텝")
        print(f"[train] effective batch = {args.batch_size} × {args.grad_accum_steps} × {accelerator.num_processes}")

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(pbar):
            # accelerate의 gradient_accumulation_steps 컨텍스트 사용
            with accelerator.accumulate(model):
                loss = compute_reinforce_loss(
                    model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["prompt_lengths"],
                    batch["rewards"],
                    normalize_rewards=args.normalize_rewards,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

            # sync_gradients: grad_accum_steps 마다 True
            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process:
                    log_entry = {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    log_history.append(log_entry)
                    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{log_entry['lr']:.2e}")

                    if global_step % args.save_steps == 0:
                        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.wait_for_everyone()
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(ckpt_path)
                        tokenizer.save_pretrained(ckpt_path)
                        print(f"\n[train] 체크포인트 저장: {ckpt_path}")

        avg_loss = epoch_loss / len(dataloader)
        if accelerator.is_main_process:
            print(f"[train] Epoch {epoch + 1} 완료 — 평균 손실: {avg_loss:.4f}")

        # 에폭별 저장
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            epoch_path = os.path.join(args.output_dir, f"epoch-{epoch + 1}")
            accelerator.unwrap_model(model).save_pretrained(epoch_path)
            tokenizer.save_pretrained(epoch_path)

    # 최종 저장
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "train_log.json"), "w") as f:
            json.dump(log_history, f, indent=2)
        print(f"[train] 완료. 모델 저장: {args.output_dir}")


if __name__ == "__main__":
    main()
