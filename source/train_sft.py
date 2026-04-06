"""
SFT 학습 스크립트 - Qwen2.5-7B-Instruct

학습 방식:
  문제 + 이전 스텝들 → 다음 스텝 텍스트 예측
  Loss는 다음 스텝 텍스트에만 계산

데이터 형식 (sft_data_v1.0.jsonl):
  {
    "problem_id":  str,
    "problem":     str,
    "gold_answer": str,
    "pred_answer": str,
    "is_right":    bool,
    "steps":       [{"step_idx": int, "type": str, "text": str, "next_gold_action": str}, ...]
  }

입력/출력 형식:
  - 이전 스텝 히스토리: "text <next_gold_action>" (text 뒤 공백 + action 토큰)
  - target: "text <next_gold_action>"

실행 예시 (GPU 4,5,6,7):
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 source/train_sft.py
  # 기존 체크포인트에서 이어서 학습:
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 source/train_sft.py \\
      --model_path /mnt/yoonju/SC/checkpoints/sft/20260403_125458/epoch5
  # 샘플 미리보기:
  CUDA_VISIBLE_DEVICES=4 python source/train_sft.py --preview
"""

import argparse
import datetime
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SYSTEM_SOLVE_SFT,
    SYSTEM_CORRECT,
    build_chat_prompt,
    load_raw_data,
    setup_tokenizer,
    collate_fn,
    _solve_user,
    _correct_user,
)


def load_config() -> dict:
    config_path = _ROOT / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


_CFG = load_config()

MODEL_ID       = _CFG["checkpoint"]["base"]
CACHE_DIR      = _CFG["checkpoint"].get("cache_dir", None)
DATA_PATH      = _CFG["data_path"]["sft_data"]
OUTPUT_DIR     = str(_ROOT / _CFG["output_path"]["sft_checkpoints"])

_sft = _CFG.get("sft", {})
LEARNING_RATE = _sft.get("learning_rate", 2e-5)
NUM_EPOCHS    = _sft.get("num_epochs", 3)
BATCH_SIZE    = _sft.get("batch_size", 1)
GRAD_ACCUM    = _sft.get("grad_accum", 32)
MAX_LENGTH    = _sft.get("max_length", 3072)
WARMUP_RATIO  = _sft.get("warmup_ratio", 0.05)
WEIGHT_DECAY  = _sft.get("weight_decay", 0.01)
MAX_GRAD_NORM = _sft.get("max_grad_norm", 1.0)
SAVE_STEPS    = _sft.get("save_steps", 100)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    각 문제의 매 스텝에 대해 하나의 학습 샘플을 생성한다.

    샘플 k:
      input  (loss 제외): system + problem + step_0 + ... + step_{k-1}
      target (loss 계산): step_k 텍스트
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.samples = []

        raw_data = load_raw_data(data_path)
        rank = int(os.environ.get("RANK", 0))

        if rank == 0:
            print(f"[Dataset] {len(raw_data)}개 문제 로드, 토크나이징 중...")

        skipped = 0
        for item in tqdm(raw_data, desc="Tokenizing", disable=(rank != 0)):
            skipped += self._process_item(item, tokenizer)

        if rank == 0:
            print(f"[Dataset] 총 학습 샘플 수: {len(self.samples)}  (max_length 초과로 제외: {skipped})")

    def _process_item(self, item: dict, tokenizer) -> int:
        """문제의 각 스텝에 대해 학습 샘플 생성. max_length 초과 수 반환."""
        problem = item["problem"]
        steps   = item["steps"]  # [{step_idx, text}, ...]
        skipped = 0

        for k in range(len(steps)):
            next_action = steps[k].get("next_gold_action", "<|solve|>")

            # # rethink 바로 전 스텝(틀린 스텝)은 학습 제외 — 모델이 오답 생성을 배우지 않도록
            # if next_action == "<|rethink|>":
            #     continue

            history     = [s["text"] + " " + s.get("next_gold_action", "<|solve|>") for s in steps[:k]]
            # target = 스텝 텍스트 + 공백 + 다음 액션 토큰
            target_text = steps[k]["text"] + " " + next_action

            step_type = steps[k].get("type", "solve")
            if step_type == "rethink":
                prefix_str = build_chat_prompt(
                    tokenizer, SYSTEM_CORRECT, _correct_user(problem, history)
                )
            else:
                prefix_str = build_chat_prompt(
                    tokenizer, SYSTEM_SOLVE_SFT, _solve_user(problem, history)
                )

            prefix_ids = tokenizer.encode(prefix_str,   add_special_tokens=False)
            target_ids = tokenizer.encode(target_text,  add_special_tokens=False)
            full_ids   = prefix_ids + target_ids

            if len(full_ids) > self.max_length:
                skipped += 1
                continue

            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels    = torch.full_like(input_ids, -100)
            labels[len(prefix_ids):] = input_ids[len(prefix_ids):]

            self.samples.append((input_ids, labels))

        return skipped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 프리뷰
# ─────────────────────────────────────────────────────────────────────────────

def _print_sample(tokenizer, item: dict, k: int, label: str):
    """item의 k번째 스텝에 대한 input/target 샘플을 출력한다."""
    problem = item["problem"]
    steps   = item["steps"]

    history     = [s["text"] + " " + s.get("next_gold_action", "<|solve|>") for s in steps[:k]]
    next_action = steps[k].get("next_gold_action", "<|solve|>")
    target_text = steps[k]["text"] + " " + next_action

    step_type = steps[k].get("type", "solve")
    if step_type == "rethink":
        prefix_str = build_chat_prompt(tokenizer, SYSTEM_CORRECT, _correct_user(problem, history))
    else:
        prefix_str = build_chat_prompt(tokenizer, SYSTEM_SOLVE_SFT, _solve_user(problem, history))

    prefix_ids = tokenizer.encode(prefix_str,  add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    sep = "─" * 70
    print(f"\n{'=' * 70}")
    print(f"[샘플: {label}]")
    print(f"문제:      {problem[:100]}...")
    print(f"gold:      {item.get('gold_answer', '')}")
    print(f"전체 스텝: {len(steps)}  |  현재 스텝: {k}  |  type: {steps[k].get('type', 'solve')}")
    print(f"\n[INPUT — loss 제외, {len(prefix_ids)} 토큰]")
    print(sep)
    print(prefix_str)
    print(f"\n[TARGET — loss 계산, {len(target_ids)} 토큰]")
    print(sep)
    print(target_text)
    print(f"\n토큰 수 — prefix: {len(prefix_ids)}  target: {len(target_ids)}  total: {len(prefix_ids)+len(target_ids)}")


def preview_sample(data_path: str, tokenizer):
    """샘플 2개 출력: (1) rethink 포함 문제의 solve→rethink 직전 스텝, (2) 같은 문제의 rethink 스텝."""
    raw_data = load_raw_data(data_path)

    # rethink가 있는 첫 번째 문제 선택
    rethink_item = next(
        (x for x in raw_data if any(s.get("type") == "rethink" or s.get("next_gold_action") == "<|rethink|>"
                                    for s in x.get("steps", []))),
        raw_data[0]
    )
    steps = rethink_item["steps"]

    # rethink 직전 스텝 인덱스 (next_gold_action == <|rethink|>)
    pre_rethink_k = next(
        (i for i, s in enumerate(steps) if s.get("next_gold_action") == "<|rethink|>"),
        len(steps) - 2
    )
    rethink_k = pre_rethink_k + 1  # rethink 스텝

    _print_sample(tokenizer, rethink_item, pre_rethink_k, "solve 스텝 (다음이 rethink)")
    _print_sample(tokenizer, rethink_item, rethink_k,     "rethink 스텝")


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (global_rank == 0)

    # ── wandb ────────────────────────────────────────────────────────────────
    use_wandb = args.wandb and is_main
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="sc-sft",
                name=args.run_name,
                config={
                    "model_path":   args.model_path,
                    "data_path":    args.data_path,
                    "lr":           args.lr,
                    "num_epochs":   args.num_epochs,
                    "batch_size":   args.batch_size,
                    "grad_accum":   args.grad_accum,
                    "max_length":   args.max_length,
                    "warmup_ratio": args.warmup_ratio,
                    "weight_decay": args.weight_decay,
                },
            )
        except ImportError:
            print("[wandb] wandb 미설치, 로깅 비활성화")
            use_wandb = False

    # ── 토크나이저 & 모델 ────────────────────────────────────────────────────
    if is_main:
        print(f"모델 로드: {args.model_path}")
    tokenizer = setup_tokenizer(args.model_path, CACHE_DIR)

    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank,
                                 shuffle=True, drop_last=True)
    loader  = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                         collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
                         num_workers=2, pin_memory=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── 옵티마이저 & 스케줄러 ─────────────────────────────────────────────────
    # GPU가 늘어도 effective batch(= N_GPUs × batch_size × grad_accum)를 일정하게 유지.
    # grad_accum을 world_size에 반비례하게 줄여서 optimizer step 수를 보존한다.
    effective_grad_accum = max(1, args.grad_accum // world_size)

    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = (len(loader) // effective_grad_accum) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=warmup_steps,
                                                   num_training_steps=total_steps)

    if is_main:
        eff_batch = world_size * args.batch_size * effective_grad_accum
        print(f"데이터 샘플 수: {len(dataset)}")
        print(f"GPU 수: {world_size}  |  grad_accum: {effective_grad_accum}  |  effective batch: {eff_batch}")
        print(f"총 옵티마이저 스텝: {total_steps}  (warmup: {warmup_steps})")

    # ── 체크포인트 저장 경로 ─────────────────────────────────────────────────
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, ts)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        print(f"저장 경로: {run_dir}")
    dist.barrier()

    global_step = 0

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        pbar       = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}") if is_main else loader
        accum_loss = 0.0
        epoch_loss = 0.0
        n_updates  = 0

        for step_in_epoch, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss / effective_grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (step_in_epoch + 1) % effective_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                n_updates   += 1
                epoch_loss  += accum_loss

                cur_lr = scheduler.get_last_lr()[0]

                if is_main:
                    pbar.set_postfix({"loss": f"{accum_loss:.4f}", "lr": f"{cur_lr:.2e}",
                                      "step": global_step})
                    if use_wandb:
                        import wandb
                        wandb.log({"train/loss": accum_loss, "train/lr": cur_lr,
                                   "train/epoch": epoch + 1}, step=global_step)

                if is_main and global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(run_dir, f"step_{global_step}")
                    model.module.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"\n[저장] {ckpt_dir}")

                accum_loss = 0.0

        if is_main:
            avg_loss = epoch_loss / max(n_updates, 1)
            print(f"\n[에폭 {epoch+1}] avg_loss={avg_loss:.4f}")
            ckpt_dir = os.path.join(run_dir, f"epoch{epoch+1}")
            model.module.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[저장] {ckpt_dir}")
            if use_wandb:
                import wandb
                wandb.log({"epoch/avg_loss": avg_loss, "epoch/epoch": epoch + 1},
                          step=global_step)

    if use_wandb and is_main:
        import wandb
        wandb.finish()

    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",    default=MODEL_ID)
    p.add_argument("--data_path",     default=DATA_PATH)
    p.add_argument("--output_dir",    default=OUTPUT_DIR)
    p.add_argument("--lr",            type=float, default=LEARNING_RATE)
    p.add_argument("--num_epochs",    type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--grad_accum",    type=int,   default=GRAD_ACCUM)
    p.add_argument("--max_length",    type=int,   default=MAX_LENGTH)
    p.add_argument("--warmup_ratio",  type=float, default=WARMUP_RATIO)
    p.add_argument("--weight_decay",  type=float, default=WEIGHT_DECAY)
    p.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    p.add_argument("--save_steps",    type=int,   default=SAVE_STEPS)
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--run_name",      default=None)
    p.add_argument("--preview",       action="store_true",
                   help="샘플 하나 출력 후 종료")
    return p.parse_args()


def main():
    args = parse_args()
    if args.preview:
        tokenizer = setup_tokenizer(args.model_path, CACHE_DIR)
        preview_sample(args.data_path, tokenizer)
        return
    train(args)


if __name__ == "__main__":
    main()
