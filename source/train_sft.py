"""
SFT 학습 스크립트 - Qwen2.5-7B-Instruct

데이터 형식 (generate_trajectory.py 출력):
  {
    "traj_id": str, "problem_id": str, "problem": str,
    "gold_answer": str, "is_right": bool, "traj_type": str,
    "steps": [
      {
        "step_idx": int, "step": str,       # "G_01", "G+_02", "P*_03"
        "inference": str,                    # 모델이 생성한 전체 텍스트
        "source": str,                       # "gen" | "rethink" | "patcher"
        "is_error": bool,
        "state": str,
        "next_gold_action": str,             # "<|solve|>" | "<|rethink|>" | "<|end|>"
        "does": str,                         # 스텝 한 줄 요약
        "PRM_critique_summary": [...],       # [{rubric, does}]
      }, ...
    ]
  }

실행 예시:
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=2 source/train_sft.py \\
      --data_path output/sft_trajectory/20260426_xxx/traj_mix.jsonl
  CUDA_VISIBLE_DEVICES=4 python source/train_sft.py --preview
"""

import argparse
import datetime
import json
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
import bitsandbytes as bnb
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_sft import setup_tokenizer, collate_fn, CONF, build_input, build_target

_sft = CONF.get("sft", {})
MODEL_ID      = CONF["checkpoint"].get("sft_checkpoint") or CONF["checkpoint"]["base"]
CACHE_DIR     = CONF["checkpoint"].get("cache_dir")
OUTPUT_DIR    = str(_ROOT / CONF["output_path"]["sft_checkpoints"])
LEARNING_RATE = _sft.get("learning_rate", 2e-5)
NUM_EPOCHS    = _sft.get("num_epochs", 3)
BATCH_PER_GPU = _sft.get("batch_per_gpu", 4)
GRAD_ACCUM    = _sft.get("grad_accum", 64)
MAX_LENGTH    = _sft.get("max_length", 3072)
WARMUP_RATIO  = _sft.get("warmup_ratio", 0.05)
WEIGHT_DECAY  = _sft.get("weight_decay", 0.01)
MAX_GRAD_NORM = _sft.get("max_grad_norm", 1.0)
SAVE_STEPS    = _sft.get("save_steps", 100)
GPU_PER_MODEL = _sft.get("gpu_per_model", 1)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────────────────────────────────────

class TrajDataset(Dataset):
    """
    generate_trajectory.py 출력 JSONL로부터 SFT 학습 샘플 생성.

    각 스텝마다 하나의 학습 샘플:
      input  (loss 제외): build_input(problem, steps, k)
      target (loss 계산): build_target(steps[k])

    기본적으로 모든 스텝을 학습하되, skip_error=True면 is_error=True 스텝 제외.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = MAX_LENGTH,
        skip_error: bool = False,
    ):
        self.max_length = max_length
        self.samples    = []

        raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"[TrajDataset] {len(raw)}개 trajectory 로드, 토크나이징 중...")

        skipped = 0
        for item in tqdm(raw, desc="Tokenizing", disable=(rank != 0)):
            skipped += self._process_item(item, tokenizer, skip_error)

        if rank == 0:
            print(f"[TrajDataset] 학습 샘플: {len(self.samples)}  (제외: {skipped})")

    def _process_item(self, item: dict, tokenizer, skip_error: bool) -> int:
        problem = item["problem"]
        steps   = item["steps"]
        skipped = 0

        for k, step in enumerate(steps):
            if skip_error and step.get("is_error"):
                continue

            prefix_str = build_input(problem, steps, k, tokenizer)
            target_str = build_target(step)

            prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
            target_ids = tokenizer.encode(target_str, add_special_tokens=False)
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

def preview(data_path: str, tokenizer, n: int = 2):
    raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    sep = "─" * 72

    count = 0
    for item in raw:
        steps   = item["steps"]
        problem = item["problem"]
        for k, step in enumerate(steps):
            if count >= n:
                return
            prefix_str = build_input(problem, steps, k, tokenizer)
            target_str = build_target(step)
            p_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
            t_ids = tokenizer.encode(target_str, add_special_tokens=False)

            print(f"\n{'='*72}")
            print(f"[traj_id={item['traj_id']}  step={step['step']}  "
                  f"source={step['source']}  is_error={step['is_error']}]")
            print(f"problem: {problem[:100]}...")
            print(f"\n[INPUT — {len(p_ids)} tok]")
            print(sep)
            print(prefix_str[-800:])   # 마지막 800자만 출력
            print(f"\n[TARGET — {len(t_ids)} tok]")
            print(sep)
            print(target_str[:400])
            print(f"\n토큰: prefix={len(p_ids)}  target={len(t_ids)}  total={len(p_ids)+len(t_ids)}")
            count += 1


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])

    gpu_per_model  = args.gpu_per_model
    n_gpus_visible = torch.cuda.device_count()
    my_gpu_ids     = list(range(local_rank * gpu_per_model,
                                local_rank * gpu_per_model + gpu_per_model))
    primary_device = torch.device(f"cuda:{my_gpu_ids[0]}")
    torch.cuda.set_device(my_gpu_ids[0])
    is_main = (global_rank == 0)

    use_wandb = args.wandb and is_main
    if use_wandb:
        try:
            import wandb
            wandb.init(project="sc-sft", name=args.run_name, config=vars(args))
        except ImportError:
            use_wandb = False

    tokenizer = setup_tokenizer(args.model_path, CACHE_DIR)
    dataset   = TrajDataset(args.data_path, tokenizer,
                            max_length=args.max_length,
                            skip_error=args.skip_error)
    sampler   = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank,
                                   shuffle=True, drop_last=True)
    loader    = DataLoader(dataset,
                           batch_size=args.batch_per_gpu * gpu_per_model,
                           sampler=sampler,
                           collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
                           num_workers=2, pin_memory=True)

    if gpu_per_model > 1:
        max_memory = {i: "85GiB" if i in my_gpu_ids else "0GiB"
                      for i in range(n_gpus_visible)}
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map="auto", max_memory=max_memory,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(primary_device)

    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    if gpu_per_model > 1:
        model = DDP(model, device_ids=None, output_device=None, find_unused_parameters=False)
    else:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    effective_grad_accum = max(1, args.grad_accum // world_size)
    optimizer   = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(loader) // effective_grad_accum) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler   = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        eff_batch = world_size * args.batch_per_gpu * gpu_per_model * effective_grad_accum
        print(f"샘플: {len(dataset)}  |  effective batch: {eff_batch}  |  steps: {total_steps}")

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
        accum_loss = epoch_loss = 0.0
        n_updates  = 0

        for step_i, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(primary_device)
            attention_mask = batch["attention_mask"].to(primary_device)
            labels         = batch["labels"].to(primary_device)

            loss = model(input_ids=input_ids, attention_mask=attention_mask,
                         labels=labels).loss.to(primary_device) / effective_grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (step_i + 1) % effective_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1; n_updates += 1; epoch_loss += accum_loss
                cur_lr = scheduler.get_last_lr()[0]

                if is_main:
                    pbar.set_postfix({"loss": f"{accum_loss:.4f}", "lr": f"{cur_lr:.2e}",
                                      "step": global_step})
                    if use_wandb:
                        import wandb
                        wandb.log({"train/loss": accum_loss, "train/lr": cur_lr,
                                   "train/epoch": epoch + 1}, step=global_step)

                if is_main and global_step % args.save_steps == 0:
                    ckpt = os.path.join(run_dir, f"step_{global_step}")
                    model.module.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
                    print(f"\n[저장] {ckpt}")

                accum_loss = 0.0

        # 에폭 끝에 grad_accum을 채우지 못한 남은 배치 flush
        remaining = len(loader) % effective_grad_accum
        if remaining > 0 and accum_loss > 0:
            # accum_loss는 remaining 스텝에 걸쳐 loss/effective_grad_accum 합산 → 실제 평균으로 보정
            accum_loss = accum_loss * effective_grad_accum / remaining
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            global_step += 1; n_updates += 1; epoch_loss += accum_loss
            if is_main and use_wandb:
                import wandb
                wandb.log({"train/loss": accum_loss, "train/lr": scheduler.get_last_lr()[0],
                           "train/epoch": epoch + 1}, step=global_step)
            accum_loss = 0.0

        if is_main:
            avg = epoch_loss / max(n_updates, 1)
            print(f"\n[에폭 {epoch+1}] avg_loss={avg:.4f}")
            ckpt = os.path.join(run_dir, f"epoch{epoch+1}")
            model.module.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
            print(f"[저장] {ckpt}")
            if use_wandb:
                import wandb
                wandb.log({"epoch/avg_loss": avg, "epoch/epoch": epoch + 1}, step=global_step)

    if use_wandb and is_main:
        import wandb; wandb.finish()
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",    default=MODEL_ID)
    p.add_argument("--data_path",     default=str(_ROOT / CONF["data_path"].get("sft_data", "")))
    p.add_argument("--output_dir",    default=OUTPUT_DIR)
    p.add_argument("--lr",            type=float, default=LEARNING_RATE)
    p.add_argument("--num_epochs",    type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch_per_gpu", type=int,   default=BATCH_PER_GPU)
    p.add_argument("--grad_accum",    type=int,   default=GRAD_ACCUM)
    p.add_argument("--max_length",    type=int,   default=MAX_LENGTH)
    p.add_argument("--warmup_ratio",  type=float, default=WARMUP_RATIO)
    p.add_argument("--weight_decay",  type=float, default=WEIGHT_DECAY)
    p.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    p.add_argument("--save_steps",    type=int,   default=SAVE_STEPS)
    p.add_argument("--gpu_per_model", type=int,   default=GPU_PER_MODEL)
    p.add_argument("--skip_error",    action="store_true",
                   help="is_error=True 스텝 학습 제외")
    p.add_argument("--preview_n",     type=int,   default=2)
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--run_name",      default=None)
    p.add_argument("--preview",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.preview:
        tok = setup_tokenizer(args.model_path, CACHE_DIR)
        preview(args.data_path, tok, n=args.preview_n)
        return
    train(args)


if __name__ == "__main__":
    main()
