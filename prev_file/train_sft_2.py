"""
SFT 학습 스크립트 - Qwen2.5-7B-Instruct  (DeepSpeed ZeRO-2)

데이터 형식 (generate_trajectory.py 출력):
  {
    "problem_id": str, "problem": str,
    "gold_answer": str, "is_right": bool, "traj_type": str,
    "steps": [
      {
        "step_idx": int, "step": str,       # "G_01", "G+_02", "P*_03"
        "inference": str,                    # 모델이 생성한 전체 텍스트
        "source": str,                       # "gen" | "rethink" | "patcher"
        "is_fail": bool,
        "state": str,
        "next_gold_action": str,             # "<|solve|>" | "<|rethink|>" | "<|end|>"
        "does": str,                         # 스텝 한 줄 요약
        "PRM_critique_summary": [...],       # [{rubric, does}]
      }, ...
    ]
  }

실행 예시:
  CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 source/train_sft.py \\
      --data_path output/SFT/xxx/sft_data/sft_preprocessed.jsonl
  CUDA_VISIBLE_DEVICES=3 python source/train_sft.py --debug
"""

import argparse
import datetime
import os
import sys
from functools import partial
from pathlib import Path

import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM
import deepspeed
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / "utils"))
from utils_sft import setup_tokenizer, collate_fn, CONF, PreprocessedDataset, debug

_sft = CONF.get("sft", {})
MODEL_ID      = CONF["checkpoint"].get("sft_checkpoint") or CONF["checkpoint"]["base"]
CACHE_DIR     = CONF["checkpoint"].get("cache_dir")
OUTPUT_DIR    = str(_ROOT / CONF["output_path"]["sft_checkpoints"])
LEARNING_RATE = _sft.get("learning_rate", 2e-5)
NUM_EPOCHS    = _sft.get("num_epochs", 3)
BATCH_PER_GPU = _sft.get("batch_per_gpu", 4)
GRAD_ACCUM    = _sft.get("grad_accum", 16)
MAX_LENGTH    = _sft.get("max_length", 3072)
WARMUP_RATIO  = _sft.get("warmup_ratio", 0.05)
WEIGHT_DECAY  = _sft.get("weight_decay", 0.01)
MAX_GRAD_NORM = _sft.get("max_grad_norm", 1.0)
SAVE_STEPS    = _sft.get("save_steps", 100)
WANDB_PROJECT = _sft.get("wandb_project", "sc-sft")


# ─────────────────────────────────────────────────────────────────────────────
# Chunked LM loss: LM head projection을 청크 단위로 계산해서
# [B×T×V] logit 텐서(~28 GB)를 ayet materialization하지 않음.
#
# 기존: hidden → lm_head → logits[B,T,V] → .float() → CE  ← OOM
# 개선: hidden → 청크별 F.linear + CE (1.5 GB/청크) → 합산
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(step, warmup_steps, total_steps, base_lr):
    if total_steps == 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def chunked_lm_loss(model_module, input_ids, attention_mask, labels,
                    chunk=16, ignore_index=-100):
    """[B×T×V] logit 텐서 materialization 없이 청크 단위로 LM loss 계산 (OOM 방지)."""
    hidden = model_module.model(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state
    if not hidden.isfinite().all():
        hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0)

    shift_hidden = hidden[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_hidden.device)
    del hidden

    n_valid = (shift_labels != ignore_index).sum().clamp(min=1).float()
    V = model_module.lm_head.weight.shape[0]
    T = shift_hidden.shape[1]

    lm_head_weight_f32 = model_module.lm_head.weight.float()
    parts = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        c_logits = F.linear(shift_hidden[:, s:e].float(), lm_head_weight_f32)
        c_loss = F.cross_entropy(c_logits.view(-1, V), shift_labels[:, s:e].reshape(-1),
                                 ignore_index=ignore_index, reduction="none")
        parts.append(c_loss.sum())
        del c_logits

    return torch.stack(parts).sum() / n_valid


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프  (DeepSpeed ZeRO-2)
# ─────────────────────────────────────────────────────────────────────────────

def _load_data_sample(data_path: str) -> dict:
    try:
        import json
        with open(data_path) as f:
            return json.loads(f.readline())
    except Exception:
        return {}


def save_run_meta(run_dir: str, args, start_time: datetime.datetime,
                  n_samples: int, total_steps: int, warmup_steps: int,
                  world_size: int):
    import json
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    meta = {
        "run_dir":      run_dir,
        "train_start":  start_time.isoformat(),
        "model":        args.model_path,
        "data_path":    args.data_path,
        "gpus":         gpu_ids,
        "world_size":   world_size,
        "hyperparams": {
            "lr":            args.lr,
            "num_epochs":    args.num_epochs,
            "batch_per_gpu": args.batch_per_gpu,
            "grad_accum":    args.grad_accum,
            "max_length":    args.max_length,
            "warmup_ratio":  args.warmup_ratio,
            "weight_decay":  args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
        },
        "flags": {
            "skip_error": args.skip_error,
        },
        "dataset": {
            "n_samples":    n_samples,
            "total_steps":  total_steps,
            "warmup_steps": warmup_steps,
        },
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_meta(ckpt: str, args, start_time: datetime.datetime, data_sample: dict):
    import json
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    meta = {
        "model": args.model_path,
        "data_path": args.data_path,
        "gpus": gpu_ids,
        "train_start": start_time.isoformat(),
        "save_time": datetime.datetime.now().isoformat(),
        "data_sample": data_sample,
    }
    with open(os.path.join(ckpt, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def train(args):
    # ── 분산 초기화 ──────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    primary_device = torch.device(f"cuda:{local_rank}")
    is_main = (global_rank == 0)

    train_start = datetime.datetime.now()
    data_sample = _load_data_sample(args.data_path) if is_main else {}

    # ── WandB ────────────────────────────────────────────────────────────────
    use_wandb = args.wandb and is_main
    if use_wandb:
        try:
            import wandb
            wandb.init(project=WANDB_PROJECT, config=vars(args))
        except ImportError:
            use_wandb = False

    # ── 데이터 ───────────────────────────────────────────────────────────────
    tokenizer = setup_tokenizer(args.model_path, CACHE_DIR)

    dataset   = PreprocessedDataset(args.data_path, tokenizer,
                                    max_length=args.max_length,
                                    skip_error=args.skip_error)
    sampler   = DistributedSampler(dataset, num_replicas=world_size,
                                   rank=global_rank, shuffle=True, drop_last=True)
    loader    = DataLoader(dataset,
                           batch_size=args.batch_per_gpu,
                           sampler=sampler,
                           collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
                           num_workers=2, pin_memory=True)

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    load_path = args.resume_checkpoint if args.resume_checkpoint else args.model_path
    model = AutoModelForCausalLM.from_pretrained(
        load_path, cache_dir=CACHE_DIR,
        dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if is_main and args.resume_checkpoint:
        print(f"[재개] 체크포인트 로드: {load_path}  (완료 에폭: {args.resume_epoch})")
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    # ── step 수 계산 ─────────────────────────────────────────────────────────
    # DeepSpeed는 gradient_accumulation_steps를 config에서 가져가므로
    # effective_grad_accum = grad_accum // world_size
    effective_grad_accum = max(1, args.grad_accum // world_size)
    total_steps  = (len(loader) // effective_grad_accum) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    if is_main:
        eff_batch = world_size * args.batch_per_gpu * effective_grad_accum
        print(f"샘플: {len(dataset)}  |  effective batch: {eff_batch}  |  steps: {total_steps}")

    # ── DeepSpeed ZeRO-2 설정 ────────────────────────────────────────────────
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_per_gpu,
        "gradient_accumulation_steps":    effective_grad_accum,
        "gradient_clipping":              args.max_grad_norm,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr":           args.lr,
                "betas":        [0.9, 0.999],
                "eps":          1e-8,
                "weight_decay": args.weight_decay,
            },
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions":          True,
            "reduce_scatter":                True,
            "overlap_comm":                  True,
            "contiguous_gradients":          True,
        },
        "steps_per_print":       9999999,
        "wall_clock_breakdown":  False,
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    # ── 저장 경로 ─────────────────────────────────────────────────────────────
    if args.run_dir:
        run_dir = args.run_dir
    else:
        ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, ts)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        print(f"저장 경로: {run_dir}")
    dist.barrier()

    # ── 학습 루프 ────────────────────────────────────────────────────────────
    steps_per_epoch = len(loader) // effective_grad_accum
    global_step     = steps_per_epoch * args.resume_epoch

    if is_main and args.resume_epoch == 0:
        save_run_meta(run_dir, args, train_start,
                      n_samples=len(dataset),
                      total_steps=total_steps,
                      warmup_steps=warmup_steps,
                      world_size=world_size)

    for epoch in range(args.resume_epoch, args.num_epochs):
        sampler.set_epoch(epoch)
        model_engine.train()

        pbar       = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}") if is_main else loader
        accum_loss = epoch_loss = 0.0
        n_updates  = 0

        for batch in pbar:
            input_ids      = batch["input_ids"].to(primary_device)
            attention_mask = batch["attention_mask"].to(primary_device)
            labels         = batch["labels"].to(primary_device)

            loss = chunked_lm_loss(model_engine.module, input_ids, attention_mask, labels)

            loss = loss / effective_grad_accum
            model_engine.backward(loss)
            accum_loss += loss.item()
            model_engine.step()

            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1
                cur_lr = cosine_lr(global_step, warmup_steps, total_steps, args.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
                n_updates   += 1
                epoch_loss  += accum_loss

                # 파라미터 NaN 진단 (step 이후)
                if is_main:
                    n_nan = sum(1 for p in model_engine.module.parameters()
                                if p.data.isnan().any() or p.data.isinf().any())
                    if n_nan > 0:
                        print(f"\n[PARAM NaN] step {global_step} 이후 {n_nan}개 파라미터 NaN/Inf", flush=True)

                if is_main:
                    pbar.set_postfix({"loss": f"{accum_loss:.4f}",
                                      "lr":   f"{cur_lr:.2e}",
                                      "step": global_step})
                    if use_wandb:
                        import wandb
                        wandb.log({"train/loss": accum_loss, "train/lr": cur_lr,
                                   "train/epoch": epoch + 1}, step=global_step)

                if is_main and global_step % args.save_steps == 0:
                    ckpt = os.path.join(run_dir, f"step_{global_step}")
                    model_engine.module.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    save_meta(ckpt, args, train_start, data_sample)
                    print(f"\n[저장] {ckpt}")

                accum_loss = 0.0

        # ── 에폭 종료: 체크포인트 저장 ───────────────────────────────────────
        if is_main:
            avg = epoch_loss / max(n_updates, 1)
            print(f"\n[에폭 {epoch+1}] avg_loss={avg:.4f}")
            ckpt = os.path.join(run_dir, f"epoch{epoch+1}")
            model_engine.module.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            save_meta(ckpt, args, train_start, data_sample)
            print(f"[저장] {ckpt}")
            if use_wandb:
                import wandb
                wandb.log({"epoch/avg_loss": avg, "epoch/epoch": epoch + 1},
                          step=global_step)
        dist.barrier()  # 저장 완료 전에 다음 에폭 진입 방지

    if use_wandb and is_main:
        import wandb; wandb.finish()
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",    default=MODEL_ID)
    p.add_argument("--use_base_model", action="store_true", default=True,
                   help="sft_checkpoint 무시하고 config의 base 모델 사용 (기본값)")
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
    p.add_argument("--skip_error",    action="store_true",
                   help="is_fail=True 스텝 학습 제외")
    p.add_argument("--wandb",         action="store_true", default=True)
    p.add_argument("--no-wandb",      dest="wandb", action="store_false",
                   help="WandB 비활성화")
    p.add_argument("--debug",         type=int, nargs="?", const=-1, default=None,
                   help="N번째 샘플 디버그 출력 후 종료 (인자 생략 시 is_fail별 자동 샘플링)")
    p.add_argument("--resume_checkpoint", default=None,
                   help="이어서 학습할 체크포인트 경로 (예: checkpoints/sft/20260505_130300/epoch2)")
    p.add_argument("--resume_epoch",  type=int, default=0,
                   help="완료된 에폭 수 (예: epoch2 체크포인트면 2)")
    p.add_argument("--run_dir",       default=None,
                   help="기존 run 디렉토리 재사용 (없으면 새 타임스탬프 디렉토리 생성)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.use_base_model:
        args.model_path = CONF["checkpoint"]["base"]
    if args.debug is not None:
        tokenizer = setup_tokenizer(args.model_path, CACHE_DIR)
        debug(args.data_path, tokenizer,
              n=None if args.debug == -1 else args.debug,
              model_path=args.model_path, cache_dir=CACHE_DIR)
        return
    train(args)


if __name__ == "__main__":
    main()