"""
SFT 학습 스크립트 — prm_critique_summary prediction

인자:
  --data_path  trajectory JSONL 경로
  --gpus       사용할 GPU 번호 comma-separated (e.g. "0,1,2,3")

입력/타겟 형식:
  input  : system + [Problem] + [Previous steps] (is_fail=False 스텝의 does) + [Current step] (inference)
  target : prm_critique_summary dict 포맷팅 문자열

실행 (run_sft_classification.sh 내에서):
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 source/train_sft.py \
      --data_path output/sft_trajectory/xxx.jsonl
"""

import argparse
import datetime
import json
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM
import deepspeed
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / "utils"))

from utils_sft import setup_tokenizer, collate_fn, CONF

_sft = CONF.get("sft", {})
MODEL_ID      = CONF["checkpoint"]["base"]
CACHE_DIR     = CONF["checkpoint"].get("cache_dir")
OUTPUT_DIR    = str(_ROOT / CONF["output_path"]["sft_checkpoints"])
LEARNING_RATE = _sft.get("learning_rate", 5e-6)
NUM_EPOCHS    = _sft.get("num_epochs", 3)
BATCH_PER_GPU = _sft.get("batch_per_gpu", 4)
GRAD_ACCUM    = _sft.get("grad_accum", 16)
MAX_LENGTH    = _sft.get("max_length", 4096)
WARMUP_RATIO  = _sft.get("warmup_ratio", 0.1)
WEIGHT_DECAY  = _sft.get("weight_decay", 0.01)
MAX_GRAD_NORM = _sft.get("max_grad_norm", 0.5)
SAVE_STEPS    = _sft.get("save_steps", 32)
WANDB_PROJECT = _sft.get("wandb_project", "sc-sft")


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 전처리
# ─────────────────────────────────────────────────────────────────────────────

def _is_fail(step: dict) -> bool:
    v = step.get("is_fail")
    if v is not None:
        return bool(v)
    return bool(step.get("is_error", False))


def _format_summary(summary: dict, verdicts: dict) -> str:
    """verdict가 있는 rubric만 포함."""
    lines = []
    for rubric, text in summary.items():
        if not text or rubric not in verdicts:
            continue
        lines.append(f"{rubric}: {text} Verdict: {verdicts[rubric]}")
    return "\n".join(lines)


def _print_rubric_stats(rubric_stats: dict) -> None:
    if not rubric_stats:
        return
    col = max(len(r) for r in rubric_stats) + 2
    sep = "─" * (col + 46)
    print(f"\n  [ Rubric 분포 ]")
    print(f"  {'Rubric':<{col}}  {'total':>7}  {'correct':>8}  {'incorrect':>10}  {'incorr%':>8}")
    print(f"  {sep}")
    for rubric in sorted(rubric_stats):
        c  = rubric_stats[rubric]["correct"]
        ic = rubric_stats[rubric]["incorrect"]
        t  = c + ic
        pct = ic / t * 100 if t else 0.0
        print(f"  {rubric:<{col}}  {t:>7}  {c:>8}  {ic:>10}  {pct:>7.1f}%")
    totals = {"correct": sum(v["correct"] for v in rubric_stats.values()),
              "incorrect": sum(v["incorrect"] for v in rubric_stats.values())}
    t = totals["correct"] + totals["incorrect"]
    pct = totals["incorrect"] / t * 100 if t else 0.0
    print(f"  {sep}")
    print(f"  {'합계':<{col}}  {t:>7}  {totals['correct']:>8}  {totals['incorrect']:>10}  {pct:>7.1f}%\n")


def _get_rubric_verdicts(step: dict) -> dict:
    """스텝에서 {rubric: "correct"/"incorrect"} 반환. prm_deep_critique 우선, 없으면 prm_fast_critique."""
    def _v(raw): return "incorrect" if (raw or "").lower() in ("incorrect", "fail") else "correct"

    dc_list = step.get("prm_deep_critique") or []
    fc_dict = step.get("prm_fast_critique") or {}
    rubrics = [d.get("rubric", "") for d in dc_list] if dc_list else list(fc_dict.keys())

    result = {}
    for rubric in rubrics:
        if not rubric:
            continue
        dc_entry = next((d for d in dc_list if d.get("rubric") == rubric), {})
        if dc_entry.get("verdict") is not None:
            result[rubric] = _v(dc_entry["verdict"])
        else:
            result[rubric] = _v((fc_dict.get(rubric) or {}).get("verdict", "correct"))
    return result


def _build_user_message(problem: str, steps: list, k: int) -> str:
    history_does = [
        (s.get("does") or "").strip()
        for s in steps[:k]
        if not _is_fail(s) and (s.get("does") or "").strip()
    ]
    current_inference = (steps[k].get("inference") or "").strip()

    lines = [f"[Problem]\n{problem}"]
    if history_does:
        lines.append("\n[Previous steps]")
        lines.extend(history_does)
    lines.append(f"\n[Current step]\n{current_inference}")
    lines.append("\nEvaluate this step.")
    return "\n".join(lines)


class SummaryDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, system_prompt: str, max_length: int = 4096):
        self.samples = []
        raw = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"[SummaryDataset] {len(raw)}개 trajectory 로드, 토크나이징 중...")

        # rubric별 correct/incorrect 집계: {rubric: {"correct": int, "incorrect": int}}
        from collections import defaultdict
        rubric_stats: dict = defaultdict(lambda: {"correct": 0, "incorrect": 0})

        skipped = 0
        for item in tqdm(raw, desc="Tokenizing", disable=(rank != 0)):
            problem = item["problem"]
            steps   = item["steps"]
            for k, step in enumerate(steps):
                if step.get("source") == "patcher":
                    skipped += 1
                    continue

                summary = step.get("prm_critique_summary")
                if not summary or not isinstance(summary, dict):
                    skipped += 1
                    continue

                verdicts   = _get_rubric_verdicts(step)
                target_str = _format_summary(summary, verdicts)
                if not target_str.strip():
                    skipped += 1
                    continue

                user_str = _build_user_message(problem, steps, k)
                msgs     = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_str},
                ]
                full_msgs  = msgs + [{"role": "assistant", "content": target_str}]
                full_str   = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
                prefix_str = tokenizer.apply_chat_template(msgs,      tokenize=False, add_generation_prompt=True)

                full_ids   = tokenizer.encode(full_str,   add_special_tokens=False)
                prefix_len = len(tokenizer.encode(prefix_str, add_special_tokens=False))

                if len(full_ids) > max_length:
                    skipped += 1
                    continue

                input_ids = torch.tensor(full_ids, dtype=torch.long)
                labels    = torch.full_like(input_ids, -100)
                labels[prefix_len:] = input_ids[prefix_len:]
                self.samples.append((input_ids, labels))

                # rubric 통계 집계
                for rubric, verdict in _get_rubric_verdicts(step).items():
                    rubric_stats[rubric][verdict] += 1

        if rank == 0:
            print(f"[SummaryDataset] 학습 샘플: {len(self.samples)}  (제외: {skipped})")
            _print_rubric_stats(rubric_stats)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 학습 유틸
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(step, warmup_steps, total_steps, base_lr):
    if total_steps == 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def compute_lm_loss(model_engine, input_ids, attention_mask, labels):
    """모델 표준 forward를 사용한 LM loss. fp16 전체 일관성 유지."""
    outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device  = torch.device(f"cuda:{local_rank}")
    is_main = (global_rank == 0)

    train_start = datetime.datetime.now()

    use_wandb = args.wandb and is_main
    if use_wandb:
        try:
            import wandb
            wandb.init(project=WANDB_PROJECT, config=vars(args))
        except ImportError:
            use_wandb = False

    # system prompt
    sys.path.insert(0, str(_ROOT / "source"))
    from preprocess import get_classification_prompt
    system_prompt = get_classification_prompt()

    tokenizer = setup_tokenizer(args.model_path, CACHE_DIR)
    dataset   = SummaryDataset(args.data_path, tokenizer, system_prompt, max_length=args.max_length)
    sampler   = DistributedSampler(dataset, num_replicas=world_size,
                                   rank=global_rank, shuffle=True, drop_last=True)
    loader    = DataLoader(dataset,
                           batch_size=args.batch_per_gpu,
                           sampler=sampler,
                           collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
                           num_workers=2, pin_memory=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, cache_dir=CACHE_DIR,
        dtype=torch.float16, trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    effective_grad_accum = max(1, args.grad_accum // world_size)
    total_steps  = (len(loader) // effective_grad_accum) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    if is_main:
        eff_batch = world_size * args.batch_per_gpu * effective_grad_accum
        print(f"샘플: {len(dataset)}  |  effective_batch: {eff_batch}  |  total_steps: {total_steps}")

    zero_opt = {
        "stage": 2,
        "allgather_partitions": True,
        "reduce_scatter":       True,
        "overlap_comm":         True,
        "contiguous_gradients": True,
    }
    if args.cpu_offload:
        zero_opt["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_per_gpu,
        "gradient_accumulation_steps":    effective_grad_accum,
        "gradient_clipping":              args.max_grad_norm,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr":           args.lr,
                "betas":        [0.9, 0.999],
                "eps":          1e-8,
                "weight_decay": args.weight_decay,
            },
        },
        "zero_optimization":    zero_opt,
        "steps_per_print":      9999999,
        "wall_clock_breakdown": False,
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config)

    ts      = train_start.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, ts)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        print(f"저장 경로: {run_dir}")
    dist.barrier()

    global_step = 0

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        model_engine.train()
        pbar       = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}") if is_main else loader
        accum_loss = epoch_loss = 0.0
        n_updates  = 0

        micro_step = 0
        cur_lr = cosine_lr(0, warmup_steps, total_steps, args.lr)

        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            micro_step    += 1

            loss = compute_lm_loss(model_engine, input_ids, attention_mask, labels)
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                n_label_tokens = (labels != -100).sum().item()
                raise RuntimeError(
                    f"Loss is {loss_val} at epoch={epoch+1} global_step={global_step} "
                    f"micro_step={micro_step} batch_shape={tuple(input_ids.shape)} "
                    f"label_tokens={n_label_tokens} — 학습 중단."
                )
            loss = loss / effective_grad_accum
            model_engine.backward(loss)
            accum_loss += loss_val / effective_grad_accum

            is_boundary = model_engine.is_gradient_accumulation_boundary()
            if is_boundary:
                global_step += 1
                cur_lr = cosine_lr(global_step, warmup_steps, total_steps, args.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr

            model_engine.step()

            if is_boundary:
                n_updates  += 1
                epoch_loss += accum_loss

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
                    print(f"\n[저장] {ckpt}")

                accum_loss = 0.0

        if is_main:
            avg  = epoch_loss / max(n_updates, 1)
            ckpt = os.path.join(run_dir, f"epoch{epoch+1}")
            model_engine.module.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"\n[에폭 {epoch+1}] avg_loss={avg:.4f}  → {ckpt}")
            if use_wandb:
                import wandb
                wandb.log({"epoch/avg_loss": avg, "epoch/epoch": epoch + 1}, step=global_step)
        dist.barrier()

    if use_wandb and is_main:
        import wandb
        wandb.finish()
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# 디버그
# ─────────────────────────────────────────────────────────────────────────────

def debug(args):
    sys.path.insert(0, str(_ROOT / "source"))
    from preprocess import get_classification_prompt
    system_prompt = get_classification_prompt()
    tokenizer     = setup_tokenizer(args.model_path, CACHE_DIR)
    dataset       = SummaryDataset(args.data_path, tokenizer, system_prompt, max_length=args.max_length)

    if not dataset.samples:
        print("샘플이 없습니다.")
        return

    sep = "─" * 72
    input_ids, labels = dataset.samples[0]

    # input: -100 마스킹 전 전체 시퀀스에서 prefix 부분
    prefix_ids = input_ids[: (labels == -100).sum().item()]
    target_ids = input_ids[(labels == -100).sum().item():]

    input_text  = tokenizer.decode(prefix_ids,  skip_special_tokens=False)
    target_text = tokenizer.decode(target_ids,  skip_special_tokens=True)

    print(f"\n{'='*72}")
    print(f"[샘플 0  |  input tokens={len(prefix_ids)}  target tokens={len(target_ids)}]")
    print(f"\n[INPUT]\n{sep}")
    print(input_text)
    print(f"\n[TARGET]\n{sep}")
    print(target_text)
    print(f"{'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",     required=True,  help="trajectory JSONL 경로")
    p.add_argument("--gpus",          default=None,   help="사용할 GPU (정보용, CUDA_VISIBLE_DEVICES는 shell에서 설정)")
    p.add_argument("--model_path",    default=MODEL_ID)
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
    p.add_argument("--cpu_offload",   action="store_true", default=True)
    p.add_argument("--no-cpu-offload", dest="cpu_offload", action="store_false")
    p.add_argument("--wandb",         action="store_true", default=True)
    p.add_argument("--no-wandb",      dest="wandb", action="store_false")
    p.add_argument("--debug",         action="store_true",
                   help="샘플 1개 출력 후 종료 (분산 학습 불필요)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        debug(args)
    else:
        train(args)
