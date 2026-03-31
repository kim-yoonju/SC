"""
SFT 학습 스크립트 - Qwen2.5-7B-Instruct

학습 방식:
  문제 + 이전 스텝들 → 다음 액션 + 스텝 텍스트 예측
  Loss는 예측 대상(액션 태그 + 텍스트)에만 계산

실행 예시 (GPU 2,3,4,5):
  CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 source/train_sft.py
  CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 source/train_sft.py --preview
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
from transformers import AutoConfig, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SYSTEM_SOLVE,
    SYSTEM_CORRECT,
    build_chat_prompt,
    extract_boxed,
    load_raw_data,
    setup_tokenizer,
    extract_step_content,
    build_target_text,
    pick_system,
    collate_fn,
    _solve_user,
    _correct_user,
)


def load_config() -> dict:
    config_path = _ROOT / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


_CFG = load_config()

MODEL_ID   = _CFG["checkpoint"]["base"]
CACHE_DIR  = _CFG["checkpoint"].get("cache_dir", None)
DATA_PATH  = str(_ROOT / _CFG["data_path"]["sft_data"])
OUTPUT_DIR = str(_ROOT / _CFG["output_path"]["sft_checkpoints"])

# 특수 토큰: 텍스트 뒤에 append되어 다음 행동을 나타낸다
# <|end|>은 강화학습용 커스텀 종료 토큰 (Qwen vocab에 없어 별도 등록 필요)
SPECIAL_TOKENS = [
    _CFG["model"]["token_solve"],
    _CFG["model"]["token_correct"],
    _CFG["model"]["token_end"],
]

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
# 데이터 전처리
# ─────────────────────────────────────────────────────────────────────────────

def build_plain_prefix(system: str, problem: str, prev_steps: list) -> str:
    """chat template 없이 plain text prefix 생성.

    형식:
        {system}

        Problem: {problem}

        {step0_text}<|action0|>

        {step1_text}<|action1|>

    """
    parts = [system, "", f"Problem: {problem}"]
    for a, t in prev_steps:
        parts.append("")
        parts.append(build_target_text(a, t))
    parts.append("")   # 마지막 개행 → 모델 생성 시작점
    return "\n".join(parts)



# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    각 문제의 매 스텝에 대해 하나의 학습 샘플을 생성한다.

    샘플 k:
      input  (loss 제외): system + problem + step_0 + ... + step_{k-1}
      target (loss 계산): <action_k>text_k</action_k>

    토크나이징은 __getitem__이 아닌 __init__에서 일괄 처리해
    학습 중 병목을 줄인다.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.samples = []   # list of (input_ids, labels) tensors

        raw_data = load_raw_data(data_path)
        rank = int(os.environ.get("RANK", 0))

        if rank == 0:
            print(f"[Dataset] {len(raw_data)}개 문제 로드, 토크나이징 중...")

        for item in tqdm(raw_data, desc="Tokenizing", disable=(rank != 0)):
            self._process_item(item, tokenizer)

        if rank == 0:
            print(f"[Dataset] 총 학습 샘플 수: {len(self.samples)}")

    def _process_item(self, item: dict, tokenizer):
        problem = item["problem"]
        steps = item["steps"]

        # 각 스텝을 (action, clean_text) 쌍으로 파싱
        step_pairs = [extract_step_content(s) for s in steps]

        last_idx = len(step_pairs) - 1

        # 스텝 k마다 하나의 샘플 생성
        for k, (action, text) in enumerate(step_pairs):
            system = pick_system(action)
            # history: 이전 스텝들의 텍스트 (액션 토큰 미포함) — prototype 추론 시와 동일 포맷
            history = [t for _, t in step_pairs[:k]]
            if action == "correct":
                reason = history[-1] if history else ""
                prefix_str = build_chat_prompt(tokenizer, system, _correct_user(problem, history, reason))
            else:
                prefix_str = build_chat_prompt(tokenizer, system, _solve_user(problem, history))

            if action == "end":
                # 최종 단계: \boxed{} 형식
                boxed = extract_boxed(text)
                if boxed is None:
                    continue
                target_str = f"Therefore, the final answer is \\boxed{{{boxed}}}.<|end|>"
            elif k == last_idx:
                # 마지막 스텝이지만 action이 "end"가 아닌 경우
                boxed = extract_boxed(text)
                if boxed is not None:
                    target_str = f"{text}\nTherefore, the final answer is \\boxed{{{boxed}}}.<|end|>"
                else:
                    target_str = build_target_text(action, text)
            else:
                # 중간 스텝: 텍스트 + 액션 토큰
                target_str = build_target_text(action, text)

            # 토크나이즈: prefix는 loss 제외(-100), target만 loss 계산
            prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
            target_ids = tokenizer.encode(target_str, add_special_tokens=False)
            full_ids   = prefix_ids + target_ids

            if len(full_ids) > self.max_length:
                continue

            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels    = torch.full_like(input_ids, -100)
            labels[len(prefix_ids):] = input_ids[len(prefix_ids):]

            self.samples.append((input_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 프리뷰: 실제 모델에 들어가는 input/output 쌍 출력
# ─────────────────────────────────────────────────────────────────────────────

def preview_sample(data_path: str, tokenizer):
    """
    첫 번째 문제의 첫 번째 스텝에 대해
    모델에 실제로 들어가는 input 문자열과 target 문자열을 출력한다.
    """
    raw_data = load_raw_data(data_path)
    item = raw_data[1]
    problem = item["problem"]
    steps = item["steps"]
    step_pairs = [extract_step_content(s) for s in steps]

    sep = "─" * 70

    last_idx = len(step_pairs) - 1

    # 마지막 스텝만 출력
    k      = last_idx
    action, text = step_pairs[k]
    system       = pick_system(action)
    history      = [t for _, t in step_pairs[:k]]
    full_prefix  = build_chat_prompt(tokenizer, system, _solve_user(problem, history))

    if action == "end":
        boxed = extract_boxed(text)
        target_str = f"Therefore, the final answer is \\boxed{{{boxed}}}.<|end|>" if boxed else build_target_text(action, text)
    else:
        boxed = extract_boxed(text)
        if boxed is not None:
            target_str = f"{text}\nTherefore, the final answer is \\boxed{{{boxed}}}.<|end|>"
        else:
            target_str = build_target_text(action, text)

    prefix_ids = tokenizer.encode(full_prefix, add_special_tokens=False)
    full_ids   = tokenizer.encode(full_prefix + target_str, add_special_tokens=False)
    target_ids = full_ids[len(prefix_ids):]

    print(f"\n{'=' * 70}")
    print(f"[스텝 {k}] INPUT (loss 제외, {len(prefix_ids)} 토큰)")
    print(sep)
    print(full_prefix)
    print(f"\n[스텝 {k}] TARGET (loss 계산, {len(target_ids)} 토큰)")
    print(sep)
    print(target_str)
    print(f"\n[스텝 {k}] 토큰 확인")
    print(f"  prefix tokens : {len(prefix_ids)}")
    print(f"  target tokens : {len(target_ids)}")
    print(f"  total tokens  : {len(prefix_ids) + len(target_ids)}")
    print(f"  action        : {action}")


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # DDP 초기화
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    is_main = (global_rank == 0)

    # ── 토크나이저 ───────────────────────────────────────────────────────────
    if is_main:
        print(f"[{global_rank}] 토크나이저 로드: {MODEL_ID}")
        print(f"[{global_rank}] 특수 토큰 추가: {SPECIAL_TOKENS}")
    tokenizer = setup_tokenizer(MODEL_ID, CACHE_DIR, special_tokens=SPECIAL_TOKENS)

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        drop_last=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=True,
    )

    # ── 모델 ─────────────────────────────────────────────────────────────────
    if is_main:
        print(f"[{global_rank}] 모델 로드: {MODEL_ID}")
    # config.vocab_size가 실제 체크포인트 가중치 크기와 다를 수 있으므로 패치 후 로드.
    # Qwen2.5 계열은 config.vocab_size=152064이지만 실제 임베딩은 151668인 경우 존재.
    _base_vocab = len(tokenizer)  # special token이 이미 base vocab에 있으면 add_special_tokens가 0 반환
    _config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
    if _config.vocab_size != _base_vocab:
        if is_main:
            print(f"[{global_rank}] config.vocab_size({_config.vocab_size}) != tokenizer vocab({_base_vocab}), 패치 후 로드")
        _config.vocab_size = _base_vocab
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=_config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # 특수 토큰 추가로 늘어난 vocab에 맞게 embedding 크기 조정
    model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── 옵티마이저 & 스케줄러 ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps  = (len(loader) // args.grad_accum) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if is_main:
        print(f"총 옵티마이저 스텝: {total_steps}  (warmup: {warmup_steps})")

    # ── 학습 ─────────────────────────────────────────────────────────────────
    # 실행 시간 기반 하위 폴더: output_dir/{YYYYMMDD_HHMMSS}/
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, ts)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        print(f"[저장 경로] {run_dir}")
    dist.barrier()   # 다른 rank들이 디렉토리 생성 전에 진입하지 않도록 대기
    global_step = 0

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        if is_main:
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        else:
            pbar = loader

        accum_loss = 0.0

        for step_in_epoch, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            # 그래디언트 누적 후 업데이트
            if (step_in_epoch + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main:
                    pbar.set_postfix({
                        "loss": f"{accum_loss:.4f}",
                        "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
                        "step": global_step,
                    })
                accum_loss = 0.0

                # 체크포인트 저장
                if is_main and global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(run_dir, f"step_{global_step}")
                    model.module.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"\n[저장] {ckpt_dir}")

        # 에폭 종료 시 저장
        if is_main:
            ckpt_dir = os.path.join(run_dir, f"epoch{epoch+1}")
            model.module.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"\n[에폭 {epoch+1} 저장] {ckpt_dir}")

    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",    default=DATA_PATH)
    p.add_argument("--output_dir",   default=OUTPUT_DIR)
    # 베이스 모델은 항상 MODEL_ID(Qwen2.5-7B-Instruct)로 고정 — 인자로 변경 불가
    p.add_argument("--lr",           type=float, default=LEARNING_RATE)
    p.add_argument("--num_epochs",   type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--grad_accum",   type=int,   default=GRAD_ACCUM)
    p.add_argument("--max_length",   type=int,   default=MAX_LENGTH)
    p.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--max_grad_norm",type=float, default=MAX_GRAD_NORM)
    p.add_argument("--save_steps",   type=int,   default=SAVE_STEPS)
    p.add_argument("--preview",      action="store_true",
                   help="input/output 샘플 하나 출력 후 종료")
    return p.parse_args()


def main():
    args = parse_args()

    if args.preview:
        # 프리뷰 모드: 토크나이저만 로드해서 샘플 출력
        tokenizer = setup_tokenizer(MODEL_ID, CACHE_DIR, special_tokens=SPECIAL_TOKENS)
        preview_sample(args.data_path, tokenizer)
        return

    train(args)


if __name__ == "__main__":
    main()
