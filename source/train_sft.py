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
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent

MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR      = "/mnt/.cache/huggingface"
DATA_PATH      = str(_ROOT / "datasets/sft_data.jsonl")
OUTPUT_DIR     = str(_ROOT / "output/sft_checkpoints")

# 특수 토큰: 텍스트 뒤에 append되어 다음 행동을 나타낸다
# <|end|>은 강화학습용 커스텀 종료 토큰 (Qwen vocab에 없어 별도 등록 필요)
SPECIAL_TOKENS = ["<|solve|>", "<|correct|>", "<|end|>"]

# 학습 하이퍼파라미터
LEARNING_RATE  = 2e-5
NUM_EPOCHS     = 3          # 데이터 1818개로 적으므로 3 epoch
BATCH_SIZE     = 1          # per GPU (배치 4는 OOM)
GRAD_ACCUM     = 32         # effective batch = 3 GPUs × 1 × 32 = 96
MAX_LENGTH     = 3072
WARMUP_RATIO   = 0.05
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 1.0
SAVE_STEPS     = 100        # 몇 스텝마다 체크포인트 저장

# 프롬프트는 source/prompts.json에서 로드 (모든 스크립트가 동일한 프롬프트 공유)
_PROMPTS_PATH = _ROOT / "source" / "prompts.json"
with open(_PROMPTS_PATH) as _f:
    _PROMPTS = json.load(_f)

SYSTEM_SOLVE   = _PROMPTS["system_solve"]
SYSTEM_CORRECT = _PROMPTS["system_correct"]


def _build_chat_prefix(tokenizer, system: str, problem: str, history: list) -> str:
    """prototype의 build_chat_prompt + _solve_user와 동일한 포맷으로 prefix 생성."""
    lines = [f"[Problem]\n{problem}"]
    if history:
        lines.append("\n[Steps so far]")
        for i, s in enumerate(history, 1):
            lines.append(f"Step {i}: {s}")
    lines.append("\nWrite the next step.")
    user_msg = "\n".join(lines)

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\n\nUser: {user_msg}\n\nAssistant:"


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 전처리
# ─────────────────────────────────────────────────────────────────────────────

def extract_boxed(text: str) -> str | None:
    """텍스트에서 마지막 \\boxed{...} 내용을 추출한다 (중첩 괄호 처리)."""
    marker = r"\boxed{"
    pos = text.rfind(marker)
    if pos == -1:
        return None
    start = pos + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def extract_step_content(step: dict) -> tuple[str, str]:
    """
    step dict에서 (action, clean_text)를 추출한다.
    두 가지 포맷 모두 처리:
      - 구 포맷: action 필드 + text 필드 (text에 <|im_end|> 포함)
      - 신 포맷: action 필드 + content 필드 + text 필드 (<solve>...</solve> 임베드)
    반환: action 문자열, 순수 텍스트 (태그/im_end 제거)
    """
    action = step["action"]

    # 신 포맷: content 필드가 있으면 우선 사용 (None 체크 포함)
    if "content" in step and step["content"] is not None:
        text = step["content"].strip()
    else:
        text = step.get("text", "")
        # <|end|> / <|im_end|> 제거
        text = text.replace("<|end|>", "").replace("<|im_end|>", "").strip()
        # <action>...</action> 태그가 임베드된 경우 내용만 추출
        for act in ["solve", "correct", "end", "review"]:
            m = re.search(rf"<{act}>(.*?)</{act}>", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
                break

    return action, text


def build_target_text(action: str, text: str) -> str:
    """텍스트 + 액션 특수 토큰을 하나의 타겟 문자열로 합친다.
    예: "reasoning...<|solve|>"  또는  "\\boxed{42}<|end|>"
    """
    return f"{text}<|{action}|>"


def pick_system(action: str) -> str:
    """현재 스텝의 액션에 맞는 system 프롬프트 반환."""
    return SYSTEM_CORRECT if action == "correct" else SYSTEM_SOLVE


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


def setup_tokenizer(model_id: str, cache_dir: str):
    """토크나이저를 로드하고 특수 토큰을 추가한다."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tokenizer


def load_raw_data(data_path: str) -> list[dict]:
    items = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


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
            prefix_str = _build_chat_prefix(tokenizer, system, problem, history)

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
# 콜레이터
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch, pad_token_id: int):
    """가변 길이 시퀀스를 패딩해 배치로 묶는다."""
    input_ids_list, labels_list = zip(*batch)

    max_len = max(x.size(0) for x in input_ids_list)

    padded_input  = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100,         dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (inp, lbl) in enumerate(zip(input_ids_list, labels_list)):
        seq_len = inp.size(0)
        padded_input[i, :seq_len]   = inp
        padded_labels[i, :seq_len]  = lbl
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids":      padded_input,
        "attention_mask": attention_mask,
        "labels":         padded_labels,
    }


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
    full_prefix  = _build_chat_prefix(tokenizer, system, problem, history)

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
    tokenizer = setup_tokenizer(MODEL_ID, CACHE_DIR)

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        drop_last=True,
    )

    from functools import partial
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
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
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
        tokenizer = setup_tokenizer(MODEL_ID, CACHE_DIR)
        preview_sample(args.data_path, tokenizer)
        return

    train(args)


if __name__ == "__main__":
    main()
