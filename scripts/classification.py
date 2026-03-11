"""
Action classification: history -> {<|solve|>, <|correct|>} 예측 (2-class)

Qwen2.5-7B backbone + classification head

레이블 규칙:
  <|correct|>: 직전 step의 temp_reward == 0.0
  <|solve|>  : 그 외 (기본값)

  action == "end" 인 step은 학습에서 제외 (boxed 출력이 곧 end)
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import format_messages, apply_chat_template


# ---------------------------------------------------------------------------
# 레이블 정의
# ---------------------------------------------------------------------------

LABEL2ID = {"solve": 0, "correct": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 2


def get_label(step: dict, prev_step: dict = None) -> int:
    """현재 step의 레이블 결정.

    - 직전 step의 temp_reward==0 → <|correct|>
    - 그 외                      → <|solve|>
    """
    if prev_step is not None and prev_step.get("temp_reward", 1.0) == 0.0:
        return LABEL2ID["correct"]
    return LABEL2ID["solve"]


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_examples(jsonl_files: List[str]) -> List[Tuple[str, list, int]]:
    """JSONL 파일들에서 (problem, history_before, label) 리스트를 반환한다."""
    examples = []
    for path in jsonl_files:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                problem = data["problem"]
                steps = data["steps"]
                for i, step in enumerate(steps):
                    if step["action"] == "end":  # end step은 제외
                        continue
                    history = step["history_before"]
                    prev_step = steps[i - 1] if i > 0 else None
                    label = get_label(step, prev_step)
                    examples.append((problem, history, label))
    return examples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ActionDataset(Dataset):
    def __init__(self, examples: List[Tuple], tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []
        for problem, history, label in examples:
            messages = format_messages(problem, history)
            prompt = apply_chat_template(tokenizer, messages)
            self.items.append((prompt, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        prompt, label = self.items[idx]
        enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    """Left-padding collate."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, -seq_len:] = item["input_ids"]
        attention_mask[i, -seq_len:] = item["attention_mask"]
        labels[i] = item["label"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# 모델: Qwen2.5-7B + classification head
# ---------------------------------------------------------------------------

class ActionClassifier(nn.Module):
    """Qwen2.5-7B backbone에 3-class linear head를 붙인 분류 모델.

    마지막 실제 토큰(패딩 제외)의 hidden state로 분류한다.
    backbone은 bfloat16으로 로드하고, classifier head는 float32로 유지한다.
    """

    def __init__(self, model_name: str, num_labels: int = 3, freeze_backbone: bool = False,
                 gpu_ids: list = None, class_weights: torch.Tensor = None):
        super().__init__()
        if gpu_ids is None:
            gpu_ids = [4, 5]

        # 지정한 GPU들에만 메모리 할당 (나머지는 0으로 막음)
        max_memory = {i: "90GiB" for i in gpu_ids}
        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        # 입력은 첫 번째 GPU, 출력(classifier head)은 마지막 GPU에 배치
        device_map = self.backbone.hf_device_map
        self.first_device = f"cuda:{gpu_ids[0]}"
        self.last_device = "cuda:" + str(max(v for v in device_map.values() if isinstance(v, int)))
        self.classifier = nn.Linear(hidden_size, num_labels).to(self.last_device)
        self.num_labels = num_labels

        # 클래스 불균형 보정용 가중치 (buffer로 등록 → device 이동 자동)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.to(self.last_device))
        else:
            self.register_buffer("class_weights", None)

    def forward(self, input_ids, attention_mask, labels=None):
        # 입력을 backbone 첫 레이어가 있는 GPU로 이동
        input_ids = input_ids.to(self.first_device)
        attention_mask = attention_mask.to(self.first_device)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state  # (B, T, H)

        # attention_mask에서 각 시퀀스의 마지막 실제 토큰 위치 추출
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        batch_size = input_ids.size(0)
        # last_hidden device로 seq_lengths 이동
        seq_lengths = seq_lengths.to(last_hidden.device)
        pooled = last_hidden[torch.arange(batch_size, device=last_hidden.device), seq_lengths]
        # (B, H) -> classifier device로 이동 후 float32 변환
        logits = self.classifier(pooled.to(self.last_device).float())  # (B, num_labels)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)

        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def train(args):
    # 데이터 로드
    jsonl_files = sorted(glob(os.path.join(args.data_dir, "rollouts_math7500_*.jsonl")))
    assert jsonl_files, f"JSONL 파일이 없습니다: {args.data_dir}"
    print(f"[cls] 데이터 파일 {len(jsonl_files)}개: {jsonl_files}")

    examples = load_examples(jsonl_files)
    print(f"[cls] 전체 예제 수: {len(examples)}")

    # 레이블 분포 & 클래스 가중치 계산 (inverse frequency)
    label_counts = [0] * NUM_LABELS
    for _, _, lbl in examples:
        label_counts[lbl] += 1
    for lid, cnt in enumerate(label_counts):
        print(f"  {ID2LABEL[lid]}: {cnt} ({100*cnt/len(examples):.1f}%)")

    total = len(examples)
    class_weights = torch.tensor(
        [total / (NUM_LABELS * cnt) if cnt > 0 else 0.0 for cnt in label_counts],
        dtype=torch.float32,
    )
    print(f"[cls] class_weights: { {ID2LABEL[i]: f'{w:.3f}' for i, w in enumerate(class_weights.tolist())} }")

    # 토크나이저 & 모델
    print(f"[cls] 토크나이저 로드: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 데이터셋 분할 (train 90% / val 10%)
    random.seed(42)
    indices = list(range(len(examples)))
    random.shuffle(indices)
    n_val = max(1, int(len(examples) * 0.1))
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    train_examples = [examples[i] for i in train_idx]
    val_examples   = [examples[i] for i in val_idx]
    print(f"[cls] train={len(train_examples)}, val={len(val_examples)}")

    train_ds = ActionDataset(train_examples, tokenizer, max_length=args.max_length)
    val_ds   = ActionDataset(val_examples,   tokenizer, max_length=args.max_length)

    # WeightedRandomSampler: 배치 구성 단계에서 minority 클래스 오버샘플링
    sample_weights = [class_weights[lbl].item() for _, _, lbl in train_examples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(f"[cls] WeightedRandomSampler 적용 (replacement=True)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               sampler=sampler, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 모델 로드
    print(f"[cls] 모델 로드: {args.model_name}")
    model = ActionClassifier(
        args.model_name,
        num_labels=NUM_LABELS,
        freeze_backbone=args.freeze_backbone,
        gpu_ids=args.gpu_ids,
        class_weights=class_weights,
    )

    # Optimizer: backbone과 head를 다른 lr로
    head_params = list(model.classifier.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    param_groups = [
        {"params": head_params, "lr": args.head_lr},
    ]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_macro_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} train")):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out["loss"] / args.grad_accum
            loss.backward()
            total_loss += out["loss"].item()

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        # --- Val ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                all_preds.append(out["logits"].argmax(dim=-1).cpu())
                all_labels.append(batch["labels"].cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        val_acc = (all_preds == all_labels).float().mean().item()

        # 클래스별 Precision / Recall / F1
        metrics = []
        for c in range(NUM_LABELS):
            tp = ((all_preds == c) & (all_labels == c)).sum().item()
            fp = ((all_preds == c) & (all_labels != c)).sum().item()
            fn = ((all_preds != c) & (all_labels == c)).sum().item()
            support = (all_labels == c).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            metrics.append((prec, rec, f1, support))

        macro_prec = sum(m[0] for m in metrics) / NUM_LABELS
        macro_rec  = sum(m[1] for m in metrics) / NUM_LABELS
        macro_f1   = sum(m[2] for m in metrics) / NUM_LABELS

        print(f"\n[Epoch {epoch}] loss={avg_loss:.4f}  acc={val_acc:.4f}")
        print(f"{'':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
        print("-" * 52)
        for c, (prec, rec, f1, sup) in enumerate(metrics):
            print(f"{ID2LABEL[c]:>10s} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {sup:>10d}")
        print("-" * 52)
        print(f"{'macro':>10s} {macro_prec:>10.4f} {macro_rec:>10.4f} {macro_f1:>10.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_path = os.path.join(args.output_dir, "best_model")
            model.backbone.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            torch.save(model.classifier.state_dict(),
                       os.path.join(save_path, "classifier_head.pt"))
            print(f"  -> 저장: {save_path} (macro_f1={macro_f1:.4f})")

    print(f"[cls] 완료. best macro_f1={best_macro_f1:.4f}")


# ---------------------------------------------------------------------------
# 추론 (단일 예시 테스트)
# ---------------------------------------------------------------------------

def predict(model: ActionClassifier, tokenizer, problem: str, history: list,
            max_length: int = 2048) -> str:
    messages = format_messages(problem, history)
    prompt = apply_chat_template(tokenizer, messages)
    enc = tokenizer(prompt, max_length=max_length, truncation=True, return_tensors="pt")

    device = next(model.classifier.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    pred_id = out["logits"].argmax(dim=-1).item()
    return f"<|{ID2LABEL[pred_id]}|>"


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Action classification 학습")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--data_dir", type=str, default="data/rollouts")
    parser.add_argument("--output_dir", type=str, default="checkpoints/action_cls")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="backbone을 고정하고 head만 학습")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[4, 5],
                        help="사용할 GPU 인덱스 목록 (예: --gpu_ids 4 5)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
