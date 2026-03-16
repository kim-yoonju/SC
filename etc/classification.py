"""
Action classification: history_before -> {<|solve|>, <|correct|>} 예측 (2-class)

Qwen2.5-7B-Instruct backbone (frozen) + linear classification head 학습

레이블 규칙:
  현재 step의 temp_reward == 0.0  →  <|correct|>
  현재 step의 temp_reward != 0.0  →  <|solve|>

  action == "end" 인 step은 학습에서 제외

저장:
  - checkpoints/action_cls/best_model/classifier_head.pt   (linear head state_dict)
  - checkpoints/action_cls/best_model/classifier_config.json (hidden_size, num_labels 등)
  → 다른 스크립트에서 nn.Linear 생성 후 load_state_dict 로 바로 로드 가능
"""

'''
이 코드는 classification head의 기능을 확인하기 위한걸로
미리 생성해둔 데이터로 linear layer 하나를 학습시킴
'''
import argparse
import json
import os
import random
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from utils import apply_chat_template, format_messages

# ---------------------------------------------------------------------------
# 레이블 정의
# ---------------------------------------------------------------------------

LABEL2ID = {"solve": 0, "correct": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 2


def get_label(step: dict) -> int:
    """현재 step의 temp_reward 기준으로 레이블 결정.

    temp_reward == 0.0  →  correct  (이 경로는 틀림 → 되돌아가야 함)
    temp_reward != 0.0  →  solve    (계속 풀기)
    """
    if step.get("temp_reward", 1.0) == 0.0:
        return LABEL2ID["correct"]
    return LABEL2ID["solve"]


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_examples(jsonl_files: List[str]) -> List[Tuple[str, list, int]]:
    """JSONL 파일들에서 (problem, history_before, label) 리스트를 반환한다.

    action == "end" 인 step은 완전히 제외한다.
    """
    examples = []
    for path in jsonl_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                problem = data["problem"]
                for step in data["steps"]:
                    if step["action"] == "end":
                        continue
                    examples.append((problem, step["history_before"], get_label(step)))
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
    """Left-padding collate (decoder-only 모델용)."""
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
# 모델: Qwen2.5-7B-Instruct (frozen) + classification head
# ---------------------------------------------------------------------------

class ActionClassifier(nn.Module):
    """Qwen2.5-7B-Instruct backbone + 2-class linear head.

    backbone은 freeze, classifier head만 학습한다.
    마지막 실제 토큰의 hidden state로 분류한다.
    """

    def __init__(self, model_name: str, num_labels: int = NUM_LABELS,
                 gpu_ids: list = None, class_weights: torch.Tensor = None,
                 label_smoothing: float = 0.1):
        super().__init__()
        if gpu_ids is None:
            gpu_ids = [2]

        # 단일 GPU: device_map으로 명시적 고정 / 멀티 GPU: max_memory로 분산
        if len(gpu_ids) == 1:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{gpu_ids[0]}"},
            )
            self.first_device = f"cuda:{gpu_ids[0]}"
            self.last_device  = f"cuda:{gpu_ids[0]}"
        else:
            max_memory = {i: "90GiB" for i in gpu_ids}
            self.backbone = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
            )
            device_map = self.backbone.hf_device_map
            self.first_device = f"cuda:{gpu_ids[0]}"
            self.last_device  = "cuda:" + str(
                max(v for v in device_map.values() if isinstance(v, int))
            )

        # backbone 완전 동결
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels).to(self.last_device)
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.to(self.last_device))
        else:
            self.register_buffer("class_weights", None)

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.to(self.first_device)
        attention_mask = attention_mask.to(self.first_device)

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, T, H)

        # 각 시퀀스의 마지막 실제 토큰 위치
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        seq_lengths = seq_lengths.to(last_hidden.device)
        pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lengths]

        logits = self.classifier(pooled.to(self.last_device).float())  # (B, num_labels)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            # class_weights: 불균형 보정 / label_smoothing: minority 과신 방지
            loss = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )(logits, labels)

        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# classifier head 저장 / 로드 유틸
# ---------------------------------------------------------------------------

def save_classifier_head(model: ActionClassifier, save_dir: str):
    """classifier head state_dict + config를 저장한다.

    다른 스크립트에서 load_classifier_head() 로 바로 로드 가능.
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        model.classifier.state_dict(),
        os.path.join(save_dir, "classifier_head.pt"),
    )
    config = {
        "hidden_size": model.hidden_size,
        "num_labels": model.num_labels,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
    }
    with open(os.path.join(save_dir, "classifier_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_classifier_head(save_dir: str, device: str = "cpu") -> Tuple[nn.Linear, dict]:
    """저장된 classifier head를 로드한다.

    Returns:
        head   : nn.Linear(hidden_size, num_labels) with loaded weights
        config : {"hidden_size", "num_labels", "id2label", "label2id"}

    사용 예:
        head, cfg = load_classifier_head("checkpoints/action_cls/best_model")
        head = head.to(device).float()
    """
    with open(os.path.join(save_dir, "classifier_config.json")) as f:
        config = json.load(f)
    head = nn.Linear(config["hidden_size"], config["num_labels"])
    state = torch.load(os.path.join(save_dir, "classifier_head.pt"), map_location=device)
    head.load_state_dict(state)
    head.eval()
    return head, config


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def train(args):
    jsonl_files = sorted(glob(os.path.join(args.data_dir, "rollouts_math7500_*.jsonl")))
    assert jsonl_files, f"JSONL 파일이 없습니다: {args.data_dir}"
    print(f"[cls] 데이터 파일 {len(jsonl_files)}개: {jsonl_files}")

    examples = load_examples(jsonl_files)
    print(f"[cls] 전체 예제 수: {len(examples)}")

    # 레이블 분포 & 클래스 가중치 (inverse frequency)
    label_counts = [0] * NUM_LABELS
    for _, _, lbl in examples:
        label_counts[lbl] += 1
    for lid, cnt in enumerate(label_counts):
        print(f"  {ID2LABEL[lid]}: {cnt} ({100 * cnt / len(examples):.1f}%)")

    total = len(examples)
    class_weights = torch.tensor(
        [total / (NUM_LABELS * cnt) if cnt > 0 else 0.0 for cnt in label_counts],
        dtype=torch.float32,
    )
    print(f"[cls] class_weights: { {ID2LABEL[i]: f'{w:.3f}' for i, w in enumerate(class_weights.tolist())} }")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # train 90% / val 10% 분할
    random.seed(42)
    indices = list(range(len(examples)))
    random.shuffle(indices)
    n_val = max(1, int(len(examples) * 0.1))
    train_examples = [examples[i] for i in indices[n_val:]]
    val_examples   = [examples[i] for i in indices[:n_val]]
    print(f"[cls] train={len(train_examples)}, val={len(val_examples)}")

    train_ds = ActionDataset(train_examples, tokenizer, max_length=args.max_length)
    val_ds   = ActionDataset(val_examples,   tokenizer, max_length=args.max_length)

    sample_weights = [class_weights[lbl].item() for _, _, lbl in train_examples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"[cls] 모델 로드: {args.model_name}")
    model = ActionClassifier(
        args.model_name,
        num_labels=NUM_LABELS,
        gpu_ids=args.gpu_ids,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # classifier head만 학습
    optimizer = AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[cls] 저장 경로: {run_dir}")
    best_macro_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step_i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} train")):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            (out["loss"] / args.grad_accum).backward()
            total_loss += out["loss"].item()

            if (step_i + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        # --- Val ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                all_preds.append(out["logits"].argmax(dim=-1).cpu())
                all_labels.append(batch["labels"].cpu())

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_acc = (all_preds == all_labels).float().mean().item()

        metrics = []
        for c in range(NUM_LABELS):
            tp = ((all_preds == c) & (all_labels == c)).sum().item()
            fp = ((all_preds == c) & (all_labels != c)).sum().item()
            fn = ((all_preds != c) & (all_labels == c)).sum().item()
            sup = (all_labels == c).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            metrics.append((prec, rec, f1, sup))

        macro_f1 = sum(m[2] for m in metrics) / NUM_LABELS

        print(f"\n[Epoch {epoch}] loss={avg_loss:.4f}  acc={val_acc:.4f}")
        print(f"{'':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
        print("-" * 52)
        for c, (prec, rec, f1, sup) in enumerate(metrics):
            print(f"{ID2LABEL[c]:>10s} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {sup:>10d}")
        print("-" * 52)
        print(f"{'macro':>10s} {sum(m[0] for m in metrics)/NUM_LABELS:>10.4f} "
              f"{sum(m[1] for m in metrics)/NUM_LABELS:>10.4f} {macro_f1:>10.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_dir = os.path.join(run_dir, "best_model")
            save_classifier_head(model, save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  -> 저장: {save_dir} (macro_f1={macro_f1:.4f})")

    print(f"[cls] 완료. best macro_f1={best_macro_f1:.4f}  저장: {run_dir}/best_model")


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------

def predict(model: ActionClassifier, tokenizer, problem: str, history: list,
            max_length: int = 2048) -> str:
    """history_before를 보고 다음 액션 예측.

    Returns:
        "<|solve|>" 또는 "<|correct|>"
    """
    messages = format_messages(problem, history)
    prompt = apply_chat_template(tokenizer, messages)
    enc = tokenizer(prompt, max_length=max_length, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(model.first_device),
            attention_mask=enc["attention_mask"].to(model.first_device),
        )
    pred_id = out["logits"].argmax(dim=-1).item()
    return f"<|{ID2LABEL[pred_id]}|>"


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Action classification 학습")
    parser.add_argument("--model_name",  type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_dir",    type=str,   default="data/rollouts")
    parser.add_argument("--output_dir",  type=str,   default="checkpoints/action_cls")
    parser.add_argument("--max_length",  type=int,   default=2048)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--grad_accum",  type=int,   default=8)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="label smoothing (minority 클래스 과신 방지)")
    parser.add_argument("--gpu_ids",     type=int,   nargs="+", default=[2],
                        help="사용할 GPU 인덱스 목록 (예: --gpu_ids 2)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
