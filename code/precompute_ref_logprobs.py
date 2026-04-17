"""
Precompute per-token log-probabilities from the reference model and save them
to a .pt file. This only needs to run once; during SFT training the cached
values are used instead of running the ref model at every step.

Memory note
-----------
Processes the sequence in small chunks (default 256 tokens) so the fp32 logits
intermediate never exceeds ~2.5 GiB at batch=16 (vs ~55 GiB for the full tensor).

  model (bf16):            ~14 GiB
  full bf16 logits (GPU):  ~28 GiB  (batch=16, seq≈5700)
  fp32 chunk (chunk=256):  ~ 2.5 GiB
  ─────────────────────────────────
  peak GPU:                ~44.5 GiB  (batch=16 on a 95 GiB GPU → very safe)

Single GPU:
    CUDA_VISIBLE_DEVICES=0 python code/precompute_ref_logprobs.py \
        --data_path ./data/train_data/sft_data.json \
        --model_path Qwen/Qwen2.5-Math-7B \
        --output_path ./data/train_data/ref_logprobs.pt \
        --model_max_length 8000 --batch_size 16

Multi-GPU (launched per-worker by train_sft.sh):
    CUDA_VISIBLE_DEVICES=6 python code/precompute_ref_logprobs.py ... \
        --gpu_id 0 --world_size 2 --output_path ref_logprobs.pt.part0
"""

import argparse
import json
import time
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer


def log(msg: str, flush: bool = True):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=flush)


def tokenize_sample(prompt: str, answer: str, tokenizer, max_length: int):
    """Replicates the tokenization in collators._llm_tokenize (single sample)."""
    text = prompt + answer
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    text_ids   = tokenizer.encode(text,   add_special_tokens=False)

    response_start_idx = len(prompt_ids)
    if prompt_ids != text_ids[:response_start_idx]:
        response_start_idx -= 1

    label = deepcopy(text_ids)
    label[:response_start_idx] = [-100] * response_start_idx

    if len(text_ids) > max_length:
        text_ids = text_ids[-max_length:]
        label    = label[-max_length:]

    return text_ids, label


@torch.no_grad()
def compute_batch_ref_logprobs(model, input_ids, attention_mask, labels, device,
                               chunk_size: int = 256):
    """
    Returns per-token log-probs processed in sequence chunks to avoid ever
    materializing the full (B, S, V) fp32 tensor on GPU.

    Memory breakdown (batch=16, seq≈5700, vocab=152064):
      - bf16 shift_logits in GPU:  ~28 GiB  (unavoidable from forward pass)
      - fp32 chunk (chunk_size=256): ~2.5 GiB  (tiny, reused each iteration)
      - model weights: ~14 GiB
      - Total peak: ~44.5 GiB  (vs 94.97 GiB available)

    Masked positions are stored as 0.0.
    Output shape per sample: (actual_seq_len - 1,)
    """
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
    )

    # shift: position i predicts token i+1
    shift_logits = outputs.logits[:, :-1, :].contiguous()  # (B, S-1, V) bf16
    del outputs  # free immediately
    shift_labels = labels[:, 1:].to(device)                # (B, S-1)

    B, S, V = shift_logits.shape
    token_logprobs = torch.zeros(B, S)  # accumulate on CPU

    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)

        # Cast only this slice to fp32 — peak ~2.5 GiB for chunk_size=256
        chunk_logits = shift_logits[:, start:end, :].float()   # (B, chunk, V) fp32
        chunk_labels = shift_labels[:, start:end].clone()       # (B, chunk)
        chunk_mask   = chunk_labels == -100
        chunk_labels[chunk_mask] = 0  # avoid index=-100 in cross_entropy

        chunk_lp = -F.cross_entropy(
            chunk_logits.view(-1, V),
            chunk_labels.view(-1),
            reduction="none",
        ).view(B, end - start)
        del chunk_logits

        chunk_lp[chunk_mask] = 0.0
        token_logprobs[:, start:end] = chunk_lp.cpu()

    del shift_logits

    return token_logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",        type=str, required=True)
    parser.add_argument("--model_path",       type=str, required=True)
    parser.add_argument("--output_path",      type=str, required=True)
    parser.add_argument("--model_max_length", type=int, default=8000)
    parser.add_argument("--batch_size",       type=int, default=16)
    parser.add_argument("--gpu_id",           type=int, default=0)
    parser.add_argument("--world_size",       type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data_path) as f:
        data = json.load(f)

    # Shard: worker i processes indices [i, i+world_size, i+2*world_size, ...]
    my_indices = list(range(args.gpu_id, len(data), args.world_size))
    my_data    = [data[idx] for idx in my_indices]

    log(f"[GPU {args.gpu_id}/{args.world_size}] {len(my_data)}/{len(data)} samples  "
        f"batch={args.batch_size}  device={device}")
    log(f"Loading model from {args.model_path} ...")

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(device).eval()

    log("Model loaded. Starting inference ...")
    t_start = time.time()
    results = []
    num_batches = (len(my_data) + args.batch_size - 1) // args.batch_size

    for bi, i in enumerate(range(0, len(my_data), args.batch_size)):
        t_batch = time.time()
        batch        = my_data[i : i + args.batch_size]
        batch_global = my_indices[i : i + args.batch_size]

        raw_input_ids, raw_labels = [], []
        for sample in batch:
            ids, lbl = tokenize_sample(
                sample["prompt"], sample["answer"], tokenizer, args.model_max_length
            )
            raw_input_ids.append(torch.tensor(ids, dtype=torch.long))
            raw_labels.append(torch.tensor(lbl, dtype=torch.long))

        input_ids      = pad_sequence(raw_input_ids, batch_first=True,
                                      padding_value=tokenizer.pad_token_id)
        labels         = pad_sequence(raw_labels,    batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        token_logprobs = compute_batch_ref_logprobs(
            model, input_ids, attention_mask, labels, device
        )

        for j, global_idx in enumerate(batch_global):
            actual_len = len(raw_input_ids[j]) - 1
            results.append((global_idx, token_logprobs[j, :actual_len]))

        elapsed   = time.time() - t_start
        per_batch = (time.time() - t_batch)
        remaining = per_batch * (num_batches - bi - 1)
        log(f"  batch [{bi+1}/{num_batches}]  "
            f"{per_batch:.1f}s/batch  "
            f"elapsed {elapsed:.0f}s  "
            f"ETA {remaining:.0f}s")

    total = time.time() - t_start
    log(f"Done. {len(results)} samples in {total:.1f}s  "
        f"({total/len(my_data):.2f}s/sample)")

    torch.save(results, args.output_path)
    log(f"Saved → {args.output_path}")


if __name__ == "__main__":
    main()
