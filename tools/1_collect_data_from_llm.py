import os
import ast
import json
import argparse
import time
import tqdm
import torch
from answer_extraction import extract_answer
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class TokenLogger(LogitsProcessor):
    """Logs token generation progress every N tokens."""
    def __init__(self, every_n=100, label="", max_new_tokens=None):
        self.every_n = every_n
        self.label = label
        self.max_new_tokens = max_new_tokens
        self.count = 0
        self.start = time.time()

    def __call__(self, input_ids, scores):
        self.count += 1
        if self.count % self.every_n == 0:
            elapsed = time.time() - self.start
            tok_s = self.count / elapsed if elapsed > 0 else 0
            total_str = f"/{self.max_new_tokens}" if self.max_new_tokens else ""
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {self.label}  → {self.count}{total_str} tokens  ({tok_s:.0f} tok/s)", flush=True)
        return scores


def _normalize_parquet_row(row, idx):
    def _as_dict(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            return ast.literal_eval(v)
        return {}

    extra_info = _as_dict(row.get("extra_info", {}))
    if extra_info.get("question"):
        problem = extra_info["question"]
    else:
        prompt_raw = row.get("prompt")
        prompt = list(prompt_raw) if prompt_raw is not None else []
        user_msgs = [m["content"] for m in prompt if isinstance(m, dict) and m.get("role") == "user"]
        problem = user_msgs[-1] if user_msgs else ""

    sol = row.get("solution")
    if sol is None or (not isinstance(sol, str) and str(sol) == "nan"):
        sol = None
    solution = sol or extra_info.get("answer", "")

    uid = row.get("unique_id")
    if uid is None or (not isinstance(uid, str) and str(uid) == "nan"):
        uid = None
    if not uid:
        src = str(row.get("data_source", "unknown")).split("/")[-1]
        uid = f"{src}_{extra_info.get('index', idx)}"

    reward_model = _as_dict(row.get("reward_model", {}))
    gold_answer = str(reward_model.get("ground_truth", ""))

    result = {
        "problem": problem,
        "solution": solution,
        "unique_id": uid,
        "gold_extracted_answer": gold_answer,
    }
    subject = row.get("subject")
    if subject and str(subject) != "nan":
        result["subject"] = subject
    level = row.get("level")
    if level and str(level) != "nan":
        result["level"] = level
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir",           type=str, required=True)
    parser.add_argument("--data_names",         type=str, nargs="+", required=True)
    parser.add_argument("--output_file",        type=str, required=True)
    parser.add_argument("--batch_size",         type=int, default=32)
    parser.add_argument("--n_samples",          type=int, default=5)
    parser.add_argument("--temperature",        type=float, default=0.7)
    parser.add_argument("--max_new_tokens",     type=int, default=2048)
    parser.add_argument("--gpu_id",             type=int, default=0)
    parser.add_argument("--world_size",         type=int, default=1)
    args = parser.parse_args()

    output_file = args.output_file

    # ── Load & merge all datasets ──────────────────────────────────────────────
    lines = []
    for data_name in args.data_names:
        data_path = os.path.join(args.data_dir, data_name)
        if data_name.endswith(".parquet"):
            import pandas as pd
            raw = pd.read_parquet(data_path).to_dict(orient="records")
            dataset = [_normalize_parquet_row(r, i) for i, r in enumerate(raw)]
        else:
            with open(data_path, "r") as f:
                dataset = [json.loads(l) for l in f.readlines()]
        print(f"  {len(dataset)} problems loaded from {data_path}")
        lines.extend(dataset)

    print(f"Total: {len(lines)} problems from {len(args.data_names)} dataset(s)")

    # Resume: 모든 part 파일을 스캔해 완료된 문제를 전체에서 제거 후 재분배
    import glob, re
    base = re.sub(r'\.part\d+$', '', output_file)
    all_parts = sorted(glob.glob(f"{base}.part*"))
    if all_parts:
        uid_n_samples = {}
        for pf in all_parts:
            with open(pf) as f:
                for line in f:
                    rec = json.loads(line)
                    uid = rec["unique_id"]
                    n = rec.get("n_samples_collected", len(rec.get("round_1_response", [])))
                    uid_n_samples[uid] = max(uid_n_samples.get(uid, 0), n)
        completed_ids = {uid for uid, n in uid_n_samples.items() if n >= args.n_samples}
        remaining = [l for l in lines if l["unique_id"] not in completed_ids]
        lines = remaining[args.gpu_id::args.world_size]
        print(f"Resume: {len(completed_ids)} completed globally, {len(remaining)} remaining → {len(lines)} for this worker")
    else:
        # 첫 실행: 인터리브 분할
        lines = lines[args.gpu_id::args.world_size]
        print(f"[GPU {args.gpu_id}/{args.world_size}] {len(lines)} problems assigned")

    if not lines:
        print("Nothing to process.")
        exit(0)

    n_batches = (len(lines) + args.batch_size - 1) // args.batch_size
    total_problems = len(lines)
    print(f"Start: {total_problems} problems / batch_size={args.batch_size} → {n_batches} batches × {args.n_samples} samples")

    # ── Load model ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},  # CUDA_VISIBLE_DEVICES로 GPU 지정
    )
    model.eval()
    log("Model loaded")

    # ── Build prompts ──────────────────────────────────────────────────────────
    prompts = []
    for line in lines:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user",   "content": line["problem"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        ))

    # ── Batch inference + immediate save ───────────────────────────────────────
    solved = 0
    pbar = tqdm.tqdm(total=total_problems, desc=f"GPU{args.gpu_id}", unit="prob", dynamic_ncols=True)
    out_f = open(output_file, "a", buffering=1)
    try:
        for batch_idx, batch_start in enumerate(range(0, len(lines), args.batch_size)):
            batch_lines   = lines[batch_start: batch_start + args.batch_size]
            batch_prompts = prompts[batch_start: batch_start + args.batch_size]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            B = len(batch_prompts)
            label = f"[GPU{args.gpu_id} B{batch_idx+1}/{n_batches}]"
            token_logger = TokenLogger(
                every_n=100,
                label=label,
                max_new_tokens=args.max_new_tokens,
            )
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=1.0,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=args.n_samples,
                    logits_processor=LogitsProcessorList([token_logger]),
                )

            # output_ids: [B * n_samples, seq_len] → [B, n_samples, new_tokens]
            generated = output_ids[:, prompt_len:].view(B, args.n_samples, -1)

            for j, data_line in enumerate(batch_lines):
                responses = [tokenizer.decode(generated[j, s], skip_special_tokens=True) for s in range(args.n_samples)]
                gold = data_line.get("gold_extracted_answer") or extract_answer(data_line["solution"])
                result = {
                    "problem":                  data_line["problem"],
                    "round_1_instruction":      data_line["problem"],
                    "prompt_1":                 batch_prompts[j],
                    "round_1_response":         responses,
                    "round_1_extracted_answer": [extract_answer(r) for r in responses],
                    "gold_extracted_answer":    gold,
                    "solution":                 data_line["solution"],
                    "unique_id":                data_line["unique_id"],
                    "subject":                  data_line.get("subject"),
                    "level":                    data_line.get("level"),
                    "n_samples_collected":      args.n_samples,
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                solved += 1
                pbar.update(1)
                tqdm.tqdm.write(f"[{time.strftime('%H:%M:%S')}] [GPU{args.gpu_id}] [{solved}/{total_problems}] saved: {data_line['unique_id']}")

    finally:
        pbar.close()
        out_f.close()

    log(f"Done! {solved} problems saved to {output_file}")
