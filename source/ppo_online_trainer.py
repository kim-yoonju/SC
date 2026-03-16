"""
Online PPO RL Training with Ray

Architecture:
  RolloutWorker (Ray actor, 1 per GPU)
    → generate_data_teacher.py 와 동일한 방식으로 trajectory 생성
    → step별로 MC rollout reward 계산 + teacher injection
  PPOTrainer (main process)
    → policy model + frozen reference model + critic value head
    → collected trajectory로 PPO update
    → 업데이트된 weights를 workers에 동기화

PPO step-level MDP:
  state  = tokenized(problem + history)
  action = 한 스텝의 생성 텍스트 (e.g. <solve>...</solve>)
  reward = final_reward from MC rollouts (generate_data_teacher.py 동일)
  log_prob = sum of token log probs for response tokens

Teacher steps (reward=None)은 PPO loss에서 제외하고 선택적으로 SFT loss에 포함.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from openai import OpenAI
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

import ray

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    apply_chat_template,
    check_answer_correct,
    extract_first_action,
    format_messages,
    parse_boxed,
    parse_step,
)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

STOP_STRINGS = ["</solve>", "</correct>", "</end>"]

# classifier head 출력 매핑 (classification.py 와 동일)
CLS_ID2LABEL = {0: "solve", 1: "correct"}


# ---------------------------------------------------------------------------
# Stopping Criteria (generate_data_teacher.py 와 동일)
# ---------------------------------------------------------------------------

class BatchedActionTagStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_ids_list: list, input_length: int, batch_size: int):
        self.stop_ids_list = stop_ids_list
        self.input_length = input_length
        self.done = [False] * batch_size

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        for i in range(len(self.done)):
            if self.done[i]:
                continue
            new_ids = input_ids[i][self.input_length:].tolist()
            for stop in self.stop_ids_list:
                n = len(stop)
                if len(new_ids) >= n and new_ids[-n:] == stop:
                    self.done[i] = True
                    break
        return all(self.done)


def _get_stop_ids(tokenizer) -> list:
    stop_ids = []
    for s in STOP_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_ids.append(ids)
    return stop_ids


# ---------------------------------------------------------------------------
# 생성 헬퍼
# ---------------------------------------------------------------------------

def generate_steps_batch(model, tokenizer, stop_ids, prompts, max_new_tokens=512, temperature=0.8,
                         prefixes=None):
    """
    prefixes: 프롬프트별 강제 액션 태그 prefix (예: ["<solve>", "<correct>", ...]).
              제공되면 각 프롬프트 뒤에 붙여서 생성하고, 결과 앞에 다시 prepend한다.
    """
    if prefixes is not None:
        prompts = [p + pfx for p, pfx in zip(prompts, prefixes)]

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList([
        BatchedActionTagStoppingCriteria(stop_ids, input_length, len(prompts))
    ])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    results = []
    for i in range(len(prompts)):
        new_tokens = outputs[i][input_length:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if prefixes is not None:
            generated = prefixes[i] + generated
        results.append(extract_first_action(generated))
    return results


def generate_teacher_step(problem: str, history: list) -> Optional[str]:
    """GPT teacher model로 correction step 생성"""
    openai_client = OpenAI()
    messages = format_messages(problem, history)
    prompt = ""
    for m in messages:
        prompt += f"{m['role']}: {m['content']}\n"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a math expert who fixes incorrect reasoning. Output exactly one action tag: <correct>...</correct>"},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[teacher] generation failed: {e}")
        return None


def run_mc_rollouts_batch(model, tokenizer, stop_ids, problem, history, gold_answer,
                          n_rollouts=8, max_steps=10, max_new_tokens=512, temperature=0.8):
    histories = [list(history) for _ in range(n_rollouts)]
    results = [{"correct": False, "num_steps": 0} for _ in range(n_rollouts)]
    done = [False] * n_rollouts

    for _ in range(max_steps):
        active = [i for i in range(n_rollouts) if not done[i]]
        if not active:
            break
        prompts = [
            apply_chat_template(tokenizer, format_messages(problem, histories[i]))
            for i in active
        ]
        step_texts = generate_steps_batch(model, tokenizer, stop_ids, prompts, max_new_tokens, temperature)
        for j, i in enumerate(active):
            step_text = step_texts[j]
            action, content = parse_step(step_text)
            if action == "end":
                is_correct = check_answer_correct(content or step_text, gold_answer)
                results[i] = {"correct": is_correct, "num_steps": len(histories[i]) + 1}
                done[i] = True
            elif action is None:
                results[i]["num_steps"] = len(histories[i]) + 1
                done[i] = True
            else:
                histories[i].append(step_text)
                results[i]["num_steps"] = len(histories[i])
    return results


# ---------------------------------------------------------------------------
# Trajectory 구조체
# ---------------------------------------------------------------------------

class StepRecord:
    """한 스텝의 trajectory 데이터"""
    __slots__ = ["problem", "history_before", "step_text", "action",
                 "final_reward", "is_teacher", "problem_id"]

    def __init__(self, problem, history_before, step_text, action, final_reward,
                 is_teacher=False, problem_id=""):
        self.problem = problem
        self.history_before = history_before
        self.step_text = step_text
        self.action = action
        self.final_reward = final_reward
        self.is_teacher = is_teacher
        self.problem_id = problem_id


# ---------------------------------------------------------------------------
# Ray Rollout Worker
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=1)
class RolloutWorker:
    """
    하나의 GPU에서 trajectory를 생성하는 Ray Actor.
    generate_data_teacher.py 의 process_problems_batch 와 동일한 로직.
    """

    def __init__(self, model_name: str, worker_id: int, dtype: str = "bfloat16",
                 n_rollouts: int = 8, max_steps: int = 10,
                 max_new_tokens: int = 512, temperature: float = 0.8,
                 jsonl_path: str = None, classifier_head_path: str = None):
        self.worker_id = worker_id
        self.n_rollouts = n_rollouts
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.jsonl_path = jsonl_path

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        print(f"[Worker {worker_id}] 모델 로드: {model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_map[dtype],
            device_map="auto",
        )
        self.model.eval()
        self.stop_ids = _get_stop_ids(self.tokenizer)

        # classification head (선택적 로드)
        self.classifier_head = None
        if classifier_head_path and os.path.isfile(classifier_head_path):
            hidden_size = self.model.config.hidden_size
            head = nn.Linear(hidden_size, 2)
            head.load_state_dict(torch.load(classifier_head_path, map_location="cpu"))
            device = next(self.model.parameters()).device
            self.classifier_head = head.to(device).float().eval()
            print(f"[Worker {worker_id}] classifier head 로드: {classifier_head_path}", flush=True)

        print(f"[Worker {worker_id}] 준비 완료", flush=True)

    def _save_step(self, record: dict):
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def predict_actions(self, prompts: list) -> list:
        """classifier head로 다음 액션 타입 예측 (solve / correct).

        policy model의 마지막 레이어 hidden state를 pooling해
        classifier head에 통과시켜 solve(0) / correct(1)을 반환한다.
        """
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        last_hidden = out.hidden_states[-1]  # (B, T, H)
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1  # (B,)
        pooled = last_hidden[torch.arange(len(prompts), device=device), seq_lengths]  # (B, H)

        cls_device = next(self.classifier_head.parameters()).device
        logits = self.classifier_head(pooled.to(cls_device).float())  # (B, 2)
        preds = logits.argmax(dim=-1).tolist()
        return [CLS_ID2LABEL[p] for p in preds]

    def generate_trajectories(self, problems_batch: list) -> list:
        """
        problems_batch: list of {"problem_id": str, "problem": str, "answer": str}
        return: list of StepRecord dicts
        """
        n = len(problems_batch)
        problem_ids = [p["problem_id"] for p in problems_batch]
        problems = [p["problem"] for p in problems_batch]
        gold_answers = [p["answer"] for p in problems_batch]

        histories = [[] for _ in range(n)]
        prev_solve_rewards = [0.0] * n
        active = list(range(n))
        all_step_records = []

        for step_idx in range(self.max_steps):
            if not active:
                break

            prompts = [
                apply_chat_template(self.tokenizer, format_messages(problems[i], histories[i]))
                for i in active
            ]

            # classifier head가 로드된 경우: 마지막 스텝을 제외하고 액션 타입 강제
            # (마지막 스텝은 자유 생성하여 <end> 자연 종료 허용)
            if self.classifier_head is not None and step_idx < self.max_steps - 1:
                predicted = self.predict_actions(prompts)
                prefixes = [f"<{a}>" for a in predicted]
            else:
                prefixes = None

            step_texts = generate_steps_batch(
                self.model, self.tokenizer, self.stop_ids, prompts,
                self.max_new_tokens, self.temperature, prefixes=prefixes,
            )

            still_active = []
            for j, i in enumerate(active):
                step_text = step_texts[j]
                action, content = parse_step(step_text)
                pid = problem_ids[i]

                if action is None:
                    continue  # 유효하지 않은 action → 이 문제 종료

                if action == "end":
                    is_correct = check_answer_correct(content or step_text, gold_answers[i])
                    num_steps = step_idx + 1
                    final_reward = (1.0 - (num_steps - 1) * 0.05) if is_correct else -1.0
                    rec = {
                        "problem_id": pid,
                        "problem": problems[i],
                        "history_before": list(histories[i]),
                        "step_text": step_text,
                        "action": action,
                        "final_reward": final_reward,
                        "is_teacher": False,
                    }
                    all_step_records.append(rec)
                    self._save_step(rec)
                    continue  # 완료

                # MC rollouts
                history_after = histories[i] + [step_text]
                rollouts = run_mc_rollouts_batch(
                    self.model, self.tokenizer, self.stop_ids,
                    problems[i], history_after, gold_answers[i],
                    n_rollouts=self.n_rollouts,
                    max_steps=self.max_steps,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                n_correct = sum(1 for r in rollouts if r["correct"])
                temp_reward = n_correct / self.n_rollouts

                if action == "solve":
                    final_reward = temp_reward
                    prev_solve_rewards[i] = final_reward
                elif action == "correct":
                    final_reward = max(0.0, temp_reward - prev_solve_rewards[i])
                else:
                    final_reward = 0.0

                rec = {
                    "problem_id": pid,
                    "problem": problems[i],
                    "history_before": list(histories[i]),
                    "step_text": step_text,
                    "action": action,
                    "final_reward": final_reward,
                    "is_teacher": False,
                }
                all_step_records.append(rec)
                self._save_step(rec)

                # Teacher injection for failed corrections
                if action == "correct" and temp_reward == 0.0:
                    teacher_text = generate_teacher_step(problems[i], histories[i])
                    if teacher_text is not None:
                        teacher_rec = {
                            "problem_id": pid,
                            "problem": problems[i],
                            "history_before": list(histories[i]),
                            "step_text": teacher_text,
                            "action": "teacher_correct",
                            "final_reward": None,  # PPO에서 제외, SFT에 포함
                            "is_teacher": True,
                        }
                        all_step_records.append(teacher_rec)
                        self._save_step(teacher_rec)
                        histories[i].append(teacher_text)
                        still_active.append(i)
                        continue

                histories[i].append(step_text)
                still_active.append(i)

            active = still_active

        return all_step_records

    def update_weights(self, state_dict):
        """Trainer로부터 새 weights 로드 (Ray가 ObjectRef를 자동 역참조해서 전달)"""
        # critic head 제외한 policy weights만 업데이트
        model_state = self.model.state_dict()
        policy_keys = {k: v for k, v in state_dict.items() if not k.startswith("critic_")}
        model_state.update(policy_keys)
        self.model.load_state_dict(model_state, strict=False)
        self.model.eval()
        print(f"[Worker {self.worker_id}] weights 업데이트 완료", flush=True)

    def ping(self):
        return f"worker_{self.worker_id}_ok"


# ---------------------------------------------------------------------------
# Critic Value Head
# ---------------------------------------------------------------------------

class CriticValueHead(nn.Module):
    """Policy LM의 마지막 hidden state → scalar value"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, H) → (B,)
        # 마지막 non-padding 토큰의 hidden state 사용
        return self.linear(hidden_states[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO 손실 계산
# ---------------------------------------------------------------------------

def compute_sequence_log_prob(logits, input_ids, prompt_lengths, attention_mask):
    """
    response 토큰들의 평균 log prob 계산
    logits: (B, T, V)
    returns: (B,) - 각 sequence의 mean response log prob
    """
    shift_logits = logits[:, :-1, :]     # (B, T-1, V)
    shift_labels = input_ids[:, 1:]      # (B, T-1)
    attn_shifted = attention_mask[:, 1:] # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    B, T1 = shift_labels.shape
    response_mask = torch.zeros(B, T1, device=input_ids.device)
    for i, plen in enumerate(prompt_lengths):
        response_mask[i, plen - 1:] = 1.0
    response_mask = response_mask * attn_shifted

    # 평균 log prob (response 토큰만)
    seq_log_prob = (token_log_probs * response_mask).sum(dim=1) / (
        response_mask.sum(dim=1).clamp(min=1)
    )
    return seq_log_prob, response_mask


def compute_ppo_loss(
    policy_model,
    ref_model,
    critic_head,
    batch,
    trainer_device,
    policy_model_raw=None,  # DataParallel일 때 hook 등록용 원본 모델
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
    vf_coef: float = 0.5,
    entropy_coef: float = 0.01,
    normalize_advantages: bool = True,
):
    """
    batch keys:
      input_ids, attention_mask, prompt_lengths, advantages, returns,
      log_probs_old, rewards, is_teacher_mask
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    prompt_lengths = batch["prompt_lengths"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    log_probs_old = batch["log_probs_old"]
    is_teacher_mask = batch["is_teacher_mask"]  # True면 SFT, False면 PPO

    # Policy forward - hook으로 마지막 hidden state 캡처
    _last_hidden = {}
    def _hook(module, inp, out):
        _last_hidden['h'] = out[0] if isinstance(out, tuple) else out
    hook_handle = policy_model_raw.model.norm.register_forward_hook(_hook)

    outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
    hook_handle.remove()

    logits = outputs.logits
    hidden_states = _last_hidden['h']

    # Value - float32로 통일
    values = critic_head(hidden_states.detach()).float()  # (B,)

    # Log probs (new) - float32로 통일
    log_probs_new, response_mask = compute_sequence_log_prob(logits, input_ids, prompt_lengths, attention_mask)
    log_probs_new = log_probs_new.float()
    response_mask = response_mask.float()

    # Reference log probs (for KL) - ref model은 CPU에 있으므로 input도 CPU로
    with torch.no_grad():
        ref_input_ids = input_ids.cpu()
        ref_attention_mask = attention_mask.cpu()
        ref_prompt_lengths = prompt_lengths.cpu()
        ref_outputs = ref_model(input_ids=ref_input_ids, attention_mask=ref_attention_mask)
        ref_log_probs, _ = compute_sequence_log_prob(
            ref_outputs.logits, ref_input_ids, ref_prompt_lengths, ref_attention_mask
        )
        ref_log_probs = ref_log_probs.to(trainer_device).float()

    # --- PPO Policy Loss (RL steps only) ---
    rl_mask = ~is_teacher_mask  # (B,) bool
    ppo_loss = torch.tensor(0.0, device=input_ids.device)
    if rl_mask.any():
        ratio = torch.exp(log_probs_new[rl_mask] - log_probs_old[rl_mask])
        adv = advantages[rl_mask]
        if normalize_advantages and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        ppo_loss = -torch.min(surr1, surr2).mean()

    # --- KL divergence penalty ---
    kl_loss = (log_probs_new - ref_log_probs).mean()

    # --- Value Loss (RL steps only) ---
    vf_loss = torch.tensor(0.0, device=input_ids.device)
    if rl_mask.any():
        vf_loss = F.mse_loss(values[rl_mask], returns[rl_mask])

    # --- Entropy bonus ---
    if entropy_coef > 0:
        shift_logits = logits[:, :-1, :].float()
        entropy = -(F.softmax(shift_logits, dim=-1) * F.log_softmax(shift_logits, dim=-1)).sum(dim=-1)
        entropy_mean = (entropy * response_mask).sum() / response_mask.sum().clamp(min=1)
    else:
        entropy_mean = torch.tensor(0.0, device=logits.device)

    # --- SFT Loss for teacher steps ---
    sft_loss = torch.tensor(0.0, device=input_ids.device)
    if is_teacher_mask.any():
        shift_logits_sft = logits[:, :-1, :].float()[is_teacher_mask]
        shift_labels_sft = input_ids[:, 1:][is_teacher_mask]
        attn_sft = attention_mask[:, 1:][is_teacher_mask]
        plen_sft = prompt_lengths[is_teacher_mask]
        B_sft, T1 = shift_labels_sft.shape
        resp_mask_sft = torch.zeros(B_sft, T1, device=input_ids.device)
        for i, plen in enumerate(plen_sft):
            resp_mask_sft[i, plen - 1:] = 1.0
        resp_mask_sft = resp_mask_sft * attn_sft
        sft_log_probs = -F.cross_entropy(
            shift_logits_sft.reshape(-1, shift_logits_sft.size(-1)),
            shift_labels_sft.reshape(-1),
            reduction="none",
        ).reshape(B_sft, T1)
        sft_loss = -(sft_log_probs * resp_mask_sft).sum() / resp_mask_sft.sum().clamp(min=1)

    total_loss = ppo_loss + kl_coef * kl_loss + vf_coef * vf_loss - entropy_coef * entropy_mean + 0.1 * sft_loss

    return total_loss, {
        "ppo_loss": ppo_loss.item(),
        "vf_loss": vf_loss.item(),
        "kl": kl_loss.item(),
        "entropy": entropy_mean.item(),
        "sft_loss": sft_loss.item(),
        "total_loss": total_loss.item(),
    }


# ---------------------------------------------------------------------------
# GAE Advantage 계산
# ---------------------------------------------------------------------------

def compute_gae_advantages(
    step_records: list,   # 동일 problem의 steps, 시간순 정렬
    values: list,         # critic 값 (각 step에 대해)
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple:
    """
    GAE advantage 계산.
    step_records는 동일 problem의 steps (시간 순서).
    returns: (advantages, returns) - numpy arrays
    """
    rewards = [s["final_reward"] for s in step_records]
    T = len(rewards)
    advantages = np.zeros(T)
    returns = np.zeros(T)

    gae = 0.0
    next_value = 0.0  # terminal state

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]

    return advantages, returns


# ---------------------------------------------------------------------------
# 배치 토크나이징
# ---------------------------------------------------------------------------

def tokenize_batch(step_records, tokenizer, max_length=2048):
    """
    step_records list → collated batch tensors
    """
    prompts = []
    full_texts = []
    for rec in step_records:
        messages = format_messages(rec["problem"], rec["history_before"])
        prompt = apply_chat_template(tokenizer, messages)
        full_text = prompt + rec["step_text"]
        prompts.append(prompt)
        full_texts.append(full_text)

    full_enc = tokenizer(
        full_texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    )
    prompt_enc = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    )
    prompt_lengths = prompt_enc["attention_mask"].sum(dim=1)

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "prompt_lengths": prompt_lengths,
    }


# ---------------------------------------------------------------------------
# 데이터셋 로드
# ---------------------------------------------------------------------------

def load_math_dataset(dataset_name: str, split: str = "train"):
    if dataset_name.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=dataset_name, split="train")
    else:
        ds = load_dataset(dataset_name, split=split)

    def normalize(example):
        if "problem" not in example:
            prompt = example.get("prompt", [])
            if isinstance(prompt, list):
                user_msgs = [m["content"] for m in prompt if m.get("role") == "user"]
                example["problem"] = user_msgs[0] if user_msgs else ""
            else:
                example["problem"] = example.get("question", "")
        if "answer" not in example:
            if "final_answer" in example:
                example["answer"] = example["final_answer"]
            else:
                reward_model = example.get("reward_model", {})
                if isinstance(reward_model, dict) and "ground_truth" in reward_model:
                    example["answer"] = reward_model["ground_truth"]
                else:
                    solution = example.get("solution", "")
                    example["answer"] = parse_boxed(solution) or solution
        return example

    return ds.map(normalize)


# ---------------------------------------------------------------------------
# 메인 PPO 학습 루프
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Online PPO RL Training with Ray")
    # 모델
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/ppo_online")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])

    # 데이터
    parser.add_argument("--dataset", type=str, default="datasets/math7500.parquet")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)

    # Rollout 설정 (generate_data_teacher.py 와 동일한 인터페이스)
    parser.add_argument("--n_rollout_workers", type=int, default=2,
                        help="Ray rollout actor 수 (각 actor는 GPU 1개 사용)")
    parser.add_argument("--n_rollouts", type=int, default=8,
                        help="MC rollout 횟수")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="문제당 최대 스텝 수")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--problems_per_rollout", type=int, default=4,
                        help="한 rollout 배치에 포함할 문제 수 (worker당)")

    # PPO 학습 설정
    parser.add_argument("--num_iterations", type=int, default=1000,
                        help="PPO 업데이트 iteration 수")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="수집된 batch에 대한 PPO epoch 수")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="PPO 업데이트 미니배치 크기")
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--normalize_advantages", action="store_true", default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # 체크포인트
    parser.add_argument("--save_every", type=int, default=100,
                        help="몇 iteration마다 체크포인트 저장할지")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="재개할 체크포인트 경로")

    # 로깅
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ppo_sc_math")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout_tag", type=str, default=None,
                        help="rollout 저장 파일 suffix (예: iter2 → online_ppo_math7500_iter2_worker0.jsonl)")
    parser.add_argument("--classifier_head_path", type=str, default=None,
                        help="학습된 classifier head 경로 (예: checkpoints/action_cls/best_model/classifier_head.pt)")
    parser.add_argument("--use_cached_rollout", action="store_true",
                        help="rollout 스킵하고 기존 worker jsonl 파일에서 데이터 로드")
    parser.add_argument("--cached_rollout_lines", type=int, default=None,
                        help="각 worker 파일에서 마지막 N줄만 사용 (None=전체)")
    parser.add_argument("--cached_rollout_skip", type=int, nargs="+", default=None,
                        help="각 worker 파일에서 앞에서 건너뛸 줄 수 (worker별로 지정: 307 333 311)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # WandB
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    log_file = open(args.log_file, "w") if args.log_file else None

    def log(msg):
        print(msg, flush=True)
        if log_file:
            print(msg, file=log_file, flush=True)

    # -----------------------------------------------------------------------
    # 1. Ray 초기화
    # -----------------------------------------------------------------------
    log("[PPO] Ray 초기화...")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # -----------------------------------------------------------------------
    # 2. 데이터셋 로드
    # -----------------------------------------------------------------------
    log(f"[PPO] 데이터셋 로드: {args.dataset}")
    dataset = load_math_dataset(args.dataset, args.split)
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    end_idx = min(end_idx, len(dataset))
    dataset = dataset.select(range(args.start_idx, end_idx))
    log(f"[PPO] 문제 수: {len(dataset)}")

    # 문제 리스트 (반복해서 샘플링)
    all_problems = [
        {
            "problem_id": str(args.start_idx + i),
            "problem": ex["problem"],
            "answer": ex["answer"],
        }
        for i, ex in enumerate(dataset)
    ]

    # -----------------------------------------------------------------------
    # 3. Rollout Workers 초기화
    # -----------------------------------------------------------------------
    log(f"[PPO] RolloutWorker {args.n_rollout_workers}개 초기화...")
    workers = [
        RolloutWorker.remote(
            model_name=args.model_name,
            worker_id=i,
            dtype=args.torch_dtype,
            n_rollouts=args.n_rollouts,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            jsonl_path=os.path.join(args.output_dir, "rollouts", f"online_ppo_{Path(args.dataset).stem}{'_' + args.rollout_tag if args.rollout_tag else ''}_worker{i}.jsonl"),
            classifier_head_path=args.classifier_head_path,
        )
        for i in range(args.n_rollout_workers)
    ]
    # 초기화 확인
    pings = ray.get([w.ping.remote() for w in workers])
    log(f"[PPO] Workers ready: {pings}")

    # -----------------------------------------------------------------------
    # JSONL 저장 설정
    # -----------------------------------------------------------------------
    rollout_dir = os.path.join(args.output_dir, "rollouts")
    os.makedirs(rollout_dir, exist_ok=True)
    dataset_stem = Path(args.dataset).stem
    jsonl_path = os.path.join(rollout_dir, f"online_ppo_{dataset_stem}{'_' + args.rollout_tag if args.rollout_tag else ''}.jsonl")
    jsonl_file = open(jsonl_path, "a", encoding="utf-8")
    log(f"[PPO] Trajectory JSONL 저장 경로: {jsonl_path}")

    def save_trajectories_to_jsonl(step_records: list, iteration: int):
        """수집된 step_records를 generate_data_teacher.py 포맷으로 JSONL에 저장"""
        from collections import defaultdict
        # problem_id로 그룹핑
        grouped = defaultdict(list)
        for rec in step_records:
            grouped[rec["problem_id"]].append(rec)
        for pid, steps in grouped.items():
            # step_idx 순서로 정렬
            entry = {
                "problem_id": pid,
                "iteration": iteration,
                "problem": steps[0]["problem"],
                "steps": [
                    {
                        "step_idx": i,
                        "action": s["action"],
                        "text": s["step_text"],
                        "history_before": s["history_before"],
                        "final_reward": s["final_reward"],
                        "is_teacher": s["is_teacher"],
                    }
                    for i, s in enumerate(steps)
                ],
            }
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        jsonl_file.flush()

    # -----------------------------------------------------------------------
    # 4. Trainer 모델 로드 (policy + ref + critic)
    # -----------------------------------------------------------------------
    log(f"[PPO] Trainer 모델 로드: {args.model_name}")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # resume 시 policy는 checkpoint에서 로드, ref는 항상 원본 모델
    policy_load_path = args.resume_from if args.resume_from else args.model_name
    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_load_path,
        torch_dtype=torch_dtype,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    )

    # Trainer가 사용할 device
    # n_rollout_workers개의 GPU를 workers가 쓰므로 나머지 GPU 사용
    if torch.cuda.device_count() > args.n_rollout_workers:
        trainer_gpu_id = args.n_rollout_workers  # workers 다음 GPU
    else:
        trainer_gpu_id = 0

    trainer_device = torch.device(f"cuda:{trainer_gpu_id}" if torch.cuda.is_available() else "cpu")
    policy_model = policy_model.to(trainer_device)
    policy_model.gradient_checkpointing_enable()  # activation 메모리 절감
    policy_model_raw = policy_model  # .config, .save_pretrained 등 원본 접근용

    # ref model은 CPU에 두고 inference 시에만 GPU로 올림 (GPU 메모리 절약)
    ref_model = ref_model.cpu()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    hidden_size = policy_model_raw.config.hidden_size
    critic_head = CriticValueHead(hidden_size).to(trainer_device).to(torch_dtype)

    # iteration 재개 정보 먼저 읽기
    start_iteration = 0
    if args.resume_from:
        meta = json.load(open(os.path.join(args.resume_from, "meta.json")))
        start_iteration = meta.get("iteration", 0) + 1
        log(f"[PPO] 체크포인트 재개: {args.resume_from} → iteration {start_iteration}부터")

    # critic resume (model과 달리 별도 파일)
    if args.resume_from:
        critic_head.load_state_dict(
            torch.load(os.path.join(args.resume_from, "critic.pt"), map_location=trainer_device)
        )

    # -----------------------------------------------------------------------
    # 5. Optimizer
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW([
        {"params": policy_model.parameters(), "lr": args.learning_rate},
        {"params": critic_head.parameters(), "lr": args.critic_lr},
    ], weight_decay=0.01)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_iterations * args.ppo_epochs,
    )

    # optimizer / scheduler 상태 복원
    if args.resume_from:
        opt_path = os.path.join(args.resume_from, "optimizer.pt")
        sch_path = os.path.join(args.resume_from, "scheduler.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=trainer_device))
        if os.path.exists(sch_path):
            scheduler.load_state_dict(torch.load(sch_path))

    # -----------------------------------------------------------------------
    # 6. 학습 루프
    # -----------------------------------------------------------------------
    log(f"[PPO] 학습 시작: {args.num_iterations} iterations")
    log(f"[PPO] Trainer device: {trainer_device}")

    problem_indices = list(range(len(all_problems)))
    np.random.shuffle(problem_indices)
    prob_ptr = 0
    all_metrics = []

    for iteration in range(start_iteration, args.num_iterations):
        iter_start = time.time()

        # ------------------------------------------------------------------
        # 6a. Rollout: 각 worker에 문제 배치 분배
        # ------------------------------------------------------------------
        n_per_worker = args.problems_per_rollout
        worker_batches = []
        for w in range(args.n_rollout_workers):
            batch_problems = []
            for _ in range(n_per_worker):
                if prob_ptr >= len(problem_indices):
                    np.random.shuffle(problem_indices)
                    prob_ptr = 0
                batch_problems.append(all_problems[problem_indices[prob_ptr]])
                prob_ptr += 1
            worker_batches.append(batch_problems)

        # 비동기 rollout 실행 (또는 캐시에서 로드)
        if args.use_cached_rollout:
            all_step_records = []
            tag = ('_' + args.rollout_tag) if args.rollout_tag else ''
            for i in range(args.n_rollout_workers):
                worker_jsonl = os.path.join(args.output_dir, "rollouts",
                    f"online_ppo_{Path(args.dataset).stem}{tag}_worker{i}.jsonl")
                if os.path.exists(worker_jsonl):
                    with open(worker_jsonl, encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    if args.cached_rollout_skip is not None:
                        skip = args.cached_rollout_skip[i] if i < len(args.cached_rollout_skip) else args.cached_rollout_skip[-1]
                        lines = lines[skip:]
                    elif args.cached_rollout_lines is not None:
                        lines = lines[-args.cached_rollout_lines:]
                    for line in lines:
                        all_step_records.append(json.loads(line))
            log(f"[PPO] 캐시된 rollout 로드: {len(all_step_records)} steps")
        else:
            rollout_futures = [
                workers[w].generate_trajectories.remote(worker_batches[w])
                for w in range(args.n_rollout_workers)
            ]
            rollout_results = ray.get(rollout_futures)

            # 결과 합치기
            all_step_records = []
            for worker_steps in rollout_results:
                all_step_records.extend(worker_steps)

        if not all_step_records:
            log(f"[PPO] iteration {iteration}: rollout 결과 없음, 스킵")
            continue

        # RL steps만 분리 (teacher steps 제외하고 value/advantage 계산)
        rl_records = [r for r in all_step_records if not r["is_teacher"]]
        teacher_records = [r for r in all_step_records if r["is_teacher"]]

        log(f"[PPO] iteration {iteration}: {len(rl_records)} RL steps, {len(teacher_records)} teacher steps")

        # JSONL 저장 (모든 step - RL + teacher)
        save_trajectories_to_jsonl(all_step_records, iteration)

        if not rl_records:
            continue

        # ------------------------------------------------------------------
        # 6b. Old log probs 및 Values 계산 (현재 policy로, gradient 없음)
        # ------------------------------------------------------------------
        policy_model.eval()
        old_log_probs_list = []
        values_list = []

        # RL records를 미니배치로 처리
        for batch_start in range(0, len(rl_records), args.batch_size):
            batch_recs = rl_records[batch_start: batch_start + args.batch_size]
            tok_batch = tokenize_batch(batch_recs, tokenizer, args.max_length)
            input_ids = tok_batch["input_ids"].to(trainer_device)
            attention_mask = tok_batch["attention_mask"].to(trainer_device)
            prompt_lengths = tok_batch["prompt_lengths"].to(trainer_device)

            with torch.no_grad():
                _last_hidden_eval = {}
                def _hook_eval(module, inp, out):
                    _last_hidden_eval['h'] = out[0] if isinstance(out, tuple) else out
                hook_handle_eval = policy_model_raw.model.norm.register_forward_hook(_hook_eval)
                outputs = policy_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hook_handle_eval.remove()
                lp, _ = compute_sequence_log_prob(
                    outputs.logits, input_ids, prompt_lengths, attention_mask
                )
                vals = critic_head(_last_hidden_eval['h'])

            old_log_probs_list.append(lp.cpu())
            values_list.append(vals.cpu())

        old_log_probs_all = torch.cat(old_log_probs_list, dim=0).float().numpy()  # (N_rl,)
        values_all = torch.cat(values_list, dim=0).float().numpy()                 # (N_rl,)

        # ------------------------------------------------------------------
        # 6c. GAE Advantage 계산 (문제별로 묶어서)
        # ------------------------------------------------------------------
        # problem_id로 그룹핑 (순서 유지)
        from collections import defaultdict
        problem_step_indices = defaultdict(list)
        for idx, rec in enumerate(rl_records):
            problem_step_indices[rec["problem_id"]].append(idx)

        advantages_all = np.zeros(len(rl_records))
        returns_all = np.zeros(len(rl_records))

        for pid, indices in problem_step_indices.items():
            steps_in_problem = [rl_records[i] for i in indices]
            vals_in_problem = [values_all[i] for i in indices]
            adv, ret = compute_gae_advantages(
                steps_in_problem, vals_in_problem, args.gamma, args.lam
            )
            for local_i, global_i in enumerate(indices):
                advantages_all[global_i] = adv[local_i]
                returns_all[global_i] = ret[local_i]

        # dataset 전체 단위로 advantage normalize (mini-batch 단위보다 안정적)
        if args.normalize_advantages and advantages_all.std() > 1e-8:
            advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-8)

        # ------------------------------------------------------------------
        # 6d. PPO Update epochs
        # ------------------------------------------------------------------
        torch.cuda.empty_cache()
        policy_model.train()
        epoch_metrics = []

        # teacher records를 RL records 뒤에 붙여서 is_teacher_mask로 구분
        combined_records = rl_records + teacher_records
        combined_adv = np.concatenate([
            advantages_all,
            np.zeros(len(teacher_records))  # teacher는 PPO에서 제외
        ])
        combined_ret = np.concatenate([
            returns_all,
            np.zeros(len(teacher_records))
        ])
        combined_lp_old = np.concatenate([
            old_log_probs_all,
            np.zeros(len(teacher_records))
        ])
        is_teacher_arr = np.array(
            [False] * len(rl_records) + [True] * len(teacher_records)
        )

        N = len(combined_records)
        indices_perm = np.arange(N)

        for ppo_epoch in range(args.ppo_epochs):
            np.random.shuffle(indices_perm)
            optimizer.zero_grad()
            accum_step = 0
            batch_metrics_list = []

            for batch_start in range(0, N, args.batch_size):
                batch_idx = indices_perm[batch_start: batch_start + args.batch_size]
                if len(batch_idx) == 0:
                    continue

                batch_recs = [combined_records[i] for i in batch_idx]
                tok_batch = tokenize_batch(batch_recs, tokenizer, args.max_length)

                train_batch = {
                    "input_ids": tok_batch["input_ids"].to(trainer_device),
                    "attention_mask": tok_batch["attention_mask"].to(trainer_device),
                    "prompt_lengths": tok_batch["prompt_lengths"].to(trainer_device),
                    "advantages": torch.tensor(combined_adv[batch_idx], dtype=torch.float32).to(trainer_device),
                    "returns": torch.tensor(combined_ret[batch_idx], dtype=torch.float32).to(trainer_device),
                    "log_probs_old": torch.tensor(combined_lp_old[batch_idx], dtype=torch.float32).to(trainer_device),
                    "rewards": torch.tensor(
                        [combined_records[i]["final_reward"] or 0.0 for i in batch_idx],
                        dtype=torch.float32,
                    ).to(trainer_device),
                    "is_teacher_mask": torch.tensor(is_teacher_arr[batch_idx], dtype=torch.bool).to(trainer_device),
                }

                loss, metrics = compute_ppo_loss(
                    policy_model, ref_model, critic_head, train_batch,
                    trainer_device=trainer_device,
                    policy_model_raw=policy_model_raw,
                    clip_eps=args.clip_eps,
                    kl_coef=args.kl_coef,
                    vf_coef=args.vf_coef,
                    entropy_coef=args.entropy_coef,
                    normalize_advantages=False,  # dataset 전체 단위로 이미 normalize 완료
                )

                # gradient accumulation
                (loss / args.grad_accum_steps).backward()
                accum_step += 1
                batch_metrics_list.append(metrics)

                if accum_step % args.grad_accum_steps == 0 or batch_start + args.batch_size >= N:
                    torch.nn.utils.clip_grad_norm_(
                        list(policy_model.parameters()) + list(critic_head.parameters()),
                        args.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # epoch 평균 metrics
            if batch_metrics_list:
                avg_epoch_metrics = {
                    k: np.mean([m[k] for m in batch_metrics_list])
                    for k in batch_metrics_list[0]
                }
                epoch_metrics.append(avg_epoch_metrics)

        # ------------------------------------------------------------------
        # 6e. Worker weights 동기화
        # ------------------------------------------------------------------
        policy_model.eval()
        state_dict = {k: v.cpu() for k, v in policy_model_raw.state_dict().items()}
        state_dict_ref = ray.put(state_dict)
        sync_futures = [w.update_weights.remote(state_dict_ref) for w in workers]
        ray.get(sync_futures)

        # ------------------------------------------------------------------
        # 6f. 로깅
        # ------------------------------------------------------------------
        iter_time = time.time() - iter_start
        avg_reward = np.mean([r["final_reward"] for r in rl_records])
        avg_metrics = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0]} if epoch_metrics else {}

        log_entry = {
            "iteration": iteration,
            "avg_reward": float(avg_reward),
            "n_rl_steps": len(rl_records),
            "n_teacher_steps": len(teacher_records),
            "iter_time_sec": round(iter_time, 1),
            **avg_metrics,
        }
        all_metrics.append(log_entry)

        log(
            f"[PPO] iter {iteration:4d} | "
            f"reward={avg_reward:.3f} | "
            f"ppo={avg_metrics.get('ppo_loss', 0):.4f} | "
            f"vf={avg_metrics.get('vf_loss', 0):.4f} | "
            f"kl={avg_metrics.get('kl', 0):.5f} | "
            f"t={iter_time:.1f}s"
        )

        if args.use_wandb:
            import wandb
            wandb.log(log_entry, step=iteration)

        # ------------------------------------------------------------------
        # 6g. 체크포인트 저장
        # ------------------------------------------------------------------
        if (iteration + 1) % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{iteration + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            policy_model_raw.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(critic_head.state_dict(), os.path.join(ckpt_dir, "critic.pt"))
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
            json.dump({"iteration": iteration}, open(os.path.join(ckpt_dir, "meta.json"), "w"))
            json.dump(all_metrics, open(os.path.join(args.output_dir, "metrics.json"), "w"), indent=2)
            log(f"[PPO] 체크포인트 저장: {ckpt_dir}")

    # -----------------------------------------------------------------------
    # 7. 최종 저장
    # -----------------------------------------------------------------------
    log("[PPO] 학습 완료. 최종 모델 저장...")
    policy_model_raw.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(critic_head.state_dict(), os.path.join(args.output_dir, "critic_final.pt"))
    json.dump(all_metrics, open(os.path.join(args.output_dir, "metrics.json"), "w"), indent=2)

    if log_file:
        log_file.close()
    jsonl_file.close()
    ray.shutdown()
    log("[PPO] 완료.")


if __name__ == "__main__":
    main()
