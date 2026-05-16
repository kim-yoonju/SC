"""
interactive_grpo.py

Interactive rollout + GRPO training (verl 없이).

구조:
  --mode rollout  : vLLM(stop_token_ids) + step_manager(base HF) 로 step-by-step 생성.
                    K rollouts per problem → group-relative advantage 계산 → queue 저장.
  --mode train    : queue에서 읽어 GRPO loss 계산 → FSDP update → checkpoint 저장.

GPU 배분 (shell이 CUDA_VISIBLE_DEVICES로 각 프로세스를 격리):
  rollout 프로세스 : CUDA_VISIBLE_DEVICES = sm_gpu,vllm_gpus
    - relative idx 0      → step_manager (base HF, subprocess로 격리)
    - relative idx 1+     → vLLM (SFT 체크포인트)
  train 프로세스   : CUDA_VISIBLE_DEVICES = train_gpus (torchrun)
    - FSDP actor + reference model

실행:
  bash scripts/run_interactive_grpo.sh
"""

import argparse
import gc
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── 설정 로드 (CUDA 초기화 전에) ─────────────────────────────────────────────

def _load_conf():
    import yaml
    return yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())

CONF         = _load_conf()
_GT_CFG      = CONF["generate_trajectory"]
_VLLM_CFG    = CONF.get("vllm", {})
_GRPO_CFG    = CONF["grpo"]["verl"]

SM_PATH      = CONF["checkpoint"]["step_manager"]
SFT_CKPT     = CONF["checkpoint"]["sft_checkpoint"]
CACHE_DIR    = CONF["checkpoint"].get("cache_dir")
ACTION_FILE  = _ROOT / CONF["prompts"]["file"]

MAX_NEW_TOKENS = _GT_CFG["max_new_tokens"]
MAX_STEPS      = _GT_CFG["max_steps"]
MAX_MODEL_LEN  = _VLLM_CFG.get("max_model_len", 32768)
ROLLOUT_N      = _GRPO_CFG["rollout_n"]          # K rollouts per problem
TEMPERATURE    = float(_GRPO_CFG["temperature"])

TOKEN_SOLVE   = CONF["model"]["token_solve"]
TOKEN_RETHINK = CONF["model"]["token_rethink"]
TOKEN_END     = CONF["model"]["token_end"]
ACTION_TOKENS = [TOKEN_SOLVE, TOKEN_RETHINK, TOKEN_END]

CLIP_EPS  = _GRPO_CFG.get("clip_ratio_low", 0.2)
KL_COEF   = _GRPO_CFG.get("kl_loss_coef", 0.04)

_SUMMARIZE_MAX_TOK = 128


# ═══════════════════════════════════════════════════════════════════════════════
# Step Manager subprocess (base HF 모델, CUDA_VISIBLE_DEVICES 격리)
# ═══════════════════════════════════════════════════════════════════════════════

def _sm_worker(sm_gpu_physical: int, req_q: mp.Queue, res_q: mp.Queue):
    """
    spawn context로 실행되는 서브프로세스.
    step_manager(base) 모델을 로드하고 요약 요청을 처리.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sm_gpu_physical)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    action_prompts = {e["name"]: e["content"] for e in json.loads(ACTION_FILE.read_text())}
    summarize_sys  = action_prompts["step_summary_system"]

    def _build_chat(text: str) -> str:
        msgs = [
            {"role": "system",  "content": summarize_sys},
            {"role": "user",    "content": f"Step:\n{text[:1200]}"},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    tok = AutoTokenizer.from_pretrained(SM_PATH, trust_remote_code=True, cache_dir=CACHE_DIR)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        SM_PATH, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True, cache_dir=CACHE_DIR,
    )
    model.eval()
    print(f"[SM worker] 로드 완료: cuda:0 (physical GPU {sm_gpu_physical})", flush=True)

    while True:
        req = req_q.get()
        if req is None:          # 종료 신호
            break
        step_text = req

        prompt = _build_chat(step_text)
        enc = tok([prompt], return_tensors="pt").to("cuda:0")
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=_SUMMARIZE_MAX_TOK,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        raw = tok.decode(out[0, input_len:], skip_special_tokens=True).strip()
        summary = raw.split("\n")[0].strip() or step_text[:300]
        res_q.put(summary)


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout 헬퍼
# ═══════════════════════════════════════════════════════════════════════════════

def _fallback_does(text: str, max_chars: int = 300) -> str:
    import re
    text = re.sub(r"^Step\s+\d+\s*[:\-]\s*", "", text.strip()).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in [". ", ".\n"]:
        pos = cut.rfind(sep)
        if pos > 50:
            return cut[:pos + 1].strip()
    return cut.strip()


def _ask_sm(req_q, res_q, step_text: str) -> str:
    """step_manager에 요약 요청 (blocking)."""
    req_q.put(step_text)
    return res_q.get()


from generate_utils import build_solve_user_msg


def _extract_step_logprobs(output) -> list[float]:
    """vLLM output에서 per-token log_probs 추출."""
    lps = []
    if output.logprobs is None:
        return lps
    for step_lp, tid in zip(output.logprobs, output.token_ids):
        entry = step_lp.get(tid)
        lps.append(entry.logprob if entry is not None else 0.0)
    return lps


def _compute_advantages(rewards: list[float]) -> list[float]:
    """GRPO group-relative advantage."""
    import statistics
    if len(rewards) <= 1:
        return [0.0] * len(rewards)
    mean = sum(rewards) / len(rewards)
    std  = statistics.stdev(rewards) + 1e-8
    return [(r - mean) / std for r in rewards]


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout 모드
# ═══════════════════════════════════════════════════════════════════════════════

def run_rollout(args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from utils_math import has_boxed, check_solved, extract_boxed
    from preprocess import build_grpo_system_prompt

    # ── run_dir: 데이터 생성 로그 (output/GRPO/{ts}/) ─────────────────────────
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log  = open(run_dir / "run.jsonl", "a", buffering=1)
    run_meta = {
        "model":       args.model_path,
        "dataset":     args.data_path,
        "num_start":   args.num_start,
        "num_end":     args.num_end,
        "rollout_n":   ROLLOUT_N,
        "max_steps":   MAX_STEPS,
        "temperature": TEMPERATURE,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))

    def _log(record: dict):
        run_log.write(json.dumps(record, ensure_ascii=False) + "\n")

    # CUDA_VISIBLE_DEVICES는 shell이 이미 설정 (sm_gpu,vllm_gpus)
    # relative 0 = sm_gpu, relative 1+ = vllm_gpus
    all_vis = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    sm_gpu_physical  = int(all_vis[0])
    vllm_gpu_physicals = [int(g) for g in all_vis[1:]] if len(all_vis) > 1 else [int(all_vis[0])]
    n_vllm_gpus      = len(vllm_gpu_physicals)

    use_sm = len(all_vis) >= 2

    req_q, res_q, sm_proc = None, None, None
    if use_sm:
        ctx    = mp.get_context("spawn")
        req_q  = ctx.Queue()
        res_q  = ctx.Queue()
        sm_proc = ctx.Process(target=_sm_worker, args=(sm_gpu_physical, req_q, res_q))
        sm_proc.start()
        # vLLM은 relative idx 1+ 만 보도록 CUDA_VISIBLE_DEVICES 재설정
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in vllm_gpu_physicals)
        print(f"[Rollout] step_manager PID={sm_proc.pid} (physical GPU {sm_gpu_physical})", flush=True)

    # ── vLLM 로드 ─────────────────────────────────────────────────────────────
    current_ckpt = args.model_path
    tok = AutoTokenizer.from_pretrained(current_ckpt, trust_remote_code=True, cache_dir=CACHE_DIR)
    tok.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    import tempfile
    tmp_tok = tempfile.mkdtemp(prefix="igrpo_tok_")
    tok.save_pretrained(tmp_tok)

    gpu_util = float(_GRPO_CFG.get("gpu_memory_utilization", 0.5))
    llm = LLM(
        model=current_ckpt, tokenizer=tmp_tok,
        dtype="bfloat16", tensor_parallel_size=n_vllm_gpus,
        trust_remote_code=True, max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_util, seed=42,
    )
    print(f"[Rollout] vLLM 로드 완료: {current_ckpt} (gpu_util={gpu_util})", flush=True)

    action_ids = [tok.convert_tokens_to_ids(t) for t in ACTION_TOKENS
                  if tok.convert_tokens_to_ids(t) != tok.unk_token_id]
    action_id_map = {tok.convert_tokens_to_ids(t): t for t in ACTION_TOKENS
                     if tok.convert_tokens_to_ids(t) != tok.unk_token_id}
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
        stop_token_ids=action_ids, logprobs=1,
    )

    system_prompt = build_grpo_system_prompt()
    queue_dir     = Path(args.queue_dir)
    ckpt_file     = queue_dir / "latest_ckpt.txt"
    queue_dir.mkdir(parents=True, exist_ok=True)

    problems = _load_problems(args.data_path, args.num_start, args.num_end)
    problems_cycle = itertools.cycle(problems)
    round_idx = 0

    print(f"[Rollout] 시작. {len(problems)}개 문제, K={ROLLOUT_N} rollouts/문제", flush=True)

    try:
        while True:
            # 새 체크포인트 확인
            if ckpt_file.exists():
                new_ckpt = ckpt_file.read_text().strip()
                if new_ckpt and new_ckpt != current_ckpt and Path(new_ckpt).exists():
                    print(f"[Rollout] 체크포인트 갱신 → {new_ckpt}", flush=True)
                    del llm; gc.collect()
                    import torch; torch.cuda.empty_cache()
                    current_ckpt = new_ckpt
                    llm = LLM(
                        model=current_ckpt, tokenizer=tmp_tok,
                        dtype="bfloat16", tensor_parallel_size=n_vllm_gpus,
                        trust_remote_code=True, max_model_len=MAX_MODEL_LEN,
                        gpu_memory_utilization=gpu_util, seed=42,
                    )

            # ── 배치: N_problems × K rollouts ─────────────────────────────────
            batch_problems = [next(problems_cycle) for _ in range(args.batch_size)]
            all_records: list[dict] = []   # 저장할 샘플들

            n_probs = len(batch_problems)
            print(f"[R{round_idx:06d}] 시작: {n_probs}문제 × K={ROLLOUT_N} = {n_probs * ROLLOUT_N}개 생성", flush=True)

            for pi, prob in enumerate(batch_problems):
                group_id   = prob.get("id", prob["problem"][:40])
                rewards    = []
                rollout_steps_list = []   # K × steps

                syms = []
                for k in range(ROLLOUT_N):
                    steps = _generate_one_trajectory(
                        prob["problem"], prob["answer"],
                        llm, tok, system_prompt,
                        action_id_map, sampling_params,
                        req_q if use_sm else None, res_q if use_sm else None,
                    )
                    # steps=None: rethink이거나 오류
                    reward = 1.0 if (steps is not None) else 0.0
                    rewards.append(reward)
                    rollout_steps_list.append(steps)
                    syms.append(f"G{k}={'✓' if reward == 1.0 else '✗'}")

                rollout_str = "  ".join(syms)
                n_solved = sum(rewards)
                print(f"  P{pi:02d}/{n_probs}  [{rollout_str}]  ({int(n_solved)}/{ROLLOUT_N} solved)", flush=True)

                advantages = _compute_advantages(rewards)

                for k, (steps, adv, rew) in enumerate(
                    zip(rollout_steps_list, advantages, rewards)
                ):
                    if steps is None:
                        continue
                    all_records.append({
                        "group_id":    group_id,
                        "rollout_idx": k,
                        "reward":      rew,
                        "advantage":   adv,
                        "steps":       steps,  # list of {prompt_token_ids, response_token_ids, old_log_probs}
                    })

            if not all_records:
                print("[Rollout] 유효 샘플 없음, 재시도", flush=True)
                continue

            # ── 큐에 저장 ────────────────────────────────────────────────────
            round_dir = queue_dir / f"round_{round_idx:06d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            with open(round_dir / "data.jsonl", "w") as f:
                for rec in all_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            (round_dir / "ready").touch()

            total_gen = n_probs * ROLLOUT_N
            avg_rew = sum(r["reward"] for r in all_records) / max(len(all_records), 1)
            print(
                f"[R{round_idx:06d}] 완료 → queue 저장 | "
                f"유효 {len(all_records)}/{total_gen}개 | avg_reward={avg_rew:.3f} | "
                f"ckpt={Path(current_ckpt).name}",
                flush=True,
            )
            _log({
                "round":      round_idx,
                "n_samples":  len(all_records),
                "n_problems": n_probs,
                "ckpt":       current_ckpt,
                "avg_reward": avg_rew,
            })
            round_idx += 1

    finally:
        run_log.close()
        if sm_proc is not None:
            req_q.put(None)
            sm_proc.join(timeout=10)
            sm_proc.terminate()


def _generate_one_trajectory(
    problem, gold_answer, llm, tok,
    system_prompt, action_id_map, sampling_params,
    req_q, res_q,
) -> list[dict] | None:
    """
    한 rollout: step-by-step 생성.
    성공이면 steps 리스트 반환, 실패(rethink/max_steps)이면 None.
    """
    from utils_math import has_boxed, check_solved, extract_boxed

    history: list[str] = []
    steps:   list[dict] = []

    for _ in range(MAX_STEPS):
        user_msg = build_solve_user_msg(problem, history)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ]
        from utils import build_chat_prompt
        prompt_str = build_chat_prompt(tok, system_prompt, user_msg)
        prompt_ids = tok.encode(prompt_str, add_special_tokens=False)

        out = llm.generate([prompt_str], sampling_params, use_tqdm=False)[0].outputs[0]
        step_text   = out.text
        stop_reason = out.stop_reason
        token_ids   = list(out.token_ids)
        old_lp      = _extract_step_logprobs(out)

        # action 결정
        if isinstance(stop_reason, int) and stop_reason in action_id_map:
            action = action_id_map[stop_reason]
        elif has_boxed(step_text):
            action = TOKEN_END
        else:
            action = TOKEN_SOLVE

        # rethink → 이 trajectory 버림
        if action == TOKEN_RETHINK:
            return None

        steps.append({
            "prompt_token_ids":   prompt_ids,
            "response_token_ids": token_ids,
            "old_log_probs":      old_lp,
        })

        if action == TOKEN_END:
            if check_solved(step_text, gold_answer):
                return steps   # 정답 ✓
            return None        # 오답 → 0 reward는 이미 처리됨

        # 다음 스텝 히스토리 업데이트
        if req_q is not None:
            does = _ask_sm(req_q, res_q, step_text)
        else:
            does = _fallback_does(step_text)
        history.append(does)

    return None  # max_steps 초과


# ═══════════════════════════════════════════════════════════════════════════════
# Training 모드
# ═══════════════════════════════════════════════════════════════════════════════

def run_train(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn.functional as F

    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    is_main = (local_rank == 0)

    def log(msg):
        if is_main:
            print(msg, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, cache_dir=CACHE_DIR)
    tok.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Actor (학습 대상)
    log(f"[Train] Actor 로드 중: {args.model_path}")
    actor_base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=CACHE_DIR,
    )
    actor_base.gradient_checkpointing_enable()
    actor = FSDP(
        actor_base,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=local_rank,
    )

    # Reference model (frozen, KL용)
    log("[Train] Reference 로드 중")
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.ref_path or args.model_path,
        torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir=CACHE_DIR,
    )
    ref = FSDP(
        ref_base,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
        device_id=local_rank,
    )
    for p in ref.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(_GRPO_CFG["lr"]),
        weight_decay=float(CONF["sft"].get("weight_decay", 0.01)),
    )

    queue_dir  = Path(args.queue_dir)
    ckpt_file  = queue_dir / "latest_ckpt.txt"
    ckpt_base  = Path(args.run_dir)
    ckpt_base.mkdir(parents=True, exist_ok=True)

    round_idx  = 0
    total_steps = 0
    log(f"[Train] 시작. queue_dir={queue_dir}")

    while total_steps < _GRPO_CFG.get("total_training_steps", 1000):
        round_dir   = queue_dir / f"round_{round_idx:06d}"
        ready_file  = round_dir / "ready"
        data_file   = round_dir / "data.jsonl"

        # 데이터 대기
        while not ready_file.exists():
            time.sleep(2)

        records = []
        with open(data_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            round_idx += 1
            continue

        log(f"[Train] Round {round_idx:06d} | {len(records)}개 샘플 | step={total_steps}")

        # ── GRPO 업데이트 ──────────────────────────────────────────────────────
        actor.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=local_rank)
        n_tokens   = 0

        for rec in records:
            advantage = rec["advantage"]
            if abs(advantage) < 1e-8:
                continue

            for step_data in rec["steps"]:
                p_ids  = torch.tensor(step_data["prompt_token_ids"],   dtype=torch.long, device=local_rank)
                r_ids  = torch.tensor(step_data["response_token_ids"], dtype=torch.long, device=local_rank)
                old_lp = torch.tensor(step_data["old_log_probs"],      dtype=torch.float32, device=local_rank)

                if r_ids.numel() == 0 or old_lp.numel() == 0:
                    continue

                input_ids = torch.cat([p_ids, r_ids]).unsqueeze(0)  # [1, seq]
                with torch.no_grad():
                    ref_logits = ref(input_ids).logits[0]  # [seq, vocab]
                actor_logits = actor(input_ids).logits[0]  # [seq, vocab]

                p_len = p_ids.shape[0]

                # response 구간의 log_probs
                new_lp = _token_log_probs(actor_logits, r_ids, offset=p_len - 1)
                ref_lp = _token_log_probs(ref_logits,   r_ids, offset=p_len - 1)

                if new_lp.numel() == 0:
                    continue

                # old_lp 길이 맞추기
                min_len = min(old_lp.shape[0], new_lp.shape[0])
                old_lp  = old_lp[:min_len]
                new_lp  = new_lp[:min_len]
                ref_lp  = ref_lp[:min_len]

                # GRPO clip loss
                ratio   = (new_lp - old_lp).exp()
                adv_t   = torch.tensor(advantage, dtype=torch.float32, device=local_rank)
                clipped = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                pg_loss = -torch.min(ratio * adv_t, clipped * adv_t).sum()

                # KL penalty
                kl_loss = (new_lp - ref_lp).sum() * KL_COEF

                total_loss = total_loss + pg_loss + kl_loss
                n_tokens  += min_len

        if n_tokens > 0:
            (total_loss / n_tokens).backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), _GRPO_CFG.get("grad_clip", 1.0))
            optimizer.step()
            log(f"[Train]   loss={total_loss.item() / n_tokens:.4f}  tokens={n_tokens}")

        total_steps += 1

        # ── 체크포인트 저장 ────────────────────────────────────────────────────
        if total_steps % _GRPO_CFG.get("save_freq", 10) == 0 and is_main:
            ckpt_path = ckpt_base / f"step_{total_steps:06d}"
            _save_fsdp_checkpoint(actor, tok, ckpt_path)
            ckpt_file.write_text(str(ckpt_path))
            log(f"[Train] 체크포인트 저장: {ckpt_path}")

        (round_dir / "done").touch()
        round_idx += 1

    dist.destroy_process_group()


def _token_log_probs(logits: "torch.Tensor", target_ids: "torch.Tensor", offset: int) -> "torch.Tensor":
    """logits[offset:offset+len(target)] 위치에서 target token들의 log_prob 반환."""
    import torch.nn.functional as F
    seq_logits = logits[offset: offset + target_ids.shape[0]]  # [resp_len, vocab]
    if seq_logits.shape[0] == 0:
        import torch
        return torch.zeros(0, device=logits.device)
    log_probs = F.log_softmax(seq_logits.float(), dim=-1)
    return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def _save_fsdp_checkpoint(model, tok, path: Path):
    """FSDP full state dict를 HuggingFace 형식으로 저장."""
    import torch
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    path.mkdir(parents=True, exist_ok=True)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()

    import torch.distributed as dist
    if dist.get_rank() == 0:
        model.module.save_pretrained(path, state_dict=state)
        tok.save_pretrained(path)


# ═══════════════════════════════════════════════════════════════════════════════
# 공통 헬퍼
# ═══════════════════════════════════════════════════════════════════════════════

def _load_problems(data_path: str, num_start=None, num_end=None) -> list[dict]:
    path = Path(data_path)
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        rows = []
        for _, r in df.iterrows():
            p = r.get("problem") or r.get("question", "")
            a = r.get("answer") or r.get("gold_answer", "")
            i = str(r.get("id", ""))
            if p and a:
                rows.append({"problem": str(p), "answer": str(a), "id": i})
        problems = rows
    else:
        problems = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d   = json.loads(line)
                p   = d.get("problem") or d.get("question", "")
                a   = d.get("answer") or d.get("gold_answer", "")
                i   = str(d.get("id", ""))
                if p and a:
                    problems.append({"problem": p, "answer": a, "id": i})

    s = num_start or 0
    e = num_end   or len(problems)
    return problems[s:e]


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        required=True, choices=["rollout", "train"])
    parser.add_argument("--queue_dir",   required=True)
    parser.add_argument("--model_path",  default=SFT_CKPT)
    parser.add_argument("--ref_path",    default=None,  help="Reference model path (default: same as model_path)")
    parser.add_argument("--run_dir",     default="checkpoints/interactive_grpo")
    parser.add_argument("--data_path",   default=_GT_CFG["base_problems"])
    parser.add_argument("--batch_size",  type=int, default=16,
                        help="Problems per round (rollouts per round = batch_size × ROLLOUT_N)")
    parser.add_argument("--num_start",   type=int, default=_GT_CFG.get("num_start"))
    parser.add_argument("--num_end",     type=int, default=_GT_CFG.get("num_end"))
    args = parser.parse_args()

    if args.mode == "rollout":
        run_rollout(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
