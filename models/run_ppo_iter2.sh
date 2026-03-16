#!/bin/bash
# Online PPO RL Training - iteration 2 재시작 (checkpoint-1에서)
#
# 사용법:
#   bash models/run_ppo_iter2.sh
#

set -e

# ---- 설정 ----
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATASET="datasets/math7500.parquet"
OUTPUT_DIR="models/ppo_online"
RESUME_FROM="models/ppo_online/checkpoint-1"

# 수정 가능한 파라미터
N_ROLLOUT_WORKERS=3
PROBLEMS_PER_ROLLOUT=50
N_ROLLOUTS=8
MAX_STEPS=10
MAX_NEW_TOKENS=512
TEMPERATURE=0.8

NUM_ITERATIONS=2
PPO_EPOCHS=1
BATCH_SIZE=8
GRAD_ACCUM=64
LR=1e-6
CRITIC_LR=1e-5
SAVE_EVERY=1

# ---- 실행 ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

/home/agi-admin/miniconda3/envs/seoyoon_bs/bin/python3 scripts/ppo_online_trainer.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --resume_from "$RESUME_FROM" \
    --n_rollout_workers $N_ROLLOUT_WORKERS \
    --problems_per_rollout $PROBLEMS_PER_ROLLOUT \
    --n_rollouts $N_ROLLOUTS \
    --max_steps $MAX_STEPS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --num_iterations $NUM_ITERATIONS \
    --ppo_epochs $PPO_EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --critic_lr $CRITIC_LR \
    --clip_eps 0.2 \
    --kl_coef 0.01 \
    --vf_coef 0.1 \
    --entropy_coef 0 \
    --gamma 0.99 \
    --lam 0.95 \
    --use_cached_rollout \
    --cached_rollout_skip 307 333 311 \
    --save_every $SAVE_EVERY \
    --use_wandb \
    --wandb_project "ppo_sc_math" \
    --log_file "$OUTPUT_DIR/train_iter2.log"

echo "Done."
