#!/bin/bash
# Online PPO RL Training 실행 스크립트
#
# 사용법:
#   bash scripts/run_ppo_online.sh
#
# GPU 배치 (CUDA_VISIBLE_DEVICES=4,5,6,7 기준 논리 인덱스):
#   논리 GPU 0 (물리 4) → RolloutWorker 0
#   논리 GPU 1 (물리 5) → RolloutWorker 1
#   논리 GPU 2 (물리 6) → RolloutWorker 2
#   논리 GPU 3 (물리 7) → Trainer (policy + ref + critic)
#
# n_rollout_workers=3 이면 논리 GPU 0,1,2 를 workers가 쓰고,
# Trainer는 논리 GPU 3

set -e

# ---- 설정 ----
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"   # 또는 로컬 경로
DATASET="datasets/math7500.parquet"
OUTPUT_DIR="models/ppo_online"

# 수정 가능한 파라미터
N_ROLLOUT_WORKERS=3          # Rollout Worker 수 (GPU 1개씩)
PROBLEMS_PER_ROLLOUT=50      # Worker당 한 iteration에 처리할 문제 수
N_ROLLOUTS=8                 # MC rollout 횟수 (generate_data_teacher.py 기본값)
MAX_STEPS=10                 # 문제당 최대 step 수
MAX_NEW_TOKENS=512
TEMPERATURE=0.8

NUM_ITERATIONS=2          # 총 PPO iteration
PPO_EPOCHS=1                # 수집된 데이터에 대한 PPO update 횟수
BATCH_SIZE=8
GRAD_ACCUM=64
LR=1e-6
CRITIC_LR=1e-5
SAVE_EVERY=1

# ---- 실행 ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

/home/seoyoon/miniconda3/envs/NRL/bin/python scripts/ppo_online_trainer.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
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
    --kl_coef 0.1 \
    --vf_coef 0.1 \
    --entropy_coef 0.01 \
    --gamma 0.99 \
    --lam 0.95 \
    --save_every $SAVE_EVERY \
    --use_wandb \
    --wandb_project "ppo_sc_math" \
    --log_file "$OUTPUT_DIR/train.log"

echo "Done."
