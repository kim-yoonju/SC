#!/bin/bash
# Online PPO 학습 + Classifier Head 재학습 통합 파이프라인
#
# 사용법:
#   bash models/run_train_online.sh
#
# GPU 배치:
#   GPU 4,5,6 → RolloutWorker 0,1,2 (각 1개)
#   GPU 7     → PPO Trainer (policy + ref + critic)

set -e

PYTHON=/home/agi-admin/miniconda3/envs/seoyoon_bs/bin/python3

# ---- 모델 / 데이터 ----
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATASET="datasets/math7500.parquet"
OUTPUT_DIR="models/ppo_online"
CLASSIFIER_HEAD="checkpoints/action_cls/best_model/classifier_head.pt"
CLS_OUTPUT_DIR="checkpoints/action_cls"

# ---- Rollout 설정 ----
N_ROLLOUT_WORKERS=3
PROBLEMS_PER_ROLLOUT=64
N_ROLLOUTS=8
MAX_STEPS=10
MAX_NEW_TOKENS=512
TEMPERATURE=0.8

# ---- PPO 학습 설정 ----
NUM_ITERATIONS=2
PPO_EPOCHS=1
BATCH_SIZE=8
GRAD_ACCUM=64
LR=1e-6
CRITIC_LR=1e-5
SAVE_EVERY=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

echo "===== [1/2] Online PPO 학습 시작 ====="
CUDA_VISIBLE_DEVICES=4,5,6,7 $PYTHON scripts/ppo_online_trainer.py \
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
    --kl_coef 0.01 \
    --vf_coef 0.1 \
    --entropy_coef 0.01 \
    --gamma 0.99 \
    --lam 0.95 \
    --save_every $SAVE_EVERY \
    --classifier_head_path "$CLASSIFIER_HEAD" \
    --use_wandb \
    --wandb_project "ppo_sc_math" \
    --log_file "$OUTPUT_DIR/train.log"

echo ""
echo "===== [2/2] Classifier Head 재학습 (backbone freeze, 새 rollout 데이터) ====="
CUDA_VISIBLE_DEVICES=4,5 $PYTHON scripts/classification.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --data_dir "$OUTPUT_DIR/rollouts" \
    --output_dir "$CLS_OUTPUT_DIR" \
    --freeze_backbone \
    --head_lr 1e-3 \
    --epochs 5 \
    --batch_size 2 \
    --grad_accum 8 \
    --gpu_ids 4 5
echo "Classifier head 업데이트 완료: $CLASSIFIER_HEAD"

echo ""
echo "Done. 다음 iteration은 업데이트된 classifier head로 실행됩니다."
