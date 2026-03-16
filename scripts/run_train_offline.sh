#!/bin/bash
# Offline REINFORCE + Classification Head 학습 스크립트
#
# GPU 설정: 아래 GPUS 변수를 수정하세요.
# GPU 수에 따라 accelerate가 자동으로 병렬 학습합니다.

set -e

# ---- GPU 설정 (첫 번째 줄) ----
GPUS="4,5"

# ---- 설정 ----
ACCELERATE=/home/yoonju/miniconda3/envs/NRL/bin/accelerate

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="checkpoints/offline_reinforce"

DATA_FILES=(
    "/mnt/yoonju/SC/data/rollouts/online_ppo_math7500_worker0.jsonl"
    "/mnt/yoonju/SC/data/rollouts/online_ppo_math7500_worker1.jsonl"
    "/mnt/yoonju/SC/data/rollouts/online_ppo_math7500_worker2.jsonl"
)

CLS_HEAD_PATH="/mnt/yoonju/SC/checkpoints/action_cls/best_model/classifier_head.pt"
CLS_COEF=0.1

BATCH_SIZE=4
GRAD_ACCUM=16
NUM_EPOCHS=1
LR=1e-5
MAX_LENGTH=2048
WARMUP_RATIO=0.1

# ---- GPU 수 자동 계산 ----
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
echo "사용 GPU: $GPUS (${NUM_GPUS}개)"

# ---- 실행 ----
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

$ACCELERATE launch \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    source/train_offline_trainer.py \
    --model_name "$MODEL_NAME" \
    --data_files "${DATA_FILES[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --cls_head_path "$CLS_HEAD_PATH" \
    --cls_coef $CLS_COEF \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LR \
    --max_length $MAX_LENGTH \
    --warmup_ratio $WARMUP_RATIO

echo "완료. 모델 저장 경로: $OUTPUT_DIR"
