#!/bin/bash
# REINFORCE 학습 스크립트 (GPU 2345, torchrun DDP)
# 사용법: bash run_train.sh

set -e

# --------------------------------------------------------------------------
# 설정 (환경 변수로 오버라이드 가능)
# --------------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B}"
DATA_DIR="${DATA_DIR:-data/rollouts}"
OUTPUT_DIR="${OUTPUT_DIR:-models/prm_trained}"

# 사용할 GPU (콤마 구분, torchrun 용)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
N_GPU=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

BATCH_SIZE="${BATCH_SIZE:-4}"          # GPU당 배치 크기
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"   # effective = BATCH × GRAD_ACCUM × N_GPU
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SEED="${SEED:-42}"
REWARD_THRESHOLD="${REWARD_THRESHOLD:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EFFECTIVE_BATCH=$(( BATCH_SIZE * GRAD_ACCUM_STEPS * N_GPU ))

# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------
echo "======================================"
echo " REINFORCE 학습 (multi-GPU DDP)"
echo "======================================"
echo "  베이스 모델    : ${MODEL_NAME}"
echo "  데이터 경로    : ${DATA_DIR}"
echo "  출력 경로      : ${OUTPUT_DIR}"
echo "  CUDA 디바이스  : ${CUDA_VISIBLE_DEVICES} (${N_GPU}개 GPU)"
echo "  GPU당 배치     : ${BATCH_SIZE}"
echo "  Grad accum     : ${GRAD_ACCUM_STEPS}"
echo "  Effective batch: ${EFFECTIVE_BATCH}"
echo "  에폭           : ${NUM_EPOCHS}"
echo "  학습률         : ${LEARNING_RATE}"
echo "======================================"

ARGS=(
    --model_name       "${MODEL_NAME}"
    --data_dir         "${DATA_DIR}"
    --output_dir       "${OUTPUT_DIR}"
    --batch_size       "${BATCH_SIZE}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --num_epochs       "${NUM_EPOCHS}"
    --learning_rate    "${LEARNING_RATE}"
    --max_length       "${MAX_LENGTH}"
    --warmup_ratio     "${WARMUP_RATIO}"
    --save_steps       "${SAVE_STEPS}"
    --seed             "${SEED}"
    --normalize_rewards
)
if [ -n "${REWARD_THRESHOLD}" ]; then
    ARGS+=(--reward_threshold "${REWARD_THRESHOLD}")
fi

export CUDA_VISIBLE_DEVICES

# torchrun으로 DDP 실행
torchrun \
    --nproc_per_node="${N_GPU}" \
    --master_port=29500 \
    "${SCRIPT_DIR}/scripts/train.py" \
    "${ARGS[@]}"

echo ""
echo "[완료] 학습 완료: ${OUTPUT_DIR}"
