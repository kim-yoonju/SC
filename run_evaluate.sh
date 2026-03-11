#!/bin/bash
# MATH500 평가 스크립트
# 사용법: bash run_evaluate.sh
#
# 예시:
#   TAG=before MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct bash run_evaluate.sh
#   TAG=after  MODEL_NAME=models/prm_trained                bash run_evaluate.sh

set -e

# --------------------------------------------------------------------------
# 설정 (환경 변수로 오버라이드 가능)
# --------------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B}"
DATASET="${DATASET:-datasets/math500.parquet}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-data/eval_results}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TAG="${TAG:-}"

# 평가는 단일 GPU (가장 여유 있는 GPU 하나)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------
echo "======================================"
echo " MATH500 평가"
echo "======================================"
echo "  모델      : ${MODEL_NAME}"
echo "  데이터셋  : ${DATASET} (${SPLIT})"
echo "  GPU       : ${CUDA_VISIBLE_DEVICES}"
echo "  출력 경로 : ${OUTPUT_DIR}"
echo "  태그      : ${TAG:-없음}"
echo "======================================"

ARGS=(
    --model_name     "${MODEL_NAME}"
    --dataset        "${DATASET}"
    --split          "${SPLIT}"
    --output_dir     "${OUTPUT_DIR}"
    --max_steps      "${MAX_STEPS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --temperature    "${TEMPERATURE}"
    --torch_dtype    "${TORCH_DTYPE}"
)
if [ -n "${MAX_PROBLEMS}" ]; then
    ARGS+=(--max_problems "${MAX_PROBLEMS}")
fi
if [ -n "${TAG}" ]; then
    ARGS+=(--tag "${TAG}")
fi

export CUDA_VISIBLE_DEVICES

python "${SCRIPT_DIR}/scripts/evaluate.py" "${ARGS[@]}"

echo ""
echo "[완료] 평가 완료: ${OUTPUT_DIR}"
