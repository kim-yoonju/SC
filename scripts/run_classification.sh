#!/bin/bash
# Action classification 학습 스크립트
#
# 사용법:
#   bash scripts/run_classification.sh          # backbone + head 전체 학습
#   bash scripts/run_classification.sh head     # head만 학습 (backbone freeze)

set -e
# python etc/classification.py --gpu_ids 2 --epochs 10
PYTHON=/home/yoonju/miniconda3/envs/NRL/bin/python3
MODE=${1:-"full"}   # 첫 번째 인수: "full"(기본) 또는 "head"

DATA_DIR="data/rollouts"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

cd "$(dirname "$0")/.."

if [ "$MODE" = "head" ]; then
    echo "===== Classification Head Only 학습 (backbone freeze) ====="
    OUTPUT_DIR="checkpoints/action_cls_head_only"
    $PYTHON etc/classification.py \
        --model_name "$MODEL_NAME" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --freeze_backbone \
        --head_lr 1e-3 \
        --epochs 10 \
        --batch_size 128 \
        --grad_accum 512 \
        --gpu_ids 6
else
    echo "===== Classification 전체 학습 (backbone + head) ====="
    OUTPUT_DIR="checkpoints/action_cls"
    $PYTHON etc/classification.py \
        --model_name "$MODEL_NAME" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --head_lr 1e-3 \
        --backbone_lr 1e-5 \
        --epochs 10 \
        --batch_size 128 \
        --grad_accum 512 \
        --gpu_ids 6
fi

echo "완료. 모델 저장 경로: $OUTPUT_DIR/best_model"
