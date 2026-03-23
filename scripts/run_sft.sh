#!/bin/bash
# GPU 4,5,6에서 SFT 학습 실행
# 사용법: bash scripts/run_sft.sh [추가 인자]
#   예시: bash scripts/run_sft.sh --num_epochs 3 --lr 2e-5

cd "$(dirname "$0")/.."

# 학습 전 input/output 샘플 출력
echo "========== SFT 샘플 미리보기 =========="
conda run -n NRL python source/train_sft.py --preview "$@"
echo "========================================"
echo ""

# 학습 시작
CUDA_VISIBLE_DEVICES=4,5,6 conda run -n NRL --no-capture-output torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    source/train_sft.py \
    "$@"
