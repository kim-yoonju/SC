#!/bin/bash
# Trajectory 데이터 생성 스크립트 (모델 학습 없음)
#
# 사용법:
#   bash scripts/generate_trajectory.sh
#   bash scripts/generate_trajectory.sh --checkpoint checkpoints/ppo/iter_0003
#   bash scripts/generate_trajectory.sh --dataset datasets/deepmath_0_5k.parquet
#   bash scripts/generate_trajectory.sh --skip_file datasets/sft_ppo.jsonl
#   bash scripts/generate_trajectory.sh --wrong_only output/base_wrong/base_wrong_trajectory.jsonl
#
# GPU 설정은 config/config.yaml의 generate_trajectory.rollout_gpus 에서 관리됩니다.
# (없으면 ppo.rollout_gpus 사용)
# 체크포인트/데이터셋/스킵 파일은 위 인자로 오버라이드 가능합니다.
# --wrong_only: 해당 jsonl의 id 목록에 해당하는 문제만 처리 (default: config base_wrong_file)

set -e

cd "$(dirname "$0")/.."

# ---- CLI 인자 파싱 ----
CHECKPOINT_ARG=""
DATASET_ARG=""
SKIP_FILE_ARG=""
WRONG_ONLY_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)  CHECKPOINT_ARG="--checkpoint $2"; shift 2 ;;
        --dataset)     DATASET_ARG="--dataset $2";       shift 2 ;;
        --skip_file)   SKIP_FILE_ARG="--skip_file $2";  shift 2 ;;
        --wrong_only)  WRONG_ONLY_ARG="--wrong_only $2"; shift 2 ;;
        *) echo "알 수 없는 인자: $1"; exit 1 ;;
    esac
done

# ---- 실행 ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 source/generate_trajectory.py \
    $CHECKPOINT_ARG \
    $DATASET_ARG \
    $SKIP_FILE_ARG \
    $WRONG_ONLY_ARG

echo "Done."
