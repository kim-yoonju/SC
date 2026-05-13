#!/bin/bash
# MC 롤아웃 데이터 생성 스크립트
#
# 사용법:
#   bash scripts/run_generate_data.sh
#
# USE_TEACHER=true 이면 generator의 <correct> 시도가 실패(temp_reward=0)했을 때
# GPT teacher가 대신 correction step을 생성한다.

set -e

PYTHON=/home/agi-admin/miniconda3/envs/seoyoon_bs/bin/python3

# ---- 설정 ----
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATASET="datasets/math7500.parquet"
OUTPUT_DIR="data/rollouts"

N_ROLLOUTS=8
MAX_STEPS=10
MAX_NEW_TOKENS=512
TEMPERATURE=0.8
PROBLEM_BATCH_SIZE=4

# teacher injection 사용 여부 (true / false)
USE_TEACHER=false

# 분산 처리: 범위를 나눠서 여러 프로세스로 실행할 경우 아래 변수를 수정
START_IDX=0
END_IDX=7500

# ---- 실행 ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

TEACHER_FLAG=""
if [ "$USE_TEACHER" = "true" ]; then
    TEACHER_FLAG="--use_teacher"
    echo "[generate] Teacher injection 활성화"
else
    echo "[generate] Teacher injection 비활성화"
fi

CUDA_VISIBLE_DEVICES=4 $PYTHON scripts/generate_data.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --n_rollouts $N_ROLLOUTS \
    --max_steps $MAX_STEPS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --problem_batch_size $PROBLEM_BATCH_SIZE \
    $TEACHER_FLAG

echo "Done."
