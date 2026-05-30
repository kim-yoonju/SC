#!/bin/bash
# SFT 학습 스크립트 — prm critique summary prediction
#
# 사용법:
#   bash scripts/run_sft_classification.sh --data_path <path> --gpus <gpu_ids>
#
# 예시:
'''
bash scripts/run_sft_classification.sh \
--data_path /mnt/yoonju/SC/output/sft_trajectory/20260525_030944_4000/traj_all.jsonl \
--gpus 0,1,2,3
'''

set -e
cd "$(dirname "$0")/.."

DATA_PATH=""
GPUS="0,1,2,3"
DEBUG=""
CPU_OFFLOAD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_path)      DATA_PATH="$2"; shift 2 ;;
        --gpus)           GPUS="$2";      shift 2 ;;
        --debug)          DEBUG="--debug"; shift 1 ;;
        --no-cpu-offload) CPU_OFFLOAD="--no-cpu-offload"; shift 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA_PATH" ]]; then
    echo "Error: --data_path 를 지정해주세요." >&2
    exit 1
fi

GPU_COUNT=$(echo "$GPUS" | awk -F',' '{print NF}')
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "====== SFT Classification 학습 ======"
echo "  data_path : $DATA_PATH"
echo "  GPUs      : $GPUS (${GPU_COUNT}개)"
echo "  debug     : ${DEBUG:-off}"

if [[ -n "$DEBUG" ]]; then
    CUDA_VISIBLE_DEVICES=${GPUS%%,*} \
    conda run -n SC_rl --no-capture-output python source/train_sft.py \
        --data_path "$DATA_PATH" \
        --gpus      "$GPUS" \
        $CPU_OFFLOAD \
        --debug
else
    echo "  port      : $MASTER_PORT"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$GPUS \
    conda run -n SC_rl --no-capture-output torchrun \
        --nproc_per_node=$GPU_COUNT \
        --master_port=$MASTER_PORT \
        source/train_sft.py \
        --data_path "$DATA_PATH" \
        --gpus      "$GPUS" \
        $CPU_OFFLOAD
fi
