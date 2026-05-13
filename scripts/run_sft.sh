#!/bin/bash
# SFT 이어서 학습
#
# 사용법:
#   bash scripts/resume_sft.sh <checkpoint_dir> <completed_epochs> <data_path>
#
# 예시 (epoch2까지 완료, epoch3부터 재개):
#   bash scripts/resume_sft.sh \
#       checkpoints/sft/20260505_130300/epoch2 \
#       2 \
#       output/SFT/20260505_130052/sft_data/sft_preprocessed.jsonl

set -e
cd "$(dirname "$0")/.."

CKPT_PATH="$1"
RESUME_EPOCH="$2"
DATA_PATH="$3"

if [[ -z "$CKPT_PATH" || -z "$RESUME_EPOCH" || -z "$DATA_PATH" ]]; then
    echo "사용법: bash scripts/resume_sft.sh <checkpoint_dir> <completed_epochs> <data_path>"
    echo "예시:   bash scripts/resume_sft.sh checkpoints/sft/20260505_130300/epoch2 2 output/SFT/20260505_130052/sft_data/sft_preprocessed.jsonl"
    exit 1
fi

RUN_DIR="$(dirname "$CKPT_PATH")"

read GPUS N_GPUS < <(conda run -n NRL python3 -c "
import yaml
with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)
sft = cfg.get('sft', {})
gpus = sft.get('train_gpus', [4, 5, 6, 7])
gpu_per_model = sft.get('gpu_per_model', 1)
n_procs = len(gpus) // gpu_per_model
print(','.join(str(g) for g in gpus), n_procs)
" 2>/dev/null)

echo "====== SFT 재개 ======"
echo "  체크포인트: $CKPT_PATH"
echo "  완료 에폭:  $RESUME_EPOCH"
echo "  데이터:     $DATA_PATH"
echo "  run_dir:    $RUN_DIR"
echo "  GPU:        $GPUS (${N_GPUS}개 프로세스)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=$GPUS \
    conda run -n NRL --no-capture-output torchrun \
    --nproc_per_node=$N_GPUS \
    --master_port=29500 \
    source/train_sft.py \
    --data_path        "$DATA_PATH" \
    --resume_checkpoint "$CKPT_PATH" \
    --resume_epoch     "$RESUME_EPOCH" \
    --run_dir          "$RUN_DIR"
