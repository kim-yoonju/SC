#!/bin/bash
# SFT 학습 실행 (config/config.yaml의 sft.train_gpus 사용)
# 사용법: bash scripts/run_sft.sh [추가 인자]
#   예시: bash scripts/run_sft.sh --num_epochs 3 --lr 2e-5

cd "$(dirname "$0")/.."

# config에서 train_gpus 읽기
GPUS=$(conda run -n NRL python3 -c "
import yaml
with open('config/config.yaml') as f:
    cfg = yaml.safe_load(f)
gpus = cfg.get('sft', {}).get('train_gpus', [4,5,6,7])
print(','.join(str(g) for g in gpus))
")
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

echo "사용 GPU: $GPUS  (${N_GPUS}개)"

# 학습 전 input/output 샘플 미리보기
echo "========== SFT 샘플 미리보기 =========="
conda run -n NRL python source/train_sft.py --preview "$@"
echo "========================================"
echo ""

# 학습 시작
CUDA_VISIBLE_DEVICES=$GPUS conda run -n NRL --no-capture-output torchrun \
    --nproc_per_node=$N_GPUS \
    --master_port=29500 \
    source/train_sft.py \
    "$@"
