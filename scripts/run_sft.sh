#!/bin/bash
# SFT 학습 실행 (config/config.yaml의 sft.train_gpus 사용)
# 사용법: bash scripts/run_sft.sh [추가 인자]
#   예시: bash scripts/run_sft.sh --num_epochs 3 --lr 2e-5
#         bash scripts/run_sft.sh --preview [--data_path ...]

cd "$(dirname "$0")/.."

# --preview 플래그 분리 (torchrun에 전달하지 않음)
PREVIEW=false
TRAIN_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--preview" ]; then
        PREVIEW=true
    else
        TRAIN_ARGS+=("$arg")
    fi
done

# config에서 train_gpus, gpu_per_model 읽기
read GPUS N_GPUS < <(conda run -n NRL python3 -c "
import yaml
with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)
sft = cfg.get('sft', {})
gpus = sft.get('train_gpus', [4,5,6,7])
gpu_per_model = sft.get('gpu_per_model', 1)
n_procs = len(gpus) // gpu_per_model
print(','.join(str(g) for g in gpus), n_procs)
" 2>/dev/null)

echo "사용 GPU: $GPUS  (${N_GPUS}개)"

# 미리보기 (--preview 단독 실행 시 학습 없이 종료)
echo "========== SFT 샘플 미리보기 =========="
conda run -n NRL python source/train_sft.py --preview "${TRAIN_ARGS[@]}"
echo "========================================"

if $PREVIEW; then
    exit 0
fi

echo ""

# 학습 시작
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS conda run -n NRL --no-capture-output torchrun \
    --nproc_per_node=$N_GPUS \
    --master_port=29500 \
    source/train_sft.py \
    "${TRAIN_ARGS[@]}"
