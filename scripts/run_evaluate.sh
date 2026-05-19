#!/bin/bash
# SFT 체크포인트 평가 스크립트 (math500)
# 사용법: bash scripts/run_evaluate.sh [--model_path <경로>]
#   예시: bash scripts/run_evaluate.sh
#         bash scripts/run_evaluate.sh --model_path output/sft_checkpoints/20260403_090000/epoch1

cd "$(dirname "$0")/.."

PYTHON=/home/yoonju/miniconda3/envs/NRL/bin/python3

# config에서 eval_gpus, evaluate checkpoint 읽기
read GPUS MODEL_PATH < <(conda run -n SC_rl python3 -c "
import yaml
with open('config/config.yaml') as f:
    cfg = yaml.safe_load(f)
gpus = cfg.get('sft', {}).get('eval_gpus', [4,5,6,7])
model = cfg.get('checkpoint', {}).get('evaluate', '')
print(','.join(str(g) for g in gpus), model)
")

echo "사용 GPU   : $GPUS"
echo "평가 모델  : $MODEL_PATH"

$PYTHON source/evaluate_step_reasoning.py \
    --gpus "$GPUS" \
    --datasets math500 \
    --model_path "$MODEL_PATH" \
    "$@"
