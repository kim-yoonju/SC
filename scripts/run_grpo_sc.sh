#!/usr/bin/env bash
# run_grpo_sc.sh — SC classification model GRPO training
#
# 사용법:
#   bash scripts/run_grpo_sc.sh
#   bash scripts/run_grpo_sc.sh --gpus 0,1,2,3
#   bash scripts/run_grpo_sc.sh --gpus 0,1 --classification_model /path/to/cls
#   bash scripts/run_grpo_sc.sh --gpus 0,1 --resume checkpoints/grpo_sc/20260520_xxx/step_200
#
# 주요 인자:
#   --gpus                사용할 GPU 번호 (쉼표 구분, 기본: config.grpo_sc.train_gpus)
#   --inference_model     추론 모델 경로 override (기본: config.checkpoint.base)
#   --classification_model 분류 모델 경로 override (기본: config.checkpoint.sft_checkpoint)
#   --resume              이어서 학습할 checkpoint 경로
#   --prm_coef            α 값 override (0.0 = PRM API 미사용)
#   --outcome_coef        β 값 override (outcome reward 가중치)
#   --min_records         학습 1회당 최소 record 수 (기본: config.grpo_sc.min_records_per_update)
#   --inf_gpu_count       inference 모델에 할당할 GPU 수 (기본: TRAIN_GPUS 절반)
#   --problem_batch_size  동시 처리 문제 수 (기본: config.grpo_sc.problem_batch_size)
#   --max_gen_batch_size  GPU 최대 배치 크기 (기본: config.grpo_sc.max_gen_batch_size)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$ROOT/source"

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
GPUS_ARG=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)                GPUS_ARG="$2";  EXTRA_ARGS="$EXTRA_ARGS --gpus $2";              shift 2 ;;
        --inference_model)                     EXTRA_ARGS="$EXTRA_ARGS --inf_checkpoint $2";    shift 2 ;;
        --classification_model)                EXTRA_ARGS="$EXTRA_ARGS --cls_checkpoint $2";    shift 2 ;;
        --resume)                              EXTRA_ARGS="$EXTRA_ARGS --resume_from $2";        shift 2 ;;
        --prm_coef)                            EXTRA_ARGS="$EXTRA_ARGS --prm_coef $2";              shift 2 ;;
        --outcome_coef)                        EXTRA_ARGS="$EXTRA_ARGS --outcome_coef $2";        shift 2 ;;
        --min_records)                         EXTRA_ARGS="$EXTRA_ARGS --min_records $2";          shift 2 ;;
        --inf_gpu_count)                       EXTRA_ARGS="$EXTRA_ARGS --inf_gpu_count $2";       shift 2 ;;
        --problem_batch_size)                  EXTRA_ARGS="$EXTRA_ARGS --problem_batch_size $2"; shift 2 ;;
        --max_gen_batch_size)                  EXTRA_ARGS="$EXTRA_ARGS --max_gen_batch_size $2"; shift 2 ;;
        --debug)                               EXTRA_ARGS="$EXTRA_ARGS --debug";                  shift 1 ;;
        *) echo "[run] unknown argument: $1" >&2; exit 1 ;;
    esac
done

GPU_DISPLAY="${GPUS_ARG:-from config (grpo_sc.train_gpus)}"

echo "============================================"
echo " SC GRPO Training"
echo " GPUs : $GPU_DISPLAY"
echo " Root : $ROOT"
echo "============================================"

python "$SRC/train_grpo_sc.py" $EXTRA_ARGS
