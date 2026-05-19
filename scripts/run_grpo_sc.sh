#!/usr/bin/env bash
# run_grpo_sc.sh — SC classification model GRPO training
#
# 사용법:
#   bash scripts/run_grpo_sc.sh
#   GPUS=4,5 bash scripts/run_grpo_sc.sh
#   GPUS=4,5 CLS_CKPT=/path/to/cls bash scripts/run_grpo_sc.sh
#   GPUS=4,5 RESUME=checkpoints/grpo_sc/20260520_xxx/step_200 bash scripts/run_grpo_sc.sh
#
# 주요 환경변수:
#   GPUS          사용할 GPU 번호 (쉼표 구분, 기본: config.grpo_sc.train_gpus)
#   INF_CKPT      추론 모델 경로 override (기본: config.checkpoint.base)
#   CLS_CKPT      분류 모델 경로 override (기본: config.checkpoint.sft_checkpoint)
#   RESUME        이어서 학습할 checkpoint 경로
#   PRM_COEF      α 값 override (0.0 = PRM API 미사용)
#   OUTCOME_COEF  β 값 override
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$ROOT/source"

# ── GPU 설정 ─────────────────────────────────────────────────────────────────
GPUS="${GPUS:-4,5}"
export CUDA_VISIBLE_DEVICES="$GPUS"

# ── config override (환경변수로 넘기면 argparse 없이 CONF를 직접 패치) ────────
# train_grpo_sc.py는 CONF에서 읽으므로, 여기서 직접 Python 변수를 override하는
# 방법 대신 임시 config 패치를 쓴다.
EXTRA_ARGS=""

if [[ -n "${INF_CKPT:-}" ]]; then
    echo "[run] INF_CKPT override: $INF_CKPT"
    EXTRA_ARGS="$EXTRA_ARGS --inf_checkpoint $INF_CKPT"
fi
if [[ -n "${CLS_CKPT:-}" ]]; then
    echo "[run] CLS_CKPT override: $CLS_CKPT"
    EXTRA_ARGS="$EXTRA_ARGS --cls_checkpoint $CLS_CKPT"
fi
if [[ -n "${RESUME:-}" ]]; then
    echo "[run] Resume from: $RESUME"
    EXTRA_ARGS="$EXTRA_ARGS --resume_from $RESUME"
fi
if [[ -n "${PRM_COEF:-}" ]]; then
    echo "[run] PRM_COEF override: $PRM_COEF"
    EXTRA_ARGS="$EXTRA_ARGS --prm_coef $PRM_COEF"
fi
if [[ -n "${OUTCOME_COEF:-}" ]]; then
    echo "[run] OUTCOME_COEF override: $OUTCOME_COEF"
    EXTRA_ARGS="$EXTRA_ARGS --outcome_coef $OUTCOME_COEF"
fi

echo "============================================"
echo " SC GRPO Training"
echo " GPUs : $GPUS"
echo " Root : $ROOT"
echo "============================================"

python "$SRC/train_grpo_sc.py" $EXTRA_ARGS
