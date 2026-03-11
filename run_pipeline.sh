#!/bin/bash
# 전체 파이프라인: 학습 전 평가 → 데이터 생성 → 학습 → 학습 후 평가
# 사용법: bash run_pipeline.sh
#
# 반복 학습 예시 (3 라운드):
#   NUM_ROUNDS=3 bash run_pipeline.sh

set -e

# --------------------------------------------------------------------------
# 공통 설정
# --------------------------------------------------------------------------
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B}"
WORK_DIR="${WORK_DIR:-.}"           # 프로젝트 루트

DATASET_TRAIN="${DATASET_TRAIN:-datasets/math7500.parquet}"
DATASET_EVAL="${DATASET_EVAL:-datasets/math500.parquet}"
SPLIT_TRAIN="${SPLIT_TRAIN:-train}"
SPLIT_EVAL="${SPLIT_EVAL:-test}"

N_ROLLOUTS="${N_ROLLOUTS:-8}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE_GEN="${TEMPERATURE_GEN:-0.8}"   # 데이터 생성 온도
TEMPERATURE_EVAL="${TEMPERATURE_EVAL:-0.0}" # 평가 온도 (greedy)

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NUM_ROUNDS="${NUM_ROUNDS:-1}"       # 반복 학습 라운드 수

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# 헬퍼 함수
# --------------------------------------------------------------------------
log() { echo ""; echo "======================================"; echo " $1"; echo "======================================"; }

# --------------------------------------------------------------------------
# 파이프라인 실행
# --------------------------------------------------------------------------
log "파이프라인 시작"
echo "  베이스 모델 : ${BASE_MODEL}"
echo "  라운드 수   : ${NUM_ROUNDS}"
echo "  작업 경로   : ${WORK_DIR}"

CURRENT_MODEL="${BASE_MODEL}"

for ROUND in $(seq 1 "${NUM_ROUNDS}"); do
    log "라운드 ${ROUND}/${NUM_ROUNDS}"

    DATA_DIR="${WORK_DIR}/data/rollouts/round${ROUND}"
    TRAIN_OUTPUT="${WORK_DIR}/models/round${ROUND}"
    EVAL_OUTPUT="${WORK_DIR}/data/eval_results"

    mkdir -p "${DATA_DIR}" "${TRAIN_OUTPUT}" "${EVAL_OUTPUT}"

    # ------------------------------------------------------------------
    # 1) 학습 전 평가 (라운드 1에서만)
    # ------------------------------------------------------------------
    if [ "${ROUND}" -eq 1 ]; then
        log "[라운드 ${ROUND}] 학습 전 평가 (베이스라인)"
        TAG="round0_before" \
        MODEL_NAME="${CURRENT_MODEL}" \
        DATASET="${DATASET_EVAL}" \
        SPLIT="${SPLIT_EVAL}" \
        OUTPUT_DIR="${EVAL_OUTPUT}" \
        TEMPERATURE="${TEMPERATURE_EVAL}" \
        MAX_STEPS="${MAX_STEPS}" \
        MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
        TORCH_DTYPE="${TORCH_DTYPE}" \
        bash "${SCRIPT_DIR}/run_evaluate.sh"
    fi

    # ------------------------------------------------------------------
    # 2) MC 롤아웃 데이터 생성
    # ------------------------------------------------------------------
    log "[라운드 ${ROUND}] MC 롤아웃 데이터 생성"
    MODEL_NAME="${CURRENT_MODEL}" \
    OUTPUT_DIR="${DATA_DIR}" \
    DATASET="${DATASET_TRAIN}" \
    SPLIT="${SPLIT_TRAIN}" \
    N_ROLLOUTS="${N_ROLLOUTS}" \
    MAX_STEPS="${MAX_STEPS}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    TEMPERATURE="${TEMPERATURE_GEN}" \
    TORCH_DTYPE="${TORCH_DTYPE}" \
    bash "${SCRIPT_DIR}/run_generate.sh"

    # ------------------------------------------------------------------
    # 3) REINFORCE 학습
    # ------------------------------------------------------------------
    log "[라운드 ${ROUND}] 학습"
    MODEL_NAME="${CURRENT_MODEL}" \
    DATA_DIR="${DATA_DIR}" \
    OUTPUT_DIR="${TRAIN_OUTPUT}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
    NUM_EPOCHS="${NUM_EPOCHS}" \
    LEARNING_RATE="${LEARNING_RATE}" \
    TORCH_DTYPE="${TORCH_DTYPE}" \
    bash "${SCRIPT_DIR}/run_train.sh"

    CURRENT_MODEL="${TRAIN_OUTPUT}"

    # ------------------------------------------------------------------
    # 4) 학습 후 평가
    # ------------------------------------------------------------------
    log "[라운드 ${ROUND}] 학습 후 평가"
    TAG="round${ROUND}_after" \
    MODEL_NAME="${CURRENT_MODEL}" \
    DATASET="${DATASET_EVAL}" \
    SPLIT="${SPLIT_EVAL}" \
    OUTPUT_DIR="${EVAL_OUTPUT}" \
    TEMPERATURE="${TEMPERATURE_EVAL}" \
    MAX_STEPS="${MAX_STEPS}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    TORCH_DTYPE="${TORCH_DTYPE}" \
    bash "${SCRIPT_DIR}/run_evaluate.sh"

    log "[라운드 ${ROUND}] 완료"
    echo "  모델 저장: ${CURRENT_MODEL}"
done

log "전체 파이프라인 완료"
echo "  최종 모델: ${CURRENT_MODEL}"
echo "  평가 결과: ${WORK_DIR}/data/eval_results/"

# --------------------------------------------------------------------------
# 결과 요약 출력
# --------------------------------------------------------------------------
echo ""
echo "=== 정확도 요약 ==="
for f in "${WORK_DIR}/data/eval_results"/summary_*.json; do
    if [ -f "${f}" ]; then
        tag=$(basename "${f}" .json | sed 's/summary_//')
        acc=$(python3 -c "import json; d=json.load(open('${f}')); print(f\"{d['accuracy']:.3f}\")" 2>/dev/null || echo "N/A")
        echo "  ${tag}: ${acc}"
    fi
done
