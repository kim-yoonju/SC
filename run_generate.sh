#!/bin/bash
# MC 롤아웃 데이터 생성 스크립트
# GPU 2345를 사용해 데이터를 4등분하여 병렬 생성한다.
# 사용법: bash run_generate.sh

set -e

# --------------------------------------------------------------------------
# 설정 (환경 변수로 오버라이드 가능)
# --------------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-data/rollouts}"

# ★ 학습 데이터셋 (로컬 .parquet 경로 또는 HuggingFace 이름)
#   예) datasets/math7500.parquet  /  zwhe99/DeepMath-103K
DATASET="${DATASET:-datasets/math7500.parquet}"
SPLIT="${SPLIT:-train}"

# ★ 전체 처리할 문제 수 (데이터셋 크기에 맞게 조정)
#   math7500 → 7500 / DeepMath-103K → 103000
TOTAL="${TOTAL:-7500}"

# 사용할 GPU ID 목록 (공백 구분)
GPUS="${GPUS:-4 5 6 7}"

N_ROLLOUTS="${N_ROLLOUTS:-8}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
# 메인 스텝 생성 시 동시에 처리할 문제 수 (VRAM 여유에 따라 조정)
# MC 롤아웃은 항상 N_ROLLOUTS 크기의 배치로 실행됨
PROBLEM_BATCH_SIZE="${PROBLEM_BATCH_SIZE:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# GPU 수에 맞게 범위 분할
# --------------------------------------------------------------------------
read -ra GPU_LIST <<< "${GPUS}"
N_GPU=${#GPU_LIST[@]}
CHUNK=$(( (TOTAL + N_GPU - 1) / N_GPU ))  # ceil division

mkdir -p "${OUTPUT_DIR}"

echo "======================================"
echo " MC 롤아웃 데이터 생성 (병렬)"
echo "======================================"
echo "  모델      : ${MODEL_NAME}"
echo "  데이터셋  : ${DATASET} (${SPLIT})"
echo "  사용 GPU  : ${GPUS}"
echo "  총 문제   : ${TOTAL}  (GPU당 ~${CHUNK}개)"
echo "  롤아웃 수 : ${N_ROLLOUTS} (배치로 병렬 실행)"
echo "  문제 배치 : ${PROBLEM_BATCH_SIZE}"
echo "  출력 경로 : ${OUTPUT_DIR}"
echo "======================================"

PIDS=()

for i in "${!GPU_LIST[@]}"; do
    GPU_ID="${GPU_LIST[$i]}"
    START=$(( i * CHUNK ))
    END=$(( START + CHUNK ))
    if [ "${END}" -gt "${TOTAL}" ]; then
        END="${TOTAL}"
    fi
    if [ "${START}" -ge "${TOTAL}" ]; then
        break
    fi

    # PID 락 파일: 같은 범위를 이미 처리 중인 프로세스가 있으면 건너뜀
    LOCK_FILE="${OUTPUT_DIR}/.lock_${START}_${END}"
    if [ -f "${LOCK_FILE}" ]; then
        EXISTING_PID=$(cat "${LOCK_FILE}")
        if kill -0 "${EXISTING_PID}" 2>/dev/null; then
            echo "  [스킵] GPU ${GPU_ID} [${START}, ${END}): PID ${EXISTING_PID} 이미 실행 중"
            continue
        else
            echo "  [재시작] GPU ${GPU_ID}: 이전 락 파일 제거 (PID ${EXISTING_PID} 종료됨)"
            rm -f "${LOCK_FILE}"
        fi
    fi

    LOG_FILE="${OUTPUT_DIR}/generate_gpu${GPU_ID}.log"
    echo "  GPU ${GPU_ID}: 문제 [${START}, ${END}) → ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${SCRIPT_DIR}/scripts/generate_data.py" \
        --model_name    "${MODEL_NAME}" \
        --output_dir    "${OUTPUT_DIR}" \
        --dataset       "${DATASET}" \
        --split         "${SPLIT}" \
        --start_idx     "${START}" \
        --end_idx       "${END}" \
        --n_rollouts    "${N_ROLLOUTS}" \
        --max_steps     "${MAX_STEPS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --temperature        "${TEMPERATURE}" \
        --torch_dtype        "${TORCH_DTYPE}" \
        --problem_batch_size "${PROBLEM_BATCH_SIZE}" \
        > "${LOG_FILE}" 2>&1 &

    NEW_PID=$!
    echo "${NEW_PID}" > "${LOCK_FILE}"
    PIDS+=("${NEW_PID}")
done

# 모든 프로세스 완료 대기
echo ""
echo "모든 GPU 프로세스 실행 중... (PID: ${PIDS[*]})"
FAILED=0
PID_IDX=0
for i in "${!GPU_LIST[@]}"; do
    GPU_ID="${GPU_LIST[$i]}"
    START=$(( i * CHUNK ))
    END=$(( START + CHUNK ))
    [ "${END}" -gt "${TOTAL}" ] && END="${TOTAL}"
    [ "${START}" -ge "${TOTAL}" ] && continue
    LOCK_FILE="${OUTPUT_DIR}/.lock_${START}_${END}"

    PID="${PIDS[$PID_IDX]}"
    PID_IDX=$(( PID_IDX + 1 ))

    if wait "${PID}"; then
        echo "  [완료] GPU ${GPU_ID} (PID ${PID})"
    else
        echo "  [실패] GPU ${GPU_ID} (PID ${PID}) — 로그: ${OUTPUT_DIR}/generate_gpu${GPU_ID}.log"
        FAILED=$(( FAILED + 1 ))
    fi
    rm -f "${LOCK_FILE}"
done

echo ""
if [ "${FAILED}" -eq 0 ]; then
    echo "[완료] 전체 데이터 생성 완료: ${OUTPUT_DIR}"
else
    echo "[경고] ${FAILED}개 프로세스 실패. 로그를 확인하세요."
    exit 1
fi
