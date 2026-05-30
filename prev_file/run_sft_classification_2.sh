#!/bin/bash
# Classification SFT 학습 스크립트
#
# 사용법:
#   bash scripts/run_sft_classification.sh \
#     --data_path /mnt/yoonju/SC/output/sft_trajectory/traj_4000_2000_deep.jsonl \
#     --rubric_weights \
#     --gpus 0,1,2,3 --debug
#
# 주요 옵션:
#   --data_path <path>   raw trajectory 또는 preprocessed jsonl 경로
#   --gpus <ids>         사용할 GPU (예: 0,1,2,3)
#   --resume <ckpt>      체크포인트에서 재개
#   --action_weights     Next action 역빈도 가중치
#   --rubric_weights     Fail rubrics focal loss
#   --debug [N]          학습 없이 전처리 후 N번째 샘플 출력 (N 생략 시 0번째)
#
# 예시:
#   bash scripts/run_sft_classification.sh --gpus 0,1,2,3 --data_path output/.../traj.jsonl
#   bash scripts/run_sft_classification.sh --gpus 0,1,2,3 --data_path output/.../traj.jsonl --debug
#   bash scripts/run_sft_classification.sh --gpus 0,1,2,3 --data_path output/.../traj.jsonl --debug 5

set -e 
cd "$(dirname "$0")/.."

RESUME_CKPT=""
DATA_PATH=""
DEBUG_N=""
GPUS_OVERRIDE=""
USE_SUMMARY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_CKPT="$2"; shift 2 ;;
        --data_path)
            DATA_PATH="$2"; shift 2 ;;
        --gpus)
            GPUS_OVERRIDE="$2"; shift 2 ;;
        --use_summary)
            USE_SUMMARY="1"; shift 1 ;;
        --debug)
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                DEBUG_N="$2"; shift 2
            else
                DEBUG_N="auto"; shift 1
            fi ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "사용법: bash scripts/run_sft_classification.sh [--data_path <path>] [--gpus <gpu_ids>] [--use_summary] [--resume <ckpt>] [--debug [N]]"
            exit 1 ;;
    esac
done

read GPUS N_GPUS GPU_PER_MODEL < <(conda run -n SC_rl python3 -c "
import yaml
with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)
sft  = cfg.get('sft', {})
gpus = sft.get('train_gpus', [4, 5, 6, 7])
gpu_per_model = sft.get('gpu_per_model', 1)
n_procs = len(gpus) // gpu_per_model
print(','.join(str(g) for g in gpus), n_procs, gpu_per_model)
" 2>/dev/null)

if [[ -n "$GPUS_OVERRIDE" ]]; then
    GPUS="$GPUS_OVERRIDE"
    GPU_COUNT=$(echo "$GPUS" | awk -F',' '{print NF}')
    N_GPUS=$((GPU_COUNT / GPU_PER_MODEL))
fi

# data_path 미지정이면 config에서 읽음
if [[ -z "$DATA_PATH" ]]; then
    DATA_PATH=$(conda run -n SC_rl python3 -c "
import yaml
with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('data_path', {}).get('sft_data', ''))
" 2>/dev/null)
fi

# 항상 전처리 실행 (기존 파일 덮어씌움)
RAW_DATA_PATH="$DATA_PATH"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "오류: 데이터 파일이 존재하지 않습니다: $DATA_PATH"
    exit 1
fi

IS_RAW=$(conda run -n SC_rl python3 -c "
import json
line = open('$DATA_PATH').readline()
item = json.loads(line)
print('1' if 'steps' in item and 'input' not in item else '0')
" 2>/dev/null)

if [[ "$IS_RAW" != "1" ]]; then
    echo "오류: DATA_PATH가 raw trajectory 파일이 아닙니다 (이미 전처리된 파일). raw 파일 경로를 지정해주세요."
    exit 1
fi

TRAJ_DIR=$(dirname "$DATA_PATH")
if [[ -n "$USE_SUMMARY" ]]; then
    PREPROCESSED_PATH="$TRAJ_DIR/cls_preprocessed_summary.jsonl"
else
    PREPROCESSED_PATH="$TRAJ_DIR/cls_preprocessed.jsonl"
fi
echo "====== Classification 전처리 (기존 파일 덮어씌움) ======"
echo "  입력:  $DATA_PATH"
echo "  출력:  $PREPROCESSED_PATH"
PREPROCESS_ARGS=(
    --data_path         "$DATA_PATH"
    --output_path       "$PREPROCESSED_PATH"
    --mode              classification
    --no-balance
    --no-filter
    --max_length        8192
    --max_target_length 4096
)
[[ -n "$USE_SUMMARY" ]] && PREPROCESS_ARGS+=(--use_summary)
conda run -n SC_rl python3 source/preprocess.py "${PREPROCESS_ARGS[@]}"
DATA_PATH="$PREPROCESSED_PATH"

EXTRA_ARGS=(--data_path "$DATA_PATH")

if [[ -n "$RESUME_CKPT" ]]; then
    EPOCH_NAME=$(basename "$RESUME_CKPT")
    if [[ "$EPOCH_NAME" =~ ^epoch([0-9]+)$ ]]; then
        RESUME_EPOCH="${BASH_REMATCH[1]}"
    else
        echo "오류: 체크포인트 폴더명이 'epochN' 형식이어야 합니다 (예: epoch2)"
        echo "  입력값: $EPOCH_NAME"
        exit 1
    fi
    RUN_DIR="$(dirname "$RESUME_CKPT")"

    echo "====== SFT 재개 ======"
    echo "  체크포인트: $RESUME_CKPT"
    echo "  완료 에폭:  $RESUME_EPOCH"
    echo "  데이터:     $DATA_PATH"
    echo "  run_dir:    $RUN_DIR"
    echo "  GPU:        $GPUS (${N_GPUS}개 프로세스)"
    [[ -n "$USE_SUMMARY" ]] && echo "  use_summary: ON (prm_critique_summary 사용)"

    EXTRA_ARGS+=(
        --resume_checkpoint "$RESUME_CKPT"
        --resume_epoch      "$RESUME_EPOCH"
        --run_dir           "$RUN_DIR"
    )
elif [[ -z "$DEBUG_N" ]]; then
    echo "====== SFT 시작 ======"
    echo "  데이터: $DATA_PATH"
    echo "  GPU:    $GPUS (${N_GPUS}개 프로세스)"
    [[ -n "$USE_SUMMARY" ]] && echo "  use_summary: ON (prm_critique_summary 사용)"
fi

if [[ -n "$DEBUG_N" ]]; then
    SAMPLE_IDX=0
    [[ "$DEBUG_N" =~ ^[0-9]+$ ]] && SAMPLE_IDX="$DEBUG_N"
    echo "====== DEBUG 모드 (샘플 $SAMPLE_IDX, 학습 안 함) ======"
    echo "  데이터: $DATA_PATH"
    echo "  GPU:    $GPUS"
    CUDA_VISIBLE_DEVICES=$GPUS \
    conda run -n SC_rl python3 source/train_sft.py \
        --data_path "$DATA_PATH" \
        --debug "$SAMPLE_IDX"
else
    MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$GPUS \
        conda run -n SC_rl --no-capture-output torchrun \
        --nproc_per_node=$N_GPUS \
        --master_port=$MASTER_PORT \
        source/train_sft.py \
        "${EXTRA_ARGS[@]}"
fi