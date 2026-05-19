#!/bin/bash
# SFT 학습 스크립트
#
# 사용법:
#   bash scripts/run_sft.sh [--resume <checkpoint_dir>] [--data_path <path>] [--debug [N]]
#
# 예시:
#   bash scripts/run_sft.sh                                              # 처음부터 학습 (config의 sft_data 사용)
#   bash scripts/run_sft.sh --resume checkpoints/sft/20260505/epoch2    # epoch2 이후 재개
#   bash scripts/run_sft.sh --data_path output/sft_trajectory/xxx/traj_all.jsonl
#   bash scripts/run_sft.sh --debug        # 0번째 샘플 출력
#   bash scripts/run_sft.sh --debug 3      # 3번째 샘플 출력
#
# --data_path에 raw trajectory(steps 키 있는 파일)를 넘기면 자동으로 전처리 후 학습합니다.

set -e
cd "$(dirname "$0")/.."

# ── 데이터 비율 설정 ──────────────────────────────────────────────────────────
SOLVE_RATIO=45
RETHINK_RATIO=45
END_RATIO=10
# ─────────────────────────────────────────────────────────────────────────────

RESUME_CKPT=""
DATA_PATH=""
DEBUG_N=""
GPUS_OVERRIDE=""
ACTION_WEIGHTS=""
RUBRIC_WEIGHTS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_CKPT="$2"; shift 2 ;;
        --data_path)
            DATA_PATH="$2"; shift 2 ;;
        --gpus)
            GPUS_OVERRIDE="$2"; shift 2 ;;
        --action_weights)
            ACTION_WEIGHTS="1"; shift 1 ;;
        --rubric_weights)
            RUBRIC_WEIGHTS="1"; shift 1 ;;
        --debug)
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                DEBUG_N="$2"; shift 2
            else
                DEBUG_N="auto"; shift 1
            fi ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "사용법: bash scripts/run_sft.sh [--resume <checkpoint_dir>] [--data_path <path>] [--gpus <gpu_ids>] [--action_weights] [--rubric_weights] [--debug [N]]"
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

# raw trajectory 파일이면 자동 전처리
RAW_DATA_PATH="$DATA_PATH"   # debug용으로 원본 경로 보존

IS_RAW=$(conda run -n SC_rl python3 -c "
import json
line = open('$DATA_PATH').readline()
item = json.loads(line)
print('1' if 'steps' in item and 'input' not in item else '0')
" 2>/dev/null)

if [[ "$IS_RAW" == "1" ]]; then
    TRAJ_DIR=$(dirname "$DATA_PATH")
    PREPROCESSED_PATH="$TRAJ_DIR/sft_preprocessed.jsonl"
    echo "====== 전처리 ======"
    echo "  입력:  $DATA_PATH"
    echo "  출력:  $PREPROCESSED_PATH"
    conda run -n SC_rl python3 source/preprocess.py \
        --data_path    "$DATA_PATH" \
        --output_path  "$PREPROCESSED_PATH" \
        --solve_ratio  "$SOLVE_RATIO" \
        --rethink_ratio "$RETHINK_RATIO" \
        --end_ratio    "$END_RATIO" \
        --no-filter \
        --no-balance
    DATA_PATH="$PREPROCESSED_PATH"
fi

EXTRA_ARGS=(--data_path "$DATA_PATH")
if [[ -n "$ACTION_WEIGHTS" ]]; then
    EXTRA_ARGS+=(--action_weights)
fi
if [[ -n "$RUBRIC_WEIGHTS" ]]; then
    EXTRA_ARGS+=(--rubric_weights)
fi

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
    echo "  체크포인트:  $RESUME_CKPT"
    echo "  완료 에폭:   $RESUME_EPOCH"
    echo "  데이터:      $DATA_PATH"
    echo "  run_dir:     $RUN_DIR"
    echo "  GPU:         $GPUS (${N_GPUS}개 프로세스)"
    [[ -n "$ACTION_WEIGHTS" ]] && echo "  action_weights: ON (Next action 역빈도 가중치)"
    [[ -n "$RUBRIC_WEIGHTS" ]] && echo "  rubric_weights: ON (Fail rubrics 역빈도 가중치)"

    EXTRA_ARGS+=(
        --resume_checkpoint "$RESUME_CKPT"
        --resume_epoch      "$RESUME_EPOCH"
        --run_dir           "$RUN_DIR"
    )
else
    echo "====== SFT 시작 ======"
    echo "  데이터: $DATA_PATH"
    echo "  GPU:    $GPUS (${N_GPUS}개 프로세스)"
    [[ -n "$ACTION_WEIGHTS" ]] && echo "  action_weights: ON (Next action 역빈도 가중치)"
    [[ -n "$RUBRIC_WEIGHTS" ]] && echo "  rubric_weights: ON (Fail rubrics 역빈도 가중치)"
fi

if [[ -n "$DEBUG_N" ]]; then
    if [[ "$DEBUG_N" == "auto" ]]; then
        echo "====== DEBUG 모드 (fail_rubrics 자동 샘플링, preprocessed: $DATA_PATH) ======"
        conda run -n SC_rl python3 source/train_sft.py \
            --data_path "$DATA_PATH" \
            --debug
    else
        echo "====== DEBUG 모드 (샘플 $DEBUG_N) ======"
        conda run -n SC_rl python3 source/train_sft.py \
            "${EXTRA_ARGS[@]}" \
            --debug "$DEBUG_N"
    fi
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