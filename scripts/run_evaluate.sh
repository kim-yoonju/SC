#!/bin/bash
# MATH500 평가 스크립트 (classifier-guided, batched, multi-worker)
#
# 사용법:
#   bash scripts/run_evaluate.sh
#
# GPUS          : 사용할 GPU 번호 (전체)
# GPUS_PER_MODEL: 모델 하나에 할당할 GPU 수 (model parallelism)
# → NUM_WORKERS = len(GPUS) / GPUS_PER_MODEL 개의 프로세스를 병렬 실행

set -e

# ---- GPU 설정 (첫 번째 줄) ----
GPUS="3,4,5,6"       # 사용할 전체 GPU
GPUS_PER_MODEL=2     # 모델 하나당 GPU 수

# ---- 설정 ----
PYTHON=/home/yoonju/miniconda3/envs/NRL/bin/python3

MODEL_NAME="checkpoints/offline_reinforce/epoch-1"
CLS_HEAD_PATH="${MODEL_NAME}/classifier_head.pt"

DATASET="datasets/math500.parquet"
OUTPUT_DIR="data/eval_results"
TAG=$(date +%Y%m%d_%H%M%S)

BATCH_SIZE=16
MAX_STEPS=10
MAX_NEW_TOKENS=512
TEMPERATURE=0.0

# ---- GPU 그룹 분할 ----
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
TOTAL_GPUS=${#GPU_ARRAY[@]}
NUM_WORKERS=$(( TOTAL_GPUS / GPUS_PER_MODEL ))

if [ $(( TOTAL_GPUS % GPUS_PER_MODEL )) -ne 0 ]; then
    echo "오류: 전체 GPU 수($TOTAL_GPUS)가 GPUS_PER_MODEL($GPUS_PER_MODEL)로 나누어 떨어져야 합니다."
    exit 1
fi

echo "===== MATH500 평가: $MODEL_NAME ====="
echo "===== 전체 GPU: $GPUS / 모델당 GPU: ${GPUS_PER_MODEL}개 / 워커 수: ${NUM_WORKERS}개 ====="
echo "===== Classifier head: $CLS_HEAD_PATH ====="
echo "===== 배치 크기: $BATCH_SIZE ====="

cd "$(dirname "$0")/.."
mkdir -p "$OUTPUT_DIR"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# ---- 워커별 병렬 실행 ----
PIDS=()
for (( w=0; w<NUM_WORKERS; w++ )); do
    START=$(( w * GPUS_PER_MODEL ))
    WORKER_GPUS=""
    for (( g=0; g<GPUS_PER_MODEL; g++ )); do
        IDX=$(( START + g ))
        [ -n "$WORKER_GPUS" ] && WORKER_GPUS="${WORKER_GPUS},"
        WORKER_GPUS="${WORKER_GPUS}${GPU_ARRAY[$IDX]}"
    done

    echo "[worker $w] GPU ${WORKER_GPUS} 시작..."
    CUDA_VISIBLE_DEVICES="$WORKER_GPUS" $PYTHON source/evaluate.py \
        --model_name "$MODEL_NAME" \
        --cls_head_path "$CLS_HEAD_PATH" \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --tag "$TAG" \
        --batch_size $BATCH_SIZE \
        --max_steps $MAX_STEPS \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --worker_id $w \
        --num_workers $NUM_WORKERS \
        > "$OUTPUT_DIR/worker${w}_${TAG}.log" 2>&1 &
    PIDS+=($!)
done

# ---- 완료 대기 ----
echo "모든 워커 실행 중... (로그: $OUTPUT_DIR/worker*_${TAG}.log)"
FAILED=0
for w in "${!PIDS[@]}"; do
    wait "${PIDS[$w]}" || { echo "[worker $w] 실패! 로그: $OUTPUT_DIR/worker${w}_${TAG}.log"; FAILED=1; }
    echo "[worker $w] 완료"
done

[ "$FAILED" -eq 1 ] && exit 1

# ---- 결과 병합 ----
echo "결과 병합 중..."

MERGED_RESULTS="$OUTPUT_DIR/results_${TAG}.jsonl"
> "$MERGED_RESULTS"
for (( w=0; w<NUM_WORKERS; w++ )); do
    cat "$OUTPUT_DIR/results_${TAG}_worker${w}.jsonl" >> "$MERGED_RESULTS"
done

$PYTHON - <<EOF
import json

tag = "${TAG}"
output_dir = "${OUTPUT_DIR}"
num_workers = ${NUM_WORKERS}

summaries = []
for i in range(num_workers):
    with open(f"{output_dir}/summary_{tag}_worker{i}.json") as f:
        summaries.append(json.load(f))

n_total   = sum(s["n_total"]      for s in summaries)
n_correct = sum(s["n_correct"]    for s in summaries)
n_term    = sum(s["n_terminated"] for s in summaries)

def merge_counts(key):
    merged = {}
    for s in summaries:
        for k, v in s.get(key, {}).items():
            merged[k] = merged.get(k, 0) + v
    return merged

merged = {
    "model":                   summaries[0]["model"],
    "cls_head":                summaries[0]["cls_head"],
    "dataset":                 summaries[0]["dataset"],
    "split":                   summaries[0]["split"],
    "batch_size":              summaries[0]["batch_size"],
    "num_workers":             num_workers,
    "n_total":                 n_total,
    "n_correct":               n_correct,
    "accuracy":                round(n_correct / n_total, 4),
    "n_terminated":            n_term,
    "termination_rate":        round(n_term / n_total, 4),
    "action_counts":           merge_counts("action_counts"),
    "predicted_action_counts": merge_counts("predicted_action_counts"),
}

with open(f"{output_dir}/summary_{tag}.json", "w") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"[merge] 최종 정확도: {merged['accuracy']:.4f} ({n_correct}/{n_total})")
print(f"[merge] 저장: {output_dir}/summary_{tag}.json")
EOF

echo "완료: $OUTPUT_DIR/results_${TAG}.jsonl"
