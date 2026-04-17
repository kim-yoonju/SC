#!/bin/bash
set -e

source "$(dirname "$0")/../configs/config.sh"

# ── 여기서 설정 (config.sh 이후에 설정해야 override 됨) ───────
GPUS="0,1,6,7"   # 사용할 GPU 번호 (쉼표로 구분, GPU 하나당 모델 1개)
BATCH_SIZE=32     # GPU당 한 번에 처리할 문제 수
# ──────────────────────────────────────────────────────────────

export HF_HOME=/mnt/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

# ── 출력 디렉토리 (RESUME_DIR 지정 시 이어서 실행) ───────────
if [ -n "$RESUME_DIR" ]; then
    RUN_DIR="$RESUME_DIR"
    echo "=== Resuming from: $RUN_DIR ==="
else
    RUN_DIR="./output/$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$RUN_DIR"
fi

exec > >(tee -a "$RUN_DIR/collect_data.log") 2>&1
echo "Run directory: $RUN_DIR"
echo "Started at: $(date)"

STEP1_OUT="$RUN_DIR/step1_output.jsonl"
STEP2_OUT="$RUN_DIR/step2_output.jsonl"
STEP3_OUT="$RUN_DIR/sft_data.json"

# Step 1: GPU별 독립 프로세스로 병렬 처리 후 결과 병합
echo "=== Step 1: Collecting responses from LLM (${DATA_NAMES[*]}) ==="
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}
echo "Launching ${NUM_GPUS} workers (GPUs: ${GPU_LIST[*]}, batch_size=${BATCH_SIZE} per GPU)"

PIDS=()
for i in "${!GPU_LIST[@]}"; do
    GPU="${GPU_LIST[$i]}"
    PART_FILE="${STEP1_OUT}.part${i}"
    echo "[GPU ${GPU}] Worker ${i} → ${PART_FILE}"
    CUDA_VISIBLE_DEVICES=$GPU python ./tools/1_collect_data_from_llm.py \
        --model_name_or_path $MODEL_PATH \
        --data_dir $DATA_DIR \
        --data_names "${DATA_NAMES[@]}" \
        --output_file "$PART_FILE" \
        --batch_size $BATCH_SIZE \
        --gpu_id $i \
        --world_size $NUM_GPUS \
        >> "$RUN_DIR/collect_data_gpu${i}.log" 2>&1 &
    PIDS+=($!)
done

# ── 터미널 진행 상황 모니터 (10초마다 저장된 문제 수 출력) ───
(while true; do
    sleep 10
    SAVED=$(cat "${STEP1_OUT}".part* 2>/dev/null | wc -l || echo 0)
    echo "[$(date '+%H:%M:%S')] Progress: ${SAVED} problems saved  (logs: $RUN_DIR/collect_data_gpu*.log)"
done) &
MONITOR_PID=$!

echo "Waiting for ${NUM_GPUS} workers to finish..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[GPU ${GPU_LIST[$i]}] Worker ${i} done."
    else
        echo "ERROR: Worker ${i} (GPU ${GPU_LIST[$i]}) failed. Check $RUN_DIR/collect_data_gpu${i}.log"
        FAILED=1
    fi
done
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null || true

[ "$FAILED" -eq 1 ] && exit 1

# 결과 병합
echo "Merging ${NUM_GPUS} part files → ${STEP1_OUT}"
cat "${STEP1_OUT}".part* > "$STEP1_OUT"
rm "${STEP1_OUT}".part*
echo "Merge done: $(wc -l < "$STEP1_OUT") lines total."

# 중복 제거: 같은 unique_id 중 n_samples_collected가 가장 많은 레코드만 유지
echo "Deduplicating (keeping most complete record per problem)..."
python3 - "$STEP1_OUT" <<'PYEOF'
import json, sys
records = {}
with open(sys.argv[1]) as f:
    for line in f:
        r = json.loads(line.strip())
        uid = r["unique_id"]
        n = r.get("n_samples_collected", len(r.get("round_1_response", [])))
        if uid not in records or n > records[uid].get("n_samples_collected", 0):
            records[uid] = r
with open(sys.argv[1], "w") as f:
    for r in records.values():
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"After dedup: {len(records)} unique problems")
PYEOF

# Step 2: Verification via GPT-4o
echo "=== Step 2: Collecting verifications (GPT-4o) ==="
python ./tools/2_collect_verification.py \
    --original_file_path $STEP1_OUT \
    --output_file $STEP2_OUT \
    --reference_file_path $STEP2_OUT \
    --api_key $API_KEY \
    --api_url $API_URL \
    --model_name_or_path $VERIFIER_MODEL \
|| { echo "ERROR: Step 2 failed. Aborting."; exit 1; }

# Step 3: Construct multi-turn SFT data
echo "=== Step 3: Constructing multi-turn training data ==="
python ./tools/3_contruct_muliti_turn_data.py \
    --response_file_path $STEP1_OUT \
    --verification_file_path $STEP2_OUT \
    --output_file $STEP3_OUT \
    --model_name_or_path $MODEL_PATH \
    --refiner_api_key $API_KEY \
    --refiner_api_url $API_URL \
    --refiner_model $REFINER_MODEL \
|| { echo "ERROR: Step 3 failed. Aborting."; exit 1; }

echo "=== Done! ==="
echo "  Step1 : $STEP1_OUT"
echo "  Step2 : $STEP2_OUT"
echo "  SFT   : $STEP3_OUT"
echo "  Log   : $RUN_DIR/collect_data.log"
echo "Finished at: $(date)"
