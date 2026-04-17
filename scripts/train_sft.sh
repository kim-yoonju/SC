#! /bin/bash

source "$(dirname "$0")/../configs/config.sh"

# --- Training (SFT) ---
GPUS="2,3,4,5"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
TOTAL_BATCH_SIZE=32
TRAIN_BATCH_SIZE_PER_GPU=4  # per_device_train_batch_size: GPU당 batch size
                            # ZeRO2 + CPU offload: activation ~6 GiB/GPU → ~34 GiB total
                            # TOTAL(32) / PER_GPU(4) / NUM_GPUS(4) = grad_accum 2
EVAL_BATCH_SIZE_PER_GPU=4
LEARNING_RATE=5e-6
MODEL_MAX_LENGTH=8000

export HF_HOME=/mnt/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

# single-node NCCL config
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

# disable deepspeed fused kernels (Blackwell arch JIT compile issue)
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_FUSED_LAMB=0
export TORCH_CUDA_ARCH_LIST="12.0"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY=$WANDB_API_KEY



REPO_DIR=./code

# ── Step 0: Precompute ref logprobs (skipped if a previous run already exists) ─
EXISTING_REF=$(ls ./output/ref_logit/*/*.pt 2>/dev/null | sort | tail -1)
if [ -n "$EXISTING_REF" ]; then
    REF_LOGPROBS_PATH="$EXISTING_REF"
    echo "=== Ref logprobs already exist, skipping precompute ==="
    echo "    Using: $REF_LOGPROBS_PATH"
else
    REF_RUN_DIR="./output/ref_logit/$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$REF_RUN_DIR"
    REF_LOGPROBS_PATH="$REF_RUN_DIR/ref_logprobs.pt"

    echo "=== Precomputing ref logprobs in parallel (GPUs: $GPUS) ==="
    echo "    Output dir: $REF_RUN_DIR"
    echo "    Started at: $(date)"

    IFS=',' read -ra GPU_LIST <<< "$GPUS"
    NUM_PRECOMPUTE_GPUS=${#GPU_LIST[@]}

    PIDS=()
    for i in "${!GPU_LIST[@]}"; do
        GPU="${GPU_LIST[$i]}"
        PART_FILE="$REF_RUN_DIR/ref_logprobs.pt.part${i}"
        LOG_FILE="$REF_RUN_DIR/precompute_gpu${i}.log"
        echo "  [GPU $GPU] worker $i → $PART_FILE  (log: $LOG_FILE)"
        PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU python -u ${REPO_DIR}/precompute_ref_logprobs.py \
            --data_path        /mnt/yoonju/NRL/S2R/data/train_data/sft_qwen2.5_math_7B.json \
            --model_path       $MODEL_PATH \
            --output_path      $PART_FILE \
            --model_max_length $MODEL_MAX_LENGTH \
            --batch_size       16 \
            --gpu_id           $i \
            --world_size       $NUM_PRECOMPUTE_GPUS \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
    done

    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || { echo "ERROR: precompute worker $i failed. Check $REF_RUN_DIR/precompute_gpu${i}.log"; exit 1; }
        echo "  [worker $i] done"
    done

    echo "=== Merging $NUM_PRECOMPUTE_GPUS part files → $REF_LOGPROBS_PATH ==="
    python3 - "$REF_LOGPROBS_PATH" "$REF_RUN_DIR" $NUM_PRECOMPUTE_GPUS <<'PYEOF'
import sys, torch, os
output_path = sys.argv[1]
run_dir     = sys.argv[2]
num_parts   = int(sys.argv[3])
all_pairs = []
for i in range(num_parts):
    part = torch.load(f"{output_path}.part{i}", map_location="cpu")
    all_pairs.extend(part)
    os.remove(f"{output_path}.part{i}")
all_pairs.sort(key=lambda x: x[0])
result = [t for _, t in all_pairs]
torch.save(result, output_path)
print(f"Merged {len(result)} samples → {output_path}")
PYEOF

    echo "=== Finished at: $(date) ==="
    echo "    ref_logprobs.pt : $REF_LOGPROBS_PATH"
    echo "    GPU logs        : $REF_RUN_DIR/precompute_gpu*.log"
fi
# ─────────────────────────────────────────────────────────────────────────────

SAVE_STEPS=50
EVAL_STEPS=2000

GRADIENT_CHECKPOINTING=True
BF16=True
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / TRAIN_BATCH_SIZE_PER_GPU / NUM_GPUS))
MODEL_NAME_OR_PATH=$MODEL_PATH
# distributed setting (single-node)
MASTER_ADDR=localhost
MASTER_PORT=6000
declare -a DATA_PATH_LIST=(
"/mnt/yoonju/NRL/S2R/data/train_data/sft_qwen2.5_math_7B.json"
# "./data/train_data/sft_data.json"
)
RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR=./checkpoints/sft/${RUN_TIMESTAMP}
mkdir -p ${OUTPUT_DIR}
LOG_FILE=${OUTPUT_DIR}/train.log

echo "=== Output dir : ${OUTPUT_DIR} ==="
echo "=== Log file   : ${LOG_FILE} ==="

for DATA_PATH in "${DATA_PATH_LIST[@]}"; do
    echo "Processing dataset: ${DATA_PATH}"

    # -------------------------------------------------------------------------------------------
    deepspeed --include localhost:${GPUS} --master_addr ${MASTER_ADDR} --master_port=${MASTER_PORT} ${REPO_DIR}/src/sft/src/sft_weighted_with_kl.py \
        --output_dir ${OUTPUT_DIR} \
        --do_train True \
        --data_paths ${DATA_PATH} \
        --model_type qwen \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --model_max_length ${MODEL_MAX_LENGTH} \
        --remove_unused_columns False \
        --report_to wandb \
        --overwrite_output_dir True \
        --per_device_train_batch_size ${TRAIN_BATCH_SIZE_PER_GPU} \
        --per_device_eval_batch_size ${EVAL_BATCH_SIZE_PER_GPU} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --num_train_epochs 3 \
        --logging_strategy steps \
        --logging_steps 1 \
        --save_strategy epoch \
        --save_steps ${SAVE_STEPS} \
        --learning_rate ${LEARNING_RATE} \
        --eval_strategy epoch \
        --eval_steps ${EVAL_STEPS} \
        --warmup_steps 5 \
        --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
        --bf16 ${BF16} \
        --ref_logprobs_path $REF_LOGPROBS_PATH \
        --lm_kl_coeff 0.01 \
        --pad_labels_with_ignore \
        --optim adamw_torch \
        --deepspeed ${REPO_DIR}/configs/ds_stage2_fast.json \
        2>&1 | tee -a ${LOG_FILE}
done


