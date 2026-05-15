#!/usr/bin/env bash
# run_interactive_grpo.sh
#
# Interactive rollout + GRPO (verl 없이).
#
# GPU 배분:
#   generate_trajectory.rollout_gpus[0]  → step_manager (base HF, subprocess)
#   generate_trajectory.rollout_gpus[1+] → vLLM (SFT 체크포인트)
#   grpo.train_gpus                      → actor FSDP + reference (torchrun)
#
# ※ rollout_gpus와 grpo.train_gpus는 반드시 서로 다른 GPU 번호를 사용해야 합니다.
#    예) config.yaml:
#          generate_trajectory.rollout_gpus: [4, 7]   # GPU4=step_manager, GPU7=vLLM
#          grpo.train_gpus:                 [2, 3, 5, 6]
#
# 사용법:
#   bash scripts/run_interactive_grpo.sh

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
WORK_DIR=$SCRIPT_DIR/..
CONF=$WORK_DIR/configs/config.yaml

py() { python3 -c "import yaml; c=yaml.safe_load(open('$CONF')); print($1)"; }

# ── GPU 설정 ──────────────────────────────────────────────────────────────────
ROLLOUT_GPUS=$(py "','.join(map(str, c['generate_trajectory']['rollout_gpus']))")
TRAIN_GPUS=$(py "','.join(map(str, c['grpo']['train_gpus']))")
N_TRAIN=$(awk -F',' '{print NF}' <<< "$TRAIN_GPUS")

SM_GPU=$(py "str(c['generate_trajectory']['rollout_gpus'][0])")
VLLM_GPUS=$(py "','.join(map(str, c['generate_trajectory']['rollout_gpus'][1:]))" 2>/dev/null || echo "$SM_GPU")

echo "========================================================"
echo " Interactive GRPO"
echo "  step_manager GPU : $SM_GPU"
echo "  vLLM GPU(s)      : $VLLM_GPUS"
echo "  Train GPU(s)     : $TRAIN_GPUS (×${N_TRAIN})"
echo "========================================================"

# GPU 중복 체크
for g in $(echo "$TRAIN_GPUS" | tr ',' ' '); do
    for r in $(echo "$ROLLOUT_GPUS" | tr ',' ' '); do
        if [ "$g" = "$r" ]; then
            echo "경고: GPU $g 가 generate_trajectory.rollout_gpus와 grpo.train_gpus에 모두 존재합니다."
            echo "      config.yaml에서 GPU 세트를 분리해주세요."
            exit 1
        fi
    done
done

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SFT_CHECKPOINT=$(py "c['checkpoint']['sft_checkpoint']")
RL_INPUT=$(py "c['data_path']['rl_data']")
NUM_START=$(py "str(c['generate_trajectory'].get('num_start', 'None'))")
NUM_END=$(py "str(c['generate_trajectory'].get('num_end', 'None'))")
BATCH_SIZE=$(py "str(c['generate_trajectory'].get('batch_per_gpu', 16))")

TS=$(date +%Y%m%d_%H%M%S)
QUEUE_DIR=$WORK_DIR/queue/igrpo_$TS
RUN_DIR=$WORK_DIR/output/GRPO/$TS          # 데이터 생성 로그
CKPT_DIR=$WORK_DIR/checkpoints/grpo/$TS   # 모델 학습 로그 + 체크포인트
mkdir -p "$QUEUE_DIR" "$RUN_DIR" "$CKPT_DIR"

echo "  SFT 체크포인트  : $SFT_CHECKPOINT"
echo "  데이터          : $RL_INPUT [$NUM_START:$NUM_END]"
echo "  Queue           : $QUEUE_DIR"
echo "  데이터생성 로그 : $RUN_DIR"
echo "  학습 체크포인트 : $CKPT_DIR"
echo "========================================================"

# ── Ray 초기화 ────────────────────────────────────────────────────────────────
echo "Ray 종료 및 초기화 중..."
ray stop --force 2>/dev/null || true
sleep 2
ray start --head --num-cpus=0 --num-gpus=0 2>/dev/null || true
echo "Ray 초기화 완료."

# ── 종료 시 cleanup ───────────────────────────────────────────────────────────
ROLLOUT_PID=""
cleanup() {
    echo "종료 중..."
    [ -n "$ROLLOUT_PID" ] && kill "$ROLLOUT_PID" 2>/dev/null || true
    wait "$ROLLOUT_PID" 2>/dev/null || true

    echo "Ray 종료 및 프로세스 정리 중..."
    ray stop --force 2>/dev/null || true
    # vLLM / torchrun 잔여 프로세스 정리
    pkill -f "interactive_grpo.py" 2>/dev/null || true
    pkill -f "ray::" 2>/dev/null || true
    sleep 2
    echo "완료"
}
trap cleanup EXIT INT TERM

# ── rollout worker 시작 (백그라운드) ─────────────────────────────────────────
ROLLOUT_LOG=$RUN_DIR/rollout.log

SLICE_ARGS=""
[ "$NUM_START" != "None" ] && SLICE_ARGS="$SLICE_ARGS --num_start $NUM_START"
[ "$NUM_END"   != "None" ] && SLICE_ARGS="$SLICE_ARGS --num_end $NUM_END"

CUDA_VISIBLE_DEVICES=$ROLLOUT_GPUS \
    python3 $WORK_DIR/source/interactive_grpo.py \
    --mode       rollout \
    --queue_dir  "$QUEUE_DIR" \
    --run_dir    "$RUN_DIR" \
    --model_path "$SFT_CHECKPOINT" \
    --data_path  "$RL_INPUT" \
    --batch_size "$BATCH_SIZE" \
    $SLICE_ARGS \
    > >(tee -a "$ROLLOUT_LOG") 2>&1 &
    

ROLLOUT_PID=$!
echo "Rollout PID: $ROLLOUT_PID  (로그: $ROLLOUT_LOG)"

# rollout worker가 뜨기까지 잠깐 대기
sleep 5

# ── training worker 시작 (포그라운드, torchrun) ───────────────────────────────
TRAIN_LOG=$CKPT_DIR/train.log

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
    --nproc_per_node="$N_TRAIN" \
    --master_port=29502 \
    $WORK_DIR/source/interactive_grpo.py \
    --mode       train \
    --queue_dir  "$QUEUE_DIR" \
    --model_path "$SFT_CHECKPOINT" \
    --run_dir    "$CKPT_DIR" \
    --data_path  "$RL_INPUT" \
    2>&1 | tee -a "$TRAIN_LOG"

echo "학습 완료."
