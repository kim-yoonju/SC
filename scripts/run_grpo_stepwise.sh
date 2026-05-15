set -euo pipefail

SCRIPT_DIR=$(cd $(dirname $0); pwd)
WORK_DIR=$SCRIPT_DIR/..
CONF=$WORK_DIR/configs/config.yaml

# config.yaml에서 값 읽기
py() { python3 -c "import yaml; c=yaml.safe_load(open('$CONF')); print($1)"; }

SFT_CHECKPOINT=$(py "c['checkpoint']['sft_checkpoint']")
GPU_IDS=$(py "','.join(map(str, c['grpo']['train_gpus']))")
NUM_START=$(py "c['grpo'].get('num_start', 'None')")
NUM_END=$(py "c['grpo'].get('num_end', 'None')")
GEN_BATCH=$(py "c['generate_trajectory']['batch_per_gpu']")

# verl 하이퍼파라미터
V="c['grpo']['verl']"
TRAIN_BSZ=$(py "${V}['train_batch_size']")
MAX_PROMPT=$(py "${V}['max_prompt_length']")
MAX_RESP=$(py "${V}['max_response_length']")
MINI_BSZ=$(py "${V}['ppo_mini_batch_size']")
MAX_TOK=$(py "${V}['ppo_max_token_len_per_gpu']")
LR=$(py "${V}['lr']")
KL_COEF=$(py "${V}['kl_loss_coef']")
GRAD_CLIP=$(py "${V}['grad_clip']")
CLIP_LO=$(py "${V}['clip_ratio_low']")
CLIP_HI=$(py "${V}['clip_ratio_high']")
ENTROPY=$(py "${V}['entropy_coeff']")
PARAM_OFF=$(py "${V}['param_offload']")
OPT_OFF=$(py "${V}['optimizer_offload']")
ROLLOUT_N=$(py "${V}['rollout_n']")
TEMP=$(py "${V}['temperature']")
GPU_UTIL=$(py "${V}['gpu_memory_utilization']")
MAX_BATCHED=$(py "${V}['max_num_batched_tokens']")
LP_MAX_TOK=$(py "${V}['log_prob_max_token_len_per_gpu']")
SAVE_FREQ=$(py "${V}['save_freq']")
TOTAL_STEPS=$(py "${V}['total_training_steps']")
RESUME_FROM=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONF')); x=c['grpo']['verl'].get('resume_from'); print('' if x is None else str(x))")

TS=$(date +%Y%m%d_%H%M%S)
CKPT_DIR=$WORK_DIR/checkpoints/grpo/$TS
RUN_DIR=$WORK_DIR/output/GRPO/$TS
mkdir -p $CKPT_DIR $RUN_DIR

# resume 설정
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARGS="trainer.resume_mode=resume_path trainer.resume_from_path=$RESUME_FROM"
    echo "Resuming from checkpoint: $RESUME_FROM"
else
    RESUME_ARGS="trainer.resume_mode=disable"
fi

N_GPUS=$(awk -F',' '{print NF}' <<<"$GPU_IDS")
export CUDA_VISIBLE_DEVICES=$GPU_IDS

cleanup() {
    echo "Stopping Ray..."
    ray stop --force
}
trap cleanup EXIT INT TERM

echo "Stopping any existing Ray instance..."
ray stop --force 2>/dev/null || true
sleep 1

# 디버그 출력 카운터 초기화
rm -f /tmp/grpo_reward_debug_count.txt

echo "Starting Ray on GPUs: $GPU_IDS"
# RAY_DISABLE_CUSTOM_METRICS prevents SIGSEGV in OpenTelemetry gRPC exporter thread
CUDA_VISIBLE_DEVICES=$GPU_IDS RAY_DISABLE_CUSTOM_METRICS=1 ray start --head --num-gpus=$N_GPUS
sleep 3

RL_INPUT=$(py "c['data_path']['rl_data']")
GRPO_PARQUET=$WORK_DIR/datasets/grpo_train.parquet
echo "GRPO 데이터 준비 중: $RL_INPUT (인덱스 $NUM_START ~ $NUM_END)"

SLICE_ARGS=""
[ "$NUM_START" != "None" ] && SLICE_ARGS="$SLICE_ARGS --num_start $NUM_START"
[ "$NUM_END"   != "None" ] && SLICE_ARGS="$SLICE_ARGS --num_end $NUM_END"

TRAIN_FILE=$(python3 $WORK_DIR/source/preprocess.py --grpo --grpo_input "$RL_INPUT" --grpo_output "$GRPO_PARQUET" $SLICE_ARGS \
    | tee /dev/stderr | grep '^TRAIN_FILE=' | cut -d= -f2)
echo "학습 데이터: $TRAIN_FILE"

echo ""
echo "====== 학습 데이터 샘플 (첫 번째 문제) ======"
python3 -c "
import pandas as pd, textwrap
df = pd.read_parquet('$TRAIN_FILE')
row = df.iloc[0]
prompt = str(row.get('prompt', row.get('question', '')))
answer = str(row.get('reward_model', {}).get('ground_truth', row.get('answer', '?')) if isinstance(row.get('reward_model'), dict) else row.get('answer', '?'))
print(f'[문제] {textwrap.shorten(prompt, width=300, placeholder=\"...\")}\n')
print(f'[정답] {answer}')
print(f'[총 샘플 수] {len(df)}')
"
echo "=============================================="
echo ""

VAL_FILE=$WORK_DIR/$(py "c['data_path']['math500_10']")

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="{
    \"env_vars\": {
        \"VLLM_USE_V1\": \"0\",
        \"VLLM_ATTENTION_BACKEND\": \"XFORMERS\",
        \"PYTHONUNBUFFERED\": \"1\",
        \"CUDA_VISIBLE_DEVICES\": \"$GPU_IDS\",
        \"HYDRA_FULL_ERROR\": \"1\",
        \"RAY_DISABLE_CUSTOM_METRICS\": \"1\"
    },
    \"pip\": [\"word2number\", \"timeout_decorator\"]
    }" -- PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    hydra.run.dir=$RUN_DIR \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.prompt_key=prompt \
    data.truncation=left \
    +data.rm_system_prompt=False \
    data.train_batch_size=$TRAIN_BSZ \
    +data.gen_batch_size=$GEN_BATCH \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESP \
    data.val_batch_size=2 \
    data.dataloader_num_workers=0 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    +algorithm.filter_groups.enable=False \
    actor_rollout_ref.model.path=$SFT_CHECKPOINT \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BSZ \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOK \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFF \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPT_OFF \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_LO \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_HI \
    actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMP \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LP_MAX_TOK \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2200 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=$PARAM_OFF \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LP_MAX_TOK \
    reward.custom_reward_function.path=$WORK_DIR/utils/reward_utils/reward_func.py \
    reward.custom_reward_function.name=reward_func \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=PRMRewardManager \
    reward.reward_manager.module.path=$WORK_DIR/utils/reward_utils/prm_reward_manager.py \
    trainer.project_name=sc-grpo-stepwise \
    trainer.experiment_name=sc-grpo-stepwise-$TS \
    +trainer.run_id=$TS \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.val_before_train=False \
    trainer.use_legacy_worker_impl=enable \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    +trainer.save_rollout=True \
    +trainer.stepwise_expand=True \
    $RESUME_ARGS \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.logger=['console','wandb'] \
    2>&1 | tee -a $CKPT_DIR/train.log

echo "Training done. Merging FSDP checkpoints to HuggingFace format..."
for STEP_DIR in $CKPT_DIR/global_step_*/; do
    ACTOR_DIR=$STEP_DIR/actor
    HF_DIR=$ACTOR_DIR/huggingface
    if [ -f "$ACTOR_DIR/fsdp_config.json" ]; then
        echo "Merging $STEP_DIR -> $HF_DIR"
        CUDA_VISIBLE_DEVICES="" python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir $ACTOR_DIR \
            --target_dir $HF_DIR \
            2>&1 | tee -a $CKPT_DIR/train.log
    fi
done
echo "All checkpoints merged."
