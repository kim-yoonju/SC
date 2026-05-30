set -e
set -u

SCRIPT_DIR=$(cd $(dirname $0); pwd)
WORK_DIR=$SCRIPT_DIR/..
CONF=$WORK_DIR/configs/config.yaml

# 인자 파싱
GPUS_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS_OVERRIDE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# config.yaml에서 값 읽기
py() { python3 -c "import yaml; c=yaml.safe_load(open('$CONF')); print($1)"; }

BASE_MODEL=$(py "c['checkpoint']['base']")
GPU_IDS=$(py "','.join(map(str, c['grpo_pure']['train_gpus']))")
if [ -n "$GPUS_OVERRIDE" ]; then
    GPU_IDS="$GPUS_OVERRIDE"
fi
GEN_BATCH=$(py "c['generate_trajectory']['batch_per_gpu']")

# verl 하이퍼파라미터
V="c['grpo_pure']['verl']"
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

# Resume 설정
RESUME_FROM=$(py "str(${V}.get('resume_from') or '')")
if [ "$RESUME_FROM" = "None" ] || [ "$RESUME_FROM" = "null" ]; then
    RESUME_FROM=""
fi

if [ -n "$RESUME_FROM" ]; then
    # 이어서 학습: 기존 체크포인트 디렉토리 재사용
    CKPT_DIR=$(dirname "$RESUME_FROM")
    TS=$(basename "$CKPT_DIR")
    RUN_DIR=$WORK_DIR/output/GRPO_PURE/$TS
    echo "Resuming from checkpoint: $RESUME_FROM"
else
    # 새로 시작
    TS=$(date +%Y%m%d_%H%M%S)
    CKPT_DIR=$WORK_DIR/checkpoints/grpo_pure/$TS
    RUN_DIR=$WORK_DIR/output/GRPO_PURE/$TS
fi
mkdir -p $CKPT_DIR $RUN_DIR

N_GPUS=$(awk -F',' '{print NF}' <<<"$GPU_IDS")
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 이전 Ray 클러스터 정리 후 재시작
ray stop --force &>/dev/null 2>&1
sleep 2
echo "Starting Ray on GPUs: $GPU_IDS"
CUDA_VISIBLE_DEVICES=$GPU_IDS ray start --head --num-gpus=$N_GPUS
sleep 3

TRAIN_FILE=$(py "c['grpo_pure']['train_data']")
VAL_FILE=$(py "c['grpo_pure']['eval_data']")

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="{
    \"env_vars\": {
        \"VLLM_USE_V1\": \"0\",
        \"VLLM_ATTENTION_BACKEND\": \"XFORMERS\",
        \"PYTHONUNBUFFERED\": \"1\",
        \"CUDA_VISIBLE_DEVICES\": \"$GPU_IDS\",
        \"TORCHDYNAMO_DISABLE\": \"1\"
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
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    +algorithm.filter_groups.enable=False \
    actor_rollout_ref.model.path=$BASE_MODEL \
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
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=$PARAM_OFF \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LP_MAX_TOK \
    reward.custom_reward_function.path=$WORK_DIR/utils/reward_func_pure.py \
    reward.custom_reward_function.name=reward_func \
    trainer.project_name=sc-grpo-pure \
    trainer.experiment_name=sc-grpo-pure-$TS \
    +trainer.run_id=$TS \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.val_before_train=False \
    trainer.use_legacy_worker_impl=enable \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    +trainer.save_rollout=True \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.logger=['console','wandb'] \
    ${RESUME_FROM:+trainer.resume_from_checkpoint=$RESUME_FROM} \
    2>&1 | tee -a $CKPT_DIR/train.log
