#!/bin/bash
# Online PPO RL Training 실행 스크립트
#
# 사용법:
#   bash scripts/run_ppo_online.sh
#
# 하이퍼파라미터는 config/config.yaml의 ppo 섹션에서 관리됩니다.
# 특정 값만 오버라이드하려면 스크립트 하단 python 호출부에 인자를 추가하세요.
#   예) --lr 5e-7 --train_batch_size 256

set -e

cd "$(dirname "$0")/.."

# ---- config/config.yaml 파싱 ----
eval "$(python3 - <<'PYEOF'
import yaml, sys

with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ppo = cfg["ppo"]

rollout_gpus = ",".join(str(g) for g in ppo["rollout_gpus"])
train_gpus   = ",".join(str(g) for g in ppo["train_gpus"])
resume       = ppo.get("resume_checkpoint") or ""

print(f'ROLLOUT_GPUS="{rollout_gpus}"')
print(f'TRAIN_GPUS="{train_gpus}"')
print(f'RESUME_CHECKPOINT="{resume}"')
print(f'MAX_ITERATIONS={ppo["max_iterations"]}')
print(f'PROBLEMS_PER_ITER={ppo["problems_per_iter"]}')
print(f'TRAIN_BATCH_SIZE={ppo["train_batch_size"]}')
print(f'LR={ppo["lr"]}')
print(f'CLIP_EPS={ppo["clip_eps"]}')
print(f'KL_COEF={ppo["kl_coef"]}')
print(f'GAMMA={ppo["gamma"]}')
print(f'MAX_SEQ_LEN={ppo["max_seq_len"]}')
PYEOF
)"

# ---- 실행 ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUME_ARG=""
if [ -n "$RESUME_CHECKPOINT" ]; then
    RESUME_ARG="--resume_checkpoint $RESUME_CHECKPOINT"
fi

# 실행 시각 타임스탬프를 한 번만 생성 → Python과 공유해서 같은 폴더 사용
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="output/train_ppo/${RUN_TS}"

mkdir -p "$RUN_DIR"
cp "$0" "${RUN_DIR}/run_ppo_online.sh"

echo "Run ts: $RUN_TS"
echo "Run dir: $RUN_DIR"

python3 source/train_ppo.py \
    --rollout_gpus   "$ROLLOUT_GPUS" \
    --train_gpus     "$TRAIN_GPUS" \
    $RESUME_ARG \
    --run_ts           "$RUN_TS" \
    --max_iterations   $MAX_ITERATIONS \
    --problems_per_iter $PROBLEMS_PER_ITER \
    --train_batch_size  $TRAIN_BATCH_SIZE \
    --lr               $LR \
    --clip_eps         $CLIP_EPS \
    --kl_coef          $KL_COEF \
    --gamma            $GAMMA \
    --max_seq_len      $MAX_SEQ_LEN \
    2>&1 | tee "${RUN_DIR}/shell.log"

echo "Done."
