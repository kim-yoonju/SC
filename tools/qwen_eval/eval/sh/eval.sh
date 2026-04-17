set -e

# ── 설정 (여기만 수정) ────────────────────────────────────────────────────────
GPUS="6,7"   # 사용할 GPU (1장: "6" / 여러 장: "6,7" → tensor parallel 자동 적용)
MODEL_NAME_OR_PATH="/mnt/yoonju/NRL/S2R/checkpoints/sft/20260416_105229/checkpoint-225"

# 평가할 데이터셋 목록 (data/eval/ 안에 있는 파일명 기준, .parquet 제외)
DATA_NAMES=(
    # "math500"
    # "aime2024"
    # "aime2025"
    # "gsm8k"
    # "math7500"
    # "olym"
    # "amc23"
    "gaokao23En"
)
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TYPE=${1:-"qwen25-math-cot"}
EVAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_BASE="${EVAL_DIR}/evaluation_results/$(basename $(dirname ${MODEL_NAME_OR_PATH}))-$(basename ${MODEL_NAME_OR_PATH})"

# math_eval.py는 같은 디렉토리의 모듈을 import하므로 해당 디렉토리에서 실행
cd "${EVAL_DIR}"

for DATA_NAME in "${DATA_NAMES[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/${DATA_NAME}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "=========================================="
    echo " Dataset : ${DATA_NAME}"
    echo " Output  : ${OUTPUT_DIR}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${GPUS} TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name ${DATA_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --split test \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample -1 \
        --max_tokens_per_call 8000 \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs
done

echo ""
echo "=========================================="
echo " 전체 평가 완료"
echo " 결과 위치: ${OUTPUT_BASE}"
echo "=========================================="
