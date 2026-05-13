#!/bin/bash
# Gen 단독 성능 평가 스크립트 (PRM 없음)
#
# 사용법:
#   bash scripts/run_eval_gen_solo.sh
#   bash scripts/run_eval_gen_solo.sh --num_data 100 --num_start 0
#   bash scripts/run_eval_gen_solo.sh --output output/my_eval
#   bash scripts/run_eval_gen_solo.sh --n_parallel 16
#   bash scripts/run_eval_gen_solo.sh --debug path/to/debug_ids.txt
#
# GPU 설정은 config/config.yaml의 generate_trajectory.rollout_gpus 에서 관리됩니다.

set -e

cd "$(dirname "$0")/.."

python3 source/evaluate_gen_solo.py "$@"

echo "Done."
