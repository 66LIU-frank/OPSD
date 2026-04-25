#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}

MODEL_PATH=${MODEL_PATH:-/data/lsg/models/Qwen3-1.7B}
DATASET_FILE=${DATASET_FILE:-/data/lsg/work/OPSD/.sotopia_data/rc_opd/sotopia_pi_rc_opd_eval.jsonl}
CKPT_DIR=${CKPT_DIR:-}
TAG=${TAG:-base}
NUM_SAMPLES=${NUM_SAMPLES:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
BATCH_SIZE=${BATCH_SIZE:-4}
OUTPUT_DIR=${OUTPUT_DIR:-/data/lsg/work/OPSD/eval/eval_results/social_response}

mkdir -p "${OUTPUT_DIR}"

CMD=(
    /home/lsg/miniconda3/envs/opsd/bin/python
    /data/lsg/work/OPSD/eval/evaluate_social_response.py
    --model_name_or_path "${MODEL_PATH}"
    --dataset_file "${DATASET_FILE}"
    --output_file "${OUTPUT_DIR}/${TAG}.json"
    --summary_csv "${OUTPUT_DIR}/summary.csv"
    --num_samples "${NUM_SAMPLES}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --batch_size "${BATCH_SIZE}"
)

if [[ -n "${CKPT_DIR}" ]]; then
    CMD+=(--checkpoint_dir "${CKPT_DIR}")
fi

"${CMD[@]}"
