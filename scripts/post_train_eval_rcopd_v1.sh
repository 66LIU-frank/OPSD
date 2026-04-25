#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/data/lsg/work/OPSD/outputs/rc_opd_sotopia_formal/rc_opd_sotopia_v1}
CKPT=${CKPT:-"${RUN_DIR}/checkpoint-200"}
DATASET_FILE=${DATASET_FILE:-/data/lsg/work/OPSD/.sotopia_data/rc_opd/sotopia_pi_rc_opd_eval.jsonl}
GPU=${GPU:-5}

while [[ ! -f "${CKPT}/adapter_model.safetensors" ]]; do
    echo "[$(date +%T)] waiting for ${CKPT}"
    sleep 60
done

/home/lsg/miniconda3/envs/opsd/bin/python /data/lsg/work/OPSD/scripts/export_trainer_log.py \
    --trainer-state "${RUN_DIR}/trainer_state.json" \
    --output-csv "${RUN_DIR}/training_curve.csv"

CUDA_VISIBLE_DEVICES="${GPU}" \
TAG=checkpoint-200 \
CKPT_DIR="${CKPT}" \
NUM_SAMPLES=128 \
DATASET_FILE="${DATASET_FILE}" \
bash /data/lsg/work/OPSD/scripts/run_social_response_eval.sh
