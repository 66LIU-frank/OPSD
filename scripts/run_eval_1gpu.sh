#!/bin/bash
# Single-GPU eval: base model + checkpoint-100 on AIME24
set -u
cd /data/lsg/work/OPSD/eval

GPU=${GPU:-1}
BASE_MODEL="/data/lsg/models/Qwen3-1.7B"
CKPT_DIR="/data/lsg/work/OPSD/outputs/qwen31b/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/checkpoint-100"
LOG=../logs/eval.log

echo "[$(date +%T)] Starting eval on GPU $GPU" | tee -a "$LOG"

echo "[$(date +%T)] === BASE model eval ===" | tee -a "$LOG"
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU" python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset aime24 \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.35 2>&1 | tee -a "$LOG"
echo "[$(date +%T)] base eval finished (exit $?)" | tee -a "$LOG"

echo "[$(date +%T)] === CHECKPOINT-100 eval ===" | tee -a "$LOG"
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU" python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset aime24 \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.35 \
    --checkpoint_dir "$CKPT_DIR" 2>&1 | tee -a "$LOG"
echo "[$(date +%T)] checkpoint eval finished (exit $?)" | tee -a "$LOG"

echo "[$(date +%T)] ALL DONE" | tee -a "$LOG"
