#!/bin/bash
# Checkpoint-100 eval with TP=2 on GPUs 4,5 for ~1.5x speedup
set -u
export LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}
cd /data/lsg/work/OPSD/eval

GPUS=${GPUS:-4,5}
BASE_MODEL="/data/lsg/models/Qwen3-1.7B"
CKPT_DIR="/data/lsg/work/OPSD/outputs/qwen31b/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/checkpoint-100"
LOG=../logs/eval_ckpt.log

echo "[$(date +%T)] parallel ckpt-100 eval TP=2 on GPUs $GPUS" | tee -a "$LOG"

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPUS" python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset aime24 \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.35 \
    --checkpoint_dir "$CKPT_DIR" 2>&1 | tee -a "$LOG"
echo "[$(date +%T)] checkpoint eval finished (exit $?)" | tee -a "$LOG"
