#!/bin/bash
# Wait for N GPUs with >= THRESH MiB free, then launch eval on them.
# Eval is vLLM inference only (no training state), so ~15GB per GPU is enough.
#
# Usage: bash scripts/wait_eval.sh [n_gpus] [threshold_mib] [poll_sec]

set -u
set -m
N_GPUS=${1:-4}
THRESH=${2:-20000}
POLL=${3:-30}
LOG=logs/wait_eval.log
EVAL_LOG=logs/eval.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

free_gpus() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -v t="$THRESH" -F', ' '($2+0) >= t { print $1 }'
}

log "wait_eval starting: need $N_GPUS GPUs with >=${THRESH} MiB free, TP=${N_GPUS}"

while true; do
    mapfile -t AVAIL < <(free_gpus)
    if [ "${#AVAIL[@]}" -ge "$N_GPUS" ]; then
        PICK=$(IFS=, ; echo "${AVAIL[*]:0:$N_GPUS}")
        log "found $N_GPUS eligible GPUs: $PICK; running eval"
        # run_eval.sh has base-model eval and checkpoint eval, use CUDA_VISIBLE_DEVICES override
        cd eval
        # Update CUDA_VISIBLE_DEVICES and TP in the eval script via env-based sed-free call:
        # run_eval.sh uses fixed values inside, so we have to override. The eval script runs
        # `python evaluate_math.py --tensor_parallel_size 4 ...` with CUDA_VISIBLE_DEVICES=1,2,3,6.
        # We'll just invoke the same python commands directly with our picked GPUs.
        BASE_MODEL="/data/lsg/models/Qwen3-1.7B"
        CKPT_DIR="/data/lsg/work/OPSD/outputs/qwen31b/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/checkpoint-100"

        log "=== evaluating BASE model ==="
        NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$PICK" python evaluate_math.py \
            --base_model "$BASE_MODEL" \
            --dataset aime24 \
            --val_n 12 \
            --temperature 1.0 \
            --tensor_parallel_size "$N_GPUS" > ../"$EVAL_LOG" 2>&1
        log "base eval done (exit $?)"

        log "=== evaluating CHECKPOINT-100 ==="
        NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$PICK" python evaluate_math.py \
            --base_model "$BASE_MODEL" \
            --dataset aime24 \
            --val_n 12 \
            --temperature 1.0 \
            --tensor_parallel_size "$N_GPUS" \
            --checkpoint_dir "$CKPT_DIR" >> ../"$EVAL_LOG" 2>&1
        log "checkpoint eval done (exit $?)"
        log "ALL EVALS COMPLETE; see $EVAL_LOG"
        exit 0
    fi
    log "eligible=${#AVAIL[@]} (need $N_GPUS); free MiB: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep "$POLL"
done
