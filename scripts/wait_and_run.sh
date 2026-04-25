#!/bin/bash
# Poll GPUs and launch training once N GPUs each have >= THRESHOLD MiB free.
# Usage: bash scripts/wait_and_run.sh [n_gpus] [threshold_mib] [poll_sec] [script]

N_GPUS=${1:-2}
THRESHOLD_MIB=${2:-48000}
POLL_SEC=${3:-30}
TRAIN_SCRIPT=${4:-scripts/run_opsd_1b_2gpu.sh}
LOG=logs/wait.log

mkdir -p logs
echo "[$(date)] watcher started: need $N_GPUS GPUs with >=${THRESHOLD_MIB} MiB free, poll every ${POLL_SEC}s, run: $TRAIN_SCRIPT" | tee -a "$LOG"

while true; do
    mapfile -t FREE_LIST < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v t="$THRESHOLD_MIB" -F', ' '$2+0 >= t {print $1}')
    COUNT=${#FREE_LIST[@]}
    NOW=$(date +%H:%M:%S)
    if [ "$COUNT" -ge "$N_GPUS" ]; then
        PICK=$(IFS=, ; echo "${FREE_LIST[*]:0:$N_GPUS}")
        echo "[$NOW] found $COUNT eligible GPUs; using: $PICK" | tee -a "$LOG"
        export CUDA_VISIBLE_DEVICES="$PICK"
        echo "[$NOW] launching: $TRAIN_SCRIPT on $PICK" | tee -a "$LOG"
        exec bash "$TRAIN_SCRIPT"
    fi
    echo "[$NOW] eligible=$COUNT (need $N_GPUS); free MiB per GPU: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')" | tee -a "$LOG"
    sleep "$POLL_SEC"
done
