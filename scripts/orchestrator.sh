#!/bin/bash
# Orchestrator:
#   Phase 1: wait for 2 GPUs (>=THRESH MiB free) -> start 2-GPU training
#   Phase 2: while 2-GPU training runs, monitor for 4-GPU availability
#            when 2 MORE GPUs become free AND a checkpoint exists:
#              stop 2-GPU training (SIGTERM to whole pgrp), wait for mem release,
#              verify 4 GPUs still free, launch 4-GPU training with resume
#
# Usage: bash scripts/orchestrator.sh

set -u
set -m      # job control: backgrounded commands get their own process group (PGID == PID)
# Relaxed mode: paired with vllm_gpu_memory_utilization=0.4 in training scripts.
# Peak ~45GB (model+DS+vLLM32GB), need >=55GB free to absorb ~10GB neighbor spikes.
THRESH=${THRESH:-55000}
POLL=${POLL:-30}
RUN_BASE="/data/lsg/work/OPSD/outputs/qwen31b/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"
LOG=logs/orchestrator.log
TRAIN_LOG_2GPU=logs/train_2gpu.log
TRAIN_LOG_4GPU=logs/train_4gpu.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

# Returns newline-separated GPU indices with >= THRESH MiB free, excluding any in $1 (csv)
free_gpus() {
    local exclude="${1:-}"
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -v t="$THRESH" -v ex="$exclude" -F', ' '
          BEGIN { n = split(ex, a, ","); for (i=1;i<=n;i++) excl[a[i]]=1 }
          ($2+0) >= t && !($1 in excl) { print $1 }'
}

latest_ckpt() {
    ls -d "$RUN_BASE"/checkpoint-* 2>/dev/null \
      | awk -F'checkpoint-' '{print $NF, $0}' \
      | sort -n | tail -1 | cut -d' ' -f2-
}

# Kill the full process group rooted at PGID with SIGTERM, fall back to SIGKILL.
# Arg: $1 = pgid (== PID of setsid-spawned bash leader)
stop_pgrp() {
    local pgid="$1"
    log "stop_pgrp: SIGTERM to pgrp -${pgid}"
    kill -TERM -- -"$pgid" 2>/dev/null || true
    # Give processes up to 90s to finish shutdown (deepspeed/vllm teardown can be slow)
    for _ in $(seq 1 90); do
        # any process alive in pgrp?
        if ! pgrep -g "$pgid" >/dev/null 2>&1; then
            log "stop_pgrp: all processes in pgrp -${pgid} exited"
            return 0
        fi
        sleep 1
    done
    log "stop_pgrp: pgrp -${pgid} still alive after 90s, SIGKILL"
    kill -KILL -- -"$pgid" 2>/dev/null || true
    sleep 3
}

# Wait until GPU memory drops on the given GPUs (csv), up to $2 seconds.
# This guards against checking free_gpus too early.
wait_mem_release() {
    local gpus="$1" max_wait="${2:-60}"
    log "wait_mem_release: waiting up to ${max_wait}s for GPUs [$gpus] memory to release"
    for _ in $(seq 1 "$max_wait"); do
        # every listed gpu must show >= THRESH free
        local ok=1
        for g in ${gpus//,/ }; do
            local free
            free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
            if [ "${free:-0}" -lt "$THRESH" ]; then ok=0; break; fi
        done
        if [ "$ok" = "1" ]; then
            log "wait_mem_release: all GPUs [$gpus] now have >=${THRESH} MiB free"
            return 0
        fi
        sleep 1
    done
    log "wait_mem_release: timeout; some GPUs in [$gpus] still busy"
    return 1
}

# Ensure our own orchestrator shuts down training if we get signaled
CHILD_PGID=""
cleanup_on_exit() {
    if [ -n "$CHILD_PGID" ] && pgrep -g "$CHILD_PGID" >/dev/null 2>&1; then
        log "orchestrator received exit signal; stopping child pgrp -${CHILD_PGID}"
        stop_pgrp "$CHILD_PGID"
    fi
}
trap cleanup_on_exit EXIT INT TERM

log "orchestrator starting; threshold=${THRESH} MiB; poll=${POLL}s; run_base=${RUN_BASE}"

# ---------- Phase 1: wait for 2 GPUs, launch 2-GPU training ----------
while true; do
    mapfile -t AVAIL < <(free_gpus)
    if [ "${#AVAIL[@]}" -ge 2 ]; then
        PICK="${AVAIL[0]},${AVAIL[1]}"
        # If a checkpoint from a previous session exists, resume from it
        EXISTING_CKPT=$(latest_ckpt)
        if [ -n "$EXISTING_CKPT" ]; then
            log "phase1: 2 GPUs available (${PICK}); resuming 2-GPU training from ${EXISTING_CKPT}"
            CUDA_VISIBLE_DEVICES="$PICK" RESUME_FROM="$EXISTING_CKPT" bash scripts/run_opsd_1b_2gpu.sh > "$TRAIN_LOG_2GPU" 2>&1 &
        else
            log "phase1: 2 GPUs available (${PICK}); launching fresh 2-GPU training"
            CUDA_VISIBLE_DEVICES="$PICK" bash scripts/run_opsd_1b_2gpu.sh > "$TRAIN_LOG_2GPU" 2>&1 &
        fi
        TRAIN_PID=$!
        CHILD_PGID="$TRAIN_PID"
        CURRENT_GPUS="$PICK"
        log "phase1: training PID=PGID=${TRAIN_PID} on GPUs ${CURRENT_GPUS}; log: ${TRAIN_LOG_2GPU}"
        break
    fi
    log "phase1: only ${#AVAIL[@]} GPU free (need 2); sleeping ${POLL}s"
    sleep "$POLL"
done

# ---------- Phase 2: monitor for 4-GPU upgrade ----------
log "phase2: monitoring for 4-GPU upgrade opportunity (poll ${POLL}s)"
while pgrep -g "$TRAIN_PID" >/dev/null 2>&1; do
    sleep "$POLL"
    pgrep -g "$TRAIN_PID" >/dev/null 2>&1 || { log "phase2: training pgrp -${TRAIN_PID} exited; stopping orchestrator"; break; }

    mapfile -t OTHER < <(free_gpus "$CURRENT_GPUS")
    CKPT=$(latest_ckpt)
    if [ "${#OTHER[@]}" -ge 2 ] && [ -n "$CKPT" ]; then
        NEW2="${OTHER[0]},${OTHER[1]}"
        log "phase2: UPGRADE trigger - 2 extra GPUs (${NEW2}) free, latest ckpt=${CKPT}"
        stop_pgrp "$TRAIN_PID"
        CHILD_PGID=""
        log "phase2: 2-GPU training stopped; waiting for GPU memory to release on [${CURRENT_GPUS}]"
        wait_mem_release "$CURRENT_GPUS" 60 || true

        # Re-check: do we have 4 eligible GPUs now?
        mapfile -t AVAIL4 < <(free_gpus)
        if [ "${#AVAIL4[@]}" -ge 4 ]; then
            PICK4="${AVAIL4[0]},${AVAIL4[1]},${AVAIL4[2]},${AVAIL4[3]}"
            log "phase2: 4 GPUs confirmed (${PICK4}); launching 4-GPU training (resume from ${CKPT})"
            CUDA_VISIBLE_DEVICES="$PICK4" RESUME_FROM="$CKPT" bash scripts/run_opsd_1b.sh > "$TRAIN_LOG_4GPU" 2>&1 &
            TRAIN_PID=$!
            CHILD_PGID="$TRAIN_PID"
            log "phase2: 4-GPU training PID=PGID=${TRAIN_PID}; log: ${TRAIN_LOG_4GPU}"
            # Wait for it to finish
            wait "$TRAIN_PID" 2>/dev/null
            CHILD_PGID=""
            log "orchestrator: 4-GPU training finished"
            exit 0
        else
            log "phase2: only ${#AVAIL4[@]} GPUs free after stop (race lost). Restarting orchestrator."
            exec "$0"
        fi
    fi
done

log "orchestrator exiting (training ended without upgrade)"
