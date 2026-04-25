#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-600}
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/tmp/hf_datasets_opsd_sotopia}

MODEL_PATH=${MODEL_PATH:-/data/lsg/models/Qwen3-1.7B}
DATASET_PATH=${DATASET_PATH:-/data/lsg/work/OPSD/.sotopia_data/rc_opd/sotopia_pi_rc_opd_train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/data/lsg/work/OPSD/outputs/rc_opd_sotopia_formal}
RUN_CONFIG=${RUN_CONFIG:-rc_opd_sotopia_v1}
NUM_PROCESSES=${NUM_PROCESSES:-1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-12961}
ACCELERATE_BIN=${ACCELERATE_BIN:-/home/lsg/miniconda3/envs/opsd/bin/accelerate}

MAX_STEPS=${MAX_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-25}
LOGGING_STEPS=${LOGGING_STEPS:-1}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-128}
MAX_REFLECTION_LENGTH=${MAX_REFLECTION_LENGTH:-192}
MAX_LENGTH=${MAX_LENGTH:-8192}

"${ACCELERATE_BIN}" launch \
    --config_file accelerate.yaml \
    --num_processes "${NUM_PROCESSES}" \
    --gradient_accumulation_steps 1 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    opsd_train.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name_or_path "${DATASET_PATH}" \
    --dataset_split train \
    --use_rc_opd \
    --rc_reflection_levels step,turn,episode \
    --rc_curriculum_schedule retract \
    --rc_curriculum_steps "${MAX_STEPS}" \
    --max_reflection_length "${MAX_REFLECTION_LENGTH}" \
    --rc_filter_bad_reflections \
    --rc_min_reflection_chars 40 \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --output_dir "${OUTPUT_DIR}" \
    --run_config "${RUN_CONFIG}" \
    --num_train_epochs 1 \
    --max_steps "${MAX_STEPS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length "${MAX_LENGTH}" \
    --beta 0 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 0.8 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --use_ema_teacher \
    --ema_decay 0.99 \
    --jsd_token_clip 0.05 \
    --wandb_project RC-OPD
