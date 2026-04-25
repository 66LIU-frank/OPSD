#!/bin/bash
# 2-GPU variant: keeps effective batch size = 32 via accum=4 (vs 4x2 on 4-GPU script)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH}
export WANDB_INIT_TIMEOUT=600

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${RESUME_FROM}"
    echo "[$(date +%T)] Resuming from: ${RESUME_FROM}"
fi

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 2 \
    --gradient_accumulation_steps 4 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path /data/lsg/models/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --output_dir  /data/lsg/work/OPSD/outputs/qwen31b/ \
    --run_config qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005 \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --wandb_project OPSD \
    $RESUME_ARG
