export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH}
export WANDB_INIT_TIMEOUT=600
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/tmp/hf_datasets_opsd}

MODEL_PATH=${MODEL_PATH:-/data/lsg/models/Qwen3-1.7B}
DATASET_PATH=${DATASET_PATH:-/data/lsg/work/OPSD/examples/rc_opd_social_sample.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/data/lsg/work/OPSD/outputs/rc_opd_social}
NUM_PROCESSES=${NUM_PROCESSES:-1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-12960}

accelerate launch \
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
    --rc_curriculum_steps 100 \
    --max_reflection_length 384 \
    --rc_filter_bad_reflections \
    --rc_min_reflection_chars 40 \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --output_dir "${OUTPUT_DIR}" \
    --run_config rc_opd_social_v0 \
    --num_train_epochs 1 \
    --max_completion_length 256 \
    --save_steps 25 \
    --logging_steps 1 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 8192 \
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
