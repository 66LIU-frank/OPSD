#!/bin/bash

BASE_MODEL="/data/lsg/models/Qwen3-1.7B"
CKPT_DIR="/data/lsg/work/OPSD/outputs/qwen31b/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/checkpoint-100"

# evaluate base model performance
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,6 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 4
wait

# after trained, evaluate the performance of the trained model
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,6 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 4 \
    --checkpoint_dir "$CKPT_DIR"
wait
    
