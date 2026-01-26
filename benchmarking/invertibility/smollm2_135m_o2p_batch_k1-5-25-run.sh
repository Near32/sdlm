#!/bin/bash

# O2P (Output-to-Prompt) batch optimization script for SmolLM2-135M
#
# Prerequisites:
# 1. Train the O2P inverse model first using train_o2p_model.py:
#    python train_o2p_model.py \
#        --llm_model_name "HuggingFaceTB/SmolLM2-135M" \
#        --t5_model_name "t5-base" \
#        --bottleneck_dim 4096 \
#        --num_tokens 64 \
#        --dataset_size 400000 \
#        --num_epochs 10 \
#        --output_dir ./o2p_checkpoints/smollm2-135m \
#        --wandb_project "o2p-training"
#
# 2. Then run this script with the trained model path

python batch_optimize_main.py \
--model_name="HuggingFaceTB/SmolLM2-135M" \
--dataset_path="data/smollm2-135m-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1" \
--output_dir="results/smollm2-135m-base_O2P_k1-5-25-run" \
--method="o2p" \
--o2p_model_path="./o2p_checkpoints/smollm2-135m/best_model" \
--o2p_num_beams=4 \
--o2p_max_length=32 \
--seq_len=80 \
--epochs=1 \
--model_precision=full \
--batch_size=1 \
--seed=42 \
--num_workers=1 \
--wandb_project="LMI-O2P"

# Notes:
# - epochs=1 since O2P is one-shot (not iterative)
# - batch_size=1 for sequential processing (O2P doesn't batch internally)
# - o2p_model_path must point to the trained inverse model
# - o2p_num_beams controls beam search quality (higher = better but slower)
# - o2p_max_length is the max prompt length the T5 decoder can generate
