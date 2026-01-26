#!/bin/bash

# O2P (Output-to-Prompt) batch optimization script for SmolLM3-3B
#
# Prerequisites:
# 1. Train the O2P inverse model first using train_o2p_model.py:
#    bash smollm3_3b_o2p_train.sh
#    OR
#    python train_o2p_model.py \
#        --llm_model_name "HuggingFaceTB/SmolLM3-3B" \
#        --t5_model_name "t5-base" \
#        --bottleneck_dim 4096 \
#        --num_tokens 64 \
#        --dataset_size 400000 \
#        --num_epochs 30 \
#        --output_dir ./o2p_checkpoints/smollm3-3b \
#        --wandb_project "o2p-training"
#
# 2. Generate the dataset if not already available:
#    # (Run the dataset generation script for SmolLM3-3B)
#
# 3. Then run this script with the trained model path

python batch_optimize_main.py \
    --model_name="HuggingFaceTB/SmolLM3-3B" \
    --dataset_path="data/smollm3-3b-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1" \
    --output_dir="results/smollm3-3b-base_O2P_k1-5-25-run" \
    --method="o2p" \
    --o2p_model_path="./o2p_checkpoints/smollm3-3b/best_model" \
    --o2p_num_beams=4 \
    --o2p_max_length=32 \
    --seq_len=80 \
    --epochs=1 \
    --model_precision=half \
    --batch_size=1 \
    --seed=42 \
    --num_workers=1 \
    --wandb_project="LMI-O2P"

# Notes:
# - epochs=1 since O2P is one-shot (not iterative)
# - batch_size=1 for sequential processing (O2P doesn't batch internally)
# - model_precision=half to save memory for the 3B model
# - o2p_model_path must point to the trained inverse model
# - o2p_num_beams controls beam search quality (higher = better but slower)
# - o2p_max_length is the max prompt length the T5 decoder can generate
