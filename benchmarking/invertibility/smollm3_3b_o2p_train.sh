#!/bin/bash

# O2P Training Script for SmolLM3-3B
#
# This script trains a T5-based inverse model for prompt reconstruction
# on the SmolLM3-3B subject LLM.
#
# After training, use smollm3_3b_o2p_batch_k1-5-25-run.sh for evaluation.

python train_o2p_model.py \
    --llm_model_name "HuggingFaceTB/SmolLM3-3B" \
    --t5_model_name "t5-base" \
    --bottleneck_dim 4096 \
    --num_tokens 64 \
    --dataset_size 400000 \
    --min_seq_length 1 \
    --max_seq_length 24 \
    --num_epochs 30 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --output_dir ./o2p_checkpoints/smollm3-3b \
    --wandb_project "o2p-training" \
    --llm_precision half \
    --seed 42

# Notes:
# - num_tokens: Controls the sequence length fed to the T5 encoder. The LLM's
#   vocabulary logits (vocab_size ~49k) are reshaped into [num_tokens, vocab_size/num_tokens]
#   and projected to T5's embedding space. This compresses the logit information into a
#   fixed-length sequence that T5 can process. Higher values = more capacity but slower.
# - bottleneck_dim: Dimension of the MLP that transforms the compressed logits to T5
#   encoder embeddings. Acts as a bottleneck for information flow.
# - dataset_size: 400k random token sequences for training
# - min/max_seq_length: Token length range for generated training data
# - batch_size: Reduced to 32 due to larger LLM (3B params needs more memory)
# - llm_precision: Use half to save memory (LLM is frozen anyway)
# - Output will be saved to ./o2p_checkpoints/smollm3-3b/best_model
