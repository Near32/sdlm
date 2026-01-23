#!/bin/bash

python -m ipdb -c c batch_optimize_main.py  \
--wandb_project="LMI-GCG" \
--model_name="HuggingFaceTB/SmolLM2-135M" \
--dataset_path="data/smollm2-135m-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1" \
--output_dir="results/smollm2-135m-base_GCG+BS=8+NC=704+TK=128+SEED=2_k1-5-25-run" \
--method="gcg" \
--baseline_backend="hf" \
--epochs=700 \
--model_precision=full \
--gradient_checkpointing=False \
--batch_size=8 \
--seed=2 \
--num_workers=1 \
--seq_len=80 \
--gcg_num_candidates=700 \
--gcg_top_k=128 \
--gcg_num_mutations=1 \
--gcg_pos_choice="uniform" \
--gcg_token_choice="uniform" \
--gcg_init_strategy="zeros"

#--target_indices="0,1"
