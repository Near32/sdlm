#!/bin/bash

python -m ipdb -c c batch_optimize_main.py  \
--wandb_project="LMI-SODA" \
--model_name="HuggingFaceTB/SmolLM2-135M" \
--dataset_path="data/smollm2-135m-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1" \
--output_dir="results/smollm2-135m-base_SODA+BS=8+LR=0.03+SEED=2_k1-5-25-run" \
--method="soda" \
--baseline_backend="hf" \
--learning_rate=0.03 \
--epochs=10000 \
--model_precision=full \
--gradient_checkpointing=False \
--batch_size=8 \
--seed=3 \
--num_workers=1 \
--seq_len=80 \
--temperature=0.05 \
--soda_decay_rate=0.98 \
--soda_beta1=0.9 \
--soda_beta2=0.995 \
--soda_reset_epoch=50 \
--soda_reinit_epoch=1500 \
--soda_bias_correction=False \
--soda_init_strategy="zeros"

#--soda_reg_weight=0.009 \
#--target_indices="0,1"
