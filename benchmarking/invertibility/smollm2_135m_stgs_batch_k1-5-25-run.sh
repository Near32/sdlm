#!/bin/bash

python -m ipdb -c c batch_optimize_main.py  \
--model_name="HuggingFaceTB/SmolLM2-135M" \
--dataset_path="data/smollm2-135m-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1" \
--output_dir="results/smollm2-135m-base_STGS+Soft+LearnTau+SoftBPTT+LearnBTau+BS=16+LR=1e-1+SEED=2_test_k1-5-25-run" \
--learning_rate=1.0e-1  \
--epochs 256 \
--model_precision full \
--gradient_checkpointing=False \
--losses="crossentropy" \
--gradient_estimator="stgs" \
--stgs_grad_variance_samples=0 \
--stgs_grad_variance_period=20 \
--stgs_grad_bias_samples=0 \
--stgs_grad_bias_period=50 \
--stgs_grad_bias_reference_samples=5 \
--stgs_grad_bias_reference_use_baseline=True \
--stgs_grad_bias_reference_reward_scale=1.0 \
--stgs_grad_bias_reference_baseline_beta=0.9 \
--batch_size=8 \
--seed=2 \
--num_workers=1 \
--seq_len=80 \
--stgs_hard=False \
--learnable_temperature=True \
--decouple_learnable_temperature=True \
--temperature=100.0 \
--bptt=False \
--bptt_stgs_hard=False \
--bptt_learnable_temperature=True \
--bptt_hidden_state_conditioning=False \
--bptt_temperature=100.0 

#--target_indices="0,1" 
#--losses=embxentropy \
# DEPR to same as batch_size : --stgs_grad_bias_reference_batch_size=64 \
