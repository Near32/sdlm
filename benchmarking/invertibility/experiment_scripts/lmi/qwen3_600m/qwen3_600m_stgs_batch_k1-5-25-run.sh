#!/bin/bash

python -m ipdb -c c ../../../batch_optimize_main.py  \
--wandb_project="LMI-GS" \
--model_name="Qwen/Qwen3-0.6B" \
--dataset_path="../../../data/qwen3-600m-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1" \
--output_dir="results/qwen3-600m-base_STGS+Soft+LearnTau+SoftBPTT+LearnBTau+BS=16+LR=1e-1+SEED=2_test_k1-5-25-run" \
--learning_rate=1.0e-1  \
--epochs 8192 \
--model_precision full \
--gradient_checkpointing=False \
--losses="hinge-margin=0" \
--loss_pos_weight_schedule="uniform" \
--loss_pos_weight_step=2.0 \
--loss_pos_weight_base=2.0 \
--teacher_forcing=True \
--promptLambda=0.0 \
--complLambda=0.0 \
--promptTfComplLambda=0.0 \
--promptDistEntropyLambda=0.0 \
\
--gradient_estimator="stgs" \
--stgs_grad_variance_samples=0 \
--stgs_grad_variance_period=20 \
--stgs_grad_bias_samples=0 \
--stgs_grad_bias_period=50 \
--stgs_grad_bias_reference_samples=5 \
--stgs_grad_bias_reference_use_baseline=True \
--stgs_grad_bias_reference_reward_scale=1.0 \
--stgs_grad_bias_reference_baseline_beta=0.9 \
\
--batch_size=32 \
--seed=10 \
--num_workers=1 \
\
--seq_len=10 \
\
--stgs_hard=False \
--stgs_hard_method="embsim-l2" \
--stgs_hard_embsim_probs="gumbel_soft" \
--logits_normalize="none" \
--logits_top_k=0 \
--logits_top_p=1.0 \
--logit_decay=0.0 \
--learnable_temperature=True \
--decouple_learnable_temperature=True \
--temperature=100.0 \
\
--logits_lora_rank=4 \
--stgs_input_dropout=0.0 \
--stgs_output_dropout=0.0 \
--gumbel_noise_scale=1.0 \
--adaptive_gumbel_noise=False \
--adaptive_gumbel_noise_beta=0.9 \
--adaptive_gumbel_noise_min_scale=0.0 \
\
--ppo_kl_lambda=0.0 \
--ppo_kl_mode="soft" \
--ppo_kl_epsilon=0.1 \
--ppo_ref_update_period=16 \
\
--commitmentLambda=0.0 \
--commitment_pos_weight_schedule="uniform" \
--commitment_pos_weight_step=100.0 \
--commitment_pos_weight_base=2.0 \
--commitment_similarity="embsim-l2" \
\
--bptt=False \
--bptt_stgs_hard=False \
--bptt_logits_normalize="none" \
--bptt_learnable_temperature=False \
--bptt_hidden_state_conditioning=False \
--bptt_temperature=100.0 \
\
--init_strategy='zeros' \
--init_std=0.0 \
--init_mlm_model=distilbert-base-uncased \
--init_mlm_top_k=100 \
\
--early_stop_on_exact_match=False \
--early_stop_loss_threshold=0.0 \
--early_stop_embsim_lcs_ratio_threshold=1.0 \
\
--run_discrete_validation=False \
--run_discrete_embsim_validation=True \
--embsim_similarity="l2" \
--embsim_teacher_forcing=False \
--embsim_use_input_logits=True \
--embsim_temperature=1.0 \
\
--temperature_anneal_schedule="none" \
--temperature_anneal_min=0.05 \
--temperature_anneal_epochs=2000 \
--temperatureAnnealRegLambda=0.01 \
--temperatureAnnealRegMode="mse" \
\
--eos_reg_lambda=0.0 \
--eos_reg_schedule=linear \
--eos_reg_alpha=1.0 \
\
--discrete_reinit_epoch=0 \
--discrete_reinit_snap="embsim-l2" \
\
--prompt_length_learnable=False \
--prompt_length_alpha_init=0.0 \
--prompt_length_beta=5.0 \
--prompt_length_reg_lambda=1.0e-4 \
--prompt_length_eos_spike=10.0 \
--prompt_length_mask_eos_attention=False \
\
--fixed_gt_prefix_n=0 \
--fixed_gt_suffix_n=0 \
--fixed_gt_prefix_rank2_n=0 \
--fixed_gt_suffix_rank2_n=0 \
\
--semantic_metrics_every_n_epochs=128 \
--bertscore_model=distilbert-base-uncased \
--sentencebert_model=all-MiniLM-L6-v2 


#--target_indices="0,1" 
#--target_indices="0,5,10,15,20" \
#--losses=embxentropy \
# DEPR to same as batch_size : --stgs_grad_bias_reference_batch_size=64 \
