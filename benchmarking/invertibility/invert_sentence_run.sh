#/bin/bash

python -m ipdb -c c invert_sentences.py \
\
--sentences="set a new record for today" \
\
--model_name="Qwen/Qwen3-0.6B" \
--model_precision=full \
--gradient_checkpointing=False \
\
--method=stgs \
--epochs=4096 \
--learning_rate=0.1 \
--temperature=100.0 \
--seq_len=0 \
--batch_size=8 \
--seed=42 \
\
--losses=crossentropy \
--promptLambda=0.0 \
--complLambda=0.0 \
--promptTfComplLambda=0.0 \
--promptDistEntropyLambda=0.0 \
\
--stgs_hard=True \
--stgs_hard_method="embsim-cos" \
--stgs_hard_embsim_probs="gumbel_soft" \
--learnable_temperature=True \
--decouple_learnable_temperature=True \
--teacher_forcing=True \
--gradient_estimator=stgs \
\
--init_strategy='normal' \
--init_std=0.0 \
--init_mlm_model=distilbert-base-uncased \
--init_mlm_top_k=100 \
\
--filter_vocab=False \
--vocab_threshold=0.5 \
--logits_top_k=0 \
--logits_top_p=1.0 \
--logits_filter_penalty=1e8 \
--max_gradient_norm=0.0 \
--eps=1e-10 \
\
--early_stop_on_exact_match=False \
--early_stop_loss_threshold=0.00001 \
\
--run_discrete_validation=False \
--run_discrete_embsim_validation=True \
--embsim_similarity="l2" \
--embsim_use_input_logits=True \
\
--temperature_anneal_schedule=none \
--temperature_anneal_min=0.1 \
--temperature_anneal_epochs=0 \
\
--gumbel_noise_scale=1.0 \
--adaptive_gumbel_noise=False \
--adaptive_gumbel_noise_beta=0.9 \
--adaptive_gumbel_noise_min_scale=0.0 \
\
--eos_reg_lambda=0.0 \
--eos_reg_schedule=linear \
--eos_reg_alpha=1.0 \
\
--commitmentLambda=100.0 \
--commitment_similarity="embsim-cos" \
\
--discrete_reinit_epoch=0 \
--discrete_reinit_snap=argmax \
\
--prompt_length_learnable=False \
--prompt_length_alpha_init=0.0 \
--prompt_length_beta=5.0 \
--prompt_length_reg_lambda=0.0 \
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
--sentencebert_model=all-MiniLM-L6-v2 \
\
--plot_every=100 \
--output_file=./output_inverted_sentences.json \
--output_dir=./invert_results \

