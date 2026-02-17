python generate_dataset_with_perplexity.py \
--seed=42 \
--model_name="Qwen/Qwen3-4B" \
--bos_token="<|im_start|>" \
--eos_token="<|im_end|>" \
--output_path="data/qwen3-4b_diverse_targets_k1-5-25_x5_seed42_TL20_NF1" \
--k_min 1 \
--k_max 25 \
--k_step 5  \
--num_samples 5 \
--max_length 20 \
--noise_factor 1.0 

