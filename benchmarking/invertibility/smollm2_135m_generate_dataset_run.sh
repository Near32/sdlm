python generate_dataset_with_perplexity.py \
--seed=42 \
--model_name="HuggingFaceTB/SmolLM2-135M" \
--bos_token="<|endoftext|>" \
--eos_token="<|endoftext|>" \
--output_path="data/smollm2-135m-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1" \
--k_min 1 \
--k_max 25 \
--k_step 5  \
--num_samples 5 \
--max_length 20 \
--noise_factor 1.0 

