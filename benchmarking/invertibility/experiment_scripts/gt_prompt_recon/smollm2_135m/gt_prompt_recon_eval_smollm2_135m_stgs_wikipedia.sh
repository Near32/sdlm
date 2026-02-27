#!/bin/bash
# GT prompt reconstruction evaluation
# Model: SmolLM2-135M
# Method: STGS
# Dataset: Wikipedia

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "$PARENT_DIR"

MODEL_NAME="HuggingFaceTB/SmolLM2-135M"
MODEL_KEY="smollm2-135m"
METHOD="stgs"
DATASET_SOURCE="wikipedia"
DATASET_SEED=42
SEED=10
EPOCHS=2048
LR=0.1

BATCH_SIZE=8 #32

#for PROMPT_LENGTH in 10 20 40 80; do
for PROMPT_LENGTH in 20 ; do
    DATASET_PATH="${PARENT_DIR}/data/soda_${MODEL_KEY}_${DATASET_SOURCE}_SEED${DATASET_SEED}_PL${PROMPT_LENGTH}_TL20_N10.json"
    OUTPUT_DIR="results/soda_eval_${MODEL_KEY}_${METHOD}_${DATASET_SOURCE}_SEED${DATASET_SEED}_PL${PROMPT_LENGTH}_EP${EPOCHS}_LR${LR}_SEED${SEED}"

    if [ ! -f "$DATASET_PATH" ]; then
        echo "Dataset not found: ${DATASET_PATH}"
        echo "Please run generate_gt_prompt_recon_wikipedia_smollm2_135m.sh first"
        continue
    fi

    echo ""
    echo "=============================================="
    echo "Running GT prompt reconstruction evaluation"
    echo "  Model: ${MODEL_NAME}"
    echo "  Method: ${METHOD}"
    echo "  Dataset: ${DATASET_PATH}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "=============================================="

    python batch_optimize_main.py \
        --model_name="${MODEL_NAME}" \
        --dataset_path="${DATASET_PATH}" \
        --output_dir="${OUTPUT_DIR}" \
        --method="${METHOD}" \
        --learning_rate=${LR} \
        --epochs=${EPOCHS} \
        --batch_size=${BATCH_SIZE} \
        --seq_len=${PROMPT_LENGTH} \
        --seed=${SEED} \
        --model_precision=full \
        --gradient_checkpointing=False \
        --num_workers=1 \
        --wandb_project="gt-prompt-reconstruction-Wikipedia-DLMI" \
	\
	--temperature=100.0 \
	\
	--losses=crossentropy \
	--promptLambda=0.0 \
	--complLambda=0.0 \
	--promptTfComplLambda=0.0 \
	--promptDistEntropyLambda=0.0 \
	\
	--stgs_hard=True \
	--stgs_hard_method="embsim-l2" \
	--stgs_hard_embsim_probs="gumbel_soft" \
	--learnable_temperature=True \
	--decouple_learnable_temperature=True \
	--teacher_forcing=True \
	--gradient_estimator=stgs \
        --bptt=False \
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
	--commitmentLambda=1.0e2 \
	--commitment_pos_weight_schedule="uniform" \
	--commitment_pos_weight_step=100.0 \
	--commitment_pos_weight_base=2.0 \
	--commitment_similarity="embsim-l2" \
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

    echo "Completed: ${OUTPUT_DIR}"
done

echo ""
echo "All evaluations completed!"
