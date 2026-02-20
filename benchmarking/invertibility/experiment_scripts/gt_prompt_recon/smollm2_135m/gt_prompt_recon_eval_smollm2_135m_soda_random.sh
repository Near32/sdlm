#!/bin/bash
# GT prompt reconstruction evaluation
# Model: SmolLM2-135M
# Method: SODA (baseline)
# Dataset: Random tokens

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "$PARENT_DIR"

MODEL_NAME="HuggingFaceTB/SmolLM2-135M"
MODEL_KEY="smollm2-135m"
METHOD="soda"
DATASET_SOURCE="random"
DATASET_SEED=42
SEED=20
#EPOCHS=100000  # SODA uses many epochs with early stopping
EPOCHS=256  # SODA uses many epochs with early stopping
LR=0.065       # SODA default learning rate

BATCH_SIZE=1   # SODA processes one at a time

# SODA-specific parameters
SODA_DECAY_RATE=0.9
SODA_BETA1=0.9
SODA_BETA2=0.995
SODA_RESET_EPOCH=50
SODA_REINIT_EPOCH=1500
SODA_TEMPERATURE=0.05

#for PROMPT_LENGTH in 10 20 40 80; do
for PROMPT_LENGTH in 10 ; do
    #DATASET_PATH="data/soda_${MODEL_KEY}_${DATASET_SOURCE}_PL${PROMPT_LENGTH}_OL25_N100.json"
    DATASET_PATH="${PARENT_DIR}/data/soda_${MODEL_KEY}_${DATASET_SOURCE}_SEED${DATASET_SEED}_PL${PROMPT_LENGTH}_TL20_N10.json"
    OUTPUT_DIR="results/soda_eval_${MODEL_KEY}_${METHOD}_${DATASET_SOURCE}_SEED${DATASET_SEED}_PL${PROMPT_LENGTH}_EP${EPOCHS}_LR${LR}_SEED${SEED}"

    if [ ! -f "$DATASET_PATH" ]; then
        echo "Dataset not found: ${DATASET_PATH}"
        echo "Please run generate_gt_prompt_recon_random_smollm2_135m.sh first"
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
        --temperature=${SODA_TEMPERATURE} \
        --soda_decay_rate=${SODA_DECAY_RATE} \
        --soda_beta1=${SODA_BETA1} \
        --soda_beta2=${SODA_BETA2} \
        --soda_reset_epoch=${SODA_RESET_EPOCH} \
        --soda_reinit_epoch=${SODA_REINIT_EPOCH} \
        --soda_bias_correction=False \
        --soda_init_strategy=zeros \
        --baseline_backend=hf \
        --num_workers=1 \
        --wandb_project="gt-prompt-reconstruction"

    echo "Completed: ${OUTPUT_DIR}"
done

echo ""
echo "All evaluations completed!"
