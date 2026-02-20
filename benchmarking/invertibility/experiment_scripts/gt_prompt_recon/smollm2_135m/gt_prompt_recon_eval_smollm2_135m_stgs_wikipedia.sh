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
        --losses=crossentropy \
        --stgs_hard=False \
        --learnable_temperature=True \
        --decouple_learnable_temperature=True \
        --temperature=100.0 \
        --teacher_forcing=True \
        --bptt=False \
        --num_workers=1 \
        --wandb_project="gt-prompt-reconstruction-Wikipedia-DLMI"

    echo "Completed: ${OUTPUT_DIR}"
done

echo ""
echo "All evaluations completed!"
