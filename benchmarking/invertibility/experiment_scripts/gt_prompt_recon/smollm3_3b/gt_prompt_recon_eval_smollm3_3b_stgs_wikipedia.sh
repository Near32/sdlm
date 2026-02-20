#!/bin/bash
# GT prompt reconstruction evaluation
# Model: SmolLM3-3B
# Method: STGS
# Dataset: Wikipedia

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "$PARENT_DIR"

MODEL_NAME="HuggingFaceTB/SmolLM3-3B-Base"
MODEL_KEY="smollm3-3b"
METHOD="stgs"
DATASET_SOURCE="wikipedia"
SEED=42
EPOCHS=2048
LR=0.1

BATCH_SIZE=32
SEQ_LEN=80

for PROMPT_LENGTH in 10 20 40 80; do
    DATASET_PATH="data/soda_${MODEL_KEY}_${DATASET_SOURCE}_PL${PROMPT_LENGTH}_OL25_N100.json"
    OUTPUT_DIR="results/soda_eval_${MODEL_KEY}_${METHOD}_${DATASET_SOURCE}_PL${PROMPT_LENGTH}_EP${EPOCHS}_LR${LR}_SEED${SEED}"

    if [ ! -f "$DATASET_PATH" ]; then
        echo "Dataset not found: ${DATASET_PATH}"
        echo "Please run generate_gt_prompt_recon_wikipedia_smollm3_3b.sh first"
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
        --seq_len=${SEQ_LEN} \
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
        --wandb_project="gt-prompt-reconstruction"

    echo "Completed: ${OUTPUT_DIR}"
done

echo ""
echo "All evaluations completed!"
