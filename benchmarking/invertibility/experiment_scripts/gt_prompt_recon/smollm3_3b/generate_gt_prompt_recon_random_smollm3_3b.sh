#!/bin/bash
# Generate GT prompt reconstruction random token dataset for SmolLM3-3B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Configuration
MODEL_NAME="HuggingFaceTB/SmolLM3-3B-Base"
MODEL_KEY="smollm3-3b"
DATASET_SOURCE="random"
NUM_SAMPLES=100
SEED=42
OUTPUT_LENGTH=25

# Generate datasets for different prompt lengths
for PROMPT_LENGTH in 10 20 40 80; do
    OUTPUT_PATH="${PARENT_DIR}/data/soda_${MODEL_KEY}_${DATASET_SOURCE}_PL${PROMPT_LENGTH}_OL${OUTPUT_LENGTH}_N${NUM_SAMPLES}.json"

    echo "Generating dataset with prompt_length=${PROMPT_LENGTH}..."
    python "${PARENT_DIR}/generate_gt_prompt_output_dataset.py" \
        --model_name="${MODEL_NAME}" \
        --dataset_source="${DATASET_SOURCE}" \
        --output_path="${OUTPUT_PATH}" \
        --prompt_length=${PROMPT_LENGTH} \
        --output_length=${OUTPUT_LENGTH} \
        --num_samples=${NUM_SAMPLES} \
        --seed=${SEED}

    echo "Saved to: ${OUTPUT_PATH}"
    echo ""
done

echo "Done generating all random datasets for SmolLM3-3B"
