#!/bin/bash
# Generate GT prompt reconstruction Wikipedia dataset for SmolLM2-135M

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Configuration
MODEL_NAME="HuggingFaceTB/SmolLM2-135M"
DATASET_SOURCE="wikipedia"
NUM_SAMPLES=10 #100
SEED=42
OUTPUT_LENGTH=20 #25

# Generate datasets for different prompt lengths
#for PROMPT_LENGTH in 10 20 40 80; do
for PROMPT_LENGTH in 10 20 80; do
    OUTPUT_PATH="${PARENT_DIR}/data/soda_smollm2-135m_wikipedia_SEED${SEED}_PL${PROMPT_LENGTH}_TL${OUTPUT_LENGTH}_N${NUM_SAMPLES}.json"

    echo "Generating dataset with prompt_length=${PROMPT_LENGTH}..."
    python "${PARENT_DIR}/generate_gt_prompt_output_dataset.py" \
        --model_name="${MODEL_NAME}" \
        --dataset_source="${DATASET_SOURCE}" \
        --output_path="${OUTPUT_PATH}" \
        --prompt_length=${PROMPT_LENGTH} \
        --output_length=${OUTPUT_LENGTH} \
        --num_samples=${NUM_SAMPLES} \
        --seed=${SEED} \
        --random_sentence \
        --random_start

    echo "Saved to: ${OUTPUT_PATH}"
    echo ""
done

echo "Done generating all Wikipedia datasets for SmolLM2-135M"
