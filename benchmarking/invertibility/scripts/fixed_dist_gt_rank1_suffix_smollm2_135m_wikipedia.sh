#!/bin/bash
# Fixed-distribution evaluation — Mode 1: GT rank-1 suffix
#
# Last N positions of the learnable prompt are held constant with the GT
# token set to rank 1 (logit = +100, all others = -100).  The remaining
# (seq_len - N) free positions are gradient-optimised as usual.
#
# Usage: bash scripts/fixed_dist_gt_rank1_suffix_smollm2_135m_wikipedia.sh
#
# The script iterates over:
#   - prompt lengths: 10, 20, 80
#   - fixed suffix sizes: 25 % and 50 % of the prompt length

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

MODEL_NAME="HuggingFaceTB/SmolLM2-135M"
MODEL_KEY="smollm2-135m"
METHOD="stgs"
DATASET_SOURCE="wikipedia"
DATASET_SEED=42
SEED=10
EPOCHS=2048
LR=0.1
BATCH_SIZE=8

# --- Dataset / suffix-size sweep ---
#
# Each entry: "PROMPT_LENGTH DATASET_FILE_TAG FIXED_N_25pct FIXED_N_50pct"
declare -a CONFIGS=(
    "10  PL10_OL20  3  5"
    "20  PL20_TL20  5  10"
    "80  PL80_OL20  20 40"
)

for CFG in "${CONFIGS[@]}"; do
    read -r PROMPT_LENGTH DATASET_TAG N_25 N_50 <<< "$CFG"

    DATASET_PATH="${PARENT_DIR}/data/soda_${MODEL_KEY}_${DATASET_SOURCE}_SEED${DATASET_SEED}_${DATASET_TAG}_N10.json"

    if [ ! -f "$DATASET_PATH" ]; then
        echo "Dataset not found: ${DATASET_PATH} — skipping."
        continue
    fi

    for FIXED_N in $N_25 $N_50; do
        OUTPUT_DIR="results/fixed_dist_gt_rank1_suffix_${MODEL_KEY}_${DATASET_SOURCE}_PL${PROMPT_LENGTH}_FN${FIXED_N}_EP${EPOCHS}_LR${LR}_SEED${SEED}"

        echo ""
        echo "=============================================="
        echo " Fixed-distribution: GT rank-1 suffix"
        echo "  Model : ${MODEL_NAME}"
        echo "  Dataset: ${DATASET_PATH}"
        echo "  seq_len=${PROMPT_LENGTH}  fixed_gt_suffix_n=${FIXED_N}  n_free=$((PROMPT_LENGTH - FIXED_N))"
        echo "  Output : ${OUTPUT_DIR}"
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
            --fixed_gt_suffix_n=${FIXED_N} \
            --wandb_project="fixed-dist-gt-rank1-suffix-Wikipedia-SmolLM2-135M"

        echo "Completed: ${OUTPUT_DIR}"
    done
done

echo ""
echo "All fixed GT rank-1 suffix evaluations completed!"
