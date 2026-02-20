#!/bin/bash
# Fixed-distribution evaluation — Mode 3: Text prefix / suffix
#
# A user-supplied text string is tokenised and inserted as a fixed
# one-hot prefix or suffix (logit = +100 at the text token, -100 elsewhere).
# The remaining (seq_len - len(text_tokens)) positions are gradient-optimised.
#
# Because seq_len = total prompt length, setting seq_len = PL keeps the
# same total length as the GT prompt: e.g. PL=20 with a 2-token prefix
# leaves 18 free positions.
#
# Token counts for SmolLM2-135M tokeniser (used below):
#   "In"                     ->  1 token
#   "According to"           ->  2 tokens
#   "Wikipedia states that"  ->  3 tokens
#
# Usage: bash experiment_scripts/gt_prompt_recon/smollm2_135m/fixed_dist_text_affixes_smollm2_135m_wikipedia.sh

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
BATCH_SIZE=8

# ---------------------------------------------------------------------------
# Experiment definitions — one entry per row in each parallel array:
#   PROMPT_LENGTHS  : total seq_len (free = seq_len - |prefix_tokens| - |suffix_tokens|)
#   DATASET_TAGS    : filename segment for the dataset
#   PREFIX_TEXTS    : fixed prefix string (empty string = disabled)
#   SUFFIX_TEXTS    : fixed suffix string (empty string = disabled)
# ---------------------------------------------------------------------------
PROMPT_LENGTHS=(10    20             20                       20             80)
DATASET_TAGS=(  "PL10_OL20" "PL20_TL20" "PL20_TL20" "PL20_TL20" "PL80_OL20")
PREFIX_TEXTS=(  "In"  "According to" "Wikipedia states that"  ""             "According to")
SUFFIX_TEXTS=(  ""    ""             ""                       "According to" "According to")

N_EXPERIMENTS=${#PROMPT_LENGTHS[@]}

for (( i=0; i<N_EXPERIMENTS; i++ )); do
    PROMPT_LENGTH="${PROMPT_LENGTHS[$i]}"
    DATASET_TAG="${DATASET_TAGS[$i]}"
    PREFIX_TEXT="${PREFIX_TEXTS[$i]}"
    SUFFIX_TEXT="${SUFFIX_TEXTS[$i]}"

    DATASET_PATH="${PARENT_DIR}/data/soda_${MODEL_KEY}_${DATASET_SOURCE}_SEED${DATASET_SEED}_${DATASET_TAG}_N10.json"

    if [ ! -f "$DATASET_PATH" ]; then
        echo "Dataset not found: ${DATASET_PATH} — skipping."
        continue
    fi

    # Build output-dir slug
    PREFIX_SLUG=$(echo "${PREFIX_TEXT}" | tr ' ' '_' | tr -cd '[:alnum:]_')
    SUFFIX_SLUG=$(echo "${SUFFIX_TEXT}" | tr ' ' '_' | tr -cd '[:alnum:]_')
    if [ -n "$PREFIX_SLUG" ] && [ -n "$SUFFIX_SLUG" ]; then
        AFFIX_SLUG="pfx_${PREFIX_SLUG}_sfx_${SUFFIX_SLUG}"
    elif [ -n "$PREFIX_SLUG" ]; then
        AFFIX_SLUG="pfx_${PREFIX_SLUG}"
    else
        AFFIX_SLUG="sfx_${SUFFIX_SLUG}"
    fi

    OUTPUT_DIR="results/fixed_dist_text_${AFFIX_SLUG}_${MODEL_KEY}_${DATASET_SOURCE}_PL${PROMPT_LENGTH}_EP${EPOCHS}_LR${LR}_SEED${SEED}"

    echo ""
    echo "=============================================="
    echo " Fixed-distribution: text affix  (exp $((i+1))/${N_EXPERIMENTS})"
    echo "  Model   : ${MODEL_NAME}"
    echo "  Dataset : ${DATASET_PATH}"
    echo "  seq_len : ${PROMPT_LENGTH}"
    echo "  prefix  : '${PREFIX_TEXT}'"
    echo "  suffix  : '${SUFFIX_TEXT}'"
    echo "  Output  : ${OUTPUT_DIR}"
    echo "=============================================="

    # Build optional flags (only pass if non-empty)
    EXTRA_ARGS=""
    [ -n "$PREFIX_TEXT" ] && EXTRA_ARGS="${EXTRA_ARGS} --fixed_prefix_text=${PREFIX_TEXT}"
    [ -n "$SUFFIX_TEXT" ] && EXTRA_ARGS="${EXTRA_ARGS} --fixed_suffix_text=${SUFFIX_TEXT}"

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
        ${EXTRA_ARGS} \
        --wandb_project="fixed-dist-text-affixes-Wikipedia-SmolLM2-135M"

    echo "Completed: ${OUTPUT_DIR}"
done

echo ""
echo "All text-affix fixed-distribution evaluations completed!"
