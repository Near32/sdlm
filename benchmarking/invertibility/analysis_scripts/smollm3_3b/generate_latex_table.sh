#!/bin/bash
# Template script for generating LaTeX tables from optimization experiments
# Modify the variables below to match your experiment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PARENT_DIR"

# === CONFIGURATION ===

# Path to dataset JSON file
#DATASET="${PARENT_DIR}/data/smollm3-3b-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1"
DATASET="${PARENT_DIR}/data/smollm3-3b-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1"

# W&B run ID (format: entity/project/runs/run_id or entity/project/run_id)
WANDB_ID="/near32/PromptOptimisationSTGS/runs/gd06ixzo" #"your-entity/your-project/runs/your-run-id"

# HuggingFace model for perplexity computation
MODEL="HuggingFaceTB/SmolLM3-3B-Base"

# Output file (leave empty for stdout)
#OUTPUT="${PARENT_DIR}/results-SmolLM3-3B-Base/table_output_2048ep.tex"
OUTPUT="${PARENT_DIR}/results-SmolLM3-3B-Base/table_output_2048ep.tex"

# Table settings
CAPTION="Optimization Results"
LABEL="tab:optimization_results"
TRUNCATE=0  # Max characters for text columns

# Optional settings (comment out to disable)
MAX_ROWS=""  # Limit number of rows (e.g., 10)
RESULTS_DIR=""  # Local results directory (overrides W&B config)
DEVICE=""  # Device for model (auto-detected if empty)

# === RUN SCRIPT ===

CMD="python -m ipdb -c c ${ANALYSIS_DIR}/generate_functional_lmi_latex_table.py"
CMD="$CMD --dataset \"$DATASET\""
CMD="$CMD --wandb-id \"$WANDB_ID\""
CMD="$CMD --model \"$MODEL\""
CMD="$CMD --caption \"$CAPTION\""
CMD="$CMD --label \"$LABEL\""
CMD="$CMD --truncate $TRUNCATE"

# Add optional arguments
[[ -n "$OUTPUT" ]] && CMD="$CMD --output \"$OUTPUT\""
[[ -n "$MAX_ROWS" ]] && CMD="$CMD --max-rows $MAX_ROWS"
[[ -n "$RESULTS_DIR" ]] && CMD="$CMD --results-dir \"$RESULTS_DIR\""
[[ -n "$DEVICE" ]] && CMD="$CMD --device \"$DEVICE\""

echo "Running: $CMD"
eval $CMD
