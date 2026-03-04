#!/bin/bash
# Generate LaTeX tables with v3 sub-row layout for SmolLM3-3B-Base.
# Renders Learned Prompt / Target Output / (Re-)sampled Output as full-width
# PNG images, each in its own sub-row.  Rank k and Sample ID span all sub-rows
# via \multirow.  A single Metrics column holds Prompt PPL / Target PPL / LCS.
#
# Modify the variables below to match your experiment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PARENT_DIR"

# === CONFIGURATION ===

# Path to dataset JSON file
DATASET="${PARENT_DIR}/data/smollm3-3b-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1"

# W&B run ID (format: entity/project/runs/run_id or entity/project/run_id)
WANDB_ID="/near32/PromptOptimisationSTGS/runs/gd06ixzo"

# HuggingFace model for perplexity computation
MODEL="HuggingFaceTB/SmolLM3-3B-Base"

# Output file (leave empty for stdout)
OUTPUT="${PARENT_DIR}/results-SmolLM3-3B-Base/table_qual_v3.tex"

# Table settings
CAPTION="Qualitative optimization results for SmolLM3-3B-Base ;"
LABEL="tab:optimization_results_v3"
TRUNCATE=0  # Max characters for text columns (0 = no truncation)

# PNG image settings
IMAGES_DIR=""       # Directory for PNG images (default: {output_dir}/table_images/)
FONT_SIZE=10         # Font size in points
IMAGE_WIDTH=12      # PNG pixel width in cm (images fill the content column)
DPI=300             # Resolution in DPI

# Layout settings
FIXED_COL_WIDTH=6.0    # Sum of fixed column widths in cm (rank+sample+metrics+separators)
METRICS_COL_WIDTH=2.5  # Width of the Metrics column in cm

# Optional settings (comment out to disable)
MAX_ROWS=""      # Limit number of logical rows (e.g., 10)
RESULTS_DIR=""   # Local results directory (overrides W&B config)
DEVICE=""        # Device for model (auto-detected if empty)

# Re-sampling settings
RESAMPLE=true                 # Set to true to re-generate outputs from learned prompts
RESAMPLE_MAX_NEW_TOKENS=20    # Max new tokens for greedy re-sampling

# === RUN SCRIPT ===

CMD="python ${ANALYSIS_DIR}/generate_functional_lmi_latex_table_v3.py"
CMD="$CMD --dataset \"$DATASET\""
CMD="$CMD --wandb-id \"$WANDB_ID\""
CMD="$CMD --model \"$MODEL\""
CMD="$CMD --caption \"$CAPTION\""
CMD="$CMD --label \"$LABEL\""
CMD="$CMD --truncate $TRUNCATE"
CMD="$CMD --font-size $FONT_SIZE"
CMD="$CMD --image-width $IMAGE_WIDTH"
CMD="$CMD --dpi $DPI"
CMD="$CMD --fixed-col-width $FIXED_COL_WIDTH"
CMD="$CMD --metrics-col-width $METRICS_COL_WIDTH"

# Add optional arguments
[[ -n "$OUTPUT" ]] && CMD="$CMD --output \"$OUTPUT\""
[[ -n "$IMAGES_DIR" ]] && CMD="$CMD --images-dir \"$IMAGES_DIR\""
[[ -n "$MAX_ROWS" ]] && CMD="$CMD --max-rows $MAX_ROWS"
[[ -n "$RESULTS_DIR" ]] && CMD="$CMD --results-dir \"$RESULTS_DIR\""
[[ -n "$DEVICE" ]] && CMD="$CMD --device \"$DEVICE\""
[[ "$RESAMPLE" == "true" ]] && CMD="$CMD --resample"
[[ -n "$RESAMPLE_MAX_NEW_TOKENS" ]] && CMD="$CMD --resample-max-new-tokens $RESAMPLE_MAX_NEW_TOKENS"

echo "Running: $CMD"
eval $CMD
