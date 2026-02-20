#!/bin/bash
# Template script for generating LaTeX tables with PNG-rendered text columns (v2)
# Renders Learned Prompt, Target Output, and Sampled Output as PNG images
# to handle multilingual/special characters that LaTeX can't render.
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
OUTPUT="${PARENT_DIR}/results-SmolLM3-3B-Base/table_output_2048ep_v2.tex"

# Table settings
CAPTION="Optimization Results"
LABEL="tab:optimization_results"
TRUNCATE=0  # Max characters for text columns (0 = no truncation)

# PNG image settings
IMAGES_DIR=""       # Directory for PNG images (default: {output_dir}/table_images/)
FONT_SIZE=10         # Font size in points (rendered at DPI resolution)
IMAGE_WIDTH=2       # Image width in cm (controls both PNG pixel width and LaTeX \includegraphics)
DPI=300             # Resolution in DPI for PNG rendering

# Optional settings (comment out to disable)
MAX_ROWS=""      # Limit number of rows (e.g., 10)
RESULTS_DIR=""   # Local results directory (overrides W&B config)
DEVICE=""        # Device for model (auto-detected if empty)

# === RUN SCRIPT ===

CMD="python ${ANALYSIS_DIR}/generate_functional_lmi_latex_table_v2.py"
CMD="$CMD --dataset \"$DATASET\""
CMD="$CMD --wandb-id \"$WANDB_ID\""
CMD="$CMD --model \"$MODEL\""
CMD="$CMD --caption \"$CAPTION\""
CMD="$CMD --label \"$LABEL\""
CMD="$CMD --truncate $TRUNCATE"
CMD="$CMD --font-size $FONT_SIZE"
CMD="$CMD --image-width $IMAGE_WIDTH"
CMD="$CMD --dpi $DPI"

# Add optional arguments
[[ -n "$OUTPUT" ]] && CMD="$CMD --output \"$OUTPUT\""
[[ -n "$IMAGES_DIR" ]] && CMD="$CMD --images-dir \"$IMAGES_DIR\""
[[ -n "$MAX_ROWS" ]] && CMD="$CMD --max-rows $MAX_ROWS"
[[ -n "$RESULTS_DIR" ]] && CMD="$CMD --results-dir \"$RESULTS_DIR\""
[[ -n "$DEVICE" ]] && CMD="$CMD --device \"$DEVICE\""

echo "Running: $CMD"
eval $CMD
