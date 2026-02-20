#!/usr/bin/env bash
# Test 5: Save output to a markdown file with --output
cd "$(dirname "$0")/../.."

PYTHON=./benchmarking/invertibility/venv/bin/python
SCRIPT=scripts/wandb_hyperparam_table_generator_v2.py
CONFIG=scripts/SmolLM3-3B-2048-GSvsREINFORCEvsBaseline-ablation-LCS.yaml
TARGET_SCRIPT=benchmarking/invertibility/batch_optimize_main.py
OUTPUT=scripts/hyperparams_examples/output_table.md

echo "=== Save to file ==="
$PYTHON $SCRIPT -c $CONFIG -s $TARGET_SCRIPT -o $OUTPUT

echo ""
echo "First 5 lines of output:"
head -5 "$OUTPUT"
echo ""
echo "Saved to: $OUTPUT"
