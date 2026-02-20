#!/usr/bin/env bash
# Test 4: Auto-discover + Description column via --script
cd "$(dirname "$0")/../.."

PYTHON=./benchmarking/invertibility/venv/bin/python
SCRIPT=./scripts/wandb_hyperparam_table_generator_v2.py
CONFIG=scripts/SmolLM3-3B-2048-GSvsREINFORCEvsBaseline-ablation-LCS.yaml
TARGET_SCRIPT=benchmarking/invertibility/batch_optimize_main.py
OUTPUT=scripts/hyperparams_examples/output_table.md

echo "=== Auto-discover + Description column (with --script) ==="
$PYTHON $SCRIPT -c $CONFIG -s $TARGET_SCRIPT -o $OUTPUT
