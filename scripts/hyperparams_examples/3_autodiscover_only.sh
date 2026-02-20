#!/usr/bin/env bash
# Test 3: Auto-discover hyperparameters from W&B run configs (no --script)
cd "$(dirname "$0")/../.."

PYTHON=./benchmarking/invertibility/venv/bin/python
SCRIPT=scripts/wandb_hyperparam_table_generator_v2.py
CONFIG=scripts/SmolLM3-3B-2048-GSvsREINFORCEvsBaseline-ablation-LCS.yaml

echo "=== Auto-discover only (no --script) ==="
$PYTHON $SCRIPT -c $CONFIG
