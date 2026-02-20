#!/usr/bin/env bash
# Test 1: Verify the script parses without syntax errors
cd "$(dirname "$0")/../.."

PYTHON=./benchmarking/invertibility/venv/bin/python
SCRIPT=scripts/wandb_hyperparam_table_generator_v2.py

echo "=== Syntax check ==="
$PYTHON -c "
import ast
with open('$SCRIPT') as f:
    ast.parse(f.read())
print('Syntax OK')
"
