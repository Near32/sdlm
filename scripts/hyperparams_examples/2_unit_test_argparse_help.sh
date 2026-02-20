#!/usr/bin/env bash
# Test 2: Unit test _extract_argparse_help on batch_optimize_main.py
cd "$(dirname "$0")/../.."

PYTHON=./benchmarking/invertibility/venv/bin/python
TARGET_SCRIPT=benchmarking/invertibility/batch_optimize_main.py

echo "=== Unit test: _extract_argparse_help ==="
$PYTHON -c "
import sys; sys.path.insert(0, 'scripts')
from wandb_hyperparam_table_generator_v2 import HyperparamTableGenerator
h = HyperparamTableGenerator._extract_argparse_help('$TARGET_SCRIPT')
print(f'Extracted {len(h)} argparse help entries')
for k in sorted(h)[:5]:
    print(f'  {k}: {h[k][:80]}')
print('...')
"
