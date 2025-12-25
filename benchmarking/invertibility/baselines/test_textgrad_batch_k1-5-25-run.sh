#!/bin/bash

python -m ipdb -c c textgrad_batch_optimize_main.py \
  --model_name distilbert/distilgpt2 \
  --dataset_path data/distilgpt2_diverse_targets_k1-5-25 \
  --output_dir results/textgrad_distilgpt2_test_k1-5-25-run \
  --seq_len 40 \
  --epochs 200 \
  --temperature 1.0 \
  --model_precision full \
  --gradient_checkpointing False \
  --filter_vocab True \
  --vocab_threshold 0.5 \
  --backward_engine experimental:openai/gpt-4o \
  --optimizer_engine experimental:openai/gpt-4o \
  --textgrad_do_sample True \
  --stop_on_done True \
  --seed 10 \
  --num_workers 1
