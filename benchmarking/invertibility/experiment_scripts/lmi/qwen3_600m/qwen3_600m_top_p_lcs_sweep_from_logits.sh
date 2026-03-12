#!/bin/bash
set -euo pipefail

# Override via environment variables if needed.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_PRECISION="${MODEL_PRECISION:-full}"

DATASET_PATH="${DATASET_PATH:-../../../data/qwen3-600m-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1}"
LEARNABLE_LOGITS_ROOT="${LEARNABLE_LOGITS_ROOT:-results/qwen3-600m-base_STGS+Soft+LearnTau+SoftBPTT+LearnBTau+BS=16+LR=1e-1+SEED=2_test_k1-5-25-run/1mkp9t21}"
LEARNABLE_TEMPERATURES_ROOT="${LEARNABLE_TEMPERATURES_ROOT:-${LEARNABLE_LOGITS_ROOT}}"
FALLBACK_LEARNABLE_TEMPERATURES="${FALLBACK_LEARNABLE_TEMPERATURES:-15.43}"
TARGET_INDICES="${TARGET_INDICES:-0,1}"   # e.g. "0,1,5" ; empty means all dataset samples

TOP_P_VALUES="${TOP_P_VALUES:-1.0,0.95,0.9,0.8,0.7,0.5,0.3,0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qwen3-600m-base_top-p-lcs-sweep/1mkp9t21}"

WANDB_PROJECT="${WANDB_PROJECT:-LMI-GS-top-p-sweep}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3_600m_top_p_lcs_sweep_from_logits}"

SEED="${SEED:-10}"
SEQ_LEN="${SEQ_LEN:-80}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GUMBEL_NOISE_SCALE="${GUMBEL_NOISE_SCALE:-0.06125}"

LOSSES="${LOSSES:-crossentropy}"
FILTER_VOCAB="${FILTER_VOCAB:-False}"
VOCAB_THRESHOLD="${VOCAB_THRESHOLD:-0.5}"

RUN_DISCRETE_VALIDATION="${RUN_DISCRETE_VALIDATION:-True}"
RUN_DISCRETE_EMBSIM_VALIDATION="${RUN_DISCRETE_EMBSIM_VALIDATION:-True}"
EMBSIM_SIMILARITY="${EMBSIM_SIMILARITY:-l2}"
EMBSIM_USE_INPUT_LOGITS="${EMBSIM_USE_INPUT_LOGITS:-True}"
EMBSIM_TEMPERATURE="${EMBSIM_TEMPERATURE:-1.0}"
TEACHER_FORCING="${TEACHER_FORCING:-False}"
BPTT_TEACHER_FORCING_VIA_DIFF_MODEL="${BPTT_TEACHER_FORCING_VIA_DIFF_MODEL:-False}"

CMD=(
  python ../../../top_p_lcs_sweep.py
  --model_name "${MODEL_NAME}"
  --model_precision "${MODEL_PRECISION}"
  --dataset_path "${DATASET_PATH}"
  --learnable_logits_root "${LEARNABLE_LOGITS_ROOT}"
  --learnable_temperatures_root "${LEARNABLE_TEMPERATURES_ROOT}"
  --top_p_values "${TOP_P_VALUES}"
  --output_dir "${OUTPUT_DIR}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --seed "${SEED}"
  --seq_len "${SEQ_LEN}"
  --learning_rate "${LEARNING_RATE}"
  --batch_size "${BATCH_SIZE}"
  --gumbel_noise_scale "${GUMBEL_NOISE_SCALE}"
  --losses "${LOSSES}"
  --filter_vocab "${FILTER_VOCAB}"
  --vocab_threshold "${VOCAB_THRESHOLD}"
  --run_discrete_validation "${RUN_DISCRETE_VALIDATION}"
  --run_discrete_embsim_validation "${RUN_DISCRETE_EMBSIM_VALIDATION}"
  --embsim_similarity "${EMBSIM_SIMILARITY}"
  --embsim_use_input_logits "${EMBSIM_USE_INPUT_LOGITS}"
  --embsim_temperature "${EMBSIM_TEMPERATURE}"
  --teacher_forcing "${TEACHER_FORCING}"
  --bptt_teacher_forcing_via_diff_model "${BPTT_TEACHER_FORCING_VIA_DIFF_MODEL}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi

if [[ -n "${TARGET_INDICES}" ]]; then
  CMD+=(--target_indices "${TARGET_INDICES}")
fi

if [[ -n "${FALLBACK_LEARNABLE_TEMPERATURES}" ]]; then
  CMD+=(--fallback_learnable_temperatures "${FALLBACK_LEARNABLE_TEMPERATURES}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
