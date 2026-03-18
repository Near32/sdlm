#!/bin/bash
set -euo pipefail

# Override via environment variables if needed.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_PRECISION="${MODEL_PRECISION:-full}"

DATASET_PATH="${DATASET_PATH:-../../../data/qwen3-600m-base_diverse_targets_k1-5-25_x5_seed42_TL20_NF1}"
#LEARNABLE_LOGITS_ROOT="${LEARNABLE_LOGITS_ROOT:-results/qwen3-600m-base_STGS+Soft+LearnTau+SoftBPTT+LearnBTau+BS=16+LR=1e-1+SEED=2_test_k1-5-25-run/1mkp9t21}"
LEARNABLE_LOGITS_ROOT="${LEARNABLE_LOGITS_ROOT:-results/qwen3-600m-base_STGS+Soft+LearnTau+SoftBPTT+LearnBTau+BS=16+LR=1e-1+SEED=2_test_k1-5-25-run/qqd01a5k}"
LEARNABLE_TEMPERATURES_ROOT="${LEARNABLE_TEMPERATURES_ROOT:-${LEARNABLE_LOGITS_ROOT}}"
FALLBACK_LEARNABLE_TEMPERATURES="${FALLBACK_LEARNABLE_TEMPERATURES:-15.43}"
TARGET_INDICES="${TARGET_INDICES:-0}"   # e.g. "0,1,5" ; empty means all dataset samples

TOP_P_VALUES="${TOP_P_VALUES:-1.0,0.99,0.98,0.97,0.96,0.95,0.9,0.8}"
#OUTPUT_DIR="${OUTPUT_DIR:-results/qwen3-600m-base_top-p-lcs-sweep/1mkp9t21}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qwen3-600m-base_top-p-lcs-sweep/qqd01a5k}"

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
TEACHER_FORCING="${TEACHER_FORCING:-True}"
BPTT_TEACHER_FORCING_VIA_DIFF_MODEL="${BPTT_TEACHER_FORCING_VIA_DIFF_MODEL:-False}"
STGS_HARD="${STGS_HARD:-False}"
SUPERPOSITION_METRIC_EVERY="${SUPERPOSITION_METRIC_EVERY:-1}"
SUPERPOSITION_METRIC_MODES="${SUPERPOSITION_METRIC_MODES:-dot,cos,l2}"
SUPERPOSITION_VOCAB_TOP_K="${SUPERPOSITION_VOCAB_TOP_K:-256}"
SUPERPOSITION_VOCAB_SOURCE="${SUPERPOSITION_VOCAB_SOURCE:-wikipedia}"
SUPERPOSITION_VOCAB_DATASET_PATH="${SUPERPOSITION_VOCAB_DATASET_PATH:-}"
SUPERPOSITION_VOCAB_HF_NAME="${SUPERPOSITION_VOCAB_HF_NAME:-lucadiliello/english_wikipedia}"
SUPERPOSITION_VOCAB_HF_SPLIT="${SUPERPOSITION_VOCAB_HF_SPLIT:-train}"
SUPERPOSITION_VOCAB_NUM_TEXTS="${SUPERPOSITION_VOCAB_NUM_TEXTS:-1000}"
SUPERPOSITION_ENTROPY_TEMPERATURE="${SUPERPOSITION_ENTROPY_TEMPERATURE:-1.0}"

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
  --stgs_hard "${STGS_HARD}"
  --superposition_metric_every "${SUPERPOSITION_METRIC_EVERY}"
  --superposition_metric_modes "${SUPERPOSITION_METRIC_MODES}"
  --superposition_vocab_top_k "${SUPERPOSITION_VOCAB_TOP_K}"
  --superposition_vocab_source "${SUPERPOSITION_VOCAB_SOURCE}"
  --superposition_vocab_hf_name "${SUPERPOSITION_VOCAB_HF_NAME}"
  --superposition_vocab_hf_split "${SUPERPOSITION_VOCAB_HF_SPLIT}"
  --superposition_vocab_num_texts "${SUPERPOSITION_VOCAB_NUM_TEXTS}"
  --superposition_entropy_temperature "${SUPERPOSITION_ENTROPY_TEMPERATURE}"
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

if [[ -n "${SUPERPOSITION_VOCAB_DATASET_PATH}" ]]; then
  CMD+=(--superposition_vocab_dataset_path "${SUPERPOSITION_VOCAB_DATASET_PATH}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
