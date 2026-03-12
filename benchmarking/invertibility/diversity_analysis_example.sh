#!/usr/bin/env bash
set -euo pipefail

# Example runner for benchmarking/invertibility/diversity_analysis.py
#
# This script exposes every active CLI argument in diversity_analysis.py,
# including STGS controls, while intentionally excluding learnable/decoupled
# temperature options.

MODEL_ID="${MODEL_ID-HuggingFaceTB/SmolLM2-135M}"
LOGITS_SOURCE="${LOGITS_SOURCE:-model}"      # random | model | /path/to/logits.pt
SEQ_LEN="${SEQ_LEN:-10}"                      # used for random/model sources
N_SAMPLES="${N_SAMPLES:-100}"                 # sampled sequences per temperature
TEMPERATURE="${TEMPERATURE:-1.0}"             # used when TEMPERATURES is empty
TEMPERATURES="${TEMPERATURES:-0.0001,0.001,0.01,0.1,0.5,1.0,2.0,4.0,10.0}"
TOP_K="${TOP_K:-8}"                           # report heatmap rows
OUTPUT_DIR="${OUTPUT_DIR:-./results/diversity-GumbelNoise006125+embsim-l2+topk_sample}"

# STGS-related controls (naming aligned with batch_optimize_main.py)
GRADIENT_ESTIMATOR="${GRADIENT_ESTIMATOR:-stgs}"            # stgs | reinforce
STGS_HARD="${STGS_HARD:-True}"
STGS_HARD_METHOD="${STGS_HARD_METHOD:-embsim-l2}"         # categorical | embsim-dot | embsim-cos | embsim-l2
STGS_HARD_EMBSIM_PROBS="${STGS_HARD_EMBSIM_PROBS:-gumbel_soft}"  # gumbel_soft | input_logits
STGS_HARD_EMBSIM_STRATEGY="${STGS_HARD_EMBSIM_STRATEGY:-topk_sample}" # nearest | topk_rerank | topk_sample | margin_fallback | lm_topk_restrict
STGS_HARD_EMBSIM_TOP_K="${STGS_HARD_EMBSIM_TOP_K:-8}"
STGS_HARD_EMBSIM_RERANK_ALPHA="${STGS_HARD_EMBSIM_RERANK_ALPHA:-0.5}"
STGS_HARD_EMBSIM_SAMPLE_TAU="${STGS_HARD_EMBSIM_SAMPLE_TAU:-1.0}"
STGS_HARD_EMBSIM_MARGIN="${STGS_HARD_EMBSIM_MARGIN:-0.0}"
STGS_HARD_EMBSIM_FALLBACK="${STGS_HARD_EMBSIM_FALLBACK:-categorical}"  # argmax | categorical
EPS="${EPS:-1e-10}"
GUMBEL_NOISE_SCALE="${GUMBEL_NOISE_SCALE:-0.06125}"
ADAPTIVE_GUMBEL_NOISE="${ADAPTIVE_GUMBEL_NOISE:-False}"
ADAPTIVE_GUMBEL_NOISE_BETA="${ADAPTIVE_GUMBEL_NOISE_BETA:-0.9}"
ADAPTIVE_GUMBEL_NOISE_MIN_SCALE="${ADAPTIVE_GUMBEL_NOISE_MIN_SCALE:-0.0}"
STGS_INPUT_DROPOUT="${STGS_INPUT_DROPOUT:-0.0}"
STGS_OUTPUT_DROPOUT="${STGS_OUTPUT_DROPOUT:-0.0}"
STGS_SNR_EMPIRICAL="${STGS_SNR_EMPIRICAL:-False}"          # True | False
LOGITS_NORMALIZE="${LOGITS_NORMALIZE:-zscore}"               # none | center | zscore
LOGITS_TOP_K="${LOGITS_TOP_K:-0}"                          # 0 disables
LOGITS_TOP_P="${LOGITS_TOP_P:-1.0}"                        # 1.0 disables
LOGITS_FILTER_PENALTY="${LOGITS_FILTER_PENALTY:-1e4}"
LOGIT_DECAY="${LOGIT_DECAY:-0.0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/diversity_analysis.py"

if [[ "${LOGITS_SOURCE}" != "random" && "${LOGITS_SOURCE}" != "model" && ! -f "${LOGITS_SOURCE}" ]]; then
  echo "[error] LOGITS_SOURCE must be 'random', 'model', or an existing .pt file path: ${LOGITS_SOURCE}" >&2
  exit 1
fi
if [[ "${LOGITS_SOURCE}" == "model" && -z "${MODEL_ID}" ]]; then
  echo "[error] MODEL_ID must be non-empty when LOGITS_SOURCE=model" >&2
  exit 1
fi

MODEL_ARG=()
if [[ -n "${MODEL_ID}" ]]; then
  MODEL_ARG=(--model "${MODEL_ID}")
fi

echo "Running diversity analysis with:"
echo "  model                 : ${MODEL_ID:-<dummy tokenizer>}"
echo "  logits_source         : ${LOGITS_SOURCE}"
echo "  seq_len               : ${SEQ_LEN}"
echo "  n_samples             : ${N_SAMPLES}"
echo "  temperature           : ${TEMPERATURE}"
echo "  temperatures          : ${TEMPERATURES}"
echo "  output_dir            : ${OUTPUT_DIR}"
echo "  gradient_estimator    : ${GRADIENT_ESTIMATOR}"
echo "  stgs_hard_method      : ${STGS_HARD_METHOD}"
echo "  embsim_strategy       : ${STGS_HARD_EMBSIM_STRATEGY}"
echo "  logits_normalize      : ${LOGITS_NORMALIZE}"

echo
python "${PY_SCRIPT}" \
  "${MODEL_ARG[@]}" \
  --logits_source "${LOGITS_SOURCE}" \
  --seq_len "${SEQ_LEN}" \
  --n_samples "${N_SAMPLES}" \
  --temperature "${TEMPERATURE}" \
  --temperatures "${TEMPERATURES}" \
  --top_k "${TOP_K}" \
  --output_dir "${OUTPUT_DIR}" \
  --gradient_estimator "${GRADIENT_ESTIMATOR}" \
  --stgs_hard "${STGS_HARD}" \
  --stgs_hard_method "${STGS_HARD_METHOD}" \
  --stgs_hard_embsim_probs "${STGS_HARD_EMBSIM_PROBS}" \
  --stgs_hard_embsim_strategy "${STGS_HARD_EMBSIM_STRATEGY}" \
  --stgs_hard_embsim_top_k "${STGS_HARD_EMBSIM_TOP_K}" \
  --stgs_hard_embsim_rerank_alpha "${STGS_HARD_EMBSIM_RERANK_ALPHA}" \
  --stgs_hard_embsim_sample_tau "${STGS_HARD_EMBSIM_SAMPLE_TAU}" \
  --stgs_hard_embsim_margin "${STGS_HARD_EMBSIM_MARGIN}" \
  --stgs_hard_embsim_fallback "${STGS_HARD_EMBSIM_FALLBACK}" \
  --eps "${EPS}" \
  --gumbel_noise_scale "${GUMBEL_NOISE_SCALE}" \
  --adaptive_gumbel_noise "${ADAPTIVE_GUMBEL_NOISE}" \
  --adaptive_gumbel_noise_beta "${ADAPTIVE_GUMBEL_NOISE_BETA}" \
  --adaptive_gumbel_noise_min_scale "${ADAPTIVE_GUMBEL_NOISE_MIN_SCALE}" \
  --stgs_input_dropout "${STGS_INPUT_DROPOUT}" \
  --stgs_output_dropout "${STGS_OUTPUT_DROPOUT}" \
  --stgs_snr_empirical "${STGS_SNR_EMPIRICAL}" \
  --logits_normalize "${LOGITS_NORMALIZE}" \
  --logits_top_k "${LOGITS_TOP_K}" \
  --logits_top_p "${LOGITS_TOP_P}" \
  --logits_filter_penalty "${LOGITS_FILTER_PENALTY}" \
  --logit_decay "${LOGIT_DECAY}"
