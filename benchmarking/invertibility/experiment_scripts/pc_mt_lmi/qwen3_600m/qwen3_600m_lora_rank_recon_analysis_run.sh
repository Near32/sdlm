#!/bin/bash
# LoRA rank vs. reconstruction quality sweep — Qwen3-0.6B
#
# Sweeps over LoRA ranks and measures how faithfully the rank-r SVD of a
# one-hot logit matrix for INITIAL_PROMPT_TEXT reproduces the target tokens.
#
# Three reconstruction methods are evaluated at each rank:
#   argmax    — greedy argmax of lora_A @ lora_B
#   embsim-l2  — nearest vocabulary embedding by L2 distance
#   embsim-cos — nearest vocabulary embedding by cosine similarity
#
# Metrics logged per (rank, method):
#   lcs_ratio   — LCS(reconstructed, target) / len(target)
#   is_exact    — 1 if token sequence matches perfectly, else 0
#   recon_error — Frobenius norm ||M_onehot - lora_A @ lora_B||_F
#
# Results are saved as a JSON file and logged to W&B as a Table +
# per-method line series (rank on x-axis).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# analysis script lives two levels up from experiment_scripts/pc_mt_lmi/qwen3_600m/
INVERTIBILITY_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "$INVERTIBILITY_DIR"

PYTHON="${INVERTIBILITY_DIR}/venv/bin/python"

# --- Model ---
MODEL_NAME="Qwen/Qwen3-0.6B"
MODEL_KEY="qwen3-600m"

# --- Prompt text ---
# The text to tokenize and use as the reconstruction target.
# Escape sequences (e.g. \n) are handled by the Python script.
INITIAL_PROMPT_TEXT="Let's decompose this problem into subproblems and think step-by-step.\n"

# --- Spike value ---
# Logit value placed at each target token position in the one-hot matrix.
# Should match the value used during actual optimization (default: 10.0).
SPIKE=20.0

# --- LoRA ranks to sweep ---
# Comma-separated list; the script iterates over all values.
LORA_RANKS="2,8,16,32,64,128"

# --- Reconstruction methods ---
# Subset of: argmax, embsim-l2, embsim-cos
RECONSTRUCTION_METHODS="argmax,embsim-l2,embsim-cos"

# --- Device ---
DEVICE="cuda"

# --- Output ---
RUN_NAME="${MODEL_KEY}_lora_rank_recon_analysis"
OUTPUT_DIR="results/lora_rank_analysis/${RUN_NAME}"
OUTPUT_PATH="${OUTPUT_DIR}/results.json"

# --- W&B ---
WANDB_PROJECT="sdlm-lora-rank-analysis"
# WANDB_ENTITY=""          # leave unset to use the logged-in W&B user/team
WANDB_RUN_NAME="${RUN_NAME}"
WANDB_TAGS="${MODEL_KEY},lora-rank-sweep,recon-analysis"

# ---------------------------------------------------------------------------

WANDB_ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ENTITY_ARGS=(--wandb_entity "${WANDB_ENTITY}")
fi

echo ""
echo "=============================================="
echo "LoRA rank reconstruction quality sweep"
echo "  Model:    ${MODEL_NAME}"
echo "  Prompt:   ${INITIAL_PROMPT_TEXT}"
echo "  Ranks:    ${LORA_RANKS}"
echo "  Methods:  ${RECONSTRUCTION_METHODS}"
echo "  Spike:    ${SPIKE}"
echo "  Output:   ${OUTPUT_PATH}"
echo "  W&B:      ${WANDB_PROJECT} / ${WANDB_RUN_NAME}"
echo "=============================================="
echo ""

"${PYTHON}" lora_rank_reconstruction_analysis.py \
    --model_name "${MODEL_NAME}" \
    --prompt_text "${INITIAL_PROMPT_TEXT}" \
    --lora_ranks "${LORA_RANKS}" \
    --reconstruction_methods "${RECONSTRUCTION_METHODS}" \
    --spike "${SPIKE}" \
    --output_path "${OUTPUT_PATH}" \
    --device "${DEVICE}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    --wandb_tags "${WANDB_TAGS}" \
    "${WANDB_ENTITY_ARGS[@]}"

echo ""
echo "Done: ${OUTPUT_PATH}"
