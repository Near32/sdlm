#!/bin/bash
# Partially-conditioned multi-target LMI — free R generation (no teacher-forcing)
# Model:   Qwen3-0.6B (chat template applied to x)
# Loss:    crossentropy on Y tokens
# Dataset: openai/gsm8k (loaded from HuggingFace)
# Input:   Step 1: [chat(x), p] -> R_generated
#           Step 2: [chat(x), p, R_generated, E] -> Y
# Gradient flows back to p through both generation steps (bptt=False: stop-grad on R).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "$PARENT_DIR"

MODEL_NAME="Qwen/Qwen3-0.6B"
MODEL_KEY="qwen3-600m"

# --- Dataset ---
TRAIN_SIZE=100
VAL_SIZE=100
TEST_SIZE=1319
TRAIN_SEED=10
VAL_SEED=10
TEST_SEED=0
OPT_SEED=10

# --- Extraction ---
EXTRACTION_PROMPT="Therefore, the final answer (formatted \$X\$ where X is a number) is \$"
MAX_NEW_TOKENS_R=512
MAX_NEW_TOKENS_Y=4

# --- Optimization ---
EPOCHS=2048
LR=0.1
SEQ_LEN=20
# initial_prompt_text: non-empty string = tokenize and use as starting prompt, overrides SEQ_LEN;
#   empty string = ignore (use SEQ_LEN and init_strategy instead)
INITIAL_PROMPT_TEXT=""
INNER_BATCH_SIZE=8
LOSSES="crossentropy"

# --- STGS (prompt p) ---
TEMPERATURE=100.0
LEARNABLE_TEMPERATURE=True
DECOUPLE_LEARNABLE_TEMPERATURE=True
STGS_HARD=False
STGS_HARD_METHOD="embsim-l2"
LOGITS_NORMALIZE="zscore"
LOGITS_LORA_RANK=4
LOGITS_TOP_K=0
LOGITS_TOP_P=1.0
GUMBEL_NOISE_SCALE=0.06125
INIT_STRATEGY="zeros"

# --- Batched forward pass + gradient accumulation ---
# use_batched_forward_pass: True = single batched GPU forward per mini-batch (faster on GPU);
#   False = serial per-sample loop (current default, matches pre-batching behavior)
USE_BATCHED_FORWARD_PASS=True
# batched_stgs_noise_mode: "shared" = one Gumbel draw broadcast to all B samples (lower variance);
#   "independent" = one draw per sample (higher exploration)
BATCHED_STGS_NOISE_MODE="shared"
# use_batched_eval: True = single batched generate per stage for all eval pairs (faster);
#   False = serial per-sample eval loop (current default)
USE_BATCHED_EVAL=True
# gradient_accumulation_steps: N = accumulate gradients over N mini-batches before optimizer.step();
#   1 = step every mini-batch (current default); >1 increases effective batch size without extra GPU memory
GRADIENT_ACCUMULATION_STEPS=1
# val_eval_before_training: True = run one validation epoch before the first training epoch (epoch -1);
#   False = no pre-training validation (default)
VAL_EVAL_BEFORE_TRAINING=True

# --- Validation ---
VAL_EVAL_EVERY=10
TEST_EVAL_EVERY=20
VAL_PROMPT_EVAL_MODE="soft"
TEST_PROMPT_EVAL_MODE="soft"

# --- Superposition metric ---
SUPERPOSITION_METRIC_EVERY=0
SUPERPOSITION_METRIC_MODES="l2" #dot,cos,l2
SUPERPOSITION_VOCAB_TOP_K=256
SUPERPOSITION_VOCAB_SOURCE="wikipedia"
SUPERPOSITION_VOCAB_DATASET_PATH=""
SUPERPOSITION_VOCAB_HF_NAME="lucadiliello/english_wikipedia"
SUPERPOSITION_VOCAB_HF_SPLIT="train"
SUPERPOSITION_VOCAB_NUM_TEXTS=1000
SUPERPOSITION_ENTROPY_TEMPERATURE=1.0

# --- Reasoning generation backend ---
REASONING_GENERATION_BACKEND="hf_generate"
REASONING_GENERATE_DO_SAMPLE=True
REASONING_GENERATE_TEMPERATURE=0.7
REASONING_GENERATE_TOP_P=1.0
REASONING_GENERATE_TOP_K=0
REASONING_GENERATE_NUM_BEAMS=1
REASONING_GENERATE_NUM_RETURN_SEQUENCES=1
REASONING_GENERATE_REPETITION_PENALTY=1.0
REASONING_GENERATE_LENGTH_PENALTY=1.0
REASONING_GENERATE_NO_REPEAT_NGRAM_SIZE=0
REASONING_GENERATE_EARLY_STOPPING=False

RUN_NAME="${MODEL_KEY}_pc_mt_free_gsm8k_SL${SEQ_LEN}_EP${EPOCHS}_LR${LR}_SEED${OPT_SEED}"
OUTPUT_DIR="results/pc_mt/${RUN_NAME}"
SUPERPOSITION_OUTPUT_DIR="${OUTPUT_DIR}/superposition"

INITIAL_PROMPT_TEXT_ARGS=()
if [[ -n "${INITIAL_PROMPT_TEXT}" ]]; then
    INITIAL_PROMPT_TEXT_ARGS=(--initial_prompt_text "${INITIAL_PROMPT_TEXT}")
fi

SUPERPOSITION_ARGS=()
if [[ ${SUPERPOSITION_METRIC_EVERY} -gt 0 ]]; then
    SUPERPOSITION_ARGS=(
        --superposition_metric_every "${SUPERPOSITION_METRIC_EVERY}"
        --superposition_metric_modes "${SUPERPOSITION_METRIC_MODES}"
        --superposition_vocab_top_k "${SUPERPOSITION_VOCAB_TOP_K}"
        --superposition_vocab_source "${SUPERPOSITION_VOCAB_SOURCE}"
        --superposition_vocab_hf_name "${SUPERPOSITION_VOCAB_HF_NAME}"
        --superposition_vocab_hf_split "${SUPERPOSITION_VOCAB_HF_SPLIT}"
        --superposition_vocab_num_texts "${SUPERPOSITION_VOCAB_NUM_TEXTS}"
        --superposition_entropy_temperature "${SUPERPOSITION_ENTROPY_TEMPERATURE}"
        --superposition_output_dir "${SUPERPOSITION_OUTPUT_DIR}"
    )
    if [[ -n "${SUPERPOSITION_VOCAB_DATASET_PATH}" ]]; then
        SUPERPOSITION_ARGS+=(
            --superposition_vocab_dataset_path "${SUPERPOSITION_VOCAB_DATASET_PATH}"
        )
    fi
fi

echo ""
echo "=============================================="
echo "PC multi-target LMI — free R generation"
echo "  Model:    ${MODEL_NAME}"
echo "  seq_len:  ${SEQ_LEN}"
echo "  epochs:   ${EPOCHS}"
echo "  lr:       ${LR}"
echo "  batch:    ${INNER_BATCH_SIZE}"
echo "  output:   ${OUTPUT_DIR}"
echo "=============================================="

python -m ipdb -c c batch_optimize_pc_main.py \
    --hf_dataset openai/gsm8k \
    --hf_dataset_subset main \
    --train_size ${TRAIN_SIZE} \
    --val_size ${VAL_SIZE} \
    --test_size ${TEST_SIZE} \
    --train_seed ${TRAIN_SEED} \
    --val_seed ${VAL_SEED} \
    --test_seed ${TEST_SEED} \
    --opt_seed ${OPT_SEED} \
    \
    --extraction_prompt "${EXTRACTION_PROMPT}" \
    --max_new_tokens_reasoning ${MAX_NEW_TOKENS_R} \
    --max_new_tokens_answer ${MAX_NEW_TOKENS_Y} \
    \
    --efficient_generate=True \
    --teacher_forcing_r False \
    --bptt False \
    --reasoning_generation_backend ${REASONING_GENERATION_BACKEND} \
    --reasoning_generate_do_sample ${REASONING_GENERATE_DO_SAMPLE} \
    --reasoning_generate_temperature ${REASONING_GENERATE_TEMPERATURE} \
    --reasoning_generate_top_p ${REASONING_GENERATE_TOP_P} \
    --reasoning_generate_top_k ${REASONING_GENERATE_TOP_K} \
    --reasoning_generate_num_beams ${REASONING_GENERATE_NUM_BEAMS} \
    --reasoning_generate_num_return_sequences ${REASONING_GENERATE_NUM_RETURN_SEQUENCES} \
    --reasoning_generate_repetition_penalty ${REASONING_GENERATE_REPETITION_PENALTY} \
    --reasoning_generate_length_penalty ${REASONING_GENERATE_LENGTH_PENALTY} \
    --reasoning_generate_no_repeat_ngram_size ${REASONING_GENERATE_NO_REPEAT_NGRAM_SIZE} \
    --reasoning_generate_early_stopping ${REASONING_GENERATE_EARLY_STOPPING} \
    --use_chat_template True \
    \
    --losses ${LOSSES} \
    --seq_len ${SEQ_LEN} \
    --epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --inner_batch_size ${INNER_BATCH_SIZE} \
    \
    --temperature ${TEMPERATURE} \
    --learnable_temperature ${LEARNABLE_TEMPERATURE} \
    --decouple_learnable_temperature ${DECOUPLE_LEARNABLE_TEMPERATURE} \
    --stgs_hard ${STGS_HARD} \
    --stgs_hard_method ${STGS_HARD_METHOD} \
    --logits_normalize ${LOGITS_NORMALIZE} \
    --logits_lora_rank ${LOGITS_LORA_RANK} \
    --logits_top_p ${LOGITS_TOP_P} \
    --logits_top_k ${LOGITS_TOP_K} \
    --stgs_input_dropout 0.0 \
    --stgs_output_dropout 0.0 \
    --gumbel_noise_scale ${GUMBEL_NOISE_SCALE} \
    --adaptive_gumbel_noise False \
    --init_strategy ${INIT_STRATEGY} \
    --init_std 0.0 \
    \
    --max_gradient_norm 0.0 \
    --logit_decay 0.0 \
    --discrete_reinit_epoch 0 \
    --temperature_anneal_schedule none \
    \
    --diff_model_temperature 1.0 \
    --diff_model_hard True \
    \
    --use_batched_forward_pass ${USE_BATCHED_FORWARD_PASS} \
    --batched_stgs_noise_mode ${BATCHED_STGS_NOISE_MODE} \
    --use_batched_eval ${USE_BATCHED_EVAL} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --val_eval_before_training ${VAL_EVAL_BEFORE_TRAINING} \
    \
    --val_eval_every ${VAL_EVAL_EVERY} \
    --test_eval_every ${TEST_EVAL_EVERY} \
    --val_prompt_eval_mode ${VAL_PROMPT_EVAL_MODE} \
    --test_prompt_eval_mode ${TEST_PROMPT_EVAL_MODE} \
    \
    --model_name "${MODEL_NAME}" \
    --model_precision full \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --wandb_project "sdlm-pc-mt-lmi" \
    --wandb_tags "qwen3-600m,free-r,chat-template,gsm8k" \
    --weave_project "sdlm-pc-mt-lmi" \
    "${INITIAL_PROMPT_TEXT_ARGS[@]}" \
    "${SUPERPOSITION_ARGS[@]}"

echo "Done: ${OUTPUT_DIR}"
