#!/bin/bash
# Partially-conditioned multi-target LMI with product-of-speaker prompt assembly.
# Model:   Qwen3-0.6B (chat template applied to x and PoS prompt)
# Loss:    crossentropy on Y tokens
# Dataset: openai/gsm8k (loaded from HuggingFace)
# Input:   [chat(x), p, R_gt, E, Y[:-1]]  — R teacher-forced from dataset CoT

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "$PARENT_DIR"

MODEL_NAME="Qwen/Qwen3-0.6B"
MODEL_KEY="qwen3-600m"

POS_LM_NAME="${POS_LM_NAME:-Qwen/Qwen3-0.6B}"
POS_PROMPT="${POS_PROMPT:-You are recovering a hidden prompt from partially conditioned input-output examples. Score the next hidden prompt token by how much it improves the explanation of all examples.}"
if [[ -z "${POS_PROMPT_TEMPLATE+x}" ]]; then
    POS_PROMPT_TEMPLATE=$'Recover the shared hidden prompt from these partially conditioned examples.\n\n{pairs}\n\nScore the next hidden prompt token that best explains all examples.'
fi
if [[ -z "${POS_PAIR_TEMPLATE+x}" ]]; then
    POS_PAIR_TEMPLATE=$'Conditioning input: {partial_conditioning_text}\nTarget: {target}'
fi
if [[ -z "${POS_PAIR_SEPARATOR+x}" ]]; then
    POS_PAIR_SEPARATOR=$'\n\n'
fi

# --- Dataset ---
TRAIN_SIZE=50
VAL_SIZE=20
TEST_SIZE=100
TRAIN_SEED=42
VAL_SEED=43
TEST_SEED=44
OPT_SEED=10

# --- Extraction ---
EXTRACTION_PROMPT="Therefore, the final answer is \$"
MAX_NEW_TOKENS_R=200
MAX_NEW_TOKENS_Y=20

# --- Optimization ---
EPOCHS=2048
LR=1.0e-1
SEQ_LEN=80
INNER_BATCH_SIZE=4
LOSSES="crossentropy"

# --- STGS (prompt p) ---
TEMPERATURE=100.0
STGS_HARD=False
STGS_HARD_METHOD="embsim-l2"
LOGITS_NORMALIZE="none"
LOGITS_LORA_RANK=32
GUMBEL_NOISE_SCALE=0.06125
INIT_STRATEGY="zeros"

# --- Validation ---
VAL_EVAL_EVERY=100
TEST_EVAL_EVERY=200
VAL_PROMPT_EVAL_MODE="discrete"
TEST_PROMPT_EVAL_MODE="discrete"

# --- Superposition metric ---
SUPERPOSITION_METRIC_EVERY=0
SUPERPOSITION_METRIC_MODES="dot,cos,l2"
SUPERPOSITION_VOCAB_TOP_K=256
SUPERPOSITION_VOCAB_SOURCE="wikipedia"
SUPERPOSITION_VOCAB_DATASET_PATH=""
SUPERPOSITION_VOCAB_HF_NAME="lucadiliello/english_wikipedia"
SUPERPOSITION_VOCAB_HF_SPLIT="train"
SUPERPOSITION_VOCAB_NUM_TEXTS=1000
SUPERPOSITION_ENTROPY_TEMPERATURE=1.0

# --- Reasoning generation backend ---
REASONING_GENERATION_BACKEND="diff"
REASONING_GENERATE_DO_SAMPLE=True
REASONING_GENERATE_TEMPERATURE=1.0
REASONING_GENERATE_TOP_P=1.0
REASONING_GENERATE_TOP_K=0
REASONING_GENERATE_NUM_BEAMS=1
REASONING_GENERATE_NUM_RETURN_SEQUENCES=1
REASONING_GENERATE_REPETITION_PENALTY=1.0
REASONING_GENERATE_LENGTH_PENALTY=1.0
REASONING_GENERATE_NO_REPEAT_NGRAM_SIZE=0
REASONING_GENERATE_EARLY_STOPPING=False

RUN_NAME="${MODEL_KEY}_pc_mt_tf_pos_gsm8k_SL${SEQ_LEN}_EP${EPOCHS}_LR${LR}_SEED${OPT_SEED}"
OUTPUT_DIR="results/pc_mt/${RUN_NAME}"
SUPERPOSITION_OUTPUT_DIR="${OUTPUT_DIR}/superposition"

POS_ARGS=(
    --assemble_strategy product_of_speaker
    --pos_lm_name "${POS_LM_NAME}"
    --pos_prompt "${POS_PROMPT}"
    --pos_prompt_template "${POS_PROMPT_TEMPLATE}"
    --pos_pair_template "${POS_PAIR_TEMPLATE}"
    --pos_pair_separator "${POS_PAIR_SEPARATOR}"
    --pos_use_chat_template True
    --pos_lm_temperature 1.0
    --pos_lm_top_p 1.0
    --pos_lm_top_k 0
)

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
echo "PC multi-target LMI — teacher-forcing R + PoS"
echo "  Model:    ${MODEL_NAME}"
echo "  seq_len:  ${SEQ_LEN}"
echo "  epochs:   ${EPOCHS}"
echo "  lr:       ${LR}"
echo "  batch:    ${INNER_BATCH_SIZE}"
echo "  output:   ${OUTPUT_DIR}"
echo "=============================================="

python batch_optimize_pc_main.py \
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
    --efficient_generate True \
    --teacher_forcing_r True \
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
    --learnable_temperature True \
    --decouple_learnable_temperature True \
    --stgs_hard ${STGS_HARD} \
    --stgs_hard_method ${STGS_HARD_METHOD} \
    --logits_normalize ${LOGITS_NORMALIZE} \
    --logits_lora_rank ${LOGITS_LORA_RANK} \
    --stgs_input_dropout 0.0 \
    --stgs_output_dropout 0.0 \
    --gumbel_noise_scale ${GUMBEL_NOISE_SCALE} \
    --adaptive_gumbel_noise False \
    --init_strategy ${INIT_STRATEGY} \
    --init_std 0.0 \
    \
    --max_gradient_norm 1.0 \
    --logit_decay 0.0 \
    --discrete_reinit_epoch 0 \
    --temperature_anneal_schedule none \
    \
    --diff_model_temperature 1.0 \
    --diff_model_hard True \
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
    --wandb_tags "qwen3-600m,teacher-forcing,pos,chat-template,gsm8k" \
    --weave_project "sdlm-pc-mt-lmi" \
    "${POS_ARGS[@]}" \
    "${SUPERPOSITION_ARGS[@]}"

echo "Done: ${OUTPUT_DIR}"
