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
TRAIN_SIZE=50
VAL_SIZE=100
TEST_SIZE=1319
TRAIN_SEED=10
VAL_SEED=10
TEST_SEED=0
OPT_SEED=10

# --- Extraction ---
EXTRACTION_PROMPT=" Therefore, the final answer (formatted as \boxed{X} where X is the final answer number) is \boxed{"
MAX_NEW_TOKENS_R=512
MAX_NEW_TOKENS_Y=4

# --- Optimization ---
EPOCHS=100
LR=0.01
SEQ_LEN=20

# --- LR schedule ---
# lr_schedule: none | cosine | linear | step | exponential
LR_SCHEDULE="none"
# lr_warmup_epochs: linear ramp from ~0 to LR over first N epochs (0 = disabled)
LR_WARMUP_EPOCHS=0
# lr_schedule_min: minimum LR for cosine/linear schedules
LR_SCHEDULE_MIN=1.0e-4
# lr_schedule_step_size: period in epochs for StepLR (ignored by other schedules)
LR_SCHEDULE_STEP_SIZE=2
# lr_schedule_gamma: multiplicative decay factor for step/exponential schedules
LR_SCHEDULE_GAMMA=0.1

# initial_prompt_text: non-empty string = tokenize and use as starting prompt, overrides SEQ_LEN;
#   empty string = ignore (use SEQ_LEN and init_strategy instead)
INITIAL_PROMPT_TEXT=""
#INITIAL_PROMPT_TEXT="Let's decompose this problem into subproblems and think step-by-step. "
# initial_prompt_lora_reconstruction: decode method used to verify the rank-r SVD approximation
#   when initial_prompt_text and logits_lora_rank > 0 are both set.
#   "argmax": greedy argmax of lora_A @ lora_B (default).
#   "embsim-l2": nearest-embedding token by L2 distance.
#   "embsim-cos": nearest-embedding token by cosine similarity.
INITIAL_PROMPT_LORA_RECONSTRUCTION="argmax"
# initial_prompt_lora_spike: logit value placed at each target token position in the
#   one-hot matrix when initializing LoRA logits from initial_prompt_text.
#   Higher values → SVD starting point closer to a hard one-hot (default: 10.0).
INITIAL_PROMPT_LORA_SPIKE=10.0
INNER_BATCH_SIZE=2
EVAL_INNER_BATCH_SIZE=64
LOSSES="crossentropy"

# --- STGS (prompt p) ---
TEMPERATURE=2.0
LEARNABLE_TEMPERATURE=True
DECOUPLE_LEARNABLE_TEMPERATURE=True
STGS_HARD=True
STGS_HARD_METHOD="categorical"
LOGITS_NORMALIZE="zscore"
LOGITS_LORA_RANK=0
LOGITS_TOP_K=0
LOGITS_TOP_P=1.0
GUMBEL_NOISE_SCALE=0.06125 #1.0
INIT_STRATEGY="zeros"

# --- SWA ---
USE_SWA=False
# SWA_START_EPOCH: leave empty to use default (75% of epochs)
SWA_START_EPOCH=""
SWA_FREQ=1

# --- Multi-Gumbel sampling ---
GUMBEL_N_SAMPLES=8

# --- Batched forward pass + gradient accumulation ---
# use_batched_forward_pass: True = single batched GPU forward per mini-batch (faster on GPU);
#   False = serial per-sample loop (current default, matches pre-batching behavior)
USE_BATCHED_FORWARD_PASS=True
# batched_stgs_noise_mode: "shared" = one Gumbel draw broadcast to all B samples (lower variance);
#   "independent" = one draw per sample (higher exploration)
BATCHED_STGS_NOISE_MODE="independent"
# use_batched_val_eval: True = single batched generate per stage for all val eval pairs (faster);
#   False = serial per-sample val eval loop (current default)
USE_BATCHED_VAL_EVAL=True
# use_batched_test_eval: True = single batched generate per stage for all test eval pairs (faster);
#   False = serial per-sample test eval loop (current default)
USE_BATCHED_TEST_EVAL=True
# gradient_accumulation_steps: N = accumulate gradients over N mini-batches before optimizer.step();
#   1 = step every mini-batch (current default); >1 increases effective batch size without extra GPU memory
GRADIENT_ACCUMULATION_STEPS=4
# val_eval_before_training: True = run one validation epoch before the first training epoch (epoch -1);
#   False = no pre-training validation (default)
VAL_EVAL_BEFORE_TRAINING=True
# test_eval_before_training: True = run one test epoch before the first training epoch (epoch -1);
#   False = no pre-training test eval (default)
TEST_EVAL_BEFORE_TRAINING=False

# --- Validation ---
VAL_EVAL_EVERY=1
TEST_EVAL_EVERY=5
VAL_PROMPT_EVAL_MODE="discrete"
TEST_PROMPT_EVAL_MODE="discrete"

# --- MAS rotational metric ---
# mas_metric_every: N = compute MAS rotational metric every N iterations (0 = disabled;
#   requires logits_lora_rank > 0)
MAS_METRIC_EVERY=0
# mas_hvp_mode: HVP method for MAS rotational metric.
#   "autograd": retains the training loss computation graph and calls create_graph=True
#               (default; ~480 MB persistent meta-graph for Qwen3-600M).
#   "jvp":      uses torch.func.vjp + jvp (no create_graph; 2 fresh forward passes,
#               ~320 MB peak freed between calls; best for teacher-forcing mode).
#               In free-generation mode the except branch fires and logs a warning.
MAS_HVP_MODE="autograd"

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
REASONING_GENERATE_DO_SAMPLE=False
REASONING_GENERATE_TEMPERATURE=0.7
REASONING_GENERATE_TOP_P=1.0
REASONING_GENERATE_TOP_K=0
REASONING_GENERATE_NUM_BEAMS=1
REASONING_GENERATE_NUM_RETURN_SEQUENCES=1
REASONING_GENERATE_REPETITION_PENALTY=1.0
REASONING_GENERATE_LENGTH_PENALTY=1.0
REASONING_GENERATE_NO_REPEAT_NGRAM_SIZE=0
REASONING_GENERATE_EARLY_STOPPING=False

# --- Offline EM ---
# Set OFFLINE_EM=True to enable EM/block-coordinate-descent prompt optimization:
#   E-step: generate and cache reasoning tokens under the current prompt.
#   M-step: optimize prompt logits against the fixed cached reasoning.
# Requires teacher_forcing_r=False and use_batched_forward_pass=True (both already set).
# OFFLINE_EM_RESAMPLE_EVERY: re-run E-step every N epochs (0 = pure offline, never resample).
# OFFLINE_EM_STGS_METHOD: how p_embeds are built from prompt logits for the E-step.
#   "soft"       = no hard selection; use the soft distribution directly as p_embeds.
#   "argmax"     = deterministic argmax token selection.
#   "embsim-l2"  = nearest-embedding (L2) hard token selection.
#   "embsim-cos" = nearest-embedding (cosine) hard token selection.
#   "embsim-dot" = nearest-embedding (dot-product) hard token selection.
# OFFLINE_EM_STGS_EMBSIM_PROBS: source distribution for E-step p_embeds.
#   "gumbel_soft"  = use Gumbel-softmax output y_soft.
#   "input_logits" = use softmax(logits/temperature) directly (no Gumbel noise).
# OFFLINE_EM_TEMPERATURE: temperature used when building E-step p_embeds.
#   "learned" = inherit the current training temperature (default).
#   <float>   = fixed override (e.g. "0.5") for the E-step only.
OFFLINE_EM=False
OFFLINE_EM_RESAMPLE_EVERY=1
OFFLINE_EM_STGS_METHOD="soft"
OFFLINE_EM_STGS_EMBSIM_PROBS="input_logits"
OFFLINE_EM_TEMPERATURE=1.0
# OFFLINE_EM_CACHE_BATCH_SIZE: mini-batch size for E-step reasoning generation.
# No gradient is computed during the E-step, so this can safely exceed INNER_BATCH_SIZE.
# 0 = fall back to INNER_BATCH_SIZE.
OFFLINE_EM_CACHE_BATCH_SIZE=16

RUN_NAME="${MODEL_KEY}_pcmt_gsm8k_SEED${OPT_SEED}"
#RUN_NAME="${MODEL_KEY}_pc_mt_free_gsm8k_SL${SEQ_LEN}_EP${EPOCHS}_LR${LR}_SEED${OPT_SEED}"
OUTPUT_DIR="results/pc_mt/${RUN_NAME}"
SUPERPOSITION_OUTPUT_DIR="${OUTPUT_DIR}/superposition"

INITIAL_PROMPT_TEXT_ARGS=()
if [[ -n "${INITIAL_PROMPT_TEXT}" ]]; then
    INITIAL_PROMPT_TEXT_ARGS=(--initial_prompt_text "${INITIAL_PROMPT_TEXT}")
fi

SWA_START_EPOCH_ARGS=()
if [[ -n "${SWA_START_EPOCH}" ]]; then
    SWA_START_EPOCH_ARGS=(--swa_start_epoch "${SWA_START_EPOCH}")
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
    --lr_schedule ${LR_SCHEDULE} \
    --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
    --lr_schedule_min ${LR_SCHEDULE_MIN} \
    --lr_schedule_step_size ${LR_SCHEDULE_STEP_SIZE} \
    --lr_schedule_gamma ${LR_SCHEDULE_GAMMA} \
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
    --mas_metric_every ${MAS_METRIC_EVERY} \
    --mas_hvp_mode ${MAS_HVP_MODE} \
    \
    --max_gradient_norm 0.0 \
    --logit_decay 0.0 \
    --discrete_reinit_epoch 0 \
    --temperature_anneal_schedule none \
    \
    --diff_model_temperature 1.0 \
    --diff_model_hard True \
    \
    --use_swa ${USE_SWA} \
    --swa_freq ${SWA_FREQ} \
    --gumbel_n_samples ${GUMBEL_N_SAMPLES} \
    \
    --use_batched_forward_pass ${USE_BATCHED_FORWARD_PASS} \
    --batched_stgs_noise_mode ${BATCHED_STGS_NOISE_MODE} \
    --use_batched_val_eval ${USE_BATCHED_VAL_EVAL} \
    --use_batched_test_eval ${USE_BATCHED_TEST_EVAL} \
    --eval_inner_batch_size ${EVAL_INNER_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --val_eval_before_training ${VAL_EVAL_BEFORE_TRAINING} \
    --test_eval_before_training ${TEST_EVAL_BEFORE_TRAINING} \
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
    --initial_prompt_lora_reconstruction "${INITIAL_PROMPT_LORA_RECONSTRUCTION}" \
    --initial_prompt_lora_spike "${INITIAL_PROMPT_LORA_SPIKE}" \
    "${SWA_START_EPOCH_ARGS[@]}" \
    --offline_em ${OFFLINE_EM} \
    --offline_em_resample_every ${OFFLINE_EM_RESAMPLE_EVERY} \
    --offline_em_stgs_method ${OFFLINE_EM_STGS_METHOD} \
    --offline_em_stgs_embsim_probs ${OFFLINE_EM_STGS_EMBSIM_PROBS} \
    --offline_em_temperature ${OFFLINE_EM_TEMPERATURE} \
    --offline_em_cache_batch_size ${OFFLINE_EM_CACHE_BATCH_SIZE} \
    "${SUPERPOSITION_ARGS[@]}"

echo "Done: ${OUTPUT_DIR}"
