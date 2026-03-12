"""
Partially-conditioned multi-target prompt inversion.

Optimizes a single shared soft prompt p across all (x, R, y) training pairs.

Generative structure:
    R <- f([x, p])          (unconstrained reasoning)
    Y <- f([x, p, R, E])    (final answer given extraction prompt E)

Loss is CE (or composable LossClass losses) on Y tokens against y.
"""

import sys
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import wandb

# Invertibility dir first so local main.py takes precedence
sys.path.insert(0, str(Path(__file__).parent))
# Project root for sdlm (installed via pip -e but add root for safety)
sys.path.insert(1, str(Path(__file__).parent.parent.parent))
# tinylora for extraction helpers
sys.path.insert(2, str(Path(__file__).parent.parent / "tinylora"))

from sdlm import STGS
from sdlm.stgs_diff_model import STGSDiffModel
from rewards import compare_answers

from main import (
    LossClass,
    initialize_learnable_inputs,
    initialize_lora_logits,
    snap_free_logits,
    snap_lora_logits,
    compute_annealed_temperature,
)

from pc_weave_logging import (
    weave_lm_train_step,
    weave_lm_eval_step,
    weave_epoch_summary,
)

logger = logging.getLogger("pc_main")


# ---------------------------------------------------------------------------
# Core forward pass
# ---------------------------------------------------------------------------

def pc_forward_pass(
    diff_model: STGSDiffModel,
    stgs_module: STGS,
    loss_instance: LossClass,
    free_logits: Tensor,            # (1, seq_len, vocab)
    embedding_weights: Tensor,      # (vocab, embed_dim) — frozen
    x_embeds: Tensor,               # (1, x_len, embed_dim)
    R_gt_embeds: Optional[Tensor],  # (1, R_len, embed_dim) or None
    E_embeds: Tensor,               # (1, E_len, embed_dim)
    Y_tokens: Tensor,               # (1, Y_len) — CE target
    teacher_forcing_r: bool,
    bptt: bool,
    max_new_tokens_reasoning: int,
    max_new_tokens_answer: int,
    gumbel_noise_scale: float = 1.0,
    accumulate_embeds: bool = False,
    tokenizer=None,
    weave_extras: Optional[dict] = None,
) -> Tensor:
    """
    Compute the loss for one (x, R_gt, y) pair.
    Gradients flow back to free_logits through p_embeds.
    Returns a scalar loss tensor.
    """
    Y_len = Y_tokens.shape[1]

    # Step 0: get soft prompt embeddings via STGS
    _, p_one_hot, _, _ = stgs_module(
        free_logits,
        gumbel_noise_scale=gumbel_noise_scale,
        embedding_weights=embedding_weights,
    )
    p_embeds = p_one_hot @ embedding_weights.detach()   # (1, seq_len, embed_dim)

    if teacher_forcing_r:
        # Case A: teacher-forced R (or no R when R_gt_embeds is None)
        # Input: [x | p | R_gt | E | Y[0:-1]]
        Y_embeds_full = embedding_weights[Y_tokens.squeeze(0)].unsqueeze(0)  # (1, Y_len, d)
        parts = [x_embeds, p_embeds]
        if R_gt_embeds is not None:
            parts.append(R_gt_embeds)
        parts.append(E_embeds)
        if Y_len > 1:
            parts.append(Y_embeds_full[:, :-1, :])
        input_embeds = torch.cat(parts, dim=1)

        output = diff_model.forward(inputs_embeds=input_embeds, output_normal_logits=True)
        y_logits = output.logits[:, -Y_len:, :]   # (1, Y_len, vocab)

    else:
        # Case B: generate R differentiably
        r_out = diff_model.generate(
            inputs_embeds=torch.cat([x_embeds, p_embeds], dim=1),
            max_new_tokens=max_new_tokens_reasoning,
            output_diff_one_hots=not accumulate_embeds,
            output_normal_logits=False,
            use_soft_embeds=True,
            use_bpttoken=bptt,
            return_dict=True,
            accumulate_embeds=accumulate_embeds,
        )
        if r_out.sampled_diff_embeds is not None:
            R_soft_embeds = r_out.sampled_diff_embeds
        else:
            R_soft_embeds = r_out.sampled_diff_one_hot @ embedding_weights.detach()
        if not bptt:
            R_soft_embeds = R_soft_embeds.detach()

        y_out = diff_model.generate(
            inputs_embeds=torch.cat([x_embeds, p_embeds, R_soft_embeds, E_embeds], dim=1),
            max_new_tokens=max_new_tokens_answer,
            output_diff_one_hots=False,
            output_normal_logits=True,
            use_soft_embeds=True,
            return_dict=True,
        )
        actual_len = y_out.logits.shape[1]
        if actual_len < Y_len:
            pad = torch.zeros(
                1, Y_len - actual_len, y_out.logits.shape[-1],
                device=y_out.logits.device, dtype=y_out.logits.dtype,
            )
            y_logits = torch.cat([y_out.logits, pad], dim=1)
        else:
            y_logits = y_out.logits[:, :Y_len, :]

    # Populate Weave trace extras with decoded text
    if weave_extras is not None and tokenizer is not None:
        weave_extras["y_pred_text"] = tokenizer.decode(
            y_logits.detach().argmax(dim=-1).squeeze(0).tolist(),
            skip_special_tokens=True,
        )
        if not teacher_forcing_r:
            weave_extras["R_gen_text"] = tokenizer.decode(
                r_out.sampled_diff_tokens.squeeze(0).tolist(),
                skip_special_tokens=True,
            )

    input_dict = {
        "generated_logits":    y_logits,
        "prompt_logits":       free_logits,
        "prompt_argmax_ids":   free_logits.argmax(dim=-1),
        "completion_ids":      Y_tokens,
        "tf_generated_logits": None,
        "prompt_input_logits": free_logits,  # for eos/entropy/commitment losses
    }
    losses_dict = loss_instance.compute_loss(input_dict)
    return losses_dict["sumloss"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _tokenize_x(x_str: str, tokenizer, device, use_chat_template: bool) -> Tensor:
    """
    Tokenize input x, optionally applying the model's chat template.

    When use_chat_template=True, wraps x as a user turn and calls
    apply_chat_template with add_generation_prompt=True so the model
    sees [chat(x), p] instead of [x, p].
    Falls back to plain tokenization if the tokenizer has no chat_template.
    """
    if use_chat_template and getattr(tokenizer, "chat_template", None) is not None:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": x_str}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    if use_chat_template:
        logger.warning(
            "use_chat_template=True but tokenizer has no chat_template; "
            "falling back to plain tokenization."
        )
    return tokenizer(x_str, return_tensors="pt").input_ids.to(device)


def evaluate_shared_prompt(
    free_logits: Tensor,
    eval_pairs: List[Tuple[str, str]],      # (x_str, y_raw_str)
    model,                                   # base HF model for discrete eval
    diff_model: STGSDiffModel,               # for soft eval
    stgs_module: STGS,
    embedding_weights: Tensor,               # (vocab, embed_dim)
    tokenizer,
    device,
    E_input_ids: Tensor,                     # (1, E_len)
    E_embeds: Tensor,                        # (1, E_len, embed_dim)
    extraction_fns: Dict[str, Callable],
    max_new_tokens_reasoning: int,
    max_new_tokens_answer: int,
    eval_mode: str = "discrete",             # "discrete" or "soft"
    use_chat_template: bool = False,
    epoch: int = 0,
    x_ids_cache: Optional[Dict] = None,
    x_embeds_cache: Optional[Dict] = None,
    accumulate_embeds: bool = False,
) -> Dict[str, float]:
    """
    Evaluate the shared prompt on eval_pairs.
    Returns accuracy metrics per extraction method plus any_correct / all_correct.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    embedding_layer = model.get_input_embeddings()

    per_method_correct: Dict[str, int] = {m: 0 for m in extraction_fns}
    any_correct_count = 0
    all_correct_count = 0
    n_eval = len(eval_pairs)

    # Pre-compute decoded p and E text for Weave logging (constant across all samples)
    p_text_weave = tokenizer.decode(
        free_logits.detach().argmax(dim=-1).squeeze(0).tolist(),
        skip_special_tokens=False,
    )
    E_text_weave = tokenizer.decode(E_input_ids[0].tolist(), skip_special_tokens=True)

    with torch.no_grad():
        if eval_mode == "discrete":
            p_ids = free_logits.argmax(dim=-1)   # (1, seq_len)

        # Hoist soft prompt embedding: constant across all eval samples
        p_embeds_eval = None
        if eval_mode == "soft":
            _, p_one_hot_eval, _, _ = stgs_module(
                free_logits,
                gumbel_noise_scale=0.0,
                embedding_weights=embedding_weights,
            )
            p_embeds_eval = p_one_hot_eval @ embedding_weights.detach()

        for x_str, y_raw_str in eval_pairs:
            x_ids = (x_ids_cache or {}).get(x_str) if x_ids_cache else None
            if x_ids is None:
                x_ids = _tokenize_x(x_str, tokenizer, device, use_chat_template)

            if eval_mode == "discrete":
                # Discrete eval: argmax prompt tokens
                prefix_ids = torch.cat([x_ids, p_ids], dim=1)
                R_ids = model.generate(
                    prefix_ids,
                    max_new_tokens=max_new_tokens_reasoning,
                    pad_token_id=pad_id,
                    do_sample=False,
                )
                # R_ids includes the prompt prefix; strip it
                R_ids_only = R_ids[:, prefix_ids.shape[1]:]
                R_text = tokenizer.decode(R_ids_only[0], skip_special_tokens=True)

                # Extractor path
                xpR_ids = torch.cat([x_ids, p_ids, R_ids_only, E_input_ids], dim=1)
                Y_ids = model.generate(
                    xpR_ids,
                    max_new_tokens=max_new_tokens_answer,
                    pad_token_id=pad_id,
                    do_sample=False,
                )
                Y_text = tokenizer.decode(Y_ids[0, xpR_ids.shape[1]:], skip_special_tokens=True)

            else:
                # Soft eval: use STGSDiffModel (p_embeds_eval hoisted above)
                x_embeds = (x_embeds_cache or {}).get(x_str) if x_embeds_cache else None
                if x_embeds is None:
                    x_embeds = embedding_layer(x_ids)

                r_out = diff_model.generate(
                    inputs_embeds=torch.cat([x_embeds, p_embeds_eval], dim=1),
                    max_new_tokens=max_new_tokens_reasoning,
                    output_diff_one_hots=not accumulate_embeds,
                    output_normal_logits=False,
                    use_soft_embeds=True,
                    return_dict=True,
                    accumulate_embeds=accumulate_embeds,
                )
                R_ids_only = r_out.sampled_diff_tokens
                R_text = tokenizer.decode(R_ids_only[0], skip_special_tokens=True)

                # Extractor path
                R_embeds = embedding_layer(R_ids_only.long())
                y_out = diff_model.generate(
                    inputs_embeds=torch.cat([x_embeds, p_embeds_eval, R_embeds, E_embeds], dim=1),
                    max_new_tokens=max_new_tokens_answer,
                    output_diff_one_hots=True,
                    output_normal_logits=False,
                    use_soft_embeds=True,
                    return_dict=True,
                )
                Y_text = tokenizer.decode(y_out.sampled_diff_tokens[0], skip_special_tokens=True)

            # Evaluate extraction methods
            method_results: Dict[str, bool] = {}
            for method, fn in extraction_fns.items():
                if method == "extractor":
                    extracted = fn(Y_text)
                else:
                    extracted = fn(R_text)
                method_results[method] = compare_answers(extracted, y_raw_str)
                if method_results[method]:
                    per_method_correct[method] += 1

            any_c = any(method_results.values())
            all_c = all(method_results.values())
            any_correct_count += int(any_c)
            all_correct_count += int(all_c)

            weave_lm_eval_step(
                split=eval_mode,
                epoch=epoch,
                x_text=x_str,
                p_text=p_text_weave,
                E_text=E_text_weave,
                R_text=R_text,
                Y_text=Y_text,
                y_target=y_raw_str,
                correct_per_method=method_results,
                any_correct=any_c,
                eval_mode=eval_mode,
            )

    metrics: Dict[str, float] = {}
    for method in extraction_fns:
        metrics[f"accuracy_{method}"] = per_method_correct[method] / max(n_eval, 1)
    metrics["any_correct"] = any_correct_count / max(n_eval, 1)
    metrics["all_correct"] = all_correct_count / max(n_eval, 1)
    metrics["n_eval"] = float(n_eval)
    return metrics


# ---------------------------------------------------------------------------
# Core optimization loop
# ---------------------------------------------------------------------------

def pc_optimize_inputs(
    model,
    diff_model: STGSDiffModel,
    tokenizer,
    device,
    model_precision: str,
    # Dataset
    train_triples: List[Tuple[str, str, str]],   # (x_str, R_gt_str, y_raw_str)
    val_pairs: Optional[List[Tuple[str, str]]],  # (x_str, y_raw_str)
    # Extraction
    extraction_prompt: str,
    extraction_fns: Dict[str, Callable],
    max_new_tokens_reasoning: int = 200,
    max_new_tokens_answer: int = 20,
    teacher_forcing_r: bool = True,
    bptt: bool = False,
    # Loss
    losses: str = "crossentropy",
    # Optimization
    seq_len: int = 20,
    epochs: int = 2000,
    learning_rate: float = 0.01,
    inner_batch_size: int = 4,
    # STGS for prompt p
    temperature: float = 1.0,
    stgs_hard: bool = True,
    stgs_hard_method: str = "categorical",
    logits_normalize: str = "none",
    stgs_input_dropout: float = 0.0,
    stgs_output_dropout: float = 0.0,
    eps: float = 1e-10,
    gumbel_noise_scale: float = 1.0,
    adaptive_gumbel_noise: bool = False,
    adaptive_gumbel_noise_beta: float = 0.9,
    adaptive_gumbel_noise_min_scale: float = 0.0,
    # Logits filtering
    logits_top_k: int = 0,
    logits_top_p: float = 1.0,
    logits_filter_penalty: float = 1e4,
    # Initialization
    init_strategy: str = "randn",
    init_std: float = 0.0,
    logits_lora_rank: int = 0,
    # Schedule
    max_gradient_norm: float = 0.0,
    temperature_anneal_schedule: str = "none",
    temperature_anneal_min: float = 0.1,
    temperature_anneal_epochs: int = 0,
    discrete_reinit_epoch: int = 0,
    discrete_reinit_snap: str = "argmax",
    logit_decay: float = 0.0,
    # Validation
    val_eval_every: int = 100,
    val_eval_mode: str = "discrete",
    per_epoch_callback: Optional[Callable] = None,
    # Chat template
    use_chat_template: bool = False,
    # Memory / speed optimizations
    accumulate_embeds: bool = False,
    # Extra kwargs forwarded to LossClass
    kwargs: dict = {},
) -> Dict:
    """
    Optimize a single shared prompt p across all (x, R_gt, y) training pairs.

    Returns:
        {
          "final_p_logits": Tensor,
          "final_p_tokens": List[int],
          "final_p_text": str,
          "loss_history": List[float],
          "val_accuracy_history": List[Dict],
        }
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    vocab_size = model.config.vocab_size
    embedding_layer = model.get_input_embeddings()
    embedding_weights = embedding_layer.weight.detach()  # (vocab, embed_dim)

    # -----------------------------------------------------------------------
    # Pre-tokenize and pre-embed all inputs
    # -----------------------------------------------------------------------
    E_input_ids = tokenizer(
        extraction_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        E_embeds = embedding_layer(E_input_ids)   # (1, E_len, d)

    # Per-sample precomputed tensors
    x_embeds_cache: Dict[str, Tensor] = {}
    x_ids_cache: Dict[str, Tensor] = {}
    R_gt_embeds_cache: Dict[str, Optional[Tensor]] = {}
    Y_tokens_cache: Dict[str, Tensor] = {}

    all_x_strs = list({t[0] for t in train_triples})
    for x_str in all_x_strs:
        x_ids = _tokenize_x(x_str, tokenizer, device, use_chat_template)
        x_ids_cache[x_str] = x_ids
        with torch.no_grad():
            x_embeds_cache[x_str] = embedding_layer(x_ids)

    for x_str, R_gt_str, _ in train_triples:
        key = (x_str, R_gt_str)
        if key not in R_gt_embeds_cache:
            if R_gt_str:
                r_ids = tokenizer(R_gt_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    R_gt_embeds_cache[key] = embedding_layer(r_ids)
            else:
                R_gt_embeds_cache[key] = None

    for _, _, y_raw_str in train_triples:
        if y_raw_str not in Y_tokens_cache:
            Y_tokens_cache[y_raw_str] = tokenizer(
                y_raw_str, return_tensors="pt"
            ).input_ids.to(device)

    # Also precompute for val pairs
    if val_pairs:
        for x_str, _ in val_pairs:
            if x_str not in x_embeds_cache:
                x_ids = _tokenize_x(x_str, tokenizer, device, use_chat_template)
                x_ids_cache[x_str] = x_ids
                with torch.no_grad():
                    x_embeds_cache[x_str] = embedding_layer(x_ids)

    # -----------------------------------------------------------------------
    # Build LossClass cache keyed by (y_raw_str, Y_len)
    # -----------------------------------------------------------------------
    lc_kwargs = dict(kwargs)
    lc_kwargs.setdefault("seq_len", seq_len)

    loss_cache: Dict[Tuple[str, int], LossClass] = {}
    for _, _, y_raw_str in train_triples:
        Y_tokens = Y_tokens_cache[y_raw_str]
        Y_len = Y_tokens.shape[1]
        key = (y_raw_str, Y_len)
        if key not in loss_cache:
            loss_cache[key] = LossClass(
                losses=losses,
                embedding_weights_subset=embedding_weights,
                target_text=y_raw_str,
                target_tokens_mapped=Y_tokens,
                model=model,
                tokenizer=tokenizer,
                kwargs=lc_kwargs,
            )

    # -----------------------------------------------------------------------
    # Initialize learnable prompt logits
    # -----------------------------------------------------------------------
    lora_A = lora_B = None
    if logits_lora_rank > 0:
        lora_A, lora_B = initialize_lora_logits(
            seq_len, vocab_size, logits_lora_rank, device,
            init_strategy=init_strategy,
            init_std=init_std,
        )
        free_logits = None
        parameters = [lora_A, lora_B]
    else:
        free_logits = initialize_learnable_inputs(
            vocab_size, seq_len, device,
            init_strategy=init_strategy,
            init_std=init_std,
        )
        parameters = [free_logits]

    # -----------------------------------------------------------------------
    # STGS module for the shared prompt p
    # -----------------------------------------------------------------------
    stgs_module = STGS(
        vocab_size=vocab_size,
        stgs_hard=stgs_hard,
        stgs_hard_method=stgs_hard_method,
        init_temperature=temperature,
        logits_normalize=logits_normalize,
        input_dropout=stgs_input_dropout,
        output_dropout=stgs_output_dropout,
        eps=eps,
        device=device,
    )
    parameters += list(stgs_module.parameters())

    optimizer = optim.Adam(parameters, lr=learning_rate)

    # -----------------------------------------------------------------------
    # Logits filter helper (top-k / top-p)
    # -----------------------------------------------------------------------
    def _apply_logit_filter(logits: Tensor) -> Tensor:
        """Apply top-k / top-p penalty to free_logits."""
        if logits_top_k <= 0 and logits_top_p >= 1.0:
            return logits
        filtered = logits.clone()
        if logits_top_k > 0:
            # Zero out tokens outside top-k per position
            topk_vals = torch.topk(filtered, min(logits_top_k, filtered.shape[-1]), dim=-1).values
            threshold = topk_vals[..., -1:].expand_as(filtered)
            filtered = torch.where(filtered < threshold, filtered - logits_filter_penalty, filtered)
        if logits_top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > logits_top_p
            sorted_logits[remove_mask] = sorted_logits[remove_mask] - logits_filter_penalty
            filtered.scatter_(-1, sorted_indices, sorted_logits)
        return filtered

    def _get_free_logits() -> Tensor:
        raw = lora_A @ lora_B if lora_A is not None else free_logits
        return _apply_logit_filter(raw)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    loss_history: List[float] = []
    val_accuracy_history: List[Dict] = []
    current_gumbel_scale = gumbel_noise_scale
    ema_loss = None
    anneal_total = temperature_anneal_epochs if temperature_anneal_epochs > 0 else epochs

    for epoch in range(epochs):
        # Temperature annealing
        current_temperature = compute_annealed_temperature(
            temperature, temperature_anneal_min, epoch, anneal_total,
            temperature_anneal_schedule,
        )
        stgs_module.init_temperature = current_temperature

        # Discrete reinitialization
        if discrete_reinit_epoch > 0 and epoch > 0 and epoch % discrete_reinit_epoch == 0:
            current_logits = _get_free_logits().detach()
            if lora_A is not None:
                snap_lora_logits(
                    lora_A, lora_B, current_logits, embedding_weights,
                    seq_len, 0, discrete_reinit_snap, optimizer,
                )
            else:
                snap_free_logits(
                    free_logits, current_logits, embedding_weights,
                    seq_len, 0, discrete_reinit_snap,
                )

        # Adaptive Gumbel noise
        if adaptive_gumbel_noise and ema_loss is not None:
            current_gumbel_scale = max(
                adaptive_gumbel_noise_min_scale,
                adaptive_gumbel_noise_beta * current_gumbel_scale
                + (1 - adaptive_gumbel_noise_beta) * (ema_loss / (ema_loss + 1.0)),
            )

        # Mini-batch selection
        if inner_batch_size >= len(train_triples):
            mini_batch = train_triples
        else:
            mini_batch = random.sample(train_triples, inner_batch_size)

        optimizer.zero_grad()
        batch_loss = 0.0
        n_batch = len(mini_batch)

        for x_str, R_gt_str, y_raw_str in mini_batch:
            x_embeds = x_embeds_cache[x_str]
            R_gt_embeds = R_gt_embeds_cache[(x_str, R_gt_str)]
            Y_tokens = Y_tokens_cache[y_raw_str]
            Y_len = Y_tokens.shape[1]
            loss_instance = loss_cache[(y_raw_str, Y_len)]

            fl = _get_free_logits()
            weave_extras: Dict = {}
            loss = pc_forward_pass(
                diff_model=diff_model,
                stgs_module=stgs_module,
                loss_instance=loss_instance,
                free_logits=fl,
                embedding_weights=embedding_weights,
                x_embeds=x_embeds,
                R_gt_embeds=R_gt_embeds,
                E_embeds=E_embeds,
                Y_tokens=Y_tokens,
                teacher_forcing_r=teacher_forcing_r,
                bptt=bptt,
                max_new_tokens_reasoning=max_new_tokens_reasoning,
                max_new_tokens_answer=max_new_tokens_answer,
                gumbel_noise_scale=current_gumbel_scale,
                accumulate_embeds=accumulate_embeds,
                tokenizer=tokenizer,
                weave_extras=weave_extras,
            )
            (loss / n_batch).backward()
            batch_loss += loss.item()

            # Weave per-LM-call trace
            weave_lm_train_step(
                epoch=epoch,
                x_text=x_str,
                p_text=tokenizer.decode(
                    fl.detach().argmax(dim=-1).squeeze(0).tolist(),
                    skip_special_tokens=False,
                ),
                R_gt_text=R_gt_str,
                R_gen_text=weave_extras.get("R_gen_text", ""),
                E_text=extraction_prompt,
                y_target=y_raw_str,
                y_pred=weave_extras.get("y_pred_text", ""),
                loss=loss.item(),
                teacher_forcing_r=teacher_forcing_r,
            )

        # Gradient clipping
        total_norm = None
        if max_gradient_norm > 0:
            all_params = [p for p in [free_logits, lora_A, lora_B] if p is not None]
            all_params += list(stgs_module.parameters())
            total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_gradient_norm).item()

        optimizer.step()

        # Logit decay
        if logit_decay > 0:
            with torch.no_grad():
                if lora_A is not None:
                    lora_A.data.mul_(1 - logit_decay)
                    lora_B.data.mul_(1 - logit_decay)
                else:
                    free_logits.data.mul_(1 - logit_decay)

        # Cache fl and p_text once per epoch after parameter update
        fl_epoch = _get_free_logits().detach()
        p_text_epoch = tokenizer.decode(
            fl_epoch.argmax(dim=-1).squeeze(0).tolist(),
            skip_special_tokens=False,
        )

        epoch_avg_loss = batch_loss / n_batch
        loss_history.append(epoch_avg_loss)
        ema_loss = epoch_avg_loss if ema_loss is None else (
            adaptive_gumbel_noise_beta * ema_loss + (1 - adaptive_gumbel_noise_beta) * epoch_avg_loss
        )

        # W&B logging
        if wandb.run is not None:
            log_dict = {
                "train/loss": epoch_avg_loss,
                "temperature": current_temperature,
                "gumbel_noise_scale": current_gumbel_scale,
                "epoch": epoch,
            }
            if total_norm is not None:
                log_dict["grad_norm"] = total_norm
            wandb.log(log_dict)

        # Weave epoch-level trace
        weave_epoch_summary(
            epoch=epoch,
            loss=epoch_avg_loss,
            p_text=p_text_epoch,
            temperature=current_temperature,
            n_samples=len(mini_batch),
        )

        # Per-epoch callback
        if per_epoch_callback is not None:
            cb_metrics = per_epoch_callback(epoch, fl_epoch)
            if cb_metrics and wandb.run is not None:
                wandb.log(cb_metrics)

        # Validation
        if val_eval_every > 0 and (epoch + 1) % val_eval_every == 0 and val_pairs:
            val_metrics = evaluate_shared_prompt(
                free_logits=fl_epoch,
                eval_pairs=val_pairs,
                model=model,
                diff_model=diff_model,
                stgs_module=stgs_module,
                embedding_weights=embedding_weights,
                tokenizer=tokenizer,
                device=device,
                E_input_ids=E_input_ids,
                E_embeds=E_embeds,
                extraction_fns=extraction_fns,
                max_new_tokens_reasoning=max_new_tokens_reasoning,
                max_new_tokens_answer=max_new_tokens_answer,
                eval_mode=val_eval_mode,
                use_chat_template=use_chat_template,
                epoch=epoch,
                x_ids_cache=x_ids_cache,
                x_embeds_cache=x_embeds_cache if val_eval_mode == "soft" else None,
                accumulate_embeds=accumulate_embeds,
            )
            val_metrics["epoch"] = epoch
            val_accuracy_history.append(val_metrics)
            if wandb.run is not None:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

    # -----------------------------------------------------------------------
    # Build final result
    # -----------------------------------------------------------------------
    final_fl = _get_free_logits().detach()
    final_p_tokens = final_fl.argmax(dim=-1).squeeze(0).tolist()
    final_p_text = tokenizer.decode(final_p_tokens, skip_special_tokens=False)

    return {
        "final_p_logits": final_fl,
        "learnable_logits": final_fl,
        "final_p_tokens": final_p_tokens,
        "final_p_text": final_p_text,
        "loss_history": loss_history,
        "val_accuracy_history": val_accuracy_history,
    }
