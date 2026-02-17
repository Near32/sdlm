"""
SODA optimization using transformer_lens - single-target variant.

This implements the SODA algorithm using the transformer_lens library API,
which is closer to the original SODA paper implementation.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import logging
import copy
import wandb

from .soda_utils import (
    CustomAdam,
    compute_lcs_ratio,
    check_exact_match,
    initialize_embedding_params,
    create_optimizer,
    apply_embedding_decay,
    reset_optimizer_state,
    reinitialize_params,
    compute_fluency_penalty,
)

logger = logging.getLogger(__name__)


def soda_optimize_inputs_tl(
    model_name: str,
    device: torch.device,
    target_text: str,
    seq_len: int,
    epochs: int = 100000,
    learning_rate: float = 0.065,
    temperature: float = 0.05,
    decay_rate: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.995),
    reset_epoch: int = 50,
    reinit_epoch: int = 1500,
    reg_weight: Optional[float] = None,
    bias_correction: bool = False,
    init_strategy: str = "zeros",
    init_std: float = 0.05,
    early_stop_on_exact_match: bool = True,
    lcs_ratio_threshold: float = 1.0,
    batch_size: int = 1,
    model: Optional[Any] = None,  # Pre-loaded model
    ground_truth_prompt_tokens: Optional[List[int]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], torch.Tensor, List[float], List[float], Dict[str, List[float]]]:
    """
    Single-target SODA optimization using transformer_lens.

    This variant uses the transformer_lens API which matches the original
    SODA paper's implementation more closely.

    Args:
        model_name: Name of the model to load via transformer_lens
        device: Device to run optimization on
        target_text: The target text we want the model to generate
        seq_len: Length of the prompt sequence to optimize
        epochs: Maximum number of optimization epochs
        learning_rate: Learning rate for optimizer
        temperature: Temperature for softmax (lower = sharper)
        decay_rate: Decay rate applied to embedding params each epoch
        betas: Adam beta parameters
        reset_epoch: Reset optimizer state every N epochs
        reinit_epoch: Reinitialize embeddings every N epochs
        reg_weight: Weight for fluency regularization (None = disabled)
        bias_correction: Use standard Adam with bias correction if True
        init_strategy: Initialization strategy ("zeros" or "normal")
        init_std: Standard deviation for normal initialization
        early_stop_on_exact_match: Stop early if exact match found
        lcs_ratio_threshold: Stop early if LCS ratio >= threshold
        batch_size: Batch size (for compatibility, not used in single-target)
        model: Pre-loaded transformer_lens model (optional)
        kwargs: Additional keyword arguments

    Returns:
        Tuple of:
            - generated_tokens: List of generated token IDs
            - optimized_inputs: Optimized learnable parameter tensor
            - losses: List of loss values per epoch
            - lcs_ratio_history: List of LCS ratios per epoch
            - prompt_metrics_history: Dict of prompt reconstruction metrics per epoch
    """
    kwargs = kwargs or {}

    # Load model if not provided
    if model is None:
        try:
            from transformer_lens import HookedTransformer
        except ImportError:
            raise ImportError(
                "transformer_lens is required for SODA TL backend. "
                "Install with: pip install transformer-lens"
            )
        model = HookedTransformer.from_pretrained(model_name, device=device)
        model = model.eval()

    # Get embedding weights
    embedding_weights = model.embed.W_E  # (vocab_size, embed_dim)
    vocab_size = embedding_weights.shape[0]

    # Tokenize the target text
    target_tokens = model.to_tokens(target_text)  # Shape: (1, target_length)
    target_length = target_tokens.shape[1]
    target_tokens_list = target_tokens[0].cpu().tolist()

    # Initialize wandb table for logging
    log_table_every = kwargs.get("log_table_every", 100)
    wandb_table = wandb.Table(columns=[
        "epoch",
        "target_text",
        "learned_input_ids",
        "learned_input_str",
        "generated_output_ids",
        "generated_output_str",
        "lcs_ratio",
        "token_overlap_ratio",
        "target_hit_ratio",
    ])

    def compute_token_overlap(prompt_tokens, target_tokens_tensor):
        """Compute token overlap and target hit ratio."""
        tt_occ = {}
        for ttoken in target_tokens_tensor:
            tt_occ[ttoken.item()] = (prompt_tokens == ttoken).int().sum().item()
        nbr_occ = sum(tt_occ.values())
        max_occ = len(prompt_tokens)
        overlap_ratio = nbr_occ / max_occ if max_occ > 0 else 0.0
        target_set = set(target_tokens_tensor.tolist())
        target_hits = {t: int(occ > 0) for t, occ in tt_occ.items()}
        target_hit_ratio = sum(target_hits.values()) / len(target_set) if target_set else 0.0
        return overlap_ratio, target_hit_ratio

    # Check if model uses positional embeddings
    model_name_lower = model_name.lower() if isinstance(model_name, str) else ""
    use_pos_embed = "gpt" in model_name_lower or "tiny" in model_name_lower

    # Initialize learnable embedding parameters
    pred_embed_pre = initialize_embedding_params(
        vocab_size=vocab_size,
        seq_len=seq_len,
        device=device,
        init_strategy=init_strategy,
        init_std=init_std,
    )
    pred_embed_pre.requires_grad = True

    # Create optimizer
    optimizer = create_optimizer(
        params=[pred_embed_pre],
        learning_rate=learning_rate,
        betas=betas,
        bias_correction=bias_correction,
    )

    # Lists to track metrics
    losses = []
    lcs_ratio_history = []

    # Initialize prompt metrics history for per-epoch tracking
    prompt_metrics_history = {
        "prompt_token_accuracy": [],
        "prompt_exact_match": [],
        "prompt_lcs_ratio": [],
        "prompt_cosine_similarity": [],
    }
    track_prompt_metrics = ground_truth_prompt_tokens is not None
    if track_prompt_metrics:
        gt_prompt_len = len(ground_truth_prompt_tokens)

    # Main optimization loop
    pbar = tqdm(range(epochs), desc="SODA TL Optimization")
    for epoch in pbar:
        optimizer.zero_grad()

        # Convert to soft one-hot via softmax
        pred_one_hot = torch.softmax(pred_embed_pre / temperature, dim=-1)

        # Get embeddings by multiplying with embedding matrix
        pred_embed = pred_one_hot @ embedding_weights

        # Concatenate with target token embeddings for teacher forcing
        target_embeds = model.embed(target_tokens[:, :-1])
        pred_embed_full = torch.cat([pred_embed, target_embeds], dim=1)

        # Add positional embeddings if needed
        if use_pos_embed:
            # Create a dummy tensor to get positional embeddings
            pos_embed = model.pos_embed(pred_embed_full[:, :, 0].detach())
            pred_embed_full = pred_embed_full + pos_embed

        # Forward pass through the model
        pred_logits = model(pred_embed_full, start_at_layer=0)

        # Extract logits for target prediction
        target_logits = pred_logits[:, seq_len - 1:, :]

        # Compute loss (log-probability difference as in original SODA)
        log_probs = F.softmax(target_logits, dim=-1).clamp(min=1e-12).log()
        target_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
        max_log_probs = log_probs.max(dim=-1).values
        loss_diff = target_log_probs - max_log_probs
        loss = -loss_diff.mean()

        # Add fluency regularization if enabled
        if reg_weight is not None and reg_weight > 0:
            prompt_logits = pred_logits[:, :seq_len, :]
            fluency_penalty = compute_fluency_penalty(prompt_logits, pred_one_hot)
            loss = loss + reg_weight * fluency_penalty.mean()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Apply decay to embeddings
        apply_embedding_decay([pred_embed_pre], decay_rate)

        # Periodic interventions
        if (epoch + 1) % reset_epoch == 0:
            reset_optimizer_state(optimizer, [pred_embed_pre])

        if (epoch + 1) % reinit_epoch == 0:
            reinitialize_params([pred_embed_pre], std=0.1)

        # Compute metrics
        with torch.no_grad():
            # Get predicted tokens
            pred_tokens = pred_one_hot.argmax(dim=-1)

            # Generate completion using discrete tokens
            pred_tokens_full = torch.cat([pred_tokens, target_tokens[:, :-1]], dim=1)
            discrete_logits = model(pred_tokens_full)
            generated_tokens = discrete_logits[:, seq_len - 1:, :].argmax(dim=-1)
            generated_tokens_list = generated_tokens[0].cpu().tolist()

            # Compute LCS ratio
            lcs_ratio = compute_lcs_ratio(generated_tokens_list, target_tokens_list)

            # Get learned input IDs and strings for logging
            learned_input_ids = pred_one_hot.argmax(dim=-1)[0].cpu().tolist()
            learned_input_str = model.to_string(learned_input_ids)
            generated_output_str = model.to_string(generated_tokens_list)

            # Compute token overlap metrics
            token_overlap_ratio, target_hit_ratio = compute_token_overlap(
                torch.tensor(learned_input_ids), target_tokens[0].cpu()
            )

            # Compute prompt reconstruction metrics if ground truth is available
            if track_prompt_metrics:
                # Prompt token accuracy (partial match)
                min_len = min(len(learned_input_ids), gt_prompt_len)
                matching = sum(
                    learned_input_ids[i] == ground_truth_prompt_tokens[i]
                    for i in range(min_len)
                )
                prompt_token_acc = matching / gt_prompt_len if gt_prompt_len > 0 else 0.0
                prompt_metrics_history["prompt_token_accuracy"].append(prompt_token_acc)

                # Prompt exact match
                prompt_exact = int(learned_input_ids[:gt_prompt_len] == ground_truth_prompt_tokens[:gt_prompt_len])
                prompt_metrics_history["prompt_exact_match"].append(prompt_exact)

                # Prompt LCS ratio
                prompt_lcs = compute_lcs_ratio(learned_input_ids, ground_truth_prompt_tokens)
                prompt_metrics_history["prompt_lcs_ratio"].append(prompt_lcs)

                # Prompt cosine similarity (using embeddings)
                opt_len = min(len(learned_input_ids), gt_prompt_len)
                opt_tokens_tensor = torch.tensor(learned_input_ids[:opt_len], dtype=torch.long, device=device)
                gt_tokens_tensor = torch.tensor(ground_truth_prompt_tokens[:opt_len], dtype=torch.long, device=device)
                opt_emb = embedding_weights[opt_tokens_tensor]
                gt_emb = embedding_weights[gt_tokens_tensor]
                opt_norm = opt_emb / (opt_emb.norm(dim=-1, keepdim=True) + 1e-9)
                gt_norm = gt_emb / (gt_emb.norm(dim=-1, keepdim=True) + 1e-9)
                cos_sim = (opt_norm * gt_norm).sum(dim=-1).mean().item()
                prompt_metrics_history["prompt_cosine_similarity"].append(cos_sim)

        losses.append(loss.item())
        lcs_ratio_history.append(lcs_ratio)

        # Log to wandb
        wandb_log = {
            "epoch": epoch + 1,
            "loss": loss.item(),
            "lcs_ratio": lcs_ratio,
            "token_overlap_ratio": token_overlap_ratio,
            "target_hit_ratio": target_hit_ratio,
            "temperature": temperature,
        }

        # Add prompt metrics to wandb_log if tracking
        if track_prompt_metrics:
            wandb_log['prompt_token_accuracy'] = prompt_token_acc
            wandb_log['prompt_exact_match'] = prompt_exact
            wandb_log['prompt_lcs_ratio'] = prompt_lcs
            wandb_log['prompt_cosine_similarity'] = cos_sim

        wandb.log(wandb_log)

        # Add to wandb table
        wandb_table.add_data(
            epoch + 1,
            target_text,
            learned_input_ids,
            learned_input_str,
            generated_tokens_list,
            generated_output_str,
            lcs_ratio,
            token_overlap_ratio,
            target_hit_ratio,
        )

        # Log table periodically
        if (epoch + 1) % log_table_every == 0:
            wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})

        # Update progress bar
        pbar.set_description(
            f"SODA TL Epoch {epoch + 1}, Loss: {loss.item():.4f}, LCS: {lcs_ratio:.4f}"
        )

        # Early stopping
        is_exact_match = check_exact_match(generated_tokens[0], target_tokens[0])
        if early_stop_on_exact_match and is_exact_match:
            logger.info(f"SODA TL: Exact match found at epoch {epoch + 1}")
            wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})
            break
        if lcs_ratio >= lcs_ratio_threshold:
            logger.info(f"SODA TL: LCS ratio threshold reached at epoch {epoch + 1}")
            wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})
            break
    else:
        # Log final table if loop completed without early stopping
        wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})

    # Return final results
    with torch.no_grad():
        final_one_hot = torch.softmax(pred_embed_pre / temperature, dim=-1)
        final_pred_tokens = final_one_hot.argmax(dim=-1)

        # Generate final completion
        final_tokens_full = torch.cat([final_pred_tokens, target_tokens[:, :-1]], dim=1)
        final_logits = model(final_tokens_full)
        final_generated = final_logits[:, seq_len - 1:, :].argmax(dim=-1)
        generated_tokens_list = final_generated[0].cpu().tolist()

    return generated_tokens_list, final_one_hot, losses, lcs_ratio_history, prompt_metrics_history
