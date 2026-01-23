"""
GCG optimization using transformer_lens - single-target variant.

This implements the GCG algorithm using the transformer_lens library API,
which is closer to the original implementation.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import logging
import copy
import wandb

from .gcg_utils import (
    compute_lcs_ratio,
    check_exact_match,
    initialize_tokens,
    select_mutation_positions,
    select_token_candidates,
    apply_mutations,
)

logger = logging.getLogger(__name__)


def gcg_optimize_inputs_tl(
    model_name: str,
    device: torch.device,
    target_text: str,
    seq_len: int,
    epochs: int = 704,
    num_candidates: int = 704,
    top_k: int = 128,
    num_mutations: int = 1,
    pos_choice: str = "uniform",
    token_choice: str = "uniform",
    init_strategy: str = "zeros",
    early_stop_on_exact_match: bool = True,
    lcs_ratio_threshold: float = 1.0,
    batch_size: int = 1,
    model: Optional[Any] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], torch.Tensor, List[float], List[float]]:
    """
    Single-target GCG optimization using transformer_lens.

    Args:
        model_name: Name of the model to load via transformer_lens
        device: Device to run optimization on
        target_text: The target text we want the model to generate
        seq_len: Length of the prompt sequence to optimize
        epochs: Maximum number of optimization epochs
        num_candidates: Number of mutation candidates to try per epoch
        top_k: Number of top tokens to consider for each position
        num_mutations: Number of positions to mutate per candidate
        pos_choice: Position selection strategy
        token_choice: Token selection strategy
        init_strategy: Initialization strategy
        early_stop_on_exact_match: Stop early if exact match found
        lcs_ratio_threshold: Stop early if LCS ratio >= threshold
        batch_size: Batch size (for compatibility)
        model: Pre-loaded transformer_lens model (optional)
        kwargs: Additional keyword arguments

    Returns:
        Tuple of:
            - generated_tokens: List of generated token IDs
            - optimized_inputs: One-hot encoding of final tokens
            - losses: List of loss values per epoch
            - lcs_ratio_history: List of LCS ratios per epoch
    """
    kwargs = kwargs or {}

    # Load model if not provided
    if model is None:
        try:
            from transformer_lens import HookedTransformer
        except ImportError:
            raise ImportError(
                "transformer_lens is required for GCG TL backend. "
                "Install with: pip install transformer-lens"
            )
        model = HookedTransformer.from_pretrained(model_name, device=device)
        model = model.eval()

    # Get embedding weights
    embedding_weights = model.embed.W_E
    vocab_size = embedding_weights.shape[0]

    # Check if model uses positional embeddings
    model_name_lower = model_name.lower() if isinstance(model_name, str) else ""
    use_pos_embed = "gpt" in model_name_lower or "tiny" in model_name_lower

    # Tokenize the target text
    target_tokens = model.to_tokens(target_text)
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

    # Initialize prediction tokens
    pred_tokens = initialize_tokens(
        vocab_size=vocab_size,
        seq_len=seq_len,
        device=device,
        init_strategy=init_strategy,
    )

    # Lists to track metrics
    losses = []
    lcs_ratio_history = []

    # Main optimization loop
    pbar = tqdm(range(epochs), desc="GCG TL Optimization")
    for epoch in pbar:
        # Get one-hot encoding of current tokens
        pred_one_hot = F.one_hot(pred_tokens, num_classes=vocab_size).float()
        pred_one_hot = pred_one_hot.to(embedding_weights.dtype).requires_grad_(True)

        # Get embeddings
        pred_embed = pred_one_hot @ embedding_weights

        # Concatenate with target token embeddings for teacher forcing
        target_embeds = model.embed(target_tokens[:, :-1])
        pred_embed_full = torch.cat([pred_embed, target_embeds], dim=1)

        # Add positional embeddings if needed
        if use_pos_embed:
            pos_embed = model.pos_embed(pred_embed_full[:, :, 0].detach())
            pred_embed_full = pred_embed_full + pos_embed

        # Forward pass
        pred_logits = model(pred_embed_full, start_at_layer=0)

        # Extract target prediction logits
        target_pred_logits = pred_logits[:, seq_len - 1:, :]

        # Compute loss
        log_probs = F.softmax(target_pred_logits, dim=-1).clamp(min=1e-12).log()
        target_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
        max_log_probs = log_probs.max(dim=-1).values
        loss_diff = target_log_probs - max_log_probs
        loss = -loss_diff.mean()

        # Backward pass to get gradients
        loss.backward()
        grad = pred_one_hot.grad.clone()
        grad = -grad  # We want to maximize the gradient direction

        # Check convergence
        with torch.no_grad():
            # Get current predictions
            generated = target_pred_logits.argmax(dim=-1)
            generated_tokens_list = generated[0].cpu().tolist()

            # Compute LCS ratio
            lcs_ratio = compute_lcs_ratio(generated_tokens_list, target_tokens_list)

            is_exact_match = check_exact_match(generated[0], target_tokens[0])

            # Get learned input IDs and strings for logging
            learned_input_ids = pred_tokens[0].cpu().tolist()
            learned_input_str = model.to_string(learned_input_ids)
            generated_output_str = model.to_string(generated_tokens_list)

            # Compute token overlap metrics
            token_overlap_ratio, target_hit_ratio = compute_token_overlap(
                torch.tensor(learned_input_ids), target_tokens[0].cpu()
            )

        losses.append(loss.item())
        lcs_ratio_history.append(lcs_ratio)

        # Log to wandb
        wandb_log = {
            "epoch": epoch + 1,
            "loss": loss.item(),
            "lcs_ratio": lcs_ratio,
            "token_overlap_ratio": token_overlap_ratio,
            "target_hit_ratio": target_hit_ratio,
        }
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
            f"GCG TL Epoch {epoch + 1}, Loss: {loss.item():.4f}, LCS: {lcs_ratio:.4f}"
        )

        # Early stopping
        if early_stop_on_exact_match and is_exact_match:
            logger.info(f"GCG TL: Exact match found at epoch {epoch + 1}")
            wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})
            break
        if lcs_ratio >= lcs_ratio_threshold:
            logger.info(f"GCG TL: LCS ratio threshold reached at epoch {epoch + 1}")
            wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})
            break

        # Try multiple candidates and keep the best
        with torch.no_grad():
            best_pred_tokens = pred_tokens.clone()
            best_loss = loss.item()

            for _ in range(num_candidates):
                # Select mutation positions
                new_token_pos = select_mutation_positions(
                    grad=grad,
                    num_mutations=num_mutations,
                    pos_choice=pos_choice,
                    batch_size=1,
                    seq_len=seq_len,
                )

                # Select token candidates
                new_token_val = select_token_candidates(
                    grad=grad,
                    new_token_pos=new_token_pos,
                    top_k=top_k,
                    token_choice=token_choice,
                    num_mutations=num_mutations,
                    batch_size=1,
                )

                # Apply mutations
                new_pred_tokens = apply_mutations(
                    pred_tokens=pred_tokens,
                    new_token_pos=new_token_pos,
                    new_token_val=new_token_val,
                )

                # Evaluate candidate
                new_pred_tokens_full = torch.cat([new_pred_tokens, target_tokens[:, :-1]], dim=1)
                new_logits = model(new_pred_tokens_full)
                new_target_logits = new_logits[:, seq_len - 1:, :]

                new_log_probs = F.softmax(new_target_logits, dim=-1).clamp(min=1e-12).log()
                new_target_log_probs = new_log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
                new_max_log_probs = new_log_probs.max(dim=-1).values
                new_loss_diff = new_target_log_probs - new_max_log_probs
                new_loss = -new_loss_diff.mean().item()

                # Keep best candidate
                if new_loss < best_loss:
                    best_pred_tokens = new_pred_tokens.clone()
                    best_loss = new_loss

            # Update tokens to best candidate
            pred_tokens = best_pred_tokens
    else:
        # Log final table if loop completed without early stopping
        wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})

    # Get final results
    with torch.no_grad():
        final_tokens_full = torch.cat([pred_tokens, target_tokens[:, :-1]], dim=1)
        final_logits = model(final_tokens_full)
        final_generated = final_logits[:, seq_len - 1:, :].argmax(dim=-1)
        generated_tokens_list = final_generated[0].cpu().tolist()

        # Return one-hot encoding of final tokens
        final_one_hot = F.one_hot(pred_tokens, num_classes=vocab_size).float()

    return generated_tokens_list, final_one_hot, losses, lcs_ratio_history
