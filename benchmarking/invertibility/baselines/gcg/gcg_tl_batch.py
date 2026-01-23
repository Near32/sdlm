"""
GCG optimization using transformer_lens - internal batching variant.

This implements the GCG algorithm with dynamic batch queue management
using transformer_lens.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import logging
import time
import copy
import wandb

from .gcg_utils import (
    DotDict,
    compute_lcs_ratio,
    check_exact_match,
    initialize_tokens,
    select_mutation_positions,
    select_token_candidates,
    apply_mutations,
)

logger = logging.getLogger(__name__)


def gcg_optimize_inputs_tl_batch(
    model_name: str,
    device: torch.device,
    targets: List[Dict[str, Any]],
    seq_len: int,
    epochs: int = 704,
    num_candidates: int = 704,
    top_k: int = 128,
    num_mutations: int = 1,
    pos_choice: str = "uniform",
    token_choice: str = "uniform",
    init_strategy: str = "zeros",
    max_batch_size: int = 285,
    early_stop_on_exact_match: bool = True,
    model: Optional[Any] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Multi-target GCG optimization using transformer_lens with internal batching.

    Args:
        model_name: Name of the model to load via transformer_lens
        device: Device to run optimization on
        targets: List of target dictionaries with 'text', 'id', etc.
        seq_len: Length of the prompt sequence to optimize
        epochs: Maximum number of optimization epochs per target
        num_candidates: Number of mutation candidates to try per epoch
        top_k: Number of top tokens to consider for each position
        num_mutations: Number of positions to mutate per candidate
        pos_choice: Position selection strategy
        token_choice: Token selection strategy
        init_strategy: Initialization strategy
        max_batch_size: Maximum batch size for parallel processing
        early_stop_on_exact_match: Stop early if exact match found
        model: Pre-loaded transformer_lens model (optional)
        kwargs: Additional keyword arguments

    Returns:
        Dictionary containing:
            - results: List of result dictionaries per target
            - elapsed_time: Total optimization time in seconds
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

    # Prepare target tokens
    target_tokens_list = []
    for target in targets:
        tokens = model.to_tokens(target["text"])[0].cpu()
        target_tokens_list.append(tokens)

    num_targets = len(targets)

    # Initialize state
    state = DotDict({
        "results": [],
        "batch_results": [],
        "target_tokens": torch.tensor([], device=device, dtype=torch.long),
        "pred_tokens": torch.tensor([], device=device, dtype=torch.long),
        "loaded_i": 0,
        "epoch": 0,
        "num_remain_items": num_targets,
        "num_success_items": 0,
        "elapsed_time": 0,
    })

    # Main optimization loop
    pbar = tqdm(total=num_targets, desc="GCG TL Batch", position=0)
    completed_prev = 0

    while state.num_remain_items != 0 or len(state.batch_results) != 0:
        start_time = time.time()
        state.epoch += 1

        with torch.no_grad():
            # Add new items to batch if have space
            if (max_batch_size - len(state.batch_results)) > 0 and state.num_remain_items != 0:
                num_new_items = min(
                    max_batch_size - len(state.batch_results),
                    state.num_remain_items
                )
                state.num_remain_items -= num_new_items

                for i in range(num_new_items):
                    idx = state.loaded_i + i
                    true_tokens = target_tokens_list[idx].to(device)

                    # Initialize result tracking
                    state.batch_results.append({
                        "target_id": targets[idx].get("id", idx),
                        "target_text": targets[idx]["text"],
                        "true_tokens": true_tokens.cpu(),
                        "pred_tokens": None,
                        "found_solution": False,
                        "done_epochs": 0,
                        "lcs_ratio_history": [],
                    })

                # Add target tokens to batch
                for i in range(num_new_items):
                    idx = state.loaded_i + i
                    tokens = target_tokens_list[idx].to(device).unsqueeze(0)
                    state.target_tokens = torch.cat([state.target_tokens, tokens], dim=0)

                # Initialize prediction tokens
                if init_strategy == "zeros":
                    new_pred = torch.zeros((num_new_items, seq_len), dtype=torch.long, device=device)
                else:
                    new_pred = torch.randint(0, vocab_size, (num_new_items, seq_len), device=device)
                state.pred_tokens = torch.cat([state.pred_tokens, new_pred], dim=0)

                state.loaded_i += num_new_items

        if len(state.batch_results) == 0:
            continue

        batch_size_current = len(state.batch_results)
        target_length = state.target_tokens.shape[1]

        # Get one-hot encoding and compute gradients
        pred_one_hot = F.one_hot(state.pred_tokens, num_classes=vocab_size)
        pred_one_hot = pred_one_hot.to(embedding_weights.dtype).requires_grad_(True)

        # Get embeddings
        pred_embed = pred_one_hot @ embedding_weights

        # Concatenate with target token embeddings
        target_embeds = model.embed(state.target_tokens[:, :-1])
        pred_embed_full = torch.cat([pred_embed, target_embeds], dim=1)

        # Add positional embeddings if needed
        if use_pos_embed:
            pos_embed = model.pos_embed(pred_embed_full[:, :, 0].detach())
            pred_embed_full = pred_embed_full + pos_embed

        # Forward pass
        pred_logits = model(pred_embed_full, start_at_layer=0)
        target_pred_logits = pred_logits[:, seq_len - 1:, :]

        # Compute loss
        log_probs = F.softmax(target_pred_logits, dim=-1).clamp(min=1e-12).log()
        target_log_probs = log_probs.gather(
            2, state.target_tokens.unsqueeze(-1)
        ).squeeze(-1)
        max_log_probs = log_probs.max(dim=-1).values
        loss_diff = target_log_probs - max_log_probs
        loss = -loss_diff.mean()

        # Get gradients
        loss.backward()
        grad = pred_one_hot.grad.clone()
        grad = -grad

        with torch.no_grad():
            # Check convergence and update epochs
            generated = target_pred_logits.argmax(dim=-1)

            # Process results in reverse order
            for i in range(len(state.batch_results) - 1, -1, -1):
                state.batch_results[i]["done_epochs"] += 1

                # Compute LCS ratio
                gen_tokens = generated[i].cpu().tolist()
                target_toks = state.target_tokens[i].cpu().tolist()
                lcs_ratio = compute_lcs_ratio(gen_tokens, target_toks)
                state.batch_results[i]["lcs_ratio_history"].append(lcs_ratio)

                # Check if converged
                have_inverted = torch.equal(state.target_tokens[i], generated[i])

                if have_inverted:
                    state.batch_results[i]["found_solution"] = True
                    state.num_success_items += 1

                # Remove if converged or max epochs reached
                if have_inverted or state.batch_results[i]["done_epochs"] >= epochs:
                    state.batch_results[i]["pred_tokens"] = state.pred_tokens[i].cpu()
                    state.batch_results[i]["generated_tokens"] = gen_tokens

                    # Remove from batch
                    state.pred_tokens = torch.cat([
                        state.pred_tokens[:i],
                        state.pred_tokens[i + 1:]
                    ], dim=0)
                    state.target_tokens = torch.cat([
                        state.target_tokens[:i],
                        state.target_tokens[i + 1:]
                    ], dim=0)
                    grad = torch.cat([grad[:i], grad[i + 1:]], dim=0)
                    state.results.append(state.batch_results.pop(i))

            # Skip GCG update if batch is empty
            if len(state.batch_results) == 0:
                state.elapsed_time += time.time() - start_time
                continue

            # Try multiple candidates and keep the best for each item
            batch_size_current = len(state.batch_results)
            best_pred_tokens = [None for _ in range(batch_size_current)]
            best_losses = [None for _ in range(batch_size_current)]

            for _ in range(num_candidates):
                # Select mutation positions
                new_token_pos = select_mutation_positions(
                    grad=grad,
                    num_mutations=num_mutations,
                    pos_choice=pos_choice,
                    batch_size=batch_size_current,
                    seq_len=seq_len,
                )

                # Select token candidates
                new_token_val = select_token_candidates(
                    grad=grad,
                    new_token_pos=new_token_pos,
                    top_k=top_k,
                    token_choice=token_choice,
                    num_mutations=num_mutations,
                    batch_size=batch_size_current,
                )

                # Apply mutations
                new_pred_tokens = apply_mutations(
                    pred_tokens=state.pred_tokens,
                    new_token_pos=new_token_pos,
                    new_token_val=new_token_val,
                )

                # Evaluate candidates
                new_pred_tokens_full = torch.cat([
                    new_pred_tokens,
                    state.target_tokens[:batch_size_current, :-1]
                ], dim=1)
                new_logits = model(new_pred_tokens_full)
                new_target_logits = new_logits[:, seq_len - 1:, :]

                new_log_probs = F.softmax(new_target_logits, dim=-1).clamp(min=1e-12).log()
                new_target_log_probs = new_log_probs.gather(
                    2, state.target_tokens[:batch_size_current].unsqueeze(-1)
                ).squeeze(-1)
                new_max_log_probs = new_log_probs.max(dim=-1).values
                new_loss_diff = new_target_log_probs - new_max_log_probs
                new_losses = -new_loss_diff.mean(dim=-1)

                # Keep best for each item
                for i in range(batch_size_current):
                    if best_losses[i] is None or new_losses[i] < best_losses[i]:
                        best_pred_tokens[i] = new_pred_tokens[i].unsqueeze(0).clone()
                        best_losses[i] = new_losses[i].clone()

            # Update to best tokens
            state.pred_tokens = torch.cat(best_pred_tokens, dim=0)

        state.elapsed_time += time.time() - start_time

        # Update progress bar
        completed = len(state.results)
        if completed > completed_prev:
            pbar.update(completed - completed_prev)
            completed_prev = completed
            pbar.set_postfix({
                "success": f"{state.num_success_items}/{completed}",
                "epoch": state.epoch
            })

        # Log to wandb (per-global-epoch)
        avg_lcs = 0.0
        if len(state.batch_results) > 0:
            lcs_values = [r["lcs_ratio_history"][-1] for r in state.batch_results if r["lcs_ratio_history"]]
            avg_lcs = sum(lcs_values) / len(lcs_values) if lcs_values else 0.0

        wandb_log = {
            "global_epoch": state.epoch,
            "active_batch_size": len(state.batch_results),
            "num_completed": len(state.results),
            "num_remaining": state.num_remain_items,
            "num_success": state.num_success_items,
            "avg_lcs_ratio": avg_lcs,
            "loss": loss.item(),
        }
        wandb.log(wandb_log)

    pbar.close()

    # Create summary table at the end
    summary_table = wandb.Table(columns=[
        "target_id",
        "target_text",
        "generated_text",
        "final_lcs_ratio",
        "found_solution",
        "epochs_used",
    ])

    for result in state.results:
        final_lcs = result["lcs_ratio_history"][-1] if result["lcs_ratio_history"] else 0.0
        gen_text = model.to_string(result["generated_tokens"]) if "generated_tokens" in result else ""
        summary_table.add_data(
            result["target_id"],
            result["target_text"],
            gen_text,
            final_lcs,
            result["found_solution"],
            result["done_epochs"],
        )

    wandb.log({
        "batch_summary_table": summary_table,
        "final_success_rate": state.num_success_items / num_targets if num_targets > 0 else 0.0,
        "total_elapsed_time": state.elapsed_time,
    })

    return {
        "results": state.results,
        "elapsed_time": round(state.elapsed_time, 3),
        "num_success": state.num_success_items,
        "num_targets": num_targets,
    }
