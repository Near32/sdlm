"""
SODA optimization using transformer_lens - internal batching variant.

This implements the SODA algorithm with dynamic batch queue management
using transformer_lens (closest to the original SODA paper implementation).
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import logging
import time
import copy
import wandb

from .soda_utils import (
    DotDict,
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


def soda_optimize_inputs_tl_batch(
    model_name: str,
    device: torch.device,
    targets: List[Dict[str, Any]],
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
    max_batch_size: int = 285,
    early_stop_on_exact_match: bool = True,
    model: Optional[Any] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Multi-target SODA optimization using transformer_lens with internal batching.

    This variant manages a dynamic queue of targets, processing them in batches
    and removing targets as they converge.

    Args:
        model_name: Name of the model to load via transformer_lens
        device: Device to run optimization on
        targets: List of target dictionaries with 'text', 'id', etc.
        seq_len: Length of the prompt sequence to optimize
        epochs: Maximum number of optimization epochs per target
        learning_rate: Learning rate for optimizer
        temperature: Temperature for softmax
        decay_rate: Decay rate applied to embedding params each epoch
        betas: Adam beta parameters
        reset_epoch: Reset optimizer state every N epochs
        reinit_epoch: Reinitialize embeddings every N epochs
        reg_weight: Weight for fluency regularization
        bias_correction: Use standard Adam with bias correction if True
        init_strategy: Initialization strategy
        init_std: Standard deviation for normal initialization
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
                "transformer_lens is required for SODA TL backend. "
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
        "optimizers": [],
        "loaded_i": 0,
        "epoch": 0,
        "num_remain_items": num_targets,
        "num_success_items": 0,
        "elapsed_time": 0,
    })

    # Main optimization loop
    pbar = tqdm(total=num_targets, desc="SODA TL Batch", position=0)
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

                    # Initialize embedding parameters
                    new_pred_embed = initialize_embedding_params(
                        vocab_size=vocab_size,
                        seq_len=seq_len,
                        device=device,
                        init_strategy=init_strategy,
                        init_std=init_std,
                    )

                    # Create separate optimizer for each target
                    params = []
                    for j in range(seq_len):
                        pos_param = new_pred_embed[:, j:j+1, :].clone()
                        pos_param.requires_grad = True
                        params.append(pos_param)

                    optimizer = create_optimizer(
                        params=params,
                        learning_rate=learning_rate,
                        betas=betas,
                        bias_correction=bias_correction,
                    )
                    state.optimizers.append(optimizer)

                # Add target tokens to batch tensor
                for i in range(num_new_items):
                    idx = state.loaded_i + i
                    tokens = target_tokens_list[idx].to(device).unsqueeze(0)
                    state.target_tokens = torch.cat([state.target_tokens, tokens], dim=0)

                state.loaded_i += num_new_items

        if len(state.batch_results) == 0:
            continue

        # Optimize batch
        for optimizer in state.optimizers:
            optimizer.zero_grad()

        # Collect all embeddings
        pred_embed_pre = torch.cat([
            torch.cat([p for p in opt.param_groups[0]['params']], dim=1)
            for opt in state.optimizers
        ], dim=0).to(device)

        # Convert to soft one-hot
        pred_one_hot = torch.softmax(pred_embed_pre / temperature, dim=-1)

        # Get embeddings
        pred_embed = pred_one_hot @ embedding_weights

        # Forward pass with target concatenation (teacher forcing)
        target_length = state.target_tokens.shape[1]

        # Get target token embeddings (all but last token)
        target_embeds = model.embed(state.target_tokens[:, :-1])

        # Concatenate prompt and target embeddings
        pred_embed_full = torch.cat([pred_embed, target_embeds], dim=1)

        # Add positional embeddings if needed
        if use_pos_embed:
            pos_embed = model.pos_embed(pred_embed_full[:, :, 0].detach())
            pred_embed_full = pred_embed_full + pos_embed

        # Get logits
        pred_logits = model(pred_embed_full, start_at_layer=0)

        # Extract target prediction logits
        target_pred_logits = pred_logits[:, seq_len - 1:, :]

        # Compute loss
        log_probs = F.softmax(target_pred_logits, dim=-1).clamp(min=1e-12).log()
        target_log_probs = log_probs.gather(
            2, state.target_tokens.unsqueeze(-1)
        ).squeeze(-1)
        max_log_probs = log_probs.max(dim=-1).values
        loss_diff = target_log_probs - max_log_probs
        loss = -loss_diff.mean()

        # Add fluency regularization
        if reg_weight is not None and reg_weight > 0:
            prompt_logits = pred_logits[:, :seq_len, :]
            fluency_penalty = compute_fluency_penalty(prompt_logits, pred_one_hot)
            loss = loss + reg_weight * fluency_penalty.mean()

        # Backward
        loss.backward()

        # Update optimizers
        for optimizer in state.optimizers:
            optimizer.step()

        # Apply decay and periodic interventions
        with torch.no_grad():
            for i, optimizer in enumerate(state.optimizers):
                params = optimizer.param_groups[0]['params']

                # Apply decay
                for param in params:
                    param.mul_(decay_rate)

                # Get done epochs for this target
                done_epochs = state.batch_results[i]["done_epochs"] + 1

                # Reset optimizer state periodically
                if done_epochs % reset_epoch == 0:
                    for param in params:
                        if param in optimizer.state:
                            del optimizer.state[param]

                # Reinitialize periodically
                if done_epochs % reinit_epoch == 0:
                    for param in params:
                        param.normal_(std=0.1)

            # Check convergence and update results
            pred_tokens = pred_one_hot.argmax(dim=-1)

            # Get discrete predictions for convergence check
            pred_tokens_full = torch.cat(
                [pred_tokens, state.target_tokens[:, :-1]], dim=1
            )
            discrete_logits = model(pred_tokens_full)
            generated = discrete_logits[:, seq_len - 1:, :].argmax(dim=-1)

            # Process results in reverse order to safely remove
            for i in range(len(state.batch_results) - 1, -1, -1):
                state.batch_results[i]["done_epochs"] += 1

                # Check if converged
                gen_tokens = generated[i].cpu().tolist()
                target_toks = state.target_tokens[i].cpu().tolist()
                lcs_ratio = compute_lcs_ratio(gen_tokens, target_toks)
                state.batch_results[i]["lcs_ratio_history"].append(lcs_ratio)

                have_inverted = torch.equal(
                    state.target_tokens[i],
                    generated[i]
                )

                if have_inverted:
                    state.batch_results[i]["found_solution"] = True
                    state.num_success_items += 1

                # Remove if converged or max epochs reached
                if have_inverted or state.batch_results[i]["done_epochs"] >= epochs:
                    state.batch_results[i]["pred_tokens"] = pred_tokens[i].cpu()
                    state.batch_results[i]["generated_tokens"] = generated[i].cpu().tolist()

                    # Remove from batch
                    del state.optimizers[i]
                    state.target_tokens = torch.cat([
                        state.target_tokens[:i],
                        state.target_tokens[i + 1:]
                    ], dim=0)
                    state.results.append(state.batch_results.pop(i))

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
            "temperature": temperature,
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
