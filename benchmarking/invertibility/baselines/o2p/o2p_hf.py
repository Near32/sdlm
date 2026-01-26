"""
O2P optimization using HuggingFace transformers - evaluation implementation.

This implements the O2P (Output-to-Prompt) method for prompt inversion
using a pre-trained T5-based inverse model with HuggingFace models.
"""

import torch
from typing import List, Tuple, Dict, Any, Optional
import logging
import copy
import wandb

from baselines.soda.soda_utils import compute_lcs_ratio
from .o2p_model import LLMInversionModel

logger = logging.getLogger(__name__)

# Global cache for loaded O2P models
_o2p_model_cache: Dict[str, LLMInversionModel] = {}


def o2p_optimize_inputs_hf(
    model,
    tokenizer,
    device: torch.device,
    target_text: str,
    o2p_model_path: str,
    seq_len: int,
    num_beams: int = 4,
    max_length: int = 32,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], torch.Tensor, List[float], List[float]]:
    """
    O2P prompt inversion using HuggingFace models.

    O2P works by:
    1. Tokenizing the target text with the LLM tokenizer
    2. Getting LLM logits for the target text
    3. Passing logits through the pre-trained T5 inverse model
    4. Generating an inverted prompt via beam search

    Args:
        model: HuggingFace AutoModelForCausalLM (subject LLM)
        tokenizer: HuggingFace AutoTokenizer for the LLM
        device: Device to run on
        target_text: The target text we want to invert
        o2p_model_path: Path to the trained O2P inverse model
        seq_len: Length of the prompt sequence for padding/truncation
        num_beams: Number of beams for T5 beam search
        max_length: Maximum generation length for T5 decoder
        kwargs: Additional keyword arguments (for compatibility)

    Returns:
        Tuple of:
            - generated_tokens: List of generated token IDs (the inverted prompt)
            - optimized_inputs: Tensor placeholder (zeros, for API compatibility)
            - losses: Single-element list [0.0] (no iterative loss for O2P)
            - lcs_ratio_history: Single-element list [final_lcs_ratio]
    """
    kwargs = kwargs or {}

    # Load or get cached O2P model
    global _o2p_model_cache

    if o2p_model_path not in _o2p_model_cache:
        logger.info(f"Loading O2P inverse model from {o2p_model_path}")
        inverse_model = LLMInversionModel.from_pretrained(
            o2p_model_path,
            external_llm=model,
            external_llm_tokenizer=tokenizer,
            device=device,
        )
        inverse_model = inverse_model.to(device)
        inverse_model.eval()
        _o2p_model_cache[o2p_model_path] = inverse_model
        logger.info("O2P inverse model loaded and cached")
    else:
        inverse_model = _o2p_model_cache[o2p_model_path]
        # Update LLM reference in case it changed
        inverse_model.llm = model
        inverse_model.llm_tokenizer = tokenizer

    # Tokenize the target text
    target_tokens = tokenizer(target_text, return_tensors="pt").input_ids.to(device)
    target_length = target_tokens.shape[1]
    target_tokens_list = target_tokens[0].cpu().tolist()

    # Create attention mask
    attention_mask = torch.ones_like(target_tokens)

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

    # Generate inverted prompt using O2P model
    with torch.no_grad():
        # Generate inverted prompt tokens
        generated_prompt_ids = inverse_model.generate(
            input_ids=target_tokens,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
        )

        # Decode the generated prompt
        generated_prompt_text = inverse_model.tokenizer.decode(
            generated_prompt_ids[0], skip_special_tokens=True
        )

        # Re-tokenize with LLM tokenizer to get prompt tokens
        prompt_tokens = tokenizer(
            generated_prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        ).input_ids.to(device)
        learned_input_ids = prompt_tokens[0].cpu().tolist()

        # Now verify: run the prompt through the LLM and see what it generates
        # Get embedding layer
        embedding_layer = model.get_input_embeddings()

        # Get embeddings for the prompt
        prompt_embeds = embedding_layer(prompt_tokens)

        # For evaluation, generate completion autoregressively
        # Use teacher forcing setup similar to SODA for fair comparison
        target_token_embeds = embedding_layer(target_tokens[:, :-1])
        combined_embeds = torch.cat([prompt_embeds, target_token_embeds], dim=1)

        # Get model outputs
        outputs = model(inputs_embeds=combined_embeds, return_dict=True)
        logits = outputs.logits

        # Extract logits for target prediction positions
        generated_logits = logits[:, seq_len - 1 : seq_len - 1 + target_length, :]
        generated_tokens = generated_logits.argmax(dim=-1)
        generated_tokens_list = generated_tokens[0].cpu().tolist()

        # Compute LCS ratio
        lcs_ratio = compute_lcs_ratio(generated_tokens_list, target_tokens_list)

        # Compute token overlap metrics
        def compute_token_overlap(prompt_tokens_tensor, target_tokens_tensor):
            """Compute token overlap and target hit ratio."""
            tt_occ = {}
            for ttoken in target_tokens_tensor:
                tt_occ[ttoken.item()] = (prompt_tokens_tensor == ttoken).int().sum().item()
            nbr_occ = sum(tt_occ.values())
            max_occ = len(prompt_tokens_tensor)
            overlap_ratio = nbr_occ / max_occ if max_occ > 0 else 0.0
            target_set = set(target_tokens_tensor.tolist())
            target_hits = {t: int(occ > 0) for t, occ in tt_occ.items()}
            target_hit_ratio = sum(target_hits.values()) / len(target_set) if target_set else 0.0
            return overlap_ratio, target_hit_ratio

        token_overlap_ratio, target_hit_ratio = compute_token_overlap(
            torch.tensor(learned_input_ids), target_tokens[0].cpu()
        )

        # Decode strings for logging
        learned_input_str = tokenizer.decode(learned_input_ids, skip_special_tokens=False)
        generated_output_str = tokenizer.decode(generated_tokens_list, skip_special_tokens=False)

    # Log to wandb
    wandb_log = {
        "epoch": 1,  # O2P is one-shot
        "loss": 0.0,  # No iterative loss for O2P
        "lcs_ratio": lcs_ratio,
        "token_overlap_ratio": token_overlap_ratio,
        "target_hit_ratio": target_hit_ratio,
        "o2p_num_beams": num_beams,
        "o2p_max_length": max_length,
    }
    wandb.log(wandb_log)

    # Add to wandb table
    wandb_table.add_data(
        1,  # epoch
        target_text,
        learned_input_ids,
        learned_input_str,
        generated_tokens_list,
        generated_output_str,
        lcs_ratio,
        token_overlap_ratio,
        target_hit_ratio,
    )

    # Log final table
    wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})

    logger.info(f"O2P: Final LCS ratio: {lcs_ratio:.4f}")

    # Create placeholder tensor for optimized_inputs (for API compatibility)
    # Use zeros with shape matching expected format
    optimized_inputs = torch.zeros(
        1, seq_len, model.config.vocab_size,
        device=device, dtype=torch.float32
    )

    # Return single-element history lists for API compatibility
    return generated_tokens_list, optimized_inputs, [0.0], [lcs_ratio]


def clear_o2p_cache():
    """Clear the O2P model cache to free memory."""
    global _o2p_model_cache
    _o2p_model_cache.clear()
    logger.info("O2P model cache cleared")


__all__ = ["o2p_optimize_inputs_hf", "clear_o2p_cache"]
